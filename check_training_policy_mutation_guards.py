"""Isolated mutation audit for executable training-policy guards.

This auditor copies only the Python/config inputs needed by the relevant
checkers into a temporary tree.  It proves every unmutated checker is green,
injects one precisely anchored defect at a time, and requires the intended
checker to turn red for the registered evidence.

Every mutation is restored from the original bytes in ``finally``.  The
temporary source snapshot must return to its pristine type/mode/byte state
after every run, and the complete real reviewed boundary must remain on one
clean unchanged HEAD.  The reviewed source manifest and its complete source
boundary are copied from one captured snapshot into a Git-environment-isolated
repository so live provenance follows the production path.  This is an
integrity mutation test, not an operating-system security sandbox: registered
checkers are synchronous and must not leave adversarial descendants running
after their subprocess exits.  No campaign data, results, or stage bundles are
copied into or left in that isolated source tree.  The bundle builder is an
audit/publication executable rather than a target-host runtime command, but it
is deliberately shipped and authenticated through the reviewed manifest.

Run serially:

    python check_training_policy_mutation_guards.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

from build_stage_bundles import _parse_source_manifest
from core.source_provenance import SOURCE_MANIFEST


REPO = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Mutation:
    name: str
    target: str
    checker: str
    original: str
    mutant: str
    evidence: str


@dataclass(frozen=True)
class BoundaryFile:
    relative: str
    payload: bytes
    filesystem_mode: int
    git_mode: str
    git_oid: str


MUTATIONS = (
    Mutation(
        name="objective_value is hardwired to aggregate MSE",
        target="core/task.py",
        checker="check_weighted_head_mse.py",
        original='    metric = policy[branch]',
        mutant='    metric = "mse"  # MUTANT: ignore objective policy',
        evidence="[FAIL] objective is SCOUR-primary when bearing heads exist",
    ),
    Mutation(
        name="scour loss range is hardwired instead of read from policy",
        target="core/task.py",
        checker="check_weighted_head_mse.py",
        original=(
            '            [head_ranges["scour"]] * n_scour_outputs(config)'
        ),
        mutant=(
            "            [60.0] * n_scour_outputs(config)"
            "  # MUTANT: ignore scour range policy"
        ),
        evidence=(
            "[FAIL] changing only loss policy changes executed head weights"
        ),
    ),
    Mutation(
        name="optimizer learning-rate key is hardwired",
        target="training/trainer.py",
        checker="check_weighted_head_mse.py",
        original='    lr_key = policy["lr_param"]',
        mutant='    lr_key = "lr"  # MUTANT: ignore optimizer lr_param policy',
        evidence=(
            "[FAIL] changing only optimizer policy changes executed "
            "learning rate"
        ),
    ),
    Mutation(
        name="scheduler eta_min is hardwired",
        target="training/trainer.py",
        checker="check_weighted_head_mse.py",
        original='    eta_min = float(policy["eta_min"])',
        mutant='    eta_min = 0.0  # MUTANT: ignore scheduler eta_min policy',
        evidence=(
            "[FAIL] changing only scheduler policy changes executed eta_min"
        ),
    ),
    Mutation(
        name="missing trial seed silently defaults to 42",
        target="training/trainer.py",
        checker="check_weighted_head_mse.py",
        original=(
            "    if key not in config:\n"
            "        raise KeyError(\n"
            "            f\"trial-seed policy requires config field {key!r}; \"\n"
            "            \"a default seed is forbidden\")\n"
            "    seed = config[key]"
        ),
        mutant=(
            "    seed = config.get(key, 42)"
            "  # MUTANT: collapse missing seed onto 42"
        ),
        evidence="[FAIL] missing trial seed fails closed",
    ),
    Mutation(
        name="cuDNN benchmark is hardwired instead of read from policy",
        target="core/utils.py",
        checker="check_environment_lock.py",
        original=(
            '    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]'
        ),
        mutant=(
            "    torch.backends.cudnn.benchmark = False"
            "  # MUTANT: ignore determinism policy"
        ),
        evidence=(
            "[FAIL] determinism behaviour is derived from its executable policy"
        ),
    ),
    Mutation(
        name="float32 matmul precision setter ignores executable policy",
        target="core/utils.py",
        checker="check_environment_lock.py",
        original=(
            '    torch.set_float32_matmul_precision('
            'policy["float32_matmul_precision"])'
        ),
        mutant=(
            '    torch.set_float32_matmul_precision("highest")'
            "  # MUTANT: ignore numeric-mode policy"
        ),
        evidence=(
            "[FAIL] numeric-mode setters and post-assertions are explicit"
        ),
    ),
    Mutation(
        name="CUDA matmul TF32 setter ignores executable policy",
        target="core/utils.py",
        checker="check_environment_lock.py",
        original=(
            "    torch.backends.cuda.matmul.allow_tf32 = \\\n"
            '        policy["cuda_matmul_allow_tf32"]'
        ),
        mutant=(
            "    torch.backends.cuda.matmul.allow_tf32 = False"
            "  # MUTANT: ignore numeric-mode policy"
        ),
        evidence=(
            "[FAIL] numeric execution mode is derived from its executable policy"
        ),
    ),
    Mutation(
        name="cuDNN TF32 setter ignores executable policy",
        target="core/utils.py",
        checker="check_environment_lock.py",
        original=(
            '    torch.backends.cudnn.allow_tf32 = '
            'policy["cudnn_allow_tf32"]'
        ),
        mutant=(
            "    torch.backends.cudnn.allow_tf32 = False"
            "  # MUTANT: ignore numeric-mode policy"
        ),
        evidence=(
            "[FAIL] numeric execution mode is derived from its executable policy"
        ),
    ),
    Mutation(
        name="float32 matmul precision postcondition launders actual state",
        target="core/utils.py",
        checker="check_environment_lock.py",
        original=(
            "            torch.get_float32_matmul_precision(),"
        ),
        mutant=(
            '            policy["float32_matmul_precision"],'
            "  # MUTANT: launder actual numeric mode"
        ),
        evidence=(
            "[FAIL] numeric-mode postcondition mismatch hard-fails"
        ),
    ),
    Mutation(
        name="cuDNN runtime equality launders the locked value",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original=(
            "    actual_cudnn = torch.backends.cudnn.version()"
        ),
        mutant=(
            '    actual_cudnn = spec.get("cudnn_runtime")'
            "  # MUTANT: ignore runtime"
        ),
        evidence="[FAIL] cuDNN runtime mismatch hard-fails",
    ),
    Mutation(
        name="trainer objective call bypasses TRAIN_PROTOCOL objective policy",
        target="training/trainer.py",
        checker="check_weighted_head_mse.py",
        original='            TRAIN_PROTOCOL["objective"],',
        mutant=(
            '            {"regression_with_bearing_heads": "mse", '
            '"default": "mse"},  # MUTANT: bypass protocol mapping'
        ),
        evidence=(
            "[FAIL] trainer consumes the protocol-hashed objective mapping"
        ),
    ),
    Mutation(
        name="production pipeline seeding omits determinism policy",
        target="training/pipeline.py",
        checker="check_campaign_controls.py",
        original=(
            '    set_global_seed(optuna_seed, TRAIN_PROTOCOL["determinism"])'
        ),
        mutant=(
            "    set_global_seed(optuna_seed)"
            "  # MUTANT: omit TRAIN_PROTOCOL determinism"
        ),
        evidence=(
            "[FAIL] production pipeline seeding consumes TRAIN_PROTOCOL "
            "determinism policy"
        ),
    ),
    Mutation(
        name="run plan accepts a missing campaign run tag",
        target="core/hyperparameter_policy.py",
        checker="check_hyperparameter_policy.py",
        original=(
            "    if not isinstance(campaign_run_tag, str):\n"
            "        raise HyperparameterPolicyError(\n"
            '            "campaign run plan lacks its exact run_tag"\n'
            "        )"
        ),
        mutant=(
            "    if False:  # MUTANT: run-plan run_tag guard disabled\n"
            "        raise HyperparameterPolicyError(\n"
            '            "campaign run plan lacks its exact run_tag"\n'
            "        )"
        ),
        evidence="[FAIL] campaign run plan without exact run tag",
    ),
    Mutation(
        name="run plan accepts an invalid execution receipt digest",
        target="core/hyperparameter_policy.py",
        checker="check_hyperparameter_policy.py",
        original=(
            '    if not _is_sha256(value["execution_receipt_sha256"]):\n'
            "        raise HyperparameterPolicyError(\n"
            '            "campaign run plan lacks a valid execution receipt '
            'SHA-256"\n'
            "        )"
        ),
        mutant=(
            "    if False:  # MUTANT: execution-receipt guard disabled\n"
            "        raise HyperparameterPolicyError(\n"
            '            "campaign run plan lacks a valid execution receipt '
            'SHA-256"\n'
            "        )"
        ),
        evidence="[FAIL] campaign run plan with invalid execution receipt",
    ),
    Mutation(
        name="follower run plan accepts a missing block-reference digest",
        target="core/hyperparameter_policy.py",
        checker="check_hyperparameter_policy.py",
        original=(
            "    elif not _is_sha256(block_reference_sha):\n"
            "        raise HyperparameterPolicyError(\n"
            '            "follower run plan lacks a valid block-reference '
            'manifest SHA-256"\n'
            "        )"
        ),
        mutant=(
            "    elif False:  # MUTANT: follower reference guard disabled\n"
            "        raise HyperparameterPolicyError(\n"
            '            "follower run plan lacks a valid block-reference '
            'manifest SHA-256"\n'
            "        )"
        ),
        evidence="[FAIL] follower run plan without block-reference digest",
    ),
)


BASELINE_EVIDENCE = {
    "check_campaign_controls.py": "CAMPAIGN CONTROLS: ALL PASS",
    "check_weighted_head_mse.py": "WEIGHTED HEAD MSE: ALL PASS",
    "check_environment_lock.py": "ENVIRONMENT LOCK: ALL PASS",
    "check_hyperparameter_policy.py": "HYPERPARAMETER POLICY: ALL PASS",
}

ROOT_INPUTS = (
    "check_campaign_controls.py",
    "check_weighted_head_mse.py",
    "check_environment_lock.py",
    "check_hyperparameter_policy.py",
    "comprehensive_ablation_multidamage.py",
    "build_stage_bundles.py",
)


def _git_environment(*, isolated_config: bool) -> dict[str, str]:
    """Drop inherited Git redirections; optionally null external config.

    Real-repository reads retain the machine's effective system/global clean
    filters (notably Windows ``core.autocrlf``).  The synthetic repository and
    checker subprocesses instead receive an explicitly isolated config.
    """
    env = os.environ.copy()
    for key in tuple(env):
        if key.upper().startswith("GIT_"):
            env.pop(key, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    if isolated_config:
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_SYSTEM"] = os.devnull
    return env


def _git(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    isolated_config: bool = False,
) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        env=_git_environment(isolated_config=isolated_config),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"isolated git {' '.join(args)} failed: "
            f"{detail or completed.returncode}"
        )
    return completed.stdout


def _git_text(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    isolated_config: bool = False,
) -> str:
    return _git(
        root,
        *args,
        input_bytes=input_bytes,
        isolated_config=isolated_config,
    ).decode("utf-8", errors="strict")


def _regular_file_snapshot(path: Path) -> tuple[bytes, int]:
    try:
        path_before = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"source boundary file is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError(
            f"source boundary path is not one regular file: {path}"
        )
    with path.open("rb") as stream:
        handle_before = os.fstat(stream.fileno())
        payload = stream.read()
        handle_after = os.fstat(stream.fileno())
    path_after = path.lstat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    before_identity = tuple(
        getattr(handle_before, field) for field in identity_fields
    )
    if (
        before_identity
        != tuple(getattr(handle_after, field) for field in identity_fields)
        or before_identity
        != tuple(getattr(path_before, field) for field in identity_fields)
        or before_identity
        != tuple(getattr(path_after, field) for field in identity_fields)
        or len(payload) != handle_before.st_size
        or path.is_symlink()
        or not stat.S_ISREG(path_after.st_mode)
    ):
        raise RuntimeError(
            f"source boundary file changed while being captured: {path}"
        )
    return payload, stat.S_IMODE(path_before.st_mode)


def _is_reparse_point(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(flag and attributes & flag)


def _assert_safe_directory_chain(root: Path, directory: Path) -> None:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"restore parent escapes isolated source: {directory}"
        ) from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(
            f"restore parent is non-canonical: {directory}"
        )
    current = root
    for part in ("", *relative.parts):
        if part:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"restore parent disappeared: {current}"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or _is_reparse_point(info)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise RuntimeError(
                f"restore parent is not one real directory: {current}"
            )


def _restore_regular_file(
    root: Path,
    target: Path,
    payload: bytes,
    filesystem_mode: int,
) -> None:
    _assert_safe_directory_chain(root, target.parent)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.restore-",
        dir=target.parent,
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, filesystem_mode)
        _assert_safe_directory_chain(root, target.parent)
        os.replace(temporary, target)
        if os.name != "nt":
            directory_descriptor = os.open(
                target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        restored, restored_mode = _regular_file_snapshot(target)
        if restored != payload or restored_mode != filesystem_mode:
            raise RuntimeError(
                f"atomic byte restoration failed for {target}"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                stat.S_ISREG(temporary_info.st_mode)
                and not stat.S_ISLNK(temporary_info.st_mode)
                and not _is_reparse_point(temporary_info)
            ):
                temporary.unlink()
            else:
                raise RuntimeError(
                    f"unsafe restore temporary remains: {temporary}"
                )


def _head_tree(root: Path, commit: str) -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", commit)
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ", 2)
            name = raw_name.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                "real HEAD contains a non-canonical tree record"
            ) from exc
        entries[name] = (mode, kind, oid)
    return entries


def _git_filtered_blob_oid(
    root: Path,
    relative: str,
    payload: bytes,
) -> str:
    oid = _git_text(
        root,
        "hash-object",
        "--stdin",
        f"--path={relative}",
        input_bytes=payload,
    ).strip()
    if not oid or any(char not in "0123456789abcdef" for char in oid):
        raise RuntimeError(
            f"Git returned an invalid filtered blob OID for {relative}: "
            f"{oid!r}"
        )
    return oid


def _assert_real_git_state(head: str) -> None:
    current = _git_text(REPO, "rev-parse", "--verify", "HEAD").strip()
    if current != head:
        raise RuntimeError(
            f"real repository HEAD moved during mutation audit: "
            f"{head} -> {current}"
        )
    dirty = _git_text(
        REPO, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if dirty:
        raise RuntimeError(
            "training mutation audit requires one clean commit-bound source "
            f"tree:\n{dirty}"
        )


def _capture_real_boundary() -> tuple[str, tuple[BoundaryFile, ...]]:
    head = _git_text(REPO, "rev-parse", "--verify", "HEAD").strip()
    if not head or any(char not in "0123456789abcdef" for char in head):
        raise RuntimeError(f"real repository HEAD is not a full hash: {head!r}")
    _assert_real_git_state(head)
    tree = _head_tree(REPO, head)
    manifest_entry = tree.get(SOURCE_MANIFEST)
    if manifest_entry is None:
        raise RuntimeError(f"{SOURCE_MANIFEST} is absent from real HEAD")
    manifest_git_mode, manifest_kind, manifest_oid = manifest_entry
    if (
        manifest_kind != "blob"
        or manifest_git_mode not in {"100644", "100755"}
    ):
        raise RuntimeError(
            f"{SOURCE_MANIFEST} is not one regular HEAD blob"
        )
    manifest_payload, manifest_filesystem_mode = _regular_file_snapshot(
        REPO / SOURCE_MANIFEST
    )
    if (
        _git_filtered_blob_oid(
            REPO, SOURCE_MANIFEST, manifest_payload
        )
        != manifest_oid
    ):
        raise RuntimeError(
            f"{SOURCE_MANIFEST} working content is not HEAD-equivalent "
            "after registered Git filters"
        )
    manifest_names = _parse_source_manifest(manifest_payload)
    if SOURCE_MANIFEST not in manifest_names:
        raise RuntimeError(
            f"{SOURCE_MANIFEST} must include itself in the reviewed boundary"
        )
    absent_roots = sorted(set(ROOT_INPUTS) - set(manifest_names))
    if absent_roots:
        raise RuntimeError(
            "mutation-harness roots are outside the reviewed source boundary: "
            f"{absent_roots}"
        )
    captured: list[BoundaryFile] = []
    for relative in manifest_names:
        entry = tree.get(relative)
        if entry is None:
            raise RuntimeError(
                f"reviewed source is absent from real HEAD: {relative}"
            )
        git_mode, kind, git_oid = entry
        if kind != "blob" or git_mode not in {"100644", "100755"}:
            raise RuntimeError(
                f"reviewed source is not one regular HEAD blob: "
                f"{relative} ({git_mode} {kind})"
            )
        if relative == SOURCE_MANIFEST:
            payload = manifest_payload
            filesystem_mode = manifest_filesystem_mode
        else:
            path = REPO.joinpath(*relative.split("/"))
            payload, filesystem_mode = _regular_file_snapshot(path)
        if _git_filtered_blob_oid(REPO, relative, payload) != git_oid:
            raise RuntimeError(
                "reviewed source working content is not HEAD-equivalent "
                f"after registered Git filters: {relative}"
            )
        captured.append(
            BoundaryFile(
                relative=relative,
                payload=payload,
                filesystem_mode=filesystem_mode,
                git_mode=git_mode,
                git_oid=git_oid,
            )
        )
    _assert_real_git_state(head)
    return head, tuple(captured)


def _copy_isolated_tree(
    destination: Path,
    sources: tuple[BoundaryFile, ...],
) -> None:
    for source in sources:
        target = destination.joinpath(*source.relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.payload)
        os.chmod(target, source.filesystem_mode)
    _assert_no_campaign_artifacts(destination)


def _initialise_isolated_git(
    root: Path,
    *,
    template_dir: Path,
    hooks_dir: Path,
) -> None:
    template_dir.mkdir(parents=True, exist_ok=False)
    hooks_dir.mkdir(parents=True, exist_ok=False)
    _git(
        root,
        "init",
        "--quiet",
        f"--template={template_dir}",
        isolated_config=True,
    )
    _git(
        root,
        "config",
        "user.name",
        "TTBI Mutation Guard",
        isolated_config=True,
    )
    _git(
        root,
        "config",
        "user.email",
        "mutation-guard@example.invalid",
        isolated_config=True,
    )
    _git(
        root,
        "config",
        "core.autocrlf",
        "false",
        isolated_config=True,
    )
    _git(
        root,
        "config",
        "core.hooksPath",
        str(hooks_dir.resolve()),
        isolated_config=True,
    )
    _git(root, "add", "--all", isolated_config=True)
    _git(
        root,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "isolated reviewed source baseline",
        isolated_config=True,
    )
    _assert_isolated_git_clean(root)


def _assert_isolated_git_clean(root: Path) -> None:
    dirty = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        isolated_config=True,
    )
    if dirty:
        raise RuntimeError(
            f"isolated reviewed source is not Git-clean:\n{dirty}"
        )


def _is_forbidden_component(name: str) -> bool:
    folded = name.casefold()
    if folded == SOURCE_MANIFEST.casefold():
        return False
    return (
        folded in {
            ".audit_tmp",
            "cache",
            "data",
            "results",
            "stale_pre_r11_bundles",
        }
        or folded.startswith("data_")
        or folded.startswith("results_")
        or folded.startswith("bundle_s")
        or folded.startswith("bundle_sha256")
        or (
            folded.endswith(".zip")
            and (
                folded.startswith("bundle_")
                or folded.startswith("multidamage_")
            )
        )
    )


def _assert_no_campaign_artifacts(root: Path) -> None:
    forbidden: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(_is_forbidden_component(part) for part in relative.parts):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise RuntimeError(
            "forbidden campaign artifacts entered isolated tree: "
            f"{sorted(forbidden)}"
        )


def _tree_source_snapshot(
    root: Path,
) -> dict[str, tuple[str, int, str | None]]:
    snapshot: dict[str, tuple[str, int, str | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        mode = path.lstat().st_mode
        name = relative.as_posix()
        if path.is_symlink():
            raise RuntimeError(
                f"symlink entered isolated source tree: {name}"
            )
        if stat.S_ISDIR(mode):
            snapshot[name] = ("directory", stat.S_IMODE(mode), None)
        elif stat.S_ISREG(mode):
            payload, filesystem_mode = _regular_file_snapshot(path)
            snapshot[name] = (
                "file",
                filesystem_mode,
                hashlib.sha256(payload).hexdigest(),
            )
        else:
            raise RuntimeError(
                f"special file entered isolated source tree: {name}"
            )
    return snapshot


def _assert_isolated_pristine(
    root: Path,
    pristine: dict[str, tuple[str, int, str | None]],
) -> None:
    _assert_no_campaign_artifacts(root)
    if _tree_source_snapshot(root) != pristine:
        raise RuntimeError("isolated source tree differs from pristine snapshot")
    _assert_isolated_git_clean(root)


def _assert_real_boundary_unchanged(
    head: str,
    before: tuple[BoundaryFile, ...],
) -> None:
    changed: list[str] = []
    for source in before:
        path = REPO.joinpath(*source.relative.split("/"))
        try:
            payload, filesystem_mode = _regular_file_snapshot(path)
        except RuntimeError:
            changed.append(source.relative)
            continue
        if (
            payload != source.payload
            or filesystem_mode != source.filesystem_mode
        ):
            changed.append(source.relative)
    _assert_real_git_state(head)
    if changed:
        raise RuntimeError(
            "real reviewed source boundary changed during isolated audit: "
            f"{changed}. Run this harness serially."
        )


def _inject_exactly_once(text: str, mutation: Mutation) -> str:
    occurrences = text.count(mutation.original)
    if occurrences != 1:
        raise RuntimeError(
            f"mutation anchor drift for {mutation.name!r}: expected exactly "
            f"one occurrence in {mutation.target}, found {occurrences}"
        )
    return text.replace(mutation.original, mutation.mutant, 1)


def _run_checker(
    isolated: Path,
    checker: str,
    timeout: int = 240,
) -> tuple[int, str]:
    env = _git_environment(isolated_config=True)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for variable in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(variable, None)
    completed = subprocess.run(
        [sys.executable, checker],
        cwd=isolated,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout


def _output_tail(output: str, lines: int = 40) -> str:
    return "\n".join(output.splitlines()[-lines:])


def main() -> int:
    tested_head, sources = _capture_real_boundary()
    caught = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="ttbi-training-policy-mutations-",
        ) as tmp:
            audit_root = Path(tmp)
            isolated = audit_root / "source"
            isolated.mkdir()
            _copy_isolated_tree(isolated, sources)
            _initialise_isolated_git(
                isolated,
                template_dir=audit_root / "empty-git-template",
                hooks_dir=audit_root / "empty-git-hooks",
            )
            pristine = _tree_source_snapshot(isolated)
            _assert_isolated_pristine(isolated, pristine)

            for checker in sorted(BASELINE_EVIDENCE):
                code, output = _run_checker(isolated, checker)
                evidence = BASELINE_EVIDENCE[checker]
                if code != 0 or evidence not in output:
                    raise RuntimeError(
                        f"isolated baseline {checker} is not GREEN: "
                        f"returncode={code}; expected={evidence!r}\n"
                        f"{_output_tail(output)}"
                    )
                if (
                    checker == "check_campaign_controls.py"
                    and "[N/A] audit-only bundle-manifest checks" in output
                ):
                    raise RuntimeError(
                        "campaign-control baseline skipped its audit-only "
                        "builder/Git branch"
                    )
                _assert_isolated_pristine(isolated, pristine)
                print(f"[BASELINE PASS] {checker} -> {evidence}")

            for index, mutation in enumerate(MUTATIONS, start=1):
                target = isolated / mutation.target
                original_bytes, original_mode = _regular_file_snapshot(target)
                try:
                    original_text = original_bytes.decode("utf-8")
                    mutated_text = _inject_exactly_once(
                        original_text, mutation)
                    target.write_bytes(mutated_text.encode("utf-8"))
                    code, output = _run_checker(
                        isolated, mutation.checker)
                    _assert_no_campaign_artifacts(isolated)
                    if code == 0 or mutation.evidence not in output:
                        raise RuntimeError(
                            "mutation was not caught for the intended reason: "
                            f"{mutation.name}\nreturncode={code}; "
                            f"expected evidence={mutation.evidence!r}\n"
                            f"{_output_tail(output)}"
                        )
                    caught += 1
                    print(
                        f"[CAUGHT {index}/{len(MUTATIONS)}] {mutation.name}\n"
                        f"  guard: {mutation.checker} -> "
                        f"{mutation.evidence}"
                    )
                finally:
                    _restore_regular_file(
                        isolated,
                        target,
                        original_bytes,
                        original_mode,
                    )
                    _assert_isolated_pristine(isolated, pristine)
                print(
                    f"  [RESTORED] {mutation.target} byte-identical; "
                    "temp source snapshot pristine"
                )
    finally:
        _assert_real_boundary_unchanged(tested_head, sources)

    if caught != len(MUTATIONS):
        raise RuntimeError(
            f"internal count mismatch: caught {caught}/{len(MUTATIONS)}")
    print(
        f"\nTRAINING POLICY MUTATION GUARDS: "
        f"{caught}/{len(MUTATIONS)} CAUGHT; 0 MISSED; "
        "ISOLATED TREE RESTORED BYTE-FOR-BYTE; REAL REVIEWED SOURCE "
        f"UNCHANGED AT {tested_head}; NO CAMPAIGN DATA/RESULTS/BUNDLES "
        "COPIED INTO OR LEFT IN THE ISOLATED SOURCE TREE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
