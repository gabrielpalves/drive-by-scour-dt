"""Build the six commit-bound Paper-1 dispatch bundles.

The publication set contains four MATLAB generation bundles and two balanced
training bundles. Each archive extracts into the repository root and runs
without source editing.

The exact source set comes from the tracked ``bundle_source_files.txt`` manifest.
It is therefore part of the reviewed Git commit rather than being inherited from
an untracked historical ZIP.
"""
from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
if _bootstrap_source_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_source_root)
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Mapping, NamedTuple
import unicodedata
import zipfile

REPO = Path(__file__).resolve().parent
BUILDER_NAME = "build_stage_bundles.py"
SOURCE_MANIFEST_NAME = "bundle_source_files.txt"
BUNDLE_IDENTITY_NAME = "paper1_bundle_identity.json"
BUNDLE_IDENTITY_SCHEMA = "ttbi-paper1-bundle-identity-v1"
DRIVER = "comprehensive_ablation_multidamage.py"
A00 = "scour_MATLAB/A00_Run.m"


class BundleBuildError(RuntimeError):
    """Fail-closed bundle precondition or publication error."""


# The only publishable package set is the settled four-stage Paper-1 campaign.
from core.paper1_dispatch import (
    GENERATION_BUNDLE_NAMES,
    TRAINING_BUNDLE_NAMES,
    generation_manifest,
    training_manifests,
)
from core.paper1_training_contract import canonical_json_bytes

BUNDLES = MappingProxyType({
    "f40s_generate": ("generation", "F40-S", "40 m scour generation"),
    "f40m_generate": ("generation", "F40-M", "40 m multi-damage generation"),
    "l99s_generate": ("generation", "L99-S", "99.6 m scour generation"),
    "l99m_generate": ("generation", "L99-M", "99.6 m multi-damage generation"),
    "train_labA": ("training", "labA", "Lab-A logical job share"),
    "train_labB": ("training", "labB", "Lab-B logical job share"),
})
EXPECTED_BUNDLE_ORDER = (
    "f40s_generate",
    "f40m_generate",
    "l99s_generate",
    "l99m_generate",
    "train_labA",
    "train_labB",
)
if tuple(BUNDLES) != EXPECTED_BUNDLE_ORDER:
    raise RuntimeError("registered Paper-1 six-bundle set drifted")


def paper1_readme(
    bundle_kind: str,
    target: str,
    purpose: str,
    source_commit: str,
) -> str:
    """Return concise instructions derived from the live bundle manifest."""

    training_manifest_command = (
        "$env:TTBI_TRAINING_JOB_MANIFEST = "
        "'<ABSOLUTE_PATH_TO_training_job_manifest.json>'"
        if bundle_kind == "training"
        else "Remove-Item Env:TTBI_TRAINING_JOB_MANIFEST "
        "-ErrorAction SilentlyContinue # generation bundle"
    )
    capacity_command = (
        "python -B capacity_preflight_compute.py --all-stages --receipt-dir "
        '"$env:TTBI_EXECUTION_RECEIPT_DIR"'
    )
    if bundle_kind == "generation":
        capacity_command = "# Training only: " + capacity_command
    lines = [
        f"# Paper-1 {bundle_kind} bundle: {target} — {purpose}",
        "",
        f"Source commit: `{source_commit}`.",
        "",
        "Verify this ZIP against `bundle_sha256.txt` and extract it into a",
        "fresh workspace. Never reuse pre-r12 data, caches, studies, or results.",
        "The extracted `paper1_bundle_identity.json` authenticates the embedded",
        "commit label, source manifest, executable source roots, and bundle",
        "manifest; a `.git` directory is neither included nor required.",
        "For Python work, install `requirements-portable.txt` with versions and",
        "a CUDA-enabled PyTorch build compatible with this PC. The pinned",
        "py313/cu128 file is only an optional known-good setup reference.",
        "All retained data use `physical8_v1`; the FRA-4 profile phase is fixed",
        "and shared across all production states and passages;",
        "wheelset proxy rows 3/4 are V&V diagnostics and are excluded from",
        "learning, whose eligible physical sensor indices are 0,1,2,5,6,7;",
        "operational EOV is enabled; track damage and wheel OOR are disabled.",
        "These six ZIPs dispatch only the complete 1,600-job primary grid.",
        "Modern-TCN/TSLANet challengers remain contract/model definitions only:",
        "they have no executor or job manifest here and are not runnable or",
        "claimable. Any later audited challenger dispatch must use only the",
        "authenticated F40-S selected pair.",
        "",
        "## Durable paths and run identity",
        "",
        "Before retained work, configure these explicitly as absolute paths.",
        "The data root must already exist and its final component must be",
        "`data`. Training job results, caches, studies, and receipts belong on",
        "durable storage outside this extracted source workspace. Generation is",
        "the deliberate exception: A00 first writes inside",
        "`scour_MATLAB/Results/<case_name>`; copy that completed folder to",
        "`TTBI_DATA_ROOT` only after generation and both parity checks finish:",
        "",
        "```powershell",
        "$env:TTBI_DATA_ROOT = '<ABSOLUTE_DURABLE_ROOT>\\data'",
        "$env:TTBI_RESULTS_ROOT = '<ABSOLUTE_DURABLE_RESULTS_ROOT>'",
        "$env:TTBI_CACHE_ROOT = '<ABSOLUTE_DURABLE_CACHE_ROOT>'",
        "$env:TTBI_STUDY_ROOT = '<ABSOLUTE_DURABLE_STUDY_ROOT>'",
        "$env:TTBI_EXECUTION_RECEIPT_DIR = '<ABSOLUTE_DURABLE_RECEIPT_ROOT>'",
        "$env:TTBI_CAMPAIGN_RUN_TAG = '<ONE_SHARED_PROSPECTIVE_RUN_TAG>'",
        training_manifest_command,
        capacity_command,
        "```",
        "",
        "LabA and LabB may execute the currently unlocked phase in parallel,",
        "but the next phase must wait for an authenticated union of both",
        "partitions' result packages and Optuna SQLite files. Use common durable",
        "roots or byte-preserving authenticated consolidation; publish once,",
        "then redistribute each artifact's absolute path and SHA-256. See",
        "`README_CAMPAIGN.md` for the exact `--publish-*` barrier commands and",
        "the `Merge-TtbiTree` helper, which writes paired source/destination",
        "SHA-256 inventory files that must be retained.",
        "",
    ]
    if bundle_kind == "generation":
        manifest = generation_manifest(target)
        lines.extend([
            "## Generation",
            "",
            f"This manifest dispatches `scour_MATLAB/A00_Run.m` as `{target}`.",
            "The A00 bytes are identical in all six ZIPs. The authenticated",
            "`generation_bundle_manifest.json` selects this stage and its exact",
            "count tuple; do not edit either file.",
            f"Expected output: `{manifest['dataset']}` with",
            f"{manifest['n_states']} states × "
            f"{manifest['passages_per_state']} passages.",
            "On this PC, run the MATLAB capability/physics smokes before retained",
            "generation: `smoke_audit`, `smoke_geometry`, `smoke_stage3`,",
            "`smoke_familytable`, `smoke_b54_overlap_parity`, and",
            "`smoke_generation_worker`. Exact MATLAB/Update/toolbox versions are",
            "recorded as provenance and need not match the reference descriptor.",
            "A00 rejects every other working directory. From MATLAB, use exactly",
            "(replace the placeholder with the extracted bundle's absolute path):",
            "",
            "```matlab",
            "cd('<ABSOLUTE_EXTRACTED_BUNDLE>\\scour_MATLAB')",
            "A00_Run",
            "```",
            "",
            "A00 writes `scour_MATLAB/Results/<case_name>` inside this workspace.",
            "Do not redirect that in-progress folder outside the workspace. After",
            "`0001.mat` completes, run the two raw-parity commands sequentially",
            "and stop generation on either failure:",
            "",
            "```matlab",
            "smoke_raw_parity('<folder>')",
            "```",
            "",
            "```powershell",
            "python check_raw_parity.py '<folder>'",
            "```",
            "",
            "For MATLAB, `<folder>` is `Results/<case_name>` from",
            "`scour_MATLAB`; for Python it is",
            "`scour_MATLAB/Results/<case_name>` from the bundle root. Only after",
            "the whole dataset, completion marker, digests, and parity checks are",
            "complete, use `Copy-TtbiDataset` from `README_CAMPAIGN.md` to copy",
            f"it byte-for-byte into `TTBI_DATA_ROOT/{manifest['dataset']}`.",
            "",
        ])
    else:
        manifest = training_manifests()[target]
        lines.extend([
            "## Training",
            "",
            "Set `TTBI_TRAINING_JOB_MANIFEST` to the absolute path of",
            "`training_job_manifest.json`, for example in PowerShell:",
            "`$env:TTBI_TRAINING_JOB_MANIFEST =",
            "(Resolve-Path '.\\training_job_manifest.json').Path`.",
            "Do not run, omit, or move jobs outside that manifest.",
            f"This machine owns {manifest['assigned_job_count']} of "
            f"{manifest['complete_job_count']} prospectively enumerated jobs.",
            "The Lab-A/Lab-B manifests are disjoint logical job partitions and",
            "may run on different PCs, GPU models, Python versions, or compatible",
            "CUDA/PyTorch stacks.",
            "Create an existing receipt directory outside this extracted",
            "workspace, set the variables above, then run",
            "`python -B capacity_preflight_compute.py --all-stages --receipt-dir",
            "\"$env:TTBI_EXECUTION_RECEIPT_DIR\"` locally on every",
            "training PC first. It creates one fresh receipt for each of the four",
            "independent execution blocks and authenticates the embedded bundle",
            "identity before every block-bound probe of the longest L99 RAW shape",
            "(batch 32 × 2 channels × 11,791 samples). OOM, Optuna",
            "FAIL, a schema mismatch, or an unassigned job is fatal.",
            "",
        ])
    lines.extend([
        "See `README_CAMPAIGN.md` and `docs/paper1_campaign_plan.md` for the",
        "complete execution protocol and claim boundary.",
        "",
    ])
    return "\n".join(lines)

class TreeEntry(NamedTuple):
    mode: str
    object_type: str
    oid: str


class BundlePlan(NamedTuple):
    repo: Path
    source_commit: str
    names: tuple[str, ...]
    blobs: Mapping[str, bytes]


class BundleBuildResult(NamedTuple):
    source_commit: str
    bundles: tuple[Path, ...]
    sha_manifest: Path


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run Git without a shell and keep filenames/output byte-exact."""
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise BundleBuildError(
            f"Git command failed ({' '.join(args)}): {stderr or proc.returncode}"
        )
    return proc


def _head_tree(repo: Path, commit: str) -> dict[str, TreeEntry]:
    """Return the exact recursive tree at *commit*, without consulting index."""
    raw = _git(
        repo, "ls-tree", "-r", "-z", "--full-tree", commit
    ).stdout
    result: dict[str, TreeEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", 1)
            mode, object_type, oid = metadata.decode("ascii").split(" ", 2)
            name = raw_name.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise BundleBuildError(
                "HEAD contains a tree entry that cannot be decoded canonically."
            ) from exc
        result[name] = TreeEntry(mode, object_type, oid)
    return result


def _read_blob(repo: Path, oid: str) -> bytes:
    return _git(repo, "cat-file", "blob", oid).stdout


def _regular_blob(
    tree: Mapping[str, TreeEntry], name: str
) -> TreeEntry:
    entry = tree.get(name)
    if entry is None:
        raise BundleBuildError(
            f"Manifest entry is not tracked by the selected HEAD: {name!r}"
        )
    if entry.object_type != "blob" or entry.mode not in {"100644", "100755"}:
        raise BundleBuildError(
            "Every bundle source must be a regular tracked HEAD blob; "
            f"{name!r} has mode/type {entry.mode} {entry.object_type}."
        )
    return entry


def _parse_source_manifest(data: bytes) -> tuple[str, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleBuildError(
            "bundle_source_files.txt must be canonical UTF-8."
        ) from exc

    names: list[str] = []
    windows_forbidden = set('<>:"|?*')
    windows_reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        # Keep this grammar identical to core.source_provenance and MATLAB's
        # generator_source_root: only empty lines and column-zero comments are
        # ignorable; whitespace-only/indented comments are non-canonical.
        if line == "" or line.startswith("#"):
            continue
        if line != line.strip():
            raise BundleBuildError(
                f"Non-canonical whitespace in manifest line {line_number}."
            )
        name = line
        parts = name.split("/")
        bad_component = any(
            not part
            or part in {".", ".."}
            or part != part.strip()
            or part.endswith(".")
            or any(ch in windows_forbidden or ord(ch) < 32 or ord(ch) == 127
                   for ch in part)
            or part.split(".", 1)[0].upper() in windows_reserved
            for part in parts
        )
        if (
            "\\" in name
            or name.startswith("/")
            or PureWindowsPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or unicodedata.normalize("NFC", name) != name
            or bad_component
        ):
            raise BundleBuildError(
                f"Unsafe/non-canonical manifest entry on line "
                f"{line_number}: {name!r}"
            )
        if name in {"README_BUNDLE.md", BUNDLE_IDENTITY_NAME}:
            raise BundleBuildError(
                f"{name} is generated and cannot be a source entry."
            )
        names.append(name)

    if names != sorted(names):
        raise BundleBuildError(
            "bundle_source_files.txt entries must be sorted."
        )
    if len(names) != len(set(names)) or len(names) != len(
        {name.casefold() for name in names}
    ):
        raise BundleBuildError(
            "bundle_source_files.txt contains duplicate or "
            "case-colliding entries."
        )
    if not names:
        raise BundleBuildError("bundle_source_files.txt is empty.")
    return tuple(names)


def _assert_worktree_head_equivalent(
    repo: Path,
    relative_name: str,
    expected: TreeEntry,
) -> None:
    """Catch modified control files even when index stat flags hide them."""
    path = repo.joinpath(*relative_name.split("/"))
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise BundleBuildError(
            f"Required control file is absent from the working tree: "
            f"{relative_name}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise BundleBuildError(
            f"Required control file is not a regular working-tree file: "
            f"{relative_name}"
        )
    actual_oid = _git(
        repo,
        "hash-object",
        f"--path={relative_name}",
        "--",
        relative_name,
    ).stdout.decode("ascii").strip()
    if actual_oid != expected.oid:
        raise BundleBuildError(
            f"Working copy of {relative_name} is not HEAD-equivalent. "
            "Commit/revert it before building (index flags cannot bypass "
            "this check)."
        )


def prepare_bundle_plan(
    repo: str | os.PathLike[str] = REPO,
) -> BundlePlan:
    """Require a clean commit and snapshot immutable package bytes from HEAD."""
    repo = Path(repo).resolve()
    source_commit = _git(
        repo, "rev-parse", "--verify", "HEAD^{commit}"
    ).stdout.decode("ascii").strip()
    tree = _head_tree(repo, source_commit)

    builder_entry = _regular_blob(tree, BUILDER_NAME)
    manifest_entry = _regular_blob(tree, SOURCE_MANIFEST_NAME)
    _assert_worktree_head_equivalent(repo, BUILDER_NAME, builder_entry)
    _assert_worktree_head_equivalent(
        repo, SOURCE_MANIFEST_NAME, manifest_entry
    )

    # The manifest itself comes from the selected commit, never from the index
    # or working tree. All payloads are then addressed by immutable blob OID.
    names = _parse_source_manifest(_read_blob(repo, manifest_entry.oid))
    entries = {name: _regular_blob(tree, name) for name in names}
    for required in (DRIVER, A00):
        if required not in entries:
            raise BundleBuildError(
                f"Required stage-preset source is absent from manifest: "
                f"{required}"
            )

    # A bundle represents one complete reviewed commit. Refuse unrelated dirty
    # or untracked files too, so the dispatched tree has one unambiguous source.
    dirty = _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if dirty:
        rendered = dirty.decode("utf-8", errors="backslashreplace").replace(
            "\0", "\n"
        ).strip()
        raise BundleBuildError(
            "Refusing to build bundles from dirty/untracked runtime source. "
            "Commit the reviewed changes first:\n" + rendered
        )

    blobs = {
        name: _read_blob(repo, entries[name].oid)
        for name in names
    }

    return BundlePlan(
        repo=repo,
        source_commit=source_commit,
        names=names,
        blobs=MappingProxyType(blobs),
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    """Stable metadata makes identical reviewed inputs byte-reproducible."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def _payload_root(
    names: tuple[str, ...], payloads: Mapping[str, bytes]
) -> tuple[str, int]:
    """Hash ordered manifest-relative payload bytes like source provenance."""

    lines = [
        f"{name}:{hashlib.sha256(payloads[name]).hexdigest()}"
        for name in names
    ]
    return (
        hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        len(lines),
    )


def bundle_identity(
    plan: BundlePlan,
    *,
    bundle_kind: str,
    target: str,
    payloads: Mapping[str, bytes],
    bundle_manifest_name: str,
    bundle_manifest_bytes: bytes,
) -> dict:
    """Bind an extracted Git-free bundle to its reviewed executable bytes."""

    if tuple(payloads) != plan.names or set(payloads) != set(plan.names):
        raise BundleBuildError(
            "bundle identity payload order differs from the source manifest"
        )
    generator_names = tuple(
        name for name in plan.names if name.startswith("scour_MATLAB/")
    )
    runtime_names = tuple(
        name for name in plan.names
        if name.endswith(".py")
    )
    if not generator_names or not runtime_names:
        raise BundleBuildError(
            "bundle identity lacks generator or Python runtime sources"
        )
    reviewed_sha, reviewed_count = _payload_root(plan.names, payloads)
    generator_sha, generator_count = _payload_root(
        generator_names, payloads
    )
    runtime_sha, runtime_count = _payload_root(runtime_names, payloads)
    return {
        "schema": BUNDLE_IDENTITY_SCHEMA,
        "source_commit": plan.source_commit,
        "bundle_kind": bundle_kind,
        "target": target,
        "source_manifest_name": SOURCE_MANIFEST_NAME,
        "source_manifest_sha256": hashlib.sha256(
            payloads[SOURCE_MANIFEST_NAME]
        ).hexdigest(),
        "source_manifest_entry_count": len(plan.names),
        "reviewed_source_root_sha256": reviewed_sha,
        "reviewed_source_file_count": reviewed_count,
        "generator_source_root_sha256": generator_sha,
        "generator_source_file_count": generator_count,
        "python_runtime_source_root_sha256": runtime_sha,
        "python_runtime_source_file_count": runtime_count,
        "bundle_manifest_name": bundle_manifest_name,
        "bundle_manifest_sha256": hashlib.sha256(
            bundle_manifest_bytes
        ).hexdigest(),
    }


def _write_paper1_bundle(
    output: Path,
    plan: BundlePlan,
    bundle_kind: str,
    target: str,
    purpose: str,
) -> None:
    payloads: dict[str, bytes] = {}
    for name in plan.names:
        # Executable source is byte-identical in all six bundles. Generation
        # stage selection lives only in the separately authenticated manifest.
        payloads[name] = plan.blobs[name]

    if bundle_kind == "generation":
        manifest_name = "generation_bundle_manifest.json"
        manifest = generation_manifest(target)
    elif bundle_kind == "training":
        manifest_name = "training_job_manifest.json"
        manifest = training_manifests()[target]
    else:
        raise BundleBuildError(f"unknown bundle kind {bundle_kind!r}")
    manifest_bytes = canonical_json_bytes(manifest)
    identity = bundle_identity(
        plan,
        bundle_kind=bundle_kind,
        target=target,
        payloads=MappingProxyType(payloads),
        bundle_manifest_name=manifest_name,
        bundle_manifest_bytes=manifest_bytes,
    )

    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name in plan.names:
            archive.writestr(_zip_info(name), payloads[name])
        archive.writestr(
            _zip_info(manifest_name), manifest_bytes
        )
        archive.writestr(
            _zip_info(BUNDLE_IDENTITY_NAME), canonical_json_bytes(identity)
        )
        archive.writestr(
            _zip_info("README_BUNDLE.md"),
            paper1_readme(
                bundle_kind,
                target,
                purpose,
                plan.source_commit,
            ).encode("utf-8"),
        )


def build_bundles(
    repo: str | os.PathLike[str] = REPO,
) -> BundleBuildResult:
    """Build and atomically publish a commit-bound complete bundle set."""
    # Every source/integrity gate completes before even a temporary ZIP exists.
    plan = prepare_bundle_plan(repo)
    repo = plan.repo
    bundle_items = tuple(BUNDLES.items())
    if (
        len(bundle_items) != len(EXPECTED_BUNDLE_ORDER)
        or tuple(key for key, _ in bundle_items) != EXPECTED_BUNDLE_ORDER
    ):
        raise BundleBuildError(
            "Refusing partial bundle publication: the registered six-bundle "
            "bundle set is incomplete or reordered."
        )

    sha_manifest = repo / "bundle_sha256.txt"
    invalid_manifest = repo / (
        "bundle_sha256.txt.INVALID_BUILD_IN_PROGRESS"
    )
    built_metadata: list[tuple[str, int, str]] = []
    bundle_paths: list[Path] = []

    # Construct and hash the complete candidate set off to the side. Only then
    # invalidate the old complete-set marker and atomically publish each file.
    with tempfile.TemporaryDirectory(
        prefix=".bundle-build-", dir=repo
    ) as staging_raw:
        staging = Path(staging_raw)
        sha_lines = [
            f"# source_commit {plan.source_commit}",
            f"# complete_bundle_count {len(bundle_items)}",
        ]
        staged_bundles: list[tuple[Path, Path]] = []
        for _key, (bundle_kind, target_key, purpose) in bundle_items:
            bundle_name = (
                GENERATION_BUNDLE_NAMES[target_key]
                if bundle_kind == "generation"
                else TRAINING_BUNDLE_NAMES[target_key]
            )
            staged = staging / bundle_name
            _write_paper1_bundle(
                staged, plan, bundle_kind, target_key, purpose
            )
            digest = hashlib.sha256(staged.read_bytes()).hexdigest()
            sha_lines.append(f"{digest}  {bundle_name}")
            target = repo / bundle_name
            staged_bundles.append((staged, target))
            built_metadata.append(
                (bundle_name, staged.stat().st_size // 1024, purpose)
            )
            bundle_paths.append(target)

        staged_manifest = staging / "bundle_sha256.txt"
        staged_manifest.write_text(
            "\n".join(sha_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staged_invalid = staging / invalid_manifest.name
        staged_invalid.write_text(
            f"INCOMPLETE bundle publication for {plan.source_commit}\n",
            encoding="utf-8",
            newline="\n",
        )

        if sha_manifest.is_file():
            os.replace(sha_manifest, invalid_manifest)
        else:
            os.replace(staged_invalid, invalid_manifest)
        for staged, target in staged_bundles:
            os.replace(staged, target)
        os.replace(staged_manifest, sha_manifest)
        if invalid_manifest.is_file():
            invalid_manifest.unlink()

    print(f"{'bundle':30} {'KB':>5}  adds")
    for (bundle_name, kb, adds), sha_line in zip(
        built_metadata, sha_lines[2:]
    ):
        print(f"{bundle_name:30} {kb:5}  {adds}")
        print(f"  sha256 {sha_line.split()[0]}")
    print(
        f"\n{len(bundle_paths)} bundles x {len(plan.names) + 3} files, "
        "contents resolved from regular tracked HEAD blobs. "
        "SHA-256 manifest -> bundle_sha256.txt"
    )
    return BundleBuildResult(
        source_commit=plan.source_commit,
        bundles=tuple(bundle_paths),
        sha_manifest=sha_manifest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate clean-commit bundle inputs without writing ZIP files",
    )
    args = parser.parse_args(argv)
    try:
        if args.check_only:
            plan = prepare_bundle_plan()
            print(
                "PAPER-1 BUNDLE INPUTS PASS: "
                f"{plan.source_commit} ({len(plan.names)} source files)"
            )
        else:
            build_bundles()
    except BundleBuildError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    main()
