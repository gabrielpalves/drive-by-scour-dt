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
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
            "imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
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
import re
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
AUDIT_REPORT_NAME = "docs/audit_r5_results.md"
DRIVER = "comprehensive_ablation_multidamage.py"
A00 = "scour_MATLAB/A00_Run.m"
EXPECTED_AUDIT_STATUS = "**Status: PAPER-1 DISPATCH AUTHORIZED.**"
EXPECTED_AUDIT_HEADING = "# Paper-1 dispatch authorization (legacy filename)"
DISPATCH_AUTHORIZATION_ENV = "TTBI_DISPATCH_AUTHORIZATION_MANIFEST"


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
    "train_labA": ("training", "labA", "Lab-A matched-GPU job share"),
    "train_labB": ("training", "labB", "Lab-B matched-GPU job share"),
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


def set_a00_stage(t, stage):
    t2, n = re.subn(r"^STAGE = '[^']*';", f"STAGE = '{stage}';", t, count=1, flags=re.M)
    if n != 1:
        raise BundleBuildError("A00 STAGE line not found/replaced exactly once.")
    return t2


def set_a00_bundle_config(text: str, stage: str) -> str:
    """Preset one complete reviewed stage tuple, never only its label."""

    counts = {
        "F40-S": (0, 50, 5, 60, 5, 0),
        "F40-M": (250, 50, 50, 5, 5, 50),
        "L99-S": (250, 50, 50, 5, 5, 50),
        "L99-M": (250, 50, 50, 5, 5, 50),
    }
    if stage not in counts:
        raise BundleBuildError(f"unregistered generation stage {stage!r}")
    text = set_a00_stage(text, stage)
    variables = (
        "n_states_multi",
        "Npass",
        "n_healthy_states",
        "n_anchor_levels",
        "n_anchor_reps",
        "n_nuisance_states",
    )
    for variable, value in zip(variables, counts[stage], strict=True):
        text, replacements = re.subn(
            rf"^{variable}\s*=\s*\d+;.*$",
            f"{variable} = {value}; % BUNDLE PRESET: {stage}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if replacements != 1:
            raise BundleBuildError(
                f"A00 {variable} line not found/replaced exactly once"
            )
    return text

def paper1_readme(
    bundle_kind: str,
    target: str,
    purpose: str,
    source_commit: str,
    tested_source_commit: str,
    authorization_sha256: str,
) -> str:
    """Return concise instructions derived from the live bundle manifest."""

    lines = [
        f"# Paper-1 {bundle_kind} bundle: {target} — {purpose}",
        "",
        f"Dispatch bundle commit B: `{source_commit}`.",
        f"Tested source commit A: `{tested_source_commit}`.",
        f"Dispatch-authorization manifest SHA-256: `{authorization_sha256}`.",
        "",
        "Verify this ZIP against `bundle_sha256.txt` and extract it into a",
        "fresh workspace. Never reuse pre-r12 data, caches, studies, or results.",
        "All retained data use `physical8_v1`; profile phase is fixed;",
        "operational EOV is enabled; track damage and wheel OOR are disabled.",
        "",
    ]
    if bundle_kind == "generation":
        manifest = generation_manifest(target)
        lines.extend([
            "## Generation",
            "",
            f"`scour_MATLAB/A00_Run.m` is preset to `{target}`.",
            "`generation_bundle_manifest.json` is authoritative; do not edit",
            "the preset or count tuple.",
            f"Expected output: `{manifest['dataset']}` with",
            f"{manifest['n_states']} states × "
            f"{manifest['passages_per_state']} passages.",
            "Run the bundle preflights and host qualification before retained",
            "generation. After `0001.mat` completes, run MATLAB and Python raw",
            "parity sequentially and stop generation on either failure.",
            "",
        ])
    else:
        manifest = training_manifests()[target]
        lines.extend([
            "## Training",
            "",
            "Set `TTBI_TRAINING_JOB_MANIFEST` to the absolute path of",
            "`training_job_manifest.json`. Do not run, omit, or move jobs outside",
            "that manifest.",
            f"This machine owns {manifest['assigned_job_count']} of "
            f"{manifest['complete_job_count']} prospectively enumerated jobs.",
            "The Lab-A/Lab-B manifests are disjoint and share one complete-grid",
            "digest; their GPU model and numeric stack must match within every",
            "scientific block.",
            "Run the 16-cell capacity preflight first, including the longest",
            "L99 RAW shape (batch 32 × 8 channels × 11,791 samples). OOM, Optuna",
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
    tested_source_commit: str
    dispatch_authorization_manifest_sha256: str
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
        if name == "README_BUNDLE.md":
            raise BundleBuildError(
                "README_BUNDLE.md is generated and cannot be a source entry."
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


class AuthorizedReport(NamedTuple):
    tested_source_commit: str
    dispatch_authorization_manifest_sha256: str


def _parse_authorized_report(audit_report: str) -> AuthorizedReport:
    """Parse a fixed early-document authorization header, never free prose."""
    lines = audit_report.splitlines()
    if (
        len(lines) < 7
        or lines[0] != EXPECTED_AUDIT_HEADING
        or lines[1] != ""
        or lines[2] != EXPECTED_AUDIT_STATUS
        or lines[3] != ""
        or lines[6] != ""
    ):
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the legacy-filename Paper-1 report "
            f"must begin with {EXPECTED_AUDIT_HEADING!r} and place "
            f"{EXPECTED_AUDIT_STATUS!r} exactly on line 3 of its document "
            "header (not in prose or a code fence)."
        )
    tested_match = re.fullmatch(
        r"\*\*Tested source commit:\*\* `([0-9a-f]{40})`",
        lines[4],
    )
    manifest_match = re.fullmatch(
        r"\*\*Dispatch authorization manifest SHA-256:\*\* "
        r"`([0-9a-f]{64})`",
        lines[5],
    )
    status_lines = [
        line for line in lines if line.startswith("**Status:")
    ]
    tested_lines = [
        line for line in lines if line.startswith("**Tested source commit:**")
    ]
    manifest_lines = [
        line for line in lines
        if line.startswith(
            "**Dispatch authorization manifest SHA-256:**"
        )
    ]
    if (
        tested_match is None
        or manifest_match is None
        or status_lines != [EXPECTED_AUDIT_STATUS]
        or tested_lines != [lines[4]]
        or manifest_lines != [lines[5]]
    ):
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the legacy-filename Paper-1 header "
            "must contain one unique status, tested-source SHA, and external "
            "dispatch-manifest SHA-256 in the fixed header."
        )
    return AuthorizedReport(
        tested_source_commit=tested_match.group(1),
        dispatch_authorization_manifest_sha256=manifest_match.group(1),
    )


def _resolve_dispatch_authorization_manifest(
    explicit: str | os.PathLike[str] | None,
) -> Path:
    environment = os.environ.get(DISPATCH_AUTHORIZATION_ENV)
    explicit_text = None if explicit is None else os.fspath(explicit)
    if explicit_text is not None and environment is not None:
        if os.path.normcase(explicit_text) != os.path.normcase(environment):
            raise BundleBuildError(
                "--dispatch-authorization-manifest and "
                f"{DISPATCH_AUTHORIZATION_ENV} disagree"
            )
    selected = explicit_text if explicit_text is not None else environment
    if not selected:
        raise BundleBuildError(
            "Refusing to build without the absolute external mechanical "
            "dispatch-evidence manifest. Pass "
            "--dispatch-authorization-manifest or set "
            f"{DISPATCH_AUTHORIZATION_ENV}."
        )
    path = Path(selected)
    if not path.is_absolute():
        raise BundleBuildError(
            "dispatch authorization manifest path must be absolute"
        )
    return path


def _verify_dispatch_authorization(
    *,
    repo: Path,
    manifest_path: Path,
    tested_source_commit: str,
    dispatch_source_commit: str,
    expected_manifest_sha256: str,
) -> None:
    try:
        import dispatch_authorization
    except ImportError as exc:
        raise BundleBuildError(
            "Refusing to build: dispatch_authorization.py is unavailable"
        ) from exc
    try:
        dispatch_authorization.verify_dispatch_authorization_manifest(
            manifest_path,
            tested_source_commit=tested_source_commit,
            dispatch_source_commit=dispatch_source_commit,
            expected_manifest_sha256=expected_manifest_sha256,
            repo=repo,
        )
    except (
        dispatch_authorization.DispatchAuthorizationError,
        ImportError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        raise BundleBuildError(
            "Refusing to build: external mechanical dispatch evidence did "
            f"not revalidate: {exc}"
        ) from exc


def prepare_bundle_plan(
    repo: str | os.PathLike[str] = REPO,
    dispatch_authorization_manifest: (
        str | os.PathLike[str] | None
    ) = None,
) -> BundlePlan:
    """Validate all gates and snapshot immutable package bytes from HEAD."""
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
    if AUDIT_REPORT_NAME not in entries:
        raise BundleBuildError(
            f"Refusing to build dispatch bundles: {AUDIT_REPORT_NAME} is not "
            "in bundle_source_files.txt."
        )
    for required in (DRIVER, A00):
        if required not in entries:
            raise BundleBuildError(
                f"Required stage-preset source is absent from manifest: "
                f"{required}"
            )

    # Preserve the explicit dirty/untracked gate for ordinary mistakes. This
    # is supplementary: payload bytes still come exclusively from HEAD, and
    # direct hashing above prevents skip-worktree/assume-unchanged bypasses of
    # the two files that control what the builder executes and packages.
    dirty = _git(
        repo,
        "--literal-pathspecs",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        BUILDER_NAME,
        SOURCE_MANIFEST_NAME,
        *names,
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
    try:
        audit_report = blobs[AUDIT_REPORT_NAME].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleBuildError(
            f"{AUDIT_REPORT_NAME} must be UTF-8."
        ) from exc
    authorized = _parse_authorized_report(audit_report)
    tested_source_commit = authorized.tested_source_commit

    # Benchmark commit A must exist, be an ancestor of dispatch commit B, and
    # A..B may contain exactly the audit report. Every command is anchored to
    # the immutable source_commit captured above, so a concurrent HEAD/index/
    # working-tree change cannot change the bytes or lineage being packaged.
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        tested_source_commit,
        source_commit,
        check=False,
    )
    if ancestor.returncode != 0:
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the R5 tested source commit "
            "is invalid or is not an ancestor of the selected HEAD "
            f"({tested_source_commit} !<= {source_commit})."
        )
    evidence_diff = _git(
        repo,
        "diff",
        "--name-only",
        f"{tested_source_commit}..{source_commit}",
    ).stdout.decode("utf-8").splitlines()
    if evidence_diff != [AUDIT_REPORT_NAME]:
        raise BundleBuildError(
            "Refusing to build dispatch bundles: after the tested source "
            f"commit, HEAD may change only {AUDIT_REPORT_NAME}. "
            f"Found commit-range paths: {evidence_diff!r}"
        )

    authorization_manifest = _resolve_dispatch_authorization_manifest(
        dispatch_authorization_manifest
    )
    _verify_dispatch_authorization(
        repo=repo,
        manifest_path=authorization_manifest,
        tested_source_commit=tested_source_commit,
        dispatch_source_commit=source_commit,
        expected_manifest_sha256=(
            authorized.dispatch_authorization_manifest_sha256
        ),
    )

    return BundlePlan(
        repo=repo,
        source_commit=source_commit,
        tested_source_commit=tested_source_commit,
        dispatch_authorization_manifest_sha256=(
            authorized.dispatch_authorization_manifest_sha256
        ),
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


def _write_paper1_bundle(
    output: Path,
    plan: BundlePlan,
    bundle_kind: str,
    target: str,
    purpose: str,
) -> None:
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name in plan.names:
            data = plan.blobs[name]
            if bundle_kind == "generation" and name == A00:
                data = set_a00_bundle_config(
                    data.decode("utf-8"), target
                ).encode("utf-8")
            archive.writestr(_zip_info(name), data)
        if bundle_kind == "generation":
            manifest_name = "generation_bundle_manifest.json"
            manifest = generation_manifest(target)
        elif bundle_kind == "training":
            manifest_name = "training_job_manifest.json"
            manifest = training_manifests()[target]
        else:
            raise BundleBuildError(f"unknown bundle kind {bundle_kind!r}")
        archive.writestr(
            _zip_info(manifest_name), canonical_json_bytes(manifest)
        )
        archive.writestr(
            _zip_info("README_BUNDLE.md"),
            paper1_readme(
                bundle_kind,
                target,
                purpose,
                plan.source_commit,
                plan.tested_source_commit,
                plan.dispatch_authorization_manifest_sha256,
            ).encode("utf-8"),
        )


def build_bundles(
    repo: str | os.PathLike[str] = REPO,
    dispatch_authorization_manifest: (
        str | os.PathLike[str] | None
    ) = None,
) -> BundleBuildResult:
    """Build and atomically publish a commit-bound complete bundle set."""
    # Crucially, every authorization/source gate completes before even a
    # temporary ZIP is created. A blocked invocation has zero bundle/manifest
    # side effects.
    plan = prepare_bundle_plan(repo, dispatch_authorization_manifest)
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
            "# dispatch_authorization_manifest_sha256 "
            f"{plan.dispatch_authorization_manifest_sha256}",
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
        built_metadata, sha_lines[3:]
    ):
        print(f"{bundle_name:30} {kb:5}  {adds}")
        print(f"  sha256 {sha_line.split()[0]}")
    print(
        f"\n{len(bundle_paths)} bundles x {len(plan.names) + 2} files, "
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
        "--dispatch-authorization-manifest",
        help=(
            "absolute external create-once evidence manifest; alternatively "
            f"set {DISPATCH_AUTHORIZATION_ENV}"
        ),
    )
    args = parser.parse_args(argv)
    try:
        build_bundles(
            dispatch_authorization_manifest=(
                args.dispatch_authorization_manifest
            )
        )
    except BundleBuildError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    main()
