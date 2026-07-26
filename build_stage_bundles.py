"""
build_stage_bundles.py  (2026-07-14)
Produce ONE self-contained bundle per stage: MATLAB generator (A00 STAGE preset)
+ Python ablation (driver STAGE preset + noise mode) + all core code + a per-stage
README. Each bundle extracts into the repo root on a Lab PC and runs with NO editing.

The exact source set comes from the tracked ``bundle_source_files.txt`` manifest.
It is therefore part of the reviewed Git commit rather than being inherited from
an untracked historical ZIP.
"""
from __future__ import annotations

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
EXPECTED_AUDIT_STATUS = "**Status: DISPATCH AUTHORIZED.**"


class BundleBuildError(RuntimeError):
    """Fail-closed bundle precondition or publication error."""


# THE LADDER (2026-07-14). Everything is regenerated from scratch (raw/Option-B
# format + the new EOV design), so every stage is a GEN stage and every stage
# ablates with `all_mult` = channel-symmetric Gaussian multiplicative noise
# (pointwise sigma = 5% of |signal|) on EVERY channel,
# injected at LOAD time onto noise-free data. One factor per rung.
STAGES = {
    # id                (noise mode, one-line what-it-adds)
    "s0_scour":       ("all_mult", "scour only - baseline + architecture selection"),
    "s11_bear":       ("all_mult", "+ bearing (HEAD)"),
    "s12_crack":      ("all_mult", "+ crack (nuisance, no bearing)"),
    "s13_bearcrack":  ("all_mult", "+ bearing + crack = all BRIDGE damages"),
    "s14_prof":       ("all_mult", "+ rail profile FRA-4 per-state = the ROUGHNESS rung"),
    "s15_track":      ("all_mult", "+ track-layer damage (ballast/hanging sleepers/pads)"),
    "s16_all":        ("all_mult", "+ wheel OOR (polygonization; flats disabled) = ALL damages"),
    "s21_scour4":     ("all_mult", "4-span L99.6, scour only"),
    "s22_bearcrack4": ("all_mult", "4-span, + bearing + crack = all BRIDGE damages"),
    "s23_all4":       ("all_mult", "4-span, all damages"),
}
def set_a00_stage(t, stage):
    t2, n = re.subn(r"^STAGE = '[^']*';", f"STAGE = '{stage}';", t, count=1, flags=re.M)
    if n != 1:
        raise BundleBuildError("A00 STAGE line not found/replaced exactly once.")
    return t2

def set_driver_stage(t, stage):
    t2, n = re.subn(r'^(STAGE = )"[^"]*"', rf'\1"{stage}"', t, count=1, flags=re.M)
    if n != 1:
        raise BundleBuildError(
            "Driver STAGE line not found/replaced exactly once."
        )
    return t2

def set_driver_noise(t, mode):
    t2, n = re.subn(
        r"^SENSOR_NOISE = None$",
        f'SENSOR_NOISE = {{"mode": "{mode}", "desvio": 0.05}}',
        t,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise BundleBuildError(
            "Driver SENSOR_NOISE line not found/replaced exactly once."
        )
    return t2

def readme(stage, mode, adds, source_commit):
    return "\n".join([
        f"# Bundle: {stage}  —  {adds}", "",
        f"Reviewed source commit: `{source_commit}`.", "",
        "Self-contained: MATLAB generator + Python ablation, both PRESET for this",
        "stage. Extract into the repo root on the Lab PC. No editing needed.", "",
        "## 0. AUDIT R5 — read first",
        "- Extract into a FRESH working copy; do NOT extract over an old run. A00",
        "  writes a gen_schema and ABORTS if you resume a folder from other code.",
        "- FAST DISPATCH PREFLIGHTS (run on every campaign PC before generation):",
        "  MATLAB: `smoke_audit`, `smoke_stage3`, `smoke_geometry`;",
        "  Python: `python check_split_grouping.py`, `python check_loader_provenance.py`,",
        "  `python check_cache_provenance.py`, `python check_protocol_hash.py`,",
        "  `python check_generation_contract.py`, `python check_benchmark_contract.py`,",
        "  `python check_paa.py`, `python check_weighted_head_mse.py`,",
        "  `python check_sensor_noise_pairing.py`, `python check_campaign_controls.py`,",
        "  `python check_statistical_inference.py`, `python check_artifact_provenance.py`,",
        "  `python check_environment_lock.py`, `python check_b54_overlap_parity.py`.",
        "  Also run MATLAB `smoke_contact_closure` before the closure study.",
        "  All must print ALL PASS. MATLAB must be available to the normal B54 parity",
        "  command; `--allow-python-only` is not a valid dispatch preflight.",
        "- AUDIT-ONLY (already run once on the reviewed source commit):",
        "  `python check_r4_mutation_guards.py`,",
        "  `python check_training_policy_mutation_guards.py`, and the heavy",
        "  `python benchmark_r5_compute.py`. They ship for review/reproducibility,",
        "  but are not per-PC preflights. The benchmark needs its immutable legacy",
        "  fixture and writes only below `.audit_tmp`; never use it for selection.",
        "  The commit-bound evidence and dispatch verdict are in",
        "  `docs/audit_r5_results.md`.",
        "- ONE-TIME CROSS-LANGUAGE CONTRACT AUDIT (reviewed source, pre-dispatch):",
        "  run MATLAB `smoke_familytable`, then from the repo root run Python",
        "  `python check_familytable_roundtrip.py`. The normal Python command must",
        "  pass; a missing genuine-MATLAB artifact is a hard failure, not a skip.",
        "- Run STAGE s0_scour FIRST: it selects the architecture and writes",
        "  results/_champion_arch_<schema>_ph-<hash>.json; every later rung reads it",
        "  and errors if it is missing (no hardcoded champion anymore). On a DIFFERENT",
        "  PC, copy that complete JSON and point CHAMPION_MANIFEST at it. Bare",
        "  architecture/pair overrides are rejected because they lack selection lineage.",
        "- PROTOCOL HASH (2026-07-19): every study/summary/manifest name carries a",
        "  SHA-256 of the full protocol (dataset fingerprint, split, seeds, trials,",
        "  pruner, search space, noise, targets). If ANY of those change, names",
        "  change and old studies are orphaned — never resumed. The exact hashed",
        "  descriptor is written to the summary dir as protocol_descriptor.json.",
        "- Study/DB/cache dirs are STAGE-prefixed, so rungs never cross-contaminate.", "",
        "## 1. MATLAB — generate the data (noise-free, RAW format)",
        f"Open `scour_MATLAB/A00_Run.m` and RUN it. `STAGE` is already `{stage}` and",
        "`use_signal_noise = false` — measurement noise is added later, at LOAD time.",
        "D01 saves the RAW, un-interpolated TIME-domain signal plus the space/crop",
        "parameters; Python rebuilds the space window at load time (Option B), so the",
        "noise model can change forever without regenerating.",
        "Output -> `scour_MATLAB/Results/<stage>_L<len>_st<N>/`; move it under `data/`.",
        "Folder names are SHORT now (Windows MAX_PATH); the full descriptor lives in",
        "`case_info.case_desc` / `case_info.txt`.", "",
        "## 2. MANDATORY after the first R9 state: verify the raw-data transform",
        "After the FIRST state exists, in MATLAB:  `smoke_raw_parity('Results/<case>')`",
        "then:  `python check_raw_parity.py \"scour_MATLAB/Results/<case>\"`",
        "Must print PARITY PASS (max|MATLAB-Python| < 1e-12) before the long run.", "",
        "## 3. Python — ablate",
        "`python comprehensive_ablation_multidamage.py` (STAGE + SENSOR_NOISE preset).",
        f"Noise: `{mode}` = zero-mean Gaussian multiplicative noise on EVERY channel",
        "with pointwise sigma = 5% of |signal|, injected at load time onto the",
        "noise-free data. 100 useful Optuna trials x 3 seeds per configuration;",
        "pair search, controls, finalist CV and the outer-test firewall are fixed",
        "by the hash-carried protocol described in README_CAMPAIGN.md.",
        f"summary -> `results/{stage}_summary_ph-<hash>/` (+ leaderboards +",
        "`protocol_descriptor.json`).", "",
        "## Heads vs nuisances",
        "HEADS = scour (per pier) + bearing (per abutment) ONLY. Crack, rail profile,",
        "track-layer and wheel damage are NUISANCES: randomized, logged, never",
        "estimated — the network must be INVARIANT to them.", "",
        "## Requirements",
        "Use Python 3.13.3 and install `requirements-campaign-py313-cu128.txt`.",
        "The driver hard-fails before creating a study if the exact hash-carried",
        "software/CUDA lock does not match. Everything is resumable.", ""])

class TreeEntry(NamedTuple):
    mode: str
    object_type: str
    oid: str


class BundlePlan(NamedTuple):
    repo: Path
    source_commit: str
    tested_source_commit: str
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
        if not line.strip() or line.lstrip().startswith("#"):
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


def _parse_authorized_report(audit_report: str) -> str:
    """Parse a fixed early-document authorization header, never free prose."""
    lines = audit_report.splitlines()
    if (
        len(lines) < 5
        or not lines[0].startswith("# Audit R5 results")
        or lines[1] != ""
        or lines[2] != EXPECTED_AUDIT_STATUS
        or lines[3] != ""
    ):
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the R5 report must place "
            f"{EXPECTED_AUDIT_STATUS!r} exactly on line 3 of its document "
            "header (not in prose or a code fence)."
        )
    tested_match = re.fullmatch(
        r"\*\*Tested source commit:\*\* `([0-9a-f]{40})`",
        lines[4],
    )
    status_lines = [
        line for line in lines if line.startswith("**Status:")
    ]
    tested_lines = [
        line for line in lines if line.startswith("**Tested source commit:**")
    ]
    if (
        tested_match is None
        or status_lines != [EXPECTED_AUDIT_STATUS]
        or tested_lines != [lines[4]]
    ):
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the R5 header must contain "
            "one unique status and one unique tested-source line with a "
            "40-character lowercase Git SHA."
        )
    return tested_match.group(1)


def prepare_bundle_plan(repo: str | os.PathLike[str] = REPO) -> BundlePlan:
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
    tested_source_commit = _parse_authorized_report(audit_report)

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

    return BundlePlan(
        repo=repo,
        source_commit=source_commit,
        tested_source_commit=tested_source_commit,
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


def _write_stage_zip(
    output: Path,
    plan: BundlePlan,
    stage: str,
    mode: str,
    adds: str,
) -> None:
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name in plan.names:
            data = plan.blobs[name]
            if name == DRIVER:
                driver = set_driver_stage(data.decode("utf-8"), stage)
                data = set_driver_noise(driver, mode).encode("utf-8")
            elif name == A00:
                data = set_a00_stage(
                    data.decode("utf-8"), stage
                ).encode("utf-8")
            archive.writestr(_zip_info(name), data)
        archive.writestr(
            _zip_info("README_BUNDLE.md"),
            readme(
                stage, mode, adds, plan.source_commit
            ).encode("utf-8"),
        )


def build_bundles(
    repo: str | os.PathLike[str] = REPO,
    *,
    stages: Mapping[str, tuple[str, str]] = STAGES,
) -> BundleBuildResult:
    """Build and atomically publish a commit-bound complete bundle set."""
    # Crucially, every authorization/source gate completes before even a
    # temporary ZIP is created. A blocked invocation has zero bundle/manifest
    # side effects.
    plan = prepare_bundle_plan(repo)
    repo = plan.repo
    stage_items = tuple(stages.items())
    if not stage_items:
        raise BundleBuildError("At least one stage is required.")

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
            f"# complete_bundle_count {len(stage_items)}",
        ]
        staged_bundles: list[tuple[Path, Path]] = []
        for stage, (mode, adds) in stage_items:
            bundle_name = f"bundle_{stage}.zip"
            staged = staging / bundle_name
            _write_stage_zip(staged, plan, stage, mode, adds)
            digest = hashlib.sha256(staged.read_bytes()).hexdigest()
            sha_lines.append(f"{digest}  {bundle_name}")
            target = repo / bundle_name
            staged_bundles.append((staged, target))
            built_metadata.append(
                (bundle_name, staged.stat().st_size // 1024, adds)
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
        f"\n{len(bundle_paths)} bundles x {len(plan.names) + 1} files, "
        "contents resolved from regular tracked HEAD blobs. "
        "SHA-256 manifest -> bundle_sha256.txt"
    )
    return BundleBuildResult(
        source_commit=plan.source_commit,
        bundles=tuple(bundle_paths),
        sha_manifest=sha_manifest,
    )


def main() -> int:
    try:
        build_bundles()
    except BundleBuildError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    main()
