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
EXPECTED_AUDIT_HEADING = "# Audit R11 results (legacy filename)"


class BundleBuildError(RuntimeError):
    """Fail-closed bundle precondition or publication error."""


# THE LADDER (2026-07-14). Everything is regenerated from scratch (raw/Option-B
# format + the new EOV design), so every stage is a GEN stage and every stage
# ablates with `all_mult` = channel-symmetric Gaussian multiplicative noise
# (pointwise sigma = 5% of |signal|) on EVERY channel,
# injected at LOAD time onto noise-free data. The controlled L60 contrast graph
# is s0->s11->s13 and s0->s12->s13->s14->s15->s16. The L99.6 stages are
# blockwise scale/stress contrasts rather than a one-factor sequence.
STAGES = MappingProxyType({
    # id                (noise mode, one-line what-it-adds)
    "s0_scour":       ("all_mult", "scour only - baseline + architecture selection"),
    "s11_bear":       ("all_mult", "+ bearing (HEAD)"),
    "s12_crack":      ("all_mult", "+ crack (nuisance, no bearing)"),
    "s13_bearcrack":  ("all_mult", "+ bearing + crack = three modeled bridge-condition mechanisms (with scour)"),
    "s14_prof":       ("all_mult", "+ rail profile FRA-4 per-state = the ROUGHNESS rung"),
    "s15_track":      ("all_mult", "+ track-layer damage (ballast/hanging sleepers/pads)"),
    "s16_all":        ("all_mult", "+ wheel OOR (polygonization; flats disabled) = all modeled vertical-pathway EOVs"),
    "s21_scour4":     ("all_mult", "L99.6 scale block: 4-span, scour only"),
    "s22_bearcrack4": ("all_mult", "L99.6 bridge-condition stress block: + bearing + crack (three modeled mechanisms with scour)"),
    "s23_all4":       ("all_mult", "L99.6 EOV stress block: all modeled vertical-pathway EOVs"),
})
EXPECTED_STAGE_ORDER = (
    "s0_scour",
    "s11_bear",
    "s12_crack",
    "s13_bearcrack",
    "s14_prof",
    "s15_track",
    "s16_all",
    "s21_scour4",
    "s22_bearcrack4",
    "s23_all4",
)
if tuple(STAGES) != EXPECTED_STAGE_ORDER:
    raise RuntimeError("registered bundle stage order is incomplete or drifted")


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

def readme(
    stage,
    mode,
    adds,
    source_commit,
    tested_source_commit,
):
    return "\n".join([
        f"# R11 bundle: {stage} — {adds}", "",
        f"Dispatch bundle commit B: `{source_commit}`.",
        f"Tested source commit A: `{tested_source_commit}`.", "",
        "This bundle was publishable only after the commit-bound audit in",
        "`docs/audit_r5_results.md` authorized that exact tested source and the",
        "report-only A→B lineage. That included report is the sole dispatch",
        "authority; its benchmark and host-qualification gates are approved.",
        "Verify this ZIP",
        "against `bundle_sha256.txt`, then extract it into a FRESH workspace.",
        "Never overlay or resume a pre-R11 data/results folder.", "",
        "## 1. FAST DISPATCH PREFLIGHTS",
        "Use the exact registered MATLAB R2025b Update 5 stack and Python 3.13.3",
        "environment. Run MATLAB `smoke_audit`, `smoke_crn_state_design`,",
        "`smoke_damage_toggles`, `smoke_stage3`, `smoke_geometry`,",
        "`smoke_r11_provenance_serialization`, and `smoke_contact_closure`.",
        "Run the no-argument Python preflights:",
        "`check_split_grouping.py`, `check_loader_provenance.py`,",
        "`check_cache_provenance.py`, `check_protocol_hash.py`,",
        "`check_generation_contract.py`, `check_benchmark_contract.py`,",
        "`check_source_provenance.py`,",
        "`check_paa.py`, `check_weighted_head_mse.py`,",
        "`check_sensor_noise_pairing.py`,",
        "`check_statistical_inference.py`, `check_artifact_provenance.py`,",
        "`check_environment_lock.py`,",
        "`check_hyperparameter_policy.py`, `check_capacity_preflight.py`,",
        "`check_execution_blocking.py`, and `check_cross_rung_inference.py`.",
        "Every listed preflight must be green on the intended host. A skip is",
        "not a pass.", "",
        "### AUDIT-ONLY (commit-A evidence; do not repeat per dispatch host)",
        "`check_campaign_controls.py`,",
        "`check_generation_release_comparison.py`,",
        "`check_r4_mutation_guards.py`,",
        "`check_training_policy_mutation_guards.py`, and",
        "`benchmark_r5_compute.py` are one-time commit-A audit gates.",
        "The included commit-B audit report has already approved the genuine",
        "eight-channel R11 benchmark against the tested commit A: one current-shape",
        "100-trial anchor study with zero Optuna FAIL/OOM and no Optuna-trial",
        "retry/replacement, plus one clean shared finalist-CV refit with",
        "`attempt_count=1`, `prior_unaccepted_attempt_count=0`,",
        "`timing_complete=true`, and `memory_complete=true`.",
        "Never reuse benchmark fixtures or studies for scientific selection.", "",
        "### ONE-TIME CROSS-LANGUAGE CONTRACT AUDIT",
        "For each authorized MATLAB environment, run MATLAB",
        "`smoke_b54_overlap_parity` and `smoke_familytable`, then run Python",
        "`check_b54_overlap_parity.py` and `check_familytable_roundtrip.py` against",
        "those fresh outputs. A missing genuine-MATLAB artifact is a hard failure;",
        "a Python-only reconstruction or skip is not dispatch evidence.",
        "Raw parity is separate and MANDATORY after the first R11 state: once",
        "`0001.mat` is complete, run MATLAB `smoke_raw_parity('Results/<case>')`",
        "and then `python check_raw_parity.py \"scour_MATLAB/Results/<case>\"`.",
        "Stop generation immediately unless both report PARITY PASS.", "",
        "Verify that this physical MATLAB PC is represented in the commit-B",
        "qualification evidence. Before authorization, every intended generation",
        "PC must have used one stable unique `TTBI_QUALIFICATION_HOST_ID` and",
        "fresh transient scripts generated from commit A for at least",
        "`s0_scour`, `s16_all`, and `s23_all4`. Each host must execute the exact",
        "source-bound script bytes; never copy or forge a host receipt.",
        "Each run must retain its authenticated `qualification_host_receipt.json`,",
        "and the required corresponding host/environment pairs must have accepted",
        "`matlab-environment-qualification-receipt-v4` comparator receipts.",
        "CPU equality is not required; equal MATLAB-environment digests are",
        "eligible only with distinct authenticated host IDs. Host identity is not",
        "part of production `gen_schema` or `gen_fingerprint`. If this PC is not",
        "covered by that evidence, do not generate; return to the audit gate.", "",
        "## 2. Generate noise-free raw data",
        f"Open `scour_MATLAB/A00_Run.m` and run it; `STAGE` is preset to `{stage}`.",
        "The generator writes raw un-interpolated time histories plus crop metadata,",
        "50 passages/state, semantic StateUIDs, named CRN streams, contact logs,",
        "the exact MATLAB stack, and reviewed source/asset roots.",
        "The modeled input comprises five vertical-acceleration and three pitch-",
        "angular-velocity response channels; these are not eight accelerometers.",
        "Every L60 rung has 450 fixed-family states; every L99.6 rung has 475.",
        "Qualification output is micro-only and the campaign loader rejects it.",
        "A00 caps the process pool at four workers; do not alter the cap or any",
        "fingerprinted campaign value.",
        "Output -> `scour_MATLAB/Results/<stage>_L<len>_st<N>/`; move the complete",
        "authenticated folder under `data/`.", "",
        "A00 does not pause and parfor may finish states out of order. Wait for",
        "the complete `0001.mat`, then run MATLAB",
        "`smoke_raw_parity('Results/<case>')`, then",
        "`python check_raw_parity.py \"scour_MATLAB/Results/<case>\"`.",
        "The MATLAB smoke writes `matlab_ref_parity.mat`, which Python consumes,",
        "so run the two checks sequentially while A00 continues. Stop A00",
        "immediately if either fails; do not admit later output until both report",
        "PARITY PASS.", "",
        "## 3. Configure the physical execution block",
        "Before an anchor starts, set all three path variables to durable ABSOLUTE",
        "paths outside this disposable workspace:",
        "  `TTBI_EXECUTION_RECEIPT_DIR` — block receipt/capacity directory;",
        "  `TTBI_HYPERPARAMETER_MANIFEST` — block full-array HPO manifest;",
        "  `CHAMPION_MANIFEST` — block reference architecture/pair manifest.",
        "After the anchor finalizes the block reference, it prints the canonical",
        "`TTBI_BLOCK_REFERENCE_SHA256=<64-lowercase-hex>` value. For every",
        "follower, export `TTBI_BLOCK_REFERENCE_SHA256` with that exact value in",
        "addition to the same three absolute paths. This fourth variable is",
        "mandatory for followers; never infer, truncate, copy across blocks, or",
        "manually recompute it.",
        "L60 is anchored independently at `s0_scour`; L99 is anchored independently",
        "at `s21_scour4`. Complete the block anchor before its followers. Never",
        "copy the L60 champion, HPO manifest, or receipt into the L99 block.",
        "Relative paths, symlinks, cross-block artifacts, and mismatched runtime or",
        "protocol hashes fail closed.",
        "Run all seven L60 Python ablations on one exact physical host/GPU/runtime",
        "and all three L99 ablations on one exact host/GPU/runtime (which may differ",
        "from L60). Do not distribute one Python block across machines.", "",
        "Before any study, the GPU must pass the durable capacity preflight for",
        "the largest registered point of all four architectures. Its threshold is",
        "total VRAM minus this process's peak reserved memory, not system-wide free",
        "VRAM, so use an otherwise idle/exclusive GPU. OOM and every Optuna FAIL",
        "state are fatal; Optuna-trial retry and replacement are forbidden.", "",
        "## 4. Run the Python ablation",
        "`python comprehensive_ablation_multidamage.py`",
        "(STAGE and SENSOR_NOISE are preset in this bundle).",
        "At each block anchor, free HPO runs ONLY on the full eight-DOF input:",
        "4 architectures x 3 seeds x 100 trials, with the registered pruner.",
        "The anchor calibration itself is its non-selectable full-array control;",
        "it is not duplicated as a singleton. Every response-channel candidate, non-anchor",
        "full-array control, and follower runs one exact singleton Optuna trial",
        "from the authenticated per-architecture/per-seed anchor parameters,",
        "with no pruner.",
        "Each anchor also runs 8 single channels x 4 architectures x 3 seeds as",
        "a frozen diagnostic; it cannot select the block reference, which comes",
        "from the complete 4-architecture x 28-two-channel matrix.",
        "`s16_all` and `s23_all4` additionally reopen 4 architectures x 28 pairs",
        "x 3 seeds as exploratory deployment selections. Those candidates remain",
        "frozen singletons; their separate winner never overwrites a block reference.",
        "The complete ten-rung design has 1,620–1,638 studies (designed/carried",
        "pair deduplication at six frozen rungs determines the range) but only",
        "3,996–4,014 actual trials. Selection uses inner validation; diagnostic",
        "5-fold x 2-repeat finalist CV runs only at s0/s16/s21/s23 and uses",
        "development states only; the immutable outer test is opened only after",
        "winner/comparator freeze. Registered within-rung MSE intervals use 2,000",
        "state-first draws (seed 42). Localisation is a passage-level point estimate",
        "for max true scour >5 percentage points, with no registered state-level CI.",
        f"Summary -> `results/{stage}_summary_ph-<hash>/`.", "",
        "## 5. Observation-model and claim boundary",
        f"`{mode}` adds pointwise zero-mean Gaussian multiplicative noise with",
        "sigma = 5% of |signal| to every selected channel at load time, keyed by",
        "global DOF. This is a symmetric relative-noise stress—not a datasheet",
        "model or sensor-hardware comparison. Claims are limited to modeled",
        "response-channel/DOF subsets under this registered policy. Physical sensor",
        "count, packaging, mounting, and hardware placement are not identified.",
        "Scour labels are support-stiffness loss, not physical scour depth. Crack",
        "is a uniform damaged-element EI block, not Sinha. Hanging sleepers use",
        "linear support removal without gap/void-depth nonlinearity. Each rung",
        "re-trains its model, so cross-rung differences are achievable performance",
        "under retraining, not zero-shot OOD robustness.", "",
        "## 6. L60 cross-rung analysis",
        "After all seven L60 summaries exist, run",
        "`analyze_cross_rung_contrasts.py` with every exact summary plus explicit",
        "canonical external L60 champion, HPO-manifest, and execution-receipt",
        "paths. Its primary population is the paired outer-test joint StateUIDs.",
        "It uses 100,000 state-first paired bootstrap draws and reports pointwise",
        "95% plus Bonferroni familywise intervals over exactly seven edges.",
        "Those bands are finite-design empirical-state uncertainty conditional on",
        "the registered anchors/LHS, not field-population coverage.",
        "L99.6 remains a separate blockwise scale/stress analysis.", "",
        "See `README_CAMPAIGN.md` and `docs/paper1_methodology.md` for the complete",
        "registered interpretation and publication limits.", ""])

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


def _parse_authorized_report(audit_report: str) -> str:
    """Parse a fixed early-document authorization header, never free prose."""
    lines = audit_report.splitlines()
    if (
        len(lines) < 5
        or lines[0] != EXPECTED_AUDIT_HEADING
        or lines[1] != ""
        or lines[2] != EXPECTED_AUDIT_STATUS
        or lines[3] != ""
    ):
        raise BundleBuildError(
            "Refusing to build dispatch bundles: the legacy-named R11 report "
            f"must begin with {EXPECTED_AUDIT_HEADING!r} and place "
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
            "Refusing to build dispatch bundles: the legacy-named R11 header "
            "must contain one unique status and one unique tested-source line "
            "with a 40-character lowercase Git SHA."
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
                stage,
                mode,
                adds,
                plan.source_commit,
                plan.tested_source_commit,
            ).encode("utf-8"),
        )


def build_bundles(
    repo: str | os.PathLike[str] = REPO,
) -> BundleBuildResult:
    """Build and atomically publish a commit-bound complete bundle set."""
    # Crucially, every authorization/source gate completes before even a
    # temporary ZIP is created. A blocked invocation has zero bundle/manifest
    # side effects.
    plan = prepare_bundle_plan(repo)
    repo = plan.repo
    stage_items = tuple(STAGES.items())
    if (
        len(stage_items) != len(EXPECTED_STAGE_ORDER)
        or tuple(stage for stage, _ in stage_items) != EXPECTED_STAGE_ORDER
    ):
        raise BundleBuildError(
            "Refusing partial bundle publication: the registered ten-stage "
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
