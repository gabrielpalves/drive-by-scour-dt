"""Deterministic mutation audit for the R4 statistical/provenance guards.

This is an auditor check, not a campaign preflight.  It copies the required
Python source into an isolated temporary tree, verifies each checker is GREEN,
then injects exactly one defect at a time and requires the intended checker to
turn RED for the expected reason.

Every mutated file is restored from its original bytes in ``finally``.  A
whole-temporary-tree SHA-256 map must match before/after every mutation, and
the real repository targets are also asserted byte-identical at exit.  Run
serially (not while another process is editing these source files):

    python check_r4_mutation_guards.py
    python check_r4_mutation_guards.py --only statistical
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Mutation:
    name: str
    group: str
    target: str
    checker: str
    original: str
    mutant: str
    evidence: str
    occurrence: int = 1


MUTATIONS = (
    Mutation(
        name="grouped-CV validation states leak into training",
        group="statistical",
        target="core/statistical_inference.py",
        checker="check_statistical_inference.py",
        original=(
            "train_states = np.setdiff1d("
            "dev_states, val_states, assume_unique=True)"
        ),
        mutant=(
            "train_states = dev_states.copy()  "
            "# MUTANT: validation states leak into training"
        ),
        evidence="group leakage in repeated CV construction",
    ),
    Mutation(
        name="grouped-CV cumulative balancing is replaced by fixed offset",
        group="statistical",
        target="core/statistical_inference.py",
        checker="check_statistical_inference.py",
        original="offset = min(candidates)[-1]",
        mutant="offset = 0  # MUTANT: every stratum starts in fold zero",
        evidence=(
            "[FAIL] cumulative fold-size balancing is non-trivial "
            "and within one state"
        ),
    ),
    Mutation(
        name="seed median is moved before state aggregation",
        group="statistical",
        target="core/statistical_inference.py",
        checker="check_statistical_inference.py",
        original=(
            "return float(np.median(per_repeat_seed, axis=1).mean())"
        ),
        mutant=(
            "return float(np.median(arr, axis=2).mean())  "
            "# MUTANT: median within states"
        ),
        evidence=(
            "[FAIL] seed median is applied after state aggregation "
            "inside the statistic"
        ),
        # The same statistic text occurs later in paired_state_contrast.
        # Occurrence 1 is specifically hierarchical_state_seed_bootstrap.
        occurrence=1,
    ),
    Mutation(
        name="matched-block inference accepts an incomplete StateUID x seed grid",
        group="statistical",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "        if set(left_cells) != expected_cells or "
            "set(right_cells) != expected_cells:"
        ),
        mutant=(
            "        if False:  "
            "# MUTANT: exact paired StateUID x seed grid ignored"
        ),
        evidence=(
            "[FAIL] missing L99 StateUID x seed cell is rejected"
        ),
    ),
    Mutation(
        name="standalone artifact verifier accepts missing provenance fields",
        group="artifact",
        target="core/artifact_provenance.py",
        checker="check_artifact_provenance.py",
        original="if missing:",
        mutant="if False:  # MUTANT: required-field guard disabled",
        evidence=(
            "[FAIL] missing required selection lineage field is rejected"
        ),
    ),
    Mutation(
        name="standalone artifact verifier skips champion SHA-256",
        group="artifact",
        target="core/artifact_provenance.py",
        checker="check_artifact_provenance.py",
        original=(
            'if _sha256_file(model_path) != '
            'metadata["champion_weights_sha256"]:'
        ),
        mutant="if False:  # MUTANT: champion hash guard disabled",
        evidence="[FAIL] one-byte champion tamper is rejected",
    ),
    Mutation(
        name="standalone artifact verifier skips protocol self-hash",
        group="artifact",
        target="core/artifact_provenance.py",
        checker="check_artifact_provenance.py",
        original=(
            'if protocol_hash(metadata["protocol_descriptor"]) '
            '!= metadata["protocol_hash"]:'
        ),
        mutant="if False:  # MUTANT: descriptor/hash guard disabled",
        evidence=(
            "[FAIL] protocol descriptor/hash disagreement is rejected"
        ),
    ),
    Mutation(
        name="study-linked artifact verifier suppresses provenance mismatches",
        group="artifact",
        target="training/pipeline.py",
        checker="check_execution_blocking.py",
        original=(
            "if mismatches:\n"
            "        raise RuntimeError(\n"
            "            f\"{config.get('name')}: champion package provenance "
            "mismatch: \"\n"
            "            f\"{mismatches}\")"
        ),
        mutant=(
            "if False:  # MUTANT: accumulated provenance mismatches ignored\n"
            "        raise RuntimeError(\n"
            "            f\"{config.get('name')}: champion package provenance "
            "mismatch: \"\n"
            "            f\"{mismatches}\")"
        ),
        evidence=(
            "[FAIL] study-linked verifier guards accumulated provenance "
            "mismatches"
        ),
    ),
    Mutation(
        name="standalone artifact verifier ignores the external block-reference pin",
        group="artifact",
        target="core/artifact_provenance.py",
        checker="check_artifact_provenance.py",
        original=(
            "if (\n"
            "        expected_block_reference_manifest_sha256\n"
            "        is not _EXPECTED_BLOCK_REFERENCE_UNSET\n"
            "    ):"
        ),
        mutant=(
            "if False:  # MUTANT: external block-reference expectation ignored"
        ),
        evidence=(
            "[FAIL] independent non-null block-reference substitution is "
            "rejected"
        ),
    ),
    Mutation(
        name="matched-block analyzer ignores exact generated StateUID inventory",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "        or inventory != list(expected_inventory)\n"
        ),
        mutant=(
            "        or False  # MUTANT: generated StateUID inventory ignored\n"
        ),
        evidence=(
            "[FAIL] missing generated F40 UID is rejected"
        ),
    ),
    Mutation(
        name="matched-block analyzer ignores endpoint pipeline binding",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "            left[\"pipeline_slot\"] != right[\"pipeline_slot\"]\n"
        ),
        mutant=(
            "            False  # MUTANT: endpoint pipeline binding ignored\n"
        ),
        evidence=(
            "[FAIL] different endpoint pipeline is rejected"
        ),
    ),
    Mutation(
        name="matched-block analyzer ignores endpoint partition alignment",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "            if left[\"partition_by_uid\"][uid] != "
            "right[\"partition_by_uid\"][uid]:"
        ),
        mutant=(
            "            if False:  "
            "# MUTANT: endpoint partition alignment ignored"
        ),
        evidence=(
            "[FAIL] matched F40 partition drift is rejected"
        ),
    ),
    Mutation(
        name="dispatch authorization bypasses the benchmark revalidator",
        group="artifact",
        target="dispatch_authorization.py",
        checker="check_dispatch_authorization.py",
        original="        evidence = benchmark.verify_completed_receipt(",
        mutant="        evidence = benchmark.trust_completed_receipt(",
        evidence=(
            "[FAIL] production gate invokes all three authoritative "
            "revalidators"
        ),
    ),
    Mutation(
        name="manifest entrypoint bypasses the exact phase executor",
        group="artifact",
        target="comprehensive_ablation_multidamage.py",
        checker="check_campaign_controls.py",
        original=(
            "    return execute_manifest_job(job, manifest)"
        ),
        mutant=(
            "    return manifest  # MUTANT: exact phase executor bypassed"
        ),
        evidence=(
            "[FAIL] post-freeze phase refuses to open data without an "
            "authenticated freeze"
        ),
    ),
    Mutation(
        name="dispatch JSON parser silently accepts duplicate evidence keys",
        group="artifact",
        target="dispatch_manifest.py",
        checker="check_dispatch_authorization.py",
        original="            object_pairs_hook=unique_pairs,",
        mutant=(
            "            object_pairs_hook=dict,  "
            "# MUTANT: duplicate JSON keys collapse silently"
        ),
        evidence=(
            "[FAIL] duplicate JSON key rejected"
        ),
    ),
    Mutation(
        name="environment verifier accepts a missing required package",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original="if missing_required:",
        mutant="if False:  # MUTANT: required-package guard disabled",
        evidence="[FAIL] missing required package hard-fails",
    ),
    Mutation(
        name="environment verifier accepts a conflicting cuBLAS setting",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original="if actual_cublas not in (None, expected_cublas):",
        mutant="if False:  # MUTANT: cuBLAS guard disabled",
        evidence=(
            "[FAIL] cuBLAS deterministic-setting conflict hard-fails"
        ),
    ),
    Mutation(
        name="environment verifier accepts unavailable required CUDA",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original=(
            "if not cuda_available:"
        ),
        mutant="if False:  # MUTANT: required-CUDA guard disabled",
        evidence="[FAIL] required CUDA becoming unavailable hard-fails",
    ),
    Mutation(
        name="environment loader accepts an unsupported lock schema",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original=(
            'if spec["schema"] != _LOCK_SCHEMA:'
        ),
        mutant="if False:  # MUTANT: lock-schema guard disabled",
        evidence="[FAIL] unsupported environment-lock schema hard-fails",
    ),
    Mutation(
        name="environment loader accepts descriptor/SHA disagreement",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original="if expected_matlab_sha != actual_matlab_sha:",
        mutant=(
            "if False:  # MUTANT: MATLAB descriptor authentication disabled"
        ),
        evidence=(
            "[FAIL] descriptor mutation without matching SHA hard-fails"
        ),
    ),
    Mutation(
        name="environment descriptor accepts unauthenticated extra fields",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original=(
            "if not isinstance(environment, dict) or set(environment) != set(\n"
            "        _MATLAB_ENVIRONMENT_FIELDS\n"
            "    ):"
        ),
        mutant=(
            "if not isinstance(environment, dict) or "
            "not set(_MATLAB_ENVIRONMENT_FIELDS).issubset(set(environment)):"
            "  # MUTANT: extra fields ignored"
        ),
        evidence="[FAIL] MATLAB descriptor extra field rejected",
    ),
    Mutation(
        name="qualification inventory ignores exact/tolerant leaf-class drift",
        group="qualification",
        # R11 split: receipt-schema parsing moved out of the inventory module.
        target="qualification_receipt_schema.py",
        checker="check_qualification_receipt_inventory.py",
        original=(
            '            payload["comparison"]["exact_leaves"],\n'
            '            payload["comparison"]["tolerant_leaves"],'
        ),
        mutant=(
            "            0,  # MUTANT: exact/tolerant classes erased\n"
            "            0,"
        ),
        evidence=(
            "schema/graph mutation "
            "'structural-leaf-class-drift-across-pairs' reached the disk "
            "comparator"
        ),
    ),
    Mutation(
        name="qualification inventory accepts a forged numerical worst path",
        group="qualification",
        # R11 split: receipt-schema parsing moved out of the inventory module.
        target="qualification_receipt_schema.py",
        checker="check_qualification_receipt_inventory.py",
        original=(
            "        worst_match = _NUMERICAL_WORST_PATH_RE.fullmatch("
            'value["worst_path"])'
        ),
        mutant=(
            "        worst_match = (\n"
            "            _NUMERICAL_WORST_PATH_RE.fullmatch("
            'value["worst_path"])\n'
            "            or _NUMERICAL_WORST_PATH_RE.fullmatch(\n"
            '                "0001.mat.data.AcelPrimVag[0]"\n'
            "            )  # MUTANT: invalid paths inherit a valid identity\n"
            "        )"
        ),
        evidence=(
            "schema/graph mutation 'forged numerical worst-path prefix' "
            "reached the disk comparator"
        ),
    ),
    Mutation(
        name="fixed/profile FRA corner drifts back to radians per metre",
        group="generation",
        target="scour_MATLAB/A04_Options.m",
        checker="check_profile_pad_contract.py",
        original="Calc.Profile.inputs(3) = 0.8245/(2*pi);",
        mutant="Calc.Profile.inputs(3) = 0.8245; % MUTANT",
        evidence="cycles-per-m corner: expected 1, found 0",
    ),
    Mutation(
        name="pad failures stop drawing once per sleeper lattice point",
        group="generation",
        target="scour_MATLAB/sample_pad_failures.m",
        checker="check_profile_pad_contract.py",
        original="failed = rand(size(lattice)) < failure_probability;",
        mutant="failed = rand() < failure_probability; % MUTANT",
        evidence="one helper draw per lattice point: expected 1, found 0",
    ),
)


BASELINE_EVIDENCE = {
    "check_statistical_inference.py": "STATISTICAL INFERENCE: ALL PASS",
    "check_artifact_provenance.py": "ARTIFACT PROVENANCE: ALL PASS",
    "check_campaign_controls.py": "CAMPAIGN CONTROLS: ALL PASS",
    "check_cross_rung_inference.py": "MATCHED-BLOCK INFERENCE: ALL PASS",
    "check_dispatch_authorization.py":
        "DISPATCH AUTHORIZATION CHECKS: ALL PASS",
    "check_execution_blocking.py": "EXECUTION BLOCKING: ALL PASS",
    "check_environment_lock.py": "ENVIRONMENT COMPATIBILITY: ALL PASS",
    "check_qualification_receipt_inventory.py":
        "QUALIFICATION RECEIPT INVENTORY: ALL CHECKS PASSED",
    "check_profile_pad_contract.py":
        "PROFILE/PAD CONTRACT: ALL CHECKS PASSED",
}

# Full source/environment verifiers deliberately authenticate large byte
# inventories and can exceed the historical three-minute subprocess budget on
# Windows/OneDrive.  Keep ordinary mutation probes bounded tightly, while giving
# the measured heavy checkers enough time to finish instead of reporting a
# harness timeout as a scientific rejection.
DEFAULT_CHECKER_TIMEOUT_SECONDS = 300
CHECKER_TIMEOUT_SECONDS = {
    "check_artifact_provenance.py": 900,
    "check_dispatch_authorization.py": 900,
    "check_environment_lock.py": 900,
    "check_qualification_receipt_inventory.py": 1800,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    """Return code/config needed by the registered isolated checkers."""

    paths = set(REPO.glob("*.py"))
    for package in ("core", "training", "plotting", "digital_twin", "TTBI_2D"):
        root = REPO / package
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    # The R11 provenance root is itself an executable dependency of the
    # artifact checker: it reads the reviewed manifest and hashes every listed
    # Python file plus both environment inputs.  Copy the manifest and the
    # requirements lock explicitly so the isolated baseline exercises the
    # production provenance path instead of failing before any guard is tested.
    paths.update({
        REPO / "bundle_source_files.txt",
        REPO / "environment" / "campaign-py313-cu128.json",
        REPO / "requirements-campaign-py313-cu128.txt",
    })
    # Qualification/contact checkers authenticate the MATLAB generator root,
    # so their isolated baseline must contain every reviewed manifest entry,
    # not merely Python modules. The current manifest is only a few MiB.
    for line in (REPO / "bundle_source_files.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if line and not line.startswith("#"):
            paths.add(REPO.joinpath(*line.split("/")))
    return sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: path.relative_to(REPO).as_posix(),
    )


def _copy_isolated_tree(destination: Path, source_paths: list[Path]) -> None:
    for source in source_paths:
        relative = source.relative_to(REPO)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _tree_fingerprint(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    }


def _replace_occurrence(
    text: str,
    original: str,
    mutant: str,
    occurrence: int,
) -> str:
    if occurrence < 1:
        raise ValueError("mutation occurrence must be one-based")
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.find(original, search_from)
        if start < 0:
            raise RuntimeError(
                f"mutation anchor occurrence {occurrence} not found: "
                f"{original!r}"
            )
        search_from = start + len(original)
    return text[:start] + mutant + text[start + len(original):]


def _validate_anchor_inventory(
    selected: list[Mutation],
    source_paths: list[Path],
) -> None:
    """Fail closed, up front, on every mutation whose anchor has moved.

    Refactors relocate code between modules, and a mutation whose ``original``
    text no longer exists in its declared ``target`` proves nothing: without
    this pass the harness aborts on the FIRST such anchor deep into the run,
    reporting only ``anchor ... not found`` with no indication that the guard
    it was supposed to exercise is now untested.  Reporting all stale anchors
    at once, together with the module that currently holds each one, makes a
    post-split retarget mechanical instead of archaeological.
    """
    stale: list[str] = []
    for mutation in selected:
        target = REPO / mutation.target
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            stale.append(
                f"{mutation.name}: target {mutation.target} is unreadable "
                f"({exc})"
            )
            continue
        if text.count(mutation.original) >= mutation.occurrence:
            continue
        elsewhere = sorted(
            path.relative_to(REPO).as_posix()
            for path in source_paths
            if path != target
            and path.suffix == ".py"
            and mutation.original
            in path.read_text(encoding="utf-8", errors="replace")
        )
        moved = f"; anchor now lives in {elsewhere}" if elsewhere else ""
        stale.append(
            f"{mutation.name}: anchor occurrence {mutation.occurrence} "
            f"absent from {mutation.target}{moved}"
        )
    if stale:
        detail = "\n  ".join(stale)
        raise RuntimeError(
            f"{len(stale)} mutation anchor(s) are stale, so the guards they "
            f"exercise are UNTESTED:\n  {detail}"
        )
    print(f"[PASS] all {len(selected)} mutation anchors resolve in their "
          "declared target module")


def _run_checker(
    root: Path,
    checker: str,
    timeout: int | None = None,
    *,
    arguments: tuple[str, ...] = (),
) -> tuple[int, str]:
    if timeout is None:
        timeout = CHECKER_TIMEOUT_SECONDS.get(
            checker, DEFAULT_CHECKER_TIMEOUT_SECONDS
        )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    completed = subprocess.run(
        [sys.executable, checker, *arguments],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout


def _output_tail(output: str, lines: int = 35) -> str:
    return "\n".join(output.splitlines()[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "statistical",
            "artifact",
            "environment",
            "qualification",
            "generation",
        ),
        default="all",
        help="limit the mutation family (default: all)",
    )
    args = parser.parse_args()
    selected = [
        mutation for mutation in MUTATIONS
        if args.only == "all" or mutation.group == args.only
    ]

    source_paths = _source_paths()
    real_target_paths = sorted({
        REPO / mutation.target for mutation in selected
    })
    real_before = {path: path.read_bytes() for path in real_target_paths}
    _validate_anchor_inventory(selected, source_paths)

    caught = 0
    with tempfile.TemporaryDirectory(prefix="ttbi-r4-mutations-") as tmp:
        isolated = Path(tmp)
        _copy_isolated_tree(isolated, source_paths)
        pristine = _tree_fingerprint(isolated)

        # A mutation's non-zero exit is meaningful only after the unmutated
        # checker is proven runnable and green in the same isolated tree.
        for checker in sorted({mutation.checker for mutation in selected}):
            code, output = _run_checker(isolated, checker)
            expected = BASELINE_EVIDENCE[checker]
            if code != 0 or expected not in output:
                raise RuntimeError(
                    f"isolated baseline {checker} is not GREEN:\n"
                    f"{_output_tail(output)}"
                )
            if _tree_fingerprint(isolated) != pristine:
                raise RuntimeError(
                    f"baseline {checker} changed the isolated source tree"
                )
            print(f"[BASELINE PASS] {checker}")

        for index, mutation in enumerate(selected, start=1):
            target = isolated / mutation.target
            original_bytes = target.read_bytes()
            try:
                text = original_bytes.decode("utf-8")
                mutated = _replace_occurrence(
                    text,
                    mutation.original,
                    mutation.mutant,
                    mutation.occurrence,
                )
                target.write_bytes(mutated.encode("utf-8"))
                mutation_arguments = (
                    ("--static-mutation-smoke",)
                    if mutation.group == "environment"
                    and mutation.checker == "check_environment_lock.py"
                    else ()
                )
                code, output = _run_checker(
                    isolated,
                    mutation.checker,
                    arguments=mutation_arguments,
                )
                if code == 0 or mutation.evidence not in output:
                    raise RuntimeError(
                        f"mutation was not caught for the intended reason: "
                        f"{mutation.name}\n"
                        f"returncode={code}; expected evidence="
                        f"{mutation.evidence!r}\n{_output_tail(output)}"
                    )
                caught += 1
                print(
                    f"[CAUGHT {index}/{len(selected)}] {mutation.name}\n"
                    f"  guard: {mutation.checker} -> {mutation.evidence}"
                )
            finally:
                target.write_bytes(original_bytes)
                if target.read_bytes() != original_bytes:
                    raise RuntimeError(
                        f"byte restoration failed for {mutation.target}"
                    )
            if _tree_fingerprint(isolated) != pristine:
                raise RuntimeError(
                    f"isolated tree differs after restoring {mutation.name}"
                )
            print(f"  [RESTORED] {mutation.target} byte-identical")

    real_after = {path: path.read_bytes() for path in real_target_paths}
    if real_after != real_before:
        changed = [
            path.relative_to(REPO).as_posix()
            for path in real_target_paths
            if real_before[path] != real_after[path]
        ]
        raise RuntimeError(
            "real repository targets changed during isolated audit: "
            f"{changed}. Run this harness serially."
        )

    print(
        f"\nR4 MUTATION GUARDS: {caught}/{len(selected)} CAUGHT; "
        "0 MISSED; ISOLATED TREE RESTORED BYTE-FOR-BYTE; "
        "REAL TREE UNTOUCHED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
