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
        name="outer paired contrast accepts different state keys",
        group="statistical",
        target="comprehensive_ablation_multidamage.py",
        checker="check_statistical_inference.py",
        original=(
            "if not np.array_equal(winner_states, comp_states):\n"
            "            raise RuntimeError(\n"
            "                f\"paired contrast state mismatch: "
            "{winner_key} vs {comparator_key}\"\n"
            "            )"
        ),
        mutant=(
            "if False:  # MUTANT: semantic state-key firewall disabled\n"
            "            raise RuntimeError(\n"
            "                f\"paired contrast state mismatch: "
            "{winner_key} vs {comparator_key}\"\n"
            "            )"
        ),
        evidence=(
            "[FAIL] outer-test paired contrast rejects differently keyed "
            "state tensors"
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
            "[FAIL] standalone verifier rejects a missing required "
            "provenance field"
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
        evidence="[FAIL] standalone verifier rejects one-byte champion tamper",
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
            "[FAIL] standalone verifier rejects descriptor/hash disagreement"
        ),
    ),
    Mutation(
        name="study-linked artifact verifier suppresses provenance mismatches",
        group="artifact",
        target="training/pipeline.py",
        checker="check_artifact_provenance.py",
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
        evidence="[FAIL] metadata protocol-descriptor tamper rejected",
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
            "[FAIL] standalone follower package rejects reference B "
            "for package A"
        ),
    ),
    Mutation(
        name="cross-rung analyzer trusts a coherently substituted internal reference",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "if champion_canonical_sha != expected_reference_sha:"
        ),
        mutant=(
            "if False:  # MUTANT: independently retained trust root ignored"
        ),
        evidence=(
            "[FAIL] coherent champion+frozen+seven-pin substitution still "
            "fails external root"
        ),
    ),
    Mutation(
        name="cross-rung analyzer ignores follower frozen-reference lineage",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "        or frozen.get(\"block_reference_manifest_sha256\")\n"
            "        != expected_artifact_reference"
        ),
        mutant=(
            "        or False  "
            "# MUTANT: frozen block-reference lineage ignored"
        ),
        evidence=(
            "[FAIL] follower frozen selection citing another block reference "
            "is rejected"
        ),
    ),
    Mutation(
        name="cross-rung analyzer ignores block-reference lineage on CSV rows",
        group="artifact",
        target="core/cross_rung_inference.py",
        checker="check_cross_rung_inference.py",
        original=(
            "    if any(\n"
            "        row.get(\"block_reference_manifest_sha256\")\n"
            "        != expected_csv_reference\n"
            "        for row in rows\n"
            "    ):"
        ),
        mutant=(
            "    if False:  "
            "# MUTANT: per-row block-reference lineage ignored"
        ),
        evidence=(
            "[FAIL] anchor metric row must retain the canonical empty "
            "anti-cycle reference"
        ),
    ),
    Mutation(
        name="benchmark restart trusts unvalidated hyperparameter lineage",
        group="artifact",
        target="benchmark_r5_compute.py",
        checker="check_benchmark_contract.py",
        original=(
            "    _validate_benchmark_hyperparameter_execution(\n"
            "        summary.get(\"hyperparameter_execution\"),"
        ),
        mutant=(
            "    _trust_benchmark_hyperparameter_execution(\n"
            "        summary.get(\"hyperparameter_execution\"),"
        ),
        evidence=(
            "[FAIL] live benchmark satisfies every static R11 invariant"
        ),
    ),
    Mutation(
        name="benchmark study receipt regresses to a split stat/hash read",
        group="artifact",
        target="benchmark_r5_compute.py",
        checker="check_benchmark_contract.py",
        original=(
            "    captured = _regular_file_snapshot(\n"
            "        receipt,\n"
            '        "immutable study receipt",'
        ),
        mutant=(
            "    captured = _unsafe_split_file_snapshot(\n"
            "        receipt,\n"
            '        "immutable study receipt",'
        ),
        evidence=(
            "[FAIL] live benchmark satisfies every static R11 invariant"
        ),
    ),
    Mutation(
        name="champion writer publishes before validating its trust-root payload",
        group="artifact",
        target="comprehensive_ablation_multidamage.py",
        checker="check_execution_blocking.py",
        original=(
            "    payload = _validate_reference_payload_for_publication(payload)"
        ),
        mutant=(
            "    payload = payload  "
            "# MUTANT: trust-root publication validator bypassed"
        ),
        evidence=(
            "[FAIL] invalid champion payload is rejected before immutable "
            "publication"
        ),
    ),
    Mutation(
        name="protocol updater treats an orphaned lock file as a live writer",
        group="artifact",
        target="comprehensive_ablation_multidamage.py",
        checker="check_execution_blocking.py",
        original=(
            "        descriptor = os.open(\n"
            "            lock_path,\n"
            "            os.O_CREAT | os.O_RDWR | "
            'getattr(os, "O_BINARY", 0),'
        ),
        mutant=(
            "        if lock_path.exists() and lock_path.stat().st_size == 0:\n"
            "            raise RuntimeError("
            '"MUTANT: orphaned lock blocks restart")\n'
            "        descriptor = os.open(\n"
            "            lock_path,\n"
            "            os.O_CREAT | os.O_RDWR | "
            'getattr(os, "O_BINARY", 0),'
        ),
        evidence=(
            "[FAIL] orphaned legacy lock file cannot block a valid crash "
            "restart"
        ),
    ),
    Mutation(
        name="campaign JSON snapshot silently accepts duplicate evidence keys",
        group="artifact",
        target="comprehensive_ablation_multidamage.py",
        checker="check_execution_blocking.py",
        original="            object_pairs_hook=unique_object,",
        mutant=(
            "            object_pairs_hook=dict,  "
            "# MUTANT: duplicate JSON keys collapse silently"
        ),
        evidence=(
            "[FAIL] reference publication rejects duplicate frozen-selection "
            "JSON keys"
        ),
    ),
    Mutation(
        name="environment verifier accepts package-version drift",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original="if actual != expected:",
        mutant="if False:  # MUTANT: package-version guard disabled",
        evidence="[FAIL] package mismatch hard-fails",
    ),
    Mutation(
        name="environment verifier accepts cuBLAS determinism drift",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original="if actual_cublas != expected_cublas:",
        mutant="if False:  # MUTANT: cuBLAS guard disabled",
        evidence="[FAIL] cuBLAS determinism mismatch hard-fails",
    ),
    Mutation(
        name="environment verifier accepts unavailable required CUDA",
        group="environment",
        target="core/environment.py",
        checker="check_environment_lock.py",
        original=(
            'if spec.get("cuda_required") and not cuda_available:'
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
            'if spec.get("schema") != _LOCK_SCHEMA:'
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
        evidence="[FAIL] extra MATLAB descriptor field hard-fails",
    ),
)


BASELINE_EVIDENCE = {
    "check_statistical_inference.py": "STATISTICAL INFERENCE: ALL PASS",
    "check_artifact_provenance.py": "ARTIFACT PROVENANCE: ALL PASS",
    "check_cross_rung_inference.py": "CROSS-RUNG INFERENCE: ALL PASS",
    "check_benchmark_contract.py":
        "R11 COMPUTE BENCHMARK CONTRACT: ALL PASS",
    "check_execution_blocking.py": "EXECUTION BLOCKING: ALL PASS",
    "check_environment_lock.py": "ENVIRONMENT LOCK: ALL PASS",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    """Return only code/config needed by the three isolated checkers."""

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


def _run_checker(root: Path, checker: str, timeout: int = 180) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    completed = subprocess.run(
        [sys.executable, checker],
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
        choices=("all", "statistical", "artifact", "environment"),
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
                code, output = _run_checker(isolated, mutation.checker)
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
