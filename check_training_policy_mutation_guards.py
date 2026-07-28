"""Isolated mutation audit for executable training-policy guards.

This auditor copies only the Python/config inputs needed by the relevant
checkers into a temporary tree.  It proves every unmutated checker is green,
injects one precisely anchored defect at a time, and requires the intended
checker to turn red for the registered evidence.

Every mutation is restored from the original bytes in ``finally``.  The
temporary-tree SHA-256 map must return to its pristine value after every run,
and the real repository mutation targets must remain byte-identical.  No
campaign data, results, stage bundles, or bundle manifest are copied or
modified.

Run serially:

    python check_training_policy_mutation_guards.py
"""

from __future__ import annotations

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
    target: str
    checker: str
    original: str
    mutant: str
    evidence: str


MUTATIONS = (
    Mutation(
        name="objective_value is hardwired to aggregate MSE",
        target="core/task.py",
        checker="check_campaign_controls.py",
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
        checker="check_campaign_controls.py",
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
        checker="check_campaign_controls.py",
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
            "[FAIL] all production seeding calls consume TRAIN_PROTOCOL "
            "determinism policy"
        ),
    ),
    Mutation(
        name="campaign driver seeding omits determinism policy",
        target="comprehensive_ablation_multidamage.py",
        checker="check_campaign_controls.py",
        original=(
            '    set_global_seed(SEEDS[0], TRAIN_PROTOCOL["determinism"])'
        ),
        mutant=(
            "    set_global_seed(SEEDS[0])"
            "  # MUTANT: omit TRAIN_PROTOCOL determinism"
        ),
        evidence=(
            "[FAIL] all production seeding calls consume TRAIN_PROTOCOL "
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
        evidence="mutation survived: campaign run-plan loses run_tag",
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
        evidence="mutation survived: campaign run-plan carries invalid receipt",
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
        evidence="mutation survived: follower run-plan loses reference",
    ),
)


BASELINE_EVIDENCE = {
    "check_campaign_controls.py": "CAMPAIGN CONTROLS: ALL PASS",
    "check_weighted_head_mse.py": "WEIGHTED HEAD MSE: ALL PASS",
    "check_environment_lock.py": "ENVIRONMENT LOCK: ALL PASS",
    "check_hyperparameter_policy.py":
        "PASS: hyperparameter policy derivation/authentication/mutations",
}

ROOT_INPUTS = (
    "check_campaign_controls.py",
    "check_weighted_head_mse.py",
    "check_environment_lock.py",
    "check_hyperparameter_policy.py",
    "comprehensive_ablation_multidamage.py",
)
FORBIDDEN_TEMP_ENTRIES = (
    "data",
    "results",
    "bundle_s21_scour4",
    "bundle_s23_all4",
    "bundle_source_files.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    paths = {REPO / relative for relative in ROOT_INPUTS}
    for package in ("core", "training"):
        paths.update((REPO / package).glob("*.py"))
    paths.add(REPO / "environment" / "campaign-py313-cu128.json")
    missing = sorted(
        path.relative_to(REPO).as_posix()
        for path in paths
        if not path.is_file()
    )
    if missing:
        raise RuntimeError(f"required isolated inputs are missing: {missing}")
    return sorted(paths, key=lambda path: path.relative_to(REPO).as_posix())


def _copy_isolated_tree(destination: Path, sources: list[Path]) -> None:
    for source in sources:
        relative = source.relative_to(REPO)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    present_forbidden = [
        entry for entry in FORBIDDEN_TEMP_ENTRIES
        if (destination / entry).exists()
    ]
    if present_forbidden:
        raise RuntimeError(
            f"forbidden campaign artifacts entered temp tree: "
            f"{present_forbidden}"
        )


def _tree_sha_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


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
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = ""
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


def _assert_real_targets_unchanged(
    before: dict[Path, bytes],
) -> None:
    changed = [
        path.relative_to(REPO).as_posix()
        for path, original in before.items()
        if not path.is_file() or path.read_bytes() != original
    ]
    if changed:
        raise RuntimeError(
            "real repository targets changed during isolated audit: "
            f"{changed}. Run this harness serially."
        )


def main() -> int:
    sources = _source_paths()
    real_targets = sorted({REPO / mutation.target for mutation in MUTATIONS})
    real_before = {path: path.read_bytes() for path in real_targets}
    caught = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="ttbi-training-policy-mutations-",
        ) as tmp:
            isolated = Path(tmp)
            _copy_isolated_tree(isolated, sources)
            pristine = _tree_sha_map(isolated)

            for checker in sorted(BASELINE_EVIDENCE):
                code, output = _run_checker(isolated, checker)
                evidence = BASELINE_EVIDENCE[checker]
                if code != 0 or evidence not in output:
                    raise RuntimeError(
                        f"isolated baseline {checker} is not GREEN: "
                        f"returncode={code}; expected={evidence!r}\n"
                        f"{_output_tail(output)}"
                    )
                if _tree_sha_map(isolated) != pristine:
                    raise RuntimeError(
                        f"baseline {checker} changed the isolated source tree"
                    )
                print(f"[BASELINE PASS] {checker} -> {evidence}")

            for index, mutation in enumerate(MUTATIONS, start=1):
                target = isolated / mutation.target
                original_bytes = target.read_bytes()
                try:
                    original_text = original_bytes.decode("utf-8")
                    mutated_text = _inject_exactly_once(
                        original_text, mutation)
                    target.write_bytes(mutated_text.encode("utf-8"))
                    code, output = _run_checker(
                        isolated, mutation.checker)
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
                    target.write_bytes(original_bytes)
                    if target.read_bytes() != original_bytes:
                        raise RuntimeError(
                            f"byte restoration failed for {mutation.target}"
                        )
                    if _tree_sha_map(isolated) != pristine:
                        raise RuntimeError(
                            "isolated tree differs after restoring "
                            f"{mutation.name}"
                        )
                print(
                    f"  [RESTORED] {mutation.target} byte-identical; "
                    "temp SHA map pristine"
                )
    finally:
        _assert_real_targets_unchanged(real_before)

    if caught != len(MUTATIONS):
        raise RuntimeError(
            f"internal count mismatch: caught {caught}/{len(MUTATIONS)}")
    print(
        f"\nTRAINING POLICY MUTATION GUARDS: "
        f"{caught}/{len(MUTATIONS)} CAUGHT; 0 MISSED; "
        "ISOLATED TREE RESTORED BYTE-FOR-BYTE; REAL TARGETS UNCHANGED; "
        "NO DATA/RESULTS/BUNDLES TOUCHED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
