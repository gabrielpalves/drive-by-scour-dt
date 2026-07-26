"""Adversarial checks for sensor-budget controls and RUN_TAG artifacts.

The campaign driver cannot be imported without a complete generated dataset,
by design.  This checker compiles the relevant function definitions directly
from its AST, so it exercises the shipped implementation without touching
``data/``, ``results/``, studies, or bundles.

Run:  python check_campaign_controls.py
"""
from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


DRIVER = Path(__file__).with_name("comprehensive_ablation_multidamage.py")
SOURCE = DRIVER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(DRIVER))
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def extracted(*names: str, namespace: dict | None = None) -> dict:
    ns = {} if namespace is None else dict(namespace)
    missing = [n for n in names if n not in FUNCTIONS]
    if missing:
        raise RuntimeError(f"driver functions missing: {missing}")
    module = ast.Module(body=[FUNCTIONS[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(DRIVER), "exec"), ns)
    return ns


print("CAMPAIGN CONTROL CHECKS")

# 1. The function that formerly crashed on ALL_DOFS must accept both pairs and
# controls, preserve the legacy pair label, and bound the all-array path.
calls = []
names = {i: f"DOF{i}" for i in range(8)}
ns = extracted(
    "phase_2_pair",
    namespace={
        "ALL_DOFS": list(range(8)),
        "IDX_TO_DOF_NAME": names,
        "run_phase": lambda label, dofs: calls.append((label, list(dofs))),
    },
)
phase = ns["phase_2_pair"]
phase([1, 3])
phase(list(range(8)))
check("two-sensor phase keeps existing pair label",
      calls[0] == ("MD0_Pair_DOF1__DOF3", [1, 3]))
check("ALL_DOFS control executes instead of unpacking/crashing",
      calls[1] == ("MD0_Control_8DOF_0_1_2_3_4_5_6_7", list(range(8))))
check("full-array phase label stays compact", len(calls[1][0]) < 64)

for bad in ([], [1, 1], [0, 8]):
    try:
        phase(bad)
    except ValueError:
        check(f"invalid sensor set rejected: {bad}", True)
    else:
        check(f"invalid sensor set rejected: {bad}", False)

# 2. At a deployment rung the selected architecture and carried architecture
# both need same-architecture 2-vs-8 controls. Otherwise sensor count is
# confounded with architecture.
control_ns = extracted(
    "_control_comparator_keys",
    namespace={
        "STAGE": "s16_all",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour", "s16_all", "s23_all4"},
        "ALL_ARCHITECTURES": [
            {"name_short": name}
            for name in ("A", "B", "C", "D")
        ],
        "CHAMPION_ARCH": "CARRIED",
        "CONTROL_SETS": [list(range(8))],
    },
)
control_keys = control_ns["_control_comparator_keys"]("WINNER")
expected = {
    (name, tuple(range(8))) for name in ("A", "B", "C", "D")
}
check("full-factorial controls cover every trained architecture",
      control_keys == expected)
same_ns = extracted(
    "_control_comparator_keys",
    namespace={
        "STAGE": "s11_bear",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour", "s16_all", "s23_all4"},
        "ALL_ARCHITECTURES": [{"name_short": "WINNER"}],
        "CHAMPION_ARCH": "WINNER",
        "CONTROL_SETS": [list(range(8))],
    },
)
check("identical winner/carried control is deduplicated",
      same_ns["_control_comparator_keys"]("WINNER")
      == {("WINNER", tuple(range(8)))})

same_pair_ns = extracted(
    "_same_pair_architecture_keys",
    namespace={
        "STAGE": "s0_scour",
        "MULTI_ARCH_PAIR_SELECTION_STAGES": {"s0_scour"},
        "ALL_ARCHITECTURES": [
            {"name_short": name} for name in ("A", "B", "C", "D")
        ],
    },
)
check("selected pair is compared under every architecture",
      same_pair_ns["_same_pair_architecture_keys"]([3, 1])
      == {(name, (1, 3)) for name in ("A", "B", "C", "D")})
check("s0 pair phase is full architecture x placement factorial",
      "if STAGE in MULTI_ARCH_PAIR_SELECTION_STAGES:" in SOURCE
      and "ARCHITECTURES = ALL_ARCHITECTURES" in SOURCE)

# 3. Median leaderboards preserve the all-array row and its sensor count.
with tempfile.TemporaryDirectory(prefix="campaign-control-") as td:
    lb_ns = extracted(
        "_write_rows_csv", "_median_leaderboard",
        namespace={
            "np": np,
            "os": os,
            "csv": csv,
            "SEEDS": [42, 1337, 2026],
            "SUMMARY_DIR": td,
        },
    )
    rows = []
    for seed, value in zip([42, 1337, 2026], [3.0, 2.0, 4.0]):
        rows.append({
            "phase": "control", "architecture": "WINNER",
            "dofs": "+".join(f"DOF{i}" for i in range(8)),
            "seed": seed, "n_sensors": 8, "inner_val_mse": value,
        })
    agg = lb_ns["_median_leaderboard"](rows, "control.csv")
    check("8-DOF control survives median leaderboard",
          len(agg) == 1 and agg[0]["n_sensors"] == 8
          and agg[0]["n_seeds"] == 3 and agg[0]["inner_val_mse"] == 3.0)

# 4. The same pre-test completeness guard works for an arbitrary-length control
# key; it must hard-fail if even one required seed/weight is missing.
with tempfile.TemporaryDirectory(prefix="campaign-preflight-") as td:
    configs = []
    for seed in [42, 1337, 2026]:
        name = f"ctrl-{seed}"
        weight_dir = Path(td, name)
        weight_dir.mkdir()
        Path(weight_dir, "DT_champion_weights.pth").write_bytes(b"fixture")
        configs.append({
            "name_short": "WINNER", "dofs": list(range(8)),
            "seed": seed, "name": name,
        })
    fake_optuna = SimpleNamespace(load_study=lambda **_: object())
    pre_ns = extracted(
        "_preflight_comparators",
        namespace={
            "_RAN_PHASES": [{
                "configs": configs, "output_dir": td, "db": "fixture",
            }],
            "IDX_TO_DOF_NAME": names,
            "SEEDS": [42, 1337, 2026],
            "N_TRIALS": 100,
            "optuna": fake_optuna,
            "_study_is_finished": lambda study, n: True,
            "verify_digital_twin_package": lambda study, cfg, out: {},
            "os": os,
        },
    )
    comparator = {("WINNER", tuple(range(8)))}
    try:
        pre_ns["_preflight_comparators"](comparator)
    except Exception as err:
        print(f"    unexpected {type(err).__name__}: {err}")
        check("8-DOF preflight accepts complete seed/weight matrix", False)
    else:
        check("8-DOF preflight accepts complete seed/weight matrix", True)
    Path(td, "ctrl-2026", "DT_champion_weights.pth").unlink()
    try:
        pre_ns["_preflight_comparators"](comparator)
    except RuntimeError:
        check("8-DOF preflight rejects missing seed weight", True)
    else:
        check("8-DOF preflight rejects missing seed weight", False)

# 5. RUN_TAG must isolate every top-level mutable selection artifact, not only
# studies and summaries.
tag_ns = extracted("_with_run_tag", namespace={"RUN_TAG": ""})
check("empty RUN_TAG preserves first-run artifact name",
      tag_ns["_with_run_tag"]("artifact") == "artifact")
tag_ns = extracted("_with_run_tag", namespace={"RUN_TAG": "replicate2"})
check("nonempty RUN_TAG suffixes artifact name",
      tag_ns["_with_run_tag"]("artifact") == "artifact_replicate2")
check("summary path uses RUN_TAG helper",
      '_with_run_tag(f"{STAGE}_summary_ph-{PROTOCOL_HASH_SHORT}")' in SOURCE)
check("champion manifest uses RUN_TAG helper",
      "_with_run_tag(\n                     f\"_champion_arch_" in SOURCE)
check("deployment manifest uses RUN_TAG helper",
      "_with_run_tag(\n                f\"_deployment_selection_" in SOURCE)
check("control architecture policy is protocol-hashed",
      '"control_arch_policy"' in
      Path(__file__).with_name("core").joinpath("protocol.py")
      .read_text(encoding="utf-8"))

# A protocol budget is exact: a manually extended best-of-120 study must never
# masquerade as the pre-registered best-of-100 experiment.
COMPLETE, PRUNED, FAIL, RUNNING, WAITING = (object() for _ in range(5))
fake_optuna_budget = SimpleNamespace(
    trial=SimpleNamespace(TrialState=SimpleNamespace(
        COMPLETE=COMPLETE, PRUNED=PRUNED, FAIL=FAIL,
        RUNNING=RUNNING, WAITING=WAITING,
    ))
)
budget_ns = extracted(
    "_study_is_finished", namespace={"optuna": fake_optuna_budget}
)
study_exact = SimpleNamespace(
    study_name="exact",
    trials=[SimpleNamespace(state=COMPLETE) for _ in range(100)],
)
study_extended = SimpleNamespace(
    study_name="extended",
    trials=[SimpleNamespace(state=COMPLETE) for _ in range(120)],
)
check("exact useful Optuna budget is accepted",
      budget_ns["_study_is_finished"](study_exact, 100))
check("manually extended Optuna budget is rejected",
      not budget_ns["_study_is_finished"](study_extended, 100))

# 6. Bundle contents come from a tracked explicit manifest, never an untracked
# historical ZIP whose bytes could change independently of source_commit.
builder = Path(__file__).with_name("build_stage_bundles.py")
source_manifest = Path(__file__).with_name("bundle_source_files.txt")
required_new = {
    "scour_MATLAB/B54_TrackVectors.m",
    "core/statistical_inference.py",
    "check_paa.py",
    "check_weighted_head_mse.py",
    "check_sensor_noise_pairing.py",
    "check_campaign_controls.py",
    "scour_MATLAB/smoke_b54_overlap_parity.m",
    "check_b54_overlap_parity.py",
    "check_statistical_inference.py",
    "check_artifact_provenance.py",
    "core/environment.py",
    "environment/campaign-py313-cu128.json",
    "requirements-campaign-py313-cu128.txt",
    "check_environment_lock.py",
    "TTBI_2D/b54_model_matrices.py",
    "scour_MATLAB/contact_closure_study.m",
    "scour_MATLAB/smoke_contact_closure.m",
    "core/artifact_provenance.py",
    "docs/audit_r4_results.md",
}
if builder.is_file():
    builder_text = builder.read_text(encoding="utf-8")
    manifest_files = {
        line.strip() for line in source_manifest.read_text(
            encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    check("bundle manifest includes every r4 runtime/test file",
          required_new <= manifest_files)
    check("every r4 bundle-manifest file exists",
          all((Path(__file__).resolve().parent / Path(p)).is_file()
              for p in required_new))
    check("bundle source manifest is tracked and explicit",
          "bundle_source_files.txt" in builder_text
          and "multidamage_stage2_bundle.zip" not in builder_text)
    invalidate_at = builder_text.find("os.replace(sha_manifest, invalid_manifest)")
    first_zip_publish_at = builder_text.find("os.replace(tmp_out, out)")
    manifest_publish_at = builder_text.find(
        "os.replace(sha_manifest_tmp, sha_manifest)")
    check("bundle-set publication is fail-closed across interruption",
          0 <= invalidate_at < first_zip_publish_at < manifest_publish_at
          and "complete_bundle_count" in builder_text)
else:
    # Stage bundles intentionally do not ship their own bundle builder/source
    # ZIP; runtime control tests above remain applicable there.
    print("  [SKIP] bundle-manifest audit (builder absent in stage bundle)")

print()
print("CAMPAIGN CONTROLS: ALL PASS" if fails == 0
      else f"CAMPAIGN CONTROLS: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
