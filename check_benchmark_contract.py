"""Adversarial, standard-library-only checks for the R11 compute benchmark.

The heavy benchmark is intentionally not imported through the training stack
and is never executed here.  These checks combine AST inspection, mutation
tests, and small pure-Python behavioural probes.  Their purpose is to make the
reviewed benchmark contract fail when any decisive workload, HPO, capacity,
failure, reporting, or finalist-refit invariant is weakened.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark_r5_compute.py"
SOURCE = BENCHMARK.read_text(encoding="utf-8")
FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    if condition:
        print(f"[PASS] {label}")
    else:
        FAILURES += 1
        print(f"[FAIL] {label}")


def _function_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assignment(tree: ast.AST, name: str) -> ast.AST | None:
    for node in getattr(tree, "body", []):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            return node.value
    return None


def _literal(tree: ast.AST, name: str) -> Any:
    node = _assignment(tree, name)
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _called_attribute(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attribute
    ]


def _called_name(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    ]


def _keyword_names(call: ast.Call) -> set[str | None]:
    return {keyword.arg for keyword in call.keywords}


def _dict_list_length(tree: ast.AST, assignment: str, key: str) -> int | None:
    node = _assignment(tree, assignment)
    if not isinstance(node, ast.Dict):
        return None
    for key_node, value_node in zip(node.keys, node.values):
        if (
            isinstance(key_node, ast.Constant)
            and key_node.value == key
            and isinstance(value_node, (ast.List, ast.Tuple))
        ):
            return len(value_node.elts)
    return None


def _range_stop_for_dofs(tree: ast.AST) -> int | None:
    node = _assignment(tree, "DOFS")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tuple"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
        and len(node.args[0].args) == 1
        and isinstance(node.args[0].args[0], ast.Constant)
    ):
        return node.args[0].args[0].value
    return None


def static_violations(source: str) -> list[str]:
    """Return decisive R11 contract violations found in arbitrary source."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["syntax"]
    funcs = _function_nodes(tree)
    violations: list[str] = []

    expected_literals = {
        "BENCHMARK_SCHEMA": "ttbi-r11-compute-benchmark-v1",
        "DESCRIPTOR_SCHEMA": "ttbi-r11-workload-benchmark-v1",
        "DERIVATION_SCHEMA": "ttbi-r11-derived-workload-v1",
        "STUDY_DATASET_NAME":
            "R11_NON_SCIENTIFIC_DERIVED_WORST_SIZE_WORKLOAD",
        "SOURCE_X_SHAPE": (12950, 2, 512),
        "SOURCE_Y_SHAPE": (12950, 2),
        "SOURCE_N_STATES": 259,
        "DEFAULT_X_SHAPE": (23750, 8, 512),
        "DEFAULT_Y_SHAPE": (23750, 5),
        "N_STATES": 475,
        "PASSAGES_PER_STATE": 50,
        "TARGET_SUPPORTS": (2, 3, 4),
        "BEARING_TARGETS": ("left", "right"),
        "SEED": 42,
        "USEFUL_TRIALS": 100,
        "EPOCHS": 50,
        "USE_PRUNER": True,
        "ANCHOR_STAGE": "s21_scour4",
        "EXECUTION_BLOCK": "l99",
    }
    for name, expected in expected_literals.items():
        if _literal(tree, name) != expected:
            violations.append(f"constant:{name}")
    if _range_stop_for_dofs(tree) != 8:
        violations.append("full-eight-dofs")
    if _dict_list_length(tree, "DERIVATION_RECIPE", "feature_channels") != 8:
        violations.append("derived-eight-channels")
    if _dict_list_length(tree, "DERIVATION_RECIPE", "label_heads") != 5:
        violations.append("derived-five-heads")

    output_node = _assignment(tree, "OUTPUT_ROOT_RELATIVE")
    output_text = ast.get_source_segment(source, output_node) or ""
    if ".audit_tmp/r11_compute_benchmark" not in output_text:
        violations.append("audit-output-root")

    if _assignment(tree, "MAX_FAIL_SLACK") is not None:
        violations.append("failure-slack-constant")
    hpo_fields = _literal(tree, "HPO_REPORT_FIELDS")
    if not isinstance(hpo_fields, tuple):
        violations.append("hpo-report-fields")
    elif any(
        name in hpo_fields
        for name in (
            "failure_slack",
            "failed_trial_durations_excluded",
            "failed_trial_duration_reason",
        )
    ):
        violations.append("failure-retry-reporting")

    required_funcs = {
        "_materialize_derived_workload",
        "_validate_derived_workload",
        "_load_runtime",
        "_build_config",
        "_prepare_study",
        "_run_registered_anchor_hpo",
        "_run_finalist_once",
        "_completed_summary_if_valid",
        "_materialize_study_storage_receipt",
        "_storage_snapshot",
        "_coordination_receipt_snapshot",
        "_immutable_evidence_snapshot",
        "_regular_file_snapshot",
        "_hyperparameter_execution_record",
        "_validate_benchmark_hyperparameter_execution",
        "main",
    }
    missing = sorted(required_funcs - set(funcs))
    if missing:
        violations.append(f"missing-functions:{','.join(missing)}")
        return violations

    build = funcs["_build_config"]
    build_text = ast.get_source_segment(source, build) or ""
    for evidence in (
        '"name_short": "PAA_LSTM_NHiTS"',
        '"use_lstm": True',
        '"use_nhits": True',
        '"dofs": list(DOFS)',
        '"target_supports": list(TARGET_SUPPORTS)',
        '"bearing_targets": list(BEARING_TARGETS)',
        '"protocol_core_hash"',
        '"execution_runtime"',
        '"hyperparameter_mode": "anchor_hpo"',
    ):
        if evidence not in build_text:
            violations.append(f"anchor-config:{evidence}")

    prepare = funcs["_prepare_study"]
    create_calls = _called_attribute(prepare, "_create_or_resume_study")
    stamp_calls = _called_attribute(prepare, "_stamp_study_protocol")
    if len(create_calls) != 1:
        violations.append("one-optuna-study")
    if len(stamp_calls) != 1:
        violations.append("one-protocol-stamp")
    elif not {
        "hyperparameter_plan",
        "capacity_receipt",
        "n_trials",
        "use_pruner",
    } <= _keyword_names(stamp_calls[0]):
        violations.append("stamp-plan-capacity")

    hpo = funcs["_run_registered_anchor_hpo"]
    execute_calls = _called_attribute(hpo, "_execute_protocol_study")
    if len(execute_calls) != 1:
        violations.append("production-protocol-study-path")
    if _called_attribute(hpo, "optimize"):
        violations.append("hpo-direct-optimize")
    if any(isinstance(node, ast.While) for node in ast.walk(hpo)):
        violations.append("hpo-retry-loop")
    if any(
        keyword.arg == "catch"
        for call in ast.walk(hpo)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    ):
        violations.append("hpo-catch")
    hpo_text = ast.get_source_segment(source, hpo) or ""
    for evidence in (
        'plan["mode"] != "anchor_hpo"',
        'plan["effective_n_trials"] != USEFUL_TRIALS',
        'tuple(plan["active_dofs"]) != DOFS',
        'plan["architecture"] != "PAA_LSTM_NHiTS"',
        'counts_after["failed"]',
        "validate_terminal_study(study, plan)",
    ):
        if evidence not in hpo_text:
            violations.append(f"hpo-gate:{evidence}")

    quiet = funcs.get("optimize")
    # There are several methods called optimize only if the source drifts.
    quiet_methods = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "_QuietProtocolStudy"
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == "optimize"
    ]
    if len(quiet_methods) != 1:
        violations.append("quiet-study-adapter")
    else:
        raw_calls = _called_attribute(quiet_methods[0], "optimize")
        if len(raw_calls) != 1:
            violations.append("one-raw-optimize")
        else:
            keywords = _keyword_names(raw_calls[0])
            if "catch" in keywords or "n_trials" not in keywords:
                violations.append("fatal-raw-optimize")
        quiet_text = ast.get_source_segment(source, quiet_methods[0]) or ""
        if (
            "callbacks=[self._receipt_callback]" not in quiet_text
            or "show_progress_bar=False" not in quiet_text
            or "callbacks != [self._production_callback]" not in quiet_text
        ):
            violations.append("quiet-output-only-adapter")
    del quiet

    main = funcs["main"]
    main_text = ast.get_source_segment(source, main) or ""
    ordered = (
        "enforce_execution_block",
        "ensure_capacity_preflight",
        "derive_execution_plan",
        "_prepare_study",
        "_run_registered_anchor_hpo",
        "_run_finalist_once",
    )
    positions = [main_text.find(token) for token in ordered]
    if any(position < 0 for position in positions) or positions != sorted(
        positions
    ):
        violations.append("main-preflight-order")
    if len(_called_name(main, "_run_finalist_once")) != 1:
        violations.append("one-finalist-refit-call")
    if len(_called_name(main, "_run_registered_anchor_hpo")) != 1:
        violations.append("one-anchor-hpo-call")
    if "derived_fixture_paths" not in main_text:
        violations.append("derived-workload-not-used")
    for evidence in (
        '"source_features"',
        '"source_labels"',
        '"derived_features"',
        '"derived_labels"',
        '"derived_manifest"',
        "_snapshot_files(all_fixture_paths)",
    ):
        if evidence not in main_text:
            violations.append(f"workload-auth:{evidence}")

    finalist = funcs["_run_finalist_once"]
    finalist_calls = _called_attribute(
        finalist, "fit_predict_finalist_fold"
    )
    if len(finalist_calls) != 1:
        violations.append("one-shared-finalist-helper")
    elif not {
        "n_epochs",
        "max_epochs",
        "n_scour_heads",
    } <= _keyword_names(finalist_calls[0]):
        violations.append("finalist-explicit-budget")

    materializer = funcs["_materialize_derived_workload"]
    materializer_text = ast.get_source_segment(source, materializer) or ""
    for evidence in (
        'mmap_mode="r"',
        "allow_pickle=False",
        "SOURCE_X_SHAPE",
        "SOURCE_Y_SHAPE",
        "open_memmap",
        "source_indices",
        "DERIVATION_RECIPE",
        "_snapshot_files(files)",
        "_atomic_json(staging / \"derived_workload.json\"",
        "os.rename(staging, derived_dir)",
    ):
        if evidence not in materializer_text:
            violations.append(f"derivation:{evidence}")
    validator_text = ast.get_source_segment(
        source, funcs["_validate_derived_workload"]
    ) or ""
    for evidence in (
        "_snapshot_files(paths)",
        "manifest.get(\"files\") != snapshot",
        "DEFAULT_X_SHAPE",
        "DEFAULT_Y_SHAPE",
        "np.memmap",
    ):
        if evidence not in validator_text:
            violations.append(f"derived-validation:{evidence}")

    runtime_loader = funcs["_load_runtime"]
    runtime_text = ast.get_source_segment(source, runtime_loader) or ""
    for evidence in (
        '"core.execution_environment"',
        '"core.hyperparameter_policy"',
        '"core.capacity_preflight"',
        '"_execute_protocol_study"',
        '"derive_execution_plan"',
        '"ensure_capacity_preflight"',
        'OPTUNA_PROTOCOL["max_fail_slack"]) != 0',
    ):
        if evidence not in runtime_text:
            violations.append(f"runtime:{evidence}")

    source_files = _literal(tree, "SOURCE_FILES")
    required_sources = {
        "benchmark_r5_compute.py",
        "bundle_source_files.txt",
        "check_benchmark_contract.py",
        "comprehensive_ablation_multidamage.py",
        "training/trainer.py",
        "training/pipeline.py",
        "core/protocol.py",
        "core/campaign_contract.py",
        "core/execution_environment.py",
        "core/hyperparameter_policy.py",
        "core/capacity_preflight.py",
        "core/source_provenance.py",
        "environment/campaign-py313-cu128.json",
        "requirements-campaign-py313-cu128.txt",
    }
    if not isinstance(source_files, tuple) or not required_sources <= set(
        source_files
    ):
        violations.append("source-hash-closure")

    prohibited_report_fields = _literal(
        tree, "PROHIBITED_REPORT_FIELD_FRAGMENTS"
    )
    for fragment in (
        "best_value",
        "trial_value",
        "objective_value",
        "prediction",
        "ground_truth",
        "metric",
        "score",
        "accuracy",
        "loss",
        "mse",
        "mae",
        "rmse",
    ):
        if fragment not in (prohibited_report_fields or ()):
            violations.append(f"report-ban:{fragment}")

    for name in (
        "_ExclusivePidLock",
        "_ActiveWallHeartbeat",
        "_PeakMemoryMonitor",
    ):
        if not any(
            isinstance(node, ast.ClassDef) and node.name == name
            for node in ast.walk(tree)
        ):
            violations.append(f"durability-class:{name}")
    for name in (
        "_atomic_json",
        "_atomic_csv",
        "_completed_summary_if_valid",
        "_materialize_study_storage_receipt",
        "_storage_snapshot",
        "_coordination_receipt_snapshot",
        "_immutable_evidence_snapshot",
        "_regular_file_snapshot",
        "_hyperparameter_execution_record",
        "_validate_benchmark_hyperparameter_execution",
    ):
        if name not in funcs:
            violations.append(f"durability-helper:{name}")
    completed_text = ast.get_source_segment(
        source, funcs["_completed_summary_if_valid"]
    ) or ""
    if "_coordination_receipt_snapshot(run_dir)" not in completed_text:
        violations.append("completed-receipt-coordination-auth")
    for name in (
        "_storage_snapshot",
        "_coordination_receipt_snapshot",
        "_immutable_evidence_snapshot",
        "_completed_summary_if_valid",
    ):
        helper_text = ast.get_source_segment(source, funcs[name]) or ""
        if "_regular_file_snapshot(" not in helper_text:
            violations.append(f"single-handle-snapshot:{name}")
        if "_sha256_file(" in helper_text or "path.stat(" in helper_text:
            violations.append(f"split-stat-hash:{name}")
    if "_json_mapping_from_snapshot(" not in completed_text:
        violations.append("summary-state-same-byte-parse")
    if "_validate_benchmark_hyperparameter_execution(" not in completed_text:
        violations.append("completed-hyperparameter-lineage-auth")
    if (
        'capacity_preflight.get("envelope_file_sha256")\n'
        '        == coordination_snapshot["cuda_capacity"]["sha256"]'
        not in completed_text
    ):
        violations.append("completed-capacity-receipt-auth")
    lineage_record = ast.get_source_segment(
        source, funcs["_hyperparameter_execution_record"]
    ) or ""
    for field in (
        '"campaign_run_tag"',
        '"execution_receipt_sha256"',
        '"block_reference_manifest_sha256"',
        '"validated_run_plan_sha256"',
        '"validated_run_plan"',
    ):
        if field not in lineage_record:
            violations.append(f"hyperparameter-lineage:{field}")
    return violations


print("=" * 78)
print("R11 COMPUTE BENCHMARK CONTRACT CHECKS")
print("=" * 78)

LIVE_VIOLATIONS = static_violations(SOURCE)
check(
    "live benchmark satisfies every static R11 invariant",
    not LIVE_VIOLATIONS,
)
if LIVE_VIOLATIONS:
    for violation in LIVE_VIOLATIONS:
        print(f"       violation: {violation}")


# Mutation discipline: each decisive guard must demonstrably fail closed.
mutations = {
    "state count": (
        SOURCE.replace("N_STATES = 475", "N_STATES = 474", 1),
        "constant:N_STATES",
    ),
    "full array": (
        SOURCE.replace(
            "DOFS = tuple(range(8))", "DOFS = tuple(range(7))", 1
        ),
        "full-eight-dofs",
    ),
    "100-trial budget": (
        SOURCE.replace("USEFUL_TRIALS = 100", "USEFUL_TRIALS = 99", 1),
        "constant:USEFUL_TRIALS",
    ),
    "largest workload": (
        SOURCE.replace(
            "DEFAULT_X_SHAPE = (23750, 8, 512)",
            "DEFAULT_X_SHAPE = (22500, 8, 512)",
            1,
        ),
        "constant:DEFAULT_X_SHAPE",
    ),
    "architecture arm": (
        SOURCE.replace(
            '"name_short": "PAA_LSTM_NHiTS"',
            '"name_short": "PAA_CNN"',
            1,
        ),
        'anchor-config:"name_short": "PAA_LSTM_NHiTS"',
    ),
    "anchor mode": (
        SOURCE.replace(
            '"hyperparameter_mode": "anchor_hpo"',
            '"hyperparameter_mode": "legacy"',
            1,
        ),
        'anchor-config:"hyperparameter_mode": "anchor_hpo"',
    ),
    "production HPO helper": (
        SOURCE.replace(
            "pipeline._execute_protocol_study(",
            "pipeline.execute_unregistered_study(",
            1,
        ),
        "production-protocol-study-path",
    ),
    "capacity preflight": (
        SOURCE.replace(
            "runtime[\"capacity_preflight\"].ensure_capacity_preflight(",
            "runtime[\"capacity_preflight\"].skip_capacity_preflight(",
            1,
        ),
        "main-preflight-order",
    ),
    "failure catch": (
        SOURCE.replace(
            "callbacks=[self._receipt_callback],\n"
            "            show_progress_bar=False,",
            "callbacks=[self._receipt_callback],\n"
            "            show_progress_bar=False,\n"
            "            catch=(Exception,),",
            1,
        ),
        "fatal-raw-optimize",
    ),
    "audit output boundary": (
        SOURCE.replace(
            'Path(".audit_tmp/r11_compute_benchmark")',
            'Path("results/r11_compute_benchmark")',
            1,
        ),
        "audit-output-root",
    ),
    "five-head derivation": (
        SOURCE.replace(
            '{"operation": "copy", "source_channel": 1},\n'
            "    ],\n"
            '    "dtype": "float32",',
            "],\n"
            '    "dtype": "float32",',
            1,
        ),
        "derived-five-heads",
    ),
    "single finalist": (
        SOURCE.replace(
            "finalist = _run_finalist_once(",
            "finalist = _run_finalist_twice(",
            1,
        ),
        "one-finalist-refit-call",
    ),
    "single-handle study receipt": (
        SOURCE.replace(
            "captured = _regular_file_snapshot(\n"
            "        receipt,\n"
            "        \"immutable study receipt\",",
            "captured = _unsafe_split_file_snapshot(\n"
            "        receipt,\n"
            "        \"immutable study receipt\",",
            1,
        ),
        "single-handle-snapshot:_storage_snapshot",
    ),
    "restart lineage validator": (
        SOURCE.replace(
            "    _validate_benchmark_hyperparameter_execution(\n"
            "        summary.get(\"hyperparameter_execution\"),",
            "    _trust_benchmark_hyperparameter_execution(\n"
            "        summary.get(\"hyperparameter_execution\"),",
            1,
        ),
        "completed-hyperparameter-lineage-auth",
    ),
    "capacity receipt bytes": (
        SOURCE.replace(
            '        and capacity_preflight.get("envelope_file_sha256")\n'
            '        == coordination_snapshot["cuda_capacity"]["sha256"]',
            '        and capacity_preflight.get("envelope_file_sha256")',
            1,
        ),
        "completed-capacity-receipt-auth",
    ),
}
for label, (mutated, expected_violation) in mutations.items():
    observed = static_violations(mutated)
    check(
        f"mutation guard rejects {label}",
        mutated != SOURCE and expected_violation in observed,
    )


# Importing this module is standard-library-only: all scientific imports are
# deliberately lazy inside benchmark execution.
spec = importlib.util.spec_from_file_location(
    "r11_benchmark_contract_module", BENCHMARK
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

check("module schema is R11", module.BENCHMARK_SCHEMA.startswith("ttbi-r11-"))
check(
    "live constants describe 475 x 50, eight channels and five heads",
    module.N_STATES == 475
    and module.PASSAGES_PER_STATE == 50
    and module.DEFAULT_X_SHAPE == (23750, 8, 512)
    and module.DEFAULT_Y_SHAPE == (23750, 5)
    and module.DOFS == tuple(range(8)),
)
check(
    "derived workload is explicitly non-scientific",
    module.DERIVATION_RECIPE["classification"] == module.DISCLAIMER,
)


class ExpectedOOM(RuntimeError):
    pass


class FakeStudy:
    def __init__(self, *, fail: bool = False):
        self.study_name = "fake"
        self.trials = []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def optimize(self, objective, **kwargs):
        self.calls.append({"objective": objective, **kwargs})
        if self.fail:
            raise ExpectedOOM("synthetic OOM")
        return "ok"


production_callback = object()
receipt_callback = object()
objective = object()
fake = FakeStudy()
adapter = module._QuietProtocolStudy(
    fake,
    production_callback=production_callback,
    receipt_callback=receipt_callback,
)
result = adapter.optimize(
    objective,
    n_trials=100,
    callbacks=[production_callback],
    show_progress_bar=True,
)
check(
    "output adapter preserves one raw optimize call without catch/replacement",
    result == "ok"
    and len(fake.calls) == 1
    and fake.calls[0] == {
        "objective": objective,
        "n_trials": 100,
        "callbacks": [receipt_callback],
        "show_progress_bar": False,
    },
)

bad_signature_rejected = False
try:
    adapter.optimize(
        objective,
        n_trials=100,
        callbacks=[],
        show_progress_bar=True,
    )
except module.ContractError:
    bad_signature_rejected = True
check(
    "output adapter rejects drift from the production helper signature",
    bad_signature_rejected,
)

oom_propagated = False
try:
    module._QuietProtocolStudy(
        FakeStudy(fail=True),
        production_callback=production_callback,
        receipt_callback=receipt_callback,
    ).optimize(
        objective,
        n_trials=100,
        callbacks=[production_callback],
        show_progress_bar=True,
    )
except ExpectedOOM:
    oom_propagated = True
check("OOM is propagated fatally by the benchmark adapter", oom_propagated)


descriptor = module._descriptor(
    git_sha="a" * 40,
    fixture_before={
        "source_features": {"size_bytes": 1, "sha256": "a" * 64},
        "source_labels": {"size_bytes": 2, "sha256": "b" * 64},
        "derived_features": {"size_bytes": 3, "sha256": "c" * 64},
        "derived_labels": {"size_bytes": 4, "sha256": "d" * 64},
    },
    source_hashes={"training/trainer.py": "e" * 64},
)
config = module._build_config(
    descriptor,
    module._canonical_sha256(descriptor),
    {
        "schema": "fixture-runtime",
        "execution_block": "l99",
    },
    campaign_run_tag="benchmark-fixture",
    execution_receipt_sha256="f" * 64,
)
check(
    "built arm is full-array PAA_LSTM_NHiTS with all five outputs",
    config["name_short"] == "PAA_LSTM_NHiTS"
    and config["dofs"] == list(range(8))
    and config["use_lstm"] is True
    and config["use_nhits"] is True
    and config["target_supports"] == [2, 3, 4]
    and config["bearing_targets"] == ["left", "right"]
    and config["hyperparameter_mode"] == "anchor_hpo"
    and config["campaign_run_tag"] == "benchmark-fixture"
    and config["execution_receipt_sha256"] == "f" * 64
    and config["block_reference_manifest_sha256"] is None
    and config["protocol_core_hash"]
        == module._canonical_sha256(descriptor["core"]),
)
check(
    "descriptor is the l99 anchor and largest registered workload",
    descriptor["rung"] == {
        "stage": "s21_scour4",
        "dataset": module.STUDY_DATASET_NAME,
        "execution_block": "l99",
        "execution_anchor": "s21_scour4",
    }
    and descriptor["workload_size_context"][
        "registered_l60_state_count"
    ] == 450
    and descriptor["workload_size_context"][
        "registered_l99_state_count"
    ] == 475
    and descriptor["workload_size_context"][
        "largest_rung_sample_count_ratio"
    ] == 1.0,
)


hyperparameter_policy = module.importlib.import_module(
    "core.hyperparameter_policy"
)
plan = {
    "schema": module.HYPERPARAMETER_RUN_PLAN_SCHEMA,
    "mode": "anchor_hpo",
    "execution_block": module.EXECUTION_BLOCK,
    "anchor_stage": module.ANCHOR_STAGE,
    "stage": module.ANCHOR_STAGE,
    "dataset": module.STUDY_DATASET_NAME,
    "protocol_hash": module._canonical_sha256(descriptor),
    "protocol_core_hash": module._canonical_sha256(descriptor["core"]),
    "architecture": "PAA_LSTM_NHiTS",
    "seed": module.SEED,
    "active_dofs": list(module.DOFS),
    "effective_n_trials": module.USEFUL_TRIALS,
    "effective_use_pruner": module.USE_PRUNER,
    "requested_n_trials": module.USEFUL_TRIALS,
    "requested_use_pruner": module.USE_PRUNER,
    "policy_sha256": hyperparameter_policy.policy_sha256(),
    "campaign_run_tag":
        f"benchmark-{module._canonical_sha256(descriptor)}",
    "execution_receipt_sha256": "f" * 64,
    "block_reference_manifest_sha256": None,
    "hyperparameter_manifest_sha256": None,
    "hyperparameter_source": None,
}


hyperparameter_execution = module._hyperparameter_execution_record(
    hyperparameter_policy,
    plan,
)
coordination = {
    "execution_block": {
        "relative_path": "execution_receipts/fixture.json",
        "size_bytes": 1,
        "sha256": "f" * 64,
    }
}
validated_plan = module._validate_benchmark_hyperparameter_execution(
    hyperparameter_execution,
    hyperparameter_policy=hyperparameter_policy,
    descriptor_sha256=module._canonical_sha256(descriptor),
    protocol_core_sha256=module._canonical_sha256(descriptor["core"]),
    coordination_receipts=coordination,
)
check(
    "published hyperparameter execution carries the full validated lineage",
    validated_plan == plan
    and hyperparameter_execution["campaign_run_tag"]
        == plan["campaign_run_tag"]
    and hyperparameter_execution["execution_receipt_sha256"] == "f" * 64
    and hyperparameter_execution["block_reference_manifest_sha256"] is None
    and hyperparameter_execution["validated_run_plan_sha256"]
        == module._canonical_sha256(plan),
)

lineage_tampers_rejected = True
for field, replacement in (
    ("campaign_run_tag", "benchmark-" + "0" * 64),
    ("execution_receipt_sha256", "0" * 64),
    ("block_reference_manifest_sha256", "0" * 64),
):
    tampered = deepcopy(hyperparameter_execution)
    tampered[field] = replacement
    tampered["validated_run_plan"][field] = replacement
    tampered["validated_run_plan_sha256"] = module._canonical_sha256(
        tampered["validated_run_plan"]
    )
    try:
        module._validate_benchmark_hyperparameter_execution(
            tampered,
            hyperparameter_policy=hyperparameter_policy,
            descriptor_sha256=module._canonical_sha256(descriptor),
            protocol_core_sha256=module._canonical_sha256(descriptor["core"]),
            coordination_receipts=coordination,
        )
    except module.ContractError:
        continue
    lineage_tampers_rejected = False
check(
    "coherent run-tag/receipt/reference lineage substitutions fail closed",
    lineage_tampers_rejected,
)


scientific_field_rejected = False
try:
    module._assert_no_scientific_report_fields(
        {"nested": {"objective_value": 1.0}}
    )
except module.ContractError:
    scientific_field_rejected = True
check(
    "scientific objective values cannot enter JSON/CSV reports",
    scientific_field_rejected,
)

with tempfile.TemporaryDirectory(prefix="r11-benchmark-contract-") as tmp:
    temporary_root = Path(tmp)
    target = temporary_root / "receipt.json"
    module._atomic_json(
        target,
        {
            "schema": module.BENCHMARK_SCHEMA,
            "classification": module.DISCLAIMER,
            "status": "fixture",
        },
    )
    check(
        "atomic benchmark JSON receipt is durable and parseable",
        target.is_file()
        and module._read_json_mapping(target, "test receipt")["status"]
        == "fixture",
    )

    raw_payload = b"one authenticated byte stream\n"
    binary_path = temporary_root / "snapshot.bin"
    binary_path.write_bytes(raw_payload)
    binary_snapshot = module._regular_file_snapshot(
        binary_path,
        "test binary snapshot",
        max_bytes=len(raw_payload),
        capture_bytes=True,
    )
    check(
        "single-handle snapshot derives bytes, size and digest coherently",
        binary_snapshot is not None
        and binary_snapshot["bytes"] == raw_payload
        and binary_snapshot["size_bytes"] == len(raw_payload)
        and binary_snapshot["sha256"]
            == module.hashlib.sha256(raw_payload).hexdigest(),
    )

    nonregular_rejected = False
    try:
        module._regular_file_snapshot(
            temporary_root,
            "directory substitution",
            max_bytes=1,
        )
    except module.ContractError:
        nonregular_rejected = True
    check(
        "snapshot reader rejects a non-regular path",
        nonregular_rejected,
    )

    symlink_rejected_or_unsupported = False
    symlink_path = temporary_root / "snapshot-link.bin"
    try:
        os.symlink(binary_path, symlink_path)
    except (NotImplementedError, OSError):
        symlink_rejected_or_unsupported = True
    else:
        try:
            module._regular_file_snapshot(
                symlink_path,
                "symlink substitution",
                max_bytes=len(raw_payload),
            )
        except module.ContractError:
            symlink_rejected_or_unsupported = True
    check(
        "snapshot reader rejects symlink substitution when supported",
        symlink_rejected_or_unsupported,
    )

    run_dir = temporary_root / "run"
    run_dir.mkdir()
    for name in module.IMMUTABLE_EVIDENCE_FILES:
        (run_dir / name).write_bytes(f"evidence:{name}\n".encode("ascii"))
    for directory_name, receipt_name in (
        ("execution_receipts", "execution.json"),
        ("capacity_receipts", "capacity.json"),
    ):
        directory = run_dir / directory_name
        directory.mkdir()
        (directory / receipt_name).write_bytes(b"{}\n")

    original_open = module.os.open
    open_counts: dict[str, int] = {}

    def counted_open(path, flags, *args):
        key = str(Path(path).resolve())
        open_counts[key] = open_counts.get(key, 0) + 1
        return original_open(path, flags, *args)

    module.os.open = counted_open
    try:
        storage = module._storage_snapshot(run_dir)
        immutable = module._immutable_evidence_snapshot(
            run_dir,
            study_storage_snapshot=storage,
        )
        coordination_snapshot = module._coordination_receipt_snapshot(run_dir)
    finally:
        module.os.open = original_open

    expected_once = [
        run_dir / name for name in module.IMMUTABLE_EVIDENCE_FILES
    ] + [
        run_dir / "execution_receipts" / "execution.json",
        run_dir / "capacity_receipts" / "capacity.json",
    ]
    check(
        "storage/receipt/evidence publication opens every input exactly once",
        set(immutable) == set(module.IMMUTABLE_EVIDENCE_FILES)
        and set(coordination_snapshot) == {"execution_block", "cuda_capacity"}
        and all(
            open_counts.get(str(path.resolve())) == 1
            for path in expected_once
        ),
    )

    completed_descriptor_sha256 = module._canonical_sha256(descriptor)
    completed_protocol_core_sha256 = module._canonical_sha256(
        descriptor["core"]
    )
    completed_plan = deepcopy(plan)
    completed_plan["execution_receipt_sha256"] = coordination_snapshot[
        "execution_block"
    ]["sha256"]
    completed_execution = module._hyperparameter_execution_record(
        hyperparameter_policy,
        completed_plan,
    )
    fixture_snapshot = {
        "fixture": {
            "path": str(binary_path.resolve()),
            "size_bytes": len(raw_payload),
            "mtime_ns": 0,
            "sha256": binary_snapshot["sha256"],
        }
    }
    source_hashes = {"benchmark_r5_compute.py": "a" * 64}
    run_id = "snapshot-contract-run"
    completed_summary = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": completed_descriptor_sha256,
        "identity": {
            "git_sha": "b" * 40,
            "run_id": run_id,
            "run_directory": str(run_dir),
            "git_tracked_dirty_at_start": False,
        },
        "command": {
            "this_invocation_measurement_completed_utc":
                "2000-01-01T00:00:00+00:00",
        },
        "execution_attestation": {
            "receipt_sha256": coordination_snapshot[
                "execution_block"
            ]["sha256"],
        },
        "capacity_preflight": {
            "passed": True,
            "envelope_file_sha256": coordination_snapshot[
                "cuda_capacity"
            ]["sha256"],
        },
        "hyperparameter_execution": completed_execution,
        "hashes": {
            "source_before": source_hashes,
            "source_after": source_hashes,
            "fixture_before": fixture_snapshot,
            "fixture_after": fixture_snapshot,
            "study_storage": storage,
            "coordination_receipts": coordination_snapshot,
            "immutable_evidence": immutable,
        },
    }
    module._atomic_json(run_dir / "summary.json", completed_summary)
    completed_summary_snapshot = module._regular_file_snapshot(
        run_dir / "summary.json",
        "completed summary fixture",
        max_bytes=module.JSON_SNAPSHOT_MAX_BYTES,
    )
    module._atomic_json(
        run_dir / "run_state.json",
        {
            "schema": module.BENCHMARK_SCHEMA,
            "classification": module.DISCLAIMER,
            "status": "completed",
            "descriptor_sha256": completed_descriptor_sha256,
            "summary": "summary.json",
            "summary_sha256": completed_summary_snapshot["sha256"],
        },
    )

    open_counts.clear()
    module.os.open = counted_open
    try:
        authenticated, repaired = module._completed_summary_if_valid(
            run_dir,
            hyperparameter_policy=hyperparameter_policy,
            descriptor_sha256=completed_descriptor_sha256,
            protocol_core_sha256=completed_protocol_core_sha256,
            git_sha="b" * 40,
            run_id=run_id,
            fixture_snapshot=fixture_snapshot,
            source_hashes=source_hashes,
            recover_torn_state=False,
        )
    finally:
        module.os.open = original_open
    authenticated_inputs = [
        run_dir / "summary.json",
        run_dir / "run_state.json",
        *expected_once,
    ]
    check(
        "restart parses and hashes summary/state/evidence from one view each",
        authenticated == completed_summary
        and repaired is False
        and all(
            open_counts.get(str(path.resolve())) == 1
            for path in authenticated_inputs
        ),
    )


if FAILURES:
    raise SystemExit(
        f"R11 COMPUTE BENCHMARK CONTRACT: {FAILURES} FAILURE(S)"
    )
print("R11 COMPUTE BENCHMARK CONTRACT: ALL PASS")
