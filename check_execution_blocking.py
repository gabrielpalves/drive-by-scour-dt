"""Adversarial checks for physical host/GPU execution blocking.

Run with the campaign interpreter:
    py -3.13 check_execution_blocking.py
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from unittest import mock

from core.campaign_contract import (
    BLOCK_REFERENCE_MANIFEST_FIELDS,
    validate_block_reference_manifest,
)
from core.execution_environment import (
    EXECUTION_BLOCK_POLICY,
    _read_regular_file,
    _receipt_path,
    canonical_execution_block_policy,
    current_execution_environment,
    enforce_execution_block,
    execution_block_for_stage,
    execution_environment_sha256,
    validate_block_reference_execution,
    validate_execution_runtime,
)
from core.hyperparameter_policy import canonical_json_sha256
from core.utils import set_global_seed
from training.trainer import TRAIN_PROTOCOL


fails = 0
REPO = Path(__file__).resolve().parent
DRIVER_PATH = REPO / "comprehensive_ablation_multidamage.py"
DRIVER_SOURCE = DRIVER_PATH.read_text(encoding="utf-8")
DRIVER_TREE = ast.parse(DRIVER_SOURCE, filename=str(DRIVER_PATH))


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def rejects(name: str, fn, message: str | None = None) -> None:
    try:
        fn()
    except RuntimeError as exc:
        check(name, message is None or message in str(exc))
    except Exception:
        check(name, False)
    else:
        check(name, False)


def fixture_environment(
    *,
    hostname: str = "fixture-host",
    uuid: str = "GPU-fixture-a",
) -> dict:
    return {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": hostname,
            "machine": "AMD64",
            "system": "Windows",
            "platform": "Windows-fixture",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "Fixture GPU",
            "uuid": uuid,
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 36,
            "total_memory_bytes": 8_000_000_000,
            "driver_version": "fixture-driver",
        },
        "numeric_stack": {
            "torch_version": "fixture-torch",
            "cuda_runtime_version": "12.8",
            "cudnn_version": 90701,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": True,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }


def _function(name: str) -> ast.FunctionDef:
    matches = [
        node for node in DRIVER_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"driver must define exactly one {name}()")
    return matches[0]


def _dict_key_sets(node: ast.AST) -> list[set[str]]:
    result: list[set[str]] = []
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Dict):
            result.append({
                key.value for key in candidate.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            })
    return result


def _string_constants(node: ast.AST) -> set[str]:
    return {
        candidate.value for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant)
        and isinstance(candidate.value, str)
    }


FIXTURE_PROTOCOL_CORE = {"code": {"schema_tag": "fixture-schema"}}
CORE_SHA = canonical_json_sha256(FIXTURE_PROTOCOL_CORE)
RUN_TAG = "fixture-run"

print("\n--- A. driver integration contract ---")
execution_keys = {
    "execution_runtime",
    "execution_environment_sha256",
    "execution_receipt_sha256",
}
reference_keys = {
    *execution_keys,
    "capacity_preflight_receipt_sha256",
    "hyperparameter_manifest_sha256",
    "frozen_selection_sha256",
}
writer = _function("_write_champion_arch")
check(
    "reference writer persists execution, capacity, HPO and frozen-selection lineage",
    any(reference_keys <= keys for keys in _dict_key_sets(writer))
    and "hyperparameter_json_sha256" in {
        candidate.func.id
        for candidate in ast.walk(writer)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
    },
)
loader = _function("_load_block_reference_manifest")
check(
    "reference loader requires exact content and full anchor lineage",
    "BLOCK_REFERENCE_SHA256" in {
        candidate.id
        for candidate in ast.walk(loader)
        if isinstance(candidate, ast.Name)
    }
    and "hyperparameter_json_sha256" in {
        candidate.func.id
        for candidate in ast.walk(loader)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
    }
    and {
        "validate_block_reference_manifest",
        "validate_execution_runtime",
    } <= {
        candidate.func.id
        for candidate in ast.walk(loader)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
    }
    and "BLOCK_REFERENCE_MANIFEST_FIELDS" in {
        candidate.id
        for candidate in ast.walk(loader)
        if isinstance(candidate, ast.Name)
    },
)
make_config_node = _function("make_config")
make_config_source = ast.get_source_segment(
    DRIVER_SOURCE, make_config_node
) or ""
check(
    "downstream config construction requires validated champion execution",
    "_CHAMPION_EXECUTION_STATUS is None" in make_config_source
    and '"execution_runtime"' in make_config_source
    and '"campaign_run_tag"' in make_config_source
    and '"execution_receipt_sha256"' in make_config_source,
)
main_nodes = [
    node for node in DRIVER_TREE.body
    if (
        isinstance(node, ast.If)
        and "__main__" in _string_constants(node.test)
    )
]
if len(main_nodes) == 1:
    main_calls = [
        (
            candidate.lineno,
            (
                candidate.func.id
                if isinstance(candidate.func, ast.Name)
                else candidate.func.attr
                if isinstance(candidate.func, ast.Attribute)
                else ""
            ),
        )
        for candidate in ast.walk(main_nodes[0])
        if isinstance(candidate, ast.Call)
    ]
    first_line = {
        name: min(line for line, call_name in main_calls if call_name == name)
        for name in (
            "enforce_execution_block",
            "_initialize_follower_reference",
            "_write_protocol_record",
        )
        if any(call_name == name for _line, call_name in main_calls)
    }
    study_lines = [
        line for line, name in main_calls
        if name in {"phase_1_single_dof", "phase_2_pair"}
    ]
else:
    first_line = {}
    study_lines = []
check(
    "main validates block reference after receipt and before study work",
    set(first_line) == {
        "enforce_execution_block",
        "_initialize_follower_reference",
        "_write_protocol_record",
    }
    and bool(study_lines)
    and first_line["enforce_execution_block"]
        < first_line["_initialize_follower_reference"]
        < first_line["_write_protocol_record"]
        < min(study_lines),
)
summary_node = _function("summarize")
auditable_summary_dicts = [
    keys for keys in _dict_key_sets(summary_node)
    if execution_keys <= keys and "protocol_hash" in keys
]
check(
    "frozen selection and detached deployment manifests carry execution lineage",
    len(auditable_summary_dicts) >= 2,
)

print("\n--- A2. behavioral block-reference authentication ---")
fixture_runtime = {
    "execution_block": "l60",
    "anchor_stage": "s0_scour",
    "execution_environment_sha256": "1" * 64,
}
loader_namespace = {
    "os": os,
    "json": json,
    "math": __import__("math"),
    "Path": Path,
    "STAGE": "s11_bear",
    "SCHEMA_TAG": "fixture-schema",
    "RUN_TAG": RUN_TAG,
    "PROTOCOL_CORE_HASH": CORE_SHA,
    "PROTOCOL_CORE_DESC": FIXTURE_PROTOCOL_CORE,
    "EXECUTION_BLOCK_POLICY": {},
    "_LADDER": {"s0_scour": ("fixture-dataset",)},
    "BLOCK_REFERENCE_MANIFEST_FIELDS": BLOCK_REFERENCE_MANIFEST_FIELDS,
    "BLOCK_REFERENCE_SHA256": "",
    "_EXECUTION_ATTESTATION": {
        "runtime": fixture_runtime,
        "receipt_sha256": "3" * 64,
    },
    "_CAPACITY_PREFLIGHT_RECEIPT": {"receipt_sha256": "c" * 64},
    "ALL_ARCHITECTURES": [
        {"name_short": "PAA_NHiTS"},
        {"name_short": "PAA_CNN"},
    ],
    "ALL_DOFS": tuple(range(8)),
    "SEEDS": (42, 1337, 2026),
    "N_TRIALS": 100,
    # Reproduce an attribution follower (s11--s15): its LOCAL switch is false
    # even though the block anchor selected its reference exhaustively.
    "EXHAUSTIVE_PAIRS": False,
    "HYPERPARAMETER_POLICY": {"frozen_singleton": {"n_trials": 1}},
    "hyperparameter_json_sha256": canonical_json_sha256,
    "hyperparameter_policy_sha256": lambda: "d" * 64,
    "execution_block_for_stage": lambda *_args: ("l60", "s0_scour"),
    "validate_execution_runtime": lambda value: value,
    "validate_block_reference_manifest": validate_block_reference_manifest,
    "_parse_json_object_snapshot": lambda path, _label: json.loads(
        Path(path).read_text(encoding="utf-8")
    ),
}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[loader], type_ignores=[])
        ),
        str(DRIVER_PATH),
        "exec",
    ),
    loader_namespace,
)
fixture_loader = loader_namespace["_load_block_reference_manifest"]
reference_fixture = {
    "champion_arch": "PAA_CNN",
    "selected_at_stage": "s0_scour",
    "dataset": "fixture-dataset",
    "schema": "fixture-schema",
    "run_tag": RUN_TAG,
    "seeds": [42, 1337, 2026],
    "n_trials": 100,
    "candidate_n_trials": 1,
    "exhaustive_pairs": True,
    "protocol_core_hash": CORE_SHA,
    "protocol_core": FIXTURE_PROTOCOL_CORE,
    "protocol_hash": "2" * 64,
    "execution_runtime": fixture_runtime,
    "execution_environment_sha256": "1" * 64,
    "execution_receipt_sha256": "3" * 64,
    "capacity_preflight_receipt_sha256": "c" * 64,
    "hyperparameter_manifest_sha256": "4" * 64,
    "hyperparameter_policy_sha256": "d" * 64,
    "champion_pair": [1, 3],
    "pair_select_metric": "inner_val_mse",
    "per_arch_median_single_dof_mse": {
        "PAA_NHiTS": 1.1,
        "PAA_CNN": 1.0,
    },
    "frozen_selection_sha256": "5" * 64,
}


def _loader_rejects() -> bool:
    try:
        fixture_loader()
    except SystemExit:
        return True
    return False


with tempfile.TemporaryDirectory(prefix="reference-auth-") as tmp:
    reference_path = Path(tmp) / "reference.json"
    loader_namespace["CHAMPION_MANIFEST"] = str(reference_path)

    def write_reference(value: dict, *, pin: bool = True) -> None:
        reference_path.write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )
        if pin:
            loader_namespace["BLOCK_REFERENCE_SHA256"] = (
                canonical_json_sha256(value)
            )

    write_reference(reference_fixture)
    check(
        "exact content-addressed block reference is accepted",
        fixture_loader() == reference_fixture,
    )
    nonexhaustive_anchor = copy.deepcopy(reference_fixture)
    nonexhaustive_anchor["exhaustive_pairs"] = False
    write_reference(nonexhaustive_anchor)
    check(
        "reference rejects a non-exhaustive anchor even on a frozen follower",
        _loader_rejects(),
    )
    valid_arch_mutation = copy.deepcopy(reference_fixture)
    valid_arch_mutation["champion_arch"] = "PAA_NHiTS"
    write_reference(valid_arch_mutation, pin=False)
    check(
        "another registered architecture is rejected against the frozen pin",
        _loader_rejects(),
    )
    valid_pair_mutation = copy.deepcopy(reference_fixture)
    valid_pair_mutation["champion_pair"] = [0, 2]
    write_reference(valid_pair_mutation, pin=False)
    check(
        "another valid pair is rejected against the frozen pin",
        _loader_rejects(),
    )
    foreign_capacity = copy.deepcopy(reference_fixture)
    foreign_capacity["capacity_preflight_receipt_sha256"] = "6" * 64
    write_reference(foreign_capacity)
    check(
        "rehashed reference still cannot substitute a capacity receipt",
        _loader_rejects(),
    )
    extra_field = {**reference_fixture, "unexpected": True}
    write_reference(extra_field)
    check(
        "rehashed reference with an extra field violates the exact schema",
        _loader_rejects(),
    )
    boolean_pair = copy.deepcopy(reference_fixture)
    boolean_pair["champion_pair"] = [True, 3]
    write_reference(boolean_pair)
    check(
        "rehashed reference rejects boolean DOF indices",
        _loader_rejects(),
    )
    boolean_budget = copy.deepcopy(reference_fixture)
    boolean_budget["candidate_n_trials"] = True
    write_reference(boolean_budget)
    check(
        "rehashed reference rejects boolean candidate budgets",
        _loader_rejects(),
    )
    negative_median = copy.deepcopy(reference_fixture)
    negative_median["per_arch_median_single_dof_mse"]["PAA_CNN"] = -0.1
    write_reference(negative_median)
    check(
        "rehashed reference rejects negative selection diagnostics",
        _loader_rejects(),
    )

print("\n--- A3. immutable and monotonic evidence publication ---")
writer_names = (
    "_canonical_pretty_json_bytes",
    "_fsync_directory",
    "_write_json_immutable",
    "_parse_json_object_snapshot",
    "_protocol_record_lock",
    "_protocol_record_transition_allowed",
    "_write_protocol_record_monotonic",
    "_validate_pair",
    "_frozen_selection_sha256_for_reference",
    "_write_champion_arch",
)
writer_namespace = {
    "json": json,
    "math": __import__("math"),
    "os": os,
    "Path": Path,
    "tempfile": tempfile,
    "_read_regular_file": _read_regular_file,
}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[_function(name) for name in writer_names],
                type_ignores=[],
            )
        ),
        str(DRIVER_PATH),
        "exec",
    ),
    writer_namespace,
)
with tempfile.TemporaryDirectory(prefix="immutable-evidence-") as tmp:
    immutable_path = Path(tmp) / "frozen.json"
    immutable_writer = writer_namespace["_write_json_immutable"]
    immutable_writer(str(immutable_path), {"winner": [1, 3]})
    first_bytes = immutable_path.read_bytes()
    immutable_writer(str(immutable_path), {"winner": [1, 3]})
    check(
        "immutable writer accepts an exact restart",
        immutable_path.read_bytes() == first_bytes,
    )
    rejects(
        "immutable writer rejects a differing same-path selection",
        lambda: immutable_writer(str(immutable_path), {"winner": [0, 2]}),
    )
    check(
        "rejected immutable overwrite preserves prior bytes",
        immutable_path.read_bytes() == first_bytes,
    )

    protocol_path = Path(tmp) / "protocol_descriptor.json"
    monotonic_writer = writer_namespace[
        "_write_protocol_record_monotonic"
    ]
    initial = {
        "protocol_hash": "a" * 64,
        "descriptor": {"fixed": True},
        "hyperparameter_manifest_sha256": None,
        "block_reference_manifest_sha256": None,
    }
    monotonic_writer(str(protocol_path), initial)
    with_hpo = copy.deepcopy(initial)
    with_hpo["hyperparameter_manifest_sha256"] = "b" * 64
    monotonic_writer(str(protocol_path), with_hpo)
    with_reference = copy.deepcopy(with_hpo)
    with_reference["block_reference_manifest_sha256"] = "c" * 64
    monotonic_writer(str(protocol_path), with_reference)
    final_bytes = protocol_path.read_bytes()
    monotonic_writer(str(protocol_path), with_reference)
    check(
        "protocol record permits only the two registered None-to-SHA steps",
        protocol_path.read_bytes() == final_bytes,
    )
    changed_reference = copy.deepcopy(with_reference)
    changed_reference["block_reference_manifest_sha256"] = "d" * 64
    rejects(
        "protocol record rejects reference SHA substitution",
        lambda: monotonic_writer(str(protocol_path), changed_reference),
    )
    changed_descriptor = copy.deepcopy(with_reference)
    changed_descriptor["descriptor"]["fixed"] = False
    rejects(
        "protocol record rejects mutation outside monotonic bindings",
        lambda: monotonic_writer(str(protocol_path), changed_descriptor),
    )
    regressed_hpo = copy.deepcopy(with_reference)
    regressed_hpo["hyperparameter_manifest_sha256"] = None
    rejects(
        "protocol record rejects SHA-to-null regression",
        lambda: monotonic_writer(str(protocol_path), regressed_hpo),
    )
    check(
        "rejected protocol mutations preserve prior evidence",
        protocol_path.read_bytes() == final_bytes,
    )

    stale_path = Path(tmp) / "stale_lock_protocol.json"
    stale_path.write_bytes(
        writer_namespace["_canonical_pretty_json_bytes"](initial)
    )
    stale_lock = stale_path.with_name(
        f".{stale_path.name}.update.lock"
    )
    stale_lock.write_bytes(b"")
    try:
        monotonic_writer(str(stale_path), with_hpo)
        stale_recovered = True
    except RuntimeError:
        stale_recovered = False
    check(
        "orphaned legacy lock file cannot block a valid crash restart",
        stale_recovered
        and json.loads(stale_path.read_text(encoding="ascii")) == with_hpo
        and stale_lock.is_file(),
    )

    frozen_runtime = {
        "execution_block": "l60",
        "anchor_stage": "s0_scour",
        "execution_environment_sha256": "1" * 64,
    }
    writer_namespace.update({
        "SUMMARY_DIR": tmp,
        "STAGE": "s0_scour",
        "DEPLOYMENT_SELECTION_STAGES": {"s16_all", "s23_all4"},
        "IDX_TO_DOF_NAME": [f"dof_{index}" for index in range(8)],
        "ALL_DOFS": list(range(8)),
        "_EXECUTION_ATTESTATION": {
            "runtime": frozen_runtime,
            "receipt_sha256": "3" * 64,
        },
        "_CAPACITY_PREFLIGHT_RECEIPT": {
            "receipt_sha256": "c" * 64,
        },
        "_HYPERPARAMETER_MANIFEST": {"fixture": True},
        "_HYPERPARAMETER_MANIFEST_SHA256": "4" * 64,
        "FROZEN_SINGLETON_MODE": "frozen_singleton",
        "RUN_TAG": RUN_TAG,
        "PROTOCOL_HASH": "2" * 64,
        "PROTOCOL_CORE_HASH": CORE_SHA,
        "TRAIN_PROTOCOL": TRAIN_PROTOCOL,
        "validate_execution_runtime": lambda value: value,
        "hyperparameter_json_sha256": canonical_json_sha256,
    })
    frozen_value = {
        "stage": "s0_scour",
        "architecture": "PAA_CNN",
        "dofs": "dof_1+dof_3",
        "selected_pair": [1, 3],
        "inner_val_mse": 1.0,
        "selection_metric": TRAIN_PROTOCOL["objective"],
        "deployment_selection": False,
        "protocol_hash": "2" * 64,
        "protocol_core_hash": CORE_SHA,
        "execution_runtime": frozen_runtime,
        "execution_environment_sha256": "1" * 64,
        "execution_receipt_sha256": "3" * 64,
        "capacity_preflight_receipt_sha256": "c" * 64,
        "hyperparameter_mode": "frozen_singleton",
        "campaign_run_tag": RUN_TAG,
        "block_reference_manifest_sha256": None,
        "hyperparameter_manifest_sha256": "4" * 64,
        "hyperparameter_source_json":
            '{"architecture":"PAA_CNN"}',
        "pre_registered_comparators": [
            "PAA_CNN on dof_1+dof_3"
        ],
    }
    frozen_path = Path(tmp) / "frozen_selection.json"

    def write_frozen(value: dict) -> None:
        frozen_path.write_text(
            json.dumps(value, sort_keys=True), encoding="utf-8"
        )

    frozen_validator = writer_namespace[
        "_frozen_selection_sha256_for_reference"
    ]
    write_frozen(frozen_value)
    check(
        "reference publication accepts the exact frozen-selection contract",
        frozen_validator("PAA_CNN", [1, 3])
        == canonical_json_sha256(frozen_value),
    )
    frozen_bool_pair = copy.deepcopy(frozen_value)
    frozen_bool_pair["selected_pair"] = [True, 3]
    write_frozen(frozen_bool_pair)
    rejects(
        "reference publication rejects boolean frozen-selection DOFs",
        lambda: frozen_validator("PAA_CNN", [1, 3]),
    )
    frozen_negative = copy.deepcopy(frozen_value)
    frozen_negative["inner_val_mse"] = -0.1
    write_frozen(frozen_negative)
    rejects(
        "reference publication rejects negative frozen inner-validation loss",
        lambda: frozen_validator("PAA_CNN", [1, 3]),
    )
    duplicate_raw = (
        '{"stage":"s0_scour",'
        + json.dumps(frozen_value, sort_keys=True)[1:]
    )
    frozen_path.write_text(duplicate_raw, encoding="utf-8")
    rejects(
        "reference publication rejects duplicate frozen-selection JSON keys",
        lambda: frozen_validator("PAA_CNN", [1, 3]),
    )

    rejected_manifest = Path(tmp) / "rejected_reference.json"
    writer_namespace.update({
        "CHAMPION_MANIFEST": str(rejected_manifest),
        "DATASET": "fixture-dataset",
        "SCHEMA_TAG": "fixture-schema",
        "SEEDS": (42, 1337, 2026),
        "N_TRIALS": 100,
        "EXHAUSTIVE_PAIRS": True,
        "HYPERPARAMETER_POLICY": {
            "frozen_singleton": {"n_trials": 1}
        },
        "PROTOCOL_CORE_DESC": FIXTURE_PROTOCOL_CORE,
        "BLOCK_REFERENCE_SHA256": "",
        "hyperparameter_policy_sha256": lambda: "d" * 64,
        "_frozen_selection_sha256_for_reference":
            lambda *_args: "5" * 64,
        "_validate_reference_payload_for_publication":
            lambda _payload: (_ for _ in ()).throw(
                RuntimeError("synthetic invalid trust root")
            ),
    })
    rejects(
        "invalid champion payload is rejected before immutable publication",
        lambda: writer_namespace["_write_champion_arch"](
            "PAA_CNN",
            {"PAA_NHiTS": 1.1, "PAA_CNN": -0.1},
            champion_pair=[1, 3],
        ),
    )
    check(
        "rejected champion payload creates no trust-root file",
        not rejected_manifest.exists(),
    )

print("\n--- B. hash-carried allocation policy ---")
policy = canonical_execution_block_policy(EXECUTION_BLOCK_POLICY)
for stage in (
    "s0_scour",
    "s11_bear",
    "s12_crack",
    "s13_bearcrack",
    "s14_prof",
    "s15_track",
    "s16_all",
):
    check(
        f"{stage} maps to L60 anchored at s0",
        execution_block_for_stage(stage, policy) == ("l60", "s0_scour"),
    )
for stage in ("s21_scour4", "s22_bearcrack4", "s23_all4"):
    check(
        f"{stage} maps to L99 anchored at s21",
        execution_block_for_stage(stage, policy)
        == ("l99", "s21_scour4"),
    )
cross = policy["cross_block_inference"]["s0_scour_to_s21_scour4"]
check(
    "s0->s21 is descriptive and non-confirmatory",
    cross["status"] == "descriptive_nonconfirmatory"
    and cross["confirmatory"] is False,
)
rejects(
    "unknown stage is rejected",
    lambda: execution_block_for_stage("s99_unknown", policy),
)
invalid_policy = copy.deepcopy(policy)
invalid_policy["blocks"]["l60"]["stages"].remove("s16_all")
rejects(
    "incomplete stage allocation is rejected",
    lambda: canonical_execution_block_policy(invalid_policy),
)
invalid_policy = copy.deepcopy(policy)
invalid_policy["cross_block_inference"]["s0_scour_to_s21_scour4"][
    "confirmatory"
] = True
rejects(
    "confirmatory cross-block claim is rejected",
    lambda: canonical_execution_block_policy(invalid_policy),
)

print("\n--- C. canonical execution identity ---")
environment = fixture_environment()
environment_sha = execution_environment_sha256(environment)
runtime = {
    "schema": "ttbi-execution-runtime-binding-v1",
    "execution_block": "l60",
    "anchor_stage": "s0_scour",
    "execution_environment_sha256": environment_sha,
    "execution_environment_descriptor": environment,
}
check(
    "runtime binding validates and reproduces its SHA",
    validate_execution_runtime(runtime)["execution_environment_sha256"]
    == environment_sha,
)
tampered_runtime = copy.deepcopy(runtime)
tampered_runtime["execution_environment_descriptor"]["host"][
    "hostname"
] = "foreign-host"
rejects(
    "descriptor mutation without SHA update is rejected",
    lambda: validate_execution_runtime(tampered_runtime),
)
missing_uuid_environment = fixture_environment()
missing_uuid_environment["accelerator"]["uuid"] = None
rejects(
    "CUDA identity without a physical GPU UUID fails closed",
    lambda: execution_environment_sha256(missing_uuid_environment),
)
missing_driver_environment = fixture_environment()
missing_driver_environment["accelerator"]["driver_version"] = None
rejects(
    "CUDA identity without an NVIDIA driver version fails closed",
    lambda: execution_environment_sha256(missing_driver_environment),
)
wrong_anchor = copy.deepcopy(runtime)
wrong_anchor["anchor_stage"] = "s21_scour4"
rejects(
    "block/anchor disagreement is rejected",
    lambda: validate_execution_runtime(wrong_anchor),
)

set_global_seed(42, TRAIN_PROTOCOL["determinism"])
actual = current_execution_environment()
actual_sha = execution_environment_sha256(actual)
check(
    "actual runtime records host, machine, accelerator and numeric state",
    len(actual_sha) == 64
    and bool(actual["host"]["hostname"])
    and bool(actual["host"]["machine"])
    and {
        "name",
        "uuid",
        "compute_capability",
        "sm_count",
        "total_memory_bytes",
        "driver_version",
    } <= set(actual["accelerator"])
    and actual["numeric_stack"]["deterministic_algorithms"] is True
    and actual["numeric_stack"]["deterministic_warn_only"] is False
    and actual["numeric_stack"]["cudnn_deterministic"] is True
    and actual["numeric_stack"]["cudnn_benchmark"] is False,
)

print("\n--- D. atomic per-block receipts ---")
with tempfile.TemporaryDirectory(prefix="execution-block-") as tmp:
    receipt_dir = Path(tmp, "receipts")
    anchor = enforce_execution_block(
        stage="s0_scour",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_dir,
        descriptor=environment,
    )
    receipt = Path(anchor["receipt_path"])
    check(
        "L60 anchor atomically creates a regular canonical receipt",
        receipt.is_file()
        and not receipt.is_symlink()
        and stat.S_ISREG(os.lstat(receipt).st_mode)
        and receipt.read_bytes().endswith(b"\n")
        and (
            json.dumps(
                json.loads(receipt.read_text(encoding="ascii")),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        == receipt.read_bytes(),
    )
    follower = enforce_execution_block(
        stage="s16_all",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_dir,
        descriptor=environment,
    )
    check(
        "later L60 node accepts the exact anchor identity",
        follower["runtime"] == anchor["runtime"]
        and follower["receipt_sha256"] == anchor["receipt_sha256"],
    )
    check(
        "L60 follower accepts only the exact s0 champion execution lineage",
        validate_block_reference_execution(
            selection_runtime=anchor["runtime"],
            selection_environment_sha256=anchor["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256=anchor["receipt_sha256"],
            current_attestation=follower,
            current_stage="s16_all",
            policy=policy,
        ) == "same_block_exact",
    )
    foreign_selection_environment = fixture_environment(
        hostname="selection-on-host-b", uuid="GPU-selection-b"
    )
    foreign_selection_runtime = {
        **anchor["runtime"],
        "execution_environment_descriptor": foreign_selection_environment,
        "execution_environment_sha256":
            execution_environment_sha256(foreign_selection_environment),
    }
    rejects(
        "GPU-B s0 reference cannot enter a GPU-A L60 receipt",
        lambda: validate_block_reference_execution(
            selection_runtime=foreign_selection_runtime,
            selection_environment_sha256=foreign_selection_runtime[
                "execution_environment_sha256"
            ],
            selection_receipt_sha256="c" * 64,
            current_attestation=follower,
            current_stage="s11_bear",
            policy=policy,
        ),
        "reference selection execution identity differs",
    )
    rejects(
        "s0 reference receipt SHA cannot be substituted inside L60",
        lambda: validate_block_reference_execution(
            selection_runtime=anchor["runtime"],
            selection_environment_sha256=anchor["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256="c" * 64,
            current_attestation=follower,
            current_stage="s11_bear",
            policy=policy,
        ),
        "reference selection execution identity differs",
    )
    rejects(
        "same block rejects a different physical host",
        lambda: enforce_execution_block(
            stage="s11_bear",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=receipt_dir,
            descriptor=fixture_environment(hostname="foreign-host"),
        ),
        "execution receipt mismatch",
    )
    rejects(
        "same block rejects a different GPU UUID",
        lambda: enforce_execution_block(
            stage="s11_bear",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=receipt_dir,
            descriptor=fixture_environment(uuid="GPU-fixture-b"),
        ),
        "execution receipt mismatch",
    )
    rejects(
        "later node cannot reuse a receipt from another core protocol",
        lambda: enforce_execution_block(
            stage="s11_bear",
            policy=policy,
            protocol_core_hash="b" * 64,
            run_tag=RUN_TAG,
            receipt_dir=receipt_dir,
            descriptor=environment,
        ),
        "Run its anchor",
    )

    # L99 deliberately has its own anchor/receipt and can use another machine.
    l99_environment = fixture_environment(
        hostname="fixture-l99-host", uuid="GPU-fixture-l99"
    )
    l99 = enforce_execution_block(
        stage="s21_scour4",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_dir,
        descriptor=l99_environment,
    )
    check(
        "independent L99 anchor may create a different execution identity",
        l99["runtime"]["execution_block"] == "l99"
        and l99["runtime"]["execution_environment_sha256"]
        != anchor["runtime"]["execution_environment_sha256"],
    )
    l99_follower = enforce_execution_block(
        stage="s22_bearcrack4",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_dir,
        descriptor=l99_environment,
    )
    check(
        "L99 follower accepts only its exact s21 reference lineage",
        validate_block_reference_execution(
            selection_runtime=l99["runtime"],
            selection_environment_sha256=l99["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256=l99["receipt_sha256"],
            current_attestation=l99_follower,
            current_stage="s22_bearcrack4",
            policy=policy,
        ) == "same_block_exact",
    )
    rejects(
        "L99 follower rejects an s0/L60 reference manifest",
        lambda: validate_block_reference_execution(
            selection_runtime=anchor["runtime"],
            selection_environment_sha256=anchor["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256=anchor["receipt_sha256"],
            current_attestation=l99_follower,
            current_stage="s22_bearcrack4",
            policy=policy,
        ),
        "not the registered l99/s21_scour4 block anchor",
    )
with tempfile.TemporaryDirectory(prefix="execution-missing-") as tmp:
    rejects(
        "non-anchor cannot create a missing block receipt",
        lambda: enforce_execution_block(
            stage="s12_crack",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=tmp,
            descriptor=environment,
        ),
        "Run its anchor",
    )

with tempfile.TemporaryDirectory(prefix="execution-malformed-") as tmp:
    created = enforce_execution_block(
        stage="s0_scour",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=tmp,
        descriptor=environment,
    )
    path = Path(created["receipt_path"])
    path.write_text("{malformed", encoding="ascii")
    rejects(
        "malformed receipt is rejected",
        lambda: enforce_execution_block(
            stage="s11_bear",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=tmp,
            descriptor=environment,
        ),
        "malformed execution receipt",
    )

with tempfile.TemporaryDirectory(prefix="execution-noncanonical-") as tmp:
    created = enforce_execution_block(
        stage="s0_scour",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=tmp,
        descriptor=environment,
    )
    path = Path(created["receipt_path"])
    value = json.loads(path.read_text(encoding="ascii"))
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    rejects(
        "noncanonical receipt bytes are rejected",
        lambda: enforce_execution_block(
            stage="s11_bear",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=tmp,
            descriptor=environment,
        ),
        "not exact canonical JSON",
    )

with tempfile.TemporaryDirectory(prefix="execution-nonregular-") as tmp:
    path = _receipt_path(
        tmp,
        block="l60",
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
    )
    path.mkdir(parents=True)
    rejects(
        "nonregular receipt target is rejected",
        lambda: enforce_execution_block(
            stage="s0_scour",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=tmp,
            descriptor=environment,
        ),
        "not a regular file",
    )

with tempfile.TemporaryDirectory(prefix="execution-symlink-") as tmp:
    path = _receipt_path(
        tmp,
        block="l60",
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    target = Path(tmp, "target.json")
    target.write_text("{}\n", encoding="ascii")
    try:
        path.symlink_to(target)
    except OSError:
        # Windows without Developer Mode cannot create the filesystem fixture.
        # Exercise the same behavioural branch by making lstat report S_IFLNK.
        with mock.patch(
            "core.execution_environment.os.lstat",
            return_value=SimpleNamespace(st_mode=stat.S_IFLNK),
        ):
            rejects(
                "symlink receipt target is rejected (mocked Windows lstat)",
                lambda: _read_regular_file(target),
                "not a regular file",
            )
    else:
        rejects(
            "symlink receipt target is rejected",
            lambda: enforce_execution_block(
                stage="s0_scour",
                policy=policy,
                protocol_core_hash=CORE_SHA,
                run_tag=RUN_TAG,
                receipt_dir=tmp,
                descriptor=environment,
            ),
            "not a regular file",
        )

print()
if fails:
    raise SystemExit(f"EXECUTION BLOCKING: {fails} FAILURE(S)")
print("EXECUTION BLOCKING: ALL PASS")
