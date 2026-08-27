"""Adversarial checks for the four-stage hyperparameter execution policy.

Run: ``python check_hyperparameter_policy.py`` after local capability checks.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import optuna
import core.hyperparameter_policy as hyperparameter_policy_module

from core.capacity_preflight import registered_capacity_cases
from core.execution_environment import (
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
from core.hyperparameter_policy import (
    ANCHOR_HPO_MODE,
    ARCHITECTURES,
    FROZEN_SINGLETON_MODE,
    HPO_ANCHOR_INPUT,
    SEEDS,
    STUDY_IDENTITY_SCHEMA,
    HyperparameterPolicyError,
    build_manifest,
    build_manifest_entry,
    canonical_json_sha256,
    derive_execution_plan,
    load_manifest,
    select_frozen_config,
    validate_manifest,
    validate_run_plan,
    validate_terminal_study,
    write_manifest,
)
from core.paper1_training_contract import FACTORIAL_CELLS
from core.protocol import protocol_hash
from training.pipeline import _create_or_resume_study, _execute_protocol_study
from training.trainer import _suggest_params


RUN_TAG = "paper1-policy-fixture"
EXECUTION_RECEIPT_SHA = "e" * 64
SOURCE_SHA = "d" * 64
SOURCE_COUNT = 123
FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def rejects(name: str, function) -> None:
    try:
        function()
    except (HyperparameterPolicyError, RuntimeError, ValueError):
        check(name, True)
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation was accepted")


def fixture_environment(
    *,
    hostname: str = "qualification-host-a",
    gpu_name: str = "qualification-gpu-a",
    gpu_uuid: str = "GPU-qualification-a",
    torch_version: str = "fixture-a",
    cuda_runtime_version: str = "12.8",
) -> dict:
    return {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": hostname,
            "machine": "AMD64",
            "system": "Windows",
            "platform": f"qualification-fixture-{hostname}",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": gpu_name,
            "uuid": gpu_uuid,
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 40,
            "total_memory_bytes": 8_589_934_592,
            "driver_version": "fixture",
        },
        "numeric_stack": {
            "torch_version": torch_version,
            "cuda_runtime_version": cuda_runtime_version,
            "cudnn_version": 1,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }


def fixture_runtime(
    environment: dict | None = None,
    *,
    execution_block: str = "f40s",
    anchor_stage: str = "F40-S",
) -> dict:
    if environment is None:
        environment = fixture_environment()
    compatibility = execution_compatibility_descriptor(environment)
    return {
        "schema": "ttbi-execution-runtime-binding-v2",
        "execution_block": execution_block,
        "anchor_stage": anchor_stage,
        "execution_environment_sha256": execution_environment_sha256(environment),
        "execution_environment_descriptor": environment,
        "execution_compatibility_sha256":
            execution_compatibility_sha256(environment),
        "execution_compatibility_descriptor": compatibility,
    }


def descriptors(dataset: str = "F40-S_fixture") -> tuple[dict, dict]:
    core = {"qualification": "paper1-hyperparameter-policy"}
    full = {
        "core": core,
        "rung": {
            "stage": "F40-S",
            "dataset": dataset,
            "execution_block": "f40s",
            "execution_anchor": "F40-S",
        },
    }
    return core, full


def architecture_config(architecture: str) -> dict:
    cell = next(cell for cell in FACTORIAL_CELLS if cell.cell_id == architecture)
    return {
        "name_short": cell.cell_id,
        "method": cell.representation,
        "model_type": "1D_MODULAR",
        "use_space2vec": cell.position_encoding,
        "use_lstm": cell.lstm,
        "use_nhits": cell.multi_rate_pooling,
    }


def anchor_config(
    architecture: str = ARCHITECTURES[0],
    seed: int = SEEDS[0],
    *,
    dofs: list[int] | None = None,
) -> dict:
    core, full = descriptors()
    return {
        "name": f"{architecture}_{seed}",
        **architecture_config(architecture),
        "seed": seed,
        "dofs": list(HPO_ANCHOR_INPUT if dofs is None else dofs),
        "protocol_descriptor": full,
        "protocol_hash": protocol_hash(full),
        "protocol_core_hash": protocol_hash(core),
        "hyperparameter_mode": ANCHOR_HPO_MODE,
        "execution_runtime": fixture_runtime(),
        "campaign_run_tag": RUN_TAG,
        "execution_receipt_sha256": EXECUTION_RECEIPT_SHA,
        "block_reference_manifest_sha256": None,
    }


def registered_params() -> dict[str, dict]:
    return {
        architecture: params
        for architecture, _config, params in registered_capacity_cases()
    }


def manifest_entries(
    runtimes: tuple[dict, ...],
    protocol_hash_value: str,
) -> list[dict]:
    if not runtimes:
        raise ValueError("at least one runtime fixture is required")
    params_by_architecture = registered_params()
    entries = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for seed_index, seed in enumerate(SEEDS):
            runtime = runtimes[
                (architecture_index + seed_index) % len(runtimes)
            ]
            params = params_by_architecture[architecture]
            identity = {
                "schema": STUDY_IDENTITY_SCHEMA,
                "execution_block": "f40s",
                "anchor_stage": "F40-S",
                "architecture": architecture,
                "seed": seed,
                "active_dofs": list(HPO_ANCHOR_INPUT),
                "study_name": f"{architecture}_{seed}",
                "protocol_hash": protocol_hash_value,
                "dataset": "F40-S_fixture",
                "model_name": f"{architecture}_{seed}",
                "execution_environment_sha256":
                    runtime["execution_environment_sha256"],
                "execution_compatibility_sha256":
                    runtime["execution_compatibility_sha256"],
                "campaign_run_tag": RUN_TAG,
                "execution_receipt_sha256": EXECUTION_RECEIPT_SHA,
                "study_protocol_record_sha256":
                    canonical_json_sha256([architecture, seed, "record"]),
                "effective_n_trials": 100,
                "effective_use_pruner": True,
                "terminal_counts": {
                    "COMPLETE": 40,
                    "PRUNED": 60,
                    "FAIL": 0,
                    "RUNNING": 0,
                    "WAITING": 0,
                    "total": 100,
                },
                "best_trial_number": seed % 100,
                "best_trial_value": float(seed) / 1_000_000,
                "best_params_sha256": canonical_json_sha256(params),
            }
            entries.append(build_manifest_entry(
                study_identity=identity,
                params=params,
            ))
    return entries


def main() -> None:
    print("PAPER1 HYPERPARAMETER POLICY CHECKS")
    # Keep this behavioural unit fixture independent of the concurrently
    # reviewed bundle inventory; production calls retain the live source root.
    hyperparameter_policy_module._runtime_source_identity = (  # noqa: SLF001
        lambda: (SOURCE_SHA, SOURCE_COUNT)
    )
    runtime = fixture_runtime()
    portable_runtime = fixture_runtime(fixture_environment(
        hostname="qualification-host-b",
        gpu_name="qualification-gpu-b",
        gpu_uuid="GPU-qualification-b",
        torch_version="fixture-b",
        cuda_runtime_version="13.0",
    ))
    anchor = anchor_config()
    plan = derive_execution_plan(
        anchor,
        dataset_name="F40-S_fixture",
        requested_n_trials=1,
        requested_use_pruner=False,
        execution_runtime=runtime,
    )
    check(
        "factorial anchor derives exact 100-trial/pruned plan",
        plan["mode"] == ANCHOR_HPO_MODE
        and plan["stage"] == "F40-S"
        and plan["active_dofs"] == [1]
        and plan["effective_n_trials"] == 100
        and plan["effective_use_pruner"] is True
        and plan["selection_artifact_sha256"] is None
        and plan["selection_slot"] is None
        and validate_run_plan(plan) == plan,
    )
    missing_run_tag = deepcopy(plan)
    missing_run_tag["campaign_run_tag"] = None
    rejects(
        "campaign run plan without exact run tag",
        lambda: validate_run_plan(missing_run_tag),
    )
    invalid_execution_receipt = deepcopy(plan)
    invalid_execution_receipt["execution_receipt_sha256"] = "invalid"
    rejects(
        "campaign run plan with invalid execution receipt",
        lambda: validate_run_plan(invalid_execution_receipt),
    )
    wrong_anchor = anchor_config(dofs=[0])
    rejects(
        "factorial HPO on a foreign channel is rejected",
        lambda: derive_execution_plan(
            wrong_anchor,
            dataset_name="F40-S_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    proxy_input = anchor_config(dofs=[3])
    proxy_input["hyperparameter_mode"] = FROZEN_SINGLETON_MODE
    rejects(
        "constrained-wheelset proxy cannot enter a campaign learning plan",
        lambda: derive_execution_plan(
            proxy_input,
            dataset_name="F40-S_fixture",
            requested_n_trials=1,
            requested_use_pruner=False,
            execution_runtime=runtime,
        ),
    )

    entries = manifest_entries(
        (runtime, portable_runtime), anchor["protocol_hash"]
    )
    manifest, manifest_sha = build_manifest(
        entries,
        execution_runtime=runtime,
        protocol_core_hash=anchor["protocol_core_hash"],
        anchor_protocol_hash=anchor["protocol_hash"],
        anchor_dataset="F40-S_fixture",
        run_tag=RUN_TAG,
        execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
        source_root_sha256=SOURCE_SHA,
        source_file_count=SOURCE_COUNT,
    )
    check(
        "anchor manifest is exact 16 cells x five portable-host restarts",
        len(entries) == len(ARCHITECTURES) * len(SEEDS) == 80
        and {
            entry["study_identity"]["execution_environment_sha256"]
            for entry in entries
        } == {
            runtime["execution_environment_sha256"],
            portable_runtime["execution_environment_sha256"],
        }
        and {
            entry["study_identity"]["execution_compatibility_sha256"]
            for entry in entries
        } == {
            runtime["execution_compatibility_sha256"],
            portable_runtime["execution_compatibility_sha256"],
        }
        and manifest_sha == canonical_json_sha256(manifest)
        and validate_manifest(
            manifest,
            expected_runtime=portable_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        ) == manifest,
    )
    missing = deepcopy(manifest); missing["entries"].pop()
    rejects("incomplete anchor manifest", lambda: validate_manifest(missing))
    duplicate = deepcopy(manifest); duplicate["entries"][-1] = deepcopy(duplicate["entries"][0])
    rejects("duplicate anchor arm", lambda: validate_manifest(duplicate))
    failed = deepcopy(manifest)
    failed["entries"][0]["study_identity"]["terminal_counts"].update({
        "COMPLETE": 39, "FAIL": 1,
    })
    rejects("failed anchor trial", lambda: validate_manifest(failed))
    wrong_dof = deepcopy(manifest)
    wrong_dof["entries"][0]["study_identity"]["active_dofs"] = [0]
    rejects("anchor identity on wrong input", lambda: validate_manifest(wrong_dof))
    rejects(
        "portable host cannot substitute the logical run tag",
        lambda: validate_manifest(
            manifest,
            expected_runtime=portable_runtime,
            expected_run_tag=f"{RUN_TAG}-other",
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        ),
    )
    rejects(
        "portable host cannot substitute the block receipt",
        lambda: validate_manifest(
            manifest,
            expected_runtime=portable_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256="f" * 64,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        ),
    )
    rejects(
        "portable host cannot substitute the reviewed source root",
        lambda: validate_manifest(
            manifest,
            expected_runtime=portable_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256="c" * 64,
            expected_source_file_count=SOURCE_COUNT,
        ),
    )
    foreign_block_runtime = fixture_runtime(
        fixture_environment(
            hostname="qualification-host-l99",
            gpu_name="qualification-gpu-l99",
            gpu_uuid="GPU-qualification-l99",
        ),
        execution_block="l99s",
        anchor_stage="L99-S",
    )
    rejects(
        "portable execution does not permit a foreign logical block/stage",
        lambda: validate_manifest(
            manifest,
            expected_runtime=foreign_block_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        ),
    )
    wrong_protocol = deepcopy(manifest)
    identity = deepcopy(wrong_protocol["entries"][0]["study_identity"])
    identity["protocol_hash"] = "c" * 64
    wrong_protocol["entries"][0] = build_manifest_entry(
        study_identity=identity,
        params=wrong_protocol["entries"][0]["params"],
    )
    rejects(
        "restart from a portable host cannot substitute protocol identity",
        lambda: validate_manifest(wrong_protocol),
    )

    with tempfile.TemporaryDirectory(prefix="paper1-hpo-manifest-") as td:
        path = Path(td).resolve() / "manifest.json"
        written_sha = write_manifest(
            path,
            manifest,
            expected_runtime=portable_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        )
        loaded = load_manifest(
            path,
            expected_sha256=written_sha,
            expected_runtime=portable_runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
        )
        check("canonical manifest write/load roundtrip", loaded == manifest)
        path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
        rejects(
            "pretty-printed/noncanonical manifest",
            lambda: load_manifest(path, expected_sha256=written_sha),
        )

    frozen_fields = select_frozen_config(
        manifest,
        architecture=ARCHITECTURES[0],
        seed=SEEDS[0],
        expected_runtime=portable_runtime,
        expected_run_tag=RUN_TAG,
        expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
        expected_source_root_sha256=SOURCE_SHA,
        expected_source_file_count=SOURCE_COUNT,
    )
    frozen = anchor_config(dofs=[0, 2])
    frozen["hyperparameter_mode"] = FROZEN_SINGLETON_MODE
    frozen.update(frozen_fields)
    frozen_plan = derive_execution_plan(
        frozen,
        dataset_name="F40-S_fixture",
        requested_n_trials=100,
        requested_use_pruner=True,
        execution_runtime=portable_runtime,
    )
    check(
        "F40-S frozen screen derives one unpruned authenticated trial",
        frozen_plan["mode"] == FROZEN_SINGLETON_MODE
        and frozen_plan["effective_n_trials"] == 1
        and frozen_plan["effective_use_pruner"] is False
        and frozen_plan["hyperparameter_manifest_sha256"] == manifest_sha
        and frozen_plan["selection_artifact_sha256"] is None,
    )
    follower_without_reference = deepcopy(frozen_plan)
    follower_without_reference["stage"] = "F40-S-FOLLOWER"
    follower_without_reference["block_reference_manifest_sha256"] = None
    rejects(
        "follower run plan without block-reference digest",
        lambda: validate_run_plan(follower_without_reference),
    )
    tampered_frozen = deepcopy(frozen)
    first_param = next(iter(tampered_frozen["frozen_hyperparameters"]))
    tampered_frozen["frozen_hyperparameters"][first_param] = "tampered"
    rejects(
        "frozen parameters differ from authenticated manifest",
        lambda: derive_execution_plan(
            tampered_frozen,
            dataset_name="F40-S_fixture",
            requested_n_trials=1,
            requested_use_pruner=False,
            execution_runtime=runtime,
        ),
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial):
        params = _suggest_params(trial, frozen)
        check("singleton live search reproduces manifest params", params == frozen[
            "frozen_hyperparameters"
        ])
        return 0.0

    study.optimize(objective, n_trials=1)
    check(
        "singleton Optuna terminal gate",
        validate_terminal_study(study, frozen_plan)["COMPLETE"] == 1,
    )
    selected_pruner = _create_or_resume_study(
        "selected-pruner-fixture",
        "sqlite:///:memory:",
        100,
        sampler_seed=42,
        use_pruner=True,
    )
    frozen_pruner = _create_or_resume_study(
        "frozen-pruner-fixture",
        "sqlite:///:memory:",
        1,
        sampler_seed=42,
        use_pruner=False,
        force_nop_pruner=True,
    )
    check(
        "live Optuna constructors preserve registered pruner distinction",
        isinstance(selected_pruner.pruner, optuna.pruners.SuccessiveHalvingPruner)
        and isinstance(frozen_pruner.pruner, optuna.pruners.NopPruner),
    )
    rejects(
        "protocol helper rejects legacy plan",
        lambda: _execute_protocol_study(
            study,
            lambda _trial: 0.0,
            derive_execution_plan(
                {"name_short": "legacy", "dofs": [0]},
                dataset_name="legacy",
                requested_n_trials=1,
                requested_use_pruner=False,
            ),
        ),
    )
    stripped = {"name_short": ARCHITECTURES[0], "dofs": [1], "selection_slot": "x"}
    rejects(
        "campaign marker without protocol hash",
        lambda: derive_execution_plan(
            stripped,
            dataset_name="fixture",
            requested_n_trials=1,
            requested_use_pruner=False,
        ),
    )

    print()
    if FAILURES:
        raise SystemExit(f"HYPERPARAMETER POLICY: {FAILURES} CHECK(S) FAILED")
    print("HYPERPARAMETER POLICY: ALL PASS")


if __name__ == "__main__":
    main()
