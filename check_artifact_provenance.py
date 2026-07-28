"""Adversarial checks for Optuna protocol and champion-weight provenance.

Run with the campaign environment:
    python check_artifact_provenance.py
"""

from __future__ import annotations

import gc
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import optuna
import torch

from core.dataset import _cache_stem
from core.artifact_provenance import verify_standalone_dt_package
from core.capacity_preflight import (
    registered_capacity_cases,
    run_capacity_preflight,
)
from core.execution_environment import execution_environment_sha256
from core.hyperparameter_policy import (
    ANCHOR_HPO_MODE,
    FROZEN_SINGLETON_MODE,
    ARCHITECTURES,
    SEEDS,
    STUDY_IDENTITY_SCHEMA,
    build_manifest,
    build_manifest_entry,
    canonical_json_sha256,
    derive_execution_plan,
    select_frozen_config,
)
from core.protocol import OPTUNA_PROTOCOL, protocol_hash
from training.pipeline import (
    _stamp_study_protocol,
    export_digital_twin_package,
    verify_digital_twin_package,
)


fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def rejects(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        check(name, True)
    else:
        check(name, False)


def rejects_with_message(name: str, fn, message: str) -> None:
    """Require the intended fail-closed error, not an incidental exception."""

    try:
        fn()
    except RuntimeError as exc:
        check(name, message in str(exc))
    except Exception:
        check(name, False)
    else:
        check(name, False)


optuna.logging.set_verbosity(optuna.logging.WARNING)
CAMPAIGN_RUN_TAG = ""
EXECUTION_RECEIPT_SHA = "e" * 64
BLOCK_REFERENCE_A = "a" * 64
BLOCK_REFERENCE_B = "b" * 64
execution_descriptor = {
    "schema": "ttbi-execution-environment-v1",
    "host": {
        "hostname": "artifact-fixture-host",
        "machine": "AMD64",
        "system": "Windows",
        "platform": "Windows-fixture",
    },
    "accelerator": {
        "backend": "cuda",
        "device_index": 0,
        "name": "Fixture GPU",
        "uuid": "GPU-fixture-uuid",
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
execution_runtime = {
    "schema": "ttbi-execution-runtime-binding-v1",
    "execution_block": "l60",
    "anchor_stage": "s0_scour",
    "execution_environment_sha256":
        execution_environment_sha256(execution_descriptor),
    "execution_environment_descriptor": execution_descriptor,
}
protocol_core = {
    "protocol_version": 6,
    "unit_fixture": "artifact-provenance-r11",
}
protocol_descriptor = {
    "core": protocol_core,
    "rung": {
        "stage": "s0_scour",
        "dataset": "fixture",
        "execution_block": "l60",
        "execution_anchor": "s0_scour",
    },
    # Tuple is intentional: Optuna's SQLite JSON round-trip returns a list.
    "search_space_fixture": {"learning_rate": (1e-4, 1e-3)},
}
config = {
    "name": "artifact-fixture",
    "name_short": "PAA_CNN",
    "method": "PAA",
    "dofs": list(range(8)),
    "task": "regression",
    "target_supports": [2, 3],
    "bearing_targets": None,
    "discretization": 1,
    "seed": 42,
    "protocol_hash": protocol_hash(protocol_descriptor),
    "protocol_core_hash": protocol_hash(protocol_core),
    "protocol_descriptor": protocol_descriptor,
    "hyperparameter_mode": ANCHOR_HPO_MODE,
    "execution_runtime": execution_runtime,
    "campaign_run_tag": CAMPAIGN_RUN_TAG,
    "execution_receipt_sha256": EXECUTION_RECEIPT_SHA,
    "block_reference_manifest_sha256": None,
}


def _capacity_probe(architecture, _config, _params, runtime):
    """Deterministic non-CUDA probe feeding the real receipt validator."""
    architecture_index = (
        "PAA_NHiTS",
        "PAA_S2V_NHiTS",
        "PAA_LSTM_NHiTS",
        "PAA_CNN",
    ).index(architecture)
    total = runtime["execution_environment_descriptor"]["accelerator"][
        "total_memory_bytes"
    ]
    reserved = 500_000_000 + 50_000_000 * architecture_index
    return {
        "peak_memory_allocated_bytes": reserved - 25_000_000,
        "peak_memory_reserved_bytes": reserved,
        "total_memory_bytes": total,
    }


capacity_receipt = run_capacity_preflight(
    execution_runtime,
    probe_runner=_capacity_probe,
)


def _synthetic_anchor_entries() -> list[dict]:
    params_by_arch = {
        architecture: params
        for architecture, _config_value, params
        in registered_capacity_cases()
    }
    entries = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            params = params_by_arch[architecture]
            identity = {
                "schema": STUDY_IDENTITY_SCHEMA,
                "execution_block": "l60",
                "anchor_stage": "s0_scour",
                "architecture": architecture,
                "seed": seed,
                "active_dofs": list(range(8)),
                "study_name": f"synthetic-{architecture}-{seed}",
                "protocol_hash": config["protocol_hash"],
                "dataset": "fixture",
                "model_name": f"synthetic-{architecture}-{seed}",
                "execution_environment_sha256":
                    execution_runtime["execution_environment_sha256"],
                "campaign_run_tag": CAMPAIGN_RUN_TAG,
                "execution_receipt_sha256": EXECUTION_RECEIPT_SHA,
                "study_protocol_record_sha256":
                    canonical_json_sha256(
                        [architecture, seed, "synthetic-record"]
                    ),
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
                "best_trial_value": float(seed) / 1000.0,
                "best_params_sha256": canonical_json_sha256(params),
            }
            entries.append(build_manifest_entry(
                study_identity=identity,
                params=params,
            ))
    return entries


hyperparameter_manifest, _hyperparameter_manifest_sha = build_manifest(
    _synthetic_anchor_entries(),
    execution_runtime=execution_runtime,
    protocol_core_hash=config["protocol_core_hash"],
    anchor_protocol_hash=config["protocol_hash"],
    anchor_dataset="fixture",
    run_tag=CAMPAIGN_RUN_TAG,
    execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
)
hyperparameter_plan = derive_execution_plan(
    config,
    dataset_name="fixture",
    # Deliberately disagree with the effective campaign settings: the plan
    # must derive 100 trials + the registered pruner from ANCHOR_HPO_MODE.
    requested_n_trials=1,
    requested_use_pruner=False,
    execution_runtime=execution_runtime,
)
assert hyperparameter_plan["effective_n_trials"] == 100
assert hyperparameter_plan["effective_use_pruner"] is True


def _follower_config(block_reference_sha256: str) -> dict:
    descriptor = json.loads(json.dumps(protocol_descriptor))
    descriptor["rung"]["stage"] = "s11_bear"
    descriptor["rung"]["dataset"] = "fixture-follower"
    follower = json.loads(json.dumps(config))
    follower.update({
        "name": "artifact-follower-fixture",
        "dofs": [1, 3],
        "protocol_descriptor": descriptor,
        "protocol_hash": protocol_hash(descriptor),
        "hyperparameter_mode": FROZEN_SINGLETON_MODE,
        "block_reference_manifest_sha256": block_reference_sha256,
    })
    follower.update(select_frozen_config(
        hyperparameter_manifest,
        architecture=follower["name_short"],
        seed=follower["seed"],
        expected_runtime=execution_runtime,
        expected_run_tag=CAMPAIGN_RUN_TAG,
        expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
    ))
    return follower


def _close_rdb_storage(storage):
    """Release SQLAlchemy's thread-local session before Windows deletes SQLite."""
    try:
        storage.remove_session()
    finally:
        storage.engine.dispose()
        gc.collect()


with ExitStack() as resources:
    tmp = resources.enter_context(
        tempfile.TemporaryDirectory(prefix="artifact-prov-")
    )
    output = Path(tmp, "out")
    cache = Path(tmp, "cache")
    output.mkdir()
    cache.mkdir()

    storage_url = f"sqlite:///{Path(tmp, 'study.db').resolve().as_posix()}"
    storage = optuna.storages.RDBStorage(storage_url)
    # Registered after TemporaryDirectory, hence executed first (LIFO), even
    # when an assertion fails midway through this adversarial test.
    resources.callback(_close_rdb_storage, storage)
    sampler_policy = OPTUNA_PROTOCOL["sampler"]
    pruner_policy = OPTUNA_PROTOCOL["pruner"]
    study = optuna.create_study(
        study_name="artifact-fixture",
        storage=storage,
        direction=OPTUNA_PROTOCOL["direction"],
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(
                10, hyperparameter_plan["effective_n_trials"] // 4
            ),
            multivariate=sampler_policy["multivariate"],
            constant_liar=sampler_policy["constant_liar"],
            warn_independent_sampling=False,
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pruner_policy["min_resource"],
            reduction_factor=pruner_policy["reduction_factor"],
            min_early_stopping_rate=
                pruner_policy["min_early_stopping_rate"],
        ),
    )
    _stamp_study_protocol(
        study,
        config=config,
        dataset_name="fixture",
        n_trials=hyperparameter_plan["effective_n_trials"],
        epochs=2,
        sampler_seed=42,
        use_pruner=hyperparameter_plan["effective_use_pruner"],
        hyperparameter_plan=hyperparameter_plan,
        capacity_receipt=capacity_receipt,
    )
    study = optuna.load_study(
        study_name="artifact-fixture", storage=storage
    )
    _stamp_study_protocol(
        study,
        config=config,
        dataset_name="fixture",
        n_trials=hyperparameter_plan["effective_n_trials"],
        epochs=2,
        sampler_seed=42,
        use_pruner=hyperparameter_plan["effective_use_pruner"],
        hyperparameter_plan=hyperparameter_plan,
        capacity_receipt=capacity_receipt,
    )
    check("SQLite restart accepts canonically identical tuple/list protocol",
          study.user_attrs["ttbi_protocol_record"]["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"])))
    study.optimize(
        lambda trial: trial.suggest_float("lr", 1e-4, 1e-3),
        n_trials=hyperparameter_plan["effective_n_trials"],
    )
    record = study.user_attrs.get("ttbi_protocol_record")
    check("study stores full protocol descriptor",
          record["protocol_hash"] == config["protocol_hash"]
          and record["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"]))
          and record["execution_environment_sha256"]
          == execution_runtime["execution_environment_sha256"]
          and record["execution_runtime"]
          == json.loads(json.dumps(execution_runtime))
          and record["campaign_run_tag"] == CAMPAIGN_RUN_TAG
          and record["execution_receipt_sha256"] == EXECUTION_RECEIPT_SHA
          and record["block_reference_manifest_sha256"] is None
          and record["schema"] == "optuna-study-provenance-v4")

    follower_a = _follower_config(BLOCK_REFERENCE_A)
    follower_plan_a = derive_execution_plan(
        follower_a,
        dataset_name="fixture-follower",
        requested_n_trials=100,
        requested_use_pruner=True,
        execution_runtime=execution_runtime,
    )
    follower_study = optuna.create_study(
        study_name="follower-reference-restart",
        direction="minimize",
    )
    _stamp_study_protocol(
        follower_study,
        config=follower_a,
        dataset_name="fixture-follower",
        n_trials=follower_plan_a["effective_n_trials"],
        epochs=2,
        sampler_seed=42,
        use_pruner=follower_plan_a["effective_use_pruner"],
        hyperparameter_plan=follower_plan_a,
        capacity_receipt=capacity_receipt,
    )
    follower_b = _follower_config(BLOCK_REFERENCE_B)
    follower_plan_b = derive_execution_plan(
        follower_b,
        dataset_name="fixture-follower",
        requested_n_trials=100,
        requested_use_pruner=True,
        execution_runtime=execution_runtime,
    )
    check(
        "block-reference outcome remains outside protocol_hash",
        follower_a["protocol_hash"] == follower_b["protocol_hash"]
        and follower_plan_a["block_reference_manifest_sha256"]
        != follower_plan_b["block_reference_manifest_sha256"],
    )
    rejects_with_message(
        "follower reference B cannot resume study stamped under reference A",
        lambda: _stamp_study_protocol(
            follower_study,
            config=follower_b,
            dataset_name="fixture-follower",
            n_trials=follower_plan_b["effective_n_trials"],
            epochs=2,
            sampler_seed=42,
            use_pruner=follower_plan_b["effective_use_pruner"],
            hyperparameter_plan=follower_plan_b,
            capacity_receipt=capacity_receipt,
        ),
        "stored Optuna protocol record differs",
    )
    follower_missing_reference = json.loads(json.dumps(follower_a))
    del follower_missing_reference["block_reference_manifest_sha256"]
    rejects(
        "follower config missing block-reference digest is rejected",
        lambda: derive_execution_plan(
            follower_missing_reference,
            dataset_name="fixture-follower",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=execution_runtime,
        ),
    )
    foreign_runtime = json.loads(json.dumps(execution_runtime))
    foreign_runtime["execution_environment_descriptor"]["host"][
        "hostname"
    ] = "foreign-host"
    foreign_runtime["execution_environment_sha256"] = (
        execution_environment_sha256(
            foreign_runtime["execution_environment_descriptor"]
        )
    )
    foreign_config = {**config, "execution_runtime": foreign_runtime}
    foreign_plan = derive_execution_plan(
        foreign_config,
        dataset_name="fixture",
        requested_n_trials=1,
        requested_use_pruner=False,
        execution_runtime=foreign_runtime,
    )
    foreign_capacity_receipt = run_capacity_preflight(
        foreign_runtime,
        probe_runner=_capacity_probe,
    )
    rejects(
        "foreign execution identity cannot resume the Optuna study",
        lambda: _stamp_study_protocol(
            study,
            config=foreign_config,
            dataset_name="fixture",
            n_trials=foreign_plan["effective_n_trials"],
            epochs=2,
            sampler_seed=42,
            use_pruner=foreign_plan["effective_use_pruner"],
            hyperparameter_plan=foreign_plan,
            capacity_receipt=foreign_capacity_receipt,
        ),
    )
    rejects(
        "protocol-hashed study rejects missing execution identity",
        lambda: _stamp_study_protocol(
            study,
            config={key: value for key, value in config.items()
                    if key != "execution_runtime"},
            dataset_name="fixture",
            n_trials=hyperparameter_plan["effective_n_trials"],
            epochs=2,
            sampler_seed=42,
            use_pruner=hyperparameter_plan["effective_use_pruner"],
            hyperparameter_plan=hyperparameter_plan,
            capacity_receipt=capacity_receipt,
        ),
    )
    wrong_block_runtime = json.loads(json.dumps(execution_runtime))
    wrong_block_runtime["execution_block"] = "l99"
    wrong_block_runtime["anchor_stage"] = "s21_scour4"
    rejects(
        "execution identity cannot be relabelled into another protocol block",
        lambda: _stamp_study_protocol(
            study,
            config={**config, "execution_runtime": wrong_block_runtime},
            dataset_name="fixture",
            n_trials=hyperparameter_plan["effective_n_trials"],
            epochs=2,
            sampler_seed=42,
            use_pruner=hyperparameter_plan["effective_use_pruner"],
            hyperparameter_plan=hyperparameter_plan,
            capacity_receipt=capacity_receipt,
        ),
    )

    trial_path = output / (
        f"weights_{config['name']}_trial_{study.best_trial.number}.pth"
    )
    torch.save({"weight": torch.arange(8, dtype=torch.float32)}, trial_path)
    scaler_path = cache / f"scaler_{_cache_stem('fixture', config)}.pkl"
    joblib.dump({"mean": [0.0, 0.0], "scale": [1.0, 1.0]}, scaler_path)
    scaler_sha = __import__("hashlib").sha256(scaler_path.read_bytes()).hexdigest()
    cache_prov = cache / f"cache_{_cache_stem('fixture', config)}_prov.json"
    cache_prov.write_text(json.dumps({
        "source": {"fixture": True},
        "artifacts": {"scaler": scaler_sha},
    }), encoding="utf-8")
    scaler_source_bytes = scaler_path.read_bytes()
    tampered_source = bytearray(scaler_source_bytes)
    tampered_source[-1] ^= 1
    scaler_path.write_bytes(tampered_source)
    rejects("pre-export cache-scaler tamper rejected",
            lambda: export_digital_twin_package(
                study, config, "fixture", str(cache), str(output)
            ))
    scaler_path.write_bytes(scaler_source_bytes)
    export_digital_twin_package(
        study, config, "fixture", str(cache), str(output)
    )
    metadata = verify_digital_twin_package(study, config, str(output))
    check("standalone deployment package verifies without Optuna DB",
          verify_standalone_dt_package(
              str(output / "DT_champion_weights.pth"),
              str(output / "DT_metadata.json"),
              str(output / "DT_scaler.pkl"),
          )["protocol_hash"] == config["protocol_hash"])
    check(
        "standalone anchor package accepts an explicit null reference expectation",
        verify_standalone_dt_package(
            str(output / "DT_champion_weights.pth"),
            str(output / "DT_metadata.json"),
            str(output / "DT_scaler.pkl"),
            expected_block_reference_manifest_sha256=None,
        )["block_reference_manifest_sha256"] is None,
    )
    check("metadata links study and best trial",
          metadata["study_name"] == study.study_name
          and metadata["best_trial_number"] == study.best_trial.number)
    check("metadata carries full protocol descriptor",
          metadata["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"])))
    check("metadata carries the exact execution identity",
           metadata["execution_environment_sha256"]
           == execution_runtime["execution_environment_sha256"]
           and metadata["execution_runtime"]
           == json.loads(json.dumps(execution_runtime))
           and metadata["campaign_run_tag"] == CAMPAIGN_RUN_TAG
           and metadata["execution_receipt_sha256"]
           == EXECUTION_RECEIPT_SHA
           and metadata["block_reference_manifest_sha256"] is None)
    check(
        "Optuna champion artifact carries v4 campaign lineage",
        study.user_attrs["ttbi_champion_artifact"]["schema"]
        == "champion-artifact-v4"
        and study.user_attrs["ttbi_champion_artifact"][
            "campaign_run_tag"
        ] == CAMPAIGN_RUN_TAG
        and study.user_attrs["ttbi_champion_artifact"][
            "execution_receipt_sha256"
        ] == EXECUTION_RECEIPT_SHA
        and study.user_attrs["ttbi_champion_artifact"][
            "block_reference_manifest_sha256"
        ] is None,
    )
    check("trial weights removed only after linked champion publication",
          not trial_path.exists()
          and (output / "DT_champion_weights.pth").is_file()
          and (output / "DT_scaler.pkl").is_file())
    export_digital_twin_package(
        study, config, "fixture", str(cache), str(output)
    )
    check("completed package export is idempotent after trial cleanup",
          not trial_path.exists()
          and verify_digital_twin_package(
              study, config, str(output)
          )["best_trial_number"] == study.best_trial.number)

    metadata_path = output / "DT_metadata.json"
    metadata_original = metadata_path.read_text(encoding="utf-8")
    missing_field = json.loads(metadata_original)
    del missing_field["scaler_sha256"]
    metadata_path.write_text(json.dumps(missing_field), encoding="utf-8")
    rejects_with_message(
        "standalone verifier rejects a missing required provenance field",
        lambda: verify_standalone_dt_package(
            str(output / "DT_champion_weights.pth"),
            str(metadata_path),
            str(output / "DT_scaler.pkl"),
        ),
        "metadata lacks provenance fields",
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    for lineage_field in (
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
    ):
        missing_lineage = json.loads(metadata_original)
        del missing_lineage[lineage_field]
        metadata_path.write_text(
            json.dumps(missing_lineage), encoding="utf-8"
        )
        rejects_with_message(
            f"standalone verifier rejects missing {lineage_field}",
            lambda: verify_standalone_dt_package(
                str(output / "DT_champion_weights.pth"),
                str(metadata_path),
                str(output / "DT_scaler.pkl"),
            ),
            "metadata lacks provenance fields",
        )
        rejects(
            f"study-linked verifier rejects missing {lineage_field}",
            lambda: verify_digital_twin_package(
                study, config, str(output)
            ),
        )
        metadata_path.write_text(metadata_original, encoding="utf-8")

    champion = output / "DT_champion_weights.pth"
    original = champion.read_bytes()
    corrupt = bytearray(original)
    corrupt[-1] ^= 1
    champion.write_bytes(corrupt)
    rejects("standalone verifier rejects one-byte champion tamper",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path),
                str(output / "DT_scaler.pkl"),
            ))
    rejects("one-byte champion tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    champion.write_bytes(original)

    exported_scaler = output / "DT_scaler.pkl"
    scaler_bytes = exported_scaler.read_bytes()
    scaler_corrupt = bytearray(scaler_bytes)
    scaler_corrupt[-1] ^= 1
    exported_scaler.write_bytes(scaler_corrupt)
    rejects("standalone verifier rejects one-byte scaler tamper",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    rejects("one-byte scaler tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    exported_scaler.write_bytes(scaler_bytes)

    altered = json.loads(metadata_original)
    altered["protocol_descriptor"]["core"]["protocol_version"] = 999
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects descriptor/hash disagreement",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    rejects("metadata protocol-descriptor tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["campaign_run_tag"] = "other-run"
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects(
        "study-linked verifier rejects campaign run_tag tamper",
        lambda: verify_digital_twin_package(study, config, str(output)),
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["execution_receipt_sha256"] = "E" * 64
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects(
        "standalone verifier rejects invalid execution receipt",
        lambda: verify_standalone_dt_package(
            str(champion), str(metadata_path), str(exported_scaler),
        ),
    )
    rejects(
        "study-linked verifier rejects execution receipt tamper",
        lambda: verify_digital_twin_package(study, config, str(output)),
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["block_reference_manifest_sha256"] = BLOCK_REFERENCE_A
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects(
        "standalone verifier rejects a reference digest on an anchor package",
        lambda: verify_standalone_dt_package(
            str(champion), str(metadata_path), str(exported_scaler),
        ),
    )
    rejects(
        "study-linked verifier rejects block-reference tamper",
        lambda: verify_digital_twin_package(study, config, str(output)),
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    follower_metadata = json.loads(metadata_original)
    follower_metadata["protocol_descriptor"]["rung"]["stage"] = "s11_bear"
    follower_metadata["protocol_descriptor"]["rung"][
        "dataset"
    ] = "fixture-follower"
    follower_metadata["protocol_hash"] = protocol_hash(
        follower_metadata["protocol_descriptor"]
    )
    follower_metadata["block_reference_manifest_sha256"] = (
        BLOCK_REFERENCE_A
    )
    metadata_path.write_text(
        json.dumps(follower_metadata), encoding="utf-8"
    )
    check(
        "standalone follower package accepts the exact external reference pin",
        verify_standalone_dt_package(
            str(champion),
            str(metadata_path),
            str(exported_scaler),
            expected_block_reference_manifest_sha256=BLOCK_REFERENCE_A,
        )["block_reference_manifest_sha256"] == BLOCK_REFERENCE_A,
    )
    rejects(
        "standalone follower package rejects reference B for package A",
        lambda: verify_standalone_dt_package(
            str(champion),
            str(metadata_path),
            str(exported_scaler),
            expected_block_reference_manifest_sha256=BLOCK_REFERENCE_B,
        ),
    )
    follower_metadata["block_reference_manifest_sha256"] = "A" * 64
    metadata_path.write_text(
        json.dumps(follower_metadata), encoding="utf-8"
    )
    rejects(
        "standalone follower package rejects malformed reference digest",
        lambda: verify_standalone_dt_package(
            str(champion), str(metadata_path), str(exported_scaler),
        ),
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["scaler_filename"] = "different_scaler.pkl"
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects scaler filename substitution",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["execution_runtime"]["execution_environment_descriptor"]["host"][
        "hostname"
    ] = "tampered-host"
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects execution descriptor/SHA tamper",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    rejects("DT verifier rejects execution descriptor/SHA tamper",
            lambda: verify_digital_twin_package(
                study, config, str(output)
            ))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["execution_runtime"] = foreign_runtime
    altered["execution_environment_sha256"] = foreign_runtime[
        "execution_environment_sha256"
    ]
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("coherent foreign execution metadata rejected by run config",
            lambda: verify_digital_twin_package(
                study, config, str(output)
            ))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["execution_runtime"]["execution_block"] = "l99"
    altered["execution_runtime"]["anchor_stage"] = "s21_scour4"
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects execution-block relabelling",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["active_dofs"] = [0]
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("deployment-semantics metadata tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    original_record = study.user_attrs["ttbi_protocol_record"]
    altered_record = json.loads(json.dumps(original_record))
    altered_record["protocol_descriptor"]["core"]["protocol_version"] = 999
    study.set_user_attr("ttbi_protocol_record", altered_record)
    rejects("Optuna protocol-record tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    study.set_user_attr("ttbi_protocol_record", original_record)

    altered_record = json.loads(json.dumps(original_record))
    altered_record["execution_runtime"] = foreign_runtime
    altered_record["execution_environment_sha256"] = foreign_runtime[
        "execution_environment_sha256"
    ]
    study.set_user_attr("ttbi_protocol_record", altered_record)
    rejects("Optuna execution-identity tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    study.set_user_attr("ttbi_protocol_record", original_record)

    for lineage_field, replacement in (
        ("campaign_run_tag", "other-run"),
        ("execution_receipt_sha256", "f" * 64),
        ("block_reference_manifest_sha256", BLOCK_REFERENCE_A),
    ):
        altered_record = json.loads(json.dumps(original_record))
        altered_record[lineage_field] = replacement
        study.set_user_attr("ttbi_protocol_record", altered_record)
        rejects(
            f"Optuna {lineage_field} tamper rejected",
            lambda: verify_digital_twin_package(
                study, config, str(output)
            ),
        )
        study.set_user_attr("ttbi_protocol_record", original_record)

    original_artifact = study.user_attrs["ttbi_champion_artifact"]
    altered_artifact = json.loads(json.dumps(original_artifact))
    altered_artifact["block_reference_manifest_sha256"] = BLOCK_REFERENCE_A
    study.set_user_attr("ttbi_champion_artifact", altered_artifact)
    rejects(
        "Optuna champion-artifact block-reference tamper rejected",
        lambda: verify_digital_twin_package(study, config, str(output)),
    )
    study.set_user_attr("ttbi_champion_artifact", original_artifact)

    altered = json.loads(metadata_path.read_text(encoding="utf-8"))
    altered["best_trial_number"] += 1
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("metadata-to-best-trial mismatch rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))

    mixed = optuna.create_study(study_name="unstamped-existing")
    mixed.optimize(lambda trial: 0.0, n_trials=1)
    rejects(
        "existing protocol-hashed trials cannot be stamped retroactively",
        lambda: _stamp_study_protocol(
            mixed,
            config={**config, "name": "unstamped-existing"},
            dataset_name="fixture",
            n_trials=hyperparameter_plan["effective_n_trials"],
            epochs=2,
            sampler_seed=42,
            use_pruner=hyperparameter_plan["effective_use_pruner"],
            hyperparameter_plan=hyperparameter_plan,
            capacity_receipt=capacity_receipt,
        ),
    )

print()
if fails:
    raise SystemExit(f"ARTIFACT PROVENANCE: {fails} FAILURE(S)")
print("ARTIFACT PROVENANCE: ALL PASS")
