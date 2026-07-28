"""Adversarial behavioural checks for the R11 hyperparameter policy."""

from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import optuna

from core.capacity_preflight import registered_capacity_cases
from core.execution_environment import execution_environment_sha256
from core.hyperparameter_policy import (
    ANCHOR_HPO_MODE,
    FROZEN_SINGLETON_MODE,
    ARCHITECTURES,
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
from core.protocol import protocol_hash
from training.pipeline import (
    _create_or_resume_study,
    _execute_protocol_study,
)
from training.trainer import _suggest_params


RUN_TAG = ""
EXECUTION_RECEIPT_SHA = "e" * 64
BLOCK_REFERENCE_SHA = "b" * 64


def _runtime(block: str = "l60") -> dict:
    anchor = "s0_scour" if block == "l60" else "s21_scour4"
    descriptor = {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": "qualification-host",
            "machine": "AMD64",
            "system": "Windows",
            "platform": "qualification-fixture",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "qualification-gpu",
            "uuid": "GPU-qualification",
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 40,
            "total_memory_bytes": 8_589_934_592,
            "driver_version": "fixture",
        },
        "numeric_stack": {
            "torch_version": "fixture",
            "cuda_runtime_version": "fixture",
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
    return {
        "schema": "ttbi-execution-runtime-binding-v1",
        "execution_block": block,
        "anchor_stage": anchor,
        "execution_environment_sha256":
            execution_environment_sha256(descriptor),
        "execution_environment_descriptor": descriptor,
    }


def _descriptor(stage: str, dataset: str, block: str = "l60") -> tuple:
    anchor = "s0_scour" if block == "l60" else "s21_scour4"
    core = {"qualification": "hyperparameter-policy-r11"}
    full = {
        "core": core,
        "rung": {
            "stage": stage,
            "dataset": dataset,
            "execution_block": block,
            "execution_anchor": anchor,
        },
    }
    return core, full


def _config(
    architecture: str,
    seed: int,
    *,
    stage: str,
    dataset: str,
    dofs: list[int],
    mode: str,
    block: str = "l60",
) -> dict:
    core, descriptor = _descriptor(stage, dataset, block)
    anchor = "s0_scour" if block == "l60" else "s21_scour4"
    flags = {
        "PAA_NHiTS": (False, False, True),
        "PAA_S2V_NHiTS": (True, False, True),
        "PAA_LSTM_NHiTS": (False, True, True),
        "PAA_CNN": (False, False, False),
    }[architecture]
    return {
        "name": f"{architecture}_{stage}_{seed}",
        "name_short": architecture,
        "seed": seed,
        "dofs": dofs,
        "method": "PAA",
        "model_type": "1D_MODULAR",
        "use_space2vec": flags[0],
        "use_lstm": flags[1],
        "use_nhits": flags[2],
        "protocol_descriptor": descriptor,
        "protocol_hash": protocol_hash(descriptor),
        "protocol_core_hash": protocol_hash(core),
        "hyperparameter_mode": mode,
        "execution_runtime": _runtime(block),
        "campaign_run_tag": RUN_TAG,
        "execution_receipt_sha256": EXECUTION_RECEIPT_SHA,
        "block_reference_manifest_sha256": (
            None if stage == anchor else BLOCK_REFERENCE_SHA
        ),
    }


def _expect_error(label: str, fn) -> None:
    try:
        fn()
    except (HyperparameterPolicyError, RuntimeError, ValueError):
        return
    raise AssertionError(f"mutation survived: {label}")


def _entries(runtime: dict, anchor_hash: str, dataset: str) -> list[dict]:
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
                "execution_block": runtime["execution_block"],
                "anchor_stage": runtime["anchor_stage"],
                "architecture": architecture,
                "seed": seed,
                "active_dofs": list(range(8)),
                "study_name": f"{architecture}_{seed}",
                "protocol_hash": anchor_hash,
                "dataset": dataset,
                "model_name": f"{architecture}_{seed}",
                "execution_environment_sha256":
                    runtime["execution_environment_sha256"],
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
                "best_trial_value": float(seed) / 1000.0,
                "best_params_sha256": canonical_json_sha256(params),
            }
            entries.append(build_manifest_entry(
                study_identity=identity,
                params=params,
            ))
    return entries


def main() -> None:
    runtime = _runtime()
    dataset = "s0_fixture"
    anchor = _config(
        ARCHITECTURES[0],
        SEEDS[0],
        stage="s0_scour",
        dataset=dataset,
        dofs=list(range(8)),
        mode=ANCHOR_HPO_MODE,
    )
    anchor_plan = derive_execution_plan(
        anchor,
        dataset_name=dataset,
        requested_n_trials=7,
        requested_use_pruner=False,
        execution_runtime=runtime,
    )
    assert anchor_plan["effective_n_trials"] == 100
    assert anchor_plan["effective_use_pruner"] is True
    assert anchor_plan["campaign_run_tag"] == RUN_TAG
    assert (
        anchor_plan["execution_receipt_sha256"]
        == EXECUTION_RECEIPT_SHA
    )
    assert anchor_plan["block_reference_manifest_sha256"] is None
    assert validate_run_plan(anchor_plan) == anchor_plan

    entries = _entries(runtime, anchor["protocol_hash"], dataset)
    assert len(entries) == len(ARCHITECTURES) * len(SEEDS) == 12
    manifest, manifest_sha = build_manifest(
        entries,
        execution_runtime=runtime,
        protocol_core_hash=anchor["protocol_core_hash"],
        anchor_protocol_hash=anchor["protocol_hash"],
        anchor_dataset=dataset,
        run_tag=RUN_TAG,
        execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
    )
    assert canonical_json_sha256(manifest) == manifest_sha
    validate_manifest(
        manifest,
        expected_runtime=runtime,
        expected_run_tag=RUN_TAG,
        expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
    )

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "hyperparameters.json"
        assert write_manifest(
            path,
            manifest,
            expected_runtime=runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
        ) == manifest_sha
        assert load_manifest(
            path,
            expected_sha256=manifest_sha,
            expected_runtime=runtime,
            expected_run_tag=RUN_TAG,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
        ) == manifest
        differing_manifest = deepcopy(manifest)
        differing_manifest["run_tag"] = "different-run"
        _expect_error(
            "differing HPO manifest overwrite",
            lambda: write_manifest(
                path,
                differing_manifest,
                expected_runtime=runtime,
            ),
        )
        pretty = path.read_text(encoding="ascii")
        path.write_text(pretty + "\n", encoding="ascii")
        _expect_error(
            "noncanonical manifest bytes",
            lambda: load_manifest(
                path,
                expected_sha256=manifest_sha,
                expected_runtime=runtime,
                expected_run_tag=RUN_TAG,
                expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
            ),
        )

    frozen_fields = select_frozen_config(
        manifest,
        architecture=ARCHITECTURES[0],
        seed=SEEDS[0],
        expected_runtime=runtime,
        expected_run_tag=RUN_TAG,
        expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
    )
    downstream = _config(
        ARCHITECTURES[0],
        SEEDS[0],
        stage="s11_bear",
        dataset="s11_fixture",
        dofs=[1, 3],
        mode=FROZEN_SINGLETON_MODE,
    )
    downstream.update(frozen_fields)
    frozen_plan = derive_execution_plan(
        downstream,
        dataset_name="s11_fixture",
        requested_n_trials=100,
        requested_use_pruner=True,
        execution_runtime=runtime,
    )
    assert frozen_plan["effective_n_trials"] == 1
    assert frozen_plan["effective_use_pruner"] is False
    assert frozen_plan["campaign_run_tag"] == RUN_TAG
    assert (
        frozen_plan["execution_receipt_sha256"]
        == EXECUTION_RECEIPT_SHA
    )
    assert (
        frozen_plan["block_reference_manifest_sha256"]
        == BLOCK_REFERENCE_SHA
    )
    assert set(frozen_plan["hyperparameter_source"]) == {
        "execution_block",
        "anchor_stage",
        "architecture",
        "seed",
        "study_identity_sha256",
        "params_sha256",
    }
    assert validate_run_plan(frozen_plan) == frozen_plan

    # The frozen arm remains a real Optuna study, but every active parameter
    # is registered as a one-point distribution and reproduces the manifest.
    fixed_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )

    def fixed_objective(trial):
        params = _suggest_params(trial, downstream)
        assert params == downstream["frozen_hyperparameters"]
        return 0.0

    fixed_study.optimize(fixed_objective, n_trials=1)
    assert fixed_study.best_params == downstream["frozen_hyperparameters"]
    for distribution in fixed_study.best_trial.distributions.values():
        if isinstance(
            distribution,
            (optuna.distributions.IntDistribution,
             optuna.distributions.FloatDistribution),
        ):
            assert distribution.low == distribution.high
        else:
            assert (
                isinstance(
                    distribution,
                    optuna.distributions.CategoricalDistribution,
                )
                and len(distribution.choices) == 1
            )

    anchor_optuna = _create_or_resume_study(
        "anchor-pruner-fixture",
        "sqlite:///:memory:",
        100,
        sampler_seed=42,
        use_pruner=True,
    )
    frozen_optuna = _create_or_resume_study(
        "frozen-pruner-fixture",
        "sqlite:///:memory:",
        1,
        sampler_seed=42,
        use_pruner=False,
        force_nop_pruner=True,
    )
    assert isinstance(
        anchor_optuna.pruner,
        optuna.pruners.SuccessiveHalvingPruner,
    )
    assert isinstance(frozen_optuna.pruner, optuna.pruners.NopPruner)
    _expect_error(
        "conflicting registered and Nop pruners",
        lambda: _create_or_resume_study(
            "conflicting-pruner-fixture",
            "sqlite:///:memory:",
            1,
            use_pruner=True,
            force_nop_pruner=True,
        ),
    )

    # Publication terminal gates derive their budgets from the registered
    # mode, rather than trusting a mutable `effective_n_trials` field.
    complete = optuna.trial.TrialState.COMPLETE
    pruned = optuna.trial.TrialState.PRUNED
    anchor_terminal = SimpleNamespace(
        trials=(
            [SimpleNamespace(state=complete) for _ in range(40)]
            + [SimpleNamespace(state=pruned) for _ in range(60)]
        )
    )
    assert validate_terminal_study(
        anchor_terminal, anchor_plan
    )["total"] == 100
    frozen_terminal = SimpleNamespace(
        trials=[SimpleNamespace(state=complete)]
    )
    assert validate_terminal_study(
        frozen_terminal, frozen_plan
    )["COMPLETE"] == 1
    mutated_budget = deepcopy(anchor_plan)
    mutated_budget["effective_n_trials"] = 1
    _expect_error(
        "anchor terminal budget changed to one",
        lambda: validate_terminal_study(frozen_terminal, mutated_budget),
    )
    mutated_plan_dofs = deepcopy(anchor_plan)
    mutated_plan_dofs["active_dofs"] = [0, 1]
    _expect_error(
        "anchor run-plan active DOFs detached",
        lambda: validate_run_plan(mutated_plan_dofs),
    )
    mutated_anchor_reference = deepcopy(anchor_plan)
    mutated_anchor_reference["block_reference_manifest_sha256"] = (
        BLOCK_REFERENCE_SHA
    )
    _expect_error(
        "anchor run-plan pre-cites reference",
        lambda: validate_run_plan(mutated_anchor_reference),
    )
    mutated_follower_reference = deepcopy(frozen_plan)
    mutated_follower_reference["block_reference_manifest_sha256"] = None
    _expect_error(
        "follower run-plan loses reference",
        lambda: validate_run_plan(mutated_follower_reference),
    )
    mutated_plan_run_tag = deepcopy(frozen_plan)
    mutated_plan_run_tag["campaign_run_tag"] = None
    _expect_error(
        "campaign run-plan loses run_tag",
        lambda: validate_run_plan(mutated_plan_run_tag),
    )
    mutated_plan_receipt = deepcopy(frozen_plan)
    mutated_plan_receipt["execution_receipt_sha256"] = "E" * 64
    _expect_error(
        "campaign run-plan carries invalid receipt",
        lambda: validate_run_plan(mutated_plan_receipt),
    )
    _expect_error(
        "frozen singleton pruned instead of complete",
        lambda: validate_terminal_study(
            SimpleNamespace(trials=[SimpleNamespace(state=pruned)]),
            frozen_plan,
        ),
    )

    class _OOMStudy:
        study_name = "synthetic_oom"
        trials = []

        def __init__(self):
            self.optimize_calls = 0

        def optimize(self, *_args, **_kwargs):
            self.optimize_calls += 1
            raise MemoryError("synthetic fatal OOM")

    oom_study = _OOMStudy()
    try:
        _execute_protocol_study(oom_study, object(), anchor_plan)
    except MemoryError:
        pass
    else:
        raise AssertionError("campaign OOM was caught or converted into retry")
    assert oom_study.optimize_calls == 1

    missing_mode = deepcopy(anchor)
    del missing_mode["hyperparameter_mode"]
    _expect_error(
        "missing protocol HPO mode",
        lambda: derive_execution_plan(
            missing_mode,
            dataset_name=dataset,
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    for missing_lineage_field in (
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
    ):
        missing_lineage = deepcopy(anchor)
        del missing_lineage[missing_lineage_field]
        _expect_error(
            f"anchor missing {missing_lineage_field}",
            lambda value=missing_lineage: derive_execution_plan(
                value,
                dataset_name=dataset,
                requested_n_trials=100,
                requested_use_pruner=True,
                execution_runtime=runtime,
            ),
        )
    anchor_with_reference = deepcopy(anchor)
    anchor_with_reference["block_reference_manifest_sha256"] = (
        BLOCK_REFERENCE_SHA
    )
    _expect_error(
        "anchor pre-cites its own reference",
        lambda: derive_execution_plan(
            anchor_with_reference,
            dataset_name=dataset,
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    anchor_subset = deepcopy(anchor)
    anchor_subset["dofs"] = [0, 1]
    _expect_error(
        "HPO on anchor sensor subset",
        lambda: derive_execution_plan(
            anchor_subset,
            dataset_name=dataset,
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    downstream_hpo = deepcopy(downstream)
    downstream_hpo["hyperparameter_mode"] = ANCHOR_HPO_MODE
    _expect_error(
        "downstream renewed HPO",
        lambda: derive_execution_plan(
            downstream_hpo,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    wrong_dataset = deepcopy(downstream)
    _expect_error(
        "pipeline/protocol dataset mismatch",
        lambda: derive_execution_plan(
            wrong_dataset,
            dataset_name="regenerated_other",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    follower_without_reference = deepcopy(downstream)
    del follower_without_reference["block_reference_manifest_sha256"]
    _expect_error(
        "follower missing block-reference digest",
        lambda: derive_execution_plan(
            follower_without_reference,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    follower_null_reference = deepcopy(downstream)
    follower_null_reference["block_reference_manifest_sha256"] = None
    _expect_error(
        "follower null block-reference digest",
        lambda: derive_execution_plan(
            follower_null_reference,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    follower_upper_reference = deepcopy(downstream)
    follower_upper_reference["block_reference_manifest_sha256"] = "B" * 64
    _expect_error(
        "follower uppercase block-reference digest",
        lambda: derive_execution_plan(
            follower_upper_reference,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    other_reference = deepcopy(downstream)
    other_reference["block_reference_manifest_sha256"] = "c" * 64
    other_reference_plan = derive_execution_plan(
        other_reference,
        dataset_name="s11_fixture",
        requested_n_trials=100,
        requested_use_pruner=True,
        execution_runtime=runtime,
    )
    assert other_reference["protocol_hash"] == downstream["protocol_hash"]
    assert (
        other_reference_plan["block_reference_manifest_sha256"]
        != frozen_plan["block_reference_manifest_sha256"]
    )

    missing_entry = deepcopy(manifest)
    missing_entry["entries"].pop()
    _expect_error(
        "incomplete architecture-seed factorial",
        lambda: validate_manifest(missing_entry, expected_runtime=runtime),
    )
    duplicate_study = deepcopy(manifest)
    duplicate_study["entries"][1]["study_identity"]["study_name"] = (
        duplicate_study["entries"][0]["study_identity"]["study_name"]
    )
    duplicate_study["entries"][1]["study_identity_sha256"] = (
        canonical_json_sha256(
            duplicate_study["entries"][1]["study_identity"]
        )
    )
    _expect_error(
        "two identities cite one Optuna study",
        lambda: validate_manifest(duplicate_study, expected_runtime=runtime),
    )
    extra_key = deepcopy(manifest)
    extra_key["unexpected"] = True
    _expect_error(
        "manifest extra field",
        lambda: validate_manifest(extra_key, expected_runtime=runtime),
    )
    changed_dataset = deepcopy(manifest)
    changed_dataset["anchor_dataset"] = "regenerated"
    _expect_error(
        "anchor dataset identity mutation",
        lambda: validate_manifest(changed_dataset, expected_runtime=runtime),
    )
    changed_run_tag = deepcopy(manifest)
    changed_run_tag["run_tag"] = "another-run"
    _expect_error(
        "manifest run_tag detached from anchor-study identities",
        lambda: validate_manifest(changed_run_tag, expected_runtime=runtime),
    )
    _expect_error(
        "campaign run-tag mutation",
        lambda: validate_manifest(
            changed_run_tag,
            expected_runtime=runtime,
            expected_run_tag=RUN_TAG,
        ),
    )
    changed_receipt = deepcopy(manifest)
    changed_receipt["execution_receipt_sha256"] = "f" * 64
    _expect_error(
        "manifest receipt detached from anchor-study identities",
        lambda: validate_manifest(changed_receipt, expected_runtime=runtime),
    )
    _expect_error(
        "execution receipt lineage mutation",
        lambda: validate_manifest(
            changed_receipt,
            expected_runtime=runtime,
            expected_execution_receipt_sha256=EXECUTION_RECEIPT_SHA,
        ),
    )
    changed_full_hash = deepcopy(manifest)
    changed_full_hash["anchor_protocol_hash"] = "c" * 64
    _expect_error(
        "anchor full protocol identity mutation",
        lambda: validate_manifest(changed_full_hash, expected_runtime=runtime),
    )
    changed_params = deepcopy(manifest)
    changed_params["entries"][0]["params"]["n_conv_layers"] = 3
    changed_params["entries"][0]["params_sha256"] = canonical_json_sha256(
        changed_params["entries"][0]["params"]
    )
    _expect_error(
        "parameters detached from study identity",
        lambda: validate_manifest(changed_params, expected_runtime=runtime),
    )
    pair_calibration = deepcopy(manifest)
    pair_calibration["entries"][0]["study_identity"]["active_dofs"] = [1, 3]
    pair_calibration["entries"][0]["study_identity_sha256"] = (
        canonical_json_sha256(
            pair_calibration["entries"][0]["study_identity"]
        )
    )
    _expect_error(
        "pair-input anchor calibration",
        lambda: validate_manifest(pair_calibration, expected_runtime=runtime),
    )
    out_of_range_trial = deepcopy(manifest)
    out_of_range_trial["entries"][0]["study_identity"][
        "best_trial_number"
    ] = 100
    out_of_range_trial["entries"][0]["study_identity_sha256"] = (
        canonical_json_sha256(
            out_of_range_trial["entries"][0]["study_identity"]
        )
    )
    _expect_error(
        "best trial outside exact budget",
        lambda: validate_manifest(out_of_range_trial, expected_runtime=runtime),
    )
    negative_mse = deepcopy(manifest)
    negative_mse["entries"][0]["study_identity"]["best_trial_value"] = -0.1
    negative_mse["entries"][0]["study_identity_sha256"] = (
        canonical_json_sha256(
            negative_mse["entries"][0]["study_identity"]
        )
    )
    _expect_error(
        "negative best MSE",
        lambda: validate_manifest(negative_mse, expected_runtime=runtime),
    )
    bad_source = deepcopy(downstream)
    bad_source["hyperparameter_source"]["extra"] = True
    _expect_error(
        "hyperparameter source extra field",
        lambda: derive_execution_plan(
            bad_source,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    wrong_expected_run = deepcopy(downstream)
    wrong_expected_run["campaign_run_tag"] = "another-run"
    _expect_error(
        "frozen derivation authenticates the expected run tag",
        lambda: derive_execution_plan(
            wrong_expected_run,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    wrong_expected_receipt = deepcopy(downstream)
    wrong_expected_receipt["execution_receipt_sha256"] = "f" * 64
    _expect_error(
        "frozen derivation authenticates the expected execution receipt",
        lambda: derive_execution_plan(
            wrong_expected_receipt,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )
    stripped_protocol_hash = deepcopy(downstream)
    del stripped_protocol_hash["protocol_hash"]
    _expect_error(
        "partially stripped campaign config cannot fall back to legacy",
        lambda: derive_execution_plan(
            stripped_protocol_hash,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )

    other_core, other_descriptor = _descriptor(
        "s11_bear", "s11_fixture"
    )
    other_core["qualification"] = "mutated-core"
    other_descriptor["core"] = other_core
    wrong_core = deepcopy(downstream)
    wrong_core["protocol_descriptor"] = other_descriptor
    wrong_core["protocol_hash"] = protocol_hash(other_descriptor)
    wrong_core["protocol_core_hash"] = protocol_hash(other_core)
    _expect_error(
        "frozen manifest from another core protocol",
        lambda: derive_execution_plan(
            wrong_core,
            dataset_name="s11_fixture",
            requested_n_trials=100,
            requested_use_pruner=True,
            execution_runtime=runtime,
        ),
    )

    legacy = derive_execution_plan(
        {"name_short": "legacy", "dofs": [0]},
        dataset_name="legacy",
        requested_n_trials=9,
        requested_use_pruner=False,
    )
    assert legacy["mode"] == "legacy"
    assert legacy["effective_n_trials"] == 9
    assert legacy["campaign_run_tag"] is None
    assert legacy["execution_receipt_sha256"] is None
    assert legacy["block_reference_manifest_sha256"] is None
    print("PASS: hyperparameter policy derivation/authentication/mutations")


if __name__ == "__main__":
    main()
