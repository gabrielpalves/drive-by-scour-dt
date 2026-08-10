"""Mutation checks for block-local freezes and sealed report adapters.

Run: ``py -3.13 check_paper1_freeze_contract.py``
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from core.campaign_contract import campaign_stage_contract
from core.paper1_dispatch import assigned_training_host, training_manifests
from core.paper1_freeze_contract import (
    BLOCK_FREEZE_ARTIFACT_ENV,
    BLOCK_FREEZE_ARTIFACT_SHA256_ENV,
    SEALED_RESULT_SCHEMA,
    SELECTED_CHAMPION_SCHEMA,
    Paper1FreezeContractError,
    build_block_freeze_artifact,
    freeze_for_slot,
    seal_sealed_result,
    validate_block_freeze_artifact,
)
from core.paper1_selection import (
    SELECTION_ARTIFACT_ENV,
    SELECTION_ARTIFACT_SHA256_ENV,
    build_selection_artifact,
)
from core.paper1_training_contract import (
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    POST_FREEZE_STABILITY_SEEDS,
    RETAINED_PIPELINE_SLOTS,
    SCREEN_REFIT_SEEDS,
    TRAINING_EPOCHS,
    canonical_json_bytes,
    complete_job_grid,
)
from training import paper1_executor as executor


FAILURES = 0
CELLS = {cell.cell_id: cell for cell in FACTORIAL_CELLS}


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    passed = bool(condition)
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not passed)


def rejects(name: str, operation) -> None:
    try:
        operation()
    except (Paper1FreezeContractError, executor.Paper1ExecutionError):
        check(name, True)
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation accepted")


def sha(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def params_for(pipeline: str) -> dict:
    cell = CELLS[pipeline]
    params = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "n_conv_layers": 2,
        "n_dense_layers": 1,
        "n_filters_l0": 16,
        "kernel_size_l0": 3,
        "pooling_l0": True,
        "n_filters_l1": 16,
        "kernel_size_l1": 3,
        "pooling_l1": True,
        "n_dense_units_l0": 32,
        "dropout_l0": 0.1,
    }
    if cell.lstm:
        params.update({"lstm_hidden_size": 32, "lstm_num_layers": 1})
    if cell.multi_rate_pooling:
        params["nhits_pool_rates_key"] = "1_2_4"
    return params


def selection_artifact() -> dict:
    return build_selection_artifact(
        campaign_run_tag="paper1-freeze-fixture",
        selected_pair=[1, 3],
        # Baseline-equal winners deliberately exercise slot alias deduplication.
        best_raw="RAW_POS0_LSTM0_MR0",
        best_paa="PAA_POS0_LSTM0_MR0",
        evidence_sha256={
            "factorial_hpo_manifest": "1" * 64,
            "development_adjudication_manifest": "2" * 64,
            "channel_screen_manifest": "3" * 64,
        },
    )


def source_lineage() -> dict:
    return {
        "environment_lock_sha256": "4" * 64,
        "python_runtime_source_root_sha256": "5" * 64,
        "python_runtime_source_file_count": 500,
        "generator_source_root_sha256": "6" * 64,
        "generator_source_file_count": 120,
        "dataset_content_root_sha256": "7" * 64,
        "generation_fingerprint": "8" * 64,
        "qualification_source_sha256": "9" * 64,
    }


def champion(
    *, stage: str, slot: str, seed: int, selection: dict,
    objective: float | None = None,
) -> dict:
    phase = (
        "f40s_selected_pair_hpo"
        if stage == "F40-S" else "block_selected_pair_hpo"
    )
    job = next(
        value for value in complete_job_grid()["phases"]["hpo"]
        if value["stage"] == stage
        and value["phase"] == phase
        and value["pipeline"] == slot
        and value["hpo_restart_seed"] == seed
    )
    pipeline = selection["slot_resolution"][slot]
    params = params_for(pipeline)
    seed_index = HPO_RESTART_SEEDS.index(seed)
    return {
        "schema": SELECTED_CHAMPION_SCHEMA,
        "stage": stage,
        "canonical_slot": slot,
        "pipeline": pipeline,
        "selected_pair": selection["selected_pair"],
        "hpo_restart_seed": seed,
        "hpo_job_id": job["job_id"],
        "hpo_identity_sha256": "a" * 64,
        "hpo_completion_sha256": "b" * 64,
        "hpo_metadata_sha256": "c" * 64,
        "hpo_study_sha256": hashlib.sha256(
            f"{stage}|{slot}|{seed}".encode("ascii")
        ).hexdigest(),
        "execution_environment_sha256": hashlib.sha256(
            f"host|{seed}".encode("ascii")
        ).hexdigest(),
        "execution_compatibility_sha256": "d" * 64,
        "execution_receipt_sha256": "e" * 64,
        "protocol_core_hash": "f" * 64,
        "protocol_hash": hashlib.sha256(stage.encode("ascii")).hexdigest(),
        "source_lineage": source_lineage(),
        "best_trial_number": seed_index,
        "best_trial_value": (
            0.1 + seed_index / 100
            if objective is None else objective
        ),
        "terminal_counts": {
            "COMPLETE": 40,
            "PRUNED": 60,
            "FAIL": 0,
            "RUNNING": 0,
            "WAITING": 0,
            "total": 100,
        },
        "params": params,
        "params_sha256": sha(params),
        "frozen_checkpoint_epochs": 7 + seed_index,
    }


def champion_inventory(stage: str, selection: dict) -> list[dict]:
    canonical_slots = [
        slot for slot in RETAINED_PIPELINE_SLOTS
        if selection["canonical_slot"][slot] == slot
    ]
    return [
        champion(stage=stage, slot=slot, seed=seed, selection=selection)
        for slot in canonical_slots
        for seed in HPO_RESTART_SEEDS
    ]


def resign_artifact(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = sha(result)
    return result


def partition() -> dict:
    return {
        "outer_split_seed": 42,
        "split_manifest_sha256": "1" * 64,
        "development_idx_sha256": executor._array_sha256(
            np.asarray([0, 1, 2], dtype=np.int64)
        ),
        "outer_test_idx_sha256": executor._array_sha256(
            np.asarray([3], dtype=np.int64)
        ),
        "development_states": [0, 1, 2],
        "outer_test_states": [3],
    }


def metrics() -> dict:
    return {
        "state": [3],
        "scour_mse": [0.25],
        "all_head_mse": [0.5],
        "predicted_max_scour_pct": [1.5],
    }


def canonical_job(phase_key: str, stage: str, slot: str, seed: int) -> dict:
    return next(
        job for job in complete_job_grid()["phases"][phase_key]
        if job["stage"] == stage
        and job["pipeline"] == slot
        and job["initialization_seed"] == seed
    )


def sealed_result(job: dict, freeze: dict) -> dict:
    freeze_stage = (
        job["stage"]
        if job["phase"] == "post_freeze_sealed_test_stability" else "F40-S"
    )
    reporting_role = (
        "primary_post_freeze_report_only"
        if job["phase"] == "post_freeze_sealed_test_stability"
        else "secondary_nonselection"
    )
    claim = freeze_for_slot(freeze, stage=freeze_stage, slot=job["pipeline"])
    phase_key = (
        "post_freeze_stability"
        if job["phase"] == "post_freeze_sealed_test_stability"
        else "secondary_frozen_transfer"
    )
    canonical = canonical_job(
        phase_key,
        job["stage"],
        claim["canonical_slot"],
        job["initialization_seed"],
    )
    config = executor._refit_config(
        job=canonical,
        pipeline=claim["pipeline"],
        dofs=claim["selected_pair"],
    )
    part = partition()
    robust = {
        "schema": "ttbi-post-freeze-stability-v1",
        "evaluation_scope": "sealed_outer_test_post_freeze",
        "selection_permitted": False,
        "outer_test_observations_accessed": True,
        "plan": {
            "config": config,
            "params": claim["winner"]["params"],
            "provenance": {
                "freeze_artifact_sha256": freeze["artifact_sha256"],
                "selection_artifact_sha256": freeze["selection_artifact"][
                    "artifact_sha256"
                ],
                "stage": job["stage"],
                "freeze_stage": freeze_stage,
                "reporting_role": reporting_role,
            },
            "development_idx_sha256": part["development_idx_sha256"],
            "sealed_outer_test_idx_sha256": part["outer_test_idx_sha256"],
            "groups_sha256": "2" * 64,
            "initialization_seeds": [job["initialization_seed"]],
            "n_epochs": claim["winner"]["frozen_checkpoint_epochs"],
            "max_epochs": TRAINING_EPOCHS,
            "n_scour_heads": len(
                campaign_stage_contract(job["stage"])["learning"][
                    "target_supports"
                ]
            ),
        },
        "complete": True,
        "runs": [{
            "initialization_seed": job["initialization_seed"],
            "outer_test_states": [3],
            "metrics": metrics(),
        }],
        "n_completed_refits": 1,
        "n_expected_refits": 1,
    }
    is_alias = job["job_id"] != canonical["job_id"]
    return seal_sealed_result({
        "schema": SEALED_RESULT_SCHEMA,
        "execution_kind": "deduplicated_alias" if is_alias else "refit",
        "campaign_run_tag": freeze["campaign_run_tag"],
        "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
        "job": job,
        "reporting_role": reporting_role,
        "selection_permitted": False,
        "selection_artifact_sha256": freeze["selection_artifact"][
            "artifact_sha256"
        ],
        "freeze_artifact_sha256": freeze["artifact_sha256"],
        "freeze_stage": freeze_stage,
        "slot_resolution": freeze["slot_resolution"],
        "canonical_slot": claim["canonical_slot"],
        "pipeline": claim["pipeline"],
        "selected_pair": claim["selected_pair"],
        "frozen_champion": claim["winner"],
        "partition": part,
        "canonical_job_id": canonical["job_id"],
        "canonical_result_sha256": "3" * 64 if is_alias else None,
        "robustness": robust,
    }, freeze_artifact=freeze)


def main() -> None:
    print("PAPER1 BLOCK FREEZE / SEALED REPORT CHECKS")
    selection = selection_artifact()
    inventory = champion_inventory("F40-S", selection)
    freeze = build_block_freeze_artifact(
        stage="F40-S", selection=selection, champions=inventory
    )
    check(
        "block freeze authenticates exact five-restart inventories and aliases",
        validate_block_freeze_artifact(freeze) == freeze
        and len(freeze["pipeline_freezes"]) == 2
        and all(
            len(item["restart_inventory"]) == 5
            for item in freeze["pipeline_freezes"]
        )
        and freeze["freeze_policy"][
            "sealed_outer_test_observations_accessed"
        ] is False,
    )
    check(
        "baseline-equal retained slots resolve to one frozen compute lineage",
        freeze_for_slot(
            freeze, stage="F40-S", slot="raw_cnn_gap_baseline"
        )["canonical_slot"] == "f40s_best_raw"
        and freeze_for_slot(
            freeze, stage="F40-S", slot="paa_cnn_gap_baseline"
        )["canonical_slot"] == "f40s_best_paa",
    )
    rejects(
        "missing selected-pair restart",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=inventory[:-1]
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["terminal_counts"].update({"FAIL": 1, "PRUNED": 59})
    rejects(
        "failed selected-pair restart",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["best_trial_value"] = float("inf")
    rejects(
        "nonfinite inner-validation objective",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["params"]["lr"] = 9.0
    mutant[0]["params_sha256"] = sha(mutant[0]["params"])
    rejects(
        "selected-pair parameter lineage drift",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["frozen_checkpoint_epochs"] = TRAINING_EPOCHS + 1
    rejects(
        "frozen checkpoint epoch drift",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["source_lineage"]["generator_source_root_sha256"] = "0" * 64
    rejects(
        "five-restart source lineage disagreement",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    foreign_slot = mutant[-1]["canonical_slot"]
    for item in mutant:
        if item["canonical_slot"] == foreign_slot:
            item["source_lineage"]["generator_source_root_sha256"] = "0" * 64
    rejects(
        "cross-pipeline source lineage disagreement",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["selected_pair"] = [0, 1]
    rejects(
        "selected pair drift",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )
    mutant = deepcopy(inventory)
    mutant[0]["stage"] = "F40-M"
    rejects(
        "selected-HPO stage drift",
        lambda: build_block_freeze_artifact(
            stage="F40-S", selection=selection, champions=mutant
        ),
    )

    tied_inventory = champion_inventory("F40-S", selection)
    for item in tied_inventory:
        if item["canonical_slot"] == "f40s_best_raw":
            item["best_trial_value"] = (
                0.05 if item["hpo_restart_seed"] in HPO_RESTART_SEEDS[:2]
                else 0.5
            )
    tied = build_block_freeze_artifact(
        stage="F40-S", selection=selection, champions=tied_inventory
    )
    tied_raw = next(
        item for item in tied["pipeline_freezes"]
        if item["canonical_slot"] == "f40s_best_raw"
    )
    check(
        "objective ties resolve by restart seed then trial number",
        tied_raw["winner"]["hpo_restart_seed"] == min(HPO_RESTART_SEEDS[:2]),
    )
    mutant_freeze = deepcopy(freeze)
    mutant_freeze["pipeline_freezes"][0]["winner"] = deepcopy(
        mutant_freeze["pipeline_freezes"][0]["restart_inventory"][-1]
    )
    rejects(
        "forged block winner",
        lambda: validate_block_freeze_artifact(resign_artifact(mutant_freeze)),
    )

    post_job = canonical_job(
        "post_freeze_stability", "F40-S", "f40s_best_raw",
        POST_FREEZE_STABILITY_SEEDS[0],
    )
    post_result = sealed_result(post_job, freeze)
    check(
        "post-freeze result is one report-only complete sealed-test refit",
        post_result["selection_permitted"] is False
        and post_result["reporting_role"]
        == "primary_post_freeze_report_only"
        and post_result["robustness"]["outer_test_observations_accessed"]
        is True,
    )
    for label, mutate in (
        ("selection-permitted misuse", lambda value: value.update({"selection_permitted": True})),
        ("reporting-role misuse", lambda value: value.update({"reporting_role": "primary"})),
        ("pair drift in sealed report", lambda value: value.update({"selected_pair": [0, 1]})),
        ("freeze-stage drift", lambda value: value.update({"freeze_stage": "F40-M"})),
        (
            "premature outer-access flag drift",
            lambda value: value["robustness"].update({
                "outer_test_observations_accessed": False
            }),
        ),
        (
            "initialization seed drift",
            lambda value: value["robustness"]["plan"].update({
                "initialization_seeds": [999]
            }),
        ),
    ):
        mutant = deepcopy(post_result)
        mutant.pop("result_sha256")
        mutate(mutant)
        rejects(
            label,
            lambda value=mutant: seal_sealed_result(
                value, freeze_artifact=freeze
            ),
        )

    alias_job = canonical_job(
        "post_freeze_stability", "F40-S", "raw_cnn_gap_baseline",
        POST_FREEZE_STABILITY_SEEDS[0],
    )
    alias_result = sealed_result(alias_job, freeze)
    check(
        "alias result binds canonical job/result without changing compute plan",
        alias_result["execution_kind"] == "deduplicated_alias"
        and alias_result["canonical_job_id"] == post_job["job_id"]
        and alias_result["canonical_result_sha256"] == "3" * 64,
    )

    # Filesystem-level proof: external selection/freeze authentication and the
    # robustness implementation remain live; only the data cache and neural
    # fold evaluator are replaced by deterministic micro fixtures.
    manifests = training_manifests()
    original_sealed_loader = executor._load_sealed_refit_data
    import core.execution_environment as execution_environment
    original_enforce = execution_environment.enforce_execution_block
    saved_env = {
        name: os.environ.get(name)
        for name in (
            executor.DATA_ROOT_ENV,
            executor.RESULTS_ROOT_ENV,
            executor.CACHE_ROOT_ENV,
            executor.RECEIPT_ROOT_ENV,
            executor.RUN_TAG_ENV,
            SELECTION_ARTIFACT_ENV,
            SELECTION_ARTIFACT_SHA256_ENV,
            BLOCK_FREEZE_ARTIFACT_ENV,
            BLOCK_FREEZE_ARTIFACT_SHA256_ENV,
        )
    }
    calls = {"data": 0, "fit": 0}
    try:
        with tempfile.TemporaryDirectory(prefix="paper1-freeze-executor-") as td:
            root = Path(td).resolve()
            for name, leaf in (
                (executor.DATA_ROOT_ENV, "data"),
                (executor.RESULTS_ROOT_ENV, "results"),
                (executor.CACHE_ROOT_ENV, "cache"),
                (executor.RECEIPT_ROOT_ENV, "receipts"),
            ):
                path = root / leaf
                path.mkdir()
                os.environ[name] = str(path)
            os.environ[executor.RUN_TAG_ENV] = selection["campaign_run_tag"]
            selection_path = root / "selection.json"
            freeze_path = root / "freeze-f40s.json"
            selection_path.write_bytes(canonical_json_bytes(selection))
            freeze_path.write_bytes(canonical_json_bytes(freeze))
            os.environ[SELECTION_ARTIFACT_ENV] = str(selection_path)
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = selection[
                "artifact_sha256"
            ]
            os.environ[BLOCK_FREEZE_ARTIFACT_ENV] = str(freeze_path)
            os.environ[BLOCK_FREEZE_ARTIFACT_SHA256_ENV] = freeze[
                "artifact_sha256"
            ]

            def fake_data(**_kwargs):
                calls["data"] += 1
                return (
                    np.zeros((4, 1, 2), dtype=np.float32),
                    np.zeros((4, 2), dtype=np.float32),
                    np.asarray([0, 1, 2, 3], dtype=np.int64),
                    np.asarray([0, 1, 2], dtype=np.int64),
                    np.asarray([3], dtype=np.int64),
                    partition(),
                )

            def fake_fit(**kwargs):
                calls["fit"] += 1
                assert kwargs["fold"].train_states.tolist() == [0, 1, 2]
                assert kwargs["fold"].val_states.tolist() == [3]
                return metrics()

            executor._load_sealed_refit_data = fake_data
            execution_environment.enforce_execution_block = lambda **_kwargs: {
                "runtime": {"schema": "fixture-runtime"},
                "receipt_sha256": "a" * 64,
            }
            host = assigned_training_host(post_job)
            completion = executor.execute_post_freeze_stability_job(
                post_job, manifests[host], fit_evaluate=fake_fit
            )
            restarted = executor.execute_post_freeze_stability_job(
                post_job, manifests[host], fit_evaluate=fake_fit
            )
            alias_completion = executor.execute_post_freeze_stability_job(
                alias_job, manifests[host], fit_evaluate=fake_fit
            )
            check(
                "post-freeze executor resumes and deduplicates slot aliases",
                completion == restarted
                and completion["completion_kind"]
                == "post_freeze_stability_refit"
                and alias_completion["completion_kind"]
                == "post_freeze_stability_alias"
                and calls == {"data": 1, "fit": 1},
            )

            premature_job = canonical_job(
                "post_freeze_stability", "F40-S", "f40s_best_raw",
                POST_FREEZE_STABILITY_SEEDS[1],
            )
            before = dict(calls)
            os.environ[BLOCK_FREEZE_ARTIFACT_SHA256_ENV] = "0" * 64
            rejects(
                "outer indices stay closed when external freeze hash is wrong",
                lambda: executor.execute_post_freeze_stability_job(
                    premature_job,
                    manifests[assigned_training_host(premature_job)],
                    fit_evaluate=fake_fit,
                ),
            )
            check("premature freeze failure performs no data/test access", calls == before)
            os.environ[BLOCK_FREEZE_ARTIFACT_SHA256_ENV] = freeze[
                "artifact_sha256"
            ]

            secondary_job = canonical_job(
                "secondary_frozen_transfer", "F40-M", "f40s_best_raw",
                SCREEN_REFIT_SEEDS[0],
            )
            secondary_completion = executor.execute_secondary_frozen_transfer_job(
                secondary_job,
                manifests[assigned_training_host(secondary_job)],
                fit_evaluate=fake_fit,
            )
            secondary_result_path = (
                root / "results" / secondary_job["stage"]
                / secondary_job["phase"] / secondary_job["job_id"]
                / "paper1_sealed_result.json"
            )
            secondary_result = __import__("json").loads(
                secondary_result_path.read_bytes()
            )
            check(
                "secondary executor uses F40-S freeze and remains nonselection",
                secondary_completion["completion_kind"]
                == "secondary_frozen_transfer_refit"
                and secondary_result["freeze_stage"] == "F40-S"
                and secondary_result["reporting_role"]
                == "secondary_nonselection"
                and secondary_result["selection_permitted"] is False
                and calls == {"data": 2, "fit": 2},
            )
    finally:
        executor._load_sealed_refit_data = original_sealed_loader
        execution_environment.enforce_execution_block = original_enforce
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    source = Path(executor.__file__).read_text(encoding="utf-8")
    check(
        "selected champion loader authenticates DB terminal states and source lineage",
        all(token in source for token in (
            "def _load_selected_hpo_champion(",
            "counts[\"COMPLETE\"] + counts[\"PRUNED\"]",
            "frozen_checkpoint_epoch_count(",
            "_source_lineage_from_metadata(",
            "verify_standalone_dt_package(",
        )),
    )

    print()
    if FAILURES:
        raise SystemExit(f"PAPER1 FREEZE CONTRACT: {FAILURES} CHECK(S) FAILED")
    print("PAPER1 FREEZE CONTRACT: ALL PASS")


if __name__ == "__main__":
    main()
