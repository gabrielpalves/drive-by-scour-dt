"""Mutation checks for Option-C adjudication and channel selection.

Run: ``py -3.13 check_paper1_refit_contract.py``
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from core.paper1_refit_contract import (
    CHANNEL_AGGREGATION_POLICY,
    CHANNEL_RESULT_SCHEMA,
    DEVELOPMENT_AGGREGATION_POLICY,
    DEVELOPMENT_RESULT_SCHEMA,
    Paper1RefitContractError,
    build_channel_selection_artifact,
    build_development_artifact,
    seal_channel_result,
    seal_development_result,
    validate_channel_selection_artifact,
    validate_development_artifact,
)
from core.paper1_selection import build_selection_artifact
from core.paper1_dispatch import assigned_training_host, training_manifests
from core.paper1_training_contract import (
    DEVELOPMENT_INIT_SEEDS,
    DEVELOPMENT_N_REPEATS,
    DEVELOPMENT_N_SPLITS,
    DEVELOPMENT_PARTITION_SEED,
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    OUTER_SPLIT_SEED,
    RETAINED_PIPELINE_SLOTS,
    SCREEN_REFIT_SEEDS,
    canonical_json_bytes,
    complete_job_grid,
)
from training import paper1_executor as executor


FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    FAILURES += int(not condition)


def rejects(name: str, function) -> None:
    try:
        function()
    except (Paper1RefitContractError, executor.Paper1ExecutionError):
        check(name, True)
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation accepted")


def sha(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def params_for(cell) -> dict:
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


def candidate(cell, restart_seed: int) -> dict:
    hpo_job = next(
        job for job in complete_job_grid()["phases"]["hpo"]
        if job["phase"] == "f40s_factorial_hpo"
        and job["pipeline"] == cell.cell_id
        and job["hpo_restart_seed"] == restart_seed
    )
    params = params_for(cell)
    return {
        "pipeline": cell.cell_id,
        "hpo_restart_seed": restart_seed,
        "hpo_job_id": hpo_job["job_id"],
        "hpo_identity_sha256": "1" * 64,
        "hpo_completion_sha256": "2" * 64,
        "hpo_metadata_sha256": "3" * 64,
        "hpo_study_sha256": "4" * 64,
        "protocol_core_hash": "5" * 64,
        "protocol_hash": "6" * 64,
        "params": params,
        "params_sha256": sha(params),
        "frozen_checkpoint_epochs": 7,
    }


def partition(*, fold_index: int | None) -> dict:
    if fold_index is None:
        train, validation = [0, 1], [2]
        seed = n_splits = n_repeats = None
    else:
        validation = [fold_index]
        train = sorted({0, 1, 2} - {fold_index})
        seed = DEVELOPMENT_PARTITION_SEED
        n_splits = DEVELOPMENT_N_SPLITS
        n_repeats = DEVELOPMENT_N_REPEATS
    return {
        "outer_split_seed": OUTER_SPLIT_SEED,
        "development_partition_seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "fold_index": fold_index,
        "split_manifest_sha256": "7" * 64,
        "development_idx_sha256": "8" * 64,
        "outer_test_idx_sha256": "9" * 64,
        "development_states": [0, 1, 2],
        "outer_test_states": [3],
        "train_states": train,
        "validation_states": validation,
    }


def metrics(states: list[int], base: float) -> dict:
    scores = [base + state / 1000 for state in states]
    return {
        "state": states,
        "scour_mse": scores,
        "all_head_mse": scores,
        "predicted_max_scour_pct": [1.0 + state for state in states],
    }


def development_results() -> list[dict]:
    cells = {cell.cell_id: cell for cell in FACTORIAL_CELLS}
    cell_index = {cell.cell_id: index for index, cell in enumerate(FACTORIAL_CELLS)}
    seed_index = {seed: index for index, seed in enumerate(HPO_RESTART_SEEDS)}
    results = []
    for job in complete_job_grid()["phases"]["development_adjudication"]:
        base = (
            1.0 + cell_index[job["pipeline"]]
            + seed_index[job["candidate_restart_seed"]] / 10
            + DEVELOPMENT_INIT_SEEDS.index(job["initialization_seed"]) / 100
        )
        part = partition(fold_index=job["fold_index"])
        results.append(seal_development_result({
            "schema": DEVELOPMENT_RESULT_SCHEMA,
            "campaign_run_tag": "paper1-refit-fixture",
            "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
            "job": job,
            "candidate": candidate(cells[job["pipeline"]], job["candidate_restart_seed"]),
            "partition": part,
            "initialization_seed": job["initialization_seed"],
            "outer_test_observations_accessed": False,
            "metrics": metrics(part["validation_states"], base),
        }))
    return results


def channel_results(adjudication: dict) -> list[dict]:
    jobs = complete_job_grid()["phases"]["channel_screen"]
    input_order = []
    for job in jobs:
        selector = job["input_selector"]
        if selector not in input_order:
            input_order.append(selector)
    results = []
    for job in jobs:
        slot = job["pipeline"]
        canonical_slot = adjudication["canonical_slot"][slot]
        pipeline = adjudication["slot_resolution"][slot]
        selector = job["input_selector"]
        # Singles deliberately look better than pairs, proving they remain
        # nonselectable. Pair [1,3] is the registered synthetic winner.
        if len(selector) == 1:
            base = 0.01 + selector[0] / 10000
        else:
            base = 0.1 + input_order.index(selector) / 1000
            if selector == [1, 3]:
                base = 0.05
        base += SCREEN_REFIT_SEEDS.index(job["initialization_seed"]) / 100000
        canonical_job = next(
            item for item in jobs
            if item["pipeline"] == canonical_slot
            and item["input_selector"] == selector
            and item["initialization_seed"] == job["initialization_seed"]
        )
        results.append(seal_channel_result({
            "schema": CHANNEL_RESULT_SCHEMA,
            "execution_kind": (
                "refit" if canonical_slot == slot else "deduplicated_alias"
            ),
            "campaign_run_tag": adjudication["campaign_run_tag"],
            "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
            "adjudication_artifact_sha256": adjudication["artifact_sha256"],
            "job": job,
            "slot_resolution": adjudication["slot_resolution"],
            "canonical_slot": canonical_slot,
            "pipeline": pipeline,
            "frozen_candidate": adjudication["winner_by_pipeline"][pipeline]["candidate"],
            "partition": partition(fold_index=None),
            "outer_test_observations_accessed": False,
            "canonical_job_id": canonical_job["job_id"],
            "metrics": metrics([2], base),
        }))
    return results


def resign_artifact(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = sha(result)
    return result


def main() -> None:
    print("PAPER1 OPTION-C REFIT / CHANNEL CONTRACT CHECKS")
    dev_results = development_results()
    adjudication = build_development_artifact(dev_results)
    check(
        "Option-C artefact is exactly 480 fits and one complete 3-fold OOF partition",
        len(adjudication["source_results"]) == 480
        and adjudication["partition_policy"] == {
            "seed": 271828,
            "n_splits": 3,
            "n_repeats": 1,
            "fold_indices": [0, 1, 2],
            "outer_test_observations_accessed": False,
        }
        and adjudication["aggregation_policy"] == DEVELOPMENT_AGGREGATION_POLICY
        and validate_development_artifact(adjudication) == adjudication,
    )
    check(
        "candidate/init folds cover development states exactly once",
        all(
            summary["development_states"] == [0, 1, 2]
            and len(summary["state_initialization_scores"]) == 6
            for summary in adjudication["candidate_summaries"]
        ),
    )

    rejects("missing OOF fit", lambda: build_development_artifact(dev_results[:-1]))
    duplicate = dev_results[:-1] + [dev_results[-2]]
    rejects("duplicate OOF fit", lambda: build_development_artifact(duplicate))
    mutant = deepcopy(dev_results[0]); mutant["partition"]["development_partition_seed"] += 1
    rejects("partition seed drift", lambda: seal_development_result(mutant))
    mutant = deepcopy(dev_results[0]); mutant["outer_test_observations_accessed"] = True
    rejects("outer-test access during adjudication", lambda: seal_development_result(mutant))
    mutant = deepcopy(dev_results[0]); mutant["candidate"]["hpo_job_id"] = "foreign"
    rejects("candidate HPO lineage drift", lambda: seal_development_result(mutant))
    mutant = deepcopy(dev_results[0]); mutant["candidate"]["params"]["lr"] = 9.0
    mutant["candidate"]["params_sha256"] = sha(mutant["candidate"]["params"])
    mutated_results = [seal_development_result(mutant), *dev_results[1:]]
    rejects("candidate parameters outside registered space", lambda: build_development_artifact(mutated_results))
    mutant_artifact = deepcopy(adjudication)
    mutant_artifact["winner_by_pipeline"][adjudication["best_raw"]]["hpo_restart_seed"] = HPO_RESTART_SEEDS[-1]
    rejects("forged candidate winner", lambda: validate_development_artifact(resign_artifact(mutant_artifact)))

    exact_development_job = dev_results[0]["job"]
    mutant_job = deepcopy(exact_development_job); mutant_job["fold_index"] = None
    rejects(
        "executor rejects missing explicit fold index",
        lambda: executor.validate_development_adjudication_job(mutant_job),
    )

    screen_results = channel_results(adjudication)
    channel = build_channel_selection_artifact(screen_results, adjudication)
    check(
        "channel artefact authenticates full 720-job tensor and alias deduplication",
        len(channel["source_results"]) == 720
        and channel["aggregation_policy"] == CHANNEL_AGGREGATION_POLICY
        and len(channel["reported_inputs"]) == 36
        and len(channel["eligible_pairs"]) == 28
        and channel["selected_pair"] == [1, 3]
        and len(channel["unique_resolved_pipelines"]) == 2
        and validate_channel_selection_artifact(
            channel, adjudication=adjudication
        ) == channel,
    )
    check(
        "singles are reported but never selection-eligible",
        all(
            entry["selection_eligible"] is (len(entry["input_selector"]) == 2)
            for entry in channel["input_scores"]
        ),
    )
    rejects("incomplete channel result tensor", lambda: build_channel_selection_artifact(screen_results[:-1], adjudication))
    alias_index = next(
        index for index, result in enumerate(screen_results)
        if result["execution_kind"] == "deduplicated_alias"
    )
    mutant = deepcopy(screen_results[alias_index]); mutant["metrics"]["scour_mse"][0] += 1
    mutated_screen = list(screen_results); mutated_screen[alias_index] = seal_channel_result(mutant)
    rejects("alias metrics differ from canonical job", lambda: build_channel_selection_artifact(mutated_screen, adjudication))
    for label, mutate in (
        ("weighting policy drift", lambda value: value["aggregation_policy"].update({"state_weighting": "passage"})),
        ("eligible pair set drift", lambda value: value["eligible_pairs"].pop()),
        ("reported input order drift", lambda value: value["reported_inputs"].reverse()),
        ("selected pair forged", lambda value: value.update({"selected_pair": [0, 1]})),
    ):
        mutant_artifact = deepcopy(channel); mutate(mutant_artifact)
        rejects(label, lambda value=resign_artifact(mutant_artifact): validate_channel_selection_artifact(value, adjudication=adjudication))

    final_selection = build_selection_artifact(
        campaign_run_tag=adjudication["campaign_run_tag"],
        selected_pair=channel["selected_pair"],
        best_raw=adjudication["best_raw"],
        best_paa=adjudication["best_paa"],
        evidence_sha256={
            "factorial_hpo_manifest": "a" * 64,
            "development_adjudication_manifest": adjudication["artifact_sha256"],
            "channel_screen_manifest": channel["artifact_sha256"],
        },
    )
    check(
        "downstream selection binds both complete aggregate artefacts",
        final_selection["selected_pair"] == [1, 3]
        and final_selection["evidence_sha256"]["development_adjudication_manifest"]
        == adjudication["artifact_sha256"]
        and final_selection["evidence_sha256"]["channel_screen_manifest"]
        == channel["artifact_sha256"],
    )

    # Filesystem-level execution/resume proof with only the expensive numerical
    # fold evaluator replaced. Manifest validation, identities, result sealing,
    # immutable completion, external artefact authentication, and alias routing
    # all remain live.
    manifests = training_manifests()
    dev_job = exact_development_job
    dev_host = assigned_training_host(dev_job)
    original_candidate = executor._load_hpo_candidate
    original_partition = executor._load_refit_data_and_partition
    import core.execution_environment as execution_environment

    original_enforce = execution_environment.enforce_execution_block
    saved_env = {name: os.environ.get(name) for name in (
        executor.DATA_ROOT_ENV,
        executor.RESULTS_ROOT_ENV,
        executor.CACHE_ROOT_ENV,
        executor.STUDY_ROOT_ENV,
        executor.RECEIPT_ROOT_ENV,
        executor.RUN_TAG_ENV,
        "TTBI_PAPER1_ADJUDICATION_ARTIFACT",
        "TTBI_PAPER1_ADJUDICATION_ARTIFACT_SHA256",
    )}
    refit_calls = {"development": 0, "channel": 0}
    try:
        with tempfile.TemporaryDirectory(prefix="paper1-refit-executor-") as td:
            root = Path(td).resolve()
            paths = {
                executor.DATA_ROOT_ENV: root / "data",
                executor.RESULTS_ROOT_ENV: root / "results",
                executor.CACHE_ROOT_ENV: root / "cache",
                executor.STUDY_ROOT_ENV: root / "studies",
                executor.RECEIPT_ROOT_ENV: root / "receipts",
            }
            for name, path in paths.items():
                path.mkdir(parents=True)
                os.environ[name] = str(path)
            os.environ[executor.RUN_TAG_ENV] = adjudication["campaign_run_tag"]

            cells = {cell.cell_id: cell for cell in FACTORIAL_CELLS}

            def fake_candidate(**kwargs):
                value = candidate(cells[kwargs["pipeline"]], kwargs["restart_seed"])
                return value, {
                    "protocol_hash": value["protocol_hash"],
                    "protocol_core_hash": value["protocol_core_hash"],
                }

            def fake_partition(*, fold_index, **_kwargs):
                part = partition(fold_index=fold_index)
                fold = SimpleNamespace(
                    train_idx=np.asarray(part["train_states"], dtype=np.int64),
                    val_idx=np.asarray(part["validation_states"], dtype=np.int64),
                    train_states=np.asarray(part["train_states"], dtype=np.int64),
                    val_states=np.asarray(part["validation_states"], dtype=np.int64),
                )
                return (
                    np.zeros((4, 1, 2), dtype=np.float32),
                    np.zeros((4, 2), dtype=np.float32),
                    np.arange(4, dtype=np.int64),
                    fold,
                    part,
                )

            executor._load_hpo_candidate = fake_candidate
            executor._load_refit_data_and_partition = fake_partition
            execution_environment.enforce_execution_block = lambda **_kwargs: {
                "runtime": {"schema": "fixture-runtime"},
                "receipt_sha256": "a" * 64,
            }

            def fake_dev_refit(**kwargs):
                refit_calls["development"] += 1
                return metrics(
                    sorted(int(value) for value in kwargs["fold"].val_states),
                    0.25,
                )

            completion = executor.execute_development_adjudication_job(
                dev_job, manifests[dev_host], refit_runner=fake_dev_refit
            )
            restarted = executor.execute_development_adjudication_job(
                dev_job, manifests[dev_host], refit_runner=fake_dev_refit
            )
            check(
                "development executor writes immutable resumable completion",
                completion == restarted
                and completion["completion_kind"] == "development_oof_refit"
                and refit_calls["development"] == 1,
            )

            adjudication_path = root / "adjudication.json"
            adjudication_path.write_bytes(canonical_json_bytes(adjudication))
            os.environ["TTBI_PAPER1_ADJUDICATION_ARTIFACT"] = str(adjudication_path)
            os.environ["TTBI_PAPER1_ADJUDICATION_ARTIFACT_SHA256"] = adjudication[
                "artifact_sha256"
            ]
            canonical_screen_job = next(
                job for job in complete_job_grid()["phases"]["channel_screen"]
                if job["pipeline"] == "f40s_best_raw"
                and job["input_selector"] == [0]
                and job["initialization_seed"] == SCREEN_REFIT_SEEDS[0]
            )
            screen_host = assigned_training_host(canonical_screen_job)

            def fake_screen_refit(**kwargs):
                refit_calls["channel"] += 1
                return metrics(
                    sorted(int(value) for value in kwargs["fold"].val_states),
                    0.5,
                )

            screen_completion = executor.execute_channel_screen_job(
                canonical_screen_job,
                manifests[screen_host],
                refit_runner=fake_screen_refit,
            )
            screen_restart = executor.execute_channel_screen_job(
                canonical_screen_job,
                manifests[screen_host],
                refit_runner=fake_screen_refit,
            )
            alias_slot = next(
                slot for slot in RETAINED_PIPELINE_SLOTS
                if adjudication["canonical_slot"][slot] == "f40s_best_raw"
                and slot != "f40s_best_raw"
            )
            alias_job = next(
                job for job in complete_job_grid()["phases"]["channel_screen"]
                if job["pipeline"] == alias_slot
                and job["input_selector"] == [0]
                and job["initialization_seed"] == SCREEN_REFIT_SEEDS[0]
            )
            alias_completion = executor.execute_channel_screen_job(
                alias_job, manifests[screen_host], refit_runner=fake_screen_refit
            )
            check(
                "channel executor resumes canonical refit and aliases without retraining",
                screen_completion == screen_restart
                and screen_completion["completion_kind"] == "channel_screen_refit"
                and alias_completion["completion_kind"] == "channel_screen_alias"
                and refit_calls["channel"] == 1,
            )
    finally:
        executor._load_hpo_candidate = original_candidate
        executor._load_refit_data_and_partition = original_partition
        execution_environment.enforce_execution_block = original_enforce
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print()
    if FAILURES:
        raise SystemExit(f"PAPER1 REFIT CONTRACT: {FAILURES} CHECK(S) FAILED")
    print("PAPER1 REFIT CONTRACT: ALL PASS")


if __name__ == "__main__":
    main()
