"""Authenticated Paper-1 development-refit and channel-screen artefacts.

The aggregate artefacts embed the complete immutable per-job result tensor.
Validation therefore recomputes coverage, OOF completeness, candidate winners,
alias deduplication, equal-state/equal-pipeline weighting, and deterministic
tie-breaks instead of trusting outcome summaries written by the executor.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.paper1_training_contract import (
    DEVELOPMENT_INIT_SEEDS,
    DEVELOPMENT_N_REPEATS,
    DEVELOPMENT_N_SPLITS,
    DEVELOPMENT_PARTITION_SEED,
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    OUTER_SPLIT_SEED,
    PAA_CNN_GAP_BASELINE_ID,
    RAW_CNN_GAP_BASELINE_ID,
    RETAINED_PIPELINE_SLOTS,
    SCREEN_REFIT_SEEDS,
    canonical_json_bytes,
    channel_screen_inputs,
    complete_job_grid,
)


DEVELOPMENT_RESULT_SCHEMA = "paper1-development-refit-result-v1"
DEVELOPMENT_ARTIFACT_SCHEMA = "paper1-development-adjudication-artifact-v1"
CHANNEL_RESULT_SCHEMA = "paper1-channel-screen-refit-result-v1"
CHANNEL_SELECTION_SCHEMA = "paper1-channel-selection-artifact-v1"

ADJUDICATION_ARTIFACT_ENV = "TTBI_PAPER1_ADJUDICATION_ARTIFACT"
ADJUDICATION_ARTIFACT_SHA256_ENV = (
    "TTBI_PAPER1_ADJUDICATION_ARTIFACT_SHA256"
)
CHANNEL_SELECTION_ARTIFACT_ENV = "TTBI_PAPER1_CHANNEL_SELECTION_ARTIFACT"

SELECTION_METRIC = "paired state-clustered grouped-development OOF MSE"
DEVELOPMENT_AGGREGATION_POLICY = {
    "metric": "scour_mse",
    "state_weighting": "equal",
    "initialization_weighting": "equal_within_state",
    "candidate_tie_break": "lowest_hpo_restart_seed",
    "architecture_tie_break": "lexicographic_cell_id",
}
CHANNEL_AGGREGATION_POLICY = {
    "metric": "scour_mse",
    "seed_weighting": "equal_within_state",
    "state_weighting": "equal_within_unique_pipeline",
    "pipeline_weighting": "equal_across_unique_resolved_pipelines",
    "alias_policy": "deduplicate_resolved_pipeline_before_weighting",
    "eligible_set": "all_28_physical8_pairs_only",
    "singles_role": "reported_nonselectable",
    "tie_break": "lexicographic_ascending_pair",
}

_HEX = frozenset("0123456789abcdef")


class Paper1RefitContractError(RuntimeError):
    """A refit result or aggregate artefact violates its exact contract."""


def _load_canonical_artifact(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise Paper1RefitContractError(
            "artefact path must be absolute, regular, and non-symlink"
        )
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1RefitContractError(f"artefact is unreadable/non-JSON: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Paper1RefitContractError("artefact bytes are not canonical JSON")
    if not _is_sha(expected_sha256) or value.get("artifact_sha256") != expected_sha256:
        raise Paper1RefitContractError("artefact differs from external SHA-256")
    return value


def load_development_artifact(
    path: str | os.PathLike[str] | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    raw_path = os.fspath(path) if path is not None else os.environ.get(
        ADJUDICATION_ARTIFACT_ENV, ""
    )
    expected = expected_sha256 if expected_sha256 is not None else os.environ.get(
        ADJUDICATION_ARTIFACT_SHA256_ENV, ""
    )
    if not raw_path:
        raise Paper1RefitContractError(f"{ADJUDICATION_ARTIFACT_ENV} is required")
    if not expected:
        raise Paper1RefitContractError(
            f"{ADJUDICATION_ARTIFACT_SHA256_ENV} is required"
        )
    return validate_development_artifact(
        _load_canonical_artifact(raw_path, expected_sha256=expected)
    )


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@lru_cache(maxsize=None)
def _phase_jobs(phase_key: str) -> tuple[dict[str, Any], ...]:
    return tuple(deepcopy(complete_job_grid()["phases"][phase_key]))


@lru_cache(maxsize=1)
def _complete_grid_sha256() -> str:
    return str(complete_job_grid()["complete_grid_sha256"])


def _json(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Paper1RefitContractError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Paper1RefitContractError(f"{label} must be finite/non-negative")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        raise Paper1RefitContractError("cannot aggregate an empty score set")
    return math.fsum(values) / len(values)


def _exact_job(job: Mapping[str, Any], phase_key: str) -> dict[str, Any]:
    value = _json(dict(job))
    expected = {
        candidate["job_id"]: candidate
        for candidate in _phase_jobs(phase_key)
    }
    if value.get("job_id") not in expected or value != expected[value["job_id"]]:
        raise Paper1RefitContractError("result cites a foreign training job")
    return value


def _validated_metrics(value: object) -> dict[str, list[Any]]:
    expected = {
        "state",
        "scour_mse",
        "all_head_mse",
        "predicted_max_scour_pct",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Paper1RefitContractError("refit metrics fields drifted")
    result = _json(dict(value))
    states = result["state"]
    if (
        not isinstance(states, list)
        or not states
        or any(isinstance(state, bool) or not isinstance(state, int) for state in states)
        or states != sorted(states)
        or len(states) != len(set(states))
        or states[0] < 0
    ):
        raise Paper1RefitContractError("metric states must be unique sorted indices")
    for field in ("scour_mse", "all_head_mse", "predicted_max_scour_pct"):
        values = result[field]
        if not isinstance(values, list) or len(values) != len(states):
            raise Paper1RefitContractError(f"metric {field} has wrong coverage")
        result[field] = [
            _finite_nonnegative(item, f"metric {field}") for item in values
        ]
    return result


_CANDIDATE_FIELDS = {
    "pipeline",
    "hpo_restart_seed",
    "hpo_job_id",
    "hpo_identity_sha256",
    "hpo_completion_sha256",
    "hpo_metadata_sha256",
    "hpo_study_sha256",
    "protocol_core_hash",
    "protocol_hash",
    "params",
    "params_sha256",
    "frozen_checkpoint_epochs",
}


def _validated_candidate(value: object, *, pipeline: str, seed: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_FIELDS:
        raise Paper1RefitContractError("candidate lineage fields drifted")
    result = _json(dict(value))
    if result["pipeline"] != pipeline or result["hpo_restart_seed"] != seed:
        raise Paper1RefitContractError("candidate pipeline/restart lineage drifted")
    if not isinstance(result["params"], dict) or not result["params"]:
        raise Paper1RefitContractError("candidate parameters are empty/invalid")
    if result["params_sha256"] != _sha(result["params"]):
        raise Paper1RefitContractError("candidate parameter digest is invalid")
    for field in (
        "hpo_identity_sha256",
        "hpo_completion_sha256",
        "hpo_metadata_sha256",
        "hpo_study_sha256",
        "protocol_core_hash",
        "protocol_hash",
    ):
        if not _is_sha(result[field]):
            raise Paper1RefitContractError(f"candidate {field} is invalid")
    if not isinstance(result["hpo_job_id"], str) or not result["hpo_job_id"]:
        raise Paper1RefitContractError("candidate HPO job id is invalid")
    expected_hpo = [
        job for job in _phase_jobs("hpo")
        if job["phase"] == "f40s_factorial_hpo"
        and job["pipeline"] == pipeline
        and job["hpo_restart_seed"] == seed
    ]
    if len(expected_hpo) != 1 or result["hpo_job_id"] != expected_hpo[0]["job_id"]:
        raise Paper1RefitContractError("candidate cites the wrong HPO job")
    epochs = result["frozen_checkpoint_epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise Paper1RefitContractError("candidate frozen epoch count is invalid")
    return result


def _validated_partition(
    value: object,
    *,
    fold_index: int | None,
    metrics_states: list[int],
) -> dict[str, Any]:
    expected = {
        "outer_split_seed",
        "development_partition_seed",
        "n_splits",
        "n_repeats",
        "fold_index",
        "split_manifest_sha256",
        "development_idx_sha256",
        "outer_test_idx_sha256",
        "development_states",
        "outer_test_states",
        "train_states",
        "validation_states",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Paper1RefitContractError("refit partition fields drifted")
    result = _json(dict(value))
    for field in (
        "split_manifest_sha256",
        "development_idx_sha256",
        "outer_test_idx_sha256",
    ):
        if not _is_sha(result[field]):
            raise Paper1RefitContractError(f"partition {field} is invalid")
    if result["outer_split_seed"] != OUTER_SPLIT_SEED:
        raise Paper1RefitContractError("outer split seed drifted")
    if fold_index is None:
        if any(
            result[field] is not None
            for field in (
                "development_partition_seed", "n_splits", "n_repeats", "fold_index"
            )
        ):
            raise Paper1RefitContractError("channel screen invented an OOF fold")
    elif (
        result["development_partition_seed"] != DEVELOPMENT_PARTITION_SEED
        or result["n_splits"] != DEVELOPMENT_N_SPLITS
        or result["n_repeats"] != DEVELOPMENT_N_REPEATS
        or result["fold_index"] != fold_index
    ):
        raise Paper1RefitContractError("development partition policy drifted")
    for field in (
        "development_states",
        "outer_test_states",
        "train_states",
        "validation_states",
    ):
        states = result[field]
        if (
            not isinstance(states, list)
            or states != sorted(states)
            or len(states) != len(set(states))
            or any(isinstance(state, bool) or not isinstance(state, int) for state in states)
        ):
            raise Paper1RefitContractError(f"partition {field} is invalid")
    development = set(result["development_states"])
    outer = set(result["outer_test_states"])
    train = set(result["train_states"])
    validation = set(result["validation_states"])
    if (
        not development
        or not outer
        or development & outer
        or train & validation
        or train | validation != development
        or result["validation_states"] != metrics_states
    ):
        raise Paper1RefitContractError(
            "partition leaks/omits states or differs from result metrics"
        )
    return result


def seal_development_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json(dict(value))
    result.pop("result_sha256", None)
    result["result_sha256"] = _sha(result)
    return validate_development_result(result)


def validate_development_result(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "campaign_run_tag",
        "complete_grid_sha256",
        "job",
        "candidate",
        "partition",
        "initialization_seed",
        "outer_test_observations_accessed",
        "metrics",
        "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Paper1RefitContractError("development result fields drifted")
    result = _json(dict(value))
    supplied = result.pop("result_sha256")
    if supplied != _sha(result):
        raise Paper1RefitContractError("development result SHA-256 is invalid")
    result["result_sha256"] = supplied
    job = _exact_job(result["job"], "development_adjudication")
    if result["schema"] != DEVELOPMENT_RESULT_SCHEMA:
        raise Paper1RefitContractError("development result schema drifted")
    if (
        result["complete_grid_sha256"]
        != _complete_grid_sha256()
        or not isinstance(result["campaign_run_tag"], str)
        or not result["campaign_run_tag"]
        or result["outer_test_observations_accessed"] is not False
        or result["initialization_seed"] != job["initialization_seed"]
    ):
        raise Paper1RefitContractError("development result identity drifted")
    metrics = _validated_metrics(result["metrics"])
    result["candidate"] = _validated_candidate(
        result["candidate"],
        pipeline=job["pipeline"],
        seed=job["candidate_restart_seed"],
    )
    result["partition"] = _validated_partition(
        result["partition"],
        fold_index=job["fold_index"],
        metrics_states=metrics["state"],
    )
    if job["development_partition_seed"] != DEVELOPMENT_PARTITION_SEED:
        raise Paper1RefitContractError("job partition seed drifted")
    result["metrics"] = metrics
    return result


def _canonical_slots(resolution: Mapping[str, str]) -> dict[str, str]:
    first: dict[str, str] = {}
    return {
        slot: first.setdefault(resolution[slot], slot)
        for slot in RETAINED_PIPELINE_SLOTS
    }


def _derive_development(source_results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = [validate_development_result(result) for result in source_results]
    expected_jobs = _phase_jobs("development_adjudication")
    expected_ids = [job["job_id"] for job in expected_jobs]
    observed_ids = [result["job"]["job_id"] for result in results]
    if sorted(observed_ids) != sorted(expected_ids) or len(observed_ids) != len(set(observed_ids)):
        raise Paper1RefitContractError(
            "development artefact does not contain every exact fit job once"
        )
    results.sort(key=lambda result: expected_ids.index(result["job"]["job_id"]))
    run_tags = {result["campaign_run_tag"] for result in results}
    if len(run_tags) != 1:
        raise Paper1RefitContractError("development results mix campaign runs")
    reference_partition = results[0]["partition"]
    for result in results[1:]:
        partition = result["partition"]
        if any(
            partition[field] != reference_partition[field]
            for field in (
                "outer_split_seed",
                "development_partition_seed",
                "n_splits",
                "n_repeats",
                "split_manifest_sha256",
                "development_idx_sha256",
                "outer_test_idx_sha256",
                "development_states",
                "outer_test_states",
            )
        ):
            raise Paper1RefitContractError(
                "development jobs do not share one authenticated partition"
            )

    candidate_summaries: list[dict[str, Any]] = []
    for cell in FACTORIAL_CELLS:
        for restart_seed in HPO_RESTART_SEEDS:
            selected = [
                result for result in results
                if result["job"]["pipeline"] == cell.cell_id
                and result["job"]["candidate_restart_seed"] == restart_seed
            ]
            if len(selected) != DEVELOPMENT_N_SPLITS * len(DEVELOPMENT_INIT_SEEDS):
                raise Paper1RefitContractError("candidate refit coverage is incomplete")
            candidate = selected[0]["candidate"]
            if any(result["candidate"] != candidate for result in selected):
                raise Paper1RefitContractError("candidate parameter lineage differs across folds")
            from core.hyperparameter_policy import validate_registered_params

            try:
                validated_params = validate_registered_params(
                    cell.cell_id, candidate["params"]
                )
            except Exception as exc:  # exact live search-space validator
                raise Paper1RefitContractError(
                    f"candidate parameters are outside the registered space: {exc}"
                ) from exc
            if validated_params != candidate["params"]:
                raise Paper1RefitContractError("candidate parameters did not reproduce")
            development_states = selected[0]["partition"]["development_states"]
            outer_states = selected[0]["partition"]["outer_test_states"]
            score_rows: list[dict[str, Any]] = []
            source_jobs: list[dict[str, str]] = []
            for init_seed in DEVELOPMENT_INIT_SEEDS:
                per_init = [
                    result for result in selected
                    if result["initialization_seed"] == init_seed
                ]
                if sorted(result["job"]["fold_index"] for result in per_init) != list(
                    range(DEVELOPMENT_N_SPLITS)
                ):
                    raise Paper1RefitContractError("candidate has missing/duplicate OOF fold")
                seen: list[int] = []
                for result in sorted(per_init, key=lambda item: item["job"]["fold_index"]):
                    partition = result["partition"]
                    if (
                        partition["development_states"] != development_states
                        or partition["outer_test_states"] != outer_states
                    ):
                        raise Paper1RefitContractError("candidate folds use different partitions")
                    seen.extend(partition["validation_states"])
                    for state, score in zip(
                        result["metrics"]["state"], result["metrics"]["scour_mse"]
                    ):
                        score_rows.append({
                            "state": state,
                            "initialization_seed": init_seed,
                            "scour_mse": score,
                        })
                    source_jobs.append({
                        "job_id": result["job"]["job_id"],
                        "result_sha256": result["result_sha256"],
                    })
                if sorted(seen) != development_states or len(seen) != len(set(seen)):
                    raise Paper1RefitContractError(
                        "candidate folds are not one complete non-overlapping OOF partition"
                    )
            score_rows.sort(key=lambda row: (row["state"], row["initialization_seed"]))
            state_scores = []
            for state in development_states:
                scores = [
                    row["scour_mse"] for row in score_rows if row["state"] == state
                ]
                if len(scores) != len(DEVELOPMENT_INIT_SEEDS):
                    raise Paper1RefitContractError("candidate state/init tensor is incomplete")
                state_scores.append({"state": state, "scour_mse": _mean(scores)})
            candidate_summaries.append({
                "pipeline": cell.cell_id,
                "hpo_restart_seed": restart_seed,
                "candidate": candidate,
                "development_states": development_states,
                "outer_test_states": outer_states,
                "state_initialization_scores": score_rows,
                "state_scores": state_scores,
                "aggregate_score": _mean(row["scour_mse"] for row in state_scores),
                "source_jobs": sorted(source_jobs, key=lambda item: item["job_id"]),
            })

    winners: dict[str, dict[str, Any]] = {}
    for cell in FACTORIAL_CELLS:
        candidates = [
            summary for summary in candidate_summaries
            if summary["pipeline"] == cell.cell_id
        ]
        winner = min(
            candidates,
            key=lambda item: (item["aggregate_score"], item["hpo_restart_seed"]),
        )
        winners[cell.cell_id] = {
            "pipeline": cell.cell_id,
            "hpo_restart_seed": winner["hpo_restart_seed"],
            "aggregate_score": winner["aggregate_score"],
            "candidate": winner["candidate"],
        }
    raw = [winner for pipeline, winner in winners.items() if pipeline.startswith("RAW_")]
    paa = [winner for pipeline, winner in winners.items() if pipeline.startswith("PAA_")]
    best_raw = min(raw, key=lambda item: (item["aggregate_score"], item["pipeline"]))
    best_paa = min(paa, key=lambda item: (item["aggregate_score"], item["pipeline"]))
    resolution = {
        "f40s_best_raw": best_raw["pipeline"],
        "f40s_best_paa": best_paa["pipeline"],
        "raw_cnn_gap_baseline": RAW_CNN_GAP_BASELINE_ID,
        "paa_cnn_gap_baseline": PAA_CNN_GAP_BASELINE_ID,
    }
    return {
        "campaign_run_tag": next(iter(run_tags)),
        "partition_policy": {
            "seed": DEVELOPMENT_PARTITION_SEED,
            "n_splits": DEVELOPMENT_N_SPLITS,
            "n_repeats": DEVELOPMENT_N_REPEATS,
            "fold_indices": list(range(DEVELOPMENT_N_SPLITS)),
            "outer_test_observations_accessed": False,
        },
        "aggregation_policy": deepcopy(DEVELOPMENT_AGGREGATION_POLICY),
        "candidate_summaries": candidate_summaries,
        "winner_by_pipeline": winners,
        "best_raw": best_raw["pipeline"],
        "best_paa": best_paa["pipeline"],
        "slot_resolution": resolution,
        "canonical_slot": _canonical_slots(resolution),
    }


def build_development_artifact(
    source_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sources = [validate_development_result(value) for value in source_results]
    derived = _derive_development(sources)
    value = {
        "schema": DEVELOPMENT_ARTIFACT_SCHEMA,
        "complete_grid_sha256": _complete_grid_sha256(),
        "selection_metric": SELECTION_METRIC,
        "source_results": sources,
        **derived,
    }
    value["artifact_sha256"] = _sha(value)
    return validate_development_artifact(value)


def validate_development_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Paper1RefitContractError("development artefact must be a mapping")
    result = _json(dict(value))
    supplied = result.pop("artifact_sha256", None)
    if not _is_sha(supplied) or supplied != _sha(result):
        raise Paper1RefitContractError("development artefact SHA-256 is invalid")
    if (
        result.get("schema") != DEVELOPMENT_ARTIFACT_SCHEMA
        or result.get("complete_grid_sha256")
        != _complete_grid_sha256()
        or result.get("selection_metric") != SELECTION_METRIC
        or not isinstance(result.get("source_results"), list)
    ):
        raise Paper1RefitContractError("development artefact header drifted")
    derived = _derive_development(result["source_results"])
    expected = {
        "schema": DEVELOPMENT_ARTIFACT_SCHEMA,
        "complete_grid_sha256": _complete_grid_sha256(),
        "selection_metric": SELECTION_METRIC,
        "source_results": [
            validate_development_result(item) for item in result["source_results"]
        ],
        **derived,
    }
    if result != expected:
        raise Paper1RefitContractError("development artefact summaries do not recompute")
    expected["artifact_sha256"] = supplied
    return expected


def seal_channel_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json(dict(value))
    result.pop("result_sha256", None)
    result["result_sha256"] = _sha(result)
    return validate_channel_result(result)


def validate_channel_result(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "execution_kind",
        "campaign_run_tag",
        "complete_grid_sha256",
        "adjudication_artifact_sha256",
        "job",
        "slot_resolution",
        "canonical_slot",
        "pipeline",
        "frozen_candidate",
        "partition",
        "outer_test_observations_accessed",
        "canonical_job_id",
        "metrics",
        "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Paper1RefitContractError("channel result fields drifted")
    result = _json(dict(value))
    supplied = result.pop("result_sha256")
    if supplied != _sha(result):
        raise Paper1RefitContractError("channel result SHA-256 is invalid")
    result["result_sha256"] = supplied
    job = _exact_job(result["job"], "channel_screen")
    if (
        result["schema"] != CHANNEL_RESULT_SCHEMA
        or result["execution_kind"] not in {"refit", "deduplicated_alias"}
        or result["complete_grid_sha256"]
        != _complete_grid_sha256()
        or not isinstance(result["campaign_run_tag"], str)
        or not result["campaign_run_tag"]
        or not _is_sha(result["adjudication_artifact_sha256"])
        or result["outer_test_observations_accessed"] is not False
    ):
        raise Paper1RefitContractError("channel result identity drifted")
    resolution = result["slot_resolution"]
    if not isinstance(resolution, dict) or set(resolution) != set(RETAINED_PIPELINE_SLOTS):
        raise Paper1RefitContractError("channel slot resolution is invalid")
    canonical = _canonical_slots(resolution)
    if result["canonical_slot"] != canonical[job["pipeline"]]:
        raise Paper1RefitContractError("channel canonical slot is invalid")
    if result["pipeline"] != resolution[job["pipeline"]]:
        raise Paper1RefitContractError("channel resolved pipeline is invalid")
    frozen = _validated_candidate(
        result["frozen_candidate"],
        pipeline=result["pipeline"],
        seed=result["frozen_candidate"].get("hpo_restart_seed"),
    )
    result["frozen_candidate"] = frozen
    metrics = _validated_metrics(result["metrics"])
    result["metrics"] = metrics
    result["partition"] = _validated_partition(
        result["partition"], fold_index=None, metrics_states=metrics["state"]
    )
    is_alias = job["pipeline"] != result["canonical_slot"]
    if is_alias != (result["execution_kind"] == "deduplicated_alias"):
        raise Paper1RefitContractError("channel alias execution kind is wrong")
    if result["execution_kind"] == "refit":
        if result["canonical_job_id"] != job["job_id"]:
            raise Paper1RefitContractError("canonical channel job id is wrong")
    elif not isinstance(result["canonical_job_id"], str) or not result["canonical_job_id"]:
        raise Paper1RefitContractError("channel alias lacks its canonical job")
    return result


def _derive_channel(
    source_results: Iterable[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    adjudication = validate_development_artifact(adjudication)
    results = [validate_channel_result(value) for value in source_results]
    expected_jobs = _phase_jobs("channel_screen")
    expected_ids = [job["job_id"] for job in expected_jobs]
    observed_ids = [result["job"]["job_id"] for result in results]
    if sorted(observed_ids) != sorted(expected_ids) or len(observed_ids) != len(set(observed_ids)):
        raise Paper1RefitContractError("channel artefact does not contain every exact job once")
    if any(
        result["campaign_run_tag"] != adjudication["campaign_run_tag"]
        or result["adjudication_artifact_sha256"] != adjudication["artifact_sha256"]
        or result["slot_resolution"] != adjudication["slot_resolution"]
        for result in results
    ):
        raise Paper1RefitContractError("channel results cite another adjudication/run")
    adjudication_partition = adjudication["source_results"][0]["partition"]
    reference_screen = results[0]["partition"]
    if (
        reference_screen["development_states"]
        != adjudication_partition["development_states"]
        or reference_screen["outer_test_states"]
        != adjudication_partition["outer_test_states"]
        or reference_screen["split_manifest_sha256"]
        != adjudication_partition["split_manifest_sha256"]
        or reference_screen["development_idx_sha256"]
        != adjudication_partition["development_idx_sha256"]
        or reference_screen["outer_test_idx_sha256"]
        != adjudication_partition["outer_test_idx_sha256"]
    ):
        raise Paper1RefitContractError(
            "channel screen partition differs from adjudication"
        )
    for result in results[1:]:
        partition = result["partition"]
        if any(
            partition[field] != reference_screen[field]
            for field in (
                "outer_split_seed",
                "split_manifest_sha256",
                "development_idx_sha256",
                "outer_test_idx_sha256",
                "development_states",
                "outer_test_states",
                "train_states",
                "validation_states",
            )
        ):
            raise Paper1RefitContractError(
                "channel jobs do not share one paired development split"
            )
    expected_frozen = adjudication["winner_by_pipeline"]
    for result in results:
        winner = expected_frozen[result["pipeline"]]["candidate"]
        if result["frozen_candidate"] != winner:
            raise Paper1RefitContractError(
                "channel result does not use the adjudicated frozen candidate"
            )
    by_id = {result["job"]["job_id"]: result for result in results}
    canonical_slots = adjudication["canonical_slot"]
    unique_slots = [
        slot for slot in RETAINED_PIPELINE_SLOTS if canonical_slots[slot] == slot
    ]
    inputs = [list(item) for item in channel_screen_inputs()]

    # Aliases must reproduce the exact canonical slot result for the same input/seed.
    for result in results:
        job = result["job"]
        canonical_slot = canonical_slots[job["pipeline"]]
        canonical_job = next(
            candidate for candidate in expected_jobs
            if candidate["pipeline"] == canonical_slot
            and candidate["input_selector"] == job["input_selector"]
            and candidate["initialization_seed"] == job["initialization_seed"]
        )
        canonical_result = by_id[canonical_job["job_id"]]
        if canonical_slot == job["pipeline"]:
            if result["execution_kind"] != "refit":
                raise Paper1RefitContractError("canonical screen job is marked alias")
        elif (
            result["execution_kind"] != "deduplicated_alias"
            or result["canonical_job_id"] != canonical_job["job_id"]
            or result["metrics"] != canonical_result["metrics"]
            or result["partition"] != canonical_result["partition"]
            or result["frozen_candidate"] != canonical_result["frozen_candidate"]
        ):
            raise Paper1RefitContractError("channel alias differs from canonical result")

    pipeline_input_scores: list[dict[str, Any]] = []
    for slot in unique_slots:
        pipeline = adjudication["slot_resolution"][slot]
        for selector in inputs:
            selected = [
                result for result in results
                if result["job"]["pipeline"] == slot
                and result["job"]["input_selector"] == selector
            ]
            if sorted(result["job"]["initialization_seed"] for result in selected) != list(
                SCREEN_REFIT_SEEDS
            ):
                raise Paper1RefitContractError("channel seed coverage is incomplete")
            states = selected[0]["metrics"]["state"]
            if any(result["metrics"]["state"] != states for result in selected):
                raise Paper1RefitContractError("channel paired seed state coverage drifted")
            rows = []
            source_jobs = []
            for result in sorted(selected, key=lambda item: item["job"]["initialization_seed"]):
                seed = result["job"]["initialization_seed"]
                rows.extend(
                    {"state": state, "initialization_seed": seed, "scour_mse": score}
                    for state, score in zip(
                        result["metrics"]["state"], result["metrics"]["scour_mse"]
                    )
                )
                source_jobs.append({
                    "job_id": result["job"]["job_id"],
                    "result_sha256": result["result_sha256"],
                })
            rows.sort(key=lambda row: (row["state"], row["initialization_seed"]))
            state_scores = []
            for state in states:
                scores = [row["scour_mse"] for row in rows if row["state"] == state]
                if len(scores) != len(SCREEN_REFIT_SEEDS):
                    raise Paper1RefitContractError("channel state/seed tensor is incomplete")
                state_scores.append({"state": state, "scour_mse": _mean(scores)})
            pipeline_input_scores.append({
                "canonical_slot": slot,
                "pipeline": pipeline,
                "input_selector": selector,
                "state_initialization_scores": rows,
                "state_scores": state_scores,
                "pipeline_score": _mean(row["scour_mse"] for row in state_scores),
                "source_jobs": source_jobs,
            })

    aggregate_inputs = []
    for selector in inputs:
        entries = [
            entry for entry in pipeline_input_scores
            if entry["input_selector"] == selector
        ]
        if len(entries) != len(unique_slots):
            raise Paper1RefitContractError("channel unique-pipeline coverage drifted")
        aggregate_inputs.append({
            "input_selector": selector,
            "pipeline_scores": [
                {"pipeline": entry["pipeline"], "score": entry["pipeline_score"]}
                for entry in entries
            ],
            "aggregate_score": _mean(entry["pipeline_score"] for entry in entries),
            "selection_eligible": len(selector) == 2,
        })
    eligible = [entry for entry in aggregate_inputs if entry["selection_eligible"]]
    selected = min(
        eligible,
        key=lambda entry: (entry["aggregate_score"], entry["input_selector"]),
    )
    return {
        "campaign_run_tag": adjudication["campaign_run_tag"],
        "adjudication_artifact_sha256": adjudication["artifact_sha256"],
        "slot_resolution": adjudication["slot_resolution"],
        "canonical_slot": canonical_slots,
        "unique_resolved_pipelines": [
            adjudication["slot_resolution"][slot] for slot in unique_slots
        ],
        "aggregation_policy": deepcopy(CHANNEL_AGGREGATION_POLICY),
        "reported_inputs": inputs,
        "eligible_pairs": [selector for selector in inputs if len(selector) == 2],
        "pipeline_input_scores": pipeline_input_scores,
        "input_scores": aggregate_inputs,
        "selected_pair": selected["input_selector"],
    }


def build_channel_selection_artifact(
    source_results: Iterable[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    sources = [validate_channel_result(value) for value in source_results]
    adjudication = validate_development_artifact(adjudication)
    derived = _derive_channel(sources, adjudication)
    value = {
        "schema": CHANNEL_SELECTION_SCHEMA,
        "complete_grid_sha256": _complete_grid_sha256(),
        "selection_metric": SELECTION_METRIC,
        "source_results": sources,
        **derived,
    }
    value["artifact_sha256"] = _sha(value)
    return validate_channel_selection_artifact(value, adjudication=adjudication)


def validate_channel_selection_artifact(
    value: Mapping[str, Any],
    *,
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    adjudication = validate_development_artifact(adjudication)
    if not isinstance(value, Mapping):
        raise Paper1RefitContractError("channel selection artefact must be a mapping")
    result = _json(dict(value))
    supplied = result.pop("artifact_sha256", None)
    if not _is_sha(supplied) or supplied != _sha(result):
        raise Paper1RefitContractError("channel selection artefact SHA-256 is invalid")
    if (
        result.get("schema") != CHANNEL_SELECTION_SCHEMA
        or result.get("complete_grid_sha256")
        != _complete_grid_sha256()
        or result.get("selection_metric") != SELECTION_METRIC
        or not isinstance(result.get("source_results"), list)
    ):
        raise Paper1RefitContractError("channel selection header drifted")
    derived = _derive_channel(result["source_results"], adjudication)
    expected = {
        "schema": CHANNEL_SELECTION_SCHEMA,
        "complete_grid_sha256": _complete_grid_sha256(),
        "selection_metric": SELECTION_METRIC,
        "source_results": [validate_channel_result(item) for item in result["source_results"]],
        **derived,
    }
    if result != expected:
        raise Paper1RefitContractError("channel selection summaries do not recompute")
    expected["artifact_sha256"] = supplied
    return expected
