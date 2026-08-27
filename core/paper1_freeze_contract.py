"""Authenticated block-local model freeze and sealed-report contracts.

Each Paper-1 block freezes one hyperparameter vector per unique resolved
pipeline only after all five selected-pair HPO studies authenticate.  The
sealed outer test is downstream of this artefact and can never participate in
    the choice.  Post-freeze and secondary-transfer results cite the locally
    authenticated freeze digest and are explicitly report-only.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.campaign_contract import campaign_stage_contract
from core.hyperparameter_policy import canonical_json_sha256, validate_registered_params
from core.paper1_refit_contract import _validated_metrics
from core.paper1_selection import validate_selection_artifact
from core.paper1_training_contract import (
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    HPO_TRIALS_PER_STUDY,
    OUTER_SPLIT_SEED,
    RETAINED_PIPELINE_SLOTS,
    STAGE_ORDER,
    TRAINING_EPOCHS,
    canonical_json_bytes,
    complete_job_grid,
)


BLOCK_FREEZE_SCHEMA = "paper1-block-freeze-artifact-v1"
SELECTED_CHAMPION_SCHEMA = "paper1-selected-pair-hpo-champion-v1"
SEALED_RESULT_SCHEMA = "paper1-sealed-report-result-v1"
BLOCK_FREEZE_ARTIFACT_ENV = "TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT"
BLOCK_FREEZE_ARTIFACT_SHA256_ENV = (
    "TTBI_PAPER1_BLOCK_FREEZE_ARTIFACT_SHA256"
)
FREEZE_OBJECTIVE = "canonical_inner_validation_objective"
FREEZE_TIE_BREAK = (
    "minimum finite objective, then ascending HPO restart seed, then "
    "ascending best-trial number"
)
_HEX = frozenset("0123456789abcdef")
_CELLS = {cell.cell_id: cell for cell in FACTORIAL_CELLS}


class Paper1FreezeContractError(RuntimeError):
    """A freeze or sealed-report record violates the registered contract."""


def _json(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Paper1FreezeContractError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Paper1FreezeContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _stage_phase(stage: str) -> str:
    if stage not in STAGE_ORDER:
        raise Paper1FreezeContractError(f"unregistered freeze stage {stage!r}")
    return (
        "f40s_selected_pair_hpo"
        if stage == "F40-S" else "block_selected_pair_hpo"
    )


def _selected_hpo_job(stage: str, slot: str, seed: int) -> dict[str, Any]:
    matches = [
        job for job in complete_job_grid()["phases"]["hpo"]
        if job["stage"] == stage
        and job["phase"] == _stage_phase(stage)
        and job["pipeline"] == slot
        and job["hpo_restart_seed"] == seed
    ]
    if len(matches) != 1:
        raise Paper1FreezeContractError(
            "selected-pair champion has no unique registered HPO job"
        )
    return matches[0]


_SOURCE_FIELDS = {
    "environment_lock_sha256",
    "python_runtime_source_root_sha256",
    "python_runtime_source_file_count",
    "generator_source_root_sha256",
    "generator_source_file_count",
    "dataset_content_root_sha256",
    "generation_fingerprint",
    "qualification_source_sha256",
}


def _validated_source_lineage(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise Paper1FreezeContractError("champion source-lineage fields drifted")
    result = _json(dict(value))
    for field in _SOURCE_FIELDS - {
        "python_runtime_source_file_count",
        "generator_source_file_count",
    }:
        if not _is_sha(result[field]):
            raise Paper1FreezeContractError(
                f"champion source lineage {field} is invalid"
            )
    _strict_int(
        result["python_runtime_source_file_count"],
        "python runtime source file count",
        minimum=1,
    )
    _strict_int(
        result["generator_source_file_count"],
        "generator source file count",
        minimum=1,
    )
    return result


_TERMINAL_FIELDS = {
    "COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING", "total"
}
_CHAMPION_FIELDS = {
    "schema",
    "stage",
    "canonical_slot",
    "pipeline",
    "selected_pair",
    "hpo_restart_seed",
    "hpo_job_id",
    "hpo_identity_sha256",
    "hpo_completion_sha256",
    "hpo_metadata_sha256",
    "hpo_study_sha256",
    "execution_environment_sha256",
    "execution_compatibility_sha256",
    "execution_receipt_sha256",
    "protocol_core_hash",
    "protocol_hash",
    "source_lineage",
    "best_trial_number",
    "best_trial_value",
    "terminal_counts",
    "params",
    "params_sha256",
    "frozen_checkpoint_epochs",
}


def validate_selected_champion(
    value: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    selection = validate_selection_artifact(selection)
    if not isinstance(value, Mapping) or set(value) != _CHAMPION_FIELDS:
        raise Paper1FreezeContractError("selected-HPO champion fields drifted")
    result = _json(dict(value))
    stage = result["stage"]
    slot = result["canonical_slot"]
    seed = result["hpo_restart_seed"]
    if (
        result["schema"] != SELECTED_CHAMPION_SCHEMA
        or stage not in STAGE_ORDER
        or slot not in RETAINED_PIPELINE_SLOTS
        or selection["canonical_slot"].get(slot) != slot
        or result["pipeline"] != selection["slot_resolution"].get(slot)
        or result["selected_pair"] != selection["selected_pair"]
        or seed not in HPO_RESTART_SEEDS
    ):
        raise Paper1FreezeContractError("selected-HPO champion identity drifted")
    expected_job = _selected_hpo_job(stage, slot, seed)
    if result["hpo_job_id"] != expected_job["job_id"]:
        raise Paper1FreezeContractError("selected-HPO champion cites another job")
    for field in (
        "hpo_identity_sha256",
        "hpo_completion_sha256",
        "hpo_metadata_sha256",
        "hpo_study_sha256",
        "execution_environment_sha256",
        "execution_compatibility_sha256",
        "execution_receipt_sha256",
        "protocol_core_hash",
        "protocol_hash",
    ):
        if not _is_sha(result[field]):
            raise Paper1FreezeContractError(f"champion {field} is invalid")
    result["source_lineage"] = _validated_source_lineage(
        result["source_lineage"]
    )
    trial_number = _strict_int(
        result["best_trial_number"], "best trial number"
    )
    if trial_number >= HPO_TRIALS_PER_STUDY:
        raise Paper1FreezeContractError("best trial number exceeds HPO budget")
    objective = result["best_trial_value"]
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
        or float(objective) < 0.0
    ):
        raise Paper1FreezeContractError(
            "champion inner-validation objective must be finite/non-negative"
        )
    result["best_trial_value"] = float(objective)
    counts = result["terminal_counts"]
    if not isinstance(counts, Mapping) or set(counts) != _TERMINAL_FIELDS:
        raise Paper1FreezeContractError("HPO terminal-state inventory drifted")
    counts = {field: _strict_int(counts[field], field) for field in counts}
    if (
        counts["total"] != HPO_TRIALS_PER_STUDY
        or counts["COMPLETE"] < 1
        or counts["COMPLETE"] + counts["PRUNED"]
        != HPO_TRIALS_PER_STUDY
        or any(counts[field] for field in ("FAIL", "RUNNING", "WAITING"))
    ):
        raise Paper1FreezeContractError(
            "selected-HPO restart is failed, incomplete, or over budget"
        )
    result["terminal_counts"] = counts
    try:
        params = validate_registered_params(result["pipeline"], result["params"])
    except Exception as exc:
        raise Paper1FreezeContractError(
            f"champion parameters are outside the registered space: {exc}"
        ) from exc
    params = _json(params)
    if result["params_sha256"] != canonical_json_sha256(params):
        raise Paper1FreezeContractError("champion parameter digest is invalid")
    result["params"] = params
    epochs = _strict_int(
        result["frozen_checkpoint_epochs"], "frozen checkpoint epochs", minimum=1
    )
    if epochs > TRAINING_EPOCHS:
        raise Paper1FreezeContractError("frozen checkpoint exceeds max epochs")
    return result


def _architecture(pipeline: str) -> dict[str, Any]:
    cell = _CELLS[pipeline]
    return {
        "pipeline": cell.cell_id,
        "representation": cell.representation,
        "position_encoding": cell.position_encoding,
        "lstm": cell.lstm,
        "multi_rate_pooling": cell.multi_rate_pooling,
    }


def _derive_freeze(
    *,
    stage: str,
    selection: Mapping[str, Any],
    champions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    selection = validate_selection_artifact(selection)
    if stage not in STAGE_ORDER:
        raise Paper1FreezeContractError(f"unregistered freeze stage {stage!r}")
    if selection["applicable_stages"] != list(STAGE_ORDER):
        raise Paper1FreezeContractError("selection does not cover all stages")
    records = [
        validate_selected_champion(value, selection=selection)
        for value in champions
    ]
    if records:
        for field in (
            "selected_pair",
            "execution_receipt_sha256",
            "protocol_core_hash",
            "protocol_hash",
            "source_lineage",
        ):
            if any(record[field] != records[0][field] for record in records[1:]):
                raise Paper1FreezeContractError(
                    f"block-local pipelines disagree on {field}"
                )
    canonical_slots = [
        slot for slot in RETAINED_PIPELINE_SLOTS
        if selection["canonical_slot"][slot] == slot
    ]
    freezes: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for slot in canonical_slots:
        pipeline = selection["slot_resolution"][slot]
        inventory = [
            record for record in records
            if record["stage"] == stage and record["canonical_slot"] == slot
        ]
        inventory.sort(key=lambda item: HPO_RESTART_SEEDS.index(
            item["hpo_restart_seed"]
        ))
        if [item["hpo_restart_seed"] for item in inventory] != list(
            HPO_RESTART_SEEDS
        ):
            raise Paper1FreezeContractError(
                f"{stage}/{slot} lacks the exact five-restart inventory"
            )
        if any(item["pipeline"] != pipeline for item in inventory):
            raise Paper1FreezeContractError("restart pipeline lineage drifted")
        if len({item["hpo_job_id"] for item in inventory}) != len(inventory):
            raise Paper1FreezeContractError("duplicate selected-HPO restart")
        observed_ids.update(item["hpo_job_id"] for item in inventory)
        for field in (
            "selected_pair",
            "execution_receipt_sha256",
            "protocol_core_hash",
            "protocol_hash",
            "source_lineage",
        ):
            if any(item[field] != inventory[0][field] for item in inventory[1:]):
                raise Paper1FreezeContractError(
                    f"five-restart inventory disagrees on {field}"
                )
        winner = min(
            inventory,
            key=lambda item: (
                item["best_trial_value"],
                item["hpo_restart_seed"],
                item["best_trial_number"],
            ),
        )
        freezes.append({
            "canonical_slot": slot,
            "pipeline": pipeline,
            "architecture": _architecture(pipeline),
            "restart_inventory": inventory,
            "winner": deepcopy(winner),
        })
    if len(records) != len(observed_ids) or len(records) != (
        len(canonical_slots) * len(HPO_RESTART_SEEDS)
    ):
        raise Paper1FreezeContractError(
            "freeze input contains foreign, duplicate, or alias HPO records"
        )
    inventory_for_hash = [
        {
            "canonical_slot": item["canonical_slot"],
            "pipeline": item["pipeline"],
            "restart_inventory": item["restart_inventory"],
        }
        for item in freezes
    ]
    return {
        "stage": stage,
        "execution_block": {
            "F40-S": "f40s", "F40-M": "f40m",
            "L99-S": "l99s", "L99-M": "l99m",
        }[stage],
        "campaign_run_tag": selection["campaign_run_tag"],
        "selection_artifact": selection,
        "selected_pair": selection["selected_pair"],
        "slot_resolution": selection["slot_resolution"],
        "canonical_slot": selection["canonical_slot"],
        "unique_canonical_slots": canonical_slots,
        "unique_resolved_pipelines": [
            selection["slot_resolution"][slot] for slot in canonical_slots
        ],
        "freeze_policy": {
            "objective": FREEZE_OBJECTIVE,
            "required_restart_seeds": list(HPO_RESTART_SEEDS),
            "required_terminal_budget": HPO_TRIALS_PER_STUDY,
            "tie_break": FREEZE_TIE_BREAK,
            "sealed_outer_test_observations_accessed": False,
        },
        "pipeline_freezes": freezes,
        "source_inventory_sha256": _sha(inventory_for_hash),
    }


def build_block_freeze_artifact(
    *,
    stage: str,
    selection: Mapping[str, Any],
    champions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    derived = _derive_freeze(
        stage=stage, selection=selection, champions=champions
    )
    value = {
        "schema": BLOCK_FREEZE_SCHEMA,
        "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
        **derived,
    }
    value["artifact_sha256"] = _sha(value)
    return validate_block_freeze_artifact(value)


def validate_block_freeze_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Paper1FreezeContractError("block freeze artefact must be a mapping")
    result = _json(dict(value))
    supplied = result.pop("artifact_sha256", None)
    if not _is_sha(supplied) or supplied != _sha(result):
        raise Paper1FreezeContractError("block freeze artefact SHA-256 is invalid")
    if (
        result.get("schema") != BLOCK_FREEZE_SCHEMA
        or result.get("complete_grid_sha256")
        != complete_job_grid()["complete_grid_sha256"]
        or not isinstance(result.get("selection_artifact"), Mapping)
        or not isinstance(result.get("pipeline_freezes"), list)
    ):
        raise Paper1FreezeContractError("block freeze artefact header drifted")
    selection = validate_selection_artifact(result["selection_artifact"])
    champions = [
        record
        for freeze in result["pipeline_freezes"]
        if isinstance(freeze, Mapping)
        for record in freeze.get("restart_inventory", [])
    ]
    derived = _derive_freeze(
        stage=result.get("stage"),
        selection=selection,
        champions=champions,
    )
    expected = {
        "schema": BLOCK_FREEZE_SCHEMA,
        "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
        **derived,
    }
    if result != expected:
        raise Paper1FreezeContractError(
            "block freeze selection/inventory does not recompute exactly"
        )
    expected["artifact_sha256"] = supplied
    return expected


def load_block_freeze_artifact(
    path: str | os.PathLike[str] | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    from_environment = path is None
    raw_path = (
        os.fspath(path)
        if path is not None
        else os.environ.get(BLOCK_FREEZE_ARTIFACT_ENV, "")
    )
    expected = (
        expected_sha256
        if expected_sha256 is not None
        else os.environ.get(BLOCK_FREEZE_ARTIFACT_SHA256_ENV, "")
        if from_environment else None
    )
    if not raw_path:
        raise Paper1FreezeContractError(
            f"{BLOCK_FREEZE_ARTIFACT_ENV} is required"
        )
    if from_environment and not expected:
        raise Paper1FreezeContractError(
            f"{BLOCK_FREEZE_ARTIFACT_SHA256_ENV} is required"
        )
    source = Path(raw_path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise Paper1FreezeContractError(
            "block freeze artefact must be an absolute regular file"
        )
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1FreezeContractError(
            f"block freeze artefact is unreadable/non-JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Paper1FreezeContractError(
            "block freeze artefact bytes are not canonical JSON"
        )
    result = validate_block_freeze_artifact(value)
    if expected is not None and (
        not _is_sha(expected) or result["artifact_sha256"] != expected
    ):
        raise Paper1FreezeContractError(
            "block freeze artefact differs from external SHA-256"
        )
    return result


def freeze_for_slot(
    artifact: Mapping[str, Any], *, stage: str, slot: str
) -> dict[str, Any]:
    value = validate_block_freeze_artifact(artifact)
    if value["stage"] != stage:
        raise Paper1FreezeContractError("freeze artefact belongs to another stage")
    if slot not in RETAINED_PIPELINE_SLOTS:
        raise Paper1FreezeContractError("unregistered retained-pipeline slot")
    canonical = value["canonical_slot"][slot]
    matches = [
        item for item in value["pipeline_freezes"]
        if item["canonical_slot"] == canonical
    ]
    if len(matches) != 1:
        raise Paper1FreezeContractError("slot has no unique frozen pipeline")
    return {
        "slot": slot,
        "canonical_slot": canonical,
        "pipeline": value["slot_resolution"][slot],
        "selected_pair": value["selected_pair"],
        "winner": matches[0]["winner"],
        "freeze_artifact_sha256": value["artifact_sha256"],
    }


def _exact_report_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = _json(dict(job))
    phase_key = (
        "post_freeze_stability"
        if value.get("phase") == "post_freeze_sealed_test_stability"
        else "secondary_frozen_transfer"
        if value.get("phase") == "secondary_frozen_hyperparameter_transfer"
        else None
    )
    if phase_key is None:
        raise Paper1FreezeContractError("sealed result cites a foreign phase")
    expected = {
        item["job_id"]: item
        for item in complete_job_grid()["phases"][phase_key]
    }
    if value.get("job_id") not in expected or value != expected[value["job_id"]]:
        raise Paper1FreezeContractError("sealed result cites a foreign job")
    return value, phase_key


def _validated_partition(value: object) -> dict[str, Any]:
    fields = {
        "outer_split_seed",
        "split_manifest_sha256",
        "development_idx_sha256",
        "outer_test_idx_sha256",
        "development_states",
        "outer_test_states",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Paper1FreezeContractError("sealed partition fields drifted")
    result = _json(dict(value))
    if result["outer_split_seed"] != OUTER_SPLIT_SEED:
        raise Paper1FreezeContractError("sealed partition seed drifted")
    for field in (
        "split_manifest_sha256",
        "development_idx_sha256",
        "outer_test_idx_sha256",
    ):
        if not _is_sha(result[field]):
            raise Paper1FreezeContractError(f"sealed partition {field} is invalid")
    for field in ("development_states", "outer_test_states"):
        states = result[field]
        if (
            not isinstance(states, list)
            or not states
            or any(isinstance(item, bool) or not isinstance(item, int) for item in states)
            or states != sorted(states)
            or len(states) != len(set(states))
            or states[0] < 0
        ):
            raise Paper1FreezeContractError(f"sealed partition {field} is invalid")
    if set(result["development_states"]) & set(result["outer_test_states"]):
        raise Paper1FreezeContractError("development and outer states overlap")
    return result


def _expected_refit_config(job: Mapping[str, Any], pipeline: str, pair: list[int]) -> dict:
    cell = _CELLS[pipeline]
    stage_contract = campaign_stage_contract(job["stage"])
    return {
        "name": f"paper1_{job['job_id']}",
        "seed": job["initialization_seed"],
        "sensor_noise": None,
        "name_short": pipeline,
        "method": cell.representation,
        "use_space2vec": cell.position_encoding,
        "use_lstm": cell.lstm,
        "use_nhits": cell.multi_rate_pooling,
        "model_type": "1D_MODULAR",
        "dofs": pair,
        "discretization": 1,
        "task": "regression",
        "target_supports": stage_contract["learning"]["target_supports"],
        "bearing_targets": stage_contract["learning"]["bearing_targets"],
    }


def validate_sealed_result(
    value: Mapping[str, Any],
    *,
    freeze_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = validate_block_freeze_artifact(freeze_artifact)
    fields = {
        "schema",
        "execution_kind",
        "campaign_run_tag",
        "complete_grid_sha256",
        "job",
        "reporting_role",
        "selection_permitted",
        "selection_artifact_sha256",
        "freeze_artifact_sha256",
        "freeze_stage",
        "slot_resolution",
        "canonical_slot",
        "pipeline",
        "selected_pair",
        "frozen_champion",
        "partition",
        "canonical_job_id",
        "canonical_result_sha256",
        "robustness",
        "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Paper1FreezeContractError("sealed result fields drifted")
    result = _json(dict(value))
    supplied = result.pop("result_sha256")
    if not _is_sha(supplied) or supplied != _sha(result):
        raise Paper1FreezeContractError("sealed result SHA-256 is invalid")
    result["result_sha256"] = supplied
    job, phase_key = _exact_report_job(result["job"])
    freeze_stage = job["stage"] if phase_key == "post_freeze_stability" else "F40-S"
    expected_role = (
        "primary_post_freeze_report_only"
        if phase_key == "post_freeze_stability" else "secondary_nonselection"
    )
    if (
        result["schema"] != SEALED_RESULT_SCHEMA
        or result["campaign_run_tag"] != freeze["campaign_run_tag"]
        or result["complete_grid_sha256"]
        != complete_job_grid()["complete_grid_sha256"]
        or result["reporting_role"] != expected_role
        or result["selection_permitted"] is not False
        or result["selection_artifact_sha256"]
        != freeze["selection_artifact"]["artifact_sha256"]
        or result["freeze_artifact_sha256"] != freeze["artifact_sha256"]
        or result["freeze_stage"] != freeze_stage
        or freeze["stage"] != freeze_stage
        or result["slot_resolution"] != freeze["slot_resolution"]
        or result["selected_pair"] != freeze["selected_pair"]
    ):
        raise Paper1FreezeContractError("sealed result identity/reporting role drifted")
    claim = freeze_for_slot(freeze, stage=freeze_stage, slot=job["pipeline"])
    if (
        result["canonical_slot"] != claim["canonical_slot"]
        or result["pipeline"] != claim["pipeline"]
        or result["frozen_champion"] != claim["winner"]
    ):
        raise Paper1FreezeContractError("sealed result frozen lineage drifted")
    is_alias = job["pipeline"] != claim["canonical_slot"]
    if is_alias != (result["execution_kind"] == "deduplicated_alias"):
        raise Paper1FreezeContractError("sealed result alias kind drifted")
    if not is_alias and result["execution_kind"] != "refit":
        raise Paper1FreezeContractError("canonical sealed job is not a refit")
    phase_jobs = complete_job_grid()["phases"][phase_key]
    canonical_jobs = [
        item for item in phase_jobs
        if item["stage"] == job["stage"]
        and item["pipeline"] == claim["canonical_slot"]
        and item["initialization_seed"] == job["initialization_seed"]
    ]
    if len(canonical_jobs) != 1:
        raise Paper1FreezeContractError("sealed alias has no unique canonical job")
    canonical_job = canonical_jobs[0]
    if result["canonical_job_id"] != canonical_job["job_id"]:
        raise Paper1FreezeContractError("sealed canonical job id drifted")
    if is_alias:
        if not _is_sha(result["canonical_result_sha256"]):
            raise Paper1FreezeContractError(
                "sealed alias lacks canonical result SHA-256"
            )
    elif result["canonical_result_sha256"] is not None:
        raise Paper1FreezeContractError(
            "canonical sealed result carries an alias result digest"
        )
    partition = _validated_partition(result["partition"])
    result["partition"] = partition
    robustness = result["robustness"]
    expected_robustness_fields = {
        "schema", "evaluation_scope", "selection_permitted",
        "outer_test_observations_accessed", "plan", "complete", "runs",
        "n_completed_refits", "n_expected_refits",
    }
    if not isinstance(robustness, Mapping) or set(robustness) != expected_robustness_fields:
        raise Paper1FreezeContractError("sealed robustness record fields drifted")
    robustness = _json(dict(robustness))
    plan = robustness["plan"]
    provenance = {
        "freeze_artifact_sha256": freeze["artifact_sha256"],
        "selection_artifact_sha256": freeze["selection_artifact"]["artifact_sha256"],
        "stage": job["stage"],
        "freeze_stage": freeze_stage,
        "reporting_role": expected_role,
    }
    expected_config = _expected_refit_config(
        canonical_job, claim["pipeline"], claim["selected_pair"]
    )
    if (
        robustness["schema"] != "ttbi-post-freeze-stability-v1"
        or robustness["evaluation_scope"] != "sealed_outer_test_post_freeze"
        or robustness["selection_permitted"] is not False
        or robustness["outer_test_observations_accessed"] is not True
        or robustness["complete"] is not True
        or robustness["n_completed_refits"] != 1
        or robustness["n_expected_refits"] != 1
        or not isinstance(plan, Mapping)
        or plan.get("config") != expected_config
        or plan.get("params") != claim["winner"]["params"]
        or plan.get("provenance") != provenance
        or plan.get("development_idx_sha256")
        != partition["development_idx_sha256"]
        or plan.get("sealed_outer_test_idx_sha256")
        != partition["outer_test_idx_sha256"]
        or not _is_sha(plan.get("groups_sha256"))
        or plan.get("initialization_seeds") != [job["initialization_seed"]]
        or plan.get("n_epochs") != claim["winner"]["frozen_checkpoint_epochs"]
        or plan.get("max_epochs") != TRAINING_EPOCHS
        or plan.get("n_scour_heads")
        != len(campaign_stage_contract(job["stage"])["learning"]["target_supports"])
        or not isinstance(robustness["runs"], list)
        or len(robustness["runs"]) != 1
    ):
        raise Paper1FreezeContractError(
            "sealed robustness plan is selectable, incomplete, or lineage-drifted"
        )
    run = robustness["runs"][0]
    if (
        not isinstance(run, Mapping)
        or set(run) != {"initialization_seed", "outer_test_states", "metrics"}
        or run["initialization_seed"] != job["initialization_seed"]
        or run["outer_test_states"] != partition["outer_test_states"]
    ):
        raise Paper1FreezeContractError("sealed robustness run identity drifted")
    metrics = _validated_metrics(run["metrics"])
    if metrics["state"] != partition["outer_test_states"]:
        raise Paper1FreezeContractError("sealed metrics do not cover outer states")
    robustness["runs"][0]["metrics"] = metrics
    result["robustness"] = robustness
    return result


def seal_sealed_result(
    value: Mapping[str, Any], *, freeze_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    result = _json(dict(value))
    result.pop("result_sha256", None)
    result["result_sha256"] = _sha(result)
    return validate_sealed_result(result, freeze_artifact=freeze_artifact)
