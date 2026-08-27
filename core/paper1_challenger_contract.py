"""Optional, secondary-only challenger grid for Paper 1.

This contract is deliberately separate from
``core.paper1_training_contract.complete_job_grid``.  The registered 16-cell
factorial, its retained slots, and every primary selection/freeze artefact are
therefore unchanged when the challenger tier is not dispatched.

The tier compares ModernTCN and TSLANet on both RAW and PAA inputs using the
same two-channel pair selected by the primary F40-S screen.  Hyperparameters
may be selected *within* each challenger arm, but no challenger result is
eligible to select another pair or alter the primary architecture, retained
slots, or block freezes.  This module registers the intended job inventory
only; it does not authorize dispatch or sealed-test access.  A future executor
must authenticate the primary selection artefact before HPO and a separate
challenger freeze before consuming the report-only outer jobs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import itertools
import json
from typing import Any, Mapping

from core.challenger_policy import (
    challenger_search_policy,
    challenger_search_policy_sha256,
)
from core.paper1_training_contract import (
    CHANNEL_SCHEMA_ID,
    DEVELOPMENT_INIT_SEEDS,
    DEVELOPMENT_N_REPEATS,
    DEVELOPMENT_N_SPLITS,
    DEVELOPMENT_PARTITION_SEED,
    ELIGIBLE_SENSOR_INDICES,
    EXCLUDED_PROXY_INDICES,
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    HPO_TRIALS_PER_STUDY,
    POST_FREEZE_STABILITY_SEEDS,
    canonical_json_bytes,
    canonical_json_sha256,
    complete_job_grid,
)


CHALLENGER_CONTRACT_SCHEMA = "paper1-challenger-contract-v2"
CHALLENGER_JOB_SCHEMA = "paper1-challenger-job-v2"
CHALLENGER_STAGE = "F40-S"
CHALLENGER_REPORTING_ROLE = "secondary_nonselection"
CHALLENGER_TIER = "optional_secondary_challenger"
CHALLENGER_PHASE_ORDER = (
    "hpo",
    "development_adjudication",
    "post_freeze_outer_report",
)
CHALLENGER_GRID_SHA256_FIELD = "complete_challenger_grid_sha256"
CHALLENGER_EXECUTION_STATUS = "contract_only_dispatch_not_implemented"
CHALLENGER_INPUT_SELECTOR = "f40s_selected_pair"
CHALLENGER_INPUT_CHANNEL_COUNT = 2
CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD = (
    "primary_selection_artifact_sha256"
)

# Prospective constraint for the still-unimplemented execution protocol.  It
# makes the intended comparison explicit without pretending that this
# contract-only module authorizes training.  A future executor must bind these
# choices (and the remaining fit details) into ``challenger_protocol_sha256``.
CHALLENGER_TRAINING_PROTOCOL_INTENT = {
    "status": "must_be_authenticated_before_dispatch",
    "initialization": "from_scratch",
    "upstream_self_supervised_pretraining_used": False,
    "upstream_training_recipe_reproduction_claimed": False,
    "optimizer": {
        "kind": "Adam",
        "lr_param": "lr",
        "weight_decay_param": "weight_decay",
    },
    "scheduler": {
        "kind": "CosineAnnealingLR",
        "eta_min": 0.0,
    },
    "source": "training.trainer.TRAIN_PROTOCOL",
    "comparison_basis": (
        "shared repository optimization protocol across model families; "
        "architecture comparison rather than upstream recipe reproduction"
    ),
    "required_identity_field": "challenger_protocol_sha256",
}

CHALLENGER_REPRESENTATIONS = ("RAW", "PAA")
CHALLENGER_MODEL_FAMILIES = ("MODERN_TCN", "TSLANET")


class Paper1ChallengerContractError(RuntimeError):
    """The optional challenger contract or one of its jobs is invalid."""


@dataclass(frozen=True)
class ChallengerArm:
    """One representation-specific challenger architecture identity."""

    arm_id: str
    representation: str
    model_family: str
    method: str
    model_type: str


def _arm_id(representation: str, model_family: str) -> str:
    return f"{representation}_{model_family}"


CHALLENGER_ARMS = tuple(
    ChallengerArm(
        arm_id=_arm_id(representation, model_family),
        representation=representation,
        model_family=model_family,
        method=representation,
        model_type=model_family,
    )
    for representation, model_family in itertools.product(
        CHALLENGER_REPRESENTATIONS, CHALLENGER_MODEL_FAMILIES
    )
)


def challenger_architecture(arm_id: str) -> dict[str, Any]:
    """Return the exact live model flags for one registered challenger arm."""

    matches = [arm for arm in CHALLENGER_ARMS if arm.arm_id == arm_id]
    if len(matches) != 1:
        raise Paper1ChallengerContractError(
            f"unregistered challenger arm {arm_id!r}"
        )
    arm = matches[0]
    return {
        "name_short": arm.arm_id,
        "method": arm.method,
        "model_type": arm.model_type,
        "use_space2vec": False,
        "use_lstm": False,
        "use_nhits": False,
    }


def all_challenger_architectures() -> list[dict[str, Any]]:
    """Return the four ordered RAW/PAA x ModernTCN/TSLANet definitions."""

    return [challenger_architecture(arm.arm_id) for arm in CHALLENGER_ARMS]


def _primary_contract_reference() -> dict[str, Any]:
    primary = complete_job_grid()
    primary_job_count = sum(len(jobs) for jobs in primary["phases"].values())
    return {
        "schema": primary["schema"],
        "complete_grid_sha256": primary["complete_grid_sha256"],
        "factorial_cell_count": len(FACTORIAL_CELLS),
        "complete_job_count": primary_job_count,
    }


_PRIMARY_CONTRACT_REFERENCE = _primary_contract_reference()
PRIMARY_COMPLETE_GRID_SHA256 = _PRIMARY_CONTRACT_REFERENCE[
    "complete_grid_sha256"
]


def _job(
    *,
    phase: str,
    arm: ChallengerArm,
    hpo_restart_seed: int | None = None,
    candidate_restart_seed: int | None = None,
    development_partition_seed: int | None = None,
    repeat_index: int | None = None,
    fold_index: int | None = None,
    initialization_seed: int | None = None,
    trials: int | None = None,
    sealed_outer_test_access_permitted: bool = False,
    requires_challenger_freeze_artifact: bool = False,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": CHALLENGER_JOB_SCHEMA,
        "tier": CHALLENGER_TIER,
        "phase": phase,
        "stage": CHALLENGER_STAGE,
        "pipeline": arm.arm_id,
        "representation": arm.representation,
        "model_family": arm.model_family,
        "model_type": arm.model_type,
        "input_selector": CHALLENGER_INPUT_SELECTOR,
        "input_channel_count": CHALLENGER_INPUT_CHANNEL_COUNT,
        "requires_primary_selection_artifact": True,
        "primary_pair_selection_eligible": False,
        "hpo_restart_seed": hpo_restart_seed,
        "candidate_restart_seed": candidate_restart_seed,
        "development_partition_seed": development_partition_seed,
        "repeat_index": repeat_index,
        "fold_index": fold_index,
        "initialization_seed": initialization_seed,
        "trials": trials,
        "reporting_role": CHALLENGER_REPORTING_ROLE,
        "primary_grid_membership": False,
        "primary_selection_eligible": False,
        "primary_artifact_mutation_permitted": False,
        "dispatch_authorized": False,
        "sealed_outer_test_access_permitted": (
            sealed_outer_test_access_permitted
        ),
        "requires_challenger_freeze_artifact": (
            requires_challenger_freeze_artifact
        ),
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "primary_complete_grid_sha256": PRIMARY_COMPLETE_GRID_SHA256,
        "challenger_search_policy_sha256": (
            challenger_search_policy_sha256()
        ),
    }
    identity["job_id"] = canonical_json_sha256(identity)[:24]
    return identity


def challenger_hpo_jobs() -> tuple[dict[str, Any], ...]:
    """Four arms x five independent 100-trial F40-S studies."""

    return tuple(
        _job(
            phase="f40s_secondary_challenger_hpo",
            arm=arm,
            hpo_restart_seed=seed,
            trials=HPO_TRIALS_PER_STUDY,
        )
        for arm in CHALLENGER_ARMS
        for seed in HPO_RESTART_SEEDS
    )


def challenger_development_adjudication_jobs() -> tuple[dict[str, Any], ...]:
    """Every HPO candidate x three grouped folds x two initializations."""

    return tuple(
        _job(
            phase="f40s_secondary_challenger_development_adjudication",
            arm=arm,
            candidate_restart_seed=restart_seed,
            development_partition_seed=DEVELOPMENT_PARTITION_SEED,
            repeat_index=repeat_index,
            fold_index=fold_index,
            initialization_seed=initialization_seed,
        )
        for arm in CHALLENGER_ARMS
        for restart_seed in HPO_RESTART_SEEDS
        for repeat_index in range(DEVELOPMENT_N_REPEATS)
        for fold_index in range(DEVELOPMENT_N_SPLITS)
        for initialization_seed in DEVELOPMENT_INIT_SEEDS
    )


def challenger_post_freeze_outer_report_jobs() -> tuple[dict[str, Any], ...]:
    """Thirty report-only sealed-test refits for each frozen challenger arm."""

    return tuple(
        _job(
            phase="f40s_secondary_challenger_post_freeze_outer_report",
            arm=arm,
            initialization_seed=initialization_seed,
            sealed_outer_test_access_permitted=True,
            requires_challenger_freeze_artifact=True,
        )
        for arm in CHALLENGER_ARMS
        for initialization_seed in POST_FREEZE_STABILITY_SEEDS
    )


def _build_challenger_job_grid() -> dict[str, Any]:
    phases = {
        "hpo": list(challenger_hpo_jobs()),
        "development_adjudication": list(
            challenger_development_adjudication_jobs()
        ),
        "post_freeze_outer_report": list(
            challenger_post_freeze_outer_report_jobs()
        ),
    }
    counts = {name: len(jobs) for name, jobs in phases.items()}
    value: dict[str, Any] = {
        "schema": CHALLENGER_CONTRACT_SCHEMA,
        "tier": CHALLENGER_TIER,
        "enabled_by_default": False,
        "execution_status": CHALLENGER_EXECUTION_STATUS,
        "dispatch_authorized": False,
        "activation_requirement": (
            "an implemented executor plus an exact separately authenticated "
            "challenger manifest are required"
        ),
        "required_before_dispatch": [
            CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD,
            "challenger_protocol_sha256",
            "implementation_source_sha256",
            "dataset_protocol_sha256",
            "capacity_receipt_sha256",
            "authenticated_manifest_sha256",
        ],
        "primary_contract": deepcopy(_PRIMARY_CONTRACT_REFERENCE),
        "primary_contract_unchanged_by_activation": True,
        "reporting_role": CHALLENGER_REPORTING_ROLE,
        "primary_selection_eligible": False,
        "primary_artifact_mutation_permitted": False,
        "stage": CHALLENGER_STAGE,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "selected_pair_policy": {
            "input_selector": CHALLENGER_INPUT_SELECTOR,
            "cardinality": CHALLENGER_INPUT_CHANNEL_COUNT,
            "source_stage": "F40-S",
            "source_phase": "f40s_channel_screen",
            "source_artifact": "paper1-f40s-selection-artifact-v1",
            "eligible_sensor_indices": list(ELIGIBLE_SENSOR_INDICES),
            "excluded_proxy_indices": list(EXCLUDED_PROXY_INDICES),
            "required_manifest_field": (
                CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD
            ),
            "resolution": (
                "authenticate the primary selection artefact, then use its "
                "selected_pair unchanged in every challenger arm"
            ),
            "challenger_pair_selection_prohibited": True,
        },
        "challenger_arms": [asdict(arm) for arm in CHALLENGER_ARMS],
        "challenger_search_policy": challenger_search_policy(),
        "challenger_search_policy_sha256": (
            challenger_search_policy_sha256()
        ),
        "challenger_training_protocol_intent": deepcopy(
            CHALLENGER_TRAINING_PROTOCOL_INTENT
        ),
        "hpo_policy": {
            "restart_seeds": list(HPO_RESTART_SEEDS),
            "trials_per_study": HPO_TRIALS_PER_STUDY,
            "study_count": counts["hpo"],
            "requested_trial_count": (
                counts["hpo"] * HPO_TRIALS_PER_STUDY
            ),
            "selection_scope": "within_arm_hyperparameter_selection_only",
        },
        "development_adjudication_policy": {
            "partition_seed": DEVELOPMENT_PARTITION_SEED,
            "n_splits": DEVELOPMENT_N_SPLITS,
            "n_repeats": DEVELOPMENT_N_REPEATS,
            "initialization_seeds": list(DEVELOPMENT_INIT_SEEDS),
            "metric": "paired state-clustered grouped-development OOF MSE",
            "aggregation": (
                "equal weight per physical state; mean over complete folds and "
                "initialization seeds"
            ),
            "candidate_record_required_fields": [
                "source_hpo_job_id",
                "source_study_identity_sha256",
                "source_completion_sha256",
                "trial_number",
                "hyperparameters",
                "hyperparameters_sha256",
            ],
            "missing_or_nonfinite_policy": (
                "disqualify that candidate; an arm with no complete finite "
                "candidate remains incomplete"
            ),
            "tie_break_order": [
                "lowest_state_balanced_oof_mse",
                "lowest_candidate_restart_seed",
            ],
            "winner_scope": "one frozen candidate per challenger arm",
            "cross_arm_winner_prohibited": True,
            "primary_selection_update_prohibited": True,
        },
        "outer_report_policy": {
            "initialization_seeds": list(POST_FREEZE_STABILITY_SEEDS),
            "requires_complete_authenticated_challenger_freeze": True,
            "contract_alone_grants_sealed_access": False,
            "selection_permitted": False,
            "report_only": True,
            "primary_selection_update_prohibited": True,
        },
        "phase_order": list(CHALLENGER_PHASE_ORDER),
        "phase_job_counts": counts,
        "complete_job_count": sum(counts.values()),
        "phases": phases,
    }
    value[CHALLENGER_GRID_SHA256_FIELD] = canonical_json_sha256(value)
    return value


def validate_challenger_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact optional grid against the live primary contract."""

    if not isinstance(value, Mapping):
        raise Paper1ChallengerContractError(
            "challenger contract must be a mapping"
        )
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Paper1ChallengerContractError(
            f"challenger contract is not finite canonical JSON: {exc}"
        ) from exc
    expected = _build_challenger_job_grid()
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise Paper1ChallengerContractError(
            "challenger contract differs from the exact optional registered grid"
        )
    supplied_sha = observed[CHALLENGER_GRID_SHA256_FIELD]
    unsigned = dict(observed)
    unsigned.pop(CHALLENGER_GRID_SHA256_FIELD)
    if supplied_sha != canonical_json_sha256(unsigned):
        raise Paper1ChallengerContractError(
            "challenger contract SHA-256 is invalid"
        )
    return observed


def complete_challenger_job_grid() -> dict[str, Any]:
    """Return a detached copy of the complete optional challenger contract."""

    return deepcopy(_REGISTERED_CHALLENGER_GRID)


def validate_challenger_job(
    value: Mapping[str, Any], *, phase_key: str | None = None
) -> dict[str, Any]:
    """Accept only one byte-for-byte canonical job from the optional grid."""

    if not isinstance(value, Mapping):
        raise Paper1ChallengerContractError("challenger job must be a mapping")
    if phase_key is not None and phase_key not in CHALLENGER_PHASE_ORDER:
        raise Paper1ChallengerContractError(
            f"unregistered challenger phase key {phase_key!r}"
        )
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Paper1ChallengerContractError(
            f"challenger job is not finite canonical JSON: {exc}"
        ) from exc
    supplied_job_id = observed.get("job_id")
    unsigned = dict(observed)
    unsigned.pop("job_id", None)
    if supplied_job_id != canonical_json_sha256(unsigned)[:24]:
        raise Paper1ChallengerContractError(
            "challenger job_id does not authenticate the complete job identity"
        )
    phases = (
        (phase_key,)
        if phase_key is not None
        else CHALLENGER_PHASE_ORDER
    )
    matches = [
        job
        for key in phases
        for job in _REGISTERED_CHALLENGER_GRID["phases"][key]
        if job["job_id"] == observed.get("job_id")
    ]
    if (
        len(matches) != 1
        or canonical_json_bytes(observed) != canonical_json_bytes(matches[0])
    ):
        raise Paper1ChallengerContractError(
            "challenger job is absent, duplicated, or differs from its contract"
        )
    return observed


def validate_contract() -> None:
    """Assert non-negotiable Paper-1 challenger invariants at import time."""

    registered = validate_challenger_contract(_REGISTERED_CHALLENGER_GRID)
    exact_arms = (
        "RAW_MODERN_TCN",
        "RAW_TSLANET",
        "PAA_MODERN_TCN",
        "PAA_TSLANET",
    )
    if tuple(arm.arm_id for arm in CHALLENGER_ARMS) != exact_arms:
        raise Paper1ChallengerContractError(
            "challenger arms drifted from the registered four-arm design"
        )
    if (
        len(HPO_RESTART_SEEDS) != 5
        or HPO_TRIALS_PER_STUDY != 100
        or DEVELOPMENT_N_SPLITS != 3
        or DEVELOPMENT_N_REPEATS != 1
        or len(DEVELOPMENT_INIT_SEEDS) != 2
        or len(POST_FREEZE_STABILITY_SEEDS) != 30
    ):
        raise Paper1ChallengerContractError(
            "challenger replicate/trial constants drifted; revise the contract "
            "and its prospective counts explicitly"
        )
    expected_counts = {
        "hpo": 20,
        "development_adjudication": 120,
        "post_freeze_outer_report": 120,
    }
    if (
        registered["phase_job_counts"] != expected_counts
        or registered["complete_job_count"] != 260
        or registered["enabled_by_default"] is not False
        or registered["dispatch_authorized"] is not False
        or registered["primary_selection_eligible"] is not False
        or registered["primary_artifact_mutation_permitted"] is not False
    ):
        raise Paper1ChallengerContractError(
            "challenger counts or fail-closed permissions drifted"
        )
    jobs = [
        job
        for phase_key in CHALLENGER_PHASE_ORDER
        for job in registered["phases"][phase_key]
    ]
    job_ids = [job["job_id"] for job in jobs]
    if len(job_ids) != 260 or len(set(job_ids)) != 260:
        raise Paper1ChallengerContractError(
            "challenger job inventory is incomplete or contains duplicate IDs"
        )
    for job in jobs:
        unsigned = dict(job)
        supplied = unsigned.pop("job_id")
        if (
            supplied != canonical_json_sha256(unsigned)[:24]
            or job["dispatch_authorized"] is not False
            or job["primary_grid_membership"] is not False
            or job["primary_selection_eligible"] is not False
            or job["primary_artifact_mutation_permitted"] is not False
            or job["input_selector"] != CHALLENGER_INPUT_SELECTOR
            or job["input_channel_count"] != CHALLENGER_INPUT_CHANNEL_COUNT
            or job["requires_primary_selection_artifact"] is not True
            or job["primary_pair_selection_eligible"] is not False
        ):
            raise Paper1ChallengerContractError(
                "challenger job identity or fail-closed permissions drifted"
            )


_REGISTERED_CHALLENGER_GRID = _build_challenger_job_grid()
validate_contract()
