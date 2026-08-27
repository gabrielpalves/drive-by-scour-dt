"""Executable Paper-1 training and evaluation design.

This module is intentionally independent of Optuna/PyTorch.  It enumerates the
complete pre-outcome job grid that the campaign driver and bundle builder must
consume.  Data-dependent winners are represented by authenticated selector
slots; they are never guessed while bundles are built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable


TRAINING_CONTRACT_SCHEMA = "paper1-training-contract-v2"
TRAINING_JOB_SCHEMA = "paper1-training-job-v2"
STAGE_ORDER = ("F40-S", "F40-M", "L99-S", "L99-M")
CHANNEL_SCHEMA_ID = "physical8_v1"
CHANNEL_NAMES = (
    "carbody_vertical_acceleration",
    "front_bogie_vertical_acceleration",
    "rear_bogie_vertical_acceleration",
    "wheelset_1_constrained_vertical_acceleration_proxy",
    "wheelset_2_constrained_vertical_acceleration_proxy",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
)
# ``physical8_v1`` is the on-disk response inventory, not an assertion that
# every stored row is an equivalent candidate sensor.  Rows 3-4 are idealized
# constrained-wheelset kinematic proxies retained for contact/solver V&V; they
# are not independent wheel DOFs or instrument models.  Scientific learning
# and sensor selection are therefore restricted to the six vehicle responses
# that have a direct accelerometer/gyroscope interpretation.
EXCLUDED_PROXY_INDICES = (3, 4)
ELIGIBLE_SENSOR_INDICES = (0, 1, 2, 5, 6, 7)
ELIGIBLE_SENSOR_NAMES = tuple(CHANNEL_NAMES[index] for index in ELIGIBLE_SENSOR_INDICES)
ANCHOR_CHANNEL_INDEX = 1
ANCHOR_CHANNEL_NAME = CHANNEL_NAMES[ANCHOR_CHANNEL_INDEX]
OUTER_SPLIT_SEED = 42
HPO_RESTART_SEEDS = (104729, 130363, 155921, 196613, 228017)
DEVELOPMENT_PARTITION_SEED = 271828
DEVELOPMENT_N_SPLITS = 3
DEVELOPMENT_N_REPEATS = 1
DEVELOPMENT_INIT_SEEDS = (424243, 8675309)
# Thirty post-selection initializations per retained pipeline.  This is the
# prospectively fixed 120-fit/block stability set (four pipelines x 30), not
# the 480-fit development adjudication used to choose F40-S vectors.
POST_FREEZE_STABILITY_SEEDS = (
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
    1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123,
    1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
)
SCREEN_REFIT_SEEDS = (4001, 4003, 4007, 4013, 4019)
HPO_TRIALS_PER_STUDY = 100
HPO_RESTARTS_PER_CONFIGURATION = 5
TRAINING_EPOCHS = 50
RETAINED_PIPELINE_SLOTS = (
    "f40s_best_raw",
    "f40s_best_paa",
    "raw_cnn_gap_baseline",
    "paa_cnn_gap_baseline",
)
REPRESENTATIONS = ("RAW", "PAA")
BOOLEAN_LEVELS = (False, True)


@dataclass(frozen=True)
class ArchitectureCell:
    cell_id: str
    representation: str
    position_encoding: bool
    lstm: bool
    multi_rate_pooling: bool


def _cell_id(
    representation: str,
    position_encoding: bool,
    lstm: bool,
    multi_rate_pooling: bool,
) -> str:
    return (
        f"{representation}_POS{int(position_encoding)}_"
        f"LSTM{int(lstm)}_MR{int(multi_rate_pooling)}"
    )


FACTORIAL_CELLS = tuple(
    ArchitectureCell(
        cell_id=_cell_id(representation, position, lstm, multi_rate),
        representation=representation,
        position_encoding=position,
        lstm=lstm,
        multi_rate_pooling=multi_rate,
    )
    for representation, position, lstm, multi_rate in itertools.product(
        REPRESENTATIONS, BOOLEAN_LEVELS, BOOLEAN_LEVELS, BOOLEAN_LEVELS
    )
)
RAW_CNN_GAP_BASELINE_ID = _cell_id("RAW", False, False, False)
PAA_CNN_GAP_BASELINE_ID = _cell_id("PAA", False, False, False)


def channel_screen_inputs() -> tuple[tuple[int, ...], ...]:
    """Return the six measurable singles and their 15 unordered pairs."""

    singles = tuple((index,) for index in ELIGIBLE_SENSOR_INDICES)
    pairs = tuple(itertools.combinations(ELIGIBLE_SENSOR_INDICES, 2))
    return singles + pairs


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _job(
    *,
    phase: str,
    stage: str,
    pipeline: str,
    input_selector: str | tuple[int, ...],
    hpo_restart_seed: int | None = None,
    development_partition_seed: int | None = None,
    fold_index: int | None = None,
    initialization_seed: int | None = None,
    candidate_restart_seed: int | None = None,
    trials: int | None = None,
    reporting_role: str = "primary",
) -> dict[str, Any]:
    if isinstance(input_selector, tuple):
        serialized_selector: str | list[int] = list(input_selector)
    else:
        serialized_selector = input_selector
    identity = {
        "schema": TRAINING_JOB_SCHEMA,
        "phase": phase,
        "stage": stage,
        "pipeline": pipeline,
        "input_selector": serialized_selector,
        "hpo_restart_seed": hpo_restart_seed,
        "development_partition_seed": development_partition_seed,
        "fold_index": fold_index,
        "initialization_seed": initialization_seed,
        "candidate_restart_seed": candidate_restart_seed,
        "trials": trials,
        "reporting_role": reporting_role,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
    }
    identity["job_id"] = canonical_json_sha256(identity)[:24]
    return identity


def hpo_jobs() -> tuple[dict[str, Any], ...]:
    """Return 160 exact 100-trial templates (16,000 requested trials)."""

    jobs: list[dict[str, Any]] = []
    # Full 2x2x2x2 architecture comparison on the prospectively selected
    # front-bogie vertical channel: 16 cells x five independent restarts.
    for cell in FACTORIAL_CELLS:
        for seed in HPO_RESTART_SEEDS:
            jobs.append(_job(
                phase="f40s_factorial_hpo",
                stage="F40-S",
                pipeline=cell.cell_id,
                input_selector=(ANCHOR_CHANNEL_INDEX,),
                hpo_restart_seed=seed,
                trials=HPO_TRIALS_PER_STUDY,
            ))

    # The four retained pipelines are re-optimized on the pair selected by the
    # complete frozen-HP screen.  Slot aliases are resolved from authenticated
    # F40-S selection artifacts and deduplicated if a winner is its baseline.
    for slot in RETAINED_PIPELINE_SLOTS:
        for seed in HPO_RESTART_SEEDS:
            jobs.append(_job(
                phase="f40s_selected_pair_hpo",
                stage="F40-S",
                pipeline=slot,
                input_selector="f40s_selected_pair",
                hpo_restart_seed=seed,
                trials=HPO_TRIALS_PER_STUDY,
            ))

    # No transport/rescue gate: every remaining scientific block receives its
    # own 4 x 5 x 100 selected-pair HPO.  Hardware matching is required within
    # each block, not across different blocks.
    for stage in STAGE_ORDER[1:]:
        for slot in RETAINED_PIPELINE_SLOTS:
            for seed in HPO_RESTART_SEEDS:
                jobs.append(_job(
                    phase="block_selected_pair_hpo",
                    stage=stage,
                    pipeline=slot,
                    input_selector="f40s_selected_pair",
                    hpo_restart_seed=seed,
                    trials=HPO_TRIALS_PER_STUDY,
                ))
    return tuple(jobs)


def development_adjudication_jobs() -> tuple[dict[str, Any], ...]:
    """Five winners x one complete three-fold OOF partition x two inits."""

    return tuple(
        _job(
            phase="f40s_development_adjudication",
            stage="F40-S",
            pipeline=cell.cell_id,
            input_selector=(ANCHOR_CHANNEL_INDEX,),
            candidate_restart_seed=restart_seed,
            development_partition_seed=DEVELOPMENT_PARTITION_SEED,
            fold_index=fold_index,
            initialization_seed=init_seed,
        )
        for cell in FACTORIAL_CELLS
        for restart_seed in HPO_RESTART_SEEDS
        for fold_index in range(DEVELOPMENT_N_SPLITS)
        for init_seed in DEVELOPMENT_INIT_SEEDS
    )


def channel_screen_jobs() -> tuple[dict[str, Any], ...]:
    """Four retained slots x (6 singles + 15 pairs) x five paired refits."""

    return tuple(
        _job(
            phase="f40s_frozen_hyperparameter_channel_screen",
            stage="F40-S",
            pipeline=slot,
            input_selector=channels,
            initialization_seed=seed,
        )
        for slot in RETAINED_PIPELINE_SLOTS
        for channels in channel_screen_inputs()
        for seed in SCREEN_REFIT_SEEDS
    )


def post_freeze_stability_jobs() -> tuple[dict[str, Any], ...]:
    """A disjoint 30-seed sealed-test refit set for every stage/pipeline."""

    return tuple(
        _job(
            phase="post_freeze_sealed_test_stability",
            stage=stage,
            pipeline=slot,
            input_selector="f40s_selected_pair",
            initialization_seed=seed,
        )
        for stage in STAGE_ORDER
        for slot in RETAINED_PIPELINE_SLOTS
        for seed in POST_FREEZE_STABILITY_SEEDS
    )


def frozen_transfer_jobs() -> tuple[dict[str, Any], ...]:
    """Secondary-only analysis of F40-S settings without block retuning."""

    return tuple(
        _job(
            phase="secondary_frozen_hyperparameter_transfer",
            stage=stage,
            pipeline=slot,
            input_selector="f40s_selected_pair",
            initialization_seed=seed,
            reporting_role="secondary_nonselection",
        )
        for stage in STAGE_ORDER[1:]
        for slot in RETAINED_PIPELINE_SLOTS
        for seed in SCREEN_REFIT_SEEDS
    )


def complete_job_grid() -> dict[str, Any]:
    phases = {
        "hpo": list(hpo_jobs()),
        "development_adjudication": list(development_adjudication_jobs()),
        "channel_screen": list(channel_screen_jobs()),
        "post_freeze_stability": list(post_freeze_stability_jobs()),
        "secondary_frozen_transfer": list(frozen_transfer_jobs()),
    }
    contract = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "stage_order": list(STAGE_ORDER),
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "channel_names": list(CHANNEL_NAMES),
        "eligible_sensor_indices": list(ELIGIBLE_SENSOR_INDICES),
        "eligible_sensor_names": list(ELIGIBLE_SENSOR_NAMES),
        "excluded_proxy_indices": list(EXCLUDED_PROXY_INDICES),
        "anchor_channel_index": ANCHOR_CHANNEL_INDEX,
        "anchor_channel_name": ANCHOR_CHANNEL_NAME,
        "outer_split_seed": OUTER_SPLIT_SEED,
        "factorial_cells": [asdict(cell) for cell in FACTORIAL_CELLS],
        "retained_pipeline_slots": list(RETAINED_PIPELINE_SLOTS),
        "slot_resolution": {
            "f40s_best_raw": "development-CV winner among eight RAW cells",
            "f40s_best_paa": "development-CV winner among eight PAA cells",
            "raw_cnn_gap_baseline": RAW_CNN_GAP_BASELINE_ID,
            "paa_cnn_gap_baseline": PAA_CNN_GAP_BASELINE_ID,
            "deduplicate_after_authenticated_resolution": True,
        },
        "hpo_trials_per_study": HPO_TRIALS_PER_STUDY,
        "training_epochs": TRAINING_EPOCHS,
        "hpo_restart_seeds": list(HPO_RESTART_SEEDS),
        "development_partition_seed": DEVELOPMENT_PARTITION_SEED,
        "development_n_splits": DEVELOPMENT_N_SPLITS,
        "development_n_repeats": DEVELOPMENT_N_REPEATS,
        "development_init_seeds": list(DEVELOPMENT_INIT_SEEDS),
        "post_freeze_stability_seeds": list(POST_FREEZE_STABILITY_SEEDS),
        "sealed_outer_test_policy": (
            "outer state groups remain unopened through HPO, development "
            "adjudication, channel screening, and selected-pair HPO"
        ),
        "selection_metric": "paired state-clustered grouped-development OOF MSE",
        "transport_rescue_policy": "withdrawn; every block has independent HPO",
        "phases": phases,
    }
    contract["complete_grid_sha256"] = canonical_json_sha256(contract)
    return contract


def resolve_retained_pipelines(
    *, best_raw: str, best_paa: str
) -> tuple[str, ...]:
    valid = {cell.cell_id for cell in FACTORIAL_CELLS}
    if best_raw not in valid or not best_raw.startswith("RAW_"):
        raise ValueError("best_raw is not a registered RAW factorial cell")
    if best_paa not in valid or not best_paa.startswith("PAA_"):
        raise ValueError("best_paa is not a registered PAA factorial cell")
    ordered = (
        best_raw,
        best_paa,
        RAW_CNN_GAP_BASELINE_ID,
        PAA_CNN_GAP_BASELINE_ID,
    )
    return tuple(dict.fromkeys(ordered))


def validate_contract() -> None:
    if len(FACTORIAL_CELLS) != 16:
        raise RuntimeError("factorial architecture grid is not 2x2x2x2")
    if len({cell.cell_id for cell in FACTORIAL_CELLS}) != 16:
        raise RuntimeError("factorial architecture identifiers collide")
    if (
        EXCLUDED_PROXY_INDICES != (3, 4)
        or ELIGIBLE_SENSOR_INDICES != (0, 1, 2, 5, 6, 7)
        or set(EXCLUDED_PROXY_INDICES) & set(ELIGIBLE_SENSOR_INDICES)
        or set(EXCLUDED_PROXY_INDICES) | set(ELIGIBLE_SENSOR_INDICES)
        != set(range(len(CHANNEL_NAMES)))
        or ELIGIBLE_SENSOR_NAMES
        != tuple(CHANNEL_NAMES[index] for index in ELIGIBLE_SENSOR_INDICES)
    ):
        raise RuntimeError("scientific sensor eligibility drifted from physical8_v1")
    if len(channel_screen_inputs()) != 21:
        raise RuntimeError("channel screen is not 6 eligible singles plus 15 pairs")
    if len(HPO_RESTART_SEEDS) != HPO_RESTARTS_PER_CONFIGURATION:
        raise RuntimeError("HPO restart seed inventory drifted")
    hpo = hpo_jobs()
    if len(hpo) != 160 or sum(job["trials"] for job in hpo) != 16000:
        raise RuntimeError("per-block HPO grid is not 160 x 100 trials")
    expected_counts = {
        "development_adjudication": 480,
        "channel_screen": 420,
        "post_freeze_stability": 480,
        "secondary_frozen_transfer": 60,
    }
    actual_counts = {
        "development_adjudication": len(development_adjudication_jobs()),
        "channel_screen": len(channel_screen_jobs()),
        "post_freeze_stability": len(post_freeze_stability_jobs()),
        "secondary_frozen_transfer": len(frozen_transfer_jobs()),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"training refit grid drifted: {actual_counts!r}")
    all_jobs = (
        hpo
        + development_adjudication_jobs()
        + channel_screen_jobs()
        + post_freeze_stability_jobs()
        + frozen_transfer_jobs()
    )
    identifiers = [job["job_id"] for job in all_jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("training job identifiers collide")


validate_contract()
