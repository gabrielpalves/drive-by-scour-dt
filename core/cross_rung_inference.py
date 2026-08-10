"""Registered matched-block inference for the four Paper-1 datasets.

The retired seven-edge L60 ladder no longer exists.  The only prospective
cross-block comparisons are:

* F40-S versus F40-M on the exact 30 generated semantic states shared by the
  two otherwise different F40 inventories; and
* L99-S versus L99-M on their complete shared 475-state inventory.

Model metrics are evaluated only on the identical sealed-test StateUID subset
declared by both endpoint payloads.  StateUID is the resampling unit; registered
training seeds remain a fixed paired axis and are never resampled.  Intervals
are finite-design resampling-sensitivity summaries, not population confidence
intervals.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.campaign_contract import (
    EXPECTED_CHANNEL_SCHEMA_ID,
    campaign_contract_sha256,
    campaign_stage_contract,
)


MATCHED_BLOCK_INPUT_SCHEMA = "paper1-matched-block-stage-metrics-v1"
MATCHED_BLOCK_INFERENCE_SCHEMA = "paper1-matched-block-inference-v1"
REGISTERED_BLOCK_PAIRS = (
    ("F40-S__F40-M", "F40-S", "F40-M", 30),
    ("L99-S__L99-M", "L99-S", "L99-M", 475),
)
MATCHED_BLOCK_BOOTSTRAP_N = 100_000
MATCHED_BLOCK_BOOTSTRAP_SEED = 42
POINTWISE_CENTRAL_MASS = 0.95
FAMILYWISE_CENTRAL_MASS = 1.0 - (1.0 - POINTWISE_CENTRAL_MASS) / len(
    REGISTERED_BLOCK_PAIRS
)
REQUIRED_INPUT_FIELDS = {
    "schema",
    "stage",
    "dataset",
    "channel_schema_id",
    "campaign_contract_sha256",
    "evaluation_role",
    "pipeline_slot",
    "input_selector",
    "metric_name",
    "registered_seeds",
    "generated_state_uids",
    "partition_by_uid",
    "metric_rows",
}
REQUIRED_METRIC_ROW_FIELDS = {"stage", "state_uid", "seed", "scour_mse"}


MATCHED_BLOCK_INFERENCE_POLICY = {
    "schema": MATCHED_BLOCK_INFERENCE_SCHEMA,
    "pairs": [
        {
            "pair_id": pair_id,
            "left_stage": left,
            "right_stage": right,
            "generated_matched_state_count": count,
        }
        for pair_id, left, right, count in REGISTERED_BLOCK_PAIRS
    ],
    "evaluation": {
        "role": "post_freeze_sealed_test_stability",
        "unit": "semantic StateUID",
        "eligibility": "same matched StateUID assigned test at both endpoints",
        "required_pipeline_and_input": "identical logical slot and selector",
        "registered_seed_axis": "fixed and paired; never resampled",
    },
    "statistic": {
        "within_seed": "mean scour_mse over eligible StateUIDs",
        "across_seed": "median over the finite registered seed set",
        "effect": "right endpoint minus left endpoint",
    },
    "bootstrap": {
        "unit": "StateUID, paired across endpoint and registered seed",
        "n": MATCHED_BLOCK_BOOTSTRAP_N,
        "seed": MATCHED_BLOCK_BOOTSTRAP_SEED,
        "pointwise_central_mass": POINTWISE_CENTRAL_MASS,
        "two_pair_familywise_central_mass": FAMILYWISE_CENTRAL_MASS,
        "interpretation": "finite-design resampling sensitivity only",
    },
    "claims": {
        "population_confidence_interval": False,
        "seed_superpopulation": False,
        "automatic_superiority_claim": False,
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _semantic_uid(
    stage: str,
    family: str,
    target: int,
    level: int,
    replica: int,
) -> str:
    geometry = campaign_stage_contract(stage)["geometry"]
    support_text = "".join(
        f"{int(value):02d}" for value in geometry["scour_supports"]
    )
    return (
        f"ttbi-state-v2|Lmm={round(1000 * geometry['L_bridge_m']):06d}|"
        f"spans={geometry['num_spans']}|scour={support_text}|"
        f"family={family}|target={target:02d}|level={level:04d}|"
        f"rep={replica:03d}"
    )


def registered_stage_uid_inventory(stage: str) -> tuple[str, ...]:
    """Return the exact generated semantic UID row order for one block."""

    if stage == "F40-S":
        rows: list[str] = []
        for severity in range(61):
            for replica in range(1, 6):
                if severity == 0:
                    rows.append(_semantic_uid(
                        stage, "target_healthy", 0, 0, replica
                    ))
                else:
                    rows.append(_semantic_uid(
                        stage, "scour_only", 2, severity, replica
                    ))
        return tuple(rows)
    if stage not in ("F40-M", "L99-S", "L99-M"):
        raise RuntimeError(f"unregistered Paper-1 stage {stage!r}")
    geometry = campaign_stage_contract(stage)["geometry"]
    rows = [
        _semantic_uid(stage, "target_healthy", 0, 0, replica)
        for replica in range(1, 51)
    ]
    for support in geometry["scour_supports"]:
        for replica in range(1, 6):
            for severity in (12, 24, 36, 48, 60):
                rows.append(_semantic_uid(
                    stage, "scour_only", int(support), severity, replica
                ))
    for abutment in (1, 2):
        for replica in range(1, 6):
            for level in range(1, 6):
                rows.append(_semantic_uid(
                    stage, "bearing_only", abutment, level, replica
                ))
    rows.extend(
        _semantic_uid(stage, "nuisance_only", 0, 0, replica)
        for replica in range(1, 51)
    )
    rows.extend(
        _semantic_uid(stage, "joint", 0, row, 1)
        for row in range(1, 251)
    )
    return tuple(rows)


def registered_matched_uids(pair_id: str) -> tuple[str, ...]:
    """Return the exact generated intersection for one registered pair."""

    try:
        _name, left, right, expected = next(
            pair for pair in REGISTERED_BLOCK_PAIRS if pair[0] == pair_id
        )
    except StopIteration as exc:
        raise RuntimeError(f"unregistered matched pair {pair_id!r}") from exc
    left_uids = registered_stage_uid_inventory(left)
    right_set = set(registered_stage_uid_inventory(right))
    matched = tuple(uid for uid in left_uids if uid in right_set)
    if len(matched) != expected:
        raise RuntimeError(
            f"{pair_id}: semantic intersection {len(matched)} != {expected}"
        )
    return matched


def _strict_seed_list(value: Any, owner: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value)
        or len(value) != len(set(value))
    ):
        raise RuntimeError(f"{owner}: registered_seeds must be unique integers")
    return tuple(value)


def _validate_payload(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != REQUIRED_INPUT_FIELDS:
        raise RuntimeError(f"{stage}: input fields differ from the exact contract")
    value = deepcopy(dict(payload))
    contract = campaign_stage_contract(stage)
    if (
        value["schema"] != MATCHED_BLOCK_INPUT_SCHEMA
        or value["stage"] != stage
        or value["dataset"] != contract["dataset"]
        or value["channel_schema_id"] != EXPECTED_CHANNEL_SCHEMA_ID
        or value["campaign_contract_sha256"] != campaign_contract_sha256(stage)
        or value["evaluation_role"] != "post_freeze_sealed_test_stability"
        or value["metric_name"] != "scour_mse"
        or not isinstance(value["pipeline_slot"], str)
        or not value["pipeline_slot"]
        or not isinstance(value["input_selector"], (str, list))
    ):
        raise RuntimeError(f"{stage}: input metadata differs from its contract")
    seeds = _strict_seed_list(value["registered_seeds"], stage)
    expected_inventory = registered_stage_uid_inventory(stage)
    inventory = value["generated_state_uids"]
    if (
        not isinstance(inventory, list)
        or inventory != list(expected_inventory)
        or len(inventory) != len(set(inventory))
    ):
        raise RuntimeError(f"{stage}: generated StateUID inventory is not exact")
    partitions = value["partition_by_uid"]
    if (
        not isinstance(partitions, dict)
        or set(partitions) != set(expected_inventory)
        or any(partition not in ("train", "val", "test")
               for partition in partitions.values())
    ):
        raise RuntimeError(f"{stage}: partition_by_uid is incomplete or invalid")
    rows = value["metric_rows"]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{stage}: metric_rows must be nonempty")
    cells: dict[tuple[str, int], float] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != REQUIRED_METRIC_ROW_FIELDS:
            raise RuntimeError(f"{stage}: metric row fields differ")
        uid, seed, raw_metric = row["state_uid"], row["seed"], row["scour_mse"]
        if (
            row["stage"] != stage
            or uid not in partitions
            or partitions[uid] != "test"
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in seeds
            or isinstance(raw_metric, bool)
            or not isinstance(raw_metric, (int, float))
            or not math.isfinite(float(raw_metric))
            or float(raw_metric) < 0.0
        ):
            raise RuntimeError(f"{stage}: invalid sealed-test metric cell")
        key = (uid, seed)
        if key in cells:
            raise RuntimeError(f"{stage}: duplicate StateUID x seed cell {key!r}")
        cells[key] = float(raw_metric)
    value["_seeds"] = seeds
    value["_cells"] = cells
    return value


def _statistic(matrix: np.ndarray) -> float:
    return float(np.median(np.asarray(matrix, dtype=np.float64).mean(axis=0)))


def _bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, int)
        or n_boot < 100
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise RuntimeError("bootstrap n/seed is not a registered integer fixture")
    if left.shape != right.shape or left.ndim != 2:
        raise RuntimeError("paired matrices are not aligned StateUID x seed")
    if left.shape[0] < 2 or left.shape[1] < 1:
        raise RuntimeError("matched inference needs >=2 StateUIDs and >=1 seed")
    estimate = _statistic(right) - _statistic(left)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 4096):
        stop = min(start + 4096, n_boot)
        positions = rng.integers(
            0, left.shape[0], size=(stop - start, left.shape[0])
        )
        draws[start:stop] = (
            np.median(right[positions].mean(axis=1), axis=1)
            - np.median(left[positions].mean(axis=1), axis=1)
        )
    return estimate, draws


def _interval(draws: np.ndarray, central_mass: float) -> tuple[float, float]:
    alpha = (1.0 - central_mass) / 2.0
    values = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(values[0]), float(values[1])


def analyze_registered_matched_blocks(
    payload_by_stage: Mapping[str, Mapping[str, Any]],
    *,
    n_boot: int = MATCHED_BLOCK_BOOTSTRAP_N,
    bootstrap_seed: int = MATCHED_BLOCK_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate all four endpoint payloads and compute two paired effects."""

    required_stages = {
        stage for _pair, left, right, _count in REGISTERED_BLOCK_PAIRS
        for stage in (left, right)
    }
    if not isinstance(payload_by_stage, Mapping) or set(payload_by_stage) != required_stages:
        raise RuntimeError("payload mapping must contain exactly four Paper-1 stages")
    stages = {
        stage: _validate_payload(stage, payload_by_stage[stage])
        for stage in sorted(required_stages)
    }
    pair_results: list[dict[str, Any]] = []
    for pair_index, (pair_id, left_stage, right_stage, matched_count) in enumerate(
        REGISTERED_BLOCK_PAIRS
    ):
        left, right = stages[left_stage], stages[right_stage]
        matched_uids = registered_matched_uids(pair_id)
        if (
            left["pipeline_slot"] != right["pipeline_slot"]
            or left["input_selector"] != right["input_selector"]
            or left["_seeds"] != right["_seeds"]
        ):
            raise RuntimeError(
                f"{pair_id}: pipeline/input/registered seed axes are not paired"
            )
        for uid in matched_uids:
            if left["partition_by_uid"][uid] != right["partition_by_uid"][uid]:
                raise RuntimeError(f"{pair_id}: matched UID partition differs")
        eligible = tuple(
            uid for uid in matched_uids
            if left["partition_by_uid"][uid] == "test"
        )
        if len(eligible) < 2:
            raise RuntimeError(f"{pair_id}: fewer than two matched test UIDs")
        seeds = left["_seeds"]
        expected_cells = {(uid, seed) for uid in eligible for seed in seeds}
        left_cells = left["_cells"]
        right_cells = right["_cells"]
        if set(left_cells) != expected_cells or set(right_cells) != expected_cells:
            raise RuntimeError(
                f"{pair_id}: metric rows are not the exact matched test UID x seed grid"
            )
        left_matrix = np.asarray([
            [left_cells[(uid, seed)] for seed in seeds] for uid in eligible
        ], dtype=np.float64)
        right_matrix = np.asarray([
            [right_cells[(uid, seed)] for seed in seeds] for uid in eligible
        ], dtype=np.float64)
        estimate, draws = _bootstrap(
            left_matrix,
            right_matrix,
            n_boot=n_boot,
            seed=bootstrap_seed + pair_index,
        )
        pointwise = _interval(draws, POINTWISE_CENTRAL_MASS)
        adjusted = _interval(draws, FAMILYWISE_CENTRAL_MASS)
        pair_results.append({
            "pair_id": pair_id,
            "left_stage": left_stage,
            "right_stage": right_stage,
            "pipeline_slot": left["pipeline_slot"],
            "input_selector": deepcopy(left["input_selector"]),
            "generated_matched_state_count": matched_count,
            "eligible_test_state_count": len(eligible),
            "eligible_test_state_uids_sha256": canonical_sha256(list(eligible)),
            "registered_seeds": list(seeds),
            "estimate_right_minus_left": estimate,
            "pointwise_resampling_sensitivity_interval": list(pointwise),
            "two_pair_adjusted_resampling_sensitivity_interval": list(adjusted),
            "bootstrap_fraction_positive": float(np.mean(draws > 0.0)),
            "population_confidence_interval": False,
            "automatic_superiority_claim": False,
        })
    result = {
        "schema": MATCHED_BLOCK_INFERENCE_SCHEMA,
        "policy": deepcopy(MATCHED_BLOCK_INFERENCE_POLICY),
        "bootstrap_n": n_boot,
        "bootstrap_seed": bootstrap_seed,
        "pairs": pair_results,
        "input_payload_sha256": {
            stage: canonical_sha256({
                key: value for key, value in payload_by_stage[stage].items()
            })
            for stage in sorted(required_stages)
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _load_canonical_payload(path: Path, stage: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{stage}: input must be an absolute regular file")
    if path.absolute() != path.resolve(strict=True):
        raise RuntimeError(f"{stage}: input path must already be canonical")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{stage}: input is not valid JSON") from exc
    if raw != _canonical_bytes(value):
        raise RuntimeError(f"{stage}: input bytes are not canonical JSON")
    return value


def analyze_registered_matched_block_files(
    payload_paths: Mapping[str, str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    n_boot: int = MATCHED_BLOCK_BOOTSTRAP_N,
    bootstrap_seed: int = MATCHED_BLOCK_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Authenticate canonical payload files and atomically publish one result."""

    payloads = {
        stage: _load_canonical_payload(Path(raw_path), stage)
        for stage, raw_path in payload_paths.items()
    }
    result = analyze_registered_matched_blocks(
        payloads, n_boot=n_boot, bootstrap_seed=bootstrap_seed
    )
    destination = Path(output_path)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise RuntimeError("output path must be an absent absolute regular-file path")
    parent = destination.parent.resolve(strict=True)
    if destination.parent.absolute() != parent:
        raise RuntimeError("output parent path must already be canonical")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("output temporary path already exists")
    temporary.write_bytes(_canonical_bytes(result))
    os.replace(temporary, destination)
    return result


def _validate_definitions() -> None:
    if FAMILYWISE_CENTRAL_MASS != 0.975:
        raise RuntimeError("two-pair familywise tail allocation drifted")
    if len(registered_matched_uids("F40-S__F40-M")) != 30:
        raise RuntimeError("F40 exact semantic matched subset drifted")
    if len(registered_matched_uids("L99-S__L99-M")) != 475:
        raise RuntimeError("L99 complete semantic pairing drifted")


_validate_definitions()
