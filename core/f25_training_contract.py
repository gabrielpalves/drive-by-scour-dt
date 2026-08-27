"""Machine-readable F25-R/F25-X training and dispatch job contract.

This module expands :mod:`core.f25_experiment_contract` into executable jobs
without importing PyTorch or Optuna.  Bundles, capacity checks, and the runtime
executor all consume the same canonical plan, so tier ordering, sensor sets,
HPO semantics, anchor provenance, and artifact isolation cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Iterable

from core.f25_experiment_contract import (
    CHANNELS,
    F25_R_CHANNELS,
    F25_R_EXPERIMENT,
    F25_R_UNFROZEN,
    F25_X_CHANNELS,
    F25_X_EXPERIMENT,
    F25_X_FROZEN_PAIRS,
    F25_X_FROZEN_SINGLES,
    F25_X_UNFROZEN_SINGLES,
    HPO_EXECUTION_SEEDS,
    REPORT_SEEDS,
    TierPlan,
    build_contract,
)


TRAINING_PLAN_SCHEMA = "f25-training-plan-v1"
TRAINING_JOB_SCHEMA = "f25-training-job-v1"
EXECUTION_BLOCK_ID = "F25-complete-balanced-v1"
SHARED_GENERATION_ROOT = (
    "f25_artifacts/shared_data/generation/fernandes-2025-f25-data-v1"
)

_ARM_TAG = {
    "RAW-CNN": "RAW",
    "PAA-CNN": "PAA",
    "PAA-multirate": "PAA-MR",
}


class F25TrainingContractError(ValueError):
    """Raised when an F25 training plan is malformed or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise F25TrainingContractError(
            f"F25 training payload is not canonical JSON: {exc}"
        ) from exc
    return text.encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _experiment(experiment_id: str):
    if experiment_id == "F25-R":
        return F25_R_EXPERIMENT
    if experiment_id == "F25-X":
        return F25_X_EXPERIMENT
    raise F25TrainingContractError("experiment_id must be F25-R or F25-X")


def _sensor_code(sensor_set: Iterable[str]) -> str:
    indices = []
    for channel in sensor_set:
        if channel not in CHANNELS:
            raise F25TrainingContractError(f"unknown physical8 channel: {channel}")
        indices.append(CHANNELS.index(channel))
    if not indices or len(set(indices)) != len(indices):
        raise F25TrainingContractError("sensor set must be nonempty and unique")
    return "-".join(f"S{index:02d}" for index in indices)


def configuration_id(arm_id: str, sensor_set: Iterable[str]) -> str:
    try:
        arm_tag = _ARM_TAG[arm_id]
    except KeyError as exc:
        raise F25TrainingContractError(f"unknown F25 arm: {arm_id}") from exc
    return f"{arm_tag}__{_sensor_code(tuple(sensor_set))}"


def _anchor_configuration(arm_id: str) -> str:
    source_arm = "PAA-CNN" if arm_id == "PAA-multirate" else arm_id
    return configuration_id(source_arm, (CHANNELS[1],))


def _job(
    *,
    experiment_id: str,
    tier: TierPlan,
    phase: str,
    arm_id: str,
    sensor_set: tuple[str, ...],
    anchor_configuration_id: str | None,
) -> dict[str, Any]:
    experiment = _experiment(experiment_id)
    config_id = configuration_id(arm_id, sensor_set)
    job_id = f"{experiment_id}/{tier.tier_id}/{phase}/{config_id}"
    if phase not in {"hpo", "report"}:
        raise F25TrainingContractError("F25 training phase must be hpo or report")
    if phase == "hpo":
        proposals = tier.hpo_proposals_per_configuration
        executions = tier.hpo_executions_per_proposal
        seeds = list(HPO_EXECUTION_SEEDS)
    else:
        proposals = 0
        executions = 0
        seeds = list(REPORT_SEEDS)
    representation = "RAW" if arm_id == "RAW-CNN" else "PAA"
    pooling = "multi-rate" if arm_id == "PAA-multirate" else "flatten+dense"
    manifest_path = str(
        PurePosixPath(experiment.manifest_root)
        / tier.tier_id
        / phase
        / config_id
    )
    results_path = str(
        PurePosixPath(experiment.results_root)
        / tier.tier_id
        / phase
        / config_id
    )
    return {
        "schema": TRAINING_JOB_SCHEMA,
        "job_id": job_id,
        "experiment_id": experiment_id,
        "tier_id": tier.tier_id,
        "regime": tier.regime,
        "phase": phase,
        "configuration_id": config_id,
        "arm_id": arm_id,
        "representation": representation,
        "pooling": pooling,
        "sensor_set": list(sensor_set),
        "sensor_indices": [CHANNELS.index(channel) for channel in sensor_set],
        "optimizer_studies": 1 if phase == "hpo" else 0,
        "hpo_proposals": proposals,
        "executions_per_proposal": executions,
        "execution_seeds": seeds,
        "report_seeds": list(REPORT_SEEDS) if phase == "report" else [],
        "anchor_configuration_id": anchor_configuration_id,
        "shared_generation_root": SHARED_GENERATION_ROOT,
        "manifest_path": manifest_path,
        "results_path": results_path,
        "execution_block_id": EXECUTION_BLOCK_ID,
        "hardware_rule": (
            "each F25 job remains on one locally capacity-qualified host; "
            "distinct jobs may use different qualified GPU models/numeric stacks"
        ),
    }


def _tier_jobs(experiment_id: str, tier: TierPlan) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for arm in tier.arms:
        for sensor_set in tier.sensor_sets:
            if tier.regime == "unfrozen":
                jobs.append(
                    _job(
                        experiment_id=experiment_id,
                        tier=tier,
                        phase="hpo",
                        arm_id=arm.arm_id,
                        sensor_set=sensor_set,
                        anchor_configuration_id=None,
                    )
                )
                report_anchor = configuration_id(arm.arm_id, sensor_set)
            else:
                report_anchor = _anchor_configuration(arm.arm_id)
            jobs.append(
                _job(
                    experiment_id=experiment_id,
                    tier=tier,
                    phase="report",
                    arm_id=arm.arm_id,
                    sensor_set=sensor_set,
                    anchor_configuration_id=report_anchor,
                )
            )
    return jobs


def build_training_plan(experiment_id: str) -> dict[str, Any]:
    experiment = _experiment(experiment_id)
    contract = build_contract()
    if experiment_id == "F25-R":
        tiers = (F25_R_UNFROZEN,)
    else:
        tiers = (
            F25_X_FROZEN_SINGLES,
            F25_X_UNFROZEN_SINGLES,
            F25_X_FROZEN_PAIRS,
        )
    jobs = [job for tier in tiers for job in _tier_jobs(experiment_id, tier)]
    body: dict[str, Any] = {
        "schema": TRAINING_PLAN_SCHEMA,
        "experiment_id": experiment_id,
        "f25_contract_sha256": contract["contract_sha256"],
        "shared_generation_root": SHARED_GENERATION_ROOT,
        "manifest_root": experiment.manifest_root,
        "cache_root": experiment.cache_root,
        "results_root": experiment.results_root,
        "bundle_name": experiment.bundle_name,
        "execution_block_id": EXECUTION_BLOCK_ID,
        "tier_order": [tier.tier_id for tier in tiers],
        "jobs": jobs,
    }
    body["plan_sha256"] = canonical_json_sha256(body)
    validate_training_plan(body)
    return body


def validate_training_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != TRAINING_PLAN_SCHEMA:
        raise F25TrainingContractError("unsupported F25 training plan schema")
    digest = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if digest != canonical_json_sha256(body):
        raise F25TrainingContractError("F25 training plan digest mismatch")
    experiment_id = plan.get("experiment_id")
    _experiment(str(experiment_id))
    if plan.get("f25_contract_sha256") != build_contract()["contract_sha256"]:
        raise F25TrainingContractError("F25 authority-contract binding drifted")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise F25TrainingContractError("F25 training plan has no jobs")
    job_ids = [job.get("job_id") for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise F25TrainingContractError("F25 training job identifiers collide")
    permitted_channels = set(
        F25_R_CHANNELS if experiment_id == "F25-R" else F25_X_CHANNELS
    )
    for job in jobs:
        if set(job) != {
            "schema",
            "job_id",
            "experiment_id",
            "tier_id",
            "regime",
            "phase",
            "configuration_id",
            "arm_id",
            "representation",
            "pooling",
            "sensor_set",
            "sensor_indices",
            "optimizer_studies",
            "hpo_proposals",
            "executions_per_proposal",
            "execution_seeds",
            "report_seeds",
            "anchor_configuration_id",
            "shared_generation_root",
            "manifest_path",
            "results_path",
            "execution_block_id",
            "hardware_rule",
        }:
            raise F25TrainingContractError("F25 job fields drifted")
        if (
            job["schema"] != TRAINING_JOB_SCHEMA
            or job["experiment_id"] != experiment_id
            or job["execution_block_id"] != EXECUTION_BLOCK_ID
            or job["shared_generation_root"] != SHARED_GENERATION_ROOT
            or job["configuration_id"]
            != configuration_id(job["arm_id"], tuple(job["sensor_set"]))
            or job["sensor_indices"]
            != [CHANNELS.index(channel) for channel in job["sensor_set"]]
        ):
            raise F25TrainingContractError("F25 job identity is inconsistent")
        if not set(job["sensor_set"]) <= permitted_channels:
            raise F25TrainingContractError(
                "F25 learning job contains an ineligible diagnostic proxy"
            )
        if job["phase"] == "hpo":
            if (
                job["regime"] != "unfrozen"
                or job["optimizer_studies"] != 1
                or job["hpo_proposals"] != 100
                or job["executions_per_proposal"] != 5
                or tuple(job["execution_seeds"]) != HPO_EXECUTION_SEEDS
                or job["report_seeds"]
            ):
                raise F25TrainingContractError("F25 HPO semantics drifted")
        elif job["phase"] == "report":
            if (
                job["optimizer_studies"] != 0
                or job["hpo_proposals"] != 0
                or job["executions_per_proposal"] != 0
                or tuple(job["report_seeds"]) != REPORT_SEEDS
                or job["anchor_configuration_id"] is None
            ):
                raise F25TrainingContractError("F25 report semantics drifted")
        else:
            raise F25TrainingContractError("unknown F25 job phase")
    counts = {
        (job["tier_id"], job["phase"]): sum(
            candidate["tier_id"] == job["tier_id"]
            and candidate["phase"] == job["phase"]
            for candidate in jobs
        )
        for job in jobs
    }
    if experiment_id == "F25-R":
        expected = {
            (F25_R_UNFROZEN.tier_id, "hpo"): 4,
            (F25_R_UNFROZEN.tier_id, "report"): 4,
        }
    else:
        expected = {
            (F25_X_FROZEN_SINGLES.tier_id, "report"): 18,
            (F25_X_UNFROZEN_SINGLES.tier_id, "hpo"): 18,
            (F25_X_UNFROZEN_SINGLES.tier_id, "report"): 18,
            (F25_X_FROZEN_PAIRS.tier_id, "report"): 45,
        }
    if counts != expected:
        raise F25TrainingContractError(
            f"F25 tier/job inventory drifted: {counts!r} != {expected!r}"
        )


def job_by_id(plan: dict[str, Any], job_id: str) -> dict[str, Any]:
    validate_training_plan(plan)
    matches = [job for job in plan["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise F25TrainingContractError(f"unknown or duplicate F25 job: {job_id}")
    return dict(matches[0])


# Import-time validation is intentionally cheap and torch-free.
build_training_plan("F25-R")
build_training_plan("F25-X")
