"""Deterministic six-bundle dispatch manifests for the Paper-1 campaign.

The four generation manifests each bind one scientific block.  The two
training manifests partition the complete pre-outcome training grid across the
matched Lab-A/Lab-B GPUs by registered seed, never by representation,
architecture, channel set, or outcome.  Consequently every compared pipeline
has the same host/seed allocation and the two manifests are disjoint while
their authenticated union is the complete training contract.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from core.campaign_contract import (
    EXPECTED_CHANNEL_SCHEMA_ID,
    STAGE_ORDER,
    campaign_contract_sha256,
    campaign_stage_contract,
)
from core.paper1_training_contract import (
    HPO_RESTART_SEEDS,
    POST_FREEZE_STABILITY_SEEDS,
    SCREEN_REFIT_SEEDS,
    canonical_json_bytes,
    canonical_json_sha256,
    complete_job_grid,
)


DISPATCH_SET_SCHEMA = "paper1-six-bundle-dispatch-v1"
GENERATION_MANIFEST_SCHEMA = "paper1-generation-bundle-manifest-v1"
TRAINING_MANIFEST_SCHEMA = "paper1-training-machine-manifest-v1"
TRAINING_HOSTS = ("labA", "labB")
GENERATION_BUNDLE_NAMES = {
    "F40-S": "bundle_f40s_generate.zip",
    "F40-M": "bundle_f40m_generate.zip",
    "L99-S": "bundle_l99s_generate.zip",
    "L99-M": "bundle_l99m_generate.zip",
}
TRAINING_BUNDLE_NAMES = {
    "labA": "bundle_train_labA.zip",
    "labB": "bundle_train_labB.zip",
}


def _all_jobs() -> tuple[dict[str, Any], ...]:
    contract = complete_job_grid()
    phases = contract["phases"]
    return tuple(
        deepcopy(job)
        for phase in (
            "hpo",
            "development_adjudication",
            "channel_screen",
            "post_freeze_stability",
            "secondary_frozen_transfer",
        )
        for job in phases[phase]
    )


def _allocation_axis(job: dict[str, Any]) -> tuple[str, int, tuple[int, ...]]:
    """Return the prospective seed axis used for machine allocation."""

    if job["hpo_restart_seed"] is not None:
        return "hpo_restart_seed", job["hpo_restart_seed"], HPO_RESTART_SEEDS
    if job["candidate_restart_seed"] is not None:
        return (
            "candidate_restart_seed",
            job["candidate_restart_seed"],
            HPO_RESTART_SEEDS,
        )
    seed = job["initialization_seed"]
    if seed is None:
        raise RuntimeError(f"training job has no allocation seed: {job!r}")
    inventory = (
        POST_FREEZE_STABILITY_SEEDS
        if job["phase"] == "post_freeze_sealed_test_stability"
        else SCREEN_REFIT_SEEDS
    )
    return "initialization_seed", seed, inventory


def assigned_training_host(job: dict[str, Any]) -> str:
    """Allocate by seed-index parity, identically for every compared cell."""

    _axis, seed, inventory = _allocation_axis(job)
    try:
        position = tuple(inventory).index(seed)
    except ValueError as exc:
        raise RuntimeError(
            f"training job carries an unregistered allocation seed: {seed}"
        ) from exc
    return TRAINING_HOSTS[position % len(TRAINING_HOSTS)]


def generation_manifest(stage: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise RuntimeError(f"unregistered generation bundle stage {stage!r}")
    contract = campaign_stage_contract(stage)
    value = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "bundle_name": GENERATION_BUNDLE_NAMES[stage],
        "stage": stage,
        "dataset": contract["dataset"],
        "n_states": contract["sampling"]["n_states"],
        "passages_per_state": contract["sampling"]["passages_per_state"],
        "channel_schema_id": EXPECTED_CHANNEL_SCHEMA_ID,
        "campaign_contract_sha256": campaign_contract_sha256(stage),
        "campaign_contract": contract,
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    return value


def training_manifests() -> dict[str, dict[str, Any]]:
    contract = complete_job_grid()
    all_jobs = _all_jobs()
    manifests: dict[str, dict[str, Any]] = {}
    for host in TRAINING_HOSTS:
        jobs = [job for job in all_jobs if assigned_training_host(job) == host]
        value = {
            "schema": TRAINING_MANIFEST_SCHEMA,
            "bundle_name": TRAINING_BUNDLE_NAMES[host],
            "machine_role": host,
            "channel_schema_id": EXPECTED_CHANNEL_SCHEMA_ID,
            "complete_grid_sha256": contract["complete_grid_sha256"],
            "complete_job_count": len(all_jobs),
            "allocation_policy": {
                "kind": "registered-seed-index-parity-v1",
                "host_order": list(TRAINING_HOSTS),
                "comparison_fields_excluded": [
                    "stage",
                    "pipeline",
                    "input_selector",
                    "representation",
                    "outcome",
                ],
                "rule": (
                    "first applicable registered HPO/candidate/init seed "
                    "index modulo two"
                ),
            },
            "assigned_job_count": len(jobs),
            "assigned_job_ids": [job["job_id"] for job in jobs],
            "jobs": jobs,
        }
        value["manifest_sha256"] = canonical_json_sha256(value)
        manifests[host] = value
    validate_training_manifests(manifests)
    return manifests


def validate_training_manifests(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(manifests, dict) or tuple(manifests) != TRAINING_HOSTS:
        raise RuntimeError("training manifests must be ordered Lab-A/Lab-B")
    expected_jobs = _all_jobs()
    expected_by_id = {job["job_id"]: job for job in expected_jobs}
    observed: dict[str, str] = {}
    validated: dict[str, dict[str, Any]] = {}
    for host in TRAINING_HOSTS:
        manifest = json.loads(canonical_json_bytes(manifests[host]))
        expected_fields = {
            "schema",
            "bundle_name",
            "machine_role",
            "channel_schema_id",
            "complete_grid_sha256",
            "complete_job_count",
            "allocation_policy",
            "assigned_job_count",
            "assigned_job_ids",
            "jobs",
            "manifest_sha256",
        }
        if set(manifest) != expected_fields:
            raise RuntimeError(f"{host} training manifest fields drifted")
        supplied_sha = manifest.pop("manifest_sha256")
        if supplied_sha != canonical_json_sha256(manifest):
            raise RuntimeError(f"{host} training manifest SHA-256 is invalid")
        manifest["manifest_sha256"] = supplied_sha
        if (
            manifest["schema"] != TRAINING_MANIFEST_SCHEMA
            or manifest["bundle_name"] != TRAINING_BUNDLE_NAMES[host]
            or manifest["machine_role"] != host
            or manifest["channel_schema_id"] != EXPECTED_CHANNEL_SCHEMA_ID
            or manifest["complete_grid_sha256"]
            != complete_job_grid()["complete_grid_sha256"]
            or manifest["complete_job_count"] != len(expected_jobs)
            or manifest["assigned_job_count"] != len(manifest["jobs"])
            or manifest["assigned_job_ids"]
            != [job.get("job_id") for job in manifest["jobs"]]
        ):
            raise RuntimeError(f"{host} training manifest metadata drifted")
        for job in manifest["jobs"]:
            job_id = job.get("job_id")
            if job_id not in expected_by_id or job != expected_by_id[job_id]:
                raise RuntimeError(f"{host} contains a foreign training job")
            if assigned_training_host(job) != host:
                raise RuntimeError(f"{job_id} is allocated to the wrong host")
            if job_id in observed:
                raise RuntimeError(
                    f"training job {job_id} is duplicated on two hosts"
                )
            observed[job_id] = host
        validated[host] = manifest
    if set(observed) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(observed))
        raise RuntimeError(f"training bundle union is incomplete: {missing}")
    return validated


def six_bundle_manifest_set() -> dict[str, Any]:
    generation = {
        stage: generation_manifest(stage) for stage in STAGE_ORDER
    }
    training = training_manifests()
    value = {
        "schema": DISPATCH_SET_SCHEMA,
        "generation": generation,
        "training": training,
        "bundle_names": [
            *(GENERATION_BUNDLE_NAMES[stage] for stage in STAGE_ORDER),
            *(TRAINING_BUNDLE_NAMES[host] for host in TRAINING_HOSTS),
        ],
    }
    value["dispatch_set_sha256"] = canonical_json_sha256(value)
    return value

