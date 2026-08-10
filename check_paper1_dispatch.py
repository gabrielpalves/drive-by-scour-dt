"""Behaviour and mutation checks for the six-bundle dispatch partition."""

from __future__ import annotations

from copy import deepcopy

from core.campaign_contract import STAGE_ORDER
from core.paper1_dispatch import (
    GENERATION_BUNDLE_NAMES,
    TRAINING_BUNDLE_NAMES,
    assigned_training_host,
    generation_manifest,
    six_bundle_manifest_set,
    training_manifests,
    validate_training_manifests,
)
from core.paper1_training_contract import complete_job_grid


def _expect_error(fn) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("dispatch mutation survived")


def main() -> None:
    complete = complete_job_grid()
    manifests = training_manifests()
    all_jobs = [
        job
        for jobs in complete["phases"].values()
        for job in jobs
    ]
    ids_a = set(manifests["labA"]["assigned_job_ids"])
    ids_b = set(manifests["labB"]["assigned_job_ids"])
    assert not ids_a.intersection(ids_b)
    assert ids_a.union(ids_b) == {job["job_id"] for job in all_jobs}
    assert len(ids_a) == 1092
    assert len(ids_b) == 808
    for job in all_jobs:
        host = assigned_training_host(job)
        assert job["job_id"] in manifests[host]["assigned_job_ids"]

    dispatch = six_bundle_manifest_set()
    assert dispatch["bundle_names"] == [
        *(GENERATION_BUNDLE_NAMES[stage] for stage in STAGE_ORDER),
        *(TRAINING_BUNDLE_NAMES[host] for host in ("labA", "labB")),
    ]
    assert len(dispatch["bundle_names"]) == 6
    for stage in STAGE_ORDER:
        item = generation_manifest(stage)
        assert item["stage"] == stage
        assert item["n_states"] in {305, 425, 475}

    missing = deepcopy(manifests)
    missing["labA"]["jobs"].pop()
    _expect_error(lambda: validate_training_manifests(missing))

    duplicate = deepcopy(manifests)
    duplicate["labB"]["jobs"].append(duplicate["labA"]["jobs"][0])
    _expect_error(lambda: validate_training_manifests(duplicate))

    method_confounded = deepcopy(manifests)
    method_confounded["labA"]["jobs"][0]["pipeline"] = (
        "PAA_POS0_LSTM0_MR0"
    )
    _expect_error(lambda: validate_training_manifests(method_confounded))
    print("PASS: six-bundle dispatch manifests and balanced seed partition")


if __name__ == "__main__":
    main()
