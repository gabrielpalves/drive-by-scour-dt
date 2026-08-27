"""Behaviour and mutation checks for the optional Paper-1 challengers.

Run: ``python check_paper1_challenger_contract.py``
"""

from __future__ import annotations

from copy import deepcopy
import itertools

from core.challenger_policy import (
    challenger_search_policy,
    challenger_search_policy_sha256,
)
from core.paper1_challenger_contract import (
    CHALLENGER_ARMS,
    CHALLENGER_GRID_SHA256_FIELD,
    CHALLENGER_INPUT_CHANNEL_COUNT,
    CHALLENGER_INPUT_SELECTOR,
    CHALLENGER_MODEL_FAMILIES,
    CHALLENGER_PHASE_ORDER,
    CHALLENGER_REPORTING_ROLE,
    CHALLENGER_REPRESENTATIONS,
    CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD,
    CHALLENGER_TRAINING_PROTOCOL_INTENT,
    Paper1ChallengerContractError,
    all_challenger_architectures,
    challenger_development_adjudication_jobs,
    challenger_hpo_jobs,
    challenger_post_freeze_outer_report_jobs,
    complete_challenger_job_grid,
    validate_challenger_contract,
    validate_challenger_job,
)
from core.paper1_selection import SELECTION_ARTIFACT_SCHEMA
from core.paper1_training_contract import (
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
    canonical_json_sha256,
    complete_job_grid,
)
from training.trainer import TRAIN_PROTOCOL


def rejects(label: str, operation) -> None:
    try:
        operation()
    except Paper1ChallengerContractError:
        return
    raise AssertionError(f"mutation was accepted: {label}")


def resign_contract(value: dict) -> dict:
    result = deepcopy(value)
    result.pop(CHALLENGER_GRID_SHA256_FIELD, None)
    result[CHALLENGER_GRID_SHA256_FIELD] = canonical_json_sha256(result)
    return result


def resign_job(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("job_id", None)
    result["job_id"] = canonical_json_sha256(result)[:24]
    return result


def main() -> None:
    primary_before = complete_job_grid()
    challenger = complete_challenger_job_grid()

    assert validate_challenger_contract(challenger) == challenger
    assert challenger["enabled_by_default"] is False
    assert challenger["dispatch_authorized"] is False
    assert challenger["reporting_role"] == CHALLENGER_REPORTING_ROLE
    assert challenger["primary_selection_eligible"] is False
    assert challenger["primary_artifact_mutation_permitted"] is False
    assert challenger["primary_contract_unchanged_by_activation"] is True
    assert challenger["primary_contract"]["complete_grid_sha256"] == (
        primary_before["complete_grid_sha256"]
    )
    assert challenger["primary_contract"]["factorial_cell_count"] == 16
    assert len(FACTORIAL_CELLS) == 16
    assert challenger["phase_order"] == list(CHALLENGER_PHASE_ORDER)
    pair_policy = challenger["selected_pair_policy"]
    assert pair_policy["input_selector"] == CHALLENGER_INPUT_SELECTOR
    assert pair_policy["cardinality"] == CHALLENGER_INPUT_CHANNEL_COUNT == 2
    assert pair_policy["source_stage"] == "F40-S"
    assert pair_policy["source_phase"] == "f40s_channel_screen"
    assert pair_policy["source_artifact"] == SELECTION_ARTIFACT_SCHEMA
    assert pair_policy["eligible_sensor_indices"] == list(
        ELIGIBLE_SENSOR_INDICES
    )
    assert pair_policy["excluded_proxy_indices"] == list(
        EXCLUDED_PROXY_INDICES
    ) == [3, 4]
    assert pair_policy["challenger_pair_selection_prohibited"] is True
    assert pair_policy["required_manifest_field"] == (
        CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD
    )
    assert CHALLENGER_PRIMARY_SELECTION_ARTIFACT_FIELD in (
        challenger["required_before_dispatch"]
    )
    assert challenger["challenger_search_policy"] == challenger_search_policy()
    assert challenger["challenger_search_policy_sha256"] == (
        challenger_search_policy_sha256()
    )
    training_intent = challenger["challenger_training_protocol_intent"]
    assert training_intent == CHALLENGER_TRAINING_PROTOCOL_INTENT
    assert training_intent["initialization"] == "from_scratch"
    assert training_intent["upstream_self_supervised_pretraining_used"] is False
    assert training_intent["upstream_training_recipe_reproduction_claimed"] is False
    assert training_intent["optimizer"] == TRAIN_PROTOCOL["optimizer"]
    assert training_intent["scheduler"] == TRAIN_PROTOCOL["scheduler"]
    assert training_intent["required_identity_field"] in (
        challenger["required_before_dispatch"]
    )

    expected_arm_pairs = list(
        itertools.product(
            CHALLENGER_REPRESENTATIONS, CHALLENGER_MODEL_FAMILIES
        )
    )
    observed_arm_pairs = [
        (arm.representation, arm.model_family) for arm in CHALLENGER_ARMS
    ]
    assert observed_arm_pairs == expected_arm_pairs
    assert [arm.arm_id for arm in CHALLENGER_ARMS] == [
        "RAW_MODERN_TCN",
        "RAW_TSLANET",
        "PAA_MODERN_TCN",
        "PAA_TSLANET",
    ]
    architectures = all_challenger_architectures()
    assert len(architectures) == 4
    assert all(
        architecture["method"] in {"RAW", "PAA"}
        and architecture["model_type"] in {"MODERN_TCN", "TSLANET"}
        and not architecture["use_space2vec"]
        and not architecture["use_lstm"]
        and not architecture["use_nhits"]
        for architecture in architectures
    )

    hpo = challenger_hpo_jobs()
    adjudication = challenger_development_adjudication_jobs()
    outer = challenger_post_freeze_outer_report_jobs()
    assert len(hpo) == 4 * 5 == 20
    assert sum(job["trials"] for job in hpo) == 2_000
    assert len(adjudication) == 4 * 5 * 3 * 2 == 120
    assert len(outer) == 4 * 30 == 120
    assert challenger["phase_job_counts"] == {
        "hpo": 20,
        "development_adjudication": 120,
        "post_freeze_outer_report": 120,
    }
    assert challenger["complete_job_count"] == 260

    expected_hpo = {
        (arm.arm_id, seed)
        for arm in CHALLENGER_ARMS
        for seed in HPO_RESTART_SEEDS
    }
    assert {
        (job["pipeline"], job["hpo_restart_seed"]) for job in hpo
    } == expected_hpo
    assert all(
        job["phase"] == "f40s_secondary_challenger_hpo"
        and job["input_selector"] == CHALLENGER_INPUT_SELECTOR
        and job["input_channel_count"] == CHALLENGER_INPUT_CHANNEL_COUNT
        and job["requires_primary_selection_artifact"] is True
        and job["primary_pair_selection_eligible"] is False
        and job["trials"] == HPO_TRIALS_PER_STUDY == 100
        and job["candidate_restart_seed"] is None
        and job["initialization_seed"] is None
        and job["sealed_outer_test_access_permitted"] is False
        and job["requires_challenger_freeze_artifact"] is False
        for job in hpo
    )

    expected_adjudication = {
        (arm.arm_id, restart, repeat, fold, initialization)
        for arm in CHALLENGER_ARMS
        for restart in HPO_RESTART_SEEDS
        for repeat in range(DEVELOPMENT_N_REPEATS)
        for fold in range(DEVELOPMENT_N_SPLITS)
        for initialization in DEVELOPMENT_INIT_SEEDS
    }
    assert {
        (
            job["pipeline"],
            job["candidate_restart_seed"],
            job["repeat_index"],
            job["fold_index"],
            job["initialization_seed"],
        )
        for job in adjudication
    } == expected_adjudication
    assert all(
        job["development_partition_seed"] == DEVELOPMENT_PARTITION_SEED
        and job["repeat_index"] == 0
        and job["hpo_restart_seed"] is None
        and job["trials"] is None
        and job["sealed_outer_test_access_permitted"] is False
        and job["requires_challenger_freeze_artifact"] is False
        for job in adjudication
    )

    expected_outer = {
        (arm.arm_id, initialization)
        for arm in CHALLENGER_ARMS
        for initialization in POST_FREEZE_STABILITY_SEEDS
    }
    assert {
        (job["pipeline"], job["initialization_seed"]) for job in outer
    } == expected_outer
    assert all(
        job["hpo_restart_seed"] is None
        and job["candidate_restart_seed"] is None
        and job["fold_index"] is None
        and job["trials"] is None
        and job["sealed_outer_test_access_permitted"] is True
        and job["requires_challenger_freeze_artifact"] is True
        for job in outer
    )

    jobs = [
        job
        for phase_key in CHALLENGER_PHASE_ORDER
        for job in challenger["phases"][phase_key]
    ]
    assert len(jobs) == 260
    assert len({job["job_id"] for job in jobs}) == 260
    assert all(validate_challenger_job(job) == job for job in jobs)
    assert all(
        job["reporting_role"] == CHALLENGER_REPORTING_ROLE
        and job["dispatch_authorized"] is False
        and job["primary_grid_membership"] is False
        and job["primary_selection_eligible"] is False
        and job["primary_artifact_mutation_permitted"] is False
        and job["input_selector"] == CHALLENGER_INPUT_SELECTOR
        and job["input_channel_count"] == 2
        and job["requires_primary_selection_artifact"] is True
        and job["primary_pair_selection_eligible"] is False
        and job["primary_complete_grid_sha256"]
        == primary_before["complete_grid_sha256"]
        and job["challenger_search_policy_sha256"]
        == challenger_search_policy_sha256()
        for job in jobs
    )
    primary_job_ids = {
        job["job_id"]
        for phase_jobs in primary_before["phases"].values()
        for job in phase_jobs
    }
    assert not primary_job_ids.intersection(job["job_id"] for job in jobs)

    unsigned = deepcopy(challenger)
    supplied_sha = unsigned.pop(CHALLENGER_GRID_SHA256_FIELD)
    assert supplied_sha == canonical_json_sha256(unsigned)

    mutant = deepcopy(challenger)
    mutant["enabled_by_default"] = True
    rejects(
        "implicit activation",
        lambda: validate_challenger_contract(resign_contract(mutant)),
    )
    mutant = deepcopy(challenger)
    mutant["primary_selection_eligible"] = True
    rejects(
        "primary selection eligibility",
        lambda: validate_challenger_contract(resign_contract(mutant)),
    )
    mutant = deepcopy(challenger)
    mutant["primary_contract"]["complete_grid_sha256"] = "0" * 64
    rejects(
        "foreign primary grid",
        lambda: validate_challenger_contract(resign_contract(mutant)),
    )
    mutant = deepcopy(challenger)
    mutant["challenger_search_policy_sha256"] = "0" * 64
    rejects(
        "foreign challenger search policy",
        lambda: validate_challenger_contract(resign_contract(mutant)),
    )
    mutant = deepcopy(challenger)
    mutant["phases"]["post_freeze_outer_report"].pop()
    mutant["complete_job_count"] -= 1
    mutant["phase_job_counts"]["post_freeze_outer_report"] -= 1
    rejects(
        "partial outer report inventory",
        lambda: validate_challenger_contract(resign_contract(mutant)),
    )

    mutant_job = deepcopy(hpo[0])
    mutant_job["input_selector"] = [1]
    rejects(
        "anchor channel substituted for authenticated pair",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    mutant_job = deepcopy(hpo[0])
    mutant_job["requires_primary_selection_artifact"] = False
    rejects(
        "challenger HPO without primary pair artefact",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    mutant_job = deepcopy(hpo[0])
    mutant_job["reporting_role"] = "primary"
    rejects(
        "primary reporting role",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    mutant_job = deepcopy(outer[0])
    mutant_job["requires_challenger_freeze_artifact"] = False
    rejects(
        "outer access without challenger freeze",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    rejects(
        "wrong phase lookup",
        lambda: validate_challenger_job(hpo[0], phase_key="development_adjudication"),
    )
    mutant_job = deepcopy(hpo[0])
    mutant_job["primary_grid_membership"] = 0
    rejects(
        "boolean field replaced by equal integer",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    mutant_job = deepcopy(hpo[0])
    mutant_job["trials"] = 100.0
    rejects(
        "integer field replaced by equal float",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )
    mutant_job = deepcopy(adjudication[0])
    mutant_job["fold_index"] = False
    rejects(
        "zero index replaced by equal boolean",
        lambda: validate_challenger_job(resign_job(mutant_job)),
    )

    assert complete_job_grid() == primary_before
    print(
        "PASS: optional Paper-1 challenger contract "
        "(20 HPO studies / 2,000 trials / 240 refits; primary grid unchanged)"
    )


if __name__ == "__main__":
    main()
