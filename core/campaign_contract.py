"""Single Python source of truth for the four Paper-1 campaign blocks.

The MATLAB generator records its complete, hash-input configuration as JSON.
Python validates that record against this module before a protocol hash, cache,
or Optuna study may be created.  The deliberate duplication across languages is
therefore executable and mutation-tested rather than an unchecked prose table.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


EXPECTED_GEN_SCHEMA = "audit-2026-08-09-r12"
EXPECTED_GENERATION_BEHAVIOR_VERSION = "generation-rules-v8"
EXPECTED_PROTOCOL_SCHEMA_TAG = "gs10a20260809r12"
EXPECTED_CHANNEL_SCHEMA_ID = "physical8_v1"
EXPECTED_RAIL_END_CLEARANCE_M = 6.0
EXPECTED_RAIL_END_CLEARANCE_DECISION_ID = (
    "paper1-rail-domain-clearance-c06-v1"
)
CAMPAIGN_CONTRACT_SCHEMA = "ttbi-campaign-stage-contract-v2"
BLOCK_REFERENCE_MANIFEST_FIELDS = frozenset({
    "champion_arch",
    "selected_at_stage",
    "dataset",
    "schema",
    "run_tag",
    "seeds",
    "n_trials",
    "candidate_n_trials",
    "exhaustive_pairs",
    "protocol_core_hash",
    "protocol_core",
    "protocol_hash",
    "execution_runtime",
    "execution_environment_sha256",
    "execution_receipt_sha256",
    "capacity_preflight_receipt_sha256",
    "hyperparameter_manifest_sha256",
    "hyperparameter_policy_sha256",
    "champion_pair",
    "pair_select_metric",
    "per_arch_median_single_dof_mse",
    "frozen_selection_sha256",
})
_LOWER_HEX = frozenset("0123456789abcdef")
STAGE_ORDER = ("F40-S", "F40-M", "L99-S", "L99-M")


# Every scalar/list generator knob that is common to the production campaign.
# The realized state matrices and the environment/source identities are checked
# separately; everything else in A00's fp_cfg must agree with this mapping.
_COMMON_GENERATION_CONFIG = {
    "schema": EXPECTED_GEN_SCHEMA,
    "generation_behavior_version": EXPECTED_GENERATION_BEHAVIOR_VERSION,
    "channel_schema_id": EXPECTED_CHANNEL_SCHEMA_ID,
    "damage_mode": "multi_scour",
    "Npass": 50,
    "state_identity_version": "semantic-state-v2",
    "n_latent_bearing_dims": 2,
    "random_stream_schedule_version": "uid-named-substreams-v2",
    "state_stream_names": [
        "operations",
        "crack",
        "profile-state",
        "track",
        "profile-phase",
    ],
    "passage_stream_names": [
        "profile-passage",
        "oor-passage",
    ],
    "max_parfor_workers": 4,
    "dano_max": 0.60,
    "include_anchors": True,
    "damage_seed": 1,
    "Bearing_Intensity": 0.0,
    "bearing_fixity_max": 0.95,
    "crack_draw": "per_state",
    "crack_p": 0.25,
    "crack_hog_ratio": 4.0,
    "crack_hog_margin": 0.175,
    "crack_frac_range": [0.10, 0.90],
    "crack_int_range": [0.05, 0.30],
    "crack_lc": 0.0,
    "profile_draw": "fixed_shared",
    "profile_jitter_sd_mm": 0.0,
    "profile_int_range": [0.5, 2.0],
    "profile_fra_classes": 4,
    "profile_fixed_phase_seed": 20260728,
    "profile_spectrum_contract": "fra-v2-class4-cycles-per-m-v1",
    "rail_end_clearance_m": EXPECTED_RAIL_END_CLEARANCE_M,
    "rail_end_clearance_decision_id":
        EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    "track_draw": "per_state",
    "track_L_app": 30.0,
    "track_L_after": 30.0,
    "hang_rate_100m": 3.0,
    "hang_group_size": [1, 5],
    "hang_p_transition": 0.6,
    "hang_trans_margin": 15.0,
    "hang_foul_mult": 3.0,
    "ballast_rate_100m": 1.2,
    "ballast_patch_len": [5, 20],
    "ballast_trans_mult": 3.0,
    "ballast_trans_margin": 20.0,
    "ballast_p_wet": 0.5,
    "ballast_eta_k_dry": [1.2, 2.0],
    "ballast_eta_c_dry": [0.4, 0.8],
    "ballast_eta_k_wet": [0.7, 0.9],
    "ballast_eta_c_wet": [1.5, 4.0],
    "pad_p_fail": 0.02,
    "pad_failure_rule": "independent-bernoulli-sleeper-lattice-v1",
    "pad_chi_range": [1.0, 3.5],
    "pad_weibull": [1.8, 2.2],
    "pad_beta_range": [0.8, 1.2],
    "oor_flats_enabled": False,
    "oor_q_bogie": 0.171,
    "oor_p_trailing": 0.40,
    "oor_p_fresh": 0.125,
    "oor_len_fresh": [0.010, 0.035],
    "oor_len_runin": [0.030, 0.060],
    "oor_radius": 0.46,
    "poly_p_wheel": 0.30,
    "poly_orders": [1, 5],
    "poly_amp_lnorm": [-10.0, 0.5],
    "poly_amp_bounds": [1e-5, 1.2e-4],
    "use_signal_noise": False,
    "Desvio": 0.05,
    "use_vehicle_variability": True,
    "use_speed_variability": True,
    "use_temp_variability": True,
    "Nveh": 5,
    "Nprop": 3,
    "temp_min": 3.0,
    "temp_max": 33.0,
    "vel_min": 70.0,
    "vel_max": 90.0,
}


_STAGE_INPUTS = {
    "F40-S": {
        "length": 40.0,
        "spans": 2,
        "scour_supports": (2,),
        "state_design_kind": "dense-scour-61x5-v1",
        "joint_lhs_design": "not-applicable-dense-scour",
        "bearing_mode": "off",
        "crack": False,
        "counts": (0, 5, 60, 5, 0),
        "family_counts": (5, 300, 0, 0, 0),
    },
    "F40-M": {
        "length": 40.0,
        "spans": 2,
        "scour_supports": (2,),
        "state_design_kind": "five-family-multidamage-v2",
        "joint_lhs_design": "master-scour-plus-two-bearing-v2",
        "bearing_mode": "target",
        "crack": True,
        "counts": (250, 50, 5, 5, 50),
        "family_counts": (50, 25, 50, 50, 250),
    },
    "L99-S": {
        "length": 99.6,
        "spans": 4,
        "scour_supports": (2, 3, 4),
        "state_design_kind": "five-family-multidamage-v2",
        "joint_lhs_design": "master-scour-plus-two-bearing-v2",
        "bearing_mode": "off",
        "crack": False,
        "counts": (250, 50, 5, 5, 50),
        "family_counts": (50, 75, 50, 50, 250),
    },
    "L99-M": {
        "length": 99.6,
        "spans": 4,
        "scour_supports": (2, 3, 4),
        "state_design_kind": "five-family-multidamage-v2",
        "joint_lhs_design": "master-scour-plus-two-bearing-v2",
        "bearing_mode": "target",
        "crack": True,
        "counts": (250, 50, 5, 5, 50),
        "family_counts": (50, 75, 50, 50, 250),
    },
}


def _build_stage_contract(stage: str) -> dict[str, Any]:
    spec = _STAGE_INPUTS[stage]
    length = spec["length"]
    spans = spec["spans"]
    scour_supports = spec["scour_supports"]
    bearing_mode = spec["bearing_mode"]
    crack = spec["crack"]
    family_names = (
        "target_healthy",
        "scour_only",
        "bearing_only",
        "nuisance_only",
        "joint",
    )
    family_counts = dict(zip(family_names, spec["family_counts"], strict=True))
    n_states = sum(family_counts.values())
    length_tag = f"{length:g}"
    dataset = f"{stage}_L{length_tag}_st{n_states}"
    return {
        "schema": CAMPAIGN_CONTRACT_SCHEMA,
        "stage": stage,
        "dataset": dataset,
        "geometry": {
            "L_bridge_m": length,
            "num_spans": spans,
            "num_supports": spans + 1,
            "scour_supports": list(scour_supports),
        },
        "scenario": {
            "damage_mode": "multi_scour",
            "bearing_mode": bearing_mode,
            "use_crack_eov": crack,
            "profile_mode": "fixed",
            "rail_end_clearance_m": EXPECTED_RAIL_END_CLEARANCE_M,
            "rail_end_clearance_decision_id":
                EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
            "use_track_eov": False,
            "use_oor_eov": False,
            "oor_flats_enabled": False,
            "use_signal_noise": False,
        },
        "sampling": {
            "n_states": n_states,
            "passages_per_state": 50,
            "scour_dano_max_frac": 0.60,
            "state_design_kind": spec["state_design_kind"],
            "driver_count_tuple": list(spec["counts"]),
            "family_counts": family_counts,
        },
        "paired_design": {
            "inventory_policy": (
                "matched-controlled-subset"
                if stage.startswith("F40")
                else "complete-within-L99-geometry"
            ),
            "state_uid_version": "semantic-state-v2",
            "joint_master_states": spec["counts"][0],
            "matched_state_count": 30 if stage.startswith("F40") else 475,
            "matched_f40_scour_percentages": (
                [0, 12, 24, 36, 48, 60]
                if stage.startswith("F40") else None
            ),
            "bearing_only_is_dormant": bearing_mode != "target",
            "nuisance_only_is_dormant": not crack,
        },
        "learning": {
            "target_supports": list(scour_supports),
            "bearing_targets": (
                ["left", "right"] if bearing_mode == "target" else None
            ),
            "f40_classification_subset_percent": (
                [0, 5, 10, 20] if stage == "F40-S" else None
            ),
        },
    }


CAMPAIGN_STAGE_CONTRACTS = {
    stage: _build_stage_contract(stage) for stage in STAGE_ORDER
}


def campaign_stage_contract(stage: str) -> dict[str, Any]:
    """Return a defensive copy of one validated rung contract."""

    if stage not in CAMPAIGN_STAGE_CONTRACTS:
        raise RuntimeError(
            f"unknown campaign stage {stage!r}; expected one of {STAGE_ORDER!r}"
        )
    return copy.deepcopy(CAMPAIGN_STAGE_CONTRACTS[stage])


def generation_config_expectations(stage: str) -> dict[str, Any]:
    """Expected non-realization fields in A00's hash-input ``fp_cfg``."""

    contract = campaign_stage_contract(stage)
    geometry = contract["geometry"]
    scenario = contract["scenario"]
    sampling = contract["sampling"]
    expected = copy.deepcopy(_COMMON_GENERATION_CONFIG)
    expected.update({
        "STAGE": stage,
        "L_bridge": geometry["L_bridge_m"],
        "num_spans": geometry["num_spans"],
        "scour_supports": geometry["scour_supports"],
        "n_states": sampling["n_states"],
        "state_design_kind": sampling["state_design_kind"],
        "joint_lhs_design": _STAGE_INPUTS[stage]["joint_lhs_design"],
        "n_anchor_levels": sampling["driver_count_tuple"][2],
        "n_healthy_states": sampling["driver_count_tuple"][1],
        "n_anchor_reps": sampling["driver_count_tuple"][3],
        "n_nuisance_states": sampling["driver_count_tuple"][4],
        "bearing_mode": scenario["bearing_mode"],
        "use_crack_eov": scenario["use_crack_eov"],
        "profile_mode": scenario["profile_mode"],
        "use_track_eov": scenario["use_track_eov"],
        "use_oor_eov": scenario["use_oor_eov"],
    })
    return expected


def canonical_campaign_contract_json(contract: dict[str, Any]) -> str:
    """Canonical JSON representation embedded in the protocol descriptor."""

    return json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def campaign_contract_sha256(stage: str) -> str:
    return hashlib.sha256(
        canonical_campaign_contract_json(
            campaign_stage_contract(stage)
        ).encode("ascii")
    ).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _LOWER_HEX for char in value)
    )


def validate_block_reference_manifest(
    manifest: dict[str, Any],
    *,
    expected_anchor_stage: str,
    expected_dataset: str,
    expected_schema: str,
    expected_run_tag: str,
    expected_seeds: list[int] | tuple[int, ...],
    expected_anchor_n_trials: int,
    expected_candidate_n_trials: int,
    expected_protocol_core_hash: str,
    expected_anchor_protocol_hash: str,
    expected_execution_runtime: dict[str, Any],
    expected_execution_receipt_sha256: str,
    expected_hyperparameter_manifest_sha256: str,
    expected_hyperparameter_policy_sha256: str,
    valid_architectures: list[str] | tuple[str, ...],
    valid_dofs: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Validate the exact, content-addressed block-reference contract.

    The function is deliberately pure: callers first authenticate the durable
    execution receipt and HPO manifest, then pass those trusted expectations
    here.  Both campaign execution and offline inference can therefore enforce
    one field/type/lineage contract without importing the large campaign driver.
    """

    if not isinstance(manifest, dict):
        raise RuntimeError("block reference must be one JSON object")
    value = copy.deepcopy(manifest)
    if set(value) != BLOCK_REFERENCE_MANIFEST_FIELDS:
        raise RuntimeError(
            "block-reference fields differ from the exact contract "
            f"(missing={sorted(BLOCK_REFERENCE_MANIFEST_FIELDS - set(value))!r}, "
            f"extra={sorted(set(value) - BLOCK_REFERENCE_MANIFEST_FIELDS)!r})"
        )

    architectures = tuple(valid_architectures)
    dof_domain = tuple(valid_dofs)
    seeds = value["seeds"]
    if (
        not isinstance(expected_anchor_stage, str)
        or not expected_anchor_stage
        or value["selected_at_stage"] != expected_anchor_stage
        or value["dataset"] != expected_dataset
        or not isinstance(value["schema"], str)
        or not value["schema"]
        or value["run_tag"] != expected_run_tag
        or not isinstance(seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int)
               for seed in seeds)
        or seeds != list(expected_seeds)
    ):
        raise RuntimeError(
            "block reference carries the wrong anchor, dataset, schema, "
            "run tag, or registered seed order"
        )

    for key, expected in (
        ("n_trials", expected_anchor_n_trials),
        ("candidate_n_trials", expected_candidate_n_trials),
    ):
        actual = value[key]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        ):
            raise RuntimeError(
                f"block-reference {key} is not the exact registered integer "
                f"budget {expected}"
            )
    if value["exhaustive_pairs"] is not True:
        raise RuntimeError(
            "block reference was not selected by the exhaustive anchor sweep"
        )

    core = value["protocol_core"]
    if not isinstance(core, dict):
        raise RuntimeError("block reference lacks its protocol-core descriptor")
    try:
        core_payload = canonical_campaign_contract_json(core).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError(
            "block-reference protocol core is not finite canonical JSON"
        ) from exc
    actual_core_hash = hashlib.sha256(core_payload).hexdigest()
    schema_tag = (
        core.get("code", {}).get("schema_tag")
        if isinstance(core.get("code"), dict) else None
    )
    if (
        not isinstance(expected_schema, str)
        or not expected_schema
        or value["schema"] != expected_schema
        or schema_tag != expected_schema
        or value["protocol_core_hash"] != expected_protocol_core_hash
        or actual_core_hash != value["protocol_core_hash"]
        or value["protocol_hash"] != expected_anchor_protocol_hash
    ):
        raise RuntimeError(
            "block-reference protocol descriptor/hash/schema lineage is "
            "inconsistent with the authenticated anchor"
        )

    if (
        value["execution_runtime"] != expected_execution_runtime
        or value["execution_environment_sha256"]
        != expected_execution_runtime.get("execution_environment_sha256")
        or value["execution_receipt_sha256"]
        != expected_execution_receipt_sha256
        or value["hyperparameter_manifest_sha256"]
        != expected_hyperparameter_manifest_sha256
        or value["hyperparameter_policy_sha256"]
        != expected_hyperparameter_policy_sha256
    ):
        raise RuntimeError(
            "block-reference runtime, receipt, HPO, or policy lineage differs "
            "from the authenticated anchor inputs"
        )
    for key in (
        "protocol_core_hash",
        "protocol_hash",
        "execution_environment_sha256",
        "execution_receipt_sha256",
        "capacity_preflight_receipt_sha256",
        "hyperparameter_manifest_sha256",
        "hyperparameter_policy_sha256",
        "frozen_selection_sha256",
    ):
        if not _is_lower_sha256(value[key]):
            raise RuntimeError(
                f"block-reference {key} must be lowercase SHA-256"
            )

    architecture = value["champion_arch"]
    if (
        not isinstance(architecture, str)
        or architecture not in architectures
    ):
        raise RuntimeError(
            "block-reference champion architecture is not registered"
        )
    pair = value["champion_pair"]
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(isinstance(dof, bool) or not isinstance(dof, int)
               for dof in pair)
        or pair != sorted(pair)
        or len(set(pair)) != 2
        or any(dof not in dof_domain for dof in pair)
    ):
        raise RuntimeError(
            "block-reference champion_pair must be two sorted, distinct, "
            "strict integer registered DOFs"
        )
    if value["pair_select_metric"] != "inner_val_mse":
        raise RuntimeError(
            "block reference cites another pair-selection metric"
        )

    medians = value["per_arch_median_single_dof_mse"]
    if (
        not isinstance(medians, dict)
        or set(medians) != set(architectures)
        or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or float(metric) < 0.0
            for metric in medians.values()
        )
    ):
        raise RuntimeError(
            "block-reference architecture diagnostics must be one finite, "
            "non-negative median per registered architecture"
        )
    return value


def _validate_definitions() -> None:
    if tuple(CAMPAIGN_STAGE_CONTRACTS) != STAGE_ORDER:
        raise RuntimeError("campaign stage definitions are missing or reordered")
    expected_datasets = {
        "F40-S": "F40-S_L40_st305",
        "F40-M": "F40-M_L40_st425",
        "L99-S": "L99-S_L99.6_st475",
        "L99-M": "L99-M_L99.6_st475",
    }
    actual = {
        stage: contract["dataset"]
        for stage, contract in CAMPAIGN_STAGE_CONTRACTS.items()
    }
    if actual != expected_datasets:
        raise RuntimeError(
            f"campaign dataset/count derivation drifted: {actual!r}"
        )


_validate_definitions()
