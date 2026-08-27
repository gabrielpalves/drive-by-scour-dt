"""Executable four-block Paper-1 generation-contract acceptance test.

This check is intentionally independent of MATLAB execution and generated
artifacts.  It cross-checks the Python contract against the reviewed MATLAB
configuration sources and reconstructs the semantic StateUID inventories from
the published grammar.  A stored ``matched_state_count`` therefore cannot make
the check pass by itself.

Run: ``python check_paper1_campaign_contract.py``
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from core.campaign_contract import (
    CAMPAIGN_STAGE_CONTRACTS,
    EXPECTED_CHANNEL_SCHEMA_ID,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_PROTOCOL_SCHEMA_TAG,
    EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    EXPECTED_RAIL_END_CLEARANCE_M,
    STAGE_ORDER,
    campaign_stage_contract,
    generation_config_expectations,
)
from core.dataset import CACHE_SCHEMA_TAG


ROOT = Path(__file__).resolve().parent
MATLAB = ROOT / "scour_MATLAB"
TTBI = MATLAB / "+ttbi"

EXPECTED_STAGES = ("F40-S", "F40-M", "L99-S", "L99-M")
EXPECTED_COUNTS = {
    "F40-S": 305,
    "F40-M": 425,
    "L99-S": 475,
    "L99-M": 475,
}
EXPECTED_DATASETS = {
    "F40-S": "F40-S_L40_st305",
    "F40-M": "F40-M_L40_st425",
    "L99-S": "L99-S_L99.6_st475",
    "L99-M": "L99-M_L99.6_st475",
}
EXPECTED_DRIVER_COUNTS = {
    "F40-S": [0, 5, 60, 5, 0],
    "F40-M": [250, 50, 5, 5, 50],
    "L99-S": [250, 50, 5, 5, 50],
    "L99-M": [250, 50, 5, 5, 50],
}
EXPECTED_FAMILY_COUNTS = {
    "F40-S": {
        "target_healthy": 5,
        "scour_only": 300,
        "bearing_only": 0,
        "nuisance_only": 0,
        "joint": 0,
    },
    "F40-M": {
        "target_healthy": 50,
        "scour_only": 25,
        "bearing_only": 50,
        "nuisance_only": 50,
        "joint": 250,
    },
    "L99-S": {
        "target_healthy": 50,
        "scour_only": 75,
        "bearing_only": 50,
        "nuisance_only": 50,
        "joint": 250,
    },
    "L99-M": {
        "target_healthy": 50,
        "scour_only": 75,
        "bearing_only": 50,
        "nuisance_only": 50,
        "joint": 250,
    },
}


class ContractError(AssertionError):
    """One cross-language campaign invariant differs from the reviewed plan."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _require_once(source: str, token: str, owner: str) -> None:
    count = source.count(token)
    if count != 1:
        raise ContractError(
            f"{owner}: expected {token!r} exactly once, found {count}"
        )


def semantic_uid(
    *,
    length_m: float,
    spans: int,
    scour_supports: tuple[int, ...],
    family: str,
    target: int,
    level: int,
    replica: int,
) -> str:
    """Independent Python rendering of ``ttbi.state_uid`` v2."""

    support_text = "".join(f"{support:02d}" for support in scour_supports)
    return (
        f"ttbi-state-v2|Lmm={round(1000 * length_m):06d}|"
        f"spans={spans}|scour={support_text}|family={family}|"
        f"target={target:02d}|level={level:04d}|rep={replica:03d}"
    )


def _uid(
    stage: str,
    family: str,
    target: int,
    level: int,
    replica: int,
) -> str:
    geometry = campaign_stage_contract(stage)["geometry"]
    return semantic_uid(
        length_m=float(geometry["L_bridge_m"]),
        spans=int(geometry["num_spans"]),
        scour_supports=tuple(geometry["scour_supports"]),
        family=family,
        target=target,
        level=level,
        replica=replica,
    )


def semantic_inventory(stage: str) -> tuple[tuple[str, str], ...]:
    """Reconstruct one stage's exact ``(family, StateUID)`` inventory."""

    if stage == "F40-S":
        rows: list[tuple[str, str]] = []
        for severity_percent in range(61):
            for replica in range(1, 6):
                if severity_percent == 0:
                    family, target, level = "target_healthy", 0, 0
                else:
                    family, target, level = "scour_only", 2, severity_percent
                rows.append((
                    family,
                    _uid(stage, family, target, level, replica),
                ))
        return tuple(rows)

    geometry = campaign_stage_contract(stage)["geometry"]
    rows = [
        ("target_healthy", _uid(stage, "target_healthy", 0, 0, replica))
        for replica in range(1, 51)
    ]
    for support in geometry["scour_supports"]:
        for replica in range(1, 6):
            for severity_percent in (12, 24, 36, 48, 60):
                rows.append((
                    "scour_only",
                    _uid(
                        stage,
                        "scour_only",
                        int(support),
                        severity_percent,
                        replica,
                    ),
                ))
    for abutment in (1, 2):
        for replica in range(1, 6):
            for level in range(1, 6):
                rows.append((
                    "bearing_only",
                    _uid(stage, "bearing_only", abutment, level, replica),
                ))
    rows.extend(
        ("nuisance_only", _uid(stage, "nuisance_only", 0, 0, replica))
        for replica in range(1, 51)
    )
    rows.extend(
        ("joint", _uid(stage, "joint", 0, joint, 1))
        for joint in range(1, 251)
    )
    return tuple(rows)


def validate_python_contract() -> None:
    if STAGE_ORDER != EXPECTED_STAGES:
        raise ContractError(f"stage order drifted: {STAGE_ORDER!r}")
    if tuple(CAMPAIGN_STAGE_CONTRACTS) != EXPECTED_STAGES:
        raise ContractError("campaign mapping is missing or reorders a block")
    if (
        EXPECTED_GEN_SCHEMA,
        EXPECTED_GENERATION_BEHAVIOR_VERSION,
        EXPECTED_CHANNEL_SCHEMA_ID,
        EXPECTED_PROTOCOL_SCHEMA_TAG,
        CACHE_SCHEMA_TAG,
    ) != (
        "audit-2026-08-09-r12",
        "generation-rules-v8",
        "physical8_v1",
        "gs10a20260809r12",
        "_gs9",
    ):
        raise ContractError("R12/v8/physical8/cache/protocol identity drifted")
    if EXPECTED_RAIL_END_CLEARANCE_M != 6.0 or (
        EXPECTED_RAIL_END_CLEARANCE_DECISION_ID
        != "paper1-rail-domain-clearance-c06-v1"
    ):
        raise ContractError("reviewed 6 m rail-domain decision drifted")

    for stage in EXPECTED_STAGES:
        contract = campaign_stage_contract(stage)
        sampling = contract["sampling"]
        scenario = contract["scenario"]
        paired = contract["paired_design"]
        generation = generation_config_expectations(stage)
        if contract["dataset"] != EXPECTED_DATASETS[stage]:
            raise ContractError(f"{stage}: dataset name/count drifted")
        if sampling["n_states"] != EXPECTED_COUNTS[stage]:
            raise ContractError(f"{stage}: state count drifted")
        if sampling["driver_count_tuple"] != EXPECTED_DRIVER_COUNTS[stage]:
            raise ContractError(f"{stage}: driver count tuple drifted")
        if sampling["family_counts"] != EXPECTED_FAMILY_COUNTS[stage]:
            raise ContractError(f"{stage}: family count inventory drifted")
        expected_design = (
            "dense-scour-61x5-v1"
            if stage == "F40-S"
            else "five-family-multidamage-v2"
        )
        if sampling["state_design_kind"] != expected_design:
            raise ContractError(f"{stage}: state_design_kind drifted")
        if any((
            scenario["use_track_eov"],
            scenario["use_oor_eov"],
            scenario["oor_flats_enabled"],
            generation["use_track_eov"],
            generation["use_oor_eov"],
            generation["oor_flats_enabled"],
        )):
            raise ContractError(f"{stage}: track/OOR entered production")
        if (
            scenario["profile_mode"] != "fixed"
            or generation["profile_mode"] != "fixed"
            or generation["profile_draw"] != "fixed_shared"
            or generation["profile_fra_classes"] != 4
            or generation["profile_fixed_phase_seed"] != 20260728
            or generation["profile_jitter_sd_mm"] != 0.0
        ):
            raise ContractError(
                f"{stage}: rail profile is not the shared fixed-phase FRA-4 "
                "realization with registered seed and zero passage jitter"
            )
        if (
            scenario["rail_end_clearance_m"]
            != EXPECTED_RAIL_END_CLEARANCE_M
            or scenario["rail_end_clearance_decision_id"]
            != EXPECTED_RAIL_END_CLEARANCE_DECISION_ID
            or generation["rail_end_clearance_m"]
            != EXPECTED_RAIL_END_CLEARANCE_M
            or generation["rail_end_clearance_decision_id"]
            != EXPECTED_RAIL_END_CLEARANCE_DECISION_ID
        ):
            raise ContractError(f"{stage}: explicit 6 m decision binding drifted")
        expected_match = 30 if stage.startswith("F40") else 475
        if paired["matched_state_count"] != expected_match:
            raise ContractError(f"{stage}: declared pairing count drifted")


def validate_semantic_pairing() -> None:
    inventories = {
        stage: semantic_inventory(stage) for stage in EXPECTED_STAGES
    }
    for stage, rows in inventories.items():
        uids = [uid for _, uid in rows]
        if len(rows) != EXPECTED_COUNTS[stage] or len(set(uids)) != len(uids):
            raise ContractError(f"{stage}: UID inventory count/collision")
        observed_families = Counter(family for family, _ in rows)
        expected_nonzero = Counter({
            family: count
            for family, count in EXPECTED_FAMILY_COUNTS[stage].items()
            if count
        })
        if observed_families != expected_nonzero:
            raise ContractError(
                f"{stage}: reconstructed families differ: {observed_families}"
            )

    f40s = {uid for _, uid in inventories["F40-S"]}
    f40m = {uid for _, uid in inventories["F40-M"]}
    expected_f40 = {
        _uid("F40-S", "target_healthy", 0, 0, replica)
        for replica in range(1, 6)
    }
    expected_f40.update(
        _uid("F40-S", "scour_only", 2, severity, replica)
        for severity in (12, 24, 36, 48, 60)
        for replica in range(1, 6)
    )
    if f40s & f40m != expected_f40 or len(expected_f40) != 30:
        raise ContractError(
            "F40-S/F40-M do not share exactly five healthy replicas plus "
            "five replicas at 12/24/36/48/60 percent"
        )

    l99s = {uid for _, uid in inventories["L99-S"]}
    l99m = {uid for _, uid in inventories["L99-M"]}
    if l99s != l99m or len(l99s) != 475:
        raise ContractError("L99-S/L99-M are not a complete 475-state pair")


def validate_matlab_sources(
    campaign_setup: str,
    state_design: str,
    state_uid_source: str,
    generation_identity: str,
    case_info: str,
    profile_config: str,
    a04_options: str,
    execution_context: str,
    execution_state: str,
    model_geometry: str,
    vv_protocol: str,
    clearance_study: str,
) -> None:
    case_names = tuple(re.findall(r"(?m)^\s*case '([^']+)'\s*$", campaign_setup))
    if case_names != EXPECTED_STAGES:
        raise ContractError(f"MATLAB stage switch drifted: {case_names!r}")
    for token in (
        "use_track_eov = false;",
        "use_oor_eov   = false;",
        "expected_counts = [0 5 60 5 0];",
        "expected_counts = [250 50 5 5 50];",
        "state_design_kind='dense-scour-61x5-v1';",
        "state_design_kind='five-family-multidamage-v2';",
        "Dano  = (0:60)/100;",
        "config.state_design_kind = state_design_kind;",
        "profile_draw         = 'fixed_shared';",
        "profile_jitter_sd_mm = 0;",
        "profile_fra_classes  = 4;",
        "profile_fixed_phase_seed = 20260728;",
        "rail_end_clearance_m = 6;",
        "'paper1-rail-domain-clearance-c06-v1';",
        "config.rail_end_clearance_m = rail_end_clearance_m;",
        "config.rail_end_clearance_decision_id = rail_end_clearance_decision_id;",
    ):
        if token not in campaign_setup:
            raise ContractError(f"campaign_setup missing {token!r}")
    uncommented_setup = re.sub(r"%.*$", "", campaign_setup, flags=re.MULTILINE)
    if uncommented_setup.count("profile_mode='fixed';") != 4:
        raise ContractError(
            "all four production stages must select profile_mode='fixed'"
        )
    if "profile_mode='psd_fra';" in uncommented_setup:
        raise ContractError("a production stage enabled state-random phases")
    for forbidden in (
        "use_track_eov = true;",
        "use_oor_eov = true;",
        "use_oor_eov   = true;",
        "oor_flats_enabled = true;",
    ):
        if forbidden in uncommented_setup:
            raise ContractError(f"production MATLAB enables {forbidden!r}")

    for token in (
        "state_identity_version = 'semantic-state-v2';",
        "joint_lhs_design = 'not-applicable-dense-scour';",
        "joint_lhs_design = 'master-scour-plus-two-bearing-v2';",
        "~isequal(reshape(Dano, 1, []), (0:60)/100)",
        "level_codes_s = round(100*levels_s);",
        "severity_pct_ = 0:60",
        "design.state_design_kind = state_design_kind;",
    ):
        if token not in state_design:
            raise ContractError(f"build_state_design missing {token!r}")
    for token in (
        "ttbi-state-v2|Lmm=%06d|spans=%d|scour=%s|",
        "family=%s|target=%02d|level=%04d|rep=%03d",
    ):
        _require_once(state_uid_source, token, "state_uid")
    for token in (
        "gen_schema = 'audit-2026-08-09-r12';",
        "generation_behavior_version = 'generation-rules-v8';",
        "channel_schema_id = 'physical8_v1';",
        "'state_design_kind', state.state_design_kind, ...",
    ):
        if token not in generation_identity:
            raise ContractError(f"build_generation_identity missing {token!r}")
    if generation_identity.count(
        "'state_design_kind', state.state_design_kind, ..."
    ) != 2:
        raise ContractError(
            "state_design_kind must bind both generation config and identity"
        )
    _require_once(
        case_info,
        "'state_design_kind', identity.state_design_kind, ...",
        "build_case_info",
    )
    for source_name, source, tokens in (
        (
            "build_profile_config",
            profile_config,
            (
                "expected_clearance_m = 6;",
                "expected_decision_id = 'paper1-rail-domain-clearance-c06-v1';",
                "'rail_end_clearance_m', config.rail_end_clearance_m, ...",
                "'rail_end_clearance_decision_id', ...",
            ),
        ),
        (
            "A04_Options",
            a04_options,
            (
                "Calc.Profile.rail_end_clearance_m = Profile_cfg.rail_end_clearance_m;",
                "Calc.Profile.rail_end_clearance_decision_id = char(decision_id);",
                "A04_Options:MainCampaignRailEndClearance",
            ),
        ),
        (
            "build_execution_context",
            execution_context,
            (
                "'rail_end_clearance_m', campaign.rail_end_clearance_m, ...",
                "campaign.rail_end_clearance_decision_id, ...",
            ),
        ),
        (
            "execute_generation_state",
            execution_state,
            (
                "'rail_end_clearance_requested_m', ...",
                "'rail_end_clearance_realized_m'};",
                "ttbi:ProductionRailEndClearanceDecisionLost",
            ),
        ),
        (
            "B43_ModelGeometry",
            model_geometry,
            (
                "noncampaign",
                "legacy_num_add_sleepers = 10;",
                "Calc.Profile.rail_end_clearance_requested_m = requested_clearance_m;",
            ),
        ),
        (
            "numerical_vv_protocol_definition",
            vv_protocol,
            (
                "C.production_decision_id = 'paper1-rail-domain-clearance-c06-v1';",
                "C.reviewed_production_clearance_m = 6;",
            ),
        ),
        (
            "rail_domain_clearance_study",
            clearance_study,
            (
                "'production_decision_id',C.production_decision_id, ...",
                "verdict.reviewed_production_decision_confirmed = logical( ...",
            ),
        ),
    ):
        for token in tokens:
            if token not in source:
                raise ContractError(f"{source_name} missing {token!r}")
    for source_name, source in (
        ("build_generation_identity", generation_identity),
        ("build_case_info", case_info),
    ):
        for token in (
            "'rail_end_clearance_m', campaign.rail_end_clearance_m, ...",
            "campaign.rail_end_clearance_decision_id, ...",
        ):
            if token not in source:
                raise ContractError(f"{source_name} missing {token!r}")


def _must_reject(name: str, **sources: str) -> None:
    baseline = {
        "campaign_setup": _read(TTBI / "campaign_setup.m"),
        "state_design": _read(TTBI / "build_state_design.m"),
        "state_uid_source": _read(TTBI / "state_uid.m"),
        "generation_identity": _read(TTBI / "build_generation_identity.m"),
        "case_info": _read(TTBI / "build_case_info.m"),
        "profile_config": _read(TTBI / "build_profile_config.m"),
        "a04_options": _read(MATLAB / "A04_Options.m"),
        "execution_context": _read(TTBI / "build_execution_context.m"),
        "execution_state": _read(TTBI / "execute_generation_state.m"),
        "model_geometry": _read(MATLAB / "B43_ModelGeometry.m"),
        "vv_protocol": _read(MATLAB / "numerical_vv_protocol_definition.m"),
        "clearance_study": _read(MATLAB / "rail_domain_clearance_study.m"),
    }
    baseline.update(sources)
    try:
        validate_matlab_sources(**baseline)
    except ContractError:
        print(f"  [PASS] mutation rejected: {name}")
        return
    raise AssertionError(f"mutation escaped campaign guard: {name}")


def main() -> None:
    validate_python_contract()
    print("  [PASS] exact four-block Python contract")
    validate_semantic_pairing()
    print("  [PASS] exact 30-state F40 and complete 475-state L99 pairing")
    baseline = {
        "campaign_setup": _read(TTBI / "campaign_setup.m"),
        "state_design": _read(TTBI / "build_state_design.m"),
        "state_uid_source": _read(TTBI / "state_uid.m"),
        "generation_identity": _read(TTBI / "build_generation_identity.m"),
        "case_info": _read(TTBI / "build_case_info.m"),
        "profile_config": _read(TTBI / "build_profile_config.m"),
        "a04_options": _read(MATLAB / "A04_Options.m"),
        "execution_context": _read(TTBI / "build_execution_context.m"),
        "execution_state": _read(TTBI / "execute_generation_state.m"),
        "model_geometry": _read(MATLAB / "B43_ModelGeometry.m"),
        "vv_protocol": _read(MATLAB / "numerical_vv_protocol_definition.m"),
        "clearance_study": _read(MATLAB / "rail_domain_clearance_study.m"),
    }
    validate_matlab_sources(**baseline)
    print("  [PASS] MATLAB R12/v8/physical8/state-design bindings")

    _must_reject(
        "F40-S count changed",
        campaign_setup=baseline["campaign_setup"].replace(
            "expected_counts = [0 5 60 5 0];",
            "expected_counts = [0 5 59 5 0];",
            1,
        ),
    )
    _must_reject(
        "production track EOV enabled",
        campaign_setup=baseline["campaign_setup"].replace(
            "use_track_eov = false;", "use_track_eov = true;", 1
        ),
    )
    _must_reject(
        "one production stage enabled state-random profile phases",
        campaign_setup=baseline["campaign_setup"].replace(
            "profile_mode='fixed';", "profile_mode='psd_fra';", 1
        ),
    )
    _must_reject(
        "shared fixed profile phase seed changed",
        campaign_setup=baseline["campaign_setup"].replace(
            "profile_fixed_phase_seed = 20260728;",
            "profile_fixed_phase_seed = 20260729;",
            1,
        ),
    )
    _must_reject(
        "physical severity replaced by ordinal level",
        state_design=baseline["state_design"].replace(
            "level_codes_s = round(100*levels_s);",
            "level_codes_s = (1:n_anchor_levels)';",
            1,
        ),
    )
    _must_reject(
        "semantic UID reverted to v1",
        state_uid_source=baseline["state_uid_source"].replace(
            "ttbi-state-v2", "ttbi-state-v1", 1
        ),
    )
    _must_reject(
        "state design omitted from fingerprint",
        generation_identity=baseline["generation_identity"].replace(
            "'state_design_kind', state.state_design_kind, ...",
            "'unbound_state_design', state.state_design_kind, ...",
            1,
        ),
    )
    _must_reject(
        "main-campaign clearance changed from 6 m",
        campaign_setup=baseline["campaign_setup"].replace(
            "rail_end_clearance_m = 6;",
            "rail_end_clearance_m = 15;",
            1,
        ),
    )
    _must_reject(
        "clearance decision identity changed",
        campaign_setup=baseline["campaign_setup"].replace(
            "paper1-rail-domain-clearance-c06-v1",
            "paper1-rail-domain-clearance-c15-v1",
            1,
        ),
    )
    _must_reject(
        "profile path stopped carrying explicit clearance",
        profile_config=baseline["profile_config"].replace(
            "'rail_end_clearance_m', config.rail_end_clearance_m, ...",
            "'unbound_clearance_m', config.rail_end_clearance_m, ...",
            1,
        ),
    )
    _must_reject(
        "clearance omitted from generation fingerprint",
        generation_identity=baseline["generation_identity"].replace(
            "'rail_end_clearance_m', campaign.rail_end_clearance_m, ...",
            "'unbound_clearance_m', campaign.rail_end_clearance_m, ...",
            1,
        ),
    )
    print("PAPER1 CAMPAIGN CONTRACT: ALL PASS")


if __name__ == "__main__":
    main()
