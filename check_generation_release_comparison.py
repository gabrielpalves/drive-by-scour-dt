"""Behavioral/adversarial checks for MATLAB environment qualification.

Fixtures are real MATLAB v5 files written inside ``TemporaryDirectory``.
Semantic mutations are followed by rebuilt digest and completion-marker
contracts, so each test reaches the intended guard rather than failing on an
unrelated stale digest.

Run:
    python check_generation_release_comparison.py
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

import compare_generation_releases as comparison
from core.environment import (
    load_environment_lock,
    matlab_environment_descriptor,
)
from make_micro_smoke import (
    main as micro_main,
    qualification_source_sha256,
    render_micro_a00,
    render_toy_driver,
    verify_qualification_source_bytes,
    write_micro_a00,
)


Mutation = Callable[[dict], None]
FIXTURE_K_REF_BEAR = 1.0e8


@dataclass(frozen=True)
class MatlabEnvironment:
    release: str
    descriptor: str
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_mat(path: Path) -> dict:
    """Parse fixture bytes through the comparator's current buffer parser."""
    return comparison._public_mat_bytes(path.read_bytes(), str(path))


def _environment(release: str, *, variant: str = "default") -> MatlabEnvironment:
    lock = load_environment_lock(
        Path(__file__).resolve().parent
        / "environment"
        / "campaign-py313-cu128.json"
    )
    spec = deepcopy(lock["spec"]["matlab_environment"])
    if release != spec["release"] or variant != "default":
        spec["release"] = release
        spec["version"] = (
            f"qualification-fixture-{release}-{variant} "
            "(deliberately distinct executable identity)"
        )
    descriptor = matlab_environment_descriptor(spec)
    return MatlabEnvironment(
        release=release,
        descriptor=descriptor,
        sha256=hashlib.sha256(descriptor.encode("utf-8")).hexdigest(),
    )


def _campaign_environment() -> MatlabEnvironment:
    policy = comparison._current_policy()
    return MatlabEnvironment(
        release=policy.campaign_matlab_release,
        descriptor=policy.campaign_matlab_environment_descriptor,
        sha256=policy.campaign_matlab_environment_sha256,
    )


def _cell(values: list[object]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def _base_data(
    actual: MatlabEnvironment,
    *,
    policy: comparison.CurrentPolicy,
    campaign: MatlabEnvironment,
    stage: str,
    n_passages: int,
    scour_vector: np.ndarray,
    bearing_vector: np.ndarray,
    family: str,
    crack_on: bool,
    state_uid: str,
    state_seed_id: int,
    latent_bearing_fixity: np.ndarray,
    latent_crack_on: bool,
    state_named_stream_seed_id: np.ndarray,
    passage_named_stream_seed_id: np.ndarray,
    qualification_sha: str,
    generation_fingerprint: str,
) -> dict:
    acel_prim = np.empty(n_passages, dtype=object)
    acel_roda = np.empty(n_passages, dtype=object)
    acel_wheelset = np.empty(n_passages, dtype=object)
    pitch = np.empty(n_passages, dtype=object)
    for passage in range(n_passages):
        acel_prim[passage] = (
            np.arange(12.0).reshape(3, 4) + passage * 0.125
        )
        acel_roda[passage] = (
            np.arange(16.0).reshape(4, 4) - passage * 0.25
        )
        acel_wheelset[passage] = (
            np.arange(16.0).reshape(4, 4) + 100.0 + passage * 0.5
        )
        pitch[passage] = (
            np.arange(12.0).reshape(3, 4) * 0.01 + passage * 0.001
        )

    stage_policy = comparison._STAGE_POLICIES[stage]
    bridge_length = float(stage_policy["L_bridge_m"])
    num_supports = int(stage_policy["num_supports"])
    scour_supports = np.asarray(
        stage_policy["scour_supports"], dtype=np.int64
    )
    bridge_samp = int(round(100 * bridge_length))
    dim_space = bridge_samp + 3000
    crop_start = 1001
    crop_end = crop_start - 1 + bridge_samp + 1831
    dim_acel = 4
    # The fixture keeps tiny signals but still reproduces the generator's exact
    # spatial-grid equation DimSpace = round(Velocidade * DimAcel / 10).
    velocity = np.full(n_passages, dim_space * 10.0 / dim_acel)
    if stage_policy["use_track_eov"]:
        track_log = _cell(
            [
                {
                    "ballast_patches": np.array(
                        [[12.5, 16.5, 0.55, 0.60]], dtype=float
                    ),
                    "hanging_groups": np.array([[20.0, 3.0]], dtype=float),
                    "pad_stiff_mult": 0.82,
                    "pad_damp_mult": 1.15,
                    "pad_failures": np.array([24.0]),
                    "x_bridge_local": 30.0,
                }
                for _ in range(n_passages)
            ]
        )
    else:
        track_log = _cell([np.empty((0, 0)) for _ in range(n_passages)])
    if stage_policy["use_oor_eov"]:
        oor_log = _cell(
            [
                {
                    "flats": np.zeros((0, 5), dtype=float),
                    "poly": np.array(
                        [[1.0, 1.0, 8.0, 2.5e-5, 0.75]], dtype=float
                    ),
                }
                for _ in range(n_passages)
            ]
        )
    else:
        oor_log = _cell([np.empty((0, 0)) for _ in range(n_passages)])
    if crack_on:
        crack_log = np.tile(
            np.array([[30.0, 0.15, 0.40]], dtype=float),
            (n_passages, 1),
        )
    else:
        crack_log = np.zeros((n_passages, 3), dtype=float)
    profile_mode = str(stage_policy["profile_mode"])
    if profile_mode == "psd_fra":
        profile_log = np.full(n_passages, 4.0)
    else:
        profile_log = np.ones(n_passages, dtype=float)

    bearing_fixity = np.asarray(bearing_vector, dtype=float)
    bearing_stiffness = (
        FIXTURE_K_REF_BEAR * bearing_fixity / (1.0 - bearing_fixity)
    )
    return {
        "AcelPrimVag": acel_prim,
        "AcelRodaPrimVag": acel_roda,
        "AcelWheelsetPrimVag": acel_wheelset,
        "PitchPrimVag": pitch,
        # Four time samples in each signal: DimAcel must be four, not dt.
        "DimAcel": np.full(n_passages, dim_acel, dtype=np.int64),
        "DimSpace": np.full(n_passages, dim_space, dtype=np.int64),
        "crop_start": np.full(n_passages, crop_start, dtype=np.int64),
        "crop_end": np.full(n_passages, crop_end, dtype=np.int64),
        "bridge_samp": np.full(n_passages, bridge_samp, dtype=np.int64),
        "L_bridge_eff": np.full(n_passages, bridge_length, dtype=float),
        "scour_vector": np.asarray(scour_vector, dtype=float),
        "scour_supports": scour_supports,
        "Dano": float(np.max(scour_vector)),
        "state_family": family,
        "state_uid": state_uid,
        "state_seed_id": np.uint32(state_seed_id),
        "random_stream_schedule_version":
            comparison._RANDOM_STREAM_SCHEDULE,
        "state_named_stream_seed_id": np.asarray(
            state_named_stream_seed_id, dtype=np.uint32
        ),
        "passage_named_stream_seed_id": np.asarray(
            passage_named_stream_seed_id, dtype=np.uint32
        ),
        "latent_bearing_fixity":
            np.asarray(latent_bearing_fixity, dtype=float),
        "latent_crack_on": bool(latent_crack_on),
        "crack_on": bool(crack_on),
        "beam_f1_Hz": 4.2 if bridge_length < 80 else 2.8,
        "bearing_vector": bearing_stiffness,
        "bearing_fixity": bearing_fixity,
        "crack_log": crack_log,
        "profile_mode": profile_mode,
        "profile_log": profile_log,
        "track_log": track_log,
        "oor_log": oor_log,
        "contact_log": np.zeros((n_passages, 4), dtype=float),
        "Temperatura": np.linspace(5.0, 25.0, n_passages),
        "Velocidade": velocity,
        "VehiclesProps": np.arange(15.0).reshape(5, 3),
        "gen_schema": policy.gen_schema,
        "gen_fingerprint": generation_fingerprint,
        "channel_schema_id": policy.channel_schema_id,
        "matlab_release": actual.release,
        "campaign_matlab_release": campaign.release,
        "actual_matlab_environment_descriptor": actual.descriptor,
        "actual_matlab_environment_sha256": actual.sha256,
        "campaign_matlab_environment_descriptor": campaign.descriptor,
        "campaign_matlab_environment_sha256": campaign.sha256,
        "generator_source_root_sha256":
            policy.generator_source_root_sha256,
        "generator_source_digest_lines":
            policy.generator_source_digest_lines,
        "generator_source_file_count":
            policy.generator_source_file_count,
        "qualification_source_sha256":
            qualification_sha,
        "release_qualification_run": True,
    }


def _fixture_labels(
    stage: str,
    n_states: int | None,
    *,
    damage_value: float,
) -> dict[str, object]:
    stage_policy = comparison._STAGE_POLICIES[stage]
    families: list[str] = []
    for family in (
        "target_healthy",
        "scour_only",
        "bearing_only",
        "nuisance_only",
        "joint",
    ):
        families.extend(
            [family] * int(stage_policy["family_counts"][family])
        )
    requested = int(n_states or stage_policy["n_states"])
    if requested < len(families):
        families = families[:requested]
    elif requested > len(families):
        families.extend(["joint"] * (requested - len(families)))

    num_supports = int(stage_policy["num_supports"])
    supports = np.asarray(stage_policy["scour_supports"], dtype=np.int64)
    damage = np.zeros((requested, num_supports), dtype=float)
    bearing = np.zeros((requested, 2), dtype=float)
    crack_on = np.zeros(requested, dtype=bool)
    anchor_target = np.zeros(requested, dtype=np.int64)
    anchor_level = np.zeros(requested, dtype=np.int64)
    latent_bearing = np.zeros((requested, 2), dtype=float)
    scour_seen = 0
    bearing_seen = 0
    for index, family in enumerate(families):
        if family == "scour_only":
            support = int(supports[(scour_seen // 4) % supports.size])
            level = scour_seen % 2 + 1
            damage[index, support - 1] = damage_value
            anchor_target[index] = support
            anchor_level[index] = level
            scour_seen += 1
        elif family == "bearing_only":
            target = (bearing_seen // 4) % 2
            level = bearing_seen % 2 + 1
            latent_bearing[index, target] = damage_value
            if stage_policy["bearing_mode"] == "target":
                bearing[index, target] = damage_value
            anchor_target[index] = target + 1
            anchor_level[index] = level
            bearing_seen += 1
        elif family == "nuisance_only":
            crack_on[index] = True
        elif family == "joint":
            damage[index, int(supports[0]) - 1] = damage_value
            latent_bearing[index] = np.array([0.15, 0.25])
            if stage_policy["bearing_mode"] == "target":
                bearing[index] = latent_bearing[index]

    geometry_tag = "".join(f"{int(value):02d}" for value in supports)
    family_occurrence: Counter[str] = Counter()
    anchor_occurrence: Counter[tuple[str, int, int]] = Counter()
    state_uids: list[str] = []
    for family, target, level in zip(
        families, anchor_target, anchor_level
    ):
        family_occurrence[family] += 1
        if family == "joint":
            uid_level = family_occurrence[family]
            replica = 1
        else:
            uid_level = int(level)
            key = (family, int(target), int(level))
            anchor_occurrence[key] += 1
            replica = anchor_occurrence[key]
        state_uids.append(
            "ttbi-state-v1|"
            f"Lmm={int(round(float(stage_policy['L_bridge_m']) * 1000)):06d}|"
            f"spans={int(stage_policy['num_spans'])}|"
            f"scour={geometry_tag}|family={family}|"
            f"target={int(target):02d}|level={uid_level:04d}|"
            f"rep={replica:03d}"
        )
    state_seed_ids = np.asarray(
        [
            int(
                hashlib.sha256(
                    (
                        "ttbi-state-seed-v1|damage_seed=1|"
                        + uid
                    ).encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            for uid in state_uids
        ],
        dtype=np.uint32,
    )
    latent_crack_hash = np.asarray(
        [
            int(
                hashlib.sha256(
                    (
                        "latent-crack-v1|damage_seed=1|" + uid
                    ).encode("utf-8")
                ).hexdigest()[:13],
                16,
            )
            / 16**13
            <= 0.25
            for uid in state_uids
        ],
        dtype=bool,
    )
    families_array = np.asarray(families)
    latent_crack = np.zeros(requested, dtype=bool)
    latent_crack[families_array == "joint"] = latent_crack_hash[
        families_array == "joint"
    ]
    latent_crack[families_array == "nuisance_only"] = True
    crack_on[:] = False
    if stage_policy["use_crack_eov"]:
        crack_on[families_array == "joint"] = latent_crack[
            families_array == "joint"
        ]
        crack_on[families_array == "nuisance_only"] = True

    return {
        "families": families,
        "damage": damage,
        "bearing": bearing,
        "crack_on": crack_on,
        "anchor_target": anchor_target,
        "anchor_level": anchor_level,
        "state_uids": state_uids,
        "state_seed_ids": state_seed_ids,
        "latent_bearing": latent_bearing,
        "latent_crack": latent_crack,
        "supports": supports,
    }


def _write_host_receipt(
    path: Path,
    *,
    host_id: str,
    actual_environment_sha256: str,
    qualification_source_sha256: str,
    qualification_executed_file_sha256: str,
) -> None:
    hostname = f"{host_id}.fixture.invalid"
    cpu_identifier = f"fixture-cpu-for-{host_id}"
    logical_processors = 16
    matlab_max_threads = 8
    computer_arch = "win64"
    descriptor = "\n".join(
        (
            f"schema={comparison._QUALIFICATION_HOST_SCHEMA}",
            f"declared_host_id={host_id}",
            f"hostname={hostname}",
            f"cpu_identifier={cpu_identifier}",
            f"logical_processors={logical_processors}",
            f"matlab_max_threads={matlab_max_threads}",
            f"computer_arch={computer_arch}",
            (
                "actual_matlab_environment_sha256="
                f"{actual_environment_sha256}"
            ),
            (
                "qualification_source_sha256="
                f"{qualification_source_sha256}"
            ),
            (
                "qualification_executed_file_sha256="
                f"{qualification_executed_file_sha256}"
            ),
        )
    )
    receipt = {
        "schema": comparison._QUALIFICATION_HOST_SCHEMA,
        "declared_host_id": host_id,
        "hostname": hostname,
        "cpu_identifier": cpu_identifier,
        "logical_processors": logical_processors,
        "matlab_max_threads": matlab_max_threads,
        "computer_arch": computer_arch,
        "actual_matlab_environment_sha256": actual_environment_sha256,
        "qualification_source_sha256": qualification_source_sha256,
        "qualification_executed_file_sha256":
            qualification_executed_file_sha256,
        "canonical_descriptor": descriptor,
        "host_diagnostic_sha256": hashlib.sha256(
            descriptor.encode("utf-8")
        ).hexdigest(),
    }
    (path / "qualification_host_receipt.json").write_text(
        json.dumps(receipt, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_dataset(
    path: Path,
    *,
    release: str,
    qualify: bool = True,
    mutate_data: Mutation | None = None,
    mutate_manifest: Mutation | None = None,
    mutate_state: Mutation | None = None,
    mutate_damage_table: Mutation | None = None,
    mutate_generation_config: Mutation | None = None,
    n_states: int | None = None,
    n_passages: int = 3,
    stage: str = "F40-S",
    actual_environment: MatlabEnvironment | None = None,
    campaign_environment: MatlabEnvironment | None = None,
    host_id: str = "fixture-host",
    damage_value: float = 0.2,
) -> None:
    path.mkdir(parents=True)
    policy = comparison._current_policy()
    actual = actual_environment or _environment(release)
    campaign = campaign_environment or _campaign_environment()
    stage_policy = comparison._STAGE_POLICIES[stage]
    labels = _fixture_labels(
        stage,
        n_states,
        damage_value=damage_value,
    )
    n_states = len(labels["families"])
    state_named_stream_seed_ids = np.empty(
        (n_states, len(comparison._STATE_STREAM_NAMES)), dtype=np.uint32
    )
    passage_named_stream_seed_ids = np.empty(
        (
            n_states,
            n_passages,
            len(comparison._PASSAGE_STREAM_NAMES),
        ),
        dtype=np.uint32,
    )
    for row, (uid, root_seed) in enumerate(
        zip(labels["state_uids"], labels["state_seed_ids"], strict=True)
    ):
        for stream_index, stream_name in enumerate(
            comparison._STATE_STREAM_NAMES
        ):
            state_named_stream_seed_ids[row, stream_index] = (
                comparison._expected_named_seed(
                    comparison._RANDOM_STREAM_SCHEDULE,
                    int(root_seed),
                    uid,
                    stream_name,
                )
            )
        for passage_index in range(n_passages):
            for stream_index, stream_name in enumerate(
                comparison._PASSAGE_STREAM_NAMES
            ):
                passage_named_stream_seed_ids[
                    row, passage_index, stream_index
                ] = comparison._expected_named_seed(
                    comparison._RANDOM_STREAM_SCHEDULE,
                    int(root_seed),
                    uid,
                    stream_name,
                    passage=passage_index + 1,
                )
    passage_named_stream_seed_ids_flat = (
        passage_named_stream_seed_ids.reshape(n_states, -1, order="F")
    )
    bridge_length = float(stage_policy["L_bridge_m"])
    scour_supports = np.asarray(labels["supports"], dtype=np.int64)
    num_supports = int(stage_policy["num_supports"])
    num_spans = int(stage_policy["num_spans"])
    qualification_sha = comparison._expected_qualification_source_sha256(
        stage, policy.source_snapshot
    )
    qualification_executable = (
        comparison._expected_qualification_source_bytes(
            stage, policy.source_snapshot
        )
    )
    (path / "qualification_executed.m").write_bytes(
        qualification_executable
    )
    _write_host_receipt(
        path,
        host_id=host_id,
        actual_environment_sha256=actual.sha256,
        qualification_source_sha256=qualification_sha,
        qualification_executed_file_sha256=hashlib.sha256(
            qualification_executable
        ).hexdigest(),
    )
    active_bearing_fixity = np.asarray(labels["bearing"], dtype=float)
    active_bearing_stiffness = (
        FIXTURE_K_REF_BEAR
        * active_bearing_fixity
        / (1.0 - active_bearing_fixity)
    )
    damage_states = {
        "DamageStates": labels["damage"],
        "BearingStates": active_bearing_stiffness,
        "BearingFixity": active_bearing_fixity,
        "LatentBearingFixity": labels["latent_bearing"],
        "StateFamily": np.array(labels["families"], dtype=object),
        "AnchorTarget": labels["anchor_target"],
        "AnchorLevel": labels["anchor_level"],
        "StateUID": np.array(labels["state_uids"], dtype=object),
        "StateSeedID": labels["state_seed_ids"],
        "StateNamedStreamSeedID": state_named_stream_seed_ids,
        "PassageNamedStreamSeedID": passage_named_stream_seed_ids,
        "PassageNamedStreamSeedIDFlat":
            passage_named_stream_seed_ids_flat,
        "random_stream_schedule_version":
            comparison._RANDOM_STREAM_SCHEDULE,
        "state_stream_names": np.asarray(
            comparison._STATE_STREAM_NAMES, dtype=object
        ),
        "passage_stream_names": np.asarray(
            comparison._PASSAGE_STREAM_NAMES, dtype=object
        ),
        "LatentCrackOn": labels["latent_crack"],
        "CrackOn": labels["crack_on"],
        "k_ref_bear": FIXTURE_K_REF_BEAR,
        "scour_supports": scour_supports,
    }
    if mutate_damage_table:
        mutate_damage_table(damage_states)

    generation_config = comparison._expected_qualification_generation_config(
        stage, policy, qualification_sha
    )
    generation_config.update(
        {
            "DamageStates":
                np.asarray(damage_states["DamageStates"]).tolist(),
            "BearingStates":
                np.asarray(damage_states["BearingStates"]).tolist(),
            "BearingFixity":
                np.asarray(damage_states["BearingFixity"]).tolist(),
            "StateFamily": [
                str(value)
                for value in np.asarray(
                    damage_states["StateFamily"], dtype=object
                ).reshape(-1)
            ],
            "AnchorTarget": [
                int(value)
                for value in np.asarray(
                    damage_states["AnchorTarget"]
                ).reshape(-1)
            ],
            "AnchorLevel": [
                int(value)
                for value in np.asarray(
                    damage_states["AnchorLevel"]
                ).reshape(-1)
            ],
            "StateUID": [
                str(value)
                for value in np.asarray(
                    damage_states["StateUID"], dtype=object
                ).reshape(-1)
            ],
            "StateSeedID": [
                int(value)
                for value in np.asarray(
                    damage_states["StateSeedID"]
                ).reshape(-1)
            ],
            "LatentBearingFixity":
                np.asarray(
                    damage_states["LatentBearingFixity"]
                ).tolist(),
            "LatentCrackOn": [
                bool(value)
                for value in np.asarray(
                    damage_states["LatentCrackOn"]
                ).reshape(-1)
            ],
            "CrackOn": [
                bool(value)
                for value in np.asarray(
                    damage_states["CrackOn"]
                ).reshape(-1)
            ],
            "StateNamedStreamSeedID":
                state_named_stream_seed_ids.tolist(),
            "PassageNamedStreamSeedIDFlat":
                passage_named_stream_seed_ids_flat.tolist(),
        }
    )
    if mutate_generation_config:
        mutate_generation_config(generation_config)
    generation_config_json = json.dumps(
        generation_config,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    generation_fingerprint = hashlib.sha256(
        generation_config_json.encode("utf-8")
    ).hexdigest()

    family_counts = Counter(labels["families"])
    manifest = {
        "case_name": "qualification_fixture",
        "case_desc": "synthetic-behavioral-contract",
        "stage": stage,
        "gen_schema": policy.gen_schema,
        "gen_fingerprint": generation_fingerprint,
        "generation_config_json": generation_config_json,
        "generation_behavior_version": policy.generation_behavior_version,
        "channel_schema_id": policy.channel_schema_id,
        "state_design_kind": generation_config.get(
            "state_design_kind",
            "dense-scour-61x5-v1"
            if stage == "F40-S"
            else "five-family-multidamage-v2",
        ),
        "matlab_release": actual.release,
        "campaign_matlab_release": campaign.release,
        "actual_matlab_environment_descriptor": actual.descriptor,
        "actual_matlab_environment_sha256": actual.sha256,
        "campaign_matlab_environment_descriptor": campaign.descriptor,
        "campaign_matlab_environment_sha256": campaign.sha256,
        "generator_source_root_sha256":
            policy.generator_source_root_sha256,
        "generator_source_digest_lines":
            policy.generator_source_digest_lines,
        "generator_source_file_count":
            policy.generator_source_file_count,
        "qualification_source_sha256": qualification_sha,
        "release_qualification_run": qualify,
        "timestamp": f"timestamp-{actual.sha256}",
        "n_states": n_states,
        "passages_per_state": n_passages,
        "max_parfor_workers": policy.max_parfor_workers,
        "L_bridge_m": bridge_length,
        "num_spans": num_spans,
        "num_supports": num_supports,
        "state_identity_version":
            generation_config["state_identity_version"],
        "joint_lhs_design": generation_config["joint_lhs_design"],
        "n_latent_bearing_dims":
            generation_config["n_latent_bearing_dims"],
        "bearing_k_ref_Nm_rad": damage_states["k_ref_bear"],
        "bearing_mode": stage_policy["bearing_mode"],
        "use_crack_eov": stage_policy["use_crack_eov"],
        "profile_mode": stage_policy["profile_mode"],
        "use_track_eov": stage_policy["use_track_eov"],
        "use_oor_eov": stage_policy["use_oor_eov"],
        "oor_flats_enabled": stage_policy["oor_flats_enabled"],
        "n_target_healthy": family_counts["target_healthy"],
        "n_scour_only": family_counts["scour_only"],
        "n_bearing_only": family_counts["bearing_only"],
        "n_nuisance_only": family_counts["nuisance_only"],
        "n_joint": family_counts["joint"],
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    savemat(
        path / "case_info.mat",
        {"case_info": manifest},
        long_field_names=True,
    )

    savemat(path / "damage_states.mat", damage_states, long_field_names=True)

    for state_index in range(1, n_states + 1):
        row = state_index - 1
        data = deepcopy(
            _base_data(
                actual,
                policy=policy,
                campaign=campaign,
                stage=stage,
                n_passages=n_passages,
                scour_vector=np.asarray(labels["damage"])[row],
                bearing_vector=np.asarray(labels["bearing"])[row],
                family=str(labels["families"][row]),
                crack_on=bool(np.asarray(labels["crack_on"])[row]),
                state_uid=str(labels["state_uids"][row]),
                state_seed_id=int(
                    np.asarray(labels["state_seed_ids"])[row]
                ),
                latent_bearing_fixity=np.asarray(
                    labels["latent_bearing"]
                )[row],
                latent_crack_on=bool(
                    np.asarray(labels["latent_crack"])[row]
                ),
                state_named_stream_seed_id=
                    state_named_stream_seed_ids[row],
                passage_named_stream_seed_id=
                    passage_named_stream_seed_ids[row],
                qualification_sha=qualification_sha,
                generation_fingerprint=generation_fingerprint,
            )
        )
        data["release_qualification_run"] = qualify
        if mutate_data:
            mutate_data(data)
        state = {
            "data": data,
            "file_gen_schema": policy.gen_schema,
            "file_gen_fingerprint": generation_fingerprint,
            "file_matlab_release": actual.release,
            "file_campaign_matlab_release": campaign.release,
            "file_release_qualification_run": qualify,
            "file_state_uid": str(labels["state_uids"][row]),
            "file_state_seed_id": np.uint32(
                np.asarray(labels["state_seed_ids"])[row]
            ),
            "file_random_stream_schedule_version":
                comparison._RANDOM_STREAM_SCHEDULE,
            "file_actual_matlab_environment_sha256": actual.sha256,
            "file_campaign_matlab_environment_sha256": campaign.sha256,
            "file_generator_source_root_sha256":
                policy.generator_source_root_sha256,
            "file_qualification_source_sha256": qualification_sha,
        }
        if mutate_state:
            mutate_state(state)
        savemat(
            path / f"{state_index:04d}.mat",
            state,
            long_field_names=True,
        )

    state_names = [f"{index:04d}.mat" for index in range(1, n_states + 1)]
    digest_names = state_names + ["case_info.mat", "damage_states.mat"]
    per_file = {name: _sha256(path / name) for name in digest_names}
    digest_lines = "\n".join(
        f"{name}:{per_file[name]}" for name in sorted(per_file)
    )
    dataset_root = hashlib.sha256(digest_lines.encode("utf-8")).hexdigest()
    savemat(
        path / "file_digests.mat",
        {
            "file_digests": {
                "schema": "source-digests-v2",
                "scope": "NNNN.mat+case_info.mat+damage_states.mat",
                "digest_lines": digest_lines,
                "root": dataset_root,
            }
        },
        long_field_names=True,
    )
    (path / "_GENERATION_COMPLETE").write_bytes(
        (
            f"{policy.gen_schema}\n{generation_fingerprint}\n"
            f"{dataset_root}\n"
        ).encode("utf-8")
    )


def _pair(
    root: Path,
    name: str,
    *,
    mutate_a: Mutation | None = None,
    mutate_b: Mutation | None = None,
    mutate_manifest_a: Mutation | None = None,
    mutate_manifest_b: Mutation | None = None,
    mutate_state_a: Mutation | None = None,
    mutate_state_b: Mutation | None = None,
    mutate_damage_table_a: Mutation | None = None,
    mutate_damage_table_b: Mutation | None = None,
    mutate_generation_config_a: Mutation | None = None,
    mutate_generation_config_b: Mutation | None = None,
    qualify_a: bool = True,
    qualify_b: bool = True,
    release_a: str | None = None,
    release_b: str = "R2023b",
    actual_environment_a: MatlabEnvironment | None = None,
    actual_environment_b: MatlabEnvironment | None = None,
    host_id_a: str = "reference-host",
    host_id_b: str = "candidate-host",
    campaign_environment: MatlabEnvironment | None = None,
    n_states: int | None = None,
    n_passages: int = 3,
    stage: str = "F40-S",
) -> tuple[Path, Path]:
    campaign = _campaign_environment()
    release_a = release_a or campaign.release
    a, b = root / f"{name}_a", root / f"{name}_b"
    common = {
        "campaign_environment": campaign_environment,
        "n_states": n_states,
        "n_passages": n_passages,
        "stage": stage,
    }
    _write_dataset(
        a,
        release=release_a,
        qualify=qualify_a,
        mutate_data=mutate_a,
        mutate_manifest=mutate_manifest_a,
        mutate_state=mutate_state_a,
        mutate_damage_table=mutate_damage_table_a,
        mutate_generation_config=mutate_generation_config_a,
        actual_environment=actual_environment_a,
        host_id=host_id_a,
        **common,
    )
    _write_dataset(
        b,
        release=release_b,
        qualify=qualify_b,
        mutate_data=mutate_b,
        mutate_manifest=mutate_manifest_b,
        mutate_state=mutate_state_b,
        mutate_damage_table=mutate_damage_table_b,
        mutate_generation_config=mutate_generation_config_b,
        actual_environment=actual_environment_b,
        host_id=host_id_b,
        **common,
    )
    return a, b


def _expect_invalid(name: str, call: Callable[[], object]) -> None:
    try:
        call()
    except comparison.QualificationInputError:
        print(f"  [PASS] invalid evidence rejected: {name}")
        return
    raise AssertionError(f"invalid qualification evidence escaped: {name}")


def _flip_last_byte(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))


def _expect_invalid_from(
    name: str,
    needle: str,
    call: Callable[[], object],
) -> None:
    """Require rejection BY A NAMED GUARD, not merely rejection.

    The atomicity bindings overlap on purpose - a late mutation is caught by the
    end-of-run reassertion even if the per-member parse-time binding is absent -
    so a probe that only asserts "something raised" cannot tell which binding is
    doing the work, and would keep passing after the binding it was written for
    is deleted.  Pinning the diagnostic makes each probe fail when ITS OWN guard
    is removed.
    """
    try:
        call()
    except comparison.QualificationInputError as exc:
        message = str(exc)
        if needle not in message:
            raise AssertionError(
                f"{name}: rejected, but not by the intended guard "
                f"(expected {needle!r} in: {message})"
            ) from exc
        print(f"  [PASS] invalid evidence rejected: {name}")
        return
    raise AssertionError(f"invalid qualification evidence escaped: {name}")


def _intra_invocation_mutation_checks(root: Path) -> None:
    """Prove atomicity WITHIN one comparison, not merely between two runs.

    A qualification directory is an ordinary mutable folder: it can be
    synchronised, copied over, or edited while a long validation is running.
    Hashing a member and then re-opening it to parse it are two separate reads,
    so without the bindings these cases exercise, a verdict could describe a
    mixture of pre- and post-change content while the evidence quoted digests
    that were never parsed.  No malicious actor is required.

    Both content windows and the separate authorization-sidecar window are
    covered:

    * hash-to-parse, inside one member - the mutation lands after the header's
      digest verification and must be caught when the payload is parsed; and
    * parse-to-publish, across members - the mutation lands after endpoint A was
      fully parsed, while endpoint B is still being read, and must be caught by
      the end-of-run digest reassertion before any verdict is returned; and
    * sidecar-parse-to-publish - each authorization sidecar outside the digest
      table is mutated after endpoint A was parsed and must be caught by its
      specifically named final sidecar guard.

    The mutations are installed by wrapping the comparator's own entry points,
    which is what makes them intra-invocation; mutating between two calls (as
    the inventory checker's stale-endpoint case does) proves only that the
    endpoint is reopened per invocation.
    """
    original_header = comparison._validate_dataset_header
    original_payload = comparison._validated_payload

    # (1) hash-to-parse: mutate a state file after the header validated every
    # digest, before _validated_payload parses that state.
    a, b = _pair(root, "toctou_hash_to_parse")
    fired = {"count": 0}

    def mutating_header(path: Path, policy: object) -> object:
        result = original_header(path, policy)
        if fired["count"] == 0:
            fired["count"] = 1
            _flip_last_byte(Path(path) / "0001.mat")
        return result

    comparison._validate_dataset_header = mutating_header
    try:
        _expect_invalid_from(
            "state file mutated between digest verification and parsing "
            "(intra-invocation)",
            "at parse time",
            lambda: comparison.compare_directories(a, b),
        )
    finally:
        comparison._validate_dataset_header = original_header
    if fired["count"] != 1:
        raise AssertionError(
            "hash-to-parse probe never fired: the comparator did not call "
            "_validate_dataset_header, so the case proves nothing"
        )

    # (2) parse-to-publish: mutate endpoint A after it was fully parsed, while
    # endpoint B is still being validated.  Only the end-of-run reassertion can
    # catch this, because every per-member parse already succeeded.
    c, d = _pair(root, "toctou_parse_to_publish")
    calls = {"count": 0}

    def mutating_payload(path: Path, policy: object) -> object:
        result = original_payload(path, policy)
        calls["count"] += 1
        if calls["count"] == 1:
            _flip_last_byte(Path(c) / "0001.mat")
        return result

    comparison._validated_payload = mutating_payload
    try:
        _expect_invalid_from(
            "endpoint mutated after it was parsed, while the other endpoint "
            "was still being read (intra-invocation)",
            "changed during validation",
            lambda: comparison.compare_directories(c, d),
        )
    finally:
        comparison._validated_payload = original_payload
    if calls["count"] < 2:
        raise AssertionError(
            "parse-to-publish probe never reached the second endpoint: the "
            "case proves nothing"
        )

    # (3) authorization sidecars are intentionally outside file_digests.mat,
    # so exercise their separate final snapshot guard one by one.  Pinning each
    # sidecar's diagnostic makes these probes fail if that specific binding is
    # removed while the digested-member reassertion remains green.
    sidecars = (
        "qualification_executed.m",
        "qualification_host_receipt.json",
        "_GENERATION_COMPLETE",
    )
    for sidecar in sidecars:
        safe_name = sidecar.lower().replace(".", "_")
        e, f = _pair(root, f"toctou_sidecar_{safe_name}")
        sidecar_calls = {"count": 0}

        def mutating_sidecar_payload(
            path: Path,
            policy: object,
            *,
            endpoint: Path = e,
            member: str = sidecar,
        ) -> object:
            result = original_payload(path, policy)
            sidecar_calls["count"] += 1
            if sidecar_calls["count"] == 1:
                _flip_last_byte(endpoint / member)
            return result

        comparison._validated_payload = mutating_sidecar_payload
        try:
            _expect_invalid_from(
                f"{sidecar} mutated after endpoint parsing "
                "(intra-invocation)",
                f"endpoint sidecar {sidecar} changed during validation",
                lambda e=e, f=f: comparison.compare_directories(e, f),
            )
        finally:
            comparison._validated_payload = original_payload
        if sidecar_calls["count"] < 2:
            raise AssertionError(
                f"{sidecar} probe never reached the second endpoint: the "
                "case proves nothing"
            )

    # The wrappers are gone; an unmutated pair must still be comparable, so a
    # failure above cannot be an artifact of leftover patching.
    e, f = _pair(root, "toctou_control")
    _expect_verdict(
        "unmutated control pair after the intra-invocation probes",
        "SEMANTICALLY-BIT-IDENTICAL",
        lambda: comparison.compare_directories(e, f),
    )


def _expect_verdict(
    name: str, expected: str, call: Callable[[], comparison.ComparisonResult]
) -> comparison.ComparisonResult:
    result = call()
    if result.verdict != expected:
        raise AssertionError(
            f"{name}: verdict {result.verdict!r} != {expected!r}; "
            f"mismatches={result.stats.mismatches}"
        )
    print(f"  [PASS] {name}: {expected}")
    return result


def _rewrite_digest_table(
    path: Path,
    mutate: Callable[[dict], None],
) -> None:
    value = deepcopy(
        _fixture_mat(path / "file_digests.mat")["file_digests"]
    )
    mutate(value)
    savemat(
        path / "file_digests.mat",
        {"file_digests": value},
        long_field_names=True,
    )
    marker_lines = (path / "_GENERATION_COMPLETE").read_bytes().decode(
        "utf-8"
    ).splitlines()
    marker_lines[2] = str(value["root"])
    (path / "_GENERATION_COMPLETE").write_bytes(
        ("\n".join(marker_lines) + "\n").encode("utf-8")
    )


def _set_digest_lines(value: dict, lines: str) -> None:
    value["digest_lines"] = lines
    value["root"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _micro_patch_checks() -> None:
    source_path = (
        Path(__file__).resolve().parent / "scour_MATLAB" / "A00_Run.m"
    )
    source = source_path.read_text(encoding="utf-8")
    expected = qualification_source_sha256("F40-M")
    assert len(expected) == 64
    qualification = render_micro_a00(
        source, qualification=True, stage="F40-M"
    )
    assert "STAGE = 'F40-M';" in qualification
    assert "qualification_run = true;" in qualification
    assert expected in qualification
    assert "Results', 'release_qualification'" in qualification
    assert "actual_matlab_environment_sha256(1:16)" in qualification
    raw_sha = verify_qualification_source_bytes(
        qualification.encode("utf-8"),
        expected,
    )
    assert raw_sha == hashlib.sha256(
        qualification.encode("utf-8")
    ).hexdigest()

    policy = comparison._current_policy()
    original_capture = policy.source_snapshot
    mutated_entries = []
    capture_changed = False
    for name, captured_file in original_capture.snapshots:
        if name == "scour_MATLAB/A00_Run.m":
            capture_changed = True
            captured_file = replace(
                captured_file,
                raw=captured_file.raw + b"\n% CAPTURE-ONLY HYBRID PROBE\n",
            )
        mutated_entries.append((name, captured_file))
    if not capture_changed:
        raise AssertionError("A00 was absent from coherent source capture")
    capture_only = replace(
        original_capture,
        snapshots=tuple(mutated_entries),
    )
    captured_digest = comparison._expected_qualification_source_sha256(
        "F40-M", capture_only
    )
    captured_executable = comparison._expected_qualification_source_bytes(
        "F40-M", capture_only
    )
    if captured_digest == expected or captured_executable == qualification.encode(
        "utf-8"
    ):
        raise AssertionError(
            "qualification helper reopened the live A00 pathname instead of "
            "using the supplied coherent capture"
        )
    verify_qualification_source_bytes(captured_executable, captured_digest)
    print(
        "  [PASS] qualification identity/executable use one supplied source "
        "capture"
    )
    assert "ttbi.qualification_script_identity(" in qualification
    assert "ttbi.preserve_qualification_evidence(" in qualification
    # Qualification publication moved into the +ttbi package.  Assert both
    # halves of that boundary: the generated script must delegate to the
    # reviewed owner, and the owner must retain the executed script and host
    # receipt.  Every package source below is part of the stamped generator
    # source root.
    package = Path(__file__).resolve().parent / "scour_MATLAB" / "+ttbi"
    preservation_src = (
        package / "preserve_qualification_evidence.m"
    ).read_text(encoding="utf-8")
    host_receipt_src = (package / "qualification_host_receipt.m").read_text(
        encoding="utf-8"
    )
    writer_src = (package / "write_qualification_host_receipt.m").read_text(
        encoding="utf-8"
    )
    assert "qualification_executed.m" in preservation_src
    assert "ttbi.qualification_host_receipt(" in preservation_src
    assert "ttbi.write_qualification_host_receipt(" in preservation_src
    assert "TTBI_QUALIFICATION_HOST_ID" in host_receipt_src
    assert "host_diagnostic_sha256" in host_receipt_src
    assert "qualification_host_receipt.json" in writer_src
    mutated = qualification.replace(
        "% Main script to run TTB-2D model",
        "% MUTATED after source stamp",
        1,
    ).encode("utf-8")
    try:
        verify_qualification_source_bytes(mutated, expected)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "post-stamp executable-byte mutation escaped self-authentication"
        )
    print(
        "  [PASS] micro qualification patch self-authenticates exact executable "
        "bytes and preserves them as evidence"
    )
    source = source_path.read_text(encoding="utf-8")
    retired_calls = (
        (lambda: render_micro_a00(source), ValueError),
        (lambda: write_micro_a00(), ValueError),
        (lambda: render_toy_driver("irrelevant"), RuntimeError),
    )
    for call, error_type in retired_calls:
        try:
            call()
        except error_type:
            pass
        else:
            raise AssertionError("retired unmarked micro API remained callable")

    for argv, message in (
        ([], "no-argument micro generation is retired"),
        (["--dryrun"], "--dryrun is retired"),
        (["--qualification"], "--qualification requires --stage"),
    ):
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                micro_main(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"retired/incomplete CLI escaped: {argv!r}")
        assert message in stderr.getvalue()
    print(
        "  [PASS] no-argument/toy micro paths fail closed; qualification "
        "requires one explicit current Paper-1 stage"
    )


def main() -> None:
    print("GENERATION ENVIRONMENT COMPARISON CHECKS")
    with tempfile.TemporaryDirectory(prefix="environment-comparison-check-") as tmp:
        root = Path(tmp)
        policy = comparison._current_policy()
        assert len(comparison._REQUIRED_TOP_LEVEL_FIELDS) == 13
        assert len(comparison._REQUIRED_DATA_FIELDS) == 48
        lock = load_environment_lock(
            Path(__file__).resolve().parent
            / "environment"
            / "campaign-py313-cu128.json"
        )
        assert (
            comparison._validate_parser_environment(lock["spec"])
            == policy.parser_environment
        )
        for key in ("python", "platform_system", "numpy", "scipy"):
            forged_runtime = dict(policy.parser_environment)
            forged_runtime[key] = f"{forged_runtime[key]}-FORGED"
            _expect_invalid(
                f"minimal parser environment drift: {key}",
                lambda runtime=forged_runtime: comparison._validate_parser_environment(
                    lock["spec"], runtime
                ),
            )
        missing_numpy = deepcopy(lock["spec"])
        del missing_numpy["packages"]["numpy"]
        _expect_invalid(
            "minimal parser lock lacks NumPy",
            lambda: comparison._validate_parser_environment(
                missing_numpy, policy.parser_environment
            ),
        )

        a, b = _pair(root, "identical")
        exact = _expect_verdict(
            "nested canonical payload",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )
        assert exact.stats.compared_signal_values > 0
        assert exact.stats.compared_leaves > 0
        exact_receipt = root / "exact-receipt.json"
        assert (
            comparison.main(
                [str(a), str(b), "--receipt", str(exact_receipt)]
            )
            == 0
        )
        exact_receipt_payload = json.loads(
            exact_receipt.read_text(encoding="utf-8")
        )
        assert (
            exact_receipt_payload["verdict"]
            == "SEMANTICALLY-BIT-IDENTICAL"
        )
        assert (
            exact_receipt_payload[
                "numerical_equivalence_explicitly_accepted"
            ]
            is False
        )
        _expect_invalid(
            "exact verdict cannot claim numerical-equivalence acceptance",
            lambda: comparison._receipt_payload(exact, accepted=True),
        )

        same_family_candidate = _environment(
            policy.campaign_matlab_release, variant="different-update"
        )
        a, b = _pair(
            root,
            "same_family_different_stack",
            release_b=policy.campaign_matlab_release,
            actual_environment_b=same_family_candidate,
        )
        _expect_verdict(
            "same release family, different authenticated stack",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )

        assert tuple(comparison._STAGE_POLICIES) == comparison.STAGE_ORDER
        for stage in comparison._STAGE_POLICIES:
            a, b = _pair(root, f"exact_policy_{stage}", stage=stage)
            result = _expect_verdict(
                f"exact four-block qualification policy: {stage}",
                "SEMANTICALLY-BIT-IDENTICAL",
                lambda a=a, b=b: comparison.compare_directories(a, b),
            )
            expected_policy = comparison._STAGE_POLICIES[stage]
            assert result.evidence_a.n_states == expected_policy["n_states"]
            assert result.evidence_a.passages_per_state == 3
            expected_total = {
                "F40-S": 31,
                "F40-M": 31,
                "L99-S": 39,
                "L99-M": 39,
            }[stage]
            assert expected_policy["n_states"] == expected_total
            assert expected_policy["family_counts"]["bearing_only"] == 8
            assert expected_policy["family_counts"]["nuisance_only"] == 6
            assert expected_policy["family_counts"]["joint"] == 10
            coverage = result.evidence_a.mechanism_coverage
            assert coverage.required is False
            assert (
                coverage.crack_active_passages,
                coverage.profile_active_passages,
                coverage.ballast_patch_rows,
                coverage.hanging_group_rows,
                coverage.pad_departure_passages,
                coverage.polygon_rows,
            ) == (0, 0, 0, 0, 0, 0)
            assert coverage.witnesses == {}

        coherent_policy_mutations = (
            ("bridge length", "F40-S", "L_bridge_m", 41.0),
            ("span count", "F40-S", "num_spans", 3),
            ("support count", "F40-S", "num_supports", 4),
            ("bearing mode", "F40-M", "bearing_mode", "off"),
            ("crack toggle", "F40-M", "use_crack_eov", False),
            ("track toggle", "F40-M", "use_track_eov", True),
            ("OOR toggle", "F40-M", "use_oor_eov", True),
            ("profile mode", "F40-M", "profile_mode", "psd_fra"),
            ("flat policy", "F40-M", "oor_flats_enabled", True),
        )
        for name, stage, key, value in coherent_policy_mutations:
            def mutate_policy(
                manifest: dict,
                *,
                key: str = key,
                value: object = value,
            ) -> None:
                manifest[key] = value

            a, b = _pair(
                root,
                f"coherent_policy_{key}",
                stage=stage,
                mutate_manifest_a=mutate_policy,
                mutate_manifest_b=mutate_policy,
            )
            _expect_invalid(
                f"same forged {name} in both inputs",
                lambda a=a, b=b: comparison.compare_directories(a, b),
            )

        generation_config_mutations: tuple[
            tuple[str, Mutation], ...
        ] = (
            (
                "missing hashed-config field",
                lambda config: config.pop("dano_max"),
            ),
            (
                "extra hashed-config field",
                lambda config: config.__setitem__(
                    "unreviewed_generator_knob", 1
                ),
            ),
            (
                "same forged hashed-config knob",
                lambda config: config.__setitem__("dano_max", 0.61),
            ),
            (
                "boolean-as-number hashed-config laundering",
                lambda config: config.__setitem__("use_crack_eov", 0),
            ),
            (
                "hashed realization differs from damage table",
                lambda config: config["DamageStates"][0].__setitem__(0, 0.1),
            ),
            (
                "hashed named-stream realization differs from table",
                lambda config: config["StateNamedStreamSeedID"][0].__setitem__(
                    0,
                    (config["StateNamedStreamSeedID"][0][0] + 1) % (2**32),
                ),
            ),
        )
        for name, mutate_config in generation_config_mutations:
            a, b = _pair(
                root,
                "generation_config_"
                + name.lower().replace(" ", "_").replace("-", "_"),
                mutate_generation_config_a=mutate_config,
                mutate_generation_config_b=mutate_config,
            )
            _expect_invalid(
                name,
                lambda a=a, b=b: comparison.compare_directories(a, b),
            )

        _expect_invalid(
            "duplicate generation-config JSON key",
            lambda: comparison._strict_json_object(
                '{"dano_max":0.6,"dano_max":0.6}', "fixture"
            ),
        )
        _expect_invalid(
            "non-finite generation-config JSON constant",
            lambda: comparison._strict_json_object(
                '{"dano_max":NaN}', "fixture"
            ),
        )

        def stale_config_json_hash(manifest: dict) -> None:
            manifest["generation_config_json"] = str(
                manifest["generation_config_json"]
            ).replace('"dano_max":0.6', '"dano_max":0.61', 1)

        a, b = _pair(
            root,
            "stale_generation_config_hash",
            mutate_manifest_a=stale_config_json_hash,
            mutate_manifest_b=stale_config_json_hash,
        )
        _expect_invalid(
            "generation-config JSON changed without fingerprint restamp",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_manifest_crn_version(manifest: dict) -> None:
            manifest["state_identity_version"] = "semantic-state-v0"

        a, b = _pair(
            root,
            "wrong_manifest_crn_version",
            mutate_manifest_a=wrong_manifest_crn_version,
            mutate_manifest_b=wrong_manifest_crn_version,
        )
        _expect_invalid(
            "same forged manifest CRN identity version in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def nonderived_state_seed(table: dict) -> None:
            values = np.asarray(table["StateSeedID"], dtype=np.uint64).copy()
            values[0] = (int(values[0]) + 1) % (2**32)
            table["StateSeedID"] = values.astype(np.uint32)

        a, b = _pair(
            root,
            "nonderived_state_seed",
            mutate_damage_table_a=nonderived_state_seed,
            mutate_damage_table_b=nonderived_state_seed,
        )
        _expect_invalid(
            "same non-derived StateSeedID restamped in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def extra_damage_field(table: dict) -> None:
            table["UnreviewedDamageLabel"] = np.zeros(
                np.asarray(table["StateSeedID"]).shape,
                dtype=np.float64,
            )

        a, b = _pair(
            root,
            "damage_table_extra_field",
            mutate_damage_table_a=extra_damage_field,
            mutate_damage_table_b=extra_damage_field,
        )
        _expect_invalid(
            "damage_states.mat has exactly 19 fields",
            lambda: comparison.compare_directories(a, b),
        )

        def transposed_damage_matrix(table: dict) -> None:
            table["DamageStates"] = np.asarray(
                table["DamageStates"]
            ).T.copy()

        a, b = _pair(
            root,
            "damage_table_transposed_shape",
            mutate_damage_table_a=transposed_damage_matrix,
            mutate_damage_table_b=transposed_damage_matrix,
        )
        _expect_invalid(
            "damage_states.mat rejects equal-size transposed shapes",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "damage_table_uint64_state_seed",
            mutate_damage_table_a=lambda table: table.__setitem__(
                "StateSeedID",
                np.asarray(table["StateSeedID"], dtype=np.uint64),
            ),
            mutate_damage_table_b=lambda table: table.__setitem__(
                "StateSeedID",
                np.asarray(table["StateSeedID"], dtype=np.uint64),
            ),
        )
        _expect_invalid(
            "damage_states StateSeedID requires MATLAB uint32 dtype",
            lambda: comparison.compare_directories(a, b),
        )

        def nonderived_named_stream(table: dict) -> None:
            values = np.asarray(
                table["StateNamedStreamSeedID"], dtype=np.uint64
            ).copy()
            values[0, 0] = (int(values[0, 0]) + 1) % (2**32)
            table["StateNamedStreamSeedID"] = values.astype(np.uint32)

        a, b = _pair(
            root,
            "nonderived_named_stream",
            mutate_damage_table_a=nonderived_named_stream,
            mutate_damage_table_b=nonderived_named_stream,
        )
        _expect_invalid(
            "same non-derived named stream restamped in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_passage_flattening(table: dict) -> None:
            values = np.asarray(
                table["PassageNamedStreamSeedIDFlat"], dtype=np.uint32
            ).copy()
            values[:, [0, 1]] = values[:, [1, 0]]
            table["PassageNamedStreamSeedIDFlat"] = values

        a, b = _pair(
            root,
            "wrong_passage_flattening",
            mutate_damage_table_a=wrong_passage_flattening,
            mutate_damage_table_b=wrong_passage_flattening,
        )
        _expect_invalid(
            "same wrong MATLAB passage-stream flattening in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def nonderived_latent_crack(table: dict) -> None:
            values = np.asarray(
                table["LatentCrackOn"], dtype=bool
            ).copy()
            values[0] = ~values[0]
            table["LatentCrackOn"] = values

        a, b = _pair(
            root,
            "nonderived_latent_crack",
            mutate_damage_table_a=nonderived_latent_crack,
            mutate_damage_table_b=nonderived_latent_crack,
        )
        _expect_invalid(
            "same non-derived latent crack draw restamped in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def duplicate_state_uid(table: dict) -> None:
            values = np.asarray(table["StateUID"], dtype=object).reshape(-1)
            values[1] = values[0]
            table["StateUID"] = values

        a, b = _pair(
            root,
            "duplicate_state_uid",
            mutate_damage_table_a=duplicate_state_uid,
            mutate_damage_table_b=duplicate_state_uid,
        )
        _expect_invalid(
            "same duplicate semantic state UID in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def active_latent_bearing_disagreement(table: dict) -> None:
            values = np.asarray(
                table["LatentBearingFixity"], dtype=float
            ).copy()
            values[0, 0] = 0.1
            table["LatentBearingFixity"] = values

        a, b = _pair(
            root,
            "active_latent_bearing_disagreement",
            stage="F40-M",
            mutate_damage_table_a=active_latent_bearing_disagreement,
            mutate_damage_table_b=active_latent_bearing_disagreement,
        )
        _expect_invalid(
            "same active/latent bearing disagreement in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def linearized_bearing_table(table: dict) -> None:
            fixity = np.asarray(table["BearingFixity"], dtype=float)
            table["BearingStates"] = FIXTURE_K_REF_BEAR * fixity

        def linearized_bearing_state(data: dict) -> None:
            fixity = np.asarray(data["bearing_fixity"], dtype=float)
            data["bearing_vector"] = FIXTURE_K_REF_BEAR * fixity

        a, b = _pair(
            root,
            "coherent_linearized_bearing_transform",
            stage="F40-M",
            mutate_a=linearized_bearing_state,
            mutate_b=linearized_bearing_state,
            mutate_damage_table_a=linearized_bearing_table,
            mutate_damage_table_b=linearized_bearing_table,
        )
        _expect_invalid(
            "coherent linear phi-to-k_r transform in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def crack_activation_policy_drift(table: dict) -> None:
            values = np.asarray(table["CrackOn"], dtype=bool).copy()
            values[0] = True
            table["CrackOn"] = values

        a, b = _pair(
            root,
            "crack_activation_policy_drift",
            stage="F40-M",
            mutate_damage_table_a=crack_activation_policy_drift,
            mutate_damage_table_b=crack_activation_policy_drift,
        )
        _expect_invalid(
            "same controlled-family crack-policy drift in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def coherent_manifest_family_counts(manifest: dict) -> None:
            manifest["n_target_healthy"] -= 1
            manifest["n_joint"] += 1

        a, b = _pair(
            root,
            "coherent_manifest_family_counts",
            mutate_manifest_a=coherent_manifest_family_counts,
            mutate_manifest_b=coherent_manifest_family_counts,
        )
        _expect_invalid(
            "same forged manifest family counts in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def coherent_table_family_counts(table: dict) -> None:
            families = np.asarray(table["StateFamily"], dtype=object).reshape(-1)
            families[0] = "joint"
            table["StateFamily"] = families

        a, b = _pair(
            root,
            "coherent_table_family_counts",
            mutate_damage_table_a=coherent_table_family_counts,
            mutate_damage_table_b=coherent_table_family_counts,
        )
        _expect_invalid(
            "same forged damage-table family counts in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def small_signal(data: dict) -> None:
            data["AcelWheelsetPrimVag"][0][0, 1] += 5e-11

        a, b = _pair(root, "small_signal", mutate_b=small_signal)
        numerical = _expect_verdict(
            "in-tolerance solver difference",
            "NUMERICALLY-EQUIVALENT",
            lambda: comparison.compare_directories(a, b),
        )
        assert numerical.default_exit_code == 3
        original_reader = comparison._read_single_link_regular_bytes
        state_reads: Counter[tuple[Path, str]] = Counter()

        def recording_reader(path: Path, owner: str) -> bytes:
            candidate = Path(path)
            if (
                candidate.parent in {Path(a), Path(b)}
                and comparison._STATE_RE.fullmatch(candidate.name)
            ):
                state_reads[(candidate.parent, candidate.name)] += 1
            return original_reader(candidate, owner)

        comparison._read_single_link_regular_bytes = recording_reader
        try:
            digest_bound_numerical = comparison.compare_directories(a, b)
        finally:
            comparison._read_single_link_regular_bytes = original_reader
        assert (
            digest_bound_numerical.raw_byte_identical_states
            == numerical.raw_byte_identical_states
        )
        expected_state_reads = {
            (endpoint, name)
            for endpoint in (Path(a), Path(b))
            for name in numerical.evidence_a.state_files
        }
        assert set(state_reads) == expected_state_reads
        assert all(count == 3 for count in state_reads.values())
        print(
            "  [PASS] raw state identity comes from authenticated digest "
            "tables, with no fourth late pathname read"
        )
        assert comparison.main([str(a), str(b)]) == 3
        assert comparison.main([str(a), str(b), "--accept-numerical"]) == 2
        pending_receipt = root / "pending-receipt.json"
        assert (
            comparison.main(
                [str(a), str(b), "--receipt", str(pending_receipt)]
            )
            == 3
        )
        pending_receipt_payload = json.loads(
            pending_receipt.read_text(encoding="utf-8")
        )
        assert pending_receipt_payload["verdict"] == "NUMERICALLY-EQUIVALENT"
        assert (
            pending_receipt_payload[
                "numerical_equivalence_explicitly_accepted"
            ]
            is False
        )
        receipt = root / "accepted-receipt.json"
        assert (
            comparison.main(
                [
                    str(a),
                    str(b),
                    "--accept-numerical",
                    "--receipt",
                    str(receipt),
                ]
            )
            == 0
        )
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert receipt_payload["verdict"] == "NUMERICALLY-EQUIVALENT"
        assert (
            receipt_payload["numerical_equivalence_explicitly_accepted"]
            is True
        )
        assert (
            receipt_payload["schema"]
            == "matlab-environment-qualification-receipt-v4"
        )
        assert receipt_payload["comparator_sha256"] == _sha256(
            Path(comparison.__file__).resolve()
        )
        assert (
            receipt_payload["environment_lock_sha256"]
            == numerical.environment_lock_sha256
        )
        assert (
            receipt_payload["parser_environment"]
            == numerical.parser_environment
        )
        assert (
            receipt_payload["python_runtime_source_root_sha256"]
            == numerical.python_runtime_source_root_sha256
        )
        assert (
            receipt_payload["python_runtime_source_file_count"]
            == numerical.python_runtime_source_file_count
        )
        assert receipt_payload["stage"] == "F40-S"
        assert (
            receipt_payload["generator_source_root_sha256"]
            == policy.generator_source_root_sha256
        )
        assert (
            receipt_payload["qualification_source_sha256"]
            == qualification_source_sha256("F40-S")
        )
        assert receipt_payload["dataset_content_roots_sha256"] == [
            numerical.evidence_a.dataset_content_root_sha256,
            numerical.evidence_b.dataset_content_root_sha256,
        ]
        assert len(set(receipt_payload["actual_matlab_environment_sha256"])) == 2
        assert receipt_payload["qualification_host_id"] == [
            "reference-host",
            "candidate-host",
        ]
        assert len(
            set(receipt_payload["qualification_host_diagnostic_sha256"])
        ) == 2
        receipt_bytes = receipt.read_bytes()
        assert (
            comparison.main(
                [
                    str(a),
                    str(b),
                    "--accept-numerical",
                    "--receipt",
                    str(receipt),
                ]
            )
            == 2
        )
        assert receipt.read_bytes() == receipt_bytes
        print(
            "  [PASS] acceptance receipt binds comparator/runtime/lock/"
            "source/environment/payload/stage"
        )
        _expect_invalid(
            "receipt with stale environment-lock bytes",
            lambda: comparison._receipt_payload(
                replace(numerical, environment_lock_sha256="0" * 64),
                accepted=True,
            ),
        )
        _expect_invalid(
            "receipt with stale Python runtime source root",
            lambda: comparison._receipt_payload(
                replace(
                    numerical,
                    python_runtime_source_root_sha256="0" * 64,
                ),
                accepted=True,
            ),
        )
        forged_parser = dict(numerical.parser_environment)
        forged_parser["scipy"] = "0.0.0"
        _expect_invalid(
            "receipt with stale parser environment",
            lambda: comparison._receipt_payload(
                replace(numerical, parser_environment=forged_parser),
                accepted=True,
            ),
        )
        _expect_invalid(
            "receipt with stale Python runtime source file count",
            lambda: comparison._receipt_payload(
                replace(
                    numerical,
                    python_runtime_source_file_count=(
                        numerical.python_runtime_source_file_count + 1
                    ),
                ),
                accepted=True,
            ),
        )
        stale_evidence_bindings = (
            ("campaign_matlab_release", "R2099a"),
            (
                "campaign_matlab_environment_descriptor",
                "release=R2099a",
            ),
            ("campaign_matlab_environment_sha256", "0" * 64),
            ("gen_schema", "foreign-schema"),
            ("channel_schema_id", "legacy_virtual8"),
            ("generation_behavior_version", "foreign-behavior"),
            ("generator_source_root_sha256", "0" * 64),
            (
                "generator_source_file_count",
                numerical.evidence_a.generator_source_file_count + 1,
            ),
            (
                "max_parfor_workers",
                numerical.evidence_a.max_parfor_workers + 1,
            ),
            ("qualification_source_sha256", "0" * 64),
            ("qualification_executed_file_sha256", "0" * 64),
        )
        for field, value in stale_evidence_bindings:
            forged_evidence = replace(
                numerical.evidence_a, **{field: value}
            )
            _expect_invalid(
                f"receipt with stale evidence binding {field}",
                lambda forged=forged_evidence: comparison._receipt_payload(
                    replace(numerical, evidence_a=forged),
                    accepted=True,
                ),
            )
        _expect_invalid(
            "receipt rejects forged ComparisonResult tolerance",
            lambda: comparison._receipt_payload(
                replace(numerical, tolerance_rtol=1e-3),
                accepted=True,
            ),
        )

        original_endpoint_reassert = comparison._reassert_endpoint_snapshot
        receipt_reassertions: list[Path] = []

        def recording_endpoint_reassert(
            path: Path,
            evidence: object,
            current_policy: comparison.CurrentPolicy,
        ) -> None:
            receipt_reassertions.append(Path(path))
            original_endpoint_reassert(path, evidence, current_policy)

        comparison._reassert_endpoint_snapshot = recording_endpoint_reassert
        try:
            comparison._receipt_payload(numerical, accepted=True)
        finally:
            comparison._reassert_endpoint_snapshot = original_endpoint_reassert
        assert receipt_reassertions == [Path(a), Path(b)]
        print("  [PASS] receipt construction reasserts both endpoints")

        original_current_policy = comparison._current_policy
        policy_calls = {"count": 0}

        def drifting_current_policy() -> comparison.CurrentPolicy:
            observed = original_current_policy()
            policy_calls["count"] += 1
            if policy_calls["count"] >= 2:
                return replace(
                    observed,
                    generator_source_root_sha256="0" * 64,
                )
            return observed

        comparison._current_policy = drifting_current_policy
        try:
            _expect_invalid(
                "receipt construction reasserts current policy at the end",
                lambda: comparison._receipt_payload(
                    numerical, accepted=True
                ),
            )
        finally:
            comparison._current_policy = original_current_policy
        assert policy_calls["count"] >= 2
        _expect_invalid(
            "looser-than-reviewed tolerance",
            lambda: comparison.compare_directories(a, b, rtol=1e-3),
        )

        def large_signal(data: dict) -> None:
            data["AcelWheelsetPrimVag"][0][0, 0] += 1e-3

        a, b = _pair(root, "large_signal", mutate_b=large_signal)
        material = _expect_verdict(
            "out-of-tolerance solver difference",
            "MATERIALLY-DIFFERENT",
            lambda: comparison.compare_directories(a, b),
        )
        assert comparison.main([str(a), str(b), "--accept-numerical"]) == 1
        material_receipt = root / "material-receipt.json"
        assert (
            comparison.main(
                [
                    str(a),
                    str(b),
                    "--accept-numerical",
                    "--receipt",
                    str(material_receipt),
                ]
            )
            == 1
        )
        material_receipt_payload = json.loads(
            material_receipt.read_text(encoding="utf-8")
        )
        assert material_receipt_payload["verdict"] == "MATERIALLY-DIFFERENT"
        assert (
            material_receipt_payload[
                "numerical_equivalence_explicitly_accepted"
            ]
            is False
        )
        _expect_invalid(
            "material verdict cannot claim numerical-equivalence acceptance",
            lambda: comparison._receipt_payload(material, accepted=True),
        )

        def random_input_drift(data: dict) -> None:
            # Move by one representable float so this exact-input probe cannot
            # become vacuous when the fixture velocity is large enough that a
            # fixed decimal increment rounds back to the original value.
            before = data["Velocidade"].copy()
            data["Velocidade"][0] = np.nextafter(
                data["Velocidade"][0], np.inf
            )
            assert not np.array_equal(before, data["Velocidade"])
            assert before.tobytes() != data["Velocidade"].tobytes()

        a, b = _pair(root, "random_drift", mutate_b=random_input_drift)
        saved_velocity_a = np.asarray(
            _fixture_mat(a / "0001.mat")["data"]["Velocidade"]
        )
        saved_velocity_b = np.asarray(
            _fixture_mat(b / "0001.mat")["data"]["Velocidade"]
        )
        assert not np.array_equal(saved_velocity_a, saved_velocity_b)
        assert saved_velocity_a.tobytes() != saved_velocity_b.tobytes()
        random_drift_result = _expect_verdict(
            "tiny random-input drift remains exact",
            "MATERIALLY-DIFFERENT",
            lambda: comparison.compare_directories(a, b),
        )
        expected_velocity_mismatches = [
            f"{state_index:04d}.mat.data.Velocidade: "
            "exact numerical values differ"
            for state_index in range(
                1, len(random_drift_result.evidence_a.state_files) + 1
            )
        ]
        assert (
            random_drift_result.stats.mismatches
            == expected_velocity_mismatches
        )

        def inject_disabled_track(data: dict) -> None:
            data["track_log"][0] = {
                "ballast_patches": np.array(
                    [[12.5, 16.5, 0.55, 0.60]], dtype=float
                ),
                "hanging_groups": np.zeros((0, 2), dtype=float),
                "pad_stiff_mult": 1.0,
                "pad_damp_mult": 1.0,
                "pad_failures": np.zeros((0,), dtype=float),
                "x_bridge_local": 30.0,
            }

        a, b = _pair(
            root,
            "disabled_track_payload",
            mutate_a=inject_disabled_track,
            mutate_b=inject_disabled_track,
            stage="F40-M",
        )
        _expect_invalid(
            "same forbidden track payload in both disabled-mechanism inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def inject_disabled_oor(data: dict) -> None:
            data["oor_log"][0] = {
                "flats": np.zeros((0, 5), dtype=float),
                "poly": np.array(
                    [[1.0, 1.0, 8.0, 2.5e-5, 0.75]], dtype=float
                ),
            }

        a, b = _pair(
            root,
            "disabled_oor_payload",
            mutate_a=inject_disabled_oor,
            mutate_b=inject_disabled_oor,
            stage="L99-M",
        )
        _expect_invalid(
            "same forbidden OOR payload in both disabled-mechanism inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_dim_acel(data: dict) -> None:
            data["DimAcel"][0] = 5

        a, b = _pair(root, "wrong_dim_acel", mutate_b=wrong_dim_acel)
        _expect_invalid(
            "RAW signal length disagrees with DimAcel",
            lambda: comparison.compare_directories(a, b),
        )

        def short_dim_space(data: dict) -> None:
            data["DimSpace"][0] = data["crop_end"][0] - 1

        a, b = _pair(root, "short_dim_space", mutate_b=short_dim_space)
        _expect_invalid(
            "RAW crop exceeds DimSpace",
            lambda: comparison.compare_directories(a, b),
        )

        def fractional_crop(data: dict) -> None:
            data["crop_start"] = data["crop_start"].astype(float)
            data["crop_start"][0] += 0.5

        a, b = _pair(root, "fractional_crop", mutate_b=fractional_crop)
        _expect_invalid(
            "fractional RAW crop index",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_dim_space_equation(data: dict) -> None:
            data["DimSpace"][0] += 1

        a, b = _pair(
            root,
            "wrong_dim_space_equation",
            mutate_b=wrong_dim_space_equation,
        )
        _expect_invalid(
            "RAW DimSpace formula mutation",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_registered_crop_start(data: dict) -> None:
            data["crop_start"][0] = 1000

        a, b = _pair(
            root,
            "wrong_registered_crop_start",
            mutate_b=wrong_registered_crop_start,
        )
        _expect_invalid(
            "RAW registered crop-start mutation",
            lambda: comparison.compare_directories(a, b),
        )

        def fatal_contact(data: dict) -> None:
            data["contact_log"][0, 1] = 1.0
            data["contact_log"][0, 2] = (
                policy.contact_fraction_tolerance / 2.0
            )
            data["contact_log"][0, 3] = policy.contact_force_tolerance_N + 1

        a, b = _pair(root, "fatal_contact", mutate_b=fatal_contact)
        _expect_invalid(
            "contact gate violation",
            lambda: comparison.compare_directories(a, b),
        )

        def negative_compression(data: dict) -> None:
            data["contact_log"][:, 3] = -87913.0

        a, b = _pair(
            root,
            "negative_compression",
            mutate_a=negative_compression,
            mutate_b=negative_compression,
        )
        _expect_verdict(
            "signed negative compression remains admissible",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )

        def inconsistent_contact_flag(data: dict) -> None:
            data["contact_log"][0, 3] = 1.0

        a, b = _pair(
            root,
            "inconsistent_contact_flag",
            mutate_b=inconsistent_contact_flag,
        )
        _expect_invalid(
            "positive signed reaction without flag/fraction",
            lambda: comparison.compare_directories(a, b),
        )

        def missing_passage_seed(data: dict) -> None:
            data.pop("passage_named_stream_seed_id")

        a, b = _pair(
            root,
            "missing_passage_seed",
            mutate_b=missing_passage_seed,
        )
        _expect_invalid(
            "missing per-state passage seed matrix",
            lambda: comparison.compare_directories(a, b),
        )

        def extra_data_field(data: dict) -> None:
            data["unexpected_scientific_alias"] = 1.0

        a, b = _pair(
            root,
            "extra_data_field",
            mutate_b=extra_data_field,
        )
        _expect_invalid(
            "unexpected nested state field",
            lambda: comparison.compare_directories(a, b),
        )

        def extra_top_field(state: dict) -> None:
            state["unexpected_top_alias"] = "forged"

        a, b = _pair(
            root,
            "extra_top_field",
            mutate_state_b=extra_top_field,
        )
        _expect_invalid(
            "unexpected top-level state field",
            lambda: comparison.compare_directories(a, b),
        )

        def unit_bearing_fixity(data: dict) -> None:
            data["bearing_fixity"][0] = 1.0

        a, b = _pair(
            root,
            "unit_bearing_fixity",
            mutate_b=unit_bearing_fixity,
        )
        _expect_invalid(
            "bearing_fixity equal to one",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "numeric_state_alias")
        shutil.copy2(a / "0001.mat", a / "1.mat")
        _expect_invalid(
            "numeric state filename alias",
            lambda: comparison.compare_directories(a, b),
        )

        def impossible_deck_frequency(data: dict) -> None:
            data["beam_f1_Hz"] = 100.0

        a, b = _pair(
            root, "impossible_deck_frequency", mutate_b=impossible_deck_frequency
        )
        _expect_invalid(
            "deck frequency outside global generator gate",
            lambda: comparison.compare_directories(a, b),
        )

        def unhealthy_healthy_frequency(data: dict) -> None:
            data["beam_f1_Hz"] = 6.5

        a, b = _pair(
            root,
            "unhealthy_healthy_frequency",
            mutate_b=unhealthy_healthy_frequency,
        )
        _expect_invalid(
            "healthy F40 deck frequency outside audited gate",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_family(data: dict) -> None:
            data["state_family"] = "joint"

        a, b = _pair(root, "wrong_family", mutate_b=wrong_family)
        _expect_invalid(
            "per-state family disagrees with damage table",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_label(data: dict) -> None:
            data["scour_vector"][0] += 0.01
            data["Dano"] += 0.01

        a, b = _pair(root, "wrong_label", mutate_b=wrong_label)
        _expect_invalid(
            "per-state label disagrees with damage table",
            lambda: comparison.compare_directories(a, b),
        )

        def missing_signal(data: dict) -> None:
            del data["AcelPrimVag"]

        a, b = _pair(root, "missing_signal", mutate_b=missing_signal)
        _expect_invalid(
            "missing nested signal",
            lambda: comparison.compare_directories(a, b),
        )

        def integer_signal(data: dict) -> None:
            data["AcelPrimVag"][0] = data["AcelPrimVag"][0].astype(np.int64)

        a, b = _pair(root, "integer_signal", mutate_b=integer_signal)
        _expect_invalid(
            "non-floating acceleration passage",
            lambda: comparison.compare_directories(a, b),
        )

        def nan_signal(data: dict) -> None:
            data["AcelPrimVag"][0][0, 0] = np.nan

        a, b = _pair(root, "nan_signal", mutate_b=nan_signal)
        _expect_invalid(
            "non-finite nested signal",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "six_passages", n_passages=6)
        _expect_invalid(
            "qualification with six passages/state",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "thirty_two_states", n_states=32)
        _expect_invalid(
            "F40-S qualification with 32 states",
            lambda: comparison.compare_directories(a, b),
        )

        forged = _environment("R2023a", variant="forged-campaign-policy")
        a, b = _pair(
            root,
            "forged_policy",
            release_a="R2023a",
            release_b="R2023b",
            campaign_environment=forged,
        )
        _expect_invalid(
            "input-forged campaign environment policy",
            lambda: comparison.compare_directories(a, b),
        )

        same_actual = _campaign_environment()
        a, b = _pair(
            root,
            "same_environment",
            release_b=same_actual.release,
            actual_environment_b=same_actual,
        )
        same_environment_result = _expect_verdict(
            "same MATLAB environment on independently identified hosts",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )
        assert (
            same_environment_result.evidence_a.qualification_host_id
            != same_environment_result.evidence_b.qualification_host_id
        )

        a, b = _pair(
            root,
            "same_environment_same_host",
            release_b=same_actual.release,
            actual_environment_b=same_actual,
            host_id_b="reference-host",
        )
        _expect_invalid(
            "same MATLAB environment and same declared host",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "missing_host_receipt")
        (b / "qualification_host_receipt.json").unlink()
        _expect_invalid(
            "missing qualification host receipt",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "mutated_host_diagnostic")
        host_path = b / "qualification_host_receipt.json"
        host_receipt = json.loads(host_path.read_text(encoding="utf-8"))
        host_receipt["cpu_identifier"] = "post-receipt-mutation"
        host_path.write_text(
            json.dumps(host_receipt, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _expect_invalid(
            "host diagnostic changed without canonical hash update",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "noncanonical_host_receipt_bytes")
        host_path = b / "qualification_host_receipt.json"
        host_receipt = json.loads(host_path.read_text(encoding="utf-8"))
        host_path.write_text(
            json.dumps(host_receipt, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _expect_invalid(
            "host receipt rejects semantically equal noncanonical JSON bytes",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "wrong_host_environment_binding")
        host_path = b / "qualification_host_receipt.json"
        host_receipt = json.loads(host_path.read_text(encoding="utf-8"))
        old_sha = host_receipt["actual_matlab_environment_sha256"]
        host_receipt["actual_matlab_environment_sha256"] = "0" * 64
        host_receipt["canonical_descriptor"] = host_receipt[
            "canonical_descriptor"
        ].replace(old_sha, "0" * 64)
        host_receipt["host_diagnostic_sha256"] = hashlib.sha256(
            host_receipt["canonical_descriptor"].encode("utf-8")
        ).hexdigest()
        host_path.write_text(
            json.dumps(host_receipt, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _expect_invalid(
            "host receipt rebound to wrong MATLAB environment",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "no_locked_reference",
            release_a="R2023a",
            release_b="R2023b",
        )
        _expect_invalid(
            "two candidates without locked campaign reference",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "bad_source_root",
            mutate_manifest_b=lambda manifest: manifest.__setitem__(
                "generator_source_root_sha256", "0" * 64
            ),
        )
        _expect_invalid(
            "manifest source root differs from current generator bytes",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "bad_channel_schema",
            mutate_manifest_b=lambda manifest: manifest.__setitem__(
                "channel_schema_id", "legacy_virtual8"
            ),
        )
        _expect_invalid(
            "manifest channel schema differs from physical8_v1",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "mixed_payload_channel_schema",
            mutate_b=lambda data: data.__setitem__(
                "channel_schema_id", "legacy_virtual8"
            ),
        )
        _expect_invalid(
            "per-state channel schema differs from manifest",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "bad_qualification_source",
            mutate_manifest_b=lambda manifest: manifest.__setitem__(
                "qualification_source_sha256", "0" * 64
            ),
        )
        _expect_invalid(
            "manifest qualification source is stale/forged",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "missing_top_env",
            mutate_state_b=lambda state: state.pop(
                "file_actual_matlab_environment_sha256"
            ),
        )
        _expect_invalid(
            "missing top-level actual-environment stamp",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "wrong_top_state_uid",
            mutate_state_b=lambda state: state.__setitem__(
                "file_state_uid", "restamped-wrong-state"
            ),
        )
        _expect_invalid(
            "wrong top-level semantic state UID",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "wrong_top_state_seed",
            mutate_state_b=lambda state: state.__setitem__(
                "file_state_seed_id", np.uint32(1)
            ),
        )
        _expect_invalid(
            "wrong top-level semantic state seed",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "uint64_top_state_seed",
            mutate_state_b=lambda state: state.__setitem__(
                "file_state_seed_id",
                np.uint64(state["file_state_seed_id"]),
            ),
        )
        _expect_invalid(
            "top-level state seed requires scalar MATLAB uint32 dtype",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "uint64_nested_state_seed",
            mutate_b=lambda data: data.__setitem__(
                "state_seed_id", np.uint64(data["state_seed_id"])
            ),
        )
        _expect_invalid(
            "nested state seed requires scalar MATLAB uint32 dtype",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_nested_named_stream(data: dict) -> None:
            values = np.asarray(
                data["state_named_stream_seed_id"], dtype=np.uint32
            ).copy()
            values[0] = np.uint32((int(values[0]) + 1) % (2**32))
            data["state_named_stream_seed_id"] = values

        a, b = _pair(
            root,
            "wrong_nested_named_stream",
            mutate_b=wrong_nested_named_stream,
        )
        _expect_invalid(
            "wrong nested named-stream row",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "wrong_top_source",
            mutate_state_b=lambda state: state.__setitem__(
                "file_generator_source_root_sha256", "0" * 64
            ),
        )
        _expect_invalid(
            "wrong top-level generator-source stamp",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_nested_environment(data: dict) -> None:
            data["actual_matlab_environment_sha256"] = "0" * 64

        a, b = _pair(
            root, "wrong_nested_environment", mutate_b=wrong_nested_environment
        )
        _expect_invalid(
            "wrong nested actual-environment stamp",
            lambda: comparison.compare_directories(a, b),
        )

        def wrong_nested_source(data: dict) -> None:
            data["generator_source_root_sha256"] = "0" * 64

        a, b = _pair(root, "wrong_nested_source", mutate_b=wrong_nested_source)
        _expect_invalid(
            "wrong nested generator-source stamp",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "unmarked", qualify_b=False)
        _expect_invalid(
            "qualification marker false",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "post_stamp_executable_mutation")
        for dataset in (a, b):
            executable = dataset / "qualification_executed.m"
            payload = executable.read_bytes()
            executable.write_bytes(
                payload.replace(
                    b"% Main script to run TTB-2D model",
                    b"% post-stamp executable mutation",
                    1,
                )
            )
        _expect_invalid(
            "same post-stamp executable-byte mutation in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "qualification_executable_symlink_guard")
        original_is_symlink = Path.is_symlink

        def mark_qualification_as_symlink(candidate: Path) -> bool:
            if candidate.name == "qualification_executed.m":
                return True
            return original_is_symlink(candidate)

        with patch.object(Path, "is_symlink", mark_qualification_as_symlink):
            _expect_invalid(
                "qualification executable symlink guard",
                lambda: comparison.compare_directories(a, b),
            )

        digest_mutations: tuple[
            tuple[str, Callable[[dict], None]], ...
        ] = (
            (
                "digest struct extra field",
                lambda value: value.__setitem__("unreviewed", "laundered"),
            ),
            (
                "legacy digest schema",
                lambda value: value.__setitem__("schema", "source-digests-v1"),
            ),
            (
                "padded digest schema",
                lambda value: value.__setitem__(
                    "schema", " source-digests-v2"
                ),
            ),
            (
                "state-only digest scope",
                lambda value: value.__setitem__("scope", "NNNN.mat"),
            ),
            (
                "uppercase digest",
                lambda value: _set_digest_lines(
                    value,
                    "\n".join(
                        [
                            (
                                row.split(":", 1)[0]
                                + ":"
                                + row.split(":", 1)[1].upper()
                                if index == 0
                                else row
                            )
                            for index, row in enumerate(
                                str(value["digest_lines"]).split("\n")
                            )
                        ]
                    ),
                ),
            ),
            (
                "unsorted digest rows",
                lambda value: _set_digest_lines(
                    value,
                    "\n".join(
                        reversed(str(value["digest_lines"]).split("\n"))
                    ),
                ),
            ),
            (
                "padded digest row",
                lambda value: _set_digest_lines(
                    value,
                    str(value["digest_lines"]).replace(":", " :", 1),
                ),
            ),
            (
                "case-colliding digest row",
                lambda value: _set_digest_lines(
                    value,
                    str(value["digest_lines"])
                    + "\nCASE_INFO.mat:"
                    + str(value["digest_lines"]).split(
                        "case_info.mat:", 1
                    )[1].split("\n", 1)[0],
                ),
            ),
            (
                "duplicate digest row",
                lambda value: _set_digest_lines(
                    value,
                    str(value["digest_lines"])
                    + "\n"
                    + str(value["digest_lines"]).split("\n", 1)[0],
                ),
            ),
        )
        for name, mutation in digest_mutations:
            a, b = _pair(root, f"digest_{name.replace(' ', '_')}")
            for dataset in (a, b):
                _rewrite_digest_table(dataset, mutation)
            _expect_invalid(
                f"same coherent {name} in both inputs",
                lambda a=a, b=b: comparison.compare_directories(a, b),
            )

        a, b = _pair(root, "digest_extra_inventory")
        for dataset in (a, b):
            extra = dataset / "extra.mat"
            extra.write_bytes(b"not a reviewed qualification input")

            def add_extra(value: dict, *, extra: Path = extra) -> None:
                rows = str(value["digest_lines"]).split("\n")
                rows.append(f"extra.mat:{_sha256(extra)}")
                _set_digest_lines(value, "\n".join(sorted(rows)))

            _rewrite_digest_table(dataset, add_extra)
        _expect_invalid(
            "same coherent extra digest inventory in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "digest_extra_top_level")
        for dataset in (a, b):
            value = _fixture_mat(dataset / "file_digests.mat")[
                "file_digests"
            ]
            savemat(
                dataset / "file_digests.mat",
                {"file_digests": value, "unreviewed": 1},
                long_field_names=True,
            )
        _expect_invalid(
            "same extra file_digests MAT variable in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "digest_nonscalar_struct")
        for dataset in (a, b):
            value = _fixture_mat(dataset / "file_digests.mat")[
                "file_digests"
            ]
            savemat(
                dataset / "file_digests.mat",
                {
                    "file_digests": np.array(
                        [deepcopy(value), deepcopy(value)], dtype=object
                    )
                },
                long_field_names=True,
            )
        _expect_invalid(
            "same nonscalar file_digests struct in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "digest_nonregular")
        for dataset in (a, b):
            digest_path = dataset / "file_digests.mat"
            digest_path.unlink()
            digest_path.mkdir()
        _expect_invalid(
            "same nonregular file_digests.mat in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "digest_symlink_guard")
        original_is_symlink = Path.is_symlink

        def mark_digest_as_symlink(candidate: Path) -> bool:
            if candidate.name == "file_digests.mat":
                return True
            return original_is_symlink(candidate)

        with patch.object(Path, "is_symlink", mark_digest_as_symlink):
            _expect_invalid(
                "file_digests symlink guard",
                lambda: comparison.compare_directories(a, b),
            )

        hardlink_members = (
            "file_digests.mat",
            "case_info.mat",
            "damage_states.mat",
            "0001.mat",
            "qualification_executed.m",
            "qualification_host_receipt.json",
            "_GENERATION_COMPLETE",
        )
        for index, member in enumerate(hardlink_members):
            a, _b = _pair(
                root,
                f"hardlink_guard_{index}_{member.replace('.', '_')}",
            )
            target = a / member
            alias = a / f"hardlink-alias-{index}.bin"
            os.link(target, alias)
            _expect_invalid_from(
                f"hard-link alias rejected for {member}",
                "single-link",
                lambda a=a: comparison._validated_payload(a, policy),
            )

        a, b = _pair(root, "tampered")
        with (b / "0001.mat").open("ab") as handle:
            handle.write(b"tamper")
        _expect_invalid(
            "post-digest state tamper",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "marker_tampered")
        with (b / "_GENERATION_COMPLETE").open("ab") as handle:
            handle.write(b"unexpected-fourth-line\n")
        _expect_invalid(
            "completion marker extension",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "marker_blank_extension")
        with (b / "_GENERATION_COMPLETE").open("ab") as handle:
            handle.write(b"\n")
        _expect_invalid(
            "completion marker blank-line extension",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "partial")
        (b / "0001.mat").unlink()
        _expect_invalid(
            "partial state inventory",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(
            root,
            "unsafe_worker_cap",
            mutate_manifest_b=lambda manifest: manifest.__setitem__(
                "max_parfor_workers", 16
            ),
        )
        _expect_invalid(
            "unsafe generation worker cap",
            lambda: comparison.compare_directories(a, b),
        )

        _intra_invocation_mutation_checks(root)

    _micro_patch_checks()
    print("GENERATION ENVIRONMENT COMPARISON: ALL ADVERSARIAL CHECKS PASS")


if __name__ == "__main__":
    main()
