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
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
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
    MICRO_DS,
    qualification_source_sha256,
    render_micro_a00,
    render_toy_driver,
    verify_qualification_source_bytes,
)


Mutation = Callable[[dict], None]


@dataclass(frozen=True)
class MatlabEnvironment:
    release: str
    descriptor: str
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    qualification_sha: str,
    generation_fingerprint: str,
) -> dict:
    acel_prim = np.empty(n_passages, dtype=object)
    acel_roda = np.empty(n_passages, dtype=object)
    pitch = np.empty(n_passages, dtype=object)
    for passage in range(n_passages):
        acel_prim[passage] = (
            np.arange(12.0).reshape(3, 4) + passage * 0.125
        )
        acel_roda[passage] = (
            np.arange(16.0).reshape(4, 4) - passage * 0.25
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

    return {
        "AcelPrimVag": acel_prim,
        "AcelRodaPrimVag": acel_roda,
        "PitchPrimVag": pitch,
        # Four time samples in each signal: DimAcel must be four, not dt.
        "DimAcel": np.full(n_passages, 4, dtype=np.int64),
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
        "latent_bearing_fixity":
            np.asarray(latent_bearing_fixity, dtype=float),
        "latent_crack_on": bool(latent_crack_on),
        "crack_on": bool(crack_on),
        "beam_f1_Hz": 4.2 if bridge_length < 80 else 2.8,
        "bearing_vector": np.asarray(bearing_vector, dtype=float),
        "bearing_fixity": np.asarray(bearing_vector, dtype=float),
        "crack_log": crack_log,
        "profile_mode": profile_mode,
        "profile_log": profile_log,
        "track_log": track_log,
        "oor_log": oor_log,
        "contact_log": np.zeros((n_passages, 4), dtype=float),
        "Temperatura": np.linspace(5.0, 25.0, n_passages),
        "Velocidade": np.linspace(70.0, 90.0, n_passages),
        "VehiclesProps": np.arange(15.0).reshape(5, 3),
        "gen_schema": policy.gen_schema,
        "gen_fingerprint": generation_fingerprint,
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
    stage: str = "s0_scour",
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
    qualification_sha = qualification_source_sha256(stage)
    source_path = Path(__file__).resolve().parent / "scour_MATLAB" / "A00_Run.m"
    qualification_executable = render_micro_a00(
        source_path.read_text(encoding="utf-8"),
        qualification=True,
        stage=stage,
    ).encode("utf-8")
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
    damage_states = {
        "DamageStates": labels["damage"],
        "BearingStates": labels["bearing"],
        "BearingFixity": labels["bearing"],
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
        "k_ref_bear": 1.0e8,
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
    (path / "_GENERATION_COMPLETE").write_text(
        f"{policy.gen_schema}\n{generation_fingerprint}\n{dataset_root}\n",
        encoding="utf-8",
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
    stage: str = "s0_scour",
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
        comparison._public_mat(path / "file_digests.mat")["file_digests"]
    )
    mutate(value)
    savemat(
        path / "file_digests.mat",
        {"file_digests": value},
        long_field_names=True,
    )
    marker_lines = (path / "_GENERATION_COMPLETE").read_text(
        encoding="utf-8"
    ).splitlines()
    marker_lines[2] = str(value["root"])
    (path / "_GENERATION_COMPLETE").write_text(
        "\n".join(marker_lines) + "\n",
        encoding="utf-8",
    )


def _set_digest_lines(value: dict, lines: str) -> None:
    value["digest_lines"] = lines
    value["root"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _micro_patch_checks() -> None:
    source_path = (
        Path(__file__).resolve().parent / "scour_MATLAB" / "A00_Run.m"
    )
    source = source_path.read_text(encoding="utf-8")
    expected = qualification_source_sha256("s16_all")
    assert len(expected) == 64
    qualification = render_micro_a00(
        source, qualification=True, stage="s16_all"
    )
    assert "STAGE = 's16_all';" in qualification
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
    assert "local_qualification_script_identity(" in qualification
    assert "qualification_executed.m" in qualification
    assert "TTBI_QUALIFICATION_HOST_ID" in qualification
    assert "qualification_host_receipt.json" in qualification
    assert "host_diagnostic_sha256" in qualification
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
    driver_source = (
        Path(__file__).resolve().parent
        / "comprehensive_ablation_multidamage.py"
    ).read_text(encoding="utf-8")
    toy_driver = render_toy_driver(driver_source)
    compile(toy_driver, "generated_micro_dryrun.py", "exec")
    assert (
        toy_driver.count(
            f'DATASET = "{MICRO_DS}"  # GENERATED MICRO-DRYRUN OVERRIDE'
        )
        == 1
    )
    assert (
        "return _micro_read_dataset_provenance(dataset_dir)" in toy_driver
    )
    toy_config = toy_driver[
        toy_driver.index("def make_config"):
        toy_driver.index("def run_phase")
    ]
    assert (
        toy_config.count(
            '"qualification_mode": "toy_nonpublication_legacy"'
        )
        == 1
    )
    for forbidden in (
        '"protocol_hash"',
        '"protocol_core_hash"',
        '"protocol_descriptor"',
        '"execution_runtime"',
        '"hyperparameter_mode"',
        '"hyperparameter_manifest"',
    ):
        assert forbidden not in toy_config
    assert "N_TRIALS       = 2            # DRY-RUN" in toy_driver
    assert "EPOCHS         = 2            # DRY-RUN" in toy_driver
    assert "USE_PRUNER     = False        # DRY-RUN" in toy_driver
    assert "SEEDS          = [42]         # DRY-RUN" in toy_driver
    assert (
        "*** TOY NON-PUBLICATION LEGACY MECHANICS RUN ***"
        in toy_driver
    )
    print(
        "  [PASS] toy driver is an explicit 2-trial/1-seed LEGACY mechanics "
        "fixture and carries no campaign/HPO/runtime identity"
    )
    broken_assignment = driver_source.replace(
        "DATASET, TARGET_SUPPORTS, BEARING_TARGETS = _LADDER[STAGE]",
        "DATASET, TARGET_SUPPORTS = _LADDER[STAGE]",
        1,
    )
    try:
        render_toy_driver(broken_assignment)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "toy-driver dataset override survived removal of its semantic anchor"
        )
    print(
        "  [PASS] toy driver overrides the resolved campaign-contract dataset "
        "without a stale literal and fails when its semantic anchor is mutated"
    )


def main() -> None:
    print("GENERATION ENVIRONMENT COMPARISON CHECKS")
    with tempfile.TemporaryDirectory(prefix="environment-comparison-check-") as tmp:
        root = Path(tmp)
        policy = comparison._current_policy()
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
                f"exact ten-rung policy: {stage}",
                "SEMANTICALLY-BIT-IDENTICAL",
                lambda a=a, b=b: comparison.compare_directories(a, b),
            )
            expected_policy = comparison._STAGE_POLICIES[stage]
            assert result.evidence_a.n_states == expected_policy["n_states"]
            assert result.evidence_a.passages_per_state == 3
            expected_total = (
                35 if expected_policy["num_spans"] == 3 else 39
            )
            assert expected_policy["n_states"] == expected_total
            assert expected_policy["family_counts"]["bearing_only"] == 8
            assert expected_policy["family_counts"]["nuisance_only"] == 6
            assert expected_policy["family_counts"]["joint"] == 10

        coherent_policy_mutations = (
            ("bridge length", "s0_scour", "L_bridge_m", 61.0),
            ("span count", "s0_scour", "num_spans", 4),
            ("support count", "s0_scour", "num_supports", 5),
            ("bearing mode", "s11_bear", "bearing_mode", "off"),
            ("crack toggle", "s16_all", "use_crack_eov", False),
            ("track toggle", "s16_all", "use_track_eov", False),
            ("OOR toggle", "s16_all", "use_oor_eov", False),
            ("profile mode", "s16_all", "profile_mode", "fixed"),
            ("flat policy", "s16_all", "oor_flats_enabled", True),
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
            stage="s11_bear",
            mutate_damage_table_a=active_latent_bearing_disagreement,
            mutate_damage_table_b=active_latent_bearing_disagreement,
        )
        _expect_invalid(
            "same active/latent bearing disagreement in both inputs",
            lambda: comparison.compare_directories(a, b),
        )

        def crack_activation_policy_drift(table: dict) -> None:
            values = np.asarray(table["CrackOn"], dtype=bool).copy()
            values[0] = True
            table["CrackOn"] = values

        a, b = _pair(
            root,
            "crack_activation_policy_drift",
            stage="s12_crack",
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
            data["AcelPrimVag"][0][0, 1] += 5e-11

        a, b = _pair(root, "small_signal", mutate_b=small_signal)
        numerical = _expect_verdict(
            "in-tolerance solver difference",
            "NUMERICALLY-EQUIVALENT",
            lambda: comparison.compare_directories(a, b),
        )
        assert numerical.default_exit_code == 3
        assert comparison.main([str(a), str(b)]) == 3
        assert comparison.main([str(a), str(b), "--accept-numerical"]) == 2
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
        assert receipt_payload["stage"] == "s0_scour"
        assert (
            receipt_payload["generator_source_root_sha256"]
            == policy.generator_source_root_sha256
        )
        assert (
            receipt_payload["qualification_source_sha256"]
            == qualification_source_sha256("s0_scour")
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
            "looser-than-reviewed tolerance",
            lambda: comparison.compare_directories(a, b, rtol=1e-3),
        )

        def large_signal(data: dict) -> None:
            data["AcelRodaPrimVag"][0][0, 0] += 1e-3

        a, b = _pair(root, "large_signal", mutate_b=large_signal)
        _expect_verdict(
            "out-of-tolerance solver difference",
            "MATERIALLY-DIFFERENT",
            lambda: comparison.compare_directories(a, b),
        )
        assert comparison.main([str(a), str(b), "--accept-numerical"]) == 1

        def random_input_drift(data: dict) -> None:
            data["Velocidade"][0] += 1e-14

        a, b = _pair(root, "random_drift", mutate_b=random_input_drift)
        _expect_verdict(
            "tiny random-input drift remains exact",
            "MATERIALLY-DIFFERENT",
            lambda: comparison.compare_directories(a, b),
        )

        def nested_drift(data: dict) -> None:
            data["track_log"][0]["ballast_patches"][0, 2] += 0.01

        a, b = _pair(
            root, "nested_drift", mutate_b=nested_drift, stage="s16_all"
        )
        _expect_verdict(
            "nested nuisance mutation",
            "MATERIALLY-DIFFERENT",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "mechanism_coverage", stage="s16_all")
        covered = _expect_verdict(
            "all-mechanism aggregate witnesses",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )
        coverage = covered.evidence_a.mechanism_coverage
        assert coverage.required
        assert min(
            coverage.crack_active_passages,
            coverage.profile_active_passages,
            coverage.ballast_patch_rows,
            coverage.hanging_group_rows,
            coverage.pad_departure_passages,
            coverage.polygon_rows,
        ) > 0
        assert set(coverage.witnesses) == {
            "crack",
            "profile",
            "ballast",
            "hanging",
            "pad",
            "polygonization",
        }

        a, b = _pair(root, "mechanism_coverage_4span", stage="s23_all4")
        _expect_verdict(
            "four-span all-mechanism aggregate witnesses",
            "SEMANTICALLY-BIT-IDENTICAL",
            lambda: comparison.compare_directories(a, b),
        )

        def remove_polygon(data: dict) -> None:
            for passage in data["oor_log"]:
                passage["poly"] = np.zeros((0, 5), dtype=float)

        a, b = _pair(
            root,
            "missing_coverage",
            mutate_a=remove_polygon,
            mutate_b=remove_polygon,
            stage="s16_all",
        )
        _expect_invalid(
            "all-mechanism stage without polygonization witness",
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

        def fatal_contact(data: dict) -> None:
            data["contact_log"][0, 3] = policy.contact_force_tolerance_N + 1

        a, b = _pair(root, "fatal_contact", mutate_b=fatal_contact)
        _expect_invalid(
            "contact gate violation",
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
            "healthy L60 deck frequency outside audited gate",
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

        a, b = _pair(root, "sixty_five_states", n_states=65)
        _expect_invalid(
            "qualification with 65 states",
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
            value = comparison._public_mat(
                dataset / "file_digests.mat"
            )["file_digests"]
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
            value = comparison._public_mat(
                dataset / "file_digests.mat"
            )["file_digests"]
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

        a, b = _pair(root, "tampered")
        with (b / "0001.mat").open("ab") as handle:
            handle.write(b"tamper")
        _expect_invalid(
            "post-digest state tamper",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "marker_tampered")
        with (b / "_GENERATION_COMPLETE").open("a", encoding="utf-8") as handle:
            handle.write("unexpected-fourth-line\n")
        _expect_invalid(
            "completion marker extension",
            lambda: comparison.compare_directories(a, b),
        )

        a, b = _pair(root, "marker_blank_extension")
        with (b / "_GENERATION_COMPLETE").open("a", encoding="utf-8") as handle:
            handle.write("\n")
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

    _micro_patch_checks()
    print("GENERATION ENVIRONMENT COMPARISON: ALL ADVERSARIAL CHECKS PASS")


if __name__ == "__main__":
    main()
