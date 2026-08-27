"""Launch the isolated F25-R HPO and ordered F25-X training tiers.

Production entry points::

    python -m training.f25_executor plan --experiment F25-R --output plan.json
    python -m training.f25_executor run-job --experiment F25-R --job-id ...

Every job is looked up in the canonical plan; free-form architecture, sensor,
split, seed, budget, or artifact-root overrides are intentionally unavailable.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.join(_bootstrap_os.path.dirname(__file__), _bootstrap_os.pardir)
)
if _bootstrap_source_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_source_root)
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.environment import (
    load_environment_lock,
    matlab_environment_descriptor,
)
from core.execution_environment import (
    current_execution_environment,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
from core.f25_experiment_contract import (
    CHANNELS,
    EARLY_STOPPING_PATIENCE_EPOCHS,
    HPO_EXECUTION_SEEDS,
    LEARNING_RATE_PLATEAU_FACTOR,
    LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
    MAXIMUM_EPOCHS,
    MINIMUM_LEARNING_RATE,
    PAA_BLOCK_COUNT,
    REPORT_SEEDS,
    SCENARIOS,
    SOURCE_CNN_SEARCH_SPACE,
    TRIMMED_WINDOW_SAMPLES,
    add_source_noise,
    apply_training_minmax,
    build_contract,
    fit_trimmed_training_minmax,
    paa_blocks_of_ten,
    partition_indices,
    partition_sha256,
)
from core.f25_models import build_f25_model, parameter_count
from core.f25_training_contract import (
    EXECUTION_BLOCK_ID,
    F25TrainingContractError,
    build_training_plan,
    canonical_json_sha256,
    job_by_id,
    validate_training_plan,
)
from core.source_provenance import generator_source_root, python_runtime_source_root
from check_f25_capacity import (
    CAPACITY_RECEIPT_ADDRESS_SCHEMA,
    SCHEMA as F25_CAPACITY_SCHEMA,
    capacity_receipt_path,
    capacity_contract_cases,
)


REPO = Path(__file__).resolve().parents[1]
RUN_RECORD_SCHEMA = "f25-training-run-record-v2"
WINNER_SCHEMA = "f25-hpo-winner-v1"
REPORT_SCHEMA = "f25-report-results-v1"
EXECUTION_RECEIPT_SCHEMA = "f25-execution-block-receipt-v2"
BUNDLE_SCHEMA = "f25-dispatch-bundle-v2"
BUNDLE_SOURCE_BINDING_SCHEMA = "f25-bundle-source-binding-v1"


class F25ExecutionError(RuntimeError):
    """Raised when an F25 job cannot prove or execute its exact contract."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise F25ExecutionError(f"stale temporary artifact exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_json_rows(
    rows: list[tuple[str, object]], owner: str
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in rows:
        if key in value:
            raise ValueError(f"{owner} has duplicate JSON key {key!r}")
        value[key] = item
    return value


def _strict_json_object(path: Path, owner: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise F25ExecutionError(f"{owner} is missing or linked: {path}")
    raw = path.read_bytes()

    def unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in rows:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise F25ExecutionError(f"{owner} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise F25ExecutionError(f"{owner} must be one JSON object")
    return parsed, raw


def _bundle_source_binding_descriptor(binding: dict[str, Any]) -> str:
    fields = (
        "schema",
        "source_commit",
        "source_manifest_sha256",
        "bundle_source_root_sha256",
        "generator_source_root_sha256",
        "generator_source_file_count",
        "f25_contract_sha256",
        "profile_asset_sha256",
        "f25_r_bundle_manifest_file_sha256",
        "f25_x_bundle_manifest_file_sha256",
    )
    return "\n".join(f"{field}={binding[field]}" for field in fields)


def _validate_bundle_source_binding(repo: Path = REPO) -> dict[str, Any]:
    """Authenticate both bundle manifests and every extracted source byte."""

    repo = repo.resolve(strict=True)
    source_manifest_path = repo / "bundle_source_files.txt"
    if not source_manifest_path.is_file() or source_manifest_path.is_symlink():
        raise F25ExecutionError("F25 reviewed source manifest is missing or linked")
    source_manifest_raw = source_manifest_path.read_bytes()
    try:
        source_manifest_text = source_manifest_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise F25ExecutionError("F25 source manifest is not UTF-8") from exc
    if "\r" in source_manifest_text or not source_manifest_text.endswith("\n"):
        raise F25ExecutionError("F25 source manifest is not canonical LF text")
    source_names = tuple(
        line
        for line in source_manifest_text.splitlines()
        if line and not line.startswith("#")
    )
    if (
        not source_names
        or source_names != tuple(sorted(source_names))
        or len(source_names) != len(set(source_names))
    ):
        raise F25ExecutionError("F25 source manifest is empty or noncanonical")

    expected_fields = {
        "schema",
        "experiment_id",
        "bundle_name",
        "source_commit",
        "source_manifest_sha256",
        "source_file_count",
        "source_files",
        "source_root_sha256",
        "f25_contract_sha256",
        "training_plan_sha256",
        "generated_artifacts",
        "shared_generation_root",
        "artifact_roots",
        "bundle_manifest_sha256",
    }
    contract = build_contract()
    manifests: dict[str, dict[str, Any]] = {}
    raw_shas: dict[str, str] = {}
    for experiment in ("F25-R", "F25-X"):
        path = repo / f"f25_bundle_manifest.{experiment}.json"
        manifest, raw = _strict_json_object(path, f"{experiment} bundle manifest")
        plan = build_training_plan(experiment)
        recorded_sha = manifest.get("bundle_manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("bundle_manifest_sha256", None)
        if (
            set(manifest) != expected_fields
            or manifest.get("schema") != BUNDLE_SCHEMA
            or manifest.get("experiment_id") != experiment
            or manifest.get("bundle_name") != plan["bundle_name"]
            or not _is_commit_sha1(manifest.get("source_commit"))
            or not _is_sha256(recorded_sha)
            or recorded_sha != canonical_json_sha256(unsigned)
            or manifest.get("f25_contract_sha256")
            != contract["contract_sha256"]
            or manifest.get("source_manifest_sha256")
            != hashlib.sha256(source_manifest_raw).hexdigest()
            or manifest.get("training_plan_sha256") != plan["plan_sha256"]
            or manifest.get("shared_generation_root")
            != plan["shared_generation_root"]
            or manifest.get("artifact_roots")
            != {
                "manifest": plan["manifest_root"],
                "cache": plan["cache_root"],
                "results": plan["results_root"],
            }
        ):
            raise F25ExecutionError(
                f"{experiment} bundle manifest identity is stale or malformed"
            )
        source_files = manifest.get("source_files")
        if (
            not isinstance(source_files, list)
            or len(source_files) != manifest.get("source_file_count")
            or len(source_files) != len(source_names)
        ):
            raise F25ExecutionError(f"{experiment} source inventory is malformed")
        rows: list[dict[str, str]] = []
        for expected_name, row in zip(source_names, source_files):
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "sha256"}
                or row.get("path") != expected_name
                or not _is_sha256(row.get("sha256"))
            ):
                raise F25ExecutionError(
                    f"{experiment} source inventory row is malformed"
                )
            posix = PurePosixPath(expected_name)
            if (
                posix.is_absolute()
                or any(part in {"", ".", ".."} for part in posix.parts)
                or posix.as_posix() != expected_name
            ):
                raise F25ExecutionError("F25 bundle source path is unsafe")
            source_path = repo.joinpath(*posix.parts)
            if source_path.is_symlink() or not source_path.is_file():
                raise F25ExecutionError(
                    f"F25 source is missing or linked: {expected_name}"
                )
            if _sha256_file(source_path) != row["sha256"]:
                raise F25ExecutionError(
                    f"F25 source differs from its bundle: {expected_name}"
                )
            rows.append({"path": expected_name, "sha256": row["sha256"]})
        if manifest.get("source_root_sha256") != canonical_json_sha256(rows):
            raise F25ExecutionError(f"{experiment} bundle source root is invalid")
        generated = manifest.get("generated_artifacts")
        expected_generated_paths = {
            "training_plan": f"f25_training_plan.{experiment}.json",
            "operator_readme": f"README_{experiment}.md",
        }
        if not isinstance(generated, dict) or set(generated) != set(
            expected_generated_paths
        ):
            raise F25ExecutionError(f"{experiment} generated inventory is invalid")
        for role, artifact in generated.items():
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256"}
                or artifact["path"] != expected_generated_paths[role]
                or not _is_sha256(artifact["sha256"])
            ):
                raise F25ExecutionError(
                    f"{experiment} generated artifact identity is malformed"
                )
            artifact_path = repo / artifact["path"]
            if (
                artifact_path.is_symlink()
                or not artifact_path.is_file()
                or _sha256_file(artifact_path) != artifact["sha256"]
            ):
                raise F25ExecutionError(
                    f"{experiment} generated artifact bytes differ from bundle"
                )
        plan_record, _plan_raw = _strict_json_object(
            repo / expected_generated_paths["training_plan"],
            f"{experiment} training plan",
        )
        try:
            validate_training_plan(plan_record)
        except F25TrainingContractError as exc:
            raise F25ExecutionError(
                f"{experiment} generated training plan is invalid"
            ) from exc
        if plan_record != plan:
            raise F25ExecutionError(
                f"{experiment} generated training plan differs from live contract"
            )
        manifests[experiment] = manifest
        raw_shas[experiment] = hashlib.sha256(raw).hexdigest()

    shared_fields = (
        "source_commit",
        "source_manifest_sha256",
        "source_file_count",
        "source_files",
        "source_root_sha256",
        "f25_contract_sha256",
        "shared_generation_root",
    )
    if any(
        manifests["F25-R"].get(field) != manifests["F25-X"].get(field)
        for field in shared_fields
    ):
        raise F25ExecutionError("F25-R/F25-X bundle source identities differ")

    generator = generator_source_root(repo)
    manifest_rows = manifests["F25-R"]["source_files"]
    generator_lines = "\n".join(
        f"{row['path']}:{row['sha256']}"
        for row in manifest_rows
        if row["path"].startswith("scour_MATLAB/")
    )
    if (
        generator.digest_lines != generator_lines
        or generator.file_count != len(generator_lines.splitlines())
        or generator.sha256 != hashlib.sha256(
            generator_lines.encode("utf-8")
        ).hexdigest()
    ):
        raise F25ExecutionError("live MATLAB source differs from bundle identity")
    profile_sha = contract["profile"]["sha256"]
    profile_row = next(
        (
            row
            for row in manifest_rows
            if row["path"] == contract["profile"]["relative_path"]
        ),
        None,
    )
    if profile_row is None or profile_row["sha256"] != profile_sha:
        raise F25ExecutionError("bundle does not bind the fixed F25 profile")

    binding = {
        "schema": BUNDLE_SOURCE_BINDING_SCHEMA,
        "source_commit": manifests["F25-R"]["source_commit"],
        "source_manifest_sha256": manifests["F25-R"][
            "source_manifest_sha256"
        ],
        "bundle_source_root_sha256": manifests["F25-R"][
            "source_root_sha256"
        ],
        "generator_source_root_sha256": generator.sha256,
        "generator_source_digest_lines": generator.digest_lines,
        "generator_source_file_count": generator.file_count,
        "f25_contract_sha256": contract["contract_sha256"],
        "profile_asset_sha256": profile_sha,
        "f25_r_bundle_manifest_file_sha256": raw_shas["F25-R"],
        "f25_x_bundle_manifest_file_sha256": raw_shas["F25-X"],
    }
    descriptor = _bundle_source_binding_descriptor(binding)
    binding["binding_descriptor"] = descriptor
    binding["binding_sha256"] = hashlib.sha256(
        descriptor.encode("utf-8")
    ).hexdigest()
    return binding


def _matlab_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise F25ExecutionError("MATLAB text field is not scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, np.ndarray):
        item = item.reshape(-1)[0]
    text = str(item)
    if not text:
        raise F25ExecutionError("MATLAB text field is empty")
    return text


def _matlab_scalar(value: Any, owner: str) -> Any:
    item: Any = value
    for _iteration in range(4):
        array = np.asarray(item)
        if array.size != 1:
            raise F25ExecutionError(f"{owner} is not scalar")
        item = array.reshape(-1)[0]
        if not isinstance(item, np.ndarray):
            return item.item() if isinstance(item, np.generic) else item
    raise F25ExecutionError(f"{owner} has excessive MATLAB scalar nesting")


def _case_text(record: np.void, field: str) -> str:
    try:
        return _matlab_text(record[field])
    except (KeyError, ValueError) as exc:
        raise F25ExecutionError(f"case_info.{field} is not text") from exc


def _case_integer(record: np.void, field: str) -> int:
    value = _matlab_scalar(record[field], f"case_info.{field}")
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise F25ExecutionError(f"case_info.{field} is not numeric")
    integer = int(value)
    if not math.isfinite(float(value)) or float(value) != integer:
        raise F25ExecutionError(f"case_info.{field} is not an integer")
    return integer


def _case_number(record: np.void, field: str, owner: str) -> float:
    value = _matlab_scalar(record[field], f"{owner}.{field}")
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise F25ExecutionError(f"{owner}.{field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise F25ExecutionError(f"{owner}.{field} is non-finite")
    return number


def _matlab_struct(value: Any, owner: str) -> np.void:
    array = np.asarray(value)
    if array.size != 1:
        raise F25ExecutionError(f"{owner} is not one scalar struct")
    record = array.reshape(-1)[0]
    if not isinstance(record, np.void) or not record.dtype.names:
        raise F25ExecutionError(f"{owner} is not one scalar struct")
    return record


def _sha_seed32(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _f25_seed_catalog() -> dict[str, Any]:
    labels = [row.label for row in SCENARIOS]
    state_uids = [f"f25-state-v1|scenario={label}" for label in labels]
    root_seeds = [
        _sha_seed32(
            "ttbi-state-seed-v1|damage_seed=2025080902|" + state_uid
        )
        for state_uid in state_uids
    ]
    schedule = "uid-named-substreams-v2"
    state_names = [
        "operations",
        "crack",
        "profile-state",
        "track",
        "profile-phase",
    ]
    passage_names = ["profile-passage", "oor-passage"]
    state_seeds = [
        [
            _sha_seed32(
                f"{schedule}|root={root}|uid={uid}|stream={stream}"
            )
            for stream in state_names
        ]
        for root, uid in zip(root_seeds, state_uids)
    ]
    passage_flat = [
        [
            _sha_seed32(
                f"{schedule}|root={root}|uid={uid}|stream={stream}|"
                f"pass={passage:05d}"
            )
            for stream in passage_names
            for passage in range(1, 201)
        ]
        for root, uid in zip(root_seeds, state_uids)
    ]
    all_seeds = [
        *root_seeds,
        *(seed for row in state_seeds for seed in row),
        *(seed for row in passage_flat for seed in row),
    ]
    if 0 in all_seeds or len(all_seeds) != len(set(all_seeds)):
        raise F25ExecutionError("F25 independently derived RNG catalogue collides")
    return {
        "StateUID": state_uids,
        "StateSeedID": root_seeds,
        "StateNamedStreamSeedID": state_seeds,
        "PassageNamedStreamSeedIDFlat": passage_flat,
    }


def _matlab_descriptor_fields(descriptor: str, owner: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in descriptor.split("\n"):
        if "=" not in line:
            raise F25ExecutionError(f"{owner} MATLAB descriptor is malformed")
        name, value = line.split("=", 1)
        if not name or not value or name in fields or any(
            character in value for character in ("\r", "\n", "\x00")
        ):
            raise F25ExecutionError(f"{owner} MATLAB descriptor is malformed")
        fields[name] = value
    expected = {
        "arch",
        "blas",
        "lapack",
        "matlab_product_version",
        "parallel_toolbox_version",
        "release",
        "statistics_toolbox_version",
        "version",
    }
    if set(fields) != expected or list(fields) != sorted(fields):
        raise F25ExecutionError(f"{owner} MATLAB descriptor fields differ")
    return fields


def _validate_generation_config(
    config: dict[str, Any],
    *,
    contract: dict[str, Any],
    binding: dict[str, Any],
    campaign_environment_sha256: str,
    actual_environment_sha256: str,
) -> None:
    scenarios = contract["scenarios"]
    expected_damage = [
        [0.0, row["central_scour_kv_loss_fraction"], 0.0]
        for row in scenarios
    ]
    expected_bearings = [
        [
            row["entrance_bearing_kr_nm_per_rad"],
            row["exit_bearing_kr_nm_per_rad"],
        ]
        for row in scenarios
    ]
    expected_crack_on = [
        bool(row["crack_ei_loss_fraction"] > 0.0) for row in scenarios
    ]
    expected_crack_location = [29.85 if active else None for active in expected_crack_on]
    expected_crack_half_length = [0.15 if active else 0.0 for active in expected_crack_on]
    expected_axis_codes = [
        [
            round(100 * (row["crack_depth_ratio"] or 0.0)),
            round(100 * row["central_scour_kv_loss_fraction"]),
            int(row["entrance_bearing_kr_nm_per_rad"] > 0.0),
        ]
        for row in scenarios
    ]
    window = {
        "schema": "f25-monitoring-window-v1",
        "source": "full_raw_passage_reconstruction",
        "samples_per_m": 100,
        "crop_start": 1001,
        "crop_end_untrimmed": 6831,
        "crop_end_trimmed": 6830,
        "post_deck_samples": 1831,
        "physical_bridge_samples": 3990,
        "source_convention_bridge_samples": 4000,
        "nominal_length_m": 58.3,
        "untrimmed_sample_count": 5831,
        "trimmed_sample_count": 5830,
        "tail_samples_trimmed": 1,
        "extra_beyond_physical_bridge_samples": 10,
    }
    expected = {
        "schema": "f25-generation-v1",
        "python_contract_schema": contract["schema"],
        "python_contract_sha256": contract["contract_sha256"],
        "shared_data_contract_id": contract["partition"][
            "shared_data_contract_id"
        ],
        "dataset_id": "fernandes-2025-f25-data-v1",
        "channel_schema_id": contract["channel_schema_id"],
        "campaign_matlab_environment_sha256": campaign_environment_sha256,
        "actual_matlab_environment_sha256": actual_environment_sha256,
        "generator_source_root_sha256": binding[
            "generator_source_root_sha256"
        ],
        "bundle_source_binding_sha256": binding["binding_sha256"],
        "qualification_source_sha256": "PRODUCTION",
        "profile_mode": "f25_stored_type2",
        "profile_asset_sha256": contract["profile"]["sha256"],
        "L_bridge": contract["geometry"]["bridge_length_m"],
        "num_spans": 2,
        "span_lengths_m": contract["geometry"]["span_lengths_m"],
        "support_locations_m": [0.0, 19.95, 39.9],
        "deck_element_length_m": contract["geometry"]["deck_mesh_m"],
        "deck_element_count": contract["geometry"]["deck_element_count"],
        "deck_mass_per_length_kg_per_m": contract["geometry"][
            "deck_mass_kg_per_m"
        ],
        "deck_E_Pa": 35.0e9,
        "deck_I_m4": 0.33,
        "deck_damping_percent": 3,
        "n_states": contract["passages"]["classes"],
        "Npass": contract["passages"]["per_class"],
        "state_design_kind": "f25-ten-scenario-v1",
        "state_identity_version": "f25-state-v1",
        "random_stream_schedule_version": "uid-named-substreams-v2",
        "state_stream_names": [
            "operations",
            "crack",
            "profile-state",
            "track",
            "profile-phase",
        ],
        "passage_stream_names": ["profile-passage", "oor-passage"],
        "DamageStates": expected_damage,
        "BearingStates": expected_bearings,
        "CrackOn": expected_crack_on,
        "CrackLocation": expected_crack_location,
        "CrackIntensity": [
            row["crack_ei_loss_fraction"] for row in scenarios
        ],
        "CrackHalfLength": expected_crack_half_length,
        "axis_codes": expected_axis_codes,
        "eov_master_seed": contract["eov_master_seed"],
        "partition_seed": contract["partition"]["seed"],
        "noise_master_seed": contract["preprocessing"]["noise"]["master_seed"],
        "Nveh": contract["eov"]["vehicle_count"],
        "Nprop": contract["eov"]["varied_properties_per_vehicle"],
        "vel_km_h": contract["eov"]["speed_km_h"],
        "temp_C": contract["eov"]["temperature_c"],
        "primary_suspension_kN_per_m": [2640, 2920],
        "secondary_suspension_kN_per_m": [942, 1042],
        "carbody_mass_kg": contract["eov"]["carbody_mass_kg"],
        "monitoring_window": window,
        "load_time_measurement_noise_fraction": contract["eov"][
            "measurement_noise_sigma_fraction"
        ],
        "noise_standard_deviation_ddof": contract["preprocessing"]["noise"][
            "standard_deviation_ddof"
        ],
        "contact_max_tension_N": 24000,
        "contact_max_tension_fraction": 0.002,
    }
    expected.update(_f25_seed_catalog())
    if set(config) != set(expected):
        raise F25ExecutionError(
            "F25 generation config field inventory differs from contract"
        )
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise F25ExecutionError(
                f"F25 generation config differs from contract at {field}"
            )


def _validate_generation_case_info(
    root: Path,
    completion_receipt: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    case_path = root / "case_info.mat"
    if case_path.is_symlink() or not case_path.is_file():
        raise F25ExecutionError("F25 case_info.mat is missing or linked")
    try:
        inventory = sio.whosmat(case_path)
    except (OSError, ValueError) as exc:
        raise F25ExecutionError("F25 case_info.mat cannot be inventoried") from exc
    if len(inventory) != 1 or inventory[0][0] != "case_info":
        raise F25ExecutionError("F25 case_info.mat must contain only case_info")
    loaded = sio.loadmat(
        case_path,
        variable_names=["case_info"],
        mat_dtype=True,
    )
    if "case_info" not in loaded or loaded["case_info"].size != 1:
        raise F25ExecutionError("F25 case_info is not one scalar struct")
    record = loaded["case_info"].reshape(-1)[0]
    names = set(record.dtype.names or ())
    required = {
        "schema",
        "dataset_id",
        "shared_data_contract_id",
        "python_contract_schema",
        "python_contract_sha256",
        "generation_schema",
        "gen_fingerprint",
        "generation_config_json",
        "channel_schema_id",
        "n_states",
        "passages_per_state",
        "state_design_kind",
        "profile_asset_sha256",
        "monitoring_window",
        "partition_seed",
        "noise_master_seed",
        "matlab_release",
        "campaign_matlab_release",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "bundle_source_binding_schema",
        "bundle_source_binding_sha256",
        "bundle_source_commit",
        "bundle_source_manifest_sha256",
        "bundle_source_root_sha256",
        "f25_r_bundle_manifest_file_sha256",
        "f25_x_bundle_manifest_file_sha256",
        "release_qualification_run",
    }
    if names != required:
        missing = sorted(required - names)
        extra = sorted(names - required)
        raise F25ExecutionError(
            f"F25 case_info field inventory differs: missing={missing}, "
            f"extra={extra}"
        )

    contract = build_contract()
    text_expected = {
        "schema": "f25-generation-case-info-v2",
        "dataset_id": "fernandes-2025-f25-data-v1",
        "shared_data_contract_id": contract["partition"][
            "shared_data_contract_id"
        ],
        "python_contract_schema": contract["schema"],
        "python_contract_sha256": contract["contract_sha256"],
        "generation_schema": "f25-generation-v1",
        "channel_schema_id": contract["channel_schema_id"],
        "profile_asset_sha256": contract["profile"]["sha256"],
        "generator_source_root_sha256": binding[
            "generator_source_root_sha256"
        ],
        "bundle_source_binding_schema": BUNDLE_SOURCE_BINDING_SCHEMA,
        "bundle_source_binding_sha256": binding["binding_sha256"],
        "bundle_source_commit": binding["source_commit"],
        "bundle_source_manifest_sha256": binding[
            "source_manifest_sha256"
        ],
        "bundle_source_root_sha256": binding["bundle_source_root_sha256"],
        "f25_r_bundle_manifest_file_sha256": binding[
            "f25_r_bundle_manifest_file_sha256"
        ],
        "f25_x_bundle_manifest_file_sha256": binding[
            "f25_x_bundle_manifest_file_sha256"
        ],
    }
    for field, expected in text_expected.items():
        if _case_text(record, field) != expected:
            raise F25ExecutionError(f"F25 case_info differs at {field}")
    if (
        _case_integer(record, "n_states") != contract["passages"]["classes"]
        or _case_integer(record, "passages_per_state")
        != contract["passages"]["per_class"]
        or _case_integer(record, "partition_seed")
        != contract["partition"]["seed"]
        or _case_integer(record, "noise_master_seed")
        != contract["preprocessing"]["noise"]["master_seed"]
        or bool(_matlab_scalar(
            record["release_qualification_run"],
            "case_info.release_qualification_run",
        ))
    ):
        raise F25ExecutionError("F25 case_info scalar contract differs")

    monitoring = _matlab_struct(
        record["monitoring_window"], "case_info.monitoring_window"
    )
    expected_monitoring = {
        "schema": "f25-monitoring-window-v1",
        "source": "full_raw_passage_reconstruction",
        "samples_per_m": 100,
        "crop_start": 1001,
        "crop_end_untrimmed": 6831,
        "crop_end_trimmed": 6830,
        "post_deck_samples": 1831,
        "physical_bridge_samples": 3990,
        "source_convention_bridge_samples": 4000,
        "nominal_length_m": 58.3,
        "untrimmed_sample_count": 5831,
        "trimmed_sample_count": 5830,
        "tail_samples_trimmed": 1,
        "extra_beyond_physical_bridge_samples": 10,
    }
    monitoring_names = set(monitoring.dtype.names or ())
    if monitoring_names != set(expected_monitoring):
        raise F25ExecutionError("F25 case_info monitoring-window fields differ")
    for field, expected_value in expected_monitoring.items():
        if isinstance(expected_value, str):
            observed: Any = _matlab_text(monitoring[field])
        else:
            observed = _case_number(
                monitoring, field, "case_info.monitoring_window"
            )
        if observed != expected_value:
            raise F25ExecutionError(
                f"F25 case_info monitoring window differs at {field}"
            )

    generator_lines = _case_text(record, "generator_source_digest_lines")
    if (
        generator_lines != binding["generator_source_digest_lines"]
        or _case_integer(record, "generator_source_file_count")
        != binding["generator_source_file_count"]
        or hashlib.sha256(generator_lines.encode("utf-8")).hexdigest()
        != binding["generator_source_root_sha256"]
    ):
        raise F25ExecutionError("F25 case_info MATLAB source identity is invalid")

    actual_descriptor = _case_text(
        record, "actual_matlab_environment_descriptor"
    )
    actual_sha = _case_text(record, "actual_matlab_environment_sha256")
    campaign_descriptor = _case_text(
        record, "campaign_matlab_environment_descriptor"
    )
    campaign_sha = _case_text(record, "campaign_matlab_environment_sha256")
    actual_fields = _matlab_descriptor_fields(actual_descriptor, "actual")
    campaign_fields = _matlab_descriptor_fields(
        campaign_descriptor, "campaign reference"
    )
    try:
        reference = load_environment_lock(
            REPO / "environment" / "campaign-py313-cu128.json"
        )["spec"]
        expected_campaign_descriptor = matlab_environment_descriptor(
            reference["matlab_environment"]
        )
    except RuntimeError as exc:
        raise F25ExecutionError(
            "F25 cannot authenticate the campaign MATLAB reference"
        ) from exc
    if (
        hashlib.sha256(actual_descriptor.encode("utf-8")).hexdigest()
        != actual_sha
        or hashlib.sha256(campaign_descriptor.encode("utf-8")).hexdigest()
        != campaign_sha
        or campaign_descriptor != expected_campaign_descriptor
        or campaign_sha != reference["matlab_environment_sha256"]
        or _case_text(record, "matlab_release") != actual_fields["release"]
        or _case_text(record, "campaign_matlab_release")
        != campaign_fields["release"]
    ):
        raise F25ExecutionError("F25 case_info MATLAB descriptor SHA is invalid")

    config_text = _case_text(record, "generation_config_json")
    fingerprint = _case_text(record, "gen_fingerprint")
    if (
        not _is_sha256(fingerprint)
        or hashlib.sha256(config_text.encode("utf-8")).hexdigest() != fingerprint
        or completion_receipt.get("gen_fingerprint") != fingerprint
    ):
        raise F25ExecutionError("F25 generation fingerprint does not authenticate")
    try:
        config = json.loads(
            config_text,
            object_pairs_hook=lambda rows: _unique_json_rows(
                rows, "F25 generation config"
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise F25ExecutionError("F25 generation config is not strict JSON") from exc
    if not isinstance(config, dict):
        raise F25ExecutionError("F25 generation config is not one object")
    _validate_generation_config(
        config,
        contract=contract,
        binding=binding,
        campaign_environment_sha256=campaign_sha,
        actual_environment_sha256=actual_sha,
    )
    return {
        "case_info_sha256": _sha256_file(case_path),
        "gen_fingerprint": fingerprint,
        "actual_matlab_environment_descriptor": actual_descriptor,
        "actual_matlab_environment_sha256": actual_sha,
        "bundle_source_binding_sha256": binding["binding_sha256"],
        "bundle_source_commit": binding["source_commit"],
        "generator_source_root_sha256": binding[
            "generator_source_root_sha256"
        ],
    }


def _validate_generation_root(root: Path) -> dict[str, Any]:
    bundle_binding = _validate_bundle_source_binding(REPO)
    root = root.resolve(strict=True)
    marker = root / "_F25_GENERATION_COMPLETE"
    digest_path = root / "file_digests.json"
    if (
        not marker.is_file()
        or marker.is_symlink()
        or not digest_path.is_file()
        or digest_path.is_symlink()
    ):
        raise F25ExecutionError("F25 shared data lack the completion boundary")
    receipt, _receipt_raw = _strict_json_object(
        digest_path, "F25 generation digest receipt"
    )
    if set(receipt) != {
        "schema",
        "gen_fingerprint",
        "python_contract_sha256",
        "files",
        "digest_root_sha256",
    } or receipt.get("schema") != "f25-generation-artifact-digests-v1":
        raise F25ExecutionError("unsupported F25 generation digest schema")
    if not _is_sha256(receipt.get("gen_fingerprint")) or not _is_sha256(
        receipt.get("digest_root_sha256")
    ):
        raise F25ExecutionError("F25 generation digest identity is malformed")
    if receipt.get("python_contract_sha256") != build_contract()["contract_sha256"]:
        raise F25ExecutionError("generated F25 data bind a foreign contract")
    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != 10:
        raise F25ExecutionError("F25 generation receipt must cover ten states")
    canonical = []
    for index, row in enumerate(files, start=1):
        expected_name = f"{index:04d}.mat"
        if not isinstance(row, dict) or row.get("name") != expected_name:
            raise F25ExecutionError("F25 state digest inventory is not canonical")
        path = root / expected_name
        expected_sha = row.get("sha256")
        if (
            not _is_sha256(expected_sha)
            or not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != expected_sha
        ):
            raise F25ExecutionError(f"F25 state bytes fail SHA-256: {expected_name}")
        canonical.append({"name": expected_name, "sha256": expected_sha})
    if receipt.get("digest_root_sha256") != canonical_json_sha256(canonical):
        # MATLAB hashes jsonencode(struct-array), whose key order is fixed but
        # differs from canonical JSON.  The marker still binds that MATLAB root;
        # Python therefore carries its own canonical verified-state root too.
        receipt = dict(receipt)
    marker_text = marker.read_text(encoding="utf-8")
    expected_marker = (
        "schema=f25-generation-complete-v1\n"
        f"gen_fingerprint={receipt['gen_fingerprint']}\n"
        f"python_contract_sha256={receipt['python_contract_sha256']}\n"
        f"digest_root_sha256={receipt['digest_root_sha256']}\n"
    )
    if marker_text != expected_marker:
        raise F25ExecutionError("F25 completion marker does not bind its receipt")
    case_binding = _validate_generation_case_info(
        root, receipt, bundle_binding
    )
    return {
        "root": str(root),
        "gen_fingerprint": receipt["gen_fingerprint"],
        "matlab_digest_root_sha256": receipt["digest_root_sha256"],
        "verified_state_root_sha256": canonical_json_sha256(canonical),
        "verified_state_files": canonical,
        **case_binding,
    }


def _load_clean_selected(
    root: Path, sensor_indices: list[int], binding: dict[str, Any]
) -> np.ndarray:
    classes: list[np.ndarray] = []
    contract = build_contract()
    contract_sha = contract["contract_sha256"]
    expected_state_sha = {
        row["name"]: row["sha256"] for row in binding["verified_state_files"]
    }
    for class_index in range(10):
        path = root / f"{class_index + 1:04d}.mat"
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != expected_state_sha[path.name]
        ):
            raise F25ExecutionError(f"{path.name} changed before payload load")
        loaded = sio.loadmat(path, variable_names=["f25_data"], mat_dtype=True)
        if "f25_data" not in loaded:
            raise F25ExecutionError(f"{path.name} lacks f25_data")
        record = loaded["f25_data"][0, 0]
        names = set(record.dtype.names or ())
        required = {
            "schema",
            "generation_schema",
            "gen_fingerprint",
            "python_contract_sha256",
            "shared_data_contract_id",
            "dataset_id",
            "channel_schema_id",
            "class_index_zero_based",
            "clean_trimmed",
            "monitoring_tail_sample",
            "source_window_samples",
            "trimmed_window_samples",
            "tail_samples_trimmed",
            "measurement_noise_applied",
            "profile_asset_sha256",
            "matlab_environment_descriptor",
            "matlab_environment_sha256",
            "generator_source_root_sha256",
            "bundle_source_binding_sha256",
            "release_qualification_run",
        }
        if not required.issubset(names):
            raise F25ExecutionError(f"{path.name} has an incomplete F25 payload")
        if (
            _matlab_text(record["schema"]) != "f25-saved-state-v1"
            or _matlab_text(record["generation_schema"])
            != "f25-generation-v1"
            or _matlab_text(record["gen_fingerprint"])
            != binding["gen_fingerprint"]
            or _matlab_text(record["python_contract_sha256"]) != contract_sha
            or _matlab_text(record["shared_data_contract_id"])
            != contract["partition"]["shared_data_contract_id"]
            or _matlab_text(record["dataset_id"])
            != "fernandes-2025-f25-data-v1"
            or _matlab_text(record["channel_schema_id"]) != "physical8_v1"
            or int(np.asarray(record["class_index_zero_based"]).squeeze())
            != class_index
            or int(np.asarray(record["source_window_samples"]).squeeze()) != 5831
            or int(np.asarray(record["trimmed_window_samples"]).squeeze()) != 5830
            or int(np.asarray(record["tail_samples_trimmed"]).squeeze()) != 1
            or bool(np.asarray(record["measurement_noise_applied"]).squeeze())
            or _matlab_text(record["profile_asset_sha256"])
            != contract["profile"]["sha256"]
            or _matlab_text(record["matlab_environment_descriptor"])
            != binding["actual_matlab_environment_descriptor"]
            or _matlab_text(record["matlab_environment_sha256"])
            != binding["actual_matlab_environment_sha256"]
            or _matlab_text(record["generator_source_root_sha256"])
            != binding["generator_source_root_sha256"]
            or _matlab_text(record["bundle_source_binding_sha256"])
            != binding["bundle_source_binding_sha256"]
            or bool(np.asarray(record["release_qualification_run"]).squeeze())
        ):
            raise F25ExecutionError(f"{path.name} violates F25 state semantics")
        clean = np.asarray(record["clean_trimmed"], dtype=np.float64)
        tail = np.asarray(record["monitoring_tail_sample"], dtype=np.float64)
        if (
            clean.shape != (200, 8, TRIMMED_WINDOW_SAMPLES)
            or tail.shape != (200, 8)
            or not np.all(np.isfinite(clean))
            or not np.all(np.isfinite(tail))
        ):
            raise F25ExecutionError(f"{path.name} has invalid monitoring arrays")
        classes.append(clean[:, sensor_indices, :])
        if _sha256_file(path) != expected_state_sha[path.name]:
            raise F25ExecutionError(f"{path.name} changed during payload load")
    return np.concatenate(classes, axis=0)


def _prepare_dataset(job: dict[str, Any], data_root: Path) -> dict[str, Any]:
    binding = _validate_generation_root(data_root)
    clean = _load_clean_selected(data_root, job["sensor_indices"], binding)
    confirmed_bundle_binding = _validate_bundle_source_binding(REPO)
    if (
        confirmed_bundle_binding["binding_sha256"]
        != binding["bundle_source_binding_sha256"]
        or confirmed_bundle_binding["generator_source_root_sha256"]
        != binding["generator_source_root_sha256"]
    ):
        raise F25ExecutionError(
            "F25 bundle/MATLAB source changed while the dataset was loaded"
        )
    noisy = np.empty(clean.shape, dtype=np.float32)
    n_channels = len(job["sensor_indices"])
    for class_index in range(10):
        for passage_index in range(200):
            row = class_index * 200 + passage_index
            for local_channel, physical_channel in enumerate(job["sensor_indices"]):
                noisy[row, local_channel] = add_source_noise(
                    clean[row, local_channel],
                    class_index=class_index,
                    passage_index=passage_index,
                    channel_index=physical_channel,
                ).astype(np.float32)
    del clean
    split = partition_indices()
    local_passage = np.tile(np.arange(200), 10)
    masks = {
        name: np.isin(local_passage, indices) for name, indices in split.items()
    }
    calibration = fit_trimmed_training_minmax(
        noisy[masks["train"]], channel_axis=1, time_axis=2
    )
    scaled = apply_training_minmax(noisy, calibration, channel_axis=1).astype(
        np.float32
    )
    del noisy
    if job["representation"] == "PAA":
        scaled = paa_blocks_of_ten(scaled, time_axis=2).astype(np.float32)
        if scaled.shape[-1] != PAA_BLOCK_COUNT:
            raise F25ExecutionError("F25 PAA did not yield 583 blocks")
    elif job["representation"] != "RAW":
        raise F25ExecutionError("unknown F25 representation")
    labels = np.repeat(np.arange(10, dtype=np.int64), 200)
    return {
        "X": scaled,
        "y": labels,
        "masks": masks,
        "minmax": {
            "minimum": list(calibration.minimum),
            "maximum": list(calibration.maximum),
        },
        "partition_sha256": partition_sha256(),
        "data_binding": binding,
        "selected_channels": n_channels,
    }


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_X)
            total_loss += float(criterion(logits, batch_y).item())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    truth = y.numpy()
    return total_loss / len(y), float(np.mean(predicted == truth)), predicted


def _train_once(
    *,
    job: dict[str, Any],
    dataset: dict[str, Any],
    params: dict[str, Any],
    seed: int,
    device: torch.device,
    max_epochs: int = MAXIMUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE_EPOCHS,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    _set_determinism(seed)
    X_array = dataset["X"]
    y_array = dataset["y"]
    masks = dataset["masks"]
    split_names = ("train", "validation", "test") if evaluate_test else (
        "train",
        "validation",
    )
    tensors = {
        name: (
            torch.from_numpy(np.ascontiguousarray(X_array[masks[name]])),
            torch.from_numpy(np.ascontiguousarray(y_array[masks[name]])),
        )
        for name in split_names
    }
    model = build_f25_model(
        arm_id=job["arm_id"],
        in_channels=dataset["selected_channels"],
        params=params,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LEARNING_RATE_PLATEAU_FACTOR,
        patience=LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
        min_lr=MINIMUM_LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()
    train_X, train_y = tensors["train"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        validation_loss, validation_accuracy, _ = _evaluate(
            model,
            *tensors["validation"],
            int(params["batch_size"]),
            device,
        )
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1.0e-12:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise F25ExecutionError("F25 training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    validation_loss, validation_accuracy, validation_prediction = _evaluate(
        model,
        *tensors["validation"],
        int(params["batch_size"]),
        device,
    )
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_prediction": validation_prediction,
        "model_state": best_state,
        "parameter_count": parameter_count(model),
        "flattened_units": model.flattened_units,
    }
    if evaluate_test:
        test_loss, test_accuracy, test_prediction = _evaluate(
            model,
            *tensors["test"],
            int(params["batch_size"]),
            device,
        )
        result.update(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_prediction": test_prediction,
                "test_truth": tensors["test"][1].numpy(),
            }
        )
    return result


def _suggest_params(trial: Any) -> dict[str, Any]:
    space = SOURCE_CNN_SEARCH_SPACE
    n_layers = trial.suggest_categorical(
        "n_conv_layers", list(space.convolution_layer_counts)
    )
    params: dict[str, Any] = {"n_conv_layers": n_layers}
    for index in range(n_layers):
        params[f"filters_l{index}"] = trial.suggest_categorical(
            f"filters_l{index}", list(space.convolution_filters)
        )
        params[f"kernel_l{index}"] = trial.suggest_categorical(
            f"kernel_l{index}", list(space.convolution_kernel_sizes)
        )
        params[f"pool_l{index}"] = trial.suggest_categorical(
            f"pool_l{index}", list(space.optional_max_pool_after_each_layer)
        )
    params["dense_units"] = trial.suggest_categorical(
        "dense_units", list(space.dense_units)
    )
    params["batch_size"] = trial.suggest_categorical(
        "batch_size", list(space.batch_sizes)
    )
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", *space.learning_rate_range, log=True
    )
    return params


def _runtime_attestation() -> dict[str, Any]:
    _set_determinism(0)
    from core.environment import load_environment_lock, validate_environment_lock

    environment_lock = load_environment_lock(
        REPO / "environment" / "campaign-py313-cu128.json"
    )
    try:
        validate_environment_lock(environment_lock)
    except RuntimeError as exc:
        raise F25ExecutionError(
            f"F25 training host failed local capability qualification: {exc}"
        ) from exc
    descriptor = current_execution_environment()
    if descriptor["accelerator"]["backend"] != "cuda":
        raise F25ExecutionError("production F25 training requires CUDA")
    return {
        "environment_lock_sha256": environment_lock["sha256"],
        "execution_environment_descriptor": descriptor,
        "execution_environment_sha256": execution_environment_sha256(descriptor),
        "execution_compatibility_descriptor": execution_compatibility_descriptor(
            descriptor
        ),
        "execution_compatibility_sha256": execution_compatibility_sha256(descriptor),
    }


def _logical_execution_block_receipt() -> dict[str, Any]:
    """Return the host-independent identity shared by every F25 job.

    The complete runtime descriptor remains in each job record, and an
    in-progress job may therefore resume only on that same runtime.  Keeping
    machine identity out of this block receipt lets distinct, locally
    qualified jobs run on different available PCs without changing the
    scientific F25 contract.
    """

    return {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "execution_block_id": EXECUTION_BLOCK_ID,
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "hardware_rule": (
            "each F25 job remains on one locally capacity-qualified host; "
            "distinct jobs may use different qualified GPU models/numeric stacks"
        ),
    }


def _bind_execution_block() -> dict[str, Any]:
    path = REPO / "f25_artifacts" / "execution_block_receipt.json"
    receipt = _logical_execution_block_receipt()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise F25ExecutionError(
                "existing F25 block receipt differs from the logical contract"
            )
    else:
        _atomic_json(path, receipt)
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def _require_capacity_receipt(
    runtime: dict[str, Any], source_root: Any
) -> dict[str, Any]:
    path = capacity_receipt_path(
        REPO,
        execution_environment_sha256_value=(
            runtime["execution_environment_sha256"]
        ),
        python_runtime_source_sha256=source_root.sha256,
    )
    if not path.is_file() or path.is_symlink():
        raise F25ExecutionError(
            "run check_f25_capacity.py on this PC and this exact source "
            f"workspace first; expected {path}"
        )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    recorded_sha = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    cases = capacity_contract_cases()
    measurements = receipt.get("measurements")
    device_total = receipt.get("device_total_memory_bytes")
    measurements_valid = (
        type(device_total) is int
        and device_total > 0
        and isinstance(measurements, list)
        and len(measurements) == len(cases)
    )
    if measurements_valid:
        for expected, measurement in zip(cases, measurements):
            if not isinstance(measurement, dict) or set(measurement) != {
                "case_id",
                "loss",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            }:
                measurements_valid = False
                break
            loss = measurement["loss"]
            allocated = measurement["peak_allocated_bytes"]
            reserved = measurement["peak_reserved_bytes"]
            if (
                measurement["case_id"] != expected["case_id"]
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
                or type(allocated) is not int
                or type(reserved) is not int
                or not (0 < allocated <= reserved <= device_total)
            ):
                measurements_valid = False
                break
    if (
        receipt.get("schema") != F25_CAPACITY_SCHEMA
        or receipt.get("accepted") is not True
        or receipt.get("contract_only") is not False
        or receipt.get("capacity_receipt_address_schema")
        != CAPACITY_RECEIPT_ADDRESS_SCHEMA
        or receipt.get("f25_contract_sha256")
        != build_contract()["contract_sha256"]
        or receipt.get("execution_environment_sha256")
        != runtime["execution_environment_sha256"]
        or receipt.get("execution_compatibility_sha256")
        != runtime["execution_compatibility_sha256"]
        or receipt.get("environment_lock_sha256")
        != runtime["environment_lock_sha256"]
        or receipt.get("execution_environment_descriptor")
        != runtime["execution_environment_descriptor"]
        or receipt.get("execution_compatibility_descriptor")
        != runtime["execution_compatibility_descriptor"]
        or receipt.get("python_runtime_source_sha256") != source_root.sha256
        or receipt.get("python_runtime_source_file_count")
        != source_root.file_count
        or recorded_sha != canonical_json_sha256(unsigned)
        or receipt.get("cases") != cases
        or receipt.get("case_contract_sha256")
        != canonical_json_sha256(cases)
        or receipt.get("device_total_memory_bytes")
        != runtime["execution_environment_descriptor"]["accelerator"][
            "total_memory_bytes"
        ]
        or not measurements_valid
    ):
        raise F25ExecutionError(
            "F25 capacity receipt is stale, foreign, incomplete, or rejected"
        )
    return receipt


def _capacity_receipt_binding(
    receipt: dict[str, Any],
    runtime: dict[str, Any],
    source_root: Any,
) -> dict[str, Any]:
    path = capacity_receipt_path(
        REPO,
        execution_environment_sha256_value=(
            runtime["execution_environment_sha256"]
        ),
        python_runtime_source_sha256=source_root.sha256,
    )
    return {
        "schema": "f25-capacity-receipt-binding-v1",
        "relative_path": path.relative_to(REPO).as_posix(),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _job_directory(job: dict[str, Any]) -> Path:
    return REPO / Path(job["manifest_path"])


def _job_completion_artifact(job: dict[str, Any]) -> Path:
    if job["phase"] == "hpo":
        return _job_directory(job) / "winner.json"
    return REPO / Path(job["results_path"]) / "metrics.json"


def _require_prior_tiers(plan: dict[str, Any], job: dict[str, Any]) -> None:
    tier_order = plan["tier_order"]
    tier_index = tier_order.index(job["tier_id"])
    prior_tiers = set(tier_order[:tier_index])
    missing = []
    for prerequisite in plan["jobs"]:
        if prerequisite["tier_id"] not in prior_tiers:
            continue
        path = _job_completion_artifact(prerequisite)
        if not path.is_file():
            missing.append(prerequisite["job_id"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_schema = (
            WINNER_SCHEMA if prerequisite["phase"] == "hpo" else REPORT_SCHEMA
        )
        if (
            payload.get("schema") != expected_schema
            or payload.get("job_id") != prerequisite["job_id"]
            or payload.get("f25_contract_sha256")
            != build_contract()["contract_sha256"]
        ):
            missing.append(prerequisite["job_id"])
    if missing:
        raise F25ExecutionError(
            f"{job['tier_id']} is locked until all prior-tier jobs complete; "
            f"missing/foreign count={len(missing)}, first={missing[0]}"
        )


def _publish_job_record(
    job: dict[str, Any],
    runtime: dict[str, Any],
    block_receipt: dict[str, Any],
    capacity_receipt: dict[str, Any],
    source_root: Any,
    data_binding: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": RUN_RECORD_SCHEMA,
        "job": job,
        "job_sha256": canonical_json_sha256(job),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "python_runtime_source_sha256": source_root.sha256,
        "python_runtime_source_file_count": source_root.file_count,
        "execution_runtime": runtime,
        "execution_block_receipt_sha256": block_receipt["receipt_sha256"],
        "capacity_receipt_binding": _capacity_receipt_binding(
            capacity_receipt, runtime, source_root
        ),
        "data_binding": data_binding,
        "partition_sha256": partition_sha256(),
    }
    path = _job_directory(job) / "run_record.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise F25ExecutionError("existing F25 job record belongs to another run")
    else:
        _atomic_json(path, record)
    return record


def _run_hpo(
    job: dict[str, Any],
    dataset: dict[str, Any],
    device: torch.device,
    run_record: dict[str, Any],
) -> None:
    import optuna

    directory = _job_directory(job)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "study.sqlite3"
    study_name = hashlib.sha256(job["job_id"].encode("utf-8")).hexdigest()
    sampler_seed = int(study_name[:8], 16)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        load_if_exists=True,
    )
    expected_attrs = {
        "job_sha256": canonical_json_sha256(job),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "run_record_sha256": canonical_json_sha256(run_record),
        "execution_seeds": list(HPO_EXECUTION_SEEDS),
    }
    for key, value in expected_attrs.items():
        if key in study.user_attrs and study.user_attrs[key] != value:
            raise F25ExecutionError(f"resumed F25 study has foreign {key}")
        study.set_user_attr(key, value)
    invalid = [trial for trial in study.trials if trial.state.name != "COMPLETE"]
    if invalid:
        raise F25ExecutionError("F25 HPO contains failed/pruned proposals")
    remaining = job["hpo_proposals"] - len(study.trials)
    if remaining < 0:
        raise F25ExecutionError("F25 HPO exceeds the frozen 100-proposal budget")

    def objective(trial: Any) -> float:
        params = _suggest_params(trial)
        scores = []
        epochs = []
        for seed in job["execution_seeds"]:
            result = _train_once(
                job=job,
                dataset=dataset,
                params=params,
                seed=int(seed),
                device=device,
            )
            scores.append(result["validation_accuracy"])
            epochs.append(result["best_epoch"])
        trial.set_user_attr("execution_seeds", list(job["execution_seeds"]))
        trial.set_user_attr("validation_accuracies", scores)
        trial.set_user_attr("best_epochs", epochs)
        return float(np.mean(scores))

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    if len(study.trials) != 100 or any(
        len(trial.user_attrs.get("validation_accuracies", [])) != 5
        for trial in study.trials
    ):
        raise F25ExecutionError("F25 HPO did not execute 100 proposals x five seeds")
    winner = {
        "schema": WINNER_SCHEMA,
        "job_id": job["job_id"],
        "job_sha256": canonical_json_sha256(job),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "run_record_sha256": canonical_json_sha256(run_record),
        "trial_number": study.best_trial.number,
        "mean_validation_accuracy": study.best_value,
        "params": study.best_params,
        "execution_seeds": list(job["execution_seeds"]),
        "validation_accuracies": study.best_trial.user_attrs[
            "validation_accuracies"
        ],
    }
    _atomic_json(directory / "winner.json", winner)


def _anchor_winner_path(job: dict[str, Any]) -> Path:
    config_id = job["anchor_configuration_id"]
    if job["experiment_id"] == "F25-R":
        experiment = "F25-R"
        tier = "F25-R-unfrozen-source-singles"
    elif job["regime"] == "unfrozen":
        experiment = "F25-X"
        tier = "F25-X-02-unfrozen-singles"
    else:
        experiment = "F25-R"
        tier = "F25-R-unfrozen-source-singles"
    return (
        REPO
        / "f25_artifacts"
        / experiment
        / "manifests"
        / tier
        / "hpo"
        / config_id
        / "winner.json"
    )


def _classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    axes = np.asarray([scenario.axis_signature for scenario in SCENARIOS])
    return {
        "overall_ten_class_accuracy": float(np.mean(truth == predicted)),
        "bearing_present_absent_accuracy": float(
            np.mean(axes[truth, 2] == axes[predicted, 2])
        ),
        "scour_level_accuracy": float(np.mean(axes[truth, 1] == axes[predicted, 1])),
        "crack_level_accuracy": float(np.mean(axes[truth, 0] == axes[predicted, 0])),
        "confusion_matrix": confusion.tolist(),
    }


def _run_report(
    job: dict[str, Any],
    dataset: dict[str, Any],
    device: torch.device,
    run_record: dict[str, Any],
) -> None:
    winner_path = _anchor_winner_path(job)
    if not winner_path.is_file():
        raise F25ExecutionError(f"required authenticated HPO winner is missing: {winner_path}")
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    if (
        winner.get("schema") != WINNER_SCHEMA
        or winner.get("f25_contract_sha256") != build_contract()["contract_sha256"]
        or not isinstance(winner.get("params"), dict)
    ):
        raise F25ExecutionError("F25 report anchor winner is malformed or foreign")
    results = []
    best_accuracy = -1.0
    best_checkpoint: dict[str, Any] | None = None
    for seed in job["report_seeds"]:
        result = _train_once(
            job=job,
            dataset=dataset,
            params=winner["params"],
            seed=int(seed),
            device=device,
            evaluate_test=True,
        )
        metrics = _classification_metrics(
            result["test_truth"], result["test_prediction"]
        )
        row = {
            "seed": int(seed),
            "best_epoch": result["best_epoch"],
            "validation_loss": result["validation_loss"],
            "validation_accuracy": result["validation_accuracy"],
            "test_loss": result["test_loss"],
            "parameter_count": result["parameter_count"],
            "flattened_units": result["flattened_units"],
            "metrics": metrics,
        }
        results.append(row)
        accuracy = metrics["overall_ten_class_accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_checkpoint = {
                "schema": "f25-best-run-visual-checkpoint-v1",
                "selection_role": (
                    "best test run retained for confusion-matrix visual comparability only"
                ),
                "job": job,
                "seed": int(seed),
                "params": winner["params"],
                "model_state_dict": result["model_state"],
                "metrics": metrics,
                "run_record_sha256": canonical_json_sha256(run_record),
            }
    if [row["seed"] for row in results] != list(REPORT_SEEDS):
        raise F25ExecutionError("F25 report did not execute the exact 20 seeds")
    artifact = {
        "schema": REPORT_SCHEMA,
        "job_id": job["job_id"],
        "job_sha256": canonical_json_sha256(job),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "run_record_sha256": canonical_json_sha256(run_record),
        "anchor_winner_path": winner_path.relative_to(REPO).as_posix(),
        "anchor_winner_sha256": _sha256_file(winner_path),
        "params": winner["params"],
        "report_seeds": list(REPORT_SEEDS),
        "results": results,
        "best_run_role": "visual comparability only; distribution uses all 20 runs",
    }
    results_dir = REPO / Path(job["results_path"])
    if best_checkpoint is None:
        raise F25ExecutionError("F25 report produced no checkpoint")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_visual_checkpoint.pt"
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    torch.save(best_checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    # Metrics are the report completion boundary. Publish them only after the
    # selected visual checkpoint is durable, so later tiers cannot accept a
    # crash-interrupted half-report as complete.
    _atomic_json(results_dir / "metrics.json", artifact)


def run_job(
    *, experiment: str, job_id: str, data_root: Path | None
) -> None:
    plan = build_training_plan(experiment)
    job = job_by_id(plan, job_id)
    _require_prior_tiers(plan, job)
    expected_root = REPO / Path(job["shared_generation_root"])
    if data_root is not None and data_root.resolve() != expected_root.resolve():
        raise F25ExecutionError("data-root override differs from the canonical F25 root")
    data_root = expected_root
    runtime = _runtime_attestation()
    source_root = python_runtime_source_root(REPO)
    capacity_receipt = _require_capacity_receipt(runtime, source_root)
    block_receipt = _bind_execution_block()
    dataset = _prepare_dataset(job, data_root)
    run_record = _publish_job_record(
        job,
        runtime,
        block_receipt,
        capacity_receipt,
        source_root,
        dataset["data_binding"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if job["phase"] == "hpo":
        _run_hpo(job, dataset, device, run_record)
    elif job["phase"] == "report":
        _run_report(job, dataset, device, run_record)
    else:  # guarded by plan validation
        raise F25ExecutionError("unknown F25 phase")


def smoke() -> None:
    block_receipt = _logical_execution_block_receipt()
    if (
        block_receipt["schema"] != EXECUTION_RECEIPT_SCHEMA
        or "execution_compatibility_sha256" in block_receipt
        or "execution_compatibility_descriptor" in block_receipt
        or "distinct jobs may use different" not in block_receipt["hardware_rule"]
    ):
        raise F25ExecutionError("F25 logical block receipt is host-bound")
    params = {
        "n_conv_layers": 1,
        "filters_l0": 32,
        "kernel_l0": 3,
        "pool_l0": True,
        "dense_units": 16,
        "batch_size": 2,
        "learning_rate": 1.0e-3,
    }
    for arm_id, samples in (
        ("RAW-CNN", 5830),
        ("PAA-CNN", 583),
        ("PAA-multirate", 583),
    ):
        model = build_f25_model(
            arm_id=arm_id, in_channels=2, params=params, device="cpu"
        )
        inputs = torch.randn(2, 2, samples)
        labels = torch.tensor([0, 9])
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        if logits.shape != (2, 10) or not torch.isfinite(loss):
            raise F25ExecutionError(f"F25 model smoke failed for {arm_id}")
    print("PASS f25_executor smoke: RAW/PAA flatten+dense and PAA multi-rate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--experiment", choices=("F25-R", "F25-X"), required=True)
    plan_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run-job")
    run_parser.add_argument("--experiment", choices=("F25-R", "F25-X"), required=True)
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--data-root", type=Path)
    subparsers.add_parser("smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan = build_training_plan(args.experiment)
        if args.output:
            _atomic_json(args.output, plan)
        else:
            print(json.dumps(plan, sort_keys=True, indent=2))
    elif args.command == "run-job":
        run_job(
            experiment=args.experiment,
            job_id=args.job_id,
            data_root=args.data_root,
        )
    else:
        smoke()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (F25ExecutionError, F25TrainingContractError) as error:
        print(f"F25 EXECUTION ABORTED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
