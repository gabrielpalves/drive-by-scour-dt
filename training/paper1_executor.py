"""Manifest-qualified execution adapters for the Paper-1 training grid.

The public :func:`execute_manifest_job` entrypoint accepts only an exact job
from one deterministic Lab-A/Lab-B manifest.  The first executable campaign
slice comprises the 80 F40-S factorial studies, artifact-resolved selected-pair
HPO in all four blocks, the complete grouped-development OOF adjudication, and
the frozen-parameter channel screen. Block-local five-restart freeze artefacts
then unlock stage-local sealed-test stability and the explicitly secondary
F40-S frozen transfer. HPO delegates optimisation, capacity
qualification, study stamping, and DT export to the existing
``training.pipeline.execute_ablation_pipeline`` implementation; refits use the
registered fixed-fold evaluator and publish authenticated aggregate artefacts.
Sealed-report adapters use the report-only robustness interface and cannot open
outer-test indices before their externally authenticated freeze exists. Nothing
falls through to the retired ten-rung workflow or to legacy training semantics.

Required environment for an executable F40-S factorial job
------------------------------------------------------------
``TTBI_DATA_ROOT``
    Absolute path to the directory containing the registered dataset folders
    (normally the repository/bundle ``data`` directory).
``TTBI_RESULTS_ROOT``
    Absolute durable root for per-job packages and identity records.
``TTBI_CACHE_ROOT``
    Absolute durable root for authenticated preprocessing caches.
``TTBI_STUDY_ROOT``
    Absolute durable root for per-job Optuna SQLite databases.
``TTBI_EXECUTION_RECEIPT_DIR``
    Absolute durable root for execution-block and capacity receipts.
``TTBI_CAMPAIGN_RUN_TAG``
    Nonempty prospective run identifier shared by the campaign dispatch.
``TTBI_PAPER1_SELECTION_ARTIFACT``
    Additionally required for selected-pair HPO; absolute path to the canonical
    F40-S pair/retained-slot selection artefact.
``TTBI_PAPER1_SELECTION_ARTIFACT_SHA256``
    External SHA-256 deposited with dispatch; prevents replacing the canonical
    artefact and merely recomputing its self-digest.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping

from core.campaign_contract import (
    EXPECTED_PROTOCOL_SCHEMA_TAG,
    campaign_stage_contract,
)
from core.paper1_dispatch import training_manifests
from core.paper1_training_contract import (
    ANCHOR_CHANNEL_INDEX,
    DEVELOPMENT_INIT_SEEDS,
    DEVELOPMENT_N_REPEATS,
    DEVELOPMENT_N_SPLITS,
    DEVELOPMENT_PARTITION_SEED,
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    HPO_TRIALS_PER_STUDY,
    OUTER_SPLIT_SEED,
    POST_FREEZE_STABILITY_SEEDS,
    RETAINED_PIPELINE_SLOTS,
    SCREEN_REFIT_SEEDS,
    TRAINING_EPOCHS,
    canonical_json_bytes,
    channel_screen_inputs,
    complete_job_grid,
)
from core.paper1_freeze_contract import (
    BLOCK_FREEZE_ARTIFACT_SHA256_ENV,
    BLOCK_FREEZE_SCHEMA,
    SEALED_RESULT_SCHEMA,
    SELECTED_CHAMPION_SCHEMA,
    Paper1FreezeContractError,
    build_block_freeze_artifact,
    freeze_for_slot,
    load_block_freeze_artifact,
    seal_sealed_result,
    validate_sealed_result,
    validate_selected_champion,
)
from core.paper1_refit_contract import (
    ADJUDICATION_ARTIFACT_SHA256_ENV,
    CHANNEL_SELECTION_ARTIFACT_ENV,
    CHANNEL_RESULT_SCHEMA,
    DEVELOPMENT_RESULT_SCHEMA,
    Paper1RefitContractError,
    build_channel_selection_artifact,
    build_development_artifact,
    load_development_artifact,
    seal_channel_result,
    seal_development_result,
    validate_channel_result,
    validate_development_result,
)
from core.paper1_selection import (
    SELECTION_ARTIFACT_ENV,
    SELECTION_ARTIFACT_SHA256_ENV,
    build_selection_artifact,
    load_selection_artifact,
    resolve_selection_claim,
)


EXECUTOR_SCHEMA = "paper1-training-executor-v1"
JOB_IDENTITY_SCHEMA = "paper1-training-job-identity-v1"
JOB_COMPLETION_SCHEMA = "paper1-training-job-completion-v1"

DATA_ROOT_ENV = "TTBI_DATA_ROOT"
RESULTS_ROOT_ENV = "TTBI_RESULTS_ROOT"
CACHE_ROOT_ENV = "TTBI_CACHE_ROOT"
STUDY_ROOT_ENV = "TTBI_STUDY_ROOT"
RECEIPT_ROOT_ENV = "TTBI_EXECUTION_RECEIPT_DIR"
RUN_TAG_ENV = "TTBI_CAMPAIGN_RUN_TAG"

_REPO = Path(__file__).resolve().parents[1]
_ENVIRONMENT_LOCK_RELATIVE = "environment/campaign-py313-cu128.json"
_ENVIRONMENT_LOCK = _REPO / Path(_ENVIRONMENT_LOCK_RELATIVE)
_RUN_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LOWER_HEX = frozenset("0123456789abcdef")


class Paper1ExecutionError(RuntimeError):
    """The manifest job cannot execute without weakening campaign controls."""


class Paper1ExecutionDependencyError(Paper1ExecutionError):
    """A registered upstream artefact or policy adapter is not yet available."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _read_canonical_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Paper1ExecutionError(
            f"campaign record is not one regular, non-symlink file: {path}"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1ExecutionError(
            f"campaign record is unreadable/non-JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Paper1ExecutionError(
            f"campaign record is not a canonical JSON object: {path}"
        )
    return value


def _write_or_verify_canonical(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable canonical record, or verify an exact restart."""

    payload = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise Paper1ExecutionError(f"campaign record cannot be a symlink: {path}")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    except FileExistsError:
        pass
    if not path.is_file() or path.read_bytes() != payload:
        raise Paper1ExecutionError(
            f"existing campaign record differs from this job: {path}"
        )


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    probe = path
    while True:
        if os.path.lexists(probe) and probe.is_symlink():
            raise Paper1ExecutionError(
                f"{label} must not traverse a symlink: {probe}"
            )
        if probe.parent == probe:
            break
        probe = probe.parent


def _required_absolute_root(name: str, *, must_exist: bool) -> Path:
    raw = os.environ.get(name, "")
    if not raw:
        raise Paper1ExecutionError(f"{name} is required")
    path = Path(raw)
    if not path.is_absolute():
        raise Paper1ExecutionError(f"{name} must be an absolute path")
    _reject_symlink_ancestors(path, name)
    if must_exist:
        if not path.is_dir():
            raise Paper1ExecutionError(
                f"{name} must name an existing directory: {path}"
            )
    else:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise Paper1ExecutionError(
                f"{name} is not a regular directory: {path}"
            )
    return path.resolve(strict=True)


def _required_run_tag() -> str:
    value = os.environ.get(RUN_TAG_ENV, "")
    if not _RUN_TAG.fullmatch(value):
        raise Paper1ExecutionError(
            f"{RUN_TAG_ENV} must match {_RUN_TAG.pattern!r}"
        )
    return value


def factorial_architecture(pipeline: str) -> dict[str, Any]:
    """Return live model flags for one exact registered factorial cell."""

    matches = [cell for cell in FACTORIAL_CELLS if cell.cell_id == pipeline]
    if len(matches) != 1:
        raise Paper1ExecutionError(
            f"pipeline {pipeline!r} is not one registered factorial cell"
        )
    cell = matches[0]
    return {
        "name_short": cell.cell_id,
        "method": cell.representation,
        "use_space2vec": cell.position_encoding,
        "use_lstm": cell.lstm,
        "use_nhits": cell.multi_rate_pooling,
        "model_type": "1D_MODULAR",
    }


def all_factorial_architectures() -> list[dict[str, Any]]:
    """Return the exact ordered 2x2x2x2 protocol architecture inventory."""

    return [factorial_architecture(cell.cell_id) for cell in FACTORIAL_CELLS]


def _validate_manifest_request(
    job: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-authenticate host allocation and the exact selected job."""

    if not isinstance(job, Mapping) or not isinstance(manifest, Mapping):
        raise Paper1ExecutionError("job and host manifest must be mappings")
    role = manifest.get("machine_role")
    expected_manifests = training_manifests()
    if role not in expected_manifests or dict(manifest) != expected_manifests[role]:
        raise Paper1ExecutionError(
            "execution requires one exact source-derived Lab-A/Lab-B manifest"
        )
    matches = [
        candidate
        for candidate in manifest["jobs"]
        if candidate["job_id"] == job.get("job_id")
    ]
    if len(matches) != 1 or dict(job) != matches[0]:
        raise Paper1ExecutionError(
            "requested job is absent, duplicated, or differs from its host manifest"
        )
    return dict(matches[0]), dict(manifest)


def validate_f40s_factorial_hpo_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact executable phase shape without importing ML code."""

    required = {
        "schema",
        "phase",
        "stage",
        "pipeline",
        "input_selector",
        "hpo_restart_seed",
        "development_partition_seed",
        "fold_index",
        "initialization_seed",
        "candidate_restart_seed",
        "trials",
        "reporting_role",
        "channel_schema_id",
        "job_id",
    }
    if not isinstance(job, Mapping) or set(job) != required:
        raise Paper1ExecutionError("HPO job fields differ from the contract")
    value = dict(job)
    if (
        value["phase"] != "f40s_factorial_hpo"
        or value["stage"] != "F40-S"
        or value["input_selector"] != [ANCHOR_CHANNEL_INDEX]
        or value["hpo_restart_seed"] not in HPO_RESTART_SEEDS
        or value["trials"] != HPO_TRIALS_PER_STUDY
        or any(
            value[name] is not None
            for name in (
                "development_partition_seed",
                "fold_index",
                "initialization_seed",
                "candidate_restart_seed",
            )
        )
    ):
        raise Paper1ExecutionError(
            "job is not an exact F40-S factorial HPO job"
        )
    factorial_architecture(value["pipeline"])
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["hpo"]
        if candidate["phase"] == "f40s_factorial_hpo"
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("F40-S factorial HPO job identity is foreign")
    return value


def validate_selected_pair_hpo_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one prospective selected-pair job before resolving outcomes."""

    if not isinstance(job, Mapping):
        raise Paper1ExecutionError("selected-pair HPO job must be a mapping")
    value = dict(job)
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["hpo"]
        if candidate["phase"] in {
            "f40s_selected_pair_hpo",
            "block_selected_pair_hpo",
        }
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("selected-pair HPO job identity is foreign")
    expected_phase = (
        "f40s_selected_pair_hpo"
        if value["stage"] == "F40-S" else "block_selected_pair_hpo"
    )
    if (
        value["phase"] != expected_phase
        or value["pipeline"] not in RETAINED_PIPELINE_SLOTS
        or value["input_selector"] != "f40s_selected_pair"
        or value["hpo_restart_seed"] not in HPO_RESTART_SEEDS
        or value["trials"] != HPO_TRIALS_PER_STUDY
        or any(
            value[name] is not None
            for name in (
                "development_partition_seed",
                "fold_index",
                "initialization_seed",
                "candidate_restart_seed",
            )
        )
    ):
        raise Paper1ExecutionError("job is not an exact selected-pair HPO job")
    return value


def validate_development_adjudication_job(
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one explicit candidate/fold/initialization OOF fit."""

    if not isinstance(job, Mapping):
        raise Paper1ExecutionError("development adjudication job must be a mapping")
    value = dict(job)
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["development_adjudication"]
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("development adjudication job identity is foreign")
    if (
        value["phase"] != "f40s_development_adjudication"
        or value["stage"] != "F40-S"
        or value["input_selector"] != [ANCHOR_CHANNEL_INDEX]
        or value["pipeline"] not in {cell.cell_id for cell in FACTORIAL_CELLS}
        or value["candidate_restart_seed"] not in HPO_RESTART_SEEDS
        or value["development_partition_seed"] != DEVELOPMENT_PARTITION_SEED
        or value["fold_index"] not in range(DEVELOPMENT_N_SPLITS)
        or value["initialization_seed"] not in DEVELOPMENT_INIT_SEEDS
        or value["hpo_restart_seed"] is not None
        or value["trials"] is not None
    ):
        raise Paper1ExecutionError("job is not one exact Option-C OOF fit")
    return value


def validate_channel_screen_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact frozen-parameter single/pair channel refit."""

    if not isinstance(job, Mapping):
        raise Paper1ExecutionError("channel-screen job must be a mapping")
    value = dict(job)
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["channel_screen"]
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("channel-screen job identity is foreign")
    if (
        value["phase"] != "f40s_frozen_hyperparameter_channel_screen"
        or value["stage"] != "F40-S"
        or value["pipeline"] not in RETAINED_PIPELINE_SLOTS
        or tuple(value["input_selector"]) not in channel_screen_inputs()
        or value["initialization_seed"] not in SCREEN_REFIT_SEEDS
        or any(
            value[field] is not None
            for field in (
                "hpo_restart_seed",
                "development_partition_seed",
                "fold_index",
                "candidate_restart_seed",
                "trials",
            )
        )
    ):
        raise Paper1ExecutionError("job is not one exact channel-screen refit")
    return value


def validate_post_freeze_stability_job(
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one predeclared stage-local sealed-test stability refit."""

    if not isinstance(job, Mapping):
        raise Paper1ExecutionError("post-freeze stability job must be a mapping")
    value = dict(job)
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["post_freeze_stability"]
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("post-freeze stability job identity is foreign")
    if (
        value["phase"] != "post_freeze_sealed_test_stability"
        or value["stage"] not in {"F40-S", "F40-M", "L99-S", "L99-M"}
        or value["pipeline"] not in RETAINED_PIPELINE_SLOTS
        or value["input_selector"] != "f40s_selected_pair"
        or value["initialization_seed"] not in POST_FREEZE_STABILITY_SEEDS
        or value["reporting_role"] != "primary"
        or any(
            value[field] is not None
            for field in (
                "hpo_restart_seed",
                "development_partition_seed",
                "fold_index",
                "candidate_restart_seed",
                "trials",
            )
        )
    ):
        raise Paper1ExecutionError("job is not one exact post-freeze stability fit")
    return value


def validate_secondary_frozen_transfer_job(
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one downstream report-only F40-S frozen transfer refit."""

    if not isinstance(job, Mapping):
        raise Paper1ExecutionError("secondary transfer job must be a mapping")
    value = dict(job)
    exact = {
        candidate["job_id"]: candidate
        for candidate in complete_job_grid()["phases"]["secondary_frozen_transfer"]
    }
    if value.get("job_id") not in exact or value != exact[value["job_id"]]:
        raise Paper1ExecutionError("secondary transfer job identity is foreign")
    if (
        value["phase"] != "secondary_frozen_hyperparameter_transfer"
        or value["stage"] not in {"F40-M", "L99-S", "L99-M"}
        or value["pipeline"] not in RETAINED_PIPELINE_SLOTS
        or value["input_selector"] != "f40s_selected_pair"
        or value["initialization_seed"] not in SCREEN_REFIT_SEEDS
        or value["reporting_role"] != "secondary_nonselection"
        or any(
            value[field] is not None
            for field in (
                "hpo_restart_seed",
                "development_partition_seed",
                "fold_index",
                "candidate_restart_seed",
                "trials",
            )
        )
    ):
        raise Paper1ExecutionError("job is not one exact secondary transfer fit")
    return value


def _protocol_inference_descriptor() -> dict[str, Any]:
    contract = complete_job_grid()
    return {
        "schema": "paper1-four-block-selection-inference-v1",
        "training_contract_schema": contract["schema"],
        "training_contract_sha256": contract["complete_grid_sha256"],
        "selection_metric": contract["selection_metric"],
        "sealed_outer_test_policy": contract["sealed_outer_test_policy"],
        "state_bootstrap": {
            "unit": "state",
            "n": 100_000,
            "seed": 42,
            "pointwise_ci": 0.95,
            "two-pair_simultaneous_ci": 0.975,
        },
        "cross_block_claim": "descriptive_nonconfirmatory",
    }


def _build_hpo_config(
    *,
    job: Mapping[str, Any],
    architecture: Mapping[str, Any],
    dofs: list[int],
    hyperparameter_mode: str,
    protocol_full: Mapping[str, Any],
    protocol_core_hash: str,
    protocol_hash: str,
    execution_attestation: Mapping[str, Any],
    run_tag: str,
    selection_artifact: Mapping[str, Any] | None = None,
    selection_slot: str | None = None,
) -> dict[str, Any]:
    """Construct the sole pipeline configuration for one HPO job."""

    contract = campaign_stage_contract(str(job["stage"]))
    config = {
        "name": f"paper1_{job['job_id']}",
        "seed": int(job["hpo_restart_seed"]),
        "sensor_noise": None,
        **dict(architecture),
        "dofs": list(dofs),
        "discretization": 1,
        "task": "regression",
        "target_supports": contract["learning"]["target_supports"],
        "bearing_targets": contract["learning"]["bearing_targets"],
        "protocol_hash": protocol_hash,
        "protocol_core_hash": protocol_core_hash,
        "protocol_descriptor": dict(protocol_full),
        "execution_runtime": dict(execution_attestation["runtime"]),
        "campaign_run_tag": run_tag,
        "execution_receipt_sha256": execution_attestation["receipt_sha256"],
        "block_reference_manifest_sha256": None,
        "hyperparameter_mode": hyperparameter_mode,
    }
    if selection_artifact is not None:
        config.update({
            "selection_artifact": dict(selection_artifact),
            "selection_artifact_sha256":
                selection_artifact["artifact_sha256"],
            "selection_slot": selection_slot,
        })
    return config


def _artifact_hashes(model_dir: Path) -> dict[str, str]:
    metadata = model_dir / "DT_metadata.json"
    weights = model_dir / "DT_champion_weights.pth"
    if not metadata.is_file() or metadata.is_symlink():
        raise Paper1ExecutionError(f"missing/unsafe DT metadata: {metadata}")
    if not weights.is_file() or weights.is_symlink():
        raise Paper1ExecutionError(f"missing/unsafe DT weights: {weights}")
    try:
        metadata_value = json.loads(metadata.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1ExecutionError(f"invalid DT metadata: {metadata}") from exc
    scaler_name = metadata_value.get("scaler_filename")
    if not isinstance(scaler_name, str) or Path(scaler_name).name != scaler_name:
        raise Paper1ExecutionError("DT metadata carries an unsafe scaler filename")
    scaler = model_dir / scaler_name
    if not scaler.is_file() or scaler.is_symlink():
        raise Paper1ExecutionError(f"missing/unsafe DT scaler: {scaler}")
    return {
        "DT_metadata.json": _sha256_file(metadata),
        "DT_champion_weights.pth": _sha256_file(weights),
        scaler_name: _sha256_file(scaler),
    }


def _verify_completion(path: Path, *, identity_sha256: str) -> dict[str, Any]:
    value = _read_canonical_json(path)
    if (
        value.get("schema") != JOB_COMPLETION_SCHEMA
        or value.get("executor_schema") != EXECUTOR_SCHEMA
        or value.get("identity_sha256") != identity_sha256
        or not _is_sha256(value.get("identity_sha256"))
    ):
        raise Paper1ExecutionError("job completion identity is invalid")
    model_dir = path.parent / value.get("model_directory", "")
    if (
        not isinstance(value.get("model_directory"), str)
        or Path(value["model_directory"]).name != value["model_directory"]
        or not model_dir.is_dir()
    ):
        raise Paper1ExecutionError("job completion model directory is invalid")
    if value.get("artifacts") != _artifact_hashes(model_dir):
        raise Paper1ExecutionError("completed job artefact bytes have changed")
    return value


def _json_value(value: Any) -> Any:
    """Convert NumPy/scalar containers into finite canonical JSON values."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    # canonical_json_bytes performs the final finite/type check.
    return json.loads(canonical_json_bytes(value))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Paper1ExecutionError(f"missing/unsafe JSON file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1ExecutionError(f"unreadable/non-JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Paper1ExecutionError(f"JSON file is not one object: {path}")
    return value


def _verify_refit_completion(
    path: Path,
    *,
    identity_sha256: str,
    expected_kind: str,
    result_validator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read_canonical_json(path)
    expected_fields = {
        "schema",
        "executor_schema",
        "completion_kind",
        "identity_sha256",
        "job_id",
        "result_file",
        "result_sha256",
    }
    if (
        set(value) != expected_fields
        or value["schema"] != JOB_COMPLETION_SCHEMA
        or value["executor_schema"] != EXECUTOR_SCHEMA
        or value["completion_kind"] != expected_kind
        or value["identity_sha256"] != identity_sha256
        or not _is_sha256(value["identity_sha256"])
        or not _is_sha256(value["result_sha256"])
        or not isinstance(value["result_file"], str)
        or Path(value["result_file"]).name != value["result_file"]
    ):
        raise Paper1ExecutionError("refit completion identity is invalid")
    result_path = path.parent / value["result_file"]
    if _sha256_file(result_path) != value["result_sha256"]:
        raise Paper1ExecutionError("completed refit result bytes have changed")
    result = result_validator(_read_canonical_json(result_path))
    if result["job"]["job_id"] != value["job_id"]:
        raise Paper1ExecutionError("refit completion cites another result job")
    return value, result


def _hpo_candidate_job(pipeline: str, restart_seed: int) -> dict[str, Any]:
    matches = [
        candidate for candidate in complete_job_grid()["phases"]["hpo"]
        if candidate["phase"] == "f40s_factorial_hpo"
        and candidate["pipeline"] == pipeline
        and candidate["hpo_restart_seed"] == restart_seed
    ]
    if len(matches) != 1:
        raise Paper1ExecutionError("candidate has no unique factorial HPO job")
    return matches[0]


def _load_hpo_candidate(
    *,
    pipeline: str,
    restart_seed: int,
    manifest: Mapping[str, Any],
    results_root: Path,
    study_root: Path,
    run_tag: str,
    require_same_host: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate one completed HPO champion and its frozen epoch count."""

    candidate_job = _hpo_candidate_job(pipeline, restart_seed)
    if require_same_host and candidate_job not in manifest["jobs"]:
        raise Paper1ExecutionDependencyError(
            "candidate HPO job is not assigned to the adjudication host"
        )
    job_dir = (
        results_root / "F40-S" / "f40s_factorial_hpo"
        / candidate_job["job_id"]
    )
    identity_path = job_dir / "paper1_job_identity.json"
    completion_path = job_dir / "paper1_job_completion.json"
    identity = _read_canonical_json(identity_path)
    identity_sha = _canonical_sha256(identity)
    if (
        identity.get("job") != candidate_job
        or identity.get("campaign_run_tag") != run_tag
        or identity.get("model_name") != f"paper1_{candidate_job['job_id']}"
        or not _is_sha256(identity.get("protocol_core_hash"))
    ):
        raise Paper1ExecutionError("candidate HPO identity differs from contract")
    completion = _verify_completion(
        completion_path, identity_sha256=identity_sha
    )
    completion_sha = _canonical_sha256(completion)
    model_dir = job_dir / completion["model_directory"]
    metadata_path = model_dir / "DT_metadata.json"
    metadata_sha = _sha256_file(metadata_path)
    if completion["artifacts"].get("DT_metadata.json") != metadata_sha:
        raise Paper1ExecutionError("candidate metadata hash differs from completion")
    metadata = _read_json_object(metadata_path)
    architecture = factorial_architecture(pipeline)
    from core.hyperparameter_policy import ANCHOR_HPO_MODE

    if (
        metadata.get("model_name") != identity["model_name"]
        or metadata.get("preprocessing_method") != architecture["method"]
        or metadata.get("active_dofs") != [ANCHOR_CHANNEL_INDEX]
        or metadata.get("architecture_flags") != {
            "use_space2vec": architecture["use_space2vec"],
            "use_lstm": architecture["use_lstm"],
            "use_nhits": architecture["use_nhits"],
            "model_type": architecture["model_type"],
        }
        or metadata.get("campaign_run_tag") != run_tag
        or metadata.get("protocol_hash") != identity.get("protocol_hash")
        or metadata.get("hyperparameter_mode") != ANCHOR_HPO_MODE
    ):
        raise Paper1ExecutionError("candidate DT metadata lineage is invalid")
    params = metadata.get("optimal_hyperparameters")
    from core.hyperparameter_policy import (
        canonical_json_sha256,
        validate_registered_params,
    )

    params = validate_registered_params(pipeline, params)
    database_path = (
        study_root / "F40-S" / "f40s_factorial_hpo"
        / f"{candidate_job['job_id']}.sqlite3"
    )
    if database_path.is_symlink() or not database_path.is_file():
        raise Paper1ExecutionDependencyError(
            f"candidate Optuna database is missing/unsafe: {database_path}"
        )
    database_sha_before = _sha256_file(database_path)
    import optuna

    study = optuna.load_study(
        study_name=identity["model_name"],
        storage=f"sqlite:///{database_path.as_posix()}",
    )
    stored_record = study.user_attrs.get("ttbi_protocol_record")
    if (
        int(study.best_trial.number) != metadata.get("best_trial_number")
        or float(study.best_value) != metadata.get("best_trial_value")
        or _json_value(study.best_params) != params
        or canonical_json_sha256(stored_record)
        != metadata.get("study_protocol_record_sha256")
    ):
        raise Paper1ExecutionError("candidate Optuna study differs from DT package")
    from core.statistical_inference import frozen_checkpoint_epoch_count

    frozen_epochs = frozen_checkpoint_epoch_count(
        study.best_trial.intermediate_values,
        max_epochs=TRAINING_EPOCHS,
    )
    database_sha_after = _sha256_file(database_path)
    if database_sha_before != database_sha_after:
        raise Paper1ExecutionError("candidate Optuna database changed while inspected")
    candidate = {
        "pipeline": pipeline,
        "hpo_restart_seed": restart_seed,
        "hpo_job_id": candidate_job["job_id"],
        "hpo_identity_sha256": identity_sha,
        "hpo_completion_sha256": completion_sha,
        "hpo_metadata_sha256": metadata_sha,
        "hpo_study_sha256": database_sha_after,
        "protocol_core_hash": identity["protocol_core_hash"],
        "protocol_hash": identity["protocol_hash"],
        "params": params,
        "params_sha256": canonical_json_sha256(params),
        "frozen_checkpoint_epochs": frozen_epochs,
    }
    return candidate, identity


def _selected_hpo_job(stage: str, slot: str, restart_seed: int) -> dict[str, Any]:
    expected_phase = (
        "f40s_selected_pair_hpo"
        if stage == "F40-S" else "block_selected_pair_hpo"
    )
    matches = [
        candidate for candidate in complete_job_grid()["phases"]["hpo"]
        if candidate["phase"] == expected_phase
        and candidate["stage"] == stage
        and candidate["pipeline"] == slot
        and candidate["hpo_restart_seed"] == restart_seed
    ]
    if len(matches) != 1:
        raise Paper1ExecutionError("selected champion has no unique HPO job")
    return matches[0]


def _source_lineage_from_metadata(
    metadata: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    descriptor = metadata.get("protocol_descriptor")
    core = descriptor.get("core") if isinstance(descriptor, Mapping) else None
    rung = descriptor.get("rung") if isinstance(descriptor, Mapping) else None
    code = core.get("code") if isinstance(core, Mapping) else None
    dataset = (
        rung.get("dataset_provenance") if isinstance(rung, Mapping) else None
    )
    if not isinstance(code, Mapping) or not isinstance(dataset, Mapping):
        raise Paper1ExecutionError(
            "selected champion lacks protocol source/dataset provenance"
        )
    value = {
        "environment_lock_sha256": identity.get("environment_lock_sha256"),
        "python_runtime_source_root_sha256": code.get(
            "python_runtime_source_root_sha256"
        ),
        "python_runtime_source_file_count": code.get(
            "python_runtime_source_file_count"
        ),
        "generator_source_root_sha256": dataset.get(
            "generator_source_root_sha256"
        ),
        "generator_source_file_count": dataset.get(
            "generator_source_file_count"
        ),
        "dataset_content_root_sha256": dataset.get(
            "dataset_content_root_sha256"
        ),
        "generation_fingerprint": dataset.get("gen_fingerprint"),
        "qualification_source_sha256": dataset.get(
            "qualification_source_sha256"
        ),
    }
    return _json_value(value)


def _load_selected_hpo_champion(
    *,
    stage: str,
    canonical_slot: str,
    restart_seed: int,
    selection: Mapping[str, Any],
    results_root: Path,
    study_root: Path,
    run_tag: str,
) -> dict[str, Any]:
    """Authenticate one completed selected-pair study without opening data."""

    job = _selected_hpo_job(stage, canonical_slot, restart_seed)
    job_dir = results_root / stage / job["phase"] / job["job_id"]
    identity = _read_canonical_json(job_dir / "paper1_job_identity.json")
    identity_sha = _canonical_sha256(identity)
    if (
        identity.get("job") != job
        or identity.get("campaign_run_tag") != run_tag
        or identity.get("model_name") != f"paper1_{job['job_id']}"
        or identity.get("selection_artifact_sha256")
        != selection["artifact_sha256"]
        or identity.get("selection_slot") != canonical_slot
        or not _is_sha256(identity.get("protocol_core_hash"))
        or not _is_sha256(identity.get("protocol_hash"))
    ):
        raise Paper1ExecutionError("selected-HPO identity differs from contract")
    from core.execution_environment import validate_execution_runtime

    runtime = validate_execution_runtime(identity.get("execution_runtime"))
    completion = _verify_completion(
        job_dir / "paper1_job_completion.json", identity_sha256=identity_sha
    )
    completion_sha = _canonical_sha256(completion)
    model_dir = job_dir / completion["model_directory"]
    metadata_path = model_dir / "DT_metadata.json"
    metadata_sha = _sha256_file(metadata_path)
    if completion["artifacts"].get("DT_metadata.json") != metadata_sha:
        raise Paper1ExecutionError("selected-HPO metadata differs from completion")
    metadata = _read_json_object(metadata_path)
    claim = resolve_selection_claim(
        selection,
        stage=stage,
        slot=canonical_slot,
        campaign_run_tag=run_tag,
        artifact_sha256=selection["artifact_sha256"],
    )
    architecture = factorial_architecture(claim["architecture"])
    from core.hyperparameter_policy import (
        SELECTED_PAIR_HPO_MODE,
        canonical_json_sha256,
        validate_registered_params,
    )

    if (
        claim["canonical_slot"] != canonical_slot
        or metadata.get("model_name") != identity["model_name"]
        or metadata.get("preprocessing_method") != architecture["method"]
        or metadata.get("active_dofs") != claim["selected_pair"]
        or metadata.get("architecture_flags") != {
            "use_space2vec": architecture["use_space2vec"],
            "use_lstm": architecture["use_lstm"],
            "use_nhits": architecture["use_nhits"],
            "model_type": architecture["model_type"],
        }
        or metadata.get("campaign_run_tag") != run_tag
        or metadata.get("protocol_hash") != identity["protocol_hash"]
        or metadata.get("execution_runtime") != runtime
        or metadata.get("execution_environment_sha256")
        != runtime["execution_environment_sha256"]
        or metadata.get("execution_compatibility_sha256")
        != runtime["execution_compatibility_sha256"]
        or metadata.get("execution_receipt_sha256")
        != identity.get("execution_receipt_sha256")
        or metadata.get("hyperparameter_mode") != SELECTED_PAIR_HPO_MODE
        or metadata.get("selection_artifact_sha256")
        != selection["artifact_sha256"]
        or metadata.get("selection_slot") != canonical_slot
        or metadata.get("block_reference_manifest_sha256") is not None
    ):
        raise Paper1ExecutionError("selected-HPO DT metadata lineage is invalid")
    params = _json_value(validate_registered_params(
        claim["architecture"], metadata.get("optimal_hyperparameters")
    ))
    scaler_name = metadata.get("scaler_filename")
    if not isinstance(scaler_name, str) or Path(scaler_name).name != scaler_name:
        raise Paper1ExecutionError("selected-HPO scaler filename is unsafe")
    from core.artifact_provenance import verify_standalone_dt_package

    verify_standalone_dt_package(
        str(model_dir / "DT_champion_weights.pth"),
        str(metadata_path),
        str(model_dir / scaler_name),
        expected_block_reference_manifest_sha256=None,
    )
    database_path = study_root / stage / job["phase"] / f"{job['job_id']}.sqlite3"
    if database_path.is_symlink() or not database_path.is_file():
        raise Paper1ExecutionDependencyError(
            f"selected-HPO Optuna database is missing/unsafe: {database_path}"
        )
    database_sha_before = _sha256_file(database_path)
    import math
    import optuna

    study = optuna.load_study(
        study_name=identity["model_name"],
        storage=f"sqlite:///{database_path.as_posix()}",
    )
    counts = {
        name: sum(trial.state.name == name for trial in study.trials)
        for name in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
    }
    counts["total"] = len(study.trials)
    best_value = float(study.best_value)
    best_number = int(study.best_trial.number)
    stored_record = study.user_attrs.get("ttbi_protocol_record")
    champion_artifact = study.user_attrs.get("ttbi_champion_artifact")
    if (
        counts["total"] != HPO_TRIALS_PER_STUDY
        or counts["COMPLETE"] < 1
        or counts["COMPLETE"] + counts["PRUNED"] != HPO_TRIALS_PER_STUDY
        or any(counts[name] for name in ("FAIL", "RUNNING", "WAITING"))
        or not math.isfinite(best_value)
        or best_value < 0.0
        or best_number != metadata.get("best_trial_number")
        or best_value != metadata.get("best_trial_value")
        or _json_value(study.best_params) != params
        or canonical_json_sha256(stored_record)
        != metadata.get("study_protocol_record_sha256")
        or not isinstance(champion_artifact, Mapping)
        or champion_artifact.get("schema") != "champion-artifact-v6"
        or champion_artifact.get("best_trial_number") != best_number
        or champion_artifact.get("best_trial_value") != best_value
        or champion_artifact.get("hyperparameter_mode")
        != SELECTED_PAIR_HPO_MODE
        or champion_artifact.get("selection_artifact_sha256")
        != selection["artifact_sha256"]
        or champion_artifact.get("selection_slot") != canonical_slot
    ):
        raise Paper1ExecutionError(
            "selected-HPO study is failed/incomplete or differs from its package"
        )
    from core.statistical_inference import frozen_checkpoint_epoch_count

    frozen_epochs = frozen_checkpoint_epoch_count(
        study.best_trial.intermediate_values,
        max_epochs=TRAINING_EPOCHS,
    )
    database_sha_after = _sha256_file(database_path)
    if database_sha_before != database_sha_after:
        raise Paper1ExecutionError("selected-HPO database changed while inspected")
    record = {
        "schema": SELECTED_CHAMPION_SCHEMA,
        "stage": stage,
        "canonical_slot": canonical_slot,
        "pipeline": claim["architecture"],
        "selected_pair": claim["selected_pair"],
        "hpo_restart_seed": restart_seed,
        "hpo_job_id": job["job_id"],
        "hpo_identity_sha256": identity_sha,
        "hpo_completion_sha256": completion_sha,
        "hpo_metadata_sha256": metadata_sha,
        "hpo_study_sha256": database_sha_after,
        "execution_environment_sha256": runtime[
            "execution_environment_sha256"
        ],
        "execution_compatibility_sha256": runtime[
            "execution_compatibility_sha256"
        ],
        "execution_receipt_sha256": identity["execution_receipt_sha256"],
        "protocol_core_hash": identity["protocol_core_hash"],
        "protocol_hash": identity["protocol_hash"],
        "source_lineage": _source_lineage_from_metadata(metadata, identity),
        "best_trial_number": best_number,
        "best_trial_value": best_value,
        "terminal_counts": counts,
        "params": params,
        "params_sha256": canonical_json_sha256(params),
        "frozen_checkpoint_epochs": frozen_epochs,
    }
    try:
        return validate_selected_champion(record, selection=selection)
    except Paper1FreezeContractError as exc:
        raise Paper1ExecutionError(f"invalid selected-HPO champion: {exc}") from exc


def _refit_config(
    *,
    job: Mapping[str, Any],
    pipeline: str,
    dofs: list[int],
) -> dict[str, Any]:
    contract = campaign_stage_contract(str(job["stage"]))
    return {
        "name": f"paper1_{job['job_id']}",
        "seed": int(job["initialization_seed"]),
        "sensor_noise": None,
        **factorial_architecture(pipeline),
        "dofs": list(dofs),
        "discretization": 1,
        "task": "regression",
        "target_supports": contract["learning"]["target_supports"],
        "bearing_targets": contract["learning"]["bearing_targets"],
    }


def _load_refit_data_and_partition(
    *,
    config: Mapping[str, Any],
    dataset: str,
    data_root: Path,
    cache_root: Path,
    fold_index: int | None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load cache and construct either one OOF fold or canonical train/val."""

    import numpy as np
    from core.dataset import canonical_grouped_splits, get_or_create_cache
    from core.statistical_inference import GroupedFold, repeated_stratified_group_folds

    with _dataset_working_directory(data_root):
        X, y, _, groups = get_or_create_cache(
            dict(config), dataset, str(cache_root)
        )
        train_idx, validation_idx, outer_idx = canonical_grouped_splits(
            len(y), groups, seed=OUTER_SPLIT_SEED, dataset_name=dataset
        )
    development_idx = np.sort(np.concatenate([train_idx, validation_idx]))
    development_states = np.unique(groups[development_idx]).astype(int)
    outer_states = np.unique(groups[outer_idx]).astype(int)
    split_path = data_root / dataset / "split_manifest.json"
    split_manifest = _read_json_object(split_path)
    strata = split_manifest.get("stratum")
    if (
        not isinstance(strata, list)
        or len(strata) != len(np.unique(groups))
        or any(not isinstance(value, str) or not value for value in strata)
    ):
        raise Paper1ExecutionError("split manifest lacks row-aligned strata")
    if fold_index is None:
        fold = GroupedFold(
            repeat=0,
            fold=0,
            train_idx=np.asarray(train_idx, dtype=np.int64),
            val_idx=np.asarray(validation_idx, dtype=np.int64),
            train_states=np.unique(groups[train_idx]).astype(np.int64),
            val_states=np.unique(groups[validation_idx]).astype(np.int64),
        )
        development_seed = None
        n_splits = None
        n_repeats = None
    else:
        folds = repeated_stratified_group_folds(
            groups,
            development_idx,
            strata,
            n_splits=DEVELOPMENT_N_SPLITS,
            n_repeats=DEVELOPMENT_N_REPEATS,
            seed=DEVELOPMENT_PARTITION_SEED,
        )
        if len(folds) != DEVELOPMENT_N_SPLITS or sorted(
            int(item.fold) for item in folds
        ) != list(range(DEVELOPMENT_N_SPLITS)):
            raise Paper1ExecutionError("development OOF constructor drifted")
        fold = next(
            item for item in folds
            if int(item.repeat) == 0 and int(item.fold) == fold_index
        )
        development_seed = DEVELOPMENT_PARTITION_SEED
        n_splits = DEVELOPMENT_N_SPLITS
        n_repeats = DEVELOPMENT_N_REPEATS
    if (
        np.intersect1d(fold.train_idx, outer_idx).size
        or np.intersect1d(fold.val_idx, outer_idx).size
        or not np.array_equal(
            np.sort(np.concatenate([fold.train_idx, fold.val_idx])),
            development_idx,
        )
    ):
        raise Paper1ExecutionError("refit partition reaches the sealed outer test")
    partition = {
        "outer_split_seed": OUTER_SPLIT_SEED,
        "development_partition_seed": development_seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "fold_index": fold_index,
        "split_manifest_sha256": _sha256_file(split_path),
        "development_idx_sha256": _array_sha256(development_idx),
        "outer_test_idx_sha256": _array_sha256(outer_idx),
        "development_states": development_states.tolist(),
        "outer_test_states": outer_states.tolist(),
        "train_states": np.sort(fold.train_states).astype(int).tolist(),
        "validation_states": np.sort(fold.val_states).astype(int).tolist(),
    }
    return X, y, groups, fold, partition


def _load_sealed_refit_data(
    *,
    config: Mapping[str, Any],
    dataset: str,
    data_root: Path,
    cache_root: Path,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """Open the outer-test indices only after an authenticated model freeze."""

    import numpy as np
    from core.dataset import canonical_grouped_splits, get_or_create_cache

    with _dataset_working_directory(data_root):
        X, y, _, groups = get_or_create_cache(
            dict(config), dataset, str(cache_root)
        )
        train_idx, validation_idx, outer_idx = canonical_grouped_splits(
            len(y), groups, seed=OUTER_SPLIT_SEED, dataset_name=dataset
        )
    development_idx = np.sort(np.concatenate([train_idx, validation_idx])).astype(
        np.int64
    )
    outer_idx = np.sort(np.asarray(outer_idx, dtype=np.int64))
    if (
        np.intersect1d(development_idx, outer_idx).size
        or not np.array_equal(
            np.sort(np.concatenate([development_idx, outer_idx])),
            np.arange(len(groups), dtype=np.int64),
        )
    ):
        raise Paper1ExecutionError(
            "development and sealed outer-test indices do not partition data"
        )
    split_path = data_root / dataset / "split_manifest.json"
    split_manifest = _read_json_object(split_path)
    assignment = split_manifest.get("assignment")
    if (
        not isinstance(assignment, list)
        or len(assignment) != len(np.unique(groups))
        or set(assignment) - {"train", "val", "test"}
    ):
        raise Paper1ExecutionError("sealed split manifest assignment is invalid")
    development_states = np.unique(groups[development_idx]).astype(int)
    outer_states = np.unique(groups[outer_idx]).astype(int)
    if np.intersect1d(development_states, outer_states).size:
        raise Paper1ExecutionError("a state crosses the sealed outer-test boundary")
    partition = {
        "outer_split_seed": OUTER_SPLIT_SEED,
        "split_manifest_sha256": _sha256_file(split_path),
        "development_idx_sha256": _array_sha256(development_idx),
        "outer_test_idx_sha256": _array_sha256(outer_idx),
        "development_states": development_states.tolist(),
        "outer_test_states": outer_states.tolist(),
    }
    return X, y, groups, development_idx, outer_idx, partition


def _write_refit_result(
    *,
    job_dir: Path,
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
    result_filename: str,
    completion_kind: str,
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    identity_path = job_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(identity_path, identity)
    identity_sha = _canonical_sha256(identity)
    result_path = job_dir / result_filename
    _write_or_verify_canonical(result_path, result)
    completion = {
        "schema": JOB_COMPLETION_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "completion_kind": completion_kind,
        "identity_sha256": identity_sha,
        "job_id": result["job"]["job_id"],
        "result_file": result_filename,
        "result_sha256": _sha256_file(result_path),
    }
    completion_path = job_dir / "paper1_job_completion.json"
    _write_or_verify_canonical(completion_path, completion)
    observed, _ = _verify_refit_completion(
        completion_path,
        identity_sha256=identity_sha,
        expected_kind=completion_kind,
        result_validator=validator,
    )
    return observed


def _complete_selected_alias(
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    claim: Mapping[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    """Complete a prospectively listed slot deduplicated after resolution."""

    canonical_jobs = [
        candidate for candidate in complete_job_grid()["phases"]["hpo"]
        if candidate["phase"] == job["phase"]
        and candidate["stage"] == job["stage"]
        and candidate["pipeline"] == claim["canonical_slot"]
        and candidate["hpo_restart_seed"] == job["hpo_restart_seed"]
    ]
    if len(canonical_jobs) != 1:
        raise Paper1ExecutionError(
            "selected-pair alias has no unique canonical manifest job"
        )
    canonical_job = canonical_jobs[0]
    if canonical_job not in manifest["jobs"]:
        raise Paper1ExecutionError(
            "selected-pair alias and canonical seed job are not on one host"
        )
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=False)
    canonical_dir = (
        results_root / canonical_job["stage"] / canonical_job["phase"]
        / canonical_job["job_id"]
    )
    canonical_identity_path = canonical_dir / "paper1_job_identity.json"
    canonical_completion_path = canonical_dir / "paper1_job_completion.json"
    canonical_identity = _read_canonical_json(canonical_identity_path)
    canonical_identity_sha = hashlib.sha256(
        canonical_json_bytes(canonical_identity)
    ).hexdigest()
    if (
        canonical_identity.get("job") != canonical_job
        or canonical_identity.get("campaign_run_tag") != run_tag
        or canonical_identity.get("selection_artifact_sha256")
        != selection["artifact_sha256"]
        or canonical_identity.get("selection_slot") != claim["canonical_slot"]
    ):
        raise Paper1ExecutionError(
            "canonical selected-pair job identity differs from alias claim"
        )
    canonical_completion = _verify_completion(
        canonical_completion_path,
        identity_sha256=canonical_identity_sha,
    )
    canonical_completion_sha = hashlib.sha256(
        canonical_json_bytes(canonical_completion)
    ).hexdigest()

    alias_dir = results_root / job["stage"] / job["phase"] / job["job_id"]
    alias_identity = {
        "schema": JOB_IDENTITY_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "execution_kind": "deduplicated-selected-slot-alias",
        "training_manifest_sha256": manifest["manifest_sha256"],
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "machine_role": manifest["machine_role"],
        "job": dict(job),
        "campaign_run_tag": run_tag,
        "selection_artifact_sha256": selection["artifact_sha256"],
        "selection_slot": claim["slot"],
        "resolved_architecture": claim["architecture"],
        "canonical_slot": claim["canonical_slot"],
        "canonical_job_id": canonical_job["job_id"],
        "canonical_identity_sha256": canonical_identity_sha,
        "canonical_completion_sha256": canonical_completion_sha,
    }
    alias_identity_path = alias_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(alias_identity_path, alias_identity)
    alias_identity_sha = hashlib.sha256(
        canonical_json_bytes(alias_identity)
    ).hexdigest()
    alias_completion = {
        "schema": JOB_COMPLETION_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "completion_kind": "deduplicated-selected-slot-alias",
        "identity_sha256": alias_identity_sha,
        "job_id": job["job_id"],
        "canonical_job_id": canonical_job["job_id"],
        "canonical_completion_sha256": canonical_completion_sha,
    }
    alias_completion_path = alias_dir / "paper1_job_completion.json"
    _write_or_verify_canonical(alias_completion_path, alias_completion)
    observed = _read_canonical_json(alias_completion_path)
    if observed != alias_completion:
        raise Paper1ExecutionError("selected-pair alias completion drifted")
    return observed


@contextmanager
def _dataset_working_directory(data_root: Path) -> Iterator[None]:
    """Expose the configured root at the legacy loader's ``data/`` pathname."""

    if data_root.name != "data":
        raise Paper1ExecutionError(
            f"{DATA_ROOT_ENV} basename must be 'data' because the authenticated "
            "loader resolves datasets through data/<dataset>"
        )
    previous = Path.cwd()
    os.chdir(data_root.parent)
    try:
        if Path("data").resolve(strict=True) != data_root:
            raise Paper1ExecutionError("data root changed during job preparation")
        yield
    finally:
        os.chdir(previous)


PipelineRunner = Callable[..., list[dict[str, Any]]]


def _execute_hpo_job(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    architecture: Mapping[str, Any],
    dofs: list[int],
    hyperparameter_mode: str,
    selection_artifact: Mapping[str, Any] | None = None,
    selection_slot: str | None = None,
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Execute/resume one already-validated 100-trial Paper-1 HPO job."""

    job, manifest = _validate_manifest_request(job, manifest)

    data_root = _required_absolute_root(DATA_ROOT_ENV, must_exist=True)
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=False)
    cache_root = _required_absolute_root(CACHE_ROOT_ENV, must_exist=False)
    study_root = _required_absolute_root(STUDY_ROOT_ENV, must_exist=False)
    receipt_root = _required_absolute_root(RECEIPT_ROOT_ENV, must_exist=False)
    run_tag = _required_run_tag()

    stage_contract = campaign_stage_contract(job["stage"])
    dataset = stage_contract["dataset"]
    dataset_dir = data_root / dataset
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise Paper1ExecutionError(
            f"registered dataset is missing or unsafe: {dataset_dir}"
        )

    # Heavy/locked imports are deliberately below exact manifest/path checks.
    from core.capacity_preflight import CAPACITY_PREFLIGHT_POLICY
    from core.environment import (
        load_environment_lock_bytes,
        validate_environment_lock,
    )
    from core.execution_environment import (
        EXECUTION_BLOCK_POLICY,
        enforce_execution_block,
    )
    from core.hyperparameter_policy import (
        HYPERPARAMETER_POLICY,
        derive_execution_plan,
    )
    from core.protocol import build_protocol_descriptors, protocol_hash
    from training.trainer import SEARCH_SPACE, TRAIN_PROTOCOL

    if _ENVIRONMENT_LOCK.is_symlink() or not _ENVIRONMENT_LOCK.is_file():
        raise Paper1ExecutionError(
            f"campaign environment lock is missing or unsafe: {_ENVIRONMENT_LOCK}"
        )
    # The lock's descriptor path is protocol material.  Authenticate bytes
    # through the absolute repository path, but record the bundle-relative
    # source name so extracting identical bundles under different directories
    # cannot create host-specific protocol hashes.
    environment_lock = load_environment_lock_bytes(
        _ENVIRONMENT_LOCK.read_bytes(), source=_ENVIRONMENT_LOCK_RELATIVE
    )
    validate_environment_lock(environment_lock)
    core_descriptor, full_descriptor = build_protocol_descriptors(
        stage=job["stage"],
        dataset=dataset,
        dataset_dir=str(dataset_dir),
        target_supports=stage_contract["learning"]["target_supports"],
        bearing_targets=stage_contract["learning"]["bearing_targets"],
        task="regression",
        discretization=1,
        seeds=list(HPO_RESTART_SEEDS),
        n_trials=HPO_TRIALS_PER_STUDY,
        epochs=TRAINING_EPOCHS,
        use_pruner=True,
        sensor_noise=None,
        architectures=all_factorial_architectures(),
        extra_pairs=[],
        control_sets=[list(range(8))],
        pair_search_stages={"F40-S"},
        arch_selection_stages={"F40-S"},
        deployment_selection_stages=set(),
        multi_arch_pair_selection_stages={"F40-S"},
        bootstrap={
            "unit": "state",
            "n_boot": 100_000,
            "seed": 42,
            "ci": 0.95,
        },
        statistical_inference=_protocol_inference_descriptor(),
        schema_tag=EXPECTED_PROTOCOL_SCHEMA_TAG,
        train_protocol=TRAIN_PROTOCOL,
        search_space=SEARCH_SPACE,
        execution_block_policy=EXECUTION_BLOCK_POLICY,
        hyperparameter_policy=HYPERPARAMETER_POLICY,
        capacity_preflight_policy=CAPACITY_PREFLIGHT_POLICY,
        environment_lock=environment_lock,
    )
    core_hash = protocol_hash(core_descriptor)
    full_hash = protocol_hash(full_descriptor)
    attestation = enforce_execution_block(
        stage=job["stage"],
        policy=EXECUTION_BLOCK_POLICY,
        protocol_core_hash=core_hash,
        run_tag=run_tag,
        receipt_dir=receipt_root,
    )
    config = _build_hpo_config(
        job=job,
        architecture=architecture,
        dofs=dofs,
        hyperparameter_mode=hyperparameter_mode,
        protocol_full=full_descriptor,
        protocol_core_hash=core_hash,
        protocol_hash=full_hash,
        execution_attestation=attestation,
        run_tag=run_tag,
        selection_artifact=selection_artifact,
        selection_slot=selection_slot,
    )
    # Exercise the live policy before creating a job record or Optuna DB.
    derive_execution_plan(
        config,
        dataset_name=dataset,
        requested_n_trials=HPO_TRIALS_PER_STUDY,
        requested_use_pruner=True,
        execution_runtime=attestation["runtime"],
    )

    job_dir = results_root / job["stage"] / job["phase"] / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["name"]
    identity = {
        "schema": JOB_IDENTITY_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "machine_role": manifest["machine_role"],
        "job": job,
        "dataset": dataset,
        "environment_lock_sha256": environment_lock["sha256"],
        "protocol_core_hash": core_hash,
        "protocol_hash": full_hash,
        "execution_runtime": attestation["runtime"],
        "execution_receipt_sha256": attestation["receipt_sha256"],
        "campaign_run_tag": run_tag,
        "model_name": model_name,
        "selection_artifact_sha256": (
            selection_artifact["artifact_sha256"]
            if selection_artifact is not None else None
        ),
        "selection_slot": selection_slot,
    }
    identity_path = job_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(identity_path, identity)
    identity_sha = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    completion_path = job_dir / "paper1_job_completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        return _verify_completion(completion_path, identity_sha256=identity_sha)

    study_dir = study_root / job["stage"] / job["phase"]
    study_dir.mkdir(parents=True, exist_ok=True)
    database_path = study_dir / f"{job['job_id']}.sqlite3"
    storage = f"sqlite:///{database_path.as_posix()}"
    stage_cache = cache_root / job["stage"]
    stage_cache.mkdir(parents=True, exist_ok=True)

    if pipeline_runner is None:
        from training.pipeline import execute_ablation_pipeline

        pipeline_runner = execute_ablation_pipeline

    previous_receipt = os.environ.get(RECEIPT_ROOT_ENV)
    os.environ[RECEIPT_ROOT_ENV] = str(receipt_root)
    try:
        with _dataset_working_directory(data_root):
            pipeline_runner(
                experiment_path=[config],
                database_name=storage,
                output_dir_name=str(job_dir),
                cache_dir_name=str(stage_cache),
                dataset=dataset,
                n_trials=HPO_TRIALS_PER_STUDY,
                epochs=TRAINING_EPOCHS,
                optuna_seed=int(job["hpo_restart_seed"]),
                use_pruner=True,
                run_robustness=False,
            )
    finally:
        if previous_receipt is None:
            os.environ.pop(RECEIPT_ROOT_ENV, None)
        else:
            os.environ[RECEIPT_ROOT_ENV] = previous_receipt

    model_dir = job_dir / model_name
    completion = {
        "schema": JOB_COMPLETION_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "identity_sha256": identity_sha,
        "job_id": job["job_id"],
        "model_directory": model_name,
        "artifacts": _artifact_hashes(model_dir),
    }
    _write_or_verify_canonical(completion_path, completion)
    return _verify_completion(completion_path, identity_sha256=identity_sha)


def execute_f40s_factorial_hpo_job(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Execute/resume one exact 100-trial F40-S factorial HPO study."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_f40s_factorial_hpo_job(job)
    from core.hyperparameter_policy import ANCHOR_HPO_MODE

    return _execute_hpo_job(
        job,
        manifest,
        architecture=factorial_architecture(job["pipeline"]),
        dofs=list(job["input_selector"]),
        hyperparameter_mode=ANCHOR_HPO_MODE,
        pipeline_runner=pipeline_runner,
    )


def execute_selected_pair_hpo_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any],
    *,
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Execute one artifact-resolved 100-trial HPO in any Paper-1 block."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_selected_pair_hpo_job(job)
    run_tag = _required_run_tag()
    expected_selection_sha = os.environ.get(
        SELECTION_ARTIFACT_SHA256_ENV, ""
    )
    if not _is_sha256(expected_selection_sha):
        raise Paper1ExecutionError(
            f"{SELECTION_ARTIFACT_SHA256_ENV} must be one lowercase SHA-256"
        )
    selection = load_selection_artifact(
        expected_sha256=expected_selection_sha
    )
    claim = resolve_selection_claim(
        selection,
        stage=job["stage"],
        slot=job["pipeline"],
        campaign_run_tag=run_tag,
        artifact_sha256=expected_selection_sha,
    )
    if claim["canonical_slot"] != claim["slot"]:
        return _complete_selected_alias(
            job=job,
            manifest=manifest,
            selection=selection,
            claim=claim,
            run_tag=run_tag,
        )
    from core.hyperparameter_policy import SELECTED_PAIR_HPO_MODE

    return _execute_hpo_job(
        job,
        manifest,
        architecture=factorial_architecture(claim["architecture"]),
        dofs=list(claim["selected_pair"]),
        hyperparameter_mode=SELECTED_PAIR_HPO_MODE,
        selection_artifact=selection,
        selection_slot=claim["slot"],
        pipeline_runner=pipeline_runner,
    )


def execute_development_adjudication_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any],
    *,
    refit_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute/resume one exact candidate/fold/initialization OOF refit."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_development_adjudication_job(job)
    data_root = _required_absolute_root(DATA_ROOT_ENV, must_exist=True)
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=False)
    cache_root = _required_absolute_root(CACHE_ROOT_ENV, must_exist=False)
    study_root = _required_absolute_root(STUDY_ROOT_ENV, must_exist=True)
    receipt_root = _required_absolute_root(RECEIPT_ROOT_ENV, must_exist=False)
    run_tag = _required_run_tag()
    candidate, hpo_identity = _load_hpo_candidate(
        pipeline=job["pipeline"],
        restart_seed=job["candidate_restart_seed"],
        manifest=manifest,
        results_root=results_root,
        study_root=study_root,
        run_tag=run_tag,
        require_same_host=True,
    )
    from core.execution_environment import EXECUTION_BLOCK_POLICY, enforce_execution_block

    attestation = enforce_execution_block(
        stage="F40-S",
        policy=EXECUTION_BLOCK_POLICY,
        protocol_core_hash=candidate["protocol_core_hash"],
        run_tag=run_tag,
        receipt_dir=receipt_root,
    )
    config = _refit_config(
        job=job,
        pipeline=job["pipeline"],
        dofs=list(job["input_selector"]),
    )
    dataset = campaign_stage_contract("F40-S")["dataset"]
    stage_cache = cache_root / "F40-S"
    stage_cache.mkdir(parents=True, exist_ok=True)
    X, y, groups, fold, partition = _load_refit_data_and_partition(
        config=config,
        dataset=dataset,
        data_root=data_root,
        cache_root=stage_cache,
        fold_index=job["fold_index"],
    )
    job_dir = results_root / "F40-S" / job["phase"] / job["job_id"]
    identity = {
        "schema": JOB_IDENTITY_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "execution_kind": "development_oof_refit",
        "training_manifest_sha256": manifest["manifest_sha256"],
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "machine_role": manifest["machine_role"],
        "job": job,
        "dataset": dataset,
        "campaign_run_tag": run_tag,
        "candidate": candidate,
        "partition": partition,
        "execution_runtime": attestation["runtime"],
        "execution_receipt_sha256": attestation["receipt_sha256"],
        "upstream_hpo_protocol_hash": hpo_identity["protocol_hash"],
    }
    identity_path = job_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(identity_path, identity)
    identity_sha = _canonical_sha256(identity)
    completion_path = job_dir / "paper1_job_completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        completion, _ = _verify_refit_completion(
            completion_path,
            identity_sha256=identity_sha,
            expected_kind="development_oof_refit",
            result_validator=validate_development_result,
        )
        return completion
    if refit_runner is None:
        from training.trainer import fit_predict_fixed_group_fold

        refit_runner = fit_predict_fixed_group_fold
    from core import task

    metrics = refit_runner(
        config=config,
        params=candidate["params"],
        X=X,
        y=y,
        groups=groups,
        fold=fold,
        seed=job["initialization_seed"],
        n_epochs=candidate["frozen_checkpoint_epochs"],
        max_epochs=TRAINING_EPOCHS,
        n_scour_heads=task.n_scour_outputs(config),
    )
    result = seal_development_result({
        "schema": DEVELOPMENT_RESULT_SCHEMA,
        "campaign_run_tag": run_tag,
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "job": job,
        "candidate": candidate,
        "partition": partition,
        "initialization_seed": job["initialization_seed"],
        "outer_test_observations_accessed": False,
        "metrics": _json_value(metrics),
    })
    return _write_refit_result(
        job_dir=job_dir,
        identity=identity,
        result=result,
        result_filename="paper1_development_result.json",
        completion_kind="development_oof_refit",
        validator=validate_development_result,
    )


def execute_channel_screen_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any],
    *,
    refit_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute/resume one frozen-HP channel refit or deduplicated slot alias."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_channel_screen_job(job)
    data_root = _required_absolute_root(DATA_ROOT_ENV, must_exist=True)
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=False)
    cache_root = _required_absolute_root(CACHE_ROOT_ENV, must_exist=False)
    receipt_root = _required_absolute_root(RECEIPT_ROOT_ENV, must_exist=False)
    run_tag = _required_run_tag()
    expected_adjudication_sha = os.environ.get(
        ADJUDICATION_ARTIFACT_SHA256_ENV, ""
    )
    if not _is_sha256(expected_adjudication_sha):
        raise Paper1ExecutionError(
            f"{ADJUDICATION_ARTIFACT_SHA256_ENV} must be one lowercase SHA-256"
        )
    try:
        adjudication = load_development_artifact(
            expected_sha256=expected_adjudication_sha
        )
    except Paper1RefitContractError as exc:
        raise Paper1ExecutionError(f"invalid adjudication artefact: {exc}") from exc
    if (
        adjudication["campaign_run_tag"] != run_tag
        or adjudication["complete_grid_sha256"] != manifest["complete_grid_sha256"]
    ):
        raise Paper1ExecutionError("adjudication artefact belongs to another run/grid")
    slot = job["pipeline"]
    canonical_slot = adjudication["canonical_slot"][slot]
    pipeline = adjudication["slot_resolution"][slot]
    frozen = adjudication["winner_by_pipeline"][pipeline]["candidate"]
    from core.hyperparameter_policy import validate_registered_params

    if validate_registered_params(pipeline, frozen["params"]) != frozen["params"]:
        raise Paper1ExecutionError("adjudicated frozen parameters do not validate")
    from core.execution_environment import EXECUTION_BLOCK_POLICY, enforce_execution_block

    attestation = enforce_execution_block(
        stage="F40-S",
        policy=EXECUTION_BLOCK_POLICY,
        protocol_core_hash=frozen["protocol_core_hash"],
        run_tag=run_tag,
        receipt_dir=receipt_root,
    )
    job_dir = results_root / "F40-S" / job["phase"] / job["job_id"]

    if canonical_slot != slot:
        exact_jobs = complete_job_grid()["phases"]["channel_screen"]
        canonical_job = next(
            candidate for candidate in exact_jobs
            if candidate["pipeline"] == canonical_slot
            and candidate["input_selector"] == job["input_selector"]
            and candidate["initialization_seed"] == job["initialization_seed"]
        )
        if canonical_job not in manifest["jobs"]:
            raise Paper1ExecutionError("channel alias canonical job is on another host")
        canonical_dir = (
            results_root / "F40-S" / canonical_job["phase"]
            / canonical_job["job_id"]
        )
        canonical_identity = _read_canonical_json(
            canonical_dir / "paper1_job_identity.json"
        )
        canonical_identity_sha = _canonical_sha256(canonical_identity)
        if (
            canonical_identity.get("job") != canonical_job
            or canonical_identity.get("campaign_run_tag") != run_tag
            or canonical_identity.get("adjudication_artifact_sha256")
            != adjudication["artifact_sha256"]
        ):
            raise Paper1ExecutionError("canonical channel identity differs from alias")
        canonical_completion, canonical_result = _verify_refit_completion(
            canonical_dir / "paper1_job_completion.json",
            identity_sha256=canonical_identity_sha,
            expected_kind="channel_screen_refit",
            result_validator=validate_channel_result,
        )
        result = seal_channel_result({
            **canonical_result,
            "execution_kind": "deduplicated_alias",
            "job": job,
            "canonical_job_id": canonical_job["job_id"],
        })
        identity = {
            "schema": JOB_IDENTITY_SCHEMA,
            "executor_schema": EXECUTOR_SCHEMA,
            "execution_kind": "channel_screen_deduplicated_alias",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "complete_grid_sha256": manifest["complete_grid_sha256"],
            "machine_role": manifest["machine_role"],
            "job": job,
            "dataset": campaign_stage_contract("F40-S")["dataset"],
            "campaign_run_tag": run_tag,
            "adjudication_artifact_sha256": adjudication["artifact_sha256"],
            "canonical_job_id": canonical_job["job_id"],
            "canonical_identity_sha256": canonical_identity_sha,
            "canonical_completion_sha256": _canonical_sha256(canonical_completion),
            "canonical_result_sha256": canonical_result["result_sha256"],
            "execution_runtime": attestation["runtime"],
            "execution_receipt_sha256": attestation["receipt_sha256"],
        }
        return _write_refit_result(
            job_dir=job_dir,
            identity=identity,
            result=result,
            result_filename="paper1_channel_result.json",
            completion_kind="channel_screen_alias",
            validator=validate_channel_result,
        )

    config = _refit_config(
        job=job,
        pipeline=pipeline,
        dofs=list(job["input_selector"]),
    )
    dataset = campaign_stage_contract("F40-S")["dataset"]
    stage_cache = cache_root / "F40-S"
    stage_cache.mkdir(parents=True, exist_ok=True)
    X, y, groups, fold, partition = _load_refit_data_and_partition(
        config=config,
        dataset=dataset,
        data_root=data_root,
        cache_root=stage_cache,
        fold_index=None,
    )
    identity = {
        "schema": JOB_IDENTITY_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "execution_kind": "channel_screen_refit",
        "training_manifest_sha256": manifest["manifest_sha256"],
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "machine_role": manifest["machine_role"],
        "job": job,
        "dataset": dataset,
        "campaign_run_tag": run_tag,
        "adjudication_artifact_sha256": adjudication["artifact_sha256"],
        "slot_resolution": adjudication["slot_resolution"],
        "canonical_slot": canonical_slot,
        "resolved_pipeline": pipeline,
        "frozen_candidate": frozen,
        "partition": partition,
        "execution_runtime": attestation["runtime"],
        "execution_receipt_sha256": attestation["receipt_sha256"],
    }
    identity_path = job_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(identity_path, identity)
    identity_sha = _canonical_sha256(identity)
    completion_path = job_dir / "paper1_job_completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        completion, _ = _verify_refit_completion(
            completion_path,
            identity_sha256=identity_sha,
            expected_kind="channel_screen_refit",
            result_validator=validate_channel_result,
        )
        return completion
    if refit_runner is None:
        from training.trainer import fit_predict_fixed_group_fold

        refit_runner = fit_predict_fixed_group_fold
    from core import task

    metrics = refit_runner(
        config=config,
        params=frozen["params"],
        X=X,
        y=y,
        groups=groups,
        fold=fold,
        seed=job["initialization_seed"],
        n_epochs=frozen["frozen_checkpoint_epochs"],
        max_epochs=TRAINING_EPOCHS,
        n_scour_heads=task.n_scour_outputs(config),
    )
    result = seal_channel_result({
        "schema": CHANNEL_RESULT_SCHEMA,
        "execution_kind": "refit",
        "campaign_run_tag": run_tag,
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "adjudication_artifact_sha256": adjudication["artifact_sha256"],
        "job": job,
        "slot_resolution": adjudication["slot_resolution"],
        "canonical_slot": canonical_slot,
        "pipeline": pipeline,
        "frozen_candidate": frozen,
        "partition": partition,
        "outer_test_observations_accessed": False,
        "canonical_job_id": job["job_id"],
        "metrics": _json_value(metrics),
    })
    return _write_refit_result(
        job_dir=job_dir,
        identity=identity,
        result=result,
        result_filename="paper1_channel_result.json",
        completion_kind="channel_screen_refit",
        validator=validate_channel_result,
    )


def _publication_path(raw: str | os.PathLike[str], label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise Paper1ExecutionError(f"{label} must be an absolute path")
    _reject_symlink_ancestors(path, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent.absolute():
        raise Paper1ExecutionError(f"{label} parent path is not canonical")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Paper1ExecutionError(f"{label} is not a regular file target")
    return path


def _collect_refit_results(
    *,
    results_root: Path,
    phase_key: str,
    run_tag: str,
    result_validator: Callable[[Mapping[str, Any]], dict[str, Any]],
    completion_kinds: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job in complete_job_grid()["phases"][phase_key]:
        job_dir = results_root / job["stage"] / job["phase"] / job["job_id"]
        identity = _read_canonical_json(job_dir / "paper1_job_identity.json")
        identity_sha = _canonical_sha256(identity)
        if identity.get("job") != job or identity.get("campaign_run_tag") != run_tag:
            raise Paper1ExecutionDependencyError(
                f"upstream refit identity is foreign/incomplete: {job['job_id']}"
            )
        completion = _read_canonical_json(
            job_dir / "paper1_job_completion.json"
        )
        kind = completion.get("completion_kind")
        if kind not in completion_kinds:
            raise Paper1ExecutionDependencyError(
                f"upstream refit completion kind is invalid: {job['job_id']}"
            )
        _, result = _verify_refit_completion(
            job_dir / "paper1_job_completion.json",
            identity_sha256=identity_sha,
            expected_kind=kind,
            result_validator=result_validator,
        )
        results.append(result)
    return results


def publish_development_adjudication_artifact(
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Publish only after all 480 exact OOF fit results authenticate."""

    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=True)
    run_tag = _required_run_tag()
    raw = (
        os.fspath(output_path)
        if output_path is not None
        else os.environ.get("TTBI_PAPER1_ADJUDICATION_ARTIFACT", "")
    )
    if not raw:
        raise Paper1ExecutionError(
            "an adjudication artefact output path is required"
        )
    path = _publication_path(raw, "adjudication artefact output")
    results = _collect_refit_results(
        results_root=results_root,
        phase_key="development_adjudication",
        run_tag=run_tag,
        result_validator=validate_development_result,
        completion_kinds={"development_oof_refit"},
    )
    artifact = build_development_artifact(results)
    _write_or_verify_canonical(path, artifact)
    if _sha256_file(path) != hashlib.sha256(canonical_json_bytes(artifact)).hexdigest():
        raise Paper1ExecutionError("published adjudication artefact bytes drifted")
    return artifact


def publish_channel_selection_artifacts(
    channel_output_path: str | os.PathLike[str] | None = None,
    selection_output_path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Publish the full channel tensor and compact downstream selection."""

    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=True)
    run_tag = _required_run_tag()
    expected_adjudication_sha = os.environ.get(
        ADJUDICATION_ARTIFACT_SHA256_ENV, ""
    )
    try:
        adjudication = load_development_artifact(
            expected_sha256=expected_adjudication_sha
        )
    except Paper1RefitContractError as exc:
        raise Paper1ExecutionError(f"invalid adjudication artefact: {exc}") from exc
    if adjudication["campaign_run_tag"] != run_tag:
        raise Paper1ExecutionError("adjudication artefact belongs to another run")
    raw_channel = (
        os.fspath(channel_output_path)
        if channel_output_path is not None
        else os.environ.get(CHANNEL_SELECTION_ARTIFACT_ENV, "")
    )
    raw_selection = (
        os.fspath(selection_output_path)
        if selection_output_path is not None
        else os.environ.get(SELECTION_ARTIFACT_ENV, "")
    )
    if not raw_channel or not raw_selection:
        raise Paper1ExecutionError(
            "channel and downstream selection output paths are required"
        )
    channel_path = _publication_path(
        raw_channel, "channel-selection artefact output"
    )
    selection_path = _publication_path(
        raw_selection, "downstream selection artefact output"
    )
    results = _collect_refit_results(
        results_root=results_root,
        phase_key="channel_screen",
        run_tag=run_tag,
        result_validator=validate_channel_result,
        completion_kinds={"channel_screen_refit", "channel_screen_alias"},
    )
    channel = build_channel_selection_artifact(results, adjudication)
    hpo_evidence = _canonical_sha256([
        {
            "pipeline": summary["pipeline"],
            "hpo_restart_seed": summary["hpo_restart_seed"],
            "hpo_job_id": summary["candidate"]["hpo_job_id"],
            "hpo_identity_sha256": summary["candidate"]["hpo_identity_sha256"],
            "hpo_completion_sha256": summary["candidate"]["hpo_completion_sha256"],
            "hpo_study_sha256": summary["candidate"]["hpo_study_sha256"],
        }
        for summary in adjudication["candidate_summaries"]
    ])
    selection = build_selection_artifact(
        campaign_run_tag=run_tag,
        selected_pair=channel["selected_pair"],
        best_raw=adjudication["best_raw"],
        best_paa=adjudication["best_paa"],
        evidence_sha256={
            "factorial_hpo_manifest": hpo_evidence,
            "development_adjudication_manifest": adjudication["artifact_sha256"],
            "channel_screen_manifest": channel["artifact_sha256"],
        },
    )
    _write_or_verify_canonical(channel_path, channel)
    _write_or_verify_canonical(selection_path, selection)
    return {"channel_selection": channel, "selection": selection}


def publish_block_freeze_artifact(
    stage: str,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Freeze one block only after every unique pipeline has five HPO runs."""

    if stage not in {"F40-S", "F40-M", "L99-S", "L99-M"}:
        raise Paper1ExecutionError(f"unregistered block-freeze stage {stage!r}")
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=True)
    study_root = _required_absolute_root(STUDY_ROOT_ENV, must_exist=True)
    run_tag = _required_run_tag()
    expected_selection_sha = os.environ.get(
        SELECTION_ARTIFACT_SHA256_ENV, ""
    )
    if not _is_sha256(expected_selection_sha):
        raise Paper1ExecutionError(
            f"{SELECTION_ARTIFACT_SHA256_ENV} must be one lowercase SHA-256"
        )
    try:
        selection = load_selection_artifact(
            expected_sha256=expected_selection_sha
        )
    except Exception as exc:
        raise Paper1ExecutionError(f"invalid selection artefact: {exc}") from exc
    if selection["campaign_run_tag"] != run_tag:
        raise Paper1ExecutionError("selection artefact belongs to another run")
    canonical_slots = [
        slot for slot in RETAINED_PIPELINE_SLOTS
        if selection["canonical_slot"][slot] == slot
    ]
    champions = [
        _load_selected_hpo_champion(
            stage=stage,
            canonical_slot=slot,
            restart_seed=seed,
            selection=selection,
            results_root=results_root,
            study_root=study_root,
            run_tag=run_tag,
        )
        for slot in canonical_slots
        for seed in HPO_RESTART_SEEDS
    ]
    try:
        artifact = build_block_freeze_artifact(
            stage=stage,
            selection=selection,
            champions=champions,
        )
    except Paper1FreezeContractError as exc:
        raise Paper1ExecutionError(f"cannot freeze {stage}: {exc}") from exc
    path = _publication_path(output_path, f"{stage} block-freeze output")
    _write_or_verify_canonical(path, artifact)
    if _sha256_file(path) != hashlib.sha256(
        canonical_json_bytes(artifact)
    ).hexdigest():
        raise Paper1ExecutionError("published block-freeze bytes drifted")
    return artifact


def _load_report_freeze(
    *, job: Mapping[str, Any], freeze_stage: str, run_tag: str
) -> dict[str, Any]:
    """Authenticate both deposited selection and freeze before data access."""

    expected_freeze_sha = os.environ.get(
        BLOCK_FREEZE_ARTIFACT_SHA256_ENV, ""
    )
    expected_selection_sha = os.environ.get(
        SELECTION_ARTIFACT_SHA256_ENV, ""
    )
    if not _is_sha256(expected_freeze_sha):
        raise Paper1ExecutionDependencyError(
            f"{BLOCK_FREEZE_ARTIFACT_SHA256_ENV} is required before sealed data"
        )
    if not _is_sha256(expected_selection_sha):
        raise Paper1ExecutionDependencyError(
            f"{SELECTION_ARTIFACT_SHA256_ENV} is required before sealed data"
        )
    try:
        freeze = load_block_freeze_artifact(
            expected_sha256=expected_freeze_sha
        )
        selection = load_selection_artifact(
            expected_sha256=expected_selection_sha
        )
    except (Paper1FreezeContractError, Exception) as exc:
        raise Paper1ExecutionDependencyError(
            f"authenticated block freeze/selection is unavailable: {exc}"
        ) from exc
    if (
        freeze["stage"] != freeze_stage
        or freeze["campaign_run_tag"] != run_tag
        or freeze["complete_grid_sha256"]
        != complete_job_grid()["complete_grid_sha256"]
        or freeze["selection_artifact"] != selection
        or selection["campaign_run_tag"] != run_tag
    ):
        raise Paper1ExecutionError(
            "block freeze belongs to another stage, run, grid, or selection"
        )
    try:
        freeze_for_slot(freeze, stage=freeze_stage, slot=job["pipeline"])
    except Paper1FreezeContractError as exc:
        raise Paper1ExecutionError(f"job has no authenticated freeze: {exc}") from exc
    return freeze


def _canonical_report_job(
    job: Mapping[str, Any], *, canonical_slot: str, phase_key: str
) -> dict[str, Any]:
    matches = [
        candidate for candidate in complete_job_grid()["phases"][phase_key]
        if candidate["stage"] == job["stage"]
        and candidate["pipeline"] == canonical_slot
        and candidate["initialization_seed"] == job["initialization_seed"]
    ]
    if len(matches) != 1:
        raise Paper1ExecutionError("report alias has no unique canonical job")
    return matches[0]


def _execute_sealed_report_job(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    freeze_stage: str,
    phase_key: str,
    reporting_role: str,
    completion_prefix: str,
    fit_evaluate: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Execute/resume one report-only development-to-outer-test refit."""

    job, manifest = _validate_manifest_request(job, manifest)
    run_tag = _required_run_tag()

    # This is the firewall: no cache, split manifest, or outer-test index is
    # loaded until the complete five-restart freeze and external digest pass.
    freeze = _load_report_freeze(
        job=job, freeze_stage=freeze_stage, run_tag=run_tag
    )
    data_root = _required_absolute_root(DATA_ROOT_ENV, must_exist=True)
    results_root = _required_absolute_root(RESULTS_ROOT_ENV, must_exist=False)
    cache_root = _required_absolute_root(CACHE_ROOT_ENV, must_exist=False)
    receipt_root = _required_absolute_root(RECEIPT_ROOT_ENV, must_exist=False)
    claim = freeze_for_slot(freeze, stage=freeze_stage, slot=job["pipeline"])
    canonical_job = _canonical_report_job(
        job, canonical_slot=claim["canonical_slot"], phase_key=phase_key
    )
    if canonical_job not in manifest["jobs"]:
        raise Paper1ExecutionError("report alias canonical seed job is on another host")
    from core.execution_environment import EXECUTION_BLOCK_POLICY, enforce_execution_block

    attestation = enforce_execution_block(
        stage=job["stage"],
        policy=EXECUTION_BLOCK_POLICY,
        protocol_core_hash=claim["winner"]["protocol_core_hash"],
        run_tag=run_tag,
        receipt_dir=receipt_root,
    )
    job_dir = results_root / job["stage"] / job["phase"] / job["job_id"]
    validator = lambda value: validate_sealed_result(  # noqa: E731
        value, freeze_artifact=freeze
    )
    canonical_completion_kind = f"{completion_prefix}_refit"
    alias_completion_kind = f"{completion_prefix}_alias"

    # A completed canonical job can be authenticated from its frozen identity
    # and result without reopening the dataset or sealed outer-test indices.
    if canonical_job["job_id"] == job["job_id"]:
        existing_completion = job_dir / "paper1_job_completion.json"
        if existing_completion.exists() or existing_completion.is_symlink():
            existing_identity = _read_canonical_json(
                job_dir / "paper1_job_identity.json"
            )
            if (
                existing_identity.get("job") != job
                or existing_identity.get("campaign_run_tag") != run_tag
                or existing_identity.get("training_manifest_sha256")
                != manifest["manifest_sha256"]
                or existing_identity.get("freeze_artifact_sha256")
                != freeze["artifact_sha256"]
                or existing_identity.get("selection_artifact_sha256")
                != freeze["selection_artifact"]["artifact_sha256"]
                or existing_identity.get("frozen_champion") != claim["winner"]
            ):
                raise Paper1ExecutionError(
                    "completed sealed-report identity differs from this freeze"
                )
            completion, _ = _verify_refit_completion(
                existing_completion,
                identity_sha256=_canonical_sha256(existing_identity),
                expected_kind=canonical_completion_kind,
                result_validator=validator,
            )
            return completion

    if canonical_job["job_id"] != job["job_id"]:
        canonical_dir = (
            results_root / canonical_job["stage"] / canonical_job["phase"]
            / canonical_job["job_id"]
        )
        canonical_identity = _read_canonical_json(
            canonical_dir / "paper1_job_identity.json"
        )
        canonical_identity_sha = _canonical_sha256(canonical_identity)
        if (
            canonical_identity.get("job") != canonical_job
            or canonical_identity.get("campaign_run_tag") != run_tag
            or canonical_identity.get("freeze_artifact_sha256")
            != freeze["artifact_sha256"]
            or canonical_identity.get("selection_artifact_sha256")
            != freeze["selection_artifact"]["artifact_sha256"]
        ):
            raise Paper1ExecutionDependencyError(
                "canonical sealed-report identity is missing or foreign"
            )
        canonical_completion, canonical_result = _verify_refit_completion(
            canonical_dir / "paper1_job_completion.json",
            identity_sha256=canonical_identity_sha,
            expected_kind=canonical_completion_kind,
            result_validator=validator,
        )
        result = seal_sealed_result({
            **canonical_result,
            "execution_kind": "deduplicated_alias",
            "job": job,
            "canonical_job_id": canonical_job["job_id"],
            "canonical_result_sha256": canonical_result["result_sha256"],
        }, freeze_artifact=freeze)
        identity = {
            "schema": JOB_IDENTITY_SCHEMA,
            "executor_schema": EXECUTOR_SCHEMA,
            "execution_kind": f"{completion_prefix}_deduplicated_alias",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "complete_grid_sha256": manifest["complete_grid_sha256"],
            "machine_role": manifest["machine_role"],
            "job": job,
            "dataset": campaign_stage_contract(job["stage"])["dataset"],
            "campaign_run_tag": run_tag,
            "reporting_role": reporting_role,
            "selection_artifact_sha256": freeze["selection_artifact"][
                "artifact_sha256"
            ],
            "freeze_artifact_sha256": freeze["artifact_sha256"],
            "freeze_stage": freeze_stage,
            "canonical_job_id": canonical_job["job_id"],
            "canonical_identity_sha256": canonical_identity_sha,
            "canonical_completion_sha256": _canonical_sha256(
                canonical_completion
            ),
            "canonical_result_sha256": canonical_result["result_sha256"],
            "execution_runtime": attestation["runtime"],
            "execution_receipt_sha256": attestation["receipt_sha256"],
        }
        return _write_refit_result(
            job_dir=job_dir,
            identity=identity,
            result=result,
            result_filename="paper1_sealed_result.json",
            completion_kind=alias_completion_kind,
            validator=validator,
        )

    config = _refit_config(
        job=job,
        pipeline=claim["pipeline"],
        dofs=list(claim["selected_pair"]),
    )
    dataset = campaign_stage_contract(job["stage"])["dataset"]
    stage_cache = cache_root / job["stage"]
    stage_cache.mkdir(parents=True, exist_ok=True)
    X, y, groups, development_idx, outer_idx, partition = (
        _load_sealed_refit_data(
            config=config,
            dataset=dataset,
            data_root=data_root,
            cache_root=stage_cache,
        )
    )
    identity = {
        "schema": JOB_IDENTITY_SCHEMA,
        "executor_schema": EXECUTOR_SCHEMA,
        "execution_kind": completion_prefix,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "machine_role": manifest["machine_role"],
        "job": job,
        "dataset": dataset,
        "campaign_run_tag": run_tag,
        "reporting_role": reporting_role,
        "selection_artifact_sha256": freeze["selection_artifact"][
            "artifact_sha256"
        ],
        "freeze_artifact_sha256": freeze["artifact_sha256"],
        "freeze_stage": freeze_stage,
        "frozen_champion": claim["winner"],
        "partition": partition,
        "execution_runtime": attestation["runtime"],
        "execution_receipt_sha256": attestation["receipt_sha256"],
    }
    identity_path = job_dir / "paper1_job_identity.json"
    _write_or_verify_canonical(identity_path, identity)
    identity_sha = _canonical_sha256(identity)
    completion_path = job_dir / "paper1_job_completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        completion, _ = _verify_refit_completion(
            completion_path,
            identity_sha256=identity_sha,
            expected_kind=canonical_completion_kind,
            result_validator=validator,
        )
        return completion
    from core import task
    from training.robustness import run_post_freeze_stability

    provenance = {
        "freeze_artifact_sha256": freeze["artifact_sha256"],
        "selection_artifact_sha256": freeze["selection_artifact"][
            "artifact_sha256"
        ],
        "stage": job["stage"],
        "freeze_stage": freeze_stage,
        "reporting_role": reporting_role,
    }
    kwargs: dict[str, Any] = {
        "config": config,
        "params": claim["winner"]["params"],
        "X": X,
        "y": y,
        "groups": groups,
        "development_idx": development_idx,
        "sealed_outer_test_idx": outer_idx,
        "initialization_seeds": [job["initialization_seed"]],
        "n_epochs": claim["winner"]["frozen_checkpoint_epochs"],
        "max_epochs": TRAINING_EPOCHS,
        "n_scour_heads": task.n_scour_outputs(config),
        "checkpoint_path": str(job_dir / "paper1_robustness_checkpoint.json"),
        "provenance": provenance,
    }
    if fit_evaluate is not None:
        kwargs["fit_evaluate"] = fit_evaluate
    robustness = run_post_freeze_stability(**kwargs)
    result = seal_sealed_result({
        "schema": SEALED_RESULT_SCHEMA,
        "execution_kind": "refit",
        "campaign_run_tag": run_tag,
        "complete_grid_sha256": manifest["complete_grid_sha256"],
        "job": job,
        "reporting_role": reporting_role,
        "selection_permitted": False,
        "selection_artifact_sha256": freeze["selection_artifact"][
            "artifact_sha256"
        ],
        "freeze_artifact_sha256": freeze["artifact_sha256"],
        "freeze_stage": freeze_stage,
        "slot_resolution": freeze["slot_resolution"],
        "canonical_slot": claim["canonical_slot"],
        "pipeline": claim["pipeline"],
        "selected_pair": claim["selected_pair"],
        "frozen_champion": claim["winner"],
        "partition": partition,
        "canonical_job_id": job["job_id"],
        "canonical_result_sha256": None,
        "robustness": _json_value(robustness),
    }, freeze_artifact=freeze)
    return _write_refit_result(
        job_dir=job_dir,
        identity=identity,
        result=result,
        result_filename="paper1_sealed_result.json",
        completion_kind=canonical_completion_kind,
        validator=validator,
    )


def execute_post_freeze_stability_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any],
    *,
    fit_evaluate: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one stage-local frozen refit and sealed-test report."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_post_freeze_stability_job(job)
    return _execute_sealed_report_job(
        job,
        manifest,
        freeze_stage=job["stage"],
        phase_key="post_freeze_stability",
        reporting_role="primary_post_freeze_report_only",
        completion_prefix="post_freeze_stability",
        fit_evaluate=fit_evaluate,
    )


def execute_secondary_frozen_transfer_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any],
    *,
    fit_evaluate: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute F40-S frozen parameters on a downstream sealed test, report-only."""

    job, manifest = _validate_manifest_request(job, manifest)
    job = validate_secondary_frozen_transfer_job(job)
    return _execute_sealed_report_job(
        job,
        manifest,
        freeze_stage="F40-S",
        phase_key="secondary_frozen_transfer",
        reporting_role="secondary_nonselection",
        completion_prefix="secondary_frozen_transfer",
        fit_evaluate=fit_evaluate,
    )


def execute_manifest_job(
    job: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Dispatch one exact manifest job to its registered phase adapter."""

    job, manifest = _validate_manifest_request(job, manifest)
    phase = job["phase"]
    if phase == "f40s_factorial_hpo":
        return execute_f40s_factorial_hpo_job(job, manifest)
    if phase in {"f40s_selected_pair_hpo", "block_selected_pair_hpo"}:
        return execute_selected_pair_hpo_job(job, manifest)
    if phase == "f40s_development_adjudication":
        return execute_development_adjudication_job(job, manifest)
    if phase == "f40s_frozen_hyperparameter_channel_screen":
        return execute_channel_screen_job(job, manifest)
    if phase == "post_freeze_sealed_test_stability":
        return execute_post_freeze_stability_job(job, manifest)
    if phase == "secondary_frozen_hyperparameter_transfer":
        return execute_secondary_frozen_transfer_job(job, manifest)
    raise Paper1ExecutionError(f"unregistered Paper-1 training phase {phase!r}")
