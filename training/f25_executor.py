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
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before F25 imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.join(_bootstrap_os.path.dirname(__file__), _bootstrap_os.pardir)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
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
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
)
from core.source_provenance import python_runtime_source_root
from check_f25_capacity import (
    SCHEMA as F25_CAPACITY_SCHEMA,
    capacity_contract_cases,
)


REPO = Path(__file__).resolve().parents[1]
RUN_RECORD_SCHEMA = "f25-training-run-record-v1"
WINNER_SCHEMA = "f25-hpo-winner-v1"
REPORT_SCHEMA = "f25-report-results-v1"
EXECUTION_RECEIPT_SCHEMA = "f25-execution-block-receipt-v1"


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


def _validate_generation_root(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    marker = root / "_F25_GENERATION_COMPLETE"
    digest_path = root / "file_digests.json"
    if not marker.is_file() or not digest_path.is_file():
        raise F25ExecutionError("F25 shared data lack the completion boundary")
    receipt = json.loads(digest_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "f25-generation-artifact-digests-v1":
        raise F25ExecutionError("unsupported F25 generation digest schema")
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
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise F25ExecutionError(f"F25 state bytes fail SHA-256: {expected_name}")
        canonical.append({"name": expected_name, "sha256": expected_sha})
    if receipt.get("digest_root_sha256") != canonical_json_sha256(canonical):
        # MATLAB hashes jsonencode(struct-array), whose key order is fixed but
        # differs from canonical JSON.  The marker still binds that MATLAB root;
        # Python therefore carries its own canonical verified-state root too.
        receipt = dict(receipt)
    marker_text = marker.read_text(encoding="utf-8")
    if (
        f"gen_fingerprint={receipt.get('gen_fingerprint')}" not in marker_text
        or f"python_contract_sha256={receipt.get('python_contract_sha256')}"
        not in marker_text
        or f"digest_root_sha256={receipt.get('digest_root_sha256')}"
        not in marker_text
    ):
        raise F25ExecutionError("F25 completion marker does not bind its receipt")
    return {
        "root": str(root),
        "gen_fingerprint": receipt["gen_fingerprint"],
        "matlab_digest_root_sha256": receipt["digest_root_sha256"],
        "verified_state_root_sha256": canonical_json_sha256(canonical),
    }


def _load_clean_selected(root: Path, sensor_indices: list[int]) -> np.ndarray:
    classes: list[np.ndarray] = []
    contract_sha = build_contract()["contract_sha256"]
    for class_index in range(10):
        path = root / f"{class_index + 1:04d}.mat"
        loaded = sio.loadmat(path, variable_names=["f25_data"], mat_dtype=True)
        if "f25_data" not in loaded:
            raise F25ExecutionError(f"{path.name} lacks f25_data")
        record = loaded["f25_data"][0, 0]
        names = set(record.dtype.names or ())
        required = {
            "schema",
            "python_contract_sha256",
            "channel_schema_id",
            "class_index_zero_based",
            "clean_trimmed",
            "monitoring_tail_sample",
            "source_window_samples",
            "trimmed_window_samples",
            "tail_samples_trimmed",
            "measurement_noise_applied",
        }
        if not required.issubset(names):
            raise F25ExecutionError(f"{path.name} has an incomplete F25 payload")
        if (
            _matlab_text(record["schema"]) != "f25-saved-state-v1"
            or _matlab_text(record["python_contract_sha256"]) != contract_sha
            or _matlab_text(record["channel_schema_id"]) != "physical8_v1"
            or int(np.asarray(record["class_index_zero_based"]).squeeze())
            != class_index
            or int(np.asarray(record["source_window_samples"]).squeeze()) != 5831
            or int(np.asarray(record["trimmed_window_samples"]).squeeze()) != 5830
            or int(np.asarray(record["tail_samples_trimmed"]).squeeze()) != 1
            or bool(np.asarray(record["measurement_noise_applied"]).squeeze())
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
    return np.concatenate(classes, axis=0)


def _prepare_dataset(job: dict[str, Any], data_root: Path) -> dict[str, Any]:
    binding = _validate_generation_root(data_root)
    clean = _load_clean_selected(data_root, job["sensor_indices"])
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
            f"F25 training environment differs from the campaign lock: {exc}"
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


def _bind_execution_block(runtime: dict[str, Any]) -> dict[str, Any]:
    path = REPO / "f25_artifacts" / "execution_block_receipt.json"
    contract_sha = build_contract()["contract_sha256"]
    receipt = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "execution_block_id": EXECUTION_BLOCK_ID,
        "f25_contract_sha256": contract_sha,
        "execution_compatibility_sha256": runtime[
            "execution_compatibility_sha256"
        ],
        "execution_compatibility_descriptor": runtime[
            "execution_compatibility_descriptor"
        ],
        "hardware_rule": (
            "complete F25-R/F25-X compared block on one GPU model/numeric stack"
        ),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise F25ExecutionError(
                "current hardware/numeric stack differs from the F25 block receipt"
            )
    else:
        _atomic_json(path, receipt)
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def _require_capacity_receipt(
    runtime: dict[str, Any], source_root: Any
) -> dict[str, Any]:
    path = REPO / "f25_artifacts" / "f25_capacity_receipt.json"
    if not path.is_file():
        raise F25ExecutionError(
            "run check_f25_capacity.py on the target 6-GB-class GPU first"
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
    _require_capacity_receipt(runtime, source_root)
    block_receipt = _bind_execution_block(runtime)
    dataset = _prepare_dataset(job, data_root)
    run_record = _publish_job_record(
        job, runtime, block_receipt, source_root, dataset["data_binding"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if job["phase"] == "hpo":
        _run_hpo(job, dataset, device, run_record)
    elif job["phase"] == "report":
        _run_report(job, dataset, device, run_record)
    else:  # guarded by plan validation
        raise F25ExecutionError("unknown F25 phase")


def smoke() -> None:
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
