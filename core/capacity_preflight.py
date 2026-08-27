"""CUDA capacity qualification for the registered campaign workload.

Before any protocol-hashed Optuna study is created, the executing GPU must
successfully train one synthetic batch through the worst registered member of
each architecture search space.  The probe includes model construction,
forward pass, the five-head loss, backward pass, and one Adam update.  Its
canonical receipt is bound to the physical runtime, the executable capacity
policy, the reviewed Python source root, and the exact probe implementation.

This is a *capacity* qualification, not a performance benchmark.  It prevents
dispatching a multi-day study to a GPU which cannot accommodate the declared
batch/input/model envelope with a prospectively source-locked memory headroom.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from core.execution_environment import validate_execution_runtime
from core.hyperparameter_policy import (
    ARCHITECTURES,
    canonical_json_bytes,
    canonical_json_sha256,
)
from core.paper1_training_contract import (
    ELIGIBLE_SENSOR_INDICES,
    FACTORIAL_CELLS,
)
from core.source_provenance import python_runtime_source_root


CAPACITY_POLICY_SCHEMA = "ttbi-cuda-capacity-policy-v3"
CAPACITY_RECEIPT_SCHEMA = "ttbi-cuda-capacity-receipt-v2"
CAPACITY_ENVELOPE_SCHEMA = "ttbi-cuda-capacity-receipt-envelope-v2"
CAPACITY_RECEIPT_DIR_ENV = "TTBI_EXECUTION_RECEIPT_DIR"

CAPACITY_PREFLIGHT_POLICY = {
    "schema": CAPACITY_POLICY_SCHEMA,
    "campaign_accelerator": "cuda",
    "architectures": list(ARCHITECTURES),
    "search_point": "maximum registered structural dimensions",
    "batch_size": 32,
    "input_scope": {
        "selector": "authenticated_selected_pair",
        "input_channels": 2,
        "eligible_sensor_indices": list(ELIGIBLE_SENSOR_INDICES),
        "diagnostic_proxy_inputs_permitted": False,
    },
    # The longest production representation is the L99.6 RAW bridge crop:
    # 9,960 bridge samples plus the source-locked 1,831-sample post window.
    # PAA is checked separately because it exercises the same model cells with
    # its actual 512-segment preprocessing contract.
    "representation_input_shapes": {
        "RAW": [32, 2, 11_791],
        "PAA": [32, 2, 512],
    },
    "largest_raw_case": {
        "stage": "L99-M",
        "bridge_length_m": 99.6,
        "bridge_samples": 9_960,
        "post_window_samples": 1_831,
    },
    "output_heads": 5,
    "output_semantics": {
        "scour_heads": 3,
        "bearing_heads": 2,
    },
    "dtype": "float32",
    "operations": [
        "model construction",
        "forward",
        "five-head registered loss",
        "backward",
        "Adam step",
    ],
    "minimum_remaining_headroom": {
        "fraction_of_total_memory": 0.20,
        "absolute_bytes": 1_073_741_824,
        "rule": "max(fraction_of_total_memory * total, absolute_bytes)",
    },
    "peak_metric": "torch.cuda.max_memory_reserved",
    "failure_policy": "fatal before Optuna study creation",
}

_RECEIPT_KEYS = {
    "schema",
    "policy",
    "policy_sha256",
    "execution_runtime",
    "execution_environment_sha256",
    "python_runtime_source_root_sha256",
    "python_runtime_source_file_count",
    "implementation_files",
    "implementation_source_sha256",
    "measurements",
    "total_memory_bytes",
    "required_headroom_bytes",
    "worst_peak_allocated_bytes",
    "worst_peak_reserved_bytes",
    "minimum_observed_headroom_bytes",
    "passed",
}
_ENVELOPE_KEYS = {"schema", "receipt", "receipt_sha256"}
_MEASUREMENT_KEYS = {
    "architecture",
    "representation",
    "input_shape",
    "params",
    "params_sha256",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "total_memory_bytes",
    "headroom_bytes",
    "passed",
}
_IMPLEMENTATION_FILES = (
    "core/capacity_preflight.py",
    "core/hyperparameter_policy.py",
    "core/models.py",
    "core/temporal_pooling.py",
    "training/trainer.py",
)
_HEX = frozenset("0123456789abcdef")


class CapacityPreflightError(RuntimeError):
    """The executing GPU does not satisfy the campaign capacity contract."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def capacity_policy_sha256() -> str:
    return canonical_json_sha256(CAPACITY_PREFLIGHT_POLICY)


def _implementation_source_sha256() -> str:
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        path = repo / relative
        if not path.is_file():
            raise CapacityPreflightError(
                f"capacity implementation source is missing: {relative}"
            )
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _architecture_config(architecture: str) -> dict:
    cells = {cell.cell_id: cell for cell in FACTORIAL_CELLS}
    if architecture not in cells:
        raise CapacityPreflightError(
            f"unregistered capacity architecture {architecture!r}"
        )
    cell = cells[architecture]
    input_shape = CAPACITY_PREFLIGHT_POLICY[
        "representation_input_shapes"
    ][cell.representation]
    return {
        "name": f"capacity_{architecture}",
        "name_short": architecture,
        "method": cell.representation,
        "model_type": "1D_MODULAR",
        "use_space2vec": cell.position_encoding,
        "use_lstm": cell.lstm,
        "use_nhits": cell.multi_rate_pooling,
        "task": "regression",
        "target_supports": [2, 3, 4],
        "bearing_targets": ["left", "right"],
        # Sensor identity does not change tensor capacity; use one registered
        # eligible pair and bind the cardinality to ``input_scope`` above.
        "dofs": list(ELIGIBLE_SENSOR_INDICES[:2]),
        "n_segments": input_shape[2],
    }


def _capacity_input_shape(config: dict) -> tuple[int, int, int]:
    try:
        shape = CAPACITY_PREFLIGHT_POLICY[
            "representation_input_shapes"
        ][config["method"]]
    except KeyError as exc:
        raise CapacityPreflightError(
            "capacity configuration has an unregistered representation"
        ) from exc
    if config.get("n_segments") != shape[2]:
        raise CapacityPreflightError(
            "capacity configuration sequence length differs from its "
            "registered representation"
        )
    expected_dofs = list(ELIGIBLE_SENSOR_INDICES[:2])
    if (
        config.get("dofs") != expected_dofs
        or len(expected_dofs) != shape[1]
        or shape[1]
        != CAPACITY_PREFLIGHT_POLICY["input_scope"]["input_channels"]
    ):
        raise CapacityPreflightError(
            "capacity configuration is not the registered eligible pair"
        )
    return tuple(shape)


def _maximum_registered_params(config: dict, search_space: dict) -> dict:
    """Derive the structural maximum directly from the live search space."""

    base = search_space["base"]
    n_conv = int(base["n_conv_layers"][2])
    n_dense = int(base["n_dense_layers"][2])
    params: dict[str, Any] = {
        "n_conv_layers": n_conv,
        "n_dense_layers": n_dense,
        "lr": float(base["lr"][2]),
        "weight_decay": float(base["weight_decay"][2]),
    }
    for index in range(n_conv):
        for key, spec in search_space["per_conv_layer"].items():
            kind = spec[0]
            if key == "pooling":
                value = False
            elif kind in {"int", "int_step", "float", "logfloat"}:
                value = spec[2]
            elif kind == "cat":
                value = max(spec[1])
            else:
                raise CapacityPreflightError(
                    f"unsupported search-space kind {kind!r}"
                )
            params[f"{key}_l{index}"] = value
    for index in range(n_dense):
        for key, spec in search_space["per_dense_layer"].items():
            params[f"{key}_l{index}"] = spec[2]
    if config["use_lstm"]:
        params["lstm_num_layers"] = search_space["lstm"][
            "lstm_num_layers"
        ][2]
        params["lstm_hidden_size"] = search_space["lstm"][
            "lstm_hidden_size"
        ][2]
        params["lstm_dropout"] = search_space["lstm"]["lstm_dropout"][2]
    if config["use_nhits"]:
        choices = search_space["nhits"]["nhits_pool_rates_key"][1]
        params["nhits_pool_rates_key"] = max(
            choices, key=lambda value: (len(value.split("_")), value)
        )
    return params


class _ExactTrial:
    """Minimal trial which proves the maximum point is valid in `_suggest_params`."""

    def __init__(self, values: dict):
        self.values = values

    def suggest_int(self, name, low, high, step=1):
        value = self.values[name]
        if low != high or value != low:
            raise CapacityPreflightError(
                f"capacity singleton registration drifted for {name!r}"
            )
        return value

    def suggest_float(self, name, low, high, **_kwargs):
        value = self.values[name]
        if low != high or float(value) != float(low):
            raise CapacityPreflightError(
                f"capacity singleton registration drifted for {name!r}"
            )
        return float(value)

    def suggest_categorical(self, name, choices):
        if list(choices) != [self.values[name]]:
            raise CapacityPreflightError(
                f"capacity singleton registration drifted for {name!r}"
            )
        return self.values[name]


def registered_capacity_cases() -> list[tuple[str, dict, dict]]:
    """Return all 16 validated architecture/config/worst-parameter cases."""

    # Lazy import prevents a pipeline -> capacity -> trainer import cycle.
    from training.trainer import SEARCH_SPACE, TRAIN_PROTOCOL, _suggest_params

    if TRAIN_PROTOCOL.get("batch_size") != CAPACITY_PREFLIGHT_POLICY["batch_size"]:
        raise CapacityPreflightError(
            "training batch size diverged from the capacity policy"
        )
    cases = []
    for architecture in ARCHITECTURES:
        config = _architecture_config(architecture)
        params = _maximum_registered_params(config, SEARCH_SPACE)
        check_config = dict(config, frozen_hyperparameters=params)
        suggested = _suggest_params(_ExactTrial(params), check_config)
        if suggested != params:
            raise CapacityPreflightError(
                f"capacity point for {architecture} does not reproduce exactly"
            )
        cases.append((architecture, config, params))
    return cases


def _run_cuda_probe(
    architecture: str,
    config: dict,
    params: dict,
    runtime: dict,
) -> dict:
    """Execute one real worst-case synthetic training step."""

    import torch

    from core import task
    from core.models import build_model
    from training.trainer import (
        TRAIN_PROTOCOL,
        make_optimizer,
    )

    accelerator = runtime["execution_environment_descriptor"]["accelerator"]
    if not torch.cuda.is_available():
        raise CapacityPreflightError(
            "campaign runtime is CUDA but torch.cuda.is_available() is false"
        )
    device_index = int(accelerator["device_index"])
    current_index = int(torch.cuda.current_device())
    if current_index != device_index:
        raise CapacityPreflightError(
            f"current CUDA device {current_index} differs from attested "
            f"device {device_index}"
        )
    props = torch.cuda.get_device_properties(device_index)
    total_memory = int(props.total_memory)
    if total_memory != int(accelerator["total_memory_bytes"]):
        raise CapacityPreflightError(
            "current CUDA total memory differs from the execution attestation"
        )
    device = torch.device(f"cuda:{device_index}")
    input_shape = _capacity_input_shape(config)
    model = optimizer = features = target = output = loss = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        model, n_outputs = build_model(
            config,
            params,
            input_shape,
            device,
        )
        if n_outputs != CAPACITY_PREFLIGHT_POLICY["output_heads"]:
            raise CapacityPreflightError(
                f"{architecture} capacity model produced {n_outputs} heads"
            )
        optimizer = make_optimizer(
            model.parameters(), params, TRAIN_PROTOCOL["optimizer"]
        )
        criterion = task.make_criterion(
            config, TRAIN_PROTOCOL["loss"]
        ).to(device)
        features = torch.randn(
            input_shape,
            device=device,
            dtype=torch.float32,
        )
        target = torch.randn(
            CAPACITY_PREFLIGHT_POLICY["batch_size"],
            CAPACITY_PREFLIGHT_POLICY["output_heads"],
            device=device,
            dtype=torch.float32,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        if tuple(output.shape) != tuple(target.shape):
            raise CapacityPreflightError(
                f"{architecture} output shape {tuple(output.shape)!r} differs "
                f"from the required {tuple(target.shape)!r}"
            )
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        return {
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
            "total_memory_bytes": total_memory,
        }
    finally:
        del loss, output, target, features, optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _strict_bytes(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CapacityPreflightError(f"{label} must be an integer byte count")
    value = int(value)
    if value < (1 if positive else 0):
        raise CapacityPreflightError(f"{label} has an invalid byte count")
    return value


def validate_capacity_receipt(
    envelope: dict,
    *,
    expected_runtime: dict | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
    expected_implementation_source_sha256: str | None = None,
) -> dict:
    """Validate a complete receipt envelope and return its canonical copy."""

    if not isinstance(envelope, dict):
        raise CapacityPreflightError("capacity receipt envelope must be a mapping")
    value = json.loads(canonical_json_bytes(envelope))
    if set(value) != _ENVELOPE_KEYS:
        raise CapacityPreflightError(
            "capacity receipt envelope fields differ from the contract"
        )
    if value["schema"] != CAPACITY_ENVELOPE_SCHEMA:
        raise CapacityPreflightError("unsupported capacity envelope schema")
    receipt = value["receipt"]
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise CapacityPreflightError(
            "capacity receipt fields differ from the contract"
        )
    if (
        not _is_sha256(value["receipt_sha256"])
        or value["receipt_sha256"] != canonical_json_sha256(receipt)
    ):
        raise CapacityPreflightError("capacity receipt SHA-256 is invalid")
    if receipt["schema"] != CAPACITY_RECEIPT_SCHEMA:
        raise CapacityPreflightError("unsupported capacity receipt schema")
    if receipt["policy"] != json.loads(
        canonical_json_bytes(CAPACITY_PREFLIGHT_POLICY)
    ):
        raise CapacityPreflightError("capacity receipt embeds another policy")
    if (
        not _is_sha256(receipt["policy_sha256"])
        or receipt["policy_sha256"] != capacity_policy_sha256()
    ):
        raise CapacityPreflightError("capacity policy SHA-256 is invalid")

    runtime = validate_execution_runtime(receipt["execution_runtime"])
    accelerator = runtime["execution_environment_descriptor"]["accelerator"]
    if accelerator["backend"] != "cuda":
        raise CapacityPreflightError(
            "campaign capacity receipt is valid only for CUDA runtimes"
        )
    if (
        receipt["execution_environment_sha256"]
        != runtime["execution_environment_sha256"]
    ):
        raise CapacityPreflightError(
            "capacity receipt execution SHA differs from its runtime"
        )
    if expected_runtime is not None and runtime != validate_execution_runtime(
        expected_runtime
    ):
        raise CapacityPreflightError(
            "capacity receipt belongs to another execution runtime"
        )

    source_sha = receipt["python_runtime_source_root_sha256"]
    source_count = _strict_bytes(
        receipt["python_runtime_source_file_count"],
        "source file count",
        positive=True,
    )
    if expected_source_root_sha256 is None:
        live = python_runtime_source_root()
        expected_source_root_sha256 = live.sha256
        if expected_source_file_count is None:
            expected_source_file_count = live.file_count
    if (
        not _is_sha256(source_sha)
        or source_sha != expected_source_root_sha256
        or (
            expected_source_file_count is not None
            and source_count != expected_source_file_count
        )
    ):
        raise CapacityPreflightError(
            "capacity receipt runtime source identity differs from expected"
        )
    if receipt["implementation_files"] != list(_IMPLEMENTATION_FILES):
        raise CapacityPreflightError(
            "capacity receipt implementation file boundary drifted"
        )
    implementation_sha = receipt["implementation_source_sha256"]
    if expected_implementation_source_sha256 is None:
        expected_implementation_source_sha256 = (
            _implementation_source_sha256()
        )
    if (
        not _is_sha256(implementation_sha)
        or implementation_sha != expected_implementation_source_sha256
    ):
        raise CapacityPreflightError(
            "capacity receipt implementation source SHA-256 differs"
        )

    total = _strict_bytes(
        receipt["total_memory_bytes"], "total GPU memory", positive=True
    )
    if total != int(accelerator["total_memory_bytes"]):
        raise CapacityPreflightError(
            "capacity total memory differs from execution attestation"
        )
    required = max(
        int(CAPACITY_PREFLIGHT_POLICY[
            "minimum_remaining_headroom"
        ]["absolute_bytes"]),
        math.ceil(
            total
            * float(CAPACITY_PREFLIGHT_POLICY[
                "minimum_remaining_headroom"
            ]["fraction_of_total_memory"])
        ),
    )
    if receipt["required_headroom_bytes"] != required:
        raise CapacityPreflightError(
            "capacity receipt uses a different headroom threshold"
        )

    measurements = receipt["measurements"]
    if not isinstance(measurements, list):
        raise CapacityPreflightError("capacity measurements must be a list")
    if [item.get("architecture") for item in measurements] != list(
        ARCHITECTURES
    ):
        raise CapacityPreflightError(
            "capacity receipt does not cover all architectures in order"
        )
    expected_params = {
        architecture: json.loads(canonical_json_bytes(params))
        for architecture, _config, params in registered_capacity_cases()
    }
    expected_configs = {
        architecture: config
        for architecture, config, _params in registered_capacity_cases()
    }
    peak_allocated_values = []
    peak_reserved_values = []
    headroom_values = []
    for measurement in measurements:
        if not isinstance(measurement, dict) or set(
            measurement
        ) != _MEASUREMENT_KEYS:
            raise CapacityPreflightError(
                "capacity measurement fields differ from the contract"
            )
        params = measurement["params"]
        config = expected_configs[measurement["architecture"]]
        expected_shape = list(_capacity_input_shape(config))
        if (
            measurement["representation"] != config["method"]
            or measurement["input_shape"] != expected_shape
        ):
            raise CapacityPreflightError(
                "capacity measurement representation/input shape differs "
                "from the live registered case"
            )
        if not isinstance(params, dict) or not params:
            raise CapacityPreflightError(
                "capacity measurement lacks its search-space point"
            )
        if params != expected_params[measurement["architecture"]]:
            raise CapacityPreflightError(
                "capacity measurement does not use the live registered "
                f"worst-case point for {measurement['architecture']}"
            )
        if (
            not _is_sha256(measurement["params_sha256"])
            or measurement["params_sha256"]
            != canonical_json_sha256(params)
        ):
            raise CapacityPreflightError(
                "capacity measurement parameter SHA-256 is invalid"
            )
        allocated = _strict_bytes(
            measurement["peak_memory_allocated_bytes"],
            "peak allocated memory",
        )
        reserved = _strict_bytes(
            measurement["peak_memory_reserved_bytes"],
            "peak reserved memory",
        )
        measurement_total = _strict_bytes(
            measurement["total_memory_bytes"],
            "measurement total memory",
            positive=True,
        )
        headroom = _strict_bytes(
            measurement["headroom_bytes"], "measurement headroom"
        )
        if (
            measurement_total != total
            or allocated > reserved
            or reserved > total
            or headroom != total - reserved
            or measurement["passed"] is not (headroom >= required)
        ):
            raise CapacityPreflightError(
                "capacity measurement arithmetic is inconsistent"
            )
        peak_allocated_values.append(allocated)
        peak_reserved_values.append(reserved)
        headroom_values.append(headroom)
    if (
        receipt["worst_peak_allocated_bytes"] != max(peak_allocated_values)
        or receipt["worst_peak_reserved_bytes"] != max(peak_reserved_values)
        or receipt["minimum_observed_headroom_bytes"] != min(headroom_values)
    ):
        raise CapacityPreflightError(
            "capacity receipt aggregate measurements are inconsistent"
        )
    passed = all(item["passed"] is True for item in measurements)
    if receipt["passed"] is not passed or not passed:
        raise CapacityPreflightError(
            "CUDA capacity preflight did not preserve required headroom"
        )
    return value


def run_capacity_preflight(
    execution_runtime: dict,
    *,
    probe_runner: Callable[[str, dict, dict, dict], dict] | None = None,
    source_root_sha256: str | None = None,
    source_file_count: int | None = None,
    implementation_source_sha256: str | None = None,
) -> dict:
    """Run all four probes and return an authenticated receipt envelope."""

    runtime = validate_execution_runtime(execution_runtime)
    accelerator = runtime["execution_environment_descriptor"]["accelerator"]
    if accelerator["backend"] != "cuda":
        raise CapacityPreflightError(
            "the scientific campaign is CUDA-only; refusing CPU preflight"
        )
    if source_root_sha256 is None or source_file_count is None:
        live = python_runtime_source_root()
        source_root_sha256 = source_root_sha256 or live.sha256
        source_file_count = (
            live.file_count if source_file_count is None else source_file_count
        )
    implementation_source_sha256 = (
        implementation_source_sha256 or _implementation_source_sha256()
    )
    total = int(accelerator["total_memory_bytes"])
    required = max(
        int(CAPACITY_PREFLIGHT_POLICY[
            "minimum_remaining_headroom"
        ]["absolute_bytes"]),
        math.ceil(
            total
            * float(CAPACITY_PREFLIGHT_POLICY[
                "minimum_remaining_headroom"
            ]["fraction_of_total_memory"])
        ),
    )
    runner = probe_runner or _run_cuda_probe
    measurements = []
    for architecture, config, params in registered_capacity_cases():
        try:
            raw = runner(architecture, config, params, runtime)
        except BaseException as exc:
            # In particular, CUDA OOM is never converted into a retryable
            # Optuna FAIL.  The preflight terminates before a study exists.
            raise CapacityPreflightError(
                f"CUDA capacity probe failed fatally for {architecture}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != {
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "total_memory_bytes",
        }:
            raise CapacityPreflightError(
                f"malformed probe result for {architecture}"
            )
        allocated = _strict_bytes(
            raw["peak_memory_allocated_bytes"], "peak allocated memory"
        )
        reserved = _strict_bytes(
            raw["peak_memory_reserved_bytes"], "peak reserved memory"
        )
        measurement_total = _strict_bytes(
            raw["total_memory_bytes"], "probe total memory", positive=True
        )
        if allocated > reserved or reserved > measurement_total:
            raise CapacityPreflightError(
                f"invalid CUDA peak measurements for {architecture}"
            )
        headroom = measurement_total - reserved
        measurements.append({
            "architecture": architecture,
            "representation": config["method"],
            "input_shape": list(_capacity_input_shape(config)),
            "params": params,
            "params_sha256": canonical_json_sha256(params),
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
            "total_memory_bytes": measurement_total,
            "headroom_bytes": headroom,
            "passed": measurement_total == total and headroom >= required,
        })

    receipt = {
        "schema": CAPACITY_RECEIPT_SCHEMA,
        "policy": CAPACITY_PREFLIGHT_POLICY,
        "policy_sha256": capacity_policy_sha256(),
        "execution_runtime": runtime,
        "execution_environment_sha256": runtime[
            "execution_environment_sha256"
        ],
        "python_runtime_source_root_sha256": source_root_sha256,
        "python_runtime_source_file_count": source_file_count,
        "implementation_files": list(_IMPLEMENTATION_FILES),
        "implementation_source_sha256": implementation_source_sha256,
        "measurements": measurements,
        "total_memory_bytes": total,
        "required_headroom_bytes": required,
        "worst_peak_allocated_bytes": max(
            item["peak_memory_allocated_bytes"] for item in measurements
        ),
        "worst_peak_reserved_bytes": max(
            item["peak_memory_reserved_bytes"] for item in measurements
        ),
        "minimum_observed_headroom_bytes": min(
            item["headroom_bytes"] for item in measurements
        ),
        "passed": all(item["passed"] for item in measurements),
    }
    envelope = {
        "schema": CAPACITY_ENVELOPE_SCHEMA,
        "receipt": receipt,
        "receipt_sha256": canonical_json_sha256(receipt),
    }
    return validate_capacity_receipt(
        envelope,
        expected_runtime=runtime,
        expected_source_root_sha256=source_root_sha256,
        expected_source_file_count=source_file_count,
        expected_implementation_source_sha256=implementation_source_sha256,
    )


def _resolved_receipt_dir(
    receipt_dir: str | os.PathLike[str] | None,
) -> Path:
    configured = (
        receipt_dir
        if receipt_dir is not None
        else os.environ.get(CAPACITY_RECEIPT_DIR_ENV)
    )
    if configured is None or not str(configured).strip():
        raise CapacityPreflightError(
            f"{CAPACITY_RECEIPT_DIR_ENV} must name an explicit absolute "
            "durable receipt directory"
        )
    destination = Path(configured)
    if not destination.is_absolute():
        raise CapacityPreflightError(
            f"{CAPACITY_RECEIPT_DIR_ENV} must be absolute, got {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise CapacityPreflightError(
            f"capacity receipt directory is not a regular directory: "
            f"{destination}"
        )
    return destination


def _receipt_identity_sha(
    runtime: dict,
    *,
    source_root_sha256: str,
    source_file_count: int,
    implementation_source_sha256: str,
) -> str:
    return canonical_json_sha256({
        "execution_runtime": runtime,
        "capacity_policy_sha256": capacity_policy_sha256(),
        "python_runtime_source_root_sha256": source_root_sha256,
        "python_runtime_source_file_count": source_file_count,
        "implementation_source_sha256": implementation_source_sha256,
    })


def capacity_receipt_path(
    execution_runtime: dict,
    *,
    receipt_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the content-addressed durable path for the live qualification."""

    destination = _resolved_receipt_dir(receipt_dir)
    runtime = validate_execution_runtime(execution_runtime)
    live = python_runtime_source_root()
    implementation_sha = _implementation_source_sha256()
    identity_sha = _receipt_identity_sha(
        runtime,
        source_root_sha256=live.sha256,
        source_file_count=live.file_count,
        implementation_source_sha256=implementation_sha,
    )
    return destination / (
        f"capacity_preflight_{runtime['execution_block']}_{identity_sha}.json"
    )


def write_capacity_receipt(
    path: str | os.PathLike[str],
    envelope: dict,
    *,
    expected_runtime: dict | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
    expected_implementation_source_sha256: str | None = None,
    require_absent: bool = False,
) -> str:
    """Atomically publish canonical receipt bytes without replacing evidence.

    ``require_absent=True`` is the qualification-publication mode: a target
    created by another process, even with identical bytes, is refused rather
    than being mistaken for this invocation's fresh evidence.
    """

    if not isinstance(require_absent, bool):
        raise CapacityPreflightError("require_absent must be boolean")

    value = validate_capacity_receipt(
        envelope,
        expected_runtime=expected_runtime,
        expected_source_root_sha256=expected_source_root_sha256,
        expected_source_file_count=expected_source_file_count,
        expected_implementation_source_sha256=(
            expected_implementation_source_sha256
        ),
    )
    payload = canonical_json_bytes(value)
    destination = Path(path)
    if not destination.is_absolute():
        raise CapacityPreflightError(
            f"capacity receipt path must be absolute: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise CapacityPreflightError(
            "capacity receipt parent is not a regular directory"
        )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard-link publish is atomic and, unlike os.replace(), cannot
            # overwrite a concurrently created qualification.
            os.link(temporary, destination)
        except FileExistsError:
            if require_absent:
                raise CapacityPreflightError(
                    "fresh capacity publication target already exists: "
                    f"{destination}"
                )
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != payload
            ):
                raise CapacityPreflightError(
                    "refusing to overwrite a differing existing capacity "
                    f"receipt: {destination}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()
    return value["receipt_sha256"]


def load_capacity_receipt(
    path: str | os.PathLike[str],
    *,
    expected_runtime: dict | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
    expected_implementation_source_sha256: str | None = None,
) -> dict:
    """Load exact canonical bytes and revalidate every live identity binding."""

    source = Path(path)
    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
    ):
        raise CapacityPreflightError(
            f"capacity receipt is not an absolute regular file: {source}"
        )
    payload = source.read_bytes()
    try:
        parsed = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapacityPreflightError(
            "capacity receipt is not canonical ASCII JSON"
        ) from exc
    if payload != canonical_json_bytes(parsed):
        raise CapacityPreflightError(
            "capacity receipt file is not in canonical byte form"
        )
    return validate_capacity_receipt(
        parsed,
        expected_runtime=expected_runtime,
        expected_source_root_sha256=expected_source_root_sha256,
        expected_source_file_count=expected_source_file_count,
        expected_implementation_source_sha256=(
            expected_implementation_source_sha256
        ),
    )


def ensure_capacity_preflight(
    execution_runtime: dict,
    *,
    receipt_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Create or reload the durable live receipt before any Optuna study.

    ``receipt_dir=None`` reads the required absolute
    ``TTBI_EXECUTION_RECEIPT_DIR``.  Every call rereads canonical bytes from
    disk, so reuse across fresh Python processes is the normal path rather than
    an untestable module-level cache.
    """

    runtime = validate_execution_runtime(execution_runtime)
    live = python_runtime_source_root()
    implementation_sha = _implementation_source_sha256()
    destination = _resolved_receipt_dir(receipt_dir)
    identity_sha = _receipt_identity_sha(
        runtime,
        source_root_sha256=live.sha256,
        source_file_count=live.file_count,
        implementation_source_sha256=implementation_sha,
    )
    path = destination / (
        f"capacity_preflight_{runtime['execution_block']}_{identity_sha}.json"
    )
    validation = {
        "expected_runtime": runtime,
        "expected_source_root_sha256": live.sha256,
        "expected_source_file_count": live.file_count,
        "expected_implementation_source_sha256": implementation_sha,
    }
    if path.exists():
        return load_capacity_receipt(path, **validation)
    envelope = run_capacity_preflight(
        runtime,
        source_root_sha256=live.sha256,
        source_file_count=live.file_count,
        implementation_source_sha256=implementation_sha,
    )
    write_capacity_receipt(path, envelope, **validation)
    return load_capacity_receipt(path, **validation)
