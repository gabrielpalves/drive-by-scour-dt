"""Capacity preflight for the worst registered F25 RAW pair envelopes."""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
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
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from core.execution_environment import (
    current_execution_environment,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
from core.f25_experiment_contract import SOURCE_CNN_SEARCH_SPACE, build_contract
from core.f25_models import build_f25_model, parameter_count
from core.f25_training_contract import canonical_json_sha256
from core.source_provenance import python_runtime_source_root


REPO = Path(__file__).resolve().parent
SCHEMA = "f25-capacity-preflight-v4"
CAPACITY_RECEIPT_ADDRESS_SCHEMA = (
    "f25-capacity-receipt-runtime-source-address-v1"
)
CAPACITY_RECEIPT_DIRECTORY = "capacity_receipts"
PAIR_INPUT_CHANNELS = 2
CAPACITY_KERNELS = (
    min(SOURCE_CNN_SEARCH_SPACE.convolution_kernel_sizes),
    max(SOURCE_CNN_SEARCH_SPACE.convolution_kernel_sizes),
)
CASES = tuple(
    (
        f"RAW-pair-b48-five-layer-k{kernel}",
        kernel,
        PAIR_INPUT_CHANNELS,
        "registered-job-envelope",
    )
    for kernel in CAPACITY_KERNELS
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def capacity_receipt_path(
    repo: str | Path,
    *,
    execution_environment_sha256_value: str,
    python_runtime_source_sha256: str,
) -> Path:
    """Return the collision-free receipt path for one exact PC/source pair."""

    if not _is_sha256(execution_environment_sha256_value):
        raise RuntimeError("F25 capacity runtime identity is not a SHA-256")
    if not _is_sha256(python_runtime_source_sha256):
        raise RuntimeError("F25 capacity source-root identity is not a SHA-256")
    return (
        Path(repo)
        / "f25_artifacts"
        / CAPACITY_RECEIPT_DIRECTORY
        / execution_environment_sha256_value
        / f"{python_runtime_source_sha256}.json"
    )


def _params(kernel: int) -> dict[str, Any]:
    space = SOURCE_CNN_SEARCH_SPACE
    params: dict[str, Any] = {
        "n_conv_layers": max(space.convolution_layer_counts),
        "dense_units": max(space.dense_units),
        "batch_size": max(space.batch_sizes),
        "learning_rate": max(space.learning_rate_range),
    }
    if kernel not in space.convolution_kernel_sizes:
        raise RuntimeError("F25 capacity kernel is outside the registered search space")
    for index in range(params["n_conv_layers"]):
        params[f"filters_l{index}"] = max(space.convolution_filters)
        params[f"kernel_l{index}"] = kernel
        params[f"pool_l{index}"] = False
    return params


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("F25 capacity receipt parent is not a regular directory")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                "F25 capacity receipt for this exact runtime/source already "
                "exists; preserve it as evidence instead of replacing it"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capacity_contract_cases() -> list[dict[str, Any]]:
    """Return the exact ordered cases that an accepted receipt must execute."""

    rows = []
    for case_id, kernel, in_channels, role in CASES:
        params = _params(kernel)
        with torch.device("meta"):
            model = build_f25_model(
                arm_id="RAW-CNN",
                in_channels=in_channels,
                params=params,
                device="meta",
            )
        rows.append(
            {
                "case_id": case_id,
                "role": role,
                "arm_id": "RAW-CNN",
                "input_shape": [48, in_channels, 5830],
                "params": params,
                "parameter_count": parameter_count(model),
                "flattened_units": model.flattened_units,
            }
        )
    return rows


def run(output: Path | None, contract_only: bool) -> dict[str, Any]:
    cases = capacity_contract_cases()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "case_contract_sha256": canonical_json_sha256(cases),
        "cases": cases,
        "accepted": False,
        "contract_only": bool(contract_only),
    }
    if contract_only:
        print(
            "PASS F25 capacity contract: worst registered RAW pair envelope "
            "at batch 48, five layers, no pooling, and kernels 2/5"
        )
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("F25 production capacity preflight requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from core.environment import load_environment_lock, validate_environment_lock

    environment_lock = load_environment_lock(
        REPO / "environment" / "campaign-py313-cu128.json"
    )
    validate_environment_lock(environment_lock)
    descriptor = current_execution_environment()
    source = python_runtime_source_root(REPO)
    runtime_sha256 = execution_environment_sha256(descriptor)
    canonical_output = capacity_receipt_path(
        REPO,
        execution_environment_sha256_value=runtime_sha256,
        python_runtime_source_sha256=source.sha256,
    )
    if output is None:
        output = canonical_output
    elif os.path.normcase(str(output.resolve())) != os.path.normcase(
        str(canonical_output.resolve())
    ):
        raise RuntimeError(
            "F25 production capacity receipts must use their canonical "
            f"runtime/source address: {canonical_output}"
        )
    device = torch.device("cuda")
    device_total_memory_bytes = int(
        torch.cuda.get_device_properties(device).total_memory
    )
    measurements = []
    for row in cases:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_f25_model(
            arm_id="RAW-CNN",
            in_channels=row["input_shape"][1],
            params=row["params"],
            device=device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=row["params"]["learning_rate"]
        )
        inputs = torch.randn(*row["input_shape"], device=device)
        labels = torch.arange(row["input_shape"][0], device=device) % 10
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if not (0 < peak_allocated <= peak_reserved <= device_total_memory_bytes):
            raise RuntimeError("F25 CUDA peak-memory accounting is invalid")
        measurements.append(
            {
                "case_id": row["case_id"],
                "loss": float(loss.detach().cpu()),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
        )
        del model, optimizer, inputs, labels, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    result.update(
        {
            "accepted": True,
            "capacity_receipt_address_schema": (
                CAPACITY_RECEIPT_ADDRESS_SCHEMA
            ),
            "environment_lock_sha256": environment_lock["sha256"],
            "execution_environment_descriptor": descriptor,
            "execution_environment_sha256": runtime_sha256,
            "execution_compatibility_descriptor": execution_compatibility_descriptor(
                descriptor
            ),
            "execution_compatibility_sha256": execution_compatibility_sha256(
                descriptor
            ),
            "python_runtime_source_sha256": source.sha256,
            "python_runtime_source_file_count": source.file_count,
            "device_total_memory_bytes": device_total_memory_bytes,
            "measurements": measurements,
        }
    )
    if python_runtime_source_root(REPO) != source:
        raise RuntimeError("F25 source root changed during capacity qualification")
    if current_execution_environment() != descriptor:
        raise RuntimeError(
            "F25 execution environment changed during capacity qualification"
        )
    unsigned = dict(result)
    result["receipt_sha256"] = canonical_json_sha256(unsigned)
    _atomic_json(output, result)
    print(f"PASS F25 CUDA capacity preflight -> {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "optional assertion of the canonical content-addressed output; "
            "normally omit"
        ),
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.contract_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
