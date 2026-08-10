"""Capacity preflight for F25 RAW jobs and the conservative full-8 stress."""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before F25 imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
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
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.execution_environment import (
    current_execution_environment,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
from core.f25_experiment_contract import build_contract
from core.f25_models import build_f25_model, parameter_count
from core.f25_training_contract import canonical_json_sha256
from core.source_provenance import python_runtime_source_root


REPO = Path(__file__).resolve().parent
SCHEMA = "f25-capacity-preflight-v2"
CASES = (
    ("RAW-pair-b48-five-layer-k2", 2, 2, "registered-job-envelope"),
    ("RAW-pair-b48-five-layer-k5", 5, 2, "registered-job-envelope"),
    (
        "RAW-full8-conservative-b48-five-layer-k2",
        2,
        8,
        "conservative-nonjob-dispatch-stress",
    ),
    (
        "RAW-full8-conservative-b48-five-layer-k5",
        5,
        8,
        "conservative-nonjob-dispatch-stress",
    ),
)


def _params(kernel: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_conv_layers": 5,
        "dense_units": 64,
        "batch_size": 48,
        "learning_rate": 1.0e-2,
    }
    for index in range(5):
        params[f"filters_l{index}"] = 128
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
                "F25 capacity receipt already exists; preserve it and use a "
                "fresh workspace for any replacement qualification"
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


def run(output: Path, contract_only: bool) -> dict[str, Any]:
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
            "PASS F25 capacity contract: registered pair envelope plus "
            "conservative full-8 RAW batch-48 five-layer no-pool k2/k5 stress"
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
            "environment_lock_sha256": environment_lock["sha256"],
            "execution_environment_descriptor": descriptor,
            "execution_environment_sha256": execution_environment_sha256(descriptor),
            "execution_compatibility_descriptor": execution_compatibility_descriptor(
                descriptor
            ),
            "execution_compatibility_sha256": execution_compatibility_sha256(
                descriptor
            ),
            "python_runtime_source_sha256": source.sha256,
            "device_total_memory_bytes": device_total_memory_bytes,
            "measurements": measurements,
        }
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
        default=REPO / "f25_artifacts" / "f25_capacity_receipt.json",
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.contract_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
