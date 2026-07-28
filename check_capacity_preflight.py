"""Mutation and behaviour checks for the CUDA capacity qualification."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from core.capacity_preflight import (
    ARCHITECTURES,
    CAPACITY_PREFLIGHT_POLICY,
    CapacityPreflightError,
    capacity_receipt_path,
    ensure_capacity_preflight,
    load_capacity_receipt,
    registered_capacity_cases,
    run_capacity_preflight,
    validate_capacity_receipt,
    write_capacity_receipt,
)
from core.execution_environment import execution_environment_sha256
from core.hyperparameter_policy import canonical_json_sha256


SOURCE_SHA = "a" * 64
IMPLEMENTATION_SHA = "b" * 64
SOURCE_COUNT = 50


def _runtime() -> dict:
    descriptor = {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": "capacity-host",
            "machine": "AMD64",
            "system": "Windows",
            "platform": "capacity-fixture",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "capacity-gpu",
            "uuid": "GPU-capacity",
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 40,
            "total_memory_bytes": 8_589_934_592,
            "driver_version": "fixture",
        },
        "numeric_stack": {
            "torch_version": "fixture",
            "cuda_runtime_version": "fixture",
            "cudnn_version": 1,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    return {
        "schema": "ttbi-execution-runtime-binding-v1",
        "execution_block": "l60",
        "anchor_stage": "s0_scour",
        "execution_environment_sha256":
            execution_environment_sha256(descriptor),
        "execution_environment_descriptor": descriptor,
    }


def _probe(architecture, _config, _params, runtime):
    index = ARCHITECTURES.index(architecture)
    total = runtime["execution_environment_descriptor"]["accelerator"][
        "total_memory_bytes"
    ]
    reserved = 600_000_000 + index * 100_000_000
    return {
        "peak_memory_allocated_bytes": reserved - 50_000_000,
        "peak_memory_reserved_bytes": reserved,
        "total_memory_bytes": total,
    }


def _expect_error(label: str, fn) -> None:
    try:
        fn()
    except (CapacityPreflightError, RuntimeError, ValueError):
        return
    raise AssertionError(f"mutation survived: {label}")


def main() -> None:
    assert CAPACITY_PREFLIGHT_POLICY["batch_size"] == 32
    assert CAPACITY_PREFLIGHT_POLICY["input_shape"] == [32, 8, 512]
    assert CAPACITY_PREFLIGHT_POLICY["output_heads"] == 5
    cases = registered_capacity_cases()
    assert [case[0] for case in cases] == list(ARCHITECTURES)
    for architecture, config, params in cases:
        assert config["dofs"] == list(range(8))
        assert params["n_conv_layers"] == 4
        assert params["n_dense_layers"] == 3
        for index in range(4):
            assert params[f"n_filters_l{index}"] == 128
            assert params[f"pooling_l{index}"] is False
        if architecture == "PAA_LSTM_NHiTS":
            assert params["lstm_num_layers"] == 2
            assert params["lstm_hidden_size"] == 128

    runtime = _runtime()
    envelope = run_capacity_preflight(
        runtime,
        probe_runner=_probe,
        source_root_sha256=SOURCE_SHA,
        source_file_count=SOURCE_COUNT,
        implementation_source_sha256=IMPLEMENTATION_SHA,
    )
    validated = validate_capacity_receipt(
        envelope,
        expected_runtime=runtime,
        expected_source_root_sha256=SOURCE_SHA,
        expected_source_file_count=SOURCE_COUNT,
        expected_implementation_source_sha256=IMPLEMENTATION_SHA,
    )
    assert validated["receipt"]["passed"] is True
    assert len(validated["receipt"]["measurements"]) == 4
    assert (
        validated["receipt"]["worst_peak_reserved_bytes"]
        == max(
            item["peak_memory_reserved_bytes"]
            for item in validated["receipt"]["measurements"]
        )
    )

    stale_hash = deepcopy(envelope)
    stale_hash["receipt"]["measurements"][0][
        "peak_memory_reserved_bytes"
    ] += 1
    _expect_error(
        "receipt body mutation",
        lambda: validate_capacity_receipt(
            stale_hash,
            expected_runtime=runtime,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    missing_architecture = deepcopy(envelope)
    missing_architecture["receipt"]["measurements"].pop()
    missing_architecture["receipt_sha256"] = canonical_json_sha256(
        missing_architecture["receipt"]
    )
    _expect_error(
        "missing architecture probe",
        lambda: validate_capacity_receipt(
            missing_architecture,
            expected_runtime=runtime,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    detached_aggregate = deepcopy(envelope)
    detached_aggregate["receipt"]["worst_peak_allocated_bytes"] += 1
    detached_aggregate["receipt_sha256"] = canonical_json_sha256(
        detached_aggregate["receipt"]
    )
    _expect_error(
        "detached aggregate peak",
        lambda: validate_capacity_receipt(
            detached_aggregate,
            expected_runtime=runtime,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )
    detached_params = deepcopy(envelope)
    detached_params["receipt"]["measurements"][0]["params"][
        "n_conv_layers"
    ] = 3
    detached_params["receipt"]["measurements"][0]["params_sha256"] = (
        canonical_json_sha256(
            detached_params["receipt"]["measurements"][0]["params"]
        )
    )
    detached_params["receipt_sha256"] = canonical_json_sha256(
        detached_params["receipt"]
    )
    _expect_error(
        "receipt search-space point detached from live maximum",
        lambda: validate_capacity_receipt(
            detached_params,
            expected_runtime=runtime,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    wrong_source = deepcopy(envelope)
    _expect_error(
        "runtime source mutation",
        lambda: validate_capacity_receipt(
            wrong_source,
            expected_runtime=runtime,
            expected_source_root_sha256="c" * 64,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )
    _expect_error(
        "implementation source mutation",
        lambda: validate_capacity_receipt(
            envelope,
            expected_runtime=runtime,
            expected_source_root_sha256=SOURCE_SHA,
            expected_source_file_count=SOURCE_COUNT,
            expected_implementation_source_sha256="d" * 64,
        ),
    )

    total = runtime["execution_environment_descriptor"]["accelerator"][
        "total_memory_bytes"
    ]
    required = max(
        CAPACITY_PREFLIGHT_POLICY["minimum_remaining_headroom"][
            "absolute_bytes"
        ],
        int(
            total
            * CAPACITY_PREFLIGHT_POLICY["minimum_remaining_headroom"][
                "fraction_of_total_memory"
            ]
        ),
    )

    def insufficient(_architecture, _config, _params, _runtime):
        reserved = total - required + 1
        return {
            "peak_memory_allocated_bytes": reserved,
            "peak_memory_reserved_bytes": reserved,
            "total_memory_bytes": total,
        }

    _expect_error(
        "insufficient headroom",
        lambda: run_capacity_preflight(
            runtime,
            probe_runner=insufficient,
            source_root_sha256=SOURCE_SHA,
            source_file_count=SOURCE_COUNT,
            implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    def oom(*_args):
        raise MemoryError("synthetic OOM")

    _expect_error(
        "OOM not fatal",
        lambda: run_capacity_preflight(
            runtime,
            probe_runner=oom,
            source_root_sha256=SOURCE_SHA,
            source_file_count=SOURCE_COUNT,
            implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    from training.trainer import TRAIN_PROTOCOL
    original_batch = TRAIN_PROTOCOL["batch_size"]
    try:
        TRAIN_PROTOCOL["batch_size"] = 16
        _expect_error(
            "training/preflight batch drift",
            registered_capacity_cases,
        )
    finally:
        TRAIN_PROTOCOL["batch_size"] = original_batch

    cpu = deepcopy(runtime)
    accelerator = cpu["execution_environment_descriptor"]["accelerator"]
    for key in set(accelerator) - {"backend"}:
        accelerator[key] = None
    accelerator["backend"] = "cpu"
    cpu["execution_environment_sha256"] = execution_environment_sha256(
        cpu["execution_environment_descriptor"]
    )
    _expect_error(
        "CPU campaign",
        lambda: run_capacity_preflight(
            cpu,
            probe_runner=_probe,
            source_root_sha256=SOURCE_SHA,
            source_file_count=SOURCE_COUNT,
            implementation_source_sha256=IMPLEMENTATION_SHA,
        ),
    )

    # A live-identity receipt is canonical on disk and reusable from a genuinely
    # fresh interpreter.  `ensure_capacity_preflight` must load it; the child
    # would otherwise attempt the real CUDA probe for this synthetic runtime.
    live_envelope = run_capacity_preflight(
        runtime,
        probe_runner=_probe,
    )
    with tempfile.TemporaryDirectory() as temporary:
        receipt_dir = Path(temporary).resolve()
        path = capacity_receipt_path(
            runtime, receipt_dir=receipt_dir
        )
        assert write_capacity_receipt(
            path, live_envelope, expected_runtime=runtime
        ) == live_envelope["receipt_sha256"]
        assert load_capacity_receipt(
            path, expected_runtime=runtime
        ) == live_envelope
        assert ensure_capacity_preflight(
            runtime, receipt_dir=receipt_dir
        ) == live_envelope

        child_code = (
            "import json, os;"
            "from core.capacity_preflight import ensure_capacity_preflight;"
            f"runtime=json.loads({json.dumps(json.dumps(runtime))});"
            "value=ensure_capacity_preflight(runtime);"
            f"assert value['receipt_sha256']=="
            f"{live_envelope['receipt_sha256']!r}"
        )
        child_environment = dict(os.environ)
        child_environment["TTBI_EXECUTION_RECEIPT_DIR"] = str(receipt_dir)
        subprocess.run(
            [sys.executable, "-c", child_code],
            check=True,
            cwd=Path(__file__).resolve().parent,
            env=child_environment,
        )

        def changed_probe(architecture, config, params, run_runtime):
            changed = _probe(architecture, config, params, run_runtime)
            changed["peak_memory_allocated_bytes"] += 1
            changed["peak_memory_reserved_bytes"] += 1
            return changed

        differing = run_capacity_preflight(
            runtime,
            probe_runner=changed_probe,
        )
        _expect_error(
            "differing durable receipt overwrite",
            lambda: write_capacity_receipt(
                path, differing, expected_runtime=runtime
            ),
        )

        payload = path.read_bytes()
        path.write_bytes(payload + b"\n")
        _expect_error(
            "noncanonical durable receipt bytes",
            lambda: load_capacity_receipt(
                path, expected_runtime=runtime
            ),
        )

    _expect_error(
        "relative durable receipt directory",
        lambda: capacity_receipt_path(
            runtime, receipt_dir="relative/receipts"
        ),
    )
    print("PASS: CUDA capacity policy/receipt/mutation guards")


if __name__ == "__main__":
    main()
