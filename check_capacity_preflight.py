"""Mutation and behaviour checks for the CUDA capacity qualification."""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
            "imports"
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

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import capacity_preflight_compute as publication
import core.execution_environment as execution_environment_module

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
from core.execution_environment import (
    EXECUTION_BLOCK_POLICY,
    current_execution_runtime_for_stage,
    enforce_execution_block,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
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
    compatibility = execution_compatibility_descriptor(descriptor)
    return {
        "schema": "ttbi-execution-runtime-binding-v2",
        "execution_block": "f40s",
        "anchor_stage": "F40-S",
        "execution_environment_sha256":
            execution_environment_sha256(descriptor),
        "execution_environment_descriptor": descriptor,
        "execution_compatibility_sha256":
            execution_compatibility_sha256(descriptor),
        "execution_compatibility_descriptor": compatibility,
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
    publication_source = Path(publication.__file__).read_text(encoding="utf-8")
    assert "ensure_capacity_preflight" not in publication_source
    assert "run_capacity_preflight(" in publication_source
    assert "require_absent=True" in publication_source
    assert "set_global_seed(CAPACITY_SETUP_SEED" in publication_source
    assert '"--receipt-dir"' in publication_source
    assert list(inspect.signature(
        publication.create_f40s_capacity_receipt
    ).parameters) == ["receipt_dir"]
    manifest_names = {
        line
        for line in (Path(__file__).resolve().parent / "bundle_source_files.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }
    assert "capacity_preflight_compute.py" in manifest_names
    assert CAPACITY_PREFLIGHT_POLICY["batch_size"] == 32
    assert CAPACITY_PREFLIGHT_POLICY["representation_input_shapes"] == {
        "RAW": [32, 8, 11791],
        "PAA": [32, 8, 512],
    }
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
        if architecture == "PAA_POS0_LSTM1_MR1":
            assert params["lstm_num_layers"] == 2
            assert params["lstm_hidden_size"] == 128
        expected_length = 11791 if architecture.startswith("RAW_") else 512
        assert config["n_segments"] == expected_length

    runtime = _runtime()
    descriptor = runtime["execution_environment_descriptor"]
    with mock.patch.object(
        execution_environment_module,
        "current_execution_environment",
        return_value=descriptor,
    ):
        assert current_execution_runtime_for_stage("F40-S") == runtime
    assert current_execution_runtime_for_stage(
        "F40-S",
        policy=EXECUTION_BLOCK_POLICY,
        descriptor=descriptor,
    ) == runtime
    _expect_error(
        "runtime helper invalid stage",
        lambda: current_execution_runtime_for_stage(
            "UNREGISTERED", descriptor=descriptor
        ),
    )
    malformed_descriptor = deepcopy(descriptor)
    malformed_descriptor["schema"] = "foreign-environment"
    _expect_error(
        "runtime helper malformed descriptor",
        lambda: current_execution_runtime_for_stage(
            "F40-S", descriptor=malformed_descriptor
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        attestation = enforce_execution_block(
            stage="F40-S",
            policy=EXECUTION_BLOCK_POLICY,
            protocol_core_hash="f" * 64,
            run_tag="capacity-runtime-helper-equivalence",
            receipt_dir=temporary,
            descriptor=descriptor,
        )
        assert attestation["runtime"] == current_execution_runtime_for_stage(
            "F40-S",
            policy=EXECUTION_BLOCK_POLICY,
            descriptor=descriptor,
        )
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
    assert len(validated["receipt"]["measurements"]) == 16
    assert {
        (item["representation"], tuple(item["input_shape"]))
        for item in validated["receipt"]["measurements"]
    } == {
        ("RAW", (32, 8, 11791)),
        ("PAA", (32, 8, 512)),
    }
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

    detached_shape = deepcopy(envelope)
    detached_shape["receipt"]["measurements"][0]["input_shape"][2] -= 1
    detached_shape["receipt_sha256"] = canonical_json_sha256(
        detached_shape["receipt"]
    )
    _expect_error(
        "RAW capacity shape detached from longest production case",
        lambda: validate_capacity_receipt(
            detached_shape,
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
        _expect_error(
            "fresh publication cannot reuse identical receipt bytes",
            lambda: write_capacity_receipt(
                path,
                live_envelope,
                expected_runtime=runtime,
                require_absent=True,
            ),
        )
        _expect_error(
            "fresh-publication switch must be boolean",
            lambda: write_capacity_receipt(
                path,
                live_envelope,
                expected_runtime=runtime,
                require_absent=1,
            ),
        )
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

    with tempfile.TemporaryDirectory() as temporary:
        boundary_root = Path(temporary).resolve()
        fake_repo = boundary_root / "repo"
        internal = fake_repo / "receipts"
        external = boundary_root / "external"
        internal.mkdir(parents=True)
        external.mkdir()
        assert publication._external_receipt_directory(
            external, fake_repo
        ) == external
        _expect_error(
            "capacity receipt directory inside repository",
            lambda: publication._external_receipt_directory(
                internal, fake_repo
            ),
        )
        _expect_error(
            "relative capacity publication directory",
            lambda: publication._external_receipt_directory(
                "relative-capacity-receipts", fake_repo
            ),
        )

    with tempfile.TemporaryDirectory() as temporary:
        git_repo = Path(temporary).resolve() / "source"
        git_repo.mkdir()
        subprocess.run(
            ["git", "init", "--quiet", str(git_repo)], check=True
        )
        subprocess.run(
            [
                "git", "-C", str(git_repo), "config",
                "user.email", "capacity-check@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git", "-C", str(git_repo), "config",
                "user.name", "Capacity Check",
            ],
            check=True,
        )
        (git_repo / "reviewed.txt").write_text(
            "reviewed\n", encoding="utf-8"
        )
        subprocess.run(
            ["git", "-C", str(git_repo), "add", "reviewed.txt"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "A"],
            check=True,
        )
        assert len(publication._require_clean_commit(git_repo)) == 40
        (git_repo / "untracked.txt").write_text(
            "dirty\n", encoding="utf-8"
        )
        _expect_error(
            "capacity qualification from dirty commit",
            lambda: publication._require_clean_commit(git_repo),
        )

    with tempfile.TemporaryDirectory() as temporary:
        receipt_dir = Path(temporary).resolve()
        target = receipt_dir / (
            "capacity_preflight_f40s_" + "c" * 64 + ".json"
        )
        source_snapshot = SimpleNamespace(
            python_runtime=SimpleNamespace(
                sha256=SOURCE_SHA,
                file_count=SOURCE_COUNT,
            ),
            assert_unchanged=mock.Mock(),
        )
        events = []

        def record_seed(seed, policy):
            assert seed == 104729
            assert policy is publication.TRAIN_PROTOCOL["determinism"]
            events.append("seed")

        def record_runtime(stage):
            assert stage == "F40-S"
            assert events and events[0] == "seed"
            events.append("runtime")
            return runtime

        with (
            mock.patch.object(
                publication,
                "_require_clean_commit",
                return_value="d" * 40,
            ) as clean_mock,
            mock.patch.object(
                publication,
                "_external_receipt_directory",
                return_value=receipt_dir,
            ),
            mock.patch.object(
                publication,
                "repository_source_snapshot",
                return_value=source_snapshot,
            ),
            mock.patch.object(
                publication,
                "load_environment_lock",
                return_value={"sha256": "e" * 64},
            ),
            mock.patch.object(
                publication, "validate_environment_lock"
            ) as lock_mock,
            mock.patch.object(
                publication, "set_global_seed", side_effect=record_seed
            ),
            mock.patch.object(
                publication,
                "current_execution_runtime_for_stage",
                side_effect=record_runtime,
            ),
            mock.patch.object(
                publication, "capacity_receipt_path", return_value=target
            ),
            mock.patch.object(
                publication, "run_capacity_preflight", return_value=envelope
            ) as run_mock,
            mock.patch.object(
                publication, "validate_capacity_receipt", return_value=envelope
            ),
            mock.patch.object(
                publication,
                "write_capacity_receipt",
                return_value=envelope["receipt_sha256"],
            ) as write_mock,
            mock.patch.object(
                publication, "canonical_existing_path", return_value=target
            ),
            mock.patch.object(
                publication, "load_capacity_receipt", return_value=envelope
            ),
        ):
            result = publication.create_f40s_capacity_receipt(receipt_dir)
        assert result["status"] == "PASS"
        assert result["architecture_probe_count"] == 16
        assert events[0] == "seed" and events.count("runtime") == 3
        assert clean_mock.call_count == 3
        assert lock_mock.call_count == 1
        source_snapshot.assert_unchanged.assert_called()
        run_mock.assert_called_once_with(
            runtime,
            source_root_sha256=SOURCE_SHA,
            source_file_count=SOURCE_COUNT,
        )
        assert write_mock.call_args.kwargs["require_absent"] is True

        target.write_text("preserve\n", encoding="utf-8")
        with mock.patch.object(
            publication, "capacity_receipt_path", return_value=target
        ):
            _expect_error(
                "capacity publication target reuse",
                lambda: publication._fresh_target(runtime, receipt_dir),
            )
    print("PASS: CUDA capacity policy/receipt/mutation guards")


if __name__ == "__main__":
    main()
