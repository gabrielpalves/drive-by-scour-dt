"""Adversarial checks for four-stage deployment-artifact provenance.

This focused suite verifies the standalone package boundary used after copying
model assets away from their Optuna database. It intentionally uses the current
runtime-v2/four-block contract and contains no retired ten-rung fixtures.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

from core.artifact_provenance import verify_standalone_dt_package
from core.execution_environment import (
    execution_block_for_stage,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
)
from core.protocol import protocol_hash


FAILURES = 0


def check(name: str, condition: bool) -> None:
    global FAILURES
    passed = bool(condition)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    FAILURES += int(not passed)


def rejects(name: str, operation, contains: str | None = None) -> None:
    try:
        operation()
    except RuntimeError as exc:
        check(name, contains is None or contains in str(exc))
    except Exception:
        check(name, False)
    else:
        check(name, False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_environment(
    *, hostname: str = "artifact-host", uuid: str = "GPU-artifact"
) -> dict:
    return {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": hostname,
            "machine": "AMD64",
            "system": "Windows",
            "platform": f"Windows-fixture-{hostname}",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "NVIDIA GeForce RTX 5060 Ti",
            "uuid": uuid,
            "compute_capability": {"major": 12, "minor": 0},
            "sm_count": 36,
            "total_memory_bytes": 17_179_869_184,
            "driver_version": "fixture-driver",
        },
        "numeric_stack": {
            "torch_version": "fixture-torch",
            "cuda_runtime_version": "12.8",
            "cudnn_version": 90701,
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


def fixture_runtime(stage: str, environment: dict | None = None) -> dict:
    environment = fixture_environment() if environment is None else environment
    block, anchor = execution_block_for_stage(stage)
    compatibility = execution_compatibility_descriptor(environment)
    return {
        "schema": "ttbi-execution-runtime-binding-v2",
        "execution_block": block,
        "anchor_stage": anchor,
        "execution_environment_sha256": execution_environment_sha256(
            environment
        ),
        "execution_environment_descriptor": environment,
        "execution_compatibility_sha256": execution_compatibility_sha256(
            environment
        ),
        "execution_compatibility_descriptor": compatibility,
    }


def fixture_metadata(
    *,
    stage: str,
    runtime: dict,
    model_path: Path,
    scaler_path: Path,
    hyperparameter_mode: str = "anchor_hpo",
    selection_artifact_sha256: str | None = None,
    selection_slot: str | None = None,
) -> dict:
    block, anchor = execution_block_for_stage(stage)
    descriptor = {
        "core": {
            "protocol_version": 12,
            "fixture": "paper1-artifact-provenance-v2",
        },
        "rung": {
            "stage": stage,
            "dataset": f"fixture-{stage}",
            "execution_block": block,
            "execution_anchor": anchor,
        },
    }
    return {
        "champion_weights_sha256": sha256_file(model_path),
        "scaler_sha256": sha256_file(scaler_path),
        "scaler_filename": scaler_path.name,
        "protocol_hash": protocol_hash(descriptor),
        "protocol_descriptor": descriptor,
        "execution_environment_sha256": runtime[
            "execution_environment_sha256"
        ],
        "execution_runtime": runtime,
        "campaign_run_tag": "paper1-artifact-fixture",
        "execution_receipt_sha256": "e" * 64,
        "block_reference_manifest_sha256": None,
        "hyperparameter_mode": hyperparameter_mode,
        "selection_artifact_sha256": selection_artifact_sha256,
        "selection_slot": selection_slot,
    }


with tempfile.TemporaryDirectory(prefix="paper1-artifact-") as temporary:
    root = Path(temporary)
    model_path = root / "DT_champion_weights.pth"
    scaler_path = root / "DT_scaler.pkl"
    metadata_path = root / "DT_metadata.json"
    model_path.write_bytes(b"fixture-model-weights\x00\x01")
    scaler_path.write_bytes(b"fixture-scaler\x02\x03")

    runtime = fixture_runtime("F40-S")
    original = fixture_metadata(
        stage="F40-S",
        runtime=runtime,
        model_path=model_path,
        scaler_path=scaler_path,
    )

    def write(value: dict) -> None:
        metadata_path.write_text(
            json.dumps(value, sort_keys=True), encoding="utf-8"
        )

    def verify(**kwargs) -> dict:
        return verify_standalone_dt_package(
            str(model_path), str(metadata_path), str(scaler_path), **kwargs
        )

    write(original)
    verified = verify()
    check(
        "F40-S anchor package verifies under runtime-v2",
        verified["protocol_hash"] == original["protocol_hash"]
        and verified["execution_runtime"] == runtime
        and verified["block_reference_manifest_sha256"] is None,
    )
    check(
        "independent null block-reference expectation is accepted",
        verify(expected_block_reference_manifest_sha256=None)[
            "block_reference_manifest_sha256"
        ] is None,
    )
    rejects(
        "independent non-null block-reference substitution is rejected",
        lambda: verify(expected_block_reference_manifest_sha256="a" * 64),
        "independently supplied expectation",
    )

    mutant = copy.deepcopy(original)
    del mutant["selection_slot"]
    write(mutant)
    rejects(
        "missing required selection lineage field is rejected",
        verify,
        "lacks provenance fields",
    )

    write(original)
    model_bytes = model_path.read_bytes()
    model_path.write_bytes(model_bytes[:-1] + bytes([model_bytes[-1] ^ 1]))
    rejects("one-byte champion tamper is rejected", verify, "weights")
    model_path.write_bytes(model_bytes)

    scaler_bytes = scaler_path.read_bytes()
    scaler_path.write_bytes(scaler_bytes[:-1] + bytes([scaler_bytes[-1] ^ 1]))
    rejects("one-byte scaler tamper is rejected", verify, "scaler")
    scaler_path.write_bytes(scaler_bytes)

    mutant = copy.deepcopy(original)
    mutant["scaler_filename"] = "foreign.pkl"
    write(mutant)
    rejects("scaler filename substitution is rejected", verify, "filename")

    mutant = copy.deepcopy(original)
    mutant["protocol_descriptor"]["core"]["protocol_version"] = 999
    write(mutant)
    rejects(
        "protocol descriptor/hash disagreement is rejected",
        verify,
        "does not reproduce",
    )

    mutant = copy.deepcopy(original)
    mutant["execution_runtime"]["execution_environment_descriptor"]["host"][
        "hostname"
    ] = "forged-host"
    write(mutant)
    rejects(
        "exact execution descriptor mutation is rejected",
        verify,
        "does not reproduce",
    )

    mutant = copy.deepcopy(original)
    mutant["execution_runtime"]["execution_compatibility_sha256"] = "a" * 64
    write(mutant)
    rejects(
        "execution compatibility mutation is rejected",
        verify,
        "compatibility descriptor/hash",
    )

    mutant = copy.deepcopy(original)
    mutant["execution_environment_sha256"] = "a" * 64
    write(mutant)
    rejects(
        "metadata/runtime exact-environment disagreement is rejected",
        verify,
        "metadata execution SHA",
    )

    mutant = copy.deepcopy(original)
    mutant["protocol_descriptor"]["rung"]["execution_block"] = "l99s"
    mutant["protocol_hash"] = protocol_hash(mutant["protocol_descriptor"])
    write(mutant)
    rejects(
        "protocol/runtime execution-block disagreement is rejected",
        verify,
        "execution block",
    )

    mutant = copy.deepcopy(original)
    mutant["protocol_descriptor"]["rung"]["stage"] = "s0_scour"
    mutant["protocol_hash"] = protocol_hash(mutant["protocol_descriptor"])
    mutant["block_reference_manifest_sha256"] = "b" * 64
    write(mutant)
    rejects(
        "retired ten-rung stage cannot be disguised as a follower package",
        verify,
        "unregistered production stage",
    )

    mutant = copy.deepcopy(original)
    mutant["execution_receipt_sha256"] = "E" * 64
    write(mutant)
    rejects("malformed execution receipt digest is rejected", verify, "receipt")

    mutant = copy.deepcopy(original)
    mutant["block_reference_manifest_sha256"] = "b" * 64
    write(mutant)
    rejects(
        "four-stage anchor cannot carry a retired follower reference",
        verify,
        "block-anchor package",
    )

    mutant = copy.deepcopy(original)
    mutant["selection_artifact_sha256"] = "c" * 64
    mutant["selection_slot"] = "f40s_best_raw"
    write(mutant)
    rejects(
        "factorial package cannot carry selected-pair lineage",
        verify,
        "non-selected-pair",
    )

    selected_runtime = fixture_runtime(
        "L99-M",
        fixture_environment(hostname="l99m-host", uuid="GPU-l99m"),
    )
    selected = fixture_metadata(
        stage="L99-M",
        runtime=selected_runtime,
        model_path=model_path,
        scaler_path=scaler_path,
        hyperparameter_mode="selected_pair_hpo",
        selection_artifact_sha256="d" * 64,
        selection_slot="paa_cnn_gap_baseline",
    )
    write(selected)
    check(
        "L99-M selected-pair package carries exact selection artefact and slot",
        verify()["selection_slot"] == "paa_cnn_gap_baseline",
    )
    mutant = copy.deepcopy(selected)
    mutant["selection_slot"] = "PAA_CNN_GAP_BASELINE"
    write(mutant)
    rejects("foreign selected-pair slot is rejected", verify, "foreign slot")
    mutant = copy.deepcopy(selected)
    mutant["selection_artifact_sha256"] = None
    write(mutant)
    rejects(
        "selected-pair package missing artefact digest is rejected",
        verify,
        "lacks its selection artefact",
    )


print()
if FAILURES:
    raise SystemExit(f"ARTIFACT PROVENANCE: {FAILURES} FAILURE(S)")
print("ARTIFACT PROVENANCE: ALL PASS")
