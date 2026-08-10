"""Mutation and behaviour checks for four-block execution qualification.

Run with the locked campaign interpreter::

    py -3.13 check_execution_blocking.py

The four Paper-1 stages are independent HPO blocks. Within one block the GPU
model and numeric stack must match exactly, while hostname, device index, and
physical GPU UUID may differ across the two matched lab machines.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import stat
import tempfile

from core.execution_environment import (
    EXECUTION_BLOCK_POLICY,
    canonical_execution_block_policy,
    enforce_execution_block,
    execution_block_for_stage,
    execution_compatibility_descriptor,
    execution_compatibility_sha256,
    execution_environment_sha256,
    validate_block_reference_execution,
    validate_execution_runtime,
)


FAILURES = 0
CORE_SHA = "a" * 64
RUN_TAG = "paper1-execution-fixture"


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


def study_linked_provenance_guard_is_live() -> bool:
    """Require the current accumulated-mismatch guard in the study verifier."""
    pipeline_path = (
        Path(__file__).resolve().parent / "training" / "pipeline.py"
    )
    tree = ast.parse(
        pipeline_path.read_text(encoding="utf-8"),
        filename=str(pipeline_path),
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "verify_digital_twin_package"
    ]
    if len(functions) != 1:
        return False

    for branch in ast.walk(functions[0]):
        if not (
            isinstance(branch, ast.If)
            and isinstance(branch.test, ast.Name)
            and branch.test.id == "mismatches"
        ):
            continue
        for candidate in branch.body:
            for node in ast.walk(candidate):
                if not (
                    isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "RuntimeError"
                ):
                    continue
                diagnostic = "".join(
                    value.value
                    for value in ast.walk(node.exc)
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                )
                references_mismatches = any(
                    isinstance(value, ast.Name)
                    and value.id == "mismatches"
                    for value in ast.walk(node.exc)
                )
                if (
                    "champion package provenance mismatch" in diagnostic
                    and references_mismatches
                ):
                    return True
    return False


def fixture_environment(
    *,
    hostname: str = "lab-a",
    uuid: str = "GPU-lab-a",
    device_index: int = 0,
    gpu_name: str = "NVIDIA GeForce RTX 5060 Ti",
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
            "device_index": device_index,
            "name": gpu_name,
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


def fixture_runtime(stage: str, environment: dict) -> dict:
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


print("\n--- A. exact four-block policy ---")
policy = canonical_execution_block_policy(EXECUTION_BLOCK_POLICY)
expected_blocks = {
    "F40-S": ("f40s", "F40-S"),
    "F40-M": ("f40m", "F40-M"),
    "L99-S": ("l99s", "L99-S"),
    "L99-M": ("l99m", "L99-M"),
}
check(
    "policy is v2 and contains exactly four independent stage blocks",
    policy["schema"] == "ttbi-execution-block-policy-v2"
    and set(policy["blocks"]) == {"f40s", "f40m", "l99s", "l99m"}
    and all(
        execution_block_for_stage(stage, policy) == expected
        for stage, expected in expected_blocks.items()
    ),
)
check(
    "all registered cross-block claims are descriptive/non-confirmatory",
    set(policy["cross_block_inference"])
    == {"F40-S_to_F40-M", "F40-S_to_L99-S", "F40-S_to_L99-M"}
    and all(
        value["status"] == "descriptive_nonconfirmatory"
        and value["confirmatory"] is False
        for value in policy["cross_block_inference"].values()
    ),
)
rejects(
    "retired production stage is rejected",
    lambda: execution_block_for_stage("s0_scour", policy),
    "absent",
)
rejects(
    "track/OOR stage cannot enter production allocation",
    lambda: execution_block_for_stage("F40-TRACK", policy),
    "absent",
)
mutant = copy.deepcopy(policy)
mutant["blocks"]["f40s"]["stages"] = ["F40-S", "F40-M"]
rejects(
    "cross-block stage reassignment is rejected",
    lambda: canonical_execution_block_policy(mutant),
)
mutant = copy.deepcopy(policy)
mutant["cross_block_inference"]["F40-S_to_F40-M"]["confirmatory"] = True
rejects(
    "confirmatory cross-block relabelling is rejected",
    lambda: canonical_execution_block_policy(mutant),
)


print("\n--- B. exact and compatibility runtime identities ---")
lab_a_environment = fixture_environment()
lab_b_environment = fixture_environment(
    hostname="lab-b", uuid="GPU-lab-b", device_index=1
)
lab_a_runtime = fixture_runtime("F40-S", lab_a_environment)
lab_b_runtime = fixture_runtime("F40-S", lab_b_environment)
check(
    "runtime-v2 authenticates exact environment and compatibility identities",
    validate_execution_runtime(lab_a_runtime) == lab_a_runtime,
)
check(
    "host/device UUID changes preserve the registered compatibility class",
    lab_a_runtime["execution_environment_sha256"]
    != lab_b_runtime["execution_environment_sha256"]
    and lab_a_runtime["execution_compatibility_sha256"]
    == lab_b_runtime["execution_compatibility_sha256"]
    and validate_execution_runtime(lab_b_runtime) == lab_b_runtime,
)
mutant = copy.deepcopy(lab_a_runtime)
mutant["execution_environment_descriptor"]["host"]["hostname"] = "forged"
rejects(
    "exact environment mutation without rehash is rejected",
    lambda: validate_execution_runtime(mutant),
    "does not reproduce",
)
mutant = copy.deepcopy(lab_a_runtime)
del mutant["execution_compatibility_sha256"]
rejects(
    "missing compatibility identity is rejected",
    lambda: validate_execution_runtime(mutant),
    "malformed",
)
mutant = copy.deepcopy(lab_a_runtime)
mutant["execution_block"] = "l99s"
rejects(
    "runtime block/anchor disagreement is rejected",
    lambda: validate_execution_runtime(mutant),
    "wrong block anchor",
)
for field in ("uuid", "driver_version"):
    mutant_environment = fixture_environment()
    mutant_environment["accelerator"][field] = None
    rejects(
        f"CUDA environment missing {field} is rejected",
        lambda value=mutant_environment: execution_environment_sha256(value),
    )


print("\n--- C. atomic receipts and same-block matched hardware ---")
with tempfile.TemporaryDirectory(prefix="paper1-execution-") as temporary:
    receipt_root = Path(temporary)
    lab_a = enforce_execution_block(
        stage="F40-S",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_root,
        descriptor=lab_a_environment,
    )
    receipt_path = Path(lab_a["receipt_path"])
    receipt_value = json.loads(receipt_path.read_text(encoding="ascii"))
    check(
        "F40-S publishes one regular canonical v2 receipt",
        receipt_path.is_file()
        and not receipt_path.is_symlink()
        and stat.S_ISREG(os.lstat(receipt_path).st_mode)
        and receipt_value["schema"] == "ttbi-execution-block-receipt-v2"
        and receipt_value["execution_block"] == "f40s"
        and receipt_value["anchor_stage"] == "F40-S"
        and receipt_path.read_bytes()
        == (
            json.dumps(
                receipt_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        ),
    )
    lab_b = enforce_execution_block(
        stage="F40-S",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_root,
        descriptor=lab_b_environment,
    )
    check(
        "matched lab host/UUID may differ without moving the F40-S block",
        lab_b["receipt_sha256"] == lab_a["receipt_sha256"]
        and lab_b["runtime"]["execution_environment_sha256"]
        != lab_a["runtime"]["execution_environment_sha256"]
        and lab_b["runtime"]["execution_compatibility_sha256"]
        == lab_a["runtime"]["execution_compatibility_sha256"],
    )
    check(
        "reference validation accepts exact receipt plus matched compatibility",
        validate_block_reference_execution(
            selection_runtime=lab_a["runtime"],
            selection_environment_sha256=lab_a["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256=lab_a["receipt_sha256"],
            current_attestation=lab_b,
            current_stage="F40-S",
            policy=policy,
        ) == "same_block_compatible_hardware_exact_stack",
    )
    rejects(
        "same block rejects a different GPU model",
        lambda: enforce_execution_block(
            stage="F40-S",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=receipt_root,
            descriptor=fixture_environment(gpu_name="NVIDIA GeForce RTX 2060"),
        ),
        "receipt mismatch",
    )
    rejects(
        "reference receipt substitution is rejected",
        lambda: validate_block_reference_execution(
            selection_runtime=lab_a["runtime"],
            selection_environment_sha256=lab_a["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256="b" * 64,
            current_attestation=lab_b,
            current_stage="F40-S",
            policy=policy,
        ),
        "compatibility differs",
    )

    l99 = enforce_execution_block(
        stage="L99-S",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=receipt_root,
        descriptor=fixture_environment(
            hostname="l99-host",
            uuid="GPU-l99",
            gpu_name="NVIDIA GeForce RTX 2060",
        ),
    )
    check(
        "L99-S has an independent receipt and may use another hardware block",
        l99["runtime"]["execution_block"] == "l99s"
        and l99["receipt_sha256"] != lab_a["receipt_sha256"],
    )
    rejects(
        "cross-block reference substitution is rejected",
        lambda: validate_block_reference_execution(
            selection_runtime=lab_a["runtime"],
            selection_environment_sha256=lab_a["runtime"][
                "execution_environment_sha256"
            ],
            selection_receipt_sha256=lab_a["receipt_sha256"],
            current_attestation=l99,
            current_stage="L99-S",
            policy=policy,
        ),
        "not the registered l99s/L99-S",
    )

with tempfile.TemporaryDirectory(prefix="paper1-execution-malformed-") as temporary:
    created = enforce_execution_block(
        stage="F40-M",
        policy=policy,
        protocol_core_hash=CORE_SHA,
        run_tag=RUN_TAG,
        receipt_dir=temporary,
        descriptor=lab_a_environment,
    )
    Path(created["receipt_path"]).write_text("{malformed", encoding="ascii")
    rejects(
        "malformed pre-existing receipt is rejected",
        lambda: enforce_execution_block(
            stage="F40-M",
            policy=policy,
            protocol_core_hash=CORE_SHA,
            run_tag=RUN_TAG,
            receipt_dir=temporary,
            descriptor=lab_a_environment,
        ),
        "malformed execution receipt",
    )


print("\n--- D. study-linked package provenance boundary ---")
check(
    "study-linked verifier guards accumulated provenance mismatches",
    study_linked_provenance_guard_is_live(),
)


print()
if FAILURES:
    raise SystemExit(f"EXECUTION BLOCKING: {FAILURES} FAILURE(S)")
print("EXECUTION BLOCKING: ALL PASS")
