"""Fail-closed physical execution blocking for the ablation campaign.

The experimental *allocation policy* in :data:`EXECUTION_BLOCK_POLICY` is
hash-carried by ``core.protocol``.  The machine/GPU observed at run time is
deliberately not part of the shared protocol hash: it is recorded separately
and enforced through one atomic receipt per pre-registered contrast block.

This distinction lets studies in one scientific block share a protocol while
still preventing a study or downstream rung from moving silently to another
host/GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import tempfile
from typing import Any


EXECUTION_BLOCK_POLICY = {
    "schema": "ttbi-execution-block-policy-v1",
    "same_physical_host_and_gpu_within_block": True,
    "blocks": {
        "l60": {
            "anchor_stage": "s0_scour",
            "stages": [
                "s0_scour",
                "s11_bear",
                "s12_crack",
                "s13_bearcrack",
                "s14_prof",
                "s15_track",
                "s16_all",
            ],
        },
        "l99": {
            "anchor_stage": "s21_scour4",
            "stages": [
                "s21_scour4",
                "s22_bearcrack4",
                "s23_all4",
            ],
        },
    },
    "cross_block_inference": {
        "s0_scour_to_s21_scour4": {
            "status": "descriptive_nonconfirmatory",
            "confirmatory": False,
            "rationale": (
                "bridge-length and support-count changes are bundled with a "
                "new execution block, so this cross-block contrast is reported "
                "descriptively and is not a confirmatory hardware-controlled "
                "effect"
            ),
        },
    },
    "receipt_contract": {
        "schema": "ttbi-execution-block-receipt-v1",
        "anchor_creates": True,
        "non_anchor_requires_existing_exact_receipt": True,
        "identity_match": "exact_canonical_execution_environment_sha256",
    },
}

_DESCRIPTOR_SCHEMA = "ttbi-execution-environment-v1"
_BINDING_SCHEMA = "ttbi-execution-runtime-binding-v1"
_RECEIPT_SCHEMA = "ttbi-execution-block-receipt-v1"
_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON byte representation used in receipts."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"execution provenance is not canonical JSON: {exc}"
        ) from exc
    return text.encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def canonical_execution_block_policy(policy: dict) -> dict:
    """Validate and return a JSON-canonical copy of the allocation policy."""

    if not isinstance(policy, dict):
        raise RuntimeError("execution-block policy must be a mapping")
    canonical = json.loads(_canonical_json_bytes(policy).decode("ascii"))
    required = {
        "schema",
        "same_physical_host_and_gpu_within_block",
        "blocks",
        "cross_block_inference",
        "receipt_contract",
    }
    if set(canonical) != required:
        raise RuntimeError(
            "execution-block policy fields differ from the registered contract"
        )
    if canonical["schema"] != "ttbi-execution-block-policy-v1":
        raise RuntimeError("unsupported execution-block policy schema")
    if canonical["same_physical_host_and_gpu_within_block"] is not True:
        raise RuntimeError(
            "campaign policy must require one physical host/GPU per block"
        )

    expected_blocks = {
        "l60": (
            "s0_scour",
            (
                "s0_scour",
                "s11_bear",
                "s12_crack",
                "s13_bearcrack",
                "s14_prof",
                "s15_track",
                "s16_all",
            ),
        ),
        "l99": (
            "s21_scour4",
            ("s21_scour4", "s22_bearcrack4", "s23_all4"),
        ),
    }
    if set(canonical["blocks"]) != set(expected_blocks):
        raise RuntimeError("execution policy must define exactly l60 and l99")
    seen: set[str] = set()
    for block, (expected_anchor, expected_stages) in expected_blocks.items():
        value = canonical["blocks"].get(block)
        if not isinstance(value, dict) or set(value) != {
            "anchor_stage", "stages"
        }:
            raise RuntimeError(f"malformed execution block {block!r}")
        if value["anchor_stage"] != expected_anchor:
            raise RuntimeError(f"wrong anchor for execution block {block!r}")
        if tuple(value["stages"]) != expected_stages:
            raise RuntimeError(f"wrong stage allocation for block {block!r}")
        overlap = seen.intersection(value["stages"])
        if overlap:
            raise RuntimeError(
                f"stages assigned to more than one execution block: {overlap}"
            )
        seen.update(value["stages"])

    cross = canonical["cross_block_inference"]
    if set(cross) != {"s0_scour_to_s21_scour4"}:
        raise RuntimeError("cross-block inference declaration is incomplete")
    contrast = cross["s0_scour_to_s21_scour4"]
    if (
        not isinstance(contrast, dict)
        or set(contrast) != {"status", "confirmatory", "rationale"}
        or contrast["status"] != "descriptive_nonconfirmatory"
        or contrast["confirmatory"] is not False
        or not isinstance(contrast["rationale"], str)
        or not contrast["rationale"].strip()
    ):
        raise RuntimeError(
            "s0->s21 must be explicitly descriptive and non-confirmatory"
        )

    receipt = canonical["receipt_contract"]
    if receipt != {
        "schema": _RECEIPT_SCHEMA,
        "anchor_creates": True,
        "non_anchor_requires_existing_exact_receipt": True,
        "identity_match": "exact_canonical_execution_environment_sha256",
    }:
        raise RuntimeError("execution receipt contract differs from v1")
    return canonical


def execution_block_for_stage(
    stage: str,
    policy: dict = EXECUTION_BLOCK_POLICY,
) -> tuple[str, str]:
    """Return ``(block, anchor_stage)`` or reject an unregistered stage."""

    canonical = canonical_execution_block_policy(policy)
    for block, value in canonical["blocks"].items():
        if stage in value["stages"]:
            return block, value["anchor_stage"]
    raise RuntimeError(
        f"stage {stage!r} is absent from the execution-block policy"
    )


def _nvidia_driver_for_uuid(gpu_uuid: str | None) -> str | None:
    """Query the display driver without a shell; return ``None`` if unavailable."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) == 2 and all(parts):
            rows.append((parts[0], parts[1]))
    if gpu_uuid:
        wanted = gpu_uuid.lower().removeprefix("gpu-")
        for candidate, driver in rows:
            actual = candidate.lower().removeprefix("gpu-")
            if actual == wanted:
                return driver
    drivers = {driver for _uuid, driver in rows}
    # NVIDIA's display driver is host-wide; multiple GPUs legitimately report
    # the same value even when an older PyTorch cannot expose a device UUID.
    return next(iter(drivers)) if len(drivers) == 1 else None


def current_execution_environment() -> dict:
    """Capture the canonical physical/numerical execution identity."""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    accelerator: dict[str, Any]
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        raw_uuid = getattr(props, "uuid", None)
        gpu_uuid = str(raw_uuid) if raw_uuid is not None else None
        accelerator = {
            "backend": "cuda",
            "device_index": device_index,
            "name": str(props.name),
            "uuid": gpu_uuid,
            "compute_capability": {
                "major": int(props.major),
                "minor": int(props.minor),
            },
            "sm_count": int(props.multi_processor_count),
            "total_memory_bytes": int(props.total_memory),
            "driver_version": _nvidia_driver_for_uuid(gpu_uuid),
        }
    else:
        accelerator = {
            "backend": "cpu",
            "device_index": None,
            "name": None,
            "uuid": None,
            "compute_capability": None,
            "sm_count": None,
            "total_memory_bytes": None,
            "driver_version": None,
        }

    cudnn = torch.backends.cudnn
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    descriptor = {
        "schema": _DESCRIPTOR_SCHEMA,
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "system": platform.system(),
            "platform": platform.platform(),
        },
        "accelerator": accelerator,
        "numeric_stack": {
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": (
                str(torch.version.cuda) if torch.version.cuda is not None
                else None
            ),
            "cudnn_version": (
                int(cudnn.version()) if cudnn.version() is not None else None
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_enabled": bool(cudnn.enabled),
            "cudnn_deterministic": bool(cudnn.deterministic),
            "cudnn_benchmark": bool(cudnn.benchmark),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": (
                bool(cuda_matmul.allow_tf32)
                if cuda_matmul is not None else None
            ),
            "float32_matmul_precision": str(
                torch.get_float32_matmul_precision()
            ),
        },
    }
    return validate_execution_environment(descriptor)


def validate_execution_environment(descriptor: dict) -> dict:
    """Validate the exact v1 descriptor and return its canonical copy."""

    if not isinstance(descriptor, dict):
        raise RuntimeError("execution-environment descriptor must be a mapping")
    value = json.loads(_canonical_json_bytes(descriptor).decode("ascii"))
    if set(value) != {"schema", "host", "accelerator", "numeric_stack"}:
        raise RuntimeError("malformed execution-environment descriptor")
    if value["schema"] != _DESCRIPTOR_SCHEMA:
        raise RuntimeError("unsupported execution-environment schema")

    host = value["host"]
    if not isinstance(host, dict) or set(host) != {
        "hostname", "machine", "system", "platform"
    }:
        raise RuntimeError("malformed execution host descriptor")
    if any(
        not isinstance(host[key], str) or not host[key].strip()
        for key in host
    ):
        raise RuntimeError("execution host fields must be nonempty strings")

    accelerator = value["accelerator"]
    accelerator_fields = {
        "backend",
        "device_index",
        "name",
        "uuid",
        "compute_capability",
        "sm_count",
        "total_memory_bytes",
        "driver_version",
    }
    if not isinstance(accelerator, dict) or set(accelerator) != accelerator_fields:
        raise RuntimeError("malformed accelerator descriptor")
    if accelerator["backend"] not in {"cuda", "cpu"}:
        raise RuntimeError("unsupported accelerator backend")
    if accelerator["backend"] == "cuda":
        if (
            not isinstance(accelerator["uuid"], str)
            or not accelerator["uuid"].strip()
        ):
            raise RuntimeError(
                "CUDA device UUID is unavailable; exact physical-GPU "
                "execution blocking cannot be certified"
            )
        if (
            isinstance(accelerator["device_index"], bool)
            or not isinstance(accelerator["device_index"], int)
            or accelerator["device_index"] < 0
            or not isinstance(accelerator["name"], str)
            or not accelerator["name"].strip()
            or not isinstance(accelerator["driver_version"], str)
            or not accelerator["driver_version"].strip()
            or isinstance(accelerator["sm_count"], bool)
            or not isinstance(accelerator["sm_count"], int)
            or accelerator["sm_count"] <= 0
            or isinstance(accelerator["total_memory_bytes"], bool)
            or not isinstance(accelerator["total_memory_bytes"], int)
            or accelerator["total_memory_bytes"] <= 0
        ):
            raise RuntimeError("invalid CUDA accelerator identity")
        capability = accelerator["compute_capability"]
        if (
            not isinstance(capability, dict)
            or set(capability) != {"major", "minor"}
            or any(
                isinstance(capability[key], bool)
                or not isinstance(capability[key], int)
                or capability[key] < 0
                for key in capability
            )
        ):
            raise RuntimeError("invalid CUDA compute capability")
    else:
        if any(
            accelerator[key] is not None
            for key in accelerator_fields - {"backend"}
        ):
            raise RuntimeError("CPU accelerator descriptor must use null fields")

    numeric = value["numeric_stack"]
    expected_numeric = {
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "float32_matmul_precision",
    }
    if not isinstance(numeric, dict) or set(numeric) != expected_numeric:
        raise RuntimeError("malformed numeric-stack descriptor")
    if not isinstance(numeric["torch_version"], str) or not numeric[
        "torch_version"
    ]:
        raise RuntimeError("numeric stack lacks torch version")
    for key in (
        "cuda_runtime_version",
        "cublas_workspace_config",
    ):
        if numeric[key] is not None and not isinstance(numeric[key], str):
            raise RuntimeError(f"numeric-stack {key} must be text or null")
    if (
        numeric["cudnn_version"] is not None
        and (
            isinstance(numeric["cudnn_version"], bool)
            or not isinstance(numeric["cudnn_version"], int)
            or numeric["cudnn_version"] < 0
        )
    ):
        raise RuntimeError("invalid cuDNN version")
    for key in (
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
    ):
        if not isinstance(numeric[key], bool):
            raise RuntimeError(f"numeric-stack {key} must be boolean")
    if (
        numeric["cuda_matmul_allow_tf32"] is not None
        and not isinstance(numeric["cuda_matmul_allow_tf32"], bool)
    ):
        raise RuntimeError(
            "numeric-stack cuda_matmul_allow_tf32 must be boolean or null"
        )
    if not isinstance(numeric["float32_matmul_precision"], str) or not numeric[
        "float32_matmul_precision"
    ]:
        raise RuntimeError("numeric stack lacks float32 matmul precision")
    return value


def execution_environment_sha256(descriptor: dict) -> str:
    """SHA-256 of the validated canonical execution descriptor."""

    return _sha256_json(validate_execution_environment(descriptor))


def validate_execution_runtime(runtime: dict) -> dict:
    """Validate a study/config/metadata execution binding."""

    if not isinstance(runtime, dict):
        raise RuntimeError("execution runtime binding must be a mapping")
    value = json.loads(_canonical_json_bytes(runtime).decode("ascii"))
    expected = {
        "schema",
        "execution_block",
        "anchor_stage",
        "execution_environment_sha256",
        "execution_environment_descriptor",
    }
    if set(value) != expected or value["schema"] != _BINDING_SCHEMA:
        raise RuntimeError("malformed execution runtime binding")
    if value["execution_block"] not in {"l60", "l99"}:
        raise RuntimeError("invalid execution block")
    expected_anchor = (
        "s0_scour" if value["execution_block"] == "l60" else "s21_scour4"
    )
    if value["anchor_stage"] != expected_anchor:
        raise RuntimeError("execution binding carries the wrong block anchor")
    if not _is_sha256(value["execution_environment_sha256"]):
        raise RuntimeError("invalid execution-environment SHA-256")
    actual = execution_environment_sha256(
        value["execution_environment_descriptor"]
    )
    if actual != value["execution_environment_sha256"]:
        raise RuntimeError(
            "execution descriptor does not reproduce its recorded SHA-256"
        )
    return value


def validate_block_reference_execution(
    *,
    selection_runtime: dict,
    selection_environment_sha256: str,
    selection_receipt_sha256: str,
    current_attestation: dict,
    current_stage: str,
    policy: dict = EXECUTION_BLOCK_POLICY,
) -> str:
    """Require a reference selection and follower to share one exact block.

    L60 is anchored by ``s0_scour`` and L99.6 by ``s21_scour4``.  A reference
    manifest is never portable across these blocks: cross-geometry reporting
    may still be descriptive, but it cannot select or authenticate a follower.
    """

    block, anchor = execution_block_for_stage(current_stage, policy)
    selected = validate_execution_runtime(selection_runtime)
    if (
        selected["execution_block"] != block
        or selected["anchor_stage"] != anchor
    ):
        raise RuntimeError(
            f"{current_stage}: reference selection is from "
            f"{selected['execution_block']}/{selected['anchor_stage']}, not "
            f"the registered {block}/{anchor} block anchor"
        )
    if (
        not _is_sha256(selection_environment_sha256)
        or selection_environment_sha256
        != selected["execution_environment_sha256"]
        or not _is_sha256(selection_receipt_sha256)
    ):
        raise RuntimeError(
            f"{current_stage}: reference selection carries invalid execution "
            "environment or receipt identity"
        )
    if not isinstance(current_attestation, dict):
        raise RuntimeError("current execution attestation is missing")
    current = validate_execution_runtime(current_attestation.get("runtime"))
    current_receipt_sha = current_attestation.get("receipt_sha256")
    if (
        current["execution_block"] != block
        or current["anchor_stage"] != anchor
        or not _is_sha256(current_receipt_sha)
    ):
        raise RuntimeError(
            f"{current_stage}: current execution attestation disagrees with "
            "the registered block"
        )
    if (
        selected != current
        or selection_environment_sha256
        != current["execution_environment_sha256"]
        or selection_receipt_sha256 != current_receipt_sha
    ):
        raise RuntimeError(
            f"{current_stage}: reference selection execution identity differs "
            f"from the current {block} block receipt"
        )
    return "same_block_exact"


def _receipt_path(
    receipt_dir: str | os.PathLike[str],
    *,
    block: str,
    protocol_core_hash: str,
    run_tag: str,
) -> Path:
    if not _is_sha256(protocol_core_hash):
        raise RuntimeError("protocol_core_hash must be lowercase SHA-256")
    run_digest = hashlib.sha256(run_tag.encode("utf-8")).hexdigest()[:12]
    return Path(receipt_dir) / (
        f"execution_{block}_pcore-{protocol_core_hash[:16]}"
        f"_run-{run_digest}.json"
    )


def _read_regular_file(path: Path, *, max_bytes: int = 1 << 20) -> bytes:
    """Read without following symlinks and reject every non-regular target."""

    try:
        before = os.lstat(path)
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"execution receipt is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open execution receipt safely: {path}") from exc
    try:
        after = os.fstat(fd)
        if (
            not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise RuntimeError(
                f"execution receipt changed during safe open: {path}"
            )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(65536, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError("execution receipt is unexpectedly large")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _parse_receipt(path: Path) -> dict:
    raw = _read_regular_file(path)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed execution receipt: {path}") from exc
    canonical = _canonical_json_bytes(value) + b"\n"
    if raw != canonical:
        raise RuntimeError(
            f"execution receipt is not exact canonical JSON: {path}"
        )
    if not isinstance(value, dict):
        raise RuntimeError("execution receipt must contain a JSON object")
    return value


def _atomic_create(path: Path, payload: bytes) -> bool:
    """Publish complete bytes atomically without replacing an existing receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(
            f"execution receipt parent is not a regular directory: {path.parent}"
        )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard-link publication is atomic and cannot replace a winner
            # created concurrently by another anchor process.
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError as exc:
            raise RuntimeError(
                f"cannot atomically publish execution receipt {path}"
            ) from exc
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def enforce_execution_block(
    *,
    stage: str,
    policy: dict,
    protocol_core_hash: str,
    run_tag: str,
    receipt_dir: str | os.PathLike[str],
    descriptor: dict | None = None,
) -> dict:
    """Create/validate this block's receipt and return its runtime attestation.

    Only the block anchor may create a missing receipt.  Every other stage
    requires the already-published canonical receipt to match exactly.
    """

    if not isinstance(run_tag, str):
        raise RuntimeError("run_tag must be text")
    block, anchor = execution_block_for_stage(stage, policy)
    environment = validate_execution_environment(
        current_execution_environment() if descriptor is None else descriptor
    )
    environment_sha = execution_environment_sha256(environment)
    runtime = validate_execution_runtime({
        "schema": _BINDING_SCHEMA,
        "execution_block": block,
        "anchor_stage": anchor,
        "execution_environment_sha256": environment_sha,
        "execution_environment_descriptor": environment,
    })
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "execution_block": block,
        "anchor_stage": anchor,
        "protocol_core_hash": protocol_core_hash,
        "run_tag": run_tag,
        "execution_runtime": runtime,
    }
    receipt_payload = _canonical_json_bytes(receipt) + b"\n"
    path = _receipt_path(
        receipt_dir,
        block=block,
        protocol_core_hash=protocol_core_hash,
        run_tag=run_tag,
    )

    if not path.exists():
        # lexists catches dangling symlinks, for which Path.exists() is False.
        if os.path.lexists(path):
            raise RuntimeError(
                f"execution receipt path is a dangling symlink: {path}"
            )
        if stage != anchor:
            raise RuntimeError(
                f"{stage}: execution-block receipt is missing for {block!r}. "
                f"Run its anchor {anchor!r} first on the physical host/GPU "
                "reserved for this block."
            )
        created = _atomic_create(path, receipt_payload)
        if not created:
            # A concurrent anchor won. It must have published exactly our bytes.
            existing = _parse_receipt(path)
            if existing != receipt:
                raise RuntimeError(
                    f"{stage}: concurrently-created execution receipt differs"
                )
    else:
        existing = _parse_receipt(path)
        if existing != receipt:
            raise RuntimeError(
                f"{stage}: execution receipt mismatch for block {block!r}; "
                "refusing to move a contrast block to another host/GPU or "
                "numeric execution state"
            )

    # Re-read the final public path even after creation. This both proves the
    # atomic bytes and closes the same symlink/nonregular/malformed checks for
    # anchors and followers.
    final = _parse_receipt(path)
    if final != receipt:
        raise RuntimeError("published execution receipt differs from expectation")
    return {
        "runtime": runtime,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }
