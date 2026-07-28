"""Pinned software-environment provenance for the ablation campaign."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path


_LOCK_SCHEMA = "ttbi-campaign-environment-v2"
_MATLAB_ENVIRONMENT_FIELDS = (
    "arch",
    "blas",
    "lapack",
    "matlab_product_version",
    "parallel_toolbox_version",
    "release",
    "statistics_toolbox_version",
    "version",
)


def matlab_environment_descriptor(environment: dict) -> str:
    """Canonical cross-language descriptor used by MATLAB and Python.

    Field names are deliberately fixed and lexicographically ordered.  MATLAB's
    ``matlab_environment_identity.m`` emits the same UTF-8 newline-delimited
    ``field=value`` bytes.
    """
    if not isinstance(environment, dict) or set(environment) != set(
        _MATLAB_ENVIRONMENT_FIELDS
    ):
        got = sorted(environment) if isinstance(environment, dict) else type(
            environment
        ).__name__
        raise RuntimeError(
            "matlab_environment must define exactly "
            f"{list(_MATLAB_ENVIRONMENT_FIELDS)!r}; got {got!r}"
        )
    values: list[str] = []
    for field in _MATLAB_ENVIRONMENT_FIELDS:
        value = environment[field]
        if (
            not isinstance(value, str)
            or not value
            or "\n" in value
            or "\r" in value
        ):
            raise RuntimeError(
                f"matlab_environment.{field} must be one nonempty text line"
            )
        values.append(f"{field}={value}")
    return "\n".join(values)


def matlab_environment_sha256(environment: dict) -> str:
    return hashlib.sha256(
        matlab_environment_descriptor(environment).encode("utf-8")
    ).hexdigest()


def load_environment_lock(path: str | Path) -> dict:
    """Return the parsed lock plus a SHA-256 of its exact bytes."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    if spec.get("schema") != _LOCK_SCHEMA:
        raise RuntimeError(f"unsupported campaign environment lock: {path}")
    expected_matlab_sha = spec.get("matlab_environment_sha256")
    actual_matlab_sha = matlab_environment_sha256(
        spec.get("matlab_environment")
    )
    if expected_matlab_sha != actual_matlab_sha:
        raise RuntimeError(
            "campaign MATLAB environment descriptor/digest mismatch: "
            f"{actual_matlab_sha!r} != {expected_matlab_sha!r}"
        )
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
    }


def validate_environment_lock(lock: dict) -> dict:
    """Hard-fail when the running ablation environment differs from the lock."""
    spec = lock["spec"]
    mismatches: dict[str, tuple[object, object]] = {}

    actual_python = platform.python_version()
    if actual_python != spec["python"]:
        mismatches["python"] = (actual_python, spec["python"])
    actual_system = platform.system()
    if actual_system != spec["platform_system"]:
        mismatches["platform_system"] = (
            actual_system, spec["platform_system"])
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected_cublas = spec.get("cublas_workspace_config")
    if actual_cublas != expected_cublas:
        mismatches["cublas_workspace_config"] = (
            actual_cublas, expected_cublas)

    actual_packages = {}
    for distribution, expected in spec["packages"].items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        actual_packages[distribution] = actual
        if actual != expected:
            mismatches[f"package:{distribution}"] = (actual, expected)

    import torch
    actual_cuda = torch.version.cuda
    if actual_cuda != spec["torch_cuda"]:
        mismatches["torch_cuda"] = (actual_cuda, spec["torch_cuda"])
    actual_cudnn = torch.backends.cudnn.version()
    expected_cudnn = spec.get("cudnn_runtime")
    if actual_cudnn != expected_cudnn:
        mismatches["cudnn_runtime"] = (actual_cudnn, expected_cudnn)
    cuda_available = bool(torch.cuda.is_available())
    if spec.get("cuda_required") and not cuda_available:
        mismatches["cuda_available"] = (cuda_available, True)

    record = {
        "lock_sha256": lock["sha256"],
        "python": actual_python,
        "platform": platform.platform(),
        "packages": actual_packages,
        "torch_cuda": actual_cuda,
        "cuda_available": cuda_available,
        "gpu": (torch.cuda.get_device_name(0) if cuda_available else None),
        "cudnn": actual_cudnn,
        "cublas_workspace_config": actual_cublas,
        # This Python validator does not launch MATLAB. A00_Run.m independently
        # captures the same canonical descriptor and hard-gates its SHA-256
        # before production generation.
        "generator_requirements": {
            "matlab_environment": spec["matlab_environment"],
            "matlab_environment_sha256":
                spec["matlab_environment_sha256"],
            "validation":
                "full descriptor hard-gated by scour_MATLAB/A00_Run.m",
        },
    }
    if mismatches:
        raise RuntimeError(
            "campaign software environment differs from the protocol lock: "
            f"{mismatches}. Install requirements-campaign-py313-cu128.txt "
            "in Python 3.13.3; do not mix environments across studies.")
    return record
