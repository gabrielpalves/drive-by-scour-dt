"""Pinned software-environment provenance for the ablation campaign."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path


def load_environment_lock(path: str | Path) -> dict:
    """Return the parsed lock plus a SHA-256 of its exact bytes."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    if spec.get("schema") != "ttbi-campaign-environment-v1":
        raise RuntimeError(f"unsupported campaign environment lock: {path}")
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
        "cudnn": (torch.backends.cudnn.version() if cuda_available else None),
        "cublas_workspace_config": actual_cublas,
        # This Python validator does not launch MATLAB. A00_Run.m independently
        # hard-gates version('-release') before generation and writes the actual
        # release into case_info for newly generated data.
        "generator_requirements": {
            "matlab_release": spec.get("matlab_release"),
            "validation": "hard-gated by scour_MATLAB/A00_Run.m",
        },
    }
    if mismatches:
        raise RuntimeError(
            "campaign software environment differs from the protocol lock: "
            f"{mismatches}. Install requirements-campaign-py313-cu128.txt "
            "in Python 3.13.3; do not mix environments across studies.")
    return record
