"""Pinned software-environment provenance for the ablation campaign."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

from core.environment_artifacts import (
    canonical_distribution_name as _canonical_distribution_name,
    current_python_runtime_descriptor as _current_python_runtime_descriptor,
    distribution_record_root as _distribution_record_root,
    installed_distribution_record_roots as _installed_distribution_record_roots,
    installed_distribution_versions as _installed_distribution_versions,
    is_sha256 as _is_sha256,
    normalise_record_path as _normalise_record_path,
    python_base_runtime_root as _python_base_runtime_root,
)


_LOCK_SCHEMA = "ttbi-campaign-environment-v2"
_PACKAGE_ARTIFACT_POLICY = "wheel-record-sha256-v1"
_LOCK_FIELDS = frozenset({
    "schema",
    "python",
    "python_runtime",
    "platform_system",
    "cuda_required",
    "torch_cuda",
    "cudnn_runtime",
    "cublas_workspace_config",
    "package_inventory_policy",
    "package_artifact_policy",
    "package_record_sha256",
    "matlab_environment",
    "matlab_environment_sha256",
    "packages",
})
_PYTHON_RUNTIME_FIELDS = frozenset({
    "version",
    "implementation",
    "implementation_version",
    "machine",
    "architecture",
    "cache_tag",
    "soabi",
    "abiflags",
    "gil_enabled",
    "build",
    "compiler",
    "base_executable_sha256",
    "venv_executable_sha256",
    "runtime_dll_sha256",
    "base_runtime_sha256",
    "base_runtime_file_count",
    "pyvenv_config_semantic_sha256",
    "virtual_environment_manager",
    "virtual_environment_manager_version",
    "include_system_site_packages",
    "user_site_enabled",
    "pythonpath_environment",
    "pythonhome_environment",
    "sys_path_roles",
    "startup_files_sha256",
})
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


def _nonempty_text(value: object, owner: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
    ):
        raise RuntimeError(f"{owner} must be one nonempty text line")
    return value


def _validate_python_runtime_spec(spec: dict) -> dict:
    runtime = spec.get("python_runtime")
    if not isinstance(runtime, dict) or set(runtime) != _PYTHON_RUNTIME_FIELDS:
        got = sorted(runtime) if isinstance(runtime, dict) else type(
            runtime
        ).__name__
        raise RuntimeError(
            "python_runtime fields differ from the campaign contract: "
            f"{got!r}"
        )
    text_fields = (
        "version",
        "implementation",
        "implementation_version",
        "machine",
        "architecture",
        "cache_tag",
        "soabi",
        "compiler",
        "virtual_environment_manager",
        "virtual_environment_manager_version",
    )
    for field in text_fields:
        _nonempty_text(runtime[field], f"python_runtime.{field}")
    if runtime["abiflags"] != "":
        raise RuntimeError("campaign CPython ABI flags must be empty")
    if runtime["implementation"] != "CPython":
        raise RuntimeError("campaign Python implementation must be CPython")
    if runtime["machine"] != "AMD64" or runtime["architecture"] != "64bit":
        raise RuntimeError("campaign Python must use the Windows AMD64 ABI")
    if runtime["gil_enabled"] is not True:
        raise RuntimeError("free-threaded Python is outside the campaign lock")
    if (
        runtime["virtual_environment_manager"] != "uv"
        or runtime["include_system_site_packages"] is not False
        or runtime["user_site_enabled"] is not False
    ):
        raise RuntimeError(
            "campaign Python must use the locked isolated uv environment"
        )
    build = runtime["build"]
    if (
        not isinstance(build, list)
        or len(build) != 2
        or any(not isinstance(value, str) or not value for value in build)
    ):
        raise RuntimeError("python_runtime.build must contain two text fields")
    for field in (
        "base_executable_sha256",
        "venv_executable_sha256",
        "runtime_dll_sha256",
        "base_runtime_sha256",
        "pyvenv_config_semantic_sha256",
    ):
        if not _is_sha256(runtime[field]):
            raise RuntimeError(f"python_runtime.{field} is not a SHA-256")
    if (
        isinstance(runtime["base_runtime_file_count"], bool)
        or not isinstance(runtime["base_runtime_file_count"], int)
        or runtime["base_runtime_file_count"] <= 0
    ):
        raise RuntimeError(
            "python_runtime.base_runtime_file_count must be positive"
        )
    if (
        runtime["pythonpath_environment"] is not None
        or runtime["pythonhome_environment"] is not None
    ):
        raise RuntimeError("PYTHONPATH and PYTHONHOME must both be absent")
    expected_path_roles = [
        "source_root",
        "base_zip",
        "base_dlls",
        "base_lib",
        "base_prefix",
        "venv_prefix",
        "venv_site_packages",
    ]
    if runtime["sys_path_roles"] != expected_path_roles:
        raise RuntimeError(
            "python_runtime.sys_path_roles differs from the fixed import path"
        )
    startup = runtime["startup_files_sha256"]
    if not isinstance(startup, dict) or not startup:
        raise RuntimeError(
            "python_runtime.startup_files_sha256 must be nonempty"
        )
    for name, digest in startup.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not _is_sha256(digest)
        ):
            raise RuntimeError("invalid Python startup-file lock entry")
    if runtime["version"] != spec.get("python"):
        raise RuntimeError(
            "top-level Python version and runtime descriptor disagree"
        )
    return runtime


def _locked_package_versions(spec: dict) -> dict[str, str]:
    """Validate and canonicalize the complete package inventory in the lock."""
    if spec.get("package_inventory_policy") != "exact":
        raise RuntimeError(
            "campaign environment must require an exact package inventory"
        )
    packages = spec.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise RuntimeError("campaign package inventory must be a nonempty object")

    locked: dict[str, str] = {}
    for declared_name, version in packages.items():
        canonical_name = _canonical_distribution_name(declared_name)
        if canonical_name != declared_name:
            raise RuntimeError(
                "campaign package names must use canonical PEP 503 spelling: "
                f"{declared_name!r} != {canonical_name!r}"
            )
        if not isinstance(version, str) or not version:
            raise RuntimeError(
                f"campaign package {declared_name!r} has no exact version"
            )
        if canonical_name in locked:
            raise RuntimeError(
                f"duplicate canonical campaign package: {canonical_name}"
            )
        locked[canonical_name] = version
    return locked


def _locked_package_record_roots(
    spec: dict,
    packages: dict[str, str],
) -> dict[str, str]:
    """Validate the one authenticated wheel-content root per distribution."""
    if spec.get("package_artifact_policy") != _PACKAGE_ARTIFACT_POLICY:
        raise RuntimeError(
            "campaign package artifact policy must authenticate wheel RECORDs"
        )
    roots = spec.get("package_record_sha256")
    if not isinstance(roots, dict) or set(roots) != set(packages):
        raise RuntimeError(
            "package RECORD-root inventory must exactly match package names"
        )
    for name, digest in roots.items():
        if name != _canonical_distribution_name(name) or not _is_sha256(digest):
            raise RuntimeError(
                f"invalid package RECORD-root entry for {name!r}"
            )
    return dict(sorted(roots.items()))


def _validate_lock_spec(spec: object) -> dict:
    """Validate the complete v2 schema before any field is trusted."""
    if not isinstance(spec, dict) or set(spec) != _LOCK_FIELDS:
        got = sorted(spec) if isinstance(spec, dict) else type(spec).__name__
        raise RuntimeError(
            f"campaign environment fields differ from v2: {got!r}"
        )
    if spec["schema"] != _LOCK_SCHEMA:
        raise RuntimeError("unsupported campaign environment schema")
    _nonempty_text(spec["python"], "python")
    _nonempty_text(spec["platform_system"], "platform_system")
    _nonempty_text(spec["torch_cuda"], "torch_cuda")
    _nonempty_text(
        spec["cublas_workspace_config"],
        "cublas_workspace_config",
    )
    if spec["platform_system"] != "Windows":
        raise RuntimeError("campaign platform_system must be Windows")
    if spec["cuda_required"] is not True:
        raise RuntimeError("campaign lock must require CUDA")
    if (
        isinstance(spec["cudnn_runtime"], bool)
        or not isinstance(spec["cudnn_runtime"], int)
        or spec["cudnn_runtime"] <= 0
    ):
        raise RuntimeError("cudnn_runtime must be a positive integer")
    _validate_python_runtime_spec(spec)
    packages = _locked_package_versions(spec)
    _locked_package_record_roots(spec, packages)
    return spec


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


def load_environment_lock_bytes(
    raw: bytes,
    *,
    source: str | Path,
) -> dict:
    """Validate one already-authenticated environment-lock byte buffer."""
    if not isinstance(raw, bytes):
        raise RuntimeError("campaign environment lock payload must be bytes")
    source_text = Path(source).as_posix()

    def unique_object(items: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        spec = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"campaign environment lock is not strict UTF-8 JSON: "
            f"{source_text}"
        ) from exc
    _validate_lock_spec(spec)
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
        "path": source_text,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "spec": spec,
    }


def load_environment_lock(path: str | Path) -> dict:
    """Return the parsed lock plus a SHA-256 of its exact bytes.

    Evidence code that already owns a secure file snapshot should call
    :func:`load_environment_lock_bytes` so parsing and hashing consume the
    same buffer.
    """
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"campaign environment lock is not a regular file: {path}"
        )
    return load_environment_lock_bytes(path.read_bytes(), source=path)


def validate_environment_lock(lock: dict) -> dict:
    """Hard-fail when the running ablation environment differs from the lock."""
    if (
        not isinstance(lock, dict)
        or set(lock) != {"path", "sha256", "spec"}
        or not isinstance(lock["path"], str)
        or not lock["path"]
        or not _is_sha256(lock["sha256"])
    ):
        raise RuntimeError("malformed loaded campaign environment lock")
    spec = lock["spec"]
    _validate_lock_spec(spec)
    mismatches: dict[str, tuple[object, object]] = {}

    expected_python_runtime = spec["python_runtime"]
    actual_python_runtime = _current_python_runtime_descriptor()
    actual_python = actual_python_runtime["version"]
    if actual_python != spec["python"]:
        mismatches["python"] = (actual_python, spec["python"])
    for field, expected in expected_python_runtime.items():
        actual = actual_python_runtime.get(field, "MISSING")
        if actual != expected:
            mismatches[f"python_runtime:{field}"] = (actual, expected)
    actual_system = platform.system()
    if actual_system != spec["platform_system"]:
        mismatches["platform_system"] = (
            actual_system, spec["platform_system"])
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected_cublas = spec.get("cublas_workspace_config")
    if actual_cublas != expected_cublas:
        mismatches["cublas_workspace_config"] = (
            actual_cublas, expected_cublas)

    expected_packages = _locked_package_versions(spec)
    actual_packages = _installed_distribution_versions()
    for distribution, expected in expected_packages.items():
        actual = actual_packages.get(distribution, "MISSING")
        if actual != expected:
            mismatches[f"package:{distribution}"] = (actual, expected)
    for distribution, actual in actual_packages.items():
        if distribution not in expected_packages:
            mismatches[f"unexpected-package:{distribution}"] = (
                actual,
                "ABSENT",
            )

    import torch
    actual_cuda = torch.version.cuda
    if actual_cuda != spec["torch_cuda"]:
        mismatches["torch_cuda"] = (actual_cuda, spec["torch_cuda"])
    actual_cudnn = torch.backends.cudnn.version()
    expected_cudnn = spec.get("cudnn_runtime")
    if actual_cudnn != expected_cudnn:
        mismatches["cudnn_runtime"] = (actual_cudnn, expected_cudnn)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        mismatches["cuda_available"] = (cuda_available, True)

    # Hashing RECORD-owned runtime files and scanning for unowned non-cache
    # package files are deferred until every cheap environmental check passes.
    # A wrong host fails quickly; an authorized campaign pays this cost once.
    if mismatches:
        raise RuntimeError(
            "campaign software environment differs from the protocol lock: "
            f"{mismatches}. Install requirements-campaign-py313-cu128.txt "
            "in the locked Python runtime; do not mix environments across "
            "studies."
        )

    expected_record_roots = _locked_package_record_roots(
        spec,
        expected_packages,
    )
    actual_record_roots = _installed_distribution_record_roots()
    for distribution, expected in expected_record_roots.items():
        actual = actual_record_roots.get(distribution, "MISSING")
        if actual != expected:
            mismatches[f"package-record:{distribution}"] = (actual, expected)
    for distribution, actual in actual_record_roots.items():
        if distribution not in expected_record_roots:
            mismatches[f"unexpected-package-record:{distribution}"] = (
                actual,
                "ABSENT",
            )
    if mismatches:
        raise RuntimeError(
            "campaign package bytes differ from the authenticated wheel "
            f"RECORD roots: {mismatches}. Rebuild the isolated environment "
            "from requirements-campaign-py313-cu128.txt."
        )

    record = {
        "lock_sha256": lock["sha256"],
        "python": actual_python,
        "python_runtime": actual_python_runtime,
        "platform": platform.platform(),
        "packages": actual_packages,
        "package_record_sha256": actual_record_roots,
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
    return record
