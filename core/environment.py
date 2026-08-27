"""Portable runtime qualification and software provenance for the campaign.

The JSON descriptor is a known-good reference environment, not an execution
allow-list.  Scientific runs are admitted by the capabilities they need
(required packages, CUDA and deterministic configuration); exact versions and
binary hashes are retained only as provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import sys
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


_LOCK_SCHEMA = "ttbi-campaign-environment-v3"
_PACKAGE_ARTIFACT_POLICY = "known-good-wheel-record-reference-v1"
_PACKAGE_INVENTORY_POLICY = "known-good-reference"
_MINIMUM_PYTHON = (3, 11)
_REQUIRED_DISTRIBUTIONS = frozenset({
    "joblib",
    "matplotlib",
    "numpy",
    "optuna",
    "pandas",
    "pywavelets",
    "scikit-learn",
    "scipy",
    "seaborn",
    "torch",
    "tqdm",
})
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
            "known-good Python reference must describe its isolated uv environment"
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
            "python_runtime.sys_path_roles differs from the known-good "
            "reference inventory"
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
    """Validate the known-good package inventory carried by the reference."""
    if spec.get("package_inventory_policy") != _PACKAGE_INVENTORY_POLICY:
        raise RuntimeError(
            "campaign environment must label package versions as a "
            "known-good reference"
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
    """Validate reference wheel roots without imposing them on live hosts."""
    if spec.get("package_artifact_policy") != _PACKAGE_ARTIFACT_POLICY:
        raise RuntimeError(
            "campaign package artifact policy must mark wheel roots as "
            "known-good reference provenance"
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
    """Validate the complete v3 reference schema before fields are trusted."""
    if not isinstance(spec, dict) or set(spec) != _LOCK_FIELDS:
        got = sorted(spec) if isinstance(spec, dict) else type(spec).__name__
        raise RuntimeError(
            f"campaign environment fields differ from v3: {got!r}"
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
    """Qualify required capabilities and record drift from the reference.

    Version, platform, environment-manager and wheel-byte differences never
    reject a host.  They are returned in ``reference_mismatches``.  A run is
    rejected only when a capability used by the campaign is absent: supported
    64-bit CPython, required distributions, CUDA, or a conflicting cuBLAS
    deterministic-workspace setting.
    """
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
    reference_mismatches: dict[str, tuple[object, object]] = {}

    expected_python_runtime = spec["python_runtime"]
    implementation = platform.python_implementation()
    actual_python = platform.python_version()
    actual_python_runtime = {
        "version": actual_python,
        "implementation": implementation,
        "implementation_version": platform.python_version(),
        "machine": platform.machine(),
        "architecture": f"{struct.calcsize('P') * 8}bit",
        "cache_tag": getattr(sys.implementation, "cache_tag", None),
        "compiler": platform.python_compiler(),
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "pythonpath_environment": os.environ.get("PYTHONPATH"),
        "pythonhome_environment": os.environ.get("PYTHONHOME"),
        "sys_path": [str(value) for value in sys.path],
    }
    if implementation != "CPython":
        raise RuntimeError(
            "campaign execution requires CPython because the numerical "
            "extensions are qualified for its ABI"
        )
    if sys.version_info < _MINIMUM_PYTHON:
        raise RuntimeError(
            "campaign execution requires CPython "
            f"{_MINIMUM_PYTHON[0]}.{_MINIMUM_PYTHON[1]} or newer"
        )
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("campaign execution requires a 64-bit Python")
    if actual_python != spec["python"]:
        reference_mismatches["python"] = (actual_python, spec["python"])
    for field, expected in expected_python_runtime.items():
        actual = actual_python_runtime.get(field, "NOT_RECORDED_PORTABLY")
        if actual != expected:
            reference_mismatches[f"python_runtime:{field}"] = (
                actual,
                expected,
            )
    actual_system = platform.system()
    if actual_system != spec["platform_system"]:
        reference_mismatches["platform_system"] = (
            actual_system, spec["platform_system"])
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected_cublas = spec.get("cublas_workspace_config")
    if actual_cublas not in (None, expected_cublas):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the deterministic "
            f"campaign setting: {actual_cublas!r} != {expected_cublas!r}"
        )
    if actual_cublas != expected_cublas:
        reference_mismatches["cublas_workspace_config"] = (
            actual_cublas, expected_cublas)

    expected_packages = _locked_package_versions(spec)
    actual_packages = _installed_distribution_versions()
    missing_required = sorted(
        distribution
        for distribution in _REQUIRED_DISTRIBUTIONS
        if distribution not in actual_packages
    )
    if missing_required:
        raise RuntimeError(
            "campaign runtime is missing required distributions: "
            f"{missing_required}. Install the project requirements; exact "
            "reference versions are not required."
        )
    for distribution, expected in expected_packages.items():
        actual = actual_packages.get(distribution, "NOT_INSTALLED")
        if actual != expected:
            reference_mismatches[f"package:{distribution}"] = (
                actual,
                expected,
            )

    import torch
    actual_cuda = torch.version.cuda
    if actual_cuda != spec["torch_cuda"]:
        reference_mismatches["torch_cuda"] = (
            actual_cuda,
            spec["torch_cuda"],
        )
    actual_cudnn = torch.backends.cudnn.version()
    expected_cudnn = spec.get("cudnn_runtime")
    if actual_cudnn != expected_cudnn:
        reference_mismatches["cudnn_runtime"] = (
            actual_cudnn,
            expected_cudnn,
        )
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError(
            "campaign training requires a CUDA device visible to PyTorch; "
            "run the local capacity preflight on this PC before dispatch"
        )

    record = {
        "qualification_policy": "required-capabilities-and-local-smokes-v1",
        "qualified": True,
        "lock_sha256": lock["sha256"],
        "reference_schema": spec["schema"],
        "python": actual_python,
        "python_runtime": actual_python_runtime,
        "platform": platform.platform(),
        "packages": actual_packages,
        # RECORD hashes in the reference remain useful forensic metadata, but
        # they are deliberately not recomputed as a portability gate.
        "package_record_sha256": {},
        "reference_mismatches": reference_mismatches,
        "required_distributions": sorted(_REQUIRED_DISTRIBUTIONS),
        "torch_cuda": actual_cuda,
        "cuda_available": cuda_available,
        "gpu": (torch.cuda.get_device_name(0) if cuda_available else None),
        "cudnn": actual_cudnn,
        "cublas_workspace_config": actual_cublas,
        # This Python validator does not launch MATLAB. A00_Run.m records the
        # actual MATLAB descriptor and qualifies required functions locally.
        "generator_requirements": {
            "matlab_environment": spec["matlab_environment"],
            "matlab_environment_sha256":
                spec["matlab_environment_sha256"],
            "validation": "reference only; capabilities and smokes gate runs",
        },
    }
    return record
