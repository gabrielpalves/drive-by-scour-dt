"""Behavioral checks for portable campaign runtime qualification.

The committed JSON is a strict, authenticated known-good reference. It is
not an allow-list for host versions: exact Python, package, CUDA and cuDNN
values are provenance. Only capabilities used by the campaign are gates.
"""

from __future__ import annotations

import copy
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

from campaign_import_guard import enforce_import_boundary

enforce_import_boundary()

import core.environment as environment  # noqa: E402
from core.environment import (  # noqa: E402
    load_environment_lock,
    load_environment_lock_bytes,
    matlab_environment_sha256,
    validate_environment_lock,
)
from core.utils import DETERMINISM_POLICY, set_global_seed  # noqa: E402


ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "environment" / "campaign-py313-cu128.json"
FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    passed = bool(condition)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    FAILURES += int(not passed)


def rejects(label: str, function, message_fragment: str = "") -> None:
    try:
        function()
    except RuntimeError as exc:
        check(label, not message_fragment or message_fragment in str(exc))
    else:
        check(label, False)


def encoded(spec: dict) -> bytes:
    return (json.dumps(spec, sort_keys=True) + "\n").encode("utf-8")


@contextmanager
def simulated_host(
    packages: dict[str, str],
    *,
    python_implementation: str = "CPython",
    python_version: str = "3.12.99",
    python_version_info: tuple[int, ...] = (3, 12, 99),
    pointer_bytes: int = 8,
    cuda_available: bool = True,
    torch_cuda: str | None = "13.7",
    cudnn: int | None = 99999,
    cublas: str | None = ":4096:8",
):
    """Patch every host-dependent qualification input, including the GPU."""
    import torch

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(
            environment,
            "_installed_distribution_versions",
            return_value=dict(packages),
        ))
        stack.enter_context(mock.patch.object(
            environment.platform,
            "python_implementation",
            return_value=python_implementation,
        ))
        stack.enter_context(mock.patch.object(
            environment.platform,
            "python_version",
            return_value=python_version,
        ))
        stack.enter_context(mock.patch.object(
            environment.platform,
            "system",
            return_value="PortableOS",
        ))
        stack.enter_context(mock.patch.object(
            environment.platform,
            "platform",
            return_value="PortableOS-test-host",
        ))
        stack.enter_context(mock.patch.object(
            environment.struct,
            "calcsize",
            return_value=pointer_bytes,
        ))
        stack.enter_context(mock.patch.object(
            environment.sys,
            "version_info",
            python_version_info,
        ))
        stack.enter_context(mock.patch.object(torch.version, "cuda", torch_cuda))
        stack.enter_context(mock.patch.object(
            torch.backends.cudnn,
            "version",
            return_value=cudnn,
        ))
        stack.enter_context(mock.patch.object(
            torch.cuda,
            "is_available",
            return_value=cuda_available,
        ))
        stack.enter_context(mock.patch.object(
            torch.cuda,
            "get_device_name",
            return_value="Simulated portable CUDA device",
        ))
        environment_values = dict(os.environ)
        environment_values.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if cublas is not None:
            environment_values["CUBLAS_WORKSPACE_CONFIG"] = cublas
        stack.enter_context(mock.patch.dict(
            os.environ,
            environment_values,
            clear=True,
        ))
        yield


def matlab_mismatch_is_warning(path: Path, identifier: str) -> bool:
    source = path.read_text(encoding="utf-8")
    marker = (
        "~strcmp(actual_matlab_environment_sha256"
        if path.name == "A00_Run.m"
        else "~strcmp(actual_environment_sha256"
    )
    start = source.find(marker)
    end = source.find("\nelse", start)
    if start < 0 or end < 0:
        return False
    branch = source[start:end]
    return (
        f"warning('{identifier}'" in branch
        and "error(" not in branch
        and "known-good MATLAB reference" in branch
        and "descriptor is recorded" in branch
    )


def main() -> int:
    print("PORTABLE ENVIRONMENT QUALIFICATION")

    raw = REFERENCE_PATH.read_bytes()
    lock = load_environment_lock(REFERENCE_PATH)
    spec = lock["spec"]
    check(
        "strict v3 reference JSON loads and authenticates its bytes",
        spec["schema"] == "ttbi-campaign-environment-v3"
        and len(lock["sha256"]) == 64,
    )
    check(
        "MATLAB reference descriptor reproduces its stored SHA-256",
        matlab_environment_sha256(spec["matlab_environment"])
        == spec["matlab_environment_sha256"],
    )

    duplicate = raw.replace(
        b'"schema": "ttbi-campaign-environment-v3",',
        b'"schema": "ttbi-campaign-environment-v3",\n  "schema": "duplicate",',
        1,
    )
    rejects(
        "duplicate JSON key rejected",
        lambda: load_environment_lock_bytes(duplicate, source="duplicate.json"),
        "strict UTF-8 JSON",
    )
    extra = copy.deepcopy(spec)
    extra["unreviewed"] = True
    rejects(
        "extra reference field rejected",
        lambda: load_environment_lock_bytes(encoded(extra), source="extra.json"),
        "fields differ",
    )
    unsupported = copy.deepcopy(spec)
    unsupported["schema"] = "ttbi-campaign-environment-v999"
    rejects(
        "unsupported environment-lock schema hard-fails",
        lambda: load_environment_lock_bytes(
            encoded(unsupported), source="schema.json"
        ),
        "unsupported campaign environment schema",
    )
    matlab_drift = copy.deepcopy(spec)
    matlab_drift["matlab_environment"]["version"] += " mutated"
    rejects(
        "descriptor mutation without matching SHA hard-fails",
        lambda: load_environment_lock_bytes(
            encoded(matlab_drift), source="matlab-drift.json"
        ),
        "descriptor/digest mismatch",
    )
    matlab_extra = copy.deepcopy(spec)
    matlab_extra["matlab_environment"]["extra"] = "unauthenticated"
    rejects(
        "MATLAB descriptor extra field rejected",
        lambda: load_environment_lock_bytes(
            encoded(matlab_extra), source="matlab-extra.json"
        ),
        "matlab_environment must define exactly",
    )

    drifted_packages = {
        name: f"portable-{index}"
        for index, name in enumerate(spec["packages"], 1)
    }
    drifted_packages["host-extra-diagnostic-package"] = "2026.8"
    with simulated_host(drifted_packages):
        record = validate_environment_lock(lock)
    mismatches = record["reference_mismatches"]
    check(
        "Python version drift is provenance and does not reject",
        record["qualified"]
        and record["python"] == "3.12.99"
        and "python" in mismatches,
    )
    check(
        "package-version drift is recorded as provenance",
        "package:numpy" in mismatches
        and mismatches["package:numpy"][0].startswith("portable-"),
    )
    check(
        "CUDA and cuDNN version drift is provenance",
        mismatches.get("torch_cuda") == ("13.7", spec["torch_cuda"])
        and mismatches.get("cudnn_runtime") == (99999, spec["cudnn_runtime"]),
    )
    check(
        "additional installed packages are accepted",
        record["packages"].get("host-extra-diagnostic-package") == "2026.8",
    )
    check(
        "simulated GPU proves checks do not require the audit PC GPU",
        record["gpu"] == "Simulated portable CUDA device"
        and record["cuda_available"],
    )

    without_numpy = dict(drifted_packages)
    without_numpy.pop("numpy")
    with simulated_host(without_numpy):
        rejects(
            "missing required package hard-fails",
            lambda: validate_environment_lock(lock),
            "missing required distributions",
        )
    with simulated_host(drifted_packages, cuda_available=False):
        rejects(
            "required CUDA becoming unavailable hard-fails",
            lambda: validate_environment_lock(lock),
            "requires a CUDA device",
        )
    with simulated_host(drifted_packages, cublas=":conflicting:"):
        rejects(
            "cuBLAS deterministic-setting conflict hard-fails",
            lambda: validate_environment_lock(lock),
            "conflicts with the deterministic campaign setting",
        )
    with simulated_host(drifted_packages, cublas=None):
        unset_record = validate_environment_lock(lock)
    check(
        "unset cuBLAS setting is accepted and recorded for entrypoint setup",
        unset_record["qualified"]
        and "cublas_workspace_config" in unset_record["reference_mismatches"],
    )
    with simulated_host(drifted_packages, python_implementation="PyPy"):
        rejects(
            "non-CPython runtime hard-fails",
            lambda: validate_environment_lock(lock),
            "requires CPython",
        )
    with simulated_host(drifted_packages, python_version_info=(3, 10, 99)):
        rejects(
            "CPython older than the supported minimum hard-fails",
            lambda: validate_environment_lock(lock),
            "3.11 or newer",
        )
    with simulated_host(drifted_packages, pointer_bytes=4):
        rejects(
            "32-bit CPython hard-fails",
            lambda: validate_environment_lock(lock),
            "64-bit Python",
        )

    a00_path = ROOT / "scour_MATLAB" / "A00_Run.m"
    f25_path = ROOT / "scour_MATLAB" / "F25_Run.m"
    check(
        "A00 MATLAB reference mismatch warns instead of equality-gating",
        matlab_mismatch_is_warning(
            a00_path, "A00:ReferenceMATLABEnvironmentDiffers"
        )
        and "A00:EnvironmentLockDigest" in a00_path.read_text(encoding="utf-8"),
    )
    check(
        "F25 MATLAB reference mismatch warns instead of equality-gating",
        matlab_mismatch_is_warning(
            f25_path, "F25_Run:ReferenceMATLABEnvironmentDiffers"
        )
        and "F25_Run:EnvironmentLockDigest" in f25_path.read_text(
            encoding="utf-8"
        ),
    )

    # Determinism stays EXACT even though versions became provenance. With the
    # host no longer pinned, the executable determinism policy is the remaining
    # control on numeric reproducibility within one host, so prove that
    # core.utils derives its behaviour from that policy rather than hardwiring
    # it. These behavioural probes were lost when this checker was rewritten
    # around capability qualification; check_training_policy_mutation_guards.py
    # mutates core/utils.py and requires exactly the two labels below.
    utils_source = (ROOT / "core" / "utils.py").read_text(encoding="utf-8")
    check(
        "numeric-mode setters and post-assertions are explicit",
        'torch.set_float32_matmul_precision('
        'policy["float32_matmul_precision"])' in utils_source
        and "torch.backends.cuda.matmul.allow_tf32 = \\\n" in utils_source
        and 'torch.backends.cudnn.allow_tf32 = policy[' in utils_source
        and "if actual_numeric_mode != expected_numeric_mode:" in utils_source,
    )
    set_global_seed(42, DETERMINISM_POLICY)
    check(
        "PyTorch nondeterministic operations are configured to hard-fail",
        __import__("torch").are_deterministic_algorithms_enabled()
        and __import__("torch").get_deterministic_debug_mode() == 2
        and __import__("torch").backends.cudnn.deterministic
        and not __import__("torch").backends.cudnn.benchmark,
    )
    try:
        set_global_seed(42, {**DETERMINISM_POLICY, "cudnn_benchmark": True})
        check(
            "determinism behaviour is derived from its executable policy",
            __import__("torch").backends.cudnn.benchmark,
        )
    finally:
        set_global_seed(42, DETERMINISM_POLICY)
    try:
        set_global_seed(
            42,
            {
                **DETERMINISM_POLICY,
                "cuda_matmul_allow_tf32": True,
                "cudnn_allow_tf32": True,
                "float32_matmul_precision": "high",
            },
        )
    except RuntimeError:
        check("numeric execution mode is derived from its executable policy", False)
    else:
        check(
            "numeric execution mode is derived from its executable policy",
            __import__("torch").backends.cuda.matmul.allow_tf32
            and __import__("torch").backends.cudnn.allow_tf32
            and __import__("torch").get_float32_matmul_precision() == "high",
        )
    finally:
        set_global_seed(42, DETERMINISM_POLICY)
    with mock.patch(
        "torch.get_float32_matmul_precision", return_value="medium"
    ):
        rejects(
            "numeric-mode postcondition mismatch hard-fails",
            lambda: set_global_seed(42, DETERMINISM_POLICY),
        )
    set_global_seed(42, DETERMINISM_POLICY)

    if FAILURES:
        print(f"\nENVIRONMENT COMPATIBILITY: {FAILURES} FAILURE(S)")
        return 1
    print("\nENVIRONMENT COMPATIBILITY: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
