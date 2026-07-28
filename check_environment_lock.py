"""Checks for the hash-carried campaign software environment lock."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

from core.environment import (
    load_environment_lock,
    matlab_environment_descriptor,
    matlab_environment_sha256,
    validate_environment_lock,
)

# Mirror the campaign entrypoint bootstrap before validation can inspect CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from core.utils import DETERMINISM_POLICY, set_global_seed  # noqa: E402

fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def rejects(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        check(name, True)
    else:
        check(name, False)


lock = load_environment_lock("environment/campaign-py313-cu128.json")
runtime = validate_environment_lock(lock)
check("current campaign environment matches exact lock",
      runtime["lock_sha256"] == lock["sha256"]
      and runtime["cuda_available"])
check("runtime record includes hardware without hiding software versions",
      bool(runtime["gpu"]) and "torch" in runtime["packages"]
      and runtime["cudnn"] == lock["spec"]["cudnn_runtime"] == 90701
      and runtime["cublas_workspace_config"] == ":4096:8"
      and runtime["generator_requirements"]["matlab_environment"]["release"]
          == "R2025b"
      and runtime["generator_requirements"]["matlab_environment_sha256"]
          == lock["spec"]["matlab_environment_sha256"])
check("MATLAB descriptor and SHA are canonical and authenticated",
      matlab_environment_sha256(lock["spec"]["matlab_environment"])
          == lock["spec"]["matlab_environment_sha256"]
      and matlab_environment_descriptor(
          lock["spec"]["matlab_environment"]
      ).splitlines()[0].startswith("arch=")
      and len(matlab_environment_descriptor(
          lock["spec"]["matlab_environment"]
      ).splitlines()) == 8)
driver_source = Path("comprehensive_ablation_multidamage.py").read_text(
    encoding="utf-8")
utils_source = Path("core/utils.py").read_text(encoding="utf-8")
check("cuBLAS determinism variable is set before torch can be imported",
      driver_source.index('CUBLAS_WORKSPACE_CONFIG')
      < driver_source.index("import torch"))
check("numeric-mode setters and post-assertions are explicit",
      'torch.set_float32_matmul_precision('
          'policy["float32_matmul_precision"])' in utils_source
      and 'torch.backends.cuda.matmul.allow_tf32 = \\\n' in utils_source
      and 'torch.backends.cudnn.allow_tf32 = policy[' in utils_source
      and 'if actual_numeric_mode != expected_numeric_mode:' in utils_source)
set_global_seed(42, DETERMINISM_POLICY)
check("PyTorch nondeterministic operations are configured to hard-fail",
      __import__("torch").are_deterministic_algorithms_enabled()
      and __import__("torch").get_deterministic_debug_mode() == 2
      and __import__("torch").backends.cudnn.deterministic
      and not __import__("torch").backends.cudnn.benchmark)
check("TF32 is disabled and float32 matmul precision is highest",
      not __import__("torch").backends.cuda.matmul.allow_tf32
      and not __import__("torch").backends.cudnn.allow_tf32
      and __import__("torch").get_float32_matmul_precision() == "highest")

mutated_determinism = {
    **DETERMINISM_POLICY,
    "cudnn_benchmark": True,
}
set_global_seed(42, mutated_determinism)
check("determinism behaviour is derived from its executable policy",
      __import__("torch").backends.cudnn.benchmark)
set_global_seed(42, DETERMINISM_POLICY)

mutated_numeric_mode = {
    **DETERMINISM_POLICY,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "high",
}
try:
    set_global_seed(42, mutated_numeric_mode)
except RuntimeError:
    check("numeric execution mode is derived from its executable policy",
          False)
else:
    check("numeric execution mode is derived from its executable policy",
          __import__("torch").backends.cuda.matmul.allow_tf32
          and __import__("torch").backends.cudnn.allow_tf32
          and __import__("torch").get_float32_matmul_precision() == "high")
finally:
    set_global_seed(42, DETERMINISM_POLICY)

with mock.patch("torch.get_float32_matmul_precision",
                return_value="medium"):
    rejects("numeric-mode postcondition mismatch hard-fails",
            lambda: set_global_seed(42, DETERMINISM_POLICY))
set_global_seed(42, DETERMINISM_POLICY)

missing_determinism_field = dict(DETERMINISM_POLICY)
missing_determinism_field.pop("torch_deterministic_warn_only")
try:
    set_global_seed(42, missing_determinism_field)
except ValueError:
    check("malformed determinism policy fails closed", True)
else:
    check("malformed determinism policy fails closed", False)

inconsistent_numeric_mode = {
    **DETERMINISM_POLICY,
    "cuda_matmul_allow_tf32": True,
}
try:
    set_global_seed(42, inconsistent_numeric_mode)
except ValueError:
    check("internally inconsistent numeric mode fails closed", True)
else:
    check("internally inconsistent numeric mode fails closed", False)

try:
    set_global_seed(-1, DETERMINISM_POLICY)
except ValueError:
    check("out-of-range seed fails closed", True)
else:
    check("out-of-range seed fails closed", False)

bad = copy.deepcopy(lock)
bad["spec"]["packages"]["numpy"] = "0.0.0"
rejects("package mismatch hard-fails", lambda: validate_environment_lock(bad))

bad_python = copy.deepcopy(lock)
bad_python["spec"]["python"] = "0.0.0"
rejects("Python-version mismatch hard-fails",
        lambda: validate_environment_lock(bad_python))

bad_platform = copy.deepcopy(lock)
bad_platform["spec"]["platform_system"] = "not-this-platform"
rejects("platform mismatch hard-fails",
        lambda: validate_environment_lock(bad_platform))

bad_cublas = copy.deepcopy(lock)
bad_cublas["spec"]["cublas_workspace_config"] = ":invalid:"
rejects("cuBLAS determinism mismatch hard-fails",
        lambda: validate_environment_lock(bad_cublas))

bad_cuda = copy.deepcopy(lock)
bad_cuda["spec"]["torch_cuda"] = "0.0"
rejects("Torch CUDA build mismatch hard-fails",
        lambda: validate_environment_lock(bad_cuda))

bad_cudnn = copy.deepcopy(lock)
bad_cudnn["spec"]["cudnn_runtime"] = 0
rejects("cuDNN runtime mismatch hard-fails",
        lambda: validate_environment_lock(bad_cudnn))

# Exercise the required-GPU branch even on the correctly configured audit PC.
# Without this mock, deleting the guard would stay green whenever CUDA happens
# to be available on the machine running the check.
with mock.patch("torch.cuda.is_available", return_value=False):
    rejects("required CUDA becoming unavailable hard-fails",
            lambda: validate_environment_lock(lock))

with tempfile.TemporaryDirectory(prefix="env-lock-") as tmp:
    source = Path("environment/campaign-py313-cu128.json").read_text(
        encoding="utf-8")
    first = Path(tmp, "first.json")
    second = Path(tmp, "second.json")
    first.write_text(source, encoding="utf-8")
    second.write_text(source + "\n", encoding="utf-8")
    check("one-byte lock change moves SHA-256",
          load_environment_lock(first)["sha256"]
          != load_environment_lock(second)["sha256"])
    malformed = Path(tmp, "malformed.json")
    malformed.write_text(
        source.replace(
            '"ttbi-campaign-environment-v2"',
            '"unsupported-environment-schema"',
            1,
        ),
        encoding="utf-8",
    )
    rejects("unsupported environment-lock schema hard-fails",
            lambda: load_environment_lock(malformed))

    source_spec = json.loads(source)

    def write_mutation(name: str, spec: dict) -> Path:
        path = Path(tmp, name)
        path.write_text(
            json.dumps(spec, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    descriptor_without_sha = copy.deepcopy(source_spec)
    descriptor_without_sha["matlab_environment"]["version"] += " MUTATED"
    rejects(
        "descriptor mutation without matching SHA hard-fails",
        lambda: load_environment_lock(
            write_mutation("descriptor-without-sha.json",
                           descriptor_without_sha)
        ),
    )

    sha_without_descriptor = copy.deepcopy(source_spec)
    sha_without_descriptor["matlab_environment_sha256"] = "0" * 64
    rejects(
        "SHA mutation without matching descriptor hard-fails",
        lambda: load_environment_lock(
            write_mutation("sha-without-descriptor.json",
                           sha_without_descriptor)
        ),
    )

    missing_descriptor_field = copy.deepcopy(source_spec)
    del missing_descriptor_field["matlab_environment"]["blas"]
    rejects(
        "missing MATLAB descriptor field hard-fails",
        lambda: load_environment_lock(
            write_mutation("missing-descriptor-field.json",
                           missing_descriptor_field)
        ),
    )

    extra_descriptor_field = copy.deepcopy(source_spec)
    extra_descriptor_field["matlab_environment"]["operating_system"] = (
        "UNAUTHENTICATED"
    )
    rejects(
        "extra MATLAB descriptor field hard-fails",
        lambda: load_environment_lock(
            write_mutation("extra-descriptor-field.json",
                           extra_descriptor_field)
        ),
    )

    missing_descriptor = copy.deepcopy(source_spec)
    del missing_descriptor["matlab_environment"]
    rejects(
        "missing MATLAB descriptor hard-fails",
        lambda: load_environment_lock(
            write_mutation("missing-descriptor.json", missing_descriptor)
        ),
    )

    missing_matlab_sha = copy.deepcopy(source_spec)
    del missing_matlab_sha["matlab_environment_sha256"]
    rejects(
        "missing MATLAB environment SHA hard-fails",
        lambda: load_environment_lock(
            write_mutation("missing-matlab-sha.json", missing_matlab_sha)
        ),
    )

if fails:
    raise SystemExit(f"ENVIRONMENT LOCK: {fails} FAILURE(S)")
print("ENVIRONMENT LOCK: ALL PASS")
