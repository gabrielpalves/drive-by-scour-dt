"""Checks for the hash-carried campaign software environment lock."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from unittest import mock

from core.environment import load_environment_lock, validate_environment_lock

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
      and runtime["cublas_workspace_config"] == ":4096:8"
      and runtime["generator_requirements"]["matlab_release"] == "R2025b")
driver_source = Path("comprehensive_ablation_multidamage.py").read_text(
    encoding="utf-8")
check("cuBLAS determinism variable is set before torch can be imported",
      driver_source.index('CUBLAS_WORKSPACE_CONFIG')
      < driver_source.index("import torch"))
set_global_seed(42, DETERMINISM_POLICY)
check("PyTorch nondeterministic operations are configured to hard-fail",
      __import__("torch").are_deterministic_algorithms_enabled()
      and __import__("torch").get_deterministic_debug_mode() == 2
      and __import__("torch").backends.cudnn.deterministic
      and not __import__("torch").backends.cudnn.benchmark)

mutated_determinism = {
    **DETERMINISM_POLICY,
    "cudnn_benchmark": True,
}
set_global_seed(42, mutated_determinism)
check("determinism behaviour is derived from its executable policy",
      __import__("torch").backends.cudnn.benchmark)
set_global_seed(42, DETERMINISM_POLICY)

missing_determinism_field = dict(DETERMINISM_POLICY)
missing_determinism_field.pop("torch_deterministic_warn_only")
try:
    set_global_seed(42, missing_determinism_field)
except ValueError:
    check("malformed determinism policy fails closed", True)
else:
    check("malformed determinism policy fails closed", False)

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
            '"ttbi-campaign-environment-v1"',
            '"unsupported-environment-schema"',
            1,
        ),
        encoding="utf-8",
    )
    rejects("unsupported environment-lock schema hard-fails",
            lambda: load_environment_lock(malformed))

if fails:
    raise SystemExit(f"ENVIRONMENT LOCK: {fails} FAILURE(S)")
print("ENVIRONMENT LOCK: ALL PASS")
