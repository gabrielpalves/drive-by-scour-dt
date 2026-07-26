"""Checks for the hash-carried campaign software environment lock."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile

from core.environment import load_environment_lock, validate_environment_lock

# Mirror the campaign entrypoint bootstrap before validation can inspect CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from core.utils import set_global_seed  # noqa: E402

fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


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
set_global_seed(42)
check("PyTorch nondeterministic operations are configured to hard-fail",
      __import__("torch").are_deterministic_algorithms_enabled()
      and __import__("torch").get_deterministic_debug_mode() == 2)

bad = copy.deepcopy(lock)
bad["spec"]["packages"]["numpy"] = "0.0.0"
try:
    validate_environment_lock(bad)
except RuntimeError:
    check("package mismatch hard-fails", True)
else:
    check("package mismatch hard-fails", False)

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

if fails:
    raise SystemExit(f"ENVIRONMENT LOCK: {fails} FAILURE(S)")
print("ENVIRONMENT LOCK: ALL PASS")
