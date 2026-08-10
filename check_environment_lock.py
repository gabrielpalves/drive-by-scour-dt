"""Checks for the hash-carried campaign software environment lock."""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
            "imports"
        )
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()


def _run_static_mutation_smoke() -> int:
    """Fast AST guard for R4 mutants after one full behavioral baseline."""
    import ast
    import json
    from pathlib import Path

    source_root = Path(__file__).resolve().parent
    environment_path = source_root / "core" / "environment.py"
    lock_path = source_root / "environment" / "campaign-py313-cu128.json"
    tree = ast.parse(
        environment_path.read_text(encoding="utf-8"),
        filename=str(environment_path),
    )

    def function(name: str) -> ast.FunctionDef | None:
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        return matches[0] if len(matches) == 1 else None

    def runtime_error_raise(node: ast.AST) -> bool:
        return any(
            isinstance(candidate, ast.Raise)
            and isinstance(candidate.exc, ast.Call)
            and isinstance(candidate.exc.func, ast.Name)
            and candidate.exc.func.id == "RuntimeError"
            for candidate in ast.walk(node)
        )

    def not_equal_names(
        node: ast.AST,
        left: str,
        right: str,
    ) -> bool:
        return (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == left
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotEq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == right
        )

    def mismatch_write(node: ast.AST, key_fragment: str) -> bool:
        for candidate in ast.walk(node):
            if not (
                isinstance(candidate, ast.Subscript)
                and isinstance(candidate.value, ast.Name)
                and candidate.value.id == "mismatches"
            ):
                continue
            fragments = [
                item.value
                for item in ast.walk(candidate.slice)
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            ]
            if any(key_fragment in fragment for fragment in fragments):
                return True
        return False

    validate = function("validate_environment_lock")
    lock_validator = function("_validate_lock_spec")
    lock_loader = function("load_environment_lock_bytes")
    matlab_descriptor = function("matlab_environment_descriptor")

    terminal_mismatch_guard = bool(
        validate is not None
        and any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "mismatches"
            and runtime_error_raise(node)
            for node in ast.walk(validate)
        )
    )

    package_guard = False
    if validate is not None and terminal_mismatch_guard:
        for loop in ast.walk(validate):
            if not (
                isinstance(loop, ast.For)
                and isinstance(loop.iter, ast.Call)
                and isinstance(loop.iter.func, ast.Attribute)
                and loop.iter.func.attr == "items"
                and isinstance(loop.iter.func.value, ast.Name)
                and loop.iter.func.value.id == "expected_packages"
            ):
                continue
            package_guard = any(
                isinstance(node, ast.If)
                and not_equal_names(node.test, "actual", "expected")
                and mismatch_write(node, "package:")
                for node in loop.body
            )

    cublas_guard = bool(
        validate is not None
        and terminal_mismatch_guard
        and any(
            isinstance(node, ast.If)
            and not_equal_names(
                node.test, "actual_cublas", "expected_cublas"
            )
            and mismatch_write(node, "cublas_workspace_config")
            for node in ast.walk(validate)
        )
    )
    cuda_guard = bool(
        validate is not None
        and terminal_mismatch_guard
        and any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "cuda_available"
            and mismatch_write(node, "cuda_available")
            for node in ast.walk(validate)
        )
    )

    schema_guard = bool(
        lock_validator is not None
        and any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Subscript)
            and isinstance(node.test.left.value, ast.Name)
            and node.test.left.value.id == "spec"
            and isinstance(node.test.left.slice, ast.Constant)
            and node.test.left.slice.value == "schema"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.NotEq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
            and node.test.comparators[0].id == "_LOCK_SCHEMA"
            and runtime_error_raise(node)
            for node in ast.walk(lock_validator)
        )
    )
    descriptor_sha_guard = bool(
        lock_loader is not None
        and any(
            isinstance(node, ast.If)
            and not_equal_names(
                node.test, "expected_matlab_sha", "actual_matlab_sha"
            )
            and runtime_error_raise(node)
            for node in ast.walk(lock_loader)
        )
    )

    def set_call(node: ast.AST, argument: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == argument
        )

    exact_matlab_fields_guard = bool(
        matlab_descriptor is not None
        and any(
            isinstance(node, ast.If)
            and runtime_error_raise(node)
            and any(
                isinstance(term, ast.Compare)
                and set_call(term.left, "environment")
                and len(term.ops) == 1
                and isinstance(term.ops[0], ast.NotEq)
                and len(term.comparators) == 1
                and set_call(
                    term.comparators[0], "_MATLAB_ENVIRONMENT_FIELDS"
                )
                for term in ast.walk(node.test)
            )
            for node in ast.walk(matlab_descriptor)
        )
    )

    lock_fields: set[str] | None = None
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_LOCK_FIELDS"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
        ):
            continue
        value = ast.literal_eval(node.value.args[0])
        if isinstance(value, set) and all(
            isinstance(item, str) for item in value
        ):
            lock_fields = value

    current_lock = json.loads(lock_path.read_text(encoding="utf-8"))

    def current_runtime_identity(spec: object) -> bool:
        return bool(
            isinstance(spec, dict)
            and lock_fields is not None
            and set(spec) == lock_fields
            and spec.get("schema") == "ttbi-campaign-environment-v2"
            and spec.get("python") == "3.13.3"
            and spec.get("torch_cuda") == "12.8"
            and spec.get("package_inventory_policy") == "exact"
            and spec.get("package_artifact_policy")
            == "wheel-record-sha256-v1"
            and isinstance(spec.get("python_runtime"), dict)
            and spec["python_runtime"].get("soabi") == "cp313-win_amd64"
            and spec["python_runtime"].get("gil_enabled") is True
            and isinstance(spec.get("matlab_environment"), dict)
            and spec["matlab_environment"].get("release") == "R2025b"
        )

    retired = json.loads(json.dumps(current_lock))
    retired["schema"] = "ttbi-campaign-environment-v1"
    retired["python"] = "3.12.0"
    retired["torch_cuda"] = "11.8"
    retired["matlab_environment"]["release"] = "R2023b"
    retired["stages"] = ["s0_scour"]

    checks = (
        ("package mismatch hard-fails", package_guard),
        ("cuBLAS determinism mismatch hard-fails", cublas_guard),
        (
            "required CUDA becoming unavailable hard-fails",
            cuda_guard,
        ),
        ("unsupported environment-lock schema hard-fails", schema_guard),
        (
            "descriptor mutation without matching SHA hard-fails",
            descriptor_sha_guard,
        ),
        (
            "extra MATLAB descriptor field hard-fails",
            exact_matlab_fields_guard,
        ),
        (
            "static smoke binds the current v2/cp313/cu128/R2025b runtime",
            current_runtime_identity(current_lock),
        ),
        (
            "static smoke rejects retired stage/runtime fixtures",
            not current_runtime_identity(retired),
        ),
    )
    failures = 0
    print("ENVIRONMENT LOCK STATIC MUTATION SMOKE")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failures += int(not passed)
    if failures:
        print(f"ENVIRONMENT LOCK STATIC MUTATION SMOKE: {failures} FAILURE(S)")
        return 1
    print("ENVIRONMENT LOCK STATIC MUTATION SMOKE: ALL PASS")
    return 0


if _bootstrap_sys.argv[1:] == ["--static-mutation-smoke"]:
    raise SystemExit(_run_static_mutation_smoke())

import base64
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest import mock

import core.environment as environment_module
import core.environment_artifacts as artifacts_module
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
REQUIREMENTS_PATH = Path("requirements-campaign-py313-cu128.txt")
EXPECTED_EXTRA_INDEX = (
    "--extra-index-url https://download.pytorch.org/whl/cu128"
)


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


def rejects_with_message(name: str, message: str, fn) -> None:
    """Require a rejection from the intended fail-closed branch."""
    try:
        fn()
    except RuntimeError as exc:
        check(name, message in str(exc))
    else:
        check(name, False)


def make_junction(alias: Path, target: Path) -> bool:
    """Create one Windows directory junction for a behavioral fixture."""
    process = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.returncode == 0


def requirement_inventory(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse the deliberately simple, fully pinned campaign requirements."""
    packages: dict[str, str] = {}
    options: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            options.append(line)
            continue
        name, separator, version = line.partition("==")
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        if (
            separator != "=="
            or not version
            or name != canonical_name
            or canonical_name in packages
        ):
            raise RuntimeError(
                f"requirement is not one unique canonical exact pin: {line!r}"
            )
        packages[canonical_name] = version
    return packages, options


def record_sha256(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.decode("ascii").rstrip("=")


lock = load_environment_lock("environment/campaign-py313-cu128.json")
runtime = validate_environment_lock(lock)
requirements_packages, requirements_options = requirement_inventory(
    REQUIREMENTS_PATH
)
check("current campaign environment matches exact lock",
      runtime["lock_sha256"] == lock["sha256"]
      and runtime["cuda_available"])
check("requirements pin the exact complete runtime distribution inventory",
      lock["spec"]["package_inventory_policy"] == "exact"
      and len(requirements_packages) == 46
      and requirements_packages == lock["spec"]["packages"]
      and runtime["packages"] == lock["spec"]["packages"]
      and requirements_options == [EXPECTED_EXTRA_INDEX])
check("RECORD runtime bytes match and unowned non-cache files are absent",
      lock["spec"]["package_artifact_policy"]
          == "wheel-record-sha256-v1"
      and len(runtime["package_record_sha256"]) == 46
      and runtime["package_record_sha256"]
          == lock["spec"]["package_record_sha256"])
check("CPython build, ABI, GIL mode, binaries, and startup hooks are exact",
      runtime["python_runtime"] == lock["spec"]["python_runtime"]
      and runtime["python_runtime"]["implementation"] == "CPython"
      and runtime["python_runtime"]["machine"] == "AMD64"
      and runtime["python_runtime"]["architecture"] == "64bit"
      and runtime["python_runtime"]["soabi"] == "cp313-win_amd64"
      and runtime["python_runtime"]["gil_enabled"] is True
      and runtime["python_runtime"]["user_site_enabled"] is False
      and runtime["python_runtime"]["pythonpath_environment"] is None
      and runtime["python_runtime"]["pythonhome_environment"] is None
      and runtime["python_runtime"]["base_runtime_file_count"] == 2696
      and len(runtime["python_runtime"]["startup_files_sha256"]) == 3)
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
benchmark_source = Path("benchmark_paper1_compute.py").read_text(
    encoding="utf-8")
guard_source = Path("campaign_import_guard/__init__.py").read_text(
    encoding="utf-8")
executor_source = Path("training/paper1_executor.py").read_text(
    encoding="utf-8")
execution_environment_source = Path(
    "core/execution_environment.py").read_text(encoding="utf-8")
utils_source = Path("core/utils.py").read_text(encoding="utf-8")
check("entrypoints reject path injection/shadows before scientific imports",
      'for variable in ("PYTHONPATH", "PYTHONHOME")' in guard_source
      and guard_source.index('for variable in ("PYTHONPATH", "PYTHONHOME")')
          < guard_source.index("    result = validate_source_tree(")
      and benchmark_source.index("enforce_import_boundary()")
          < benchmark_source.index("    import numpy as np")
      and benchmark_source.index("enforce_import_boundary()")
          < benchmark_source.index("    import torch")
      and driver_source.index(
          'for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME")'
      ) < driver_source.index("from core.campaign_contract import")
      and driver_source.index("_enforce_import_boundary()")
          < driver_source.index("from core.campaign_contract import")
      and "import numpy" not in driver_source
      and "import torch" not in driver_source
      and "return execute_manifest_job(job, manifest)" in driver_source)
check("cuBLAS determinism is authenticated before campaign output creation",
      '"CUBLAS_WORKSPACE_CONFIG"' in execution_environment_source
      and executor_source.index("attestation = enforce_execution_block(")
          < executor_source.index("job_dir.mkdir(parents=True, exist_ok=True)")
      and utils_source.index('"CUBLAS_WORKSPACE_CONFIG", required_workspace')
          < utils_source.index("torch.cuda.manual_seed(seed)"))
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

with mock.patch.object(
    environment_module,
    "_installed_distribution_versions",
    return_value={**runtime["packages"], "unregistered-package": "1.0"},
):
    rejects(
        "unexpected installed distribution hard-fails",
        lambda: validate_environment_lock(lock),
    )

missing_distribution = dict(runtime["packages"])
missing_distribution.pop("six")
with mock.patch.object(
    environment_module,
    "_installed_distribution_versions",
    return_value=missing_distribution,
):
    rejects(
        "missing transitive distribution hard-fails",
        lambda: validate_environment_lock(lock),
    )

bad_inventory_policy = copy.deepcopy(lock)
bad_inventory_policy["spec"]["package_inventory_policy"] = "declared-only"
rejects(
    "non-exact package inventory policy hard-fails",
    lambda: validate_environment_lock(bad_inventory_policy),
)

bad_artifact_policy = copy.deepcopy(lock)
bad_artifact_policy["spec"]["package_artifact_policy"] = "versions-only"
rejects(
    "non-cryptographic package artifact policy hard-fails",
    lambda: validate_environment_lock(bad_artifact_policy),
)

bad_record_root = copy.deepcopy(lock)
bad_record_root["spec"]["package_record_sha256"]["numpy"] = "0" * 64
with mock.patch.object(
    environment_module,
    "_installed_distribution_record_roots",
    return_value=runtime["package_record_sha256"],
):
    rejects(
        "package RECORD-root mismatch hard-fails",
        lambda: validate_environment_lock(bad_record_root),
    )

missing_record_root = copy.deepcopy(lock)
missing_record_root["spec"]["package_record_sha256"].pop("six")
rejects(
    "incomplete package RECORD-root inventory hard-fails",
    lambda: validate_environment_lock(missing_record_root),
)

cuda_optional = copy.deepcopy(lock)
cuda_optional["spec"]["cuda_required"] = False
rejects(
    "CUDA cannot be made optional in the campaign lock",
    lambda: validate_environment_lock(cuda_optional),
)

cuda_integer = copy.deepcopy(lock)
cuda_integer["spec"]["cuda_required"] = 1
rejects(
    "CUDA requirement must be the Boolean true",
    lambda: validate_environment_lock(cuda_integer),
)

extra_lock_field = copy.deepcopy(lock)
extra_lock_field["spec"]["unreviewed"] = True
rejects(
    "extra environment-lock field hard-fails",
    lambda: validate_environment_lock(extra_lock_field),
)

missing_lock_field = copy.deepcopy(lock)
missing_lock_field["spec"].pop("torch_cuda")
rejects(
    "missing environment-lock field hard-fails",
    lambda: validate_environment_lock(missing_lock_field),
)

actual_arm_runtime = copy.deepcopy(runtime["python_runtime"])
actual_arm_runtime["machine"] = "ARM64"
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_arm_runtime,
):
    rejects(
        "wrong Python machine ABI hard-fails",
        lambda: validate_environment_lock(lock),
    )

actual_free_threaded_runtime = copy.deepcopy(runtime["python_runtime"])
actual_free_threaded_runtime["gil_enabled"] = False
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_free_threaded_runtime,
):
    rejects(
        "free-threaded Python hard-fails",
        lambda: validate_environment_lock(lock),
    )

actual_rebuilt_python = copy.deepcopy(runtime["python_runtime"])
actual_rebuilt_python["base_executable_sha256"] = "0" * 64
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_rebuilt_python,
):
    rejects(
        "different CPython executable build hard-fails",
        lambda: validate_environment_lock(lock),
    )

actual_changed_venv_launcher = copy.deepcopy(runtime["python_runtime"])
actual_changed_venv_launcher["venv_executable_sha256"] = "0" * 64
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_changed_venv_launcher,
):
    rejects(
        "different virtual-environment launcher hard-fails",
        lambda: validate_environment_lock(lock),
    )

actual_changed_pyvenv = copy.deepcopy(runtime["python_runtime"])
actual_changed_pyvenv["pyvenv_config_semantic_sha256"] = "0" * 64
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_changed_pyvenv,
):
    rejects(
        "different pyvenv.cfg semantics hard-fail",
        lambda: validate_environment_lock(lock),
    )

actual_changed_stdlib = copy.deepcopy(runtime["python_runtime"])
actual_changed_stdlib["base_runtime_sha256"] = "0" * 64
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_changed_stdlib,
):
    rejects(
        "different CPython standard-library/DLL tree hard-fails",
        lambda: validate_environment_lock(lock),
    )

actual_startup_injection = copy.deepcopy(runtime["python_runtime"])
actual_startup_injection["startup_files_sha256"]["injected.pth"] = "1" * 64
with mock.patch.object(
    environment_module,
    "_current_python_runtime_descriptor",
    return_value=actual_startup_injection,
):
    rejects(
        "unregistered Python startup hook hard-fails",
        lambda: validate_environment_lock(lock),
    )

with mock.patch.dict(os.environ, {"PYTHONPATH": "unreviewed"}, clear=False):
    rejects(
        "PYTHONPATH injection hard-fails",
        lambda: validate_environment_lock(lock),
    )

with mock.patch.dict(os.environ, {"PYTHONHOME": "unreviewed"}, clear=False):
    rejects(
        "PYTHONHOME injection hard-fails",
        lambda: validate_environment_lock(lock),
    )

with mock.patch.object(sys, "path", [*sys.path, "unreviewed-import-root"]):
    rejects(
        "unregistered sys.path entry hard-fails",
        lambda: validate_environment_lock(lock),
    )

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

bad_cudnn_lock = copy.deepcopy(lock)
bad_cudnn_lock["spec"]["cudnn_runtime"] = 0
rejects("nonpositive cuDNN lock value hard-fails",
        lambda: validate_environment_lock(bad_cudnn_lock))
with mock.patch(
    "torch.backends.cudnn.version",
    return_value=lock["spec"]["cudnn_runtime"] + 1,
):
    rejects("cuDNN runtime mismatch hard-fails",
            lambda: validate_environment_lock(lock))

# Exercise the required-GPU branch even on the correctly configured audit PC.
# Without this mock, deleting the guard would stay green whenever CUDA happens
# to be available on the machine running the check.
with mock.patch("torch.cuda.is_available", return_value=False):
    rejects("required CUDA becoming unavailable hard-fails",
            lambda: validate_environment_lock(lock))

with tempfile.TemporaryDirectory(prefix="env-lock-") as tmp:
    source = Path("environment/campaign-py313-cu128.json").read_text(
        encoding="utf-8")
    rejects(
        "non-regular environment-lock path hard-fails",
        lambda: load_environment_lock(tmp),
    )
    with mock.patch.object(Path, "is_symlink", return_value=True):
        rejects(
            "symlink environment-lock path hard-fails",
            lambda: load_environment_lock(
                "environment/campaign-py313-cu128.json"
            ),
        )
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

    def exercise_startup_customizer(module_name: str) -> None:
        source_root = Path(tmp, f"{module_name}-root")
        module_path = Path(
            source_root,
            "core",
            "environment_artifacts.py",
        )
        module_path.parent.mkdir(parents=True)
        module_path.write_text("# path-role fixture\n", encoding="utf-8")
        Path(source_root, f"{module_name}.py").write_text(
            "# unreviewed startup mutation\n",
            encoding="utf-8",
        )
        mutated_sys_path = [str(source_root), *sys.path[1:]]
        with (
            mock.patch.object(
                artifacts_module,
                "__file__",
                str(module_path),
            ),
            mock.patch.object(sys, "path", mutated_sys_path),
        ):
            rejects_with_message(
                f"{module_name} injection hard-fails",
                "external Python startup customizer is forbidden",
                artifacts_module.sys_path_roles,
            )

    exercise_startup_customizer("sitecustomize")
    exercise_startup_customizer("usercustomize")

    check(
        "dedicated non-vacuous import-boundary checker is present",
        Path("check_import_path_guard.py").is_file()
        and Path("campaign_import_guard", "__init__.py").is_file()
        and Path("core", "__init__.py").is_file()
        and Path("training", "__init__.py").is_file()
        and Path("TTBI_2D", "__init__.py").is_file(),
    )

    launcher_venv = Path(tmp, "launcher-venv")
    launcher_path = Path(launcher_venv, "Scripts", "python.exe")
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_bytes(b"UV-LAUNCHER-A")
    with (
        mock.patch.object(sys, "prefix", str(launcher_venv)),
        mock.patch.object(sys, "executable", str(launcher_path)),
    ):
        launcher_sha_before = artifacts_module.venv_executable_sha256()
        launcher_path.write_bytes(b"UV-LAUNCHER-B")
        launcher_sha_after = artifacts_module.venv_executable_sha256()
    check(
        "virtual-environment launcher byte mutation moves SHA-256",
        launcher_sha_before != launcher_sha_after,
    )
    with (
        mock.patch.object(sys, "prefix", str(launcher_venv)),
        mock.patch.object(
            sys,
            "executable",
            str(Path(launcher_venv, "unreviewed-python.exe")),
        ),
    ):
        rejects_with_message(
            "noncanonical virtual-environment launcher path hard-fails",
            "not the uv environment launcher",
            artifacts_module.venv_executable_sha256,
        )

    config_base = Path(tmp, "config-base")
    config_venv = Path(tmp, "config-venv")
    config_base.mkdir()
    config_venv.mkdir()
    config_path = Path(config_venv, "pyvenv.cfg")

    def write_pyvenv_config(
        *,
        target: Path = config_path,
        home: Path = config_base,
        version: str = "3.13.3",
        uv_version: str = "0.8.22",
        compact: bool = False,
    ) -> None:
        separator = "=" if compact else " = "
        entries = (
            ("home", str(home)),
            ("implementation", "CPython"),
            ("uv", uv_version),
            ("version_info", version),
            ("include-system-site-packages", "false"),
            ("seed", "true"),
        )
        target.write_text(
            "".join(
                f"{key}{separator}{value}\n"
                for key, value in entries
            ),
            encoding="utf-8",
        )

    with (
        mock.patch.object(sys, "prefix", str(config_venv)),
        mock.patch.object(sys, "base_prefix", str(config_base)),
        mock.patch.object(
            artifacts_module.platform,
            "python_version",
            return_value="3.13.3",
        ),
    ):
        write_pyvenv_config()
        _values, config_sha_spaced = (
            artifacts_module.parse_pyvenv_config()
        )
        write_pyvenv_config(compact=True)
        _values, config_sha_compact = (
            artifacts_module.parse_pyvenv_config()
        )
        check(
            "semantic-equivalent pyvenv.cfg bytes share canonical SHA-256",
            config_sha_spaced == config_sha_compact,
        )
        write_pyvenv_config(uv_version="0.8.23")
        _values, config_sha_other_uv = (
            artifacts_module.parse_pyvenv_config()
        )
        check(
            "pyvenv.cfg semantic mutation moves canonical SHA-256",
            config_sha_other_uv != config_sha_spaced,
        )
        write_pyvenv_config(version="3.13.2")
        rejects_with_message(
            "pyvenv.cfg version_info mismatch hard-fails",
            "version_info differs",
            artifacts_module.parse_pyvenv_config,
        )
        write_pyvenv_config(home=Path(tmp, "wrong-config-home"))
        rejects_with_message(
            "pyvenv.cfg home mismatch hard-fails",
            "home differs",
            artifacts_module.parse_pyvenv_config,
        )

    relocated_base = Path(tmp, "relocated-config-base")
    relocated_venv = Path(tmp, "relocated-config-venv")
    relocated_base.mkdir()
    relocated_venv.mkdir()
    relocated_config = Path(relocated_venv, "pyvenv.cfg")
    write_pyvenv_config(
        target=relocated_config,
        home=relocated_base,
        compact=True,
    )
    with (
        mock.patch.object(sys, "prefix", str(relocated_venv)),
        mock.patch.object(sys, "base_prefix", str(relocated_base)),
        mock.patch.object(
            artifacts_module.platform,
            "python_version",
            return_value="3.13.3",
        ),
    ):
        _values, relocated_config_sha = (
            artifacts_module.parse_pyvenv_config()
        )
    check(
        "pyvenv.cfg relocation preserves canonical semantic SHA-256",
        relocated_config_sha == config_sha_spaced,
    )

    base_fixture = Path(tmp, "base-runtime")
    Path(base_fixture, "Lib").mkdir(parents=True)
    Path(base_fixture, "DLLs").mkdir()
    stdlib_fixture = Path(base_fixture, "Lib", "science.py")
    native_fixture = Path(base_fixture, "DLLs", "science.pyd")
    stdlib_fixture.write_bytes(b"CONSTANT = 1\n")
    native_fixture.write_bytes(b"NATIVE-FIXTURE")
    base_root_before, base_count = (
        environment_module._python_base_runtime_root(base_fixture)
    )
    cache_dir = Path(base_fixture, "Lib", "__pycache__")
    cache_dir.mkdir()
    Path(cache_dir, "science.cpython-313.pyc").write_bytes(b"MUTABLE")
    base_root_with_cache, cached_count = (
        environment_module._python_base_runtime_root(base_fixture)
    )
    stdlib_fixture.write_bytes(b"CONSTANT = 2\n")
    base_root_after, changed_count = (
        environment_module._python_base_runtime_root(base_fixture)
    )
    check(
        "CPython base root covers stdlib/native bytes but excludes caches",
        base_count == cached_count == changed_count == 2
        and base_root_before == base_root_with_cache
        and base_root_after != base_root_before,
    )
    base_junction_target = Path(tmp, "base-junction-target")
    base_junction_target.mkdir()
    Path(base_junction_target, "aliased.py").write_bytes(b"ALIASED = True\n")
    base_junction = Path(base_fixture, "Lib", "joined-runtime")
    created = make_junction(base_junction, base_junction_target)
    try:
        is_real_junction = (
            created
            and not base_junction.is_symlink()
            and bool(
                getattr(base_junction, "is_junction", lambda: False)()
            )
        )
        check(
            "CPython base reparse fixture is a real junction",
            is_real_junction,
        )
        rejects_with_message(
            "internal CPython base junction hard-fails",
            "reparse/aliased path",
            lambda: environment_module._python_base_runtime_root(
                base_fixture
            ),
        )
    finally:
        if base_junction.exists():
            os.rmdir(base_junction)

    rejects(
        "unsafe wheel RECORD parent traversal hard-fails",
        lambda: environment_module._normalise_record_path(
            "../../../outside.py"
        ),
    )
    check(
        "standard wheel data relocation remains explicit",
        environment_module._normalise_record_path(
            "../../share/man/man1/tool.1"
        ) == "../../share/man/man1/tool.1",
    )

    fixture_root = Path(tmp, "wheel-fixture")
    package_dir = Path(fixture_root, "fixture_pkg")
    dist_info = Path(fixture_root, "fixture_pkg-1.0.dist-info")
    package_dir.mkdir(parents=True)
    dist_info.mkdir()
    payload_path = Path(package_dir, "__init__.py")
    metadata_path = Path(dist_info, "METADATA")
    record_path = Path(dist_info, "RECORD")
    original_payload = b"SCIENTIFIC_CONSTANT = 1\n"
    changed_payload = b"SCIENTIFIC_CONSTANT = 2\n"
    metadata_payload = (
        b"Metadata-Version: 2.1\nName: fixture-pkg\nVersion: 1.0\n"
    )
    payload_path.write_bytes(original_payload)
    metadata_path.write_bytes(metadata_payload)

    def write_fixture_record(payload: bytes) -> None:
        rows = [
            (
                "fixture_pkg/__init__.py",
                record_sha256(payload),
                str(len(payload)),
            ),
            (
                "fixture_pkg-1.0.dist-info/METADATA",
                record_sha256(metadata_payload),
                str(len(metadata_payload)),
            ),
            ("fixture_pkg-1.0.dist-info/RECORD", "", ""),
        ]
        record_path.write_text(
            "".join(",".join(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_fixture_record(original_payload)
    fixture_distribution = metadata.PathDistribution(
        dist_info
    )
    fixture_root_sha = environment_module._distribution_record_root(
        fixture_distribution,
        verify_files=True,
        allowed_prefix=fixture_root,
    )
    check(
        "wheel RECORD fixture authenticates its installed bytes",
        len(fixture_root_sha) == 64,
    )
    payload_path.write_bytes(changed_payload)
    rejects(
        "same-version installed-file mutation hard-fails",
        lambda: environment_module._distribution_record_root(
            metadata.PathDistribution(dist_info),
            verify_files=True,
            allowed_prefix=fixture_root,
        ),
    )
    write_fixture_record(changed_payload)
    changed_root_sha = environment_module._distribution_record_root(
        metadata.PathDistribution(dist_info),
        verify_files=True,
        allowed_prefix=fixture_root,
    )
    check(
        "coherent same-version wheel replacement moves authenticated root",
        changed_root_sha != fixture_root_sha,
    )

    inventory_fixture = Path(tmp, "package-inventory")
    owned_file = Path(inventory_fixture, "owned_pkg", "__init__.py")
    installer_file = Path(
        inventory_fixture,
        "owned_pkg-1.0.dist-info",
        "RECORD",
    )
    owned_file.parent.mkdir(parents=True)
    installer_file.parent.mkdir()
    owned_file.write_bytes(b"OWNED = True\n")
    installer_file.write_bytes(b"installer fixture\n")
    cache_file = Path(
        inventory_fixture,
        "owned_pkg",
        "__pycache__",
        "__init__.cpython-313.pyc",
    )
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"DERIVED CACHE")
    inventory_count = artifacts_module.validate_site_package_inventory(
        {owned_file.resolve()},
        {installer_file.resolve()},
        set(),
        library_roots=(inventory_fixture,),
    )
    check(
        "exact package-directory fixture accepts only declared files",
        inventory_count == 2,
    )
    Path(inventory_fixture, "shadow_module.py").write_bytes(
        b"UNOWNED = True\n"
    )
    rejects_with_message(
        "unowned package-directory module hard-fails",
        "unowned file in Python package directory",
        lambda: artifacts_module.validate_site_package_inventory(
            {owned_file.resolve()},
            {installer_file.resolve()},
            set(),
            library_roots=(inventory_fixture,),
        ),
    )
    junction_inventory = Path(tmp, "junction-package-inventory")
    junction_inventory.mkdir()
    declared_file = Path(junction_inventory, "declared.py")
    declared_file.write_bytes(b"DECLARED = True\n")
    package_junction_target = Path(tmp, "package-junction-target")
    package_junction_target.mkdir()
    aliased_package_file = Path(package_junction_target, "__init__.py")
    aliased_package_file.write_bytes(b"ALIASED_PACKAGE = True\n")
    package_junction = Path(junction_inventory, "joined_pkg")
    created = make_junction(package_junction, package_junction_target)
    try:
        is_real_junction = (
            created
            and not package_junction.is_symlink()
            and bool(
                getattr(package_junction, "is_junction", lambda: False)()
            )
        )
        check(
            "package-inventory reparse fixture is a real junction",
            is_real_junction,
        )
        rejects_with_message(
            "internal package-directory junction hard-fails",
            "reparse/aliased path",
            lambda: artifacts_module.validate_site_package_inventory(
                {
                    declared_file.resolve(),
                    aliased_package_file.resolve(),
                },
                set(),
                set(),
                library_roots=(junction_inventory,),
            ),
        )
    finally:
        if package_junction.exists():
            os.rmdir(package_junction)

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

    mutated_requirements = Path(tmp, "requirements.txt")
    mutated_requirements.write_text(
        REQUIREMENTS_PATH.read_text(encoding="utf-8").replace(
            "six==1.17.0",
            "six==1.16.0",
            1,
        ),
        encoding="utf-8",
    )
    mutated_packages, mutated_options = requirement_inventory(
        mutated_requirements
    )
    check(
        "requirements drift is detected against the environment inventory",
        mutated_packages != lock["spec"]["packages"]
        and mutated_options == [EXPECTED_EXTRA_INDEX],
    )

if fails:
    raise SystemExit(f"ENVIRONMENT LOCK: {fails} FAILURE(S)")
print("ENVIRONMENT LOCK: ALL PASS")
