"""Retirement boundary for the archived R11 compute benchmark.

The R11 workload remains named and frozen as R11.  It is not Paper-1 dispatch
evidence once the live generation schema moves on; the genuine Paper-1 v2
benchmark and its own contract checker are the only dispatch-bound lineage.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import benchmark_paper1_compute as paper1_benchmark
import benchmark_r5_compute as legacy_benchmark
from core.campaign_contract import EXPECTED_GEN_SCHEMA
import dispatch_manifest


ROOT = Path(__file__).resolve().parent
LEGACY_PATH = ROOT / "benchmark_r5_compute.py"
PAPER1_PATH = ROOT / "benchmark_paper1_compute.py"
PAPER1_CHECKER = ROOT / "check_paper1_benchmark_contract.py"
DISPATCH_AUTHORIZATION_PATH = ROOT / "dispatch_authorization.py"
FROZEN_R11_GEN_SCHEMA = "audit-2026-07-27-r11"

FAILURES = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    passed = bool(condition)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          + (f" - {detail}" if detail else ""))
    FAILURES += int(not passed)


def retirement_source_contract(source: str) -> bool:
    """Recognize the minimal immutable-R11/fail-first source boundary."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    assignments = {
        target.id: node.value
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = {"_assert_legacy_r11_schema_boundary", "main"}
    if not required <= set(functions):
        return False
    frozen = assignments.get("FROZEN_R11_GEN_SCHEMA")
    benchmark_schema = assignments.get("BENCHMARK_SCHEMA")
    main = functions["main"]
    boundary = functions["_assert_legacy_r11_schema_boundary"]
    if not main.body:
        return False
    first = main.body[0]
    first_is_boundary = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "_assert_legacy_r11_schema_boundary"
        and not first.value.args
        and not first.value.keywords
    )
    boundary_source = ast.get_source_segment(source, boundary) or ""
    return (
        isinstance(frozen, ast.Constant)
        and frozen.value == FROZEN_R11_GEN_SCHEMA
        and isinstance(benchmark_schema, ast.Constant)
        and benchmark_schema.value == "ttbi-r11-compute-benchmark-v1"
        and first_is_boundary
        and "EXPECTED_GEN_SCHEMA != FROZEN_R11_GEN_SCHEMA" in boundary_source
        and "benchmark_paper1_compute.py" in boundary_source
        and all(token not in boundary_source for token in (
            ".mkdir(", ".open(", ".write_", "os.open(", "Path(",
        ))
    )


print("LEGACY R11 BENCHMARK RETIREMENT CHECKS")
legacy_source = LEGACY_PATH.read_text(encoding="utf-8")
check(
    "legacy benchmark remains explicitly frozen and named R11",
    legacy_benchmark.FROZEN_R11_GEN_SCHEMA == FROZEN_R11_GEN_SCHEMA
    and legacy_benchmark.BENCHMARK_SCHEMA
    == "ttbi-r11-compute-benchmark-v1"
    and retirement_source_contract(legacy_source),
)
check(
    "live campaign has moved beyond the frozen R11 generation schema",
    EXPECTED_GEN_SCHEMA != FROZEN_R11_GEN_SCHEMA,
    f"live={EXPECTED_GEN_SCHEMA!r}, frozen={FROZEN_R11_GEN_SCHEMA!r}",
)

try:
    legacy_benchmark._assert_legacy_r11_schema_boundary()
except legacy_benchmark.ContractError as exc:
    boundary_message = str(exc)
    check(
        "live schema mismatch raises the explicit retirement error",
        "legacy R11 compute benchmark is retired" in boundary_message
        and repr(EXPECTED_GEN_SCHEMA) in boundary_message
        and repr(FROZEN_R11_GEN_SCHEMA) in boundary_message
        and "benchmark_paper1_compute.py" in boundary_message,
        boundary_message,
    )
except Exception as exc:  # noqa: BLE001 - diagnostic must name wrong failure
    check(
        "live schema mismatch raises the explicit retirement error",
        False,
        f"unexpected {type(exc).__name__}: {exc}",
    )
else:
    check(
        "live schema mismatch raises the explicit retirement error",
        False,
        "legacy boundary accepted the live schema",
    )

with tempfile.TemporaryDirectory(prefix="r11-retirement-check-") as temp:
    temp_path = Path(temp)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    process = subprocess.run(
        [sys.executable, "-B", str(LEGACY_PATH)],
        cwd=temp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = process.stdout + process.stderr
    check(
        "legacy CLI fails at the retirement boundary before filesystem writes",
        process.returncode != 0
        and "legacy R11 compute benchmark is retired" in combined
        and list(temp_path.iterdir()) == [],
        f"returncode={process.returncode}",
    )

check(
    "mutation guard rejects removing the fail-first boundary",
    not retirement_source_contract(legacy_source.replace(
        "    _assert_legacy_r11_schema_boundary()\n"
        "    args = _parser().parse_args(argv)",
        "    args = _parser().parse_args(argv)",
        1,
    )),
)
check(
    "mutation guard rejects relabelling the archived benchmark",
    not retirement_source_contract(legacy_source.replace(
        'BENCHMARK_SCHEMA = "ttbi-r11-compute-benchmark-v1"',
        'BENCHMARK_SCHEMA = "ttbi-paper1-compute-benchmark-v2"',
        1,
    )),
)

dispatch_sources = set(dispatch_manifest.POLICY_SOURCE_FILES)
dispatch_authorization_source = DISPATCH_AUTHORIZATION_PATH.read_text(
    encoding="utf-8"
)
check(
    "Paper-1 dispatch requires genuine benchmark authorization schema v2",
    paper1_benchmark.BENCHMARK_SCHEMA
    == "ttbi-paper1-compute-benchmark-v2"
    and paper1_benchmark.AUTHORIZATION_SCHEMA
    == "ttbi-paper1-benchmark-authorization-evidence-v2"
    and dispatch_manifest.REQUIRED_BENCHMARK_SCHEMA
    == paper1_benchmark.AUTHORIZATION_SCHEMA,
)
check(
    "dispatch invokes the genuine Paper-1 benchmark revalidator",
    "import benchmark_paper1_compute as benchmark"
    in dispatch_authorization_source
    and "import benchmark_r5_compute as benchmark"
    not in dispatch_authorization_source,
)
check(
    "dispatch policy binds the Paper-1 benchmark checker, not legacy R11",
    PAPER1_PATH.is_file()
    and PAPER1_CHECKER.is_file()
    and "benchmark_paper1_compute.py" in dispatch_sources
    and "check_paper1_benchmark_contract.py" in dispatch_sources
    and "benchmark_r5_compute.py" not in dispatch_sources
    and "check_benchmark_contract.py" not in dispatch_sources,
)

if FAILURES:
    raise SystemExit(
        f"LEGACY R11 BENCHMARK RETIREMENT: {FAILURES} CHECK(S) FAILED"
    )
print("LEGACY R11 BENCHMARK RETIREMENT: ALL PASS")
