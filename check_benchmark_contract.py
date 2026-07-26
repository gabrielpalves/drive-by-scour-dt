"""Lightweight contract checks for ``benchmark_r5_compute.py``.

The checker uses only the Python standard library.  It parses the benchmark
AST and exercises its standard-library safety helpers against temporary
fixtures.  It does not import NumPy, Torch, Optuna, the training stack, or a
generated dataset, and it never launches the heavy benchmark.

Run:

    python check_benchmark_contract.py
"""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark_r5_compute.py"
SOURCE = BENCHMARK.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(BENCHMARK))
FAILURES = 0


def check(name: str, condition: bool) -> None:
    global FAILURES
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        FAILURES += 1


def expect_raises(name: str, exception_type: type[BaseException], call) -> None:
    try:
        call()
    except exception_type:
        check(name, True)
    except Exception as exc:
        print(
            f"  [FAIL] {name}: raised {type(exc).__name__}, "
            f"expected {exception_type.__name__}"
        )
        global FAILURES
        FAILURES += 1
    else:
        check(name, False)


def literal_assignments() -> dict[str, object]:
    result = {}
    for node in TREE.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return result


def functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def call_attributes(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return names


print("R5 COMPUTE BENCHMARK CONTRACT CHECKS")
assignments = literal_assignments()
funcs = functions()

# 1. The fixed workload and compute budget are source-level constants.  There
# are deliberately no CLI flags for smaller or larger scientific-looking runs.
expected_literals = {
    "DISCLAIMER": "NON-SCIENTIFIC WORKLOAD FIXTURE",
    "DEFAULT_X_SHAPE": (12950, 2, 512),
    "DEFAULT_Y_SHAPE": (12950, 2),
    "N_STATES": 259,
    "PASSAGES_PER_STATE": 50,
    "DOFS": (2, 5),
    "TARGET_SUPPORTS": (2, 3),
    "SEED": 42,
    "USEFUL_TRIALS": 100,
    "EPOCHS": 50,
    "USE_PRUNER": True,
    "MAX_FAIL_SLACK": 20,
    "ACTIVE_WALL_HEARTBEAT_SECONDS": 5.0,
    "AUTHENTICATED_JSON_READ_ATTEMPTS": 30,
    "LOCK_UNREADABLE_STALE_SECONDS": 300,
    "FINALIST_N_SPLITS": 5,
    "FINALIST_N_REPEATS": 2,
    "FINALIST_SPLIT_SEED": 271828,
}
for name, expected in expected_literals.items():
    check(f"{name} is pinned to {expected!r}", assignments.get(name) == expected)

expected_source_files = (
    "benchmark_r5_compute.py",
    "check_benchmark_contract.py",
    "comprehensive_ablation_multidamage.py",
    "training/trainer.py",
    "training/pipeline.py",
    "training/robustness.py",
    "plotting/confusion.py",
    "plotting/robustness_plots.py",
    "core/dataset.py",
    "core/models.py",
    "core/preprocessing.py",
    "core/task.py",
    "core/utils.py",
    "core/statistical_inference.py",
    "core/protocol.py",
    "core/environment.py",
    "environment/campaign-py313-cu128.json",
)
check(
    "source hash closure names every direct and transitive runtime module",
    assignments.get("SOURCE_FILES") == expected_source_files,
)
expected_finalist_fields = (
    "schema",
    "classification",
    "status",
    "descriptor_sha256",
    "helper",
    "selected_trial_number",
    "selected_parameter_sha256",
    "frozen_checkpoint_epoch_count",
    "durably_accepted_refits",
    "execution_semantics",
    "attempt_count",
    "prior_unaccepted_attempt_count",
    "repeat",
    "fold",
    "n_splits",
    "n_repeats",
    "split_seed",
    "train_state_count",
    "validation_state_count",
    "train_sample_count",
    "validation_sample_count",
    "train_states_sha256",
    "validation_states_sha256",
    "scale_train_infer_seconds",
    "active_wall_seconds_cumulative",
    "active_wall_checkpoint_interval_seconds",
    "active_wall_semantics",
    "timing_complete",
    "nominal_unrecorded_tail_seconds_per_abrupt_stop",
    "unrecorded_tail_bound",
    "memory",
    "memory_scope",
    "memory_complete",
    "returned_values_finite",
    "returned_values_discarded",
    "completed_utc",
)
check(
    "finalist durable receipt has an exact reviewer-visible schema",
    assignments.get("FINALIST_REPORT_FIELDS") == expected_finalist_fields,
)

check(
    "no CLI budget/epoch override exists",
    all(flag not in SOURCE for flag in ("--trials", "--epochs", "--n-trials")),
)
check(
    "output root is pinned below .audit_tmp",
    ".audit_tmp/r5_compute_benchmark" in SOURCE,
)
check(
    "Optuna study name is unmistakably non-publishable",
    assignments.get("STUDY_NAME_PREFIX")
    == "BENCHMARK_ONLY_DO_NOT_PUBLISH__",
)

# 2. No heavyweight import is allowed at module scope.  This keeps this checker
# and ``--help`` independent of GPU/runtime availability.
heavy_roots = {
    "numpy", "torch", "optuna", "training", "core", "sklearn", "matplotlib"
}
module_import_roots = set()
for node in TREE.body:
    if isinstance(node, ast.Import):
        module_import_roots.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        module_import_roots.add(node.module.split(".")[0])
check(
    "heavy runtime has no module-scope imports",
    not (module_import_roots & heavy_roots),
)

# 3. The fixture loader must use read-only memmaps for both arrays, with pickle
# disabled.  Shape/order and sampled/full finiteness guards must be in the same
# live function.
loader = funcs.get("_load_and_validate_fixture")
load_calls = []
if loader is not None:
    for node in ast.walk(loader):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ):
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            load_calls.append(keywords)
check("_load_and_validate_fixture exists", loader is not None)
check("both fixture arrays use np.load", len(load_calls) == 2)
check(
    "both fixture arrays use mmap_mode='r'",
    len(load_calls) == 2
    and all(
        isinstance(keywords.get("mmap_mode"), ast.Constant)
        and keywords["mmap_mode"].value == "r"
        for keywords in load_calls
    ),
)
check(
    "both fixture loads disable pickle",
    len(load_calls) == 2
    and all(
        isinstance(keywords.get("allow_pickle"), ast.Constant)
        and keywords["allow_pickle"].value is False
        for keywords in load_calls
    ),
)
loader_source = ast.get_source_segment(SOURCE, loader) if loader else ""
for evidence in (
    "sampled_finite",
    "full_finite",
    "block_order_ok",
    "N_STATES * PASSAGES_PER_STATE",
):
    check(f"fixture loader enforces {evidence}", evidence in loader_source)

# 4. The context manager may assign only the two trainer entry points named by
# the design, and must restore both in ``finally``.
patcher = funcs.get("_patched_trainer_fixture")
patched_attributes = set()
if patcher is not None:
    for node in ast.walk(patcher):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "trainer_module"
            ):
                patched_attributes.add(target.attr)
decorators = {
    decorator.attr if isinstance(decorator, ast.Attribute)
    else decorator.id if isinstance(decorator, ast.Name)
    else ""
    for decorator in (patcher.decorator_list if patcher else [])
}
check("_patched_trainer_fixture is a context manager", "contextmanager" in decorators)
check(
    "trainer adapter patches exactly two entry points",
    patched_attributes
    == {"get_or_create_cache", "canonical_train_val_split"},
)
patcher_source = ast.get_source_segment(SOURCE, patcher) if patcher else ""
check(
    "trainer adapter restores get_or_create_cache",
    "trainer_module.get_or_create_cache = original_get_cache" in patcher_source,
)
check(
    "trainer adapter restores canonical_train_val_split",
    "trainer_module.canonical_train_val_split = original_split" in patcher_source,
)

# 5. Production API and useful-budget semantics must remain directly visible.
all_calls = call_attributes(TREE)
for production_call in (
    "_create_or_resume_study",
    "_stamp_study_protocol",
    "Objective",
    "optimize",
    "repeated_stratified_group_folds",
    "frozen_checkpoint_epoch_count",
    "fit_predict_finalist_fold",
):
    check(
        f"production call is present: {production_call}",
        production_call in all_calls or production_call in SOURCE,
    )
optimize_source = ast.get_source_segment(
    SOURCE, funcs["_optimize_useful_budget"]
)
check(
    "useful budget counts COMPLETE plus PRUNED",
    'state in (trial_state.COMPLETE, trial_state.PRUNED)' in SOURCE,
)
check(
    "retry loop uses useful deficit and total failure slack",
    "USEFUL_TRIALS - counts[\"useful\"]" in optimize_source
    and "(USEFUL_TRIALS + MAX_FAIL_SLACK) - counts[\"total\"]"
    in optimize_source,
)
check(
    "Optuna progress bar cannot print scientific values",
    "show_progress_bar=False" in optimize_source,
)
check(
    "production best_value is never read or reported",
    not any(
        isinstance(node, ast.Attribute) and node.attr == "best_value"
        for node in ast.walk(TREE)
    ),
)
check(
    "no production best callback is used",
    "print_best_callback" not in SOURCE,
)
optimize_node = funcs["_optimize_useful_budget"]
check(
    "HPO recovery acknowledgement is a required keyword-only argument",
    [argument.arg for argument in optimize_node.args.kwonlyargs]
    == ["recover_stale"]
    and len(optimize_node.args.kw_defaults) == 1
    and optimize_node.args.kw_defaults[0] is None,
)
main_source = ast.get_source_segment(SOURCE, funcs["main"])
main_hpo_calls = [
    node for node in ast.walk(funcs["main"])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_useful_budget"
    )
]
check(
    "main passes explicit stale-recovery authority into HPO",
    len(main_hpo_calls) == 1
    and any(
        keyword.arg == "recover_stale"
        and isinstance(keyword.value, ast.Attribute)
        and keyword.value.attr == "recover_stale"
        for keyword in main_hpo_calls[0].keywords
    ),
)
check(
    "benchmark refuses dirty tracked runtime source",
    "if git_tracked_dirty_at_start:" in main_source
    and "Commit the converged code before" in main_source,
)
tracked_guard_source = ast.get_source_segment(
    SOURCE, funcs["_assert_sources_tracked_at_head"])
check(
    "every benchmark runtime source must resolve to its HEAD blob",
    "rev-parse" in tracked_guard_source
    and "hash-object" in tracked_guard_source
    and "_assert_sources_tracked_at_head(repo)" in main_source,
)
check(
    "cuBLAS lock is bootstrapped before the first heavy-stack import",
    0 <= main_source.find("_bootstrap_cublas_environment(repo)")
    < main_source.find("_load_runtime(descriptor, descriptor_sha256)"),
)
check(
    "a completed receipt is authenticated before any study/runtime rewrite",
    0 <= main_source.find("_completed_summary_if_valid(")
    < main_source.find("_configure_runtime_outputs(run_dir)")
    < main_source.find("_load_runtime(descriptor, descriptor_sha256)")
    and "was preserved byte-for-byte" in main_source,
)
check(
    "CUDA JIT cache is confined below the benchmark run",
    '"CUDA_CACHE_PATH"' in SOURCE
    and 'str(cache_root / "cuda")' in SOURCE,
)
check(
    "benchmark records start/end GPU telemetry",
    "gpu_telemetry_this_invocation_start" in SOURCE
    and "gpu_telemetry_this_invocation_end" in SOURCE,
)
check(
    "benchmark persists cumulative active wall time and timing quantiles",
    "hpo_active_wall_seconds_cumulative" in SOURCE
    and "useful_trial_duration_seconds_quantiles" in SOURCE
    and "active_compute_seconds_cumulative" in SOURCE,
)
check(
    "abrupt-stop timing is an honest heartbeat lower bound and excludes stale FAIL duration",
    "_ActiveWallHeartbeat" in optimize_source
    and "active_wall_heartbeat.json" in optimize_source
    and 'row["state"] in ("COMPLETE", "PRUNED")' in optimize_source
    and "failed_trial_durations_excluded" in optimize_source
    and "not a strict tail bound" in SOURCE
    and "unrecorded_tail_bound" in optimize_source,
)
check(
    "HPO interruption and Optuna recovery histories are cumulative",
    "_merge_hpo_interruption_history" in optimize_source
    and "optuna_recovery_events" in optimize_source
    and "all_stale_trial_numbers_recovered" in optimize_source,
)
check(
    "HPO live peak snapshots survive a torn final publication",
    "memory_monitor.snapshot()" in optimize_source
    and "_merge_memory_receipts" in optimize_source,
)
check(
    "completed HPO receipt is hashed and reused before Objective construction",
    "_canonical_sha256(stored_report)" in optimize_source
    and '"report_sha256": report_sha256' in optimize_source
    and 0 <= optimize_source.find(
        'return {"study": study, "report": stored_report}')
    < optimize_source.find("trainer.Objective"),
)
check(
    "completed HPO receipt authenticates trial CSV bytes and logical rows",
    '"trial_compute_sha256": trial_compute_sha256' in optimize_source
    and "_authenticated_file_sha256" in optimize_source
    and "_csv_rows_sha256" in optimize_source,
)
check(
    "Optuna INFO output is suppressed while objective values exist",
    "_suppress_optuna_value_logging(optuna)" in optimize_source,
)
finalist_runner = funcs["_run_finalist_once"]
finalist_calls = [
    node for node in ast.walk(finalist_runner)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit_predict_finalist_fold"
    )
]
finalist_keywords = (
    {keyword.arg for keyword in finalist_calls[0].keywords}
    if len(finalist_calls) == 1 else set()
)
check(
    "shared finalist helper receives explicit production horizon/head count",
    len(finalist_calls) == 1
    and {"n_epochs", "max_epochs", "n_scour_heads"} <= finalist_keywords,
)
main_finalist_calls = [
    node for node in ast.walk(funcs["main"])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_finalist_once"
    )
]
check(
    "main passes explicit stale-recovery authority into the finalist refit",
    len(main_finalist_calls) == 1
    and any(
        keyword.arg == "recover_stale"
        for keyword in main_finalist_calls[0].keywords
    ),
)
finalist_source = ast.get_source_segment(SOURCE, finalist_runner)
check(
    "finalist receipt claims durable acceptance rather than impossible exactly-once execution",
    "durably_accepted_refits" in finalist_source
    and "prior_unaccepted_attempt_count" in finalist_source
    and "may have physically completed" in finalist_source
    and "exactly one completed refit" not in finalist_source,
)
check(
    "published finalist marker authenticates and repairs only its torn attempt pointer",
    "_load_active_wall_checkpoint" in finalist_source
    and "finalist marker and attempt counters contradict" in finalist_source
    and "torn attempt pointer" in finalist_source
    and "_atomic_json(attempt_path" in finalist_source,
)
completed_receipt_source = ast.get_source_segment(
    SOURCE, funcs["_completed_summary_if_valid"])
check(
    "completed summary authenticates every announced immutable evidence artifact",
    "_immutable_evidence_snapshot(run_dir)" in completed_receipt_source
    and "_completion_pointer_history_valid" in completed_receipt_source,
)
check(
    "SQLite completion identity uses an immutable logical backup, not WAL sidecars",
    "_materialize_study_storage_receipt" in main_source
    and "study_receipt.sqlite3" in SOURCE
    and "study.sqlite3-wal" not in ast.get_source_segment(
        SOURCE, funcs["_storage_snapshot"]),
)
check(
    "recovered summary labels invocation-only wall clock and environment scope",
    "wall_seconds_before_summary_publication" in main_source
    and "active_compute_seconds_cumulative" in main_source
    and "environment_scope" in SOURCE,
)
lock_class = next(
    node for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "_ExclusivePidLock"
)
lock_class_source = ast.get_source_segment(SOURCE, lock_class)
check(
    "same-host lock identity detects PID reuse across reboot/process creation",
    "_process_identity_token" in lock_class_source
    and "process_identity" in lock_class_source,
)
pid_lock_enter = next(
    (
        node for node in lock_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__enter__"
    ),
    None,
)
pid_lock_enter_source = (
    ast.get_source_segment(SOURCE, pid_lock_enter)
    if pid_lock_enter is not None else ""
)
check(
    "PID-lock classify, archive and O_EXCL publish share the OS mutex",
    "with _exclusive_file_mutex(self.guard_path):"
    in pid_lock_enter_source
    and pid_lock_enter_source.find("_classify_existing()")
    < pid_lock_enter_source.find("_archive_existing")
    < pid_lock_enter_source.find("os.open"),
)

# 6. Publications must be atomic, guarded against scientific field names, and
# stale recovery must preserve evidence instead of removing an output tree.
replace_helper = funcs["_atomic_replace"]
check(
    "_atomic_replace owns the bounded os.replace retry",
    "replace" in call_attributes(replace_helper)
    and "attempts = 30" in ast.get_source_segment(SOURCE, replace_helper),
)
json_reader = funcs["_read_json_mapping"]
json_reader_source = ast.get_source_segment(SOURCE, json_reader)
check(
    "authenticated JSON reader owns bounded transient retries",
    "AUTHENTICATED_JSON_READ_ATTEMPTS" in json_reader_source
    and "PermissionError" in json_reader_source
    and "JSONDecodeError" not in json_reader_source
    and "parse_constant" in json_reader_source
    and "sleep" in call_attributes(json_reader),
)
check(
    "HPO and active-wall receipts use the authenticated JSON reader",
    "_read_json_mapping" in call_attributes(optimize_node)
    and "_read_json_mapping"
    in call_attributes(funcs["_load_active_wall_checkpoint"]),
)
for writer_name in ("_atomic_json", "_atomic_csv"):
    writer = funcs[writer_name]
    calls = call_attributes(writer)
    check(f"{writer_name} uses the atomic replace helper",
          "_atomic_replace" in calls)
    check(
        f"{writer_name} applies report-field guard",
        "_assert_no_scientific_report_fields" in calls,
    )
check("no recursive output deletion exists", "rmtree" not in all_calls)
check("no raw os.remove call exists", ".remove(" not in SOURCE)
check(
    "stale recovery requires explicit CLI switch",
    "--recover-stale" in SOURCE and "recover_stale" in SOURCE,
)

# Import only the benchmark's standard-library surface for temporary-fixture
# tests.  Suppress bytecode so this checker itself changes no repository file.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("r5_benchmark_contract_module", BENCHMARK)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load benchmark module spec: {BENCHMARK}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

naive_local = datetime(2026, 7, 25, 12, 34, 56, 123456)
naive_utc_text = module._datetime_as_utc_text(naive_local)
naive_utc = datetime.fromisoformat(naive_utc_text)
check(
    "naive Optuna local timestamp is converted to aware UTC",
    naive_utc.utcoffset() == timedelta(0)
    and abs(naive_utc.timestamp() - naive_local.timestamp()) < 1e-6,
)
aware_offset = datetime(
    2026, 7, 25, 12, 34, 56,
    tzinfo=timezone(timedelta(hours=-7)),
)
check(
    "aware Optuna timestamp is normalized to UTC",
    module._datetime_as_utc_text(aware_offset)
    == "2026-07-25T19:34:56+00:00",
)
check(
    "missing Optuna timestamp stays explicitly empty",
    module._datetime_as_utc_text(None) == "",
)
expect_raises(
    "non-datetime Optuna timestamp is rejected",
    module.ContractError,
    lambda: module._datetime_as_utc_text("2026-07-25"),
)

checker_tmp_root = ROOT / ".audit_tmp" / "check_benchmark_contract"
checker_tmp_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(
    prefix="r5-benchmark-contract-",
    dir=checker_tmp_root,
) as tmp_text:
    tmp = Path(tmp_text)
    check(
        "checker temporary tree stays below repository .audit_tmp",
        tmp.resolve().is_relative_to((ROOT / ".audit_tmp").resolve()),
    )

    lock_repo = tmp / "bootstrap-repo"
    lock_dir = lock_repo / "environment"
    lock_dir.mkdir(parents=True)
    (lock_dir / "campaign-py313-cu128.json").write_text(
        json.dumps({"cublas_workspace_config": ":temporary:test"}),
        encoding="utf-8",
    )
    previous_cublas = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        check(
            "cuBLAS bootstrap reads and installs the lock value",
            module._bootstrap_cublas_environment(lock_repo)
            == ":temporary:test"
            and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":temporary:test",
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":conflict:"
        expect_raises(
            "cuBLAS bootstrap rejects a conflicting parent environment",
            module.ContractError,
            lambda: module._bootstrap_cublas_environment(lock_repo),
        )
    finally:
        if previous_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas

    # 7. SHA/size/mtime snapshots catch a one-byte fixture mutation.
    fixture_dir = tmp / "fixture"
    fixture_dir.mkdir()
    x_path = fixture_dir / "features.npy"
    y_path = fixture_dir / "labels.npy"
    x_path.write_bytes(b"\x93NUMPY-temporary-feature-fixture")
    y_path.write_bytes(b"\x93NUMPY-temporary-label-fixture")
    paths = {"features": x_path, "labels": y_path}
    before = module._snapshot_files(paths)
    same = module._snapshot_files(paths)
    check("temporary fixture snapshot is stable", module._snapshot_equal(before, same))
    x_path.write_bytes(x_path.read_bytes()[:-1] + b"X")
    mutated = module._snapshot_files(paths)
    check(
        "one-byte temporary fixture mutation is detected",
        not module._snapshot_equal(before, mutated),
    )

    # 8. Atomic JSON/CSV writes work and reject scientific field names.
    publication_dir = tmp / "publication"
    json_path = publication_dir / "record.json"
    csv_path = publication_dir / "record.csv"
    module._atomic_json(json_path, {
        "classification": module.DISCLAIMER,
        "duration_seconds": 1.25,
    })
    check(
        "atomic JSON publishes complete payload",
        json.loads(json_path.read_text(encoding="utf-8"))["duration_seconds"]
        == 1.25,
    )
    module._atomic_csv(
        csv_path,
        [{"state": "COMPLETE", "duration_seconds": 2.5}],
        ("state", "duration_seconds"),
    )
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    check(
        "atomic CSV publishes complete payload",
        csv_rows == [{"state": "COMPLETE", "duration_seconds": "2.5"}],
    )
    expect_raises(
        "JSON scientific field is rejected",
        module.ContractError,
        lambda: module._atomic_json(
            publication_dir / "forbidden.json",
            {"best_value": 0.0},
        ),
    )
    expect_raises(
        "CSV scientific field is rejected",
        module.ContractError,
        lambda: module._atomic_csv(
            publication_dir / "forbidden.csv",
            [{"mse": 0.0}],
            ("mse",),
        ),
    )
    check(
        "logical CSV hash matches published UTF-8 bytes",
        module._csv_rows_sha256(
            [{"state": "COMPLETE", "duration_seconds": 2.5}],
            ("state", "duration_seconds"),
        )
        == module._sha256_file(csv_path),
    )

    fake_logging = SimpleNamespace(WARNING=30, verbosity=20)
    fake_logging.get_verbosity = lambda: fake_logging.verbosity
    fake_logging.set_verbosity = (
        lambda value: setattr(fake_logging, "verbosity", value)
    )
    with module._suppress_optuna_value_logging(
        SimpleNamespace(logging=fake_logging)
    ):
        check(
            "Optuna objective-value logging is suppressed inside optimization",
            fake_logging.verbosity == fake_logging.WARNING,
        )
    check(
        "Optuna logging verbosity is restored after optimization",
        fake_logging.verbosity == 20,
    )

    class FlakyJsonPath:
        def __init__(self):
            self.attempts = 0

        def read_text(self, *, encoding):
            self.attempts += 1
            if self.attempts == 1:
                raise PermissionError("temporary sharing lock")
            if self.attempts == 2:
                return "{"
            return '{"authenticated": true}'

        def __str__(self):
            return "temporary-flaky-json-path"

    flaky_json_path = FlakyJsonPath()
    check(
        "authenticated JSON read retries sharing lock and torn parse",
        module._read_json_mapping(
            flaky_json_path, "temporary authenticated JSON")
        == {"authenticated": True}
        and flaky_json_path.attempts == 3,
    )
    nonfinite_path = publication_dir / "nonfinite.json"
    nonfinite_path.write_text('{"counter": NaN}', encoding="utf-8")
    original_read_attempts = module.AUTHENTICATED_JSON_READ_ATTEMPTS
    module.AUTHENTICATED_JSON_READ_ATTEMPTS = 1
    try:
        expect_raises(
            "authenticated JSON read rejects non-finite tokens",
            module.ContractError,
            lambda: module._read_json_mapping(
                nonfinite_path, "temporary non-finite JSON"),
        )
    finally:
        module.AUTHENTICATED_JSON_READ_ATTEMPTS = original_read_attempts

    heartbeat_path = publication_dir / "active_wall_heartbeat.json"
    with module._ActiveWallHeartbeat(
        heartbeat_path,
        "c" * 64,
        previous_seconds=2.0,
        interval_seconds=0.02,
    ) as heartbeat:
        time.sleep(0.06)
        check(
            "active-wall heartbeat advances atomically",
            heartbeat_path.is_file()
            and heartbeat.current_seconds() > 2.0,
        )
    final_heartbeat = json.loads(
        heartbeat_path.read_text(encoding="utf-8"))
    check(
        "active-wall heartbeat closes with nominal-cadence checkpoint metadata",
        final_heartbeat["status"] == "completed"
        and final_heartbeat["checkpoint_interval_seconds"] == 0.02
        and final_heartbeat["active_wall_seconds_cumulative"] >= 2.0,
    )
    completed_heartbeat_bytes = heartbeat_path.read_bytes()
    module._complete_or_verify_active_wall_checkpoint(
        heartbeat_path,
        "c" * 64,
        final_heartbeat["active_wall_seconds_cumulative"],
        allow_explicit_recovery=False,
    )
    check(
        "completed heartbeat authentication is byte-preserving",
        heartbeat_path.read_bytes() == completed_heartbeat_bytes,
    )

    active_heartbeat = {**final_heartbeat, "status": "active"}
    module._atomic_json(heartbeat_path, active_heartbeat)
    expect_raises(
        "active heartbeat cannot support a completed receipt implicitly",
        module.ContractError,
        lambda: module._complete_or_verify_active_wall_checkpoint(
            heartbeat_path,
            "c" * 64,
            final_heartbeat["active_wall_seconds_cumulative"],
            allow_explicit_recovery=False,
        ),
    )
    recovered_heartbeat = (
        module._complete_or_verify_active_wall_checkpoint(
            heartbeat_path,
            "c" * 64,
            final_heartbeat["active_wall_seconds_cumulative"],
            allow_explicit_recovery=True,
        )
    )
    check(
        "explicit recovery seals but does not invent heartbeat time",
        recovered_heartbeat["status"] == "completed"
        and recovered_heartbeat["completion_recovered_from_status"] == "active"
        and recovered_heartbeat["active_wall_seconds_cumulative"]
        == final_heartbeat["active_wall_seconds_cumulative"],
    )
    expect_raises(
        "heartbeat recovery rejects contradictory cumulative time",
        module.ContractError,
        lambda: module._complete_or_verify_active_wall_checkpoint(
            heartbeat_path,
            "c" * 64,
            final_heartbeat["active_wall_seconds_cumulative"] + 1.0,
            allow_explicit_recovery=True,
        ),
    )
    missing_heartbeat = publication_dir / "missing_heartbeat.json"
    expect_raises(
        "completed receipt cannot fabricate a missing heartbeat",
        module.ContractError,
        lambda: module._complete_or_verify_active_wall_checkpoint(
            missing_heartbeat,
            "c" * 64,
            0.0,
            allow_explicit_recovery=True,
        ),
    )

    expect_raises(
        "unfinished HPO segment needs explicit recovery without RUNNING trial",
        module.ContractError,
        lambda: module._merge_hpo_interruption_history(
            [],
            active_heartbeat,
            recover_stale=False,
            completed_receipt_missing=True,
        ),
    )
    interruption_history = module._merge_hpo_interruption_history(
        [],
        active_heartbeat,
        recover_stale=True,
        completed_receipt_missing=True,
    )
    check(
        "explicit HPO recovery records authenticated incomplete-tail evidence",
        len(interruption_history) == 1
        and interruption_history[0]["heartbeat_status"] == "active"
        and interruption_history[0]["active_time_tail_may_be_incomplete"]
        and interruption_history[0]["compute_receipt_was_missing"],
    )
    check(
        "same HPO recovery evidence is cumulative and deduplicated",
        module._merge_hpo_interruption_history(
            interruption_history,
            active_heartbeat,
            recover_stale=False,
            completed_receipt_missing=True,
        )
        == interruption_history,
    )
    tampered_history = [{**interruption_history[0], "pid": 1}]
    expect_raises(
        "tampered HPO interruption identity is rejected",
        module.ContractError,
        lambda: module._validated_hpo_interruption_history(tampered_history),
    )
    expect_raises(
        "boolean HPO interruption PID is rejected",
        module.ContractError,
        lambda: module._validated_hpo_interruption_history([
            {**interruption_history[0], "pid": True}
        ]),
    )
    completed_interruption = module._merge_hpo_interruption_history(
        [],
        final_heartbeat,
        recover_stale=True,
        completed_receipt_missing=True,
    )
    check(
        "torn HPO receipt after completed heartbeat preserves exact timing",
        len(completed_interruption) == 1
        and completed_interruption[0]["heartbeat_status"] == "completed"
        and not completed_interruption[0][
            "active_time_tail_may_be_incomplete"],
    )
    recovery_event = {
        "recovered_utc": "2000-01-01T00:00:00+00:00",
        "trial_numbers": [7],
        "policy": "state changed to FAIL; no deletion",
    }
    check(
        "Optuna recovery history preserves validated cumulative trials",
        module._validated_study_recovery_events([recovery_event])
        == [recovery_event],
    )
    expect_raises(
        "boolean Optuna recovery trial number is rejected",
        module.ContractError,
        lambda: module._validated_study_recovery_events([
            {**recovery_event, "trial_numbers": [True]}
        ]),
    )
    expect_raises(
        "duplicate cumulative Optuna recovery trial is rejected",
        module.ContractError,
        lambda: module._validated_study_recovery_events([
            recovery_event,
            {
                **recovery_event,
                "recovered_utc": "2000-01-02T00:00:00+00:00",
            },
        ]),
    )
    check(
        "cumulative memory receipt preserves each prior invocation peak",
        module._merge_memory_receipts(
            {
                "rss_peak_bytes": 20,
                "cuda_peak_allocated_bytes": 5,
                "cuda_peak_reserved_bytes": None,
            },
            {
                "rss_peak_bytes": 10,
                "cuda_peak_allocated_bytes": 8,
                "cuda_peak_reserved_bytes": 12,
            },
        )
        == {
            "rss_peak_bytes": 20,
            "cuda_peak_allocated_bytes": 8,
            "cuda_peak_reserved_bytes": 12,
        },
    )

    # A clean completed HPO receipt is a fast-return boundary: neither it nor
    # trial_compute.csv may be rewritten, and both the byte hash and logical
    # rows must authenticate against the resumed study.
    complete_state = SimpleNamespace(name="COMPLETE")
    pruned_state = SimpleNamespace(name="PRUNED")
    failed_state = SimpleNamespace(name="FAIL")
    running_state = SimpleNamespace(name="RUNNING")
    waiting_state = SimpleNamespace(name="WAITING")
    trial_state = SimpleNamespace(
        COMPLETE=complete_state,
        PRUNED=pruned_state,
        FAIL=failed_state,
        RUNNING=running_state,
        WAITING=waiting_state,
    )
    completed_trials = [
        SimpleNamespace(
            number=number,
            state=complete_state,
            intermediate_values={},
            duration=None,
            datetime_start=None,
            datetime_complete=None,
        )
        for number in range(module.USEFUL_TRIALS)
    ]
    receipt_study = SimpleNamespace(
        study_name="temporary-completed-hpo",
        trials=completed_trials,
        user_attrs={},
    )
    receipt_runtime = {
        "optuna": SimpleNamespace(
            trial=SimpleNamespace(TrialState=trial_state)),
        "torch": object(),
        "trainer": object(),
    }
    completed_counts = module._study_counts(
        receipt_runtime["optuna"], receipt_study)
    empty_counts = {
        "complete": 0,
        "pruned": 0,
        "failed": 0,
        "running": 0,
        "waiting": 0,
        "useful": 0,
        "total": 0,
    }
    receipt_run_dir = tmp / "completed-hpo-receipt"
    receipt_run_dir.mkdir()
    receipt_trial_csv = receipt_run_dir / "trial_compute.csv"
    module._write_trial_csv(
        receipt_trial_csv, receipt_study, "d" * 64)
    receipt_trial_sha256 = module._sha256_file(receipt_trial_csv)
    receipt_memory = {
        "rss_peak_bytes": 100,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    receipt_adapter_calls = {
        "get_cache": 1,
        "canonical_split": 1,
    }
    completed_hpo_report = {
        "study_name": receipt_study.study_name,
        "resumed_with_existing_trials": False,
        "counts_before": empty_counts,
        "counts_after": completed_counts,
        "useful_budget": module.USEFUL_TRIALS,
        "failure_slack": module.MAX_FAIL_SLACK,
        "epoch_cap_per_trial": module.EPOCHS,
        "useful_trial_duration_seconds_sum": 0.0,
        "useful_trial_duration_seconds_quantiles": {},
        "useful_epochs_reported_sum": 0,
        "useful_epochs_reported_quantiles": {},
        "failed_trial_durations_excluded": True,
        "failed_trial_duration_reason": "temporary",
        "hpo_wall_seconds_this_invocation": 4.0,
        "hpo_active_wall_seconds_cumulative": 4.0,
        "active_wall_checkpoint_interval_seconds":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "active_wall_semantics": "temporary",
        "timing_complete": True,
        "hpo_interruption_history": [],
        "optuna_recovery_events": [],
        "stale_inflight_trial_numbers_recovered_this_invocation": [],
        "all_stale_trial_numbers_recovered": [],
        "nominal_unrecorded_tail_seconds_per_abrupt_stop":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "unrecorded_tail_bound": "temporary",
        "checkpoint_files_removed_during_hpo": 0,
        "trial_compute_sha256": receipt_trial_sha256,
        "adapter_calls": receipt_adapter_calls,
        "adapter_calls_scope": "this invocation",
        "memory": receipt_memory,
        "memory_scope": "entire HPO workload",
        "memory_complete": True,
    }
    check(
        "completed HPO checker fixture covers exact report fields",
        set(completed_hpo_report) == set(module.HPO_REPORT_FIELDS),
    )
    receipt_hpo_path = receipt_run_dir / "hpo_compute.json"
    module._atomic_json(receipt_hpo_path, {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": "d" * 64,
        "study_name": receipt_study.study_name,
        "study_counts": completed_counts,
        "active_wall_seconds_cumulative": 4.0,
        "checkpoint_files_removed": 0,
        "hpo_interruption_history": [],
        "optuna_recovery_events": [],
        "timing_complete": True,
        "memory_complete": True,
        "memory": receipt_memory,
        "adapter_calls": receipt_adapter_calls,
        "report": completed_hpo_report,
        "report_sha256": module._canonical_sha256(completed_hpo_report),
        "trial_compute_sha256": receipt_trial_sha256,
        "completed_utc": "2000-01-01T00:00:00+00:00",
    })
    receipt_heartbeat_path = (
        receipt_run_dir / "active_wall_heartbeat.json"
    )
    module._atomic_json(receipt_heartbeat_path, {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": "d" * 64,
        "active_wall_seconds_cumulative": 4.0,
        "checkpoint_interval_seconds":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "pid": os.getpid(),
        "updated_utc": "2000-01-01T00:00:00+00:00",
    })
    hpo_bytes_before = receipt_hpo_path.read_bytes()
    trial_bytes_before = receipt_trial_csv.read_bytes()
    hpo_heartbeat_bytes_before = receipt_heartbeat_path.read_bytes()
    reused_hpo = module._optimize_useful_budget(
        receipt_runtime,
        receipt_study,
        {},
        {},
        {},
        receipt_run_dir / "weights",
        receipt_run_dir,
        "d" * 64,
        [],
        recover_stale=False,
    )
    check(
        "completed HPO receipt returns without rewriting authenticated evidence",
        reused_hpo["report"] == completed_hpo_report
        and receipt_hpo_path.read_bytes() == hpo_bytes_before
        and receipt_trial_csv.read_bytes() == trial_bytes_before
        and receipt_heartbeat_path.read_bytes()
        == hpo_heartbeat_bytes_before,
    )
    receipt_heartbeat_path.unlink()
    expect_raises(
        "completed HPO fast path rejects a missing heartbeat",
        module.ContractError,
        lambda: module._optimize_useful_budget(
            receipt_runtime,
            receipt_study,
            {},
            {},
            {},
            receipt_run_dir / "weights",
            receipt_run_dir,
            "d" * 64,
            [],
            recover_stale=True,
        ),
    )
    receipt_heartbeat_path.write_bytes(hpo_heartbeat_bytes_before)
    stale_hpo_heartbeat = json.loads(
        hpo_heartbeat_bytes_before.decode("utf-8"))
    stale_hpo_heartbeat["status"] = "active"
    module._atomic_json(receipt_heartbeat_path, stale_hpo_heartbeat)
    expect_raises(
        "completed HPO fast path rejects a stale active heartbeat",
        module.ContractError,
        lambda: module._optimize_useful_budget(
            receipt_runtime,
            receipt_study,
            {},
            {},
            {},
            receipt_run_dir / "weights",
            receipt_run_dir,
            "d" * 64,
            [],
            recover_stale=True,
        ),
    )
    receipt_heartbeat_path.write_bytes(hpo_heartbeat_bytes_before)
    receipt_trial_csv.write_bytes(trial_bytes_before[:-1] + b"X")
    expect_raises(
        "completed HPO fast return rejects tampered trial CSV",
        module.ContractError,
        lambda: module._optimize_useful_budget(
            receipt_runtime,
            receipt_study,
            {},
            {},
            {},
            receipt_run_dir / "weights",
            receipt_run_dir,
            "d" * 64,
            [],
            recover_stale=False,
        ),
    )

    # 9. Descriptor resumption accepts an identical descriptor, rejects drift,
    # and does not overwrite the authenticated file.
    descriptor_dir = tmp / "descriptor-run"
    descriptor_dir.mkdir()
    payload = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "descriptor_sha256": "a" * 64,
        "descriptor": {"purpose": "temporary compute fixture"},
    }
    descriptor_path = module._write_or_verify_descriptor(descriptor_dir, payload)
    first_bytes = descriptor_path.read_bytes()
    module._write_or_verify_descriptor(descriptor_dir, payload)
    check(
        "identical descriptor resume preserves exact bytes",
        descriptor_path.read_bytes() == first_bytes,
    )
    expect_raises(
        "descriptor drift is rejected",
        module.ContractError,
        lambda: module._write_or_verify_descriptor(
            descriptor_dir,
            {**payload, "descriptor_sha256": "b" * 64},
        ),
    )
    check(
        "descriptor drift does not overwrite stored descriptor",
        descriptor_path.read_bytes() == first_bytes,
    )

    # 10. The liveness probe must be observational.  On Windows,
    # os.kill(pid, 0) terminates rather than probes, so exercise another live
    # process and require it to survive the check before testing lock semantics.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        child_alive = module._pid_alive(child.pid)
        time.sleep(0.10)
        check(
            "PID liveness probe leaves the inspected process alive",
            child_alive and child.poll() is None,
        )
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)

    # The system-level guard excludes other processes and is crash-released.
    # This makes stale classification + archival + replacement one transaction.
    mutex_path = tmp / "direct-os-mutex.guard"
    mutex_probe_code = """
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("mutex_benchmark", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    with module._exclusive_file_mutex(Path(sys.argv[2])):
        print("acquired")
except module.ContractError:
    print("rejected")
"""
    with module._exclusive_file_mutex(mutex_path):
        blocked_mutex_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                mutex_probe_code,
                str(BENCHMARK),
                str(mutex_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    released_mutex_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            mutex_probe_code,
            str(BENCHMARK),
            str(mutex_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    check(
        "OS mutex excludes another process and releases cleanly",
        blocked_mutex_probe.stdout.strip() == "rejected"
        and released_mutex_probe.stdout.strip() == "acquired",
    )

    # O_EXCL PID lock rejects a concurrent holder.  A dead-PID lock cannot be
    # recovered implicitly; explicit recovery archives it without deletion.
    lock_dir = tmp / "lock-run"
    lock_dir.mkdir()
    with module._ExclusivePidLock(
        lock_dir,
        recover_stale=False,
        command="temporary-contract-check",
    ):
        expect_raises(
            "concurrent PID lock is rejected",
            module.ContractError,
            lambda: module._ExclusivePidLock(
                lock_dir,
                recover_stale=True,
                command="second-holder",
            ).__enter__(),
        )
    check("clean lock release archives evidence", not (lock_dir / "run.lock").exists())

    fresh_descriptor_dir = tmp / "fresh-descriptor-under-lock"
    fresh_descriptor_payload = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "descriptor_sha256": "9" * 64,
        "descriptor": {"purpose": "temporary lock-order fixture"},
    }
    with module._ExclusivePidLock(
        fresh_descriptor_dir,
        recover_stale=False,
        command="fresh-descriptor-contract-check",
    ):
        fresh_descriptor = module._write_or_verify_descriptor(
            fresh_descriptor_dir, fresh_descriptor_payload)
        check(
            "fresh descriptor accepts the persistent OS lock guard",
            fresh_descriptor.is_file(),
        )

    stale_payload = {
        "schema": "r5-compute-pid-lock-v1",
        "pid": 99_999_999,
        "hostname": socket.gethostname(),
        "created_utc": "2000-01-01T00:00:00+00:00",
        "command": "crashed",
        "token": "stale",
    }
    (lock_dir / "run.lock").write_text(
        json.dumps(stale_payload),
        encoding="utf-8",
    )
    expect_raises(
        "stale PID lock needs --recover-stale",
        module.ContractError,
        lambda: module._ExclusivePidLock(
            lock_dir,
            recover_stale=False,
            command="implicit-recovery",
        ).__enter__(),
    )
    with module._ExclusivePidLock(
        lock_dir,
        recover_stale=True,
        command="explicit-recovery",
    ):
        check("explicit stale recovery acquires lock", (lock_dir / "run.lock").is_file())
    history_names = [path.name for path in (lock_dir / "lock_history").iterdir()]
    check(
        "stale lock evidence is preserved",
        any(name.startswith("stale-recovered-") for name in history_names),
    )

    # Two explicit stale recoverers launched together must never both enter the
    # same run. The OS guard is crash-released and serialises classify+archive+
    # O_EXCL publication as one critical section.
    race_dir = tmp / "stale-recovery-race"
    race_dir.mkdir()
    (race_dir / "run.lock").write_text(
        json.dumps(stale_payload), encoding="utf-8")
    race_start = race_dir / "start"
    race_release = race_dir / "release"
    race_code = """
import importlib.util
from pathlib import Path
import sys
import time

spec = importlib.util.spec_from_file_location("race_benchmark", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run_dir = Path(sys.argv[2])
start = Path(sys.argv[3])
release = Path(sys.argv[4])
ready = Path(sys.argv[5])
result = Path(sys.argv[6])
ready.write_text("ready", encoding="utf-8")
while not start.exists():
    time.sleep(0.005)
try:
    with module._ExclusivePidLock(
        run_dir, recover_stale=True, command="race-recoverer"
    ):
        result.write_text("acquired", encoding="utf-8")
        while not release.exists():
            time.sleep(0.005)
except Exception as exc:
    result.write_text(
        "rejected:" + type(exc).__name__, encoding="utf-8"
    )
"""
    race_ready = [race_dir / f"ready-{index}" for index in range(2)]
    race_results = [race_dir / f"result-{index}" for index in range(2)]
    race_processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                race_code,
                str(BENCHMARK),
                str(race_dir),
                str(race_start),
                str(race_release),
                str(race_ready[index]),
                str(race_results[index]),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for index in range(2)
    ]
    try:
        ready_deadline = time.time() + 15.0
        while (
            not all(path.exists() for path in race_ready)
            and time.time() < ready_deadline
        ):
            time.sleep(0.01)
        check(
            "both stale-recovery contenders reached the barrier",
            all(path.exists() for path in race_ready),
        )
        race_start.touch()
        result_deadline = time.time() + 15.0
        while (
            not all(path.exists() for path in race_results)
            and time.time() < result_deadline
        ):
            time.sleep(0.01)
        race_values = [
            path.read_text(encoding="utf-8")
            if path.exists() else "missing"
            for path in race_results
        ]
        check(
            "simultaneous stale recovery admits exactly one process",
            race_values.count("acquired") == 1
            and sum(value.startswith("rejected:") for value in race_values)
            == 1,
        )
    finally:
        race_release.touch(exist_ok=True)
        for process in race_processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    reused_pid_dir = tmp / "reused-pid-lock"
    reused_pid_dir.mkdir()
    (reused_pid_dir / "run.lock").write_text(
        json.dumps({
            "pid": 12345,
            "hostname": socket.gethostname(),
            "process_identity": "old-process-creation",
        }),
        encoding="utf-8",
    )
    original_pid_alive = module._pid_alive
    original_process_identity = module._process_identity_token
    try:
        module._pid_alive = lambda pid: True
        module._process_identity_token = lambda pid: "new-process-creation"
        pid_reused_stale, pid_reused_detail = module._ExclusivePidLock(
            reused_pid_dir,
            recover_stale=True,
            command="pid-reuse-check",
        )._classify_existing()
    finally:
        module._pid_alive = original_pid_alive
        module._process_identity_token = original_process_identity
    check(
        "same-host live PID with a different creation identity is stale",
        pid_reused_stale and "identity_match=False" in pid_reused_detail,
    )

    unreadable_lock_dir = tmp / "unreadable-stale-lock"
    unreadable_lock_dir.mkdir()
    unreadable_lock_path = unreadable_lock_dir / "run.lock"
    unreadable_lock_path.write_bytes(b"{torn")
    old_timestamp = (
        time.time() - module.LOCK_UNREADABLE_STALE_SECONDS - 1.0
    )
    os.utime(unreadable_lock_path, (old_timestamp, old_timestamp))
    previous_read_attempts = module.AUTHENTICATED_JSON_READ_ATTEMPTS
    module.AUTHENTICATED_JSON_READ_ATTEMPTS = 1
    try:
        unreadable_stale, unreadable_detail = module._ExclusivePidLock(
            unreadable_lock_dir,
            recover_stale=True,
            command="unreadable-lock-check",
        )._classify_existing()
    finally:
        module.AUTHENTICATED_JSON_READ_ATTEMPTS = previous_read_attempts
    check(
        "old unreadable lock becomes explicitly recoverable after five minutes",
        unreadable_stale and "unreadable lock age=" in unreadable_detail,
    )

    # 11. Exercise the trainer adapter with standard-library fakes.  It must
    # return the supplied fixture/split and restore the exact original callables.
    original_cache = lambda *args, **kwargs: "original-cache"
    original_split = lambda *args, **kwargs: "original-split"
    fake_trainer = SimpleNamespace(
        get_or_create_cache=original_cache,
        canonical_train_val_split=original_split,
    )
    fake_np = SimpleNamespace(array_equal=lambda left, right: left == right)
    groups = tuple(range(module.DEFAULT_X_SHAPE[0]))
    fake_fixture = {
        "X": object(),
        "y": object(),
        "groups": groups,
    }
    fake_splits = {
        "indices": {
            "train": [0, 1],
            "inner": [2, 3],
        },
    }
    fake_config = {
        "method": "PAA",
        "dofs": [2, 5],
        "task": "regression",
        "target_supports": [2, 3],
        "use_lstm": True,
        "use_nhits": True,
    }
    with module._patched_trainer_fixture(
        fake_trainer,
        fake_np,
        fake_fixture,
        fake_splits,
        fake_config,
    ) as calls:
        cache_result = fake_trainer.get_or_create_cache(
            fake_config,
            module.STUDY_DATASET_NAME,
            "ignored",
        )
        split_result = fake_trainer.canonical_train_val_split(
            module.DEFAULT_X_SHAPE[0],
            groups,
            seed=module.SEED,
            dataset_name=module.STUDY_DATASET_NAME,
        )
        check("trainer cache adapter returns fixture", cache_result[:2] == (
            fake_fixture["X"], fake_fixture["y"]
        ))
        check("trainer split adapter returns fixed indices", split_result == (
            [0, 1], [2, 3]
        ))
        check("trainer adapter records both calls", calls == {
            "get_cache": 1, "canonical_split": 1
        })
    check(
        "trainer cache entry point is restored",
        fake_trainer.get_or_create_cache is original_cache,
    )
    check(
        "trainer split entry point is restored",
        fake_trainer.canonical_train_val_split is original_split,
    )

    # 12. Exact trial-number paths are cleaned; an unrecognised weight remains.
    weights_dir = tmp / "weights"
    weights_dir.mkdir()
    expected_0 = weights_dir / "weights_r5c_trial_0.pth"
    expected_3 = weights_dir / "weights_r5c_trial_3.pth"
    decoy = weights_dir / "weights_r5c_trial_999.pth"
    for path in (expected_0, expected_3, decoy):
        path.write_bytes(b"temporary")
    cleanup = module._clean_exact_trial_weights(
        weights_dir,
        "r5c",
        [SimpleNamespace(number=0), SimpleNamespace(number=3)],
    )
    check(
        "only exact trial-number weights are removed",
        not expected_0.exists() and not expected_3.exists() and decoy.exists(),
    )
    check(
        "exact cleanup reports two removals",
        cleanup["exact_files_removed"] == 2
        and cleanup["unexpected_paths_touched"] is False,
    )

    # 13. A published finalist marker is the durable commit. If power fails
    # before the attempt pointer flips to completed, recovery repairs only the
    # pointer and must not need (or be able) to invoke the trainer again.
    finalist_dir = tmp / "finalist-torn-pointer"
    finalist_dir.mkdir()
    complete_state = "COMPLETE"
    selected_trial = SimpleNamespace(
        state=complete_state,
        params={"temporary_parameter": 1},
        intermediate_values={0: 1.0},
        number=7,
    )
    finalist_study = SimpleNamespace(best_trial=selected_trial)
    finalist_runtime = {
        "optuna": SimpleNamespace(
            trial=SimpleNamespace(
                TrialState=SimpleNamespace(COMPLETE=complete_state)
            )
        ),
        "inference": SimpleNamespace(
            frozen_checkpoint_epoch_count=lambda values, max_epochs: 3
        ),
        # Deliberately no np/torch/trainer: the durable-marker fast path must
        # return before any compute runtime is touched.
    }
    finalist_descriptor = "f" * 64
    selected_parameter_sha256 = module._canonical_sha256(
        selected_trial.params)
    completed_utc = "2000-01-01T00:00:00+00:00"
    finalist_active_wall = 2.5
    finalist_identity = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "descriptor_sha256": finalist_descriptor,
        "selected_trial_number": selected_trial.number,
        "selected_parameter_sha256": selected_parameter_sha256,
        "frozen_checkpoint_epoch_count": 3,
        "helper": "training.trainer.fit_predict_finalist_fold",
    }
    finalist_marker = {
        **finalist_identity,
        "status": "completed",
        "durably_accepted_refits": 1,
        "execution_semantics": "temporary durable acceptance fixture",
        "attempt_count": 1,
        "prior_unaccepted_attempt_count": 0,
        "repeat": module.FINALIST_REPEAT,
        "fold": module.FINALIST_FOLD,
        "n_splits": module.FINALIST_N_SPLITS,
        "n_repeats": module.FINALIST_N_REPEATS,
        "split_seed": module.FINALIST_SPLIT_SEED,
        "train_state_count": 4,
        "validation_state_count": 1,
        "train_sample_count": 4 * module.PASSAGES_PER_STATE,
        "validation_sample_count": module.PASSAGES_PER_STATE,
        "train_states_sha256": "a" * 64,
        "validation_states_sha256": "b" * 64,
        "scale_train_infer_seconds": 1.25,
        "active_wall_seconds_cumulative": finalist_active_wall,
        "active_wall_checkpoint_interval_seconds":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "active_wall_semantics": "temporary exact clean-exit fixture",
        "timing_complete": True,
        "nominal_unrecorded_tail_seconds_per_abrupt_stop":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "unrecorded_tail_bound": "temporary nominal-cadence caveat",
        "memory": {
            "rss_peak_bytes": 1024,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        },
        "memory_scope": "temporary accepted attempt",
        "memory_complete": True,
        "returned_values_finite": True,
        "returned_values_discarded": True,
        "completed_utc": completed_utc,
    }
    finalist_running_attempt = {
        **finalist_identity,
        "status": "running",
        "attempt_count": 1,
        "prior_unaccepted_attempt_count": 0,
    }
    finalist_heartbeat = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": finalist_descriptor,
        "active_wall_seconds_cumulative": finalist_active_wall,
        "checkpoint_interval_seconds":
            module.ACTIVE_WALL_HEARTBEAT_SECONDS,
        "pid": os.getpid(),
        "updated_utc": completed_utc,
    }
    finalist_marker_path = finalist_dir / "finalist_compute.json"
    finalist_attempt_path = finalist_dir / "finalist_attempt_state.json"
    finalist_heartbeat_path = (
        finalist_dir / "finalist_active_wall_heartbeat.json"
    )
    module._atomic_json(finalist_marker_path, finalist_marker)
    module._atomic_json(finalist_attempt_path, finalist_running_attempt)
    module._atomic_json(finalist_heartbeat_path, finalist_heartbeat)
    marker_bytes = finalist_marker_path.read_bytes()
    heartbeat_bytes = finalist_heartbeat_path.read_bytes()
    expect_raises(
        "torn finalist attempt pointer needs explicit recovery",
        module.ContractError,
        lambda: module._run_finalist_once(
            finalist_runtime,
            finalist_study,
            {},
            {},
            {},
            finalist_dir,
            finalist_descriptor,
            recover_stale=False,
        ),
    )
    recovered_marker = module._run_finalist_once(
        finalist_runtime,
        finalist_study,
        {},
        {},
        {},
        finalist_dir,
        finalist_descriptor,
        recover_stale=True,
    )
    repaired_attempt = module._read_json_mapping(
        finalist_attempt_path, "repaired finalist attempt")
    check(
        "explicit finalist recovery repairs only the attempt pointer",
        recovered_marker == finalist_marker
        and repaired_attempt["status"] == "completed"
        and repaired_attempt["completion_marker"] == finalist_marker_path.name
        and finalist_marker_path.read_bytes() == marker_bytes
        and finalist_heartbeat_path.read_bytes() == heartbeat_bytes,
    )
    repaired_artifact_bytes = {
        path.name: path.read_bytes()
        for path in (
            finalist_marker_path,
            finalist_attempt_path,
            finalist_heartbeat_path,
        )
    }
    module._run_finalist_once(
        finalist_runtime,
        finalist_study,
        {},
        {},
        {},
        finalist_dir,
        finalist_descriptor,
        recover_stale=False,
    )
    check(
        "completed finalist fast path is a byte-preserving compute no-op",
        repaired_artifact_bytes == {
            path.name: path.read_bytes()
            for path in (
                finalist_marker_path,
                finalist_attempt_path,
                finalist_heartbeat_path,
            )
        },
    )
    finalist_heartbeat_path.write_bytes(
        heartbeat_bytes.replace(b"2.5", b"3.5"))
    expect_raises(
        "finalist marker rejects a contradictory heartbeat",
        module.ContractError,
        lambda: module._run_finalist_once(
            finalist_runtime,
            finalist_study,
            {},
            {},
            {},
            finalist_dir,
            finalist_descriptor,
            recover_stale=True,
        ),
    )
    finalist_heartbeat_path.write_bytes(heartbeat_bytes)

    missing_attempt_dir = tmp / "finalist-missing-attempt"
    missing_attempt_dir.mkdir()
    (missing_attempt_dir / "finalist_compute.json").write_bytes(marker_bytes)
    (missing_attempt_dir / "finalist_active_wall_heartbeat.json").write_bytes(
        heartbeat_bytes)
    expect_raises(
        "finalist marker rejects a missing attempt artifact",
        module.ContractError,
        lambda: module._run_finalist_once(
            finalist_runtime,
            finalist_study,
            {},
            {},
            {},
            missing_attempt_dir,
            finalist_descriptor,
            recover_stale=True,
        ),
    )
    orphan_heartbeat_dir = tmp / "finalist-orphan-heartbeat"
    orphan_heartbeat_dir.mkdir()
    (
        orphan_heartbeat_dir / "finalist_active_wall_heartbeat.json"
    ).write_bytes(heartbeat_bytes)
    expect_raises(
        "orphan finalist heartbeat is never relabelled as a first attempt",
        module.ContractError,
        lambda: module._run_finalist_once(
            finalist_runtime,
            finalist_study,
            {},
            {},
            {},
            orphan_heartbeat_dir,
            finalist_descriptor,
            recover_stale=True,
        ),
    )
    check(
        "complete genuine finalist marker passes the shared validator",
        module._validated_finalist_marker(
            finalist_marker, finalist_identity) == finalist_marker,
    )
    finalist_mutations = {
        "truncated finalist marker is rejected": {
            key: value for key, value in finalist_marker.items()
            if key != "repeat"
        },
        "finalist state/sample contradiction is rejected": {
            **finalist_marker,
            "validation_sample_count":
                finalist_marker["validation_sample_count"] + 1,
        },
        "numeric finalist completeness flag is rejected": {
            **finalist_marker,
            "timing_complete": 1,
        },
    }
    for label, mutated_marker in finalist_mutations.items():
        expect_raises(
            label,
            module.ContractError,
            lambda value=mutated_marker:
                module._validated_finalist_marker(
                    value, finalist_identity),
        )

    # 14. The final summary authenticates every announced receipt and supports
    # only explicit, evidence-preserving repair of its separate run-state
    # pointer. Repeated authentication is a byte-preserving no-op.
    summary_dir = tmp / "completed-summary"
    summary_dir.mkdir()
    source_study_path = summary_dir / "study.sqlite3"
    connection = sqlite3.connect(source_study_path)
    try:
        connection.execute("CREATE TABLE receipt_test (value INTEGER)")
        connection.execute("INSERT INTO receipt_test VALUES (7)")
        connection.commit()
    finally:
        connection.close()
    study_receipt_path = module._materialize_study_storage_receipt(
        summary_dir)
    connection = sqlite3.connect(study_receipt_path)
    try:
        receipt_value = connection.execute(
            "SELECT value FROM receipt_test").fetchone()
    finally:
        connection.close()
    check(
        "immutable SQLite receipt is a valid logical backup",
        receipt_value == (7,),
    )

    for name in module.IMMUTABLE_EVIDENCE_FILES:
        path = summary_dir / name
        if path == study_receipt_path:
            continue
        path.write_bytes(f"temporary evidence: {name}\n".encode("utf-8"))
    summary_fixture_path = tmp / "summary-fixture.bin"
    summary_fixture_path.write_bytes(b"immutable fixture bytes")
    summary_fixture = module._snapshot_files({
        "features": summary_fixture_path,
    })
    summary_source_hashes = {
        "benchmark_r5_compute.py": "a" * 64,
    }
    summary_descriptor = "b" * 64
    summary_git_sha = "c" * 40
    summary_run_id = "temporary-completed-run"
    immutable_evidence = module._immutable_evidence_snapshot(summary_dir)
    completed_summary = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": summary_descriptor,
        "identity": {
            "git_sha": summary_git_sha,
            "run_id": summary_run_id,
            "run_directory": str(summary_dir.resolve()),
            "git_tracked_dirty_at_start": False,
        },
        "command": {
            "this_invocation_measurement_completed_utc":
                "2000-01-01T00:00:00+00:00",
        },
        "hashes": {
            "source_before": summary_source_hashes,
            "source_after": summary_source_hashes,
            "fixture_before": summary_fixture,
            "fixture_after": summary_fixture,
            "study_storage": module._storage_snapshot(summary_dir),
            "immutable_evidence": immutable_evidence,
        },
    }
    summary_path = summary_dir / "summary.json"
    state_path = summary_dir / "run_state.json"
    module._atomic_json(summary_path, completed_summary)
    summary_sha256 = module._sha256_file(summary_path)
    clean_completed_state = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": summary_descriptor,
        "summary": "summary.json",
        "summary_sha256": summary_sha256,
    }
    module._atomic_json(state_path, clean_completed_state)

    def authenticate_summary(*, recover: bool):
        return module._completed_summary_if_valid(
            summary_dir,
            descriptor_sha256=summary_descriptor,
            git_sha=summary_git_sha,
            run_id=summary_run_id,
            fixture_snapshot=summary_fixture,
            source_hashes=summary_source_hashes,
            recover_torn_state=recover,
        )

    immutable_paths = [
        summary_dir / name for name in module.IMMUTABLE_EVIDENCE_FILES
    ]
    no_op_before = {
        path.name: path.read_bytes()
        for path in [summary_path, state_path, *immutable_paths]
    }
    authenticated_summary, pointer_repaired = authenticate_summary(
        recover=False)
    authenticated_summary_again, pointer_repaired_again = authenticate_summary(
        recover=False)
    no_op_after = {
        path.name: path.read_bytes()
        for path in [summary_path, state_path, *immutable_paths]
    }
    check(
        "completed-summary authentication is a byte-preserving no-op",
        authenticated_summary == completed_summary
        and authenticated_summary_again == completed_summary
        and not pointer_repaired
        and not pointer_repaired_again
        and no_op_after == no_op_before,
    )

    for path in immutable_paths:
        original = path.read_bytes()
        path.write_bytes(original + b"X")
        expect_raises(
            f"completed summary rejects mutated {path.name}",
            module.ContractError,
            lambda: authenticate_summary(recover=True),
        )
        path.write_bytes(original)

    failed_pointer = {
        "schema": module.BENCHMARK_SCHEMA,
        "classification": module.DISCLAIMER,
        "status": "failed",
        "descriptor_sha256": summary_descriptor,
    }
    module._atomic_json(state_path, failed_pointer)
    failed_pointer_bytes = state_path.read_bytes()
    expect_raises(
        "incomplete summary pointer needs explicit recovery",
        module.ContractError,
        lambda: authenticate_summary(recover=False),
    )
    summary_bytes_before_repair = summary_path.read_bytes()
    evidence_before_repair = {
        path.name: path.read_bytes() for path in immutable_paths
    }
    _, pointer_repaired = authenticate_summary(recover=True)
    repaired_state = module._read_json_mapping(
        state_path, "repaired summary pointer")
    archived_pointer = (
        summary_dir / repaired_state["prior_completion_pointer"]["archive"]
    )
    check(
        "summary-pointer recovery archives prior bytes and preserves receipts",
        pointer_repaired
        and archived_pointer.read_bytes() == failed_pointer_bytes
        and module._sha256_file(archived_pointer)
        == repaired_state["prior_completion_pointer"]["sha256"]
        and summary_path.read_bytes() == summary_bytes_before_repair
        and evidence_before_repair == {
            path.name: path.read_bytes() for path in immutable_paths
        },
    )
    authenticate_summary(recover=False)

    repaired_state_bytes = state_path.read_bytes()
    contradictory_state = dict(repaired_state)
    contradictory_state["summary_sha256"] = "d" * 64
    module._atomic_json(state_path, contradictory_state)
    expect_raises(
        "contradictory completed summary pointer is never repaired",
        module.ContractError,
        lambda: authenticate_summary(recover=True),
    )
    state_path.write_bytes(repaired_state_bytes)

    state_path.write_bytes(b"{torn")
    torn_state_bytes = state_path.read_bytes()
    previous_read_attempts = module.AUTHENTICATED_JSON_READ_ATTEMPTS
    module.AUTHENTICATED_JSON_READ_ATTEMPTS = 1
    try:
        expect_raises(
            "unreadable summary pointer needs explicit recovery",
            module.ContractError,
            lambda: authenticate_summary(recover=False),
        )
        _, unreadable_repaired = authenticate_summary(recover=True)
    finally:
        module.AUTHENTICATED_JSON_READ_ATTEMPTS = previous_read_attempts
    unreadable_state = module._read_json_mapping(
        state_path, "repaired unreadable summary pointer")
    unreadable_archive = (
        summary_dir
        / unreadable_state["prior_completion_pointer"]["archive"]
    )
    check(
        "explicit recovery preserves unreadable pointer bytes",
        unreadable_repaired
        and unreadable_archive.read_bytes() == torn_state_bytes,
    )

    state_path.unlink()
    expect_raises(
        "missing summary pointer needs explicit recovery",
        module.ContractError,
        lambda: authenticate_summary(recover=False),
    )
    _, missing_repaired = authenticate_summary(recover=True)
    missing_state = module._read_json_mapping(
        state_path, "repaired missing summary pointer")
    check(
        "explicit recovery recreates only a missing completion pointer",
        missing_repaired
        and missing_state["prior_completion_pointer"] == {
            "status": "missing",
            "archive": None,
            "sha256": None,
        }
        and summary_path.read_bytes() == summary_bytes_before_repair,
    )

print()
if FAILURES:
    raise SystemExit(f"R5 COMPUTE BENCHMARK CONTRACT: {FAILURES} FAILURE(S)")
print("R5 COMPUTE BENCHMARK CONTRACT: ALL PASS")
