"""Adversarial, solver-free checks for the Paper-1 compute benchmark."""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace

from campaign_import_guard import enforce_import_boundary

enforce_import_boundary()


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "benchmark_paper1_compute.py"
PIPELINE_PATH = ROOT / "training" / "pipeline.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
PIPELINE_SOURCE = PIPELINE_PATH.read_text(encoding="utf-8")
FAILURES = 0


def check(label: str, condition: bool) -> None:
    global FAILURES
    if condition:
        print(f"[PASS] {label}")
    else:
        FAILURES += 1
        print(f"[FAIL] {label}")


def rejects(label: str, action) -> None:
    try:
        action()
    except Exception:
        check(label, True)
    else:
        check(label, False)


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def static_violations(source: str, pipeline_source: str) -> list[str]:
    try:
        tree = ast.parse(source)
        pipeline_tree = ast.parse(pipeline_source)
    except SyntaxError:
        return ["syntax"]
    violations: list[str] = []
    expected = {
        "BENCHMARK_SCHEMA": "ttbi-paper1-compute-benchmark-v2",
        "DESCRIPTOR_SCHEMA": "ttbi-paper1-compute-benchmark-descriptor-v2",
        "AUTHORIZATION_SCHEMA": "ttbi-paper1-benchmark-authorization-evidence-v2",
        "ARCHITECTURE_ID": "RAW_POS1_LSTM1_MR1",
        "STAGE": "F40-S",
        "ACTIVE_DOFS": (1,),
        "TRIAL_SEED": 104729,
        "N_TRIALS": 100,
        "EPOCHS": 50,
        "N_STATES": 305,
        "PASSAGES_PER_STATE": 50,
        "N_SEVERITIES": 61,
        "RAW_LENGTH": 5831,
        "N_CHANNELS": 1,
        "TARGET_COUNT": 1,
        "TRAIN_GROUP_COUNT": 183,
        "VALIDATION_GROUP_COUNT": 61,
        "SEALED_TEST_GROUP_COUNT": 61,
    }
    for name, value in expected.items():
        if _literal(tree, name) != value:
            violations.append(f"constant:{name}")
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = {
        "run_benchmark",
        "_registered_execution_environment",
        "verify_completed_receipt",
        "_fixture_arrays",
        "_patched_trainer_fixture",
        "_semantic_study_evidence",
        "_validate_champion_state",
        "_validate_summary",
        "_read_trial_csv",
    }
    if not required <= set(functions):
        violations.append("producer-functions")
        return violations
    pipeline_functions = {
        node.name: node
        for node in pipeline_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    if "execute_registered_hpo_study" not in pipeline_functions:
        violations.append("registered-hpo-helper")
    run_text = ast.get_source_segment(source, functions["run_benchmark"]) or ""
    for token in (
        "tested_commit = _require_clean_commit_a(REPO)",
        "repository_source_snapshot(REPO)",
        "validate_environment_lock(environment_lock)",
        "load_capacity_receipt(",
        "_registered_execution_environment()",
        "current_execution_environment()",
        "enforce_execution_block(",
        "pipeline.execute_registered_hpo_study(",
        "require_fresh=True",
        "callbacks=()",
        "show_progress_bar=False",
        "torch.cuda.reset_peak_memory_stats()",
        "source_snapshot.assert_unchanged()",
        "verify_completed_receipt(run_dir, tested_commit, repo=REPO)",
    ):
        if token not in run_text:
            violations.append(f"run:{token}")
    environment_text = ast.get_source_segment(
        source, functions["_registered_execution_environment"]
    ) or ""
    seed_call = 'set_global_seed(TRIAL_SEED, TRAIN_PROTOCOL["determinism"])'
    capture_call = "current_execution_environment()"
    if (
        seed_call not in environment_text
        or capture_call not in environment_text
        or environment_text.index(seed_call) > environment_text.index(capture_call)
    ):
        violations.append("registered-environment-order")
    if ".optimize(" in run_text or "remaining =" in run_text:
        violations.append("parallel-optimize-path")
    fixture_text = ast.get_source_segment(source, functions["_fixture_arrays"]) or ""
    if any(token in fixture_text for token in ("np.sin", "np.cos", "np.exp")):
        violations.append("transcendental-fixture")
    verify_text = ast.get_source_segment(
        source, functions["verify_completed_receipt"]
    ) or ""
    for token in (
        "current_head = _require_clean_tested_or_report_commit(",
        "_outside_repository(directory, repository)",
        "_assert_final_inventory(directory)",
        "_semantic_study_evidence(",
        "snapshot.assert_unchanged()",
        '"schema": AUTHORIZATION_SCHEMA',
    ):
        if token not in verify_text:
            violations.append(f"verify:{token}")
    helper_text = ast.get_source_segment(
        pipeline_source, pipeline_functions.get("execute_registered_hpo_study")
    ) if "execute_registered_hpo_study" in pipeline_functions else ""
    ablation_text = ast.get_source_segment(
        pipeline_source, pipeline_functions.get("execute_ablation_pipeline")
    ) if "execute_ablation_pipeline" in pipeline_functions else ""
    if "execute_registered_hpo_study(" not in (ablation_text or ""):
        violations.append("ablation-does-not-share-helper")
    for token in (
        "derive_execution_plan(",
        "validate_capacity_receipt(",
        "_create_or_resume_study(",
        "if require_fresh and study.trials:",
        "_stamp_study_protocol(",
        "objective = Objective(",
        "_execute_protocol_study(",
    ):
        if token not in (helper_text or ""):
            violations.append(f"helper:{token}")
    return violations


print("PAPER-1 BENCHMARK CONTRACT CHECKS")
check(
    "live producer and shared campaign helper satisfy static contract",
    not static_violations(SOURCE, PIPELINE_SOURCE),
)

for label, old, new in (
    ("99 trials", "N_TRIALS = 100", "N_TRIALS = 99"),
    ("short fixture", "N_STATES = 305", "N_STATES = 30"),
    ("five passages", "PASSAGES_PER_STATE = 50", "PASSAGES_PER_STATE = 5"),
    ("short RAW", "RAW_LENGTH = 5831", "RAW_LENGTH = 583"),
    ("PAA cell", 'ARCHITECTURE_ID = "RAW_POS1_LSTM1_MR1"', 'ARCHITECTURE_ID = "PAA_POS1_LSTM1_MR1"'),
    ("two channels", "ACTIVE_DOFS = (1,)", "ACTIVE_DOFS = (0, 1)"),
    ("unregistered seed", "TRIAL_SEED = 104729", "TRIAL_SEED = 42"),
    ("partial resume", "require_fresh=True", "require_fresh=False"),
    ("objective progress", "show_progress_bar=False", "show_progress_bar=True"),
    ("missing clean gate", "tested_commit = _require_clean_commit_a(REPO)", "tested_commit = _resolved_commit(REPO)"),
    (
        "environment captured before determinism policy",
        'set_global_seed(TRIAL_SEED, TRAIN_PROTOCOL["determinism"])',
        'current_execution_environment(); set_global_seed(TRIAL_SEED, TRAIN_PROTOCOL["determinism"])',
    ),
):
    mutated = SOURCE.replace(old, new, 1)
    check(
        f"static mutation rejected: {label}",
        mutated != SOURCE and bool(static_violations(mutated, PIPELINE_SOURCE)),
    )

mutated_pipeline = PIPELINE_SOURCE.replace(
    "if require_fresh and study.trials:", "if False and study.trials:", 1
)
check(
    "static mutation rejected: shared helper permits resume",
    mutated_pipeline != PIPELINE_SOURCE
    and bool(static_violations(SOURCE, mutated_pipeline)),
)


import benchmark_paper1_compute as benchmark  # noqa: E402


import torch
from core.execution_environment import (
    current_execution_environment,
    execution_environment_sha256,
)
from training.trainer import TRAIN_PROTOCOL

# Reproduce the entry-state drift that previously made the initial benchmark
# identity differ from the state established inside the registered HPO helper.
torch.use_deterministic_algorithms(False, warn_only=True)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
registered_environment = benchmark._registered_execution_environment()
registered_numeric = registered_environment["numeric_stack"]
determinism = TRAIN_PROTOCOL["determinism"]
check(
    "benchmark applies registered numeric state before environment capture",
    registered_numeric["deterministic_algorithms"]
    is determinism["torch_deterministic_algorithms"]
    and registered_numeric["deterministic_warn_only"]
    is determinism["torch_deterministic_warn_only"]
    and registered_numeric["cudnn_deterministic"]
    is determinism["cudnn_deterministic"]
    and registered_numeric["cudnn_benchmark"]
    is determinism["cudnn_benchmark"]
    and registered_numeric["cudnn_allow_tf32"]
    is determinism["cudnn_allow_tf32"]
    and registered_numeric["cuda_matmul_allow_tf32"]
    is determinism["cuda_matmul_allow_tf32"]
    and registered_numeric["float32_matmul_precision"]
    == determinism["float32_matmul_precision"]
    and execution_environment_sha256(registered_environment)
    == execution_environment_sha256(current_execution_environment()),
)


policy = benchmark._benchmark_policy()
check(
    "registered benchmark is the largest-mechanism RAW one-channel anchor",
    policy["architecture"] == {
        "name_short": "RAW_POS1_LSTM1_MR1",
        "method": "RAW",
        "use_space2vec": True,
        "use_lstm": True,
        "use_nhits": True,
        "model_type": "1D_MODULAR",
    }
    and policy["active_dofs"] == [1]
    and policy["study"]["trials"] == 100
    and policy["study"]["epochs"] == 50
    and policy["study"]["fresh_uninterrupted_only"] is True,
)

x, y, groups = benchmark._fixture_arrays()
fixture_digest = benchmark._fixture_digest((x, y, groups))
train_groups, inner_groups, sealed_groups = benchmark._partition_groups()
check(
    "full-size deterministic fixture and exact 60/20/20 groups are present",
    x.shape == (15250, 1, 5831)
    and y.shape == (15250, 1)
    and groups.shape == (15250,)
    and (len(train_groups), len(inner_groups), len(sealed_groups)) == (183, 61, 61)
    and set(train_groups).isdisjoint(inner_groups)
    and set(train_groups).isdisjoint(sealed_groups)
    and set(inner_groups).isdisjoint(sealed_groups)
    and set(train_groups) | set(inner_groups) | set(sealed_groups) == set(range(305)),
)
check(
    "fixture uses exact binary modular construction and 61 severity labels",
    float(x.min()) == -1.0
    and float(x.max()) == 1023.0 / 1024.0
    and sorted(set(y[:, 0].tolist())) == [float(i) for i in range(61)]
    and float(x[0, 0, 0]) == -1.0
    and float(x[1, 0, 0]) == (-927.0 / 1024.0),
)


class _FakeTrainer:
    def get_or_create_cache(self):
        raise AssertionError

    def canonical_train_val_split(self):
        raise AssertionError


fake_trainer = _FakeTrainer()
config_stub = {"identity": "fixture"}
with benchmark._patched_trainer_fixture(
    fake_trainer, x, y, groups, config_stub
) as calls:
    observed = fake_trainer.get_or_create_cache(
        config_stub, benchmark.STUDY_DATASET, "unused"
    )
    train_idx, val_idx = fake_trainer.canonical_train_val_split(
        len(observed[1]), observed[3], dataset_name=benchmark.STUDY_DATASET
    )
    selected_groups = (
        set(observed[3][train_idx].tolist())
        | set(observed[3][val_idx].tolist())
    )
    check(
        "trainer adapter passes only train+inner validation, never sealed test",
        observed[0].shape == (12200, 1, 5831)
        and observed[1].shape == (12200, 1)
        and set(observed[3].tolist()) == set(train_groups) | set(inner_groups)
        and set(observed[3][train_idx].tolist()) == set(train_groups)
        and set(observed[3][val_idx].tolist()) == set(inner_groups)
        and selected_groups.isdisjoint(sealed_groups)
        and calls == {
            "get_or_create_cache": 1,
            "canonical_train_val_split": 1,
        },
    )
del x, y, groups
x2, y2, groups2 = benchmark._fixture_arrays()
check(
    "full fixture bytes reproduce deterministically",
    fixture_digest == benchmark._fixture_digest((x2, y2, groups2)),
)
del x2, y2, groups2


def _runtime() -> dict:
    from core.execution_environment import (
        execution_compatibility_descriptor,
        execution_compatibility_sha256,
        execution_environment_sha256,
    )

    descriptor = {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": "benchmark-contract-host",
            "machine": "AMD64",
            "system": "Windows",
            "platform": "benchmark-contract-fixture",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "benchmark-contract-gpu",
            "uuid": "GPU-benchmark-contract",
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 40,
            "total_memory_bytes": 8_589_934_592,
            "driver_version": "fixture",
        },
        "numeric_stack": {
            "torch_version": "fixture",
            "cuda_runtime_version": "fixture",
            "cudnn_version": 1,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    compatibility = execution_compatibility_descriptor(descriptor)
    return {
        "schema": "ttbi-execution-runtime-binding-v2",
        "execution_block": "f40s",
        "anchor_stage": "F40-S",
        "execution_environment_sha256": execution_environment_sha256(descriptor),
        "execution_environment_descriptor": descriptor,
        "execution_compatibility_sha256": execution_compatibility_sha256(descriptor),
        "execution_compatibility_descriptor": compatibility,
    }


def _probe(architecture, _config, _params, runtime):
    from core.capacity_preflight import ARCHITECTURES

    index = ARCHITECTURES.index(architecture)
    total = runtime["execution_environment_descriptor"]["accelerator"][
        "total_memory_bytes"
    ]
    reserved = 600_000_000 + index * 100_000_000
    return {
        "peak_memory_allocated_bytes": reserved - 50_000_000,
        "peak_memory_reserved_bytes": reserved,
        "total_memory_bytes": total,
    }


class _Chooser:
    def __init__(self):
        self.distributions = {}

    def suggest_int(self, name, low, high, *, step=1, log=False):
        import optuna

        self.distributions[name] = optuna.distributions.IntDistribution(
            low, high, step=step, log=log
        )
        return int(low)

    def suggest_float(self, name, low, high, *, step=None, log=False):
        import optuna

        self.distributions[name] = optuna.distributions.FloatDistribution(
            low, high, step=step, log=log
        )
        return float(low)

    def suggest_categorical(self, name, choices):
        import optuna

        choices = tuple(choices)
        self.distributions[name] = optuna.distributions.CategoricalDistribution(
            choices
        )
        return choices[0]


def _happy_path(root: Path):
    import optuna
    import torch
    from core.capacity_preflight import run_capacity_preflight
    from core.hyperparameter_policy import derive_execution_plan
    from core.models import build_model
    from core.source_provenance import python_runtime_source_root
    from training import pipeline, trainer

    runtime = _runtime()
    source_root = python_runtime_source_root()
    capacity = run_capacity_preflight(
        runtime,
        probe_runner=_probe,
        source_root_sha256=source_root.sha256,
        source_file_count=source_root.file_count,
    )
    snapshot = SimpleNamespace(
        generator=SimpleNamespace(sha256="a" * 64, file_count=7),
        python_runtime=source_root,
    )
    descriptor, _receipt = benchmark._expected_descriptor(
        tested_source_commit="1" * 40,
        source_snapshot=snapshot,
        capacity_envelope=capacity,
        environment_lock={"sha256": "b" * 64},
        fixture_sha256="c" * 64,
    )
    config = benchmark._study_config(descriptor)
    database = root / "study.sqlite3"
    storage = f"sqlite:///{database.as_posix()}"
    study = pipeline._create_or_resume_study(
        descriptor["study_name"],
        storage,
        benchmark.N_TRIALS,
        sampler_seed=benchmark.TRIAL_SEED,
        use_pruner=True,
    )
    plan = derive_execution_plan(
        config,
        dataset_name=benchmark.STUDY_DATASET,
        requested_n_trials=benchmark.N_TRIALS,
        requested_use_pruner=True,
        execution_runtime=runtime,
    )
    pipeline._stamp_study_protocol(
        study,
        config=config,
        dataset_name=benchmark.STUDY_DATASET,
        n_trials=benchmark.N_TRIALS,
        epochs=benchmark.EPOCHS,
        sampler_seed=benchmark.TRIAL_SEED,
        use_pruner=True,
        hyperparameter_plan=plan,
        capacity_receipt=capacity,
    )
    chooser = _Chooser()
    params = trainer._suggest_params(chooser, config)
    started = datetime(2026, 8, 9, 12, 0, 0)
    for number in range(benchmark.N_TRIALS):
        state = (
            optuna.trial.TrialState.COMPLETE
            if number % 4 == 0 else optuna.trial.TrialState.PRUNED
        )
        base = 0.125 + number
        intermediate = {0: base + 2.0, 1: base + 1.0, 2: base}
        trial = optuna.trial.FrozenTrial(
            number=number,
            state=state,
            value=base,
            datetime_start=started + timedelta(seconds=number * 10),
            datetime_complete=started + timedelta(seconds=number * 10 + 5),
            params=dict(params),
            distributions=dict(chooser.distributions),
            user_attrs={},
            system_attrs={},
            intermediate_values=intermediate,
            trial_id=-1,
        )
        study.add_trial(trial)
    champion = root / "champion.pth"
    model, n_outputs = build_model(
        config,
        params,
        (
            benchmark.N_STATES * benchmark.PASSAGES_PER_STATE,
            benchmark.N_CHANNELS,
            benchmark.RAW_LENGTH,
        ),
        torch.device("cpu"),
    )
    assert n_outputs == 1
    torch.save(model.state_dict(), champion)
    storage_object = getattr(study, "_storage", None)
    if storage_object is not None and hasattr(storage_object, "remove_session"):
        storage_object.remove_session()
    backend = getattr(storage_object, "_backend", storage_object)
    engine = getattr(backend, "engine", None)
    if engine is not None:
        engine.dispose()
    del study
    return descriptor, capacity, database, champion


with tempfile.TemporaryDirectory(prefix="paper1-benchmark-optuna-check-") as raw:
    root = Path(raw)
    descriptor, capacity, database, champion = _happy_path(root)
    semantic = benchmark._semantic_study_evidence(
        database, descriptor=descriptor, champion_path=champion
    )
    check(
        "real Optuna happy path proves schema, 100 trials, best, and state_dict",
        semantic["counts"] == {
            "COMPLETE": 25,
            "PRUNED": 75,
            "FAIL": 0,
            "RUNNING": 0,
            "WAITING": 0,
            "TOTAL": 100,
        }
        and semantic["selected_trial_number"] == 0
        and len(semantic["rows"]) == 100,
    )

    def mutated_database(label: str, statements) -> None:
        safe_label = label.replace(" ", "-").replace("/", "-")
        target = root / f"mutation-{safe_label}.sqlite3"
        shutil.copyfile(database, target)
        connection = sqlite3.connect(target)
        try:
            if isinstance(statements, tuple) and isinstance(statements[0], str):
                statements = [statements]
            for statement, parameters in statements:
                connection.execute(statement, parameters)
            connection.commit()
        finally:
            connection.close()
        rejects(
            f"real Optuna verifier rejects {label}",
            lambda: benchmark._semantic_study_evidence(
                target, descriptor=descriptor, champion_path=champion
            ),
        )

    wrong_name = deepcopy(descriptor)
    wrong_name["study_name"] += "-foreign"
    rejects(
        "real Optuna verifier rejects foreign study name",
        lambda: benchmark._semantic_study_evidence(
            database, descriptor=wrong_name, champion_path=champion
        ),
    )
    mutated_database(
        "maximize direction",
        ("UPDATE study_directions SET direction='MAXIMIZE'", ()),
    )
    mutated_database(
        "extra study attribute",
        (
            "INSERT INTO study_user_attributes(study_id,key,value_json) "
            "SELECT study_id,'foreign','1' FROM studies",
            (),
        ),
    )
    mutated_database(
        "missing terminal trial",
        ("DELETE FROM trials WHERE number=99", ()),
    )
    mutated_database(
        "failed trial state",
        ("UPDATE trials SET state='FAIL' WHERE number=99", ()),
    )
    mutated_database(
        "reordered trial inventory",
        ("UPDATE trials SET number=100 WHERE number=99", ()),
    )
    mutated_database(
        "out-of-range parameter",
        (
            "UPDATE trial_params SET param_value=999 WHERE param_id=("
            "SELECT MIN(param_id) FROM trial_params)",
            (),
        ),
    )
    connection = sqlite3.connect(database)
    distribution_json = connection.execute(
        "SELECT distribution_json FROM trial_params ORDER BY param_id LIMIT 1"
    ).fetchone()[0]
    connection.close()
    changed_distribution = json.loads(distribution_json)
    changed_distribution["attributes"]["high"] = 999
    mutated_database(
        "foreign parameter distribution",
        (
            "UPDATE trial_params SET distribution_json=? WHERE param_id=("
            "SELECT MIN(param_id) FROM trial_params)",
            (json.dumps(changed_distribution),),
        ),
    )
    mutated_database(
        "noncontiguous intermediate history",
        (
            "DELETE FROM trial_intermediate_values WHERE step=1 AND trial_id=("
            "SELECT trial_id FROM trials WHERE number=0)",
            (),
        ),
    )
    mutated_database(
        "objective/history disagreement",
        (
            "UPDATE trial_values SET value=999 WHERE trial_id=("
            "SELECT trial_id FROM trials WHERE number=0)",
            (),
        ),
    )
    mutated_database(
        "non-finite objective",
        (
            "UPDATE trial_values SET value=NULL,value_type='INF_POS' WHERE trial_id=("
            "SELECT trial_id FROM trials WHERE number=0)",
            (),
        ),
    )
    mutated_database(
        "unordered timestamps",
        (
            "UPDATE trials SET datetime_complete=datetime_start WHERE number=0",
            (),
        ),
    )
    mutated_database(
        "trial user attribute",
        (
            "INSERT INTO trial_user_attributes(trial_id,key,value_json) "
            "SELECT trial_id,'foreign','1' FROM trials WHERE number=0",
            (),
        ),
    )
    mutated_database(
        "foreign pruner system attribute",
        (
            "INSERT INTO trial_system_attributes(trial_id,key,value_json) "
            "SELECT trial_id,'foreign','1' FROM trials WHERE number=0",
            (),
        ),
    )

    bad_champion = root / "bad-champion.pth"
    import torch

    torch.save({"foreign": torch.zeros(1)}, bad_champion)
    rejects(
        "champion must load against registered shape and best params",
        lambda: benchmark._semantic_study_evidence(
            database, descriptor=descriptor, champion_path=bad_champion
        ),
    )

    handcrafted = root / "handcrafted.sqlite3"
    connection = sqlite3.connect(handcrafted)
    connection.execute("CREATE TABLE trials(number INTEGER,state TEXT)")
    connection.commit()
    connection.close()
    rejects(
        "handcrafted two-column SQLite is not Optuna evidence",
        lambda: benchmark._semantic_study_evidence(
            handcrafted, descriptor=descriptor, champion_path=champion
        ),
    )

    connection = sqlite3.connect(database)
    record_json = connection.execute(
        "SELECT value_json FROM study_user_attributes WHERE key='ttbi_protocol_record'"
    ).fetchone()[0]
    capacity_json = connection.execute(
        "SELECT value_json FROM study_user_attributes "
        "WHERE key='ttbi_capacity_preflight_receipt'"
    ).fetchone()[0]
    connection.close()
    changed_record = json.loads(record_json)
    changed_record["epochs"] = 49
    mutated_database(
        "mutated protocol stamp",
        (
            "UPDATE study_user_attributes SET value_json=? "
            "WHERE key='ttbi_protocol_record'",
            (json.dumps(changed_record),),
        ),
    )
    changed_capacity = json.loads(capacity_json)
    changed_capacity["receipt_sha256"] = "f" * 64
    mutated_database(
        "mutated capacity stamp",
        (
            "UPDATE study_user_attributes SET value_json=? "
            "WHERE key='ttbi_capacity_preflight_receipt'",
            (json.dumps(changed_capacity),),
        ),
    )

    csv_path = root / "trial_compute.csv"
    rows = semantic["rows"]
    csv_path.write_bytes(benchmark._csv_bytes(rows))
    check(
        "strict objective-free canonical trial CSV round-trips",
        benchmark._read_trial_csv(csv_path) == rows
        and "value" not in benchmark.CSV_FIELDS
        and "objective" not in benchmark.CSV_FIELDS,
    )
    for label, transform in (
        (
            "failed CSV state",
            lambda text: text.replace("99,PRUNED", "99,FAIL", 1),
        ),
        (
            "non-finite CSV duration",
            lambda text: text.replace("5.000000000", "inf", 1),
        ),
        (
            "noncanonical CSV float",
            lambda text: text.replace("5.000000000", "5", 1),
        ),
        (
            "forged CSV epoch count",
            lambda text: text.replace(",3,2\n", ",2,2\n", 1),
        ),
    ):
        mutated_csv = root / f"{label.replace(' ', '-')}.csv"
        mutated_csv.write_text(
            transform(csv_path.read_text(encoding="ascii")), encoding="ascii"
        )
        rejects(label, lambda path=mutated_csv: benchmark._read_trial_csv(path))

    artifacts = {name: "d" * 64 for name in benchmark.REQUIRED_ARTIFACTS}
    evidence_root = benchmark._evidence_root(artifacts)
    durations = [float(row["duration_seconds"]) for row in rows]
    total = capacity["receipt"]["total_memory_bytes"]
    good_summary = {
        "schema": benchmark.BENCHMARK_SCHEMA,
        "classification": benchmark.CLASSIFICATION,
        "status": benchmark.STATUS,
        "tested_source_commit": descriptor["tested_source_commit"],
        "descriptor_sha256": benchmark._canonical_sha256(descriptor),
        "evidence_root_sha256": evidence_root,
        "run_directory": str(root.resolve()),
        "study_name": descriptor["study_name"],
        "protocol_hash": descriptor["protocol_hash"],
        "protocol_core_hash": descriptor["protocol_core_hash"],
        "execution_receipt_sha256": descriptor["execution_receipt_sha256"],
        "study_counts": semantic["counts"],
        "selected_trial_number": semantic["selected_trial_number"],
        "trial_duration_seconds_sum": sum(durations),
        "trial_duration_seconds_mean": sum(durations) / 100,
        "benchmark_wall_seconds": sum(durations) + 1.0,
        "peak_cuda_allocated_bytes": 100,
        "peak_cuda_reserved_bytes": 200,
        "device_total_memory_bytes": total,
        "capacity_receipt_sha256": capacity["receipt_sha256"],
        "execution_environment_sha256": capacity["receipt"][
            "execution_environment_sha256"
        ],
        "fixture_sha256": descriptor["fixture_sha256"],
        "artifact_sha256": artifacts,
        "adapter_calls": {
            "get_or_create_cache": 100,
            "canonical_train_val_split": 100,
        },
        "started_utc": "2026-08-09T12:00:00Z",
        "completed_utc": "2026-08-09T12:10:00Z",
        "objective_values_exported_to_summary_csv": False,
        "objective_values_retained_in_sqlite": True,
        "optuna_info_logging_suppressed": True,
        "progress_display_suppressed": True,
        "qualifying_run_was_fresh_uninterrupted": True,
    }
    benchmark._validate_summary(
        good_summary,
        descriptor=descriptor,
        capacity=capacity,
        directory=root.resolve(),
        artifacts=artifacts,
        evidence_root=evidence_root,
        semantic=semantic,
        rows=rows,
    )
    check("recomputed strict summary happy path accepted", True)

    def summary_mutation(label: str, mutate) -> None:
        value = deepcopy(good_summary)
        mutate(value)
        rejects(
            f"summary rejects {label}",
            lambda: benchmark._validate_summary(
                value,
                descriptor=descriptor,
                capacity=capacity,
                directory=root.resolve(),
                artifacts=artifacts,
                evidence_root=evidence_root,
                semantic=semantic,
                rows=rows,
            ),
        )

    for label, mutate in (
        (
            "boolean trial count",
            lambda value: value["study_counts"].__setitem__("TOTAL", True),
        ),
        (
            "selected trial not Optuna best",
            lambda value: value.__setitem__("selected_trial_number", 4),
        ),
        (
            "forged duration sum",
            lambda value: value.__setitem__("trial_duration_seconds_sum", 1.0),
        ),
        (
            "forged duration mean",
            lambda value: value.__setitem__("trial_duration_seconds_mean", 1.0),
        ),
        (
            "non-finite wall time",
            lambda value: value.__setitem__("benchmark_wall_seconds", math.inf),
        ),
        (
            "integer wall time",
            lambda value: value.__setitem__("benchmark_wall_seconds", 501),
        ),
        (
            "wall shorter than trials",
            lambda value: value.__setitem__("benchmark_wall_seconds", 1.0),
        ),
        (
            "zero allocated CUDA bytes",
            lambda value: value.__setitem__("peak_cuda_allocated_bytes", 0),
        ),
        (
            "allocated exceeds reserved",
            lambda value: value.__setitem__("peak_cuda_allocated_bytes", 201),
        ),
        (
            "reserved exceeds total",
            lambda value: value.__setitem__("peak_cuda_reserved_bytes", total + 1),
        ),
        (
            "device total differs from capacity",
            lambda value: value.__setitem__("device_total_memory_bytes", total - 1),
        ),
        (
            "environment SHA differs from capacity",
            lambda value: value.__setitem__("execution_environment_sha256", "e" * 64),
        ),
        (
            "adapter call count differs",
            lambda value: value["adapter_calls"].__setitem__("get_or_create_cache", 99),
        ),
        (
            "noncanonical UTC",
            lambda value: value.__setitem__("started_utc", "2026-08-09 12:00:00"),
        ),
        (
            "reversed UTC order",
            lambda value: value.__setitem__("completed_utc", "2026-08-09T11:00:00Z"),
        ),
        (
            "objective-export flag",
            lambda value: value.__setitem__("objective_values_exported_to_summary_csv", True),
        ),
        (
            "SQLite objective-retention denial",
            lambda value: value.__setitem__("objective_values_retained_in_sqlite", False),
        ),
        (
            "progress-display claim",
            lambda value: value.__setitem__("progress_display_suppressed", False),
        ),
        (
            "partial-resume claim",
            lambda value: value.__setitem__("qualifying_run_was_fresh_uninterrupted", False),
        ),
    ):
        summary_mutation(label, mutate)

    from training import pipeline

    rejects(
        "shared HPO helper rejects every pre-existing trial in fresh mode",
        lambda: pipeline.execute_registered_hpo_study(
            config=benchmark._study_config(descriptor),
            dataset_name=benchmark.STUDY_DATASET,
            storage=f"sqlite:///{database.as_posix()}",
            output_dir=str(root / "unused-output"),
            cache_dir=str(root / "unused-cache"),
            requested_n_trials=100,
            epochs=50,
            sampler_seed=benchmark.TRIAL_SEED,
            requested_use_pruner=True,
            capacity_receipt=capacity,
            require_fresh=True,
            callbacks=(),
            show_progress_bar=False,
        ),
    )


good_counts = {
    "COMPLETE": 25,
    "PRUNED": 75,
    "FAIL": 0,
    "RUNNING": 0,
    "WAITING": 0,
    "TOTAL": 100,
}
benchmark._validate_terminal_counts(good_counts)
for label, mutate in (
    ("failed state", lambda value: value.__setitem__("FAIL", 1)),
    ("99 total", lambda value: value.__setitem__("TOTAL", 99)),
    ("no complete", lambda value: (value.__setitem__("COMPLETE", 0), value.__setitem__("PRUNED", 100))),
    ("boolean count", lambda value: value.__setitem__("TOTAL", True)),
):
    changed = deepcopy(good_counts)
    mutate(changed)
    rejects(label, lambda value=changed: benchmark._validate_terminal_counts(value))

artifact_inventory = {
    name: "a" * 64 for name in benchmark.REQUIRED_ARTIFACTS
}
root_a = benchmark._evidence_root(artifact_inventory)
changed_inventory = dict(artifact_inventory)
changed_inventory[benchmark.REQUIRED_ARTIFACTS[0]] = "b" * 64
check(
    "artifact inventory and root bind every required file including attestation",
    "execution_receipt.json" in benchmark.REQUIRED_ARTIFACTS
    and root_a != benchmark._evidence_root(changed_inventory),
)
partial_inventory = dict(artifact_inventory)
partial_inventory.pop("study.sqlite3")
rejects(
    "partial artifact inventory is rejected",
    lambda: benchmark._evidence_root(partial_inventory),
)

with tempfile.TemporaryDirectory(prefix="paper1-benchmark-git-boundary-") as raw:
    git_root = Path(raw)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(git_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Benchmark Contract")
    git("config", "user.email", "benchmark-contract@example.invalid")
    (git_root / "docs").mkdir()
    report = git_root / "docs" / "audit_r5_results.md"
    foreign = git_root / "foreign.txt"
    report.write_text("blocked\n", encoding="utf-8")
    foreign.write_text("source A\n", encoding="utf-8")
    git("add", "docs/audit_r5_results.md", "foreign.txt")
    git("commit", "-m", "commit A")
    commit_a = git("rev-parse", "HEAD")
    check(
        "benchmark git boundary accepts exact clean tested A",
        benchmark._require_clean_tested_or_report_commit(git_root, commit_a)
        == commit_a,
    )
    report.write_text("authorized\n", encoding="utf-8")
    git("add", "docs/audit_r5_results.md")
    git("commit", "-m", "report-only B")
    commit_b = git("rev-parse", "HEAD")
    check(
        "benchmark git boundary accepts exact clean report-only B",
        commit_b != commit_a
        and benchmark._require_clean_tested_or_report_commit(git_root, commit_a)
        == commit_b,
    )
    foreign.write_text("dirty\n", encoding="utf-8")
    rejects(
        "benchmark git boundary rejects dirty report-only B",
        lambda: benchmark._require_clean_tested_or_report_commit(
            git_root, commit_a
        ),
    )
    foreign.write_text("source A\n", encoding="utf-8")
    foreign.write_text("foreign descendant\n", encoding="utf-8")
    git("add", "foreign.txt")
    git("commit", "-m", "foreign descendant")
    rejects(
        "benchmark git boundary rejects a foreign descendant of A",
        lambda: benchmark._require_clean_tested_or_report_commit(
            git_root, commit_a
        ),
    )

rejects(
    "authorization rejects malformed tested commit before evidence use",
    lambda: benchmark.verify_completed_receipt(ROOT, "not-a-commit", repo=ROOT),
)

if FAILURES:
    raise SystemExit(f"PAPER-1 BENCHMARK CONTRACT: {FAILURES} FAILURE(S)")
print("PAPER-1 BENCHMARK CONTRACT: ALL PASS")
