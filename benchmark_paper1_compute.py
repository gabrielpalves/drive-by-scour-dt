"""Genuine Paper-1 RAW production-path compute benchmark.

This is non-scientific qualification evidence.  It executes one fresh,
uninterrupted, registered 100-trial F40-S anchor-HPO study through the same
execution-plan, capacity, Optuna-stamping, Objective, training, and model
construction path as the campaign.  The deterministic fixture has the full
F40-S population shape (305 states x 50 passages, one RAW channel of length
5831) but contains no generator output and no transcendental construction.

Objective values are deliberately absent from the JSON/CSV/console summaries.
They necessarily remain in the authenticated Optuna SQLite database because
Optuna needs those values to define the selected champion.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before benchmark "
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

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping

REPO = Path(__file__).resolve().parent
BENCHMARK_SCHEMA = "ttbi-paper1-compute-benchmark-v2"
DESCRIPTOR_SCHEMA = "ttbi-paper1-compute-benchmark-descriptor-v2"
RUN_STATE_SCHEMA = "ttbi-paper1-compute-benchmark-run-state-v2"
AUTHORIZATION_SCHEMA = "ttbi-paper1-benchmark-authorization-evidence-v2"
COMPLETION_SCHEMA = "ttbi-paper1-compute-benchmark-completion-v2"
PROTOCOL_CORE_SCHEMA = "ttbi-paper1-benchmark-protocol-core-v1"
PROTOCOL_RUNG_SCHEMA = "ttbi-paper1-benchmark-protocol-rung-v1"
STATUS = "PASS"
CLASSIFICATION = "NON_SCIENTIFIC_COMPUTE_AND_CAPACITY_EVIDENCE"
ARCHITECTURE_ID = "RAW_POS1_LSTM1_MR1"
STAGE = "F40-S"
ACTIVE_DOFS = (1,)
# A registered campaign HPO seed is mandatory for ANCHOR_HPO_MODE.
TRIAL_SEED = 104729
N_TRIALS = 100
EPOCHS = 50
N_STATES = 305
PASSAGES_PER_STATE = 50
N_SEVERITIES = 61
RAW_LENGTH = 5831
N_CHANNELS = 1
TARGET_COUNT = 1
TRAIN_GROUP_COUNT = 183
VALIDATION_GROUP_COUNT = 61
SEALED_TEST_GROUP_COUNT = 61
STUDY_DATASET = "PAPER1_NONSCIENTIFIC_F40S_RAW_BENCHMARK"
STUDY_NAME_PREFIX = "paper1-raw-benchmark-"
CSV_FIELDS = (
    "trial_number",
    "state",
    "duration_seconds",
    "epochs_reported",
    "last_epoch_index",
)
REQUIRED_ARTIFACTS = (
    "capacity_receipt.json",
    "champion.pth",
    "descriptor.json",
    "execution_receipt.json",
    "study.sqlite3",
    "trial_compute.csv",
)
FINAL_INVENTORY = frozenset({
    *REQUIRED_ARTIFACTS,
    "summary.json",
    "run_state.json",
    "_COMPLETE",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_PRUNER_SYSTEM_ATTR = re.compile(r"^completed_rung_[0-9]+$")


class ContractError(RuntimeError):
    """Paper-1 benchmark production or verification failed closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"noncanonical benchmark JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{label} is not a regular nonsymlink file: {path}")
    return path


def _regular_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ContractError(
            f"{label} is not a regular nonsymlink directory: {path}"
        )
    return path.resolve(strict=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if os.path.lexists(temporary):
        raise ContractError(f"stale benchmark temporary file exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json_bytes(dict(value)))


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    payload = _regular_file(path, label).read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict ASCII JSON") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise ContractError(f"{label} is not canonical JSON")
    return value


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=check,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"git command failed: {args!r}") from exc


def _resolved_commit(repo: Path, value: str = "HEAD") -> str:
    commit = _git(repo, "rev-parse", "--verify", f"{value}^{{commit}}").stdout
    text = commit.decode("ascii").strip()
    if not _HEX40.fullmatch(text):
        raise ContractError("git did not return one full commit SHA")
    return text


def _require_clean_commit_a(repo: Path) -> str:
    dirty = _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if dirty:
        rendered = dirty.decode("utf-8", errors="backslashreplace").replace(
            "\0", "\n"
        ).strip()
        raise ContractError(
            "Paper-1 benchmark verification requires the exact clean tested "
            "commit A:\n" + rendered
        )
    return _resolved_commit(repo)


def _require_clean_tested_or_report_commit(
    repo: Path, tested_source_commit: str
) -> str:
    """Accept exact clean A or its sole exact report-only B descendant."""

    if not _HEX40.fullmatch(tested_source_commit):
        raise ContractError("tested source commit is malformed")
    if _resolved_commit(repo, tested_source_commit) != tested_source_commit:
        raise ContractError("tested source commit does not resolve exactly")
    head = _require_clean_commit_a(repo)
    if head == tested_source_commit:
        return head
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        tested_source_commit,
        head,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ContractError("current HEAD is not a descendant of tested commit A")
    changed = _git(
        repo,
        "diff",
        "--name-only",
        f"{tested_source_commit}..{head}",
    ).stdout
    try:
        paths = changed.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError("A..HEAD path inventory is not UTF-8") from exc
    if paths != ["docs/audit_r5_results.md"]:
        raise ContractError(
            "benchmark revalidation after commit A permits exactly one "
            "report-only descendant changing docs/audit_r5_results.md; "
            f"found {paths!r}"
        )
    return head


def _strict_int(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ContractError(f"{label} must be a {qualifier}integer")
    return value


def _strict_finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ContractError(f"{label} must be finite and positive")
    return result


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"{label} must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc or value != (
        parsed.isoformat().replace("+00:00", "Z")
    ):
        raise ContractError(f"{label} is not a canonical UTC timestamp")
    return parsed


def _current_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _architecture() -> dict[str, Any]:
    from core.paper1_training_contract import FACTORIAL_CELLS

    matches = [cell for cell in FACTORIAL_CELLS if cell.cell_id == ARCHITECTURE_ID]
    if len(matches) != 1:
        raise ContractError("registered benchmark architecture is absent")
    cell = matches[0]
    value = {
        "name_short": cell.cell_id,
        "method": cell.representation,
        "use_space2vec": cell.position_encoding,
        "use_lstm": cell.lstm,
        "use_nhits": cell.multi_rate_pooling,
        "model_type": "1D_MODULAR",
    }
    expected = {
        "name_short": ARCHITECTURE_ID,
        "method": "RAW",
        "use_space2vec": True,
        "use_lstm": True,
        "use_nhits": True,
        "model_type": "1D_MODULAR",
    }
    if value != expected:
        raise ContractError("registered largest-mechanism RAW cell drifted")
    return value


def _partition_groups() -> tuple[list[int], list[int], list[int]]:
    train = [group for group in range(N_STATES) if group % 5 in (0, 1, 2)]
    inner = [group for group in range(N_STATES) if group % 5 == 3]
    sealed = [group for group in range(N_STATES) if group % 5 == 4]
    if (
        len(train) != TRAIN_GROUP_COUNT
        or len(inner) != VALIDATION_GROUP_COUNT
        or len(sealed) != SEALED_TEST_GROUP_COUNT
        or set(train) | set(inner) | set(sealed) != set(range(N_STATES))
        or set(train) & set(inner)
        or set(train) & set(sealed)
        or set(inner) & set(sealed)
    ):
        raise ContractError("benchmark group partition drifted")
    return train, inner, sealed


def _fixture_policy() -> dict[str, Any]:
    train, inner, sealed = _partition_groups()
    return {
        "schema": "ttbi-paper1-raw-benchmark-fixture-v2",
        "classification": CLASSIFICATION,
        "stage": STAGE,
        "architecture_id": ARCHITECTURE_ID,
        "active_dofs": list(ACTIVE_DOFS),
        "shape": [N_STATES * PASSAGES_PER_STATE, N_CHANNELS, RAW_LENGTH],
        "label_shape": [N_STATES * PASSAGES_PER_STATE, TARGET_COUNT],
        "group_count": N_STATES,
        "passages_per_group": PASSAGES_PER_STATE,
        "severity_count": N_SEVERITIES,
        "train_groups": train,
        "inner_validation_groups": inner,
        "sealed_unused_test_groups": sealed,
        "split_fractions": [0.6, 0.2, 0.2],
        "row_order": "train groups, inner-validation groups, sealed-test groups",
        "test_partition_exposed_to_objective": False,
        "construction": (
            "float32((37*raw_coordinate_index + 53*state + 97*passage) "
            "mod 2048 - 1024) / 1024; exact binary arithmetic, no "
            "transcendental functions, no generator output"
        ),
        "label_construction": "float32(floor(state/5)); values 0..60",
    }


def _fixture_arrays():
    import numpy as np

    sample_count = N_STATES * PASSAGES_PER_STATE
    x = np.empty((sample_count, N_CHANNELS, RAW_LENGTH), dtype=np.float32)
    train_groups, inner_groups, sealed_groups = _partition_groups()
    # Partition-major row order makes the development population one
    # zero-copy prefix.  The trainer adapter can therefore withhold sealed
    # inputs and labels entirely, rather than merely excluding their indices.
    ordered_groups = np.asarray(
        [*train_groups, *inner_groups, *sealed_groups], dtype=np.int64
    )
    groups = np.repeat(ordered_groups, PASSAGES_PER_STATE)
    passages = np.tile(
        np.arange(PASSAGES_PER_STATE, dtype=np.uint32), N_STATES
    )
    y = (groups // 5).astype(np.float32).reshape(sample_count, TARGET_COUNT)
    coordinate = np.arange(RAW_LENGTH, dtype=np.uint32)
    # Bounded chunks avoid a second full-size allocation.  Division by 1024
    # and the integer range are exactly representable in float32.
    for first in range(0, sample_count, 64):
        last = min(first + 64, sample_count)
        state_term = groups[first:last].astype(np.uint32)[:, None] * 53
        passage_term = passages[first:last, None] * 97
        code = (
            coordinate[None, :] * 37 + state_term + passage_term
        ) & np.uint32(2047)
        x[first:last, 0, :] = (
            code.astype(np.float32) - np.float32(1024.0)
        ) / np.float32(1024.0)
    if (
        x.shape != (15250, 1, 5831)
        or y.shape != (15250, 1)
        or len(np.unique(groups)) != 305
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(61, dtype=np.float32))
    ):
        raise ContractError("constructed benchmark fixture is malformed")
    return x, y, groups


def _array_digest(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    descriptor = {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)}
    digest = hashlib.sha256(_canonical_json_bytes(descriptor))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _fixture_digest(arrays: tuple[Any, Any, Any] | None = None) -> str:
    x, y, groups = _fixture_arrays() if arrays is None else arrays
    return _canonical_sha256({
        "policy": _fixture_policy(),
        "x_sha256": _array_digest(x),
        "y_sha256": _array_digest(y),
        "groups_sha256": _array_digest(groups),
    })


def _benchmark_policy() -> dict[str, Any]:
    from core.capacity_preflight import CAPACITY_PREFLIGHT_POLICY
    from core.execution_environment import EXECUTION_BLOCK_POLICY
    from core.hyperparameter_policy import HYPERPARAMETER_POLICY
    from core.protocol import OPTUNA_PROTOCOL
    from training.trainer import SEARCH_SPACE, TRAIN_PROTOCOL

    return {
        "schema": "ttbi-paper1-compute-benchmark-policy-v2",
        "classification": CLASSIFICATION,
        "architecture": _architecture(),
        "active_dofs": list(ACTIVE_DOFS),
        "study": {
            "trials": N_TRIALS,
            "epochs": EPOCHS,
            "seed": TRIAL_SEED,
            "mode": "anchor_hpo",
            "registered_pruner": True,
            "fresh_uninterrupted_only": True,
            "failure_policy": (
                "any pre-existing trial or any FAIL/RUNNING/WAITING trial is "
                "fatal; choose a new output directory; no partial resume claim"
            ),
        },
        "fixture": _fixture_policy(),
        "trainer_protocol": TRAIN_PROTOCOL,
        "search_space": SEARCH_SPACE,
        "optuna_protocol": OPTUNA_PROTOCOL,
        "hyperparameter_policy": HYPERPARAMETER_POLICY,
        "execution_block_policy": EXECUTION_BLOCK_POLICY,
        "capacity_preflight_policy": CAPACITY_PREFLIGHT_POLICY,
        "execution_path": [
            "training.pipeline.execute_registered_hpo_study",
            "core.hyperparameter_policy.derive_execution_plan",
            "training.pipeline._stamp_study_protocol",
            "training.trainer.Objective",
            "training.trainer.train_and_evaluate",
            "core.models.build_model",
        ],
        "reporting": {
            "objective_values_in_json_csv_console": False,
            "objective_values_necessarily_retained_in_authenticated_sqlite": True,
            "optuna_info_logging": False,
            "progress_bar": False,
        },
    }


def _expected_execution_receipt(
    *, runtime: Mapping[str, Any], protocol_core_hash: str, run_tag: str
) -> dict[str, Any]:
    return {
        "schema": "ttbi-execution-block-receipt-v2",
        "execution_block": runtime["execution_block"],
        "anchor_stage": runtime["anchor_stage"],
        "protocol_core_hash": protocol_core_hash,
        "run_tag": run_tag,
        "execution_compatibility_sha256": (
            runtime["execution_compatibility_sha256"]
        ),
        "execution_compatibility_descriptor": (
            runtime["execution_compatibility_descriptor"]
        ),
    }


def _expected_descriptor(
    *,
    tested_source_commit: str,
    source_snapshot: Any,
    capacity_envelope: Mapping[str, Any],
    environment_lock: Mapping[str, Any],
    fixture_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from core.execution_environment import (
        EXECUTION_BLOCK_POLICY,
        execution_block_for_stage,
        validate_execution_runtime,
    )
    from core.protocol import protocol_hash

    capacity = dict(capacity_envelope)
    runtime = validate_execution_runtime(
        capacity["receipt"]["execution_runtime"]
    )
    block, anchor = execution_block_for_stage(STAGE, EXECUTION_BLOCK_POLICY)
    if (runtime["execution_block"], runtime["anchor_stage"]) != (block, anchor):
        raise ContractError("capacity receipt is not bound to the F40-S block")
    policy = _benchmark_policy()
    core = {
        "schema": PROTOCOL_CORE_SCHEMA,
        "classification": CLASSIFICATION,
        "execution_blocking": EXECUTION_BLOCK_POLICY,
        "benchmark_policy": policy,
        "fixture_sha256": fixture_sha256,
        "tested_source_commit": tested_source_commit,
        "generator_source_root_sha256": source_snapshot.generator.sha256,
        "generator_source_file_count": source_snapshot.generator.file_count,
        "python_runtime_source_root_sha256": source_snapshot.python_runtime.sha256,
        "python_runtime_source_file_count": source_snapshot.python_runtime.file_count,
        "environment_lock_sha256": environment_lock["sha256"],
        "capacity_receipt_sha256": capacity["receipt_sha256"],
    }
    rung = {
        "schema": PROTOCOL_RUNG_SCHEMA,
        "stage": STAGE,
        "dataset": STUDY_DATASET,
        "execution_block": block,
        "execution_anchor": anchor,
    }
    protocol_descriptor = {"core": core, "rung": rung}
    core_hash = protocol_hash(core)
    full_hash = protocol_hash(protocol_descriptor)
    run_tag = (
        f"paper1-benchmark-{tested_source_commit[:12]}-{core_hash[:12]}"
    )
    execution_receipt = _expected_execution_receipt(
        runtime=runtime,
        protocol_core_hash=core_hash,
        run_tag=run_tag,
    )
    execution_receipt_sha = hashlib.sha256(
        _canonical_json_bytes(execution_receipt)
    ).hexdigest()
    descriptor = {
        "schema": DESCRIPTOR_SCHEMA,
        "classification": CLASSIFICATION,
        "tested_source_commit": tested_source_commit,
        "generator_source_root_sha256": source_snapshot.generator.sha256,
        "generator_source_file_count": source_snapshot.generator.file_count,
        "python_runtime_source_root_sha256": source_snapshot.python_runtime.sha256,
        "python_runtime_source_file_count": source_snapshot.python_runtime.file_count,
        "environment_lock_sha256": environment_lock["sha256"],
        "capacity_receipt_sha256": capacity["receipt_sha256"],
        "execution_runtime": runtime,
        "execution_receipt_sha256": execution_receipt_sha,
        "campaign_run_tag": run_tag,
        "fixture_sha256": fixture_sha256,
        "policy": policy,
        "policy_sha256": _canonical_sha256(policy),
        "protocol_descriptor": protocol_descriptor,
        "protocol_core_hash": core_hash,
        "protocol_hash": full_hash,
        "study_name": f"{STUDY_NAME_PREFIX}{full_hash[:16]}",
    }
    return descriptor, execution_receipt


def _study_config(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    from core.hyperparameter_policy import ANCHOR_HPO_MODE

    return {
        "name": descriptor["study_name"],
        "seed": TRIAL_SEED,
        "sensor_noise": None,
        **_architecture(),
        "dofs": list(ACTIVE_DOFS),
        "discretization": 1,
        "task": "regression",
        "target_supports": [2],
        "bearing_targets": None,
        "protocol_hash": descriptor["protocol_hash"],
        "protocol_core_hash": descriptor["protocol_core_hash"],
        "protocol_descriptor": descriptor["protocol_descriptor"],
        "execution_runtime": descriptor["execution_runtime"],
        "campaign_run_tag": descriptor["campaign_run_tag"],
        "execution_receipt_sha256": descriptor["execution_receipt_sha256"],
        "block_reference_manifest_sha256": None,
        "hyperparameter_mode": ANCHOR_HPO_MODE,
    }


@contextmanager
def _patched_trainer_fixture(
    trainer: Any,
    x: Any,
    y: Any,
    groups: Any,
    config: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    import numpy as np

    original_cache = trainer.get_or_create_cache
    original_split = trainer.canonical_train_val_split
    calls = {"get_or_create_cache": 0, "canonical_train_val_split": 0}
    train_groups, inner_groups, sealed_groups = _partition_groups()
    train_mask = np.isin(groups, np.asarray(train_groups, dtype=np.int64))
    inner_mask = np.isin(groups, np.asarray(inner_groups, dtype=np.int64))
    sealed_mask = np.isin(groups, np.asarray(sealed_groups, dtype=np.int64))
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    val_idx = np.flatnonzero(inner_mask).astype(np.int64)
    sealed_idx = np.flatnonzero(sealed_mask).astype(np.int64)
    development_count = (
        TRAIN_GROUP_COUNT + VALIDATION_GROUP_COUNT
    ) * PASSAGES_PER_STATE
    development_x = x[:development_count]
    development_y = y[:development_count]
    development_groups = groups[:development_count]
    if (
        train_idx.size != TRAIN_GROUP_COUNT * PASSAGES_PER_STATE
        or val_idx.size != VALIDATION_GROUP_COUNT * PASSAGES_PER_STATE
        or sealed_idx.size != SEALED_TEST_GROUP_COUNT * PASSAGES_PER_STATE
        or np.intersect1d(train_idx, val_idx).size
        or np.intersect1d(train_idx, sealed_idx).size
        or np.intersect1d(val_idx, sealed_idx).size
        or not np.array_equal(
            development_groups,
            np.concatenate((
                np.repeat(np.asarray(train_groups), PASSAGES_PER_STATE),
                np.repeat(np.asarray(inner_groups), PASSAGES_PER_STATE),
            )),
        )
    ):
        raise ContractError("benchmark sample split is malformed")

    def benchmark_cache(
        observed_config: Mapping[str, Any],
        dataset_name: str,
        cache_dir: str,
    ):
        del cache_dir
        if dataset_name != STUDY_DATASET or dict(observed_config) != dict(config):
            raise ContractError("trainer requested a foreign benchmark fixture")
        calls["get_or_create_cache"] += 1
        return development_x, development_y, None, development_groups

    def benchmark_split(
        n_samples: int,
        observed_groups: Any = None,
        seed: int = 42,
        dataset_name: str | None = None,
    ):
        if (
            n_samples != len(development_y)
            or seed != 42
            or dataset_name != STUDY_DATASET
            or observed_groups is None
            or not np.array_equal(observed_groups, development_groups)
        ):
            raise ContractError("trainer requested a foreign benchmark split")
        calls["canonical_train_val_split"] += 1
        # sealed_idx and its arrays are deliberately unavailable to the
        # selection objective; train/val indices are relative to the exact
        # development-only prefix returned above.
        return train_idx.copy(), val_idx.copy()

    trainer.get_or_create_cache = benchmark_cache
    trainer.canonical_train_val_split = benchmark_split
    try:
        yield calls
    finally:
        trainer.get_or_create_cache = original_cache
        trainer.canonical_train_val_split = original_split


def _validate_terminal_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    expected_keys = {"COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING", "TOTAL"}
    if not isinstance(counts, Mapping) or set(counts) != expected_keys:
        raise ContractError("benchmark terminal-count fields differ")
    result = {
        key: _strict_int(counts[key], f"study_counts.{key}")
        for key in expected_keys
    }
    if (
        result["TOTAL"] != N_TRIALS
        or result["COMPLETE"] < 1
        or result["COMPLETE"] + result["PRUNED"] != N_TRIALS
        or any(result[name] for name in ("FAIL", "RUNNING", "WAITING"))
    ):
        raise ContractError(f"benchmark study is not exactly terminal: {result!r}")
    return result


class _DistributionReplay:
    """Replay ``_suggest_params`` against stored params/distributions."""

    def __init__(self, params: Mapping[str, Any]):
        self.params = dict(params)
        self.distributions: dict[str, Any] = {}

    def _take(self, name: str, distribution: Any) -> Any:
        if name not in self.params or name in self.distributions:
            raise ContractError(f"trial parameter replay failed at {name!r}")
        value = self.params[name]
        if not distribution._contains(distribution.to_internal_repr(value)):
            raise ContractError(f"trial parameter {name!r} is outside its distribution")
        self.distributions[name] = distribution
        return value

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        import optuna

        return self._take(
            name, optuna.distributions.IntDistribution(low, high, log=log, step=step)
        )

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        import optuna

        return self._take(
            name,
            optuna.distributions.FloatDistribution(
                low, high, log=log, step=step
            ),
        )

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        import optuna

        return self._take(
            name, optuna.distributions.CategoricalDistribution(tuple(choices))
        )


def _trial_param_type(value: Any, distribution: Any, label: str) -> None:
    import optuna

    if isinstance(distribution, optuna.distributions.IntDistribution):
        if type(value) is not int:
            raise ContractError(f"{label} is not a strict integer")
    elif isinstance(distribution, optuna.distributions.FloatDistribution):
        if type(value) is not float or not math.isfinite(value):
            raise ContractError(f"{label} is not a finite strict float")
    elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"{label} is a non-finite category")
    else:
        raise ContractError(f"{label} uses an unregistered distribution type")


def _validate_champion_state(
    path: Path,
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    import torch
    from core.models import build_model

    _regular_file(path, "benchmark champion weights")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError("benchmark champion state_dict is unreadable") from exc
    if not isinstance(state, Mapping) or not state:
        raise ContractError("benchmark champion is not a nonempty state_dict")
    model, n_outputs = build_model(
        dict(config),
        dict(params),
        (N_STATES * PASSAGES_PER_STATE, N_CHANNELS, RAW_LENGTH),
        torch.device("cpu"),
    )
    if n_outputs != TARGET_COUNT:
        raise ContractError("registered benchmark model has the wrong output shape")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ContractError("champion state_dict keys differ from the registered model")
    for name, tensor in state.items():
        expected_tensor = expected[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
        ):
            raise ContractError(f"champion tensor {name!r} shape/dtype differs")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ContractError(f"champion tensor {name!r} contains non-finite values")
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise ContractError("champion state_dict cannot load strictly") from exc


@contextmanager
def _copied_optuna_database(database: Path):
    """Yield one disposable real RDB storage and always close its engine."""

    import optuna

    original_sha = _file_sha256(
        _regular_file(database, "Optuna SQLite database")
    )
    temporary = tempfile.TemporaryDirectory(
        prefix="paper1-benchmark-optuna-", ignore_cleanup_errors=True
    )
    rdb_storage = None
    try:
        copied = Path(temporary.name) / "study.sqlite3"
        shutil.copyfile(database, copied)
        if _file_sha256(copied) != original_sha:
            raise ContractError("Optuna SQLite copy changed during verification")
        storage = f"sqlite:///{copied.as_posix()}"
        rdb_storage = optuna.storages.RDBStorage(url=storage)
        yield rdb_storage, original_sha
    finally:
        if rdb_storage is not None:
            rdb_storage.remove_session()
            rdb_storage.engine.dispose()
        temporary.cleanup()


def _semantic_study_evidence(
    database: Path,
    *,
    descriptor: Mapping[str, Any],
    champion_path: Path | None,
) -> dict[str, Any]:
    """Validate a copied real Optuna study and derive authenticated rows."""

    import optuna
    from training import pipeline, trainer

    with _copied_optuna_database(database) as (rdb_storage, original_sha):
        try:
            summaries = optuna.study.get_all_study_summaries(storage=rdb_storage)
            if len(summaries) != 1:
                raise ContractError("Optuna database must contain exactly one study")
            summary = summaries[0]
            if (
                summary.study_name != descriptor["study_name"]
                or summary.direction is not optuna.study.StudyDirection.MINIMIZE
            ):
                raise ContractError("Optuna study name/direction differs")
            study = optuna.load_study(
                study_name=descriptor["study_name"], storage=rdb_storage
            )
        except (KeyError, RuntimeError, ValueError, optuna.exceptions.OptunaError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError("Optuna database schema/study is invalid") from exc

        if study.direction is not optuna.study.StudyDirection.MINIMIZE:
            raise ContractError("Optuna study direction is not minimize")
        study_system_attrs = study._storage.get_study_system_attrs(
            study._study_id
        )
        if study_system_attrs:
            raise ContractError("Optuna study carries unexpected system attributes")
        if set(study.user_attrs) != {
            "ttbi_protocol_record", "ttbi_capacity_preflight_receipt"
        }:
            raise ContractError("Optuna study attribute inventory differs")
        config = _study_config(descriptor)
        try:
            plan, capacity = pipeline._validated_study_hyperparameter_record(
                study, config
            )
        except (RuntimeError, ValueError) as exc:
            raise ContractError("Optuna protocol/capacity stamp is invalid") from exc
        record = study.user_attrs["ttbi_protocol_record"]
        if (
            record.get("dataset") != STUDY_DATASET
            or record.get("model_name") != descriptor["study_name"]
            or record.get("sampler_seed") != TRIAL_SEED
            or record.get("seed") != TRIAL_SEED
            or record.get("n_trials") != N_TRIALS
            or record.get("epochs") != EPOCHS
            or record.get("use_pruner") is not True
        ):
            raise ContractError("Optuna study protocol scalar fields differ")
        if (
            plan["effective_n_trials"] != N_TRIALS
            or plan["effective_use_pruner"] is not True
            or capacity["receipt_sha256"] != descriptor["capacity_receipt_sha256"]
        ):
            raise ContractError("Optuna derived plan/capacity binding differs")

        trials = study.get_trials(deepcopy=True)
        if [trial.number for trial in trials] != list(range(N_TRIALS)):
            raise ContractError("Optuna trials are missing, duplicated, or reordered")
        counts = {
            name: sum(
                trial.state is getattr(optuna.trial.TrialState, name)
                for trial in trials
            )
            for name in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
        }
        counts["TOTAL"] = len(trials)
        counts = _validate_terminal_counts(counts)
        rows: list[dict[str, str]] = []
        for trial in trials:
            label = f"trial {trial.number}"
            if trial.state not in {
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            }:
                raise ContractError(f"{label} is not terminal-useful")
            if trial.datetime_start is None or trial.datetime_complete is None:
                raise ContractError(f"{label} lacks start/completion timestamps")
            if trial.datetime_complete <= trial.datetime_start:
                raise ContractError(f"{label} timestamps are not ordered")
            duration = trial.duration
            if duration is None:
                raise ContractError(f"{label} lacks duration")
            duration_seconds = _strict_finite(
                duration.total_seconds(), f"{label} duration", positive=True
            )
            delta = (trial.datetime_complete - trial.datetime_start).total_seconds()
            if duration_seconds != delta:
                raise ContractError(f"{label} duration disagrees with timestamps")
            if type(trial.number) is not int:
                raise ContractError(f"{label} number is not a strict integer")
            if trial.user_attrs:
                raise ContractError(f"{label} carries unexpected user attributes")
            if any(
                not _PRUNER_SYSTEM_ATTR.fullmatch(key)
                or type(value) is not float
                or not math.isfinite(value)
                for key, value in trial.system_attrs.items()
            ):
                raise ContractError(f"{label} carries invalid pruner attributes")
            replay = _DistributionReplay(trial.params)
            try:
                replayed = trainer._suggest_params(replay, config)
            except (KeyError, TypeError, ValueError, ContractError) as exc:
                raise ContractError(f"{label} search-space replay failed") from exc
            if replayed != trial.params or replay.distributions != trial.distributions:
                raise ContractError(f"{label} params/distributions differ from policy")
            for name, distribution in trial.distributions.items():
                _trial_param_type(
                    trial.params[name], distribution, f"{label} param {name!r}"
                )
            intermediates = trial.intermediate_values
            steps = list(intermediates)
            if (
                not steps
                or steps != list(range(len(steps)))
                or steps[-1] >= EPOCHS
                or any(
                    type(value) is not float
                    or not math.isfinite(value)
                    for value in intermediates.values()
                )
            ):
                raise ContractError(f"{label} intermediate-value history is invalid")
            value = trial.value
            if (
                type(value) is not float
                or not math.isfinite(value)
            ):
                raise ContractError(f"{label} objective value is not finite")
            if trial.state is optuna.trial.TrialState.COMPLETE:
                expected_value = min(intermediates.values())
            else:
                expected_value = intermediates[steps[-1]]
            if value != expected_value:
                raise ContractError(f"{label} objective disagrees with its history")
            rows.append({
                "trial_number": str(trial.number),
                "state": trial.state.name,
                "duration_seconds": format(duration_seconds, ".9f"),
                "epochs_reported": str(len(steps)),
                "last_epoch_index": str(steps[-1]),
            })

        try:
            best = study.best_trial
        except ValueError as exc:
            raise ContractError("Optuna study has no complete champion") from exc
        if best.state is not optuna.trial.TrialState.COMPLETE:
            raise ContractError("selected Optuna champion is not COMPLETE")
        best_number = int(best.number)
        if champion_path is not None:
            _validate_champion_state(
                champion_path, config=config, params=best.params
            )
        storage_object = getattr(study, "_storage", None)
        if storage_object is not None and hasattr(storage_object, "remove_session"):
            storage_object.remove_session()
    if _file_sha256(database) != original_sha:
        raise ContractError("Optuna SQLite changed during semantic verification")
    return {
        "counts": counts,
        "rows": rows,
        "selected_trial_number": best_number,
        "plan": plan,
    }


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    try:
        return stream.getvalue().encode("ascii")
    except UnicodeEncodeError as exc:
        raise ContractError("trial CSV is not ASCII") from exc


def _validate_trial_rows(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list) or len(rows) != N_TRIALS:
        raise ContractError("trial compute CSV does not contain 100 rows")
    normalized: list[dict[str, str]] = []
    for expected, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != set(CSV_FIELDS):
            raise ContractError("trial compute CSV row fields differ")
        row = {key: raw[key] for key in CSV_FIELDS}
        if (
            row["trial_number"] != str(expected)
            or row["state"] not in {"COMPLETE", "PRUNED"}
            or not row["epochs_reported"].isdigit()
            or not row["last_epoch_index"].isdigit()
        ):
            raise ContractError("trial compute CSV identity/state/history differs")
        epochs = int(row["epochs_reported"])
        last = int(row["last_epoch_index"])
        if not 1 <= epochs <= EPOCHS or last != epochs - 1:
            raise ContractError("trial compute CSV epoch history is invalid")
        try:
            duration = float(row["duration_seconds"])
        except ValueError as exc:
            raise ContractError("trial compute CSV duration is invalid") from exc
        if (
            not math.isfinite(duration)
            or duration <= 0.0
            or row["duration_seconds"] != format(duration, ".9f")
        ):
            raise ContractError("trial compute CSV duration is not canonical")
        normalized.append(row)
    return normalized


def _read_trial_csv(path: Path) -> list[dict[str, str]]:
    payload = _regular_file(path, "trial compute CSV").read_bytes()
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ContractError("trial compute CSV is not ASCII") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != CSV_FIELDS:
        raise ContractError("trial compute CSV header differs")
    rows = _validate_trial_rows(list(reader))
    if payload != _csv_bytes(rows):
        raise ContractError("trial compute CSV is not canonical")
    return rows


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {
        name: _file_sha256(_regular_file(run_dir / name, name))
        for name in REQUIRED_ARTIFACTS
    }


def _evidence_root(artifacts: Mapping[str, str]) -> str:
    if set(artifacts) != set(REQUIRED_ARTIFACTS) or any(
        not _HEX64.fullmatch(value) for value in artifacts.values()
    ):
        raise ContractError("benchmark artifact hash inventory is malformed")
    return _canonical_sha256(dict(artifacts))


def _assert_final_inventory(run_dir: Path) -> None:
    names = {item.name for item in run_dir.iterdir()}
    if names != FINAL_INVENTORY:
        raise ContractError(
            f"completed benchmark artifact inventory differs: {sorted(names)!r}"
        )
    for item in run_dir.iterdir():
        _regular_file(item, f"completed benchmark artifact {item.name}")


def _outside_repository(directory: Path, repository: Path) -> None:
    try:
        directory.relative_to(repository)
    except ValueError:
        return
    raise ContractError("benchmark run directory must be outside the repository")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any],
    capacity: Mapping[str, Any],
    directory: Path,
    artifacts: Mapping[str, str],
    evidence_root: str,
    semantic: Mapping[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    required = {
        "schema",
        "classification",
        "status",
        "tested_source_commit",
        "descriptor_sha256",
        "evidence_root_sha256",
        "run_directory",
        "study_name",
        "protocol_hash",
        "protocol_core_hash",
        "execution_receipt_sha256",
        "study_counts",
        "selected_trial_number",
        "trial_duration_seconds_sum",
        "trial_duration_seconds_mean",
        "benchmark_wall_seconds",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "device_total_memory_bytes",
        "capacity_receipt_sha256",
        "execution_environment_sha256",
        "fixture_sha256",
        "artifact_sha256",
        "adapter_calls",
        "started_utc",
        "completed_utc",
        "objective_values_exported_to_summary_csv",
        "objective_values_retained_in_sqlite",
        "optuna_info_logging_suppressed",
        "progress_display_suppressed",
        "qualifying_run_was_fresh_uninterrupted",
    }
    if not isinstance(summary, Mapping) or set(summary) != required:
        raise ContractError("benchmark summary fields differ")
    descriptor_sha = _canonical_sha256(descriptor)
    expected_identity = {
        "schema": BENCHMARK_SCHEMA,
        "classification": CLASSIFICATION,
        "status": STATUS,
        "tested_source_commit": descriptor["tested_source_commit"],
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "run_directory": str(directory),
        "study_name": descriptor["study_name"],
        "protocol_hash": descriptor["protocol_hash"],
        "protocol_core_hash": descriptor["protocol_core_hash"],
        "execution_receipt_sha256": descriptor["execution_receipt_sha256"],
        "capacity_receipt_sha256": capacity["receipt_sha256"],
        "execution_environment_sha256": capacity["receipt"][
            "execution_environment_sha256"
        ],
        "fixture_sha256": descriptor["fixture_sha256"],
        "artifact_sha256": dict(artifacts),
        "objective_values_exported_to_summary_csv": False,
        "objective_values_retained_in_sqlite": True,
        "optuna_info_logging_suppressed": True,
        "progress_display_suppressed": True,
        "qualifying_run_was_fresh_uninterrupted": True,
    }
    mismatches = {
        key: (summary.get(key), expected)
        for key, expected in expected_identity.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"benchmark summary identity/provenance differs: {mismatches}")
    counts = _validate_terminal_counts(summary["study_counts"])
    if counts != semantic["counts"]:
        raise ContractError("summary counts differ from real Optuna states")
    selected = _strict_int(
        summary["selected_trial_number"], "selected_trial_number"
    )
    if selected != semantic["selected_trial_number"]:
        raise ContractError("summary selection differs from Optuna best_trial")
    durations = [float(row["duration_seconds"]) for row in rows]
    duration_sum = sum(durations)
    duration_mean = duration_sum / N_TRIALS
    for key in (
        "trial_duration_seconds_sum",
        "trial_duration_seconds_mean",
        "benchmark_wall_seconds",
    ):
        if type(summary[key]) is not float:
            raise ContractError(f"benchmark summary field {key} must be a float")
    if (
        _strict_finite(
            summary["trial_duration_seconds_sum"],
            "trial duration sum",
            positive=True,
        )
        != duration_sum
        or _strict_finite(
            summary["trial_duration_seconds_mean"],
            "trial duration mean",
            positive=True,
        )
        != duration_mean
    ):
        raise ContractError("summary duration aggregates were not recomputed")
    wall = _strict_finite(
        summary["benchmark_wall_seconds"], "benchmark wall time", positive=True
    )
    if wall + 1e-6 < duration_sum:
        raise ContractError("benchmark wall time is shorter than summed trials")
    allocated = _strict_int(
        summary["peak_cuda_allocated_bytes"],
        "peak CUDA allocated bytes",
        positive=True,
    )
    reserved = _strict_int(
        summary["peak_cuda_reserved_bytes"],
        "peak CUDA reserved bytes",
        positive=True,
    )
    total = _strict_int(
        summary["device_total_memory_bytes"],
        "CUDA device total bytes",
        positive=True,
    )
    capacity_total = capacity["receipt"]["total_memory_bytes"]
    runtime_total = capacity["receipt"]["execution_runtime"][
        "execution_environment_descriptor"
    ]["accelerator"]["total_memory_bytes"]
    if not (0 < allocated <= reserved <= total == capacity_total == runtime_total):
        raise ContractError("benchmark CUDA memory arithmetic/binding differs")
    if summary["adapter_calls"] != {
        "get_or_create_cache": N_TRIALS,
        "canonical_train_val_split": N_TRIALS,
    }:
        raise ContractError("benchmark trainer-adapter call ledger differs")
    started = _utc(summary["started_utc"], "benchmark started_utc")
    completed = _utc(summary["completed_utc"], "benchmark completed_utc")
    if completed <= started:
        raise ContractError("benchmark UTC timestamps are not ordered")
    return dict(summary)


def _registered_execution_environment() -> dict[str, Any]:
    """Apply the training numeric policy before capturing live identity."""

    from core.execution_environment import current_execution_environment
    from core.utils import set_global_seed
    from training.trainer import TRAIN_PROTOCOL

    set_global_seed(TRIAL_SEED, TRAIN_PROTOCOL["determinism"])
    return current_execution_environment()


def run_benchmark(
    output_dir: str | os.PathLike[str],
    capacity_receipt: str | os.PathLike[str],
) -> dict[str, Any]:
    """Execute one fresh qualifying benchmark and publish its receipt."""

    from core.capacity_preflight import load_capacity_receipt
    from core.environment import load_environment_lock, validate_environment_lock
    from core.execution_environment import (
        EXECUTION_BLOCK_POLICY,
        current_execution_environment,
        enforce_execution_block,
        execution_environment_sha256,
    )
    from core.source_provenance import repository_source_snapshot

    tested_commit = _require_clean_commit_a(REPO)
    source_snapshot = repository_source_snapshot(REPO)
    environment_lock = load_environment_lock(
        REPO / "environment" / "campaign-py313-cu128.json"
    )
    validate_environment_lock(environment_lock)
    capacity_path = Path(capacity_receipt)
    if not capacity_path.is_absolute():
        raise ContractError("capacity receipt path must be absolute")
    capacity_path = _regular_file(capacity_path, "capacity receipt").resolve(
        strict=True
    )
    capacity = load_capacity_receipt(
        capacity_path,
        expected_source_root_sha256=source_snapshot.python_runtime.sha256,
        expected_source_file_count=source_snapshot.python_runtime.file_count,
    )
    current_environment_sha = execution_environment_sha256(
        _registered_execution_environment()
    )
    if capacity["receipt"]["execution_environment_sha256"] != current_environment_sha:
        raise ContractError(
            "benchmark must run on the exact CUDA device/runtime that produced "
            "the bound 16-cell capacity receipt"
        )

    raw_output = Path(output_dir)
    if not raw_output.is_absolute() or raw_output.name in {"", ".", ".."}:
        raise ContractError("benchmark output directory must be one absolute child")
    resolved_parent = _regular_directory(raw_output.parent, "output parent")
    run_dir = resolved_parent / raw_output.name
    _outside_repository(run_dir, REPO.resolve(strict=True))
    if os.path.lexists(run_dir):
        _regular_directory(run_dir, "benchmark run directory")
        raise ContractError(
            "qualifying benchmarks are always fresh: the output directory "
            "already exists, so any trials there are ineligible for this run; "
            "preserve it for verification/diagnosis and choose a new directory"
        )

    arrays = _fixture_arrays()
    fixture_sha = _fixture_digest(arrays)
    descriptor, expected_execution_receipt = _expected_descriptor(
        tested_source_commit=tested_commit,
        source_snapshot=source_snapshot,
        capacity_envelope=capacity,
        environment_lock=environment_lock,
        fixture_sha256=fixture_sha,
    )
    descriptor_sha = _canonical_sha256(descriptor)
    with tempfile.TemporaryDirectory(
        prefix=".paper1-benchmark-attestation-", dir=resolved_parent
    ) as receipt_temp:
        attestation = enforce_execution_block(
            stage=STAGE,
            policy=EXECUTION_BLOCK_POLICY,
            protocol_core_hash=descriptor["protocol_core_hash"],
            run_tag=descriptor["campaign_run_tag"],
            receipt_dir=receipt_temp,
        )
        attested_path = _regular_file(
            Path(attestation["receipt_path"]), "execution attestation"
        )
        attested_payload = attested_path.read_bytes()
        if (
            attestation["runtime"] != descriptor["execution_runtime"]
            or attestation["receipt_sha256"]
            != descriptor["execution_receipt_sha256"]
            or attested_payload != _canonical_json_bytes(expected_execution_receipt)
        ):
            raise ContractError("live execution-block attestation differs")

    run_dir.mkdir()
    _atomic_json(run_dir / "descriptor.json", descriptor)
    _atomic_bytes(run_dir / "capacity_receipt.json", capacity_path.read_bytes())
    _atomic_bytes(run_dir / "execution_receipt.json", attested_payload)
    database = run_dir / "study.sqlite3"
    if os.path.lexists(database):
        raise ContractError("fresh benchmark database path already exists")
    weights_dir = run_dir / "weights_incomplete"
    weights_dir.mkdir()

    import optuna
    import torch
    from training import pipeline, trainer

    config = _study_config(descriptor)
    started_utc = _current_utc()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise ContractError("Paper-1 benchmark requires the capacity-bound CUDA GPU")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    prior_verbosity = optuna.logging.get_verbosity()
    with _patched_trainer_fixture(trainer, *arrays, config) as calls:
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study, plan, observed_capacity = pipeline.execute_registered_hpo_study(
                config=config,
                dataset_name=STUDY_DATASET,
                storage=f"sqlite:///{database.as_posix()}",
                output_dir=str(weights_dir),
                cache_dir=str(run_dir / "unused_cache_adapter"),
                requested_n_trials=N_TRIALS,
                epochs=EPOCHS,
                sampler_seed=TRIAL_SEED,
                requested_use_pruner=True,
                capacity_receipt=capacity,
                require_fresh=True,
                callbacks=(),
                show_progress_bar=False,
            )
        finally:
            optuna.logging.set_verbosity(prior_verbosity)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    device_total = int(
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    )
    if (
        plan["effective_n_trials"] != N_TRIALS
        or plan["effective_use_pruner"] is not True
        or observed_capacity != capacity
    ):
        raise ContractError("registered HPO helper returned foreign plan/capacity")
    if calls != {
        "get_or_create_cache": N_TRIALS,
        "canonical_train_val_split": N_TRIALS,
    }:
        raise ContractError("benchmark objective did not call the exact adapter ledger")

    try:
        champion_number = int(study.best_trial.number)
    except ValueError as exc:
        raise ContractError("fresh benchmark produced no complete champion") from exc
    champion_source = weights_dir / (
        f"weights_{config['name']}_trial_{champion_number}.pth"
    )
    _regular_file(champion_source, "benchmark champion weights")
    shutil.copyfile(champion_source, run_dir / "champion.pth")
    storage_object = getattr(study, "_storage", None)
    if storage_object is not None and hasattr(storage_object, "remove_session"):
        storage_object.remove_session()
    backend = getattr(storage_object, "_backend", storage_object)
    engine = getattr(backend, "engine", None)
    if engine is not None:
        engine.dispose()
    del study

    semantic = _semantic_study_evidence(
        database,
        descriptor=descriptor,
        champion_path=run_dir / "champion.pth",
    )
    if semantic["selected_trial_number"] != champion_number:
        raise ContractError("copied champion is not the real Optuna best_trial")
    rows = _validate_trial_rows(semantic["rows"])
    _atomic_bytes(run_dir / "trial_compute.csv", _csv_bytes(rows))
    # This directory is inside the newly-created run and contains only
    # disposable per-trial states; the authenticated champion has been copied.
    shutil.rmtree(weights_dir)

    source_snapshot.assert_unchanged()
    if _require_clean_commit_a(REPO) != tested_commit:
        raise ContractError("tested commit changed during benchmark")
    final_environment_sha = execution_environment_sha256(
        current_execution_environment()
    )
    if (
        final_environment_sha != current_environment_sha
        or final_environment_sha
        != capacity["receipt"]["execution_environment_sha256"]
    ):
        raise ContractError("CUDA execution environment changed during benchmark")
    artifacts = _artifact_hashes(run_dir)
    evidence_root = _evidence_root(artifacts)
    durations = [float(row["duration_seconds"]) for row in rows]
    completed_utc = _current_utc()
    summary = {
        "schema": BENCHMARK_SCHEMA,
        "classification": CLASSIFICATION,
        "status": STATUS,
        "tested_source_commit": tested_commit,
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "run_directory": str(run_dir),
        "study_name": descriptor["study_name"],
        "protocol_hash": descriptor["protocol_hash"],
        "protocol_core_hash": descriptor["protocol_core_hash"],
        "execution_receipt_sha256": descriptor["execution_receipt_sha256"],
        "study_counts": semantic["counts"],
        "selected_trial_number": champion_number,
        "trial_duration_seconds_sum": sum(durations),
        "trial_duration_seconds_mean": sum(durations) / N_TRIALS,
        "benchmark_wall_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": device_total,
        "capacity_receipt_sha256": capacity["receipt_sha256"],
        "execution_environment_sha256": current_environment_sha,
        "fixture_sha256": fixture_sha,
        "artifact_sha256": artifacts,
        "adapter_calls": calls,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "objective_values_exported_to_summary_csv": False,
        "objective_values_retained_in_sqlite": True,
        "optuna_info_logging_suppressed": True,
        "progress_display_suppressed": True,
        "qualifying_run_was_fresh_uninterrupted": True,
    }
    _validate_summary(
        summary,
        descriptor=descriptor,
        capacity=capacity,
        directory=run_dir,
        artifacts=artifacts,
        evidence_root=evidence_root,
        semantic=semantic,
        rows=rows,
    )
    _atomic_json(run_dir / "summary.json", summary)
    summary_sha = _file_sha256(run_dir / "summary.json")
    run_state = {
        "schema": RUN_STATE_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "COMPLETE",
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "summary_sha256": summary_sha,
        "study_counts": semantic["counts"],
        "completed_utc": completed_utc,
    }
    _atomic_json(run_dir / "run_state.json", run_state)
    state_sha = _file_sha256(run_dir / "run_state.json")
    _atomic_json(run_dir / "_COMPLETE", {
        "schema": COMPLETION_SCHEMA,
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "summary_sha256": summary_sha,
        "run_state_sha256": state_sha,
    })
    return verify_completed_receipt(run_dir, tested_commit, repo=REPO)


def verify_completed_receipt(
    run_dir: str | os.PathLike[str],
    tested_source_commit: str,
    *,
    repo: str | os.PathLike[str] = REPO,
) -> dict[str, Any]:
    """Revalidate one completed benchmark and return dispatch evidence."""

    from core.capacity_preflight import load_capacity_receipt
    from core.environment import load_environment_lock, validate_environment_lock
    from core.source_provenance import repository_source_snapshot

    repository = _regular_directory(Path(repo), "repository")
    if not _HEX40.fullmatch(tested_source_commit):
        raise ContractError("tested source commit is malformed")
    current_head = _require_clean_tested_or_report_commit(
        repository, tested_source_commit
    )
    directory = _regular_directory(Path(run_dir), "benchmark run directory")
    _outside_repository(directory, repository)
    _assert_final_inventory(directory)
    snapshot = repository_source_snapshot(repository)
    environment_lock = load_environment_lock(
        repository / "environment" / "campaign-py313-cu128.json"
    )
    validate_environment_lock(environment_lock)
    capacity = load_capacity_receipt(
        directory / "capacity_receipt.json",
        expected_source_root_sha256=snapshot.python_runtime.sha256,
        expected_source_file_count=snapshot.python_runtime.file_count,
    )
    arrays = _fixture_arrays()
    fixture_sha = _fixture_digest(arrays)
    del arrays
    expected_descriptor, expected_execution_receipt = _expected_descriptor(
        tested_source_commit=tested_source_commit,
        source_snapshot=snapshot,
        capacity_envelope=capacity,
        environment_lock=environment_lock,
        fixture_sha256=fixture_sha,
    )
    descriptor = _strict_json(directory / "descriptor.json", "descriptor")
    if descriptor != expected_descriptor:
        raise ContractError("benchmark descriptor does not match live policy/source")
    descriptor_sha = _canonical_sha256(descriptor)
    execution_receipt = _strict_json(
        directory / "execution_receipt.json", "execution receipt"
    )
    if (
        execution_receipt != expected_execution_receipt
        or _file_sha256(directory / "execution_receipt.json")
        != descriptor["execution_receipt_sha256"]
    ):
        raise ContractError("benchmark execution receipt differs from its binding")

    summary = _strict_json(directory / "summary.json", "benchmark summary")
    run_state = _strict_json(directory / "run_state.json", "benchmark run state")
    marker = _strict_json(directory / "_COMPLETE", "benchmark completion marker")
    artifacts = _artifact_hashes(directory)
    evidence_root = _evidence_root(artifacts)
    summary_sha = _file_sha256(directory / "summary.json")
    state_sha = _file_sha256(directory / "run_state.json")
    expected_marker = {
        "schema": COMPLETION_SCHEMA,
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "summary_sha256": summary_sha,
        "run_state_sha256": state_sha,
    }
    if marker != expected_marker:
        raise ContractError("benchmark completion marker differs")

    semantic = _semantic_study_evidence(
        directory / "study.sqlite3",
        descriptor=descriptor,
        champion_path=directory / "champion.pth",
    )
    rows = _read_trial_csv(directory / "trial_compute.csv")
    if rows != semantic["rows"]:
        raise ContractError("trial CSV differs from real Optuna trial evidence")
    summary = _validate_summary(
        summary,
        descriptor=descriptor,
        capacity=capacity,
        directory=directory,
        artifacts=artifacts,
        evidence_root=evidence_root,
        semantic=semantic,
        rows=rows,
    )
    required_state = {
        "schema",
        "classification",
        "status",
        "descriptor_sha256",
        "evidence_root_sha256",
        "summary_sha256",
        "study_counts",
        "completed_utc",
    }
    if (
        set(run_state) != required_state
        or run_state["schema"] != RUN_STATE_SCHEMA
        or run_state["classification"] != CLASSIFICATION
        or run_state["status"] != "COMPLETE"
        or run_state["descriptor_sha256"] != descriptor_sha
        or run_state["evidence_root_sha256"] != evidence_root
        or run_state["summary_sha256"] != summary_sha
        or run_state["study_counts"] != semantic["counts"]
        or run_state["completed_utc"] != summary["completed_utc"]
    ):
        raise ContractError("benchmark run-state receipt differs")
    _utc(run_state["completed_utc"], "run-state completed_utc")
    snapshot.assert_unchanged()
    _assert_final_inventory(directory)
    if (
        _artifact_hashes(directory) != artifacts
        or _file_sha256(directory / "summary.json") != summary_sha
        or _file_sha256(directory / "run_state.json") != state_sha
        or _strict_json(
            directory / "_COMPLETE", "benchmark completion marker recheck"
        ) != marker
    ):
        raise ContractError("benchmark evidence changed during verification")
    if _require_clean_tested_or_report_commit(
        repository, tested_source_commit
    ) != current_head:
        raise ContractError("repository changed during benchmark verification")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": STATUS,
        "tested_source_commit": tested_source_commit,
        "descriptor_sha256": descriptor_sha,
        "evidence_root_sha256": evidence_root,
        "summary_sha256": summary_sha,
        "run_state_sha256": state_sha,
        "run_directory": str(directory),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capacity-receipt", required=True)
    args = parser.parse_args(argv)
    try:
        evidence = run_benchmark(args.output_dir, args.capacity_receipt)
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PAPER-1 BENCHMARK REFUSED: {exc}") from exc
    print("PAPER-1 COMPUTE BENCHMARK PASS")
    print(json.dumps(evidence, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    main()
