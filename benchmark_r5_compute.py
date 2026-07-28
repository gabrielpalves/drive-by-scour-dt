"""R11 compute-only benchmark using the registered production training path.

This program is deliberately *not* an experiment and its legacy cache is not
scientific evidence.  It exists only to measure whether a campaign machine can
execute the production HPO and finalist-refit workload in a practical amount
of time.

The benchmark is intentionally strict:

* the legacy fixture is opened read-only and used only to materialise a
  deterministic, authenticated 475-state x 50-passage, eight-channel workload;
* every derived byte lives below ``.audit_tmp/r11_compute_benchmark``;
* only ``training.trainer.get_or_create_cache`` and
  ``training.trainer.canonical_train_val_split`` are temporarily adapted;
* the real ``derive_execution_plan``, ``_stamp_study_protocol``,
  ``_execute_protocol_study`` and ``training.trainer.Objective`` path is used;
* a durable, real-CUDA capacity qualification is completed before the single
  registered 100-trial full-array PAA_LSTM_NHiTS anchor study is created;
* an OOM or any other failed trial is fatal and is never caught or replaced;
* exactly one shared-production finalist CV refit is durably accepted;
* results contain compute/provenance data only.  Model-objective and returned
  finalist values are never exported to JSON/CSV or printed;
* all durable and transient outputs live below
  ``.audit_tmp/r11_compute_benchmark``.

The shared production ``training.trainer.fit_predict_finalist_fold`` helper is
loaded as a hard preflight.  Keep this import boundary aligned with the
production driver when that helper is moved out of the driver.  Its benchmark
call supplies both ``max_epochs`` and ``n_scour_heads`` explicitly, so the
shared implementation never depends on driver globals.

Run only in the pinned campaign environment, from the repository root:

    python benchmark_r5_compute.py

After a crashed process, recovery is explicit and non-destructive:

    python benchmark_r5_compute.py --recover-stale
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
from importlib import metadata
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from typing import Any
import uuid


DISCLAIMER = "NON-SCIENTIFIC WORKLOAD FIXTURE"
DISCLAIMER_DETAIL = (
    "COMPUTE/THROUGHPUT BENCHMARK ONLY. The legacy fixture is not admissible "
    "for model selection, performance claims, scientific inference, or paper "
    "tables. Objective and returned finalist values are not exported."
)

BENCHMARK_SCHEMA = "ttbi-r11-compute-benchmark-v1"
DESCRIPTOR_SCHEMA = "ttbi-r11-workload-benchmark-v1"
DERIVATION_SCHEMA = "ttbi-r11-derived-workload-v1"
STUDY_DATASET_NAME = "R11_NON_SCIENTIFIC_DERIVED_WORST_SIZE_WORKLOAD"
STUDY_NAME_PREFIX = "BENCHMARK_ONLY_DO_NOT_PUBLISH__"

SOURCE_X_SHAPE = (12950, 2, 512)
SOURCE_Y_SHAPE = (12950, 2)
SOURCE_N_STATES = 259
DEFAULT_X_SHAPE = (23750, 8, 512)
DEFAULT_Y_SHAPE = (23750, 5)
N_STATES = 475
PASSAGES_PER_STATE = 50
DOFS = tuple(range(8))
TARGET_SUPPORTS = (2, 3, 4)
BEARING_TARGETS = ("left", "right")
SEED = 42
USEFUL_TRIALS = 100
EPOCHS = 50
USE_PRUNER = True
ANCHOR_STAGE = "s21_scour4"
EXECUTION_BLOCK = "l99"

FINALIST_N_SPLITS = 5
FINALIST_N_REPEATS = 2
FINALIST_REPEAT = 0
FINALIST_FOLD = 0
FINALIST_SPLIT_SEED = 271828

SAMPLED_FINITE_COUNT = 1024
FULL_FINITE_CHUNK_SAMPLES = 256
LOCK_FOREIGN_STALE_SECONDS = 24 * 60 * 60
LOCK_UNREADABLE_STALE_SECONDS = 300
ACTIVE_WALL_HEARTBEAT_SECONDS = 5.0
JSON_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
REPORT_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
SQLITE_SNAPSHOT_MAX_BYTES = 4 * 1024 * 1024 * 1024
SNAPSHOT_CHUNK_BYTES = 1 << 20
SNAPSHOT_OPEN_ATTEMPTS = 30

BENCHMARK_HYPERPARAMETER_EXECUTION_SCHEMA = (
    "ttbi-r11-benchmark-hyperparameter-execution-v1"
)
HYPERPARAMETER_RUN_PLAN_SCHEMA = "ttbi-hyperparameter-run-plan-v2"
HYPERPARAMETER_RUN_PLAN_FIELDS = frozenset({
    "schema",
    "mode",
    "execution_block",
    "anchor_stage",
    "stage",
    "dataset",
    "protocol_hash",
    "protocol_core_hash",
    "architecture",
    "seed",
    "active_dofs",
    "effective_n_trials",
    "effective_use_pruner",
    "requested_n_trials",
    "requested_use_pruner",
    "policy_sha256",
    "campaign_run_tag",
    "execution_receipt_sha256",
    "block_reference_manifest_sha256",
    "hyperparameter_manifest_sha256",
    "hyperparameter_source",
})

DEFAULT_FIXTURE_DIR = Path(
    "results/Stage0/cache/"
    "MD0_Pair_RearBogie_Vert__CarBody_Pitch_all_DOF_2_5_disc1"
)
DEFAULT_X_GLOB = "cache_*_PAA_dofs_2_5_disc1_reg_t2_3.npy"
DEFAULT_Y_GLOB = "cache_*_PAA_dofs_2_5_disc1_reg_t2_3_labels.npy"
OUTPUT_ROOT_RELATIVE = Path(".audit_tmp/r11_compute_benchmark")

DERIVATION_RECIPE = {
    "schema": DERIVATION_SCHEMA,
    "source_layout": {
        "features_shape": list(SOURCE_X_SHAPE),
        "labels_shape": list(SOURCE_Y_SHAPE),
        "states": SOURCE_N_STATES,
        "passages_per_state": PASSAGES_PER_STATE,
    },
    "derived_layout": {
        "features_shape": list(DEFAULT_X_SHAPE),
        "labels_shape": list(DEFAULT_Y_SHAPE),
        "states": N_STATES,
        "passages_per_state": PASSAGES_PER_STATE,
    },
    "state_mapping": "derived_state modulo 259; passage index preserved",
    "feature_channels": [
        {"source_channel": 0, "scale": 1.0},
        {"source_channel": 1, "scale": 1.0},
        {"source_channel": 0, "scale": -1.0},
        {"source_channel": 1, "scale": -1.0},
        {"source_channel": 0, "scale": 0.5},
        {"source_channel": 1, "scale": 0.5},
        {"source_channel": 0, "scale": 1.5},
        {"source_channel": 1, "scale": 1.5},
    ],
    "label_heads": [
        {"operation": "copy", "source_channel": 0},
        {"operation": "copy", "source_channel": 1},
        {"operation": "mean", "source_channels": [0, 1]},
        {"operation": "copy", "source_channel": 0},
        {"operation": "copy", "source_channel": 1},
    ],
    "dtype": "float32",
    "classification": DISCLAIMER,
}

IMMUTABLE_EVIDENCE_FILES = (
    "descriptor.json",
    "trial_compute.csv",
    "hpo_compute.json",
    "active_wall_heartbeat.json",
    "finalist_compute.json",
    "finalist_attempt_state.json",
    "finalist_active_wall_heartbeat.json",
    "study_receipt.sqlite3",
)

SOURCE_FILES = (
    "benchmark_r5_compute.py",
    "bundle_source_files.txt",
    "check_benchmark_contract.py",
    "comprehensive_ablation_multidamage.py",
    "core/campaign_contract.py",
    "core/capacity_preflight.py",
    "core/dataset.py",
    "core/environment.py",
    "core/execution_environment.py",
    "core/hyperparameter_policy.py",
    "core/models.py",
    "core/preprocessing.py",
    "core/protocol.py",
    "core/source_provenance.py",
    "core/statistical_inference.py",
    "core/task.py",
    "core/utils.py",
    "environment/campaign-py313-cu128.json",
    "plotting/confusion.py",
    "plotting/robustness_plots.py",
    "requirements-campaign-py313-cu128.txt",
    "training/pipeline.py",
    "training/robustness.py",
    "training/trainer.py",
)

# These are field-name fragments, not a ban on explanatory prose.  Every JSON
# and CSV publication passes this guard.  Optuna's internal scalar control
# values remain confined to its descriptor-stamped SQLite study, as required
# to invoke the real Objective and pruner; they are never copied into reports.
PROHIBITED_REPORT_FIELD_FRAGMENTS = (
    "best_value",
    "trial_value",
    "objective_value",
    "prediction",
    "ground_truth",
    "metric",
    "score",
    "accuracy",
    "loss",
    "mse",
    "mae",
    "rmse",
)

TRIAL_CSV_FIELDS = (
    "disclaimer",
    "descriptor_sha256",
    "study_name",
    "trial_number",
    "state",
    "duration_seconds",
    "epochs_reported",
    "last_epoch_count",
    "started_utc",
    "completed_utc",
)

HPO_REPORT_FIELDS = (
    "study_name",
    "resumed_with_existing_trials",
    "counts_before",
    "counts_after",
    "useful_budget",
    "epoch_cap_per_trial",
    "useful_trial_duration_seconds_sum",
    "useful_trial_duration_seconds_quantiles",
    "useful_epochs_reported_sum",
    "useful_epochs_reported_quantiles",
    "fatal_failure_policy",
    "hpo_wall_seconds_this_invocation",
    "hpo_active_wall_seconds_cumulative",
    "active_wall_checkpoint_interval_seconds",
    "active_wall_semantics",
    "timing_complete",
    "hpo_interruption_history",
    "optuna_recovery_events",
    "stale_inflight_trial_numbers_recovered_this_invocation",
    "all_stale_trial_numbers_recovered",
    "nominal_unrecorded_tail_seconds_per_abrupt_stop",
    "unrecorded_tail_bound",
    "checkpoint_files_removed_during_hpo",
    "trial_compute_sha256",
    "adapter_calls",
    "adapter_calls_scope",
    "memory",
    "memory_scope",
    "memory_complete",
)

FINALIST_REPORT_FIELDS = (
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


class ContractError(RuntimeError):
    """A benchmark contract or immutable-input invariant was violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_snapshot(
    path: Path,
    label: str,
    *,
    max_bytes: int,
    capture_bytes: bool = False,
    allow_missing: bool = False,
) -> dict[str, Any] | None:
    """Read one immutable regular-file view through one no-follow handle.

    The accepted size and digest are computed from the exact byte stream read
    from the opened handle.  Pre/open/post identities are compared so a path
    substitution, truncation, or concurrent rewrite cannot combine metadata
    from one object with bytes from another.  ``O_NOFOLLOW`` is used where the
    platform exposes it; the lstat/open identity comparison is the fallback on
    Windows.
    """

    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise ContractError("snapshot max_bytes must be a non-negative integer")
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    before: os.stat_result | None = None
    for attempt in range(SNAPSHOT_OPEN_ATTEMPTS):
        try:
            before = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            if allow_missing:
                return None
            raise ContractError(f"{label} is missing: {path}") from None
        except OSError as exc:
            retryable = (
                isinstance(exc, PermissionError)
                or (
                    os.name == "nt"
                    and getattr(exc, "winerror", None) in (5, 32)
                )
            )
            if not retryable or attempt + 1 == SNAPSHOT_OPEN_ATTEMPTS:
                raise ContractError(f"cannot lstat {label}: {path}") from exc
            time.sleep(min(0.02 * (attempt + 1), 0.25))
            continue
        if stat.S_ISLNK(before.st_mode):
            raise ContractError(f"{label} must not be a symbolic link: {path}")
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"{label} is not a regular file: {path}")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise ContractError(
                f"{label} size {before.st_size} exceeds limit "
                f"{max_bytes}: {path}"
            )
        try:
            descriptor = os.open(path, flags)
            break
        except FileNotFoundError:
            raise ContractError(
                f"{label} disappeared before open: {path}") from None
        except OSError as exc:
            retryable = (
                isinstance(exc, PermissionError)
                or (
                    os.name == "nt"
                    and getattr(exc, "winerror", None) in (5, 32)
                )
            )
            if not retryable or attempt + 1 == SNAPSHOT_OPEN_ATTEMPTS:
                raise ContractError(
                    f"cannot open no-follow snapshot for {label}: {path}"
                ) from exc
            time.sleep(min(0.02 * (attempt + 1), 0.25))
    if descriptor is None or before is None:
        raise AssertionError("unreachable snapshot-open retry state")

    digest = hashlib.sha256()
    captured = bytearray() if capture_bytes else None
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        same_object = (
            stat.S_ISREG(opened.st_mode)
            and opened.st_dev == before.st_dev
            and (
                before.st_ino == 0
                or opened.st_ino == 0
                or opened.st_ino == before.st_ino
            )
        )
        if not same_object:
            raise ContractError(
                f"{label} path identity changed while opening: {path}"
            )
        if opened.st_size < 0 or opened.st_size > max_bytes:
            raise ContractError(
                f"{label} size {opened.st_size} exceeds limit "
                f"{max_bytes}: {path}"
            )
        while True:
            block = os.read(descriptor, SNAPSHOT_CHUNK_BYTES)
            if not block:
                break
            byte_count += len(block)
            if byte_count > max_bytes:
                raise ContractError(
                    f"{label} byte stream exceeds limit {max_bytes}: {path}"
                )
            digest.update(block)
            if captured is not None:
                captured.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ContractError(f"cannot snapshot {label}: {path}") from exc
    finally:
        os.close(descriptor)

    stable_metadata = (
        stat.S_ISREG(after.st_mode)
        and opened.st_dev == after.st_dev
        and opened.st_ino == after.st_ino
        and opened.st_size == after.st_size == byte_count
        and getattr(opened, "st_mtime_ns", None)
        == getattr(after, "st_mtime_ns", None)
        and getattr(opened, "st_ctime_ns", None)
        == getattr(after, "st_ctime_ns", None)
    )
    if not stable_metadata:
        raise ContractError(f"{label} changed while being read: {path}")
    result: dict[str, Any] = {
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }
    if captured is not None:
        result["bytes"] = bytes(captured)
    return result


def _public_file_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Strip private captured bytes from a snapshot before publication."""

    return {
        "size_bytes": int(snapshot["size_bytes"]),
        "sha256": str(snapshot["sha256"]),
    }


def _json_mapping_from_snapshot(
    snapshot: Mapping[str, Any],
    path: Path,
    label: str,
) -> dict[str, Any]:
    """Parse the exact bytes whose size and SHA-256 were authenticated."""

    raw = snapshot.get("bytes")
    if not isinstance(raw, bytes):
        raise ContractError(f"{label} snapshot did not retain JSON bytes")

    def reject_nonfinite_json(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = child
        return value

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_nonfinite_json,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_json_int(
    record: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(
            f"JSON field {key!r} must be an integer >= {minimum}; got "
            f"{value!r}")
    return value


def _required_json_nonnegative_float(
    record: Mapping[str, Any],
    key: str,
) -> float:
    value = record.get(key)
    if isinstance(value, bool):
        raise ContractError(
            f"JSON field {key!r} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"JSON field {key!r} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise ContractError(
            f"JSON field {key!r} must be a finite non-negative number")
    return number


def _assert_no_scientific_report_fields(value: Any, prefix: str = "$") -> None:
    """Reject metric-bearing field names before publishing JSON or CSV."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text
                   for fragment in PROHIBITED_REPORT_FIELD_FRAGMENTS):
                raise ContractError(
                    f"scientific field {key!r} is forbidden in benchmark "
                    f"report payload at {prefix}"
                )
            _assert_no_scientific_report_fields(
                child, f"{prefix}.{key_text}"
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_scientific_report_fields(
                child, f"{prefix}[{index}]"
            )


def _atomic_replace(source: Path, destination: Path) -> None:
    """Publish one file, tolerating short Windows reader/OneDrive locks."""

    attempts = 30
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            retryable_windows_lock = (
                os.name == "nt"
                and getattr(exc, "winerror", None) in (5, 32)
            )
            if not retryable_windows_lock or attempt + 1 == attempts:
                raise
            time.sleep(min(0.02 * (attempt + 1), 0.25))
    raise AssertionError("unreachable atomic-replace retry state")


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    """Read one authenticated JSON snapshot from a regular file."""

    snapshot = _regular_file_snapshot(
        path,
        label,
        max_bytes=JSON_SNAPSHOT_MAX_BYTES,
        capture_bytes=True,
    )
    assert snapshot is not None
    return _json_mapping_from_snapshot(snapshot, path, label)


def _authenticated_file_sha256(path: Path, label: str) -> str:
    """Hash the exact accepted bytes of one required regular file."""

    snapshot = _regular_file_snapshot(
        path,
        label,
        max_bytes=REPORT_SNAPSHOT_MAX_BYTES,
    )
    assert snapshot is not None
    return str(snapshot["sha256"])


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _assert_no_scientific_report_fields(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Durably publish exact already-authenticated bytes."""

    if not isinstance(payload, bytes):
        raise ContractError("atomic binary payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise ContractError(
                        f"short write while publishing {path}")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _atomic_replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    fieldnames = tuple(fieldnames)
    _assert_no_scientific_report_fields({key: None for key in fieldnames})
    expected = set(fieldnames)
    for row in rows:
        if set(row) != expected:
            raise ContractError(
                f"CSV row fields {sorted(row)} != {sorted(expected)}"
            )
        _assert_no_scientific_report_fields(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    """Hash the exact UTF-8 bytes emitted by ``_atomic_csv``."""

    fieldnames = tuple(fieldnames)
    _assert_no_scientific_report_fields({key: None for key in fieldnames})
    expected = set(fieldnames)
    for row in rows:
        if set(row) != expected:
            raise ContractError(
                f"CSV row fields {sorted(row)} != {sorted(expected)}"
            )
        _assert_no_scientific_report_fields(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_within(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    if not _is_within(resolved, parent):
        raise ContractError(
            f"{label} must remain below {parent.resolve()}, got {resolved}"
        )
    return resolved


def _snapshot_files(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for role, path in sorted(paths.items()):
        resolved = path.resolve()
        stat = resolved.stat()
        if not resolved.is_file():
            raise FileNotFoundError(f"{role} fixture is not a file: {resolved}")
        snapshot[role] = {
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _sha256_file(resolved),
        }
    return snapshot


def _snapshot_equal(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Compare immutable content identity; retain mtime only as provenance."""

    if set(before) != set(after):
        return False
    keys = ("path", "size_bytes", "sha256")
    return all(
        all(before[role].get(key) == after[role].get(key) for key in keys)
        for role in before
    )


def _source_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_text in SOURCE_FILES:
        path = repo / relative_text
        if not path.is_file():
            raise FileNotFoundError(
                f"benchmark runtime source is missing: {relative_text}"
            )
        result[relative_text] = _sha256_file(path)
    return result


def _assert_sources_tracked_at_head(repo: Path) -> None:
    """Require every runtime source to resolve exactly to a HEAD Git blob.

    ``git status --untracked-files=no`` is intentionally tolerant of the user's
    data/results, but that tolerance must never admit an untracked benchmark
    module. ``git hash-object --path`` applies the repository's normal text
    filters, so this remains correct under Windows CRLF checkout settings.
    """

    failures: list[str] = []
    for relative_text in SOURCE_FILES:
        head = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative_text}"],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        working = subprocess.run(
            [
                "git", "hash-object", f"--path={relative_text}",
                "--", relative_text,
            ],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        head_blob = head.stdout.strip().lower()
        working_blob = working.stdout.strip().lower()
        if (
            head.returncode != 0
            or working.returncode != 0
            or not re.fullmatch(r"[0-9a-f]{40}", head_blob)
            or not re.fullmatch(r"[0-9a-f]{40}", working_blob)
            or head_blob != working_blob
        ):
            failures.append(relative_text)
    if failures:
        raise ContractError(
            "benchmark runtime sources must all be tracked and byte-equivalent "
            f"to HEAD after Git filters; failures: {failures}")


def _git_sha(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sha = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ContractError(f"unexpected git SHA from repository: {sha!r}")
    return sha


def _git_dirty(repo: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return bool(completed.stdout.strip())


def _safe_hostname() -> str:
    raw = socket.gethostname().strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "-", raw).strip(".-")
    if not safe:
        raise ContractError(f"hostname cannot form a run identity: {raw!r}")
    return safe[:48]


def _render_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def _resolve_one_fixture(
    fixture_dir: Path,
    explicit: str | None,
    pattern: str,
    role: str,
) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"{role} fixture not found: {candidate}")
        return candidate
    matches = sorted(path.resolve() for path in fixture_dir.glob(pattern))
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one {role} fixture matching {pattern!r} in "
            f"{fixture_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def _resolve_fixture_paths(args: argparse.Namespace) -> dict[str, Path]:
    fixture_dir = Path(args.fixture_dir)
    if not fixture_dir.is_absolute():
        fixture_dir = Path.cwd() / fixture_dir
    fixture_dir = fixture_dir.resolve()
    if not fixture_dir.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {fixture_dir}")
    x_path = _resolve_one_fixture(
        fixture_dir, args.fixture_x, DEFAULT_X_GLOB, "features"
    )
    y_path = _resolve_one_fixture(
        fixture_dir, args.fixture_y, DEFAULT_Y_GLOB, "labels"
    )
    if x_path == y_path:
        raise ContractError("feature and label fixtures resolve to one file")
    return {"features": x_path, "labels": y_path}


def _portable_file_identity(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Drop host/copy metadata while retaining exact content identity."""

    return {
        role: {
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for role, record in sorted(snapshot.items())
    }


def _validate_derived_workload(
    np: Any,
    derived_dir: Path,
    *,
    source_identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Authenticate the deterministic derived workload and its exact arrays."""

    manifest_path = derived_dir / "derived_workload.json"
    manifest = _read_json_mapping(
        manifest_path, "derived-workload manifest"
    )
    expected_keys = {
        "schema",
        "classification",
        "recipe",
        "recipe_sha256",
        "source_files",
        "files",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema") != DERIVATION_SCHEMA
        or manifest.get("classification") != DISCLAIMER
        or manifest.get("recipe") != DERIVATION_RECIPE
        or manifest.get("recipe_sha256")
        != _canonical_sha256(DERIVATION_RECIPE)
        or manifest.get("source_files") != dict(source_identity)
    ):
        raise ContractError(
            f"derived workload manifest differs from R11: {manifest_path}"
        )
    paths = {
        "features": derived_dir / "features_475x50x8x512_f32.npy",
        "labels": derived_dir / "labels_475x50x5_f32.npy",
    }
    snapshot = _portable_file_identity(_snapshot_files(paths))
    if manifest.get("files") != snapshot:
        raise ContractError(
            "derived workload bytes differ from their authenticated manifest"
        )
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    try:
        if (
            not isinstance(features, np.memmap)
            or not isinstance(labels, np.memmap)
            or features.flags.writeable
            or labels.flags.writeable
            or tuple(features.shape) != DEFAULT_X_SHAPE
            or tuple(labels.shape) != DEFAULT_Y_SHAPE
            or features.dtype != np.dtype("float32")
            or labels.dtype != np.dtype("float32")
        ):
            raise ContractError(
                "authenticated derived arrays do not have the R11 workload "
                "shape, dtype, and read-only memmap contract"
            )
    finally:
        features._mmap.close()
        labels._mmap.close()
    return {**paths, "manifest": manifest_path}


def _materialize_derived_workload(
    np: Any,
    *,
    source_paths: Mapping[str, Path],
    source_snapshot: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Path]:
    """Create the campaign-worst-size synthetic workload content-addressably.

    A unique staging directory is populated and then renamed to its immutable
    content identity.  A power loss can leave only an unreferenced staging
    directory; it can never make a partial array look published.
    """

    source_identity = _portable_file_identity(source_snapshot)
    identity = {
        "schema": DERIVATION_SCHEMA,
        "recipe": DERIVATION_RECIPE,
        "source_files": source_identity,
    }
    identity_sha = _canonical_sha256(identity)
    root = _require_within(
        output_root / "derived_workloads",
        output_root,
        "derived-workload root",
    )
    root.mkdir(parents=True, exist_ok=True)
    derived_dir = _require_within(
        root / identity_sha,
        root,
        "derived-workload directory",
    )
    if derived_dir.exists():
        return _validate_derived_workload(
            np,
            derived_dir,
            source_identity=source_identity,
        )

    source_x = np.load(
        source_paths["features"], mmap_mode="r", allow_pickle=False
    )
    source_y = np.load(
        source_paths["labels"], mmap_mode="r", allow_pickle=False
    )
    staging = _require_within(
        root / (
            f".materializing-{identity_sha}-{os.getpid()}-{uuid.uuid4().hex}"
        ),
        root,
        "derived-workload staging directory",
    )
    staging.mkdir()
    x_path = staging / "features_475x50x8x512_f32.npy"
    y_path = staging / "labels_475x50x5_f32.npy"
    derived_x = derived_y = None
    try:
        if (
            not isinstance(source_x, np.memmap)
            or not isinstance(source_y, np.memmap)
            or source_x.flags.writeable
            or source_y.flags.writeable
            or tuple(source_x.shape) != SOURCE_X_SHAPE
            or tuple(source_y.shape) != SOURCE_Y_SHAPE
            or source_x.dtype != np.dtype("float32")
            or source_y.dtype != np.dtype("float32")
        ):
            raise ContractError(
                "legacy source fixture differs from its immutable "
                "259-state/two-channel float32 contract"
            )
        source_labels = np.asarray(source_y).reshape(
            SOURCE_N_STATES,
            PASSAGES_PER_STATE,
            SOURCE_Y_SHAPE[1],
        )
        if not np.array_equal(
            source_labels,
            np.broadcast_to(source_labels[:, :1, :], source_labels.shape),
        ):
            raise ContractError(
                "legacy source labels are not contiguous 50-passage state blocks"
            )

        derived_x = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float32, shape=DEFAULT_X_SHAPE
        )
        derived_y = np.lib.format.open_memmap(
            y_path, mode="w+", dtype=np.float32, shape=DEFAULT_Y_SHAPE
        )
        chunk = FULL_FINITE_CHUNK_SAMPLES
        for start in range(0, DEFAULT_X_SHAPE[0], chunk):
            stop = min(start + chunk, DEFAULT_X_SHAPE[0])
            derived_indices = np.arange(start, stop, dtype=np.int64)
            states = derived_indices // PASSAGES_PER_STATE
            passages = derived_indices % PASSAGES_PER_STATE
            source_indices = (
                (states % SOURCE_N_STATES) * PASSAGES_PER_STATE + passages
            )
            source_features = np.asarray(source_x[source_indices])
            source_targets = np.asarray(source_y[source_indices])
            for channel, rule in enumerate(
                DERIVATION_RECIPE["feature_channels"]
            ):
                derived_x[start:stop, channel, :] = (
                    source_features[:, rule["source_channel"], :]
                    * np.float32(rule["scale"])
                )
            derived_y[start:stop, 0] = source_targets[:, 0]
            derived_y[start:stop, 1] = source_targets[:, 1]
            derived_y[start:stop, 2] = (
                source_targets[:, 0] + source_targets[:, 1]
            ) * np.float32(0.5)
            derived_y[start:stop, 3] = source_targets[:, 0]
            derived_y[start:stop, 4] = source_targets[:, 1]
        derived_x.flush()
        derived_y.flush()
        derived_x._mmap.close()
        derived_y._mmap.close()
        derived_x = derived_y = None

        files = {
            "features": x_path,
            "labels": y_path,
        }
        manifest = {
            "schema": DERIVATION_SCHEMA,
            "classification": DISCLAIMER,
            "recipe": DERIVATION_RECIPE,
            "recipe_sha256": _canonical_sha256(DERIVATION_RECIPE),
            "source_files": source_identity,
            "files": _portable_file_identity(_snapshot_files(files)),
        }
        _atomic_json(staging / "derived_workload.json", manifest)
        try:
            os.rename(staging, derived_dir)
        except FileExistsError:
            # A concurrent content-identical publisher won. Its public bytes
            # are reauthenticated below; this process's complete staging
            # directory remains as explicit, non-public evidence.
            pass
    finally:
        if derived_x is not None:
            derived_x._mmap.close()
        if derived_y is not None:
            derived_y._mmap.close()
        source_x._mmap.close()
        source_y._mmap.close()

    return _validate_derived_workload(
        np,
        derived_dir,
        source_identity=source_identity,
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is NOT a read-only liveness probe on Windows:
        # CPython routes every non-console-event value through TerminateProcess.
        # Query the process exit code through a least-privilege Win32 handle
        # instead, so a second benchmark invocation can never kill the owner
        # whose lock it is trying to inspect.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        error_invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            # Invalid/dead PIDs return ERROR_INVALID_PARAMETER (87). Any other
            # failure (including access denied for a protected process) is
            # conservatively treated as live and cannot authorize recovery.
            return ctypes.get_last_error() != error_invalid_parameter
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)):
                # The handle was opened successfully, so failure to inspect it
                # must not authorize stale-lock recovery.
                return True
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_identity_token(pid: int) -> str | None:
    """Return a boot/process-creation identity so PID reuse is detectable."""

    if pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _FileTime(ctypes.Structure):
                _fields_ = (
                    ("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD),
                )

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(_FileTime),
                ctypes.POINTER(_FileTime),
                ctypes.POINTER(_FileTime),
                ctypes.POINTER(_FileTime),
            )
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                creation = _FileTime()
                exit_time = _FileTime()
                kernel_time = _FileTime()
                user_time = _FileTime()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    return None
                ticks = (
                    int(creation.dwHighDateTime) << 32
                ) | int(creation.dwLowDateTime)
                return f"windows-filetime-100ns:{ticks}"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields_after_name = stat_text[stat_text.rfind(")") + 2:].split()
        start_ticks = int(fields_after_name[19])
        return f"linux-proc:{boot_id}:{start_ticks}"
    except Exception:
        return None


@contextlib.contextmanager
def _exclusive_file_mutex(path: Path):
    """Hold a one-byte OS lock that is released automatically on process exit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT)
    locked = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        last_error: OSError | None = None
        for attempt in range(30):
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                locked = True
                break
            except OSError as exc:
                last_error = exc
                if attempt + 1 < 30:
                    time.sleep(min(0.01 * (attempt + 1), 0.10))
        if not locked:
            raise ContractError(
                f"benchmark lock acquisition/recovery is already in "
                f"progress: {path}"
            ) from last_error
        yield
    finally:
        if locked:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class _ExclusivePidLock:
    """O_EXCL PID lock with explicit, evidence-preserving stale recovery."""

    def __init__(
        self,
        run_dir: Path,
        *,
        recover_stale: bool,
        command: str,
    ):
        self.run_dir = run_dir.resolve()
        self.lock_path = self.run_dir / "run.lock"
        self.guard_path = self.run_dir / "run.lock.guard"
        self.history_dir = self.run_dir / "lock_history"
        self.recover_stale = bool(recover_stale)
        self.command = command
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _archive_existing(self, classification: str) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        destination = self.history_dir / (
            f"{classification}-{time.time_ns()}-{uuid.uuid4().hex}.json"
        )
        _atomic_replace(self.lock_path, destination)

    def _classify_existing(self) -> tuple[bool, str]:
        try:
            payload = _read_json_mapping(
                self.lock_path, "benchmark PID lock")
            pid = int(payload["pid"])
            host = str(payload["hostname"]).lower()
        except Exception:
            age = time.time() - self.lock_path.stat().st_mtime
            return (
                age >= LOCK_UNREADABLE_STALE_SECONDS,
                f"unreadable lock age={age:.1f}s",
            )
        this_host = socket.gethostname().lower()
        if host == this_host:
            alive = _pid_alive(pid)
            stored_identity = payload.get("process_identity")
            current_identity = (
                _process_identity_token(pid) if alive else None
            )
            pid_reused = (
                alive
                and isinstance(stored_identity, str)
                and bool(stored_identity)
                and current_identity is not None
                and stored_identity != current_identity
            )
            return (
                not alive or pid_reused,
                f"same-host pid={pid} alive={alive} "
                f"identity_match={not pid_reused}",
            )
        age = time.time() - self.lock_path.stat().st_mtime
        return (
            age >= LOCK_FOREIGN_STALE_SECONDS,
            f"foreign-host={host!r} pid={pid} age={age:.1f}s",
        )

    def __enter__(self) -> "_ExclusivePidLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "r11-compute-pid-lock-v1",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_utc": _utc_now(),
            "command": self.command,
            "token": self.token,
            "process_identity": _process_identity_token(os.getpid()),
        }
        if not payload["process_identity"]:
            raise ContractError(
                "cannot establish current process creation identity for the "
                "benchmark lock")
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        with _exclusive_file_mutex(self.guard_path):
            # Classification, evidence archival and O_EXCL publication are one
            # critical section.  Without this OS mutex, two simultaneous
            # --recover-stale callers could both classify the old lock and the
            # loser could archive the winner's newly published live lock.
            if self.lock_path.exists():
                stale, detail = self._classify_existing()
                if not stale:
                    raise ContractError(
                        f"benchmark run is locked ({detail}): {self.lock_path}"
                    )
                if not self.recover_stale:
                    raise ContractError(
                        f"stale benchmark lock detected ({detail}). Re-run "
                        "with --recover-stale; no study/output is deleted."
                    )
                self._archive_existing("stale-recovered")
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                )
            except FileExistsError as exc:
                raise ContractError(
                    f"benchmark lock was acquired concurrently: "
                    f"{self.lock_path}"
                ) from exc
            try:
                os.write(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.acquired:
            return
        try:
            with _exclusive_file_mutex(self.guard_path):
                payload = _read_json_mapping(
                    self.lock_path, "held benchmark PID lock")
                if payload.get("token") != self.token:
                    raise ContractError(
                        "benchmark lock token changed while held: "
                        f"{self.lock_path}"
                    )
                self._archive_existing(
                    "released-ok" if exc_type is None
                    else "released-error"
                )
        finally:
            self.acquired = False


def _write_or_verify_descriptor(
    run_dir: Path,
    payload: Mapping[str, Any],
) -> Path:
    descriptor_path = run_dir / "descriptor.json"
    if descriptor_path.exists():
        stored = _read_json_mapping(
            descriptor_path, "existing benchmark descriptor")
        if stored != payload:
            raise ContractError(
                f"existing benchmark descriptor differs: {descriptor_path}. "
                "Refusing to mix runs."
            )
        return descriptor_path

    allowed_without_descriptor = {
        "run.lock",
        "run.lock.guard",
        "lock_history",
    }
    unexpected = sorted(
        path.name for path in run_dir.iterdir()
        if path.name not in allowed_without_descriptor
    )
    if unexpected:
        raise ContractError(
            f"run directory has payload but no descriptor ({unexpected}); "
            "refusing retroactive adoption or deletion"
        )
    _atomic_json(descriptor_path, payload)
    return descriptor_path


def _configure_runtime_outputs(run_dir: Path) -> dict[str, str]:
    """Keep library caches and temporary files inside the benchmark run."""

    runtime_tmp = run_dir / "runtime_tmp"
    cache_root = run_dir / "runtime_cache"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    values = {
        "TMP": str(runtime_tmp),
        "TEMP": str(runtime_tmp),
        "TMPDIR": str(runtime_tmp),
        "TORCH_HOME": str(cache_root / "torch"),
        "CUDA_CACHE_PATH": str(cache_root / "cuda"),
        "XDG_CACHE_HOME": str(cache_root),
        "MPLCONFIGDIR": str(cache_root / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache_root / "numba"),
    }
    for key, value in values.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    return values


def _bootstrap_cublas_environment(repo: Path) -> str:
    """Set the lock-declared cuBLAS workspace before importing PyTorch."""

    lock_path = repo / "environment" / "campaign-py313-cu128.json"
    try:
        spec = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = spec["cublas_workspace_config"]
    except Exception as exc:
        raise ContractError(
            f"cannot bootstrap cuBLAS from environment lock: {lock_path}"
        ) from exc
    if not isinstance(expected, str) or not expected:
        raise ContractError(
            "environment lock needs a non-empty cublas_workspace_config")
    actual = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", expected)
    if actual != expected:
        raise ContractError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the campaign lock before "
            f"PyTorch import: {actual!r} != {expected!r}")
    return actual


def _current_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(
            Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
        )
        return int(page_size * resident_pages)
    except Exception:
        return None


class _PeakMemoryMonitor:
    def __init__(self, torch_module: Any, interval_seconds: float = 0.20):
        self.torch = torch_module
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss: int | None = None
        self._sample_lock = threading.Lock()
        self.result: dict[str, int | None] | None = None

    def _sample(self) -> None:
        current = _current_rss_bytes()
        if current is not None:
            with self._sample_lock:
                self._peak_rss = (
                    current
                    if self._peak_rss is None
                    else max(self._peak_rss, current)
                )

    def snapshot(
        self,
        *,
        synchronize_cuda: bool = False,
    ) -> dict[str, int | None]:
        """Return the largest observed memory counters without resetting them."""

        self._sample()
        with self._sample_lock:
            peak_rss = self._peak_rss
        cuda_allocated = cuda_reserved = None
        if self.torch.cuda.is_available():
            if synchronize_cuda:
                self.torch.cuda.synchronize()
            cuda_allocated = int(self.torch.cuda.max_memory_allocated())
            cuda_reserved = int(self.torch.cuda.max_memory_reserved())
        return {
            "rss_peak_bytes": peak_rss,
            "cuda_peak_allocated_bytes": cuda_allocated,
            "cuda_peak_reserved_bytes": cuda_reserved,
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "_PeakMemoryMonitor":
        self._sample()
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(
            target=self._loop,
            name="r5-compute-memory-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.result = self.snapshot(synchronize_cuda=True)


class _ActiveWallHeartbeat:
    """Crash-resilient lower-bound clock for active HPO process time.

    Optuna's per-trial callback cannot run if power is lost mid-trial. Without
    an independent heartbeat, a resumed benchmark would silently omit that
    active interval while a stale trial's database duration could include hours
    of powered-off time. Atomic checkpoints provide a persisted lower bound
    without counting downtime. Five seconds is the nominal cadence, not a
    strict maximum tail under OS scheduling, GIL or filesystem delays.
    """

    def __init__(
        self,
        path: Path,
        descriptor_sha256: str,
        previous_seconds: float,
        interval_seconds: float = ACTIVE_WALL_HEARTBEAT_SECONDS,
    ):
        if not math.isfinite(previous_seconds) or previous_seconds < 0:
            raise ContractError("previous active-wall time cannot be negative")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ContractError("active-wall heartbeat interval must be positive")
        self.path = path
        self.descriptor_sha256 = descriptor_sha256
        self.previous_seconds = float(previous_seconds)
        self.interval_seconds = float(interval_seconds)
        self._started: float | None = None
        self._final_seconds: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._error: BaseException | None = None

    def current_seconds(self) -> float:
        if self._error is not None:
            raise ContractError(
                "active-wall heartbeat failed"
            ) from self._error
        if self._final_seconds is not None:
            return self._final_seconds
        if self._started is None:
            return self.previous_seconds
        return self.previous_seconds + time.perf_counter() - self._started

    def _write(self, status: str) -> None:
        with self._write_lock:
            _atomic_json(self.path, {
                "schema": BENCHMARK_SCHEMA,
                "classification": DISCLAIMER,
                "status": status,
                "descriptor_sha256": self.descriptor_sha256,
                "active_wall_seconds_cumulative": self.current_seconds(),
                "checkpoint_interval_seconds": self.interval_seconds,
                "pid": os.getpid(),
                "updated_utc": _utc_now(),
            })

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write("active")
            except BaseException as exc:
                self._error = exc
                self._stop.set()
                return

    def __enter__(self) -> "_ActiveWallHeartbeat":
        self._started = time.perf_counter()
        self._write("active")
        self._thread = threading.Thread(
            target=self._loop,
            name="r5-compute-active-wall-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(15.0, self.interval_seconds + 10.0))
            if self._thread.is_alive():
                self._error = RuntimeError(
                    "active-wall heartbeat worker did not stop")
        self._final_seconds = (
            self.previous_seconds + time.perf_counter() - self._started
            if self._started is not None
            else self.previous_seconds
        )
        if self._error is None:
            self._write("completed" if exc_type is None else "interrupted")
        elif exc_type is None:
            raise ContractError(
                "active-wall heartbeat failed"
            ) from self._error


def _load_active_wall_checkpoint(
    path: Path,
    descriptor_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = _read_json_mapping(path, "active-wall heartbeat")
    seconds = _required_json_nonnegative_float(
        checkpoint, "active_wall_seconds_cumulative")
    interval = _required_json_nonnegative_float(
        checkpoint, "checkpoint_interval_seconds")
    pid = _required_json_int(checkpoint, "pid", minimum=1)
    if (
        checkpoint.get("schema") != BENCHMARK_SCHEMA
        or checkpoint.get("classification") != DISCLAIMER
        or checkpoint.get("descriptor_sha256") != descriptor_sha256
        or checkpoint.get("status") not in (
            "active", "completed", "interrupted")
        or interval <= 0
        or not isinstance(checkpoint.get("updated_utc"), str)
    ):
        raise ContractError(
            f"active-wall heartbeat identity/value differs: {path}")
    return {
        **checkpoint,
        "active_wall_seconds_cumulative": seconds,
        "checkpoint_interval_seconds": interval,
        "pid": pid,
    }


def _read_active_wall_checkpoint(
    path: Path,
    descriptor_sha256: str,
) -> float:
    checkpoint = _load_active_wall_checkpoint(path, descriptor_sha256)
    return (
        0.0
        if checkpoint is None
        else float(checkpoint["active_wall_seconds_cumulative"])
    )


def _complete_or_verify_active_wall_checkpoint(
    path: Path,
    descriptor_sha256: str,
    expected_seconds: float,
    *,
    allow_explicit_recovery: bool,
) -> dict[str, Any]:
    """Authenticate a completed heartbeat, or explicitly seal a torn one.

    A clean compute path must have closed its heartbeat before publishing a
    completed receipt.  The only permitted repair is for an already recorded
    ``active``/``interrupted`` heartbeat after ``--recover-stale``; its
    persisted active time is retained exactly and the recovery is labelled.
    Missing or numerically contradictory timing evidence is never fabricated.
    """

    record = _load_active_wall_checkpoint(path, descriptor_sha256)
    if record is None:
        raise ContractError(
            f"completed compute receipt requires an active-wall heartbeat: "
            f"{path}"
        )
    actual_seconds = float(record["active_wall_seconds_cumulative"])
    if not math.isclose(
        actual_seconds,
        float(expected_seconds),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ContractError(
            "completed compute receipt and active-wall heartbeat disagree: "
            f"{actual_seconds} != {expected_seconds}"
        )
    if record.get("status") == "completed":
        return record
    if not allow_explicit_recovery:
        raise ContractError(
            "completed compute receipt requires a completed active-wall "
            "heartbeat; explicit interrupted-run recovery was not authorized"
        )
    prior_status = str(record["status"])
    _atomic_json(path, {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": descriptor_sha256,
        "active_wall_seconds_cumulative": actual_seconds,
        "checkpoint_interval_seconds":
            float(record["checkpoint_interval_seconds"]),
        "pid": int(record["pid"]),
        "updated_utc": _utc_now(),
        "completion_recovered_from_status": prior_status,
        "completion_recovered_utc": _utc_now(),
    })
    return _load_active_wall_checkpoint(path, descriptor_sha256) or {}


def _validated_hpo_interruption_history(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError("HPO interruption history must be a JSON list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractError(
                f"HPO interruption history item {index} is not an object")
        event = dict(item)
        event_id = str(event.get("event_id", "")).lower()
        status = event.get("heartbeat_status")
        raw_seconds = event.get("active_wall_seconds_cumulative", -1.0)
        raw_pid = event.get("pid", -1)
        if (
            isinstance(raw_seconds, bool)
            or not isinstance(raw_seconds, (int, float))
            or isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
        ):
            raise ContractError(
                f"HPO interruption history item {index} has invalid numbers")
        try:
            seconds = float(raw_seconds)
            pid = int(raw_pid)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"HPO interruption history item {index} has invalid numbers"
            ) from exc
        if (
            not re.fullmatch(r"[0-9a-f]{64}", event_id)
            or event_id in seen
            or status not in ("active", "completed", "interrupted", "missing")
            or not math.isfinite(seconds)
            or seconds < 0
            or (
                (status == "missing" and pid != 0)
                or (status != "missing" and pid <= 0)
            )
            or not isinstance(event.get("last_checkpoint_utc"), str)
            or not event["last_checkpoint_utc"]
            or not isinstance(event.get("acknowledged_utc"), str)
            or not event["acknowledged_utc"]
            or not isinstance(
                event.get("active_time_tail_may_be_incomplete"), bool)
            or not isinstance(event.get("compute_receipt_was_missing"), bool)
            or event["active_time_tail_may_be_incomplete"]
            is not (status in ("active", "missing"))
        ):
            raise ContractError(
                f"invalid HPO interruption history item {index}")
        expected_event_id = _canonical_sha256({
            "heartbeat_status": status,
            "pid": pid,
            "last_checkpoint_utc": event["last_checkpoint_utc"],
            "active_wall_seconds_cumulative": seconds,
            "compute_receipt_was_missing":
                event["compute_receipt_was_missing"],
        })
        if event_id != expected_event_id:
            raise ContractError(
                f"HPO interruption history item {index} identity differs")
        seen.add(event_id)
        event["event_id"] = event_id
        event["active_wall_seconds_cumulative"] = seconds
        event["pid"] = pid
        result.append(event)
    return result


def _merge_hpo_interruption_history(
    existing: Any,
    heartbeat: Mapping[str, Any] | None,
    *,
    recover_stale: bool,
    completed_receipt_missing: bool,
) -> list[dict[str, Any]]:
    """Acknowledge an unfinished HPO segment exactly once and preserve it."""

    history = _validated_hpo_interruption_history(existing)
    if heartbeat is None:
        if not completed_receipt_missing:
            return history
        identity = {
            "heartbeat_status": "missing",
            "pid": 0,
            "last_checkpoint_utc": "missing",
            "active_wall_seconds_cumulative": 0.0,
        }
        tail_incomplete = True
    else:
        status = str(heartbeat["status"])
        if status == "completed" and not completed_receipt_missing:
            return history
        identity = {
            "heartbeat_status": status,
            "pid": int(heartbeat["pid"]),
            "last_checkpoint_utc": str(heartbeat["updated_utc"]),
            "active_wall_seconds_cumulative": float(
                heartbeat["active_wall_seconds_cumulative"]),
        }
        tail_incomplete = status == "active"
    event_id = _canonical_sha256({
        **identity,
        "compute_receipt_was_missing": bool(completed_receipt_missing),
    })
    if any(event["event_id"] == event_id for event in history):
        return history
    if not recover_stale:
        raise ContractError(
            "an interrupted or incompletely published HPO segment is recorded; "
            "explicit --recover-stale is required even when Optuna has no "
            "RUNNING trial"
        )
    history.append({
        "event_id": event_id,
        **identity,
        "active_time_tail_may_be_incomplete": tail_incomplete,
        "compute_receipt_was_missing": bool(completed_receipt_missing),
        "acknowledged_utc": _utc_now(),
    })
    return history


def _validated_study_recovery_events(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError("Optuna recovery history must be a JSON list")
    result: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractError(
                f"Optuna recovery history item {index} is not an object")
        raw_numbers = item.get("trial_numbers")
        if not isinstance(raw_numbers, list):
            raise ContractError(
                f"Optuna recovery history item {index} has invalid trials")
        try:
            numbers = [int(number) for number in raw_numbers]
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"Optuna recovery history item {index} has invalid trials"
            ) from exc
        if (
            any(
                isinstance(raw, bool) or not isinstance(raw, int)
                for raw in raw_numbers
            )
            or any(number < 0 for number in numbers)
            or len(numbers) != len(set(numbers))
            or bool(seen_numbers.intersection(numbers))
            or not isinstance(item.get("recovered_utc"), str)
            or not item["recovered_utc"]
            or item.get("policy") != "state changed to FAIL; no deletion"
        ):
            raise ContractError(
                f"invalid Optuna recovery history item {index}")
        seen_numbers.update(numbers)
        result.append({
            "recovered_utc": str(item["recovered_utc"]),
            "trial_numbers": numbers,
            "policy": str(item["policy"]),
        })
    return result


def _validated_memory_receipt(
    value: Any,
    label: str,
) -> dict[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} memory receipt is not an object")
    keys = (
        "rss_peak_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    if set(value) != set(keys):
        raise ContractError(f"{label} memory receipt fields differ")
    result: dict[str, int | None] = {}
    for key in keys:
        raw = value[key]
        if raw is None:
            result[key] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ContractError(f"{label} memory receipt {key} is invalid")
        integer = raw
        if integer < 0:
            raise ContractError(f"{label} memory receipt {key} is invalid")
        result[key] = integer
    return result


def _merge_memory_receipts(
    *values: Mapping[str, Any] | None,
) -> dict[str, int | None] | None:
    """Merge persisted invocation peaks without losing an earlier maximum."""

    merged: dict[str, int | None] | None = None
    for index, value in enumerate(values):
        receipt = _validated_memory_receipt(
            value, f"HPO memory segment {index}")
        if receipt is None:
            continue
        if merged is None:
            merged = dict(receipt)
            continue
        for key, candidate in receipt.items():
            previous = merged[key]
            if candidate is not None:
                merged[key] = (
                    candidate
                    if previous is None
                    else max(previous, candidate)
                )
    return merged


def _validated_adapter_calls(
    value: Any,
    label: str,
) -> dict[str, int | None] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"get_cache", "canonical_split"}
    ):
        raise ContractError(f"{label} adapter-call receipt differs")
    result: dict[str, int | None] = {}
    for key in ("get_cache", "canonical_split"):
        raw = value[key]
        if raw is None:
            result[key] = None
        elif isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ContractError(f"{label} adapter-call count is invalid")
        else:
            result[key] = int(raw)
    return result


def _synchronize_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


@contextlib.contextmanager
def _suppress_optuna_value_logging(optuna_module: Any):
    """Prevent Optuna's INFO logger from printing benchmark objective values."""

    previous = optuna_module.logging.get_verbosity()
    optuna_module.logging.set_verbosity(optuna_module.logging.WARNING)
    try:
        yield
    finally:
        optuna_module.logging.set_verbosity(previous)


def _load_runtime(descriptor: Mapping[str, Any], descriptor_sha256: str) -> dict:
    """Import the pinned heavy stack only after output isolation is configured."""

    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    optuna = importlib.import_module("optuna")
    trainer = importlib.import_module("training.trainer")
    pipeline = importlib.import_module("training.pipeline")
    inference = importlib.import_module("core.statistical_inference")
    protocol = importlib.import_module("core.protocol")
    environment = importlib.import_module("core.environment")
    execution_environment = importlib.import_module(
        "core.execution_environment"
    )
    hyperparameter_policy = importlib.import_module(
        "core.hyperparameter_policy"
    )
    capacity_preflight = importlib.import_module(
        "core.capacity_preflight"
    )
    utils = importlib.import_module("core.utils")

    if protocol.protocol_hash(dict(descriptor)) != descriptor_sha256:
        raise ContractError(
            "benchmark descriptor hash differs from core.protocol.protocol_hash"
        )
    if int(protocol.OPTUNA_PROTOCOL["max_fail_slack"]) != 0:
        raise ContractError(
            "registered campaign failure policy is not fail-closed"
        )
    for module, name in (
        (pipeline, "_create_or_resume_study"),
        (pipeline, "_stamp_study_protocol"),
        (pipeline, "_execute_protocol_study"),
        (inference, "repeated_stratified_group_folds"),
        (inference, "frozen_checkpoint_epoch_count"),
        (trainer, "Objective"),
        (trainer, "fit_predict_finalist_fold"),
        (hyperparameter_policy, "derive_execution_plan"),
        (hyperparameter_policy, "validate_terminal_study"),
        (capacity_preflight, "ensure_capacity_preflight"),
        (execution_environment, "enforce_execution_block"),
    ):
        if not callable(getattr(module, name, None)):
            raise ContractError(
                f"required shared production helper is unavailable: "
                f"{module.__name__}.{name}"
            )

    lock = environment.load_environment_lock(
        "environment/campaign-py313-cu128.json"
    )
    environment_record = environment.validate_environment_lock(lock)
    return {
        "np": np,
        "torch": torch,
        "optuna": optuna,
        "trainer": trainer,
        "pipeline": pipeline,
        "inference": inference,
        "protocol": protocol,
        "environment_record": environment_record,
        "execution_environment": execution_environment,
        "hyperparameter_policy": hyperparameter_policy,
        "capacity_preflight": capacity_preflight,
        "utils": utils,
    }


def _physical_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def _nvidia_smi_snapshot() -> dict[str, Any]:
    """Best-effort hardware/thermal receipt; never changes benchmark state."""

    fields = (
        "index",
        "name",
        "driver_version",
        "memory.total",
        "temperature.gpu",
        "power.limit",
        "pstate",
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": type(exc).__name__,
            "captured_utc": _utc_now(),
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": f"exit-{completed.returncode}",
            "captured_utc": _utc_now(),
        }
    keys = (
        "index",
        "name",
        "driver_version",
        "memory_total_mib",
        "temperature_gpu_c",
        "power_limit_w",
        "pstate",
    )
    devices = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(keys):
            raise ContractError(
                f"unexpected nvidia-smi telemetry row: {line!r}")
        devices.append(dict(zip(keys, values)))
    return {
        "available": bool(devices),
        "captured_utc": _utc_now(),
        "devices": devices,
    }


def _runtime_environment(runtime: Mapping[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]
    packages = {}
    for distribution in (
        "numpy",
        "optuna",
        "torch",
        "scikit-learn",
        "joblib",
        "tqdm",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = "MISSING"
    cuda_available = bool(torch.cuda.is_available())
    return {
        "campaign_lock_validation": runtime["environment_record"],
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "packages": packages,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "environment_scope": (
            "this invocation; earlier interrupted-invocation telemetry is "
            "not reconstructed"
        ),
        "gpu_telemetry_this_invocation_start": _nvidia_smi_snapshot(),
    }


def _load_and_validate_fixture(
    np: Any,
    fixture_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Open immutable fixture memmaps and validate shape/order/finiteness."""

    X = np.load(
        fixture_paths["features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    y = np.load(
        fixture_paths["labels"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if not isinstance(X, np.memmap) or not isinstance(y, np.memmap):
        raise ContractError("both workload arrays must be numpy.memmap objects")
    if X.flags.writeable or y.flags.writeable:
        raise ContractError("workload memmaps unexpectedly allow writes")
    if tuple(X.shape) != DEFAULT_X_SHAPE:
        raise ContractError(
            f"fixture X shape {tuple(X.shape)} != {DEFAULT_X_SHAPE}"
        )
    if tuple(y.shape) != DEFAULT_Y_SHAPE:
        raise ContractError(
            f"fixture y shape {tuple(y.shape)} != {DEFAULT_Y_SHAPE}"
        )
    if X.dtype != np.dtype("float32") or y.dtype != np.dtype("float32"):
        raise ContractError(
            f"fixture dtypes must be float32/float32, got {X.dtype}/{y.dtype}"
        )
    if len(X) != N_STATES * PASSAGES_PER_STATE:
        raise ContractError(
            "derived workload sample count does not equal 475 states x "
            "50 passages"
        )

    sample_count = min(SAMPLED_FINITE_COUNT, len(X))
    sampled_idx = np.unique(
        np.linspace(0, len(X) - 1, sample_count, dtype=np.int64)
    )
    sampled_finite = bool(
        np.isfinite(X[sampled_idx]).all()
        and np.isfinite(y[sampled_idx]).all()
    )
    if not sampled_finite:
        raise ContractError("sampled fixture finiteness check failed")

    full_finite = True
    for start in range(0, len(X), FULL_FINITE_CHUNK_SAMPLES):
        stop = min(start + FULL_FINITE_CHUNK_SAMPLES, len(X))
        if not np.isfinite(X[start:stop]).all():
            full_finite = False
            break
    if full_finite:
        for start in range(0, len(y), FULL_FINITE_CHUNK_SAMPLES):
            stop = min(start + FULL_FINITE_CHUNK_SAMPLES, len(y))
            if not np.isfinite(y[start:stop]).all():
                full_finite = False
                break
    if not full_finite:
        raise ContractError("full fixture finiteness check failed")

    labels_by_state = np.asarray(y).reshape(
        N_STATES, PASSAGES_PER_STATE, DEFAULT_Y_SHAPE[1]
    )
    block_order_ok = bool(
        np.array_equal(
            labels_by_state,
            np.broadcast_to(
                labels_by_state[:, :1, :],
                labels_by_state.shape,
            ),
        )
    )
    if not block_order_ok:
        raise ContractError(
            "fixture is not ordered as 259 contiguous 50-passage state blocks"
        )
    groups = np.repeat(
        np.arange(N_STATES, dtype=np.int64),
        PASSAGES_PER_STATE,
    )
    return {
        "X": X,
        "y": y,
        "groups": groups,
        "report": {
            "features_shape": list(X.shape),
            "labels_shape": list(y.shape),
            "features_dtype": str(X.dtype),
            "labels_dtype": str(y.dtype),
            "features_memmap_read_only": True,
            "labels_memmap_read_only": True,
            "state_count": N_STATES,
            "passages_per_state": PASSAGES_PER_STATE,
            "state_block_order_verified": block_order_ok,
            "sampled_finiteness_verified": sampled_finite,
            "sampled_finiteness_sample_count": int(len(sampled_idx)),
            "full_finiteness_verified": full_finite,
            "full_finiteness_chunk_samples": FULL_FINITE_CHUNK_SAMPLES,
        },
    }


def _close_fixture(fixture: Mapping[str, Any] | None) -> None:
    if not fixture:
        return
    for name in ("X", "y"):
        array = fixture.get(name)
        mmap_handle = getattr(array, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


def _index_sha256(np: Any, values: Any) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _make_workload_splits(np: Any, groups: Any) -> dict[str, Any]:
    """Deterministic 60/20/20 state split for the non-scientific workload."""

    state_order = np.random.default_rng(SEED).permutation(N_STATES)
    pattern = ("train", "outer", "inner", "train", "train")
    state_parts: dict[str, list[int]] = {
        "train": [],
        "inner": [],
        "outer": [],
    }
    for position, state in enumerate(state_order):
        state_parts[pattern[position % len(pattern)]].append(int(state))
    sample_idx = np.arange(len(groups), dtype=np.int64)
    indices = {
        part: sample_idx[np.isin(groups, np.asarray(states, dtype=np.int64))]
        for part, states in state_parts.items()
    }
    development_idx = np.sort(
        np.concatenate([indices["train"], indices["inner"]])
    )
    if np.intersect1d(development_idx, indices["outer"]).size:
        raise ContractError("workload development and outer partitions overlap")
    if len(development_idx) + len(indices["outer"]) != len(groups):
        raise ContractError("workload split does not cover the fixture")
    report = {
        "seed": SEED,
        "assignment_pattern": list(pattern),
        "stratum_count": 1,
        "stratum_label": "workload",
        "train_state_count": len(state_parts["train"]),
        "inner_state_count": len(state_parts["inner"]),
        "outer_state_count": len(state_parts["outer"]),
        "train_sample_count": int(len(indices["train"])),
        "inner_sample_count": int(len(indices["inner"])),
        "outer_sample_count": int(len(indices["outer"])),
        "development_sample_count": int(len(development_idx)),
        "train_states_sha256": _index_sha256(
            np, sorted(state_parts["train"])
        ),
        "inner_states_sha256": _index_sha256(
            np, sorted(state_parts["inner"])
        ),
        "outer_states_sha256": _index_sha256(
            np, sorted(state_parts["outer"])
        ),
        "train_indices_sha256": _index_sha256(np, indices["train"]),
        "inner_indices_sha256": _index_sha256(np, indices["inner"]),
        "outer_indices_sha256": _index_sha256(np, indices["outer"]),
    }
    return {
        "indices": indices,
        "development_idx": development_idx,
        "report": report,
    }


@contextlib.contextmanager
def _patched_trainer_fixture(
    trainer_module: Any,
    np: Any,
    fixture: Mapping[str, Any],
    splits: Mapping[str, Any],
    expected_config: Mapping[str, Any],
):
    """Temporarily adapt exactly the two data-bound trainer entry points."""

    original_get_cache = trainer_module.get_or_create_cache
    original_split = trainer_module.canonical_train_val_split
    calls = {"get_cache": 0, "canonical_split": 0}

    def benchmark_get_or_create_cache(
        config: Mapping[str, Any],
        dataset_name: str,
        cache_dir: str,
    ):
        del cache_dir
        if dataset_name != STUDY_DATASET_NAME:
            raise ContractError(
                f"Objective requested unexpected dataset {dataset_name!r}"
            )
        for key in (
            "method",
            "dofs",
            "task",
            "target_supports",
            "use_lstm",
            "use_nhits",
        ):
            if config.get(key) != expected_config.get(key):
                raise ContractError(
                    f"Objective config drift at {key!r}: "
                    f"{config.get(key)!r} != {expected_config.get(key)!r}"
                )
        calls["get_cache"] += 1
        return (
            fixture["X"],
            fixture["y"],
            None,
            fixture["groups"],
        )

    def benchmark_canonical_train_val_split(
        n_samples: int,
        groups: Any = None,
        seed: int = SEED,
        dataset_name: str | None = None,
    ):
        if n_samples != DEFAULT_X_SHAPE[0]:
            raise ContractError(
                f"Objective split requested {n_samples} samples"
            )
        if seed != SEED:
            raise ContractError(f"Objective split seed drifted to {seed}")
        if dataset_name != STUDY_DATASET_NAME:
            raise ContractError(
                f"Objective split requested dataset {dataset_name!r}"
            )
        if groups is None or not np.array_equal(groups, fixture["groups"]):
            raise ContractError("Objective split groups differ from fixture")
        calls["canonical_split"] += 1
        return (
            splits["indices"]["train"].copy(),
            splits["indices"]["inner"].copy(),
        )

    trainer_module.get_or_create_cache = benchmark_get_or_create_cache
    trainer_module.canonical_train_val_split = (
        benchmark_canonical_train_val_split
    )
    try:
        yield calls
    finally:
        trainer_module.get_or_create_cache = original_get_cache
        trainer_module.canonical_train_val_split = original_split


def _study_counts(optuna_module: Any, study: Any) -> dict[str, int]:
    trial_state = optuna_module.trial.TrialState
    states = [trial.state for trial in study.trials]
    return {
        "complete": sum(state == trial_state.COMPLETE for state in states),
        "pruned": sum(state == trial_state.PRUNED for state in states),
        "failed": sum(state == trial_state.FAIL for state in states),
        "running": sum(state == trial_state.RUNNING for state in states),
        "waiting": sum(state == trial_state.WAITING for state in states),
        "useful": sum(
            state in (trial_state.COMPLETE, trial_state.PRUNED)
            for state in states
        ),
        "total": len(states),
    }


def _trial_compute_rows(
    study: Any,
    descriptor_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in sorted(study.trials, key=lambda item: item.number):
        steps = sorted(int(step) for step in trial.intermediate_values)
        duration = (
            trial.duration.total_seconds()
            if trial.duration is not None
            else ""
        )
        rows.append({
            "disclaimer": DISCLAIMER,
            "descriptor_sha256": descriptor_sha256,
            "study_name": study.study_name,
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "duration_seconds": duration,
            "epochs_reported": len(steps),
            "last_epoch_count": (steps[-1] + 1 if steps else 0),
            "started_utc": _datetime_as_utc_text(trial.datetime_start),
            "completed_utc": _datetime_as_utc_text(
                trial.datetime_complete),
        })
    return rows


def _datetime_as_utc_text(value: datetime | None) -> str:
    """Serialize Optuna datetimes honestly as timezone-aware UTC.

    Optuna 4.3 supplies naive local datetimes on the pinned Windows runtime.
    Python's ``astimezone`` deliberately interprets a naive datetime in the
    machine's local zone before converting it, avoiding the false ``*_utc``
    labels that a direct ``isoformat`` would produce.
    """

    if value is None:
        return ""
    if not isinstance(value, datetime):
        raise ContractError(
            f"Optuna trial timestamp has type {type(value).__name__}, "
            "expected datetime"
        )
    return value.astimezone(timezone.utc).isoformat()


def _write_trial_csv(
    path: Path,
    study: Any,
    descriptor_sha256: str,
) -> None:
    _atomic_csv(
        path,
        _trial_compute_rows(study, descriptor_sha256),
        TRIAL_CSV_FIELDS,
    )


def _quantile_summary(np: Any, values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        key: float(value)
        for key, value in zip(
            ("p0", "p25", "p50", "p75", "p95", "p100"),
            quantiles,
        )
    }


def _recover_inflight_trials(
    runtime: Mapping[str, Any],
    study: Any,
    *,
    recover_stale: bool,
) -> list[int]:
    trial_state = runtime["optuna"].trial.TrialState
    inflight = [
        trial for trial in study.trials
        if trial.state in (trial_state.RUNNING, trial_state.WAITING)
    ]
    if not inflight:
        return []
    numbers = [int(trial.number) for trial in inflight]
    if not recover_stale:
        raise ContractError(
            f"study contains RUNNING/WAITING trials {numbers}; explicit "
            "--recover-stale is required. No trial or study is deleted."
        )
    for trial in inflight:
        changed = study._storage.set_trial_state_values(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            trial_state.FAIL,
        )
        if changed is False:
            raise ContractError(
                f"could not mark stale trial {trial.number} as FAIL"
            )
    events = _validated_study_recovery_events(
        study.user_attrs.get("r11_compute_recovery_events", [])
    )
    events.append({
        "recovered_utc": _utc_now(),
        "trial_numbers": numbers,
        "policy": "state changed to FAIL; no deletion",
    })
    study.set_user_attr("r11_compute_recovery_events", events)
    return numbers


def _build_config(
    descriptor: Mapping[str, Any],
    descriptor_sha256: str,
    execution_runtime: Mapping[str, Any],
    *,
    campaign_run_tag: str,
    execution_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "name": "r11c",
        "seed": SEED,
        "sensor_noise": None,
        "name_short": "PAA_LSTM_NHiTS",
        "method": "PAA",
        "dofs": list(DOFS),
        "discretization": 1,
        "use_space2vec": False,
        "use_lstm": True,
        "use_nhits": True,
        "model_type": "1D_MODULAR",
        "task": "regression",
        "target_supports": list(TARGET_SUPPORTS),
        "bearing_targets": list(BEARING_TARGETS),
        "protocol_hash": descriptor_sha256,
        "protocol_core_hash": _canonical_sha256(descriptor["core"]),
        "protocol_descriptor": dict(descriptor),
        "execution_runtime": dict(execution_runtime),
        # This is an anchor-HPO benchmark: the block reference does not exist
        # until after selection, but the run and execution receipt already do.
        "campaign_run_tag": campaign_run_tag,
        "execution_receipt_sha256": execution_receipt_sha256,
        "block_reference_manifest_sha256": None,
        "hyperparameter_mode": "anchor_hpo",
    }


def _prepare_study(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    hyperparameter_plan: Mapping[str, Any],
    capacity_receipt: Mapping[str, Any],
    run_dir: Path,
    descriptor_sha256: str,
    *,
    recover_stale: bool,
) -> tuple[Any, list[int]]:
    pipeline = runtime["pipeline"]
    database_path = run_dir / "study.sqlite3"
    storage = f"sqlite:///{database_path.as_posix()}"
    study_name = f"{STUDY_NAME_PREFIX}{descriptor_sha256[:16]}"
    study = pipeline._create_or_resume_study(
        study_name,
        storage,
        USEFUL_TRIALS,
        sampler_seed=SEED,
        use_pruner=USE_PRUNER,
    )
    pipeline._stamp_study_protocol(
        study,
        config=dict(config),
        dataset_name=STUDY_DATASET_NAME,
        n_trials=USEFUL_TRIALS,
        epochs=EPOCHS,
        sampler_seed=SEED,
        use_pruner=USE_PRUNER,
        hyperparameter_plan=dict(hyperparameter_plan),
        capacity_receipt=dict(capacity_receipt),
    )
    contract_attr = {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "descriptor_sha256": descriptor_sha256,
        "reporting": (
            "compute/provenance fields only; scientific values are not exported"
        ),
    }
    previous = study.user_attrs.get("r11_compute_contract")
    if previous is None:
        study.set_user_attr("r11_compute_contract", contract_attr)
    elif previous != contract_attr:
        raise ContractError("stored R11 compute-study contract differs")
    recovered = _recover_inflight_trials(
        runtime,
        study,
        recover_stale=recover_stale,
    )
    return study, recovered


class _QuietProtocolStudy:
    """Output-only adapter around the real durable Optuna study.

    The production helper deliberately supplies its normal champion-printing
    callback and progress bar.  Those surfaces reveal objective values, which
    this non-scientific benchmark forbids publishing.  The adapter verifies
    that exact call signature, substitutes the compute-receipt callback, and
    invokes the underlying Optuna study without ``catch``.  Trial sampling,
    pruning, objective execution, storage and terminal validation are unchanged.
    """

    def __init__(
        self,
        raw_study: Any,
        *,
        production_callback: Any,
        receipt_callback: Any,
    ):
        self._raw_study = raw_study
        self._production_callback = production_callback
        self._receipt_callback = receipt_callback

    @property
    def study_name(self) -> str:
        return self._raw_study.study_name

    @property
    def trials(self):
        return self._raw_study.trials

    def optimize(
        self,
        objective: Any,
        *,
        n_trials: int,
        callbacks: Sequence[Any],
        show_progress_bar: bool,
    ) -> Any:
        if (
            callbacks != [self._production_callback]
            or show_progress_bar is not True
        ):
            raise ContractError(
                "training.pipeline._execute_protocol_study call signature "
                "drifted from the reviewed production path"
            )
        # Deliberately no catch= argument: CPU/CUDA OOM and every unexpected
        # exception are fatal, produce a FAIL trial, and receive no replacement.
        return self._raw_study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[self._receipt_callback],
            show_progress_bar=False,
        )


def _run_registered_anchor_hpo(
    runtime: Mapping[str, Any],
    study: Any,
    config: Mapping[str, Any],
    hyperparameter_plan: Mapping[str, Any],
    fixture: Mapping[str, Any],
    splits: Mapping[str, Any],
    weights_dir: Path,
    run_dir: Path,
    descriptor_sha256: str,
    recovered_trial_numbers: Sequence[int],
    *,
    recover_stale: bool,
) -> dict[str, Any]:
    """Run the one exact registered anchor-HPO study without value output."""

    optuna = runtime["optuna"]
    torch = runtime["torch"]
    trainer = runtime["trainer"]
    pipeline = runtime["pipeline"]
    plan = runtime["hyperparameter_policy"].validate_run_plan(
        dict(hyperparameter_plan)
    )
    if (
        plan["mode"] != "anchor_hpo"
        or plan["effective_n_trials"] != USEFUL_TRIALS
        or plan["effective_use_pruner"] is not USE_PRUNER
        or tuple(plan["active_dofs"]) != DOFS
        or plan["architecture"] != "PAA_LSTM_NHiTS"
        or plan["stage"] != ANCHOR_STAGE
        or plan["execution_block"] != EXECUTION_BLOCK
    ):
        raise ContractError(
            "benchmark did not receive the exact registered R11 anchor plan"
        )
    trial_csv = run_dir / "trial_compute.csv"
    progress_path = run_dir / "run_state.json"
    hpo_checkpoint_path = run_dir / "hpo_compute.json"
    heartbeat_path = run_dir / "active_wall_heartbeat.json"
    counts_before = _study_counts(optuna, study)
    if counts_before["useful"] > USEFUL_TRIALS:
        raise ContractError(
            f"study was extended past useful budget: {counts_before}"
        )
    if counts_before["total"] > USEFUL_TRIALS:
        raise ContractError(
            f"study was extended past its exact registered budget: "
            f"{counts_before}"
        )

    study_recovery_events = _validated_study_recovery_events(
        study.user_attrs.get("r11_compute_recovery_events", [])
    )
    all_recovered_trial_numbers = sorted({
        int(number)
        for event in study_recovery_events
        for number in event["trial_numbers"]
    })
    if not set(int(number) for number in recovered_trial_numbers).issubset(
            all_recovered_trial_numbers):
        raise ContractError(
            "current stale-trial recovery is absent from cumulative study history"
        )

    previous_active_wall = 0.0
    previous_checkpoint_removals = 0
    previous_memory: dict[str, int | None] | None = None
    previous_adapter_calls: dict[str, int] | None = None
    interruption_history: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    if hpo_checkpoint_path.exists():
        checkpoint = _read_json_mapping(
            hpo_checkpoint_path, "HPO compute checkpoint")
        if (
            checkpoint.get("schema") != BENCHMARK_SCHEMA
            or checkpoint.get("classification") != DISCLAIMER
            or checkpoint.get("descriptor_sha256") != descriptor_sha256
            or checkpoint.get("study_name") not in (None, study.study_name)
        ):
            raise ContractError(
                f"HPO compute checkpoint identity differs: "
                f"{hpo_checkpoint_path}")
        previous_active_wall = _required_json_nonnegative_float(
            checkpoint, "active_wall_seconds_cumulative")
        previous_checkpoint_removals = _required_json_int(
            checkpoint, "checkpoint_files_removed")
        interruption_history = _validated_hpo_interruption_history(
            checkpoint.get("hpo_interruption_history", [])
        )
        previous_memory = _validated_memory_receipt(
            checkpoint.get("memory"), "HPO checkpoint")
        raw_adapter_calls = checkpoint.get("adapter_calls")
        if raw_adapter_calls is not None:
            if (
                not isinstance(raw_adapter_calls, Mapping)
                or set(raw_adapter_calls)
                != {"get_cache", "canonical_split"}
            ):
                raise ContractError(
                    "HPO checkpoint adapter-call receipt differs")
            previous_adapter_calls = {}
            for key in ("get_cache", "canonical_split"):
                value = raw_adapter_calls[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ContractError(
                        "HPO checkpoint adapter-call count is invalid")
                previous_adapter_calls[key] = int(value)

    heartbeat_checkpoint = _load_active_wall_checkpoint(
        heartbeat_path, descriptor_sha256)
    previous_active_wall = max(
        previous_active_wall,
        (
            0.0
            if heartbeat_checkpoint is None
            else float(
                heartbeat_checkpoint["active_wall_seconds_cumulative"])
        ),
    )

    budget_complete_before = (
        counts_before["failed"] == 0
        and counts_before["running"] == 0
        and counts_before["waiting"] == 0
        and counts_before["useful"] == USEFUL_TRIALS
        and counts_before["total"] == USEFUL_TRIALS
        and counts_before["complete"] >= 1
    )
    stored_report: dict[str, Any] | None = None
    if checkpoint is not None and checkpoint.get("status") == "completed":
        raw_report = checkpoint.get("report")
        if raw_report is not None:
            if not isinstance(raw_report, Mapping):
                raise ContractError(
                    "completed HPO checkpoint report is not an object")
            stored_report = dict(raw_report)
            if set(stored_report) != set(HPO_REPORT_FIELDS):
                raise ContractError(
                    "completed HPO checkpoint report fields differ")
            _assert_no_scientific_report_fields(stored_report)
            report_sha256 = str(
                checkpoint.get("report_sha256", "")).lower()
            if (
                not re.fullmatch(r"[0-9a-f]{64}", report_sha256)
                or report_sha256 != _canonical_sha256(stored_report)
            ):
                raise ContractError(
                    "completed HPO checkpoint report hash differs")
            trial_compute_sha256 = str(
                checkpoint.get("trial_compute_sha256", "")).lower()
            actual_trial_compute_sha256 = _authenticated_file_sha256(
                trial_csv, "completed HPO trial-compute receipt")
            expected_trial_compute_sha256 = _csv_rows_sha256(
                _trial_compute_rows(study, descriptor_sha256),
                TRIAL_CSV_FIELDS,
            )
            if (
                not re.fullmatch(r"[0-9a-f]{64}", trial_compute_sha256)
                or stored_report.get("trial_compute_sha256")
                != trial_compute_sha256
                or actual_trial_compute_sha256 != trial_compute_sha256
                or expected_trial_compute_sha256 != trial_compute_sha256
            ):
                raise ContractError(
                    "completed HPO trial-compute receipt differs from study")
            stored_history = _validated_hpo_interruption_history(
                stored_report.get("hpo_interruption_history", [])
            )
            stored_recovery = _validated_study_recovery_events(
                stored_report.get("optuna_recovery_events", [])
            )
            stored_memory = _validated_memory_receipt(
                stored_report.get("memory"), "completed HPO report")
            stored_adapter_calls = _validated_adapter_calls(
                stored_report.get("adapter_calls"), "completed HPO report")
            raw_this_invocation = stored_report.get(
                "stale_inflight_trial_numbers_recovered_this_invocation")
            if (
                not isinstance(raw_this_invocation, list)
                or any(
                    isinstance(number, bool) or not isinstance(number, int)
                    or number < 0
                    for number in raw_this_invocation
                )
                or len(raw_this_invocation) != len(set(raw_this_invocation))
                or not set(raw_this_invocation).issubset(
                    all_recovered_trial_numbers)
            ):
                raise ContractError(
                    "completed HPO receipt has invalid invocation recovery list")
            stored_active_wall = _required_json_nonnegative_float(
                stored_report, "hpo_active_wall_seconds_cumulative")
            _complete_or_verify_active_wall_checkpoint(
                heartbeat_path,
                descriptor_sha256,
                stored_active_wall,
                allow_explicit_recovery=False,
            )
            if (
                not budget_complete_before
                or stored_active_wall != previous_active_wall
                or checkpoint.get("study_counts") != counts_before
                or stored_report.get("study_name") != study.study_name
                or stored_report.get("counts_after") != counts_before
                or stored_report.get("useful_budget") != USEFUL_TRIALS
                or stored_history != interruption_history
                or stored_recovery != study_recovery_events
                or stored_report.get("all_stale_trial_numbers_recovered")
                != all_recovered_trial_numbers
                or stored_report.get(
                    "checkpoint_files_removed_during_hpo")
                != previous_checkpoint_removals
                or stored_memory != previous_memory
                or stored_adapter_calls != previous_adapter_calls
                or not isinstance(
                    stored_report.get("timing_complete"), bool)
                or not isinstance(
                    stored_report.get("memory_complete"), bool)
                or checkpoint.get("timing_complete")
                is not stored_report.get("timing_complete")
                or checkpoint.get("memory_complete")
                is not stored_report.get("memory_complete")
            ):
                raise ContractError(
                    "completed HPO compute receipt differs from study/history"
                )
            return {"study": study, "report": stored_report}

    prior_compute_without_complete_receipt = (
        stored_report is None
        and (
            counts_before["total"] > 0
            or checkpoint is not None
            or heartbeat_checkpoint is not None
        )
    )
    old_history = list(interruption_history)
    interruption_history = _merge_hpo_interruption_history(
        interruption_history,
        heartbeat_checkpoint,
        recover_stale=recover_stale,
        completed_receipt_missing=prior_compute_without_complete_receipt,
    )
    timing_complete = (
        not study_recovery_events
        and not any(
            event["active_time_tail_may_be_incomplete"]
            for event in interruption_history
        )
    )
    memory_complete_so_far = (
        not study_recovery_events and not interruption_history
    )

    if checkpoint is not None and checkpoint.get("status") == "completed":
        if stored_report is None and not recover_stale:
            raise ContractError(
                "completed HPO checkpoint lacks its authenticated compute "
                "receipt; explicit --recover-stale is required")
    if interruption_history != old_history:
        _atomic_json(hpo_checkpoint_path, {
            "schema": BENCHMARK_SCHEMA,
            "classification": DISCLAIMER,
            "status": "resume-preflight",
            "descriptor_sha256": descriptor_sha256,
            "study_name": study.study_name,
            "study_counts": counts_before,
            "active_wall_seconds_cumulative": previous_active_wall,
            "checkpoint_files_removed": previous_checkpoint_removals,
            "hpo_interruption_history": interruption_history,
            "optuna_recovery_events": study_recovery_events,
            "timing_complete": timing_complete,
            "memory_complete": False,
            "memory": previous_memory,
            "adapter_calls": previous_adapter_calls,
            "updated_utc": _utc_now(),
        })

    invocation_start = time.perf_counter()
    checkpoint_files_removed = previous_checkpoint_removals
    did_run_trials = counts_before["useful"] < USEFUL_TRIALS
    heartbeat: _ActiveWallHeartbeat | None = None
    memory_monitor: _PeakMemoryMonitor | None = None
    adapter_calls: Mapping[str, int | None] = (
        previous_adapter_calls
        if previous_adapter_calls is not None
        else {"get_cache": None, "canonical_split": None}
    )

    def callback(callback_study: Any, completed_trial: Any) -> None:
        nonlocal checkpoint_files_removed
        removal = _clean_exact_trial_weights(
            weights_dir,
            str(config["name"]),
            [completed_trial],
        )
        checkpoint_files_removed += int(removal["exact_files_removed"])
        _write_trial_csv(trial_csv, callback_study, descriptor_sha256)
        counts = _study_counts(optuna, callback_study)
        active_wall = (
            heartbeat.current_seconds()
            if heartbeat is not None
            else previous_active_wall
        )
        memory_checkpoint = _merge_memory_receipts(
            previous_memory,
            (
                memory_monitor.snapshot()
                if memory_monitor is not None
                else None
            ),
        )
        _atomic_json(hpo_checkpoint_path, {
            "schema": BENCHMARK_SCHEMA,
            "classification": DISCLAIMER,
            "status": "optimizing",
            "descriptor_sha256": descriptor_sha256,
            "study_name": callback_study.study_name,
            "study_counts": counts,
            "active_wall_seconds_cumulative": active_wall,
            "checkpoint_files_removed": checkpoint_files_removed,
            "hpo_interruption_history": interruption_history,
            "optuna_recovery_events": study_recovery_events,
            "timing_complete": timing_complete,
            "memory_complete": False,
            "memory": memory_checkpoint,
            "adapter_calls": dict(adapter_calls),
            "updated_utc": _utc_now(),
        })
        _atomic_json(progress_path, {
            "schema": BENCHMARK_SCHEMA,
            "classification": DISCLAIMER,
            "status": "optimizing",
            "descriptor_sha256": descriptor_sha256,
            "study_name": callback_study.study_name,
            "study_counts": counts,
            "hpo_wall_seconds_this_invocation": (
                time.perf_counter() - invocation_start
            ),
            "updated_utc": _utc_now(),
        })
        print(
            f"[R11 COMPUTE] terminal={counts['useful']}/{USEFUL_TRIALS} "
            f"complete={counts['complete']} pruned={counts['pruned']} "
            f"failed={counts['failed']}"
        )

    current_memory: dict[str, int | None] | None = None
    if did_run_trials:
        objective = trainer.Objective(
            config=dict(config),
            dataset_name=STUDY_DATASET_NAME,
            n_epochs=EPOCHS,
            cache_dir=str(run_dir / "fixture_adapter"),
            output_dir=str(weights_dir),
        )
        _write_trial_csv(trial_csv, study, descriptor_sha256)
        with _patched_trainer_fixture(
            trainer,
            runtime["np"],
            fixture,
            splits,
            config,
        ) as live_adapter_calls:
            adapter_calls = live_adapter_calls
            with (
                _suppress_optuna_value_logging(optuna),
                _PeakMemoryMonitor(torch) as live_memory_monitor,
                _ActiveWallHeartbeat(
                    heartbeat_path,
                    descriptor_sha256,
                    previous_active_wall,
                ) as heartbeat,
            ):
                memory_monitor = live_memory_monitor
                _synchronize_cuda(torch)
                start = time.perf_counter()
                quiet_study = _QuietProtocolStudy(
                    study,
                    production_callback=pipeline.print_best_callback,
                    receipt_callback=callback,
                )
                pipeline._execute_protocol_study(
                    quiet_study,
                    objective,
                    plan,
                )
                _synchronize_cuda(torch)
                wall_seconds = time.perf_counter() - start
            current_memory = live_memory_monitor.result
    else:
        wall_seconds = 0.0

    if did_run_trials and current_memory is None:
        raise ContractError("HPO memory monitor did not publish its receipt")

    counts_after = _study_counts(optuna, study)
    if (
        counts_after["failed"]
        or counts_after["running"]
        or counts_after["waiting"]
        or counts_after["useful"] != USEFUL_TRIALS
        or counts_after["total"] != USEFUL_TRIALS
        or counts_after["complete"] < 1
    ):
        raise ContractError(
            "benchmark study did not finish the exact useful budget: "
            f"{counts_after}"
        )
    runtime["hyperparameter_policy"].validate_terminal_study(study, plan)
    _write_trial_csv(trial_csv, study, descriptor_sha256)
    rows = _trial_compute_rows(study, descriptor_sha256)
    trial_compute_sha256 = _authenticated_file_sha256(
        trial_csv, "HPO trial-compute receipt")
    if trial_compute_sha256 != _csv_rows_sha256(rows, TRIAL_CSV_FIELDS):
        raise ContractError(
            "published HPO trial-compute receipt differs from study rows")
    useful_rows = [
        row for row in rows
        if row["state"] in ("COMPLETE", "PRUNED")
    ]
    useful_durations = [
        float(row["duration_seconds"])
        for row in useful_rows
        if row["duration_seconds"] != ""
    ]
    useful_epochs_reported = [
        int(row["epochs_reported"]) for row in useful_rows
    ]
    useful_duration_sum = sum(useful_durations)
    useful_epoch_sum = sum(useful_epochs_reported)
    active_wall_cumulative = (
        heartbeat.current_seconds()
        if did_run_trials else previous_active_wall
    )
    _complete_or_verify_active_wall_checkpoint(
        heartbeat_path,
        descriptor_sha256,
        active_wall_cumulative,
        allow_explicit_recovery=(
            not did_run_trials and recover_stale
        ),
    )
    reported_memory = _merge_memory_receipts(
        previous_memory,
        current_memory if did_run_trials else None,
    )
    if reported_memory is None:
        reported_memory = {
            "rss_peak_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    memory_complete = (
        memory_complete_so_far
        and did_run_trials
        and counts_before["total"] == 0
    )
    if did_run_trials:
        memory_scope = (
            "entire HPO workload"
            if memory_complete
            else (
                "maximum of persisted invocation peaks; an interrupted "
                "in-flight tail may be higher"
            )
        )
    else:
        memory_scope = (
            "unavailable because the prior completed HPO receipt was torn"
            if previous_memory is None
            else "maximum of persisted prior-invocation peak snapshots"
        )
    report = {
        "study_name": study.study_name,
        "resumed_with_existing_trials": counts_before["total"] > 0,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "useful_budget": USEFUL_TRIALS,
        "epoch_cap_per_trial": EPOCHS,
        "useful_trial_duration_seconds_sum": useful_duration_sum,
        "useful_trial_duration_seconds_quantiles": _quantile_summary(
            runtime["np"], useful_durations),
        "useful_epochs_reported_sum": useful_epoch_sum,
        "useful_epochs_reported_quantiles": _quantile_summary(
            runtime["np"], useful_epochs_reported),
        "fatal_failure_policy": (
            "FAIL=0 is required; OOM and every other exception are uncaught "
            "and no replacement trial is permitted"
        ),
        "hpo_wall_seconds_this_invocation": wall_seconds,
        "hpo_active_wall_seconds_cumulative": active_wall_cumulative,
        "active_wall_checkpoint_interval_seconds":
            ACTIVE_WALL_HEARTBEAT_SECONDS,
        "active_wall_semantics": (
            "exact on clean exit; after abrupt stop the persisted value is a "
            "lower bound sampled at a nominal cadence, not a strict tail bound"
        ),
        "timing_complete": timing_complete,
        "hpo_interruption_history": interruption_history,
        "optuna_recovery_events": study_recovery_events,
        "stale_inflight_trial_numbers_recovered_this_invocation": [
            int(number) for number in recovered_trial_numbers
        ],
        "all_stale_trial_numbers_recovered":
            all_recovered_trial_numbers,
        "nominal_unrecorded_tail_seconds_per_abrupt_stop":
            ACTIVE_WALL_HEARTBEAT_SECONDS,
        "unrecorded_tail_bound": (
            "not guaranteed; OS scheduling, the GIL and filesystem locks can "
            "delay a heartbeat beyond its nominal cadence"
        ),
        "checkpoint_files_removed_during_hpo":
            checkpoint_files_removed,
        "trial_compute_sha256": trial_compute_sha256,
        "adapter_calls": dict(adapter_calls),
        "adapter_calls_scope": (
            "this invocation" if did_run_trials else "last persisted checkpoint"
        ),
        "memory": reported_memory,
        "memory_scope": memory_scope,
        "memory_complete": memory_complete,
    }
    if set(report) != set(HPO_REPORT_FIELDS):
        raise AssertionError("internal HPO report field contract differs")
    report_sha256 = _canonical_sha256(report)
    completed_utc = _utc_now()
    _atomic_json(hpo_checkpoint_path, {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": descriptor_sha256,
        "study_name": study.study_name,
        "study_counts": counts_after,
        "active_wall_seconds_cumulative": active_wall_cumulative,
        "checkpoint_files_removed": checkpoint_files_removed,
        "hpo_interruption_history": interruption_history,
        "optuna_recovery_events": study_recovery_events,
        "timing_complete": timing_complete,
        "memory_complete": memory_complete,
        "memory": reported_memory,
        "adapter_calls": dict(adapter_calls),
        "report": report,
        "report_sha256": report_sha256,
        "trial_compute_sha256": trial_compute_sha256,
        "completed_utc": completed_utc,
    })
    return {"study": study, "report": report}


def _select_complete_trial(
    runtime: Mapping[str, Any],
    study: Any,
) -> tuple[Any, int, str]:
    trial_state = runtime["optuna"].trial.TrialState
    selected = study.best_trial
    if selected.state != trial_state.COMPLETE:
        raise ContractError("Optuna selected trial is not COMPLETE")
    if not selected.params:
        raise ContractError("selected COMPLETE trial has no parameters")
    refit_epochs = runtime["inference"].frozen_checkpoint_epoch_count(
        selected.intermediate_values,
        max_epochs=EPOCHS,
    )
    return (
        selected,
        int(refit_epochs),
        _canonical_sha256(dict(selected.params)),
    )


def _assert_returned_values_finite(np: Any, value: Any) -> None:
    """Validate the shared helper's returned values without serialising them."""

    if isinstance(value, Mapping):
        if not value:
            raise ContractError("shared finalist helper returned an empty mapping")
        for child in value.values():
            _assert_returned_values_finite(np, child)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise ContractError("shared finalist helper returned an empty sequence")
        for child in value:
            _assert_returned_values_finite(np, child)
        return
    array = np.asarray(value)
    if array.size == 0:
        raise ContractError("shared finalist helper returned an empty array")
    if array.dtype.kind not in "biufc":
        raise ContractError(
            f"shared finalist helper returned non-numeric dtype {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ContractError(
            "shared finalist helper returned a non-finite value"
        )


def _validated_finalist_marker(
    value: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete durable finalist-refit receipt."""

    marker = dict(value)
    if set(marker) != set(FINALIST_REPORT_FIELDS):
        missing = sorted(set(FINALIST_REPORT_FIELDS) - set(marker))
        extra = sorted(set(marker) - set(FINALIST_REPORT_FIELDS))
        raise ContractError(
            "finalist completion marker fields differ; "
            f"missing={missing}, extra={extra}"
        )
    _assert_no_scientific_report_fields(marker)
    if any(marker.get(key) != expected for key, expected in identity.items()):
        raise ContractError("finalist completion marker identity differs")
    if marker.get("status") != "completed":
        raise ContractError("finalist completion marker is not completed")

    attempt_count = _required_json_int(
        marker, "attempt_count", minimum=1)
    prior_unaccepted = _required_json_int(
        marker, "prior_unaccepted_attempt_count")
    accepted_refits = _required_json_int(
        marker, "durably_accepted_refits", minimum=1)
    if accepted_refits != 1 or attempt_count != prior_unaccepted + 1:
        raise ContractError(
            "finalist completion marker has impossible attempt counters")

    expected_ints = {
        "repeat": FINALIST_REPEAT,
        "fold": FINALIST_FOLD,
        "n_splits": FINALIST_N_SPLITS,
        "n_repeats": FINALIST_N_REPEATS,
        "split_seed": FINALIST_SPLIT_SEED,
    }
    for key, expected in expected_ints.items():
        if _required_json_int(marker, key) != expected:
            raise ContractError(
                f"finalist completion marker {key} differs")
    train_state_count = _required_json_int(
        marker, "train_state_count", minimum=1)
    validation_state_count = _required_json_int(
        marker, "validation_state_count", minimum=1)
    train_sample_count = _required_json_int(
        marker, "train_sample_count", minimum=1)
    validation_sample_count = _required_json_int(
        marker, "validation_sample_count", minimum=1)
    if (
        train_sample_count != train_state_count * PASSAGES_PER_STATE
        or validation_sample_count
        != validation_state_count * PASSAGES_PER_STATE
    ):
        raise ContractError(
            "finalist state/sample counts violate the fixture block size")

    for key in (
        "selected_parameter_sha256",
        "train_states_sha256",
        "validation_states_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(marker.get(key, ""))):
            raise ContractError(
                f"finalist completion marker {key} is not SHA-256")

    elapsed = _required_json_nonnegative_float(
        marker, "scale_train_infer_seconds")
    active_wall = _required_json_nonnegative_float(
        marker, "active_wall_seconds_cumulative")
    interval = _required_json_nonnegative_float(
        marker, "active_wall_checkpoint_interval_seconds")
    nominal_tail = _required_json_nonnegative_float(
        marker, "nominal_unrecorded_tail_seconds_per_abrupt_stop")
    if (
        active_wall + 1e-9 < elapsed
        or interval != ACTIVE_WALL_HEARTBEAT_SECONDS
        or nominal_tail != ACTIVE_WALL_HEARTBEAT_SECONDS
    ):
        raise ContractError(
            "finalist completion marker timing invariants differ")

    for key in (
        "timing_complete",
        "memory_complete",
        "returned_values_finite",
        "returned_values_discarded",
    ):
        if not isinstance(marker.get(key), bool):
            raise ContractError(
                f"finalist completion marker {key} must be boolean")
    expected_complete = prior_unaccepted == 0
    if (
        marker["timing_complete"] is not expected_complete
        or marker["memory_complete"] is not expected_complete
        or marker["returned_values_finite"] is not True
        or marker["returned_values_discarded"] is not True
    ):
        raise ContractError(
            "finalist completion marker completeness flags contradict "
            "its attempt history")
    if _validated_memory_receipt(
            marker.get("memory"), "finalist completion marker") is None:
        raise ContractError(
            "finalist completion marker lacks its memory receipt")

    for key in (
        "execution_semantics",
        "active_wall_semantics",
        "unrecorded_tail_bound",
        "memory_scope",
    ):
        if not isinstance(marker.get(key), str) or not marker[key].strip():
            raise ContractError(
                f"finalist completion marker {key} must be non-empty")
    completed_utc = marker.get("completed_utc")
    try:
        parsed_completed = datetime.fromisoformat(str(completed_utc))
    except ValueError as exc:
        raise ContractError(
            "finalist completion marker completed_utc is invalid"
        ) from exc
    if (
        parsed_completed.tzinfo is None
        or parsed_completed.utcoffset()
        != timezone.utc.utcoffset(None)
    ):
        raise ContractError(
            "finalist completion marker completed_utc is not aware UTC")
    return marker


def _run_finalist_once(
    runtime: Mapping[str, Any],
    study: Any,
    config: Mapping[str, Any],
    fixture: Mapping[str, Any],
    splits: Mapping[str, Any],
    run_dir: Path,
    descriptor_sha256: str,
    *,
    recover_stale: bool,
) -> dict[str, Any]:
    marker_path = run_dir / "finalist_compute.json"
    selected, refit_epochs, selected_parameter_sha256 = (
        _select_complete_trial(runtime, study)
    )
    attempt_path = run_dir / "finalist_attempt_state.json"
    heartbeat_path = run_dir / "finalist_active_wall_heartbeat.json"
    attempt_identity = {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "descriptor_sha256": descriptor_sha256,
        "selected_trial_number": int(selected.number),
        "selected_parameter_sha256": selected_parameter_sha256,
        "frozen_checkpoint_epoch_count": int(refit_epochs),
        "helper": "training.trainer.fit_predict_finalist_fold",
    }
    if marker_path.exists():
        marker = _validated_finalist_marker(
            _read_json_mapping(
                marker_path, "finalist completion marker"),
            attempt_identity,
        )
        marker_attempt_count = marker["attempt_count"]
        marker_prior_unaccepted = marker[
            "prior_unaccepted_attempt_count"]
        marker_active_wall = _required_json_nonnegative_float(
            marker, "active_wall_seconds_cumulative")
        if not attempt_path.exists():
            raise ContractError(
                "finalist completion marker exists but its attempt state is "
                f"missing: {attempt_path}")
        previous_attempt = _read_json_mapping(
            attempt_path, "finalist attempt state")
        if any(
            previous_attempt.get(key) != value
            for key, value in attempt_identity.items()
        ):
            raise ContractError(
                f"finalist attempt identity differs: {attempt_path}")
        counters_match = (
            previous_attempt.get("attempt_count") == marker_attempt_count
            and previous_attempt.get("prior_unaccepted_attempt_count")
            == marker_prior_unaccepted
        )
        if not counters_match:
            raise ContractError(
                "finalist marker and attempt counters contradict each other")
        heartbeat_record = _load_active_wall_checkpoint(
            heartbeat_path, descriptor_sha256)
        if (
            heartbeat_record is None
            or heartbeat_record.get("status") != "completed"
            or not math.isclose(
                float(heartbeat_record["active_wall_seconds_cumulative"]),
                marker_active_wall,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ContractError(
                "finalist completion marker and active-wall heartbeat "
                "contradict each other")
        if previous_attempt.get("status") == "running":
            if not recover_stale:
                raise ContractError(
                    "a finalist completion marker has a torn attempt pointer; "
                    "explicit --recover-stale is required to repair only the "
                    "pointer without rerunning the refit")
            _atomic_json(attempt_path, {
                **attempt_identity,
                "status": "completed",
                "attempt_count": marker_attempt_count,
                "prior_unaccepted_attempt_count":
                    marker_prior_unaccepted,
                "active_wall_seconds_cumulative":
                    marker_active_wall,
                "completion_marker": marker_path.name,
                "completed_utc": marker["completed_utc"],
                "completion_pointer_repaired_utc": _utc_now(),
            })
        elif previous_attempt.get("status") == "completed":
            attempt_active_wall = _required_json_nonnegative_float(
                previous_attempt, "active_wall_seconds_cumulative")
            if (
                not math.isclose(
                    attempt_active_wall,
                    marker_active_wall,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or previous_attempt.get("completion_marker")
                != marker_path.name
                or previous_attempt.get("completed_utc")
                != marker.get("completed_utc")
            ):
                raise ContractError(
                    "finalist completed attempt contradicts its marker")
        else:
            raise ContractError(
                "finalist completion marker has an unsupported attempt "
                f"status {previous_attempt.get('status')!r}")
        return dict(marker)

    if heartbeat_path.exists() and not attempt_path.exists():
        raise ContractError(
            "finalist active-wall heartbeat exists without its earlier "
            "attempt-state record; this ordering cannot be produced by an "
            "abrupt stop and is not recoverable automatically"
        )

    np = runtime["np"]
    torch = runtime["torch"]
    inference = runtime["inference"]
    trainer = runtime["trainer"]
    strata = ["workload"] * N_STATES
    folds = inference.repeated_stratified_group_folds(
        fixture["groups"],
        splits["development_idx"],
        strata,
        n_splits=FINALIST_N_SPLITS,
        n_repeats=FINALIST_N_REPEATS,
        seed=FINALIST_SPLIT_SEED,
    )
    matching = [
        fold for fold in folds
        if fold.repeat == FINALIST_REPEAT and fold.fold == FINALIST_FOLD
    ]
    if len(matching) != 1:
        raise ContractError(
            "production repeated-fold constructor did not return exactly one "
            "requested finalist fold"
        )
    fold = matching[0]
    outer_idx = splits["indices"]["outer"]
    if (
        np.intersect1d(fold.train_idx, fold.val_idx).size
        or np.intersect1d(fold.train_idx, outer_idx).size
        or np.intersect1d(fold.val_idx, outer_idx).size
    ):
        raise ContractError("finalist workload fold crosses a state firewall")

    previous_attempt_count = 0
    prior_unaccepted_attempt_count = 0
    if attempt_path.exists():
        previous_attempt = _read_json_mapping(
            attempt_path, "finalist attempt state")
        if any(
            previous_attempt.get(key) != value
            for key, value in attempt_identity.items()
        ):
            raise ContractError(
                f"finalist attempt identity differs: {attempt_path}")
        previous_attempt_count = _required_json_int(
            previous_attempt, "attempt_count", minimum=1)
        prior_unaccepted_attempt_count = _required_json_int(
            previous_attempt, "prior_unaccepted_attempt_count")
        if (
            previous_attempt_count < 1
            or prior_unaccepted_attempt_count < 0
            or prior_unaccepted_attempt_count >= previous_attempt_count
        ):
            raise ContractError(
                f"invalid finalist attempt counters: {attempt_path}")
        previous_status = previous_attempt.get("status")
        if previous_status == "running":
            if not recover_stale:
                raise ContractError(
                    "an interrupted finalist refit is recorded; explicit "
                    "--recover-stale is required before an at-least-once retry"
                )
            prior_unaccepted_attempt_count += 1
        elif previous_status == "completed":
            raise ContractError(
                "finalist attempt says completed but its completion marker "
                "is missing")
        else:
            raise ContractError(
                f"unsupported finalist attempt status {previous_status!r}")

    previous_finalist_active_wall = _read_active_wall_checkpoint(
        heartbeat_path, descriptor_sha256)
    attempt_count = previous_attempt_count + 1
    _atomic_json(attempt_path, {
        **attempt_identity,
        "status": "running",
        "attempt_count": attempt_count,
        "prior_unaccepted_attempt_count":
            prior_unaccepted_attempt_count,
        "active_wall_seconds_before_attempt": previous_finalist_active_wall,
        "started_utc": _utc_now(),
        "pid": os.getpid(),
    })

    with _PeakMemoryMonitor(torch) as memory:
        with _ActiveWallHeartbeat(
            heartbeat_path,
            descriptor_sha256,
            previous_finalist_active_wall,
        ) as finalist_heartbeat:
            _synchronize_cuda(torch)
            start = time.perf_counter()
            returned = trainer.fit_predict_finalist_fold(
                dict(config),
                dict(selected.params),
                fixture["X"],
                fixture["y"],
                fixture["groups"],
                fold,
                SEED,
                n_epochs=refit_epochs,
                max_epochs=EPOCHS,
                n_scour_heads=len(TARGET_SUPPORTS),
            )
            _synchronize_cuda(torch)
            elapsed = time.perf_counter() - start
        finalist_active_wall = finalist_heartbeat.current_seconds()
    _assert_returned_values_finite(np, returned)
    if isinstance(returned, Mapping) and "state" in returned:
        if not np.array_equal(
            np.asarray(returned["state"], dtype=np.int64),
            fold.val_states,
        ):
            raise ContractError(
                "shared finalist helper returned state IDs that differ from "
                "the requested validation fold"
            )
    del returned

    marker = {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": descriptor_sha256,
        "helper": "training.trainer.fit_predict_finalist_fold",
        "selected_trial_number": int(selected.number),
        "selected_parameter_sha256": selected_parameter_sha256,
        "frozen_checkpoint_epoch_count": int(refit_epochs),
        "durably_accepted_refits": 1,
        "execution_semantics": (
            "exactly one durably accepted completion; at-least-once "
            "execution after explicit recovery. A prior unaccepted attempt "
            "may have physically completed before its marker was published."
        ),
        "attempt_count": attempt_count,
        "prior_unaccepted_attempt_count":
            prior_unaccepted_attempt_count,
        "repeat": FINALIST_REPEAT,
        "fold": FINALIST_FOLD,
        "n_splits": FINALIST_N_SPLITS,
        "n_repeats": FINALIST_N_REPEATS,
        "split_seed": FINALIST_SPLIT_SEED,
        "train_state_count": int(len(fold.train_states)),
        "validation_state_count": int(len(fold.val_states)),
        "train_sample_count": int(len(fold.train_idx)),
        "validation_sample_count": int(len(fold.val_idx)),
        "train_states_sha256": _index_sha256(np, fold.train_states),
        "validation_states_sha256": _index_sha256(np, fold.val_states),
        "scale_train_infer_seconds": elapsed,
        "active_wall_seconds_cumulative": finalist_active_wall,
        "active_wall_checkpoint_interval_seconds":
            ACTIVE_WALL_HEARTBEAT_SECONDS,
        "active_wall_semantics": (
            "exact on clean exit; after abrupt stop the persisted value is a "
            "lower bound sampled at a nominal cadence, not a strict tail bound"
        ),
        "timing_complete": prior_unaccepted_attempt_count == 0,
        "nominal_unrecorded_tail_seconds_per_abrupt_stop":
            ACTIVE_WALL_HEARTBEAT_SECONDS,
        "unrecorded_tail_bound": (
            "not guaranteed; OS scheduling, the GIL and filesystem locks can "
            "delay a heartbeat beyond its nominal cadence"
        ),
        "memory": memory.result,
        "memory_scope": (
            "durably accepted attempt only; may understate peaks from prior "
            "unaccepted attempts"
        ),
        "memory_complete": prior_unaccepted_attempt_count == 0,
        "returned_values_finite": True,
        "returned_values_discarded": True,
        "completed_utc": _utc_now(),
    }
    marker = _validated_finalist_marker(marker, attempt_identity)
    _atomic_json(marker_path, marker)
    _atomic_json(attempt_path, {
        **attempt_identity,
        "status": "completed",
        "attempt_count": attempt_count,
        "prior_unaccepted_attempt_count":
            prior_unaccepted_attempt_count,
        "active_wall_seconds_cumulative": finalist_active_wall,
        "completion_marker": marker_path.name,
        "completed_utc": marker["completed_utc"],
    })
    return marker


def _clean_exact_trial_weights(
    weights_dir: Path,
    model_name: str,
    trials: Sequence[Any],
) -> dict[str, Any]:
    """Remove only trial-number-derived benchmark paths inside weights_dir."""

    weights_dir = weights_dir.resolve()
    removed: list[str] = []
    for trial in sorted(trials, key=lambda item: int(item.number)):
        path = (
            weights_dir
            / f"weights_{model_name}_trial_{int(trial.number)}.pth"
        ).resolve()
        _require_within(path, weights_dir, "trial weight cleanup path")
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return {
        "exact_paths_considered": len(trials),
        "exact_files_removed": len(removed),
        "unexpected_paths_touched": False,
    }


def _materialize_study_storage_receipt(run_dir: Path) -> Path:
    """Publish a stable SQLite backup after every study operation is finished.

    Hashing the live ``study.sqlite3`` plus optional WAL/SHM files is not a
    stable completion receipt: SQLite may checkpoint or remove those sidecars
    when the process closes, making an untouched second invocation appear
    corrupt.  ``Connection.backup`` captures one committed logical snapshot in
    a standalone database, which is then atomically published and never opened
    by Optuna.
    """

    source = run_dir / "study.sqlite3"
    destination = run_dir / "study_receipt.sqlite3"
    if not source.is_file():
        raise ContractError(f"Optuna study database is missing: {source}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    try:
        with (
            contextlib.closing(
                sqlite3.connect(source_uri, uri=True)
            ) as source_connection,
            contextlib.closing(
                sqlite3.connect(str(temporary))
            ) as receipt_connection,
        ):
            source_connection.backup(receipt_connection)
            receipt_connection.commit()
            row = receipt_connection.execute("PRAGMA quick_check").fetchone()
            if row != ("ok",):
                raise ContractError(
                    f"SQLite backup quick_check failed for {temporary}: {row!r}"
                )
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _atomic_replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _storage_snapshot(run_dir: Path) -> dict[str, dict[str, Any]]:
    receipt = run_dir / "study_receipt.sqlite3"
    captured = _regular_file_snapshot(
        receipt,
        "immutable study receipt",
        max_bytes=SQLITE_SNAPSHOT_MAX_BYTES,
    )
    assert captured is not None
    return {
        receipt.name: _public_file_snapshot(captured)
    }


def _coordination_receipt_snapshot(
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Authenticate the one execution and one CUDA-capacity public receipt."""

    paths: dict[str, Path] = {}
    for role, directory_name in (
        ("execution_block", "execution_receipts"),
        ("cuda_capacity", "capacity_receipts"),
    ):
        directory = run_dir / directory_name
        try:
            directory_stat = os.stat(directory, follow_symlinks=False)
        except FileNotFoundError:
            matches: list[Path] = []
        except OSError as exc:
            raise ContractError(
                f"cannot inspect {role} receipt directory: {directory}"
            ) from exc
        else:
            if stat.S_ISLNK(directory_stat.st_mode):
                raise ContractError(
                    f"{role} receipt directory must not be a symlink: "
                    f"{directory}"
                )
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise ContractError(
                    f"{role} receipt path is not a directory: {directory}"
                )
            try:
                with os.scandir(directory) as entries:
                    matches = sorted(
                        (Path(entry.path) for entry in entries
                         if entry.name.endswith(".json")),
                        key=lambda path: path.name,
                    )
            except OSError as exc:
                raise ContractError(
                    f"cannot enumerate {role} receipts: {directory}"
                ) from exc
        if len(matches) != 1:
            raise ContractError(
                f"expected exactly one regular {role} receipt below "
                f"{directory}, found {matches}"
            )
        paths[role] = matches[0]
    snapshot: dict[str, dict[str, Any]] = {}
    for role, path in sorted(paths.items()):
        captured = _regular_file_snapshot(
            path,
            f"{role} coordination receipt",
            max_bytes=JSON_SNAPSHOT_MAX_BYTES,
        )
        assert captured is not None
        snapshot[role] = {
            "relative_path": path.relative_to(run_dir).as_posix(),
            **_public_file_snapshot(captured),
        }
    return snapshot


def _immutable_evidence_snapshot(
    run_dir: Path,
    *,
    study_storage_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash every durable artifact that the completed summary announces."""

    snapshot: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in IMMUTABLE_EVIDENCE_FILES:
        path = run_dir / name
        if name == "study_receipt.sqlite3" and study_storage_snapshot is not None:
            stored = study_storage_snapshot.get(name)
            if not isinstance(stored, Mapping):
                missing.append(name)
                continue
            snapshot[name] = {
                "size_bytes": int(stored.get("size_bytes", -1)),
                "sha256": str(stored.get("sha256", "")),
            }
            continue
        maximum = (
            SQLITE_SNAPSHOT_MAX_BYTES
            if path.suffix == ".sqlite3"
            else REPORT_SNAPSHOT_MAX_BYTES
        )
        captured = _regular_file_snapshot(
            path,
            f"immutable benchmark evidence {name}",
            max_bytes=maximum,
            allow_missing=True,
        )
        if captured is None:
            missing.append(name)
            continue
        snapshot[name] = _public_file_snapshot(captured)
    if missing:
        raise ContractError(
            f"immutable benchmark evidence is incomplete: {missing}")
    return snapshot


def _hyperparameter_execution_record(
    hyperparameter_policy: Any,
    hyperparameter_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the complete policy-validated benchmark run plan."""

    validated_plan = hyperparameter_policy.validate_run_plan(
        dict(hyperparameter_plan)
    )
    if not isinstance(validated_plan, dict):
        raise ContractError(
            "hyperparameter policy did not return a validated run-plan object"
        )
    plan_sha256 = hyperparameter_policy.canonical_json_sha256(validated_plan)
    record = {
        "schema": BENCHMARK_HYPERPARAMETER_EXECUTION_SCHEMA,
        "campaign_run_tag": validated_plan["campaign_run_tag"],
        "execution_receipt_sha256":
            validated_plan["execution_receipt_sha256"],
        "block_reference_manifest_sha256":
            validated_plan["block_reference_manifest_sha256"],
        "validated_run_plan_sha256": plan_sha256,
        "validated_run_plan": validated_plan,
    }
    _assert_no_scientific_report_fields(record)
    return record


def _validate_benchmark_hyperparameter_execution(
    record: Any,
    *,
    hyperparameter_policy: Any,
    descriptor_sha256: str,
    protocol_core_sha256: str,
    coordination_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on any restart-time run/receipt/reference substitution."""

    outer_fields = {
        "schema",
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
        "validated_run_plan_sha256",
        "validated_run_plan",
    }
    if not isinstance(record, Mapping) or set(record) != outer_fields:
        raise ContractError(
            "benchmark hyperparameter-execution fields differ from contract"
        )
    if record["schema"] != BENCHMARK_HYPERPARAMETER_EXECUTION_SCHEMA:
        raise ContractError(
            "benchmark hyperparameter-execution schema differs")
    plan = record["validated_run_plan"]
    if not isinstance(plan, Mapping) or set(plan) != HYPERPARAMETER_RUN_PLAN_FIELDS:
        raise ContractError(
            "benchmark validated run-plan fields differ from contract"
        )
    plan = dict(plan)
    try:
        policy_validated_plan = hyperparameter_policy.validate_run_plan(
            dict(plan)
        )
        policy_plan_sha256 = hyperparameter_policy.canonical_json_sha256(
            policy_validated_plan
        )
    except Exception as exc:
        raise ContractError(
            "benchmark run plan fails the registered hyperparameter policy"
        ) from exc
    if policy_validated_plan != plan:
        raise ContractError(
            "registered hyperparameter policy changed the published run plan"
        )
    expected_run_tag = f"benchmark-{descriptor_sha256}"
    execution_snapshot = coordination_receipts.get("execution_block")
    if not isinstance(execution_snapshot, Mapping):
        raise ContractError(
            "benchmark coordination snapshot lacks execution receipt")
    expected_receipt_sha256 = execution_snapshot.get("sha256")
    if (
        not isinstance(expected_receipt_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256)
    ):
        raise ContractError(
            "benchmark execution receipt snapshot has invalid SHA-256")
    plan_sha256 = _canonical_sha256(plan)
    if (
        record["validated_run_plan_sha256"] != plan_sha256
        or policy_plan_sha256 != plan_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256)
    ):
        raise ContractError(
            "benchmark validated run-plan SHA-256 differs from its contents"
        )
    if (
        record["campaign_run_tag"] != expected_run_tag
        or plan["campaign_run_tag"] != expected_run_tag
    ):
        raise ContractError("benchmark campaign run_tag differs")
    if (
        record["execution_receipt_sha256"] != expected_receipt_sha256
        or plan["execution_receipt_sha256"] != expected_receipt_sha256
    ):
        raise ContractError(
            "benchmark hyperparameter plan does not cite the exact execution "
            "receipt bytes"
        )
    if (
        record["block_reference_manifest_sha256"] is not None
        or plan["block_reference_manifest_sha256"] is not None
    ):
        raise ContractError(
            "benchmark block anchor cannot cite a reference manifest")

    exact_plan_fields = {
        "schema": HYPERPARAMETER_RUN_PLAN_SCHEMA,
        "mode": "anchor_hpo",
        "execution_block": EXECUTION_BLOCK,
        "anchor_stage": ANCHOR_STAGE,
        "stage": ANCHOR_STAGE,
        "dataset": STUDY_DATASET_NAME,
        "protocol_hash": descriptor_sha256,
        "protocol_core_hash": protocol_core_sha256,
        "architecture": "PAA_LSTM_NHiTS",
        "seed": SEED,
        "active_dofs": list(DOFS),
        "effective_n_trials": USEFUL_TRIALS,
        "effective_use_pruner": USE_PRUNER,
        "requested_n_trials": USEFUL_TRIALS,
        "requested_use_pruner": USE_PRUNER,
        "hyperparameter_manifest_sha256": None,
        "hyperparameter_source": None,
    }
    for key, expected in exact_plan_fields.items():
        if plan.get(key) != expected:
            raise ContractError(
                f"benchmark validated run-plan field {key!r} differs")
    if (
        not isinstance(plan.get("policy_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", plan["policy_sha256"])
    ):
        raise ContractError(
            "benchmark validated run plan lacks policy SHA-256")
    return plan


def _completion_pointer_history_valid(
    run_dir: Path,
    state: Mapping[str, Any],
) -> bool:
    record = state.get("prior_completion_pointer")
    if record is None:
        return "completion_pointer_recovery" not in state
    if not isinstance(record, Mapping) or set(record) != {
        "status", "archive", "sha256"
    }:
        return False
    if not isinstance(record["status"], str):
        return False
    archive = record["archive"]
    digest = record["sha256"]
    if archive is None or digest is None:
        return (
            archive is None
            and digest is None
            and record["status"] == "missing"
        )
    if (
        not isinstance(archive, str)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return False
    path = (run_dir / archive).resolve()
    history_dir = (run_dir / "completion_pointer_history").resolve()
    if not _is_within(path, history_dir):
        return False
    try:
        snapshot = _regular_file_snapshot(
            path,
            "prior benchmark completion pointer",
            max_bytes=JSON_SNAPSHOT_MAX_BYTES,
        )
    except ContractError:
        return False
    assert snapshot is not None
    return snapshot["sha256"] == digest


def _completed_summary_if_valid(
    run_dir: Path,
    *,
    hyperparameter_policy: Any,
    descriptor_sha256: str,
    protocol_core_sha256: str,
    git_sha: str,
    run_id: str,
    fixture_snapshot: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    recover_torn_state: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Authenticate an immutable receipt and, explicitly, repair its pointer.

    ``summary.json`` is published before ``run_state.json``.  A power loss in
    that narrow interval must not force the expensive workload to run again,
    but it must also never authorize an implicit repair.  With
    ``recover_torn_state=True`` an independently authenticated summary may
    recreate only the completion pointer; the summary itself is preserved
    byte-for-byte.  A contradictory *completed* pointer remains a hard error.
    """

    summary_path = run_dir / "summary.json"
    state_path = run_dir / "run_state.json"
    summary_snapshot = _regular_file_snapshot(
        summary_path,
        "existing benchmark summary",
        max_bytes=JSON_SNAPSHOT_MAX_BYTES,
        capture_bytes=True,
        allow_missing=True,
    )
    if summary_snapshot is None:
        state_snapshot = _regular_file_snapshot(
            state_path,
            "incomplete benchmark run state",
            max_bytes=JSON_SNAPSHOT_MAX_BYTES,
            capture_bytes=True,
            allow_missing=True,
        )
        if state_snapshot is not None:
            state = _json_mapping_from_snapshot(
                state_snapshot,
                state_path,
                "incomplete benchmark run state",
            )
            if state.get("status") == "completed":
                raise ContractError(
                    "run_state says completed but summary.json is missing")
        return None, False
    summary = _json_mapping_from_snapshot(
        summary_snapshot,
        summary_path,
        "existing benchmark summary",
    )
    _assert_no_scientific_report_fields(summary)
    identity = summary.get("identity", {})
    hashes = summary.get("hashes", {})
    if not isinstance(identity, Mapping) or not isinstance(hashes, Mapping):
        raise ContractError(
            "existing benchmark summary lacks identity/hash mappings")
    storage_snapshot = _storage_snapshot(run_dir)
    coordination_snapshot = _coordination_receipt_snapshot(run_dir)
    immutable_snapshot = _immutable_evidence_snapshot(
        run_dir,
        study_storage_snapshot=storage_snapshot,
    )
    _validate_benchmark_hyperparameter_execution(
        summary.get("hyperparameter_execution"),
        hyperparameter_policy=hyperparameter_policy,
        descriptor_sha256=descriptor_sha256,
        protocol_core_sha256=protocol_core_sha256,
        coordination_receipts=coordination_snapshot,
    )
    execution_attestation = summary.get("execution_attestation")
    capacity_preflight = summary.get("capacity_preflight")
    summary_valid = (
        summary.get("schema") == BENCHMARK_SCHEMA
        and summary.get("classification") == DISCLAIMER
        and summary.get("status") == "completed"
        and summary.get("descriptor_sha256") == descriptor_sha256
        and identity.get("git_sha") == git_sha
        and identity.get("run_id") == run_id
        and identity.get("run_directory") == str(run_dir)
        and identity.get("git_tracked_dirty_at_start") is False
        and hashes.get("source_before") == dict(source_hashes)
        and hashes.get("source_after") == dict(source_hashes)
        and isinstance(hashes.get("fixture_before"), Mapping)
        and isinstance(hashes.get("fixture_after"), Mapping)
        and _snapshot_equal(
            hashes["fixture_before"], fixture_snapshot)
        and _snapshot_equal(
            hashes["fixture_after"], fixture_snapshot)
        and hashes.get("study_storage") == storage_snapshot
        and hashes.get("coordination_receipts")
        == coordination_snapshot
        and hashes.get("immutable_evidence")
        == immutable_snapshot
        and isinstance(execution_attestation, Mapping)
        and execution_attestation.get("receipt_sha256")
        == coordination_snapshot["execution_block"]["sha256"]
        and isinstance(capacity_preflight, Mapping)
        and capacity_preflight.get("passed") is True
        and capacity_preflight.get("envelope_file_sha256")
        == coordination_snapshot["cuda_capacity"]["sha256"]
    )
    if not summary_valid:
        raise ContractError(
            "existing summary/storage no longer matches the "
            "immutable completed benchmark receipt")

    summary_sha256 = str(summary_snapshot["sha256"])
    state: dict[str, Any] | None = None
    state_snapshot = _regular_file_snapshot(
        state_path,
        "benchmark completion state",
        max_bytes=JSON_SNAPSHOT_MAX_BYTES,
        capture_bytes=True,
        allow_missing=True,
    )
    state_read_error: Exception | None = None
    if state_snapshot is not None:
        try:
            state = _json_mapping_from_snapshot(
                state_snapshot,
                state_path,
                "benchmark completion state",
            )
        except Exception as exc:
            state_read_error = exc
    state_valid = (
        state is not None
        and state.get("schema") == BENCHMARK_SCHEMA
        and state.get("classification") == DISCLAIMER
        and state.get("status") == "completed"
        and state.get("descriptor_sha256") == descriptor_sha256
        and state.get("summary") == "summary.json"
        and state.get("summary_sha256") == summary_sha256
        and _completion_pointer_history_valid(run_dir, state)
    )
    if state_valid:
        return summary, False

    if state is not None and state.get("status") == "completed":
        raise ContractError(
            "completed run_state contradicts the authenticated immutable "
            "summary; refusing to repair it")
    if not recover_torn_state:
        detail = (
            f"unreadable run_state ({type(state_read_error).__name__})"
            if state_read_error is not None
            else "missing or incomplete run_state"
        )
        raise ContractError(
            f"authenticated summary has a torn completion pointer: {detail}. "
            "Re-run with --recover-stale to repair only run_state.json; the "
            "summary will remain byte-for-byte unchanged."
        )

    prior_pointer = {
        "status": "missing",
        "archive": None,
        "sha256": None,
    }
    if state_snapshot is not None:
        prior_sha256 = str(state_snapshot["sha256"])
        history_dir = run_dir / "completion_pointer_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        archive_path = history_dir / (
            f"recovered-{time.time_ns()}-{uuid.uuid4().hex}.bin"
        )
        state_bytes = state_snapshot.get("bytes")
        if not isinstance(state_bytes, bytes):
            raise ContractError(
                "completion state snapshot did not retain archive bytes")
        _atomic_bytes(archive_path, state_bytes)
        prior_pointer = {
            "status": (
                str(state.get("status", "unknown"))
                if state is not None
                else "unreadable"
            ),
            "archive": archive_path.relative_to(run_dir).as_posix(),
            "sha256": prior_sha256,
        }

    repaired_state = {
        "schema": BENCHMARK_SCHEMA,
        "classification": DISCLAIMER,
        "status": "completed",
        "descriptor_sha256": descriptor_sha256,
        "completed_utc": str(
            summary.get("command", {}).get(
                "this_invocation_measurement_completed_utc", "")
        ),
        "summary": "summary.json",
        "summary_sha256": summary_sha256,
        "completion_pointer_repaired_utc": _utc_now(),
        "completion_pointer_recovery": (
            "explicit --recover-stale; authenticated summary preserved"
        ),
        "prior_completion_pointer": prior_pointer,
    }
    _atomic_json(state_path, repaired_state)
    return summary, True


def _descriptor(
    *,
    git_sha: str,
    fixture_before: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    # Keep the workload hash comparable across machines: hostnames, absolute
    # paths, and copy-specific mtimes belong in the run report, not the
    # descriptor.  File bytes/sizes and source hashes are the portable identity.
    portable_fixture = _portable_file_identity(fixture_before)
    core = {
        "schema": "ttbi-r11-benchmark-protocol-core-v1",
        "classification": DISCLAIMER,
        "execution_blocking": {
            "block": EXECUTION_BLOCK,
            "anchor_stage": ANCHOR_STAGE,
            "same_physical_host_and_gpu_within_block": True,
        },
        "workload": {
            "derived_files": {
                key: value
                for key, value in portable_fixture.items()
                if key.startswith("derived_")
            },
            "derivation_recipe": DERIVATION_RECIPE,
            "states": N_STATES,
            "passages_per_state": PASSAGES_PER_STATE,
            "channels": len(DOFS),
            "heads": len(TARGET_SUPPORTS) + len(BEARING_TARGETS),
        },
        "training": {
            "architecture": "PAA_LSTM_NHiTS",
            "seed": SEED,
            "epoch_cap": EPOCHS,
        },
        "source_sha256": dict(source_hashes),
    }
    return {
        "schema": DESCRIPTOR_SCHEMA,
        "classification": DISCLAIMER,
        "classification_detail": DISCLAIMER_DETAIL,
        "purpose": "production-path compute and throughput measurement only",
        "git_sha": git_sha,
        "core": core,
        "rung": {
            "stage": ANCHOR_STAGE,
            "dataset": STUDY_DATASET_NAME,
            "execution_block": EXECUTION_BLOCK,
            "execution_anchor": ANCHOR_STAGE,
        },
        "fixture": {
            "files": portable_fixture,
            "expected_features_shape": list(DEFAULT_X_SHAPE),
            "expected_labels_shape": list(DEFAULT_Y_SHAPE),
            "state_count": N_STATES,
            "passages_per_state": PASSAGES_PER_STATE,
            "read_mode": "numpy mmap_mode=r, allow_pickle=false",
            "sampled_finiteness_count": SAMPLED_FINITE_COUNT,
            "full_finiteness_chunk_samples": FULL_FINITE_CHUNK_SAMPLES,
        },
        "configuration": {
            "architecture": "PAA_LSTM_NHiTS",
            "method": "PAA",
            "dofs": list(DOFS),
            "task": "regression",
            "target_supports": list(TARGET_SUPPORTS),
            "bearing_targets": list(BEARING_TARGETS),
            "seed": SEED,
        },
        "workload_size_context": {
            "benchmark_state_count": N_STATES,
            "registered_l60_state_count": 450,
            "registered_l99_state_count": 475,
            "largest_rung_state_count": 475,
            "largest_rung_sample_count_ratio": 1.0,
            "interpretation": (
                "the benchmark uses the largest registered state/sample count; "
                "it remains throughput evidence, not a model-quality estimate"
            ),
        },
        "study_policy": {
            "useful_trials": USEFUL_TRIALS,
            "epochs": EPOCHS,
            "pruner_enabled": USE_PRUNER,
            "useful_states": ["COMPLETE", "PRUNED"],
            "failed_trials_allowed": 0,
            "running_or_waiting_trials_allowed": 0,
            "oom_policy": "fatal, uncaught, no replacement",
            "sampler_seed": SEED,
            "production_helpers": [
                "training.pipeline._create_or_resume_study",
                "core.hyperparameter_policy.derive_execution_plan",
                "training.pipeline._stamp_study_protocol",
                "training.pipeline._execute_protocol_study",
                "training.trainer.Objective",
                "core.capacity_preflight.ensure_capacity_preflight",
            ],
        },
        "split_policy": {
            "hpo_assignment_pattern": [
                "train", "outer", "inner", "train", "train"
            ],
            "seed": SEED,
            "state_grouped": True,
            "stratum": "workload",
        },
        "finalist_policy": {
            "helper": "training.trainer.fit_predict_finalist_fold",
            "fold_constructor": (
                "core.statistical_inference."
                "repeated_stratified_group_folds"
            ),
            "checkpoint_rule": (
                "core.statistical_inference."
                "frozen_checkpoint_epoch_count"
            ),
            "n_splits": FINALIST_N_SPLITS,
            "n_repeats": FINALIST_N_REPEATS,
            "repeat": FINALIST_REPEAT,
            "fold": FINALIST_FOLD,
            "seed": FINALIST_SPLIT_SEED,
            "stratum": "workload",
            "returned_values_policy": "validate finite in memory, then discard",
            "execution_semantics": (
                "exactly one durably accepted completion; at-least-once "
                "execution only after explicit recovery. A prior unaccepted "
                "attempt may have physically completed before publication."
            ),
            "interrupted_attempt_timing": (
                "cumulative active wall uses a nominal-cadence heartbeat; an "
                "abrupt-stop receipt is a lower bound, not a strict tail "
                "bound, and prior-attempt memory peaks are labelled incomplete"
            ),
            "explicit_helper_arguments": {
                "max_epochs": EPOCHS,
                "n_scour_heads": len(TARGET_SUPPORTS),
            },
        },
        "report_policy": {
            "json_csv": "compute/provenance fields only",
            "scientific_values": "not exported or printed",
            "optuna_console_logging": (
                "WARNING or higher during optimization; INFO trial values "
                "suppressed and prior verbosity restored"
            ),
            "sqlite": (
                "descriptor-verified benchmark-only Optuna storage; final "
                "logical state atomically copied to a standalone immutable "
                "SQLite receipt so WAL/SHM lifecycle cannot change its hash"
            ),
            "active_wall_checkpoint_interval_seconds":
                ACTIVE_WALL_HEARTBEAT_SECONDS,
            "active_wall_abrupt_stop_semantics": (
                "persisted lower bound at nominal cadence; no mathematical "
                "maximum tail because scheduling/GIL/filesystem delays apply"
            ),
            "failed_trial_policy": (
                "no completed receipt is possible when any FAIL exists"
            ),
        },
        "source_sha256": dict(source_hashes),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"{DISCLAIMER}: run the fixed R11 production-path compute benchmark"
        )
    )
    parser.add_argument(
        "--fixture-dir",
        default=str(DEFAULT_FIXTURE_DIR),
        help="legacy fixture-cache directory (read-only)",
    )
    parser.add_argument(
        "--fixture-x",
        default=None,
        help="explicit feature .npy path (otherwise the pinned glob is used)",
    )
    parser.add_argument(
        "--fixture-y",
        default=None,
        help="explicit label .npy path (otherwise the pinned glob is used)",
    )
    parser.add_argument(
        "--recover-stale",
        action="store_true",
        help=(
            "explicitly archive stale lock/pointer evidence, acknowledge "
            "interrupted HPO/finalist segments, and mark stale "
            "RUNNING/WAITING trials FAIL; never delete the study/output"
        ),
    )
    return parser


def _print_banner() -> None:
    border = "!" * 78
    print(border)
    print(f"!!! {DISCLAIMER} !!!")
    print(DISCLAIMER_DETAIL)
    print("No model-quality number from this run may be reported or cited.")
    print(border)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _print_banner()
    started_utc = _utc_now()
    started = time.perf_counter()
    repo = Path(__file__).resolve().parent
    if Path.cwd().resolve() != repo:
        raise ContractError(
            f"run from repository root {repo}; current directory is {Path.cwd()}"
        )
    command_argv = [sys.executable, str(Path(__file__).resolve())]
    command_argv.extend(list(sys.argv[1:] if argv is None else argv))
    command = _render_command(command_argv)

    source_fixture_paths = _resolve_fixture_paths(args)
    source_fixture_before = _snapshot_files(source_fixture_paths)
    source_before = _source_hashes(repo)
    git_sha = _git_sha(repo)
    _assert_sources_tracked_at_head(repo)
    git_tracked_dirty_at_start = _git_dirty(repo)
    if git_tracked_dirty_at_start:
        raise ContractError(
            "tracked runtime source is dirty. Commit the converged code before "
            "benchmarking so every timing is attached to one immutable commit."
        )
    output_root = (repo / OUTPUT_ROOT_RELATIVE).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_cublas_environment(repo)
    derivation_np = importlib.import_module("numpy")
    derived_fixture_paths = _materialize_derived_workload(
        derivation_np,
        source_paths=source_fixture_paths,
        source_snapshot=source_fixture_before,
        output_root=output_root,
    )
    all_fixture_paths = {
        "source_features": source_fixture_paths["features"],
        "source_labels": source_fixture_paths["labels"],
        "derived_features": derived_fixture_paths["features"],
        "derived_labels": derived_fixture_paths["labels"],
        "derived_manifest": derived_fixture_paths["manifest"],
    }
    fixture_before = _snapshot_files(all_fixture_paths)
    hostname = _safe_hostname()
    descriptor = _descriptor(
        git_sha=git_sha,
        fixture_before=fixture_before,
        source_hashes=source_before,
    )
    descriptor_sha256 = _canonical_sha256(descriptor)
    restart_hyperparameter_policy = importlib.import_module(
        "core.hyperparameter_policy"
    )
    run_id = f"{git_sha}-{hostname}-r11c-{descriptor_sha256[:12]}"
    run_dir = _require_within(
        output_root / run_id,
        output_root,
        "benchmark run directory",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = _require_within(
        run_dir / "weights",
        run_dir,
        "benchmark weights directory",
    )

    runtime: dict[str, Any] | None = None
    fixture: dict[str, Any] | None = None
    study = None
    cleanup_report = None
    fixture_after = None
    source_after = None
    with _ExclusivePidLock(
        run_dir,
        recover_stale=args.recover_stale,
        command=command,
    ):
        descriptor_payload = {
            "schema": BENCHMARK_SCHEMA,
            "classification": DISCLAIMER,
            "descriptor_sha256": descriptor_sha256,
            "descriptor": descriptor,
        }
        _write_or_verify_descriptor(run_dir, descriptor_payload)
        completed_summary, completion_pointer_repaired = (
            _completed_summary_if_valid(
                run_dir,
                hyperparameter_policy=restart_hyperparameter_policy,
                descriptor_sha256=descriptor_sha256,
                protocol_core_sha256=_canonical_sha256(descriptor["core"]),
                git_sha=git_sha,
                run_id=run_id,
                fixture_snapshot=fixture_before,
                source_hashes=source_before,
                recover_torn_state=args.recover_stale,
            )
        )
        if completed_summary is not None:
            print("!" * 78)
            if completion_pointer_repaired:
                print(
                    "Authenticated summary found after an interrupted final "
                    "publication; run_state.json was repaired explicitly."
                )
            print(
                f"{DISCLAIMER}: existing completed receipt is valid and "
                "was preserved byte-for-byte."
            )
            print(f"Compute/provenance report: {run_dir / 'summary.json'}")
            print("No benchmark study, timing, telemetry, or report was rewritten.")
            print("!" * 78)
            return 0
        runtime_dirs = _configure_runtime_outputs(run_dir)
        _bootstrap_cublas_environment(repo)
        weights_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "run_state.json", {
            "schema": BENCHMARK_SCHEMA,
            "classification": DISCLAIMER,
            "status": "starting",
            "descriptor_sha256": descriptor_sha256,
            "started_utc": started_utc,
            "command": {
                "argv": command_argv,
                "rendered": command,
                "cwd": str(repo),
            },
        })
        try:
            runtime = _load_runtime(descriptor, descriptor_sha256)
            runtime["utils"].set_global_seed(
                SEED,
                runtime["trainer"].TRAIN_PROTOCOL["determinism"],
            )
            environment_record = _runtime_environment(runtime)
            fixture = _load_and_validate_fixture(
                runtime["np"],
                derived_fixture_paths,
            )
            splits = _make_workload_splits(
                runtime["np"],
                fixture["groups"],
            )
            protocol_core_hash = runtime["protocol"].protocol_hash(
                descriptor["core"]
            )
            if protocol_core_hash != _canonical_sha256(descriptor["core"]):
                raise ContractError(
                    "benchmark protocol-core hash differs between canonical "
                    "hash implementations"
                )
            execution_attestation = (
                runtime["execution_environment"].enforce_execution_block(
                    stage=ANCHOR_STAGE,
                    policy=runtime[
                        "execution_environment"
                    ].EXECUTION_BLOCK_POLICY,
                    protocol_core_hash=protocol_core_hash,
                    run_tag=f"benchmark-{descriptor_sha256}",
                    receipt_dir=(
                        run_dir / "execution_receipts"
                    ).resolve(),
                )
            )
            capacity_receipt = (
                runtime["capacity_preflight"].ensure_capacity_preflight(
                    execution_attestation["runtime"],
                    receipt_dir=(
                        run_dir / "capacity_receipts"
                    ).resolve(),
                )
            )
            config = _build_config(
                descriptor,
                descriptor_sha256,
                execution_attestation["runtime"],
                campaign_run_tag=f"benchmark-{descriptor_sha256}",
                execution_receipt_sha256=
                    execution_attestation["receipt_sha256"],
            )
            hyperparameter_plan = (
                runtime["hyperparameter_policy"].derive_execution_plan(
                    config,
                    dataset_name=STUDY_DATASET_NAME,
                    requested_n_trials=USEFUL_TRIALS,
                    requested_use_pruner=USE_PRUNER,
                    execution_runtime=execution_attestation["runtime"],
                )
            )
            study, recovered_trials = _prepare_study(
                runtime,
                config,
                hyperparameter_plan,
                capacity_receipt,
                run_dir,
                descriptor_sha256,
                recover_stale=args.recover_stale,
            )
            hpo = _run_registered_anchor_hpo(
                runtime,
                study,
                config,
                hyperparameter_plan,
                fixture,
                splits,
                weights_dir,
                run_dir,
                descriptor_sha256,
                recovered_trials,
                recover_stale=args.recover_stale,
            )
            fixture_report = dict(fixture["report"])
            finalist = _run_finalist_once(
                runtime,
                study,
                config,
                fixture,
                splits,
                run_dir,
                descriptor_sha256,
                recover_stale=args.recover_stale,
            )
            cleanup_report = _clean_exact_trial_weights(
                weights_dir,
                config["name"],
                study.trials,
            )
            _close_fixture(fixture)
            fixture = None
            fixture_after = _snapshot_files(all_fixture_paths)
            source_after = _source_hashes(repo)
            if not _snapshot_equal(fixture_before, fixture_after):
                raise ContractError(
                    "fixture path, size or SHA-256 changed during benchmark"
                )
            if source_before != source_after:
                raise ContractError(
                    "runtime source hashes changed during benchmark"
                )
            if _git_sha(repo) != git_sha:
                raise ContractError("git HEAD changed during benchmark")

            study_receipt_path = _materialize_study_storage_receipt(
                run_dir)
            storage_snapshot = _storage_snapshot(run_dir)
            coordination_snapshot = _coordination_receipt_snapshot(run_dir)
            immutable_snapshot = _immutable_evidence_snapshot(
                run_dir,
                study_storage_snapshot=storage_snapshot,
            )
            hyperparameter_execution = _hyperparameter_execution_record(
                runtime["hyperparameter_policy"],
                hyperparameter_plan,
            )
            _validate_benchmark_hyperparameter_execution(
                hyperparameter_execution,
                hyperparameter_policy=runtime["hyperparameter_policy"],
                descriptor_sha256=descriptor_sha256,
                protocol_core_sha256=protocol_core_hash,
                coordination_receipts=coordination_snapshot,
            )
            if (
                execution_attestation.get("receipt_sha256")
                != coordination_snapshot["execution_block"]["sha256"]
            ):
                raise ContractError(
                    "execution attestation does not cite the exact published "
                    "receipt bytes"
                )
            completed_utc = _utc_now()
            total_wall = time.perf_counter() - started
            environment_record[
                "gpu_telemetry_this_invocation_end"
            ] = _nvidia_smi_snapshot()
            summary = {
                "schema": BENCHMARK_SCHEMA,
                "classification": DISCLAIMER,
                "classification_detail": DISCLAIMER_DETAIL,
                "status": "completed",
                "descriptor_sha256": descriptor_sha256,
                "identity": {
                    "git_sha": git_sha,
                    "git_tracked_dirty_at_start": git_tracked_dirty_at_start,
                    "hostname": hostname,
                    "run_id": run_id,
                    "run_directory": str(run_dir),
                },
                "command": {
                    "argv": command_argv,
                    "rendered": command,
                    "cwd": str(repo),
                    "this_invocation_started_utc": started_utc,
                    "this_invocation_measurement_completed_utc":
                        completed_utc,
                    "wall_seconds_before_summary_publication": total_wall,
                    "wall_clock_semantics": (
                        "this invocation from immediately after the warning "
                        "banner through immutable study-receipt creation; "
                        "excludes only final telemetry/summary/run-state writes"
                    ),
                },
                "environment": environment_record,
                "execution_attestation": execution_attestation,
                "capacity_preflight": {
                    "receipt_sha256": capacity_receipt["receipt_sha256"],
                    "envelope_file_sha256":
                        coordination_snapshot["cuda_capacity"]["sha256"],
                    "policy_sha256": capacity_receipt[
                        "receipt"
                    ]["policy_sha256"],
                    "passed": capacity_receipt["receipt"]["passed"],
                },
                "hyperparameter_execution": hyperparameter_execution,
                "runtime_output_directories": runtime_dirs,
                "hashes": {
                    "descriptor_sha256": descriptor_sha256,
                    "source_before": source_before,
                    "source_after": source_after,
                    "fixture_before": fixture_before,
                    "fixture_after": fixture_after,
                    "study_storage": storage_snapshot,
                    "coordination_receipts": coordination_snapshot,
                    "immutable_evidence": immutable_snapshot,
                },
                "fixture": fixture_report,
                "splits": splits["report"],
                "study_compute": hpo["report"],
                "stale_trials_recovered_this_invocation":
                    recovered_trials,
                "finalist_compute": finalist,
                "active_compute_seconds_cumulative": (
                    hpo["report"]["hpo_active_wall_seconds_cumulative"]
                    + finalist["active_wall_seconds_cumulative"]
                ),
                "cleanup": cleanup_report,
                "artifacts": {
                    "descriptor": "descriptor.json",
                    "study": "study.sqlite3",
                    "immutable_study_receipt": study_receipt_path.name,
                    "trial_compute": "trial_compute.csv",
                    "hpo_compute": "hpo_compute.json",
                    "active_wall_heartbeat": "active_wall_heartbeat.json",
                    "finalist_compute": "finalist_compute.json",
                    "finalist_attempt_state":
                        "finalist_attempt_state.json",
                    "finalist_active_wall_heartbeat":
                        "finalist_active_wall_heartbeat.json",
                    "summary": "summary.json",
                },
            }
            final_fixture_snapshot = _snapshot_files(all_fixture_paths)
            if not _snapshot_equal(fixture_before, final_fixture_snapshot):
                raise ContractError(
                    "fixture changed during final report validation"
                )
            summary["hashes"]["fixture_after"] = final_fixture_snapshot
            _atomic_json(run_dir / "summary.json", summary)
            published_summary = _regular_file_snapshot(
                run_dir / "summary.json",
                "published benchmark summary",
                max_bytes=JSON_SNAPSHOT_MAX_BYTES,
            )
            assert published_summary is not None
            summary_sha256 = str(published_summary["sha256"])
            _atomic_json(run_dir / "run_state.json", {
                "schema": BENCHMARK_SCHEMA,
                "classification": DISCLAIMER,
                "status": "completed",
                "descriptor_sha256": descriptor_sha256,
                "completed_utc": completed_utc,
                "summary": "summary.json",
                "summary_sha256": summary_sha256,
            })
        except BaseException as exc:
            if study is not None:
                try:
                    _write_trial_csv(
                        run_dir / "trial_compute.csv",
                        study,
                        descriptor_sha256,
                    )
                except Exception:
                    pass
                try:
                    cleanup_report = _clean_exact_trial_weights(
                        weights_dir,
                        "r11c",
                        study.trials,
                    )
                except Exception:
                    cleanup_report = {
                        "exact_paths_considered": 0,
                        "exact_files_removed": 0,
                        "unexpected_paths_touched": False,
                    }
            _close_fixture(fixture)
            try:
                fixture_after = _snapshot_files(all_fixture_paths)
            except Exception:
                fixture_after = {}
            try:
                source_after = _source_hashes(repo)
            except Exception:
                source_after = {}
            _atomic_json(run_dir / "run_state.json", {
                "schema": BENCHMARK_SCHEMA,
                "classification": DISCLAIMER,
                "status": "failed",
                "descriptor_sha256": descriptor_sha256,
                "failed_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "fixture_before": fixture_before,
                "fixture_after": fixture_after,
                "fixture_unchanged": _snapshot_equal(
                    fixture_before, fixture_after
                ),
                "source_before": source_before,
                "source_after": source_after,
                "source_unchanged": source_before == source_after,
                "cleanup": cleanup_report,
            })
            raise

    print("!" * 78)
    print(f"{DISCLAIMER}: compute benchmark completed.")
    print(f"Compute/provenance report: {run_dir / 'summary.json'}")
    print("No model-quality values were reported.")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
