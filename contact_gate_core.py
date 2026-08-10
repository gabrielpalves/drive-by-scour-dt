"""Shared types, constants, and strict parsing primitives for the contact gate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable

from core.environment import (
    matlab_environment_descriptor,
    matlab_environment_sha256,
)
from contact_gate_path_safety import GateError

ROOT = Path(__file__).resolve().parent
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
STAGES = ("F40-S", "F40-M", "L99-S", "L99-M")
EXPECTED_PASSAGES = 3
EXPECTED_STATES = {"F40-S": 31, "F40-M": 31, "L99-S": 39, "L99-M": 39}
EXPECTED_CASES = {
    stage: count * EXPECTED_PASSAGES
    for stage, count in EXPECTED_STATES.items()
}
EXPECTED_TOTAL_CASES = sum(EXPECTED_CASES.values())
EXPECTED_FAMILIES = {
    "F40-S": {
        "target_healthy": 3,
        "scour_only": 4,
        "bearing_only": 8,
        "nuisance_only": 6,
        "joint": 10,
    },
    "F40-M": {
        "target_healthy": 3,
        "scour_only": 4,
        "bearing_only": 8,
        "nuisance_only": 6,
        "joint": 10,
    },
    "L99-S": {
        "target_healthy": 3,
        "scour_only": 12,
        "bearing_only": 8,
        "nuisance_only": 6,
        "joint": 10,
    },
    "L99-M": {
        "target_healthy": 3,
        "scour_only": 12,
        "bearing_only": 8,
        "nuisance_only": 6,
        "joint": 10,
    },
}
CHANNELS = (
    "carbody_vertical_acceleration",
    "front_bogie_vertical_acceleration",
    "rear_bogie_vertical_acceleration",
    "wheelset_1_constrained_vertical_acceleration_proxy",
    "wheelset_2_constrained_vertical_acceleration_proxy",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
)
CHANNEL_SCHEMA_ID = "physical8_v1"
EXPECTED_GEN_SCHEMA = "audit-2026-08-09-r12"
EXPECTED_GENERATION_BEHAVIOR_VERSION = "generation-rules-v8"
POLICY_SCHEMA = "contact-closure-gate-v2"
SUMMARY_SCHEMA = "contact-closure-gate-summary-v2"
PLAN_SCHEMA = "contact-closure-plan-v2"
AUTHORIZATION_RECEIPT_SCHEMA = "contact-closure-authorization-receipt-v2"
STUDY_SCHEMA = "contact-closure-v3"
DT_MS = (1.0, 0.5, 0.25)
GATES_N = (0.0, 12000.0, 24000.0)
FRACTION_GATE = 0.002
COMMON_DX_M = 0.01
# Registered production crop span beyond deck entry: the D01 crop keeps 1831
# post-deck samples = 1830 grid intervals at 0.01 m = 18.30 m.  The study's
# comparison window must end at exactly 10 + L_bridge + 18.30 m.
POST_DECK_WINDOW_M = 18.30
COMPARISON_WINDOW_ATOL = 1e-9
EXPECTED_L_BRIDGE_M = {
    "F40-S": 40.0,
    "F40-M": 40.0,
    "L99-S": 99.6,
    "L99-M": 99.6,
}
RECON_RTOL = 1e-10
RECON_ATOL = 1e-12
COARSE_NRMSE = 0.05
COARSE_NMAX = 0.10
COARSE_CORR = 0.995
MEDIUM_NRMSE = 0.02
MEDIUM_NMAX = 0.05
MEDIUM_CORR = 0.999
GCI_FS = 1.25
GCI_METHOD = "actual-step-generalized-richardson-v1"
EQUIVALENCE_RTOL = 1e-10
EQUIVALENCE_ATOL = 1e-12
GCI_P_MIN = 1e-8
GCI_P_MAX = 50.0
TIME_GRID_ULPS = 8
WAVEFORM_MONOTONIC_ATOL = 1e-12
FINEST_IDENTITY_ATOL = 1e-12
CLOSURE_INTERPRETATION = "bounded-numerical-tension-engineering-v1"
EXPECTED_MATLAB_RELEASE = "R2025b"
EXPECTED_MATLAB_VERSION = "25.2.0.3177638 (R2025b) Update 5"
EXPECTED_MATLAB_ENVIRONMENT_SHA256 = (
    "958e7fe28f70577e9cb77aba0443c127d0a99726042a4618f7cce88d557fce79"
)
ENVIRONMENT_LOCK = ROOT / "environment" / "campaign-py313-cu128.json"
NUMERIC_HASH_SELFCHECK = hashlib.sha256(
    struct.pack("<2d", 0.0, 1.0)
).hexdigest()
PLAIN_REPORT_FIELDS = {
    "study_schema", "created_utc", "matlab_release", "dataset_dir",
    "state_file", "state_file_sha256", "dataset_integrity", "stage",
    "case_name", "gen_schema", "generation_behavior_version",
    "channel_schema_id", "gen_fingerprint", "state_index",
    "passage_index", "passage_selector", "state_uid", "state_family",
    "profile_phase_stream_index", "profile_phase_seed", "dt_requested_ms",
    "gates_n", "fraction_gate", "common_dx_m", "reconstruction_rtol",
    "reconstruction_atol", "saved_contact_log", "saved_gate_pass",
    "harness_sha256", "b66_sha256", "solver_source_sha256",
    "solver_execution_root_sha256", "current_generator_source_root_sha256",
    "current_matlab_environment_sha256", "dry_run",
    "numeric_hash_selfcheck", "status", "reference_dt_ms",
    "comparison_window_m", "descriptor", "run_table", "channel_table",
    "channel_qoi_table", "saved_baseline_table", "saved_baseline_note",
    "saved_baseline_mode", "direct_reconstruction_pass",
    "rerun_contact_log_1ms", "saved_contact_reconstruction_pass",
    "signal_common_sha256", "contact_peak_delta_vs_finest_N",
    "tension_fraction_delta_vs_finest",
}


@dataclass(frozen=True)
class SelectionRow:
    ordinal: int
    stage: str
    state_index: int
    passage_index: int
    state_uid: str
    state_family: str
    state_file_sha256: str
    saved_bridge_flag: float
    saved_track_flag: float
    saved_fraction: float
    saved_signed_peak_n: float


@dataclass(frozen=True)
class DatasetDescriptor:
    stage: str
    dataset_dir_sha256: str
    content_root: str
    case_info: str
    damage_states: str
    file_digests: str
    complete: str
    host_receipt: str
    fingerprint: str
    qual_source: str
    qual_executed: str
    host_diagnostic: str
    raw_line: str

def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"required regular file is missing: {path}")
    return _sha256_bytes(path.read_bytes())


def _normalised_lf(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise GateError(f"{path} is not UTF-8 text") from exc


def _canonical_lf_file(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{label} is missing/non-regular/symlinked: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"{label} is not UTF-8") from exc
    if "\r" in text or not text.endswith("\n") or text.endswith("\n\n"):
        raise GateError(f"{label} must use canonical LF with one final LF")
    return text


def _strict_json_text(text: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise GateError(f"{label} contains forbidden nonfinite token {value}")

    def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise GateError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except json.JSONDecodeError as exc:
        raise GateError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} root must be an object")
    return value


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{label} is missing/non-regular/symlinked: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"{label} is not UTF-8") from exc
    if "\r" in text or not text.endswith("\n") or text.endswith("\n\n"):
        raise GateError(f"{label} must use canonical LF with one final LF")
    return _strict_json_text(text, label)


def _locked_matlab_environment() -> tuple[dict[str, str], str, str]:
    lock = _strict_json_file(ENVIRONMENT_LOCK, "campaign environment lock")
    if lock.get("schema") != "ttbi-campaign-environment-v2":
        raise GateError("campaign environment lock has the wrong schema")
    environment = lock.get("matlab_environment")
    if not isinstance(environment, dict):
        raise GateError("environment lock lacks matlab_environment")
    try:
        descriptor = matlab_environment_descriptor(environment)
        digest = matlab_environment_sha256(environment)
    except RuntimeError as exc:
        raise GateError(f"malformed locked MATLAB descriptor: {exc}") from exc
    if (
        digest != EXPECTED_MATLAB_ENVIRONMENT_SHA256
        or lock.get("matlab_environment_sha256") != digest
        or environment.get("release") != EXPECTED_MATLAB_RELEASE
        or environment.get("version") != EXPECTED_MATLAB_VERSION
    ):
        raise GateError("campaign lock is not exact R2025b Update 5")
    return environment, descriptor, digest


def _as_list(value: Any) -> list[Any]:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, list):
        return value
    return [value]


def _strict_number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise GateError(f"{label} is not strict JSON numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{label} contains NaN/Inf")
    return result


def _strict_integer(value: Any, label: str) -> int:
    result = _strict_number(value, label)
    if int(result) != result:
        raise GateError(f"{label} is not an exact integer")
    return int(result)


def _float_list(value: Any, *, count: int, label: str) -> list[float]:
    values = _as_list(value)
    if len(values) != count:
        raise GateError(f"{label} has {len(values)} values, expected {count}")
    return [
        _strict_number(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]


def _same_float_list(a: Iterable[float], b: Iterable[float]) -> bool:
    aa, bb = list(a), list(b)
    return len(aa) == len(bb) and all(x == y for x, y in zip(aa, bb))


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise GateError(
            f"{label} field set mismatch "
            f"(missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)})"
        )
    return value


def _allclose(a: float, b: float, *, rtol: float, atol: float) -> bool:
    return (
        math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    )


def _validate_utc_pair(started: Any, completed: Any, label: str) -> None:
    parsed: list[datetime] = []
    for field, value in (("started_utc", started), ("completed_utc", completed)):
        if not isinstance(value, str) or not value:
            raise GateError(f"{label} {field} must be canonical UTC text")
        try:
            item = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise GateError(f"{label} {field} is not ISO-8601") from exc
        if item.tzinfo is None or item.utcoffset().total_seconds() != 0:
            raise GateError(f"{label} {field} is not UTC")
        parsed.append(item)
    if parsed[1] < parsed[0]:
        raise GateError(f"{label} completion precedes start")


def _strict_json_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if type(left) in {int, float} or type(right) in {int, float}:
        return (
            type(left) in {int, float}
            and type(right) in {int, float}
            and math.isfinite(float(left))
            and math.isfinite(float(right))
            and float(left) == float(right)
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                _strict_json_equivalent(a, b)
                for a, b in zip(left, right)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(
                _strict_json_equivalent(left[key], right[key])
                for key in left
            )
        )
    return type(left) is type(right) and left == right


def _first_json_mismatch(
    left: Any, right: Any, path: str = "$",
) -> str | None:
    if _strict_json_equivalent(left, right):
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return (
                f"{path} field set differs "
                f"(left-only={sorted(set(left) - set(right))}, "
                f"right-only={sorted(set(right) - set(left))})"
            )
        for key in left:
            mismatch = _first_json_mismatch(
                left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatch = _first_json_mismatch(
                left_item, right_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
    return (
        f"{path}: {left!r} ({type(left).__name__}) != "
        f"{right!r} ({type(right).__name__})"
    )
