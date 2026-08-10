#!/usr/bin/env python3
"""Independent integrity/decision checker for rail-domain clearance packages.

This checker independently pins the 18-case matrix, endpoint inventory,
waveform thresholds, conjunctive comparison decisions, and final 6/15/unresolved
selection. It does not convert the study into physical validation or general
numerical V&V.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


PROTOCOL_SCHEMA = "rail-domain-clearance-protocol-v1"
MANIFEST_SCHEMA = "rail-domain-clearance-manifest-v1"
VERDICT_SCHEMA = "rail-domain-clearance-verdict-v1"
FULL_KIND = "rail_domain_clearance_full"
SMOKE_KIND = "rail_domain_clearance_smoke"
SMOKE_CASE = "F40_2span_V70_T3_C15"
PRODUCTION_DECISION_ID = "paper1-rail-domain-clearance-c06-v1"
REVIEWED_PRODUCTION_CLEARANCE_M = 6.0
CLEARANCES = (6.0, 15.0, 30.0)
GEOMETRIES = {
    "F40_2span": (40.0, 2, 3, 2),
    "L99p6_4span": (99.6, 4, 2, 2),
}
OPERATING = {
    "V70_T3": (70.0, 3.0),
    "V80_T18": (80.0, 18.0),
    "V90_T33": (90.0, 33.0),
}
COMPARISONS = {
    "C06_vs_C30": (6.0, 30.0, 0.05, 0.10, 0.995),
    "C15_vs_C30": (15.0, 30.0, 0.02, 0.05, 0.999),
}
CHANNELS = (
    "carbody_vertical_acc",
    "front_bogie_vertical_acc",
    "rear_bogie_vertical_acc",
    "rail_eulerian_vertical_acc_under_wheel_1",
    "rail_eulerian_vertical_acc_under_wheel_2",
    "rail_eulerian_vertical_acc_under_wheel_3",
    "rail_eulerian_vertical_acc_under_wheel_4",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
    "wheelset_total_vertical_acc_1",
    "wheelset_total_vertical_acc_2",
    "wheelset_total_vertical_acc_3",
    "wheelset_total_vertical_acc_4",
    "bridge_midspan_displacement",
    "bridge_midspan_acceleration",
    "contact_force_wheel_1",
    "contact_force_wheel_2",
    "contact_force_wheel_3",
    "contact_force_wheel_4",
)
REQUIRED_ARTIFACTS = {
    "input_descriptor.json",
    "protocol_snapshot.json",
    "source_manifest.txt",
    "case_table.csv",
    "wheel_end_clearances.csv",
    "waveform_metrics.csv",
    "scalar_metrics.csv",
    "comparison_summary.csv",
    "raw_clearance_runs.mat",
    "rail_domain_clearance_report.md",
    "verdict.json",
}
EXPECTED_FILES = REQUIRED_ARTIFACTS | {"manifest.json", "_RUN_COMPLETE"}
GEOMETRY_TOL = 1e-10
TIME_TOL = 1e-12
PROFILE_TOL = 1e-12


class VerificationError(RuntimeError):
    """Fail-closed package error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path.name} must contain one object")
    return value


def read_csv(path: Path, required: Iterable[str], *, allow_empty: bool = False) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = set(required) - fields
            if missing:
                raise VerificationError(f"{path.name} missing columns {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise VerificationError(f"cannot read {path.name}: {exc}") from exc
    if not rows and not allow_empty:
        raise VerificationError(f"{path.name} must not be empty")
    return rows


def number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise VerificationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise VerificationError(f"{label} must be finite")
    return result


def integer(value: Any, label: str) -> int:
    result = number(value, label)
    if result != int(result):
        raise VerificationError(f"{label} must be an integer")
    return int(result)


def flag(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise VerificationError(f"{label} must be boolean")


def close(actual: Any, expected: float, label: str, tol: float = 1e-12) -> float:
    result = number(actual, label)
    if abs(result - expected) > tol:
        raise VerificationError(f"{label}={result:.17g}, expected {expected:.17g}")
    return result


def expected_cases() -> list[tuple[str, str, str, float]]:
    return [
        (f"{geometry}_{operating}_C{int(clearance):02d}", geometry, operating, clearance)
        for geometry in GEOMETRIES
        for operating in OPERATING
        for clearance in CLEARANCES
    ]


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise VerificationError("protocol schema mismatch")
    if protocol.get("full_run_kind") != FULL_KIND or protocol.get("smoke_run_kind") != SMOKE_KIND:
        raise VerificationError("protocol run-kind contract mismatch")
    if flag(protocol.get("numerical_verification_claim_authorized"), "numerical claim flag"):
        raise VerificationError("protocol escalates numerical qualification")
    if flag(protocol.get("physical_validation_claim_authorized"), "physical claim flag"):
        raise VerificationError("protocol escalates physical validation")
    if protocol.get("production_decision_id") != PRODUCTION_DECISION_ID:
        raise VerificationError("production clearance decision identity mismatch")
    close(
        protocol.get("reviewed_production_clearance_m"),
        REVIEWED_PRODUCTION_CLEARANCE_M,
        "reviewed production clearance",
    )
    if tuple(number(v, "clearance") for v in protocol.get("clearance_m", [])) != CLEARANCES:
        raise VerificationError("clearance sequence is not source locked")
    if integer(protocol.get("expected_full_solve_count"), "full solve count") != 18:
        raise VerificationError("full solve count is not 18")
    if protocol.get("smoke_case_id") != SMOKE_CASE:
        raise VerificationError("smoke case mismatch")

    geometries = protocol.get("geometries")
    if not isinstance(geometries, list) or len(geometries) != len(GEOMETRIES):
        raise VerificationError("geometry inventory mismatch")
    for row in geometries:
        if not isinstance(row, dict) or row.get("id") not in GEOMETRIES:
            raise VerificationError("unregistered geometry")
        length, spans, bridge_count, rail_count = GEOMETRIES[row["id"]]
        close(row.get("bridge_length_m"), length, "bridge length")
        if integer(row.get("num_spans"), "span count") != spans:
            raise VerificationError("span count mismatch")
        if integer(row.get("expected_bridge_elements_per_sleeper_bay"), "bridge mesh") != bridge_count:
            raise VerificationError("bridge mesh mismatch")
        if integer(row.get("rail_elements_per_sleeper_bay"), "rail mesh") != rail_count:
            raise VerificationError("rail mesh mismatch")

    points = protocol.get("operating_points")
    if not isinstance(points, list) or {p.get("id") for p in points if isinstance(p, dict)} != set(OPERATING):
        raise VerificationError("operating-point inventory mismatch")
    for point in points:
        speed, temperature = OPERATING[point["id"]]
        close(point.get("speed_kmh"), speed, "speed")
        close(point.get("temperature_degC"), temperature, "temperature")

    profile = protocol.get("profile")
    if not isinstance(profile, dict) or profile.get("mode") != "fixed":
        raise VerificationError("profile mode mismatch")
    if integer(profile.get("fra_class"), "FRA class") != 4 or integer(profile.get("phase_seed"), "phase seed") != 20260728:
        raise VerificationError("fixed FRA contract mismatch")
    if profile.get("spectrum_contract") != "fra-v2-class4-cycles-per-m-v1":
        raise VerificationError("spectrum contract mismatch")
    if [number(v, "profile bound") for v in profile.get("common_x_bounds_m", [])] != [-30.0, 390.0]:
        raise VerificationError("common profile bounds mismatch")

    retained = protocol.get("retained_window")
    if not isinstance(retained, dict):
        raise VerificationError("missing retained-window contract")
    close(retained.get("dx_m"), 0.01, "retained dx")
    close(retained.get("crop_start_m"), 10.0, "retained start")
    close(retained.get("post_deck_m"), 18.3, "post-deck window")
    tolerances = protocol.get("invariance_tolerance")
    if not isinstance(tolerances, dict):
        raise VerificationError("missing invariance tolerances")
    close(tolerances.get("geometry_m"), GEOMETRY_TOL, "geometry tolerance", 1e-20)
    close(tolerances.get("time_s"), TIME_TOL, "time tolerance", 1e-22)
    close(tolerances.get("profile_elevation_m"), PROFILE_TOL, "profile tolerance", 1e-22)
    endpoint_policy = protocol.get("wheel_endpoint_policy")
    if not isinstance(endpoint_policy, dict):
        raise VerificationError("missing wheel endpoint policy")
    if not flag(endpoint_policy.get("requested_padding_exact_on_sleeper_lattice"), "exact padding policy") or not flag(endpoint_policy.get("every_wheel_at_least_requested"), "per-wheel minimum policy"):
        raise VerificationError("wheel endpoint policy was weakened")
    close(endpoint_policy.get("expected_nominal_train_start_surplus_m"), 0.5, "start surplus")
    close(endpoint_policy.get("expected_leading_exit_surplus_m"), 0.0, "exit surplus")
    if tuple(protocol.get("waveform_channels", ())) != CHANNELS:
        raise VerificationError("20-channel waveform inventory mismatch")
    comparisons = protocol.get("comparisons")
    if not isinstance(comparisons, list) or {c.get("id") for c in comparisons if isinstance(c, dict)} != set(COMPARISONS):
        raise VerificationError("comparison inventory mismatch")
    for comparison in comparisons:
        test, reference, nrmse, nmax, corr = COMPARISONS[comparison["id"]]
        close(comparison.get("test_clearance_m"), test, "test clearance")
        close(comparison.get("reference_clearance_m"), reference, "reference clearance")
        close(comparison.get("maximum_nrmse"), nrmse, "NRMSE threshold")
        close(comparison.get("maximum_normalized_max_error"), nmax, "NMAX threshold")
        close(comparison.get("minimum_correlation"), corr, "correlation threshold")
    if set(protocol.get("required_artifacts", ())) != REQUIRED_ARTIFACTS:
        raise VerificationError("protocol artifact inventory mismatch")


def validate_manifest(root: Path, manifest: dict[str, Any], run_kind: str) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("protocol_schema") != PROTOCOL_SCHEMA:
        raise VerificationError("manifest schema mismatch")
    if manifest.get("run_kind") != run_kind or manifest.get("status") != "COMPLETE":
        raise VerificationError("manifest run status mismatch")
    if flag(manifest.get("numerical_verification_claim_authorized"), "manifest numerical flag"):
        raise VerificationError("manifest escalates numerical qualification")
    if flag(manifest.get("physical_validation_claim_authorized"), "manifest physical flag"):
        raise VerificationError("manifest escalates physical validation")
    if manifest.get("production_decision_id") != PRODUCTION_DECISION_ID:
        raise VerificationError("manifest clearance decision identity mismatch")
    close(
        manifest.get("reviewed_production_clearance_m"),
        REVIEWED_PRODUCTION_CLEARANCE_M,
        "manifest reviewed production clearance",
    )
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list) or {r.get("path") for r in inventory if isinstance(r, dict)} != REQUIRED_ARTIFACTS:
        raise VerificationError("manifest artifact inventory mismatch")
    for row in inventory:
        name = row.get("path")
        path = root / str(name)
        if path.is_symlink() or not path.is_file():
            raise VerificationError(f"artifact is not regular/unlinked: {name}")
        if integer(row.get("bytes"), f"{name} bytes") != path.stat().st_size:
            raise VerificationError(f"artifact size mismatch: {name}")
        if row.get("sha256") != sha256_file(path):
            raise VerificationError(f"artifact digest mismatch: {name}")
    completion = (root / "_RUN_COMPLETE").read_text(encoding="utf-8")
    expected_marker = (
        "schema=rail-domain-clearance-completion-v1\n"
        f"run_kind={run_kind}\nstatus=COMPLETE\n"
        f"manifest_sha256={sha256_file(root / 'manifest.json')}\n"
    )
    if completion != expected_marker:
        raise VerificationError("completion marker does not bind the manifest")


def validate_source_manifest(root: Path, manifest: dict[str, Any], repo_root: Path | None) -> None:
    try:
        lines = (root / "source_manifest.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"cannot read source manifest: {exc}") from exc
    if not lines or lines != sorted(lines) or len(lines) != len(set(lines)):
        raise VerificationError("source manifest must be nonempty, sorted, and unique")
    for line in lines:
        if ":" not in line:
            raise VerificationError("malformed source manifest line")
        relative, digest = line.rsplit(":", 1)
        if not relative.startswith("scour_MATLAB/") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VerificationError("malformed source manifest identity")
        if repo_root is not None:
            source = repo_root.joinpath(*relative.split("/"))
            if source.is_symlink() or not source.is_file() or sha256_file(source) != digest:
                raise VerificationError(f"current source mismatch: {relative}")
    root_digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    if manifest.get("generator_source_root_sha256") != root_digest:
        raise VerificationError("generator source root mismatch")
    if integer(manifest.get("generator_source_file_count"), "source file count") != len(lines):
        raise VerificationError("generator source file count mismatch")


CASE_FIELDS = {
    "case_id", "geometry_id", "operating_point_id", "clearance_m",
    "speed_kmh", "temperature_degC", "bridge_length_m", "num_spans",
    "bridge_elements_per_sleeper_bay", "rail_elements_per_sleeper_bay",
    "rail_length_m", "rail_domain_translation_m", "lead_to_deck_m",
    "lead_travel_m", "solve_duration_s", "actual_dt_s", "solver_sample_count",
    "retained_start_m", "retained_end_m", "retained_dx_m", "retained_sample_count",
    "minimum_start_clearance_m", "minimum_end_clearance_m",
    "profile_retained_sha256", "contact_admissible", "contact_lost_track",
    "source_commit", "input_hash",
}


def validate_cases(root: Path, run_kind: str, manifest: dict[str, Any], input_data: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv(root / "case_table.csv", CASE_FIELDS)
    expected = expected_cases() if run_kind == FULL_KIND else [next(row for row in expected_cases() if row[0] == SMOKE_CASE)]
    if len(rows) != len(expected) or {row["case_id"] for row in rows} != {row[0] for row in expected}:
        raise VerificationError("case table inventory mismatch")
    if integer(manifest.get("case_count"), "manifest case count") != len(expected):
        raise VerificationError("manifest case count mismatch")
    if integer(input_data.get("case_count"), "input case count") != len(expected):
        raise VerificationError("input case count mismatch")
    input_cases = input_data.get("cases")
    if not isinstance(input_cases, list) or {r.get("case_id") for r in input_cases if isinstance(r, dict)} != {r[0] for r in expected}:
        raise VerificationError("input case inventory mismatch")
    input_hash = manifest.get("input_hash")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        _, geometry, operating, clearance = next(v for v in expected if v[0] == case_id)
        length, spans, bridge_count, rail_count = GEOMETRIES[geometry]
        speed, temperature = OPERATING[operating]
        if row["geometry_id"] != geometry or row["operating_point_id"] != operating:
            raise VerificationError(f"case descriptor mismatch: {case_id}")
        close(row["clearance_m"], clearance, f"{case_id} clearance")
        close(row["speed_kmh"], speed, f"{case_id} speed")
        close(row["temperature_degC"], temperature, f"{case_id} temperature")
        close(row["bridge_length_m"], length, f"{case_id} length")
        if integer(row["num_spans"], f"{case_id} spans") != spans:
            raise VerificationError(f"{case_id} spans mismatch")
        if integer(row["bridge_elements_per_sleeper_bay"], f"{case_id} bridge mesh") != bridge_count:
            raise VerificationError(f"{case_id} bridge mesh mismatch")
        if integer(row["rail_elements_per_sleeper_bay"], f"{case_id} rail mesh") != rail_count:
            raise VerificationError(f"{case_id} rail mesh mismatch")
        close(row["rail_domain_translation_m"], clearance - 6.0, f"{case_id} translation", GEOMETRY_TOL)
        close(row["lead_to_deck_m"], 10.2, f"{case_id} lead-to-deck", GEOMETRY_TOL)
        close(row["retained_start_m"], 10.0, f"{case_id} retained start")
        close(row["retained_end_m"], 10.0 + length + 18.3, f"{case_id} retained end", GEOMETRY_TOL)
        close(row["retained_dx_m"], 0.01, f"{case_id} retained dx", 1e-10)
        expected_samples = round((length + 18.3) / 0.01) + 1
        if integer(row["retained_sample_count"], f"{case_id} retained samples") != expected_samples:
            raise VerificationError(f"{case_id} retained sample count mismatch")
        if number(row["minimum_start_clearance_m"], f"{case_id} start clearance") < clearance - GEOMETRY_TOL:
            raise VerificationError(f"{case_id} start clearance below request")
        close(row["minimum_start_clearance_m"], clearance + 0.5, f"{case_id} controlling start clearance", GEOMETRY_TOL)
        if number(row["minimum_end_clearance_m"], f"{case_id} end clearance") < clearance - GEOMETRY_TOL:
            raise VerificationError(f"{case_id} end clearance below request")
        close(row["minimum_end_clearance_m"], clearance, f"{case_id} controlling end clearance", GEOMETRY_TOL)
        digest = row["profile_retained_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VerificationError(f"{case_id} profile digest malformed")
        if not flag(row["contact_admissible"], f"{case_id} contact admissibility"):
            raise VerificationError(f"{case_id} violates registered contact policy")
        if flag(row["contact_lost_track"], f"{case_id} contact lost"):
            raise VerificationError(f"{case_id} records tensile contact demand")
        if row["input_hash"] != input_hash:
            raise VerificationError(f"{case_id} input hash mismatch")
        by_id[case_id] = row

    if run_kind == FULL_KIND:
        for geometry in GEOMETRIES:
            for operating in OPERATING:
                group = [by_id[f"{geometry}_{operating}_C{int(c):02d}"] for c in CLEARANCES]
                lead_travel = [number(r["lead_travel_m"], "lead travel") for r in group]
                duration = [number(r["solve_duration_s"], "duration") for r in group]
                samples = [integer(r["solver_sample_count"], "solver samples") for r in group]
                if max(lead_travel) - min(lead_travel) > GEOMETRY_TOL:
                    raise VerificationError("lead travel changed across clearance arms")
                if max(duration) - min(duration) > TIME_TOL or len(set(samples)) != 1:
                    raise VerificationError("solve time/grid changed across clearance arms")
    return rows, by_id


ENDPOINT_FIELDS = {
    "case_id", "geometry_id", "operating_point_id", "vehicle_index", "wheel_index",
    "requested_clearance_m", "start_clearance_m", "end_clearance_m", "source_commit", "input_hash",
}


def validate_endpoints(root: Path, case_rows: list[dict[str, str]], cases: dict[str, dict[str, str]]) -> dict[str, bool]:
    rows = read_csv(root / "wheel_end_clearances.csv", ENDPOINT_FIELDS)
    if len(rows) != 20 * len(case_rows):
        raise VerificationError("per-wheel endpoint row count mismatch")
    grouped: dict[str, list[dict[str, str]]] = {case_id: [] for case_id in cases}
    for row in rows:
        if row["case_id"] not in grouped:
            raise VerificationError("endpoint row references a foreign case")
        grouped[row["case_id"]].append(row)
    result: dict[str, bool] = {}
    for case_id, group in grouped.items():
        keys = [(integer(r["vehicle_index"], "vehicle"), integer(r["wheel_index"], "wheel")) for r in group]
        expected_keys = [(vehicle, wheel) for vehicle in range(1, 6) for wheel in range(1, 5)]
        if sorted(keys) != expected_keys:
            raise VerificationError(f"{case_id} endpoint wheel inventory mismatch")
        clearance = number(cases[case_id]["clearance_m"], "case clearance")
        starts = []
        ends = []
        for row in group:
            close(row["requested_clearance_m"], clearance, "endpoint request")
            starts.append(number(row["start_clearance_m"], "start clearance"))
            ends.append(number(row["end_clearance_m"], "end clearance"))
            if row["input_hash"] != cases[case_id]["input_hash"]:
                raise VerificationError("endpoint input hash mismatch")
        if min(starts) < clearance - GEOMETRY_TOL or min(ends) < clearance - GEOMETRY_TOL:
            raise VerificationError(f"{case_id} endpoint minimum below request")
        close(min(starts), number(cases[case_id]["minimum_start_clearance_m"], "case start minimum"), "start minimum", GEOMETRY_TOL)
        close(min(ends), number(cases[case_id]["minimum_end_clearance_m"], "case end minimum"), "end minimum", GEOMETRY_TOL)
        result[case_id] = True
    return result


WAVE_FIELDS = {
    "geometry_id", "operating_point_id", "comparison_id", "test_clearance_m",
    "reference_clearance_m", "maximum_nrmse", "maximum_normalized_max_error",
    "minimum_correlation", "channel_pass", "channel_id", "nrmse",
    "normalized_max_error", "correlation", "source_commit", "input_hash",
}


def validate_metrics(root: Path, run_kind: str, manifest: dict[str, Any]) -> tuple[dict[tuple[str, str, str], bool], list[dict[str, str]]]:
    rows = read_csv(root / "waveform_metrics.csv", WAVE_FIELDS, allow_empty=run_kind == SMOKE_KIND)
    scalar = read_csv(root / "scalar_metrics.csv", {"geometry_id", "operating_point_id", "comparison_id", "qoi_id", "test_value", "reference_value", "signed_difference", "relative_difference", "input_hash"}, allow_empty=run_kind == SMOKE_KIND)
    if run_kind == SMOKE_KIND:
        if rows or scalar:
            raise VerificationError("smoke package must not contain comparison metrics")
        return {}, []
    expected_keys = {(g, o, c, channel) for g in GEOMETRIES for o in OPERATING for c in COMPARISONS for channel in CHANNELS}
    actual_keys: list[tuple[str, str, str, str]] = []
    grouped: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        key3 = (row["geometry_id"], row["operating_point_id"], row["comparison_id"])
        key4 = key3 + (row["channel_id"],)
        actual_keys.append(key4)
        if key4 not in expected_keys:
            raise VerificationError("unregistered waveform metric identity")
        test, reference, nrmse_limit, nmax_limit, corr_limit = COMPARISONS[row["comparison_id"]]
        close(row["test_clearance_m"], test, "metric test clearance")
        close(row["reference_clearance_m"], reference, "metric reference clearance")
        close(row["maximum_nrmse"], nrmse_limit, "metric NRMSE limit")
        close(row["maximum_normalized_max_error"], nmax_limit, "metric NMAX limit")
        close(row["minimum_correlation"], corr_limit, "metric correlation limit")
        nrmse = number(row["nrmse"], "metric NRMSE")
        nmax = number(row["normalized_max_error"], "metric NMAX")
        corr = number(row["correlation"], "metric correlation")
        expected_pass = nrmse <= nrmse_limit and nmax <= nmax_limit and corr >= corr_limit
        if flag(row["channel_pass"], "channel pass") != expected_pass:
            raise VerificationError("channel pass flag does not follow frozen thresholds")
        if row["input_hash"] != manifest.get("input_hash"):
            raise VerificationError("waveform metric input hash mismatch")
        grouped.setdefault(key3, []).append(expected_pass)
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise VerificationError("waveform metric inventory missing, extra, or duplicated")
    for row in scalar:
        if (row["geometry_id"], row["operating_point_id"], row["comparison_id"]) not in grouped:
            raise VerificationError("scalar metric references foreign comparison")
        test = number(row["test_value"], "scalar test")
        reference = number(row["reference_value"], "scalar reference")
        signed = number(row["signed_difference"], "scalar signed difference")
        relative = number(row["relative_difference"], "scalar relative difference")
        close(signed, test - reference, "scalar signed difference", 1e-9 * max(abs(test), abs(reference), 1.0))
        expected_relative = abs(test - reference) / max(abs(reference), 1e-15)
        close(relative, expected_relative, "scalar relative difference", 1e-9 * max(expected_relative, 1.0))
        if row["input_hash"] != manifest.get("input_hash"):
            raise VerificationError("scalar metric input hash mismatch")
    if not scalar:
        raise VerificationError("full package has no scalar QoI comparisons")
    return {key: all(values) for key, values in grouped.items()}, rows


SUMMARY_FIELDS = {
    "geometry_id", "operating_point_id", "comparison_id", "maximum_nrmse",
    "maximum_normalized_max_error", "minimum_correlation", "worst_nrmse",
    "worst_normalized_max_error", "worst_correlation", "profile_max_abs_delta_m",
    "waveform_pass", "geometry_invariant", "profile_invariant", "clearance_enforced",
    "contact_admissible", "comparison_pass", "input_hash",
}


def validate_summary(root: Path, run_kind: str, manifest: dict[str, Any], cases: dict[str, dict[str, str]], endpoint_pass: dict[str, bool], metric_pass: dict[tuple[str, str, str], bool], wave_rows: list[dict[str, str]]) -> dict[str, bool]:
    rows = read_csv(root / "comparison_summary.csv", SUMMARY_FIELDS, allow_empty=run_kind == SMOKE_KIND)
    if run_kind == SMOKE_KIND:
        if rows:
            raise VerificationError("smoke package must not contain comparison summary")
        return {}
    expected = {(g, o, c) for g in GEOMETRIES for o in OPERATING for c in COMPARISONS}
    if len(rows) != len(expected) or {(r["geometry_id"], r["operating_point_id"], r["comparison_id"]) for r in rows} != expected:
        raise VerificationError("comparison summary inventory mismatch")
    wave_group: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in wave_rows:
        wave_group.setdefault((row["geometry_id"], row["operating_point_id"], row["comparison_id"]), []).append(row)
    result: dict[str, list[bool]] = {comparison: [] for comparison in COMPARISONS}
    for row in rows:
        key = (row["geometry_id"], row["operating_point_id"], row["comparison_id"])
        test, reference, nrmse_limit, nmax_limit, corr_limit = COMPARISONS[row["comparison_id"]]
        metrics = wave_group[key]
        close(row["maximum_nrmse"], nrmse_limit, "summary NRMSE limit")
        close(row["maximum_normalized_max_error"], nmax_limit, "summary NMAX limit")
        close(row["minimum_correlation"], corr_limit, "summary correlation limit")
        close(row["worst_nrmse"], max(number(v["nrmse"], "NRMSE") for v in metrics), "summary worst NRMSE", 1e-12)
        close(row["worst_normalized_max_error"], max(number(v["normalized_max_error"], "NMAX") for v in metrics), "summary worst NMAX", 1e-12)
        close(row["worst_correlation"], min(number(v["correlation"], "corr") for v in metrics), "summary worst corr", 1e-12)
        waveform_ok = metric_pass[key]
        if flag(row["waveform_pass"], "summary waveform pass") != waveform_ok:
            raise VerificationError("summary waveform pass mismatch")
        test_case = cases[f"{key[0]}_{key[1]}_C{int(test):02d}"]
        reference_case = cases[f"{key[0]}_{key[1]}_C{int(reference):02d}"]
        geometry_ok = (
            abs(number(test_case["lead_to_deck_m"], "lead deck") - number(reference_case["lead_to_deck_m"], "lead deck")) <= GEOMETRY_TOL
            and abs(number(test_case["lead_travel_m"], "lead travel") - number(reference_case["lead_travel_m"], "lead travel")) <= GEOMETRY_TOL
            and abs(number(test_case["solve_duration_s"], "duration") - number(reference_case["solve_duration_s"], "duration")) <= TIME_TOL
            and integer(test_case["solver_sample_count"], "samples") == integer(reference_case["solver_sample_count"], "samples")
        )
        if flag(row["geometry_invariant"], "summary geometry invariant") != geometry_ok:
            raise VerificationError("summary geometry invariant mismatch")
        profile_ok = number(row["profile_max_abs_delta_m"], "profile delta") <= PROFILE_TOL
        if flag(row["profile_invariant"], "summary profile invariant") != profile_ok:
            raise VerificationError("summary profile invariant mismatch")
        clearance_ok = endpoint_pass[test_case["case_id"]] and endpoint_pass[reference_case["case_id"]]
        if flag(row["clearance_enforced"], "summary clearance flag") != clearance_ok:
            raise VerificationError("summary clearance flag mismatch")
        contact_ok = flag(test_case["contact_admissible"], "test contact") and flag(reference_case["contact_admissible"], "reference contact")
        if flag(row["contact_admissible"], "summary contact flag") != contact_ok:
            raise VerificationError("summary contact flag mismatch")
        expected_pass = waveform_ok and geometry_ok and profile_ok and clearance_ok and contact_ok
        if flag(row["comparison_pass"], "comparison pass") != expected_pass:
            raise VerificationError("comparison pass is not conjunctive")
        if row["input_hash"] != manifest.get("input_hash"):
            raise VerificationError("comparison summary input hash mismatch")
        result[row["comparison_id"]].append(expected_pass)
    return {comparison: all(values) for comparison, values in result.items()}


def validate_verdict(root: Path, run_kind: str, manifest: dict[str, Any], comparison_pass: dict[str, bool]) -> None:
    verdict = load_json(root / "verdict.json")
    if verdict.get("schema") != VERDICT_SCHEMA or verdict.get("run_kind") != run_kind:
        raise VerificationError("verdict schema/run-kind mismatch")
    if flag(verdict.get("numerical_verification_claim_authorized"), "verdict numerical flag") or flag(verdict.get("physical_validation_claim_authorized"), "verdict physical flag"):
        raise VerificationError("verdict escalates validation scope")
    if verdict.get("production_decision_id") != PRODUCTION_DECISION_ID:
        raise VerificationError("verdict clearance decision identity mismatch")
    close(
        verdict.get("reviewed_production_clearance_m"),
        REVIEWED_PRODUCTION_CLEARANCE_M,
        "verdict reviewed production clearance",
    )
    if verdict.get("input_hash") != manifest.get("input_hash") or verdict.get("generator_source_root_sha256") != manifest.get("generator_source_root_sha256"):
        raise VerificationError("verdict provenance mismatch")
    close(verdict.get("diagnostic_clearance_m"), 30.0, "diagnostic clearance")
    if run_kind == SMOKE_KIND:
        if verdict.get("overall_status") != "UNVERIFIED" or verdict.get("production_selection_status") != "NOT_EVALUATED" or verdict.get("production_clearance_m") not in (None, []):
            raise VerificationError("smoke verdict attempts a production decision")
        if flag(verdict.get("reviewed_production_decision_confirmed"), "smoke decision confirmation"):
            raise VerificationError("smoke verdict confirms the production decision")
        return
    coarse = comparison_pass["C06_vs_C30"]
    primary = comparison_pass["C15_vs_C30"]
    if flag(verdict.get("coarse_comparison_pass"), "verdict coarse flag") != coarse or flag(verdict.get("primary_comparison_pass"), "verdict primary flag") != primary:
        raise VerificationError("verdict comparison flags mismatch")
    if coarse and primary:
        expected = ("PASS", "SELECTED", 6.0)
    elif primary:
        expected = ("PASS", "SELECTED", 15.0)
    else:
        expected = ("UNRESOLVED", "UNRESOLVED", None)
    if verdict.get("overall_status") != expected[0] or verdict.get("production_selection_status") != expected[1]:
        raise VerificationError("verdict selection status violates frozen rule")
    actual_clearance = verdict.get("production_clearance_m")
    if expected[2] is None:
        if actual_clearance not in (None, []):
            raise VerificationError("unresolved verdict authorizes a clearance")
    else:
        close(actual_clearance, expected[2], "selected production clearance")
    expected_confirmation = expected[2] == REVIEWED_PRODUCTION_CLEARANCE_M
    if flag(
        verdict.get("reviewed_production_decision_confirmed"),
        "reviewed production decision confirmation",
    ) != expected_confirmation:
        raise VerificationError("verdict decision confirmation is inconsistent")


def verify_package(root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise VerificationError("package root must be a regular directory")
    root = root.resolve()
    entries = {path.name for path in root.iterdir()}
    if entries != EXPECTED_FILES or any(path.is_dir() or path.is_symlink() for path in root.iterdir()):
        raise VerificationError("package file inventory is missing, extra, linked, or nested")
    manifest = load_json(root / "manifest.json")
    run_kind = manifest.get("run_kind")
    if run_kind not in {FULL_KIND, SMOKE_KIND}:
        raise VerificationError("unregistered run kind")
    validate_manifest(root, manifest, run_kind)
    validate_source_manifest(root, manifest, repo_root.resolve() if repo_root else None)
    protocol = load_json(root / "protocol_snapshot.json")
    validate_protocol(protocol)
    input_data = load_json(root / "input_descriptor.json")
    if input_data.get("schema") != "rail-domain-clearance-input-v1" or input_data.get("run_kind") != run_kind:
        raise VerificationError("input descriptor mismatch")
    if input_data.get("source_root_sha256") != manifest.get("generator_source_root_sha256"):
        raise VerificationError("input/source root mismatch")
    if input_data.get("production_decision_id") != PRODUCTION_DECISION_ID:
        raise VerificationError("input clearance decision identity mismatch")
    close(
        input_data.get("reviewed_production_clearance_m"),
        REVIEWED_PRODUCTION_CLEARANCE_M,
        "input reviewed production clearance",
    )
    if flag(input_data.get("physical_validation_requested"), "physical request") or flag(input_data.get("general_numerical_qualification_requested"), "qualification request"):
        raise VerificationError("input requests an unauthorized claim")
    case_rows, cases = validate_cases(root, run_kind, manifest, input_data)
    endpoints = validate_endpoints(root, case_rows, cases)
    metric_pass, wave_rows = validate_metrics(root, run_kind, manifest)
    comparison_pass = validate_summary(root, run_kind, manifest, cases, endpoints, metric_pass, wave_rows)
    validate_verdict(root, run_kind, manifest, comparison_pass)
    return {
        "status": "PASS",
        "run_kind": run_kind,
        "case_count": len(case_rows),
        "production_selection": load_json(root / "verdict.json").get("production_clearance_m"),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fixture_protocol() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "full_run_kind": FULL_KIND,
        "smoke_run_kind": SMOKE_KIND,
        "claim_scope": "finite-rail-domain-model-form-selection-only",
        "numerical_verification_claim_authorized": False,
        "physical_validation_claim_authorized": False,
        "production_decision_id": PRODUCTION_DECISION_ID,
        "reviewed_production_clearance_m": REVIEWED_PRODUCTION_CLEARANCE_M,
        "clearance_m": list(CLEARANCES),
        "geometries": [
            {"id": key, "bridge_length_m": value[0], "num_spans": value[1], "expected_bridge_elements_per_sleeper_bay": value[2], "rail_elements_per_sleeper_bay": value[3]}
            for key, value in GEOMETRIES.items()
        ],
        "operating_points": [{"id": key, "speed_kmh": value[0], "temperature_degC": value[1]} for key, value in OPERATING.items()],
        "expected_full_solve_count": 18,
        "smoke_case_id": SMOKE_CASE,
        "profile": {"mode": "fixed", "fra_class": 4, "phase_seed": 20260728, "spectrum_contract": "fra-v2-class4-cycles-per-m-v1", "common_x_bounds_m": [-30, 390]},
        "retained_window": {"dx_m": 0.01, "crop_start_m": 10.0, "post_deck_m": 18.3},
        "invariance_tolerance": {"geometry_m": GEOMETRY_TOL, "time_s": TIME_TOL, "profile_elevation_m": PROFILE_TOL},
        "wheel_endpoint_policy": {"requested_padding_exact_on_sleeper_lattice": True, "every_wheel_at_least_requested": True, "expected_nominal_train_start_surplus_m": 0.5, "expected_leading_exit_surplus_m": 0.0},
        "waveform_channels": list(CHANNELS),
        "comparisons": [
            {"id": key, "test_clearance_m": value[0], "reference_clearance_m": value[1], "maximum_nrmse": value[2], "maximum_normalized_max_error": value[3], "minimum_correlation": value[4]}
            for key, value in COMPARISONS.items()
        ],
        "required_artifacts": sorted(REQUIRED_ARTIFACTS),
    }


def build_fixture(root: Path) -> None:
    root.mkdir()
    source_bytes = b"fixture source\n"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source_line = f"scour_MATLAB/fixture.m:{source_digest}"
    source_root = hashlib.sha256(source_line.encode()).hexdigest()
    (root / "source_manifest.txt").write_text(source_line + "\n", encoding="utf-8")
    protocol = fixture_protocol()
    (root / "protocol_snapshot.json").write_text(json.dumps(protocol), encoding="utf-8")
    cases_contract = [
        {"case_id": case_id, "geometry_id": geometry, "operating_point_id": operating, "clearance_m": clearance}
        for case_id, geometry, operating, clearance in expected_cases()
    ]
    input_data = {"schema": "rail-domain-clearance-input-v1", "run_kind": FULL_KIND, "case_count": 18, "cases": cases_contract, "production_decision_id": PRODUCTION_DECISION_ID, "reviewed_production_clearance_m": REVIEWED_PRODUCTION_CLEARANCE_M, "source_root_sha256": source_root, "physical_validation_requested": False, "general_numerical_qualification_requested": False}
    (root / "input_descriptor.json").write_text(json.dumps(input_data), encoding="utf-8")
    input_hash = "a" * 64
    case_fields = sorted(CASE_FIELDS)
    case_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for case_id, geometry, operating, clearance in expected_cases():
        length, spans, bridge_count, rail_count = GEOMETRIES[geometry]
        speed, temperature = OPERATING[operating]
        row = {field: 0 for field in case_fields}
        row.update({"case_id": case_id, "geometry_id": geometry, "operating_point_id": operating, "clearance_m": clearance, "speed_kmh": speed, "temperature_degC": temperature, "bridge_length_m": length, "num_spans": spans, "bridge_elements_per_sleeper_bay": bridge_count, "rail_elements_per_sleeper_bay": rail_count, "rail_length_m": 300 + 2 * clearance, "rail_domain_translation_m": clearance - 6, "lead_to_deck_m": 10.2, "lead_travel_m": 200.0, "solve_duration_s": 9.0, "actual_dt_s": 0.001, "solver_sample_count": 9001, "retained_start_m": 10.0, "retained_end_m": 10 + length + 18.3, "retained_dx_m": 0.01, "retained_sample_count": round((length + 18.3) / 0.01) + 1, "minimum_start_clearance_m": clearance + 0.5, "minimum_end_clearance_m": clearance, "profile_retained_sha256": "b" * 64, "contact_admissible": 1, "contact_lost_track": 0, "source_commit": "fixture", "input_hash": input_hash})
        case_rows.append(row)
        for vehicle in range(1, 6):
            for wheel in range(1, 5):
                endpoint_rows.append({"case_id": case_id, "geometry_id": geometry, "operating_point_id": operating, "vehicle_index": vehicle, "wheel_index": wheel, "requested_clearance_m": clearance, "start_clearance_m": clearance + 0.5 + (5 - vehicle) * 22 + (4 - wheel), "end_clearance_m": clearance + (vehicle - 1) * 22 + (wheel - 1), "source_commit": "fixture", "input_hash": input_hash})
    write_csv(root / "case_table.csv", case_fields, case_rows)
    write_csv(root / "wheel_end_clearances.csv", sorted(ENDPOINT_FIELDS), endpoint_rows)

    wave_fields = sorted(WAVE_FIELDS | {"reference_rms", "test_rms", "reference_abs_peak", "test_abs_peak", "peak_amplitude_error", "peak_position_error_m"})
    wave_rows: list[dict[str, Any]] = []
    summary_fields = sorted(SUMMARY_FIELDS | {"test_clearance_m", "reference_clearance_m", "source_commit"})
    summary_rows: list[dict[str, Any]] = []
    scalar_fields = ["geometry_id", "operating_point_id", "comparison_id", "qoi_id", "unit", "test_value", "reference_value", "signed_difference", "relative_difference", "source_commit", "input_hash"]
    scalar_rows: list[dict[str, Any]] = []
    for geometry in GEOMETRIES:
        for operating in OPERATING:
            for comparison, values in COMPARISONS.items():
                test, reference, nrmse_limit, nmax_limit, corr_limit = values
                for channel in CHANNELS:
                    row = {field: 0 for field in wave_fields}
                    row.update({"geometry_id": geometry, "operating_point_id": operating, "comparison_id": comparison, "test_clearance_m": test, "reference_clearance_m": reference, "maximum_nrmse": nrmse_limit, "maximum_normalized_max_error": nmax_limit, "minimum_correlation": corr_limit, "channel_pass": 1, "channel_id": channel, "nrmse": 0, "normalized_max_error": 0, "correlation": 1, "source_commit": "fixture", "input_hash": input_hash})
                    wave_rows.append(row)
                scalar_rows.append({"geometry_id": geometry, "operating_point_id": operating, "comparison_id": comparison, "qoi_id": "fixture_qoi", "unit": "1", "test_value": 1, "reference_value": 1, "signed_difference": 0, "relative_difference": 0, "source_commit": "fixture", "input_hash": input_hash})
                row = {field: 0 for field in summary_fields}
                row.update({"geometry_id": geometry, "operating_point_id": operating, "comparison_id": comparison, "test_clearance_m": test, "reference_clearance_m": reference, "maximum_nrmse": nrmse_limit, "maximum_normalized_max_error": nmax_limit, "minimum_correlation": corr_limit, "worst_nrmse": 0, "worst_normalized_max_error": 0, "worst_correlation": 1, "profile_max_abs_delta_m": 0, "waveform_pass": 1, "geometry_invariant": 1, "profile_invariant": 1, "clearance_enforced": 1, "contact_admissible": 1, "comparison_pass": 1, "source_commit": "fixture", "input_hash": input_hash})
                summary_rows.append(row)
    write_csv(root / "waveform_metrics.csv", wave_fields, wave_rows)
    write_csv(root / "scalar_metrics.csv", scalar_fields, scalar_rows)
    write_csv(root / "comparison_summary.csv", summary_fields, summary_rows)
    (root / "raw_clearance_runs.mat").write_bytes(b"fixture mat\n")
    (root / "rail_domain_clearance_report.md").write_text("# Fixture\n", encoding="utf-8")
    verdict = {"schema": VERDICT_SCHEMA, "run_kind": FULL_KIND, "claim_scope": "finite-rail-domain-model-form-selection-only", "numerical_verification_claim_authorized": False, "physical_validation_claim_authorized": False, "production_decision_id": PRODUCTION_DECISION_ID, "reviewed_production_clearance_m": REVIEWED_PRODUCTION_CLEARANCE_M, "reviewed_production_decision_confirmed": True, "input_hash": input_hash, "generator_source_root_sha256": source_root, "completed_case_count": 18, "diagnostic_clearance_m": 30, "overall_status": "PASS", "production_selection_status": "SELECTED", "production_clearance_m": 6, "coarse_comparison_pass": True, "primary_comparison_pass": True}
    (root / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    reseal_fixture(root, source_root, input_hash)


def reseal_fixture(root: Path, source_root: str | None = None, input_hash: str | None = None) -> None:
    old = load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    if source_root is None:
        source_root = old["generator_source_root_sha256"]
    if input_hash is None:
        input_hash = old["input_hash"]
    artifacts = [{"path": name, "sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size} for name in sorted(REQUIRED_ARTIFACTS)]
    manifest = {"schema": MANIFEST_SCHEMA, "protocol_schema": PROTOCOL_SCHEMA, "run_kind": FULL_KIND, "status": "COMPLETE", "claim_scope": "finite-rail-domain-model-form-selection-only", "numerical_verification_claim_authorized": False, "physical_validation_claim_authorized": False, "production_decision_id": PRODUCTION_DECISION_ID, "reviewed_production_clearance_m": REVIEWED_PRODUCTION_CLEARANCE_M, "generator_source_root_sha256": source_root, "generator_source_file_count": 1, "input_hash": input_hash, "case_count": 18, "artifacts": artifacts}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "_RUN_COMPLETE").write_text(f"schema=rail-domain-clearance-completion-v1\nrun_kind={FULL_KIND}\nstatus=COMPLETE\nmanifest_sha256={sha256_file(root / 'manifest.json')}\n", encoding="utf-8")


def expect_rejected(root: Path, label: str) -> None:
    try:
        verify_package(root)
    except VerificationError:
        return
    raise AssertionError(f"mutation was accepted: {label}")


def run_self_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="rail_clearance_checker_") as temp:
        base = Path(temp) / "base"
        build_fixture(base)
        verify_package(base)

        mutation = Path(temp) / "verdict_mutation"
        shutil.copytree(base, mutation)
        verdict = load_json(mutation / "verdict.json")
        verdict["production_clearance_m"] = 15
        (mutation / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
        reseal_fixture(mutation)
        expect_rejected(mutation, "selection changed")

        mutation = Path(temp) / "metric_mutation"
        shutil.copytree(base, mutation)
        rows = read_csv(mutation / "waveform_metrics.csv", WAVE_FIELDS)
        rows[0]["nrmse"] = "0.9"
        write_csv(mutation / "waveform_metrics.csv", list(rows[0]), rows)
        reseal_fixture(mutation)
        expect_rejected(mutation, "threshold/pass inconsistency")

        mutation = Path(temp) / "endpoint_mutation"
        shutil.copytree(base, mutation)
        rows = read_csv(mutation / "wheel_end_clearances.csv", ENDPOINT_FIELDS)
        rows[0]["start_clearance_m"] = "0"
        write_csv(mutation / "wheel_end_clearances.csv", list(rows[0]), rows)
        reseal_fixture(mutation)
        expect_rejected(mutation, "clearance below request")

        mutation = Path(temp) / "inventory_mutation"
        shutil.copytree(base, mutation)
        rows = read_csv(mutation / "case_table.csv", CASE_FIELDS)
        write_csv(mutation / "case_table.csv", list(rows[0]), rows[:-1])
        reseal_fixture(mutation)
        expect_rejected(mutation, "missing case")

        mutation = Path(temp) / "hash_mutation"
        shutil.copytree(base, mutation)
        (mutation / "raw_clearance_runs.mat").write_bytes(b"changed")
        expect_rejected(mutation, "artifact hash")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        print("rail-domain clearance checker mutations: PASS")
    if args.package is not None:
        result = verify_package(args.package, args.repo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    if not args.self_test and args.package is None:
        parser.error("provide a package path or --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
