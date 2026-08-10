#!/usr/bin/env python3
"""Independent nonqualifying-integrity check for numerical V&V packages.

Scientific qualification is deliberately unavailable and fails closed until
an independent, source-bound verifier is implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


MANIFEST_SCHEMA = "numerical-vv-manifest-v1"
VERDICT_SCHEMA = "numerical-vv-verdict-v1"
MICRO_KIND = "nonqualifying_micro"
MICRO_FILES = {
    "case_table.csv",
    "descriptor_hashes.csv",
    "static_equilibrium_checks.csv",
    "modal_matching.csv",
    "mesh_scalar_qoi.csv",
    "time_grid_realization.csv",
    "support_alignment.csv",
    "tolerance_rationale.csv",
    "raw_bridge_fixture.mat",
    "protocol_snapshot.json",
    "input_descriptor.json",
    "vv_verdict.json",
}
SLEEPER_SPACING_M = 0.6
MAX_GCI_OBSERVED_ORDER = 10
MESH_LEVEL_ROLES = {
    "M0": "current-production",
    "M1": "primary-finer-comparison",
    "M2": "second-primary-finer-comparison",
    "M3": "conditional-resolution",
}
GEOMETRY_CONTRACT: dict[str, dict[str, Any]] = {
    "L60_3span": {
        "bridge_length_m": 60.0,
        "num_spans": 3,
        "levels": {
            "M0": (3, 2),
            "M1": (6, 4),
            "M2": (12, 8),
            "M3": (24, 16),
        },
    },
    "L99p6_4span": {
        "bridge_length_m": 99.6,
        "num_spans": 4,
        "levels": {
            "M0": (2, 2),
            "M1": (4, 4),
            "M2": (8, 8),
            "M3": (16, 16),
        },
    },
}
DESCRIPTOR_FIELDS = {
    "case_id",
    "geometry_id",
    "mesh_level",
    "bridge_length_m",
    "bridge_elements_per_sleeper_bay",
    "rail_elements_per_sleeper_bay",
    "bridge_nominal_h_m",
    "rail_nominal_h_m",
    "rail_mesh_executed",
    "point_load_N",
    "scope",
}


class VerificationError(RuntimeError):
    """A fail-closed package verification error."""


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
        raise VerificationError(f"cannot parse JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path.name} must contain a JSON object")
    return value


def require_regular_unlinked(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"expected a regular non-symlink file: {path}")


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not required.issubset(fields):
                missing = sorted(required - fields)
                raise VerificationError(f"{path.name} missing columns: {missing}")
            rows = list(reader)
    except OSError as exc:
        raise VerificationError(f"cannot read {path.name}: {exc}") from exc
    if not rows:
        raise VerificationError(f"{path.name} must not be empty")
    return rows


def finite_positive(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise VerificationError(f"{label} is not numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise VerificationError(f"{label} must be finite and positive")
    return value


def positive_integer(text: str, label: str) -> int:
    value = finite_positive(text, label)
    if value != int(value):
        raise VerificationError(f"{label} must be an integer")
    return int(value)


def binary_flag(text: str, label: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise VerificationError(f"{label} must be a binary flag")


def numeric_value(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise VerificationError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise VerificationError(f"{label} must be finite")
    return number


def integer_value(value: Any, label: str) -> int:
    number = numeric_value(value, label)
    if number != int(number):
        raise VerificationError(f"{label} must be an integer")
    return int(number)


def flag_value(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return binary_flag(value, label)
    raise VerificationError(f"{label} must be a binary flag")


def close_value(left: Any, right: float, label: str, *, tolerance: float | None = None) -> float:
    actual = numeric_value(left, label)
    if tolerance is None:
        tolerance = 64 * math.ulp(max(abs(actual), abs(right), 1.0))
    if abs(actual - right) > tolerance:
        raise VerificationError(
            f"{label}={actual:.17g} differs from source-locked {right:.17g}"
        )
    return actual


def source_mesh(geometry_id: str, mesh_level: str) -> tuple[dict[str, Any], int, int]:
    geometry = GEOMETRY_CONTRACT.get(geometry_id)
    if geometry is None or mesh_level not in geometry["levels"]:
        raise VerificationError(
            f"unregistered source-locked geometry/mesh pair: {geometry_id}/{mesh_level}"
        )
    bridge_count, rail_count = geometry["levels"][mesh_level]
    return geometry, bridge_count, rail_count


def validate_alignment_records(records: Any, label: str) -> None:
    if not isinstance(records, list):
        raise VerificationError(f"{label} must be an array")
    expected_keys = {
        (geometry_id, level, support_number)
        for geometry_id, geometry in GEOMETRY_CONTRACT.items()
        for level in MESH_LEVEL_ROLES
        for support_number in range(1, int(geometry["num_spans"]) + 2)
    }
    actual_keys: list[tuple[str, str, int]] = []
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise VerificationError(f"{label} row {index} must be an object")
        geometry_id = row.get("geometry_id")
        mesh_level = row.get("mesh_level")
        if not isinstance(geometry_id, str) or not isinstance(mesh_level, str):
            raise VerificationError(f"{label} row {index} has invalid identity")
        geometry, bridge_count, rail_count = source_mesh(geometry_id, mesh_level)
        support_number = integer_value(
            row.get("support_number"), f"{label} row {index} support_number"
        )
        key = (geometry_id, mesh_level, support_number)
        actual_keys.append(key)
        length = float(geometry["bridge_length_m"])
        spans = int(geometry["num_spans"])
        bridge_h = SLEEPER_SPACING_M / bridge_count
        rail_h = SLEEPER_SPACING_M / rail_count
        n_elements = round(length / bridge_h)
        nominal = (support_number - 1) * length / spans
        expected_node = round(nominal / bridge_h) + 1
        tolerance = max(256, 2 * n_elements) * math.ulp(max(length, 1.0))
        if integer_value(
            row.get("bridge_elements_per_sleeper_bay"),
            f"{label} row {index} bridge count",
        ) != bridge_count or integer_value(
            row.get("rail_elements_per_sleeper_bay"),
            f"{label} row {index} rail count",
        ) != rail_count:
            raise VerificationError(f"{label} row {index} mesh counts are not source locked")
        close_value(row.get("bridge_nominal_h_m"), bridge_h, f"{label} row {index} bridge h")
        close_value(row.get("rail_nominal_h_m"), rail_h, f"{label} row {index} rail h")
        close_value(row.get("bridge_actual_h_m"), bridge_h, f"{label} row {index} actual h")
        close_value(row.get("nominal_coordinate_m"), nominal, f"{label} row {index} nominal support")
        if integer_value(
            row.get("realized_node_number"), f"{label} row {index} realized node"
        ) != expected_node:
            raise VerificationError(f"{label} row {index} realized node is not source locked")
        realized = numeric_value(
            row.get("realized_coordinate_m"), f"{label} row {index} realized support"
        )
        if abs(realized - nominal) > tolerance:
            raise VerificationError(f"{label} row {index} changes physical support geometry")
        signed = numeric_value(
            row.get("signed_offset_m"), f"{label} row {index} signed offset"
        )
        # CSV rendering can round the realized coordinate to the nominal
        # value while retaining the smaller pre-rounding signed offset.  Bind
        # both independently to the source-locked roundoff envelope.
        if abs(signed) > tolerance:
            raise VerificationError(f"{label} row {index} signed offset exceeds tolerance")
        close_value(
            row.get("absolute_offset_m"),
            abs(signed),
            f"{label} row {index} absolute offset",
        )
        close_value(
            row.get("alignment_tolerance_m"),
            tolerance,
            f"{label} row {index} alignment tolerance",
        )
        if not flag_value(
            row.get("exactly_aligned_for_qualification"),
            f"{label} row {index} alignment flag",
        ):
            raise VerificationError(f"{label} row {index} is not aligned")
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise VerificationError(f"{label} inventory is missing, extra, or duplicated")


def validate_protocol_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("schema") != "numerical-vv-protocol-v1":
        raise VerificationError("protocol snapshot schema mismatch")
    close_value(snapshot.get("sleeper_spacing_m"), SLEEPER_SPACING_M, "protocol sleeper spacing")
    if snapshot.get("micro_run_kind") != MICRO_KIND:
        raise VerificationError("protocol snapshot micro run kind mismatch")
    if integer_value(
        snapshot.get("max_gci_observed_order"), "protocol GCI order ceiling"
    ) != MAX_GCI_OBSERVED_ORDER:
        raise VerificationError("protocol snapshot GCI order ceiling is not source locked")
    if flag_value(
        snapshot.get("qualification_verifier_implemented"),
        "protocol qualification verifier flag",
    ) or flag_value(snapshot.get("qualification_ready"), "protocol qualification-ready flag"):
        raise VerificationError("protocol snapshot escalates qualification capability")

    mesh_levels = snapshot.get("mesh_levels")
    if not isinstance(mesh_levels, list) or [
        (row.get("id"), row.get("role")) for row in mesh_levels if isinstance(row, dict)
    ] != list(MESH_LEVEL_ROLES.items()):
        raise VerificationError("protocol snapshot mesh-level roles mismatch")

    geometries = snapshot.get("geometries")
    if not isinstance(geometries, list) or len(geometries) != len(GEOMETRY_CONTRACT):
        raise VerificationError("protocol snapshot geometry inventory mismatch")
    seen_geometries: set[str] = set()
    for row in geometries:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise VerificationError("protocol snapshot contains an invalid geometry")
        geometry_id = row["id"]
        geometry = GEOMETRY_CONTRACT.get(geometry_id)
        if geometry is None or geometry_id in seen_geometries:
            raise VerificationError("protocol snapshot geometry is extra or duplicated")
        seen_geometries.add(geometry_id)
        close_value(
            row.get("bridge_length_m"),
            float(geometry["bridge_length_m"]),
            f"protocol {geometry_id} length",
        )
        if integer_value(row.get("num_spans"), f"protocol {geometry_id} spans") != geometry["num_spans"]:
            raise VerificationError(f"protocol {geometry_id} span count mismatch")

    sequences = snapshot.get("geometry_mesh_sequences")
    if not isinstance(sequences, list):
        raise VerificationError("protocol mesh sequences must be an array")
    expected_sequence_keys = {
        (geometry_id, level)
        for geometry_id in GEOMETRY_CONTRACT
        for level in MESH_LEVEL_ROLES
    }
    sequence_keys: list[tuple[str, str]] = []
    for index, row in enumerate(sequences, 1):
        if not isinstance(row, dict):
            raise VerificationError(f"protocol mesh sequence {index} is not an object")
        geometry_id = row.get("geometry_id")
        mesh_level = row.get("level_id")
        if not isinstance(geometry_id, str) or not isinstance(mesh_level, str):
            raise VerificationError(f"protocol mesh sequence {index} has invalid identity")
        geometry, bridge_count, rail_count = source_mesh(geometry_id, mesh_level)
        del geometry
        sequence_keys.append((geometry_id, mesh_level))
        if integer_value(row.get("bridge_elements_per_sleeper_bay"), "protocol bridge count") != bridge_count:
            raise VerificationError("protocol bridge count differs from source lock")
        if integer_value(row.get("rail_elements_per_sleeper_bay"), "protocol rail count") != rail_count:
            raise VerificationError("protocol rail count differs from source lock")
        close_value(row.get("bridge_nominal_h_m"), SLEEPER_SPACING_M / bridge_count, "protocol bridge h")
        close_value(row.get("rail_nominal_h_m"), SLEEPER_SPACING_M / rail_count, "protocol rail h")
    if len(sequence_keys) != len(expected_sequence_keys) or set(sequence_keys) != expected_sequence_keys:
        raise VerificationError("protocol mesh sequence inventory is missing, extra, or duplicated")

    policy = snapshot.get("support_alignment_policy")
    if not isinstance(policy, dict) or policy.get("current_policy_status") != "SOURCE_LOCKED_GEOMETRY_SPECIFIC_ALIGNED":
        raise VerificationError("protocol support-alignment policy status mismatch")
    validate_alignment_records(snapshot.get("registered_support_alignment"), "protocol alignment")


def validate_case_row(row: dict[str, str], index: int) -> dict[str, Any]:
    geometry_id = row["geometry_id"]
    mesh_level = row["mesh_level"]
    geometry, bridge_count, rail_count = source_mesh(geometry_id, mesh_level)
    length = float(geometry["bridge_length_m"])
    bridge_h = SLEEPER_SPACING_M / bridge_count
    rail_h = SLEEPER_SPACING_M / rail_count
    n_elements = round(length / bridge_h)
    alignment_tolerance = max(256, 2 * n_elements) * math.ulp(max(length, 1.0))
    expected_case_id = f"{geometry_id}_{mesh_level}_ss_micro"
    if row["case_id"] != expected_case_id:
        raise VerificationError(f"case row {index} case_id is not source locked")
    if row["study_kind"] != "simply_supported_bridge_fixture":
        raise VerificationError(f"case row {index} study kind mismatch")
    close_value(row["bridge_length_m"], length, f"case row {index} bridge length")
    if positive_integer(row["bridge_elements_per_sleeper_bay"], f"case row {index} bridge count") != bridge_count:
        raise VerificationError(f"case row {index} bridge count differs from source lock")
    if positive_integer(row["rail_elements_per_sleeper_bay"], f"case row {index} rail count") != rail_count:
        raise VerificationError(f"case row {index} rail count differs from source lock")
    close_value(row["bridge_nominal_h_m"], bridge_h, f"case row {index} bridge nominal h")
    close_value(row["rail_nominal_h_m"], rail_h, f"case row {index} rail nominal h")
    close_value(row["bridge_actual_h_m"], bridge_h, f"case row {index} bridge actual h")
    if positive_integer(row["bridge_n_elements"], f"case row {index} bridge element total") != n_elements:
        raise VerificationError(f"case row {index} bridge element total mismatch")
    if positive_integer(row["bridge_n_nodes"], f"case row {index} bridge node total") != n_elements + 1:
        raise VerificationError(f"case row {index} bridge node total mismatch")
    support_offset = numeric_value(row["max_support_offset_m"], f"case row {index} support offset")
    if support_offset < 0 or support_offset > alignment_tolerance:
        raise VerificationError(f"case row {index} support offset changes physical geometry")
    if not binary_flag(row["support_alignment_pass"], f"case row {index} support alignment"):
        raise VerificationError(f"case row {index} does not pass support alignment")
    if not binary_flag(row["bridge_mesh_executed"], f"case row {index} bridge execution"):
        raise VerificationError("bridge-only micro did not execute bridge mesh")
    if binary_flag(row["rail_mesh_executed"], f"case row {index} rail execution"):
        raise VerificationError("bridge-only micro must not claim a rail solve")
    try:
        rail_actual = float(row["rail_actual_h_m"])
    except ValueError as exc:
        raise VerificationError(f"case row {index} rail actual h is not numeric") from exc
    if not math.isnan(rail_actual):
        raise VerificationError("bridge-only micro must record rail_actual_h_m=NaN")
    return {
        "geometry_id": geometry_id,
        "mesh_level": mesh_level,
        "bridge_length_m": length,
        "bridge_count": bridge_count,
        "rail_count": rail_count,
        "bridge_h": bridge_h,
        "rail_h": rail_h,
    }


def validate_descriptor(
    row: dict[str, str], case_row: dict[str, str], case_contract: dict[str, Any]
) -> dict[str, Any]:
    try:
        descriptor = json.loads(row["descriptor_json"])
    except json.JSONDecodeError as exc:
        raise VerificationError("descriptor preimage is not valid JSON") from exc
    if not isinstance(descriptor, dict) or set(descriptor) != DESCRIPTOR_FIELDS:
        raise VerificationError("descriptor JSON schema mismatch")
    if row["descriptor_schema"] != "numerical-vv-bridge-fixture-v1":
        raise VerificationError("descriptor table schema mismatch")
    if row["geometry_id"] != case_row["geometry_id"] or row["mesh_level"] != case_row["mesh_level"]:
        raise VerificationError("descriptor table geometry/mesh identity mismatch")
    exact_text = {
        "case_id": case_row["case_id"],
        "geometry_id": case_row["geometry_id"],
        "mesh_level": case_row["mesh_level"],
        "scope": "nonqualifying-simply-supported-production-matrix-micro",
    }
    for field, expected in exact_text.items():
        if descriptor.get(field) != expected:
            raise VerificationError(f"descriptor {field} differs from its case")
    close_value(descriptor.get("bridge_length_m"), case_contract["bridge_length_m"], "descriptor bridge length")
    if integer_value(descriptor.get("bridge_elements_per_sleeper_bay"), "descriptor bridge count") != case_contract["bridge_count"]:
        raise VerificationError("descriptor bridge count differs from its case")
    if integer_value(descriptor.get("rail_elements_per_sleeper_bay"), "descriptor rail count") != case_contract["rail_count"]:
        raise VerificationError("descriptor rail count differs from its case")
    close_value(descriptor.get("bridge_nominal_h_m"), case_contract["bridge_h"], "descriptor bridge h")
    close_value(descriptor.get("rail_nominal_h_m"), case_contract["rail_h"], "descriptor rail h")
    if flag_value(descriptor.get("rail_mesh_executed"), "descriptor rail execution"):
        raise VerificationError("descriptor falsely claims a rail solve")
    point_load = numeric_value(descriptor.get("point_load_N"), "descriptor point load")
    if point_load <= 0:
        raise VerificationError("descriptor point load must be positive")
    return descriptor


def normalize_json_list(value: Any, label: str) -> list[Any]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise VerificationError(f"{label} must not be empty")
    return values


def validate_input_descriptor(
    input_descriptor: dict[str, Any], manifest: dict[str, Any], case_rows: list[dict[str, str]]
) -> None:
    if input_descriptor.get("schema") != "numerical-vv-micro-input-v1":
        raise VerificationError("input descriptor schema mismatch")
    if flag_value(input_descriptor.get("dynamic_solver_executed"), "input dynamic solver flag"):
        raise VerificationError("micro input descriptor claims a dynamic solve")
    if flag_value(input_descriptor.get("qualification_requested"), "input qualification flag"):
        raise VerificationError("micro input descriptor requests qualification")
    if manifest.get("input_descriptor") != input_descriptor:
        raise VerificationError("manifest/input_descriptor.json semantic mismatch")
    canonical = json.dumps(input_descriptor, separators=(",", ":"), ensure_ascii=False)
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if computed != manifest.get("input_hash"):
        raise VerificationError("manifest input_hash does not match input descriptor")
    lengths = [numeric_value(value, "input bridge length") for value in normalize_json_list(input_descriptor.get("bridge_lengths_m"), "input bridge lengths")]
    mesh_ids = normalize_json_list(input_descriptor.get("mesh_level_ids"), "input mesh levels")
    if any(not isinstance(level, str) for level in mesh_ids):
        raise VerificationError("input mesh levels must be text")
    if len(set(lengths)) != len(lengths) or len(set(mesh_ids)) != len(mesh_ids):
        raise VerificationError("input case dimensions contain duplicates")
    requested_geometry_ids: list[str] = []
    for length in lengths:
        matches = [
            geometry_id
            for geometry_id, geometry in GEOMETRY_CONTRACT.items()
            if abs(length - float(geometry["bridge_length_m"]))
            <= 64 * math.ulp(max(abs(length), 1.0))
        ]
        if len(matches) != 1:
            raise VerificationError("input descriptor contains unregistered bridge length")
        requested_geometry_ids.append(matches[0])
    if any(level not in MESH_LEVEL_ROLES for level in mesh_ids):
        raise VerificationError("input descriptor contains unregistered mesh level")
    expected = {
        (geometry_id, level) for geometry_id in requested_geometry_ids for level in mesh_ids
    }
    actual = {(row["geometry_id"], row["mesh_level"]) for row in case_rows}
    if len(case_rows) != len(expected) or actual != expected:
        raise VerificationError("input descriptor and case inventory differ")


def verify_package(
    package: Path,
    *,
    allow_nonqualifying_micro: bool,
    require_qualification: bool,
    source_root: Path | None,
) -> dict[str, Any]:
    if require_qualification:
        raise VerificationError(
            "qualification verification is not implemented; this checker "
            "can authenticate only an explicitly nonqualifying micro package"
        )
    # Reserved for the future independent source-bound verifier.  Keeping the
    # argument is CLI/API compatibility, not evidence that it is consumed.
    del source_root
    if package.is_symlink():
        raise VerificationError("package root must not be a symlink")
    package = package.resolve(strict=True)
    if package.is_symlink() or not package.is_dir():
        raise VerificationError("package root must be a regular directory")
    if (package / "_RUN_INCOMPLETE").exists():
        raise VerificationError("_RUN_INCOMPLETE is present")
    manifest_path = package / "manifest.json"
    verdict_path = package / "vv_verdict.json"
    marker_path = package / "_RUN_COMPLETE"
    for path in (manifest_path, verdict_path, marker_path):
        require_regular_unlinked(path)
    manifest = load_json(manifest_path)
    verdict = load_json(verdict_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise VerificationError("manifest schema mismatch")
    if verdict.get("schema") != VERDICT_SCHEMA:
        raise VerificationError("verdict schema mismatch")
    run_kind = manifest.get("run_kind")
    if run_kind != MICRO_KIND:
        raise VerificationError(
            "only run_kind=nonqualifying_micro is supported; scientific "
            "qualification requires a future independent verifier"
        )
    manifest_hash = sha256_file(manifest_path)
    marker = marker_path.read_text(encoding="utf-8")
    if f"manifest_sha256={manifest_hash}" not in marker:
        raise VerificationError("completion marker does not authenticate manifest")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise VerificationError("manifest artifacts must be a nonempty array")
    names: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise VerificationError("artifact entries must be objects")
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise VerificationError("artifact path must be one safe flat filename")
        if name in names:
            raise VerificationError(f"duplicate artifact path: {name}")
        names.append(name)
        path = package / name
        require_regular_unlinked(path)
        if path.stat().st_size != entry.get("bytes"):
            raise VerificationError(f"artifact size mismatch: {name}")
        if sha256_file(path) != entry.get("sha256"):
            raise VerificationError(f"artifact digest mismatch: {name}")
    actual = {path.name for path in package.iterdir()}
    expected = set(names) | {"manifest.json", "_RUN_COMPLETE"}
    if actual != expected:
        raise VerificationError(
            f"artifact inventory mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    if not MICRO_FILES.issubset(names):
        raise VerificationError(f"micro foundation files missing: {sorted(MICRO_FILES-set(names))}")

    protocol_snapshot = load_json(package / "protocol_snapshot.json")
    validate_protocol_snapshot(protocol_snapshot)
    support_rows = read_csv(
        package / "support_alignment.csv",
        {
            "geometry_id",
            "mesh_level",
            "bridge_elements_per_sleeper_bay",
            "rail_elements_per_sleeper_bay",
            "bridge_nominal_h_m",
            "rail_nominal_h_m",
            "bridge_actual_h_m",
            "support_number",
            "nominal_coordinate_m",
            "realized_node_number",
            "realized_coordinate_m",
            "signed_offset_m",
            "absolute_offset_m",
            "alignment_tolerance_m",
            "exactly_aligned_for_qualification",
        },
    )
    validate_alignment_records(support_rows, "support_alignment.csv")

    case_rows = read_csv(
        package / "case_table.csv",
        {
            "case_id",
            "geometry_id",
            "study_kind",
            "mesh_level",
            "bridge_length_m",
            "bridge_elements_per_sleeper_bay",
            "rail_elements_per_sleeper_bay",
            "bridge_nominal_h_m",
            "rail_nominal_h_m",
            "bridge_actual_h_m",
            "rail_actual_h_m",
            "bridge_mesh_executed",
            "rail_mesh_executed",
            "bridge_n_elements",
            "bridge_n_nodes",
            "max_support_offset_m",
            "support_alignment_pass",
            "source_commit",
            "input_hash",
        },
    )
    if len(case_rows) != manifest.get("completed_mesh_case_count"):
        raise VerificationError("case count does not match manifest")
    case_ids = [row["case_id"] for row in case_rows]
    if len(set(case_ids)) != len(case_ids):
        raise VerificationError("duplicate case_id")
    case_contracts: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(case_rows, 1):
        case_contracts[row["case_id"]] = validate_case_row(row, index)
        if row["source_commit"] != manifest.get("source_commit"):
            raise VerificationError("case source_commit does not match manifest")
        if row["input_hash"] != manifest.get("input_hash"):
            raise VerificationError("case input_hash does not match manifest")

    input_descriptor = load_json(package / "input_descriptor.json")
    validate_input_descriptor(input_descriptor, manifest, case_rows)

    descriptor_rows = read_csv(
        package / "descriptor_hashes.csv",
        {
            "case_id",
            "geometry_id",
            "mesh_level",
            "descriptor_schema",
            "descriptor_json",
            "descriptor_hash",
            "source_commit",
            "input_hash",
        },
    )
    descriptor_ids = [row["case_id"] for row in descriptor_rows]
    if (
        len(descriptor_rows) != len(case_rows)
        or len(set(descriptor_ids)) != len(descriptor_ids)
        or set(descriptor_ids) != set(case_ids)
    ):
        raise VerificationError("descriptor/case identity mismatch")
    case_by_id = {row["case_id"]: row for row in case_rows}
    for row in descriptor_rows:
        digest = row["descriptor_hash"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise VerificationError("invalid descriptor SHA-256")
        computed = hashlib.sha256(row["descriptor_json"].encode("utf-8")).hexdigest()
        if computed != digest:
            raise VerificationError("descriptor hash does not match its preimage")
        matching_case = case_by_id[row["case_id"]]
        if row["source_commit"] != matching_case["source_commit"]:
            raise VerificationError("descriptor source_commit mismatch")
        if row["input_hash"] != matching_case["input_hash"]:
            raise VerificationError("descriptor input_hash mismatch")
        descriptor = validate_descriptor(
            row, matching_case, case_contracts[row["case_id"]]
        )
        close_value(
            descriptor["point_load_N"],
            numeric_value(input_descriptor.get("point_load_N"), "input point load"),
            "descriptor/input point load",
        )

    if not allow_nonqualifying_micro:
        raise VerificationError("nonqualifying micro requires explicit opt-in")
    forbidden = (
        bool(manifest.get("numerical_verification_claim_authorized"))
        or bool(manifest.get("physical_validation_claim_authorized"))
        or bool(verdict.get("numerical_verification_claim_authorized"))
        or bool(verdict.get("physical_validation_claim_authorized"))
        or bool(verdict.get("production_resolution_qualified"))
        or verdict.get("overall_status") != "UNVERIFIED"
    )
    if forbidden:
        raise VerificationError("micro escalates an unauthorized claim")

    return {
        "schema": "numerical-vv-python-receipt-v1",
        "package": str(package),
        "manifest_sha256": manifest_hash,
        "run_kind": run_kind,
        "status": "NONQUALIFYING_INTEGRITY_PASS",
        "integrity_pass": True,
        "qualification_pass": False,
        "numerical_verification_claim_authorized": False,
        "physical_validation_claim_authorized": False,
    }


def write_receipt(path: Path, receipt: dict[str, Any], package: Path) -> None:
    resolved_parent = path.parent.resolve(strict=True)
    resolved_target = resolved_parent / path.name
    try:
        resolved_target.relative_to(package.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise VerificationError("receipt must be outside the immutable package")
    if resolved_target.exists():
        raise VerificationError("receipt target already exists")
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=resolved_parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved_target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def self_test_qualification_refusal() -> None:
    """Prove qualification is refused before any package can self-attest."""
    try:
        verify_package(
            Path("qualification-package-must-never-be-read"),
            allow_nonqualifying_micro=True,
            require_qualification=True,
            source_root=None,
        )
    except VerificationError as exc:
        if "qualification verification is not implemented" not in str(exc):
            raise VerificationError(
                "qualification refusal raised the wrong failure"
            ) from exc
        return
    raise VerificationError(
        "qualification request escaped the mandatory refusal gate"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path)
    parser.add_argument("--allow-nonqualifying-micro", action="store_true")
    parser.add_argument("--require-qualification", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--self-test-qualification-refusal", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test_qualification_refusal:
            self_test_qualification_refusal()
            print("NUMERICAL V&V QUALIFICATION REFUSAL: PASS")
            return 0
        if args.package is None:
            parser.error("package is required unless a self-test is requested")
        receipt = verify_package(
            args.package,
            allow_nonqualifying_micro=args.allow_nonqualifying_micro,
            require_qualification=args.require_qualification,
            source_root=args.source_root,
        )
        if args.receipt is not None:
            write_receipt(args.receipt, receipt, args.package)
    except (OSError, VerificationError) as exc:
        print(f"NUMERICAL V&V PACKAGE: FAIL — {exc}")
        return 1
    print(f"NUMERICAL V&V PACKAGE: {receipt['status']}")
    print(f"manifest_sha256={receipt['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
