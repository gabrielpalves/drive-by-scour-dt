"""Frozen policy, selection parsing, and inventory checks."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from contact_gate_path_safety import GateError
from contact_gate_core import (
    CHANNELS,
    CHANNEL_SCHEMA_ID,
    CLOSURE_INTERPRETATION,
    COARSE_CORR,
    COARSE_NMAX,
    COARSE_NRMSE,
    COMMON_DX_M,
    DT_MS,
    DatasetDescriptor,
    EQUIVALENCE_ATOL,
    EQUIVALENCE_RTOL,
    EXPECTED_CASES,
    EXPECTED_FAMILIES,
    EXPECTED_PASSAGES,
    EXPECTED_STATES,
    EXPECTED_TOTAL_CASES,
    FINEST_IDENTITY_ATOL,
    FRACTION_GATE,
    GATES_N,
    GCI_FS,
    GCI_METHOD,
    GCI_P_MAX,
    GCI_P_MIN,
    MEDIUM_CORR,
    MEDIUM_NMAX,
    MEDIUM_NRMSE,
    POLICY_SCHEMA,
    RECON_ATOL,
    RECON_RTOL,
    SHA256_RE,
    STAGES,
    SelectionRow,
    TIME_GRID_ULPS,
    WAVEFORM_MONOTONIC_ATOL,
    _canonical_lf_file,
    _sha256_bytes,
)

def _expected_policy_fields(source_commit: str) -> dict[str, str]:
    return {
        "schema": POLICY_SCHEMA,
        "closure_interpretation": CLOSURE_INTERPRETATION,
        "source_commit": source_commit,
        "stages": ",".join(STAGES),
        "expected_states": ",".join(str(EXPECTED_STATES[stage]) for stage in STAGES),
        "expected_passages": str(EXPECTED_PASSAGES),
        "expected_cases": str(EXPECTED_TOTAL_CASES),
        "dt_ms": "1,0.5,0.25",
        "gates_n": "0,12000,24000",
        "fraction_gate": "0.002",
        "common_dx_m": "0.01",
        "reconstruction_rtol": "1e-10",
        "reconstruction_atol": "9.9999999999999998e-13",
        "coarse_nrmse_max": "0.050000000000000003",
        "coarse_nmax_max": "0.10000000000000001",
        "coarse_corr_min": "0.995",
        "medium_nrmse_max": "0.02",
        "medium_nmax_max": "0.050000000000000003",
        "medium_corr_min": "0.999",
        "gci_safety_factor": "1.25",
        "gci_method": GCI_METHOD,
        "equivalence_rtol": "1e-10",
        "equivalence_atol": "9.9999999999999998e-13",
        "gci_p_min": "1e-08",
        "gci_p_max": "50",
        "qoi_gci_required": "true",
        "time_grid_ulps": "8",
        "waveform_monotonic_atol": "9.9999999999999998e-13",
        "finest_identity_atol": "9.9999999999999998e-13",
        "expected_channels": ",".join(CHANNELS),
        "channel_schema_id": CHANNEL_SCHEMA_ID,
    }


def _expected_policy_json(source_commit: str) -> dict[str, Any]:
    return {
        "schema": POLICY_SCHEMA,
        "closure_interpretation": CLOSURE_INTERPRETATION,
        "source_commit": source_commit,
        "stages": list(STAGES),
        "expected_states": [EXPECTED_STATES[stage] for stage in STAGES],
        "expected_passages": EXPECTED_PASSAGES,
        "expected_cases": EXPECTED_TOTAL_CASES,
        "dt_ms": list(DT_MS),
        "gates_n": list(GATES_N),
        "fraction_gate": FRACTION_GATE,
        "common_dx_m": COMMON_DX_M,
        "reconstruction_rtol": RECON_RTOL,
        "reconstruction_atol": RECON_ATOL,
        "coarse_nrmse_max": COARSE_NRMSE,
        "coarse_nmax_max": COARSE_NMAX,
        "coarse_corr_min": COARSE_CORR,
        "medium_nrmse_max": MEDIUM_NRMSE,
        "medium_nmax_max": MEDIUM_NMAX,
        "medium_corr_min": MEDIUM_CORR,
        "gci_safety_factor": GCI_FS,
        "gci_method": GCI_METHOD,
        "equivalence_rtol": EQUIVALENCE_RTOL,
        "equivalence_atol": EQUIVALENCE_ATOL,
        "gci_p_min": GCI_P_MIN,
        "gci_p_max": GCI_P_MAX,
        "qoi_gci_required": True,
        "time_grid_ulps": TIME_GRID_ULPS,
        "waveform_monotonic_atol": WAVEFORM_MONOTONIC_ATOL,
        "finest_identity_atol": FINEST_IDENTITY_ATOL,
        "expected_channels": list(CHANNELS),
        "channel_schema_id": CHANNEL_SCHEMA_ID,
    }


def _parse_policy(
    path: Path, source_commit: str,
) -> tuple[dict[str, str], str, str]:
    text = _canonical_lf_file(path, "closure policy")
    lines = text.splitlines()
    if len(lines) < 2 or not lines[-1].startswith("sha256="):
        raise GateError("closure_policy.txt lacks terminal sha256")
    declared = lines[-1].split("=", 1)[1]
    descriptor = "\n".join(lines[:-1])
    actual = _sha256_bytes(descriptor.encode("utf-8"))
    if declared != actual or not SHA256_RE.fullmatch(declared):
        raise GateError("closure policy digest mismatch")
    fields: dict[str, str] = {}
    for line in lines[:-1]:
        if "=" not in line:
            raise GateError(f"malformed policy line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise GateError(f"duplicate/empty policy field: {key!r}")
        fields[key] = value
    expected = _expected_policy_fields(source_commit)
    if fields != expected:
        missing = sorted(set(expected) - set(fields))
        extra = sorted(set(fields) - set(expected))
        drift = sorted(
            key for key in set(expected) & set(fields)
            if expected[key] != fields[key]
        )
        raise GateError(
            f"closure policy differs from checker (missing={missing}, "
            f"extra={extra}, drift={drift})"
        )
    return fields, declared, descriptor


def _parse_dataset_line(raw: str, expected_stage: str) -> DatasetDescriptor:
    parts = raw.split("\t")
    keys = (
        "dataset_dir_sha256", "content_root", "case_info", "damage_states",
        "file_digests", "complete", "host_receipt", "fingerprint",
        "qual_source", "qual_executed", "host_diagnostic",
    )
    if len(parts) != 2 + len(keys) or parts[:2] != ["#dataset", expected_stage]:
        raise GateError(f"malformed/out-of-order dataset descriptor: {raw!r}")
    values: dict[str, str] = {}
    for token, expected_key in zip(parts[2:], keys):
        if token.count("=") != 1:
            raise GateError(f"malformed dataset token: {token!r}")
        key, value = token.split("=", 1)
        if key != expected_key or not SHA256_RE.fullmatch(value):
            raise GateError(
                f"dataset {expected_stage} field {expected_key} is malformed"
            )
        values[key] = value
    return DatasetDescriptor(
        stage=expected_stage,
        raw_line=raw,
        **values,
    )


def _parse_selection(
    path: Path,
) -> tuple[list[SelectionRow], str, list[DatasetDescriptor]]:
    text = _canonical_lf_file(path, "selection manifest")
    actual_sha = _sha256_bytes(text.encode("utf-8"))
    lines = text.splitlines()
    dataset_count = len(STAGES)
    dataset_lines = lines[:dataset_count]
    if len(lines) < dataset_count + 1 or any(
        not line.startswith("#dataset\t") for line in dataset_lines
    ) or any(
        line.startswith("#dataset\t") for line in lines[dataset_count:]
    ):
        raise GateError(
            f"selection must bind exactly {dataset_count} dataset descriptors"
        )
    datasets = [
        _parse_dataset_line(raw, stage)
        for raw, stage in zip(dataset_lines, STAGES)
    ]
    header_index = dataset_count
    expected_header = (
        "ordinal\tstage\tstate_index\tpassage_index\tstate_uid\tstate_family\t"
        "state_file_sha256\tsaved_bridge_flag\tsaved_track_flag\t"
        "saved_fraction\tsaved_signed_peak_N"
    )
    if lines[header_index] != expected_header:
        raise GateError("selection header mismatch")
    rows: list[SelectionRow] = []
    for raw in lines[header_index + 1:]:
        parts = raw.split("\t")
        if len(parts) != 11:
            raise GateError(f"selection row has {len(parts)} columns")
        try:
            row = SelectionRow(
                ordinal=int(parts[0]),
                stage=parts[1],
                state_index=int(parts[2]),
                passage_index=int(parts[3]),
                state_uid=parts[4],
                state_family=parts[5],
                state_file_sha256=parts[6],
                saved_bridge_flag=float(parts[7]),
                saved_track_flag=float(parts[8]),
                saved_fraction=float(parts[9]),
                saved_signed_peak_n=float(parts[10]),
            )
        except ValueError as exc:
            raise GateError("selection row has invalid numeric content") from exc
        rows.append(row)
    _validate_inventory(rows)
    return rows, actual_sha, datasets


def _validate_inventory(rows: list[SelectionRow]) -> None:
    if len(rows) != EXPECTED_TOTAL_CASES:
        raise GateError(
            f"selection has {len(rows)} rows, expected {EXPECTED_TOTAL_CASES}"
        )
    if [row.ordinal for row in rows] != list(
        range(1, EXPECTED_TOTAL_CASES + 1)
    ):
        raise GateError(
            f"selection ordinals are not exactly 1..{EXPECTED_TOTAL_CASES}"
        )
    expected_stage_sequence = [
        stage for stage in STAGES for _ in range(EXPECTED_CASES[stage])
    ]
    if [row.stage for row in rows] != expected_stage_sequence:
        raise GateError("selection stage order/count differs")
    for stage in STAGES:
        stage_rows = [row for row in rows if row.stage == stage]
        keys = [(row.state_index, row.passage_index) for row in stage_rows]
        expected_keys = [
            (state, passage)
            for state in range(1, EXPECTED_STATES[stage] + 1)
            for passage in range(1, EXPECTED_PASSAGES + 1)
        ]
        if keys != expected_keys:
            raise GateError(f"{stage} is not the complete state x passage product")
        uid_by_state: dict[int, str] = {}
        family_by_state: dict[int, str] = {}
        sha_by_state: dict[int, str] = {}
        for row in stage_rows:
            if not SHA256_RE.fullmatch(row.state_file_sha256):
                raise GateError(f"{stage} has malformed state SHA")
            if row.saved_bridge_flag not in (0.0, 1.0) \
                    or row.saved_track_flag not in (0.0, 1.0) \
                    or not 0 <= row.saved_fraction <= 1 \
                    or not math.isfinite(row.saved_signed_peak_n):
                raise GateError(f"{stage} has malformed saved contact data")
            for mapping, value, label in (
                (uid_by_state, row.state_uid, "UID"),
                (family_by_state, row.state_family, "family"),
                (sha_by_state, row.state_file_sha256, "SHA"),
            ):
                previous = mapping.setdefault(row.state_index, value)
                if previous != value:
                    raise GateError(f"{stage} state {row.state_index} changes {label}")
        if len(set(uid_by_state.values())) != EXPECTED_STATES[stage]:
            raise GateError(f"{stage} StateUIDs are not unique")
        observed = {
            family: sum(value == family for value in family_by_state.values())
            for family in EXPECTED_FAMILIES[stage]
        }
        if observed != EXPECTED_FAMILIES[stage]:
            raise GateError(f"{stage} family inventory differs: {observed}")


def _expected_selection_records(
    rows: list[SelectionRow],
    summary: dict[str, Any],
) -> dict[str, Any]:
    directory_by_stage = {
        item["stage"]: item["dataset_dir"]
        for item in summary["datasets"]
    }
    return {
        "ordinal": [row.ordinal for row in rows],
        "stage": [row.stage for row in rows],
        "dataset_dir": [directory_by_stage[row.stage] for row in rows],
        "state_index": [row.state_index for row in rows],
        "passage_index": [row.passage_index for row in rows],
        "state_uid": [row.state_uid for row in rows],
        "state_family": [row.state_family for row in rows],
        "state_file_sha256": [row.state_file_sha256 for row in rows],
        "saved_contact_lost_bridge": [
            row.saved_bridge_flag for row in rows
        ],
        "saved_contact_lost_track": [row.saved_track_flag for row in rows],
        "saved_tension_fraction": [row.saved_fraction for row in rows],
        "saved_peak_signed_N": [row.saved_signed_peak_n for row in rows],
    }
