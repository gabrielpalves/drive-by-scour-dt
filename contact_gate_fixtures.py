"""Synthetic fixture builders for the contact-closure checker self-tests.

This module owns ONLY test-fixture construction: synthetic selection rows,
dataset descriptors, policy/selection text, fully-valid case JSON payloads,
plain-report projections and the complete synthetic 420-case gate tree.  It
is imported exclusively by ``contact_gate_selftests`` (the self-test driver
dispatched by ``check_contact_closure_gate.py`` when run without arguments).
Although no real-verification path executes fixture code, its bytes are part
of the explicit transitive verifier root so the argument-free behavioral suite
cannot drift independently of the checker it qualifies.

All identity constants and recomputation primitives are imported from
``check_contact_closure_gate`` so a fixture can never silently drift from
the verifier's own definitions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from check_contact_closure_gate import (
    CHANNELS,
    CHANNEL_SCHEMA_ID,
    COMMON_DX_M,
    DT_MS,
    EXPECTED_FAMILIES,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_L_BRIDGE_M,
    EXPECTED_MATLAB_RELEASE,
    EXPECTED_PASSAGES,
    EXPECTED_STATES,
    EXPECTED_TOTAL_CASES,
    FRACTION_GATE,
    GATES_N,
    NUMERIC_HASH_SELFCHECK,
    POST_DECK_WINDOW_M,
    RECON_ATOL,
    RECON_RTOL,
    ROOT,
    SOLVER_MODULES,
    STAGES,
    STUDY_SCHEMA,
    SUMMARY_SCHEMA,
    DatasetDescriptor,
    SelectionRow,
    _expected_policy_fields,
    _gate_execution_root,
    _gci,
    _generator_source_identity,
    _locked_matlab_environment,
    _sha256_bytes,
    _sha256_file,
    _solver_execution_identity,
)


def _synthetic_rows() -> list[SelectionRow]:
    rows: list[SelectionRow] = []
    ordinal = 0
    for stage in STAGES:
        families: list[str] = []
        for family, count in EXPECTED_FAMILIES[stage].items():
            families.extend([family] * count)
        for state_index, family in enumerate(families, 1):
            for passage in range(1, EXPECTED_PASSAGES + 1):
                ordinal += 1
                rows.append(SelectionRow(
                    ordinal=ordinal,
                    stage=stage,
                    state_index=state_index,
                    passage_index=passage,
                    state_uid=f"{stage}|state={state_index}",
                    state_family=family,
                    state_file_sha256=f"{state_index:064x}"[-64:],
                    saved_bridge_flag=0,
                    saved_track_flag=0,
                    saved_fraction=0,
                    saved_signed_peak_n=-100000,
                ))
    return rows


def _write_canonical_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes((
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8"))


def _synthetic_hash(stage: str, field: str) -> str:
    return _sha256_bytes(f"synthetic:{stage}:{field}".encode("utf-8"))


def _synthetic_descriptors(
    dataset_dirs: list[Path],
) -> list[DatasetDescriptor]:
    descriptors: list[DatasetDescriptor] = []
    keys = (
        "dataset_dir_sha256", "content_root", "case_info", "damage_states",
        "file_digests", "complete", "host_receipt", "fingerprint",
        "qual_source", "qual_executed", "host_diagnostic",
    )
    for stage, dataset_dir in zip(STAGES, dataset_dirs):
        values = {
            key: (
                _sha256_bytes(str(dataset_dir).encode("utf-8"))
                if key == "dataset_dir_sha256"
                else _synthetic_hash(stage, key)
            )
            for key in keys
        }
        raw = "\t".join((
            "#dataset", stage,
            *(f"{key}={values[key]}" for key in keys),
        ))
        descriptors.append(DatasetDescriptor(
            stage=stage,
            raw_line=raw,
            **values,
        ))
    return descriptors


def _synthetic_selection_text(
    rows: list[SelectionRow],
    descriptors: list[DatasetDescriptor],
) -> str:
    header = (
        "ordinal\tstage\tstate_index\tpassage_index\tstate_uid\tstate_family\t"
        "state_file_sha256\tsaved_bridge_flag\tsaved_track_flag\t"
        "saved_fraction\tsaved_signed_peak_N"
    )
    lines = [descriptor.raw_line for descriptor in descriptors]
    lines.append(header)
    lines.extend(
        "\t".join((
            str(row.ordinal),
            row.stage,
            str(row.state_index),
            str(row.passage_index),
            row.state_uid,
            row.state_family,
            row.state_file_sha256,
            format(row.saved_bridge_flag, ".17g"),
            format(row.saved_track_flag, ".17g"),
            format(row.saved_fraction, ".17g"),
            format(row.saved_signed_peak_n, ".17g"),
        ))
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _synthetic_policy_text(source_commit: str) -> tuple[str, str, str]:
    descriptor = "\n".join(
        f"{key}={value}"
        for key, value in _expected_policy_fields(source_commit).items()
    )
    digest = _sha256_bytes(descriptor.encode("utf-8"))
    return descriptor + f"\nsha256={digest}\n", digest, descriptor


def _synthetic_case(
    row: SelectionRow,
    *,
    dataset: DatasetDescriptor,
    dataset_dir: Path,
    policy_sha: str,
    selection_sha: str,
    source_root: str,
    environment_sha: str,
    solver_root: str,
    harness_sha: str,
    b66_sha: str,
) -> dict[str, Any]:
    t_end = 1.234567
    intervals = [
        math.ceil(t_end / (requested * 1e-3))
        for requested in DT_MS
    ]
    actual_ms = [1000 * t_end / item for item in intervals]
    actual_s = [item * 1e-3 for item in actual_ms]
    signed = [-100000.0] * 3
    peak = [0.0] * 3
    fraction = [0.0] * 3
    peak_gci = _gci(peak, actual_s, GATES_N[-1])[1]
    fraction_gci = _gci(fraction, actual_s, FRACTION_GATE)[1]

    metric_dt: list[float] = []
    metric_channel: list[str] = []
    nrmse: list[float] = []
    nmax: list[float] = []
    correlation: list[float] = []
    qoi_rms: list[float] = []
    qoi_peak: list[float] = []
    for channel_index, channel in enumerate(CHANNELS, 1):
        for dt_index, requested in enumerate(DT_MS):
            metric_dt.append(requested)
            metric_channel.append(channel)
            nrmse.append((0.01, 0.005, 0.0)[dt_index])
            nmax.append((0.02, 0.01, 0.0)[dt_index])
            correlation.append((0.999, 0.9995, 1.0)[dt_index])
            qoi_rms.append(float(channel_index))
            qoi_peak.append(float(2 * channel_index))
    qoi_gci = []
    for channel in CHANNELS:
        positions = [
            index for index, name in enumerate(metric_channel)
            if name == channel
        ]
        qoi_gci.append([
            _gci([qoi_rms[index] for index in positions], actual_s, math.inf)[1],
            _gci([qoi_peak[index] for index in positions], actual_s, math.inf)[1],
        ])

    reconstruction = {
        "channel": list(CHANNELS),
        "nrmse_rerun_vs_saved": [0.0] * len(CHANNELS),
        "nmax_rerun_vs_saved": [0.0] * len(CHANNELS),
        "correlation_rerun_vs_saved": [1.0] * len(CHANNELS),
        "max_abs_rerun_vs_saved": [0.0] * len(CHANNELS),
        "max_rel_rerun_vs_saved": [0.0] * len(CHANNELS),
        "max_tolerance_ratio": [0.0] * len(CHANNELS),
        "within_tolerance": [True] * len(CHANNELS),
    }
    classes = [
        [p <= gate and f <= FRACTION_GATE for gate in GATES_N]
        for p, f in zip(peak, fraction)
    ]
    case = {
        "schema": "contact-closure-case-v1",
        "status": "PASS",
        "ordinal": row.ordinal,
        "stage": row.stage,
        "state_index": row.state_index,
        "passage_index": row.passage_index,
        "state_uid": row.state_uid,
        "state_file_sha256": row.state_file_sha256,
        "policy_sha256": policy_sha,
        "selection_sha256": selection_sha,
        "generator_source_root_sha256": source_root,
        "matlab_environment_sha256": environment_sha,
        "started_utc": "2026-07-28T00:00:00Z",
        "completed_utc": "2026-07-28T00:00:01Z",
        "failure_reasons": [],
        "error_identifier": "",
        "error_message": "",
        "study_schema": STUDY_SCHEMA,
        "report_status": "COMPLETED",
        "report_stage": row.stage,
        "report_state_index": row.state_index,
        "report_passage_index": row.passage_index,
        "report_state_uid": row.state_uid,
        "report_state_family": row.state_family,
        "report_state_file_sha256": row.state_file_sha256,
        "report_gen_fingerprint": dataset.fingerprint,
        "report_dataset_dir_sha256": dataset.dataset_dir_sha256,
        "report_generator_source_root_sha256": source_root,
        "report_matlab_environment_sha256": environment_sha,
        "report_harness_sha256": harness_sha,
        "report_b66_sha256": b66_sha,
        "report_solver_execution_root_sha256": solver_root,
        "report_dataset_manifest_root": dataset.content_root,
        "report_case_info_sha256": dataset.case_info,
        "report_damage_states_sha256": dataset.damage_states,
        "report_file_digests_sha256": dataset.file_digests,
        "report_completion_marker_sha256": dataset.complete,
        "report_host_receipt_sha256": dataset.host_receipt,
        "profile_phase_stream_index": 5,
        "profile_phase_seed": 123456789,
        "dt_requested_ms": list(DT_MS),
        "gates_n": list(GATES_N),
        "fraction_gate": FRACTION_GATE,
        "common_dx_m": COMMON_DX_M,
        "reconstruction_rtol": RECON_RTOL,
        "reconstruction_atol": RECON_ATOL,
        "saved_baseline_mode": "direct_raw_samples",
        "direct_reconstruction_pass": True,
        "saved_contact_reconstruction_pass": True,
        "requested_dt_ms": list(DT_MS),
        "actual_dt_ms": actual_ms,
        "t_end_s": [t_end] * 3,
        "n_samples": [item + 1 for item in intervals],
        "peak_contact_signed_N": signed,
        "peak_tension_N": peak,
        "tension_fraction": fraction,
        "contact_lost_track": [0, 0, 0],
        "contact_lost_bridge": [0, 0, 0],
        "saved_contact_log": [0, 0, 0, -100000],
        "rerun_contact_log_1ms": [0, 0, 0, -100000],
        "channel_metrics": {
            "requested_dt_ms": metric_dt,
            "channel": metric_channel,
            "nrmse_vs_finest": nrmse,
            "nmax_vs_finest": nmax,
            "correlation_vs_finest": correlation,
        },
        "channel_qoi": {
            "requested_dt_ms": metric_dt,
            "channel": metric_channel,
            "signal_rms": qoi_rms,
            "signal_abs_peak": qoi_peak,
        },
        "saved_reconstruction": reconstruction,
        "acceptance": {
            "contact_peak_N": peak,
            "contact_fraction": fraction,
            "contact_classification": classes,
            "contact_classification_stable": True,
            "contact_flags_consistent": True,
            "bilateral_tension_present_diagnostic": False,
            "contact_peak_contracts": True,
            "contact_fraction_contracts": True,
            "peak_gci": peak_gci,
            "fraction_gci": fraction_gci,
            "waveform_pass": True,
            "channel_qoi_gci": qoi_gci,
            "channel_qoi_gci_all_pass": True,
        },
    }
    case["report_plain"] = _synthetic_plain_report(
        case,
        row,
        dataset,
        dataset_dir=dataset_dir,
        source_root=source_root,
        environment_sha=environment_sha,
        solver_root=solver_root,
        harness_sha=harness_sha,
        b66_sha=b66_sha,
    )
    return case


def _case_artifact_root(gate_dir: Path) -> str:
    lines: list[str] = []
    for ordinal in range(1, EXPECTED_TOTAL_CASES + 1):
        for extension in ("mat", "json"):
            path = gate_dir / "cases" / f"{ordinal:04d}_case.{extension}"
            lines.append(f"{path.name}:{_sha256_file(path)}")
    return _sha256_bytes("\n".join(lines).encode("utf-8"))


def _synthetic_plain_report(
    case: dict[str, Any],
    row: SelectionRow,
    dataset: DatasetDescriptor,
    *,
    dataset_dir: Path,
    source_root: str,
    environment_sha: str,
    solver_root: str,
    harness_sha: str,
    b66_sha: str,
) -> dict[str, Any]:
    run_table = {
        "requested_dt_ms": case["requested_dt_ms"],
        "actual_dt_ms": case["actual_dt_ms"],
        "t_end_s": case["t_end_s"],
        "n_samples": case["n_samples"],
        "peak_contact_signed_N": case["peak_contact_signed_N"],
        "peak_tension_N": case["peak_tension_N"],
        "tension_fraction": case["tension_fraction"],
        "contact_lost_track": [False, False, False],
        "contact_lost_bridge": [False, False, False],
    }
    for gate in GATES_N:
        run_table[f"pass_gate_{int(gate)}_N"] = [
            peak <= gate and fraction <= FRACTION_GATE
            for peak, fraction in zip(
                case["peak_tension_N"], case["tension_fraction"])
        ]
    solver_paths = [
        str((ROOT / "scour_MATLAB" / f"{module}.m").resolve())
        for module in SOLVER_MODULES
    ]
    solver_hashes = [_sha256_file(Path(path)) for path in solver_paths]
    l_bridge = EXPECTED_L_BRIDGE_M[row.stage]
    return {
        "study_schema": STUDY_SCHEMA,
        "created_utc": "2026-07-28T00:00:00Z",
        "matlab_release": EXPECTED_MATLAB_RELEASE,
        "dataset_dir": str(dataset_dir),
        "state_file": str(dataset_dir / f"{row.state_index:04d}.mat"),
        "state_file_sha256": row.state_file_sha256,
        "dataset_integrity": {
            "status": "VERIFIED",
            "manifest_root": dataset.content_root,
            "state_digest_match": True,
            "marker_match": True,
            "state_table_match": True,
            "identity_table_match": True,
            "case_info_sha256": dataset.case_info,
            "damage_states_sha256": dataset.damage_states,
            "completion_marker_sha256": dataset.complete,
            "file_digests_sha256": dataset.file_digests,
            "qualification_host_receipt_sha256": dataset.host_receipt,
        },
        "stage": row.stage,
        "case_name": f"synthetic-{row.stage}",
        "gen_schema": EXPECTED_GEN_SCHEMA,
        "generation_behavior_version": EXPECTED_GENERATION_BEHAVIOR_VERSION,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "gen_fingerprint": dataset.fingerprint,
        "state_index": row.state_index,
        "passage_index": row.passage_index,
        "passage_selector": str(row.passage_index),
        "state_uid": row.state_uid,
        "state_family": row.state_family,
        "profile_phase_stream_index": 5,
        "profile_phase_seed": case["profile_phase_seed"],
        "dt_requested_ms": list(DT_MS),
        "gates_n": list(GATES_N),
        "fraction_gate": FRACTION_GATE,
        "common_dx_m": COMMON_DX_M,
        "reconstruction_rtol": RECON_RTOL,
        "reconstruction_atol": RECON_ATOL,
        "saved_contact_log": case["saved_contact_log"],
        "saved_gate_pass": [True, True, True],
        "harness_sha256": harness_sha,
        "b66_sha256": b66_sha,
        "solver_source_sha256": {
            "module": list(SOLVER_MODULES),
            "path": solver_paths,
            "sha256": solver_hashes,
        },
        "solver_execution_root_sha256": solver_root,
        "current_generator_source_root_sha256": source_root,
        "current_matlab_environment_sha256": environment_sha,
        "dry_run": False,
        "numeric_hash_selfcheck": NUMERIC_HASH_SELFCHECK,
        "status": "COMPLETED",
        "reference_dt_ms": case["actual_dt_ms"][2],
        "comparison_window_m": [10.0, 10.0 + l_bridge + POST_DECK_WINDOW_M],
        "descriptor": {
            "L_bridge_m": l_bridge,
            "num_spans": 1.0,
            "velocity_kmh": 120.0,
            "temperature_C": 20.0,
            "scour_vector": [0.0, 0.0],
            "bearing_vector_Nm_rad": [0.0, 0.0],
            "crack_row": [0.0, 0.0, 0.0],
            "profile_mode": "fixed",
            "profile_value": 0.0,
            "profile_phase_seed": case["profile_phase_seed"],
            "profile_phase_stream_index": 5,
            "state_uid": row.state_uid,
            "state_family": row.state_family,
            "has_track_eov": False,
            "n_flats": 0.0,
            "n_polygonization": 0.0,
        },
        "run_table": run_table,
        "channel_table": case["channel_metrics"],
        "channel_qoi_table": case["channel_qoi"],
        "saved_baseline_table": case["saved_reconstruction"],
        "saved_baseline_note": "synthetic direct reconstruction",
        "saved_baseline_mode": "direct_raw_samples",
        "direct_reconstruction_pass": True,
        "rerun_contact_log_1ms": case["rerun_contact_log_1ms"],
        "saved_contact_reconstruction_pass": True,
        "signal_common_sha256": [
            _synthetic_hash(row.stage, f"signal_{index}")
            for index in range(3)
        ],
        "contact_peak_delta_vs_finest_N": [0.0, 0.0, 0.0],
        "tension_fraction_delta_vs_finest": [0.0, 0.0, 0.0],
    }


def _build_synthetic_gate(base: Path) -> dict[str, Any]:
    gate_dir = base / "gate"
    gate_dir.mkdir()
    cases_dir = gate_dir / "cases"
    cases_dir.mkdir()
    dataset_dirs = [
        (base / f"dataset_{stage}").resolve()
        for stage in STAGES
    ]
    for stage, path in zip(STAGES, dataset_dirs):
        path.mkdir()
        for state_index in range(1, EXPECTED_STATES[stage] + 1):
            (path / f"{state_index:04d}.mat").write_bytes(
                f"{stage}-state-{state_index}\n".encode("ascii"))
        for name in (
            "case_info.mat",
            "damage_states.mat",
            "file_digests.mat",
            "_GENERATION_COMPLETE",
            "qualification_host_receipt.json",
            "qualification_executed.m",
        ):
            (path / name).write_bytes(
                f"{stage}-{name}\n".encode("ascii"))
    rows = _synthetic_rows()
    descriptors = _synthetic_descriptors(dataset_dirs)
    source_commit = "a" * 40
    policy_text, policy_sha, _ = _synthetic_policy_text(source_commit)
    (gate_dir / "closure_policy.txt").write_bytes(policy_text.encode("utf-8"))
    (gate_dir / "closure_policy.mat").write_bytes(b"synthetic-policy-mat\n")
    selection_text = _synthetic_selection_text(rows, descriptors)
    (gate_dir / "selection_manifest.tsv").write_bytes(
        selection_text.encode("utf-8"))
    selection_sha = _sha256_bytes(selection_text.encode("utf-8"))
    (gate_dir / "selection_manifest.mat").write_bytes(
        b"synthetic-selection-mat\n")

    source_identity = _generator_source_identity()
    _, locked_descriptor, environment_sha = _locked_matlab_environment()
    solver_root, harness_sha, b66_sha = _solver_execution_identity()
    descriptor_by_stage = {
        item.stage: item for item in descriptors
    }
    expected_physical_descriptors: dict[
        tuple[str, int, int], dict[str, Any]
    ] = {}
    first_case: dict[str, Any] | None = None
    for row in rows:
        case = _synthetic_case(
            row,
            dataset=descriptor_by_stage[row.stage],
            dataset_dir=dataset_dirs[STAGES.index(row.stage)],
            policy_sha=policy_sha,
            selection_sha=selection_sha,
            source_root=source_identity.sha256,
            environment_sha=environment_sha,
            solver_root=solver_root,
            harness_sha=harness_sha,
            b66_sha=b66_sha,
        )
        if first_case is None:
            first_case = case
        expected_physical_descriptors[
            (row.stage, row.state_index, row.passage_index)
        ] = json.loads(json.dumps(case["report_plain"]["descriptor"]))
        stem = cases_dir / f"{row.ordinal:04d}_case"
        stem.with_suffix(".mat").write_bytes(
            f"synthetic-case-{row.ordinal}\n".encode("ascii"))
        _write_canonical_json(stem.with_suffix(".json"), case)

    host = {
        "qualification_hostname": "synthetic-host",
        "qualification_cpu_identifier": "synthetic-cpu",
        "qualification_logical_processors": 8,
        "qualification_matlab_max_threads": 8,
        "qualification_computer_arch": "win64",
    }
    summary_datasets: list[dict[str, Any]] = []
    for descriptor, dataset_dir in zip(descriptors, dataset_dirs):
        summary_datasets.append({
            "stage": descriptor.stage,
            "dataset_dir": str(dataset_dir),
            "dataset_dir_sha256": descriptor.dataset_dir_sha256,
            "dataset_content_root_sha256": descriptor.content_root,
            "case_info_sha256": descriptor.case_info,
            "damage_states_sha256": descriptor.damage_states,
            "file_digests_sha256": descriptor.file_digests,
            "completion_marker_sha256": descriptor.complete,
            "qualification_host_receipt_sha256": descriptor.host_receipt,
            "gen_fingerprint": descriptor.fingerprint,
            "qualification_source_sha256": descriptor.qual_source,
            "qualification_executed_file_sha256": descriptor.qual_executed,
            "qualification_host_diagnostic_sha256": descriptor.host_diagnostic,
            **host,
        })
    closure_host_descriptor = "\n".join((
        "schema=contact-closure-host-attestation-v1",
        "declared_host_id=synthetic-host",
        "hostname=synthetic-host",
        "cpu_identifier=synthetic-cpu",
        "logical_processors=8",
        "matlab_max_threads=8",
        "computer_arch=win64",
        f"matlab_environment_sha256={environment_sha}",
    ))
    closure_host = {
        "schema": "contact-closure-host-attestation-v1",
        "declared_host_id": "synthetic-host",
        "hostname": "synthetic-host",
        "cpu_identifier": "synthetic-cpu",
        "logical_processors": 8,
        "matlab_max_threads": 8,
        "computer_arch": "win64",
        "matlab_environment_sha256": environment_sha,
        "canonical_descriptor": closure_host_descriptor,
        "sha256": _sha256_bytes(closure_host_descriptor.encode("utf-8")),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "policy_sha256": policy_sha,
        "selection_sha256": selection_sha,
        "expected_cases": EXPECTED_TOTAL_CASES,
        "completed_cases": EXPECTED_TOTAL_CASES,
        "pass_cases": EXPECTED_TOTAL_CASES,
        "fail_cases": 0,
        "error_cases": 0,
        "case_artifact_root_sha256": _case_artifact_root(gate_dir),
        "declared_host_id": "synthetic-host",
        "closure_host_attestation": closure_host,
        "generator_source_root_sha256": source_identity.sha256,
        "generator_source_digest_lines": source_identity.digest_lines,
        "generator_source_file_count": source_identity.file_count,
        "matlab_environment_sha256": environment_sha,
        "matlab_environment_descriptor": locked_descriptor,
        "matlab_release": EXPECTED_MATLAB_RELEASE,
        "gate_execution_root_sha256": _gate_execution_root(),
        "datasets": summary_datasets,
        "started_utc": "2026-07-28T00:00:00Z",
        "completed_utc": "2026-07-28T00:10:00Z",
    }
    (gate_dir / "gate_summary.mat").write_bytes(b"synthetic-summary-mat\n")
    _write_canonical_json(gate_dir / "gate_summary.json", summary)
    (gate_dir / "GATE_STATUS.txt").write_bytes(
        (
            f"PASS\nexpected={EXPECTED_TOTAL_CASES}\n"
            f"completed={EXPECTED_TOTAL_CASES}\n"
            f"pass={EXPECTED_TOTAL_CASES}\nfail=0\nerror=0\n"
        ).encode("ascii")
    )
    assert first_case is not None
    return {
        "gate_dir": gate_dir,
        "source_commit": source_commit,
        "rows": rows,
        "descriptors": descriptors,
        "policy_sha": policy_sha,
        "selection_sha": selection_sha,
        "source_root": source_identity.sha256,
        "environment_sha": environment_sha,
        "solver_root": solver_root,
        "harness_sha": harness_sha,
        "b66_sha": b66_sha,
        "first_case": first_case,
        "expected_physical_descriptors": expected_physical_descriptors,
    }
