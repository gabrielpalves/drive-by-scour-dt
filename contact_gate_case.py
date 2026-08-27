"""Per-case scientific evidence recomputation and source identities."""
from __future__ import annotations

import math
from typing import Any

from core.source_provenance import SourceProvenanceError, generator_source_root
from contact_gate_path_safety import GateError
from contact_gate_source_contract import (
    GATE_SOURCE_FILES,
    SOLVER_MODULES,
    STUDY_HARNESS_FILES,
    matlab_source_path,
)
from contact_gate_core import (
    CHANNELS,
    COARSE_CORR,
    COARSE_NMAX,
    COARSE_NRMSE,
    COMMON_DX_M,
    DT_MS,
    DatasetDescriptor,
    EQUIVALENCE_ATOL,
    EQUIVALENCE_RTOL,
    FINEST_IDENTITY_ATOL,
    FRACTION_GATE,
    GATES_N,
    MEDIUM_CORR,
    MEDIUM_NMAX,
    MEDIUM_NRMSE,
    RECON_ATOL,
    RECON_RTOL,
    ROOT,
    SelectionRow,
    STUDY_SCHEMA,
    TIME_GRID_ULPS,
    WAVEFORM_MONOTONIC_ATOL,
    _allclose,
    _as_list,
    _exact_keys,
    _float_list,
    _same_float_list,
    _sha256_bytes,
    _sha256_file,
    _strict_integer,
    _strict_number,
    _validate_utc_pair,
)
from contact_gate_numerics import (
    _contracts,
    _gci,
    _metric_columns,
    _validate_plain_report,
    _validate_public_gci,
)

def _study_harness_root() -> str:
    """SHA-256 root of the STUDY EXECUTABLE SET.

    Mirrors local_study_harness_root in
    ``scour_MATLAB/contact_closure_common.m`` byte-for-byte: LF-joined,
    lexicographically sorted ``<name>:<sha256>`` lines with no terminal LF,
    hashed as UTF-8.  The MATLAB study writes this value into
    ``report.harness_sha256``, the gate re-verifies it during report binding,
    and this independent recomputation must agree with both.
    """
    lines = [
        f"{name}:{_sha256_file(matlab_source_path(name))}"
        for name in STUDY_HARNESS_FILES
    ]
    return _sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))


def _gate_execution_root() -> str:
    """SHA-256 root of the GATE EXECUTABLE SET.

    Mirrors local_gate_execution_root in
    ``scour_MATLAB/contact_closure_common.m`` under the identical grammar used
    by :func:`_study_harness_root`.  The gate writes its value into
    ``gate_summary.json``; this independent recomputation must agree, which is
    what binds the modules that DECIDE acceptance to the reviewed bytes rather
    than to whatever MATLAB happened to resolve on the operator's path.
    """
    lines = [
        f"{name}:{_sha256_file(matlab_source_path(name))}"
        for name in GATE_SOURCE_FILES
    ]
    return _sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))


def _solver_execution_identity() -> tuple[str, str, str]:
    source_dir = ROOT / "scour_MATLAB"
    lines: list[str] = []
    for module in SOLVER_MODULES:
        path = matlab_source_path(f"{module}.m")
        lines.append(f"{module}:{_sha256_file(path)}")
    root = _sha256_bytes("\n".join(lines).encode("utf-8"))
    return (
        root,
        _study_harness_root(),
        _sha256_file(source_dir / "B66_ContactForce.m"),
    )


def _generator_source_identity() -> Any:
    try:
        return generator_source_root(ROOT)
    except SourceProvenanceError as exc:
        raise GateError(f"generator source provenance is invalid: {exc}") from exc


def _recompute_case(
    case: dict[str, Any],
    row: SelectionRow,
    *,
    dataset: DatasetDescriptor,
    policy_sha: str,
    selection_sha: str,
    source_root: str,
    environment_sha: str,
    solver_root: str,
    harness_sha: str,
    b66_sha: str,
    expected_descriptor: dict[str, Any] | None = None,
) -> None:
    identity_fields = {
        "schema", "status", "ordinal", "stage", "state_index",
        "passage_index", "state_uid", "state_file_sha256", "policy_sha256",
        "selection_sha256", "generator_source_root_sha256",
        "matlab_environment_sha256", "started_utc", "completed_utc",
        "failure_reasons", "error_identifier", "error_message",
    }
    report_fields = {
        "study_schema", "report_status", "report_stage",
        "report_state_index", "report_passage_index", "report_state_uid",
        "report_state_family", "report_state_file_sha256",
        "report_gen_fingerprint", "report_dataset_dir_sha256",
        "report_generator_source_root_sha256",
        "report_matlab_environment_sha256", "report_harness_sha256",
        "report_b66_sha256", "report_solver_execution_root_sha256",
        "report_dataset_manifest_root", "report_case_info_sha256",
        "report_damage_states_sha256", "report_file_digests_sha256",
        "report_completion_marker_sha256", "report_host_receipt_sha256",
        "profile_phase_stream_index", "profile_phase_seed",
        "dt_requested_ms", "gates_n", "fraction_gate", "common_dx_m",
        "reconstruction_rtol", "reconstruction_atol",
        "saved_baseline_mode", "direct_reconstruction_pass",
        "saved_contact_reconstruction_pass", "requested_dt_ms",
        "actual_dt_ms", "t_end_s", "n_samples",
        "peak_contact_signed_N", "peak_tension_N", "tension_fraction",
        "contact_lost_track", "contact_lost_bridge", "saved_contact_log",
        "rerun_contact_log_1ms", "channel_metrics", "channel_qoi",
        "saved_reconstruction", "report_plain", "acceptance",
    }
    _exact_keys(case, identity_fields | report_fields, f"case {row.ordinal}")
    expected_identity = {
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
    }
    for key, expected in expected_identity.items():
        if case[key] != expected:
            raise GateError(
                f"case {row.ordinal} identity {key}={case[key]!r}, "
                f"expected {expected!r}"
            )
    if (
        type(case["ordinal"]) is not int
        or type(case["state_index"]) is not int
        or type(case["passage_index"]) is not int
        or type(case["report_state_index"]) is not int
        or type(case["report_passage_index"]) is not int
        or any(
            not isinstance(case[key], str)
            for key in (
                "schema", "status", "stage", "state_uid",
                "state_file_sha256", "policy_sha256", "selection_sha256",
                "generator_source_root_sha256", "matlab_environment_sha256",
                "study_schema", "report_status", "report_stage",
                "report_state_uid", "report_state_family",
                "report_state_file_sha256", "report_gen_fingerprint",
                "report_dataset_dir_sha256",
                "report_generator_source_root_sha256",
                "report_matlab_environment_sha256", "report_harness_sha256",
                "report_b66_sha256", "report_solver_execution_root_sha256",
                "report_dataset_manifest_root", "report_case_info_sha256",
                "report_damage_states_sha256", "report_file_digests_sha256",
                "report_completion_marker_sha256",
                "report_host_receipt_sha256",
            )
        )
    ):
        raise GateError(f"case {row.ordinal} identity types differ")
    _validate_utc_pair(
        case["started_utc"], case["completed_utc"], f"case {row.ordinal}")
    if (
        case["failure_reasons"] != []
        or case["error_identifier"] != ""
        or case["error_message"] != ""
    ):
        raise GateError(f"case {row.ordinal} PASS carries failure/error data")
    if _strict_integer(
        case["profile_phase_stream_index"],
        f"case {row.ordinal} profile-phase stream index",
    ) != 5:
        raise GateError(f"case {row.ordinal} profile-phase stream index is not 5")
    seed = case["profile_phase_seed"]
    seed_integer = _strict_integer(
        seed, f"case {row.ordinal} named phase seed")
    if not 1 <= seed_integer <= 2**32 - 1:
        raise GateError(f"case {row.ordinal} has invalid named phase seed")
    if (
        case["direct_reconstruction_pass"] is not True
        or case["saved_contact_reconstruction_pass"] is not True
        or case["saved_baseline_mode"] != "direct_raw_samples"
    ):
        raise GateError(f"case {row.ordinal} failed direct raw reconstruction")
    if (
        _strict_number(
            case["reconstruction_rtol"], "reconstruction_rtol") != RECON_RTOL
        or _strict_number(
            case["reconstruction_atol"], "reconstruction_atol") != RECON_ATOL
        or not _same_float_list(
            _float_list(case["dt_requested_ms"], count=3, label="dt policy"),
            DT_MS,
        )
        or not _same_float_list(
            _float_list(case["gates_n"], count=3, label="gate policy"),
            GATES_N,
        )
        or _strict_number(case["fraction_gate"], "fraction_gate")
        != FRACTION_GATE
        or _strict_number(case["common_dx_m"], "common_dx_m")
        != COMMON_DX_M
    ):
        raise GateError(f"case {row.ordinal} report policy drift")

    requested = _float_list(
        case["requested_dt_ms"], count=3, label="requested_dt_ms")
    actual_ms = _float_list(
        case["actual_dt_ms"], count=3, label="actual_dt_ms")
    t_end = _float_list(case["t_end_s"], count=3, label="t_end_s")
    samples_raw = _as_list(case["n_samples"])
    if len(samples_raw) != 3 or any(
        type(item) not in {int, float}
        or not math.isfinite(float(item))
        or int(item) != item
        or int(item) < 2
        for item in samples_raw
    ):
        raise GateError(f"case {row.ordinal} malformed sample counts")
    samples = [int(item) for item in samples_raw]
    if requested != list(DT_MS) or not actual_ms[0] > actual_ms[1] > actual_ms[2]:
        raise GateError(f"case {row.ordinal} time-step order differs")
    if max(abs(item - t_end[0]) for item in t_end) > (
        TIME_GRID_ULPS * math.ulp(max(1.0, *(abs(item) for item in t_end)))
    ):
        raise GateError(f"case {row.ordinal} t_end changes across refinements")
    for req_ms, observed_ms, end_s, count in zip(
        requested, actual_ms, t_end, samples
    ):
        expected_intervals = math.ceil(end_s / (req_ms * 1e-3))
        reconstructed_ms = 1000 * end_s / (count - 1)
        if (
            count - 1 != expected_intervals
            or abs(observed_ms - reconstructed_ms)
            > TIME_GRID_ULPS * math.ulp(max(abs(reconstructed_ms), 1.0))
            or observed_ms <= 0
            or observed_ms
            > req_ms + TIME_GRID_ULPS * math.ulp(max(abs(req_ms), 1.0))
        ):
            raise GateError(f"case {row.ordinal} violates B11 time-grid identity")
    actual_s = [item * 1e-3 for item in actual_ms]

    signed = _float_list(
        case["peak_contact_signed_N"], count=3, label="signed peak")
    peak = _float_list(case["peak_tension_N"], count=3, label="peak")
    fraction = _float_list(
        case["tension_fraction"], count=3, label="fraction")
    if any(item < 0 for item in peak) or any(
        not 0 <= item <= 1 for item in fraction
    ):
        raise GateError(f"case {row.ordinal} contact metric outside domain")
    for observed, signed_value in zip(peak, signed):
        if not _allclose(
            observed,
            max(0.0, signed_value),
            rtol=EQUIVALENCE_RTOL,
            atol=EQUIVALENCE_ATOL,
        ):
            raise GateError(f"case {row.ordinal} signed/derived peak mismatch")
    track = _as_list(case["contact_lost_track"])
    bridge = _as_list(case["contact_lost_bridge"])
    if len(track) != 3 or len(bridge) != 3 or any(
        type(item) not in {int, float} or item not in (0, 1)
        for item in track + bridge
    ):
        raise GateError(f"case {row.ordinal} contact flags are not exact 0/1")
    if any(
        bool(t) != (s > 0) or (bool(b) and not bool(t))
        for t, b, s in zip(track, bridge, signed)
    ):
        raise GateError(f"case {row.ordinal} contact flag algebra differs")

    frozen = [
        row.saved_bridge_flag, row.saved_track_flag,
        row.saved_fraction, row.saved_signed_peak_n,
    ]
    saved = _float_list(
        case["saved_contact_log"], count=4, label="saved contact")
    rerun = _float_list(
        case["rerun_contact_log_1ms"], count=4, label="rerun contact")
    for observed in (saved, rerun):
        if observed[:2] != frozen[:2] or any(
            not _allclose(
                a, b, rtol=RECON_RTOL, atol=RECON_ATOL
            )
            for a, b in zip(observed[2:], frozen[2:])
        ):
            raise GateError(
                f"case {row.ordinal} contact vector not frozen/reconstructed"
            )
    if rerun[:2] != [bridge[0], track[0]] or any(
        not _allclose(a, b, rtol=RECON_RTOL, atol=RECON_ATOL)
        for a, b in zip(rerun[2:], [fraction[0], signed[0]])
    ):
        raise GateError(f"case {row.ordinal} 1-ms report row is not reconstructed")
    if any(item > GATES_N[-1] for item in peak) or any(
        item > FRACTION_GATE for item in fraction
    ):
        raise GateError(f"case {row.ordinal} exceeds registered bilateral envelope")
    classes = [
        tuple(p <= gate and f <= FRACTION_GATE for gate in GATES_N)
        for p, f in zip(peak, fraction)
    ]
    if not classes[0] == classes[1] == classes[2]:
        raise GateError(f"case {row.ordinal} contact classification changes")
    if not _contracts(peak) or not _contracts(fraction):
        raise GateError(f"case {row.ordinal} contact metric does not contract")
    peak_gci_pass, peak_gci = _gci(peak, actual_s, GATES_N[-1])
    fraction_gci_pass, fraction_gci = _gci(
        fraction, actual_s, FRACTION_GATE)
    if not peak_gci_pass or not fraction_gci_pass:
        raise GateError(f"case {row.ordinal} actual-step GCI bound fails")

    metrics = _metric_columns(case["channel_metrics"], "channel_metrics")
    metric_fields = {
        "requested_dt_ms", "channel", "nrmse_vs_finest",
        "nmax_vs_finest", "correlation_vs_finest",
    }
    channel_table_rows = len(DT_MS) * len(CHANNELS)
    if set(metrics) != metric_fields or any(
        len(metrics[key]) != channel_table_rows for key in metric_fields
    ):
        raise GateError(f"case {row.ordinal} malformed channel table")
    dt_column = [
        _strict_number(item, f"channel requested_dt_ms[{index}]")
        for index, item in enumerate(metrics["requested_dt_ms"])
    ]
    names = [str(item) for item in metrics["channel"]]
    if any(not isinstance(item, str) for item in metrics["channel"]):
        raise GateError(f"case {row.ordinal} channel names are not JSON strings")
    nrmse = [
        _strict_number(item, f"channel NRMSE[{index}]")
        for index, item in enumerate(metrics["nrmse_vs_finest"])
    ]
    nmax = [
        _strict_number(item, f"channel NMAX[{index}]")
        for index, item in enumerate(metrics["nmax_vs_finest"])
    ]
    corr = [
        _strict_number(item, f"channel correlation[{index}]")
        for index, item in enumerate(metrics["correlation_vs_finest"])
    ]
    if (
        not all(math.isfinite(item) for item in nrmse + nmax + corr)
        or any(item < 0 for item in nrmse + nmax)
        or any(not -1 <= item <= 1 for item in corr)
    ):
        raise GateError(f"case {row.ordinal} invalid channel metric domain")
    for channel in CHANNELS:
        positions = [i for i, name in enumerate(names) if name == channel]
        if len(positions) != 3 \
                or [dt_column[i] for i in positions] != list(DT_MS):
            raise GateError(f"case {row.ordinal} channel inventory differs")
        coarse, medium, finest = positions
        if (
            nrmse[coarse] > COARSE_NRMSE
            or nmax[coarse] > COARSE_NMAX
            or corr[coarse] < COARSE_CORR
            or nrmse[medium] > MEDIUM_NRMSE
            or nmax[medium] > MEDIUM_NMAX
            or corr[medium] < MEDIUM_CORR
            or nrmse[medium] > nrmse[coarse] + WAVEFORM_MONOTONIC_ATOL
            or nmax[medium] > nmax[coarse] + WAVEFORM_MONOTONIC_ATOL
            or nrmse[finest] > FINEST_IDENTITY_ATOL
            or nmax[finest] > FINEST_IDENTITY_ATOL
            or abs(corr[finest] - 1) > FINEST_IDENTITY_ATOL
        ):
            raise GateError(
                f"case {row.ordinal} waveform gate failed for {channel}"
            )
    if set(names) != set(CHANNELS):
        raise GateError(f"case {row.ordinal} has an extra/missing channel")

    qoi = _metric_columns(case["channel_qoi"], "channel_qoi")
    qoi_fields = {"requested_dt_ms", "channel", "signal_rms", "signal_abs_peak"}
    if set(qoi) != qoi_fields or any(
        len(qoi[key]) != channel_table_rows for key in qoi_fields
    ):
        raise GateError(f"case {row.ordinal} malformed channel QOI table")
    if any(not isinstance(item, str) for item in qoi["channel"]):
        raise GateError(f"case {row.ordinal} QOI channel names are not strings")
    qoi_names = list(qoi["channel"])
    qoi_dt = [
        _strict_number(item, f"QOI requested_dt_ms[{index}]")
        for index, item in enumerate(qoi["requested_dt_ms"])
    ]
    qoi_rms = [
        _strict_number(item, f"QOI RMS[{index}]")
        for index, item in enumerate(qoi["signal_rms"])
    ]
    qoi_peak = [
        _strict_number(item, f"QOI peak[{index}]")
        for index, item in enumerate(qoi["signal_abs_peak"])
    ]
    qoi_values = qoi_rms + qoi_peak
    if not all(math.isfinite(item) and item >= 0 for item in qoi_values):
        raise GateError(f"case {row.ordinal} invalid channel QOI domain")
    qoi_gci_expected: list[list[dict[str, Any]]] = []
    qoi_gci_all_pass = True
    for channel in CHANNELS:
        positions = [i for i, name in enumerate(qoi_names) if name == channel]
        if len(positions) != 3 or [qoi_dt[i] for i in positions] != list(DT_MS):
            raise GateError(f"case {row.ordinal} QOI inventory differs")
        rms_pass, rms_gci = _gci(
            [qoi_rms[i] for i in positions], actual_s, math.inf)
        peak_pass, peak_qoi_gci = _gci(
            [qoi_peak[i] for i in positions], actual_s, math.inf)
        qoi_gci_all_pass = qoi_gci_all_pass and rms_pass and peak_pass
        qoi_gci_expected.append([rms_gci, peak_qoi_gci])
    if not qoi_gci_all_pass:
        raise GateError(f"case {row.ordinal} channel QOI GCI gate fails")

    reconstruction = _metric_columns(
        case["saved_reconstruction"], "saved_reconstruction")
    recon_fields = {
        "channel", "nrmse_rerun_vs_saved", "nmax_rerun_vs_saved",
        "correlation_rerun_vs_saved", "max_abs_rerun_vs_saved",
        "max_rel_rerun_vs_saved", "max_tolerance_ratio",
        "within_tolerance",
    }
    if set(reconstruction) != recon_fields or any(
        len(reconstruction[key]) != len(CHANNELS) for key in recon_fields
    ) or [str(item) for item in reconstruction["channel"]] != list(CHANNELS):
        raise GateError(f"case {row.ordinal} malformed reconstruction table")
    recon_numeric = {
        key: [
            _strict_number(item, f"reconstruction {key}[{index}]")
            for index, item in enumerate(reconstruction[key])
        ]
        for key in recon_fields - {"channel", "within_tolerance"}
    }
    if (
        not all(
            math.isfinite(item)
            for values in recon_numeric.values()
            for item in values
        )
        or any(
            item < 0
            for key, values in recon_numeric.items()
            if key != "correlation_rerun_vs_saved"
            for item in values
        )
        or any(
            not -1 <= item <= 1
            for item in recon_numeric["correlation_rerun_vs_saved"]
        )
        or any(item > 1 for item in recon_numeric["max_tolerance_ratio"])
        or any(item is not True for item in reconstruction["within_tolerance"])
    ):
        raise GateError(f"case {row.ordinal} reconstruction evidence failed")

    acceptance = _exact_keys(
        case["acceptance"],
        {
            "contact_peak_N", "contact_fraction", "contact_classification",
            "contact_classification_stable", "contact_flags_consistent",
            "bilateral_tension_present_diagnostic", "contact_peak_contracts",
            "contact_fraction_contracts", "peak_gci", "fraction_gci",
            "waveform_pass", "channel_qoi_gci",
            "channel_qoi_gci_all_pass",
        },
        f"case {row.ordinal} acceptance",
    )
    for key in (
        "contact_classification_stable", "contact_flags_consistent",
        "contact_peak_contracts", "contact_fraction_contracts",
        "waveform_pass", "channel_qoi_gci_all_pass",
    ):
        if acceptance[key] is not True:
            raise GateError(f"case {row.ordinal} acceptance {key} is not true")
    if acceptance["bilateral_tension_present_diagnostic"] is not any(
        bool(item) for item in track
    ):
        raise GateError(f"case {row.ordinal} tension diagnostic differs")
    public_classes = acceptance["contact_classification"]
    if (
        not isinstance(public_classes, list)
        or len(public_classes) != 3
        or any(
            not isinstance(class_row, list)
            or len(class_row) != 3
            or any(type(item) is not bool for item in class_row)
            for class_row in public_classes
        )
    ):
        raise GateError(
            f"case {row.ordinal} contact classification is not bool[3][3]")
    if (
        _float_list(
            acceptance["contact_peak_N"], count=3,
            label="acceptance contact peak",
        ) != peak
        or _float_list(
            acceptance["contact_fraction"], count=3,
            label="acceptance contact fraction",
        ) != fraction
        or public_classes != [
            list(item) for item in classes
        ]
    ):
        raise GateError(f"case {row.ordinal} projected acceptance metrics differ")
    _validate_public_gci(
        acceptance["peak_gci"], peak_gci,
        f"case {row.ordinal} acceptance peak GCI",
    )
    _validate_public_gci(
        acceptance["fraction_gci"], fraction_gci,
        f"case {row.ordinal} acceptance fraction GCI",
    )
    public_qoi_gci = acceptance["channel_qoi_gci"]
    if (
        not isinstance(public_qoi_gci, list)
        or len(public_qoi_gci) != len(CHANNELS)
    ):
        raise GateError(f"case {row.ordinal} acceptance QOI GCI shape differs")
    for channel_index, (observed_row, expected_row) in enumerate(
        zip(public_qoi_gci, qoi_gci_expected)
    ):
        if not isinstance(observed_row, list) or len(observed_row) != 2:
            raise GateError(
                f"case {row.ordinal} acceptance QOI GCI shape differs")
        for qoi_index, (observed, expected) in enumerate(
            zip(observed_row, expected_row)
        ):
            _validate_public_gci(
                observed,
                expected,
                f"case {row.ordinal} acceptance QOI GCI "
                f"{CHANNELS[channel_index]}[{qoi_index}]",
            )
    _validate_plain_report(
        case["report_plain"],
        case,
        row,
        dataset,
        source_root=source_root,
        environment_sha=environment_sha,
        solver_root=solver_root,
        harness_sha=harness_sha,
        b66_sha=b66_sha,
        expected_descriptor=expected_descriptor,
    )
