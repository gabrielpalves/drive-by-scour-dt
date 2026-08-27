"""Numerical convergence and public-report recomputation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from contact_gate_path_safety import GateError
from contact_gate_source_contract import SOLVER_MODULES, matlab_source_path
from contact_gate_core import (
    CHANNEL_SCHEMA_ID,
    CHANNELS,
    COARSE_CORR,
    COARSE_NMAX,
    COARSE_NRMSE,
    COMMON_DX_M,
    COMPARISON_WINDOW_ATOL,
    DT_MS,
    DatasetDescriptor,
    EQUIVALENCE_ATOL,
    EQUIVALENCE_RTOL,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_L_BRIDGE_M,
    EXPECTED_MATLAB_RELEASE,
    FINEST_IDENTITY_ATOL,
    FRACTION_GATE,
    GATES_N,
    GCI_FS,
    GCI_P_MAX,
    GCI_P_MIN,
    MEDIUM_CORR,
    MEDIUM_NMAX,
    MEDIUM_NRMSE,
    NUMERIC_HASH_SELFCHECK,
    PLAIN_REPORT_FIELDS,
    POST_DECK_WINDOW_M,
    RECON_ATOL,
    RECON_RTOL,
    ROOT,
    SHA256_RE,
    SelectionRow,
    STUDY_SCHEMA,
    TIME_GRID_ULPS,
    WAVEFORM_MONOTONIC_ATOL,
    _allclose,
    _as_list,
    _exact_keys,
    _float_list,
    _sha256_bytes,
    _strict_integer,
    _strict_json_equivalent,
    _strict_number,
    _validate_utc_pair,
)

def _contracts(phi: list[float]) -> bool:
    scale = max(1.0, *(abs(item) for item in phi))
    tol = EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * scale
    return abs(phi[1] - phi[2]) <= abs(phi[0] - phi[2]) + tol


def _gci(
    phi: list[float],
    actual_step_s: list[float],
    limit: float,
) -> tuple[bool, dict[str, Any]]:
    result: dict[str, Any] = {
        "status": "INVALID",
        "observed_order": None,
        "extrapolated": None,
        "fine_uncertainty": None,
        "upper_bound": None,
        "actual_step_s": list(actual_step_s),
        "coarse_medium_ratio": None,
        "medium_fine_ratio": None,
    }
    if (
        len(phi) != 3
        or len(actual_step_s) != 3
        or not all(math.isfinite(item) for item in phi + actual_step_s)
        or any(item <= 0 for item in actual_step_s)
        or not actual_step_s[0] > actual_step_s[1] > actual_step_s[2]
    ):
        return False, result
    result["coarse_medium_ratio"] = actual_step_s[0] / actual_step_s[1]
    result["medium_fine_ratio"] = actual_step_s[1] / actual_step_s[2]
    scale = max(1.0, *(abs(item) for item in phi))
    tol = EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * scale
    e32 = phi[0] - phi[1]
    e21 = phi[1] - phi[2]
    if abs(e32) <= tol and abs(e21) <= tol:
        upper = max(0.0, phi[2])
        result.update({
            "status": "EQUIVALENT",
            "extrapolated": phi[2],
            "fine_uncertainty": 0.0,
            "upper_bound": upper,
        })
        return upper <= limit, result
    if abs(e21) <= tol and abs(e32) > tol:
        upper = max(0.0, phi[2])
        result.update({
            "status": "FINE_PAIR_EQUIVALENT",
            "extrapolated": phi[2],
            "fine_uncertainty": 0.0,
            "upper_bound": upper,
        })
        return upper <= limit, result
    if abs(e32) <= tol or e32 * e21 <= 0:
        result["status"] = "OSCILLATORY_OR_STALLED"
        return False, result
    observed_ratio = abs(e32 / e21)
    r31 = actual_step_s[0] / actual_step_s[2]
    r21 = actual_step_s[1] / actual_step_s[2]

    def residual(order: float) -> float:
        model = (
            (r31**order - r21**order)
            / (r21**order - 1.0)
        )
        return model - observed_ratio

    lo, hi = GCI_P_MIN, GCI_P_MAX
    flo, fhi = residual(lo), residual(hi)
    if (
        not math.isfinite(flo)
        or not math.isfinite(fhi)
        or flo > 0
        or fhi < 0
    ):
        result["status"] = "NO_POSITIVE_ORDER"
        return False, result
    for _ in range(200):
        middle = (lo + hi) / 2
        fm = residual(middle)
        if fm < 0:
            lo = middle
        else:
            hi = middle
    p = (lo + hi) / 2
    if not math.isfinite(p) or not GCI_P_MIN <= p <= GCI_P_MAX:
        result["status"] = "ORDER_SOLVE_FAILED"
        return False, result
    denominator = r21**p - 1
    extrapolated = (r21**p * phi[2] - phi[1]) / denominator
    uncertainty = GCI_FS * abs(phi[2] - phi[1]) / denominator
    upper = max(0.0, extrapolated) + uncertainty
    result.update({
        "status": "MONOTONIC",
        "observed_order": p,
        "extrapolated": extrapolated,
        "fine_uncertainty": uncertainty,
        "upper_bound": upper,
    })
    return upper <= limit, result


def _metric_columns(block: Any, label: str) -> dict[str, list[Any]]:
    if not isinstance(block, dict):
        raise GateError(f"{label} is not an object")
    result: dict[str, list[Any]] = {}
    for key, value in block.items():
        result[key] = _as_list(value)
    return result


def _validate_public_gci(
    value: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    fields = {
        "status", "observed_order", "extrapolated", "fine_uncertainty",
        "upper_bound", "actual_step_s", "coarse_medium_ratio",
        "medium_fine_ratio",
    }
    observed = _exact_keys(value, fields, label)
    if observed["status"] != expected["status"]:
        raise GateError(f"{label} status differs")
    for key in (
        "observed_order", "extrapolated", "fine_uncertainty",
        "upper_bound", "coarse_medium_ratio", "medium_fine_ratio",
    ):
        actual = observed[key]
        wanted = expected[key]
        if wanted is None:
            if actual is not None:
                raise GateError(f"{label} {key} must be JSON null")
        elif type(actual) not in {int, float} or not _allclose(
            float(actual), float(wanted),
            rtol=1e-9, atol=1e-12,
        ):
            raise GateError(f"{label} {key} differs")
    steps = _float_list(
        observed["actual_step_s"], count=3, label=f"{label} actual_step_s")
    if any(
        not _allclose(a, b, rtol=1e-12, atol=1e-15)
        for a, b in zip(steps, expected["actual_step_s"])
    ):
        raise GateError(f"{label} actual steps differ")


def _validate_plain_report(
    report: Any,
    case: dict[str, Any],
    row: SelectionRow,
    dataset: DatasetDescriptor,
    *,
    source_root: str,
    environment_sha: str,
    solver_root: str,
    harness_sha: str,
    b66_sha: str,
    expected_descriptor: dict[str, Any] | None = None,
) -> None:
    data = _exact_keys(
        report, PLAIN_REPORT_FIELDS, f"case {row.ordinal} plain report")
    scalar_expected = {
        "study_schema": STUDY_SCHEMA,
        "matlab_release": EXPECTED_MATLAB_RELEASE,
        "state_file_sha256": row.state_file_sha256,
        "stage": row.stage,
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
        "fraction_gate": FRACTION_GATE,
        "common_dx_m": COMMON_DX_M,
        "reconstruction_rtol": RECON_RTOL,
        "reconstruction_atol": RECON_ATOL,
        "harness_sha256": harness_sha,
        "b66_sha256": b66_sha,
        "solver_execution_root_sha256": solver_root,
        "current_generator_source_root_sha256": source_root,
        "current_matlab_environment_sha256": environment_sha,
        "dry_run": False,
        "status": "COMPLETED",
        "saved_baseline_mode": "direct_raw_samples",
        "direct_reconstruction_pass": True,
        "saved_contact_reconstruction_pass": True,
    }
    for key, expected in scalar_expected.items():
        if not _strict_json_equivalent(data[key], expected):
            raise GateError(
                f"case {row.ordinal} plain report {key} differs")
    _validate_utc_pair(
        data["created_utc"], data["created_utc"],
        f"case {row.ordinal} plain report creation")
    dataset_dir = data["dataset_dir"]
    state_file = data["state_file"]
    if (
        not isinstance(dataset_dir, str)
        or not isinstance(state_file, str)
        or _sha256_bytes(dataset_dir.encode("utf-8"))
        != dataset.dataset_dir_sha256
        or Path(state_file).name != f"{row.state_index:04d}.mat"
        or str(Path(state_file).parent) != str(Path(dataset_dir))
    ):
        raise GateError(f"case {row.ordinal} plain report paths differ")
    for key, expected in (
        ("dt_requested_ms", list(DT_MS)),
        ("gates_n", list(GATES_N)),
        ("saved_contact_log", case["saved_contact_log"]),
        ("rerun_contact_log_1ms", case["rerun_contact_log_1ms"]),
    ):
        if not _strict_json_equivalent(data[key], expected):
            raise GateError(f"case {row.ordinal} plain report {key} differs")
    if type(data["saved_gate_pass"]) is not list \
            or any(type(item) is not bool for item in data["saved_gate_pass"]):
        raise GateError(f"case {row.ordinal} saved gate classification malformed")
    saved_peak = max(0.0, row.saved_signed_peak_n)
    expected_saved_gate = [
        saved_peak <= gate and row.saved_fraction <= FRACTION_GATE
        for gate in GATES_N
    ]
    if data["saved_gate_pass"] != expected_saved_gate:
        raise GateError(f"case {row.ordinal} saved gate classification differs")

    run = _exact_keys(
        data["run_table"],
        {
            "requested_dt_ms", "actual_dt_ms", "t_end_s", "n_samples",
            "peak_contact_signed_N", "peak_tension_N", "tension_fraction",
            "contact_lost_track", "contact_lost_bridge",
            "pass_gate_0_N", "pass_gate_12000_N", "pass_gate_24000_N",
        },
        f"case {row.ordinal} plain run table",
    )
    run_projection = {
        "requested_dt_ms": case["requested_dt_ms"],
        "actual_dt_ms": case["actual_dt_ms"],
        "t_end_s": case["t_end_s"],
        "n_samples": case["n_samples"],
        "peak_contact_signed_N": case["peak_contact_signed_N"],
        "peak_tension_N": case["peak_tension_N"],
        "tension_fraction": case["tension_fraction"],
        "contact_lost_track": [
            bool(item) for item in case["contact_lost_track"]
        ],
        "contact_lost_bridge": [
            bool(item) for item in case["contact_lost_bridge"]
        ],
    }
    for key, expected in run_projection.items():
        if not _strict_json_equivalent(run[key], expected):
            raise GateError(
                f"case {row.ordinal} plain run table {key} differs")
    for gate_index, gate in enumerate(GATES_N):
        key = f"pass_gate_{int(gate)}_N"
        expected = [
            peak <= gate and fraction <= FRACTION_GATE
            for peak, fraction in zip(
                case["peak_tension_N"], case["tension_fraction"])
        ]
        if (
            not isinstance(run[key], list)
            or any(type(item) is not bool for item in run[key])
            or run[key] != expected
        ):
            raise GateError(
                f"case {row.ordinal} plain run table {key} differs")
    for source_key, public_key in (
        ("channel_table", "channel_metrics"),
        ("channel_qoi_table", "channel_qoi"),
        ("saved_baseline_table", "saved_reconstruction"),
    ):
        if not _strict_json_equivalent(data[source_key], case[public_key]):
            raise GateError(
                f"case {row.ordinal} plain report {source_key} differs")

    solver = _exact_keys(
        data["solver_source_sha256"],
        {"module", "path", "sha256"},
        f"case {row.ordinal} solver source table",
    )
    modules = _as_list(solver["module"])
    paths = _as_list(solver["path"])
    hashes = _as_list(solver["sha256"])
    if (
        modules != list(SOLVER_MODULES)
        or len(paths) != len(SOLVER_MODULES)
        or len(hashes) != len(SOLVER_MODULES)
        or any(not isinstance(item, str) for item in paths + hashes)
    ):
        raise GateError(f"case {row.ordinal} solver source inventory differs")
    for module, path_text, digest in zip(modules, paths, hashes):
        expected_path = matlab_source_path(f"{module}.m").resolve()
        if (
            Path(path_text).resolve() != expected_path
            or not SHA256_RE.fullmatch(digest)
        ):
            raise GateError(
                f"case {row.ordinal} solver source {module} differs")
    observed_solver_root = _sha256_bytes("\n".join(
        f"{module}:{digest}"
        for module, digest in zip(modules, hashes)
    ).encode("utf-8"))
    if observed_solver_root != solver_root:
        raise GateError(f"case {row.ordinal} solver source root differs")

    signal_hashes = _as_list(data["signal_common_sha256"])
    if len(signal_hashes) != 3 or any(
        not isinstance(item, str) or not SHA256_RE.fullmatch(item)
        for item in signal_hashes
    ):
        raise GateError(f"case {row.ordinal} signal hashes are malformed")
    if data["numeric_hash_selfcheck"] != NUMERIC_HASH_SELFCHECK:
        raise GateError(
            f"case {row.ordinal} numeric hash convention selfcheck differs")
    # These three digests are diagnostic commitments to the unpersisted
    # common-grid waveforms.  Acceptance is independently recomputed from
    # the persisted waveform metrics/QOIs, never inferred from these hashes.
    expected_peak_delta = [
        item - case["peak_tension_N"][2]
        for item in case["peak_tension_N"]
    ]
    expected_fraction_delta = [
        item - case["tension_fraction"][2]
        for item in case["tension_fraction"]
    ]
    for key, expected in (
        ("contact_peak_delta_vs_finest_N", expected_peak_delta),
        ("tension_fraction_delta_vs_finest", expected_fraction_delta),
    ):
        observed = _float_list(data[key], count=3, label=key)
        if any(
            not _allclose(a, b, rtol=1e-12, atol=1e-15)
            for a, b in zip(observed, expected)
        ):
            raise GateError(f"case {row.ordinal} {key} differs")
    if not _allclose(
        _strict_number(data["reference_dt_ms"], "reference_dt_ms"),
        case["actual_dt_ms"][2],
        rtol=1e-12,
        atol=1e-15,
    ):
        raise GateError(f"case {row.ordinal} reference dt differs")
    comparison_window = _float_list(
        data["comparison_window_m"], count=2, label="comparison window")
    if row.stage not in EXPECTED_L_BRIDGE_M:
        raise GateError(
            f"case {row.ordinal} stage {row.stage!r} has no registered "
            "bridge length")
    expected_l_bridge = EXPECTED_L_BRIDGE_M[row.stage]
    expected_window_end = 10.0 + expected_l_bridge + POST_DECK_WINDOW_M
    if comparison_window[0] != 10.0:
        raise GateError(f"case {row.ordinal} comparison window malformed")
    if abs(comparison_window[1] - expected_window_end) > COMPARISON_WINDOW_ATOL:
        raise GateError(
            f"case {row.ordinal} comparison window end differs: expected "
            f"{expected_window_end!r} m (10 m skip + {expected_l_bridge!r} m "
            f"bridge + {POST_DECK_WINDOW_M!r} m registered post-deck crop "
            f"span for stage {row.stage}), observed {comparison_window[1]!r}")

    integrity = _exact_keys(
        data["dataset_integrity"],
        {
            "status", "manifest_root", "state_digest_match", "marker_match",
            "state_table_match", "identity_table_match", "case_info_sha256",
            "damage_states_sha256", "completion_marker_sha256",
            "file_digests_sha256", "qualification_host_receipt_sha256",
        },
        f"case {row.ordinal} dataset integrity",
    )
    if integrity["status"] != "VERIFIED" or any(
        integrity[key] is not True
        for key in (
            "state_digest_match", "marker_match",
            "state_table_match", "identity_table_match",
        )
    ):
        raise GateError(f"case {row.ordinal} dataset integrity is not verified")
    integrity_expected = {
        "manifest_root": dataset.content_root,
        "case_info_sha256": dataset.case_info,
        "damage_states_sha256": dataset.damage_states,
        "completion_marker_sha256": dataset.complete,
        "file_digests_sha256": dataset.file_digests,
        "qualification_host_receipt_sha256": dataset.host_receipt,
    }
    for key, expected in integrity_expected.items():
        if integrity[key] != expected:
            raise GateError(
                f"case {row.ordinal} dataset integrity {key} differs")

    descriptor = _exact_keys(
        data["descriptor"],
        {
            "L_bridge_m", "num_spans", "velocity_kmh", "temperature_C",
            "scour_vector", "bearing_vector_Nm_rad", "crack_row",
            "profile_mode", "profile_value", "profile_phase_seed",
            "profile_phase_stream_index", "state_uid", "state_family",
            "has_track_eov", "n_flats", "n_polygonization",
        },
        f"case {row.ordinal} descriptor",
    )
    for key, expected in (
        ("profile_phase_seed", case["profile_phase_seed"]),
        ("profile_phase_stream_index", 5),
        ("state_uid", row.state_uid),
        ("state_family", row.state_family),
    ):
        if not _strict_json_equivalent(descriptor[key], expected):
            raise GateError(f"case {row.ordinal} descriptor {key} differs")
    for key in (
        "L_bridge_m", "velocity_kmh", "temperature_C", "profile_value",
    ):
        _strict_number(descriptor[key], f"descriptor {key}")
    if not _allclose(
        _strict_number(descriptor["L_bridge_m"], "descriptor L_bridge_m"),
        expected_l_bridge,
        rtol=0.0,
        atol=COMPARISON_WINDOW_ATOL,
    ):
        raise GateError(
            f"case {row.ordinal} descriptor L_bridge_m differs from the "
            f"registered stage geometry ({expected_l_bridge!r} m for "
            f"{row.stage}); expected comparison window end "
            f"{expected_window_end!r} m")
    for key in ("num_spans", "n_flats", "n_polygonization"):
        if _strict_integer(
            descriptor[key], f"descriptor {key}"
        ) < (1 if key == "num_spans" else 0):
            raise GateError(
                f"case {row.ordinal} descriptor {key} is outside domain")
    for key in ("scour_vector", "bearing_vector_Nm_rad", "crack_row"):
        values = _as_list(descriptor[key])
        if not values:
            raise GateError(
                f"case {row.ordinal} descriptor {key} is empty")
        for index, value in enumerate(values):
            _strict_number(value, f"descriptor {key}[{index}]")
    if not isinstance(descriptor["profile_mode"], str) \
            or not descriptor["profile_mode"]:
        raise GateError(
            f"case {row.ordinal} descriptor profile_mode is not text")
    if type(descriptor["has_track_eov"]) is not bool:
        raise GateError(f"case {row.ordinal} descriptor track flag is not bool")
    if expected_descriptor is not None and not _strict_json_equivalent(
        descriptor, expected_descriptor
    ):
        raise GateError(
            f"case {row.ordinal} physical descriptor differs from "
            "authenticated state/passage data"
        )
    if not isinstance(data["case_name"], str) \
            or not isinstance(data["saved_baseline_note"], str):
        raise GateError(f"case {row.ordinal} report text field malformed")
