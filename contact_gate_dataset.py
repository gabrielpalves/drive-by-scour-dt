"""Qualification-dataset and MATLAB publication verification."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from contact_gate_path_safety import GateError
from contact_gate_core import (
    DatasetDescriptor,
    EXPECTED_PASSAGES,
    RECON_ATOL,
    RECON_RTOL,
    STAGES,
    SelectionRow,
    _exact_keys,
    _first_json_mismatch,
    _sha256_file,
    _strict_json_equivalent,
    _strict_json_file,
)
from contact_gate_policy import (
    _expected_policy_json,
    _expected_selection_records,
)

def _descriptor_scalar(value: Any, label: str) -> Any:
    try:
        import numpy as np
        array = np.asarray(value, dtype=object)
    except Exception as exc:
        raise GateError(f"{label} cannot be scalarized") from exc
    if array.size != 1:
        raise GateError(f"{label} is not scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _passage_container_value(
    value: Any,
    passage_index: int,
    passage_count: int,
    label: str,
) -> Any:
    try:
        import numpy as np
        array = np.asarray(value, dtype=object)
    except Exception as exc:
        raise GateError(f"{label} cannot be indexed") from exc
    if array.size == 0:
        return None
    flattened = array.reshape(-1, order="F")
    if flattened.size != passage_count:
        raise GateError(
            f"{label} has {flattened.size} entries, expected {passage_count}")
    item = flattened[passage_index - 1]
    if type(item).__name__ == "mat_struct":
        return _mat_jsonish(item, f"{label}[{passage_index}]")
    return item


def _matlab_row_count(value: Any, label: str) -> int:
    try:
        import numpy as np
        array = np.asarray(value)
    except Exception as exc:
        raise GateError(f"{label} cannot be counted") from exc
    if array.size == 0:
        return 0
    if array.ndim <= 1:
        return 1
    return int(array.shape[0])


def _expected_physical_descriptor(
    manifest: dict[str, Any],
    data: dict[str, Any],
    row: SelectionRow,
    *,
    passage_count: int,
) -> dict[str, Any]:
    try:
        import numpy as np

        velocity = np.asarray(data["Velocidade"], dtype=float).reshape(-1)
        temperature = np.asarray(
            data["Temperatura"], dtype=float).reshape(-1)
        scour = np.asarray(data["scour_vector"], dtype=float).reshape(-1)
        bearing = np.asarray(
            data.get("bearing_vector", [0.0, 0.0]),
            dtype=float,
        ).reshape(-1)
        crack = np.asarray(data["crack_log"], dtype=float)
        profile = np.asarray(data["profile_log"], dtype=float).reshape(-1)
        named_seeds = np.asarray(
            data["state_named_stream_seed_id"], dtype=np.uint64
        ).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(
            f"{row.stage}/{row.state_index:04d} descriptor source malformed"
        ) from exc
    if (
        velocity.size != passage_count
        or temperature.size != passage_count
        or profile.size != passage_count
        or crack.ndim != 2
        or crack.shape[0] != passage_count
        or named_seeds.size < 5
        or any(
            not np.all(np.isfinite(values))
            for values in (velocity, temperature, scour, bearing, crack, profile)
        )
    ):
        raise GateError(
            f"{row.stage}/{row.state_index:04d} descriptor source shape differs")
    profile_mode_value = data.get("profile_mode", manifest.get("profile_mode"))
    profile_mode = _descriptor_scalar(
        profile_mode_value,
        f"{row.stage}/{row.state_index:04d} profile_mode",
    )
    if not isinstance(profile_mode, str) or not profile_mode:
        raise GateError(
            f"{row.stage}/{row.state_index:04d} profile_mode is malformed")
    track = _passage_container_value(
        data.get("track_log", []),
        row.passage_index,
        passage_count,
        f"{row.stage}/{row.state_index:04d} track_log",
    )
    oor = _passage_container_value(
        data.get("oor_log", []),
        row.passage_index,
        passage_count,
        f"{row.stage}/{row.state_index:04d} oor_log",
    )
    has_track = track is not None and np.asarray(
        track, dtype=object).size != 0
    n_flats = 0
    n_polygonization = 0
    if oor is not None and np.asarray(oor, dtype=object).size != 0:
        if not isinstance(oor, dict):
            raise GateError(
                f"{row.stage}/{row.state_index:04d} oor_log entry is malformed")
        if "flats" in oor:
            n_flats = _matlab_row_count(
                oor["flats"], f"{row.stage} oor flats")
        if "poly" in oor:
            n_polygonization = _matlab_row_count(
                oor["poly"], f"{row.stage} oor polygonization")
    passage = row.passage_index - 1
    crack_row = crack[passage, :].reshape(-1).tolist()
    return {
        "L_bridge_m": float(_descriptor_scalar(
            manifest["L_bridge_m"], f"{row.stage} L_bridge_m")),
        "num_spans": float(_descriptor_scalar(
            manifest["num_spans"], f"{row.stage} num_spans")),
        "velocity_kmh": float(velocity[passage] * 3.6),
        "temperature_C": float(temperature[passage]),
        "scour_vector": scour.tolist(),
        "bearing_vector_Nm_rad": bearing.tolist(),
        "crack_row": crack_row,
        "profile_mode": profile_mode,
        "profile_value": float(profile[passage]),
        "profile_phase_seed": int(named_seeds[4]),
        "profile_phase_stream_index": 5,
        "state_uid": row.state_uid,
        "state_family": row.state_family,
        "has_track_eov": bool(has_track),
        "n_flats": n_flats,
        "n_polygonization": n_polygonization,
    }


def _validate_datasets_with_comparator(
    paths: list[Path],
    descriptors: list[DatasetDescriptor],
    rows: list[SelectionRow],
    *,
    source_root: str,
    environment_sha: str,
    host_id: str,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    try:
        import numpy as np
        from compare_generation_releases import (
            _current_policy,
            _validated_payload,
        )
    except Exception as exc:
        raise GateError(f"cannot load locked qualification comparator: {exc}") from exc
    policy = _current_policy()
    evidence_by_stage: dict[str, Any] = {}
    manifests_by_stage: dict[str, dict[str, Any]] = {}
    states_by_stage: dict[str, dict[str, Any]] = {}
    for path, frozen in zip(paths, descriptors):
        evidence, manifest, _, states = _validated_payload(path, policy)
        expected = {
            "stage": frozen.stage,
            "path": str(path.resolve()),
            "actual_matlab_environment_sha256": environment_sha,
            "generator_source_root_sha256": source_root,
            "gen_fingerprint": frozen.fingerprint,
            "qualification_source_sha256": frozen.qual_source,
            "qualification_executed_file_sha256": frozen.qual_executed,
            "qualification_host_id": host_id,
            "qualification_host_diagnostic_sha256": frozen.host_diagnostic,
            "dataset_content_root_sha256": frozen.content_root,
        }
        for key, wanted in expected.items():
            if getattr(evidence, key) != wanted:
                raise GateError(
                    f"comparator evidence {frozen.stage}.{key} differs"
                )
        raw_hashes = {
            "case_info": _sha256_file(path / "case_info.mat"),
            "damage_states": _sha256_file(path / "damage_states.mat"),
            "file_digests": _sha256_file(path / "file_digests.mat"),
            "complete": _sha256_file(path / "_GENERATION_COMPLETE"),
            "host_receipt": _sha256_file(
                path / "qualification_host_receipt.json"),
            "qual_executed": _sha256_file(path / "qualification_executed.m"),
        }
        for key, observed in raw_hashes.items():
            if observed != getattr(frozen, key):
                raise GateError(f"frozen dataset {frozen.stage}.{key} differs")
        evidence_by_stage[frozen.stage] = evidence
        manifests_by_stage[frozen.stage] = manifest
        states_by_stage[frozen.stage] = states
    expected_descriptors: dict[
        tuple[str, int, int], dict[str, Any]
    ] = {}
    for row in rows:
        state_name = f"{row.state_index:04d}.mat"
        path = paths[STAGES.index(row.stage)]
        if _sha256_file(path / state_name) != row.state_file_sha256:
            raise GateError(f"selection state SHA differs for row {row.ordinal}")
        loaded = states_by_stage[row.stage][state_name]
        data = loaded.get("data")
        if not isinstance(data, dict):
            raise GateError(f"comparator state lacks data: {row.stage}/{state_name}")
        contact = np.asarray(data.get("contact_log"), dtype=float)
        if contact.shape != (EXPECTED_PASSAGES, 4) or not np.all(
            np.isfinite(contact)
        ):
            raise GateError(f"state contact_log malformed: {row.stage}/{state_name}")
        frozen_contact = np.asarray([
            row.saved_bridge_flag, row.saved_track_flag,
            row.saved_fraction, row.saved_signed_peak_n,
        ])
        if not np.array_equal(contact[row.passage_index - 1, :2], frozen_contact[:2]) \
                or not np.allclose(
                    contact[row.passage_index - 1, 2:],
                    frozen_contact[2:],
                    rtol=RECON_RTOL,
                    atol=RECON_ATOL,
                ):
            raise GateError(f"selection contact row differs at {row.ordinal}")
        expected_descriptors[
            (row.stage, row.state_index, row.passage_index)
        ] = _expected_physical_descriptor(
            manifests_by_stage[row.stage],
            data,
            row,
            passage_count=EXPECTED_PASSAGES,
        )
    if len(expected_descriptors) != len(rows):
        raise GateError("authenticated physical descriptor inventory differs")
    return expected_descriptors


def _mat_scalar_text(value: Any, label: str) -> str:
    try:
        import numpy as np
        array = np.asarray(value, dtype=object)
    except Exception as exc:
        raise GateError(f"{label} cannot be scalarized") from exc
    if array.size != 1:
        raise GateError(f"{label} is not scalar")
    return str(array.reshape(-1)[0])


def _mat_jsonish(value: Any, label: str) -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise GateError(f"{label} requires locked numpy") from exc
    if type(value).__name__ == "MatlabOpaque":
        raise GateError(f"{label} unexpectedly contains an opaque MATLAB object")
    if type(value).__name__ == "mat_struct":
        fields = getattr(value, "_fieldnames", None)
        if (
            not isinstance(fields, list)
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise GateError(f"{label} contains a malformed MATLAB struct")
        return {
            field: _mat_jsonish(getattr(value, field), f"{label}.{field}")
            for field in fields
        }
    if isinstance(value, dict):
        return {
            str(key): _mat_jsonish(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _mat_jsonish(item, f"{label}[]")
            for item in value
        ]
    if isinstance(value, np.ndarray):
        if value.size == 0 and value.dtype.kind in {"U", "S"}:
            return ""
        if value.ndim == 0:
            return _mat_jsonish(value.item(), label)
        return _mat_jsonish(value.tolist(), label)
    if isinstance(value, np.generic):
        return _mat_jsonish(value.item(), label)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateError(f"{label} contains non-UTF-8 bytes") from exc
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise GateError(f"{label} has unsupported MATLAB type {type(value).__name__}")


def _validate_mat_sources(
    gate_dir: Path,
    *,
    policy_descriptor: str,
    policy_sha: str,
    selection_descriptor: str,
    selection_sha: str,
    summary: dict[str, Any],
    rows: list[SelectionRow],
) -> None:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise GateError(f"locked scipy is required to verify MAT sources: {exc}") from exc

    def public(path: Path) -> dict[str, Any]:
        try:
            loaded = loadmat(path, simplify_cells=True, mat_dtype=True)
        except Exception as exc:
            raise GateError(f"cannot read MAT source {path}: {exc}") from exc
        return {
            str(key): value
            for key, value in loaded.items()
            if not str(key).startswith("__")
        }

    policy_mat = public(gate_dir / "closure_policy.mat")
    if set(policy_mat) != {"frozen_policy"} \
            or not isinstance(policy_mat["frozen_policy"], dict):
        raise GateError("closure_policy.mat has the wrong exact root")
    frozen_policy = policy_mat["frozen_policy"]
    _exact_keys(
        frozen_policy,
        {"canonical_policy", "descriptor", "sha256"},
        "closure_policy.mat frozen_policy",
    )
    if (
        _mat_scalar_text(frozen_policy.get("descriptor"), "policy descriptor")
        != policy_descriptor
        or _mat_scalar_text(frozen_policy.get("sha256"), "policy sha")
        != policy_sha
    ):
        raise GateError("closure_policy.mat differs from authenticated TXT")
    observed_policy = _mat_jsonish(
        frozen_policy["canonical_policy"],
        "closure_policy.mat canonical policy",
    )
    if not _strict_json_equivalent(
        observed_policy, _expected_policy_json(summary["source_commit"])
    ):
        raise GateError(
            "closure_policy.mat canonical policy differs from TXT/checker")

    selection_mat = public(gate_dir / "selection_manifest.mat")
    if set(selection_mat) != {"frozen_selection"} \
            or not isinstance(selection_mat["frozen_selection"], dict):
        raise GateError("selection_manifest.mat has the wrong exact root")
    frozen_selection = selection_mat["frozen_selection"]
    _exact_keys(
        frozen_selection,
        {"selection_records", "datasets", "descriptor", "sha256"},
        "selection_manifest.mat frozen_selection",
    )
    if (
        _mat_scalar_text(
            frozen_selection.get("descriptor"), "selection descriptor")
        != selection_descriptor
        or _mat_scalar_text(frozen_selection.get("sha256"), "selection sha")
        != selection_sha
    ):
        raise GateError("selection_manifest.mat differs from authenticated TSV")
    observed_records = _mat_jsonish(
        frozen_selection["selection_records"],
        "selection_manifest.mat selection records",
    )
    if not _strict_json_equivalent(
        observed_records, _expected_selection_records(rows, summary)
    ):
        raise GateError(
            "selection_manifest.mat records differ from authenticated TSV")
    if not _strict_json_equivalent(_mat_jsonish(
        frozen_selection["datasets"],
        "selection_manifest.mat frozen datasets",
    ), summary["datasets"]):
        raise GateError(
            "selection_manifest.mat datasets differ from summary JSON")

    summary_mat = public(gate_dir / "gate_summary.mat")
    if set(summary_mat) != {"publication"} \
            or not isinstance(summary_mat["publication"], dict):
        raise GateError("gate_summary.mat has the wrong exact root")
    publication = summary_mat["publication"]
    _exact_keys(
        publication,
        {"summary", "selection_descriptor", "selection_sha256"},
        "gate_summary.mat publication",
    )
    mat_summary = publication.get("summary")
    if not isinstance(mat_summary, dict):
        raise GateError("gate_summary.mat lacks publication.summary")
    if not _strict_json_equivalent(
        _mat_jsonish(mat_summary, "gate_summary.mat summary"), summary
    ):
        raise GateError("gate_summary.mat summary differs from summary JSON")
    if (
        _mat_scalar_text(
            publication["selection_descriptor"],
            "gate_summary.mat selection descriptor",
        ) != selection_descriptor
        or _mat_scalar_text(
            publication["selection_sha256"],
            "gate_summary.mat selection sha",
        ) != selection_sha
    ):
        raise GateError(
            "gate_summary.mat selection identity differs")

    for row in rows:
        stem = f"{row.ordinal:04d}_case"
        case_mat = public(gate_dir / "cases" / f"{stem}.mat")
        if set(case_mat) != {"canonical_case"}:
            raise GateError(f"{stem}.mat lacks exact canonical_case root")
        canonical_case = _mat_jsonish(
            case_mat["canonical_case"], f"{stem}.mat canonical_case")
        case_json = _strict_json_file(
            gate_dir / "cases" / f"{stem}.json", f"{stem} JSON")
        if not _strict_json_equivalent(canonical_case, case_json):
            raise GateError(
                f"{stem}.mat canonical_case differs from JSON: "
                f"{_first_json_mismatch(canonical_case, case_json)}"
            )
