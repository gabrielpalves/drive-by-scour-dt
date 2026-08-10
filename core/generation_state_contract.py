"""Closed contracts for one generated MATLAB state file.

The MATLAB writer, resume validator, release comparator, and training loader
must agree on these invariants.  Keeping the small, representation-independent
checks here prevents each Python consumer from growing a subtly different
version of the state schema.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

import numpy as np


STATE_TOP_LEVEL_FIELDS = frozenset(
    {
        "data",
        "file_actual_matlab_environment_sha256",
        "file_campaign_matlab_environment_sha256",
        "file_campaign_matlab_release",
        "file_gen_fingerprint",
        "file_gen_schema",
        "file_generator_source_root_sha256",
        "file_matlab_release",
        "file_qualification_source_sha256",
        "file_random_stream_schedule_version",
        "file_release_qualification_run",
        "file_state_seed_id",
        "file_state_uid",
    }
)

STATE_DATA_FIELDS = frozenset(
    {
        "AcelPrimVag",
        "AcelRodaPrimVag",
        "AcelWheelsetPrimVag",
        "Dano",
        "DimAcel",
        "DimSpace",
        "L_bridge_eff",
        "PitchPrimVag",
        "Temperatura",
        "VehiclesProps",
        "Velocidade",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "beam_f1_Hz",
        "bearing_fixity",
        "bearing_vector",
        "bridge_samp",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "campaign_matlab_release",
        "channel_schema_id",
        "contact_log",
        "crack_log",
        "crack_on",
        "crop_end",
        "crop_start",
        "gen_fingerprint",
        "gen_schema",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "generator_source_root_sha256",
        "latent_bearing_fixity",
        "latent_crack_on",
        "matlab_release",
        "oor_log",
        "passage_named_stream_seed_id",
        "profile_log",
        "profile_mode",
        "qualification_source_sha256",
        "random_stream_schedule_version",
        "release_qualification_run",
        "scour_supports",
        "scour_vector",
        "state_family",
        "state_named_stream_seed_id",
        "state_seed_id",
        "state_uid",
        "track_log",
    }
)

DAMAGE_STATE_FIELDS = frozenset(
    {
        "AnchorLevel",
        "AnchorTarget",
        "BearingFixity",
        "BearingStates",
        "CrackOn",
        "DamageStates",
        "LatentBearingFixity",
        "LatentCrackOn",
        "PassageNamedStreamSeedID",
        "PassageNamedStreamSeedIDFlat",
        "StateFamily",
        "StateNamedStreamSeedID",
        "StateSeedID",
        "StateUID",
        "k_ref_bear",
        "passage_stream_names",
        "random_stream_schedule_version",
        "scour_supports",
        "state_stream_names",
    }
)

_NUMERIC_MAT_NAME = re.compile(r"^\d+\.[mM][aA][tT]$")


def require_exact_fields(
    observed: Iterable[str],
    expected: frozenset[str],
    owner: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    """Reject both missing and unexpected fields in a closed schema."""
    actual = {str(field) for field in observed}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise error_type(
            f"{owner} field inventory differs from the closed contract; "
            f"missing={missing}, extra={extra}"
        )


def require_canonical_state_names(
    entry_names: Iterable[str],
    n_states: int,
    owner: str,
    *,
    error_type: type[Exception] = ValueError,
) -> tuple[str, ...]:
    """Return ``0001.mat..NNNN.mat`` and reject every numeric alias.

    Aliases such as ``1.mat``, ``00001.mat``, ``0001.MAT``, ``0000.mat``,
    and out-of-range numeric names are fatal instead of being ignored.
    """
    expected = tuple(f"{index:04d}.mat" for index in range(1, n_states + 1))
    numeric_names = tuple(
        sorted(
            str(name)
            for name in entry_names
            if _NUMERIC_MAT_NAME.fullmatch(str(name))
        )
    )
    if numeric_names != expected:
        raise error_type(
            f"{owner} numeric MAT inventory must be exactly "
            f"0001.mat..{n_states:04d}.mat; found={numeric_names}"
        )
    return expected


def require_damage_table_shapes(
    table: Mapping[str, Any],
    *,
    n_states: int,
    n_passages: int,
    n_supports: int,
    n_scour_supports: int,
    n_state_streams: int,
    n_passage_streams: int,
    owner: str,
    error_type: type[Exception] = ValueError,
) -> None:
    """Require the exact post-``loadmat(simplify_cells=True)`` shapes."""
    require_exact_fields(
        table, DAMAGE_STATE_FIELDS, owner, error_type=error_type
    )
    expected = {
        "DamageStates": (n_states, n_supports),
        "BearingStates": (n_states, 2),
        "BearingFixity": (n_states, 2),
        "LatentBearingFixity": (n_states, 2),
        "k_ref_bear": (),
        # ``simplify_cells=True`` implies ``squeeze_me=True``.  MATLAB's
        # one-support vector therefore arrives as a scalar for F40, while the
        # three-support L99 vector remains one-dimensional.
        "scour_supports": (
            () if n_scour_supports == 1 else (n_scour_supports,)
        ),
        "StateFamily": (n_states,),
        "AnchorTarget": (n_states,),
        "AnchorLevel": (n_states,),
        "StateUID": (n_states,),
        "StateSeedID": (n_states,),
        "StateNamedStreamSeedID": (n_states, n_state_streams),
        "PassageNamedStreamSeedID": (
            n_states,
            n_passages,
            n_passage_streams,
        ),
        "PassageNamedStreamSeedIDFlat": (
            n_states,
            n_passages * n_passage_streams,
        ),
        "random_stream_schedule_version": (),
        "state_stream_names": (n_state_streams,),
        "passage_stream_names": (n_passage_streams,),
        "LatentCrackOn": (n_states,),
        "CrackOn": (n_states,),
    }
    mismatches = {
        field: (np.asarray(table[field]).shape, shape)
        for field, shape in expected.items()
        if np.asarray(table[field]).shape != shape
    }
    if mismatches:
        raise error_type(
            f"{owner}: field shapes differ from the exact 19-field damage "
            f"contract: {mismatches}"
        )


def _numeric_vector(
    value: Any,
    length: int,
    owner: str,
    *,
    integer: bool = False,
    positive: bool = False,
    error_type: type[Exception],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise error_type(f"{owner} must be a real numeric vector")
    vector = raw.astype(np.float64, copy=False).reshape(-1)
    if vector.size != length or not np.all(np.isfinite(vector)):
        raise error_type(
            f"{owner} must contain exactly {length} finite values"
        )
    if integer and not np.all(vector == np.floor(vector)):
        raise error_type(f"{owner} must be integer-valued")
    if positive and np.any(vector <= 0):
        raise error_type(f"{owner} must be strictly positive")
    return vector


def _matlab_round_positive(values: np.ndarray) -> np.ndarray:
    """MATLAB ``round`` for the nonnegative quantities in this contract."""
    return np.floor(values + 0.5)


def validate_raw_metadata(
    data: Mapping[str, Any],
    *,
    n_passages: int,
    bridge_length_m: float,
    error_type: type[Exception] = ValueError,
    owner: str = "data",
) -> dict[str, np.ndarray]:
    """Validate the exact registered RAW space/crop equations."""
    integer_fields = (
        "DimAcel",
        "DimSpace",
        "crop_start",
        "crop_end",
        "bridge_samp",
    )
    values = {
        field: _numeric_vector(
            data[field],
            n_passages,
            f"{owner}.{field}",
            integer=True,
            positive=True,
            error_type=error_type,
        )
        for field in integer_fields
    }
    values["L_bridge_eff"] = _numeric_vector(
        data["L_bridge_eff"],
        n_passages,
        f"{owner}.L_bridge_eff",
        positive=True,
        error_type=error_type,
    )
    values["Velocidade"] = _numeric_vector(
        data["Velocidade"],
        n_passages,
        f"{owner}.Velocidade",
        positive=True,
        error_type=error_type,
    )

    expected_length = np.full(n_passages, float(bridge_length_m))
    expected_bridge = np.full(
        n_passages,
        _matlab_round_positive(np.asarray([100.0 * bridge_length_m]))[0],
    )
    expected_start = np.full(n_passages, 1001.0)
    expected_space = _matlab_round_positive(
        values["Velocidade"] * values["DimAcel"] / 10.0
    )
    expected_end = np.minimum(
        expected_start - 1.0 + expected_bridge + 1831.0,
        values["DimSpace"],
    )
    crop_length = values["crop_end"] - values["crop_start"] + 1.0

    valid = (
        np.array_equal(values["L_bridge_eff"], expected_length)
        and np.array_equal(values["bridge_samp"], expected_bridge)
        and np.array_equal(values["crop_start"], expected_start)
        and np.array_equal(values["crop_end"], expected_end)
        and np.array_equal(values["DimSpace"], expected_space)
        and np.all(values["DimAcel"] >= 2)
        and np.all(values["crop_start"] <= values["crop_end"])
        and np.all(values["crop_end"] <= values["DimSpace"])
        and np.all(crop_length >= values["bridge_samp"])
    )
    if not valid:
        raise error_type(
            f"{owner} does not reproduce the registered RAW "
            "DimSpace/bridge/crop equations"
        )
    return values


def validate_contact_log(
    value: Any,
    *,
    n_passages: int,
    max_tension_N: float,
    max_tension_fraction: float,
    error_type: type[Exception] = ValueError,
    owner: str = "data.contact_log",
) -> np.ndarray:
    """Validate the signed bilateral-contact diagnostic.

    Column four is signed: negative compression is valid.  Only positive
    reaction is a tensile artifact, and its flag/fraction must agree exactly.
    """
    contact = np.asarray(value)
    if (
        contact.dtype != np.dtype(np.float64)
        or contact.shape != (n_passages, 4)
        or not np.all(np.isfinite(contact))
    ):
        raise error_type(
            f"{owner} must be one finite float64 ({n_passages}, 4) matrix"
        )
    bridge_flag = contact[:, 0]
    track_flag = contact[:, 1]
    tension_fraction = contact[:, 2]
    signed_max_force = contact[:, 3]
    valid = (
        np.all(np.isin(contact[:, :2], (0.0, 1.0)))
        and np.all(bridge_flag <= track_flag)
        and np.array_equal(
            track_flag,
            (signed_max_force > 0.0).astype(np.float64),
        )
        and np.array_equal(
            track_flag,
            (tension_fraction > 0.0).astype(np.float64),
        )
        and np.all(tension_fraction >= 0.0)
        and np.all(tension_fraction <= 1.0)
        and np.all(tension_fraction <= max_tension_fraction)
        and np.all(signed_max_force <= max_tension_N)
    )
    if not valid:
        raise error_type(
            f"{owner} has inconsistent flags/fraction/signed force or exceeds "
            "the registered tensile-artifact gate"
        )
    return contact


def validate_bearing_fixity(
    value: Any,
    *,
    owner: str,
    length: int | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Require finite bearing fixity in the physical half-open range [0, 1)."""
    vector = np.asarray(value)
    if vector.dtype.kind not in "iuf" or vector.dtype.kind == "b":
        raise error_type(f"{owner} must be real numeric")
    vector = vector.astype(np.float64, copy=False).reshape(-1)
    if (
        (length is not None and vector.size != length)
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
        or np.any(vector >= 1.0)
    ):
        expected_length = "" if length is None else f" with length {length}"
        raise error_type(
            f"{owner} must be finite in [0, 1){expected_length}"
        )
    return vector
