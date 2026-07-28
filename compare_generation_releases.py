"""Fail-closed comparison of two MATLAB environment-qualification datasets.

This program authenticates two deliberately reduced MATLAB runs against the
*current repository policy* and then compares their canonical scientific
payloads.  The policy is never inferred from either input directory.

Each input must also carry a self-authenticating qualification-only host
diagnostic receipt. Equal MATLAB-environment digests are allowed only when the
receipts name independently identified hosts; host/CPU identity never enters
the production generation schema and hardware equality is not required.

``SEMANTICALLY-BIT-IDENTICAL`` means that every compared canonical value is
exactly equal after excluding only the explicitly named executable-environment
identity fields and MAT-file container metadata.  It does **not** claim that
the raw MAT-file bytes are identical: MATLAB v5 headers normally contain
timestamps.  Only solver-derived floating outputs may receive the reviewed
numerical tolerance.

Numerical equivalence is pending by default.  Accepting it requires both
``--accept-numerical`` and a new machine-readable ``--receipt`` path.

Exit status
-----------
0  canonical semantics are exact, or numerical equivalence was explicitly
   accepted and bound to a new receipt;
1  authenticated payloads are materially different;
2  an input/evidence contract is invalid;
3  numerically equivalent within tolerance, but explicit acceptance is pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np
import scipy
from scipy.io import loadmat

from core.campaign_contract import (
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_GEN_SCHEMA,
    STAGE_ORDER,
    campaign_stage_contract,
    generation_config_expectations,
)
from core.environment import (
    load_environment_lock,
    matlab_environment_descriptor,
)
from core.source_provenance import (
    generator_source_root,
    python_runtime_source_root,
)


_ROOT = Path(__file__).resolve().parent
_ENVIRONMENT_LOCK = _ROOT / "environment" / "campaign-py313-cu128.json"
_STATE_RE = re.compile(r"^\d{4}\.mat$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MATLAB_RELEASE_RE = re.compile(r"^R20\d{2}[ab]$")
_QUALIFICATION_HOST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_QUALIFICATION_HOST_SCHEMA = "ttbi-matlab-qualification-host-v1"
_RANDOM_STREAM_SCHEDULE = "uid-named-substreams-v2"
_STATE_STREAM_NAMES = (
    "operations",
    "crack",
    "profile-state",
    "track",
    "profile-phase",
)
_PASSAGE_STREAM_NAMES = ("profile-passage", "oor-passage")
_QUALIFICATION_HOST_FIELDS = frozenset(
    {
        "schema",
        "declared_host_id",
        "hostname",
        "cpu_identifier",
        "logical_processors",
        "matlab_max_threads",
        "computer_arch",
        "actual_matlab_environment_sha256",
        "qualification_source_sha256",
        "qualification_executed_file_sha256",
        "canonical_descriptor",
        "host_diagnostic_sha256",
    }
)
_REQUIRED_SIGNALS = ("AcelPrimVag", "AcelRodaPrimVag", "PitchPrimVag")
_SIGNAL_ROWS = {
    "AcelPrimVag": 3,
    "AcelRodaPrimVag": 4,
    "PitchPrimVag": 3,
}
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "stage",
        "gen_schema",
        "gen_fingerprint",
        "generation_config_json",
        "generation_behavior_version",
        "matlab_release",
        "campaign_matlab_release",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "qualification_source_sha256",
        "release_qualification_run",
        "n_states",
        "passages_per_state",
        "max_parfor_workers",
        "L_bridge_m",
        "num_spans",
        "num_supports",
        "state_identity_version",
        "joint_lhs_design",
        "n_latent_bearing_dims",
        "bearing_k_ref_Nm_rad",
        "bearing_mode",
        "use_crack_eov",
        "profile_mode",
        "use_track_eov",
        "use_oor_eov",
        "oor_flats_enabled",
        "n_target_healthy",
        "n_scour_only",
        "n_bearing_only",
        "n_nuisance_only",
        "n_joint",
    }
)
_REQUIRED_DATA_FIELDS = frozenset(
    {
        *_REQUIRED_SIGNALS,
        "DimAcel",
        "DimSpace",
        "crop_start",
        "crop_end",
        "bridge_samp",
        "L_bridge_eff",
        "scour_vector",
        "scour_supports",
        "Dano",
        "state_family",
        "state_uid",
        "state_seed_id",
        "latent_bearing_fixity",
        "latent_crack_on",
        "crack_on",
        "beam_f1_Hz",
        "bearing_vector",
        "bearing_fixity",
        "crack_log",
        "profile_mode",
        "profile_log",
        "track_log",
        "oor_log",
        "contact_log",
        "Temperatura",
        "Velocidade",
        "VehiclesProps",
        "gen_schema",
        "gen_fingerprint",
        "matlab_release",
        "campaign_matlab_release",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
        "generator_source_digest_lines",
        "generator_source_file_count",
        "qualification_source_sha256",
        "release_qualification_run",
        "random_stream_schedule_version",
        "state_named_stream_seed_id",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "data",
        "file_gen_schema",
        "file_gen_fingerprint",
        "file_matlab_release",
        "file_campaign_matlab_release",
        "file_release_qualification_run",
        "file_state_uid",
        "file_state_seed_id",
        "file_random_stream_schedule_version",
        "file_actual_matlab_environment_sha256",
        "file_campaign_matlab_environment_sha256",
        "file_generator_source_root_sha256",
        "file_qualification_source_sha256",
    }
)
_DEFAULT_RTOL = 1e-10
_DEFAULT_ATOL = 1e-12
_STATE_FAMILIES = frozenset(
    {"target_healthy", "scour_only", "bearing_only", "nuisance_only", "joint"}
)
_MECHANISM_COVERAGE_STAGES = frozenset({"s16_all", "s23_all4"})
_FAMILY_MANIFEST_FIELDS = {
    "target_healthy": "n_target_healthy",
    "scour_only": "n_scour_only",
    "bearing_only": "n_bearing_only",
    "nuisance_only": "n_nuisance_only",
    "joint": "n_joint",
}

def _qualification_stage_policy(stage: str) -> dict[str, Any]:
    """Overlay MICRO counts on the single scientific campaign-stage contract."""
    contract = campaign_stage_contract(stage)
    geometry = contract["geometry"]
    scenario = contract["scenario"]
    family_counts = {
        "target_healthy": 3,
        "scour_only": 4 * len(geometry["scour_supports"]),
        # Strong-CRN qualification keeps the latent family inventory fixed
        # across rungs. Mechanism toggles affect active physics, never rows.
        "bearing_only": 8,
        "nuisance_only": 6,
        "joint": 10,
    }
    return {
        "L_bridge_m": geometry["L_bridge_m"],
        "num_spans": geometry["num_spans"],
        "num_supports": geometry["num_supports"],
        "scour_supports": tuple(geometry["scour_supports"]),
        "bearing_mode": scenario["bearing_mode"],
        "use_crack_eov": scenario["use_crack_eov"],
        "profile_mode": scenario["profile_mode"],
        "use_track_eov": scenario["use_track_eov"],
        "use_oor_eov": scenario["use_oor_eov"],
        "oor_flats_enabled": scenario["oor_flats_enabled"],
        "n_states": sum(family_counts.values()),
        "family_counts": family_counts,
    }


_STAGE_POLICIES: dict[str, dict[str, Any]] = {
    stage: _qualification_stage_policy(stage) for stage in STAGE_ORDER
}
_GENERATION_REALIZATION_FIELDS = frozenset(
    {
        "DamageStates",
        "BearingStates",
        "BearingFixity",
        "StateFamily",
        "AnchorTarget",
        "AnchorLevel",
        "StateUID",
        "StateSeedID",
        "LatentBearingFixity",
        "LatentCrackOn",
        "CrackOn",
        "StateNamedStreamSeedID",
        "PassageNamedStreamSeedIDFlat",
    }
)

# Solver outputs only. Random inputs, labels, geometry, state families and
# nuisance realisations remain exact.
_TOLERANT_FLOAT_FIELDS = frozenset(
    {*_REQUIRED_SIGNALS, "contact_log", "beam_f1_Hz"}
)
_MANIFEST_IGNORED_KEYS = frozenset(
    {
        "matlab_release",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
        "timestamp",
    }
)
_TOP_IGNORED_KEYS = frozenset(
    {"data", "file_matlab_release", "file_actual_matlab_environment_sha256"}
)
_DATA_IGNORED_KEYS = frozenset(
    {
        "matlab_release",
        "actual_matlab_environment_descriptor",
        "actual_matlab_environment_sha256",
    }
)


class QualificationInputError(RuntimeError):
    """A directory cannot serve as environment-qualification evidence."""


@dataclass(frozen=True)
class CurrentPolicy:
    environment_lock_sha256: str
    parser_environment: dict[str, str]
    python_runtime_source_root_sha256: str
    python_runtime_source_file_count: int
    campaign_matlab_release: str
    campaign_matlab_environment_descriptor: str
    campaign_matlab_environment_sha256: str
    generator_source_root_sha256: str
    generator_source_digest_lines: str
    generator_source_file_count: int
    gen_schema: str
    generation_behavior_version: str
    max_parfor_workers: int
    contact_force_tolerance_N: float
    contact_fraction_tolerance: float


@dataclass(frozen=True)
class MechanismCoverage:
    required: bool
    crack_active_passages: int
    profile_active_passages: int
    ballast_patch_rows: int
    hanging_group_rows: int
    pad_departure_passages: int
    polygon_rows: int
    witnesses: dict[str, str]


@dataclass(frozen=True)
class DatasetEvidence:
    path: str
    stage: str
    matlab_release: str
    actual_matlab_environment_descriptor: str
    actual_matlab_environment_sha256: str
    campaign_matlab_release: str
    campaign_matlab_environment_descriptor: str
    campaign_matlab_environment_sha256: str
    gen_schema: str
    generation_behavior_version: str
    gen_fingerprint: str
    generator_source_root_sha256: str
    generator_source_file_count: int
    qualification_source_sha256: str
    qualification_executed_file_sha256: str
    qualification_host_id: str
    qualification_hostname: str
    qualification_cpu_identifier: str
    qualification_logical_processors: int
    qualification_matlab_max_threads: int
    qualification_computer_arch: str
    qualification_host_canonical_descriptor: str
    qualification_host_diagnostic_sha256: str
    n_states: int
    passages_per_state: int
    num_supports: int
    max_parfor_workers: int
    dataset_content_root_sha256: str
    state_files: tuple[str, ...]
    mechanism_coverage: MechanismCoverage


@dataclass
class ComparisonStats:
    compared_leaves: int = 0
    compared_numeric_values: int = 0
    compared_signal_values: int = 0
    tolerant_leaves: int = 0
    exact_leaves: int = 0
    max_absolute_difference: float = 0.0
    max_relative_difference: float = 0.0
    worst_path: str = ""
    numerical_difference: bool = False
    mismatches: list[str] = field(default_factory=list)

    def mismatch(self, message: str) -> None:
        if len(self.mismatches) < 50:
            self.mismatches.append(message)


@dataclass(frozen=True)
class ComparisonResult:
    verdict: str
    evidence_a: DatasetEvidence
    evidence_b: DatasetEvidence
    stats: ComparisonStats
    raw_byte_identical_states: int
    tolerance_rtol: float
    tolerance_atol: float
    environment_lock_sha256: str
    parser_environment: dict[str, str]
    python_runtime_source_root_sha256: str
    python_runtime_source_file_count: int

    @property
    def default_exit_code(self) -> int:
        if self.verdict == "SEMANTICALLY-BIT-IDENTICAL":
            return 0
        if self.verdict == "NUMERICALLY-EQUIVALENT":
            return 3
        return 1


@dataclass(frozen=True)
class DamageLabels:
    damage_states: np.ndarray
    bearing_states: np.ndarray
    bearing_fixity: np.ndarray
    state_families: tuple[str, ...]
    anchor_target: np.ndarray
    anchor_level: np.ndarray
    state_uids: tuple[str, ...]
    state_seed_ids: np.ndarray
    random_stream_schedule_version: str
    state_stream_names: tuple[str, ...]
    passage_stream_names: tuple[str, ...]
    state_named_stream_seed_ids: np.ndarray
    passage_named_stream_seed_ids: np.ndarray
    passage_named_stream_seed_ids_flat: np.ndarray
    latent_bearing_fixity: np.ndarray
    latent_crack_on: np.ndarray
    scour_supports: np.ndarray
    crack_on: np.ndarray


@dataclass
class _CoverageAccumulator:
    crack_active_passages: int = 0
    profile_active_passages: int = 0
    ballast_patch_rows: int = 0
    hanging_group_rows: int = 0
    pad_departure_passages: int = 0
    polygon_rows: int = 0
    witnesses: dict[str, str] = field(default_factory=dict)

    def witness(self, mechanism: str, location: str) -> None:
        self.witnesses.setdefault(mechanism, location)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _single_literal(path: Path, pattern: str, label: str) -> str:
    found = re.findall(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if len(found) != 1:
        raise QualificationInputError(
            f"cannot resolve exactly one {label} from {path}"
        )
    return str(found[0])


def _expected_schema() -> str:
    """Require MATLAB and Python to agree on the current generator schema."""
    matlab = _single_literal(
        _ROOT / "scour_MATLAB" / "A00_Run.m",
        r"^\s*gen_schema\s*=\s*'([^']+)';",
        "gen_schema",
    )
    if matlab != EXPECTED_GEN_SCHEMA:
        raise QualificationInputError(
            "repository schema sources disagree: "
            f"A00={matlab!r}, campaign_contract={EXPECTED_GEN_SCHEMA!r}"
        )
    return matlab


def _expected_behavior_version() -> str:
    matlab = _single_literal(
        _ROOT / "scour_MATLAB" / "A00_Run.m",
        r"^\s*generation_behavior_version\s*=\s*'([^']+)';",
        "generation_behavior_version",
    )
    if matlab != EXPECTED_GENERATION_BEHAVIOR_VERSION:
        raise QualificationInputError(
            "repository behavior-version sources disagree: "
            f"A00={matlab!r}, "
            f"campaign_contract={EXPECTED_GENERATION_BEHAVIOR_VERSION!r}"
        )
    return matlab


def _expected_max_parfor_workers() -> int:
    value = _single_literal(
        _ROOT / "scour_MATLAB" / "A00_Run.m",
        r"^\s*max_parfor_workers\s*=\s*(\d+);",
        "max_parfor_workers",
    )
    if int(value) <= 0:
        raise QualificationInputError("max_parfor_workers must be positive")
    return int(value)


def _source_float(name: str) -> float:
    value = _single_literal(
        _ROOT / "core" / "dataset.py",
        rf"^\s*{re.escape(name)}\s*=\s*([0-9.eE+-]+)",
        name,
    )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise QualificationInputError(f"{name} is not finite/nonnegative")
    return number


def _current_generator_identity() -> tuple[str, str, int]:
    try:
        identity = generator_source_root(_ROOT)
    except Exception as exc:
        raise QualificationInputError(
            f"cannot authenticate current generator source root: {exc}"
        ) from exc
    lines = "\n".join(
        f"{name}:{sha256(_ROOT.joinpath(*name.split('/')))}"
        for name in identity.files
    )
    calculated = _sha256_text(lines)
    if calculated != identity.sha256:
        raise QualificationInputError(
            "Python generator source-root helper is internally inconsistent"
        )
    return identity.sha256, lines, identity.file_count


def _current_parser_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _validate_parser_environment(
    spec: dict[str, Any],
    actual: dict[str, str] | None = None,
) -> dict[str, str]:
    """Authenticate the minimal non-GPU stack that parses qualification MATs."""
    observed = dict(actual or _current_parser_environment())
    if set(observed) != {"python", "platform_system", "numpy", "scipy"}:
        raise QualificationInputError(
            "qualification parser environment has the wrong field set"
        )
    packages = spec.get("packages")
    if not isinstance(packages, dict):
        raise QualificationInputError(
            "campaign lock lacks an exact package version map"
        )
    expected = {
        "python": spec.get("python"),
        "platform_system": spec.get("platform_system"),
        "numpy": packages.get("numpy"),
        "scipy": packages.get("scipy"),
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise QualificationInputError(
            "campaign lock lacks the minimal qualification parser versions"
        )
    mismatches = {
        key: (observed.get(key), expected[key])
        for key in expected
        if observed.get(key) != expected[key]
    }
    if mismatches:
        raise QualificationInputError(
            "qualification comparator is running outside the exact minimal "
            f"parser lock: {mismatches}"
        )
    return observed


def _current_policy() -> CurrentPolicy:
    if (
        not _ENVIRONMENT_LOCK.is_file()
        or _ENVIRONMENT_LOCK.is_symlink()
    ):
        raise QualificationInputError(
            "campaign environment lock must be one regular non-symlink file"
        )
    try:
        lock = load_environment_lock(_ENVIRONMENT_LOCK)
        environment = lock["spec"]["matlab_environment"]
        descriptor = matlab_environment_descriptor(environment)
        environment_sha = lock["spec"]["matlab_environment_sha256"]
    except Exception as exc:
        raise QualificationInputError(
            f"cannot authenticate current MATLAB environment lock: {exc}"
        ) from exc
    release = str(environment.get("release", ""))
    if not _MATLAB_RELEASE_RE.fullmatch(release):
        raise QualificationInputError(
            f"campaign lock has malformed MATLAB release {release!r}"
        )
    if _sha256_text(descriptor) != environment_sha:
        raise QualificationInputError(
            "campaign MATLAB descriptor SHA differs from the lock"
        )
    generator_sha, generator_lines, generator_count = (
        _current_generator_identity()
    )
    try:
        python_identity = python_runtime_source_root(_ROOT)
    except Exception as exc:
        raise QualificationInputError(
            f"cannot authenticate current Python runtime source root: {exc}"
        ) from exc
    parser_environment = _validate_parser_environment(lock["spec"])
    return CurrentPolicy(
        environment_lock_sha256=lock["sha256"],
        parser_environment=parser_environment,
        python_runtime_source_root_sha256=python_identity.sha256,
        python_runtime_source_file_count=python_identity.file_count,
        campaign_matlab_release=release,
        campaign_matlab_environment_descriptor=descriptor,
        campaign_matlab_environment_sha256=environment_sha,
        generator_source_root_sha256=generator_sha,
        generator_source_digest_lines=generator_lines,
        generator_source_file_count=generator_count,
        gen_schema=_expected_schema(),
        generation_behavior_version=_expected_behavior_version(),
        max_parfor_workers=_expected_max_parfor_workers(),
        contact_force_tolerance_N=_source_float("_CONTACT_F_TOL_N"),
        contact_fraction_tolerance=_source_float("_CONTACT_FRAC_TOL"),
    )


def _strict_json_object(text: str, owner: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationInputError(
            f"{owner} is not one strict JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QualificationInputError(f"{owner} must be one JSON object")
    return value


def _json_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return (
            math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and float(actual) == float(expected)
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_values_equal(left, right)
            for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_values_equal(actual[key], expected[key])
            for key in expected
        )
    return type(actual) is type(expected) and actual == expected


def _expected_qualification_generation_config(
    stage: str,
    policy: CurrentPolicy,
    qualification_sha256: str,
) -> dict[str, Any]:
    expected = generation_config_expectations(stage)
    micro = _STAGE_POLICIES[stage]
    expected.update(
        {
            "Npass": 3,
            "n_anchor_levels": 2,
            "n_healthy_states": 3,
            "n_anchor_reps": 2,
            "n_nuisance_states": 6,
            "n_states": micro["n_states"],
            "campaign_matlab_release": policy.campaign_matlab_release,
            "campaign_matlab_environment_sha256":
                policy.campaign_matlab_environment_sha256,
            "generator_source_root_sha256":
                policy.generator_source_root_sha256,
            "qualification_source_sha256": qualification_sha256,
        }
    )
    profile_asset = (
        _ROOT / "scour_MATLAB" / "Calc.ProfileData15_05.mat"
    )
    expected["profile_asset_sha256"] = (
        sha256(profile_asset)
        if profile_asset.is_file() and not profile_asset.is_symlink()
        else "ABSENT"
    )
    return expected


def _expected_qualification_source_sha256(stage: str) -> str:
    try:
        from make_micro_smoke import qualification_source_sha256

        digest = qualification_source_sha256(stage)
    except Exception as exc:
        raise QualificationInputError(
            "cannot recompute the current deterministic qualification-source "
            f"identity for {stage!r}: {exc}"
        ) from exc
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise QualificationInputError(
            "make_micro_smoke.qualification_source_sha256() did not return "
            "one lowercase SHA-256"
        )
    return digest


def _expected_qualification_source_bytes(stage: str) -> bytes:
    try:
        from make_micro_smoke import render_micro_a00

        source_path = _ROOT / "scour_MATLAB" / "A00_Run.m"
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError("A00_Run.m is not a regular source file")
        rendered = render_micro_a00(
            source_path.read_text(encoding="utf-8"),
            qualification=True,
            stage=stage,
        )
    except Exception as exc:
        raise QualificationInputError(
            "cannot reconstruct the exact current qualification executable "
            f"for {stage!r}: {exc}"
        ) from exc
    return rendered.encode("utf-8")


def _public_mat(path: Path) -> dict[str, Any]:
    try:
        loaded = loadmat(path, simplify_cells=True)
    except Exception as exc:  # scipy exceptions vary by corruption mode
        raise QualificationInputError(f"cannot read MAT file {path}: {exc}") from exc
    return {str(k): v for k, v in loaded.items() if not str(k).startswith("__")}


def _manifest(path: Path) -> dict[str, Any]:
    mat = _public_mat(path / "case_info.mat")
    value = mat.get("case_info")
    if not isinstance(value, dict):
        raise QualificationInputError(
            f"{path}: case_info.mat lacks a scalar case_info struct"
        )
    return value


def _scalar(value: Any, label: str) -> Any:
    array = np.asarray(value, dtype=object)
    if array.size != 1:
        raise QualificationInputError(f"{label} must be scalar, got shape {array.shape}")
    return array.reshape(-1)[0]


def _required_string(mapping: dict[str, Any], key: str, owner: str) -> str:
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    value = str(value).strip()
    if not value:
        raise QualificationInputError(f"{owner}.{key} must be a nonempty string")
    return value


def _required_exact_text(mapping: dict[str, Any], key: str, owner: str) -> str:
    """Read canonical text without silently trimming provenance bytes."""
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    value = str(value)
    if not value or value != value.strip() or "\r" in value:
        raise QualificationInputError(
            f"{owner}.{key} is not canonical nonempty LF text"
        )
    return value


def _required_sha(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = _required_string(mapping, key, owner)
    if not _SHA256_RE.fullmatch(value):
        raise QualificationInputError(
            f"{owner}.{key} must be one lowercase SHA-256"
        )
    return value


def _required_positive_int(mapping: dict[str, Any], key: str, owner: str) -> int:
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationInputError(f"{owner}.{key} is not numeric") from exc
    if not math.isfinite(number) or number <= 0 or number != int(number):
        raise QualificationInputError(
            f"{owner}.{key} must be a positive integer, got {value!r}"
        )
    return int(number)


def _qualification_host_text(
    mapping: dict[str, Any], key: str, owner: str
) -> str:
    value = _required_exact_text(mapping, key, owner)
    if "\n" in value or "\0" in value or len(value) > 1024:
        raise QualificationInputError(
            f"{owner}.{key} must be one nonempty canonical line (<=1024 chars)"
        )
    return value


def _load_qualification_host_receipt(
    dataset: Path,
    *,
    actual_environment_sha256: str,
    qualification_source_sha256: str,
    qualification_executed_file_sha256: str,
) -> dict[str, Any]:
    path = dataset / "qualification_host_receipt.json"
    if not path.is_file() or path.is_symlink():
        raise QualificationInputError(
            f"{dataset}: missing regular non-symlink "
            "qualification_host_receipt.json"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QualificationInputError(
            f"{path}: cannot read strict UTF-8 host receipt: {exc}"
        ) from exc
    if "\r" in raw or not raw.endswith("\n") or raw.endswith("\n\n"):
        raise QualificationInputError(
            f"{path}: host receipt must use canonical LF text with one final LF"
        )
    value = _strict_json_object(raw, str(path))
    observed_fields = set(value)
    if observed_fields != _QUALIFICATION_HOST_FIELDS:
        raise QualificationInputError(
            f"{path}: host-receipt field set mismatch; "
            f"missing={sorted(_QUALIFICATION_HOST_FIELDS - observed_fields)}, "
            f"extra={sorted(observed_fields - _QUALIFICATION_HOST_FIELDS)}"
        )
    schema = _qualification_host_text(value, "schema", str(path))
    if schema != _QUALIFICATION_HOST_SCHEMA:
        raise QualificationInputError(
            f"{path}: host-receipt schema {schema!r} is not "
            f"{_QUALIFICATION_HOST_SCHEMA!r}"
        )
    declared_host_id = _qualification_host_text(
        value, "declared_host_id", str(path)
    )
    if not _QUALIFICATION_HOST_ID_RE.fullmatch(declared_host_id):
        raise QualificationInputError(
            f"{path}: declared_host_id must match "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    hostname = _qualification_host_text(value, "hostname", str(path))
    cpu_identifier = _qualification_host_text(
        value, "cpu_identifier", str(path)
    )
    computer_arch = _qualification_host_text(value, "computer_arch", str(path))
    logical_processors = value["logical_processors"]
    matlab_max_threads = value["matlab_max_threads"]
    for field_name, number in (
        ("logical_processors", logical_processors),
        ("matlab_max_threads", matlab_max_threads),
    ):
        if type(number) is not int or number < 1:
            raise QualificationInputError(
                f"{path}.{field_name} must be one positive JSON integer"
            )
    actual_sha = _required_sha(
        value, "actual_matlab_environment_sha256", str(path)
    )
    qualification_sha = _required_sha(
        value, "qualification_source_sha256", str(path)
    )
    executable_sha = _required_sha(
        value, "qualification_executed_file_sha256", str(path)
    )
    expected_bindings = (
        (actual_sha, actual_environment_sha256, "actual MATLAB environment"),
        (
            qualification_sha,
            qualification_source_sha256,
            "qualification source",
        ),
        (
            executable_sha,
            qualification_executed_file_sha256,
            "qualification executable",
        ),
    )
    for observed, expected, label in expected_bindings:
        if observed != expected:
            raise QualificationInputError(
                f"{path}: host receipt is not bound to the authenticated "
                f"{label} ({observed} != {expected})"
            )
    descriptor = "\n".join(
        (
            f"schema={schema}",
            f"declared_host_id={declared_host_id}",
            f"hostname={hostname}",
            f"cpu_identifier={cpu_identifier}",
            f"logical_processors={logical_processors}",
            f"matlab_max_threads={matlab_max_threads}",
            f"computer_arch={computer_arch}",
            f"actual_matlab_environment_sha256={actual_sha}",
            f"qualification_source_sha256={qualification_sha}",
            f"qualification_executed_file_sha256={executable_sha}",
        )
    )
    observed_descriptor = _required_exact_text(
        value, "canonical_descriptor", str(path)
    )
    if observed_descriptor != descriptor:
        raise QualificationInputError(
            f"{path}: canonical_descriptor does not exactly encode the "
            "qualification host diagnostics"
        )
    diagnostic_sha = _required_sha(
        value, "host_diagnostic_sha256", str(path)
    )
    if _sha256_text(descriptor) != diagnostic_sha:
        raise QualificationInputError(
            f"{path}: host diagnostic descriptor/SHA mismatch"
        )
    return value


def _required_nonnegative_int(
    mapping: dict[str, Any], key: str, owner: str
) -> int:
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationInputError(f"{owner}.{key} is not numeric") from exc
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise QualificationInputError(
            f"{owner}.{key} must be a nonnegative integer, got {value!r}"
        )
    return int(number)


def _required_positive_float(mapping: dict[str, Any], key: str, owner: str) -> float:
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    try:
        number = float(_scalar(mapping[key], f"{owner}.{key}"))
    except (TypeError, ValueError) as exc:
        raise QualificationInputError(f"{owner}.{key} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise QualificationInputError(
            f"{owner}.{key} must be finite and positive"
        )
    return number


def _required_bool(mapping: dict[str, Any], key: str, owner: str) -> bool:
    if key not in mapping:
        raise QualificationInputError(f"{owner}: missing required field {key!r}")
    value = _scalar(mapping[key], f"{owner}.{key}")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number in (0.0, 1.0):
            return bool(int(number))
    raise QualificationInputError(
        f"{owner}.{key} must be a scalar logical/0/1, got {value!r}"
    )


def _parse_environment_descriptor(
    descriptor: str, owner: str, policy: CurrentPolicy
) -> dict[str, str]:
    expected_fields = tuple(
        line.split("=", 1)[0]
        for line in policy.campaign_matlab_environment_descriptor.splitlines()
    )
    raw_lines = descriptor.splitlines()
    if len(raw_lines) != len(expected_fields):
        raise QualificationInputError(
            f"{owner} has {len(raw_lines)} environment fields; "
            f"expected {len(expected_fields)}"
        )
    parsed: dict[str, str] = {}
    observed: list[str] = []
    for line in raw_lines:
        if "=" not in line:
            raise QualificationInputError(
                f"{owner} contains a malformed descriptor line {line!r}"
            )
        key, value = line.split("=", 1)
        if not key or not value or key in parsed:
            raise QualificationInputError(
                f"{owner} contains an empty/duplicate environment field"
            )
        parsed[key] = value
        observed.append(key)
    if tuple(observed) != expected_fields:
        raise QualificationInputError(
            f"{owner} fields/order {observed!r} != canonical {expected_fields!r}"
        )
    return parsed


def _validate_stage_policy(
    manifest: dict[str, Any],
    stage: str,
    owner: str,
) -> dict[str, Any]:
    expected = _STAGE_POLICIES.get(stage)
    if expected is None:
        raise QualificationInputError(
            f"{owner}.stage={stage!r} is not one of the ten qualification stages"
        )
    observed_states = _required_positive_int(manifest, "n_states", owner)
    if observed_states != expected["n_states"]:
        raise QualificationInputError(
            f"{owner}: {stage} requires exactly {expected['n_states']} states, "
            f"got {observed_states}"
        )
    observed_passages = _required_positive_int(
        manifest, "passages_per_state", owner
    )
    if observed_passages != 3:
        raise QualificationInputError(
            f"{owner}: release qualification requires exactly 3 passages/state"
        )
    for key in ("num_spans", "num_supports"):
        observed = _required_positive_int(manifest, key, owner)
        if observed != expected[key]:
            raise QualificationInputError(
                f"{owner}: {stage} requires {key}={expected[key]}, got {observed}"
            )
    bridge_length = _required_positive_float(manifest, "L_bridge_m", owner)
    if not math.isclose(
        bridge_length, expected["L_bridge_m"], rel_tol=0, abs_tol=1e-12
    ):
        raise QualificationInputError(
            f"{owner}: {stage} requires L_bridge_m={expected['L_bridge_m']}, "
            f"got {bridge_length}"
        )
    if _required_string(manifest, "bearing_mode", owner) != expected["bearing_mode"]:
        raise QualificationInputError(
            f"{owner}: {stage} has the wrong bearing_mode"
        )
    if _required_string(manifest, "profile_mode", owner) != expected["profile_mode"]:
        raise QualificationInputError(
            f"{owner}: {stage} has the wrong profile_mode"
        )
    for key in (
        "use_crack_eov",
        "use_track_eov",
        "use_oor_eov",
        "oor_flats_enabled",
    ):
        if _required_bool(manifest, key, owner) != expected[key]:
            raise QualificationInputError(
                f"{owner}: {stage} requires {key}={expected[key]}"
            )
    observed_families = {
        family: _required_nonnegative_int(
            manifest, manifest_field, owner
        )
        for family, manifest_field in _FAMILY_MANIFEST_FIELDS.items()
    }
    if observed_families != expected["family_counts"]:
        raise QualificationInputError(
            f"{owner}: {stage} family counts {observed_families} differ from "
            f"the exact qualification design {expected['family_counts']}"
        )
    if sum(observed_families.values()) != observed_states:
        raise QualificationInputError(
            f"{owner}: family counts do not sum to n_states"
        )
    return expected


def _read_digest_table(path: Path) -> tuple[dict[str, str], str]:
    digest_path = path / "file_digests.mat"
    if not digest_path.is_file() or digest_path.is_symlink():
        raise QualificationInputError(
            f"{path}: file_digests.mat must be a regular non-symlink file"
        )
    mat = _public_mat(digest_path)
    if set(mat) != {"file_digests"}:
        raise QualificationInputError(
            f"{path}: file_digests.mat must contain exactly file_digests"
        )
    value = mat.get("file_digests")
    if not isinstance(value, dict):
        raise QualificationInputError(
            f"{path}: file_digests.mat lacks a scalar file_digests struct"
        )
    digest_owner = f"{path}.file_digests"
    expected_fields = {"schema", "scope", "digest_lines", "root"}
    if set(value) != expected_fields:
        raise QualificationInputError(
            f"{path}: file_digests must contain exactly "
            f"{sorted(expected_fields)!r}"
        )
    if _required_exact_text(
        value, "schema", digest_owner
    ) != "source-digests-v2":
        raise QualificationInputError(
            f"{path}: unsupported dataset digest-table schema"
        )
    if (
        _required_exact_text(value, "scope", digest_owner)
        != "NNNN.mat+case_info.mat+damage_states.mat"
    ):
        raise QualificationInputError(
            f"{path}: unsupported/incomplete dataset digest-table scope"
        )
    lines = _required_exact_text(value, "digest_lines", digest_owner)
    root = _required_sha(value, "root", f"{path}.file_digests")

    raw_lines = lines.split("\n")
    if not raw_lines or any(not raw for raw in raw_lines):
        raise QualificationInputError(
            f"{path}: digest_lines contains an empty/noncanonical row"
        )
    per_file: dict[str, str] = {}
    folded_names: set[str] = set()
    observed_names: list[str] = []
    for raw in raw_lines:
        if raw.count(":") != 1:
            raise QualificationInputError(
                f"{path}: malformed digest line: {raw!r}"
            )
        name, digest = raw.split(":", 1)
        folded = name.casefold()
        if (
            not name
            or name in per_file
            or folded in folded_names
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name != name.strip()
        ):
            raise QualificationInputError(
                f"{path}: unsafe, duplicate or case-colliding digest "
                f"filename {name!r}"
            )
        if not _SHA256_RE.fullmatch(digest):
            raise QualificationInputError(
                f"{path}: invalid SHA-256 for {name!r}: {digest!r}"
            )
        per_file[name] = digest
        folded_names.add(folded)
        observed_names.append(name)

    canonical = "\n".join(f"{name}:{per_file[name]}" for name in sorted(per_file))
    if observed_names != sorted(observed_names) or lines != canonical:
        raise QualificationInputError(
            f"{path}: digest_lines is not exact lowercase sorted canonical text"
        )
    calculated_root = _sha256_text(canonical)
    if calculated_root != root:
        raise QualificationInputError(
            f"{path}: digest table is internally inconsistent "
            f"({calculated_root} != {root})"
        )
    return per_file, root


def _validate_dataset_header(
    path: Path, policy: CurrentPolicy
) -> tuple[DatasetEvidence, dict[str, Any], dict[str, Any]]:
    if not path.is_dir():
        raise QualificationInputError(f"not a directory: {path}")
    if path.is_symlink():
        raise QualificationInputError(f"dataset root may not be a symlink: {path}")

    manifest = _manifest(path)
    owner = str(path / "case_info.mat")
    missing = sorted(_REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise QualificationInputError(f"{owner}: missing required fields {missing}")

    schema = _required_string(manifest, "gen_schema", owner)
    if schema != policy.gen_schema:
        raise QualificationInputError(
            f"{path}: gen_schema={schema!r}, current policy expects "
            f"{policy.gen_schema!r}"
        )
    behavior = _required_string(manifest, "generation_behavior_version", owner)
    if behavior != policy.generation_behavior_version:
        raise QualificationInputError(
            f"{path}: generation_behavior_version={behavior!r}, current policy "
            f"expects {policy.generation_behavior_version!r}"
        )
    fingerprint = _required_sha(manifest, "gen_fingerprint", owner)
    stage = _required_string(manifest, "stage", owner)
    stage_policy = _validate_stage_policy(manifest, stage, owner)

    release = _required_string(manifest, "matlab_release", owner)
    if not _MATLAB_RELEASE_RE.fullmatch(release):
        raise QualificationInputError(
            f"{path}: malformed matlab_release {release!r}; expected R####a/b"
        )
    campaign_release = _required_string(
        manifest, "campaign_matlab_release", owner
    )
    if campaign_release != policy.campaign_matlab_release:
        raise QualificationInputError(
            f"{path}: input-declared campaign release {campaign_release!r} "
            f"does not match current locked policy "
            f"{policy.campaign_matlab_release!r}"
        )

    actual_descriptor = _required_exact_text(
        manifest, "actual_matlab_environment_descriptor", owner
    )
    actual_sha = _required_sha(
        manifest, "actual_matlab_environment_sha256", owner
    )
    actual_environment = _parse_environment_descriptor(
        actual_descriptor,
        f"{owner}.actual_matlab_environment_descriptor",
        policy,
    )
    if _sha256_text(actual_descriptor) != actual_sha:
        raise QualificationInputError(
            f"{path}: actual MATLAB environment descriptor/SHA mismatch"
        )
    if actual_environment["release"] != release:
        raise QualificationInputError(
            f"{path}: matlab_release differs from the authenticated actual "
            "environment descriptor"
        )

    campaign_descriptor = _required_exact_text(
        manifest, "campaign_matlab_environment_descriptor", owner
    )
    campaign_sha = _required_sha(
        manifest, "campaign_matlab_environment_sha256", owner
    )
    if (
        campaign_descriptor
        != policy.campaign_matlab_environment_descriptor
        or campaign_sha != policy.campaign_matlab_environment_sha256
        or _sha256_text(campaign_descriptor) != campaign_sha
    ):
        raise QualificationInputError(
            f"{path}: input-declared campaign MATLAB environment is not the "
            "current authenticated repository policy"
        )

    generator_sha = _required_sha(
        manifest, "generator_source_root_sha256", owner
    )
    generator_lines = _required_exact_text(
        manifest, "generator_source_digest_lines", owner
    )
    generator_count = _required_positive_int(
        manifest, "generator_source_file_count", owner
    )
    if (
        generator_sha != policy.generator_source_root_sha256
        or generator_lines != policy.generator_source_digest_lines
        or generator_count != policy.generator_source_file_count
        or _sha256_text(generator_lines) != generator_sha
        or len(generator_lines.splitlines()) != generator_count
    ):
        raise QualificationInputError(
            f"{path}: qualification evidence is not bound to the current "
            "reviewed generator source bytes"
        )

    qualification_sha = _required_sha(
        manifest, "qualification_source_sha256", owner
    )
    expected_qualification_sha = _expected_qualification_source_sha256(stage)
    if qualification_sha != expected_qualification_sha:
        raise QualificationInputError(
            f"{path}: qualification_source_sha256={qualification_sha} does not "
            f"match the current deterministic {stage} template "
            f"{expected_qualification_sha}"
        )
    qualification_evidence_path = path / "qualification_executed.m"
    if (
        not qualification_evidence_path.is_file()
        or qualification_evidence_path.is_symlink()
    ):
        raise QualificationInputError(
            f"{path}: missing regular non-symlink qualification_executed.m"
        )
    expected_qualification_bytes = _expected_qualification_source_bytes(stage)
    qualification_evidence_bytes = qualification_evidence_path.read_bytes()
    if qualification_evidence_bytes != expected_qualification_bytes:
        raise QualificationInputError(
            f"{path}: qualification_executed.m is not the exact current "
            f"{stage} executable produced by make_micro_smoke.py"
        )
    qualification_executed_file_sha = hashlib.sha256(
        qualification_evidence_bytes
    ).hexdigest()
    qualification_host = _load_qualification_host_receipt(
        path,
        actual_environment_sha256=actual_sha,
        qualification_source_sha256=qualification_sha,
        qualification_executed_file_sha256=qualification_executed_file_sha,
    )
    config_json = _required_exact_text(
        manifest, "generation_config_json", owner
    )
    config_sha = _sha256_text(config_json)
    if config_sha != fingerprint:
        raise QualificationInputError(
            f"{path}: generation_config_json hashes to {config_sha}, not "
            f"gen_fingerprint={fingerprint}"
        )
    generation_config = _strict_json_object(
        config_json, f"{owner}.generation_config_json"
    )
    expected_config = _expected_qualification_generation_config(
        stage, policy, qualification_sha
    )
    expected_keys = set(expected_config) | set(_GENERATION_REALIZATION_FIELDS)
    actual_keys = set(generation_config)
    if actual_keys != expected_keys:
        raise QualificationInputError(
            f"{path}: hashed generation configuration has the wrong exact "
            f"field set; missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    config_mismatches = {
        field: (generation_config.get(field), wanted)
        for field, wanted in expected_config.items()
        if not _json_values_equal(generation_config.get(field), wanted)
    }
    if config_mismatches:
        raise QualificationInputError(
            f"{path}: hashed generation configuration disagrees with the "
            f"current exact {stage} qualification contract: "
            f"{config_mismatches}"
        )
    crn_manifest_fields = (
        "state_identity_version",
        "joint_lhs_design",
        "n_latent_bearing_dims",
    )
    missing_crn_contract = [
        field for field in crn_manifest_fields if field not in expected_config
    ]
    if missing_crn_contract:
        raise QualificationInputError(
            "current shared campaign contract does not define the complete "
            f"strong-CRN policy: missing={missing_crn_contract}"
        )
    for field in ("state_identity_version", "joint_lhs_design"):
        observed = _required_string(manifest, field, owner)
        if not _json_values_equal(observed, expected_config[field]):
            raise QualificationInputError(
                f"{path}: manifest {field}={observed!r} disagrees with the "
                f"hashed/current value {expected_config[field]!r}"
            )
    latent_dims = _required_positive_int(
        manifest, "n_latent_bearing_dims", owner
    )
    if not _json_values_equal(
        latent_dims, expected_config["n_latent_bearing_dims"]
    ):
        raise QualificationInputError(
            f"{path}: manifest n_latent_bearing_dims={latent_dims} disagrees "
            f"with the hashed/current value "
            f"{expected_config['n_latent_bearing_dims']!r}"
        )
    if not _required_bool(manifest, "release_qualification_run", owner):
        raise QualificationInputError(
            f"{path}: release_qualification_run is false; only deliberately "
            "marked qualification output may enter this comparison"
        )

    n_states = int(stage_policy["n_states"])
    n_passages = 3
    max_workers = _required_positive_int(manifest, "max_parfor_workers", owner)
    if max_workers != policy.max_parfor_workers:
        raise QualificationInputError(
            f"{path}: max_parfor_workers={max_workers} does not match current "
            f"reviewed value {policy.max_parfor_workers}"
        )
    num_spans = int(stage_policy["num_spans"])
    num_supports = int(stage_policy["num_supports"])
    if num_supports != num_spans + 1:
        raise QualificationInputError(
            f"{path}: num_supports={num_supports} must equal "
            f"num_spans+1={num_spans + 1}"
        )

    state_candidates = tuple(
        sorted(
            item.name
            for item in path.iterdir()
            if re.fullmatch(r"\d{4}\.mat", item.name, flags=re.IGNORECASE)
        )
    )
    expected_names = tuple(f"{index:04d}.mat" for index in range(1, n_states + 1))
    if state_candidates != expected_names:
        raise QualificationInputError(
            f"{path}: state inventory is not exactly 0001.mat..{n_states:04d}.mat "
            f"(found {len(state_candidates)} case-insensitive candidates)"
        )
    state_names = expected_names

    per_file, dataset_root = _read_digest_table(path)
    required_digest_names = set(state_names) | {"case_info.mat", "damage_states.mat"}
    if set(per_file) != required_digest_names:
        missing_digest = sorted(required_digest_names - set(per_file))
        extra_digest = sorted(set(per_file) - required_digest_names)
        raise QualificationInputError(
            f"{path}: digest inventory mismatch; missing={missing_digest}, "
            f"extra={extra_digest}"
        )
    for name, expected_digest in sorted(per_file.items()):
        target = path / name
        if not target.is_file() or target.is_symlink():
            raise QualificationInputError(
                f"{path}: digested entry is missing, non-regular or symlinked: {name}"
            )
        actual = sha256(target)
        if actual != expected_digest:
            raise QualificationInputError(
                f"{path}: SHA-256 mismatch for {name}: {actual} != {expected_digest}"
            )

    marker = path / "_GENERATION_COMPLETE"
    if not marker.is_file() or marker.is_symlink():
        raise QualificationInputError(f"{path}: missing regular completion marker")
    expected_marker = f"{schema}\n{fingerprint}\n{dataset_root}\n"
    if marker.read_text(encoding="utf-8") != expected_marker:
        raise QualificationInputError(
            f"{path}: completion marker does not exactly bind "
            "schema/fingerprint/dataset-content-root in canonical three-line form"
        )

    empty_coverage = MechanismCoverage(
        required=stage in _MECHANISM_COVERAGE_STAGES,
        crack_active_passages=0,
        profile_active_passages=0,
        ballast_patch_rows=0,
        hanging_group_rows=0,
        pad_departure_passages=0,
        polygon_rows=0,
        witnesses={},
    )
    evidence = DatasetEvidence(
        path=str(path.resolve()),
        stage=stage,
        matlab_release=release,
        actual_matlab_environment_descriptor=actual_descriptor,
        actual_matlab_environment_sha256=actual_sha,
        campaign_matlab_release=campaign_release,
        campaign_matlab_environment_descriptor=campaign_descriptor,
        campaign_matlab_environment_sha256=campaign_sha,
        gen_schema=schema,
        generation_behavior_version=behavior,
        gen_fingerprint=fingerprint,
        generator_source_root_sha256=generator_sha,
        generator_source_file_count=generator_count,
        qualification_source_sha256=qualification_sha,
        qualification_executed_file_sha256=qualification_executed_file_sha,
        qualification_host_id=qualification_host["declared_host_id"],
        qualification_hostname=qualification_host["hostname"],
        qualification_cpu_identifier=qualification_host["cpu_identifier"],
        qualification_logical_processors=qualification_host[
            "logical_processors"
        ],
        qualification_matlab_max_threads=qualification_host[
            "matlab_max_threads"
        ],
        qualification_computer_arch=qualification_host["computer_arch"],
        qualification_host_canonical_descriptor=qualification_host[
            "canonical_descriptor"
        ],
        qualification_host_diagnostic_sha256=qualification_host[
            "host_diagnostic_sha256"
        ],
        n_states=n_states,
        passages_per_state=n_passages,
        num_supports=num_supports,
        max_parfor_workers=max_workers,
        dataset_content_root_sha256=dataset_root,
        state_files=state_names,
        mechanism_coverage=empty_coverage,
    )
    return evidence, manifest, generation_config


def _finite_numeric_array(
    value: Any, label: str, *, allow_empty: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or (array.size == 0 and not allow_empty):
        raise QualificationInputError(f"{label} must be a finite numerical array")
    if not np.all(np.isfinite(array)):
        raise QualificationInputError(f"{label} contains non-finite values")
    return array


def _vector(
    value: Any,
    label: str,
    *,
    length: int | None = None,
    integer: bool = False,
    positive: bool = False,
) -> np.ndarray:
    array = _finite_numeric_array(value, label).astype(
        np.float64, copy=False
    ).reshape(-1)
    if length is not None and array.size != length:
        raise QualificationInputError(
            f"{label} has {array.size} values, expected {length}"
        )
    if integer and not np.all(array == np.round(array)):
        raise QualificationInputError(f"{label} must be integer-valued")
    if positive and not np.all(array > 0):
        raise QualificationInputError(f"{label} must be positive")
    return array


def _canonical_text_vector(
    value: Any, label: str, *, length: int | None = None
) -> tuple[str, ...]:
    values = np.asarray(value, dtype=object).reshape(-1)
    result: list[str] = []
    for index, item in enumerate(values):
        scalar = _scalar(item, f"{label}[{index}]")
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8", errors="strict")
        text = str(scalar)
        if (
            not text
            or text != text.strip()
            or "\r" in text
            or "\n" in text
            or "\0" in text
        ):
            raise QualificationInputError(
                f"{label}[{index}] is not one canonical text value"
            )
        result.append(text)
    if length is not None and len(result) != length:
        raise QualificationInputError(
            f"{label} has {len(result)} values, expected {length}"
        )
    return tuple(result)


def _expected_named_seed(
    schedule: str,
    root_seed: int,
    state_uid: str,
    stream_name: str,
    *,
    passage: int | None = None,
) -> int:
    key = (
        f"{schedule}|root={root_seed}|uid={state_uid}|stream={stream_name}"
    )
    if passage is not None:
        key += f"|pass={passage:05d}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _matrix(
    value: Any, label: str, rows: int, columns: int
) -> np.ndarray:
    array = _finite_numeric_array(value, label).astype(
        np.float64, copy=False
    )
    if array.size != rows * columns:
        raise QualificationInputError(
            f"{label} has {array.size} values; expected {rows}x{columns}"
        )
    array = array.reshape(rows, columns)
    return array


def _row_matrix(value: Any, label: str, columns: int) -> np.ndarray:
    array = _finite_numeric_array(value, label, allow_empty=True).astype(
        np.float64, copy=False
    )
    if array.size == 0:
        return np.empty((0, columns), dtype=float)
    if array.size % columns:
        raise QualificationInputError(
            f"{label} has {array.size} values, not rows of {columns}"
        )
    return array.reshape(-1, columns)


def _passages(value: Any, n_passages: int, label: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        passages = list(value)
    else:
        array = np.asarray(value)
        if array.dtype == object:
            passages = list(array.flat)
        elif n_passages == 1:
            passages = [value]
        else:
            raise QualificationInputError(
                f"{label} must contain {n_passages} MATLAB-cell passages"
            )
    if len(passages) != n_passages:
        raise QualificationInputError(
            f"{label} has {len(passages)} passages, expected {n_passages}"
        )
    return passages


def _validate_signal_passages(
    value: Any,
    label: str,
    n_passages: int,
    expected_rows: int,
    dim_acel: np.ndarray,
) -> int:
    passages = _passages(value, n_passages, label)
    total = 0
    for index, passage in enumerate(passages):
        array = np.asarray(passage)
        if array.dtype != np.dtype("float64") or array.size == 0:
            raise QualificationInputError(
                f"{label}[{index}] must be a nonempty MATLAB-double array"
            )
        expected_shape = (expected_rows, int(dim_acel[index]))
        if array.ndim != 2 or array.shape != expected_shape:
            raise QualificationInputError(
                f"{label}[{index}] has shape {array.shape}; expected "
                f"{expected_shape} from the RAW metadata"
            )
        if not np.all(np.isfinite(array)):
            raise QualificationInputError(
                f"{label}[{index}] contains non-finite values"
            )
        total += int(array.size)
    return total


def _validate_damage_table(
    table: dict[str, Any],
    evidence: DatasetEvidence,
    manifest: dict[str, Any],
) -> DamageLabels:
    required = {
        "StateFamily",
        "AnchorTarget",
        "AnchorLevel",
        "CrackOn",
        "DamageStates",
        "BearingStates",
        "BearingFixity",
        "LatentBearingFixity",
        "StateUID",
        "StateSeedID",
        "StateNamedStreamSeedID",
        "PassageNamedStreamSeedID",
        "PassageNamedStreamSeedIDFlat",
        "random_stream_schedule_version",
        "state_stream_names",
        "passage_stream_names",
        "LatentCrackOn",
        "k_ref_bear",
        "scour_supports",
    }
    missing = sorted(required - set(table))
    owner = f"{evidence.path}/damage_states.mat"
    if missing:
        raise QualificationInputError(f"{owner}: missing fields {missing}")

    supports = _vector(
        table["scour_supports"],
        f"{owner}.scour_supports",
        integer=True,
        positive=True,
    )
    if len(set(int(value) for value in supports)) != supports.size:
        raise QualificationInputError(f"{owner}.scour_supports contains duplicates")

    n_states = evidence.n_states
    families = tuple(
        str(item) for item in np.asarray(table["StateFamily"]).reshape(-1)
    )
    if len(families) != n_states:
        raise QualificationInputError(
            f"{owner}: StateFamily has {len(families)} rows, expected {n_states}"
        )
    unknown = sorted(set(families) - _STATE_FAMILIES)
    if unknown:
        raise QualificationInputError(f"{owner}: unknown StateFamily {unknown}")
    expected_family_counts = _STAGE_POLICIES[evidence.stage]["family_counts"]
    observed_family_counts = {
        family: Counter(families).get(family, 0)
        for family in _FAMILY_MANIFEST_FIELDS
    }
    if observed_family_counts != expected_family_counts:
        raise QualificationInputError(
            f"{owner}: family counts {observed_family_counts} differ from exact "
            f"{evidence.stage} qualification design {expected_family_counts}"
        )

    anchor_target = _vector(
        table["AnchorTarget"],
        f"{owner}.AnchorTarget",
        length=n_states,
        integer=True,
    )
    anchor_level = _vector(
        table["AnchorLevel"],
        f"{owner}.AnchorLevel",
        length=n_states,
        integer=True,
    )
    if np.any(anchor_target < 0) or np.any(anchor_level < 0):
        raise QualificationInputError(
            f"{owner}: AnchorTarget/AnchorLevel must be nonnegative integers"
        )
    crack_values = _vector(
        table["CrackOn"], f"{owner}.CrackOn", length=n_states
    )
    if not np.all(np.isin(crack_values, (0, 1))):
        raise QualificationInputError(f"{owner}.CrackOn is not logical/0/1")
    crack_on = crack_values.astype(bool)
    latent_crack_values = _vector(
        table["LatentCrackOn"],
        f"{owner}.LatentCrackOn",
        length=n_states,
    )
    if not np.all(np.isin(latent_crack_values, (0, 1))):
        raise QualificationInputError(
            f"{owner}.LatentCrackOn is not logical/0/1"
        )
    latent_crack_on = latent_crack_values.astype(bool)
    state_uids = tuple(
        str(item) for item in np.asarray(table["StateUID"]).reshape(-1)
    )
    if (
        len(state_uids) != n_states
        or len(set(state_uids)) != n_states
        or any(not uid or uid != uid.strip() or "\r" in uid for uid in state_uids)
    ):
        raise QualificationInputError(
            f"{owner}.StateUID must contain {n_states} unique canonical strings"
        )
    state_seed_raw = np.asarray(table["StateSeedID"])
    state_seed_ids = _vector(
        state_seed_raw,
        f"{owner}.StateSeedID",
        length=n_states,
        integer=True,
    )
    if (
        state_seed_raw.dtype != np.dtype("uint32")
        or np.any(state_seed_ids < 1)
        or np.any(state_seed_ids > np.iinfo(np.uint32).max)
        or len(set(int(value) for value in state_seed_ids)) != n_states
    ):
        raise QualificationInputError(
            f"{owner}.StateSeedID must contain {n_states} unique nonzero "
            "MATLAB-uint32 values"
        )
    schedule = _required_string(
        table, "random_stream_schedule_version", owner
    )
    state_stream_names = _canonical_text_vector(
        table["state_stream_names"],
        f"{owner}.state_stream_names",
        length=len(_STATE_STREAM_NAMES),
    )
    passage_stream_names = _canonical_text_vector(
        table["passage_stream_names"],
        f"{owner}.passage_stream_names",
        length=len(_PASSAGE_STREAM_NAMES),
    )
    if (
        schedule != _RANDOM_STREAM_SCHEDULE
        or state_stream_names != _STATE_STREAM_NAMES
        or passage_stream_names != _PASSAGE_STREAM_NAMES
    ):
        raise QualificationInputError(
            f"{owner}: foreign named-substream schedule/names "
            f"({schedule!r}, {state_stream_names!r}, "
            f"{passage_stream_names!r})"
        )
    state_named_raw = np.asarray(table["StateNamedStreamSeedID"])
    passage_named_raw = np.asarray(table["PassageNamedStreamSeedID"])
    passage_flat_raw = np.asarray(table["PassageNamedStreamSeedIDFlat"])
    expected_state_shape = (n_states, len(state_stream_names))
    expected_passage_shape = (
        n_states,
        evidence.passages_per_state,
        len(passage_stream_names),
    )
    expected_flat_shape = (
        n_states,
        evidence.passages_per_state * len(passage_stream_names),
    )
    if (
        state_named_raw.dtype != np.dtype("uint32")
        or passage_named_raw.dtype != np.dtype("uint32")
        or passage_flat_raw.dtype != np.dtype("uint32")
        or state_named_raw.shape != expected_state_shape
        or passage_named_raw.shape != expected_passage_shape
        or passage_flat_raw.shape != expected_flat_shape
        or np.any(state_named_raw == 0)
        or np.any(passage_named_raw == 0)
        or not np.array_equal(
            passage_flat_raw,
            passage_named_raw.reshape(n_states, -1, order="F"),
        )
    ):
        raise QualificationInputError(
            f"{owner}: named RNG stream matrices are not nonzero uint32 "
            "arrays with the registered state/passage/flattened shapes"
        )
    expected_state_named = np.empty(expected_state_shape, dtype=np.uint32)
    expected_passage_named = np.empty(
        expected_passage_shape, dtype=np.uint32
    )
    for row, (uid, root_seed) in enumerate(
        zip(state_uids, state_seed_ids, strict=True)
    ):
        for stream_index, stream_name in enumerate(state_stream_names):
            expected_state_named[row, stream_index] = _expected_named_seed(
                schedule, int(root_seed), uid, stream_name
            )
        for passage_index in range(evidence.passages_per_state):
            for stream_index, stream_name in enumerate(
                passage_stream_names
            ):
                expected_passage_named[
                    row, passage_index, stream_index
                ] = _expected_named_seed(
                    schedule,
                    int(root_seed),
                    uid,
                    stream_name,
                    passage=passage_index + 1,
                )
    all_seed_ids = np.concatenate(
        (
            state_seed_ids.astype(np.uint32),
            state_named_raw.reshape(-1),
            passage_named_raw.reshape(-1),
        )
    )
    if (
        not np.array_equal(state_named_raw, expected_state_named)
        or not np.array_equal(passage_named_raw, expected_passage_named)
        or np.unique(all_seed_ids).size != all_seed_ids.size
    ):
        raise QualificationInputError(
            f"{owner}: named RNG streams are misderived or collide across "
            "root/state/passage namespaces"
        )

    damage = _matrix(
        table["DamageStates"],
        f"{owner}.DamageStates",
        n_states,
        evidence.num_supports,
    )
    bearing_states = _matrix(
        table["BearingStates"], f"{owner}.BearingStates", n_states, 2
    )
    bearing_fixity = _matrix(
        table["BearingFixity"], f"{owner}.BearingFixity", n_states, 2
    )
    latent_bearing_fixity = _matrix(
        table["LatentBearingFixity"],
        f"{owner}.LatentBearingFixity",
        n_states,
        2,
    )
    k_ref_bear = _required_positive_float(table, "k_ref_bear", owner)
    manifest_owner = str(Path(evidence.path) / "case_info.mat")
    manifest_k_ref = _required_positive_float(
        manifest, "bearing_k_ref_Nm_rad", manifest_owner
    )
    if k_ref_bear != manifest_k_ref:
        raise QualificationInputError(
            f"{owner}.k_ref_bear does not exactly match the manifest"
        )
    if (
        np.any(damage < 0)
        or np.any(damage > 1)
        or np.any(bearing_states < 0)
        or np.any(bearing_fixity < 0)
        or np.any(bearing_fixity > 1)
        or np.any(latent_bearing_fixity < 0)
        or np.any(latent_bearing_fixity > 1)
    ):
        raise QualificationInputError(
            f"{owner}: damage/fixity labels are outside their physical range"
        )
    if np.any(supports <= 1) or np.any(supports >= evidence.num_supports):
        raise QualificationInputError(
            f"{owner}.scour_supports must index internal piers of the "
            f"{evidence.num_supports}-support bridge"
        )
    expected_supports = np.asarray(
        _STAGE_POLICIES[evidence.stage]["scour_supports"],
        dtype=np.int64,
    )
    if not np.array_equal(supports.astype(np.int64), expected_supports):
        raise QualificationInputError(
            f"{owner}.scour_supports must be exactly "
            f"{expected_supports.tolist()} for {evidence.stage}"
        )
    non_targets = np.ones(evidence.num_supports, dtype=bool)
    non_targets[supports.astype(int) - 1] = False
    if np.any(damage[:, non_targets] != 0):
        raise QualificationInputError(
            f"{owner}.DamageStates contains scour outside declared target piers"
        )
    stage_policy = _STAGE_POLICIES[evidence.stage]
    if stage_policy["bearing_mode"] == "target":
        if not np.array_equal(bearing_fixity, latent_bearing_fixity):
            raise QualificationInputError(
                f"{owner}: active bearing fixity must exactly equal its latent "
                "CRN realization on bearing-target rungs"
            )
    elif np.any(bearing_states != 0) or np.any(bearing_fixity != 0):
        raise QualificationInputError(
            f"{owner}: bearing-off rung contains active bearing damage"
        )
    families_array = np.asarray(families)
    expected_crack_on = np.zeros(n_states, dtype=bool)
    if stage_policy["use_crack_eov"]:
        joint = families_array == "joint"
        nuisance = families_array == "nuisance_only"
        expected_crack_on[joint] = latent_crack_on[joint]
        expected_crack_on[nuisance] = True
    if not np.array_equal(crack_on, expected_crack_on):
        raise QualificationInputError(
            f"{owner}.CrackOn does not follow the exact active/latent "
            f"{evidence.stage} crack policy"
        )
    return DamageLabels(
        damage_states=damage,
        bearing_states=bearing_states,
        bearing_fixity=bearing_fixity,
        state_families=families,
        anchor_target=anchor_target.astype(np.int64),
        anchor_level=anchor_level.astype(np.int64),
        state_uids=state_uids,
        state_seed_ids=state_seed_ids.astype(np.uint32),
        random_stream_schedule_version=schedule,
        state_stream_names=state_stream_names,
        passage_stream_names=passage_stream_names,
        state_named_stream_seed_ids=state_named_raw,
        passage_named_stream_seed_ids=passage_named_raw,
        passage_named_stream_seed_ids_flat=passage_flat_raw,
        latent_bearing_fixity=latent_bearing_fixity,
        latent_crack_on=latent_crack_on,
        scour_supports=supports.astype(np.int64),
        crack_on=crack_on,
    )


def _validate_generation_realizations(
    generation_config: dict[str, Any],
    labels: DamageLabels,
    evidence: DatasetEvidence,
) -> None:
    """Bind the hashed JSON realization arrays to damage_states.mat exactly."""
    owner = f"{evidence.path}/case_info.mat.generation_config_json"
    expected = {
        "DamageStates": labels.damage_states.tolist(),
        "BearingStates": labels.bearing_states.tolist(),
        "BearingFixity": labels.bearing_fixity.tolist(),
        "StateFamily": list(labels.state_families),
        "AnchorTarget": labels.anchor_target.tolist(),
        "AnchorLevel": labels.anchor_level.tolist(),
        "StateUID": list(labels.state_uids),
        "StateSeedID": [
            int(value) for value in labels.state_seed_ids
        ],
        "LatentBearingFixity": labels.latent_bearing_fixity.tolist(),
        "LatentCrackOn": labels.latent_crack_on.tolist(),
        "CrackOn": labels.crack_on.tolist(),
        "StateNamedStreamSeedID":
            labels.state_named_stream_seed_ids.tolist(),
        "PassageNamedStreamSeedIDFlat":
            labels.passage_named_stream_seed_ids_flat.tolist(),
    }
    mismatches = {
        field: (generation_config.get(field), wanted)
        for field, wanted in expected.items()
        if not _json_values_equal(generation_config.get(field), wanted)
    }
    if mismatches:
        raise QualificationInputError(
            f"{owner}: hashed realization fields do not exactly reproduce "
            f"damage_states.mat: {mismatches}"
        )
    damage_seed = generation_config.get("damage_seed")
    if (
        isinstance(damage_seed, bool)
        or not isinstance(damage_seed, (int, float))
        or not math.isfinite(float(damage_seed))
        or float(damage_seed) < 0
        or float(damage_seed) != int(float(damage_seed))
    ):
        raise QualificationInputError(
            f"{owner}.damage_seed must be one nonnegative integer"
        )
    expected_seed_ids = np.asarray(
        [
            int(
                hashlib.sha256(
                    (
                        "ttbi-state-seed-v1|damage_seed="
                        f"{int(float(damage_seed))}|{uid}"
                    ).encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            for uid in labels.state_uids
        ],
        dtype=np.uint32,
    )
    if not np.array_equal(labels.state_seed_ids, expected_seed_ids):
        raise QualificationInputError(
            f"{owner}: StateSeedID is not the reviewed SHA-256 derivation of "
            "StateUID and damage_seed"
        )
    crack_p = generation_config.get("crack_p")
    if (
        isinstance(crack_p, bool)
        or not isinstance(crack_p, (int, float))
        or not math.isfinite(float(crack_p))
        or not 0 <= float(crack_p) <= 1
    ):
        raise QualificationInputError(
            f"{owner}.crack_p must be finite and in [0,1]"
        )
    uid_hash_latent_crack = np.asarray(
        [
            (
                int(
                    hashlib.sha256(
                        (
                            "latent-crack-v1|damage_seed="
                            f"{int(float(damage_seed))}|{uid}"
                        ).encode("utf-8")
                    ).hexdigest()[:13],
                    16,
                )
                / 16**13
            )
            <= float(crack_p)
            for uid in labels.state_uids
        ],
        dtype=bool,
    )
    families = np.asarray(labels.state_families)
    expected_latent_crack = np.zeros(len(labels.state_families), dtype=bool)
    expected_latent_crack[families == "joint"] = uid_hash_latent_crack[
        families == "joint"
    ]
    expected_latent_crack[families == "nuisance_only"] = True
    if not np.array_equal(labels.latent_crack_on, expected_latent_crack):
        raise QualificationInputError(
            f"{owner}: LatentCrackOn is not the reviewed controlled-family "
            "policy (anchors off, nuisance on, joint UID-keyed SHA-256)"
        )


def _validate_state_provenance(
    loaded: dict[str, Any],
    evidence: DatasetEvidence,
    manifest: dict[str, Any],
    labels: DamageLabels,
    state_name: str,
    state_index: int,
    policy: CurrentPolicy,
) -> dict[str, Any]:
    owner = f"{evidence.path}/{state_name}"
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(loaded))
    if missing:
        raise QualificationInputError(f"{owner}: missing top-level fields {missing}")

    top_expected_strings = {
        "file_gen_schema": evidence.gen_schema,
        "file_gen_fingerprint": evidence.gen_fingerprint,
        "file_matlab_release": evidence.matlab_release,
        "file_campaign_matlab_release": evidence.campaign_matlab_release,
        "file_random_stream_schedule_version":
            labels.random_stream_schedule_version,
        "file_actual_matlab_environment_sha256":
            evidence.actual_matlab_environment_sha256,
        "file_campaign_matlab_environment_sha256":
            evidence.campaign_matlab_environment_sha256,
        "file_generator_source_root_sha256":
            evidence.generator_source_root_sha256,
        "file_qualification_source_sha256":
            evidence.qualification_source_sha256,
    }
    for key, expected in top_expected_strings.items():
        if _required_string(loaded, key, owner) != expected:
            raise QualificationInputError(f"{owner}: {key} mismatch")
    if (
        _required_exact_text(loaded, "file_state_uid", owner)
        != labels.state_uids[state_index]
    ):
        raise QualificationInputError(f"{owner}: file_state_uid mismatch")
    top_seed_raw = np.asarray(loaded["file_state_seed_id"])
    top_seed = _scalar(top_seed_raw, f"{owner}.file_state_seed_id")
    if (
        isinstance(top_seed, (bool, np.bool_))
        or not math.isfinite(float(top_seed))
        or float(top_seed) != int(float(top_seed))
        or int(top_seed) != int(labels.state_seed_ids[state_index])
    ):
        raise QualificationInputError(
            f"{owner}: file_state_seed_id is not the registered uint32-valued "
            "row seed"
        )
    if not _required_bool(loaded, "file_release_qualification_run", owner):
        raise QualificationInputError(
            f"{owner}: file_release_qualification_run is not true"
        )

    data = loaded["data"]
    if not isinstance(data, dict):
        raise QualificationInputError(f"{owner}: data is not a scalar struct")
    missing_data = sorted(_REQUIRED_DATA_FIELDS - set(data))
    if missing_data:
        raise QualificationInputError(f"{owner}.data: missing fields {missing_data}")

    data_expected_strings = {
        "gen_schema": evidence.gen_schema,
        "gen_fingerprint": evidence.gen_fingerprint,
        "matlab_release": evidence.matlab_release,
        "campaign_matlab_release": evidence.campaign_matlab_release,
        "actual_matlab_environment_descriptor":
            evidence.actual_matlab_environment_descriptor,
        "actual_matlab_environment_sha256":
            evidence.actual_matlab_environment_sha256,
        "campaign_matlab_environment_descriptor":
            evidence.campaign_matlab_environment_descriptor,
        "campaign_matlab_environment_sha256":
            evidence.campaign_matlab_environment_sha256,
        "generator_source_root_sha256":
            evidence.generator_source_root_sha256,
        "generator_source_digest_lines":
            policy.generator_source_digest_lines,
        "qualification_source_sha256":
            evidence.qualification_source_sha256,
        "random_stream_schedule_version":
            labels.random_stream_schedule_version,
    }
    for key, expected in data_expected_strings.items():
        reader = (
            _required_exact_text
            if key.endswith("_descriptor") or key == "generator_source_digest_lines"
            else _required_string
        )
        if reader(data, key, f"{owner}.data") != expected:
            raise QualificationInputError(f"{owner}.data.{key} mismatch")
    if (
        _required_positive_int(
            data, "generator_source_file_count", f"{owner}.data"
        )
        != evidence.generator_source_file_count
    ):
        raise QualificationInputError(
            f"{owner}.data.generator_source_file_count mismatch"
        )
    if not _required_bool(data, "release_qualification_run", f"{owner}.data"):
        raise QualificationInputError(
            f"{owner}.data.release_qualification_run is not true"
        )

    n_passages = evidence.passages_per_state
    dim_acel = _vector(
        data["DimAcel"],
        f"{owner}.data.DimAcel",
        length=n_passages,
        integer=True,
        positive=True,
    )
    dim_space = _vector(
        data["DimSpace"],
        f"{owner}.data.DimSpace",
        length=n_passages,
        integer=True,
        positive=True,
    )
    crop_start = _vector(
        data["crop_start"],
        f"{owner}.data.crop_start",
        length=n_passages,
        integer=True,
        positive=True,
    )
    crop_end = _vector(
        data["crop_end"],
        f"{owner}.data.crop_end",
        length=n_passages,
        integer=True,
        positive=True,
    )
    bridge_samp = _vector(
        data["bridge_samp"],
        f"{owner}.data.bridge_samp",
        length=n_passages,
        integer=True,
        positive=True,
    )
    bridge_length = _vector(
        data["L_bridge_eff"],
        f"{owner}.data.L_bridge_eff",
        length=n_passages,
        positive=True,
    )
    manifest_bridge = _required_positive_float(
        manifest, "L_bridge_m", str(Path(evidence.path) / "case_info.mat")
    )
    if (
        np.any(crop_start > crop_end)
        or np.any(crop_end > dim_space)
        or np.any(bridge_samp > (crop_end - crop_start + 1))
        or not np.allclose(bridge_length, manifest_bridge, rtol=0, atol=1e-12)
        or not np.array_equal(bridge_samp, np.round(100 * bridge_length))
    ):
        raise QualificationInputError(
            f"{owner}.data RAW crop/bridge metadata is internally inconsistent"
        )

    for field_name in _REQUIRED_SIGNALS:
        _validate_signal_passages(
            data[field_name],
            f"{owner}.data.{field_name}",
            n_passages,
            _SIGNAL_ROWS[field_name],
            dim_acel,
        )

    contact = _finite_numeric_array(
        data["contact_log"], f"{owner}.data.contact_log"
    )
    if contact.shape != (n_passages, 4):
        raise QualificationInputError(
            f"{owner}.data.contact_log has shape {contact.shape}; expected "
            f"({n_passages}, 4)"
        )
    if (
        not np.all(np.isin(contact[:, :2], (0, 1)))
        or np.any(contact[:, 2] < 0)
        or np.any(contact[:, 2] > 1)
        or np.any(contact[:, 2] > policy.contact_fraction_tolerance)
        or np.any(contact[:, 3] > policy.contact_force_tolerance_N)
    ):
        raise QualificationInputError(
            f"{owner}.data.contact_log violates the reviewed logical/range/contact gate"
        )

    scour = _vector(
        data["scour_vector"],
        f"{owner}.data.scour_vector",
        length=evidence.num_supports,
    )
    supports = _vector(
        data["scour_supports"],
        f"{owner}.data.scour_supports",
        length=labels.scour_supports.size,
        integer=True,
        positive=True,
    ).astype(np.int64)
    bearing_vector = _vector(
        data["bearing_vector"],
        f"{owner}.data.bearing_vector",
        length=2,
    )
    bearing_fixity = _vector(
        data["bearing_fixity"],
        f"{owner}.data.bearing_fixity",
        length=2,
    )
    dano = float(_scalar(data["Dano"], f"{owner}.data.Dano"))
    family = _required_string(data, "state_family", f"{owner}.data")
    state_uid = _required_exact_text(
        data, "state_uid", f"{owner}.data"
    )
    state_seed_array = np.asarray(data["state_seed_id"])
    state_seed_raw = _scalar(
        state_seed_array, f"{owner}.data.state_seed_id"
    )
    try:
        state_seed_number = float(state_seed_raw)
    except (TypeError, ValueError) as exc:
        raise QualificationInputError(
            f"{owner}.data.state_seed_id is not numeric"
        ) from exc
    if (
        not math.isfinite(state_seed_number)
        or state_seed_number < 1
        or state_seed_number > np.iinfo(np.uint32).max
        or state_seed_number != int(state_seed_number)
    ):
        raise QualificationInputError(
            f"{owner}.data.state_seed_id is not uint32-valued"
        )
    state_named_raw = np.asarray(data["state_named_stream_seed_id"])
    state_named = _vector(
        state_named_raw,
        f"{owner}.data.state_named_stream_seed_id",
        length=len(labels.state_stream_names),
        integer=True,
        positive=True,
    )
    if (
        state_named_raw.dtype != np.dtype("uint32")
        or not np.array_equal(
            state_named.astype(np.uint32),
            labels.state_named_stream_seed_ids[state_index],
        )
    ):
        raise QualificationInputError(
            f"{owner}.data.state_named_stream_seed_id is not the "
            "registered uint32 namespace row"
        )
    latent_bearing = _vector(
        data["latent_bearing_fixity"],
        f"{owner}.data.latent_bearing_fixity",
        length=2,
    )
    latent_crack = _required_bool(
        data, "latent_crack_on", f"{owner}.data"
    )
    active_crack = _required_bool(data, "crack_on", f"{owner}.data")
    if (
        not np.array_equal(scour, labels.damage_states[state_index])
        or not np.array_equal(supports, labels.scour_supports)
        or not np.array_equal(
            bearing_vector, labels.bearing_states[state_index]
        )
        or not np.array_equal(
            bearing_fixity, labels.bearing_fixity[state_index]
        )
        or family != labels.state_families[state_index]
        or state_uid != labels.state_uids[state_index]
        or int(state_seed_number) != int(labels.state_seed_ids[state_index])
        or not np.array_equal(
            latent_bearing, labels.latent_bearing_fixity[state_index]
        )
        or latent_crack != bool(labels.latent_crack_on[state_index])
        or active_crack != bool(labels.crack_on[state_index])
        or not math.isfinite(dano)
        or dano != float(np.max(scour))
    ):
        raise QualificationInputError(
            f"{owner}.data labels/family do not agree exactly with damage_states.mat"
        )
    if (
        np.any(scour < 0)
        or np.any(scour > 1)
        or np.any(bearing_vector < 0)
        or np.any(bearing_fixity < 0)
        or np.any(bearing_fixity > 1)
    ):
        raise QualificationInputError(f"{owner}.data label range is invalid")

    crack = _finite_numeric_array(data["crack_log"], f"{owner}.data.crack_log")
    if crack.shape != (n_passages, 3) or np.any(crack[:, 1:] < 0):
        raise QualificationInputError(
            f"{owner}.data.crack_log must be finite ({n_passages}, 3) with "
            "nonnegative loss/length"
        )
    crack_active = crack[:, 1] > 0
    if (
        labels.crack_on[state_index] and not np.all(crack_active)
    ) or (
        not labels.crack_on[state_index] and np.any(crack_active)
    ):
        raise QualificationInputError(
            f"{owner}.data.crack_log activation disagrees with "
            "damage_states.mat CrackOn"
        )
    profile = _vector(
        data["profile_log"],
        f"{owner}.data.profile_log",
        length=n_passages,
    )
    if np.any(profile <= 0):
        raise QualificationInputError(f"{owner}.data.profile_log must be positive")
    if (
        _required_string(data, "profile_mode", f"{owner}.data")
        != _required_string(manifest, "profile_mode", str(Path(evidence.path) / "case_info.mat"))
    ):
        raise QualificationInputError(f"{owner}.data.profile_mode mismatch")

    _vector(
        data["Temperatura"],
        f"{owner}.data.Temperatura",
        length=n_passages,
    )
    _vector(
        data["Velocidade"],
        f"{owner}.data.Velocidade",
        length=n_passages,
        positive=True,
    )
    _finite_numeric_array(data["VehiclesProps"], f"{owner}.data.VehiclesProps")
    beam_f1 = float(_scalar(data["beam_f1_Hz"], f"{owner}.data.beam_f1_Hz"))
    if not math.isfinite(beam_f1) or not 0.2 <= beam_f1 <= 15:
        raise QualificationInputError(
            f"{owner}.data.beam_f1_Hz is outside the generator's global "
            "0.2--15 Hz sanity gate"
        )
    if family == "target_healthy":
        bounds = (3.0, 6.0) if manifest_bridge < 80 else (2.0, 4.0)
        if not bounds[0] <= beam_f1 <= bounds[1]:
            raise QualificationInputError(
                f"{owner}.data.beam_f1_Hz violates the healthy "
                f"{manifest_bridge:g} m bridge gate {bounds}"
            )

    # Require passage cardinality even where a mechanism is disabled. Detailed
    # struct validation is performed for the all-mechanism coverage stages.
    _passages(data["track_log"], n_passages, f"{owner}.data.track_log")
    _passages(data["oor_log"], n_passages, f"{owner}.data.oor_log")
    return data


def _update_mechanism_coverage(
    accumulator: _CoverageAccumulator,
    data: dict[str, Any],
    evidence: DatasetEvidence,
    state_name: str,
) -> None:
    if evidence.stage not in _MECHANISM_COVERAGE_STAGES:
        return
    crack = np.asarray(data["crack_log"], dtype=float).reshape(
        evidence.passages_per_state, 3
    )
    for passage, row in enumerate(crack):
        if row[1] > 0:
            accumulator.crack_active_passages += 1
            accumulator.witness("crack", f"{state_name}:passage-{passage + 1}")

    if _required_string(data, "profile_mode", f"{state_name}.data") != "psd_fra":
        raise QualificationInputError(
            f"{evidence.path}/{state_name}: all-mechanism stage is not psd_fra"
        )
    profile = np.asarray(data["profile_log"], dtype=float).reshape(-1)
    active_profile = int(np.count_nonzero(profile > 0))
    accumulator.profile_active_passages += active_profile
    if active_profile:
        accumulator.witness("profile", f"{state_name}:passage-1")

    track_passages = _passages(
        data["track_log"],
        evidence.passages_per_state,
        f"{evidence.path}/{state_name}.data.track_log",
    )
    oor_passages = _passages(
        data["oor_log"],
        evidence.passages_per_state,
        f"{evidence.path}/{state_name}.data.oor_log",
    )
    for passage, track in enumerate(track_passages):
        location = f"{state_name}:passage-{passage + 1}"
        if not isinstance(track, dict):
            raise QualificationInputError(
                f"{evidence.path}/{location}: track_log must be a struct"
            )
        required_track = {
            "ballast_patches",
            "hanging_groups",
            "pad_stiff_mult",
            "pad_damp_mult",
            "pad_failures",
            "x_bridge_local",
        }
        missing = sorted(required_track - set(track))
        if missing:
            raise QualificationInputError(
                f"{evidence.path}/{location}: track_log lacks {missing}"
            )
        ballast = _row_matrix(
            track["ballast_patches"], f"{location}.ballast_patches", 4
        )
        hanging = _row_matrix(
            track["hanging_groups"], f"{location}.hanging_groups", 2
        )
        pad_failures = _finite_numeric_array(
            track["pad_failures"], f"{location}.pad_failures", allow_empty=True
        ).reshape(-1)
        pad_stiff = float(_scalar(track["pad_stiff_mult"], f"{location}.pad_stiff_mult"))
        pad_damp = float(_scalar(track["pad_damp_mult"], f"{location}.pad_damp_mult"))
        x_bridge = float(_scalar(track["x_bridge_local"], f"{location}.x_bridge_local"))
        if not np.all(
            np.isfinite([pad_stiff, pad_damp, x_bridge])
        ) or pad_stiff <= 0 or pad_damp <= 0 or x_bridge < 0:
            raise QualificationInputError(
                f"{evidence.path}/{location}: invalid track scalar"
            )
        accumulator.ballast_patch_rows += ballast.shape[0]
        accumulator.hanging_group_rows += hanging.shape[0]
        if ballast.shape[0]:
            accumulator.witness("ballast", location)
        if hanging.shape[0]:
            accumulator.witness("hanging", location)
        if (
            not math.isclose(pad_stiff, 1.0, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(pad_damp, 1.0, rel_tol=0, abs_tol=1e-15)
            or pad_failures.size
        ):
            accumulator.pad_departure_passages += 1
            accumulator.witness("pad", location)

    for passage, oor in enumerate(oor_passages):
        location = f"{state_name}:passage-{passage + 1}"
        if not isinstance(oor, dict) or not {"flats", "poly"} <= set(oor):
            raise QualificationInputError(
                f"{evidence.path}/{location}: oor_log must contain flats/poly"
            )
        flats = _row_matrix(oor["flats"], f"{location}.flats", 5)
        polygon = _row_matrix(oor["poly"], f"{location}.poly", 5)
        if flats.shape[0]:
            raise QualificationInputError(
                f"{evidence.path}/{location}: wheel flats are disabled by the "
                "reviewed bilateral-contact policy"
            )
        accumulator.polygon_rows += polygon.shape[0]
        if polygon.shape[0]:
            accumulator.witness("polygonization", location)


def _finalise_coverage(
    accumulator: _CoverageAccumulator, evidence: DatasetEvidence, manifest: dict[str, Any]
) -> MechanismCoverage:
    required = evidence.stage in _MECHANISM_COVERAGE_STAGES
    if required:
        owner = str(Path(evidence.path) / "case_info.mat")
        expected_toggles = {
            "use_crack_eov": True,
            "use_track_eov": True,
            "use_oor_eov": True,
            "oor_flats_enabled": False,
        }
        for key, expected in expected_toggles.items():
            if _required_bool(manifest, key, owner) != expected:
                raise QualificationInputError(
                    f"{evidence.path}: {evidence.stage} requires {key}={expected}"
                )
        if _required_string(manifest, "profile_mode", owner) != "psd_fra":
            raise QualificationInputError(
                f"{evidence.path}: {evidence.stage} requires profile_mode=psd_fra"
            )
        counts = {
            "crack": accumulator.crack_active_passages,
            "profile": accumulator.profile_active_passages,
            "ballast": accumulator.ballast_patch_rows,
            "hanging": accumulator.hanging_group_rows,
            "pad": accumulator.pad_departure_passages,
            "polygonization": accumulator.polygon_rows,
        }
        absent = sorted(name for name, count in counts.items() if count <= 0)
        missing_witness = sorted(set(counts) - set(accumulator.witnesses))
        if absent or missing_witness:
            raise QualificationInputError(
                f"{evidence.path}: {evidence.stage} lacks required aggregate "
                f"mechanism witnesses; zero={absent}, unwitnessed={missing_witness}"
            )
    return MechanismCoverage(
        required=required,
        crack_active_passages=accumulator.crack_active_passages,
        profile_active_passages=accumulator.profile_active_passages,
        ballast_patch_rows=accumulator.ballast_patch_rows,
        hanging_group_rows=accumulator.hanging_group_rows,
        pad_departure_passages=accumulator.pad_departure_passages,
        polygon_rows=accumulator.polygon_rows,
        witnesses=dict(sorted(accumulator.witnesses.items())),
    )


def _numeric_array(value: Any) -> np.ndarray | None:
    array = np.asarray(value)
    if array.dtype.kind in "biufc":
        return array
    return None


def _compare_numeric(
    a: np.ndarray,
    b: np.ndarray,
    *,
    path: str,
    tolerant: bool,
    rtol: float,
    atol: float,
    stats: ComparisonStats,
) -> None:
    if a.shape != b.shape:
        stats.mismatch(f"{path}: shape {a.shape} != {b.shape}")
        return
    if a.dtype != b.dtype:
        stats.mismatch(f"{path}: dtype {a.dtype} != {b.dtype}")
        return
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        stats.mismatch(f"{path}: non-finite numerical value")
        return

    stats.compared_leaves += 1
    stats.compared_numeric_values += int(a.size)
    leaf_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if leaf_name in _REQUIRED_SIGNALS:
        stats.compared_signal_values += int(a.size)

    if not tolerant or a.dtype.kind in "biu" or b.dtype.kind in "biu":
        stats.exact_leaves += 1
        if a.dtype.kind != b.dtype.kind or not np.array_equal(a, b):
            stats.mismatch(f"{path}: exact numerical values differ")
        return

    stats.tolerant_leaves += 1
    if a.size == 0:
        return
    abs_diff = np.abs(a - b)
    max_abs = float(np.max(abs_diff))
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny)
    max_rel = float(np.max(abs_diff / denom))
    if max_abs > stats.max_absolute_difference:
        stats.max_absolute_difference = max_abs
        stats.worst_path = path
    stats.max_relative_difference = max(stats.max_relative_difference, max_rel)
    if not np.array_equal(a, b):
        stats.numerical_difference = True
    allowed = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    if not np.all(abs_diff <= allowed):
        stats.mismatch(
            f"{path}: solver output exceeds tolerance "
            f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e})"
        )


def _compare_node(
    a: Any,
    b: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
    stats: ComparisonStats,
    force_exact: bool = False,
) -> None:
    """Recursively compare scipy ``simplify_cells`` values."""
    if a is None or b is None:
        stats.compared_leaves += 1
        stats.exact_leaves += 1
        if a is not None or b is not None:
            stats.mismatch(f"{path}: None/non-None value differs")
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            stats.mismatch(f"{path}: mapping type differs")
            return
        ka, kb = set(a), set(b)
        if ka != kb:
            stats.mismatch(
                f"{path}: key set differs; only-A={sorted(ka-kb)}, "
                f"only-B={sorted(kb-ka)}"
            )
            return
        for key in sorted(ka):
            _compare_node(
                a[key],
                b[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                stats=stats,
                force_exact=force_exact,
            )
        return

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            stats.mismatch(f"{path}: sequence type differs")
            return
        if len(a) != len(b):
            stats.mismatch(f"{path}: sequence length {len(a)} != {len(b)}")
            return
        for index, (left, right) in enumerate(zip(a, b)):
            _compare_node(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                stats=stats,
                force_exact=force_exact,
            )
        return

    aa, bb = np.asarray(a), np.asarray(b)
    if aa.dtype == object or bb.dtype == object:
        if aa.dtype != object or bb.dtype != object:
            stats.mismatch(f"{path}: object/non-object representation differs")
            return
        if aa.shape != bb.shape:
            stats.mismatch(f"{path}: object-array shape {aa.shape} != {bb.shape}")
            return
        for index, (left, right) in enumerate(zip(aa.flat, bb.flat)):
            _compare_node(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                stats=stats,
                force_exact=force_exact,
            )
        return

    na, nb = _numeric_array(a), _numeric_array(b)
    if na is not None or nb is not None:
        if na is None or nb is None:
            stats.mismatch(f"{path}: numerical/non-numerical type differs")
            return
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        _compare_numeric(
            na,
            nb,
            path=path,
            tolerant=(not force_exact and leaf in _TOLERANT_FLOAT_FIELDS),
            rtol=rtol,
            atol=atol,
            stats=stats,
        )
        return

    if aa.shape != bb.shape:
        stats.mismatch(f"{path}: shape {aa.shape} != {bb.shape}")
        return
    stats.compared_leaves += 1
    stats.exact_leaves += 1
    if aa.dtype.kind != bb.dtype.kind or not np.array_equal(aa, bb):
        stats.mismatch(f"{path}: exact categorical/string values differ")


def _validated_payload(
    path: Path, policy: CurrentPolicy
) -> tuple[
    DatasetEvidence,
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    evidence, manifest, generation_config = _validate_dataset_header(path, policy)
    label_table = _public_mat(path / "damage_states.mat")
    labels = _validate_damage_table(label_table, evidence, manifest)
    _validate_generation_realizations(generation_config, labels, evidence)
    coverage = _CoverageAccumulator()
    states: dict[str, dict[str, Any]] = {}
    for index, state_name in enumerate(evidence.state_files):
        loaded = _public_mat(path / state_name)
        data = _validate_state_provenance(
            loaded,
            evidence,
            manifest,
            labels,
            state_name,
            index,
            policy,
        )
        _update_mechanism_coverage(coverage, data, evidence, state_name)
        states[state_name] = loaded
    final_coverage = _finalise_coverage(coverage, evidence, manifest)
    evidence = replace(evidence, mechanism_coverage=final_coverage)
    return evidence, manifest, label_table, states


def compare_directories(
    dir_a: Path | str,
    dir_b: Path | str,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> ComparisonResult:
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise QualificationInputError("rtol and atol must be finite and nonnegative")
    if rtol > _DEFAULT_RTOL or atol > _DEFAULT_ATOL:
        raise QualificationInputError(
            "qualification tolerances may be made stricter but not looser than "
            f"the reviewed policy (rtol={_DEFAULT_RTOL:.1e}, "
            f"atol={_DEFAULT_ATOL:.1e})"
        )

    policy = _current_policy()
    path_a, path_b = Path(dir_a), Path(dir_b)
    ev_a, manifest_a, labels_a, states_a = _validated_payload(path_a, policy)
    ev_b, manifest_b, labels_b, states_b = _validated_payload(path_b, policy)

    if (
        ev_a.actual_matlab_environment_sha256
        == ev_b.actual_matlab_environment_sha256
        and ev_a.qualification_host_id == ev_b.qualification_host_id
    ):
        raise QualificationInputError(
            "two inputs with the same authenticated MATLAB environment must "
            "come from independently identified hosts; both receipts declare "
            f"{ev_a.qualification_host_id!r}"
        )
    locked_sha = policy.campaign_matlab_environment_sha256
    if locked_sha not in {
        ev_a.actual_matlab_environment_sha256,
        ev_b.actual_matlab_environment_sha256,
    }:
        raise QualificationInputError(
            "one input must have been generated by the current locked campaign "
            f"MATLAB environment {locked_sha}; input declarations cannot replace "
            "that repository policy"
        )
    for attribute in (
        "stage",
        "campaign_matlab_release",
        "campaign_matlab_environment_descriptor",
        "campaign_matlab_environment_sha256",
        "gen_schema",
        "generation_behavior_version",
        "gen_fingerprint",
        "generator_source_root_sha256",
        "generator_source_file_count",
        "qualification_source_sha256",
        "qualification_executed_file_sha256",
        "n_states",
        "passages_per_state",
        "num_supports",
        "max_parfor_workers",
        "state_files",
    ):
        if getattr(ev_a, attribute) != getattr(ev_b, attribute):
            raise QualificationInputError(
                f"qualification inputs differ in {attribute}: "
                f"{getattr(ev_a, attribute)!r} != {getattr(ev_b, attribute)!r}"
            )

    stats = ComparisonStats()
    clean_manifest_a = {
        key: value
        for key, value in manifest_a.items()
        if key not in _MANIFEST_IGNORED_KEYS
    }
    clean_manifest_b = {
        key: value
        for key, value in manifest_b.items()
        if key not in _MANIFEST_IGNORED_KEYS
    }
    _compare_node(
        clean_manifest_a,
        clean_manifest_b,
        path="case_info",
        rtol=rtol,
        atol=atol,
        stats=stats,
        force_exact=True,
    )
    _compare_node(
        labels_a,
        labels_b,
        path="damage_states",
        rtol=rtol,
        atol=atol,
        stats=stats,
        force_exact=True,
    )

    raw_identical = 0
    for state_name in ev_a.state_files:
        if sha256(path_a / state_name) == sha256(path_b / state_name):
            raw_identical += 1
        loaded_a, loaded_b = states_a[state_name], states_b[state_name]
        top_a = {
            key: value
            for key, value in loaded_a.items()
            if key not in _TOP_IGNORED_KEYS
        }
        top_b = {
            key: value
            for key, value in loaded_b.items()
            if key not in _TOP_IGNORED_KEYS
        }
        _compare_node(
            top_a,
            top_b,
            path=f"{state_name}.top",
            rtol=rtol,
            atol=atol,
            stats=stats,
            force_exact=True,
        )
        semantic_a = {
            key: value
            for key, value in loaded_a["data"].items()
            if key not in _DATA_IGNORED_KEYS
        }
        semantic_b = {
            key: value
            for key, value in loaded_b["data"].items()
            if key not in _DATA_IGNORED_KEYS
        }
        _compare_node(
            semantic_a,
            semantic_b,
            path=f"{state_name}.data",
            rtol=rtol,
            atol=atol,
            stats=stats,
        )

    if stats.compared_leaves == 0 or stats.compared_signal_values == 0:
        raise QualificationInputError(
            "comparison reached zero semantic leaves or zero signal values"
        )
    if stats.mismatches:
        verdict = "MATERIALLY-DIFFERENT"
    elif stats.numerical_difference:
        verdict = "NUMERICALLY-EQUIVALENT"
    else:
        verdict = "SEMANTICALLY-BIT-IDENTICAL"
    return ComparisonResult(
        verdict=verdict,
        evidence_a=ev_a,
        evidence_b=ev_b,
        stats=stats,
        raw_byte_identical_states=raw_identical,
        tolerance_rtol=rtol,
        tolerance_atol=atol,
        environment_lock_sha256=policy.environment_lock_sha256,
        parser_environment=dict(policy.parser_environment),
        python_runtime_source_root_sha256=
            policy.python_runtime_source_root_sha256,
        python_runtime_source_file_count=
            policy.python_runtime_source_file_count,
    )


def _receipt_payload(result: ComparisonResult, *, accepted: bool) -> dict[str, Any]:
    current = _current_policy()
    if (
        current.environment_lock_sha256 != result.environment_lock_sha256
        or current.parser_environment != result.parser_environment
        or current.python_runtime_source_root_sha256
        != result.python_runtime_source_root_sha256
        or current.python_runtime_source_file_count
        != result.python_runtime_source_file_count
    ):
        raise QualificationInputError(
            "repository/parser provenance changed after comparison; rerun the "
            "comparison before writing a receipt"
        )
    return {
        "schema": "matlab-environment-qualification-receipt-v4",
        "semantic_exactness_definition": (
            "Exact canonical payload equality after excluding only declared "
            "actual-environment fields and MAT-container metadata; not raw "
            "MAT-file byte identity."
        ),
        "comparator_sha256": sha256(Path(__file__).resolve()),
        "environment_lock_sha256": result.environment_lock_sha256,
        "parser_environment": result.parser_environment,
        "python_runtime_source_root_sha256":
            result.python_runtime_source_root_sha256,
        "python_runtime_source_file_count":
            result.python_runtime_source_file_count,
        "verdict": result.verdict,
        "numerical_equivalence_explicitly_accepted": bool(accepted),
        "stage": result.evidence_a.stage,
        "generator_source_root_sha256":
            result.evidence_a.generator_source_root_sha256,
        "qualification_source_sha256":
            result.evidence_a.qualification_source_sha256,
        "dataset_content_roots_sha256": [
            result.evidence_a.dataset_content_root_sha256,
            result.evidence_b.dataset_content_root_sha256,
        ],
        "actual_matlab_environment_sha256": [
            result.evidence_a.actual_matlab_environment_sha256,
            result.evidence_b.actual_matlab_environment_sha256,
        ],
        "qualification_host_id": [
            result.evidence_a.qualification_host_id,
            result.evidence_b.qualification_host_id,
        ],
        "qualification_host_diagnostic_sha256": [
            result.evidence_a.qualification_host_diagnostic_sha256,
            result.evidence_b.qualification_host_diagnostic_sha256,
        ],
        "tolerances": {
            "rtol": result.tolerance_rtol,
            "atol": result.tolerance_atol,
            "tolerant_fields": sorted(_TOLERANT_FLOAT_FIELDS),
        },
        "dataset_a": asdict(result.evidence_a),
        "dataset_b": asdict(result.evidence_b),
        "comparison": asdict(result.stats),
        "raw_byte_identical_states": result.raw_byte_identical_states,
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(encoded)
    except FileExistsError as exc:
        raise QualificationInputError(
            f"refusing to overwrite existing qualification receipt: {path}"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.read_text(encoding="utf-8") != encoded
    ):
        raise QualificationInputError(
            f"qualification receipt did not persist as the exact regular file: {path}"
        )


def _print_result(result: ComparisonResult) -> None:
    print("GENERATION-ENVIRONMENT QUALIFICATION COMPARISON")
    print(
        f"  A: {result.evidence_a.path} "
        f"({result.evidence_a.matlab_release}; "
        f"{result.evidence_a.actual_matlab_environment_sha256}; "
        f"host={result.evidence_a.qualification_host_id})"
    )
    print(
        f"  B: {result.evidence_b.path} "
        f"({result.evidence_b.matlab_release}; "
        f"{result.evidence_b.actual_matlab_environment_sha256}; "
        f"host={result.evidence_b.qualification_host_id})"
    )
    print(
        "  locked campaign environment: "
        f"{result.evidence_a.campaign_matlab_environment_sha256}"
    )
    print(f"  stage/schema: {result.evidence_a.stage}/{result.evidence_a.gen_schema}")
    print(f"  fingerprint: {result.evidence_a.gen_fingerprint}")
    print(
        "  generator/qualification source: "
        f"{result.evidence_a.generator_source_root_sha256}/"
        f"{result.evidence_a.qualification_source_sha256}"
    )
    print(
        "  authenticated states: "
        f"{result.evidence_a.n_states}; raw-container SHA matches: "
        f"{result.raw_byte_identical_states}/{result.evidence_a.n_states}"
    )
    if result.evidence_a.mechanism_coverage.required:
        print(
            "  mechanism coverage: "
            f"{asdict(result.evidence_a.mechanism_coverage)}"
        )
    print(
        f"  compared leaves: {result.stats.compared_leaves}; "
        f"numeric values: {result.stats.compared_numeric_values}; "
        f"signal values: {result.stats.compared_signal_values}"
    )
    print(
        "  worst solver difference: "
        f"abs={result.stats.max_absolute_difference:.3e}, "
        f"rel={result.stats.max_relative_difference:.3e} "
        f"at {result.stats.worst_path or '(none)'}"
    )
    for mismatch in result.stats.mismatches:
        print(f"  DIFFERENCE: {mismatch}")
    print(f"\nVERDICT: {result.verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dir_a", type=Path)
    parser.add_argument("dir_b", type=Path)
    parser.add_argument(
        "--rtol",
        type=float,
        default=_DEFAULT_RTOL,
        help="relative tolerance; may be stricter, never looser, than 1e-10",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=_DEFAULT_ATOL,
        help="absolute tolerance; may be stricter, never looser, than 1e-12",
    )
    parser.add_argument(
        "--accept-numerical",
        action="store_true",
        help="explicitly accept an in-tolerance numerical verdict; requires --receipt",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="new path for a machine-readable receipt (existing files are refused)",
    )
    args = parser.parse_args(argv)
    try:
        result = compare_directories(
            args.dir_a, args.dir_b, rtol=args.rtol, atol=args.atol
        )
        _print_result(result)
        if result.verdict == "NUMERICALLY-EQUIVALENT":
            if not args.accept_numerical:
                if args.receipt:
                    _write_receipt(
                        args.receipt, _receipt_payload(result, accepted=False)
                    )
                print(
                    "PENDING: numerical equivalence is not automatic qualification. "
                    "Review it, then use --accept-numerical --receipt <new-file>."
                )
                return 3
            if args.receipt is None:
                raise QualificationInputError(
                    "--accept-numerical requires --receipt so acceptance is bound "
                    "to these exact sources, environments, payload roots and tolerances"
                )
        if args.receipt:
            _write_receipt(
                args.receipt,
                _receipt_payload(
                    result,
                    accepted=(
                        result.verdict == "SEMANTICALLY-BIT-IDENTICAL"
                        or (
                            result.verdict == "NUMERICALLY-EQUIVALENT"
                            and args.accept_numerical
                        )
                    ),
                ),
            )
            print(f"  receipt: {args.receipt}")
        return result.default_exit_code if not args.accept_numerical else (
            0 if result.verdict != "MATERIALLY-DIFFERENT" else 1
        )
    except QualificationInputError as exc:
        print(f"INVALID QUALIFICATION EVIDENCE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
