"""Strict JSON schema layer for pairwise MATLAB qualification receipts.

Contract: this module authenticates the *content* of one retained
``matlab-environment-qualification-receipt-v4`` file against the current
repository policy - exact field sets, canonical JSON bytes, provenance
pinning, verdict/acceptance coherence, and per-endpoint identity extraction.
It exists as its own module so the receipt grammar can be audited in
isolation from graph closure (``qualification_receipt_inventory``) and from
retained-endpoint disk revalidation (``qualification_endpoint_revalidation``).

Rationale: the schema layer deliberately NEVER touches dataset directories.
The ``path`` recorded inside a dataset block is validated here only as a
nonempty string; reopening that directory and rerunning the comparator is the
exclusive responsibility of ``qualification_endpoint_revalidation``, which the
inventory invokes unconditionally.  Keeping the layers separate makes it
impossible for a "shape-only" code path to silently become the authorization
path: the inventory root digest binds all three modules together.

Every deviation fails closed through ``QualificationInventoryError``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import compare_generation_releases as comparator


REQUIRED_STAGES = ("F40-S", "F40-M", "L99-S", "L99-M")
QUALIFICATION_STATE_COUNTS = {
    "F40-S": 31,
    "F40-M": 31,
    "L99-S": 39,
    "L99-M": 39,
}
QUALIFICATION_PASSAGES_PER_STATE = 3
RECEIPT_SCHEMA = "matlab-environment-qualification-receipt-v4"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HOST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MATLAB_RELEASE_RE = re.compile(r"^R20\d{2}[ab]$")
_SEMANTIC_EXACTNESS_DEFINITION = (
    "Exact canonical payload equality after excluding only declared "
    "actual-environment fields and MAT-container metadata; not raw "
    "MAT-file byte identity."
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "semantic_exactness_definition",
        "comparator_sha256",
        "environment_lock_sha256",
        "parser_environment",
        "python_runtime_source_root_sha256",
        "python_runtime_source_file_count",
        "verdict",
        "numerical_equivalence_explicitly_accepted",
        "stage",
        "generator_source_root_sha256",
        "qualification_source_sha256",
        "dataset_content_roots_sha256",
        "actual_matlab_environment_sha256",
        "qualification_host_id",
        "qualification_host_diagnostic_sha256",
        "tolerances",
        "dataset_a",
        "dataset_b",
        "comparison",
        "raw_byte_identical_states",
    }
)
_DATASET_FIELDS = frozenset(
    field.name for field in fields(comparator.DatasetEvidence)
)
_MECHANISM_FIELDS = frozenset(
    field.name for field in fields(comparator.MechanismCoverage)
)
_COMPARISON_FIELDS = frozenset(
    field.name for field in fields(comparator.ComparisonStats)
)
_TOLERANCE_FIELDS = frozenset({"rtol", "atol", "tolerant_fields"})
_REQUIRED_MECHANISM_WITNESSES = frozenset(
    {"crack", "profile", "ballast", "hanging", "pad", "polygonization"}
)
_MECHANISM_WITNESS_RE = re.compile(
    r"^(?P<state>\d{4})\.mat:passage-(?P<passage>[1-9]\d*)$"
)
_NUMERICAL_WORST_PATH_RE = re.compile(
    r"^(?P<state>\d{4})\.mat\.data\."
    r"(?:(?P<signal>AcelPrimVag|AcelRodaPrimVag|AcelWheelsetPrimVag|PitchPrimVag)"
    r"\[(?P<passage>0|[1-9]\d*)\]|"
    r"(?P<array>contact_log|beam_f1_Hz))$"
)
_SHA_DATASET_FIELDS = (
    "actual_matlab_environment_sha256",
    "campaign_matlab_environment_sha256",
    "generator_source_root_sha256",
    "qualification_source_sha256",
    "qualification_executed_file_sha256",
    "qualification_host_diagnostic_sha256",
    "dataset_content_root_sha256",
)
_POSITIVE_DATASET_INTS = (
    "generator_source_file_count",
    "qualification_logical_processors",
    "qualification_matlab_max_threads",
    "n_states",
    "passages_per_state",
    "num_supports",
    "max_parfor_workers",
)
_HOST_DESCRIPTOR_FIELDS = (
    ("schema", comparator._QUALIFICATION_HOST_SCHEMA),
    ("declared_host_id", "qualification_host_id"),
    ("hostname", "qualification_hostname"),
    ("cpu_identifier", "qualification_cpu_identifier"),
    ("logical_processors", "qualification_logical_processors"),
    ("matlab_max_threads", "qualification_matlab_max_threads"),
    ("computer_arch", "qualification_computer_arch"),
    (
        "actual_matlab_environment_sha256",
        "actual_matlab_environment_sha256",
    ),
    ("qualification_source_sha256", "qualification_source_sha256"),
    (
        "qualification_executed_file_sha256",
        "qualification_executed_file_sha256",
    ),
)


class QualificationInventoryError(RuntimeError):
    """The supplied receipt collection is not complete/current evidence."""


@dataclass(frozen=True)
class CurrentInventoryPolicy:
    comparator_sha256: str
    checker_sha256: str
    environment_lock_sha256: str
    parser_environment: dict[str, str]
    python_runtime_source_root_sha256: str
    python_runtime_source_file_count: int
    generator_source_root_sha256: str
    generator_source_file_count: int
    campaign_matlab_release: str
    campaign_matlab_environment_descriptor: str
    campaign_matlab_environment_sha256: str
    gen_schema: str
    channel_schema_id: str
    generation_behavior_version: str
    max_parfor_workers: int
    qualification_source_sha256: dict[str, str]
    qualification_executed_file_sha256: dict[str, str]


@dataclass(frozen=True)
class EndpointBinding:
    dataset_content_root_sha256: str
    actual_matlab_environment_descriptor: str
    actual_matlab_environment_sha256: str
    matlab_release: str
    gen_fingerprint: str
    generator_source_root_sha256: str
    generator_source_file_count: int
    qualification_source_sha256: str
    qualification_executed_file_sha256: str
    qualification_host_canonical_descriptor: str
    qualification_host_diagnostic_sha256: str
    qualification_hostname: str
    qualification_cpu_identifier: str
    qualification_logical_processors: int
    qualification_matlab_max_threads: int
    qualification_computer_arch: str
    mechanism_coverage_sha256: str


@dataclass(frozen=True)
class HostHardwareBinding:
    """Stage-invariant SELF-ATTESTED host diagnostics for one intended host.

    Threat model - read before strengthening any claim built on this type.
    These fields are diagnostics the qualification run reports about itself,
    bound together by a SHA-256 over their own canonical descriptor. There is
    no pre-registered signing key, no hardware or remote attestation, and no
    independent witness, so this binding proves INTERNAL CONSISTENCY of the
    retained artifacts, not the physical origin of the computation. A trusted
    operator who runs the reviewed bytes on distinct machines produces distinct
    bindings; an operator who fabricates coherent receipts can also produce
    distinct bindings, and this layer cannot tell the two apart. The
    repository's own fixture builder demonstrates the boundary: it writes
    acceptable host receipts and datasets from Python with SciPy, and the
    resulting graph validates. Describe this as self-attested host diagnostics
    plus retained-artifact integrity under a trusted-operator threat model -
    never as proof that MATLAB ran on independent physical hosts.
    """

    actual_matlab_environment_descriptor: str
    actual_matlab_environment_sha256: str
    matlab_release: str
    qualification_hostname: str
    qualification_cpu_identifier: str
    qualification_logical_processors: int
    qualification_matlab_max_threads: int
    qualification_computer_arch: str


@dataclass(frozen=True)
class ReceiptEdge:
    stage: str
    host_a: str
    host_b: str
    receipt_sha256: str
    path: Path
    structural_comparison_counts: tuple[int, int, int, int, int]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.stage, self.host_a, self.host_b)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _strict_json_object(text: str, owner: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationInventoryError(
            f"{owner}: not one strict JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QualificationInventoryError(
            f"{owner}: top-level JSON value must be an object"
        )
    return value


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exact_fields(
    mapping: dict[str, Any],
    expected: frozenset[str],
    owner: str,
) -> None:
    observed = set(mapping)
    if observed != expected:
        raise QualificationInventoryError(
            f"{owner}: field-set mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _text(value: Any, owner: str) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise QualificationInventoryError(
            f"{owner}: expected one nonempty JSON string"
        )
    return value


def _canonical_host_text(value: Any, owner: str) -> str:
    text = _text(value, owner)
    if (
        text != text.strip()
        or "\r" in text
        or "\n" in text
        or "\0" in text
        or len(text) > 1024
    ):
        raise QualificationInventoryError(
            f"{owner}: expected one canonical nonempty line of at most "
            "1024 characters"
        )
    return text


def _sha(value: Any, owner: str) -> str:
    text = _text(value, owner)
    if not _SHA256_RE.fullmatch(text):
        raise QualificationInventoryError(
            f"{owner}: expected one lowercase SHA-256"
        )
    return text


def _positive_int(value: Any, owner: str) -> int:
    if type(value) is not int or value < 1:
        raise QualificationInventoryError(
            f"{owner}: expected one positive JSON integer"
        )
    return value


def _nonnegative_int(value: Any, owner: str) -> int:
    if type(value) is not int or value < 0:
        raise QualificationInventoryError(
            f"{owner}: expected one nonnegative JSON integer"
        )
    return value


def _finite_nonnegative_number(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationInventoryError(
            f"{owner}: expected one finite nonnegative JSON number"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise QualificationInventoryError(
            f"{owner}: expected one finite nonnegative JSON number"
        )
    return number


def _validate_host_descriptor(dataset: dict[str, Any], owner: str) -> None:
    lines: list[str] = []
    for label, source in _HOST_DESCRIPTOR_FIELDS:
        value: Any = source if label == "schema" else dataset[source]
        if "\n" in str(value) or "\0" in str(value):
            raise QualificationInventoryError(
                f"{owner}: host diagnostic field {label!r} is not one line"
            )
        lines.append(f"{label}={value}")
    descriptor = "\n".join(lines)
    if dataset["qualification_host_canonical_descriptor"] != descriptor:
        raise QualificationInventoryError(
            f"{owner}: qualification host canonical descriptor mismatch"
        )
    if (
        _sha256_text(descriptor)
        != dataset["qualification_host_diagnostic_sha256"]
    ):
        raise QualificationInventoryError(
            f"{owner}: qualification host diagnostic SHA mismatch"
        )


def _validate_mechanism(
    value: Any,
    *,
    stage: str,
    n_states: int,
    passages_per_state: int,
    owner: str,
) -> None:
    if not isinstance(value, dict):
        raise QualificationInventoryError(f"{owner}: expected one JSON object")
    _exact_fields(value, _MECHANISM_FIELDS, owner)
    expected_required = stage in comparator._MECHANISM_COVERAGE_STAGES
    if type(value["required"]) is not bool or value["required"] != expected_required:
        raise QualificationInventoryError(
            f"{owner}.required does not match the current stage policy"
        )
    count_fields = (
        "crack_active_passages",
        "profile_active_passages",
        "ballast_patch_rows",
        "hanging_group_rows",
        "pad_departure_passages",
        "polygon_rows",
    )
    counts = {
        field: _nonnegative_int(value[field], f"{owner}.{field}")
        for field in count_fields
    }
    witnesses = value["witnesses"]
    if not isinstance(witnesses, dict):
        raise QualificationInventoryError(
            f"{owner}.witnesses must be one JSON object"
        )
    if expected_required:
        absent = sorted(field for field, count in counts.items() if count <= 0)
        if absent:
            raise QualificationInventoryError(
                f"{owner}: required mechanism counts must all be positive; "
                f"zero={absent}"
            )
        if set(witnesses) != _REQUIRED_MECHANISM_WITNESSES:
            raise QualificationInventoryError(
                f"{owner}: required witness-key set mismatch; "
                f"missing={sorted(_REQUIRED_MECHANISM_WITNESSES - set(witnesses))}, "
                f"extra={sorted(set(witnesses) - _REQUIRED_MECHANISM_WITNESSES)}"
            )
    elif any(counts.values()) or witnesses:
        raise QualificationInventoryError(
            f"{owner}: non-coverage stage must retain zero counts and no witnesses"
        )
    for key, witness in witnesses.items():
        _text(key, f"{owner}.witnesses key")
        location = _text(witness, f"{owner}.witnesses[{key!r}]")
        match = _MECHANISM_WITNESS_RE.fullmatch(location)
        if match is None:
            raise QualificationInventoryError(
                f"{owner}.witnesses[{key!r}] is not a canonical state/passage "
                "location"
            )
        state_index = int(match.group("state"))
        passage_index = int(match.group("passage"))
        if not (1 <= state_index <= n_states) or not (
            1 <= passage_index <= passages_per_state
        ):
            raise QualificationInventoryError(
                f"{owner}.witnesses[{key!r}] lies outside the authenticated "
                "state/passage inventory"
            )


def _validate_dataset(
    value: Any,
    *,
    expected_host: str,
    stage: str,
    policy: CurrentInventoryPolicy,
    owner: str,
) -> tuple[dict[str, Any], EndpointBinding]:
    if not isinstance(value, dict):
        raise QualificationInventoryError(f"{owner}: expected one JSON object")
    _exact_fields(value, _DATASET_FIELDS, owner)

    for field in _SHA_DATASET_FIELDS:
        _sha(value[field], f"{owner}.{field}")
    for field in _POSITIVE_DATASET_INTS:
        _positive_int(value[field], f"{owner}.{field}")
    for field in (
        "path",
        "actual_matlab_environment_descriptor",
        "campaign_matlab_environment_descriptor",
        "gen_schema",
        "channel_schema_id",
        "generation_behavior_version",
        "gen_fingerprint",
        "qualification_host_id",
        "qualification_host_canonical_descriptor",
    ):
        _text(value[field], f"{owner}.{field}")
    for field in (
        "qualification_hostname",
        "qualification_cpu_identifier",
        "qualification_computer_arch",
    ):
        _canonical_host_text(value[field], f"{owner}.{field}")
    _sha(value["gen_fingerprint"], f"{owner}.gen_fingerprint")
    for field in ("matlab_release", "campaign_matlab_release"):
        release = _text(value[field], f"{owner}.{field}")
        if not _MATLAB_RELEASE_RE.fullmatch(release):
            raise QualificationInventoryError(
                f"{owner}.{field}: malformed MATLAB release"
            )

    expected_stage_policy = comparator._STAGE_POLICIES[stage]
    expected_state_count = QUALIFICATION_STATE_COUNTS[stage]
    if int(expected_stage_policy["n_states"]) != expected_state_count:
        raise QualificationInventoryError(
            f"{owner}: comparator qualification count drifted for {stage}"
        )
    exact_expected: dict[str, Any] = {
        "stage": stage,
        "matlab_release": policy.campaign_matlab_release,
        "actual_matlab_environment_descriptor": (
            policy.campaign_matlab_environment_descriptor
        ),
        "actual_matlab_environment_sha256": (
            policy.campaign_matlab_environment_sha256
        ),
        "campaign_matlab_release": policy.campaign_matlab_release,
        "campaign_matlab_environment_descriptor": (
            policy.campaign_matlab_environment_descriptor
        ),
        "campaign_matlab_environment_sha256": (
            policy.campaign_matlab_environment_sha256
        ),
        "gen_schema": policy.gen_schema,
        "channel_schema_id": policy.channel_schema_id,
        "generation_behavior_version": policy.generation_behavior_version,
        "generator_source_root_sha256": policy.generator_source_root_sha256,
        "generator_source_file_count": policy.generator_source_file_count,
        "qualification_source_sha256": (
            policy.qualification_source_sha256[stage]
        ),
        "qualification_executed_file_sha256": (
            policy.qualification_executed_file_sha256[stage]
        ),
        "qualification_host_id": expected_host,
        "n_states": expected_state_count,
        "passages_per_state": QUALIFICATION_PASSAGES_PER_STATE,
        "num_supports": int(expected_stage_policy["num_supports"]),
        "max_parfor_workers": policy.max_parfor_workers,
    }
    for field, expected in exact_expected.items():
        if value[field] != expected:
            raise QualificationInventoryError(
                f"{owner}.{field} does not match current qualification policy"
            )
    if _sha256_text(
        value["actual_matlab_environment_descriptor"]
    ) != value["actual_matlab_environment_sha256"]:
        raise QualificationInventoryError(
            f"{owner}: actual MATLAB environment descriptor/SHA mismatch"
        )

    state_files = value["state_files"]
    expected_states = [
        f"{index:04d}.mat" for index in range(1, value["n_states"] + 1)
    ]
    if state_files != expected_states:
        raise QualificationInventoryError(
            f"{owner}.state_files is not the exact current micro inventory"
        )
    _validate_mechanism(
        value["mechanism_coverage"],
        stage=stage,
        n_states=value["n_states"],
        passages_per_state=value["passages_per_state"],
        owner=f"{owner}.mechanism_coverage",
    )
    _validate_host_descriptor(value, owner)

    binding = EndpointBinding(
        dataset_content_root_sha256=value["dataset_content_root_sha256"],
        actual_matlab_environment_descriptor=(
            value["actual_matlab_environment_descriptor"]
        ),
        actual_matlab_environment_sha256=(
            value["actual_matlab_environment_sha256"]
        ),
        matlab_release=value["matlab_release"],
        gen_fingerprint=value["gen_fingerprint"],
        generator_source_root_sha256=value["generator_source_root_sha256"],
        generator_source_file_count=value["generator_source_file_count"],
        qualification_source_sha256=value["qualification_source_sha256"],
        qualification_executed_file_sha256=(
            value["qualification_executed_file_sha256"]
        ),
        qualification_host_canonical_descriptor=(
            value["qualification_host_canonical_descriptor"]
        ),
        qualification_host_diagnostic_sha256=(
            value["qualification_host_diagnostic_sha256"]
        ),
        qualification_hostname=value["qualification_hostname"],
        qualification_cpu_identifier=value["qualification_cpu_identifier"],
        qualification_logical_processors=(
            value["qualification_logical_processors"]
        ),
        qualification_matlab_max_threads=(
            value["qualification_matlab_max_threads"]
        ),
        qualification_computer_arch=value["qualification_computer_arch"],
        mechanism_coverage_sha256=_sha256_bytes(
            json.dumps(
                value["mechanism_coverage"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
    )
    return value, binding


def _validate_comparison(
    value: Any,
    *,
    verdict: str,
    n_states: int,
    passages_per_state: int,
    raw_byte_identical_states: Any,
    owner: str,
) -> None:
    if not isinstance(value, dict):
        raise QualificationInventoryError(f"{owner}: expected one JSON object")
    _exact_fields(value, _COMPARISON_FIELDS, owner)
    for field in (
        "compared_leaves",
        "compared_numeric_values",
        "compared_signal_values",
    ):
        _positive_int(value[field], f"{owner}.{field}")
    for field in ("tolerant_leaves", "exact_leaves"):
        _nonnegative_int(value[field], f"{owner}.{field}")
    if value["compared_leaves"] != (
        value["exact_leaves"] + value["tolerant_leaves"]
    ):
        raise QualificationInventoryError(
            f"{owner}: compared_leaves must equal exact_leaves + "
            "tolerant_leaves"
        )
    if value["compared_signal_values"] > value["compared_numeric_values"]:
        raise QualificationInventoryError(
            f"{owner}: compared signal values exceed all compared numeric values"
        )
    if value["exact_leaves"] < 1 or value["tolerant_leaves"] < 1:
        raise QualificationInventoryError(
            f"{owner}: current qualification comparison must exercise both "
            "exact and tolerant leaves"
        )
    if value["tolerant_leaves"] > value["compared_numeric_values"]:
        raise QualificationInventoryError(
            f"{owner}: tolerant numeric leaves exceed compared numeric values"
        )
    absolute = _finite_nonnegative_number(
        value["max_absolute_difference"],
        f"{owner}.max_absolute_difference",
    )
    relative = _finite_nonnegative_number(
        value["max_relative_difference"],
        f"{owner}.max_relative_difference",
    )
    _text(value["worst_path"], f"{owner}.worst_path") if value[
        "worst_path"
    ] else None
    if type(value["numerical_difference"]) is not bool:
        raise QualificationInventoryError(
            f"{owner}.numerical_difference must be one JSON boolean"
        )
    if value["mismatches"] != []:
        raise QualificationInventoryError(
            f"{owner}.mismatches must be the exact empty list for acceptance"
        )
    if verdict == "SEMANTICALLY-BIT-IDENTICAL":
        if (
            value["numerical_difference"]
            or absolute != 0.0
            or relative != 0.0
            or value["worst_path"] != ""
        ):
            raise QualificationInventoryError(
                f"{owner}: exact verdict has numerical-difference evidence"
            )
    else:
        if (
            not value["numerical_difference"]
            or absolute <= 0.0
            or relative <= 0.0
            or not value["worst_path"]
        ):
            raise QualificationInventoryError(
                f"{owner}: numerical verdict lacks numerical-difference evidence"
            )
        worst_match = _NUMERICAL_WORST_PATH_RE.fullmatch(value["worst_path"])
        if worst_match is None:
            raise QualificationInventoryError(
                f"{owner}: numerical worst path is not one canonical "
                "state-payload tolerant-field path"
            )
        state_index = int(worst_match.group("state"))
        if not 1 <= state_index <= n_states:
            raise QualificationInventoryError(
                f"{owner}: numerical worst path state lies outside the "
                "authenticated state inventory"
            )
        passage = worst_match.group("passage")
        if (
            passage is not None
            and not 0 <= int(passage) < passages_per_state
        ):
            raise QualificationInventoryError(
                f"{owner}: numerical worst path passage index lies outside "
                "the three-passage qualification payload"
            )
    raw_count = _nonnegative_int(
        raw_byte_identical_states,
        f"{owner}.raw_byte_identical_states",
    )
    if raw_count > n_states:
        raise QualificationInventoryError(
            f"{owner}.raw_byte_identical_states exceeds authenticated states"
        )
    if verdict == "NUMERICALLY-EQUIVALENT" and raw_count == n_states:
        raise QualificationInventoryError(
            f"{owner}: numerical differences contradict raw identity of every "
            "authenticated state file"
        )


def _validate_tolerances(value: Any, owner: str) -> None:
    if not isinstance(value, dict):
        raise QualificationInventoryError(f"{owner}: expected one JSON object")
    _exact_fields(value, _TOLERANCE_FIELDS, owner)
    rtol = _finite_nonnegative_number(value["rtol"], f"{owner}.rtol")
    atol = _finite_nonnegative_number(value["atol"], f"{owner}.atol")
    if rtol > comparator._DEFAULT_RTOL or atol > comparator._DEFAULT_ATOL:
        raise QualificationInventoryError(
            f"{owner}: tolerances are looser than current comparator policy"
        )
    if value["tolerant_fields"] != sorted(comparator._TOLERANT_FLOAT_FIELDS):
        raise QualificationInventoryError(
            f"{owner}.tolerant_fields differs from current comparator policy"
        )


def _load_receipt(
    path: Path,
    *,
    intended_hosts: frozenset[str],
    policy: CurrentInventoryPolicy,
    receipt_bytes: bytes | None = None,
) -> tuple[
    ReceiptEdge,
    dict[tuple[str, str], EndpointBinding],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
]:
    """Authenticate one receipt and return every downstream-safe view of it.

    The final item is the payload parsed from the exact canonical bytes whose
    SHA-256 is stored in ``ReceiptEdge``.  Returning that object lets the graph
    validator rerun the pairwise comparison without reopening the mutable
    receipt pathname between schema validation and comparison revalidation.
    """
    try:
        raw = path.read_bytes() if receipt_bytes is None else receipt_bytes
        if not isinstance(raw, bytes):
            raise TypeError("receipt_bytes must be immutable bytes")
        text = raw.decode("utf-8")
    except (OSError, TypeError, UnicodeError) as exc:
        raise QualificationInventoryError(
            f"{path}: cannot read strict UTF-8 receipt: {exc}"
        ) from exc
    if (
        b"\r" in raw
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        raise QualificationInventoryError(
            f"{path}: receipt must use canonical LF text with one final LF"
        )
    payload = _strict_json_object(text, str(path))
    if raw != _canonical_json_bytes(payload):
        raise QualificationInventoryError(
            f"{path}: receipt is not the comparator's canonical JSON encoding"
        )
    _exact_fields(payload, _RECEIPT_FIELDS, str(path))

    current_expected: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "semantic_exactness_definition": _SEMANTIC_EXACTNESS_DEFINITION,
        "comparator_sha256": policy.comparator_sha256,
        "environment_lock_sha256": policy.environment_lock_sha256,
        "parser_environment": policy.parser_environment,
        "python_runtime_source_root_sha256": (
            policy.python_runtime_source_root_sha256
        ),
        "python_runtime_source_file_count": (
            policy.python_runtime_source_file_count
        ),
        "generator_source_root_sha256": policy.generator_source_root_sha256,
    }
    for field, expected in current_expected.items():
        if payload[field] != expected:
            raise QualificationInventoryError(
                f"{path}.{field} does not match current comparator/runtime/source "
                "provenance"
            )

    stage = _text(payload["stage"], f"{path}.stage")
    if stage not in REQUIRED_STAGES:
        raise QualificationInventoryError(
            f"{path}.stage={stage!r} is outside the mandatory inventory"
        )
    if (
        payload["qualification_source_sha256"]
        != policy.qualification_source_sha256[stage]
    ):
        raise QualificationInventoryError(
            f"{path}.qualification_source_sha256 is stale for {stage}"
        )

    verdict = _text(payload["verdict"], f"{path}.verdict")
    if verdict not in {
        "SEMANTICALLY-BIT-IDENTICAL",
        "NUMERICALLY-EQUIVALENT",
    }:
        raise QualificationInventoryError(
            f"{path}: verdict {verdict!r} is not acceptable"
        )
    acceptance = payload["numerical_equivalence_explicitly_accepted"]
    if type(acceptance) is not bool:
        raise QualificationInventoryError(
            f"{path}: numerical-equivalence acceptance must be a JSON boolean"
        )
    if verdict == "SEMANTICALLY-BIT-IDENTICAL" and acceptance:
        raise QualificationInventoryError(
            f"{path}: an exact verdict must not claim explicit numerical "
            "equivalence acceptance"
        )
    if verdict == "NUMERICALLY-EQUIVALENT" and not acceptance:
        raise QualificationInventoryError(
            f"{path}: numerical equivalence remains pending without explicit "
            "acceptance"
        )
    _validate_tolerances(payload["tolerances"], f"{path}.tolerances")

    host_list = payload["qualification_host_id"]
    if (
        not isinstance(host_list, list)
        or len(host_list) != 2
        or any(type(host) is not str for host in host_list)
    ):
        raise QualificationInventoryError(
            f"{path}.qualification_host_id must contain exactly two strings"
        )
    if host_list[0] == host_list[1]:
        raise QualificationInventoryError(
            f"{path}: a qualification edge must join distinct hosts"
        )
    if set(host_list) - intended_hosts:
        raise QualificationInventoryError(
            f"{path}: receipt names undeclared intended host IDs"
        )

    datasets: list[dict[str, Any]] = []
    bindings: dict[tuple[str, str], EndpointBinding] = {}
    retained_datasets: dict[tuple[str, str], dict[str, Any]] = {}
    for index, (field, host) in enumerate(
        zip(("dataset_a", "dataset_b"), host_list, strict=True)
    ):
        dataset, binding = _validate_dataset(
            payload[field],
            expected_host=host,
            stage=stage,
            policy=policy,
            owner=f"{path}.{field}",
        )
        datasets.append(dataset)
        bindings[(host, stage)] = binding
        retained_datasets[(host, stage)] = dataset
        for top_field, dataset_field in (
            ("dataset_content_roots_sha256", "dataset_content_root_sha256"),
            (
                "actual_matlab_environment_sha256",
                "actual_matlab_environment_sha256",
            ),
            ("qualification_host_id", "qualification_host_id"),
            (
                "qualification_host_diagnostic_sha256",
                "qualification_host_diagnostic_sha256",
            ),
        ):
            values = payload[top_field]
            if not isinstance(values, list) or len(values) != 2:
                raise QualificationInventoryError(
                    f"{path}.{top_field} must contain exactly two values"
                )
            if values[index] != dataset[dataset_field]:
                raise QualificationInventoryError(
                    f"{path}.{top_field}[{index}] is not bound to {field}"
                )

    if (
        datasets[0]["qualification_executed_file_sha256"]
        != datasets[1]["qualification_executed_file_sha256"]
    ):
        raise QualificationInventoryError(
            f"{path}: endpoint qualification executable identities differ"
        )
    if datasets[0]["gen_fingerprint"] != datasets[1]["gen_fingerprint"]:
        raise QualificationInventoryError(
            f"{path}: endpoint generation fingerprints differ"
        )
    if datasets[0]["mechanism_coverage"] != datasets[1]["mechanism_coverage"]:
        raise QualificationInventoryError(
            f"{path}: endpoint aggregate mechanism coverage differs"
        )
    if (
        datasets[0]["qualification_host_diagnostic_sha256"]
        == datasets[1]["qualification_host_diagnostic_sha256"]
    ):
        raise QualificationInventoryError(
            f"{path}: distinct hosts reuse one host-diagnostic identity"
        )

    _validate_comparison(
        payload["comparison"],
        verdict=verdict,
        n_states=datasets[0]["n_states"],
        passages_per_state=datasets[0]["passages_per_state"],
        raw_byte_identical_states=payload["raw_byte_identical_states"],
        owner=f"{path}.comparison",
    )
    if (
        datasets[0]["dataset_content_root_sha256"]
        == datasets[1]["dataset_content_root_sha256"]
        and payload["raw_byte_identical_states"] != datasets[0]["n_states"]
    ):
        raise QualificationInventoryError(
            f"{path}: equal dataset-content roots require raw identity of every "
            "authenticated state file"
        )
    hosts = sorted(host_list)
    edge = ReceiptEdge(
        stage=stage,
        host_a=hosts[0],
        host_b=hosts[1],
        receipt_sha256=_sha256_bytes(raw),
        path=path,
        structural_comparison_counts=(
            payload["comparison"]["compared_leaves"],
            payload["comparison"]["compared_numeric_values"],
            payload["comparison"]["compared_signal_values"],
            payload["comparison"]["exact_leaves"],
            payload["comparison"]["tolerant_leaves"],
        ),
    )
    return edge, bindings, retained_datasets, payload
