"""Authenticated F40-S selection artefact for Paper-1 downstream training.

The outcome-dependent values needed after factorial HPO, development
adjudication, and frozen-hyperparameter channel screening live in one canonical
JSON object.  The object resolves every prospective retained-pipeline slot and
the single selected physical sensor pair.  Downstream jobs cite its SHA-256;
they do not infer winners from result directories or accept a command-line
architecture/pair override.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.paper1_training_contract import (
    CHANNEL_SCHEMA_ID,
    ELIGIBLE_SENSOR_INDICES,
    FACTORIAL_CELLS,
    PAA_CNN_GAP_BASELINE_ID,
    RAW_CNN_GAP_BASELINE_ID,
    RETAINED_PIPELINE_SLOTS,
    STAGE_ORDER,
    canonical_json_bytes,
    complete_job_grid,
)


SELECTION_ARTIFACT_SCHEMA = "paper1-f40s-selection-artifact-v1"
SELECTION_ARTIFACT_ENV = "TTBI_PAPER1_SELECTION_ARTIFACT"
SELECTION_ARTIFACT_SHA256_ENV = "TTBI_PAPER1_SELECTION_ARTIFACT_SHA256"
SELECTION_METRIC = "paired state-clustered grouped-development OOF MSE"
_LOWER_HEX = frozenset("0123456789abcdef")
_FIELDS = {
    "schema",
    "channel_schema_id",
    "selection_stage",
    "applicable_stages",
    "campaign_run_tag",
    "complete_grid_sha256",
    "selection_metric",
    "selected_pair",
    "slot_resolution",
    "canonical_slot",
    "evidence_sha256",
    "artifact_sha256",
}
_EVIDENCE_FIELDS = {
    "factorial_hpo_manifest",
    "development_adjudication_manifest",
    "channel_screen_manifest",
}


class Paper1SelectionError(RuntimeError):
    """A selected pair/slot claim is absent, malformed, or unauthenticated."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _validate_pipeline(value: object, *, representation: str | None = None) -> str:
    registered = {cell.cell_id: cell for cell in FACTORIAL_CELLS}
    if not isinstance(value, str) or value not in registered:
        raise Paper1SelectionError(
            f"selection resolved an unregistered pipeline {value!r}"
        )
    if representation is not None and registered[value].representation != representation:
        raise Paper1SelectionError(
            f"selection pipeline {value!r} is not {representation}"
        )
    return value


def _expected_canonical_slots(resolution: Mapping[str, str]) -> dict[str, str]:
    first_for_pipeline: dict[str, str] = {}
    result: dict[str, str] = {}
    for slot in RETAINED_PIPELINE_SLOTS:
        pipeline = resolution[slot]
        canonical = first_for_pipeline.setdefault(pipeline, slot)
        result[slot] = canonical
    return result


def build_selection_artifact(
    *,
    campaign_run_tag: str,
    selected_pair: list[int] | tuple[int, int],
    best_raw: str,
    best_paa: str,
    evidence_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build and validate one canonical outcome-selection object."""

    resolution = {
        "f40s_best_raw": best_raw,
        "f40s_best_paa": best_paa,
        "raw_cnn_gap_baseline": RAW_CNN_GAP_BASELINE_ID,
        "paa_cnn_gap_baseline": PAA_CNN_GAP_BASELINE_ID,
    }
    value: dict[str, Any] = {
        "schema": SELECTION_ARTIFACT_SCHEMA,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "selection_stage": "F40-S",
        "applicable_stages": list(STAGE_ORDER),
        "campaign_run_tag": campaign_run_tag,
        "complete_grid_sha256": complete_job_grid()["complete_grid_sha256"],
        "selection_metric": SELECTION_METRIC,
        "selected_pair": list(selected_pair),
        "slot_resolution": resolution,
        "canonical_slot": _expected_canonical_slots(resolution),
        "evidence_sha256": dict(evidence_sha256),
    }
    value["artifact_sha256"] = _artifact_sha256(value)
    return validate_selection_artifact(value)


def validate_selection_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact fields, pair, slots, evidence hashes, and self-digest."""

    if not isinstance(value, Mapping):
        raise Paper1SelectionError("selection artefact must be a mapping")
    try:
        result = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Paper1SelectionError(
            f"selection artefact is not canonical finite JSON: {exc}"
        ) from exc
    if set(result) != _FIELDS:
        raise Paper1SelectionError("selection artefact fields differ from contract")
    if result["schema"] != SELECTION_ARTIFACT_SCHEMA:
        raise Paper1SelectionError("unsupported selection artefact schema")
    if result["channel_schema_id"] != CHANNEL_SCHEMA_ID:
        raise Paper1SelectionError("selection uses a foreign channel schema")
    if result["selection_stage"] != "F40-S":
        raise Paper1SelectionError("selection stage must be F40-S")
    if result["applicable_stages"] != list(STAGE_ORDER):
        raise Paper1SelectionError("selection applicable-stage order drifted")
    if not isinstance(result["campaign_run_tag"], str) or not result[
        "campaign_run_tag"
    ]:
        raise Paper1SelectionError("selection campaign run tag is empty/invalid")
    if (
        result["complete_grid_sha256"]
        != complete_job_grid()["complete_grid_sha256"]
    ):
        raise Paper1SelectionError("selection training-contract digest drifted")
    if result["selection_metric"] != SELECTION_METRIC:
        raise Paper1SelectionError("selection metric drifted")
    pair = result["selected_pair"]
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
        or pair != sorted(pair)
        or len(set(pair)) != 2
        or any(index not in ELIGIBLE_SENSOR_INDICES for index in pair)
    ):
        raise Paper1SelectionError(
            "selected pair must be two distinct ascending eligible-sensor indices"
        )
    resolution = result["slot_resolution"]
    if not isinstance(resolution, dict) or set(resolution) != set(
        RETAINED_PIPELINE_SLOTS
    ):
        raise Paper1SelectionError("selection slot inventory/order drifted")
    _validate_pipeline(resolution["f40s_best_raw"], representation="RAW")
    _validate_pipeline(resolution["f40s_best_paa"], representation="PAA")
    if resolution["raw_cnn_gap_baseline"] != RAW_CNN_GAP_BASELINE_ID:
        raise Paper1SelectionError("RAW baseline slot was changed")
    if resolution["paa_cnn_gap_baseline"] != PAA_CNN_GAP_BASELINE_ID:
        raise Paper1SelectionError("PAA baseline slot was changed")
    for pipeline in resolution.values():
        _validate_pipeline(pipeline)
    canonical = result["canonical_slot"]
    expected_canonical = _expected_canonical_slots(resolution)
    if not isinstance(canonical, dict) or canonical != expected_canonical:
        raise Paper1SelectionError("selection slot deduplication mapping is invalid")
    evidence = result["evidence_sha256"]
    if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_FIELDS:
        raise Paper1SelectionError("selection evidence inventory drifted")
    if not all(_is_sha256(digest) for digest in evidence.values()):
        raise Paper1SelectionError("selection evidence SHA-256 is invalid")
    if (
        not _is_sha256(result["artifact_sha256"])
        or result["artifact_sha256"] != _artifact_sha256(result)
    ):
        raise Paper1SelectionError("selection artefact SHA-256 is invalid")
    return result


def load_selection_artifact(
    path: str | os.PathLike[str] | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one canonical selection artefact and verify an external digest.

    Production callers resolve ``path`` and ``expected_sha256`` from separate
    environment variables.  Supplying ``path`` explicitly remains useful to
    offline publication/checking code, where the returned self-digest can be
    recorded independently before dispatch.
    """

    from_environment = path is None
    raw_path = (
        os.fspath(path)
        if path is not None
        else os.environ.get(SELECTION_ARTIFACT_ENV, "")
    )
    if not raw_path:
        raise Paper1SelectionError(f"{SELECTION_ARTIFACT_ENV} is required")
    source = Path(raw_path)
    if not source.is_absolute():
        raise Paper1SelectionError(
            f"{SELECTION_ARTIFACT_ENV} must be an absolute path"
        )
    if source.is_symlink() or not source.is_file():
        raise Paper1SelectionError(
            "selection artefact must be one regular, non-symlink file"
        )
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Paper1SelectionError(
            f"selection artefact is unreadable/non-JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise Paper1SelectionError("selection artefact bytes are not canonical JSON")
    result = validate_selection_artifact(value)
    expected = (
        expected_sha256
        if expected_sha256 is not None
        else os.environ.get(SELECTION_ARTIFACT_SHA256_ENV, "")
        if from_environment
        else None
    )
    if from_environment and not expected:
        raise Paper1SelectionError(
            f"{SELECTION_ARTIFACT_SHA256_ENV} is required"
        )
    if expected is not None:
        if not _is_sha256(expected):
            raise Paper1SelectionError(
                "expected selection artefact SHA-256 is invalid"
            )
        if result["artifact_sha256"] != expected:
            raise Paper1SelectionError(
                "selection artefact differs from its external SHA-256"
            )
    return result


def resolve_selection_claim(
    artifact: Mapping[str, Any],
    *,
    stage: str,
    slot: str,
    pair: list[int] | tuple[int, ...] | None = None,
    architecture: str | None = None,
    campaign_run_tag: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve and cross-check one downstream stage/slot/pair claim."""

    value = validate_selection_artifact(artifact)
    if stage not in STAGE_ORDER or stage not in value["applicable_stages"]:
        raise Paper1SelectionError(f"selection does not apply to stage {stage!r}")
    if slot not in RETAINED_PIPELINE_SLOTS:
        raise Paper1SelectionError(f"unregistered retained-pipeline slot {slot!r}")
    selected_pair = value["selected_pair"]
    resolved_architecture = value["slot_resolution"][slot]
    if pair is not None and list(pair) != selected_pair:
        raise Paper1SelectionError("job pair differs from selection artefact")
    if architecture is not None and architecture != resolved_architecture:
        raise Paper1SelectionError("job architecture differs from selected slot")
    if campaign_run_tag is not None and campaign_run_tag != value[
        "campaign_run_tag"
    ]:
        raise Paper1SelectionError("job run tag differs from selection artefact")
    if artifact_sha256 is not None and artifact_sha256 != value[
        "artifact_sha256"
    ]:
        raise Paper1SelectionError("job selection SHA-256 is wrong")
    return {
        "stage": stage,
        "slot": slot,
        "canonical_slot": value["canonical_slot"][slot],
        "architecture": resolved_architecture,
        "selected_pair": selected_pair,
        "artifact_sha256": value["artifact_sha256"],
        "campaign_run_tag": value["campaign_run_tag"],
    }
