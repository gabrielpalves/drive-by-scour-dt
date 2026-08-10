"""Lightweight verification for exported digital-twin deployment assets.

This module deliberately depends only on the standard library plus the
import-light ``core.protocol`` and ``core.execution_environment`` validators.
Campaign bundles can therefore validate copied deployment artifacts without
importing the online digital-twin simulator (and its much larger physics
dependency graph).
"""

from __future__ import annotations

import hashlib
import json
import os

from core.execution_environment import (
    execution_block_for_stage,
    validate_execution_runtime,
)
from core.protocol import protocol_hash


_EXPECTED_BLOCK_REFERENCE_UNSET = object()
_LOWER_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_standalone_dt_package(
    model_path: str,
    metadata_path: str,
    scaler_path: str,
    *,
    expected_block_reference_manifest_sha256=(
        _EXPECTED_BLOCK_REFERENCE_UNSET
    ),
) -> dict:
    """Verify a copied deployment package without access to its Optuna DB."""
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    required = (
        "champion_weights_sha256",
        "scaler_sha256",
        "scaler_filename",
        "protocol_hash",
        "protocol_descriptor",
        "execution_environment_sha256",
        "execution_runtime",
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
        "selection_artifact_sha256",
        "selection_slot",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(
            f"DT package metadata lacks provenance fields: {missing}"
        )
    if _sha256_file(model_path) != metadata["champion_weights_sha256"]:
        raise RuntimeError("DT champion weights fail metadata SHA-256.")
    if os.path.basename(scaler_path) != metadata["scaler_filename"]:
        raise RuntimeError(
            "DT scaler path does not match metadata.scaler_filename."
        )
    if _sha256_file(scaler_path) != metadata["scaler_sha256"]:
        raise RuntimeError("DT scaler fails metadata SHA-256.")
    if protocol_hash(metadata["protocol_descriptor"]) != metadata["protocol_hash"]:
        raise RuntimeError(
            "DT protocol descriptor does not reproduce metadata.protocol_hash."
        )
    runtime = validate_execution_runtime(metadata["execution_runtime"])
    if (
        metadata["execution_environment_sha256"]
        != runtime["execution_environment_sha256"]
    ):
        raise RuntimeError(
            "DT execution descriptor does not reproduce metadata execution SHA."
        )
    descriptor = metadata["protocol_descriptor"]
    rung = descriptor.get("rung") if isinstance(descriptor, dict) else None
    if (
        not isinstance(rung, dict)
        or (
            runtime["execution_block"],
            runtime["anchor_stage"],
        ) != (
            rung.get("execution_block"),
            rung.get("execution_anchor"),
        )
    ):
        raise RuntimeError(
            "DT execution block disagrees with its protocol rung."
        )
    if not isinstance(metadata["campaign_run_tag"], str):
        raise RuntimeError("DT campaign run_tag must be text.")
    if not _is_sha256(metadata["execution_receipt_sha256"]):
        raise RuntimeError("DT execution receipt SHA-256 is invalid.")
    stage = rung.get("stage")
    anchor = rung.get("execution_anchor")
    block_reference_sha = metadata["block_reference_manifest_sha256"]
    if not isinstance(stage, str) or not stage:
        raise RuntimeError("DT protocol rung lacks its stage.")
    if not isinstance(anchor, str) or not anchor:
        raise RuntimeError("DT protocol rung lacks its execution anchor.")
    try:
        registered_block, registered_anchor = execution_block_for_stage(stage)
    except RuntimeError as exc:
        raise RuntimeError(
            "DT protocol rung carries an unregistered production stage."
        ) from exc
    if (rung.get("execution_block"), anchor) != (
        registered_block,
        registered_anchor,
    ):
        raise RuntimeError(
            "DT protocol rung disagrees with the registered execution block."
        )
    if stage == anchor:
        if block_reference_sha is not None:
            raise RuntimeError(
                "DT block-anchor package cannot cite its not-yet-published "
                "reference manifest."
            )
    elif not _is_sha256(block_reference_sha):
        raise RuntimeError(
            "DT follower package lacks a valid block-reference manifest SHA-256."
        )
    if (
        expected_block_reference_manifest_sha256
        is not _EXPECTED_BLOCK_REFERENCE_UNSET
    ):
        expected = expected_block_reference_manifest_sha256
        if expected is not None and not _is_sha256(expected):
            raise RuntimeError(
                "Expected block-reference manifest SHA-256 is invalid."
            )
        if block_reference_sha != expected:
            raise RuntimeError(
                "DT block-reference manifest SHA-256 differs from the "
                "independently supplied expectation."
            )
    selection_sha = metadata["selection_artifact_sha256"]
    selection_slot = metadata["selection_slot"]
    if metadata.get("hyperparameter_mode") == "selected_pair_hpo":
        if not _is_sha256(selection_sha):
            raise RuntimeError(
                "selected-pair DT package lacks its selection artefact SHA-256."
            )
        if selection_slot not in {
            "f40s_best_raw",
            "f40s_best_paa",
            "raw_cnn_gap_baseline",
            "paa_cnn_gap_baseline",
        }:
            raise RuntimeError("selected-pair DT package has a foreign slot.")
    elif selection_sha is not None or selection_slot is not None:
        raise RuntimeError(
            "non-selected-pair DT package carries selection artefact lineage."
        )
    return metadata
