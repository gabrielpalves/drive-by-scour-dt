"""Lightweight verification for exported digital-twin deployment assets.

This module deliberately depends only on the standard library and
``core.protocol``.  Campaign bundles can therefore validate copied deployment
artifacts without importing the online digital-twin simulator (and its much
larger physics dependency graph).
"""

from __future__ import annotations

import hashlib
import json
import os

from core.protocol import protocol_hash


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
    return metadata
