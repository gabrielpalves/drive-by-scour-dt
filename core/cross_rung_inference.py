"""Registered common-random-number inference across the seven L60 edges.

The independent unit is a generated semantic ``StateUID``.  Only the immutable
outer-test subset of the 250-state joint master population enters the primary
contrasts.  Controlled anchors and dormant/rung-specific mechanisms remain
diagnostics and are deliberately excluded from the primary treatment estimand.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from core.campaign_contract import (
    EXPECTED_PROTOCOL_SCHEMA_TAG,
    campaign_stage_contract,
    validate_block_reference_manifest,
)
from core.dataset import SPLIT_SEED, STRATIFY_PATTERN
from core.execution_environment import (
    _read_regular_file,
    validate_execution_runtime,
)
from core.hyperparameter_policy import (
    HYPERPARAMETER_POLICY,
    canonical_json_sha256 as hyperparameter_json_sha256,
    policy_sha256 as hyperparameter_policy_sha256,
    validate_manifest as validate_hyperparameter_manifest,
)
from core.protocol import canonical_json, protocol_hash
from core.utils import IDX_TO_DOF_NAME


REGISTERED_L60_STAGES = (
    "s0_scour",
    "s11_bear",
    "s12_crack",
    "s13_bearcrack",
    "s14_prof",
    "s15_track",
    "s16_all",
)
REGISTERED_L60_EDGES = (
    ("s0_scour", "s11_bear"),
    ("s0_scour", "s12_crack"),
    ("s11_bear", "s13_bearcrack"),
    ("s12_crack", "s13_bearcrack"),
    ("s13_bearcrack", "s14_prof"),
    ("s14_prof", "s15_track"),
    ("s15_track", "s16_all"),
)
N_PRIMARY_EDGES = len(REGISTERED_L60_EDGES)
CROSS_RUNG_BOOTSTRAP_N = 100_000
CROSS_RUNG_BOOTSTRAP_SEED = 42
CROSS_RUNG_POINTWISE_COVERAGE = 0.95
CROSS_RUNG_FAMILYWISE_COVERAGE = 0.95
CROSS_RUNG_BONFERRONI_EDGE_COVERAGE = (
    1.0 - (1.0 - CROSS_RUNG_FAMILYWISE_COVERAGE) / N_PRIMARY_EDGES
)
SEMANTIC_SPLIT_SEED = SPLIT_SEED
SEMANTIC_SPLIT_PATTERN = tuple(STRATIFY_PATTERN)
CRN_INVARIANT_IDENTITY_FIELDS = (
    "damage_seed",
    "random_stream_schedule_version",
    "state_stream_names",
    "passage_stream_names",
    "passages_per_state",
    "state_uid_inventory",
    "state_uid_inventory_sha256",
    "state_seed_id_by_uid_sha256",
    "state_named_stream_by_uid_sha256",
    "passage_named_stream_by_uid_sha256",
    "joint_state_uid_inventory",
    "joint_state_uid_inventory_sha256",
    "family_counts",
    "latent_design_root_sha256",
)
HYPERPARAMETER_SOURCE_FIELDS = {
    "execution_block",
    "anchor_stage",
    "architecture",
    "seed",
    "study_identity_sha256",
    "params_sha256",
}

CROSS_RUNG_INFERENCE_POLICY = {
    "schema": "ttbi-l60-crn-cross-rung-inference-v1",
    "role": "pre-registered confirmatory L60 edge effects",
    "stages": list(REGISTERED_L60_STAGES),
    "primary_edges": [
        {"left": left, "right": right, "effect": f"{right} minus {left}"}
        for left, right in REGISTERED_L60_EDGES
    ],
    "reference_model": {
        "source": "authenticated s0 champion manifest",
        "architecture": "carried s0 reference architecture only",
        "sensor_pair": "carried s0 reference two-DOF pair only",
        "hyperparameters": "authenticated L60/s0 full-eight-DOF calibration, "
                           "frozen exactly by architecture and training seed",
        "deployment_winner_substitution": "forbidden",
    },
    "execution": {
        "block": "l60",
        "require_same_protocol_core_hash": True,
        "require_same_execution_receipt_sha256": True,
    },
    "generated_population": {
        "family": "joint",
        "master_joint_state_uids": 250,
        "require_identical_full_uid_inventory_across_l60": True,
        "require_identical_uid_partition_across_l60": True,
        "semantic_split": {
            "schema": "ttbi-semantic-split-v1",
            "seed": SEMANTIC_SPLIT_SEED,
            "pattern": list(SEMANTIC_SPLIT_PATTERN),
            "assignment": "SHA-256-seeded permutation of lexicographically "
                          "sorted StateUIDs within each semantic stratum",
        },
        "require_identical_crn_identity_fields":
            list(CRN_INVARIANT_IDENTITY_FIELDS),
        "active_treatment_fields_excluded_from_latent_design_root": [
            "CrackOn", "BearingFixity"
        ],
    },
    "evaluation_population": {
        "family": "joint",
        "partition": "canonical immutable outer test",
        "rule": "exact common outer-joint StateUID set at both endpoints",
        "anchors": "controlled-anchor and other-family results are diagnostics "
                   "only and excluded from the primary delta",
    },
    "cells": {
        "layout": "StateUID x registered training seed",
        "completeness": "every exact outer-joint UID x seed cell once",
        "pairing": "same UID and same registered seed at both endpoints",
    },
    "metric": {
        "name": "state-level scour MSE",
        "units": "squared percentage points",
        "effect_direction": "right-stage minus left-stage; positive means "
                            "higher error after adding the edge mechanism",
    },
    "statistic": {
        "within_seed": "mean across outer-joint states",
        "across_seed": "median across the finite registered seed set",
        "edge_effect": "right-stage statistic minus left-stage statistic",
        "seed_scope": "conditional on the registered finite seed set; seeds are "
                      "paired but never resampled",
    },
    "bootstrap": {
        "unit": "StateUID first, paired across endpoints and seeds",
        "n_boot": CROSS_RUNG_BOOTSTRAP_N,
        "seed": CROSS_RUNG_BOOTSTRAP_SEED,
        "pointwise_coverage": CROSS_RUNG_POINTWISE_COVERAGE,
        "familywise_method": "Bonferroni across exactly seven primary edges",
        "familywise_coverage": CROSS_RUNG_FAMILYWISE_COVERAGE,
        "per_edge_familywise_coverage":
            CROSS_RUNG_BONFERRONI_EDGE_COVERAGE,
        "minimum_expected_draws_per_familywise_tail":
            CROSS_RUNG_BOOTSTRAP_N
            * (1.0 - CROSS_RUNG_BONFERRONI_EDGE_COVERAGE) / 2.0,
        "sign_claim_rule": "only the Bonferroni-familywise interval may support "
                           "an across-ladder sign claim",
    },
    "estimand": {
        "scope": "performance change after adding each registered mechanism, "
                 "for the carried s0 architecture/pair with the exact "
                 "architecture-by-seed hyperparameters calibrated once on the "
                 "s0 full-eight-DOF control and frozen across every L60 rung",
        "training": "models are refit independently at every rung with paired "
                    "training seeds; only HPO is frozen",
        "rung_specific_hpo": "forbidden",
        "conditioning": "conditional on the s0-selected reference model, the "
                        "registered generated-state population, and finite "
                        "training-seed set",
    },
    "secondary_interaction": {
        "name": "bearing-by-crack difference-in-differences",
        "formula": "(s13_bearcrack - s12_crack) - "
                   "(s11_bear - s0_scour)",
        "population": "same exact outer-joint StateUID x seed cells",
        "status": "secondary exploratory; not an eighth primary edge",
        "interval": "pointwise 95% state-first paired bootstrap only",
    },
}


def _validate_registered_policy() -> None:
    exact_edges = (
        ("s0_scour", "s11_bear"),
        ("s0_scour", "s12_crack"),
        ("s11_bear", "s13_bearcrack"),
        ("s12_crack", "s13_bearcrack"),
        ("s13_bearcrack", "s14_prof"),
        ("s14_prof", "s15_track"),
        ("s15_track", "s16_all"),
    )
    if REGISTERED_L60_EDGES != exact_edges:
        raise RuntimeError("registered L60 primary-edge family was mutated")
    expected_policy_edges = [
        {"left": left, "right": right, "effect": f"{right} minus {left}"}
        for left, right in exact_edges
    ]
    if (
        CROSS_RUNG_INFERENCE_POLICY.get("primary_edges")
        != expected_policy_edges
        or CROSS_RUNG_INFERENCE_POLICY.get("stages")
        != list(REGISTERED_L60_STAGES)
        or CROSS_RUNG_INFERENCE_POLICY.get("bootstrap", {}).get(
            "familywise_method"
        ) != "Bonferroni across exactly seven primary edges"
        or CROSS_RUNG_INFERENCE_POLICY.get("bootstrap", {}).get("n_boot")
        != 100_000
        or CROSS_RUNG_INFERENCE_POLICY.get("bootstrap", {}).get(
            "pointwise_coverage"
        ) != 0.95
        or CROSS_RUNG_INFERENCE_POLICY.get("bootstrap", {}).get(
            "familywise_coverage"
        ) != 0.95
        or CROSS_RUNG_INFERENCE_POLICY.get("bootstrap", {}).get(
            "per_edge_familywise_coverage"
        ) != 1.0 - 0.05 / 7.0
        or CROSS_RUNG_BOOTSTRAP_N != 100_000
        or CROSS_RUNG_BOOTSTRAP_SEED != 42
        or CROSS_RUNG_POINTWISE_COVERAGE != 0.95
        or CROSS_RUNG_FAMILYWISE_COVERAGE != 0.95
        or CROSS_RUNG_BONFERRONI_EDGE_COVERAGE != 1.0 - 0.05 / 7.0
        or N_PRIMARY_EDGES != 7
        or CROSS_RUNG_INFERENCE_POLICY.get(
            "generated_population", {}
        ).get("semantic_split") != {
            "schema": "ttbi-semantic-split-v1",
            "seed": 42,
            "pattern": ["train", "test", "val", "train", "train"],
            "assignment": "SHA-256-seeded permutation of lexicographically "
                          "sorted StateUIDs within each semantic stratum",
        }
        or SEMANTIC_SPLIT_SEED != 42
        or SEMANTIC_SPLIT_PATTERN
        != ("train", "test", "val", "train", "train")
    ):
        raise RuntimeError("registered L60 inference policy was mutated")


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty registered result {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _regular_file(path: Path, owner: str) -> Path:
    if not os.path.lexists(path):
        raise RuntimeError(f"{owner}: missing required file {path}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{owner}: {path} must be one regular local file")
    return path


def _load_json_snapshot(
    path: Path,
    owner: str,
    *,
    max_bytes: int = 16 << 20,
) -> tuple[dict, str]:
    """Parse and hash one safely-opened immutable snapshot of a JSON input."""

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        raw = _read_regular_file(path, max_bytes=max_bytes)
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{owner}: missing required JSON {path}") from exc
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{owner}: malformed JSON {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{owner}: {path} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_csv_snapshot(
    path: Path,
    owner: str,
    *,
    max_bytes: int = 64 << 20,
) -> tuple[list[dict[str, str]], list[str], str]:
    """Parse and hash the same safely-opened byte snapshot of one CSV input."""

    try:
        raw = _read_regular_file(path, max_bytes=max_bytes)
        text = raw.decode("utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"{owner}: missing required CSV {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"{owner}: unreadable CSV {path}") from exc
    try:
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
    except csv.Error as exc:
        raise RuntimeError(f"{owner}: malformed CSV {path}") from exc
    return rows, fields, hashlib.sha256(raw).hexdigest()


def _hex64(value, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"{owner} must be one lowercase SHA-256 hex digest")
    return value


def _registered_execution_receipt(receipt_path: Path) -> dict:
    if not receipt_path.is_absolute():
        raise RuntimeError(
            "execution receipt input must be an absolute durable path"
        )
    try:
        raw = _read_regular_file(receipt_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"missing required external execution receipt {receipt_path}"
        ) from exc

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-finite constant {value!r}")

    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("execution receipt is not strict ASCII JSON") from exc
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema",
            "execution_block",
            "anchor_stage",
            "protocol_core_hash",
            "run_tag",
            "execution_runtime",
        }
        or receipt.get("schema") != "ttbi-execution-block-receipt-v1"
        or receipt.get("execution_block") != "l60"
        or receipt.get("anchor_stage") != "s0_scour"
        or not isinstance(receipt.get("run_tag"), str)
    ):
        raise RuntimeError("malformed registered L60 execution receipt")
    canonical = canonical_json(receipt).encode("ascii") + b"\n"
    if raw != canonical:
        raise RuntimeError("execution receipt bytes are not exact canonical JSON")
    runtime = validate_execution_runtime(receipt["execution_runtime"])
    if (
        runtime["execution_block"] != "l60"
        or runtime["anchor_stage"] != "s0_scour"
    ):
        raise RuntimeError("execution receipt runtime is not the L60/s0 binding")
    return {
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_core_hash": _hex64(
            receipt["protocol_core_hash"],
            "execution receipt protocol_core_hash",
        ),
        "run_tag": receipt["run_tag"],
        "execution_runtime": runtime,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _registered_hyperparameter_reference(
    manifest_path: Path,
    *,
    receipt_reference: dict,
) -> dict:
    if not manifest_path.is_absolute():
        raise RuntimeError(
            "hyperparameter manifest input must be an absolute durable path"
        )
    loaded_manifest, manifest_file_sha = _load_json_snapshot(
        manifest_path, "hyperparameter manifest"
    )
    manifest = validate_hyperparameter_manifest(
        loaded_manifest,
        expected_runtime=receipt_reference["execution_runtime"],
        expected_run_tag=receipt_reference["run_tag"],
        expected_execution_receipt_sha256=
            receipt_reference["receipt_sha256"],
    )
    manifest_sha = hyperparameter_json_sha256(manifest)
    if manifest_file_sha != manifest_sha:
        raise RuntimeError(
            "hyperparameter manifest file is not the unique canonical JSON "
            "representation authenticated by its SHA-256"
        )
    if (
        manifest.get("policy") != HYPERPARAMETER_POLICY
        or manifest.get("execution_block") != "l60"
        or manifest.get("anchor_stage") != "s0_scour"
    ):
        raise RuntimeError(
            "cross-rung analysis requires the authenticated L60/s0 "
            "hyperparameter manifest"
        )
    sources = {}
    for entry in manifest["entries"]:
        key = (entry["architecture"], int(entry["seed"]))
        sources[key] = {
            "execution_block": "l60",
            "anchor_stage": "s0_scour",
            "architecture": entry["architecture"],
            "seed": int(entry["seed"]),
            "study_identity_sha256": entry["study_identity_sha256"],
            "params_sha256": entry["params_sha256"],
        }
    return {
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_file_sha,
        "protocol_core_hash": manifest["protocol_core_hash"],
        "anchor_protocol_hash": manifest["anchor_protocol_hash"],
        "anchor_dataset": manifest["anchor_dataset"],
        "run_tag": manifest["run_tag"],
        "execution_receipt_sha256":
            manifest["execution_receipt_sha256"],
        "execution_runtime": manifest["execution_runtime"],
        "sources": sources,
    }


def _registered_reference(
    champion_path: Path,
    *,
    expected_block_reference_sha256: str,
    hyperparameter_reference: dict,
    hyperparameter_manifest_sha256: str,
    receipt_reference: dict,
) -> dict:
    if not champion_path.is_absolute():
        raise RuntimeError(
            "block-reference manifest input must be an absolute durable path"
        )
    expected_reference_sha = _hex64(
        expected_block_reference_sha256,
        "externally supplied block-reference SHA-256",
    )
    champion, champion_file_sha = _load_json_snapshot(
        champion_path, "champion manifest"
    )
    champion_canonical_sha = hyperparameter_json_sha256(champion)
    if champion_canonical_sha != expected_reference_sha:
        raise RuntimeError(
            "supplied block-reference manifest differs from the independently "
            "supplied canonical SHA-256"
        )
    champion = validate_block_reference_manifest(
        champion,
        expected_anchor_stage="s0_scour",
        expected_dataset=campaign_stage_contract("s0_scour")["dataset"],
        expected_schema=EXPECTED_PROTOCOL_SCHEMA_TAG,
        expected_run_tag=receipt_reference["run_tag"],
        expected_seeds=HYPERPARAMETER_POLICY["seeds"],
        expected_anchor_n_trials=HYPERPARAMETER_POLICY[
            "anchor_hpo"
        ]["n_trials"],
        expected_candidate_n_trials=HYPERPARAMETER_POLICY[
            "frozen_singleton"
        ]["n_trials"],
        expected_protocol_core_hash=
            hyperparameter_reference["protocol_core_hash"],
        expected_anchor_protocol_hash=
            hyperparameter_reference["anchor_protocol_hash"],
        expected_execution_runtime=receipt_reference["execution_runtime"],
        expected_execution_receipt_sha256=
            receipt_reference["receipt_sha256"],
        expected_hyperparameter_manifest_sha256=
            hyperparameter_manifest_sha256,
        expected_hyperparameter_policy_sha256=
            hyperparameter_policy_sha256(),
        valid_architectures=HYPERPARAMETER_POLICY["architectures"],
        valid_dofs=tuple(IDX_TO_DOF_NAME),
    )
    champion_runtime = validate_execution_runtime(
        champion["execution_runtime"]
    )
    champion_receipt_sha = champion["execution_receipt_sha256"]
    frozen_selection_sha = champion["frozen_selection_sha256"]
    architecture = champion["champion_arch"]
    pair = list(champion["champion_pair"])
    seeds = champion["seeds"]
    return {
        "architecture": architecture,
        "pair": pair,
        "dofs_label": "+".join(IDX_TO_DOF_NAME[value] for value in pair),
        "seeds": list(seeds),
        "protocol_core_hash": _hex64(
            champion["protocol_core_hash"],
            "champion protocol_core_hash",
        ),
        "execution_receipt_sha256": champion_receipt_sha,
        "run_tag": receipt_reference["run_tag"],
        "hyperparameter_manifest_sha256":
            hyperparameter_manifest_sha256,
        "execution_runtime": champion_runtime,
        "execution_environment_sha256":
            champion_runtime["execution_environment_sha256"],
        "champion_manifest_sha256":
            champion_canonical_sha,
        "externally_supplied_block_reference_sha256":
            expected_reference_sha,
        "champion_file_sha256": champion_file_sha,
        "capacity_preflight_receipt_sha256":
            champion["capacity_preflight_receipt_sha256"],
        "protocol_core": champion["protocol_core"],
        "schema": champion["schema"],
        "frozen_selection_sha256": frozen_selection_sha,
    }


def _canonical_hyperparameter_source(raw: str, owner: str) -> dict:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{owner}: empty hyperparameter_source_json")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-finite constant {value!r}")

    try:
        source = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{owner}: malformed hyperparameter_source_json"
        ) from exc
    if (
        not isinstance(source, dict)
        or set(source) != HYPERPARAMETER_SOURCE_FIELDS
        or canonical_json(source) != raw
    ):
        raise RuntimeError(
            f"{owner}: hyperparameter_source_json is noncanonical or has "
            "the wrong field set"
        )
    return source


def _validate_state_identity(stage: str, provenance: dict) -> tuple[dict, dict]:
    identity = provenance.get("state_identity")
    split = provenance.get("semantic_split")
    if not isinstance(identity, dict) or not isinstance(split, dict):
        raise RuntimeError(
            f"{stage}: protocol lacks semantic state identity/split provenance"
        )
    identity_fields = {
        "schema",
        "damage_seed",
        "random_stream_schedule_version",
        "state_stream_names",
        "passage_stream_names",
        "passages_per_state",
        "state_uid_count",
        "state_uid_inventory",
        "state_uid_inventory_sha256",
        "state_uid_row_order",
        "state_uid_row_order_sha256",
        "state_seed_id_by_uid_sha256",
        "state_named_stream_by_uid_sha256",
        "passage_named_stream_by_uid_sha256",
        "joint_state_uid_count",
        "joint_state_uid_inventory",
        "joint_state_uid_inventory_sha256",
        "family_counts",
        "latent_design_root_sha256",
        "state_identity_root_sha256",
    }
    if (
        set(identity) != identity_fields
        or identity.get("schema") != "ttbi-semantic-state-identity-v2"
    ):
        raise RuntimeError(f"{stage}: wrong semantic-state identity schema")
    inventory = identity.get("state_uid_inventory")
    row_order = identity.get("state_uid_row_order")
    joint = identity.get("joint_state_uid_inventory")
    if (
        not isinstance(inventory, list)
        or not isinstance(row_order, list)
        or not isinstance(joint, list)
        or inventory != sorted(inventory)
        or joint != sorted(joint)
        or len(inventory) != len(set(inventory))
        or len(row_order) != len(set(row_order))
        or set(inventory) != set(row_order)
        or not set(joint).issubset(inventory)
    ):
        raise RuntimeError(f"{stage}: malformed UID inventory/row mapping")
    expected_states = campaign_stage_contract(stage)["sampling"]["n_states"]
    expected_sampling = campaign_stage_contract(stage)["sampling"]
    if len(inventory) != expected_states:
        raise RuntimeError(
            f"{stage}: UID inventory has {len(inventory)} states, expected "
            f"{expected_states}"
        )
    if len(joint) != 250 or identity.get("joint_state_uid_count") != 250:
        raise RuntimeError(
            f"{stage}: registered joint master population must contain exactly "
            "250 semantic UIDs"
        )
    if identity.get("state_uid_count") != len(inventory):
        raise RuntimeError(f"{stage}: state_uid_count disagrees with inventory")
    if identity.get("state_uid_row_order_sha256") != hashlib.sha256(
        canonical_json(row_order).encode("utf-8")
    ).hexdigest():
        raise RuntimeError(f"{stage}: UID row-order digest mismatch")
    if identity.get("family_counts") != expected_sampling["family_counts"]:
        raise RuntimeError(
            f"{stage}: semantic family counts differ from campaign contract"
        )
    if (
        isinstance(identity.get("damage_seed"), bool)
        or not isinstance(identity.get("damage_seed"), int)
        or not 0 <= identity["damage_seed"] <= np.iinfo(np.uint32).max
        or identity.get("random_stream_schedule_version")
        != "uid-named-substreams-v2"
        or identity.get("state_stream_names")
        != ["operations", "crack", "profile-state", "track", "profile-phase"]
        or identity.get("passage_stream_names")
        != ["profile-passage", "oor-passage"]
        or identity.get("passages_per_state")
        != expected_sampling["passages_per_state"]
    ):
        raise RuntimeError(f"{stage}: malformed registered CRN schedule identity")
    for field in (
        "state_seed_id_by_uid_sha256",
        "state_named_stream_by_uid_sha256",
        "passage_named_stream_by_uid_sha256",
        "latent_design_root_sha256",
        "state_identity_root_sha256",
    ):
        _hex64(identity.get(field), f"{stage} {field}")
    if identity.get("state_uid_inventory_sha256") != hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest():
        raise RuntimeError(f"{stage}: UID inventory digest mismatch")
    if identity.get("joint_state_uid_inventory_sha256") != hashlib.sha256(
        canonical_json(joint).encode("utf-8")
    ).hexdigest():
        raise RuntimeError(f"{stage}: joint UID inventory digest mismatch")
    split_fields = {
        "schema",
        "seed",
        "assignment_by_uid",
        "assignment_by_uid_sha256",
        "partition_counts",
    }
    records = split.get("assignment_by_uid")
    if (
        set(split) != split_fields
        or split.get("schema") != "ttbi-semantic-split-v1"
        or split.get("seed") != SEMANTIC_SPLIT_SEED
        or not isinstance(records, list)
        or len(records) != len(inventory)
    ):
        raise RuntimeError(f"{stage}: malformed semantic split descriptor")
    by_uid = {}
    observed_seed_ids: set[int] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "state_uid", "state_seed_id", "stratum", "partition"
        }:
            raise RuntimeError(f"{stage}: malformed UID split record")
        uid = record["state_uid"]
        if uid in by_uid or uid not in set(inventory):
            raise RuntimeError(f"{stage}: duplicate/foreign UID in split record")
        state_seed_id = record["state_seed_id"]
        if (
            isinstance(state_seed_id, bool)
            or not isinstance(state_seed_id, int)
            or not 1 <= state_seed_id <= np.iinfo(np.uint32).max
            or state_seed_id in observed_seed_ids
        ):
            raise RuntimeError(f"{stage}: malformed/duplicate state-stream root")
        observed_seed_ids.add(state_seed_id)
        if record["partition"] not in ("train", "val", "test"):
            raise RuntimeError(f"{stage}: invalid UID partition")
        if not isinstance(record["stratum"], str) or not record["stratum"]:
            raise RuntimeError(f"{stage}: empty/non-text semantic stratum")
        by_uid[uid] = record
    if sorted(by_uid) != inventory:
        raise RuntimeError(f"{stage}: split records do not cover UID inventory")
    if [record["state_uid"] for record in records] != inventory:
        raise RuntimeError(
            f"{stage}: semantic split records are not in canonical UID order"
        )
    if split.get("assignment_by_uid_sha256") != hashlib.sha256(
        canonical_json(records).encode("utf-8")
    ).hexdigest():
        raise RuntimeError(f"{stage}: semantic split assignment digest mismatch")
    seed_id_pairs = [
        [uid, int(by_uid[uid]["state_seed_id"])] for uid in sorted(by_uid)
    ]
    if identity.get("state_seed_id_by_uid_sha256") != hashlib.sha256(
        canonical_json(seed_id_pairs).encode("utf-8")
    ).hexdigest():
        raise RuntimeError(
            f"{stage}: UID-to-state-stream digest disagrees with split records"
        )
    observed_counts = {
        partition: sum(
            record["partition"] == partition for record in records
        )
        for partition in ("train", "val", "test")
    }
    if split.get("partition_counts") != observed_counts:
        raise RuntimeError(
            f"{stage}: semantic split partition counts are inconsistent"
        )
    expected_partition: dict[str, str] = {}
    for stratum in sorted({record["stratum"] for record in records}):
        members = sorted(
            record["state_uid"]
            for record in records
            if record["stratum"] == stratum
        )
        digest = hashlib.sha256(
            f"{SEMANTIC_SPLIT_SEED}|{stratum}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        permutation = rng.permutation(len(members))
        for position, member_position in enumerate(permutation):
            expected_partition[members[int(member_position)]] = (
                SEMANTIC_SPLIT_PATTERN[
                    position % len(SEMANTIC_SPLIT_PATTERN)
                ]
            )
    observed_partition = {
        record["state_uid"]: record["partition"] for record in records
    }
    if observed_partition != expected_partition:
        raise RuntimeError(
            f"{stage}: UID partition is not the registered deterministic "
            "within-stratum assignment"
        )
    return identity, split


def _read_stage(
    stage: str,
    summary_dir: Path,
    reference: dict,
    hyperparameter_reference: dict,
) -> dict:
    if summary_dir.is_symlink() or not summary_dir.is_dir():
        raise RuntimeError(f"{stage}: summary path must be a regular directory")
    protocol_path = _regular_file(
        summary_dir / "protocol_descriptor.json", stage
    )
    frozen_path = _regular_file(summary_dir / "frozen_selection.json", stage)
    metrics_path = _regular_file(
        summary_dir / "outer_test_state_metrics.csv", stage
    )
    record, protocol_file_sha = _load_json_snapshot(protocol_path, stage)
    descriptor = record.get("descriptor")
    if not isinstance(descriptor, dict) or set(descriptor) != {"core", "rung"}:
        raise RuntimeError(f"{stage}: malformed full protocol descriptor")
    full_hash = _hex64(record.get("protocol_hash"), f"{stage} protocol_hash")
    core_hash = _hex64(
        record.get("protocol_core_hash"), f"{stage} protocol_core_hash"
    )
    if protocol_hash(descriptor) != full_hash:
        raise RuntimeError(f"{stage}: full protocol hash does not match descriptor")
    if protocol_hash(descriptor["core"]) != core_hash:
        raise RuntimeError(f"{stage}: core protocol hash does not match descriptor")
    if descriptor["core"] != reference["protocol_core"]:
        raise RuntimeError(
            f"{stage}: protocol-core descriptor differs from the exact "
            "block-reference descriptor"
        )
    if core_hash != reference["protocol_core_hash"]:
        raise RuntimeError(f"{stage}: protocol core differs from s0 reference")
    if core_hash != hyperparameter_reference["protocol_core_hash"]:
        raise RuntimeError(
            f"{stage}: protocol core differs from the authenticated "
            "hyperparameter calibration manifest"
        )
    if (
        stage == "s0_scour"
        and full_hash != hyperparameter_reference["anchor_protocol_hash"]
    ):
        raise RuntimeError(
            "s0 full protocol hash differs from the hyperparameter anchor study"
        )
    embedded_policy = (
        descriptor["core"].get("selection", {})
        .get("statistical_inference", {})
        .get("cross_rung_crn")
    )
    if embedded_policy != CROSS_RUNG_INFERENCE_POLICY:
        raise RuntimeError(
            f"{stage}: hash-carried cross-rung inference policy differs from "
            "the executing registered policy"
        )
    rung = descriptor["rung"]
    expected_dataset = campaign_stage_contract(stage)["dataset"]
    if (
        rung.get("stage") != stage
        or rung.get("dataset") != expected_dataset
        or rung.get("execution_block") != "l60"
        or rung.get("execution_anchor") != "s0_scour"
        or Path(summary_dir).name == ""
    ):
        raise RuntimeError(
            f"{stage}: protocol rung/dataset does not match registered contract"
        )
    seeds = (
        descriptor.get("core", {})
        .get("optuna", {})
        .get("seeds")
    )
    if seeds != reference["seeds"]:
        raise RuntimeError(f"{stage}: protocol seed set/order differs from champion")
    receipt = _hex64(
        record.get("execution_receipt_sha256"),
        f"{stage} execution receipt",
    )
    if receipt != reference["execution_receipt_sha256"]:
        raise RuntimeError(
            f"{stage}: L60 stage uses a different execution receipt"
        )
    if (
        record.get("block_reference_manifest_sha256")
        != reference["champion_manifest_sha256"]
    ):
        raise RuntimeError(
            f"{stage}: protocol record does not pin the supplied canonical "
            "block-reference manifest SHA-256"
        )
    capacity_record = record.get("capacity_preflight_receipt")
    if (
        record.get("run_tag") != reference["run_tag"]
        or record.get("hyperparameter_manifest_sha256")
        != hyperparameter_reference["manifest_sha256"]
        or not isinstance(capacity_record, dict)
        or capacity_record.get("receipt_sha256")
        != reference["capacity_preflight_receipt_sha256"]
    ):
        raise RuntimeError(
            f"{stage}: protocol record does not reproduce the authenticated "
            "run/HPO/capacity lineage"
        )
    stage_runtime = validate_execution_runtime(
        record.get("execution_runtime")
    )
    if (
        stage_runtime != reference["execution_runtime"]
        or stage_runtime != hyperparameter_reference["execution_runtime"]
    ):
        raise RuntimeError(
            f"{stage}: execution runtime differs across protocol, champion, "
            "and hyperparameter calibration manifest"
        )
    identity, split = _validate_state_identity(
        stage, rung.get("dataset_provenance", {})
    )
    frozen, frozen_file_sha = _load_json_snapshot(frozen_path, stage)
    frozen_canonical_sha = hyperparameter_json_sha256(frozen)
    frozen_runtime = validate_execution_runtime(
        frozen.get("execution_runtime")
    )
    expected_artifact_reference = (
        None
        if stage == "s0_scour"
        else reference["externally_supplied_block_reference_sha256"]
    )
    if (
        frozen.get("stage") != stage
        or frozen.get("protocol_hash") != full_hash
        or frozen.get("protocol_core_hash") != core_hash
        or "campaign_run_tag" not in frozen
        or frozen.get("campaign_run_tag") != reference["run_tag"]
        or "block_reference_manifest_sha256" not in frozen
        or frozen.get("block_reference_manifest_sha256")
        != expected_artifact_reference
        or frozen.get("execution_receipt_sha256") != receipt
        or frozen.get("capacity_preflight_receipt_sha256")
        != reference["capacity_preflight_receipt_sha256"]
        or frozen.get("hyperparameter_manifest_sha256")
        != hyperparameter_reference["manifest_sha256"]
        or frozen_runtime != stage_runtime
        or frozen.get("execution_environment_sha256")
        != stage_runtime["execution_environment_sha256"]
    ):
        raise RuntimeError(f"{stage}: frozen selection provenance mismatch")
    if stage == "s0_scour":
        selected_pair = frozen.get("selected_pair")
        if (
            not isinstance(selected_pair, list)
            or len(selected_pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in selected_pair
            )
            or selected_pair != sorted(selected_pair)
            or len(set(selected_pair)) != 2
            or any(value not in IDX_TO_DOF_NAME for value in selected_pair)
        ):
            raise RuntimeError(
                "s0 frozen selection requires two sorted, distinct, strict "
                "integer registered DOFs"
            )
        if (
            frozen.get("architecture") != reference["architecture"]
            or selected_pair != reference["pair"]
            or frozen.get("deployment_selection") is not False
            or frozen_canonical_sha != reference["frozen_selection_sha256"]
        ):
            raise RuntimeError(
                "s0 frozen selection does not equal the champion reference"
            )

    rows, metric_fields, metrics_file_sha = _load_csv_snapshot(
        metrics_path, stage
    )
    required_columns = {
        "stage", "protocol_hash", "protocol_core_hash",
        "execution_receipt_sha256", "architecture", "dofs", "seed", "repeat",
        "state", "state_uid", "state_seed_id", "family", "scour_mse",
        "campaign_run_tag", "block_reference_manifest_sha256",
        "hyperparameter_manifest_sha256", "hyperparameter_source_json",
    }
    if not rows or not metric_fields:
        raise RuntimeError(f"{stage}: outer-test state metrics are empty")
    if len(metric_fields) != len(set(metric_fields)):
        raise RuntimeError(f"{stage}: outer metrics contain duplicate CSV headers")
    if not required_columns.issubset(metric_fields):
        raise RuntimeError(
            f"{stage}: outer metrics lack "
            f"{sorted(required_columns - set(metric_fields))}"
        )
    expected_csv_reference = (
        ""
        if expected_artifact_reference is None
        else expected_artifact_reference
    )
    if any(row.get("campaign_run_tag") != reference["run_tag"] for row in rows):
        raise RuntimeError(
            f"{stage}: outer metrics do not all cite the authenticated "
            "campaign run tag"
        )
    if any(
        row.get("block_reference_manifest_sha256")
        != expected_csv_reference
        for row in rows
    ):
        raise RuntimeError(
            f"{stage}: outer metrics do not all carry the registered "
            "anchor-null/follower-pinned block-reference lineage"
        )
    selected = [
        row for row in rows
        if row["architecture"] == reference["architecture"]
        and row["dofs"] == reference["dofs_label"]
    ]
    if not selected:
        raise RuntimeError(
            f"{stage}: outer metrics do not contain the carried s0 reference; "
            "a deployment winner cannot substitute for it"
        )
    row_order = identity["state_uid_row_order"]
    joint_inventory = set(identity["joint_state_uid_inventory"])
    expected_outer_joint = sorted(
        record["state_uid"]
        for record in split["assignment_by_uid"]
        if record["state_uid"] in joint_inventory
        and record["partition"] == "test"
    )
    seed_id_by_uid = {
        record["state_uid"]: int(record["state_seed_id"])
        for record in split["assignment_by_uid"]
    }
    if len(expected_outer_joint) < 2:
        raise RuntimeError(f"{stage}: outer-joint partition has fewer than 2 states")
    cells: dict[tuple[str, int], float] = {}
    state_by_uid: dict[str, int] = {}
    for row in selected:
        if (
            row["stage"] != stage
            or row["protocol_hash"] != full_hash
            or row["protocol_core_hash"] != core_hash
            or row["execution_receipt_sha256"] != receipt
            or row["hyperparameter_manifest_sha256"]
            != hyperparameter_reference["manifest_sha256"]
        ):
            raise RuntimeError(f"{stage}: per-row outer-metric provenance mismatch")
        try:
            seed = int(row["seed"])
            repeat = int(row["repeat"])
            state = int(row["state"])
            state_seed_id = int(row["state_seed_id"])
            error = float(row["scour_mse"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{stage}: malformed numeric metric cell") from exc
        if (
            seed not in reference["seeds"]
            or repeat != 0
            or not math.isfinite(error)
            or error < 0.0
        ):
            raise RuntimeError(
                f"{stage}: unregistered seed/repeat or invalid scour MSE"
            )
        source = _canonical_hyperparameter_source(
            row["hyperparameter_source_json"],
            f"{stage}: outer metric seed {seed}",
        )
        expected_source = hyperparameter_reference["sources"].get(
            (reference["architecture"], seed)
        )
        if source != expected_source:
            raise RuntimeError(
                f"{stage}: outer metric does not cite the exact authenticated "
                f"s0 hyperparameters for architecture={reference['architecture']!r}, "
                f"seed={seed}"
            )
        uid = row["state_uid"]
        if (
            row["family"] != "joint"
            and uid in joint_inventory
        ):
            raise RuntimeError(f"{stage}: joint UID was relabelled as another family")
        if row["family"] != "joint":
            continue
        if uid not in joint_inventory:
            raise RuntimeError(f"{stage}: foreign UID labelled joint")
        if not (0 <= state < len(row_order)) or row_order[state] != uid:
            raise RuntimeError(
                f"{stage}: state row {state} does not map to UID {uid!r}"
            )
        if state_seed_id != seed_id_by_uid[uid]:
            raise RuntimeError(
                f"{stage}: state-stream root does not map to UID {uid!r}"
            )
        key = (uid, seed)
        if key in cells:
            raise RuntimeError(f"{stage}: duplicate outer UID x seed cell {key}")
        cells[key] = error
        previous = state_by_uid.setdefault(uid, state)
        if previous != state:
            raise RuntimeError(f"{stage}: UID maps to different rows across seeds")
    observed_uids = sorted({uid for uid, _seed in cells})
    if observed_uids != expected_outer_joint:
        raise RuntimeError(
            f"{stage}: outer reference joint UID set differs from the registered "
            "semantic split"
        )
    expected_cells = {
        (uid, seed)
        for uid in expected_outer_joint
        for seed in reference["seeds"]
    }
    if set(cells) != expected_cells:
        raise RuntimeError(
            f"{stage}: incomplete/extra outer-joint UID x seed cells"
        )
    matrix = np.asarray([
        [cells[(uid, seed)] for seed in reference["seeds"]]
        for uid in expected_outer_joint
    ], dtype=np.float64)
    return {
        "stage": stage,
        "protocol_hash": full_hash,
        "protocol_core_hash": core_hash,
        "execution_receipt_sha256": receipt,
        "campaign_run_tag": reference["run_tag"],
        "block_reference_manifest_sha256": expected_artifact_reference,
        "hyperparameter_manifest_sha256":
            hyperparameter_reference["manifest_sha256"],
        "identity": identity,
        "split": split,
        "outer_joint_uids": expected_outer_joint,
        "matrix": matrix,
        "input_sha256": {
            "protocol_descriptor.json": protocol_file_sha,
            "frozen_selection.json": frozen_file_sha,
            "outer_test_state_metrics.csv": metrics_file_sha,
        },
    }


def _statistic(matrix: np.ndarray) -> float:
    """Median of registered-seed mean state MSEs."""

    return float(np.median(np.asarray(matrix, dtype=np.float64).mean(axis=0)))


def _bootstrap_statistics(
    matrix: np.ndarray,
    state_positions: np.ndarray,
) -> np.ndarray:
    """Vectorized statistic for a batch of state-resampling index matrices."""

    # matrix[state_positions] is bootstrap x state x registered-seed. State
    # aggregation precedes the finite-seed median exactly as preregistered.
    return np.median(matrix[state_positions].mean(axis=1), axis=1)


def _edge_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if left.shape != right.shape or left.ndim != 2:
        raise RuntimeError("paired edge matrices must be UID x seed and aligned")
    if left.shape[0] < 2 or left.shape[1] < 1:
        raise RuntimeError("paired edge needs >=2 states and >=1 seed")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("paired edge contains non-finite errors")
    estimate = _statistic(right) - _statistic(left)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    batch_size = 4096
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        state_positions = rng.integers(
            0,
            left.shape[0],
            size=(stop - start, left.shape[0]),
        )
        draws[start:stop] = (
            _bootstrap_statistics(right, state_positions)
            - _bootstrap_statistics(left, state_positions)
        )
    return estimate, draws


def _interval(draws: np.ndarray, coverage: float) -> tuple[float, float]:
    alpha = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def analyze_registered_l60_contrasts(
    stage_summary_dirs: Mapping[str, str | os.PathLike[str]],
    champion_manifest: str | os.PathLike[str],
    hyperparameter_manifest: str | os.PathLike[str],
    execution_receipt: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_block_reference_sha256: str,
) -> dict:
    """Validate all registered artifacts, compute, and atomically publish results."""

    _validate_registered_policy()
    if set(stage_summary_dirs) != set(REGISTERED_L60_STAGES):
        raise RuntimeError(
            "stage-summary mapping must contain exactly the seven registered "
            f"L60 stages {REGISTERED_L60_STAGES}"
        )
    champion_path = Path(champion_manifest)
    receipt_reference = _registered_execution_receipt(
        Path(execution_receipt)
    )
    hyperparameter_path = _regular_file(
        Path(hyperparameter_manifest),
        "hyperparameter manifest",
    )
    hyperparameter_reference = _registered_hyperparameter_reference(
        hyperparameter_path,
        receipt_reference=receipt_reference,
    )
    reference = _registered_reference(
        champion_path,
        expected_block_reference_sha256=
            expected_block_reference_sha256,
        hyperparameter_reference=hyperparameter_reference,
        hyperparameter_manifest_sha256=
            hyperparameter_reference["manifest_sha256"],
        receipt_reference=receipt_reference,
    )
    if (
        reference["protocol_core_hash"]
        != hyperparameter_reference["protocol_core_hash"]
        or reference["protocol_core_hash"]
        != receipt_reference["protocol_core_hash"]
        or campaign_stage_contract("s0_scour")["dataset"]
        != hyperparameter_reference["anchor_dataset"]
        or reference["execution_runtime"]
        != hyperparameter_reference["execution_runtime"]
        or reference["execution_runtime"]
        != receipt_reference["execution_runtime"]
        or hyperparameter_reference["execution_receipt_sha256"]
        != receipt_reference["receipt_sha256"]
        or hyperparameter_reference["run_tag"]
        != receipt_reference["run_tag"]
    ):
        raise RuntimeError(
            "champion/protocol campaign identity differs across the supplied "
            "hyperparameter calibration manifest and execution receipt"
        )
    stages = {
        stage: _read_stage(
            stage,
            Path(stage_summary_dirs[stage]),
            reference,
            hyperparameter_reference,
        )
        for stage in REGISTERED_L60_STAGES
    }
    baseline = stages["s0_scour"]
    for stage, record in stages.items():
        changed_crn_fields = [
            field for field in CRN_INVARIANT_IDENTITY_FIELDS
            if record["identity"].get(field)
            != baseline["identity"].get(field)
        ]
        if (
            changed_crn_fields
            or record["split"]["assignment_by_uid"]
            != baseline["split"]["assignment_by_uid"]
            or record["outer_joint_uids"] != baseline["outer_joint_uids"]
        ):
            raise RuntimeError(
                f"{stage}: L60 CRN identity/partition differs from s0 "
                f"(changed CRN fields={changed_crn_fields}); paired cross-rung "
                "inference is not valid"
            )
    uids = baseline["outer_joint_uids"]
    seeds = reference["seeds"]
    summary_rows: list[dict] = []
    cell_rows: list[dict] = []
    for edge_index, (left_stage, right_stage) in enumerate(
        REGISTERED_L60_EDGES, start=1
    ):
        left = stages[left_stage]["matrix"]
        right = stages[right_stage]["matrix"]
        estimate, draws = _edge_bootstrap(
            left,
            right,
            n_boot=CROSS_RUNG_BOOTSTRAP_N,
            seed=CROSS_RUNG_BOOTSTRAP_SEED,
        )
        point_lo, point_hi = _interval(
            draws, CROSS_RUNG_POINTWISE_COVERAGE
        )
        family_lo, family_hi = _interval(
            draws, CROSS_RUNG_BONFERRONI_EDGE_COVERAGE
        )
        edge_name = f"{left_stage}->{right_stage}"
        summary_rows.append({
            "analysis_role": "primary preregistered edge",
            "edge_index": edge_index,
            "edge": edge_name,
            "left_stage": left_stage,
            "right_stage": right_stage,
            "effect": "right minus left state-level scour MSE",
            "estimate": estimate,
            "pointwise_ci95_lo": point_lo,
            "pointwise_ci95_hi": point_hi,
            "familywise_bonferroni_ci95_lo": family_lo,
            "familywise_bonferroni_ci95_hi": family_hi,
            "familywise_per_edge_coverage":
                CROSS_RUNG_BONFERRONI_EDGE_COVERAGE,
            "descriptive_bootstrap_fraction_positive":
                float(np.mean(draws > 0.0)),
            "n_outer_joint_state_uids": len(uids),
            "n_registered_seeds": len(seeds),
            "bootstrap_n": CROSS_RUNG_BOOTSTRAP_N,
            "bootstrap_seed": CROSS_RUNG_BOOTSTRAP_SEED,
            "inference_scope": "conditional on registered finite seed set",
            "estimand": CROSS_RUNG_INFERENCE_POLICY["estimand"]["scope"],
        })
        for uid_index, uid in enumerate(uids):
            for seed_index, registered_seed in enumerate(seeds):
                left_error = float(left[uid_index, seed_index])
                right_error = float(right[uid_index, seed_index])
                cell_rows.append({
                    "analysis_role": "primary preregistered edge",
                    "edge": edge_name,
                    "state_uid": uid,
                    "seed": registered_seed,
                    "left_scour_mse": left_error,
                    "right_scour_mse": right_error,
                    "difference_right_minus_left": right_error - left_error,
                })

    # Secondary 2x2 interaction; it is intentionally outside the seven-edge
    # confirmatory family and receives only its pointwise exploratory interval.
    s0 = stages["s0_scour"]["matrix"]
    s11 = stages["s11_bear"]["matrix"]
    s12 = stages["s12_crack"]["matrix"]
    s13 = stages["s13_bearcrack"]["matrix"]
    interaction_estimate = (
        (_statistic(s13) - _statistic(s12))
        - (_statistic(s11) - _statistic(s0))
    )
    rng = np.random.default_rng(CROSS_RUNG_BOOTSTRAP_SEED)
    interaction_draws = np.empty(CROSS_RUNG_BOOTSTRAP_N, dtype=np.float64)
    batch_size = 4096
    for start in range(0, CROSS_RUNG_BOOTSTRAP_N, batch_size):
        stop = min(start + batch_size, CROSS_RUNG_BOOTSTRAP_N)
        positions = rng.integers(
            0,
            len(uids),
            size=(stop - start, len(uids)),
        )
        interaction_draws[start:stop] = (
            (
                _bootstrap_statistics(s13, positions)
                - _bootstrap_statistics(s12, positions)
            )
            - (
                _bootstrap_statistics(s11, positions)
                - _bootstrap_statistics(s0, positions)
            )
        )
    interaction_lo, interaction_hi = _interval(
        interaction_draws, CROSS_RUNG_POINTWISE_COVERAGE
    )
    summary_rows.append({
        "analysis_role": "secondary exploratory interaction",
        "edge_index": "",
        "edge": "bearing_x_crack_DiD",
        "left_stage": "(s11_bear-s0_scour)",
        "right_stage": "(s13_bearcrack-s12_crack)",
        "effect": "(s13-s12)-(s11-s0)",
        "estimate": interaction_estimate,
        "pointwise_ci95_lo": interaction_lo,
        "pointwise_ci95_hi": interaction_hi,
        "familywise_bonferroni_ci95_lo": "",
        "familywise_bonferroni_ci95_hi": "",
        "familywise_per_edge_coverage": "",
        "descriptive_bootstrap_fraction_positive":
            float(np.mean(interaction_draws > 0.0)),
        "n_outer_joint_state_uids": len(uids),
        "n_registered_seeds": len(seeds),
        "bootstrap_n": CROSS_RUNG_BOOTSTRAP_N,
        "bootstrap_seed": CROSS_RUNG_BOOTSTRAP_SEED,
        "inference_scope": "secondary exploratory; conditional on registered "
                           "finite seed set",
        "estimand": "bearing-by-crack interaction for the carried s0 reference "
                    "with s0-calibrated architecture-by-seed hyperparameters "
                    "frozen across the four cells",
    })
    for uid_index, uid in enumerate(uids):
        for seed_index, registered_seed in enumerate(seeds):
            cell_value = (
                (s13[uid_index, seed_index] - s12[uid_index, seed_index])
                - (s11[uid_index, seed_index] - s0[uid_index, seed_index])
            )
            cell_rows.append({
                "analysis_role": "secondary exploratory interaction",
                "edge": "bearing_x_crack_DiD",
                "state_uid": uid,
                "seed": registered_seed,
                "left_scour_mse": "",
                "right_scour_mse": "",
                "difference_right_minus_left": float(cell_value),
            })

    destination = Path(output_dir)
    if os.path.lexists(destination) and (
        destination.is_symlink() or not destination.is_dir()
    ):
        raise RuntimeError("output_dir must be a regular local directory")
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "registered_l60_cross_rung_contrasts.csv"
    cells_path = destination / "registered_l60_cross_rung_state_cells.csv"
    manifest_path = destination / "registered_l60_cross_rung_manifest.json"
    _atomic_csv(summary_path, summary_rows)
    _atomic_csv(cells_path, cell_rows)
    manifest = {
        "schema": "ttbi-l60-crn-cross-rung-results-v1",
        "policy": CROSS_RUNG_INFERENCE_POLICY,
        "policy_sha256": protocol_hash(CROSS_RUNG_INFERENCE_POLICY),
        "reference": reference,
        "externally_supplied_block_reference_sha256":
            reference["externally_supplied_block_reference_sha256"],
        "protocol_core_hash": reference["protocol_core_hash"],
        "execution_receipt_sha256": reference["execution_receipt_sha256"],
        "execution_receipt": {
            "file_sha256": receipt_reference["file_sha256"],
            "protocol_core_hash": receipt_reference["protocol_core_hash"],
            "run_tag": receipt_reference["run_tag"],
        },
        "hyperparameter_manifest": {
            "canonical_sha256":
                hyperparameter_reference["manifest_sha256"],
            "file_sha256":
                hyperparameter_reference["manifest_file_sha256"],
            "execution_receipt_sha256":
                hyperparameter_reference["execution_receipt_sha256"],
            "run_tag": hyperparameter_reference["run_tag"],
        },
        "state_uid_inventory_sha256":
            baseline["identity"]["state_uid_inventory_sha256"],
        "joint_state_uid_inventory_sha256":
            baseline["identity"]["joint_state_uid_inventory_sha256"],
        "semantic_split_assignment_sha256":
            baseline["split"]["assignment_by_uid_sha256"],
        "outer_joint_state_uids": uids,
        "outer_joint_state_uid_sha256": hashlib.sha256(
            canonical_json(uids).encode("utf-8")
        ).hexdigest(),
        "registered_seeds": seeds,
        "stage_inputs": {
            stage: {
                "protocol_hash": record["protocol_hash"],
                "input_sha256": record["input_sha256"],
            }
            for stage, record in stages.items()
        },
        "outputs": {
            summary_path.name: _sha256_file(summary_path),
            cells_path.name: _sha256_file(cells_path),
        },
        "interpretation": {
            "primary": "seven edge effects with pointwise and Bonferroni-"
                       "familywise 95% intervals; only familywise intervals "
                       "support across-ladder sign claims",
            "secondary": "bearing-by-crack difference-in-differences is "
                         "exploratory and not an eighth primary edge",
            "seed_scope": "all intervals are conditional on the registered "
                          "finite training-seed set",
        },
    }
    _atomic_json(manifest_path, manifest)
    return {
        "summary_rows": summary_rows,
        "cell_rows": cell_rows,
        "manifest": manifest,
        "summary_path": str(summary_path),
        "cells_path": str(cells_path),
        "manifest_path": str(manifest_path),
    }
