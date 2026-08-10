"""Fail-closed disk revalidation of retained qualification evidence.

Contract: pair receipts are unsigned JSON, so a coherent fabricated receipt
graph - or a genuine graph whose underlying datasets were later deleted,
moved, or mutated - is indistinguishable from real evidence by schema and
graph checks alone.  This module closes that gap:

* ``revalidate_endpoint`` canonically binds one retained dataset directory
  (absolute, self-resolving, real, reached without any symlink/junction),
  recomputes its complete evidence from disk through the same comparator
  entry point the original qualification run used
  (``compare_generation_releases._validated_payload``: full per-file SHA-256
  digest-table verification, ``_GENERATION_COMPLETE`` marker, host receipt,
  and every state payload's provenance), and requires the recomputed
  ``DatasetEvidence`` to equal the retained dataset block field for field.
* ``revalidate_edge_comparison`` reruns the complete pairwise comparison
  (``compare_generation_releases.compare_directories``) between the two
  retained endpoints at the retained tolerances and requires the recomputed
  verdict, both evidence blocks, every comparison statistic, and the
  raw-byte-identity count to equal the retained receipt exactly.

Rationale: this module deliberately holds its own path canonicaliser instead
of importing ``dispatch_authorization`` (which would be circular through
``qualification_receipt_inventory``).  Neither function accepts any argument
that can skip or weaken the recomputation; every deviation fails closed
through ``QualificationInventoryError`` and names the differing field.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import compare_generation_releases as comparator
from qualification_receipt_schema import (
    QualificationInventoryError,
    _COMPARISON_FIELDS,
    _DATASET_FIELDS,
    _exact_fields,
    _finite_nonnegative_number,
    _text,
)


def current_comparator_policy() -> comparator.CurrentPolicy:
    """Authenticate the current repository-side comparator policy."""
    try:
        return comparator._current_policy()
    except comparator.QualificationInputError as exc:
        raise QualificationInventoryError(
            f"cannot authenticate current comparator policy: {exc}"
        ) from exc


def _is_junction(path: Path) -> bool:
    predicate = getattr(os.path, "isjunction", None)
    return bool(predicate is not None and predicate(path))


def _canonical_existing_directory(raw: Any, owner: str) -> Path:
    """Require ``raw`` to be the canonical resolved path of one real directory.

    Deliberately mirrors dispatch_authorization's evidence-path canonicaliser
    (duplicated here to keep the import graph acyclic): the retained text must
    be absolute, must resolve strictly to itself, must denote a real directory,
    and no path component may be a symlink or junction.
    """
    text = _text(raw, owner)
    supplied = Path(text)
    if not supplied.is_absolute():
        raise QualificationInventoryError(
            f"{owner}: retained dataset path must be absolute: {text!r}"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise QualificationInventoryError(
            f"{owner}: retained dataset directory cannot be resolved "
            f"(missing, deleted, or moved after qualification): {supplied}"
        ) from exc
    if text != str(resolved):
        raise QualificationInventoryError(
            f"{owner}: retained dataset path is not the canonical resolved "
            f"path: {text!r} != {str(resolved)!r}"
        )
    current = supplied
    while True:
        try:
            if current.is_symlink() or _is_junction(current):
                raise QualificationInventoryError(
                    f"{owner}: retained dataset path traverses a "
                    f"symlink/junction: {current}"
                )
        except OSError as exc:
            raise QualificationInventoryError(
                f"{owner}: retained dataset path component cannot be "
                f"inspected: {current}"
            ) from exc
        if current.parent == current:
            break
        current = current.parent
    try:
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except OSError as exc:
        raise QualificationInventoryError(
            f"{owner}: retained dataset directory cannot be inspected: "
            f"{resolved}"
        ) from exc
    if not stat.S_ISDIR(mode):
        raise QualificationInventoryError(
            f"{owner}: retained dataset path must be one real directory: "
            f"{resolved}"
        )
    return resolved


def _retained_json_view(value: Any) -> Any:
    """Render recomputed dataclass content in exact retained-JSON form."""
    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False)
    )


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Type-strict JSON value equality (True is never equal to 1)."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _require_identical_fields(
    recomputed: dict[str, Any],
    retained: dict[str, Any],
    field_names: frozenset[str],
    *,
    owner: str,
) -> None:
    differing = sorted(
        name
        for name in field_names
        if not _strict_json_equal(recomputed[name], retained[name])
    )
    if differing:
        raise QualificationInventoryError(
            f"{owner}: retained evidence differs from the full from-disk "
            f"recomputation at field(s) {differing}"
        )


def _require_result_policy_bindings(
    result: comparator.ComparisonResult,
    retained: dict[str, Any],
    policy: comparator.CurrentPolicy,
    *,
    owner: str,
) -> None:
    """Bind every repository/tolerance field used by the comparison."""
    top_level = {
        "environment_lock_sha256": result.environment_lock_sha256,
        "parser_environment": result.parser_environment,
        "python_runtime_source_root_sha256": (
            result.python_runtime_source_root_sha256
        ),
        "python_runtime_source_file_count": (
            result.python_runtime_source_file_count
        ),
        "stage": result.evidence_a.stage,
        "generator_source_root_sha256": (
            result.evidence_a.generator_source_root_sha256
        ),
        "qualification_source_sha256": (
            result.evidence_a.qualification_source_sha256
        ),
    }
    differing = sorted(
        field
        for field, expected in top_level.items()
        if field not in retained
        or not _strict_json_equal(expected, retained[field])
    )
    if differing:
        raise QualificationInventoryError(
            f"{owner}: retained comparison-policy bindings differ from the "
            f"recomputed result at field(s) {differing}"
        )

    expected_tolerances = {
        "rtol": result.tolerance_rtol,
        "atol": result.tolerance_atol,
        "tolerant_fields": sorted(comparator._TOLERANT_FLOAT_FIELDS),
    }
    if not _strict_json_equal(
        expected_tolerances, retained.get("tolerances")
    ):
        raise QualificationInventoryError(
            f"{owner}: retained tolerances differ from the exact tolerances "
            "used by the recomputed ComparisonResult"
        )

    result_policy = {
        "environment_lock_sha256": result.environment_lock_sha256,
        "parser_environment": result.parser_environment,
        "python_runtime_source_root_sha256": (
            result.python_runtime_source_root_sha256
        ),
        "python_runtime_source_file_count": (
            result.python_runtime_source_file_count
        ),
    }
    current_result_policy = {
        "environment_lock_sha256": policy.environment_lock_sha256,
        "parser_environment": policy.parser_environment,
        "python_runtime_source_root_sha256": (
            policy.python_runtime_source_root_sha256
        ),
        "python_runtime_source_file_count": (
            policy.python_runtime_source_file_count
        ),
    }
    if not _strict_json_equal(result_policy, current_result_policy):
        raise QualificationInventoryError(
            f"{owner}: ComparisonResult repository/parser provenance is no "
            "longer the current authenticated policy"
        )

    try:
        expected_qualification = (
            comparator._expected_qualification_source_sha256(
                result.evidence_a.stage,
                policy.source_snapshot,
            )
        )
        expected_executable = hashlib.sha256(
            comparator._expected_qualification_source_bytes(
                result.evidence_a.stage,
                policy.source_snapshot,
            )
        ).hexdigest()
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"{owner}: cannot authenticate the current qualification "
            f"source/executable binding: {exc}"
        ) from exc
    for name, evidence in (
        ("dataset_a", result.evidence_a),
        ("dataset_b", result.evidence_b),
    ):
        bindings = {
            "campaign_matlab_release": policy.campaign_matlab_release,
            "campaign_matlab_environment_descriptor": (
                policy.campaign_matlab_environment_descriptor
            ),
            "campaign_matlab_environment_sha256": (
                policy.campaign_matlab_environment_sha256
            ),
            "gen_schema": policy.gen_schema,
            "generation_behavior_version": (
                policy.generation_behavior_version
            ),
            "generator_source_root_sha256": (
                policy.generator_source_root_sha256
            ),
            "generator_source_file_count": policy.generator_source_file_count,
            "max_parfor_workers": policy.max_parfor_workers,
            "qualification_source_sha256": expected_qualification,
            "qualification_executed_file_sha256": expected_executable,
        }
        changed = sorted(
            field
            for field, expected in bindings.items()
            if not _strict_json_equal(getattr(evidence, field), expected)
        )
        if changed:
            raise QualificationInventoryError(
                f"{owner}.{name}: recomputed evidence is stale against the "
                f"current policy at field(s) {changed}"
            )


def revalidate_endpoint(
    path_text: Any,
    retained_dataset_block: Any,
    *,
    policy: comparator.CurrentPolicy,
    owner: str,
) -> Path:
    """Reopen one retained endpoint directory and re-authenticate it fully.

    Recomputes the dataset's complete ``DatasetEvidence`` from disk (re-hashing
    every digested file, re-verifying the completion marker, host receipt, and
    every state payload's provenance) and requires field-for-field equality
    with the retained dataset block.  There is no argument that can skip this.
    """
    if not isinstance(retained_dataset_block, dict):
        raise QualificationInventoryError(
            f"{owner}: retained dataset evidence must be one JSON object"
        )
    _exact_fields(retained_dataset_block, _DATASET_FIELDS, owner)
    canonical = _canonical_existing_directory(path_text, owner)
    if retained_dataset_block["path"] != str(canonical):
        raise QualificationInventoryError(
            f"{owner}: retained dataset block path is not canonically bound "
            f"to the revalidated directory: "
            f"{retained_dataset_block['path']!r} != {str(canonical)!r}"
        )
    try:
        evidence, _manifest, _labels, _states = comparator._validated_payload(
            canonical, policy
        )
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"{owner}: retained dataset directory failed full comparator "
            f"revalidation from disk: {exc}"
        ) from exc
    recomputed = _retained_json_view(asdict(evidence))
    _require_identical_fields(
        recomputed,
        retained_dataset_block,
        _DATASET_FIELDS,
        owner=owner,
    )
    # Re-verify the endpoint after the field-for-field comparison, so a
    # directory mutated during its own revalidation cannot be reported as
    # matching the retained block (the per-member binding inside
    # _validated_payload proves each parse; this proves the whole endpoint was
    # stable for the duration).
    try:
        comparator._reassert_endpoint_snapshot(canonical, evidence, policy)
        comparator._assert_policy_sources_unchanged(
            policy, f"{owner} endpoint revalidation"
        )
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"{owner}: retained dataset directory changed while it was being "
            f"revalidated: {exc}"
        ) from exc
    if current_comparator_policy() != policy:
        raise QualificationInventoryError(
            f"{owner}: comparator policy changed during endpoint revalidation"
        )
    return canonical


def revalidate_edge_comparison(
    retained_receipt_payload: Any,
    *,
    owner: str,
) -> None:
    """Rerun the complete pairwise comparison behind one retained receipt.

    Reopens both retained endpoint directories, reruns
    ``compare_directories`` at the retained tolerances, and requires the
    recomputed verdict, both dataset evidence blocks, every comparison
    statistic, and the raw-byte-identical state count to equal the retained
    receipt exactly.  There is no argument that can skip this.
    """
    if not isinstance(retained_receipt_payload, dict):
        raise QualificationInventoryError(
            f"{owner}: retained pair receipt must be one JSON object"
        )
    required = {
        "verdict",
        "environment_lock_sha256",
        "parser_environment",
        "python_runtime_source_root_sha256",
        "python_runtime_source_file_count",
        "stage",
        "generator_source_root_sha256",
        "qualification_source_sha256",
        "tolerances",
        "dataset_a",
        "dataset_b",
        "comparison",
        "raw_byte_identical_states",
    }
    missing = sorted(required - set(retained_receipt_payload))
    if missing:
        raise QualificationInventoryError(
            f"{owner}: retained pair receipt lacks comparison-revalidation "
            f"fields: {missing}"
        )
    policy_before = current_comparator_policy()
    tolerances = retained_receipt_payload["tolerances"]
    if not isinstance(tolerances, dict) or not {
        "rtol",
        "atol",
        "tolerant_fields",
    } <= set(tolerances):
        raise QualificationInventoryError(
            f"{owner}: retained tolerances block is malformed"
        )
    rtol = _finite_nonnegative_number(
        tolerances["rtol"], f"{owner}.tolerances.rtol"
    )
    atol = _finite_nonnegative_number(
        tolerances["atol"], f"{owner}.tolerances.atol"
    )
    if tolerances["tolerant_fields"] != sorted(
        comparator._TOLERANT_FLOAT_FIELDS
    ):
        raise QualificationInventoryError(
            f"{owner}: retained tolerant-field policy differs from the "
            "current comparator policy"
        )
    for name in ("dataset_a", "dataset_b"):
        block = retained_receipt_payload[name]
        if not isinstance(block, dict):
            raise QualificationInventoryError(
                f"{owner}.{name}: retained dataset evidence must be one "
                "JSON object"
            )
        _exact_fields(block, _DATASET_FIELDS, f"{owner}.{name}")
    canonical_a = _canonical_existing_directory(
        retained_receipt_payload["dataset_a"]["path"], f"{owner}.dataset_a"
    )
    canonical_b = _canonical_existing_directory(
        retained_receipt_payload["dataset_b"]["path"], f"{owner}.dataset_b"
    )
    try:
        result = comparator.compare_directories(
            canonical_a,
            canonical_b,
            rtol=rtol,
            atol=atol,
        )
    except (comparator.QualificationInputError, OSError) as exc:
        raise QualificationInventoryError(
            f"{owner}: retained pair failed full pairwise comparator "
            f"revalidation from disk: {exc}"
        ) from exc
    _require_result_policy_bindings(
        result,
        retained_receipt_payload,
        policy_before,
        owner=owner,
    )
    if result.verdict != retained_receipt_payload["verdict"]:
        raise QualificationInventoryError(
            f"{owner}: recomputed verdict {result.verdict!r} differs from "
            f"retained verdict {retained_receipt_payload['verdict']!r}"
        )
    for name, evidence in (
        ("dataset_a", result.evidence_a),
        ("dataset_b", result.evidence_b),
    ):
        _require_identical_fields(
            _retained_json_view(asdict(evidence)),
            retained_receipt_payload[name],
            _DATASET_FIELDS,
            owner=f"{owner}.{name}",
        )
    retained_stats = retained_receipt_payload["comparison"]
    if not isinstance(retained_stats, dict):
        raise QualificationInventoryError(
            f"{owner}.comparison: retained statistics must be one JSON object"
        )
    _exact_fields(retained_stats, _COMPARISON_FIELDS, f"{owner}.comparison")
    _require_identical_fields(
        _retained_json_view(asdict(result.stats)),
        retained_stats,
        _COMPARISON_FIELDS,
        owner=f"{owner}.comparison",
    )
    if not _strict_json_equal(
        _retained_json_view(result.raw_byte_identical_states),
        retained_receipt_payload["raw_byte_identical_states"],
    ):
        raise QualificationInventoryError(
            f"{owner}: recomputed raw-byte-identical state count "
            f"{result.raw_byte_identical_states} differs from retained "
            f"{retained_receipt_payload['raw_byte_identical_states']!r}"
        )
    try:
        comparator._reassert_endpoint_snapshot(
            Path(result.evidence_a.path), result.evidence_a, policy_before
        )
        comparator._reassert_endpoint_snapshot(
            Path(result.evidence_b.path), result.evidence_b, policy_before
        )
        comparator._assert_policy_sources_unchanged(
            policy_before, f"{owner} edge revalidation"
        )
    except (comparator.QualificationInputError, OSError, RuntimeError) as exc:
        raise QualificationInventoryError(
            f"{owner}: endpoint changed after result/receipt comparison: {exc}"
        ) from exc
    policy_after = current_comparator_policy()
    if policy_after != policy_before:
        raise QualificationInventoryError(
            f"{owner}: current comparator policy changed during edge "
            "revalidation"
        )
