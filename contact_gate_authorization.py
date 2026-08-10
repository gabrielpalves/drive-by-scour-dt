"""Snapshot orchestration and create-once authorization receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from contact_gate_path_safety import (
    GateError,
    canonical_existing_directory,
    canonical_existing_file,
    canonical_receipt_path,
)
from contact_gate_source_contract import (
    GATE_SOURCE_FILES,
    SOLVER_SOURCE_FILES,
    STUDY_HARNESS_FILES,
    _contact_source_set,
    _validate_matlab_source_contract,
)
from contact_gate_verifier_identity import verifier_source_root
from contact_gate_core import (
    AUTHORIZATION_RECEIPT_SCHEMA,
    CHANNEL_SCHEMA_ID,
    COMMIT_RE,
    ENVIRONMENT_LOCK,
    EXPECTED_MATLAB_RELEASE,
    EXPECTED_TOTAL_CASES,
    ROOT,
    SUMMARY_SCHEMA,
    STAGES,
    _canonical_lf_file,
    _exact_keys,
    _locked_matlab_environment,
    _normalised_lf,
    _sha256_bytes,
    _sha256_file,
    _strict_json_file,
    _validate_utc_pair,
)
from contact_gate_policy import _parse_policy, _parse_selection
from contact_gate_case import (
    _gate_execution_root,
    _generator_source_identity,
    _recompute_case,
    _solver_execution_identity,
)
from contact_gate_artifacts import (
    _dataset_digest_snapshot,
    _freeze_dataset_snapshot,
    _freeze_gate_snapshot,
    _gate_digest_snapshot,
    _git_clean_head,
    _validate_closure_host_attestation,
    _validate_gate_inventory,
    _validate_plan_marker,
    _validate_summary_datasets,
)
from contact_gate_dataset import (
    _validate_datasets_with_comparator,
    _validate_mat_sources,
)

def _verify_gate_snapshot(
    gate_dir: Path,
    live_gate_dir: Path,
    source_commit: str,
    *,
    frozen_dataset_paths: list[Path],
    live_dataset_snapshots: list[tuple[Path, str, dict[str, str]]],
    git_check: Any = _git_clean_head,
    dataset_validator: Any = _validate_datasets_with_comparator,
    mat_validator: Any = _validate_mat_sources,
) -> dict[str, Any]:
    if not gate_dir.is_dir() or gate_dir.is_symlink():
        raise GateError(f"gate directory is missing/not regular: {gate_dir}")
    if not COMMIT_RE.fullmatch(source_commit):
        raise GateError("source commit must be 40 lowercase hex")
    _validate_gate_inventory(gate_dir)
    gate_snapshot = _gate_digest_snapshot(gate_dir)
    checker_sha = verifier_source_root()
    environment_lock_sha = _sha256_file(ENVIRONMENT_LOCK)
    git_check(source_commit)
    _, locked_descriptor, locked_environment_sha = (
        _locked_matlab_environment()
    )
    source_identity = _generator_source_identity()
    source_root = source_identity.sha256
    _validate_matlab_source_contract(
        _contact_source_set(STUDY_HARNESS_FILES),
        _contact_source_set(GATE_SOURCE_FILES),
        solver_sources=_contact_source_set(SOLVER_SOURCE_FILES),
    )
    solver_root, harness_sha, b66_sha = _solver_execution_identity()

    _, policy_sha, policy_descriptor = _parse_policy(
        gate_dir / "closure_policy.txt", source_commit)
    rows, selection_sha, descriptors = _parse_selection(
        gate_dir / "selection_manifest.tsv")
    selection_descriptor = _normalised_lf(
        gate_dir / "selection_manifest.tsv")
    summary_path = gate_dir / "gate_summary.json"
    summary = _strict_json_file(summary_path, "gate summary")
    summary_fields = {
        "schema", "status", "source_commit", "policy_sha256",
        "selection_sha256", "expected_cases", "completed_cases",
        "pass_cases", "fail_cases", "error_cases",
        "case_artifact_root_sha256", "declared_host_id",
        "closure_host_attestation",
        "generator_source_root_sha256", "generator_source_digest_lines",
        "generator_source_file_count", "matlab_environment_sha256",
        "matlab_environment_descriptor", "matlab_release",
        "gate_execution_root_sha256", "datasets",
        "started_utc", "completed_utc",
    }
    _exact_keys(summary, summary_fields, "gate summary")
    for field in (
        "expected_cases", "completed_cases", "pass_cases",
        "fail_cases", "error_cases", "generator_source_file_count",
    ):
        if type(summary[field]) is not int:
            raise GateError(f"summary {field} is not an exact JSON integer")
    expected_summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "policy_sha256": policy_sha,
        "selection_sha256": selection_sha,
        "expected_cases": EXPECTED_TOTAL_CASES,
        "completed_cases": EXPECTED_TOTAL_CASES,
        "pass_cases": EXPECTED_TOTAL_CASES,
        "fail_cases": 0,
        "error_cases": 0,
        "generator_source_root_sha256": source_root,
        "generator_source_digest_lines": source_identity.digest_lines,
        "generator_source_file_count": source_identity.file_count,
        "matlab_environment_sha256": locked_environment_sha,
        "matlab_environment_descriptor": locked_descriptor,
        "matlab_release": EXPECTED_MATLAB_RELEASE,
        "gate_execution_root_sha256": _gate_execution_root(),
    }
    for key, expected in expected_summary.items():
        if summary[key] != expected:
            raise GateError(
                f"summary {key}={summary[key]!r}, expected {expected!r}"
            )
    environment_sha = summary["matlab_environment_sha256"]
    host_id = summary["declared_host_id"]
    if not isinstance(host_id, str) \
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", host_id):
        raise GateError("summary host ID malformed")
    _validate_utc_pair(
        summary["started_utc"], summary["completed_utc"], "summary")

    dataset_paths = _validate_summary_datasets(
        summary["datasets"], descriptors)
    if len(frozen_dataset_paths) != len(dataset_paths) \
            or len(live_dataset_snapshots) != len(dataset_paths):
        raise GateError("frozen qualification dataset inventory differs")
    _validate_closure_host_attestation(
        summary["closure_host_attestation"],
        declared_host_id=host_id,
        environment_sha=environment_sha,
        summary_datasets=summary["datasets"],
    )
    expected_descriptors = dataset_validator(
        frozen_dataset_paths,
        descriptors,
        rows,
        source_root=source_root,
        environment_sha=environment_sha,
        host_id=host_id,
    )
    if expected_descriptors is not None and not isinstance(
        expected_descriptors, dict
    ):
        raise GateError(
            "dataset validator returned malformed descriptor evidence")
    plan_sha = _validate_plan_marker(
        gate_dir,
        source_commit=source_commit,
        policy_sha=policy_sha,
        selection_sha=selection_sha,
    )
    mat_validator(
        gate_dir,
        policy_descriptor=policy_descriptor,
        policy_sha=policy_sha,
        selection_descriptor=selection_descriptor,
        selection_sha=selection_sha,
        summary=summary,
        rows=rows,
    )

    cases_dir = gate_dir / "cases"
    artifact_lines: list[str] = []
    dataset_by_stage = {
        descriptor.stage: descriptor for descriptor in descriptors
    }
    for row in rows:
        stem = f"{row.ordinal:04d}_case"
        mat_path = cases_dir / f"{stem}.mat"
        json_path = cases_dir / f"{stem}.json"
        artifact_lines.extend((
            f"{mat_path.name}:{gate_snapshot[f'cases/{mat_path.name}']}",
            f"{json_path.name}:{gate_snapshot[f'cases/{json_path.name}']}",
        ))
        case = _strict_json_file(json_path, f"case {row.ordinal} JSON")
        _recompute_case(
            case,
            row,
            dataset=dataset_by_stage[row.stage],
            policy_sha=policy_sha,
            selection_sha=selection_sha,
            source_root=source_root,
            environment_sha=environment_sha,
            solver_root=solver_root,
            harness_sha=harness_sha,
            b66_sha=b66_sha,
            expected_descriptor=(
                expected_descriptors.get(
                    (row.stage, row.state_index, row.passage_index)
                )
                if expected_descriptors is not None
                else None
            ),
        )
        if expected_descriptors is not None and (
            row.stage, row.state_index, row.passage_index
        ) not in expected_descriptors:
            raise GateError(
                f"dataset validator omitted descriptor for case {row.ordinal}")
    observed_case_root = _sha256_bytes("\n".join(artifact_lines).encode("utf-8"))
    if summary.get("case_artifact_root_sha256") != observed_case_root:
        raise GateError("case artifact root differs from summary")
    status_text = _canonical_lf_file(
        gate_dir / "GATE_STATUS.txt", "gate status")
    if status_text != (
        f"PASS\nexpected={EXPECTED_TOTAL_CASES}\n"
        f"completed={EXPECTED_TOTAL_CASES}\n"
        f"pass={EXPECTED_TOTAL_CASES}\nfail=0\nerror=0\n"
    ):
        raise GateError("GATE_STATUS.txt differs")
    gate_artifacts = {
        name: gate_snapshot[name]
        for name in (
            "closure_policy.mat", "closure_policy.txt",
            "selection_manifest.mat", "selection_manifest.tsv",
            "gate_summary.mat", "gate_summary.json", "GATE_STATUS.txt",
        )
    }
    if plan_sha is not None:
        if gate_snapshot["PLAN_ONLY_NONQUALIFYING.json"] != plan_sha:
            raise GateError("plan marker changed during validation")
        gate_artifacts["PLAN_ONLY_NONQUALIFYING.json"] = plan_sha
    if (
        _gate_digest_snapshot(gate_dir) != gate_snapshot
        or _gate_digest_snapshot(live_gate_dir) != gate_snapshot
    ):
        raise GateError("gate artifact snapshot changed during validation")
    for (
        live_dataset_path,
        stage,
        frozen_digest,
    ), frozen_dataset_path, expected_live_path in zip(
        live_dataset_snapshots, frozen_dataset_paths, dataset_paths
    ):
        if live_dataset_path.resolve() != expected_live_path.resolve():
            raise GateError(f"{stage} frozen/live dataset path binding differs")
        if (
            _dataset_digest_snapshot(
                frozen_dataset_path, stage) != frozen_digest
            or _dataset_digest_snapshot(
                live_dataset_path, stage) != frozen_digest
        ):
            raise GateError(
                f"{stage} qualification dataset changed during validation")
    source_identity_end = _generator_source_identity()
    solver_identity_end = _solver_execution_identity()
    if (
        source_identity_end != source_identity
        or solver_identity_end != (solver_root, harness_sha, b66_sha)
        or verifier_source_root() != checker_sha
        or _sha256_file(ENVIRONMENT_LOCK) != environment_lock_sha
    ):
        raise GateError(
            "checker/source/environment snapshot changed during validation")
    return {
        "schema": AUTHORIZATION_RECEIPT_SCHEMA,
        "status": "ACCEPTED",
        "source_commit": source_commit,
        "declared_host_id": host_id,
        "matlab_environment_sha256": environment_sha,
        "generator_source_root_sha256": source_root,
        "policy_sha256": policy_sha,
        "selection_sha256": selection_sha,
        "case_artifact_root_sha256": observed_case_root,
        "gate_summary_sha256": gate_snapshot["gate_summary.json"],
        "gate_artifact_sha256": gate_artifacts,
        "checker_sha256": checker_sha,
        "environment_lock_sha256": environment_lock_sha,
        "dataset_descriptors": [
            descriptor.raw_line for descriptor in descriptors
        ],
        "expected_cases": EXPECTED_TOTAL_CASES,
        "accepted_cases": EXPECTED_TOTAL_CASES,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        # This field is deliberately
        # deterministic: the gate's authenticated completion instant, not
        # the wall-clock time at which this Python process wrote the receipt.
        "validated_utc": summary["completed_utc"],
    }


def verify_gate(
    gate_dir: Path,
    source_commit: str,
    *,
    git_check: Any = _git_clean_head,
    dataset_validator: Any = _validate_datasets_with_comparator,
    mat_validator: Any = _validate_mat_sources,
) -> dict[str, Any]:
    gate_dir = canonical_existing_directory(gate_dir, "gate directory")
    if not COMMIT_RE.fullmatch(source_commit):
        raise GateError("source commit must be 40 lowercase hex")
    with tempfile.TemporaryDirectory(
        prefix="contact_gate_frozen_snapshot_"
    ) as raw_snapshot:
        snapshot_root = Path(raw_snapshot)
        frozen_gate_dir = snapshot_root / "gate"
        frozen_gate_dir.mkdir()
        _freeze_gate_snapshot(gate_dir, frozen_gate_dir)
        frozen_summary = _strict_json_file(
            frozen_gate_dir / "gate_summary.json",
            "frozen gate summary",
        )
        dataset_items = frozen_summary.get("datasets")
        if not isinstance(dataset_items, list) \
                or len(dataset_items) != len(STAGES):
            raise GateError(
                "frozen gate summary lacks exactly four datasets")
        datasets_root = snapshot_root / "datasets"
        datasets_root.mkdir()
        frozen_dataset_paths: list[Path] = []
        live_dataset_snapshots: list[
            tuple[Path, str, dict[str, str]]
        ] = []
        for item, stage in zip(dataset_items, STAGES):
            if (
                not isinstance(item, dict)
                or item.get("stage") != stage
                or not isinstance(item.get("dataset_dir"), str)
            ):
                raise GateError(
                    f"frozen gate summary dataset order differs at {stage}")
            live_dataset_path = canonical_existing_directory(
                Path(item["dataset_dir"]),
                f"{stage} qualification dataset",
            )
            frozen_dataset_path = datasets_root / stage
            captured = _freeze_dataset_snapshot(
                live_dataset_path, frozen_dataset_path, stage)
            frozen_dataset_paths.append(frozen_dataset_path)
            live_dataset_snapshots.append(
                (live_dataset_path, stage, captured))
        return _verify_gate_snapshot(
            frozen_gate_dir,
            gate_dir,
            source_commit,
            frozen_dataset_paths=frozen_dataset_paths,
            live_dataset_snapshots=live_dataset_snapshots,
            git_check=git_check,
            dataset_validator=dataset_validator,
            mat_validator=mat_validator,
        )


def _publish_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path = canonical_receipt_path(path, "authorization receipt")
    if path.exists() or path.is_symlink():
        raise GateError(f"receipt is create-once and already exists: {path}")
    payload = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise GateError(f"temporary receipt path already exists: {tmp}")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        tmp.unlink(missing_ok=True)
        raise GateError(
            f"receipt create-once collision; final was not overwritten: {path}"
        ) from exc
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.unlink()
    try:
        installed = path.read_bytes()
    except OSError as exc:
        raise GateError(
            f"cannot authenticate installed receipt bytes: {path}") from exc
    if installed != payload:
        raise GateError(
            "installed receipt bytes changed during create-once publication")


def _revalidate_existing_receipt(
    path: Path,
    recomputed: dict[str, Any],
) -> None:
    path = canonical_existing_file(path, "authorization receipt")
    raw = path.read_bytes()
    existing = _strict_json_file(path, "authorization receipt")
    canonical = (
        json.dumps(existing, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise GateError("retained receipt bytes are not canonical")
    expected_bytes = (
        json.dumps(
            recomputed, sort_keys=True, indent=2, ensure_ascii=True
        ) + "\n"
    ).encode("utf-8")
    if raw != expected_bytes:
        raise GateError("retained receipt bytes differ from recomputed evidence")


def _is_within(path: Path, root: Path) -> bool:
    path_text = os.path.normcase(str(path))
    root_text = os.path.normcase(str(root))
    try:
        return os.path.commonpath((path_text, root_text)) == root_text
    except ValueError:
        return False


def _validate_receipt_location(path: Path, gate_dir: Path) -> None:
    gate_dir = canonical_existing_directory(gate_dir, "gate directory")
    resolved = canonical_receipt_path(path, "authorization receipt")
    summary = _strict_json_file(
        gate_dir / "gate_summary.json", "gate summary for receipt location")
    datasets = summary.get("datasets")
    if not isinstance(datasets, list):
        raise GateError("cannot resolve protected dataset roots")
    protected = [
        canonical_existing_directory(ROOT, "reviewed repository"),
        gate_dir,
    ]
    for item in datasets:
        if not isinstance(item, dict) or not isinstance(
            item.get("dataset_dir"), str
        ):
            raise GateError("cannot resolve protected dataset root")
        protected.append(
            canonical_existing_directory(
                Path(item["dataset_dir"]),
                "protected qualification dataset",
            )
        )
    for root in protected:
        if _is_within(resolved, root):
            raise GateError(
                f"receipt path must be outside repo/gate/datasets: {root}")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        raise GateError(f"temporary receipt path already exists: {tmp}")


def _authorize_gate(
    gate_dir: Path,
    source_commit: str,
    receipt_path: Path,
    *,
    revalidate_receipt: bool,
    git_check: Any = _git_clean_head,
    dataset_validator: Any = _validate_datasets_with_comparator,
    mat_validator: Any = _validate_mat_sources,
) -> dict[str, Any]:
    _validate_receipt_location(receipt_path, gate_dir)
    recomputed = verify_gate(
        gate_dir,
        source_commit,
        git_check=git_check,
        dataset_validator=dataset_validator,
        mat_validator=mat_validator,
    )
    _validate_receipt_location(receipt_path, gate_dir)
    if revalidate_receipt:
        if not receipt_path.is_absolute():
            raise GateError("receipt path must be absolute")
        _revalidate_existing_receipt(receipt_path, recomputed)
    else:
        _publish_receipt(receipt_path, recomputed)
    return recomputed


def verify_existing_authorization_receipt(
    gate_dir: Path,
    source_commit: str,
    receipt_path: Path,
    *,
    git_check: Any = _git_clean_head,
    dataset_validator: Any = _validate_datasets_with_comparator,
    mat_validator: Any = _validate_mat_sources,
) -> dict[str, Any]:
    """Recompute all evidence and authenticate, without replacing, a receipt."""
    return _authorize_gate(
        Path(gate_dir),
        source_commit,
        Path(receipt_path),
        revalidate_receipt=True,
        git_check=git_check,
        dataset_validator=dataset_validator,
        mat_validator=mat_validator,
    )
