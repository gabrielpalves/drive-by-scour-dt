"""Immutable gate/dataset snapshots and host/path bindings."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

from contact_gate_path_safety import (
    GateError,
    canonical_existing_directory,
    canonical_existing_file,
)
from contact_gate_core import (
    DatasetDescriptor,
    EXPECTED_TOTAL_CASES,
    EXPECTED_STATES,
    PLAN_SCHEMA,
    ROOT,
    _exact_keys,
    _sha256_bytes,
    _sha256_file,
    _strict_integer,
    _strict_json_file,
)

def _require_clean_status(status: str) -> None:
    if status:
        raise GateError(
            "working tree/index has tracked or untracked content: "
            f"{status.splitlines()[:5]!r}"
        )


def _git_clean_head(source_commit: str) -> None:
    env = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != source_commit:
        raise GateError(f"clean HEAD {head} != requested source {source_commit}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require_clean_status(status)


def _validate_gate_inventory(gate_dir: Path) -> None:
    expected_files = {
        "closure_policy.mat", "closure_policy.txt",
        "selection_manifest.mat", "selection_manifest.tsv",
        "gate_summary.mat", "gate_summary.json", "GATE_STATUS.txt",
    }
    optional_files = {"PLAN_ONLY_NONQUALIFYING.json"}
    expected_dirs = {"cases"}
    entries = {item.name: item for item in gate_dir.iterdir()}
    allowed = expected_files | optional_files | expected_dirs
    if set(entries) - allowed or expected_files - set(entries) \
            or expected_dirs - set(entries):
        raise GateError(
            "gate top-level inventory differs "
            f"(missing={sorted((expected_files | expected_dirs) - set(entries))}, "
            f"extra={sorted(set(entries) - allowed)})"
        )
    for name in expected_files | (optional_files & set(entries)):
        path = entries[name]
        canonical_existing_file(path, f"gate artifact {name}")
    cases = entries["cases"]
    canonical_existing_directory(cases, "gate cases")
    expected_case_files = {
        f"{ordinal:04d}_case.{extension}"
        for ordinal in range(1, EXPECTED_TOTAL_CASES + 1)
        for extension in ("mat", "json")
    }
    case_entries = {item.name: item for item in cases.iterdir()}
    if set(case_entries) != expected_case_files:
        raise GateError(
            f"cases inventory is not exactly {2 * EXPECTED_TOTAL_CASES} artifacts "
            f"(missing={len(expected_case_files - set(case_entries))}, "
            f"extra={sorted(set(case_entries) - expected_case_files)[:5]})"
        )
    for name, path in case_entries.items():
        canonical_existing_file(path, f"case artifact {name}")


def _gate_digest_snapshot(gate_dir: Path) -> dict[str, str]:
    _validate_gate_inventory(gate_dir)
    paths = [
        item for item in gate_dir.iterdir()
        if item.is_file() and not item.is_symlink()
    ]
    paths.extend(
        item for item in (gate_dir / "cases").iterdir()
        if item.is_file() and not item.is_symlink()
    )
    return {
        path.relative_to(gate_dir).as_posix(): _sha256_file(path)
        for path in sorted(
            paths,
            key=lambda item: item.relative_to(gate_dir).as_posix(),
        )
    }


def _freeze_gate_snapshot(source: Path, destination: Path) -> dict[str, str]:
    """Copy one hash-checked gate snapshot for all semantic parsing.

    Parsing live pathnames after hashing admits an ABA substitution: a file
    can be swapped to valid bytes only while a validator has it open and then
    restored to the forged bytes whose digest enters the receipt.  Every
    validator instead reads this private copy, and the live tree must still
    match its initial digest map after all checks finish.
    """
    initial = _gate_digest_snapshot(source)
    cases = destination / "cases"
    cases.mkdir()
    for relative, expected_sha in initial.items():
        source_path = source / Path(relative)
        payload = source_path.read_bytes()
        if _sha256_bytes(payload) != expected_sha:
            raise GateError(
                f"gate artifact changed while freezing snapshot: {relative}")
        destination_path = destination / Path(relative)
        destination_path.write_bytes(payload)
    frozen = _gate_digest_snapshot(destination)
    if frozen != initial or _gate_digest_snapshot(source) != initial:
        raise GateError("gate artifact tree changed while freezing snapshot")
    return initial


def _dataset_digest_snapshot(
    dataset_dir: Path, stage: str,
) -> dict[str, str]:
    if stage not in EXPECTED_STATES:
        raise GateError(f"cannot snapshot unknown qualification stage {stage}")
    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        raise GateError(
            f"qualification dataset is missing/not regular: {dataset_dir}")
    names = {
        *(f"{index:04d}.mat"
          for index in range(1, EXPECTED_STATES[stage] + 1)),
        "case_info.mat",
        "damage_states.mat",
        "file_digests.mat",
        "_GENERATION_COMPLETE",
        "qualification_host_receipt.json",
        "qualification_executed.m",
    }
    snapshot: dict[str, str] = {}
    for name in sorted(names):
        path = dataset_dir / name
        if not path.is_file() or path.is_symlink():
            raise GateError(
                f"{stage} decisive dataset artifact is missing/not regular: "
                f"{name}")
        snapshot[name] = _sha256_file(path)
    return snapshot


def _freeze_dataset_snapshot(
    source: Path,
    destination: Path,
    stage: str,
) -> dict[str, str]:
    """Copy all comparator-visible dataset evidence exactly once."""
    if not source.is_dir() or source.is_symlink():
        raise GateError(f"{stage} dataset source is missing/not regular")
    destination.mkdir()
    names = {
        *(f"{index:04d}.mat"
          for index in range(1, EXPECTED_STATES[stage] + 1)),
        "case_info.mat",
        "damage_states.mat",
        "file_digests.mat",
        "_GENERATION_COMPLETE",
        "qualification_host_receipt.json",
        "qualification_executed.m",
    }
    captured: dict[str, str] = {}
    for name in sorted(names):
        source_path = source / name
        if not source_path.is_file() or source_path.is_symlink():
            raise GateError(
                f"{stage} decisive dataset artifact is missing/not regular: "
                f"{name}")
        payload = source_path.read_bytes()
        captured[name] = _sha256_bytes(payload)
        (destination / name).write_bytes(payload)
    return captured


def _validate_plan_marker(
    gate_dir: Path,
    *,
    source_commit: str,
    policy_sha: str,
    selection_sha: str,
) -> str | None:
    path = gate_dir / "PLAN_ONLY_NONQUALIFYING.json"
    if not path.exists():
        return None
    value = _strict_json_file(path, "plan marker")
    expected = {
        "schema": PLAN_SCHEMA,
        "status": "PLAN_ONLY_NONQUALIFYING",
        "source_commit": source_commit,
        "policy_sha256": policy_sha,
        "selection_sha256": selection_sha,
        "expected_cases": EXPECTED_TOTAL_CASES,
    }
    if value != expected:
        raise GateError("retained plan marker differs from frozen gate")
    return _sha256_file(path)


def _validate_summary_datasets(
    value: Any,
    descriptors: list[DatasetDescriptor],
) -> list[Path]:
    if not isinstance(value, list) or len(value) != len(descriptors):
        raise GateError(
            "summary.datasets must match the exact four-stage descriptor array"
        )
    fields = {
        "stage", "dataset_dir", "dataset_dir_sha256",
        "dataset_content_root_sha256", "case_info_sha256",
        "damage_states_sha256", "file_digests_sha256",
        "completion_marker_sha256", "qualification_host_receipt_sha256",
        "gen_fingerprint", "qualification_source_sha256",
        "qualification_executed_file_sha256",
        "qualification_host_diagnostic_sha256", "qualification_hostname",
        "qualification_cpu_identifier", "qualification_logical_processors",
        "qualification_matlab_max_threads", "qualification_computer_arch",
    }
    paths: list[Path] = []
    host_tuples: list[tuple[Any, ...]] = []
    for item, frozen in zip(value, descriptors):
        data = _exact_keys(item, fields, f"summary dataset {frozen.stage}")
        expected = {
            "stage": frozen.stage,
            "dataset_dir_sha256": frozen.dataset_dir_sha256,
            "dataset_content_root_sha256": frozen.content_root,
            "case_info_sha256": frozen.case_info,
            "damage_states_sha256": frozen.damage_states,
            "file_digests_sha256": frozen.file_digests,
            "completion_marker_sha256": frozen.complete,
            "qualification_host_receipt_sha256": frozen.host_receipt,
            "gen_fingerprint": frozen.fingerprint,
            "qualification_source_sha256": frozen.qual_source,
            "qualification_executed_file_sha256": frozen.qual_executed,
            "qualification_host_diagnostic_sha256": frozen.host_diagnostic,
        }
        for key, wanted in expected.items():
            if data[key] != wanted:
                raise GateError(
                    f"summary dataset {frozen.stage} {key} differs"
                )
        supplied_dataset_path = Path(data["dataset_dir"])
        if (
            not supplied_dataset_path.is_absolute()
            or _sha256_bytes(str(supplied_dataset_path).encode("utf-8"))
            != frozen.dataset_dir_sha256
        ):
            raise GateError(
                "summary dataset path is not frozen: "
                f"{supplied_dataset_path}"
            )
        dataset_path = canonical_existing_directory(
            supplied_dataset_path,
            f"summary dataset {frozen.stage}",
        )
        host_tuple = (
            data["qualification_hostname"],
            data["qualification_cpu_identifier"],
            data["qualification_logical_processors"],
            data["qualification_matlab_max_threads"],
            data["qualification_computer_arch"],
        )
        if (
            not all(isinstance(part, str) and part for part in host_tuple[:2])
            or type(host_tuple[2]) is not int
            or type(host_tuple[3]) is not int
            or host_tuple[2] < 1
            or host_tuple[3] < 1
            or not isinstance(host_tuple[4], str)
            or not host_tuple[4]
        ):
            raise GateError(f"summary dataset {frozen.stage} host tuple malformed")
        host_tuples.append(host_tuple)
        paths.append(dataset_path)
    if len(set(host_tuples)) != 1:
        raise GateError(
            "the four Paper-1 blocks do not carry one identical set of self-attested host "
            "diagnostics"
        )
    if len({item.fingerprint for item in descriptors}) != len(descriptors) \
            or len({item.qual_source for item in descriptors}) != len(descriptors):
        raise GateError("stage fingerprints/qualification sources are not distinct")
    return paths


def _validate_closure_host_attestation(
    value: Any,
    *,
    declared_host_id: str,
    environment_sha: str,
    summary_datasets: list[dict[str, Any]],
) -> None:
    fields = {
        "schema", "declared_host_id", "hostname", "cpu_identifier",
        "logical_processors", "matlab_max_threads", "computer_arch",
        "matlab_environment_sha256", "canonical_descriptor", "sha256",
    }
    data = _exact_keys(value, fields, "closure host attestation")
    text_fields = (
        "schema", "declared_host_id", "hostname", "cpu_identifier",
        "computer_arch", "matlab_environment_sha256",
        "canonical_descriptor", "sha256",
    )
    if any(
        not isinstance(data[key], str) or not data[key]
        for key in text_fields
    ):
        raise GateError("closure host attestation text field malformed")
    logical_processors = _strict_integer(
        data["logical_processors"], "closure logical_processors")
    matlab_threads = _strict_integer(
        data["matlab_max_threads"], "closure matlab_max_threads")
    if logical_processors < 1 or matlab_threads < 1:
        raise GateError("closure host processor/thread counts must be positive")
    if (
        data["schema"] != "contact-closure-host-attestation-v1"
        or data["declared_host_id"] != declared_host_id
        or data["matlab_environment_sha256"] != environment_sha
    ):
        raise GateError("closure host attestation identity differs")
    descriptor = "\n".join((
        f"schema={data['schema']}",
        f"declared_host_id={data['declared_host_id']}",
        f"hostname={data['hostname']}",
        f"cpu_identifier={data['cpu_identifier']}",
        f"logical_processors={logical_processors}",
        f"matlab_max_threads={matlab_threads}",
        f"computer_arch={data['computer_arch']}",
        f"matlab_environment_sha256={data['matlab_environment_sha256']}",
    ))
    if (
        data["canonical_descriptor"] != descriptor
        or data["sha256"] != _sha256_bytes(descriptor.encode("utf-8"))
    ):
        raise GateError("closure host descriptor/hash does not recompute")
    expected_tuple = (
        data["hostname"],
        data["cpu_identifier"],
        logical_processors,
        matlab_threads,
        data["computer_arch"],
    )
    for item in summary_datasets:
        observed = (
            item["qualification_hostname"],
            item["qualification_cpu_identifier"],
            item["qualification_logical_processors"],
            item["qualification_matlab_max_threads"],
            item["qualification_computer_arch"],
        )
        if observed != expected_tuple:
            raise GateError(
                f"closure host differs from {item['stage']} qualification host")
