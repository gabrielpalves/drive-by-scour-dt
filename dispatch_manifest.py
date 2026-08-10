"""Manifest grammar, snapshot, and publication layer for dispatch authorization.

Contract: this module owns the mechanical, evidence-agnostic building blocks
of the Paper-1 dispatch-authorization gate:

* the exact ``ttbi-paper1-dispatch-authorization-manifest-v2`` field sets and
  their structural/cross-binding validation (``_validate_manifest_shape``);
* canonical JSON encode/parse (``_canonical_json_bytes``,
  ``_strict_json_bytes``) and digest helpers;
* TOCTOU-hardened file snapshots and canonical evidence-path validation
  (``_snapshot_regular``, ``_assert_snapshot_unchanged``,
  ``_canonical_existing_path``, junction/symlink/hard-link guards);
* the create-once external manifest publication (``_publish_create_once``).

Rationale: splitting these out of ``dispatch_authorization`` keeps that file
a short, readable orchestrator of the git boundary and the three authoritative
evidence revalidators, while this layer stays free of any evidence semantics.
Nothing here reads benchmark, qualification, or contact evidence; the shape
validator never touches disk.  Every deviation fails closed through
``DispatchAuthorizationError``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any

from contact_gate_verifier_identity import (
    VERIFIER_SOURCE_FILES as CONTACT_GATE_VERIFIER_SOURCE_FILES,
)


MANIFEST_SCHEMA = "ttbi-paper1-dispatch-authorization-manifest-v2"
POLICY_SCHEMA = "ttbi-paper1-dispatch-authorization-policy-v2"
REQUIRED_BENCHMARK_SCHEMA = (
    "ttbi-paper1-benchmark-authorization-evidence-v2"
)
REQUIRED_QUALIFICATION_STAGES = ("F40-S", "F40-M", "L99-S", "L99-M")
EXPECTED_CONTACT_CASES = 420
REQUIRED_CHANNEL_SCHEMA_ID = "physical8_v1"
STATUS = "AUTHORIZED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_AUTH_FILE_BYTES = 256 * 1024 * 1024

_DISPATCH_POLICY_SOURCE_FILES = {
    "benchmark_paper1_compute.py",
    "build_stage_bundles.py",
    "bundle_source_files.txt",
    "check_paper1_benchmark_contract.py",
    "check_campaign_controls.py",
    "check_dispatch_authorization.py",
    "check_generation_release_comparison.py",
    "check_import_path_guard.py",
    "check_qualification_receipt_inventory.py",
    "dispatch_authorization.py",
    "dispatch_manifest.py",
    "environment/campaign-py313-cu128.json",
    "qualification_endpoint_revalidation.py",
    "qualification_path_safety.py",
    "qualification_receipt_inventory.py",
    "qualification_receipt_schema.py",
    "training/__init__.py",
    "TTBI_2D/__init__.py",
}
# Dispatch revalidates contact evidence by executing that verifier.  Reuse its
# audited transitive inventory verbatim, so a future contact-module split
# cannot be omitted from policy A by updating one manually duplicated tuple.
POLICY_SOURCE_FILES = tuple(sorted(
    _DISPATCH_POLICY_SOURCE_FILES | set(CONTACT_GATE_VERIFIER_SOURCE_FILES)
))

TOP_FIELDS = frozenset({
    "schema",
    "status",
    "tested_source_commit",
    "policy",
    "benchmark",
    "qualification",
    "contact_closure",
})
POLICY_FIELDS = frozenset({
    "schema",
    "tested_source_tree_oid",
    "source_sha256",
    "source_root_sha256",
})
BENCHMARK_FIELDS = frozenset({
    "schema",
    "status",
    "tested_source_commit",
    "descriptor_sha256",
    "evidence_root_sha256",
    "summary_sha256",
    "run_state_sha256",
    "run_directory",
})
QUALIFICATION_FIELDS = frozenset({
    "required_stages",
    "intended_host_ids",
    "pair_receipt_paths",
    "inventory_receipt_path",
    "inventory_receipt_sha256",
    "inventory_root_sha256",
    "accepted_pairwise_receipt_count",
    "generator_source_root_sha256",
    "matlab_environment_sha256",
})
CONTACT_FIELDS = frozenset({
    "gate_directory",
    "authorization_receipt_path",
    "authorization_receipt_sha256",
    "receipt_schema",
    "status",
    "declared_host_id",
    "matlab_environment_sha256",
    "generator_source_root_sha256",
    "policy_sha256",
    "selection_sha256",
    "case_artifact_root_sha256",
    "gate_summary_sha256",
    "gate_artifact_root_sha256",
    "dataset_descriptors_root_sha256",
    "expected_cases",
    "accepted_cases",
    "channel_schema_id",
})
CONTACT_RECEIPT_FIELDS = frozenset({
    "schema",
    "status",
    "source_commit",
    "declared_host_id",
    "matlab_environment_sha256",
    "generator_source_root_sha256",
    "policy_sha256",
    "selection_sha256",
    "case_artifact_root_sha256",
    "gate_summary_sha256",
    "gate_artifact_sha256",
    "checker_sha256",
    "environment_lock_sha256",
    "dataset_descriptors",
    "expected_cases",
    "accepted_cases",
    "channel_schema_id",
    "validated_utc",
})


class DispatchAuthorizationError(RuntimeError):
    """The retained evidence does not authorize bundle dispatch."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    size: int
    mtime_ns: int
    file_id: tuple[int, int]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DispatchAuthorizationError(
            "authorization value is not strict JSON"
        ) from exc
    return (rendered + "\n").encode("utf-8")


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(
    value: Any,
    expected: frozenset[str] | set[str],
    owner: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DispatchAuthorizationError(f"{owner} must be a JSON object")
    observed = set(value)
    if observed != set(expected):
        raise DispatchAuthorizationError(
            f"{owner} fields differ; "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )
    return value


def _strict_json_bytes(raw: bytes, owner: str) -> dict[str, Any]:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise DispatchAuthorizationError(f"{owner} is unexpectedly large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DispatchAuthorizationError(f"{owner} is not strict UTF-8") from exc
    if (
        text.startswith("\ufeff")
        or "\r" in text
        or not text.endswith("\n")
        or text.endswith("\n\n")
    ):
        raise DispatchAuthorizationError(
            f"{owner} must use UTF-8, canonical LF and one final LF"
        )

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DispatchAuthorizationError(f"{owner} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DispatchAuthorizationError(f"{owner} root must be an object")
    if raw != _canonical_json_bytes(value):
        raise DispatchAuthorizationError(
            f"{owner} is not the registered canonical JSON encoding"
        )
    return value


def _is_junction(path: Path) -> bool:
    predicate = getattr(os.path, "isjunction", None)
    return bool(predicate is not None and predicate(path))


def _canonical_existing_path(
    raw: str,
    owner: str,
    *,
    kind: str,
) -> Path:
    if type(raw) is not str or not raw or "\0" in raw:
        raise DispatchAuthorizationError(
            f"{owner} must be one nonempty absolute path string"
        )
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise DispatchAuthorizationError(f"{owner} must be absolute")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise DispatchAuthorizationError(
            f"{owner} cannot be resolved: {supplied}"
        ) from exc
    if raw != str(resolved):
        raise DispatchAuthorizationError(
            f"{owner} is not the canonical resolved path: {raw!r}"
        )
    current = supplied
    while True:
        try:
            if current.is_symlink() or _is_junction(current):
                raise DispatchAuthorizationError(
                    f"{owner} traverses a symlink/junction: {current}"
                )
        except OSError as exc:
            raise DispatchAuthorizationError(
                f"{owner} path component cannot be inspected: {current}"
            ) from exc
        if current.parent == current:
            break
        current = current.parent
    try:
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except OSError as exc:
        raise DispatchAuthorizationError(
            f"{owner} cannot be inspected: {resolved}"
        ) from exc
    valid = (
        stat.S_ISREG(mode) if kind == "file"
        else stat.S_ISDIR(mode) if kind == "directory"
        else False
    )
    if not valid:
        raise DispatchAuthorizationError(
            f"{owner} must be one real {kind}: {resolved}"
        )
    if kind == "file":
        links = int(getattr(os.stat(
            resolved, follow_symlinks=False), "st_nlink", 1))
        if links != 1:
            raise DispatchAuthorizationError(
                f"{owner} must not have a hard-link alias: {resolved}"
            )
    return resolved


def _canonical_output_path(raw: str | os.PathLike[str], repo: Path) -> Path:
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise DispatchAuthorizationError(
            "dispatch authorization manifest output must be absolute"
        )
    try:
        parent = supplied.parent.resolve(strict=True)
    except OSError as exc:
        raise DispatchAuthorizationError(
            "manifest output parent must already exist"
        ) from exc
    canonical = parent / supplied.name
    if str(supplied) != str(canonical):
        raise DispatchAuthorizationError(
            "manifest output path must already be canonical"
        )
    _canonical_existing_path(str(parent), "manifest output parent",
                             kind="directory")
    try:
        canonical.relative_to(repo.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise DispatchAuthorizationError(
            "dispatch authorization manifest must be external to the repository"
        )
    return canonical


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _require_external_to_repo(path: Path, repo: Path, owner: str) -> None:
    if _is_within(path, repo):
        raise DispatchAuthorizationError(
            f"{owner} must be external to the repository: {path}")


def _snapshot_regular(path: Path, owner: str) -> FileSnapshot:
    canonical = _canonical_existing_path(str(path), owner, kind="file")
    flags = os.O_RDONLY
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise DispatchAuthorizationError(
            f"cannot securely open {owner}") from exc
    try:
        before = os.fstat(descriptor)
        if before.st_size < 0 or before.st_size > MAX_AUTH_FILE_BYTES:
            raise DispatchAuthorizationError(
                f"{owner} exceeds the authorization snapshot limit")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DispatchAuthorizationError(f"cannot snapshot {owner}") from exc
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        path_after = os.stat(canonical, follow_symlinks=False)
    except OSError as exc:
        raise DispatchAuthorizationError(
            f"{owner} path disappeared during snapshot") from exc
    identity_before = (
        int(getattr(before, "st_dev", 0)),
        int(getattr(before, "st_ino", 0)),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(getattr(before, "st_nlink", 1)),
    )
    identity_after = (
        int(getattr(after, "st_dev", 0)),
        int(getattr(after, "st_ino", 0)),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(getattr(after, "st_nlink", 1)),
    )
    path_identity_after = (
        int(getattr(path_after, "st_dev", 0)),
        int(getattr(path_after, "st_ino", 0)),
        int(path_after.st_size),
        int(path_after.st_mtime_ns),
        int(getattr(path_after, "st_nlink", 1)),
    )
    if (
        identity_before != identity_after
        or identity_after != path_identity_after
        or not stat.S_ISREG(before.st_mode)
        or identity_before[-1] != 1
        or len(raw) != before.st_size
    ):
        raise DispatchAuthorizationError(
            f"{owner} changed while it was being read"
        )
    return FileSnapshot(
        path=canonical,
        raw=raw,
        sha256=_sha256_bytes(raw),
        size=int(before.st_size),
        mtime_ns=int(before.st_mtime_ns),
        file_id=(
            int(getattr(before, "st_dev", 0)),
            int(getattr(before, "st_ino", 0)),
        ),
    )


def _assert_snapshot_unchanged(snapshot: FileSnapshot, owner: str) -> None:
    current = _snapshot_regular(snapshot.path, owner)
    if (
        current.raw != snapshot.raw
        or current.sha256 != snapshot.sha256
        or current.size != snapshot.size
        or current.mtime_ns != snapshot.mtime_ns
        or current.file_id != snapshot.file_id
    ):
        raise DispatchAuthorizationError(
            f"{owner} changed/replaced during authorization"
        )


def _validate_manifest_shape(payload: dict[str, Any]) -> None:
    _exact_keys(payload, TOP_FIELDS, "dispatch authorization manifest")
    if (
        payload["schema"] != MANIFEST_SCHEMA
        or payload["status"] != STATUS
        or type(payload["tested_source_commit"]) is not str
        or not COMMIT_RE.fullmatch(payload["tested_source_commit"])
    ):
        raise DispatchAuthorizationError(
            "dispatch authorization manifest identity/status is invalid")
    policy = _exact_keys(payload["policy"], POLICY_FIELDS,
                         "dispatch policy")
    if (
        policy["schema"] != POLICY_SCHEMA
        or type(policy["tested_source_tree_oid"]) is not str
        or not COMMIT_RE.fullmatch(policy["tested_source_tree_oid"])
        or type(policy["source_sha256"]) is not dict
        or not set(POLICY_SOURCE_FILES).issubset(
            policy["source_sha256"])
        or any(
            type(name) is not str
            or not name
            or "\\" in name
            or name.startswith("/")
            or PureWindowsPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or any(
                not part or part in {".", ".."}
                for part in name.split("/")
            )
            for name in policy["source_sha256"]
        )
        or any(
            type(value) is not str or not SHA256_RE.fullmatch(value)
            for value in policy["source_sha256"].values()
        )
        or type(policy["source_root_sha256"]) is not str
        or not SHA256_RE.fullmatch(policy["source_root_sha256"])
    ):
        raise DispatchAuthorizationError(
            "dispatch authorization policy is malformed")
    benchmark = _exact_keys(
        payload["benchmark"], BENCHMARK_FIELDS, "benchmark evidence")
    if (
        benchmark["schema"] != REQUIRED_BENCHMARK_SCHEMA
        or benchmark["status"] != "PASS"
        or benchmark["tested_source_commit"]
        != payload["tested_source_commit"]
    ):
        raise DispatchAuthorizationError(
            "benchmark evidence identity/status is invalid")
    for key in (
        "descriptor_sha256",
        "evidence_root_sha256",
        "summary_sha256",
        "run_state_sha256",
    ):
        if type(benchmark[key]) is not str or not SHA256_RE.fullmatch(
                benchmark[key]):
            raise DispatchAuthorizationError(
                f"benchmark evidence {key} is malformed")
    qualification = _exact_keys(
        payload["qualification"], QUALIFICATION_FIELDS,
        "qualification evidence",
    )
    if (
        qualification["required_stages"] != list(REQUIRED_QUALIFICATION_STAGES)
        or not isinstance(qualification["intended_host_ids"], list)
        or len(qualification["intended_host_ids"]) < 2
        or not isinstance(qualification["pair_receipt_paths"], list)
        or type(qualification["accepted_pairwise_receipt_count"]) is not int
        or len(set(qualification["intended_host_ids"]))
        != len(qualification["intended_host_ids"])
        or qualification["accepted_pairwise_receipt_count"]
        != len(REQUIRED_QUALIFICATION_STAGES)
        * (
            len(qualification["intended_host_ids"])
            * (len(qualification["intended_host_ids"]) - 1)
            // 2
        )
        or len(qualification["pair_receipt_paths"])
        != qualification["accepted_pairwise_receipt_count"]
    ):
        raise DispatchAuthorizationError(
            "qualification evidence cardinalities are malformed")
    for key in (
        "inventory_receipt_sha256",
        "inventory_root_sha256",
        "generator_source_root_sha256",
        "matlab_environment_sha256",
    ):
        if type(qualification[key]) is not str or not SHA256_RE.fullmatch(
                qualification[key]):
            raise DispatchAuthorizationError(
                f"qualification evidence {key} is malformed")
    contact = _exact_keys(
        payload["contact_closure"], CONTACT_FIELDS,
        "contact-closure evidence",
    )
    if (
        contact["receipt_schema"]
        != "contact-closure-authorization-receipt-v2"
        or contact["status"] != "ACCEPTED"
        or contact["expected_cases"] != EXPECTED_CONTACT_CASES
        or contact["accepted_cases"] != EXPECTED_CONTACT_CASES
        or contact["channel_schema_id"] != REQUIRED_CHANNEL_SCHEMA_ID
    ):
        raise DispatchAuthorizationError(
            "contact-closure evidence status/cardinality is invalid")
    for key in (
        "authorization_receipt_sha256",
        "matlab_environment_sha256",
        "generator_source_root_sha256",
        "policy_sha256",
        "selection_sha256",
        "case_artifact_root_sha256",
        "gate_summary_sha256",
        "gate_artifact_root_sha256",
        "dataset_descriptors_root_sha256",
    ):
        if type(contact[key]) is not str or not SHA256_RE.fullmatch(
                contact[key]):
            raise DispatchAuthorizationError(
                f"contact-closure evidence {key} is malformed")
    if (
        contact["generator_source_root_sha256"]
        != qualification["generator_source_root_sha256"]
    ):
        raise DispatchAuthorizationError(
            "qualification and contact evidence bind different generator "
            "source roots")
    if (
        contact["matlab_environment_sha256"]
        != qualification["matlab_environment_sha256"]
    ):
        raise DispatchAuthorizationError(
            "qualification and contact evidence bind different MATLAB "
            "environment identities")
    if contact["declared_host_id"] not in qualification[
            "intended_host_ids"]:
        raise DispatchAuthorizationError(
            "contact reference host is absent from the intended generation "
            "host inventory")


def _publish_create_once(path: Path, payload: dict[str, Any]) -> str:
    raw = _canonical_json_bytes(payload)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink():
        raise DispatchAuthorizationError(
            f"manifest is create-once and already exists: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise DispatchAuthorizationError(
            "legacy/stale manifest temporary requires manual forensic review: "
            f"{temporary}")
    # Publish directly through O_EXCL/create-once.  A temporary-hardlink
    # protocol would need a path-based unlink after publication; another
    # writer could replace that pathname in the check/unlink gap and cause us
    # to delete unrelated bytes.  Direct creation may expose a partial file to
    # a concurrent reader, but that reader can only fail closed, and any write
    # failure leaves the create-once path for forensic review.
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DispatchAuthorizationError(
            "manifest final path was created concurrently") from exc
    except OSError as exc:
        raise DispatchAuthorizationError(
            "cannot persist the create-once manifest; any created path was "
            "left untouched for forensic review"
        ) from exc
    final = _snapshot_regular(path, "published dispatch manifest")
    if final.raw != raw:
        raise DispatchAuthorizationError(
            "published dispatch manifest differs from intended bytes")
    return final.sha256
