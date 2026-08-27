"""Fresh Paper-1 stage-bound CUDA-capacity receipt publisher.

This non-scientific qualification entry point runs all 16 registered
worst-case capacity probes on the CUDA runtime of the current PC. It publishes
one content-addressed canonical receipt per selected execution block in an
existing external directory. Versions and hardware are provenance; successful
real training probes and VRAM headroom qualify this host. Qualification is
fresh-only.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
if _bootstrap_source_root not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, _bootstrap_source_root)
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from core.campaign_contract import STAGE_ORDER
from core.capacity_preflight import (
    capacity_receipt_path,
    load_capacity_receipt,
    run_capacity_preflight,
    validate_capacity_receipt,
    write_capacity_receipt,
)
from core.environment import load_environment_lock, validate_environment_lock
from core.execution_environment import current_execution_runtime_for_stage
from core.paper1_dispatch import generation_manifest, training_manifests
from core.paper1_training_contract import HPO_RESTART_SEEDS, canonical_json_bytes
from core.source_provenance import repository_source_snapshot
from core.utils import set_global_seed
from qualification_path_safety import (
    ReceiptSnapshot,
    assert_snapshot_unchanged,
    canonical_existing_path,
    snapshot_receipt,
)
from training.trainer import TRAIN_PROTOCOL


REPO = Path(__file__).resolve().parent
CAPACITY_SETUP_SEED = HPO_RESTART_SEEDS[0]
OUTPUT_SCHEMA = "ttbi-paper1-capacity-publication-v1"
OUTPUT_SET_SCHEMA = "ttbi-paper1-capacity-publication-set-v1"
CLASSIFICATION = "NON_SCIENTIFIC_COMPUTE_AND_CAPACITY_EVIDENCE"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
BUNDLE_IDENTITY_NAME = "paper1_bundle_identity.json"
BUNDLE_IDENTITY_SCHEMA = "ttbi-paper1-bundle-identity-v1"
SOURCE_MANIFEST_NAME = "bundle_source_files.txt"
_BUNDLE_IDENTITY_FIELDS = frozenset({
    "schema",
    "source_commit",
    "bundle_kind",
    "target",
    "source_manifest_name",
    "source_manifest_sha256",
    "source_manifest_entry_count",
    "reviewed_source_root_sha256",
    "reviewed_source_file_count",
    "generator_source_root_sha256",
    "generator_source_file_count",
    "python_runtime_source_root_sha256",
    "python_runtime_source_file_count",
    "bundle_manifest_name",
    "bundle_manifest_sha256",
})


class CapacityPublicationError(RuntimeError):
    """The fresh Paper-1 capacity publication was refused."""


@dataclass(frozen=True)
class _AuthenticatedSource:
    """Stable Git or extracted-bundle identity used around the CUDA probe."""

    mode: str
    source_commit: str
    source_snapshot: Any
    bundle_identity: dict[str, Any] | None = None
    bundle_identity_snapshot: ReceiptSnapshot | None = None
    bundle_manifest_snapshot: ReceiptSnapshot | None = None
    bundle_source_snapshots: tuple[ReceiptSnapshot, ...] = ()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapacityPublicationError(
            f"git command failed for {args!r}"
        ) from exc


def _require_clean_commit(repo: Path) -> str:
    """Return exact clean HEAD for one canonical repository root."""

    repository = canonical_existing_path(
        repo, "capacity source repository", kind="directory"
    )
    try:
        reported_root = _git(
            repository, "rev-parse", "--show-toplevel"
        ).stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CapacityPublicationError(
            "git repository root is not UTF-8"
        ) from exc
    try:
        resolved_root = Path(reported_root).resolve(strict=True)
    except OSError as exc:
        raise CapacityPublicationError(
            "git reported an unavailable repository root"
        ) from exc
    if os.path.normcase(str(resolved_root)) != os.path.normcase(str(repository)):
        raise CapacityPublicationError(
            "capacity source directory is not the exact git worktree root"
        )

    dirty = _git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if dirty:
        rendered = dirty.decode(
            "utf-8", errors="backslashreplace"
        ).replace("\0", "\n").strip()
        raise CapacityPublicationError(
            "Paper-1 capacity qualification requires exact clean commit A:\n"
            + rendered
        )
    try:
        commit = _git(
            repository, "rev-parse", "--verify", "HEAD^{commit}"
        ).stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CapacityPublicationError(
            "git did not return an ASCII commit identity"
        ) from exc
    if not _HEX40.fullmatch(commit):
        raise CapacityPublicationError(
            "git did not return one full SHA-1 commit identity"
        )
    return commit


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityPublicationError(
                f"duplicate JSON key in embedded bundle identity: {key!r}"
            )
        result[key] = value
    return result


def _canonical_json_object(
    snapshot: ReceiptSnapshot, label: str
) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.raw.decode("ascii"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapacityPublicationError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or snapshot.raw != canonical_json_bytes(value)
    ):
        raise CapacityPublicationError(
            f"{label} bytes are not one canonical JSON object"
        )
    return value


def _snapshot_root(
    names: tuple[str, ...], snapshots: dict[str, ReceiptSnapshot]
) -> tuple[str, int]:
    lines = [f"{name}:{snapshots[name].sha256}" for name in names]
    return (
        hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        len(lines),
    )


def _embedded_bundle_source_identity(
    repo: Path, *, expected_bundle_kind: str = "training"
) -> _AuthenticatedSource:
    """Authenticate one extracted Paper-1 ZIP without requiring Git."""

    root = canonical_existing_path(
        repo, "capacity bundle source root", kind="directory"
    )
    if os.path.lexists(root / ".git"):
        raise CapacityPublicationError(
            "embedded-bundle identity is only valid when .git is absent"
        )
    identity_snapshot = snapshot_receipt(
        root / BUNDLE_IDENTITY_NAME, "embedded bundle identity"
    )
    identity = _canonical_json_object(
        identity_snapshot, "embedded bundle identity"
    )
    if set(identity) != _BUNDLE_IDENTITY_FIELDS:
        raise CapacityPublicationError(
            "embedded bundle identity field inventory drifted"
        )
    expected_targets = (
        {"labA", "labB"}
        if expected_bundle_kind == "training"
        else {"F40-S", "F40-M", "L99-S", "L99-M"}
        if expected_bundle_kind == "generation"
        else set()
    )
    if (
        identity.get("schema") != BUNDLE_IDENTITY_SCHEMA
        or identity.get("bundle_kind") != expected_bundle_kind
        or identity.get("target") not in expected_targets
        or identity.get("source_manifest_name") != SOURCE_MANIFEST_NAME
        or not isinstance(identity.get("source_commit"), str)
        or _HEX40.fullmatch(identity["source_commit"]) is None
    ):
        raise CapacityPublicationError(
            "embedded bundle identity header is not the required bundle kind"
        )
    digest_fields = (
        "source_manifest_sha256",
        "reviewed_source_root_sha256",
        "generator_source_root_sha256",
        "python_runtime_source_root_sha256",
        "bundle_manifest_sha256",
    )
    count_fields = (
        "source_manifest_entry_count",
        "reviewed_source_file_count",
        "generator_source_file_count",
        "python_runtime_source_file_count",
    )
    if any(
        not isinstance(identity.get(field), str)
        or _HEX64.fullmatch(identity[field]) is None
        for field in digest_fields
    ) or any(
        isinstance(identity.get(field), bool)
        or not isinstance(identity.get(field), int)
        or identity[field] <= 0
        for field in count_fields
    ):
        raise CapacityPublicationError(
            "embedded bundle identity digest/count fields are malformed"
        )

    source_snapshot = repository_source_snapshot(root)
    names = source_snapshot.manifest_names
    if (
        identity["source_manifest_entry_count"] != len(names)
        or identity["reviewed_source_file_count"] != len(names)
    ):
        raise CapacityPublicationError(
            "embedded source-manifest count differs from bundle identity"
        )
    source_manifest_sha256 = hashlib.sha256(
        source_snapshot.manifest_snapshot.raw
    ).hexdigest()
    if identity["source_manifest_sha256"] != source_manifest_sha256:
        raise CapacityPublicationError(
            "embedded source manifest differs from bundle identity"
        )

    source_snapshots = tuple(
        snapshot_receipt(
            root.joinpath(*name.split("/")),
            f"embedded reviewed source {name}",
        )
        for name in names
    )
    indexed = {
        name: snapshot for name, snapshot in zip(names, source_snapshots, strict=True)
    }
    reviewed_sha, reviewed_count = _snapshot_root(names, indexed)
    if (
        identity["reviewed_source_root_sha256"] != reviewed_sha
        or identity["reviewed_source_file_count"] != reviewed_count
        or identity["generator_source_root_sha256"]
        != source_snapshot.generator.sha256
        or identity["generator_source_file_count"]
        != source_snapshot.generator.file_count
        or identity["python_runtime_source_root_sha256"]
        != source_snapshot.python_runtime.sha256
        or identity["python_runtime_source_file_count"]
        != source_snapshot.python_runtime.file_count
    ):
        raise CapacityPublicationError(
            "embedded executable source roots differ from bundle identity"
        )

    expected_manifest_name = (
        "training_job_manifest.json"
        if expected_bundle_kind == "training"
        else "generation_bundle_manifest.json"
    )
    if identity.get("bundle_manifest_name") != expected_manifest_name:
        raise CapacityPublicationError(
            "embedded bundle identity names a foreign bundle manifest"
        )
    manifest_snapshot = snapshot_receipt(
        root / expected_manifest_name, "embedded bundle manifest"
    )
    expected_manifest = canonical_json_bytes(
        training_manifests()[identity["target"]]
        if expected_bundle_kind == "training"
        else generation_manifest(identity["target"])
    )
    if (
        manifest_snapshot.raw != expected_manifest
        or identity["bundle_manifest_sha256"] != manifest_snapshot.sha256
    ):
        raise CapacityPublicationError(
            "embedded manifest differs from its source-derived identity"
        )

    source_snapshot.assert_unchanged()
    assert_snapshot_unchanged(identity_snapshot, "embedded bundle identity")
    assert_snapshot_unchanged(
        manifest_snapshot, "embedded bundle manifest"
    )
    for name, snapshot in zip(names, source_snapshots, strict=True):
        assert_snapshot_unchanged(snapshot, f"embedded reviewed source {name}")
    return _AuthenticatedSource(
        mode="embedded-bundle",
        source_commit=identity["source_commit"],
        source_snapshot=source_snapshot,
        bundle_identity=identity,
        bundle_identity_snapshot=identity_snapshot,
        bundle_manifest_snapshot=manifest_snapshot,
        bundle_source_snapshots=source_snapshots,
    )


def _authenticate_source(repo: Path) -> _AuthenticatedSource:
    """Prefer clean-Git evidence; otherwise require the embedded bundle seal."""

    if os.path.lexists(repo / ".git"):
        commit = _require_clean_commit(repo)
        return _AuthenticatedSource(
            mode="git-clean",
            source_commit=commit,
            source_snapshot=repository_source_snapshot(repo),
        )
    return _embedded_bundle_source_identity(repo)


def authenticate_training_execution_source(
    expected_machine_role: str,
    repo: Path = REPO,
) -> _AuthenticatedSource:
    """Authenticate source before one training job can create any evidence.

    A clean Git worktree carries its identity through HEAD. An extracted ZIP
    carries the same reviewed source boundary through its embedded bundle seal,
    which must also name the exact Lab-A/Lab-B manifest already selected by the
    driver.
    """

    if expected_machine_role not in training_manifests():
        raise CapacityPublicationError(
            "training execution requested an unregistered machine role"
        )
    authenticated = _authenticate_source(repo)
    if (
        authenticated.mode == "embedded-bundle"
        and (
            authenticated.bundle_identity is None
            or authenticated.bundle_identity.get("bundle_kind") != "training"
            or authenticated.bundle_identity.get("target")
            != expected_machine_role
        )
    ):
        raise CapacityPublicationError(
            "embedded training bundle identity does not match the selected "
            f"{expected_machine_role} manifest"
        )
    _assert_authenticated_source_unchanged(authenticated, repo)
    return authenticated


def _assert_authenticated_source_unchanged(
    source: _AuthenticatedSource,
    repo: Path = REPO,
) -> None:
    source.source_snapshot.assert_unchanged()
    if source.mode == "git-clean":
        if _require_clean_commit(repo) != source.source_commit:
            raise CapacityPublicationError(
                "source commit changed during capacity qualification"
            )
        return
    if (
        source.mode != "embedded-bundle"
        or source.bundle_identity_snapshot is None
        or source.bundle_manifest_snapshot is None
        or source.bundle_identity is None
    ):
        raise CapacityPublicationError("capacity source identity mode is invalid")
    assert_snapshot_unchanged(
        source.bundle_identity_snapshot, "embedded bundle identity"
    )
    assert_snapshot_unchanged(
        source.bundle_manifest_snapshot, "embedded bundle manifest"
    )
    for name, snapshot in zip(
        source.source_snapshot.manifest_names,
        source.bundle_source_snapshots,
        strict=True,
    ):
        assert_snapshot_unchanged(snapshot, f"embedded reviewed source {name}")


def _external_receipt_directory(
    raw: str | os.PathLike[str], repo: Path = REPO
) -> Path:
    """Require one existing canonical directory outside the repository."""

    repository = canonical_existing_path(
        repo, "capacity source repository", kind="directory"
    )
    directory = canonical_existing_path(
        raw, "capacity receipt directory", kind="directory"
    )
    try:
        common = os.path.commonpath((str(directory), str(repository)))
    except ValueError:
        common = ""
    if os.path.normcase(common) == os.path.normcase(str(repository)):
        raise CapacityPublicationError(
            "capacity receipt directory must remain outside the repository"
        )
    return directory


def _fresh_target(runtime: dict[str, Any], receipt_dir: Path) -> Path:
    """Resolve and reserve the expected content-addressed target identity."""

    target = capacity_receipt_path(runtime, receipt_dir=receipt_dir)
    expected_name = re.fullmatch(
        rf"capacity_preflight_{re.escape(runtime['execution_block'])}_"
        r"([0-9a-f]{64})\.json",
        target.name,
    )
    if (
        not target.is_absolute()
        or os.path.normcase(str(target.parent))
        != os.path.normcase(str(receipt_dir))
        or expected_name is None
    ):
        raise CapacityPublicationError(
            "capacity core returned a noncanonical receipt target"
        )
    if os.path.lexists(target):
        raise CapacityPublicationError(
            "fresh capacity receipt target already exists; preserve it and "
            f"do not claim a replacement run: {target}"
        )
    return target


def create_stage_capacity_receipt(
    stage: str,
    receipt_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run and freshly publish one block-bound 16-cell CUDA preflight."""

    if stage not in STAGE_ORDER:
        raise CapacityPublicationError(
            f"capacity stage must be one of {STAGE_ORDER}, got {stage!r}"
        )

    authenticated_source = _authenticate_source(REPO)
    tested_commit = authenticated_source.source_commit
    destination = _external_receipt_directory(receipt_dir, REPO)
    source_snapshot = authenticated_source.source_snapshot
    environment_lock = load_environment_lock(
        REPO / "environment" / "campaign-py313-cu128.json"
    )
    validate_environment_lock(environment_lock)

    if CAPACITY_SETUP_SEED != 104729:
        raise CapacityPublicationError(
            "capacity setup seed is not the registered first HPO seed"
        )
    set_global_seed(CAPACITY_SETUP_SEED, TRAIN_PROTOCOL["determinism"])
    runtime = current_execution_runtime_for_stage(stage)
    target = _fresh_target(runtime, destination)

    envelope = run_capacity_preflight(
        runtime,
        source_root_sha256=source_snapshot.python_runtime.sha256,
        source_file_count=source_snapshot.python_runtime.file_count,
    )
    envelope = validate_capacity_receipt(
        envelope,
        expected_runtime=runtime,
        expected_source_root_sha256=source_snapshot.python_runtime.sha256,
        expected_source_file_count=source_snapshot.python_runtime.file_count,
    )

    _assert_authenticated_source_unchanged(authenticated_source)
    if current_execution_runtime_for_stage(stage) != runtime:
        raise CapacityPublicationError(
            "execution environment changed during the capacity probes"
        )
    if _fresh_target(runtime, destination) != target:
        raise CapacityPublicationError(
            "content-addressed capacity target changed during qualification"
        )

    receipt_sha256 = write_capacity_receipt(
        target,
        envelope,
        expected_runtime=runtime,
        expected_source_root_sha256=source_snapshot.python_runtime.sha256,
        expected_source_file_count=source_snapshot.python_runtime.file_count,
        require_absent=True,
    )
    published = canonical_existing_path(
        target, "published capacity receipt", kind="file"
    )
    loaded = load_capacity_receipt(
        published,
        expected_runtime=runtime,
        expected_source_root_sha256=source_snapshot.python_runtime.sha256,
        expected_source_file_count=source_snapshot.python_runtime.file_count,
    )
    if loaded != envelope or receipt_sha256 != envelope["receipt_sha256"]:
        raise CapacityPublicationError(
            "published capacity receipt differs from the qualified envelope"
        )

    _assert_authenticated_source_unchanged(authenticated_source)
    if current_execution_runtime_for_stage(stage) != runtime:
        raise CapacityPublicationError(
            "execution environment changed during capacity publication"
        )
    receipt = loaded["receipt"]
    result = {
        "schema": OUTPUT_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PASS",
        "stage": stage,
        "determinism_seed": CAPACITY_SETUP_SEED,
        "tested_source_commit": tested_commit,
        "source_identity_mode": authenticated_source.mode,
        "environment_lock_sha256": environment_lock["sha256"],
        "execution_environment_sha256": (
            runtime["execution_environment_sha256"]
        ),
        "python_runtime_source_root_sha256": (
            source_snapshot.python_runtime.sha256
        ),
        "python_runtime_source_file_count": (
            source_snapshot.python_runtime.file_count
        ),
        "capacity_policy_sha256": receipt["policy_sha256"],
        "capacity_receipt_sha256": loaded["receipt_sha256"],
        "capacity_receipt_path": str(published),
        "architecture_probe_count": len(receipt["measurements"]),
        "minimum_observed_headroom_bytes": (
            receipt["minimum_observed_headroom_bytes"]
        ),
    }
    if (
        result["architecture_probe_count"] != 16
        or not _HEX64.fullmatch(result["capacity_receipt_sha256"])
    ):
        raise CapacityPublicationError(
            "published capacity result is incomplete or malformed"
        )
    return result


def create_f40s_capacity_receipt(
    receipt_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Backward-compatible single-block publisher for F40-S."""

    return create_stage_capacity_receipt("F40-S", receipt_dir)


def create_all_stage_capacity_receipts(
    receipt_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Freshly qualify all four independent Paper-1 execution blocks."""

    results = [
        create_stage_capacity_receipt(stage, receipt_dir)
        for stage in STAGE_ORDER
    ]
    if (
        [result.get("stage") for result in results] != list(STAGE_ORDER)
        or any(result.get("status") != "PASS" for result in results)
    ):
        raise CapacityPublicationError(
            "four-block capacity publication returned an incomplete result set"
        )
    return {
        "schema": OUTPUT_SET_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PASS",
        "stages": list(STAGE_ORDER),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="freshly qualify only this stage/execution block",
    )
    selection.add_argument(
        "--all-stages",
        action="store_true",
        help="freshly qualify F40-S, F40-M, L99-S, and L99-M in order",
    )
    parser.add_argument(
        "--receipt-dir",
        required=True,
        help=(
            "existing canonical absolute directory outside the repository; "
            "the content-addressed target must not already exist"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = (
            create_all_stage_capacity_receipts(args.receipt_dir)
            if args.all_stages
            else create_stage_capacity_receipt(args.stage, args.receipt_dir)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PAPER-1 CAPACITY PREFLIGHT REFUSED: {exc}") from exc
    print("PAPER-1 CUDA CAPACITY PREFLIGHT PASS")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    main()
