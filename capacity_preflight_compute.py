"""Fresh Paper-1 F40-S CUDA-capacity receipt publisher.

This non-scientific qualification entry point runs all 16 registered
worst-case capacity probes on the exact locked CUDA runtime that will execute
the Paper-1 compute benchmark.  It publishes one content-addressed canonical
receipt in an existing external directory.  Qualification is fresh-only: an
existing target is never reused or replaced.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before capacity "
            "imports"
        )

_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
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
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from core.capacity_preflight import (
    capacity_receipt_path,
    load_capacity_receipt,
    run_capacity_preflight,
    validate_capacity_receipt,
    write_capacity_receipt,
)
from core.environment import load_environment_lock, validate_environment_lock
from core.execution_environment import current_execution_runtime_for_stage
from core.paper1_training_contract import HPO_RESTART_SEEDS
from core.source_provenance import repository_source_snapshot
from core.utils import set_global_seed
from qualification_path_safety import canonical_existing_path
from training.trainer import TRAIN_PROTOCOL


REPO = Path(__file__).resolve().parent
STAGE = "F40-S"
CAPACITY_SETUP_SEED = HPO_RESTART_SEEDS[0]
OUTPUT_SCHEMA = "ttbi-paper1-capacity-publication-v1"
CLASSIFICATION = "NON_SCIENTIFIC_COMPUTE_AND_CAPACITY_EVIDENCE"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CapacityPublicationError(RuntimeError):
    """The fresh Paper-1 capacity publication was refused."""


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


def create_f40s_capacity_receipt(
    receipt_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run and freshly publish the benchmark-bound 16-cell CUDA preflight."""

    tested_commit = _require_clean_commit(REPO)
    destination = _external_receipt_directory(receipt_dir, REPO)
    source_snapshot = repository_source_snapshot(REPO)
    environment_lock = load_environment_lock(
        REPO / "environment" / "campaign-py313-cu128.json"
    )
    validate_environment_lock(environment_lock)

    if CAPACITY_SETUP_SEED != 104729:
        raise CapacityPublicationError(
            "capacity setup seed is not the registered first F40-S HPO seed"
        )
    set_global_seed(CAPACITY_SETUP_SEED, TRAIN_PROTOCOL["determinism"])
    runtime = current_execution_runtime_for_stage(STAGE)
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

    source_snapshot.assert_unchanged()
    if current_execution_runtime_for_stage(STAGE) != runtime:
        raise CapacityPublicationError(
            "execution environment changed during the capacity probes"
        )
    if _require_clean_commit(REPO) != tested_commit:
        raise CapacityPublicationError(
            "source commit changed during the capacity probes"
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

    source_snapshot.assert_unchanged()
    if current_execution_runtime_for_stage(STAGE) != runtime:
        raise CapacityPublicationError(
            "execution environment changed during capacity publication"
        )
    if _require_clean_commit(REPO) != tested_commit:
        raise CapacityPublicationError(
            "source commit changed during capacity publication"
        )
    receipt = loaded["receipt"]
    result = {
        "schema": OUTPUT_SCHEMA,
        "classification": CLASSIFICATION,
        "status": "PASS",
        "stage": STAGE,
        "determinism_seed": CAPACITY_SETUP_SEED,
        "tested_source_commit": tested_commit,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        result = create_f40s_capacity_receipt(args.receipt_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"PAPER-1 CAPACITY PREFLIGHT REFUSED: {exc}") from exc
    print("PAPER-1 CUDA CAPACITY PREFLIGHT PASS")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    main()
