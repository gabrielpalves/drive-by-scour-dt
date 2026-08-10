"""Mechanical, create-once authorization evidence for Paper-1 dispatch.

The final audit report is deliberately small and report-only.  It is therefore
not trusted as a substitute for the retained machine-readable evidence.  This
module creates and later revalidates one canonical external manifest that
points to:

* the completed production-path compute benchmark;
* the complete pairwise MATLAB-host qualification graph and its retained
  aggregate inventory receipt - every retained endpoint dataset directory is
  reopened and re-authenticated from disk, and the complete pairwise
  comparison behind every retained pair receipt is rerun at authorization
  time (``qualification_endpoint_revalidation``); and
* the exhaustive reference-host contact/time-step closure and its retained
  create-once receipt.

Creation is allowed only at the clean tested source commit A.  Revalidation is
allowed at A or at a clean report-only commit B whose sole A..B path is
``docs/audit_r5_results.md``.  There is no skip, trust, or synthetic backend in
the production API.

Module layout: the evidence-agnostic building blocks (manifest grammar,
canonical JSON, TOCTOU snapshots, canonical paths, create-once publication)
live in ``dispatch_manifest``; this file remains the short orchestrator that
binds the git boundary, the policy-source identity, and the three
authoritative evidence revalidators together.
"""
from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before evidence "
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
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
from typing import Any, Sequence
import unicodedata

from dispatch_manifest import (  # noqa: F401  (re-exported gate surface)
    BENCHMARK_FIELDS,
    COMMIT_RE,
    CONTACT_FIELDS,
    CONTACT_RECEIPT_FIELDS,
    DispatchAuthorizationError,
    FileSnapshot,
    MANIFEST_SCHEMA,
    MAX_AUTH_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    POLICY_FIELDS,
    POLICY_SCHEMA,
    POLICY_SOURCE_FILES,
    QUALIFICATION_FIELDS,
    EXPECTED_CONTACT_CASES,
    REQUIRED_BENCHMARK_SCHEMA,
    REQUIRED_CHANNEL_SCHEMA_ID,
    REQUIRED_QUALIFICATION_STAGES,
    SHA256_RE,
    STATUS,
    TOP_FIELDS,
    _assert_snapshot_unchanged,
    _canonical_existing_path,
    _canonical_json_bytes,
    _canonical_output_path,
    _canonical_value_sha256,
    _exact_keys,
    _is_junction,
    _is_within,
    _publish_create_once,
    _require_external_to_repo,
    _sha256_bytes,
    _sha256_file,
    _snapshot_regular,
    _strict_json_bytes,
    _validate_manifest_shape,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = "docs/audit_r5_results.md"


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode != 0:
        detail = process.stderr.decode(
            "utf-8", errors="replace").strip()
        raise DispatchAuthorizationError(
            f"Git command failed ({' '.join(args)}): "
            f"{detail or process.returncode}"
        )
    return process


def _git_head(repo: Path) -> str:
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}"
                ).stdout.decode("ascii").strip()
    if not COMMIT_RE.fullmatch(head):
        raise DispatchAuthorizationError("Git HEAD is not one SHA-1 commit")
    return head


def _assert_git_boundary(
    repo: Path,
    tested_source_commit: str,
    dispatch_source_commit: str,
) -> None:
    if (
        not COMMIT_RE.fullmatch(tested_source_commit)
        or not COMMIT_RE.fullmatch(dispatch_source_commit)
    ):
        raise DispatchAuthorizationError(
            "tested/dispatch source commits must be 40 lowercase hex"
        )
    if _git_head(repo) != dispatch_source_commit:
        raise DispatchAuthorizationError(
            "current HEAD changed or differs from the dispatch commit"
        )
    dirty = _git(
        repo,
        "--literal-pathspecs",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if dirty:
        rendered = dirty.decode(
            "utf-8", errors="backslashreplace").replace("\0", "\n").strip()
        raise DispatchAuthorizationError(
            "authorization requires an entirely clean worktree:\n" + rendered
        )
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        tested_source_commit,
        dispatch_source_commit,
        check=False,
    )
    if ancestor.returncode != 0:
        raise DispatchAuthorizationError(
            "tested source A is not an ancestor of dispatch source"
        )
    changed = _git(
        repo,
        "diff",
        "--name-only",
        f"{tested_source_commit}..{dispatch_source_commit}",
    ).stdout.decode("utf-8").splitlines()
    expected = [] if tested_source_commit == dispatch_source_commit \
        else [REPORT_PATH]
    if changed != expected:
        raise DispatchAuthorizationError(
            "A..current may be empty at creation or contain only the final "
            f"report at dispatch; observed={changed!r}"
        )


def _tracked_blob_bytes(repo: Path, commit: str, relative: str) -> bytes:
    process = _git(
        repo, "show", f"{commit}:{relative}", check=False)
    if process.returncode != 0:
        raise DispatchAuthorizationError(
            f"required policy source is absent from tested A: {relative}"
        )
    entry = _git(
        repo, "ls-tree", commit, "--", relative).stdout
    fields = entry.split(b"\t", 1)[0].split()
    if len(fields) != 3 or fields[0] not in {b"100644", b"100755"} \
            or fields[1] != b"blob":
        raise DispatchAuthorizationError(
            f"policy source is not a regular tracked blob: {relative}"
        )
    return process.stdout


def _policy_source_names(manifest_raw: bytes) -> tuple[str, ...]:
    try:
        text = manifest_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DispatchAuthorizationError(
            "bundle source manifest is not UTF-8") from exc
    names: list[str] = []
    windows_forbidden = set('<>:"|?*')
    windows_reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("/")
        bad_component = any(
            not part
            or part in {".", ".."}
            or part != part.strip()
            or part.endswith(".")
            or any(
                character in windows_forbidden
                or ord(character) < 32
                or ord(character) == 127
                for character in part
            )
            or part.split(".", 1)[0].upper() in windows_reserved
            for part in parts
        )
        if (
            line != line.strip()
            or "\\" in line
            or line.startswith("/")
            or PureWindowsPath(line).is_absolute()
            or PurePosixPath(line).as_posix() != line
            or unicodedata.normalize("NFC", line) != line
            or bad_component
        ):
            raise DispatchAuthorizationError(
                "bundle source manifest has an unsafe/noncanonical entry at "
                f"line {line_number}")
        names.append(line)
    if (
        not names
        or names != sorted(names)
        or len(names) != len(set(names))
        or len(names) != len({name.casefold() for name in names})
    ):
        raise DispatchAuthorizationError(
            "bundle source manifest is empty, unsorted, or duplicate")
    selected = {
        name for name in names
        if name.endswith(".py")
        or name in {
            "environment/campaign-py313-cu128.json",
            "requirements-campaign-py313-cu128.txt",
        }
    }
    selected.add("bundle_source_files.txt")
    # Audit/publication executables are shipped for source-complete review even
    # when target hosts do not invoke them during a campaign run.
    required_in_manifest = set(POLICY_SOURCE_FILES)
    missing = required_in_manifest - selected
    if missing:
        raise DispatchAuthorizationError(
            "bundle source manifest omits dispatch-verification source: "
            f"{sorted(missing)!r}")
    selected.update(POLICY_SOURCE_FILES)
    return tuple(sorted(selected))


def _current_policy(
    repo: Path,
    tested_source_commit: str,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    snapshots: list[FileSnapshot] = []
    hashes: dict[str, str] = {}
    manifest_path = repo / "bundle_source_files.txt"
    manifest_snapshot = _snapshot_regular(
        manifest_path, "policy source bundle_source_files.txt")
    tested_manifest = _tracked_blob_bytes(
        repo, tested_source_commit, "bundle_source_files.txt")
    if manifest_snapshot.raw != tested_manifest:
        raise DispatchAuthorizationError(
            "current bundle_source_files.txt differs from tested source A")
    source_names = _policy_source_names(manifest_snapshot.raw)
    for relative in source_names:
        if relative == "bundle_source_files.txt":
            snapshot = manifest_snapshot
            tested_bytes = tested_manifest
        else:
            path = repo.joinpath(*relative.split("/"))
            snapshot = _snapshot_regular(path, f"policy source {relative}")
            tested_bytes = _tracked_blob_bytes(
                repo, tested_source_commit, relative)
        if snapshot.raw != tested_bytes:
            raise DispatchAuthorizationError(
                f"current {relative} bytes differ from tested source A"
            )
        snapshots.append(snapshot)
        hashes[relative] = snapshot.sha256
    tree_oid = _git(
        repo, "rev-parse", f"{tested_source_commit}^{{tree}}"
    ).stdout.decode("ascii").strip()
    if not COMMIT_RE.fullmatch(tree_oid):
        raise DispatchAuthorizationError(
            "tested source tree is not a 40-character Git object ID"
        )
    root_material = "".join(
        f"{name}:{hashes[name]}\n" for name in sorted(hashes)
    ).encode("utf-8")
    return {
        "schema": POLICY_SCHEMA,
        "tested_source_tree_oid": tree_oid,
        "source_sha256": hashes,
        "source_root_sha256": _sha256_bytes(root_material),
    }, tuple(snapshots)


def _qualification_evidence(
    host_ids: Sequence[str],
    pair_receipt_paths: Sequence[str],
    inventory_receipt_path: str,
    *,
    repo: Path,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    import qualification_endpoint_revalidation as endpoint_revalidation
    import qualification_receipt_inventory as inventory

    try:
        hosts = inventory.validate_host_ids(host_ids)
    except inventory.QualificationInventoryError as exc:
        raise DispatchAuthorizationError(
            f"qualification host inventory is invalid: {exc}") from exc
    canonical_pair_paths = tuple(
        _canonical_existing_path(
            raw, f"qualification pair receipt {index}", kind="file")
        for index, raw in enumerate(pair_receipt_paths, 1)
    )
    if tuple(str(path) for path in canonical_pair_paths) != tuple(
            sorted(str(path) for path in canonical_pair_paths)):
        raise DispatchAuthorizationError(
            "qualification pair receipt paths must be sorted canonically"
        )
    if len(set(canonical_pair_paths)) != len(canonical_pair_paths):
        raise DispatchAuthorizationError(
            "qualification pair receipt paths must be duplicate-free"
        )
    for path in canonical_pair_paths:
        _require_external_to_repo(
            path, repo, "qualification pair receipt")
    pair_snapshots = tuple(
        _snapshot_regular(path, f"qualification pair receipt {index}")
        for index, path in enumerate(canonical_pair_paths, 1)
    )
    inventory_path = _canonical_existing_path(
        inventory_receipt_path,
        "retained qualification inventory receipt",
        kind="file",
    )
    inventory_snapshot = _snapshot_regular(
        inventory_path, "retained qualification inventory receipt")
    _require_external_to_repo(
        inventory_path, repo, "qualification inventory receipt")
    if inventory_path in canonical_pair_paths:
        raise DispatchAuthorizationError(
            "qualification inventory and pair receipts must be path-disjoint")
    pair_receipt_directories = {
        path.parent for path in canonical_pair_paths
    }
    if inventory_path.parent in pair_receipt_directories:
        raise DispatchAuthorizationError(
            "aggregate inventory receipt must remain outside every dedicated "
            "pair-receipt directory")
    try:
        result = inventory.validate_inventory(hosts, canonical_pair_paths)
        # ``validate_inventory`` performs its own secure reads.  Bind its
        # returned graph explicitly to the earlier dispatch snapshots so an
        # A/B pathname swap cannot make the aggregate describe bytes other
        # than those retained for this authorization pass.
        returned_edges = {str(edge.path): edge for edge in result.edges}
        expected_paths = {str(snapshot.path) for snapshot in pair_snapshots}
        if (
            len(result.edges) != len(pair_snapshots)
            or len(returned_edges) != len(result.edges)
            or set(returned_edges) != expected_paths
        ):
            raise DispatchAuthorizationError(
                "qualification inventory returned a different/duplicate "
                "receipt-path inventory from the dispatch input snapshots"
            )
        mismatched_receipts = [
            str(snapshot.path)
            for snapshot in pair_snapshots
            if returned_edges[str(snapshot.path)].receipt_sha256
            != snapshot.sha256
        ]
        if mismatched_receipts:
            raise DispatchAuthorizationError(
                "qualification inventory receipt SHA/path bindings differ "
                "from the dispatch input snapshots: "
                f"{mismatched_receipts}"
            )
        expected_payload = inventory.inventory_receipt_payload(result)
        expected_bytes = inventory._canonical_json_bytes(expected_payload)
    except inventory.QualificationInventoryError as exc:
        raise DispatchAuthorizationError(
            f"qualification receipt graph failed revalidation: {exc}"
        ) from exc
    if inventory_snapshot.raw != expected_bytes:
        raise DispatchAuthorizationError(
            "retained inventory receipt is not the exact canonical "
            "recomputation from the supplied pairwise receipts"
        )
    # validate_inventory above already reran the COMPLETE pairwise comparator
    # behind every graph edge.  At the dispatch boundary, repeat that
    # revalidation from the exact receipt bytes captured in pair_snapshots.
    # This defence-in-depth pass binds authorization to those immutable input
    # snapshots and again requires exact verdict/evidence/statistics/raw-count
    # agreement.  No argument can skip either pass.
    for index, snapshot in enumerate(pair_snapshots, 1):
        retained_pair = _strict_json_bytes(
            snapshot.raw, f"qualification pair receipt {index}")
        try:
            endpoint_revalidation.revalidate_edge_comparison(
                retained_pair, owner=str(snapshot.path))
        except inventory.QualificationInventoryError as exc:
            raise DispatchAuthorizationError(
                f"qualification pair receipt {index} failed full pairwise "
                f"comparator revalidation: {exc}"
            ) from exc
    for index, snapshot in enumerate(pair_snapshots, 1):
        _assert_snapshot_unchanged(
            snapshot, f"qualification pair receipt {index}")
    _assert_snapshot_unchanged(
        inventory_snapshot, "retained qualification inventory receipt")
    return {
        "required_stages": list(REQUIRED_QUALIFICATION_STAGES),
        "intended_host_ids": list(hosts),
        "pair_receipt_paths": [
            str(path) for path in canonical_pair_paths
        ],
        "inventory_receipt_path": str(inventory_path),
        "inventory_receipt_sha256": inventory_snapshot.sha256,
        "inventory_root_sha256": result.inventory_root_sha256,
        "accepted_pairwise_receipt_count": len(result.edges),
        "generator_source_root_sha256": (
            result.policy.generator_source_root_sha256
        ),
        "matlab_environment_sha256": (
            result.policy.campaign_matlab_environment_sha256
        ),
    }, (*pair_snapshots, inventory_snapshot)


def _contact_evidence(
    gate_directory: str,
    authorization_receipt_path: str,
    tested_source_commit: str,
    *,
    repo: Path,
    dispatch_source_commit: str,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    import check_contact_closure_gate as contact

    gate_dir = _canonical_existing_path(
        gate_directory, "contact gate directory", kind="directory")
    receipt_path = _canonical_existing_path(
        authorization_receipt_path,
        "contact authorization receipt",
        kind="file",
    )
    _require_external_to_repo(
        gate_dir, repo, "contact gate directory")
    _require_external_to_repo(
        receipt_path, repo, "contact authorization receipt")
    if _is_within(receipt_path, gate_dir):
        raise DispatchAuthorizationError(
            "contact authorization receipt must be outside the gate directory")
    receipt_snapshot = _snapshot_regular(
        receipt_path, "contact authorization receipt")
    retained = _strict_json_bytes(
        receipt_snapshot.raw, "contact authorization receipt")
    _exact_keys(retained, CONTACT_RECEIPT_FIELDS,
                "contact authorization receipt")

    def report_only_git_check(source_commit: str) -> None:
        if source_commit != tested_source_commit:
            raise contact.GateError(
                "contact checker requested a foreign tested source")
        try:
            _assert_git_boundary(
                repo, tested_source_commit, dispatch_source_commit)
        except DispatchAuthorizationError as exc:
            raise contact.GateError(str(exc)) from exc

    try:
        recomputed = contact.verify_existing_authorization_receipt(
            gate_dir,
            tested_source_commit,
            receipt_path,
            git_check=report_only_git_check,
        )
    except (contact.GateError, OSError, subprocess.SubprocessError) as exc:
        raise DispatchAuthorizationError(
            f"contact-closure evidence failed revalidation: {exc}"
        ) from exc
    _exact_keys(recomputed, CONTACT_RECEIPT_FIELDS,
                "recomputed contact authorization")
    for key in CONTACT_RECEIPT_FIELDS - {"validated_utc"}:
        if retained[key] != recomputed[key]:
            raise DispatchAuthorizationError(
                f"contact retained/recomputed receipt differs at {key}"
            )
    _assert_snapshot_unchanged(
        receipt_snapshot, "contact authorization receipt")
    gate_artifacts = _exact_keys(
        retained["gate_artifact_sha256"],
        set(retained["gate_artifact_sha256"]),
        "contact gate artifact hashes",
    )
    if not gate_artifacts or any(
        type(key) is not str
        or type(value) is not str
        or not SHA256_RE.fullmatch(value)
        for key, value in gate_artifacts.items()
    ):
        raise DispatchAuthorizationError(
            "contact gate artifact hashes are malformed"
        )
    descriptors = retained["dataset_descriptors"]
    if not isinstance(descriptors, list) or not descriptors or any(
            type(item) is not str for item in descriptors):
        raise DispatchAuthorizationError(
            "contact dataset descriptor list is malformed")
    return {
        "gate_directory": str(gate_dir),
        "authorization_receipt_path": str(receipt_path),
        "authorization_receipt_sha256": receipt_snapshot.sha256,
        "receipt_schema": retained["schema"],
        "status": retained["status"],
        "declared_host_id": retained["declared_host_id"],
        "matlab_environment_sha256": (
            retained["matlab_environment_sha256"]
        ),
        "generator_source_root_sha256": (
            retained["generator_source_root_sha256"]
        ),
        "policy_sha256": retained["policy_sha256"],
        "selection_sha256": retained["selection_sha256"],
        "case_artifact_root_sha256": (
            retained["case_artifact_root_sha256"]
        ),
        "gate_summary_sha256": retained["gate_summary_sha256"],
        "gate_artifact_root_sha256": (
            _canonical_value_sha256(gate_artifacts)
        ),
        "dataset_descriptors_root_sha256": (
            _canonical_value_sha256(descriptors)
        ),
        "expected_cases": retained["expected_cases"],
        "accepted_cases": retained["accepted_cases"],
        "channel_schema_id": retained["channel_schema_id"],
    }, (receipt_snapshot,)


def _benchmark_evidence(
    run_directory: str,
    tested_source_commit: str,
    *,
    repo: Path,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    import benchmark_paper1_compute as benchmark

    run_dir = _canonical_existing_path(
        run_directory, "benchmark run directory", kind="directory")
    summary_snapshot = _snapshot_regular(
        run_dir / "summary.json", "benchmark summary")
    state_snapshot = _snapshot_regular(
        run_dir / "run_state.json", "benchmark run state")
    try:
        evidence = benchmark.verify_completed_receipt(
            run_dir,
            tested_source_commit,
            repo=repo,
        )
    except (benchmark.ContractError, OSError, ValueError) as exc:
        raise DispatchAuthorizationError(
            f"benchmark evidence failed revalidation: {exc}"
        ) from exc
    _exact_keys(evidence, BENCHMARK_FIELDS, "benchmark authorization evidence")
    if evidence["schema"] != REQUIRED_BENCHMARK_SCHEMA:
        raise DispatchAuthorizationError(
            "benchmark verifier returned legacy/non-Paper-1 evidence; a "
            "fresh Paper-1 benchmark authorization schema is required"
        )
    if evidence["run_directory"] != str(run_dir):
        raise DispatchAuthorizationError(
            "benchmark verifier returned a noncanonical run directory")
    if (
        evidence["summary_sha256"] != summary_snapshot.sha256
        or evidence["run_state_sha256"] != state_snapshot.sha256
    ):
        raise DispatchAuthorizationError(
            "benchmark returned hashes differ from the pre-verification "
            "snapshots"
        )
    _assert_snapshot_unchanged(summary_snapshot, "benchmark summary")
    _assert_snapshot_unchanged(state_snapshot, "benchmark run state")
    return dict(evidence), (summary_snapshot, state_snapshot)


def _collect_evidence(
    *,
    repo: Path,
    tested_source_commit: str,
    dispatch_source_commit: str,
    benchmark_run_directory: str,
    qualification_host_ids: Sequence[str],
    qualification_pair_receipts: Sequence[str],
    qualification_inventory_receipt: str,
    contact_gate_directory: str,
    contact_authorization_receipt: str,
) -> tuple[dict[str, Any], tuple[FileSnapshot, ...]]:
    _assert_git_boundary(
        repo, tested_source_commit, dispatch_source_commit)
    policy, policy_snapshots = _current_policy(
        repo, tested_source_commit)
    benchmark, benchmark_snapshots = _benchmark_evidence(
        benchmark_run_directory, tested_source_commit, repo=repo)
    qualification, qualification_snapshots = _qualification_evidence(
        qualification_host_ids,
        qualification_pair_receipts,
        qualification_inventory_receipt,
        repo=repo,
    )
    contact, contact_snapshots = _contact_evidence(
        contact_gate_directory,
        contact_authorization_receipt,
        tested_source_commit,
        repo=repo,
        dispatch_source_commit=dispatch_source_commit,
    )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "status": STATUS,
        "tested_source_commit": tested_source_commit,
        "policy": policy,
        "benchmark": benchmark,
        "qualification": qualification,
        "contact_closure": contact,
    }
    _validate_manifest_shape(payload)
    all_receipt_paths = {
        Path(qualification["inventory_receipt_path"]),
        *(Path(path) for path in qualification["pair_receipt_paths"]),
        Path(contact["authorization_receipt_path"]),
    }
    if len(all_receipt_paths) != (
            2 + len(qualification["pair_receipt_paths"])):
        raise DispatchAuthorizationError(
            "qualification/contact authorization receipts must be "
            "mutually path-disjoint")
    benchmark_dir = Path(benchmark["run_directory"])
    contact_dir = Path(contact["gate_directory"])
    if (
        _is_within(benchmark_dir, contact_dir)
        or _is_within(contact_dir, benchmark_dir)
    ):
        raise DispatchAuthorizationError(
            "benchmark and contact gate directories must be path-disjoint")
    snapshots = (
        *policy_snapshots,
        *benchmark_snapshots,
        *qualification_snapshots,
        *contact_snapshots,
    )
    for snapshot in snapshots:
        _assert_snapshot_unchanged(
            snapshot, f"authorization input {snapshot.path}")
    _assert_git_boundary(
        repo, tested_source_commit, dispatch_source_commit)
    return payload, snapshots


def create_dispatch_authorization_manifest(
    output_path: str | os.PathLike[str],
    *,
    tested_source_commit: str,
    benchmark_run_directory: str,
    qualification_host_ids: Sequence[str],
    qualification_pair_receipts: Sequence[str],
    qualification_inventory_receipt: str,
    contact_gate_directory: str,
    contact_authorization_receipt: str,
    repo: str | os.PathLike[str] = ROOT,
) -> str:
    """Revalidate all evidence and publish one external create-once manifest."""

    root = Path(repo).resolve(strict=True)
    output = _canonical_output_path(output_path, root)
    head = _git_head(root)
    if head != tested_source_commit:
        raise DispatchAuthorizationError(
            "manifest creation must run at clean tested source commit A")
    payload, snapshots = _collect_evidence(
        repo=root,
        tested_source_commit=tested_source_commit,
        dispatch_source_commit=head,
        benchmark_run_directory=benchmark_run_directory,
        qualification_host_ids=qualification_host_ids,
        qualification_pair_receipts=qualification_pair_receipts,
        qualification_inventory_receipt=qualification_inventory_receipt,
        contact_gate_directory=contact_gate_directory,
        contact_authorization_receipt=contact_authorization_receipt,
    )
    evidence_files = {snapshot.path for snapshot in snapshots}
    if output in evidence_files:
        raise DispatchAuthorizationError(
            "dispatch manifest output must not alias an evidence file")
    if output.parent in {
        Path(path).parent
        for path in payload["qualification"]["pair_receipt_paths"]
    }:
        raise DispatchAuthorizationError(
            "dispatch manifest output must remain outside every dedicated "
            "pair-receipt directory")
    for owner, directory in (
        ("benchmark run", Path(payload["benchmark"]["run_directory"])),
        ("contact gate", Path(payload["contact_closure"]["gate_directory"])),
    ):
        if _is_within(output, directory):
            raise DispatchAuthorizationError(
                f"dispatch manifest output must be outside the {owner} "
                "directory")
    digest = _publish_create_once(output, payload)
    expected_manifest_bytes = _canonical_json_bytes(payload)
    try:
        published_snapshot = _snapshot_regular(
            output, "newly published dispatch manifest")
    except BaseException:
        # There is no authenticated snapshot from which a safe rollback can
        # be made.  Leave the path untouched and fail closed.
        raise
    if (
        published_snapshot.sha256 != digest
        or published_snapshot.raw != expected_manifest_bytes
    ):
        raise DispatchAuthorizationError(
            "newly published dispatch manifest changed before the "
            "post-publication evidence check; it was left untouched for "
            "forensic review"
        )
    try:
        # Reopen the COMPLETE evidence graph after the create-once file has
        # become visible.  This is intentionally not a receipt-only snapshot
        # pass: it reruns benchmark verification, every qualification edge
        # comparator, and contact-gate/dataset verification.
        recomputed, post_publish_snapshots = _collect_evidence(
            repo=root,
            tested_source_commit=tested_source_commit,
            dispatch_source_commit=head,
            benchmark_run_directory=(
                payload["benchmark"]["run_directory"]
            ),
            qualification_host_ids=(
                payload["qualification"]["intended_host_ids"]
            ),
            qualification_pair_receipts=(
                payload["qualification"]["pair_receipt_paths"]
            ),
            qualification_inventory_receipt=(
                payload["qualification"]["inventory_receipt_path"]
            ),
            contact_gate_directory=(
                payload["contact_closure"]["gate_directory"]
            ),
            contact_authorization_receipt=(
                payload["contact_closure"]["authorization_receipt_path"]
            ),
        )
        if recomputed != payload:
            raise DispatchAuthorizationError(
                "evidence changed across dispatch-manifest publication")
        for snapshot in (*snapshots, *post_publish_snapshots):
            _assert_snapshot_unchanged(
                snapshot, f"authorization input {snapshot.path}")
        _assert_snapshot_unchanged(
            published_snapshot, "newly published dispatch manifest")
        _assert_git_boundary(root, tested_source_commit, head)
    except BaseException as validation_error:
        # Never unlink through a mutable pathname after publication: an
        # attacker or synchronisation process could replace the checked entry
        # between validation and unlink.  The create-once artifact remains as
        # explicit forensic evidence and can never be mistaken for a success,
        # because this call raises and returns no authorization digest.
        raise DispatchAuthorizationError(
            "post-publication evidence validation failed; the create-once "
            "dispatch manifest was retained, unmodified, for forensic review: "
            f"{validation_error}"
        ) from validation_error
    return digest


def verify_dispatch_authorization_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    tested_source_commit: str,
    dispatch_source_commit: str,
    expected_manifest_sha256: str,
    repo: str | os.PathLike[str] = ROOT,
) -> dict[str, Any]:
    """Reopen all retained evidence named by an external canonical manifest."""

    root = Path(repo).resolve(strict=True)
    if (
        type(expected_manifest_sha256) is not str
        or not SHA256_RE.fullmatch(expected_manifest_sha256)
    ):
        raise DispatchAuthorizationError(
            "expected manifest SHA-256 must be 64 lowercase hex")
    raw_manifest_path = str(manifest_path)
    path = _canonical_existing_path(
        raw_manifest_path, "dispatch authorization manifest", kind="file")
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise DispatchAuthorizationError(
            "dispatch authorization manifest must be external to the repository"
        )
    manifest_snapshot = _snapshot_regular(
        path, "dispatch authorization manifest")
    if manifest_snapshot.sha256 != expected_manifest_sha256:
        raise DispatchAuthorizationError(
            "dispatch report/manifest SHA-256 binding differs")
    retained = _strict_json_bytes(
        manifest_snapshot.raw, "dispatch authorization manifest")
    _validate_manifest_shape(retained)
    if retained["tested_source_commit"] != tested_source_commit:
        raise DispatchAuthorizationError(
            "manifest tested source differs from the report")
    for owner, directory in (
        ("benchmark run", Path(retained["benchmark"]["run_directory"])),
        (
            "contact gate",
            Path(retained["contact_closure"]["gate_directory"]),
        ),
    ):
        if _is_within(path, directory):
            raise DispatchAuthorizationError(
                f"dispatch manifest must be outside the {owner} directory")
    if path.parent in {
        Path(receipt).parent
        for receipt in retained["qualification"]["pair_receipt_paths"]
    }:
        raise DispatchAuthorizationError(
            "dispatch manifest must remain outside every dedicated "
            "pair-receipt directory")
    qualification = retained["qualification"]
    recomputed, snapshots = _collect_evidence(
        repo=root,
        tested_source_commit=tested_source_commit,
        dispatch_source_commit=dispatch_source_commit,
        benchmark_run_directory=retained["benchmark"]["run_directory"],
        qualification_host_ids=qualification["intended_host_ids"],
        qualification_pair_receipts=qualification["pair_receipt_paths"],
        qualification_inventory_receipt=(
            qualification["inventory_receipt_path"]
        ),
        contact_gate_directory=(
            retained["contact_closure"]["gate_directory"]
        ),
        contact_authorization_receipt=(
            retained["contact_closure"]["authorization_receipt_path"]
        ),
    )
    if recomputed != retained:
        raise DispatchAuthorizationError(
            "retained dispatch manifest differs from current full evidence "
            "recomputation")
    for snapshot in snapshots:
        _assert_snapshot_unchanged(
            snapshot, f"authorization input {snapshot.path}")
    _assert_snapshot_unchanged(
        manifest_snapshot, "dispatch authorization manifest")
    _assert_git_boundary(
        root, tested_source_commit, dispatch_source_commit)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS",
        "tested_source_commit": tested_source_commit,
        "dispatch_source_commit": dispatch_source_commit,
        "manifest_path": str(path),
        "manifest_sha256": manifest_snapshot.sha256,
        "benchmark_evidence_root_sha256": (
            retained["benchmark"]["evidence_root_sha256"]
        ),
        "qualification_inventory_root_sha256": (
            retained["qualification"]["inventory_root_sha256"]
        ),
        "contact_case_artifact_root_sha256": (
            retained["contact_closure"]["case_artifact_root_sha256"]
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="revalidate evidence and create one external manifest")
    create.add_argument("--output", required=True)
    create.add_argument("--tested-source-commit", required=True)
    create.add_argument("--benchmark-run-directory", required=True)
    create.add_argument(
        "--qualification-host", action="append", required=True)
    create.add_argument(
        "--qualification-pair-receipt", action="append", required=True)
    create.add_argument("--qualification-inventory-receipt", required=True)
    create.add_argument("--contact-gate-directory", required=True)
    create.add_argument("--contact-authorization-receipt", required=True)

    verify = subparsers.add_parser(
        "verify", help="reopen an existing manifest and all retained evidence")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--tested-source-commit", required=True)
    verify.add_argument("--dispatch-source-commit", required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            digest = create_dispatch_authorization_manifest(
                args.output,
                tested_source_commit=args.tested_source_commit,
                benchmark_run_directory=args.benchmark_run_directory,
                qualification_host_ids=args.qualification_host,
                qualification_pair_receipts=(
                    args.qualification_pair_receipt
                ),
                qualification_inventory_receipt=(
                    args.qualification_inventory_receipt
                ),
                contact_gate_directory=args.contact_gate_directory,
                contact_authorization_receipt=(
                    args.contact_authorization_receipt
                ),
            )
            print("DISPATCH AUTHORIZATION MANIFEST: CREATED")
            print(f"  path: {Path(args.output).resolve()}")
            print(f"  sha256: {digest}")
        else:
            result = verify_dispatch_authorization_manifest(
                args.manifest,
                tested_source_commit=args.tested_source_commit,
                dispatch_source_commit=args.dispatch_source_commit,
                expected_manifest_sha256=(
                    args.expected_manifest_sha256
                ),
            )
            print("DISPATCH AUTHORIZATION MANIFEST: PASS")
            print(f"  tested source A: {result['tested_source_commit']}")
            print(f"  dispatch source: {result['dispatch_source_commit']}")
            print(f"  manifest sha256: {result['manifest_sha256']}")
        return 0
    except (
        DispatchAuthorizationError,
        ImportError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"DISPATCH AUTHORIZATION MANIFEST: FAIL: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
