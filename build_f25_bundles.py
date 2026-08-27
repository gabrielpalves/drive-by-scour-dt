"""Build the isolated, commit-bound F25-R and F25-X dispatch bundles."""

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
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
from typing import Any
import zipfile

from core.f25_experiment_contract import build_contract
from core.f25_training_contract import (
    build_training_plan,
    canonical_json_sha256,
    validate_training_plan,
)


REPO = Path(__file__).resolve().parent
SOURCE_MANIFEST = "bundle_source_files.txt"
BUNDLE_BUILDER = "build_f25_bundles.py"
BUNDLE_SCHEMA = "f25-dispatch-bundle-v2"
REGULAR_BLOB_MODES = frozenset({"100644", "100755"})
REQUIRED_F25_SOURCE = {
    BUNDLE_BUILDER,
    SOURCE_MANIFEST,
    "check_channel_semantics.py",
    "check_f25_capacity.py",
    "check_f25_experiment_contract.py",
    "check_f25_production.py",
    "core/f25_experiment_contract.py",
    "core/f25_models.py",
    "core/f25_training_contract.py",
    "core/temporal_pooling.py",
    "training/f25_executor.py",
    "scour_MATLAB/Calc.ProfileData15_05.mat",
    "scour_MATLAB/F25_Run.m",
    "scour_MATLAB/smoke_f25_contract.m",
    "scour_MATLAB/smoke_f25_solver.m",
    "scour_MATLAB/+ttbi/f25_bundle_source_binding.m",
    "scour_MATLAB/+ttbi/f25_damage_for_state.m",
    "scour_MATLAB/+ttbi/f25_execute_generation_state.m",
    "scour_MATLAB/+ttbi/f25_experiment_config.m",
    "scour_MATLAB/+ttbi/f25_extract_monitoring_signals.m",
    "scour_MATLAB/+ttbi/f25_generation_identity.m",
    "scour_MATLAB/+ttbi/f25_monitoring_window.m",
    "scour_MATLAB/+ttbi/f25_sample_operations.m",
    "scour_MATLAB/+ttbi/f25_scenario_catalog.m",
    "scour_MATLAB/+ttbi/f25_state_design.m",
}


class F25BundleError(RuntimeError):
    """Raised before publication when an F25 bundle boundary is not exact."""


def generated_names(experiment: str) -> tuple[str, str, str]:
    if experiment not in {"F25-R", "F25-X"}:
        raise F25BundleError("generated F25 artifact names require F25-R or F25-X")
    return (
        f"f25_training_plan.{experiment}.json",
        f"f25_bundle_manifest.{experiment}.json",
        f"README_{experiment}.md",
    )


@dataclass(frozen=True)
class SourceSnapshot:
    """Immutable source bytes and Git identities captured from one commit tree."""

    source_commit: str
    entries: tuple[str, ...]
    manifest_bytes: bytes
    blob_oids: tuple[tuple[str, str], ...]
    payloads: tuple[tuple[str, bytes], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_like(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction is not None and junction())


def _parse_source_manifest(raw: bytes) -> tuple[str, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise F25BundleError("source manifest is not UTF-8") from exc
    if "\r" in text or not text.endswith("\n"):
        raise F25BundleError("source manifest must be LF-only with final newline")
    entries = tuple(
        line for line in text.splitlines() if line and not line.startswith("#")
    )
    if entries != tuple(sorted(entries)) or len(entries) != len(set(entries)):
        raise F25BundleError("source manifest is not sorted and unique")
    missing_contract = sorted(REQUIRED_F25_SOURCE - set(entries))
    if missing_contract:
        raise F25BundleError(
            f"source manifest lacks F25 production closure: {missing_contract}"
        )
    for name in entries:
        posix = PurePosixPath(name)
        if (
            not name
            or Path(name).is_absolute()
            or posix.is_absolute()
            or name != posix.as_posix()
            or any(part in {"", ".", ".."} for part in posix.parts)
            or "\\" in name
            or any(ord(character) < 32 for character in name)
        ):
            raise F25BundleError(f"unsafe source path: {name}")
    return entries


def _run_git(args: list[str], repo: Path) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise F25BundleError(f"Git command failed ({' '.join(args)}){suffix}") from exc
    return result.stdout


def _git(args: list[str], repo: Path) -> str:
    return _run_git(args, repo).decode("ascii").strip()


def _git_difference(args: list[str], repo: Path) -> bool:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=False, capture_output=True
    )
    if result.returncode not in (0, 1):
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise F25BundleError(
            f"Git difference check failed ({' '.join(args)}){suffix}"
        )
    return result.returncode == 1


def _resolve_head_commit(repo: Path) -> str:
    # Peel HEAD before consulting any checkout path. Publication is tied to this
    # exact commit tree, not to index/worktree normalization or smudge filters.
    commit = _git(["rev-parse", "--verify", "HEAD^{commit}"], repo)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise F25BundleError("could not resolve HEAD^{commit} to one full source commit")
    return commit


def _tree_blob(repo: Path, commit: str, name: str) -> tuple[str, str, bytes]:
    listing = _run_git(
        ["ls-tree", "-z", "--full-tree", commit, "--", name], repo
    )
    rows = [row for row in listing.split(b"\0") if row]
    if len(rows) != 1:
        raise F25BundleError(f"source path is absent or ambiguous in commit: {name}")
    try:
        metadata, encoded_name = rows[0].split(b"\t", 1)
        mode_raw, kind_raw, oid_raw = metadata.split(b" ", 2)
        listed_name = encoded_name.decode("utf-8")
        mode = mode_raw.decode("ascii")
        kind = kind_raw.decode("ascii")
        oid = oid_raw.decode("ascii")
    except (UnicodeDecodeError, ValueError) as exc:
        raise F25BundleError(f"invalid Git tree entry for source path: {name}") from exc
    if listed_name != name:
        raise F25BundleError(f"Git tree path mismatch: requested {name}, got {listed_name}")
    if kind != "blob" or mode not in REGULAR_BLOB_MODES:
        raise F25BundleError(
            f"source is not a regular 100644/100755 blob in commit: {name} "
            f"({mode} {kind})"
        )
    return mode, oid, _run_git(["cat-file", "blob", oid], repo)


def _commit_source_snapshot(repo: Path, source_commit: str) -> SourceSnapshot:
    _mode, _manifest_oid, manifest_bytes = _tree_blob(
        repo, source_commit, SOURCE_MANIFEST
    )
    entries = _parse_source_manifest(manifest_bytes)
    blobs: list[tuple[str, str]] = []
    payloads: list[tuple[str, bytes]] = []
    for name in entries:
        _mode, oid, payload = _tree_blob(repo, source_commit, name)
        blobs.append((name, oid))
        payloads.append((name, payload))
    return SourceSnapshot(
        source_commit=source_commit,
        entries=entries,
        manifest_bytes=manifest_bytes,
        blob_oids=tuple(blobs),
        payloads=tuple(payloads),
    )


def _development_source_payloads(
    repo: Path,
) -> tuple[tuple[str, ...], bytes, tuple[tuple[str, bytes], ...]]:
    """Read a checkout for check-only diagnostics, never for publication."""

    manifest_path = repo / SOURCE_MANIFEST
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise F25BundleError(f"cannot read development source manifest: {exc}") from exc
    entries = _parse_source_manifest(manifest_bytes)
    payloads: list[tuple[str, bytes]] = []
    for name in entries:
        path = repo / name
        try:
            info = path.lstat()
        except OSError as exc:
            raise F25BundleError(f"cannot inspect development source: {name}") from exc
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise F25BundleError(f"source is not one regular unlinked file: {name}")
        payloads.append((name, path.read_bytes()))
    return entries, manifest_bytes, tuple(payloads)


def _operator_readme(experiment: str, source_commit: str) -> bytes:
    plan_name, manifest_name, readme_name = generated_names(experiment)
    other = "F25-X" if experiment == "F25-R" else "F25-R"
    other_plan, other_manifest, other_readme = generated_names(other)
    text = f"""# {experiment} dispatch bundle

This archive is one authenticated half of the complete F25-R/F25-X comparison
block from source commit `{source_commit}`. Distinct jobs may run on different
locally qualified PCs, GPU models, or compatible numeric stacks, but one
in-progress job must remain on the same host. The registered science jobs use
one or two channels; the capacity gate executes the two worst registered RAW
pair envelopes, with batch 48, five layers, no pooling, and kernels 2/5.

## Prepare byte-identical workspaces

Retain the SHA-256 values printed by `build_f25_bundles.py` and transfer both
`bundle_F25-R.zip` and `bundle_F25-X.zip`. First extract either archive to a
temporary bootstrap directory and use its builder to verify the pair:

```powershell
<qualified-python> -B <bootstrap>\\build_f25_bundles.py --verify-pair `
  <absolute-bundle_F25-R.zip> <absolute-bundle_F25-X.zip> `
  --expected-f25-r-sha256 <retained-R-SHA-256> `
  --expected-f25-x-sha256 <retained-X-SHA-256> `
  --expected-source-commit <retained-clean-A-commit>
```

Only after that command passes, co-extract both archives into one new empty
workspace on every participating PC. Common source paths are byte-identical by
the pair verifier; the experiment-qualified evidence files do not collide.
Never overlay an old workspace, cache, result, receipt, or generated dataset.
Preserve all six generated evidence files: `{plan_name}`, `{manifest_name}`,
`{readme_name}`, `{other_plan}`, `{other_manifest}`, and `{other_readme}`.

```powershell
$workspace = '<absolute-new-F25-workspace>'
New-Item -ItemType Directory -Path $workspace | Out-Null
Expand-Archive -LiteralPath <absolute-bundle_F25-R.zip> `
  -DestinationPath $workspace
Expand-Archive -LiteralPath <absolute-bundle_F25-X.zip> `
  -DestinationPath $workspace -Force
```

`-Force` is permitted only in that newly created empty workspace and only
after `--verify-pair` proves every common path has identical bytes.

## Capacity, one shared generation, and ordered jobs

Use any supported 64-bit CPython/MATLAB installation that passes the local
capability, physics and capacity checks. Exact Python, package, CUDA, MATLAB,
Update and toolbox versions are recorded as provenance; they need not match the
known-good reference. Select the target GPU if necessary. The executor sets the
registered deterministic `CUBLAS_WORKSPACE_CONFIG=:4096:8` when absent.
Install the direct dependencies from `requirements-portable.txt`; the pinned
py313/cu128 requirements file is only an optional known-good fallback.

On **every PC that will execute training jobs**, run the genuine capacity gate
without `--contract-only`. Each host writes a collision-free receipt below
`f25_artifacts/capacity_receipts/<runtime-sha256>/<source-root-sha256>.json`;
never rename a receipt or copy one into another runtime directory. Choose one
PC as the consolidation coordinator and generate the common dataset there
exactly once:

```powershell
<qualified-python> -B check_f25_capacity.py
matlab -batch "cd('<workspace>/scour_MATLAB'); F25_Run('F25-R',4)"
```

After generation completes, stop all F25 processes and distribute the exact
dataset and coordinator receipts to every worker with the authenticated,
append-only merge. `SOURCE_WORKSPACE` and `DESTINATION_WORKSPACE` are absolute
workspace roots, not their `f25_artifacts` subdirectories:

```powershell
<qualified-python> -B build_f25_bundles.py --merge-artifacts `
  <absolute-coordinator-workspace> <absolute-worker-workspace>
```

The merge first hashes every collision and copies nothing if any existing path
has different bytes. It then copies only missing regular files and verifies
them again. It never overwrites SQLite databases, winners, manifests, results,
capacity receipts, or dataset states.

Assign each job ID to exactly one PC and always invoke the executor as a module
from that PC's workspace root. Execute the dependency rounds below in order;
jobs within one round may be distributed, but no later round may start early:

1. F25-R HPO jobs.
2. F25-R report jobs.
3. F25-X tier 01 report jobs.
4. F25-X tier 02 HPO jobs.
5. F25-X tier 02 report jobs.
6. F25-X tier 03 report jobs.

At the end of **every** round, stop all F25 processes. Merge each worker into
the coordinator, one at a time, then merge the coordinator back into every
worker before starting the next round:

```powershell
<qualified-python> -B build_f25_bundles.py --merge-artifacts `
  <absolute-worker-workspace> <absolute-coordinator-workspace>
<qualified-python> -B build_f25_bundles.py --merge-artifacts `
  <absolute-coordinator-workspace> <absolute-worker-workspace>
```

This barrier is mandatory because prior-tier completion and frozen anchor
winners are authenticated from local artifact paths. A divergent collision
means two PCs produced different bytes for one logical path: stop and audit;
never use Explorer, `Copy-Item -Force`, or archive extraction to resolve it.
One in-progress job must never be split or migrated between PCs.

For a single-PC execution, run every job in generated-plan order:

```powershell
$r = Get-Content f25_training_plan.F25-R.json -Raw | ConvertFrom-Json
foreach ($job in $r.jobs) {{
  <qualified-python> -B -m training.f25_executor run-job `
    --experiment F25-R --job-id $job.job_id
}}
$x = Get-Content f25_training_plan.F25-X.json -Raw | ConvertFrom-Json
foreach ($job in $x.jobs) {{
  <qualified-python> -B -m training.f25_executor run-job `
    --experiment F25-X --job-id $job.job_id
}}
```

Stop on any nonzero exit. Preserve the fully consolidated `f25_artifacts` with
the archive hashes and all per-PC capacity receipts as run evidence. Each job's
create-once `run_record.json` binds its exact runtime and capacity receipt.
"""
    return text.encode("utf-8")


def _worktree_filtered_blob_oid(repo: Path, name: str) -> str:
    path = repo / name
    try:
        info = path.lstat()
    except OSError as exc:
        raise F25BundleError(f"cannot inspect required working source: {name}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise F25BundleError(f"required working source is not a regular file: {name}")
    # --path applies the repository's clean/EOL policy. Thus a clean CRLF
    # checkout is compared with its LF blob identity, while assume-unchanged
    # cannot conceal a substantive builder or manifest edit.
    return _git(["hash-object", f"--path={name}", "--", name], repo)


def _require_publication_boundary(repo: Path, snapshot: SourceSnapshot) -> None:
    # Inspect the two files that define the archive boundary directly before
    # consulting porcelain. Git index hints such as assume-unchanged must not
    # conceal a modified builder or source manifest.
    committed_oids = dict(snapshot.blob_oids)
    for name in (BUNDLE_BUILDER, SOURCE_MANIFEST):
        expected = committed_oids.get(name)
        if expected is None:
            raise F25BundleError(f"publication boundary omits required source: {name}")
        observed = _worktree_filtered_blob_oid(repo, name)
        if observed != expected:
            raise F25BundleError(
                f"working {name} does not match HEAD after clean filtering"
            )

    status = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        repo,
    )
    # Reconcile a possible CRLF-only porcelain refresh, but inspect the whole
    # repository rather than only bundle-manifest entries.
    if status and (
        _git_difference(["diff", "--quiet", "--no-ext-diff"], repo)
        or _git_difference(
            ["diff", "--cached", "--quiet", "--no-ext-diff", "HEAD"],
            repo,
        )
        or bool(
            _run_git(
                ["ls-files", "--others", "--exclude-standard", "-z"],
                repo,
            )
        )
    ):
        raise F25BundleError(
            "F25 bundles require one globally clean reviewed source commit; "
            "dirty paths remain"
        )


def _payloads(
    experiment: str,
    source_payloads: tuple[tuple[str, bytes], ...],
    manifest_bytes: bytes,
    source_commit: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    names = tuple(name for name, _payload in source_payloads)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise F25BundleError("immutable source payloads are not sorted and unique")
    payloads = dict(source_payloads)
    source_digests = [
        {"path": name, "sha256": _sha256_bytes(payloads[name])} for name in names
    ]
    plan = build_training_plan(experiment)
    plan_bytes = (
        json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    plan_name, manifest_name, readme_name = generated_names(experiment)
    readme_bytes = _operator_readme(experiment, source_commit)
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "experiment_id": experiment,
        "bundle_name": plan["bundle_name"],
        "source_commit": source_commit,
        "source_manifest_sha256": _sha256_bytes(manifest_bytes),
        "source_file_count": len(names),
        "source_files": source_digests,
        "source_root_sha256": canonical_json_sha256(source_digests),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "training_plan_sha256": plan["plan_sha256"],
        "generated_artifacts": {
            "training_plan": {
                "path": plan_name,
                "sha256": _sha256_bytes(plan_bytes),
            },
            "operator_readme": {
                "path": readme_name,
                "sha256": _sha256_bytes(readme_bytes),
            },
        },
        "shared_generation_root": plan["shared_generation_root"],
        "artifact_roots": {
            "manifest": plan["manifest_root"],
            "cache": plan["cache_root"],
            "results": plan["results_root"],
        },
    }
    bundle["bundle_manifest_sha256"] = canonical_json_sha256(bundle)
    bundle_bytes = (
        json.dumps(bundle, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    payloads[plan_name] = plan_bytes
    payloads[manifest_name] = bundle_bytes
    payloads[readme_name] = readme_bytes
    return payloads, bundle


def _write_zip(path: Path, payloads: dict[str, bytes]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            info.create_system = 3
            archive.writestr(info, payloads[name])
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None or archive.namelist() != sorted(payloads):
            raise F25BundleError(f"ZIP verification failed: {path.name}")
        for name, expected in payloads.items():
            if archive.read(name) != expected:
                raise F25BundleError(f"ZIP payload mismatch: {path.name}:{name}")


def _canonical_archive_path(path: Path, owner: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise F25BundleError(f"F25 archive is missing or linked: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise F25BundleError(f"F25 archive cannot be resolved: {path}") from exc
    if os.path.normcase(str(path)) != os.path.normcase(str(resolved)):
        raise F25BundleError(f"{owner} archive path is aliased or noncanonical")
    return resolved


def _expected_archive_sha256(value: str, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise F25BundleError(f"{owner} expected archive SHA-256 is malformed")
    return value


def _expected_source_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise F25BundleError("expected clean-A source commit is malformed")
    return value


def _verify_one_archive(path: Path, experiment: str) -> dict[str, Any]:
    plan_name, manifest_name, readme_name = generated_names(experiment)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if archive.testzip() is not None or names != sorted(names):
                raise F25BundleError(f"F25 archive fails ZIP integrity: {path}")
            if len(names) != len(set(names)):
                raise F25BundleError(f"F25 archive has duplicate paths: {path}")
            try:
                manifest = json.loads(archive.read(manifest_name).decode("ascii"))
                plan = json.loads(archive.read(plan_name).decode("ascii"))
                readme = archive.read(readme_name)
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise F25BundleError(
                    f"F25 archive generated evidence is malformed: {path}"
                ) from exc
            if not isinstance(manifest, dict):
                raise F25BundleError("F25 bundle manifest is not an object")
            recorded_manifest_sha = manifest.get("bundle_manifest_sha256")
            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("bundle_manifest_sha256", None)
            if (
                manifest.get("schema") != BUNDLE_SCHEMA
                or manifest.get("experiment_id") != experiment
                or recorded_manifest_sha != canonical_json_sha256(unsigned_manifest)
            ):
                raise F25BundleError("F25 bundle manifest identity does not verify")
            validate_training_plan(plan)
            if (
                plan.get("experiment_id") != experiment
                or plan.get("plan_sha256") != manifest.get("training_plan_sha256")
            ):
                raise F25BundleError("F25 training plan does not bind its bundle")
            if manifest.get("generated_artifacts") != {
                "training_plan": {
                    "path": plan_name,
                    "sha256": _sha256_bytes(archive.read(plan_name)),
                },
                "operator_readme": {
                    "path": readme_name,
                    "sha256": _sha256_bytes(readme),
                },
            }:
                raise F25BundleError(
                    "F25 generated plan/README bytes do not bind the manifest"
                )
            source_files = manifest.get("source_files")
            if not isinstance(source_files, list) or len(source_files) != manifest.get(
                "source_file_count"
            ):
                raise F25BundleError("F25 bundle source inventory is malformed")
            source_names: list[str] = []
            for row in source_files:
                if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                    raise F25BundleError("F25 bundle source row is malformed")
                name = row["path"]
                digest = row["sha256"]
                if not isinstance(name, str) or not isinstance(digest, str):
                    raise F25BundleError("F25 bundle source identity is malformed")
                try:
                    payload = archive.read(name)
                except KeyError as exc:
                    raise F25BundleError(
                        f"F25 archive omits source payload: {name}"
                    ) from exc
                if len(digest) != 64 or _sha256_bytes(payload) != digest:
                    raise F25BundleError(f"F25 source payload fails SHA-256: {name}")
                source_names.append(name)
            if source_names != sorted(set(source_names)):
                raise F25BundleError("F25 bundle source paths are not canonical")
            if names != sorted(source_names + [plan_name, manifest_name, readme_name]):
                raise F25BundleError("F25 archive payload closure is not exact")
            if manifest.get("source_root_sha256") != canonical_json_sha256(
                source_files
            ):
                raise F25BundleError("F25 bundle source root does not verify")
            try:
                source_manifest = archive.read(SOURCE_MANIFEST)
            except KeyError as exc:
                raise F25BundleError("F25 archive omits its source manifest") from exc
            manifest_entries = _parse_source_manifest(source_manifest)
            if tuple(source_names) != manifest_entries:
                raise F25BundleError(
                    "F25 archive inventory differs from its safe source manifest"
                )
            if manifest.get("source_manifest_sha256") != _sha256_bytes(
                source_manifest
            ):
                raise F25BundleError("F25 source-manifest bytes do not verify")
            if (
                not readme.endswith(b"\n")
                or manifest.get("source_commit", "").encode("ascii") not in readme
            ):
                raise F25BundleError("F25 operator README does not bind the commit")
            return manifest
    except (OSError, zipfile.BadZipFile) as exc:
        raise F25BundleError(f"cannot read F25 archive: {path}") from exc


def verify_pair_archives(
    f25_r: Path,
    f25_x: Path,
    *,
    expected_f25_r_sha256: str,
    expected_f25_x_sha256: str,
    expected_source_commit: str,
) -> None:
    """Verify that two transferred archives can form one shared workspace."""

    r_path = _canonical_archive_path(f25_r, "F25-R")
    x_path = _canonical_archive_path(f25_x, "F25-X")
    expected_r = _expected_archive_sha256(expected_f25_r_sha256, "F25-R")
    expected_x = _expected_archive_sha256(expected_f25_x_sha256, "F25-X")
    expected_commit = _expected_source_commit(expected_source_commit)
    if _sha256_bytes(r_path.read_bytes()) != expected_r:
        raise F25BundleError("F25-R archive differs from its retained SHA-256")
    if _sha256_bytes(x_path.read_bytes()) != expected_x:
        raise F25BundleError("F25-X archive differs from its retained SHA-256")
    r_manifest = _verify_one_archive(r_path, "F25-R")
    x_manifest = _verify_one_archive(x_path, "F25-X")
    if (
        r_manifest.get("source_commit") != expected_commit
        or x_manifest.get("source_commit") != expected_commit
    ):
        raise F25BundleError("F25 archives do not bind the retained clean-A commit")
    for field in ("source_commit", "source_files", "source_root_sha256"):
        if r_manifest[field] != x_manifest[field]:
            raise F25BundleError(f"F25-R/F25-X archive {field} differs")
    print(
        "PASS F25 archive pair: byte-identical shared source closure, "
        "experiment-qualified plans/manifests/READMEs"
    )


def _canonical_workspace(path: Path, owner: str) -> Path:
    if not path.is_absolute() or _link_like(path) or not path.is_dir():
        raise F25BundleError(
            f"{owner} workspace must be one absolute, unlinked directory"
        )
    resolved = path.resolve(strict=True)
    if os.path.normcase(str(path)) != os.path.normcase(str(resolved)):
        raise F25BundleError(f"{owner} workspace path is aliased/noncanonical")
    return resolved


def _require_same_extracted_bundle_pair(source: Path, destination: Path) -> None:
    evidence = [SOURCE_MANIFEST]
    evidence.extend(
        name
        for experiment in ("F25-R", "F25-X")
        for name in generated_names(experiment)
    )
    for name in evidence:
        source_path = source / name
        destination_path = destination / name
        for path, owner in (
            (source_path, "source"),
            (destination_path, "destination"),
        ):
            try:
                info = path.lstat()
            except OSError as exc:
                raise F25BundleError(
                    f"{owner} workspace lacks paired F25 evidence: {name}"
                ) from exc
            if (
                not stat.S_ISREG(info.st_mode)
                or _link_like(path)
                or info.st_nlink != 1
            ):
                raise F25BundleError(
                    f"{owner} F25 evidence is linked/nonregular: {name}"
                )
        if _sha256_path(source_path) != _sha256_path(destination_path):
            raise F25BundleError(
                f"F25 workspaces derive from different bundle bytes: {name}"
            )


def _regular_artifact_inventory(root: Path) -> dict[str, tuple[Path, str]]:
    if not root.exists():
        return {}
    if _link_like(root) or not root.is_dir():
        raise F25BundleError("f25_artifacts is linked or non-directory")
    inventory: dict[str, tuple[Path, str]] = {}
    casefolded: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        parent = Path(directory)
        for child_name in directory_names:
            child = parent / child_name
            info = child.lstat()
            if _link_like(child) or not stat.S_ISDIR(info.st_mode):
                raise F25BundleError(
                    f"artifact tree contains a linked/non-directory path: {child}"
                )
        for child_name in file_names:
            child = parent / child_name
            info = child.lstat()
            if (
                _link_like(child)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise F25BundleError(
                    f"artifact tree contains a linked/nonregular file: {child}"
                )
            relative = child.relative_to(root).as_posix()
            folded = relative.casefold()
            if folded in casefolded:
                raise F25BundleError(
                    f"artifact tree has a case-colliding path: {relative}"
                )
            casefolded.add(folded)
            inventory[relative] = (child, _sha256_path(child))
    return inventory


def merge_artifact_tree(source_workspace: Path, destination_workspace: Path) -> int:
    """Append one stopped worker's artifacts without overwriting any byte."""

    source = _canonical_workspace(source_workspace, "source")
    destination = _canonical_workspace(destination_workspace, "destination")
    if os.path.normcase(str(source)) == os.path.normcase(str(destination)):
        raise F25BundleError("artifact merge source and destination are identical")
    _require_same_extracted_bundle_pair(source, destination)
    source_root = source / "f25_artifacts"
    destination_root = destination / "f25_artifacts"
    source_inventory = _regular_artifact_inventory(source_root)
    destination_inventory = _regular_artifact_inventory(destination_root)
    divergent = sorted(
        name
        for name, (_path, digest) in source_inventory.items()
        if name in destination_inventory
        and destination_inventory[name][1] != digest
    )
    if divergent:
        raise F25BundleError(
            "artifact merge found divergent bytes and copied nothing: "
            f"{divergent}"
        )
    missing = sorted(set(source_inventory) - set(destination_inventory))
    lock_path = destination / ".f25-artifact-merge.lock"
    lock_owned = False
    try:
        with lock_path.open("x", encoding="ascii") as lock:
            lock_owned = True
            lock.write(f"source={source.as_posix()}\n")
            lock.flush()
            os.fsync(lock.fileno())
        destination_root.mkdir(parents=False, exist_ok=True)
        for name in missing:
            source_path, expected_sha = source_inventory[name]
            target = destination_root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.f25-merge-{os.getpid()}.tmp"
            )
            if temporary.exists():
                raise F25BundleError(
                    f"stale artifact-merge temporary exists: {temporary}"
                )
            try:
                with source_path.open("rb") as source_handle, temporary.open(
                    "xb"
                ) as target_handle:
                    while True:
                        chunk = source_handle.read(1 << 20)
                        if not chunk:
                            break
                        target_handle.write(chunk)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                if (
                    _sha256_path(temporary) != expected_sha
                    or _sha256_path(source_path) != expected_sha
                ):
                    raise F25BundleError(
                        f"artifact changed while being copied: {name}"
                    )
                if target.exists():
                    if _sha256_path(target) != expected_sha:
                        raise F25BundleError(
                            f"artifact appeared with divergent bytes: {name}"
                        )
                    temporary.unlink()
                else:
                    os.replace(temporary, target)
            except BaseException:
                if temporary.exists():
                    temporary.unlink()
                raise
        confirmed = _regular_artifact_inventory(destination_root)
        if any(
            name not in confirmed or confirmed[name][1] != expected_sha
            for name, (_path, expected_sha) in source_inventory.items()
        ):
            raise F25BundleError("artifact merge failed post-copy verification")
    finally:
        if lock_owned and lock_path.exists():
            lock_path.unlink()
    print(
        f"PASS F25 artifact merge: {len(missing)} new files, "
        f"{len(source_inventory) - len(missing)} byte-identical files"
    )
    return len(missing)


def check(repo: Path = REPO) -> None:
    """Exercise bundle structure from a checkout without publication claims."""

    repo = repo.resolve(strict=True)
    entries, manifest_bytes, source_payloads = _development_source_payloads(repo)
    bundles: list[dict[str, Any]] = []
    for experiment in ("F25-R", "F25-X"):
        payloads, bundle = _payloads(
            experiment, source_payloads, manifest_bytes, "0" * 40
        )
        if len(payloads) != len(entries) + len(generated_names(experiment)):
            raise F25BundleError("F25 bundle payload closure is incomplete")
        if bundle["experiment_id"] != experiment:
            raise F25BundleError("F25 bundle experiment identity drifted")
        bundles.append(bundle)
    if (
        bundles[0]["source_root_sha256"] != bundles[1]["source_root_sha256"]
        or bundles[0]["source_files"] != bundles[1]["source_files"]
    ):
        raise F25BundleError("F25-R/F25-X source closures differ")
    print(
        f"PASS F25 development-only bundle contract: {len(entries)} reviewed "
        "checkout sources + three experiment-qualified generated artifacts per "
        "archive; not publication evidence"
    )


def build(repo: Path = REPO) -> tuple[Path, Path]:
    """Publish two bundles whose source payloads come only from HEAD blobs."""

    repo = repo.resolve(strict=True)
    source_commit = _resolve_head_commit(repo)
    snapshot = _commit_source_snapshot(repo, source_commit)
    _require_publication_boundary(repo, snapshot)
    staged: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".f25-bundles-", dir=repo) as raw:
        staging = Path(raw)
        for experiment in ("F25-R", "F25-X"):
            payloads, bundle = _payloads(
                experiment,
                snapshot.payloads,
                snapshot.manifest_bytes,
                snapshot.source_commit,
            )
            temporary = staging / bundle["bundle_name"]
            _write_zip(temporary, payloads)
            staged.append((temporary, repo / bundle["bundle_name"]))
        for temporary, target in staged:
            os.replace(temporary, target)
    for _temporary, target in staged:
        print(f"{target.name}: {_sha256_bytes(target.read_bytes())}")
    return tuple(target for _temporary, target in staged)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument(
        "--verify-pair",
        nargs=2,
        type=Path,
        metavar=("F25_R_ZIP", "F25_X_ZIP"),
    )
    mode.add_argument(
        "--merge-artifacts",
        nargs=2,
        type=Path,
        metavar=("SOURCE_WORKSPACE", "DESTINATION_WORKSPACE"),
    )
    parser.add_argument("--expected-f25-r-sha256")
    parser.add_argument("--expected-f25-x-sha256")
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args()
    if args.check_only:
        check()
    elif args.verify_pair:
        if (
            not args.expected_f25_r_sha256
            or not args.expected_f25_x_sha256
            or not args.expected_source_commit
        ):
            parser.error(
                "--verify-pair requires both retained archive SHA-256 values "
                "and --expected-source-commit"
            )
        verify_pair_archives(
            args.verify_pair[0],
            args.verify_pair[1],
            expected_f25_r_sha256=args.expected_f25_r_sha256,
            expected_f25_x_sha256=args.expected_f25_x_sha256,
            expected_source_commit=args.expected_source_commit,
        )
    elif args.merge_artifacts:
        if (
            args.expected_f25_r_sha256
            or args.expected_f25_x_sha256
            or args.expected_source_commit
        ):
            parser.error(
                "expected archive identities are valid only with --verify-pair"
            )
        merge_artifact_tree(
            args.merge_artifacts[0], args.merge_artifacts[1]
        )
    elif (
        args.expected_f25_r_sha256
        or args.expected_f25_x_sha256
        or args.expected_source_commit
    ):
        parser.error("expected F25 identities are valid only with --verify-pair")
    else:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
