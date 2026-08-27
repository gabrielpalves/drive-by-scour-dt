"""Acceptance and mutation checks for the production F25 integration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import zipfile

import torch
from torch import nn

import build_f25_bundles
import check_f25_capacity
from core.f25_experiment_contract import (
    CHANNELS,
    ELIGIBLE_SENSOR_INDICES,
    F25_X_CHANNELS,
    build_contract,
)
from core.f25_models import F25CNN, build_f25_model
from core.f25_training_contract import (
    F25TrainingContractError,
    build_training_plan,
    canonical_json_sha256,
    configuration_id,
    validate_training_plan,
)
import training.f25_executor as f25_executor


ROOT = Path(__file__).resolve().parent


def _rejects(action, label: str) -> None:
    try:
        action()
    except (F25TrainingContractError, ValueError):
        return
    raise AssertionError(f"mutation was accepted: {label}")


def _capacity_receipt_smoke() -> None:
    cases = check_f25_capacity.capacity_contract_cases()
    source_root = SimpleNamespace(sha256="c" * 64, file_count=123)

    def runtime(environment_character: str, compatibility_character: str) -> dict:
        return {
            "environment_lock_sha256": "a" * 64,
            "execution_environment_sha256": environment_character * 64,
            "execution_compatibility_sha256": compatibility_character * 64,
            "execution_environment_descriptor": {
                "host": {"hostname": f"pc-{environment_character}"},
                "accelerator": {"total_memory_bytes": 1_000_000},
            },
            "execution_compatibility_descriptor": {
                "gpu": f"fixture-{compatibility_character}"
            },
        }

    def receipt(runtime_value: dict) -> dict:
        return {
            "schema": check_f25_capacity.SCHEMA,
            "accepted": True,
            "contract_only": False,
            "capacity_receipt_address_schema": (
                check_f25_capacity.CAPACITY_RECEIPT_ADDRESS_SCHEMA
            ),
            "f25_contract_sha256": build_contract()["contract_sha256"],
            "case_contract_sha256": canonical_json_sha256(cases),
            "cases": cases,
            "environment_lock_sha256": runtime_value[
                "environment_lock_sha256"
            ],
            "execution_environment_sha256": runtime_value[
                "execution_environment_sha256"
            ],
            "execution_compatibility_sha256": runtime_value[
                "execution_compatibility_sha256"
            ],
            "execution_environment_descriptor": runtime_value[
                "execution_environment_descriptor"
            ],
            "execution_compatibility_descriptor": runtime_value[
                "execution_compatibility_descriptor"
            ],
            "python_runtime_source_sha256": source_root.sha256,
            "python_runtime_source_file_count": source_root.file_count,
            "device_total_memory_bytes": 1_000_000,
            "measurements": [
                {
                    "case_id": row["case_id"],
                    "loss": 1.0,
                    "peak_allocated_bytes": 100_000,
                    "peak_reserved_bytes": 200_000,
                }
                for row in cases
            ],
        }

    def signed(value: dict) -> dict:
        value = copy.deepcopy(value)
        value["receipt_sha256"] = canonical_json_sha256(value)
        return value

    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="f25-capacity-receipt-") as raw:
        root = Path(raw)
        runtime_a = runtime("e", "b")
        runtime_b = runtime("f", "9")
        base = receipt(runtime_a)
        base_b = receipt(runtime_b)
        receipt_path = check_f25_capacity.capacity_receipt_path(
            root,
            execution_environment_sha256_value=(
                runtime_a["execution_environment_sha256"]
            ),
            python_runtime_source_sha256=source_root.sha256,
        )
        receipt_path_b = check_f25_capacity.capacity_receipt_path(
            root,
            execution_environment_sha256_value=(
                runtime_b["execution_environment_sha256"]
            ),
            python_runtime_source_sha256=source_root.sha256,
        )
        original_repo = f25_executor.REPO
        f25_executor.REPO = root
        try:
            # The former singleton must not authorize this PC. Another PC's
            # valid receipt must also not substitute for the current runtime.
            write(
                root / "f25_artifacts" / "f25_capacity_receipt.json",
                signed(base),
            )
            try:
                f25_executor._require_capacity_receipt(runtime_a, source_root)
            except f25_executor.F25ExecutionError:
                pass
            else:
                raise AssertionError("legacy singleton capacity receipt was used")
            write(receipt_path_b, signed(base_b))
            try:
                f25_executor._require_capacity_receipt(runtime_a, source_root)
            except f25_executor.F25ExecutionError:
                pass
            else:
                raise AssertionError("another PC's capacity receipt was used")

            write(receipt_path, signed(base))
            accepted_a = f25_executor._require_capacity_receipt(
                runtime_a, source_root
            )
            accepted_b = f25_executor._require_capacity_receipt(
                runtime_b, source_root
            )
            assert receipt_path != receipt_path_b
            assert receipt_path.is_file() and receipt_path_b.is_file()
            assert accepted_a["execution_environment_sha256"] == "e" * 64
            assert accepted_b["execution_environment_sha256"] == "f" * 64
            mutations = []
            missing = copy.deepcopy(base)
            missing["cases"].pop()
            missing["case_contract_sha256"] = canonical_json_sha256(
                missing["cases"]
            )
            mutations.append((missing, "missing registered pair-envelope case"))
            extra_full_array = copy.deepcopy(base)
            extra_case = copy.deepcopy(extra_full_array["cases"][-1])
            extra_case["case_id"] = "RAW-full-array-unregistered"
            extra_case["input_shape"][1] = len(CHANNELS)
            extra_full_array["cases"].append(extra_case)
            extra_full_array["case_contract_sha256"] = canonical_json_sha256(
                extra_full_array["cases"]
            )
            extra_measurement = copy.deepcopy(
                extra_full_array["measurements"][-1]
            )
            extra_measurement["case_id"] = extra_case["case_id"]
            extra_full_array["measurements"].append(extra_measurement)
            mutations.append(
                (extra_full_array, "unregistered full-array capacity case")
            )
            reordered = copy.deepcopy(base)
            reordered["cases"] = list(reversed(reordered["cases"]))
            reordered["case_contract_sha256"] = canonical_json_sha256(
                reordered["cases"]
            )
            mutations.append((reordered, "reordered pair-envelope cases"))
            shape = copy.deepcopy(base)
            shape["cases"][0]["input_shape"][1] = 1
            shape["case_contract_sha256"] = canonical_json_sha256(shape["cases"])
            mutations.append((shape, "pair-envelope channel count changed"))
            stale_digest = copy.deepcopy(base)
            stale_digest["case_contract_sha256"] = "d" * 64
            mutations.append((stale_digest, "stale case-contract digest"))
            legacy_schema = copy.deepcopy(base)
            legacy_schema["schema"] = "f25-capacity-preflight-v2"
            mutations.append((legacy_schema, "legacy full-array capacity schema"))
            contract_only = copy.deepcopy(base)
            contract_only["contract_only"] = True
            mutations.append((contract_only, "contract-only receipt"))
            foreign_environment = copy.deepcopy(base)
            foreign_environment["execution_environment_sha256"] = "f" * 64
            mutations.append((foreign_environment, "foreign target GPU/runtime"))
            foreign_source_count = copy.deepcopy(base)
            foreign_source_count["python_runtime_source_file_count"] += 1
            mutations.append((foreign_source_count, "foreign source inventory"))
            zero_memory = copy.deepcopy(base)
            zero_memory["measurements"][1]["peak_allocated_bytes"] = 0
            mutations.append((zero_memory, "zero-compute pair-envelope case"))
            for mutation, label in mutations:
                write(receipt_path, signed(mutation))
                try:
                    f25_executor._require_capacity_receipt(
                        runtime_a, source_root
                    )
                except f25_executor.F25ExecutionError:
                    continue
                raise AssertionError(f"capacity mutation was accepted: {label}")
        finally:
            f25_executor.REPO = original_repo


def _run_record_runtime_smoke() -> None:
    source_root = SimpleNamespace(sha256="c" * 64, file_count=123)
    runtime_a = {
        "execution_environment_sha256": "e" * 64,
        "host": "PC-A",
    }
    runtime_b = {
        "execution_environment_sha256": "f" * 64,
        "host": "PC-B",
    }
    capacity_a = {"receipt_sha256": "1" * 64}
    capacity_b = {"receipt_sha256": "2" * 64}
    block = {"receipt_sha256": "3" * 64}
    job = {
        "job_id": "fixture-job",
        "manifest_path": "f25_artifacts/fixture-job",
    }
    data_binding = {"dataset": "fixture"}
    with tempfile.TemporaryDirectory(prefix="f25-run-record-") as raw:
        root = Path(raw)
        original_repo = f25_executor.REPO
        f25_executor.REPO = root
        try:
            first = f25_executor._publish_job_record(
                job,
                runtime_a,
                block,
                capacity_a,
                source_root,
                data_binding,
            )
            repeated = f25_executor._publish_job_record(
                job,
                runtime_a,
                block,
                capacity_a,
                source_root,
                data_binding,
            )
            assert first == repeated
            assert first["schema"] == "f25-training-run-record-v2"
            assert (
                first["capacity_receipt_binding"]["receipt_sha256"]
                == "1" * 64
            )
            for changed_runtime, changed_capacity, label in (
                (runtime_b, capacity_b, "runtime migration"),
                (runtime_a, capacity_b, "capacity receipt substitution"),
            ):
                try:
                    f25_executor._publish_job_record(
                        job,
                        changed_runtime,
                        block,
                        changed_capacity,
                        source_root,
                        data_binding,
                    )
                except f25_executor.F25ExecutionError:
                    continue
                raise AssertionError(f"run record accepted {label}")
        finally:
            f25_executor.REPO = original_repo


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def _fixture_repository(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "F25 fixture")
    _git(repo, "config", "user.email", "f25-fixture@example.invalid")
    _git(repo, "config", "core.autocrlf", "true")
    (repo / ".gitattributes").write_bytes(
        b"* text eol=crlf\n*.mat -text\n.gitattributes text eol=lf\n"
    )

    entries = tuple(
        sorted(build_f25_bundles.REQUIRED_F25_SOURCE | {"fixture_payload.txt"})
    )
    for name in entries:
        if name == build_f25_bundles.SOURCE_MANIFEST:
            continue
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "scour_MATLAB/Calc.ProfileData15_05.mat":
            path.write_bytes((ROOT / name).read_bytes())
        else:
            path.write_bytes(
                f"fixture source: {name}\nsecond line\n".encode("utf-8")
            )
    manifest = (
        "# F25 temporary Git fixture.\n" + "\n".join(entries) + "\n"
    ).encode("utf-8")
    (repo / build_f25_bundles.SOURCE_MANIFEST).write_bytes(manifest)
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", "fixture")

    # Materialize a CRLF checkout while preserving LF commit blobs. A correct
    # publication build must package the latter and remain clean under Git's
    # configured conversion policy.
    for name in entries:
        if name == "scour_MATLAB/Calc.ProfileData15_05.mat":
            continue
        path = repo / name
        payload = path.read_bytes()
        path.write_bytes(payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert not _git(repo, "diff", "--", *entries)
    assert not _git(repo, "diff", "--cached", "HEAD", "--", *entries)
    assert b"\r\n" in (repo / build_f25_bundles.SOURCE_MANIFEST).read_bytes()
    assert b"\r\n" in (repo / build_f25_bundles.BUNDLE_BUILDER).read_bytes()
    return repo


def _assert_bundle_rejected(repo: Path, expected: str) -> None:
    try:
        build_f25_bundles.build(repo)
    except build_f25_bundles.F25BundleError as exc:
        assert expected in str(exc), (expected, str(exc))
        return
    raise AssertionError(f"F25 publication mutation was accepted: {expected}")


def _git_blob(repo: Path, commit: str, name: str) -> bytes:
    return _git(repo, "show", f"{commit}:{name}")


def bundle_commit_blob_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="f25-blob-test-") as raw:
        repo = _fixture_repository(Path(raw))
        commit = _git(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
        snapshot = build_f25_bundles._commit_source_snapshot(repo, commit)
        archives = build_f25_bundles.build(repo)
        expected_files = [
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in snapshot.payloads
        ]
        bundle_manifests = []
        for experiment, archive_path in zip(("F25-R", "F25-X"), archives):
            with zipfile.ZipFile(archive_path, "r") as archive:
                for name, payload in snapshot.payloads:
                    assert payload == _git_blob(repo, commit, name)
                    assert archive.read(name) == payload
                _plan_name, manifest_name, readme_name = (
                    build_f25_bundles.generated_names(experiment)
                )
                bundle_manifest = json.loads(
                    archive.read(manifest_name).decode("ascii")
                )
                assert bundle_manifest["source_commit"] == commit
                assert bundle_manifest["source_files"] == expected_files
                assert bundle_manifest["source_manifest_sha256"] == hashlib.sha256(
                    snapshot.manifest_bytes
                ).hexdigest()
                readme = archive.read(readme_name)
                assert commit.encode("ascii") in readme
                assert b"f25_artifacts/capacity_receipts/" in readme
                assert b"every PC that will execute training jobs" in readme
                assert b"--merge-artifacts" in readme
                assert b"copies nothing if any existing path" in readme
                assert b"end of **every** round" in readme
                assert b"never use Explorer" in readme
                assert b"never be split or migrated between PCs" in readme
                bundle_manifests.append(bundle_manifest)
        assert bundle_manifests[0]["source_files"] == bundle_manifests[1]["source_files"]
        assert (
            bundle_manifests[0]["source_root_sha256"]
            == bundle_manifests[1]["source_root_sha256"]
        )
        r_sha = hashlib.sha256(archives[0].read_bytes()).hexdigest()
        x_sha = hashlib.sha256(archives[1].read_bytes()).hexdigest()
        build_f25_bundles.verify_pair_archives(
            archives[0],
            archives[1],
            expected_f25_r_sha256=r_sha,
            expected_f25_x_sha256=x_sha,
            expected_source_commit=commit,
        )
        for mutation in (
            {
                "expected_f25_r_sha256": "0" * 64,
                "expected_f25_x_sha256": x_sha,
                "expected_source_commit": commit,
            },
            {
                "expected_f25_r_sha256": r_sha,
                "expected_f25_x_sha256": x_sha,
                "expected_source_commit": "0" * 40,
            },
        ):
            try:
                build_f25_bundles.verify_pair_archives(
                    archives[0], archives[1], **mutation
                )
            except build_f25_bundles.F25BundleError:
                pass
            else:
                raise AssertionError("F25 pair accepted a foreign trusted identity")
        workspace = Path(raw) / "paired-workspace"
        workspace.mkdir()
        for archive_path in archives:
            with zipfile.ZipFile(archive_path, "r") as archive:
                for name in archive.namelist():
                    destination = workspace / name
                    payload = archive.read(name)
                    if destination.exists():
                        assert destination.read_bytes() == payload
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(payload)
        generated = {
            name
            for experiment in ("F25-R", "F25-X")
            for name in build_f25_bundles.generated_names(experiment)
        }
        assert all((workspace / name).is_file() for name in generated)

        manifest = json.loads(
            (workspace / "f25_bundle_manifest.F25-R.json").read_text(
                encoding="ascii"
            )
        )
        matlab_rows = [
            row
            for row in manifest["source_files"]
            if row["path"].startswith("scour_MATLAB/")
        ]
        digest_lines = "\n".join(
            f"{row['path']}:{row['sha256']}" for row in matlab_rows
        )
        fixture_source_root = SimpleNamespace(
            sha256=hashlib.sha256(digest_lines.encode("utf-8")).hexdigest(),
            file_count=len(matlab_rows),
            digest_lines=digest_lines,
        )
        original_generator_source_root = f25_executor.generator_source_root
        f25_executor.generator_source_root = lambda _repo: fixture_source_root
        try:
            source_binding = f25_executor._validate_bundle_source_binding(
                workspace
            )
            assert (
                source_binding["generator_source_root_sha256"]
                == fixture_source_root.sha256
            )
            live_matlab = workspace / "scour_MATLAB" / "F25_Run.m"
            original_matlab = live_matlab.read_bytes()
            live_matlab.write_bytes(original_matlab + b"mutated after extraction\n")
            try:
                f25_executor._validate_bundle_source_binding(workspace)
            except f25_executor.F25ExecutionError as exc:
                assert "source differs from its bundle" in str(exc)
            else:
                raise AssertionError(
                    "training accepted MATLAB source edited after extraction"
                )
            live_matlab.write_bytes(original_matlab)
        finally:
            f25_executor.generator_source_root = original_generator_source_root

        worker = Path(raw) / "paired-worker"
        worker.mkdir()
        for archive_path in archives:
            with zipfile.ZipFile(archive_path, "r") as archive:
                for name in archive.namelist():
                    destination = worker / name
                    payload = archive.read(name)
                    if destination.exists():
                        assert destination.read_bytes() == payload
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(payload)
        coordinator_artifact = (
            workspace / "f25_artifacts" / "F25-R" / "manifests" / "winner.json"
        )
        coordinator_artifact.parent.mkdir(parents=True)
        coordinator_artifact.write_bytes(b"coordinator winner\n")
        worker_artifact = (
            worker / "f25_artifacts" / "F25-X" / "results" / "metrics.json"
        )
        worker_artifact.parent.mkdir(parents=True)
        worker_artifact.write_bytes(b"worker metrics\n")
        assert build_f25_bundles.merge_artifact_tree(workspace, worker) == 1
        assert worker_artifact.is_file() and coordinator_artifact.read_bytes() == (
            worker / coordinator_artifact.relative_to(workspace)
        ).read_bytes()
        assert build_f25_bundles.merge_artifact_tree(worker, workspace) == 1
        assert build_f25_bundles.merge_artifact_tree(workspace, worker) == 0
        collision_relative = Path("f25_artifacts/F25-R/results/collision.bin")
        source_collision = workspace / collision_relative
        worker_collision = worker / collision_relative
        source_collision.parent.mkdir(parents=True, exist_ok=True)
        worker_collision.parent.mkdir(parents=True, exist_ok=True)
        source_collision.write_bytes(b"source bytes")
        worker_collision.write_bytes(b"different worker bytes")
        source_only = workspace / "f25_artifacts" / "source-only-after-collision.bin"
        source_only.write_bytes(b"must not be copied")
        try:
            build_f25_bundles.merge_artifact_tree(workspace, worker)
        except build_f25_bundles.F25BundleError as exc:
            assert "divergent bytes and copied nothing" in str(exc)
        else:
            raise AssertionError("artifact merge overwrote divergent bytes")
        assert not (
            worker / "f25_artifacts" / source_only.name
        ).exists()

        dirty = repo / "core" / "f25_models.py"
        dirty.write_bytes(dirty.read_bytes() + b"substantive dirty source\r\n")
        _assert_bundle_rejected(repo, "dirty paths remain")
        _git(repo, "restore", "--worktree", "--", "core/f25_models.py")

        builder = repo / build_f25_bundles.BUNDLE_BUILDER
        _git(
            repo,
            "update-index",
            "--assume-unchanged",
            build_f25_bundles.BUNDLE_BUILDER,
        )
        builder.write_bytes(builder.read_bytes() + b"concealed builder change\r\n")
        assert not _git(
            repo,
            "status",
            "--porcelain=v1",
            "--",
            build_f25_bundles.BUNDLE_BUILDER,
        )
        _assert_bundle_rejected(repo, "does not match HEAD after clean filtering")

    with tempfile.TemporaryDirectory(prefix="f25-mode-test-") as raw:
        repo = _fixture_repository(Path(raw))
        name = "core/f25_models.py"
        oid = _git(repo, "rev-parse", f"HEAD:{name}").decode("ascii").strip()
        _git(repo, "update-index", "--add", "--cacheinfo", "120000", oid, name)
        _git(repo, "commit", "--quiet", "-m", "nonregular mode")
        _assert_bundle_rejected(repo, "not a regular 100644/100755 blob")

    with tempfile.TemporaryDirectory(prefix="f25-manifest-test-") as raw:
        repo = _fixture_repository(Path(raw))
        name = build_f25_bundles.SOURCE_MANIFEST
        manifest_lf = _git(repo, "show", f"HEAD:{name}")
        manifest_crlf = manifest_lf.replace(b"\n", b"\r\n")
        oid = _git(repo, "hash-object", "-w", "--stdin", input_bytes=manifest_crlf)
        oid_text = oid.decode("ascii").strip()
        _git(repo, "update-index", "--add", "--cacheinfo", "100644", oid_text, name)
        _git(repo, "commit", "--quiet", "-m", "CRLF manifest blob")
        _assert_bundle_rejected(repo, "source manifest must be LF-only")

    unsafe_entries = sorted(
        build_f25_bundles.REQUIRED_F25_SOURCE | {"../escape.txt"}
    )
    unsafe_manifest = ("\n".join(unsafe_entries) + "\n").encode("utf-8")
    try:
        build_f25_bundles._parse_source_manifest(unsafe_manifest)
    except build_f25_bundles.F25BundleError as exc:
        assert "unsafe source path" in str(exc)
    else:
        raise AssertionError("F25 source manifest accepted parent traversal")


def main() -> None:
    plan_r = build_training_plan("F25-R")
    plan_x = build_training_plan("F25-X")
    assert plan_r["f25_contract_sha256"] == build_contract()["contract_sha256"]
    assert len(plan_r["jobs"]) == 8
    assert len(plan_x["jobs"]) == 99
    assert sum(job["phase"] == "hpo" for job in plan_r["jobs"]) == 4
    assert sum(job["phase"] == "report" for job in plan_r["jobs"]) == 4
    assert sum(job["phase"] == "hpo" for job in plan_x["jobs"]) == 18
    assert sum(job["phase"] == "report" for job in plan_x["jobs"]) == 81
    assert sum(
        job["hpo_proposals"] * job["executions_per_proposal"]
        for job in plan_r["jobs"]
    ) == 2_000
    assert sum(
        job["hpo_proposals"] * job["executions_per_proposal"]
        for job in plan_x["jobs"]
    ) == 9_000
    assert sum(len(job["report_seeds"]) for job in plan_r["jobs"]) == 80
    assert sum(len(job["report_seeds"]) for job in plan_x["jobs"]) == 1_620
    frozen = [
        job
        for job in plan_x["jobs"]
        if job["regime"] == "frozen" and job["phase"] == "report"
    ]
    assert len(frozen) == 63
    assert all(job["anchor_configuration_id"].endswith("S01") for job in frozen)
    assert all(len(job["sensor_set"]) <= 2 for job in plan_x["jobs"])
    assert all(
        set(job["sensor_indices"]) <= set(ELIGIBLE_SENSOR_INDICES)
        for job in plan_x["jobs"]
    )
    assert {
        tuple(job["sensor_set"])
        for job in plan_x["jobs"]
        if job["tier_id"] == "F25-X-03-frozen-hp-pairs"
    } == {
        tuple(sorted((left, right)))
        for index, left in enumerate(F25_X_CHANNELS)
        for right in F25_X_CHANNELS[index + 1 :]
    }

    mutated = copy.deepcopy(plan_r)
    mutated["jobs"][0]["hpo_proposals"] = 99
    _rejects(lambda: validate_training_plan(mutated), "99 HPO proposals")
    mutated = copy.deepcopy(plan_x)
    mutated["jobs"][-1]["sensor_indices"].reverse()
    mutated.pop("plan_sha256")
    from core.f25_training_contract import canonical_json_sha256

    mutated["plan_sha256"] = canonical_json_sha256(mutated)
    _rejects(lambda: validate_training_plan(mutated), "sensor identity drift")

    mutated = copy.deepcopy(plan_x)
    proxy_job = mutated["jobs"][0]
    proxy_index = 3
    old_configuration = proxy_job["configuration_id"]
    new_configuration = configuration_id(
        proxy_job["arm_id"], (CHANNELS[proxy_index],)
    )
    proxy_job["sensor_set"] = [CHANNELS[proxy_index]]
    proxy_job["sensor_indices"] = [proxy_index]
    proxy_job["configuration_id"] = new_configuration
    proxy_job["job_id"] = proxy_job["job_id"].replace(
        old_configuration, new_configuration
    )
    proxy_job["manifest_path"] = proxy_job["manifest_path"].replace(
        old_configuration, new_configuration
    )
    proxy_job["results_path"] = proxy_job["results_path"].replace(
        old_configuration, new_configuration
    )
    mutated.pop("plan_sha256")
    mutated["plan_sha256"] = canonical_json_sha256(mutated)
    _rejects(
        lambda: validate_training_plan(mutated),
        "wheelset diagnostic proxy admitted as an F25-X sensor",
    )

    params = {
        "n_conv_layers": 2,
        "filters_l0": 32,
        "kernel_l0": 2,
        "pool_l0": False,
        "filters_l1": 48,
        "kernel_l1": 3,
        "pool_l1": False,
        "dense_units": 48,
        "batch_size": 24,
        "learning_rate": 1.0e-3,
    }
    source_model = build_f25_model(
        arm_id="PAA-CNN", in_channels=1, params=params, device="cpu"
    )
    extension_model = build_f25_model(
        arm_id="PAA-multirate", in_channels=1, params=params, device="cpu"
    )
    assert isinstance(source_model, F25CNN)
    assert isinstance(source_model.aggregation, nn.Flatten)
    assert source_model.flattened_units == 48 * 580 == 27_840
    assert extension_model.pooling_kind == "multi-rate"
    assert extension_model.flattened_units == 48 * 7
    assert source_model(torch.zeros(2, 1, 583)).shape == (2, 10)
    assert extension_model(torch.zeros(2, 1, 583)).shape == (2, 10)

    capacity = check_f25_capacity.run(
        ROOT / ".audit_tmp" / "must-not-write.json", contract_only=True
    )
    assert check_f25_capacity.SCHEMA == "f25-capacity-preflight-v4"
    assert capacity["accepted"] is False and len(capacity["cases"]) == 2
    assert [row["case_id"] for row in capacity["cases"]] == [
        "RAW-pair-b48-five-layer-k2",
        "RAW-pair-b48-five-layer-k5",
    ]
    assert [row["input_shape"] for row in capacity["cases"]] == [
        [48, 2, 5830],
        [48, 2, 5830],
    ]
    assert [row["role"] for row in capacity["cases"]] == [
        "registered-job-envelope",
        "registered-job-envelope",
    ]
    assert [row["parameter_count"] for row in capacity["cases"]] == [
        47_851_338,
        47_925_834,
    ]
    _capacity_receipt_smoke()
    _run_record_runtime_smoke()

    matlab_driver = (ROOT / "scour_MATLAB" / "F25_Run.m").read_text(
        encoding="utf-8"
    )
    executor = (
        ROOT / "scour_MATLAB" / "+ttbi" / "f25_execute_generation_state.m"
    ).read_text(encoding="utf-8")
    assert "ttbi.f25_experiment_config" in matlab_driver
    assert "ttbi.f25_generation_identity" in matlab_driver
    assert "ttbi.f25_bundle_source_binding" in matlab_driver
    assert "ttbi.f25_execute_generation_state" in matlab_driver
    assert "~isequaln(previous.case_info, case_info)" in matlab_driver
    assert "file_matlab_environment_descriptor" in matlab_driver
    assert "file_bundle_source_binding_sha256" in matlab_driver
    assert "A01_Train" in executor and "B00_Calculations" in executor
    assert "D01_DataProcessing" in executor
    assert "ttbi.f25_extract_monitoring_signals" in executor
    assert "clean_trimmed" in executor and "monitoring_tail_sample" in executor
    assert "matlab_environment_descriptor" in executor
    assert "bundle_source_binding_sha256" in executor

    python_executor = (ROOT / "training" / "f25_executor.py").read_text(
        encoding="utf-8"
    )
    assert "bundle_binding = _validate_bundle_source_binding(REPO)" in python_executor
    assert "_validate_generation_case_info(" in python_executor
    assert "confirmed_bundle_binding = _validate_bundle_source_binding(REPO)" in (
        python_executor
    )

    build_f25_bundles.check(ROOT)
    bundle_commit_blob_smoke()
    f25_executor.smoke()
    print(
        "PASS check_f25_production: shared generation, 107 training jobs, "
        "flatten+dense reconstruction, tier anchors, capacity and bundles"
    )


if __name__ == "__main__":
    main()
