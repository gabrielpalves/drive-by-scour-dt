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
from core.f25_experiment_contract import CHANNELS, build_contract
from core.f25_models import F25CNN, build_f25_model
from core.f25_training_contract import (
    F25TrainingContractError,
    build_training_plan,
    canonical_json_sha256,
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
    runtime = {
        "environment_lock_sha256": "a" * 64,
        "execution_environment_sha256": "e" * 64,
        "execution_compatibility_sha256": "b" * 64,
        "execution_environment_descriptor": {
            "accelerator": {"total_memory_bytes": 1_000_000}
        },
        "execution_compatibility_descriptor": {"gpu": "fixture"},
    }
    source_root = SimpleNamespace(sha256="c" * 64)
    base = {
        "schema": check_f25_capacity.SCHEMA,
        "accepted": True,
        "contract_only": False,
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "case_contract_sha256": canonical_json_sha256(cases),
        "cases": cases,
        "environment_lock_sha256": runtime["environment_lock_sha256"],
        "execution_environment_sha256": runtime["execution_environment_sha256"],
        "execution_compatibility_sha256": runtime[
            "execution_compatibility_sha256"
        ],
        "execution_environment_descriptor": runtime[
            "execution_environment_descriptor"
        ],
        "execution_compatibility_descriptor": runtime[
            "execution_compatibility_descriptor"
        ],
        "python_runtime_source_sha256": source_root.sha256,
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
        receipt_path = root / "f25_artifacts" / "f25_capacity_receipt.json"
        original_repo = f25_executor.REPO
        f25_executor.REPO = root
        try:
            write(receipt_path, signed(base))
            f25_executor._require_capacity_receipt(runtime, source_root)
            mutations = []
            missing = copy.deepcopy(base)
            missing["cases"].pop()
            missing["case_contract_sha256"] = canonical_json_sha256(
                missing["cases"]
            )
            mutations.append((missing, "missing full-eight case"))
            reordered = copy.deepcopy(base)
            reordered["cases"][2:] = reversed(reordered["cases"][2:])
            reordered["case_contract_sha256"] = canonical_json_sha256(
                reordered["cases"]
            )
            mutations.append((reordered, "reordered full-eight cases"))
            shape = copy.deepcopy(base)
            shape["cases"][2]["input_shape"][1] = 2
            shape["case_contract_sha256"] = canonical_json_sha256(shape["cases"])
            mutations.append((shape, "full-eight shape changed"))
            stale_digest = copy.deepcopy(base)
            stale_digest["case_contract_sha256"] = "d" * 64
            mutations.append((stale_digest, "stale case-contract digest"))
            contract_only = copy.deepcopy(base)
            contract_only["contract_only"] = True
            mutations.append((contract_only, "contract-only receipt"))
            foreign_environment = copy.deepcopy(base)
            foreign_environment["execution_environment_sha256"] = "f" * 64
            mutations.append((foreign_environment, "foreign target GPU/runtime"))
            zero_memory = copy.deepcopy(base)
            zero_memory["measurements"][3]["peak_allocated_bytes"] = 0
            mutations.append((zero_memory, "zero-compute full-eight stress"))
            for mutation, label in mutations:
                write(receipt_path, signed(mutation))
                try:
                    f25_executor._require_capacity_receipt(runtime, source_root)
                except f25_executor.F25ExecutionError:
                    continue
                raise AssertionError(f"capacity mutation was accepted: {label}")
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
        b"* text eol=crlf\n.gitattributes text eol=lf\n"
    )

    entries = tuple(
        sorted(build_f25_bundles.REQUIRED_F25_SOURCE | {"fixture_payload.txt"})
    )
    for name in entries:
        if name == build_f25_bundles.SOURCE_MANIFEST:
            continue
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture source: {name}\nsecond line\n".encode("utf-8"))
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
                assert commit.encode("ascii") in archive.read(readme_name)
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
    assert len(plan_x["jobs"]) == 156
    assert sum(job["phase"] == "hpo" for job in plan_r["jobs"]) == 4
    assert sum(job["phase"] == "report" for job in plan_r["jobs"]) == 4
    assert sum(job["phase"] == "hpo" for job in plan_x["jobs"]) == 24
    assert sum(job["phase"] == "report" for job in plan_x["jobs"]) == 132
    assert sum(
        job["hpo_proposals"] * job["executions_per_proposal"]
        for job in plan_r["jobs"]
    ) == 2_000
    assert sum(
        job["hpo_proposals"] * job["executions_per_proposal"]
        for job in plan_x["jobs"]
    ) == 12_000
    assert sum(len(job["report_seeds"]) for job in plan_r["jobs"]) == 80
    assert sum(len(job["report_seeds"]) for job in plan_x["jobs"]) == 2_640
    frozen = [
        job
        for job in plan_x["jobs"]
        if job["regime"] == "frozen" and job["phase"] == "report"
    ]
    assert len(frozen) == 108
    assert all(job["anchor_configuration_id"].endswith("S01") for job in frozen)
    assert all(len(job["sensor_set"]) <= 2 for job in plan_x["jobs"])
    assert {
        tuple(job["sensor_set"])
        for job in plan_x["jobs"]
        if job["tier_id"] == "F25-X-03-frozen-hp-pairs"
    } == {
        tuple(sorted((left, right)))
        for index, left in enumerate(CHANNELS)
        for right in CHANNELS[index + 1 :]
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
    assert capacity["accepted"] is False and len(capacity["cases"]) == 4
    assert [row["case_id"] for row in capacity["cases"]] == [
        "RAW-pair-b48-five-layer-k2",
        "RAW-pair-b48-five-layer-k5",
        "RAW-full8-conservative-b48-five-layer-k2",
        "RAW-full8-conservative-b48-five-layer-k5",
    ]
    assert [row["input_shape"] for row in capacity["cases"]] == [
        [48, 2, 5830],
        [48, 2, 5830],
        [48, 8, 5830],
        [48, 8, 5830],
    ]
    assert [row["role"] for row in capacity["cases"]] == [
        "registered-job-envelope",
        "registered-job-envelope",
        "conservative-nonjob-dispatch-stress",
        "conservative-nonjob-dispatch-stress",
    ]
    assert [row["parameter_count"] for row in capacity["cases"]] == [
        47_851_338,
        47_925_834,
        47_852_874,
        47_929_674,
    ]
    _capacity_receipt_smoke()

    matlab_driver = (ROOT / "scour_MATLAB" / "F25_Run.m").read_text(
        encoding="utf-8"
    )
    executor = (
        ROOT / "scour_MATLAB" / "+ttbi" / "f25_execute_generation_state.m"
    ).read_text(encoding="utf-8")
    assert "ttbi.f25_experiment_config" in matlab_driver
    assert "ttbi.f25_generation_identity" in matlab_driver
    assert "ttbi.f25_execute_generation_state" in matlab_driver
    assert "A01_Train" in executor and "B00_Calculations" in executor
    assert "D01_DataProcessing" in executor
    assert "ttbi.f25_extract_monitoring_signals" in executor
    assert "clean_trimmed" in executor and "monitoring_tail_sample" in executor

    build_f25_bundles.check(ROOT)
    bundle_commit_blob_smoke()
    f25_executor.smoke()
    print(
        "PASS check_f25_production: shared generation, 164 training jobs, "
        "flatten+dense reconstruction, tier anchors, capacity and bundles"
    )


if __name__ == "__main__":
    main()
