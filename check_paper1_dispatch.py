"""Behaviour and mutation checks for the six-bundle dispatch partition."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import build_stage_bundles

from core.campaign_contract import STAGE_ORDER
from core.paper1_dispatch import (
    GENERATION_BUNDLE_NAMES,
    TRAINING_BUNDLE_NAMES,
    _allocation_axis,
    assigned_training_host,
    generation_manifest,
    six_bundle_manifest_set,
    training_manifests,
    validate_training_manifests,
)
from core.paper1_training_contract import complete_job_grid


ROOT = Path(__file__).resolve().parent


def _expect_error(fn) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("dispatch mutation survived")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _portable_builder_smoke() -> None:
    """Build the real six-ZIP shape from a temporary clean Git commit."""

    names = build_stage_bundles._parse_source_manifest(
        (ROOT / build_stage_bundles.SOURCE_MANIFEST_NAME).read_bytes()
    )
    with tempfile.TemporaryDirectory(prefix="paper1-bundle-portable-") as raw:
        repo = Path(raw) / "repo"
        repo.mkdir()
        for name in names:
            source = ROOT.joinpath(*name.split("/"))
            target = repo.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _git(repo, "init", "--quiet")
        _git(repo, "config", "user.name", "Paper1 fixture")
        _git(repo, "config", "user.email", "paper1@example.invalid")
        _git(repo, "add", "--all")
        _git(repo, "commit", "--quiet", "-m", "portable fixture")

        dirty = repo / "unrelated-untracked.txt"
        dirty.write_text("must block\n", encoding="utf-8")
        _expect_error(lambda: build_stage_bundles.prepare_bundle_plan(repo))
        dirty.unlink()

        built = build_stage_bundles.build_bundles(repo)
        assert len(built.bundles) == 6
        sha_lines = built.sha_manifest.read_text(encoding="utf-8").splitlines()
        assert sha_lines[0].startswith("# source_commit ")
        assert sha_lines[1] == "# complete_bundle_count 6"
        assert len(sha_lines) == 8
        a00_hashes: set[str] = set()
        extracted_training: Path | None = None
        extracted_training_role: str | None = None
        for archive_path, digest_line in zip(built.bundles, sha_lines[2:]):
            expected_sha, expected_name = digest_line.split("  ", 1)
            assert expected_name == archive_path.name
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_sha
            with zipfile.ZipFile(archive_path) as archive:
                readme = archive.read("README_BUNDLE.md").decode("utf-8")
                assert "authorization" not in readme.lower()
                assert "Source commit:" in readme
                for variable in (
                    "TTBI_DATA_ROOT",
                    "TTBI_RESULTS_ROOT",
                    "TTBI_CACHE_ROOT",
                    "TTBI_STUDY_ROOT",
                    "TTBI_EXECUTION_RECEIPT_DIR",
                    "TTBI_CAMPAIGN_RUN_TAG",
                    "TTBI_TRAINING_JOB_MANIFEST",
                ):
                    assert variable in readme
                assert "--receipt-dir" in readme
                assert "--all-stages" in readme
                assert "Merge-TtbiTree" in readme
                assert "source/destination" in readme
                assert "paper1_bundle_identity.json" in archive.namelist()
                identity = json.loads(
                    archive.read("paper1_bundle_identity.json")
                )
                assert identity["schema"] == (
                    build_stage_bundles.BUNDLE_IDENTITY_SCHEMA
                )
                assert identity["source_commit"] == built.source_commit
                a00_bytes = archive.read(build_stage_bundles.A00)
                a00_hashes.add(hashlib.sha256(a00_bytes).hexdigest())
                assert a00_bytes == (repo / build_stage_bundles.A00).read_bytes()

                extracted = Path(raw) / f"extract-{archive_path.stem}"
                extracted.mkdir()
                archive.extractall(extracted)
                assert not (extracted / ".git").exists()
                if archive_path.name in GENERATION_BUNDLE_NAMES.values():
                    stage = next(
                        stage for stage, name in GENERATION_BUNDLE_NAMES.items()
                        if name == archive_path.name
                    )
                    assert identity["bundle_kind"] == "generation"
                    assert identity["target"] == stage
                    assert archive.read(
                        "generation_bundle_manifest.json"
                    ) == build_stage_bundles.canonical_json_bytes(
                        generation_manifest(stage)
                    )
                    assert "cd('<ABSOLUTE_EXTRACTED_BUNDLE>\\scour_MATLAB')" in readme
                    assert "A00_Run" in readme
                    assert "smoke_raw_parity('<folder>')" in readme
                    assert "python check_raw_parity.py '<folder>'" in readme
                    assert "scour_MATLAB/Results/<case_name>" in readme
                    assert "copy that completed folder" in readme
                else:
                    host = next(
                        host for host, name in TRAINING_BUNDLE_NAMES.items()
                        if name == archive_path.name
                    )
                    assert identity["bundle_kind"] == "training"
                    assert identity["target"] == host
                    assert archive.read(
                        "training_job_manifest.json"
                    ) == build_stage_bundles.canonical_json_bytes(
                        training_manifests()[host]
                    )
                    extracted_training = extracted
                    extracted_training_role = host

                import capacity_preflight_compute as capacity_publication

                authenticated = (
                    capacity_publication._embedded_bundle_source_identity(
                        extracted,
                        expected_bundle_kind=identity["bundle_kind"],
                    )
                )
                assert authenticated.mode == "embedded-bundle"
                assert authenticated.source_commit == built.source_commit
                if identity["bundle_kind"] == "training":
                    execution_source = (
                        capacity_publication
                        .authenticate_training_execution_source(
                            identity["target"], repo=extracted
                        )
                    )
                    assert execution_source.source_commit == built.source_commit

        assert len(a00_hashes) == 1
        assert extracted_training is not None
        assert extracted_training_role is not None
        import capacity_preflight_compute as capacity_publication

        _expect_error(
            lambda: capacity_publication.authenticate_training_execution_source(
                "labA" if extracted_training_role == "labB" else "labB",
                repo=extracted_training,
            )
        )

        victim = extracted_training / "capacity_preflight_compute.py"
        original = victim.read_bytes()
        victim.write_bytes(original + b"\n")
        _expect_error(
            lambda: capacity_publication._embedded_bundle_source_identity(
                extracted_training
            )
        )
        victim.write_bytes(original)

        manifest_path = extracted_training / "training_job_manifest.json"
        original_manifest = manifest_path.read_bytes()
        mutated_manifest = json.loads(original_manifest)
        mutated_manifest["machine_role"] = "labA" if (
            mutated_manifest["machine_role"] == "labB"
        ) else "labB"
        manifest_path.write_bytes(
            build_stage_bundles.canonical_json_bytes(mutated_manifest)
        )
        _expect_error(
            lambda: capacity_publication._embedded_bundle_source_identity(
                extracted_training
            )
        )
        manifest_path.write_bytes(original_manifest)


def main() -> None:
    complete = complete_job_grid()
    manifests = training_manifests()
    all_jobs = [
        job
        for jobs in complete["phases"].values()
        for job in jobs
    ]
    ids_a = set(manifests["labA"]["assigned_job_ids"])
    ids_b = set(manifests["labB"]["assigned_job_ids"])
    assert not ids_a.intersection(ids_b)
    assert ids_a.union(ids_b) == {job["job_id"] for job in all_jobs}
    assert len(ids_a) == 912
    assert len(ids_b) == 688
    for job in all_jobs:
        host = assigned_training_host(job)
        assert job["job_id"] in manifests[host]["assigned_job_ids"]

    # The v3 execution policy admits different GPUs and numeric stacks inside
    # one scientific block.  That is sound ONLY because of two properties of
    # ``assigned_training_host`` that nothing else asserts:
    #
    #   (a) one allocation seed maps to exactly one host campaign-wide, so a
    #       contrast paired on seed is automatically paired within a host and
    #       any host effect cancels in the difference; and
    #   (b) every compared pipeline receives an identical host mix, so no arm
    #       is advantaged by landing more often on the faster or slower PC.
    #
    # A future "balance the queue by machine speed" change would keep the
    # partition disjoint and complete -- and would silently destroy both.
    seed_hosts: dict[tuple[str, int], set[str]] = {}
    pipeline_mix: dict[str, dict[str, int]] = {}
    for job in all_jobs:
        axis, seed, _inventory = _allocation_axis(job)
        host = assigned_training_host(job)
        seed_hosts.setdefault((axis, seed), set()).add(host)
        mix = pipeline_mix.setdefault(job["pipeline"], {})
        mix[host] = mix.get(host, 0) + 1
    split_seeds = {key for key, hosts in seed_hosts.items() if len(hosts) > 1}
    assert not split_seeds, f"allocation seed spans hosts: {sorted(split_seeds)}"
    factorial_mixes = {
        pipeline: tuple(sorted(mix.items()))
        for pipeline, mix in pipeline_mix.items()
        if pipeline in {cell["cell_id"] for cell in complete["factorial_cells"]}
    }
    assert len(factorial_mixes) == 16
    assert len(set(factorial_mixes.values())) == 1, (
        f"factorial cells receive unequal host mixes: {factorial_mixes}"
    )
    slot_mixes = {
        pipeline: tuple(sorted(mix.items()))
        for pipeline, mix in pipeline_mix.items()
        if pipeline in set(complete["retained_pipeline_slots"])
    }
    assert len(slot_mixes) == len(complete["retained_pipeline_slots"])
    assert len(set(slot_mixes.values())) == 1, (
        f"retained slots receive unequal host mixes: {slot_mixes}"
    )

    dispatch = six_bundle_manifest_set()
    assert dispatch["bundle_names"] == [
        *(GENERATION_BUNDLE_NAMES[stage] for stage in STAGE_ORDER),
        *(TRAINING_BUNDLE_NAMES[host] for host in ("labA", "labB")),
    ]
    assert len(dispatch["bundle_names"]) == 6
    for stage in STAGE_ORDER:
        item = generation_manifest(stage)
        assert item["stage"] == stage
        assert item["n_states"] in {305, 425, 475}

    missing = deepcopy(manifests)
    missing["labA"]["jobs"].pop()
    _expect_error(lambda: validate_training_manifests(missing))

    duplicate = deepcopy(manifests)
    duplicate["labB"]["jobs"].append(duplicate["labA"]["jobs"][0])
    _expect_error(lambda: validate_training_manifests(duplicate))

    method_confounded = deepcopy(manifests)
    method_confounded["labA"]["jobs"][0]["pipeline"] = (
        "PAA_POS0_LSTM0_MR0"
    )
    _expect_error(lambda: validate_training_manifests(method_confounded))

    builder_source = inspect.getsource(build_stage_bundles)
    assert "TTBI_DISPATCH_AUTHORIZATION_MANIFEST" not in builder_source
    assert "--dispatch-authorization-manifest" not in builder_source
    assert "tested_source_commit" not in builder_source
    assert "--check-only" in builder_source
    assert "set_a00_bundle_config" not in builder_source
    assert "set_a00_stage" not in builder_source
    assert tuple(build_stage_bundles.BUNDLES) == (
        "f40s_generate",
        "f40m_generate",
        "l99s_generate",
        "l99m_generate",
        "train_labA",
        "train_labB",
    )
    generation_readme = build_stage_bundles.paper1_readme(
        "generation", "F40-S", "test", "a" * 40
    )
    training_readme = build_stage_bundles.paper1_readme(
        "training", "labA", "test", "a" * 40
    )
    assert "capability/physics smokes" in generation_readme
    assert "Exact MATLAB/Update/toolbox versions" in generation_readme
    assert "batch 32 × 2 channels × 11,791 samples" in training_readme
    assert "one fresh receipt for each of the four" in training_readme
    assert "may run on different PCs" in training_readme
    assert "matched-GPU" not in training_readme
    assert "authorization" not in generation_readme.lower()
    assert "authorization" not in training_readme.lower()
    required_runtime_variables = (
        "TTBI_DATA_ROOT",
        "TTBI_RESULTS_ROOT",
        "TTBI_CACHE_ROOT",
        "TTBI_STUDY_ROOT",
        "TTBI_EXECUTION_RECEIPT_DIR",
        "TTBI_CAMPAIGN_RUN_TAG",
        "TTBI_TRAINING_JOB_MANIFEST",
    )
    for readme in (generation_readme, training_readme):
        assert all(variable in readme for variable in required_runtime_variables)
        assert "authenticated union" in readme
        assert "Merge-TtbiTree" in readme
        assert "source/destination" in readme
    assert "--all-stages --receipt-dir" in builder_source
    assert "smoke_raw_parity('<folder>')" in generation_readme
    assert "python check_raw_parity.py '<folder>'" in generation_readme
    assert "cd('<ABSOLUTE_EXTRACTED_BUNDLE>\\scour_MATLAB')" in generation_readme
    assert "scour_MATLAB/Results/<case_name>" in generation_readme

    a00_source = (ROOT / build_stage_bundles.A00).read_text(encoding="utf-8")
    selector_source = (
        ROOT / "scour_MATLAB" / "+ttbi"
        / "load_generation_bundle_manifest.m"
    ).read_text(encoding="utf-8")
    assert "ttbi.load_generation_bundle_manifest" in a00_source
    assert "build_stage_bundles.py writes" not in a00_source
    assert "if ~qualification_run" in a00_source
    for stage in STAGE_ORDER:
        manifest_bytes = build_stage_bundles.canonical_json_bytes(
            generation_manifest(stage)
        )
        assert hashlib.sha256(manifest_bytes).hexdigest() in selector_source

    operator_guide = (ROOT / "README_CAMPAIGN.md").read_text(encoding="utf-8")
    driver_source = (
        ROOT / "comprehensive_ablation_multidamage.py"
    ).read_text(encoding="utf-8")
    for action in (
        "--execute-job",
        "--publish-adjudication",
        "--publish-channel-selection",
        "--publish-block-freeze",
    ):
        assert action in driver_source
        assert action in operator_guide
    for dataset in (
        "F40-S_L40_st305",
        "F40-M_L40_st425",
        "L99-S_L99.6_st475",
        "L99-M_L99.6_st475",
    ):
        assert f"data/{dataset}" in operator_guide
    for phase in (
        "f40s_factorial_hpo",
        "f40s_development_adjudication",
        "f40s_frozen_hyperparameter_channel_screen",
        "f40s_selected_pair_hpo",
        "block_selected_pair_hpo",
        "post_freeze_sealed_test_stability",
        "secondary_frozen_hyperparameter_transfer",
    ):
        assert phase in operator_guide
    assert "Merge-TtbiTree" in operator_guide
    assert "--all-stages" in operator_guide
    assert "smoke_raw_parity('<folder>')" in operator_guide
    assert "python check_raw_parity.py '<folder>'" in operator_guide
    assert "cd('<ABSOLUTE_EXTRACTED_BUNDLE>\\scour_MATLAB')" in operator_guide
    assert "$EvidencePrefix.source.sha256" in operator_guide
    assert "$EvidencePrefix.destination.sha256" in operator_guide
    assert "Set-Content" in operator_guide
    assert "Write-Output \"WROTE $sourceInventoryPath\"" in operator_guide
    assert "Copy-TtbiDataset" in operator_guide
    assert "publisher-only copy is insufficient" in operator_guide
    assert "1,600-job primary grid" in operator_guide
    assert "no challenger executor" in operator_guide
    _portable_builder_smoke()
    print("PASS: six-bundle dispatch manifests and balanced seed partition")
    print("PASS: portable clean-commit builder publishes six authenticated ZIPs")


if __name__ == "__main__":
    main()
