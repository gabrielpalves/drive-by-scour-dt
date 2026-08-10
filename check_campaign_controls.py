"""Adversarial checks for the four-block training/dispatch controls.

Run: ``py -3.13 check_campaign_controls.py``
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import comprehensive_ablation_multidamage as driver
from training import paper1_executor as executor
from core.campaign_contract import (
    EXPECTED_PROTOCOL_SCHEMA_TAG,
    STAGE_ORDER as GENERATION_STAGE_ORDER,
)
from core.paper1_dispatch import (
    TRAINING_HOSTS,
    assigned_training_host,
    training_manifests,
    validate_training_manifests,
)
from core.paper1_training_contract import (
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    RETAINED_PIPELINE_SLOTS,
    STAGE_ORDER,
    canonical_json_bytes,
    channel_screen_inputs,
    complete_job_grid,
    development_adjudication_jobs,
    frozen_transfer_jobs,
    hpo_jobs,
    post_freeze_stability_jobs,
)


FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def raises(name: str, function, expected: type[BaseException]) -> None:
    try:
        function()
    except expected:
        check(name, True)
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation was accepted")


def _all_jobs() -> list[dict]:
    phases = complete_job_grid()["phases"]
    return [
        job
        for phase in (
            "hpo",
            "development_adjudication",
            "channel_screen",
            "post_freeze_stability",
            "secondary_frozen_transfer",
        )
        for job in phases[phase]
    ]


def main() -> None:
    print("PAPER1 CAMPAIGN CONTROL CHECKS")
    check(
        "generation and training use one four-block order",
        STAGE_ORDER == GENERATION_STAGE_ORDER
        == ("F40-S", "F40-M", "L99-S", "L99-M"),
    )
    check("architecture comparison is exact 2x2x2x2", len(FACTORIAL_CELLS) == 16)
    check(
        "channel screen is all 8 singles plus 28 pairs",
        Counter(map(len, channel_screen_inputs())) == {1: 8, 2: 28},
    )
    hpo_counts = Counter((job["stage"], job["phase"]) for job in hpo_jobs())
    check(
        "per-block HPO replaces transport/rescue",
        hpo_counts == {
            ("F40-S", "f40s_factorial_hpo"): 80,
            ("F40-S", "f40s_selected_pair_hpo"): 20,
            ("F40-M", "block_selected_pair_hpo"): 20,
            ("L99-S", "block_selected_pair_hpo"): 20,
            ("L99-M", "block_selected_pair_hpo"): 20,
        }
        and {job["trials"] for job in hpo_jobs()} == {100}
        and len(HPO_RESTART_SEEDS) == 5,
    )
    check(
        "registered refit grids are complete",
        len(development_adjudication_jobs()) == 480
        and len(post_freeze_stability_jobs()) == 480
        and len(frozen_transfer_jobs()) == 60,
    )

    manifests = training_manifests()
    validated = validate_training_manifests(manifests)
    all_jobs = _all_jobs()
    expected_ids = {job["job_id"] for job in all_jobs}
    ids_by_host = {
        host: set(validated[host]["assigned_job_ids"])
        for host in TRAINING_HOSTS
    }
    check(
        "Lab-A/Lab-B manifests are disjoint and exhaustive",
        not (ids_by_host["labA"] & ids_by_host["labB"])
        and ids_by_host["labA"] | ids_by_host["labB"] == expected_ids,
    )
    check(
        "every assigned job is source-identical to the complete grid",
        sum(len(value) for value in ids_by_host.values()) == len(all_jobs)
        and all(
            assigned_training_host(job) == host
            for host in TRAINING_HOSTS
            for job in validated[host]["jobs"]
        ),
    )

    # Hardware allocation may depend only on the first applicable registered
    # seed.  Mutating compared scientific fields must never move a job.
    allocation_invariant = True
    for job in all_jobs:
        expected_host = assigned_training_host(job)
        mutant = deepcopy(job)
        mutant["stage"] = "foreign-stage"
        mutant["pipeline"] = "foreign-pipeline"
        mutant["input_selector"] = [7, 6, 5]
        if assigned_training_host(mutant) != expected_host:
            allocation_invariant = False
            break
    check(
        "host allocation excludes stage/pipeline/channel method",
        allocation_invariant,
    )

    # Within every phase/seed inventory, each compared pipeline sees the same
    # host pattern.  This is stronger than merely balancing aggregate counts.
    patterns: dict[tuple[str, str, str], dict[str, tuple[str, ...]]] = defaultdict(dict)
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for job in all_jobs:
        selector = json.dumps(job["input_selector"], sort_keys=True)
        key = (job["phase"], job["stage"], selector, job["pipeline"])
        grouped[key].append(job)
    for (phase, stage, selector, pipeline), jobs in grouped.items():
        hosts = tuple(sorted(assigned_training_host(job) for job in jobs))
        patterns[(phase, stage, selector)][pipeline] = hosts
    same_pattern = all(
        len(set(by_pipeline.values())) == 1
        for by_pipeline in patterns.values()
        if len(by_pipeline) > 1
    )
    check(
        "every compared pipeline has the same Lab-A/Lab-B seed pattern",
        same_pattern,
    )

    mutant_manifests = deepcopy(manifests)
    mutant_manifests["labA"]["jobs"][0]["pipeline"] = "outcome-picked"
    raises(
        "foreign/outcome-picked manifest job is rejected",
        lambda: validate_training_manifests(mutant_manifests),
        RuntimeError,
    )
    duplicate_manifests = deepcopy(manifests)
    stolen = deepcopy(duplicate_manifests["labA"]["jobs"][0])
    duplicate_manifests["labB"]["jobs"].append(stolen)
    duplicate_manifests["labB"]["assigned_job_ids"].append(stolen["job_id"])
    duplicate_manifests["labB"]["assigned_job_count"] += 1
    raises(
        "cross-host duplicate job is rejected",
        lambda: validate_training_manifests(duplicate_manifests),
        RuntimeError,
    )

    check(
        "retired driver is replaced by manifest-qualified execution",
        driver.SCHEMA_TAG == EXPECTED_PROTOCOL_SCHEMA_TAG
        and driver.MIGRATION_STATUS
        == "manifest-qualified-four-stage-execution"
        and tuple(RETAINED_PIPELINE_SLOTS)
        == (
            "f40s_best_raw",
            "f40s_best_paa",
            "raw_cnn_gap_baseline",
            "paa_cnn_gap_baseline",
        ),
    )
    factorial_job = next(
        job for job in all_jobs if job["phase"] == "f40s_factorial_hpo"
    )
    validated_factorial = executor.validate_f40s_factorial_hpo_job(
        factorial_job
    )
    architecture = executor.factorial_architecture(
        validated_factorial["pipeline"]
    )
    check(
        "F40-S factorial jobs map exactly onto live model flags",
        validated_factorial == factorial_job
        and architecture["name_short"] == factorial_job["pipeline"]
        and architecture["method"] in {"RAW", "PAA"}
        and set(architecture)
        == {
            "name_short", "method", "use_space2vec", "use_lstm",
            "use_nhits", "model_type",
        },
    )
    mutant_factorial = deepcopy(factorial_job)
    mutant_factorial["trials"] = 99
    raises(
        "factorial adapter rejects a reduced HPO budget",
        lambda: executor.validate_f40s_factorial_hpo_job(mutant_factorial),
        executor.Paper1ExecutionError,
    )
    adjudication_job = next(
        job for job in all_jobs
        if job["phase"] == "f40s_development_adjudication"
    )
    screen_job = next(
        job for job in all_jobs
        if job["phase"] == "f40s_frozen_hyperparameter_channel_screen"
    )
    check(
        "Option-C adjudication and channel jobs have executable exact validators",
        executor.validate_development_adjudication_job(adjudication_job)
        == adjudication_job
        and executor.validate_channel_screen_job(screen_job) == screen_job,
    )
    nonfactorial = next(
        job for job in all_jobs
        if job["phase"] == "post_freeze_sealed_test_stability"
        and assigned_training_host(job) == "labA"
    )
    secondary = next(
        job for job in all_jobs
        if job["phase"] == "secondary_frozen_hyperparameter_transfer"
        and assigned_training_host(job) == "labA"
    )
    check(
        "sealed-report jobs have exact executable validators",
        executor.validate_post_freeze_stability_job(nonfactorial)
        == nonfactorial
        and executor.validate_secondary_frozen_transfer_job(secondary)
        == secondary,
    )
    saved_run_tag = os.environ.get(executor.RUN_TAG_ENV)
    os.environ[executor.RUN_TAG_ENV] = "campaign-controls-fixture"
    try:
        raises(
            "post-freeze phase refuses to open data without a deposited freeze",
            lambda: driver.execute_registered_job(
                nonfactorial, manifests["labA"]
            ),
            executor.Paper1ExecutionDependencyError,
        )
    finally:
        if saved_run_tag is None:
            os.environ.pop(executor.RUN_TAG_ENV, None)
        else:
            os.environ[executor.RUN_TAG_ENV] = saved_run_tag

    saved = os.environ.pop(driver.TRAINING_JOB_MANIFEST_ENV, None)
    try:
        raises(
            "driver rejects missing TTBI_TRAINING_JOB_MANIFEST",
            driver.load_training_job_manifest,
            driver.TrainingManifestError,
        )
        with tempfile.TemporaryDirectory(prefix="paper1-training-manifest-") as td:
            root = Path(td).resolve()
            valid_path = root / "labA.json"
            valid_path.write_bytes(canonical_json_bytes(manifests["labA"]))
            loaded = driver.load_training_job_manifest(valid_path)
            check(
                "driver authenticates exact canonical Lab-A manifest",
                loaded == manifests["labA"],
            )
            first_job = loaded["jobs"][0]
            check(
                "driver materializes only an assigned exact job",
                driver.select_registered_job(loaded, first_job["job_id"])
                == first_job,
            )
            raises(
                "driver rejects a foreign job ID",
                lambda: driver.select_registered_job(loaded, "0" * 24),
                driver.TrainingManifestError,
            )
            noncanonical = root / "noncanonical.json"
            noncanonical.write_bytes(
                json.dumps(manifests["labA"], indent=2).encode("ascii")
            )
            raises(
                "driver rejects noncanonical manifest bytes",
                lambda: driver.load_training_job_manifest(noncanonical),
                driver.TrainingManifestError,
            )
    finally:
        if saved is not None:
            os.environ[driver.TRAINING_JOB_MANIFEST_ENV] = saved

    source = Path(driver.__file__).read_text(encoding="utf-8")
    executor_source = Path(executor.__file__).read_text(encoding="utf-8")
    pipeline_source = (
        Path(__file__).resolve().parent / "training" / "pipeline.py"
    ).read_text(encoding="utf-8")
    check(
        "production pipeline seeding consumes TRAIN_PROTOCOL determinism policy",
        'set_global_seed(optuna_seed, TRAIN_PROTOCOL["determinism"])'
        in pipeline_source,
    )
    check(
        "HPO adapter delegates to the production pipeline and its Objective",
        "from training.pipeline import execute_ablation_pipeline" in executor_source
        and "pipeline_runner(" in executor_source
        and "objective = Objective(" in pipeline_source
        and "paper1_job_identity.json" in executor_source
        and "paper1_job_completion.json" in executor_source,
    )
    check(
        "selected-pair HPO resolves its authenticated artefact into production HPO",
        all(
            token in executor_source
            for token in (
                "def execute_selected_pair_hpo_job(",
                "SELECTION_ARTIFACT_SHA256_ENV",
                "load_selection_artifact(",
                "expected_sha256=expected_selection_sha",
                "SELECTED_PAIR_HPO_MODE",
                "return _execute_hpo_job(",
                "return _complete_selected_alias(",
            )
        ),
    )
    check(
        "adjudication and channel-screen adapters execute authenticated refits",
        all(
            token in executor_source
            for token in (
                "def execute_development_adjudication_job(",
                'metadata.get("hyperparameter_mode") != ANCHOR_HPO_MODE',
                "fit_predict_fixed_group_fold",
                "seal_development_result(",
                "def execute_channel_screen_job(",
                "load_development_artifact(",
                "seal_channel_result(",
                "def publish_development_adjudication_artifact(",
                "def publish_channel_selection_artifacts(",
            )
        )
        and '"factorial_anchor_hpo"' not in executor_source,
    )
    check(
        "final report-only phases use authenticated freeze and robustness adapters",
        all(token in executor_source for token in (
            "def publish_block_freeze_artifact(",
            "def execute_post_freeze_stability_job(",
            "def execute_secondary_frozen_transfer_job(",
            "run_post_freeze_stability",
            "_load_sealed_refit_data(",
            "seal_sealed_result(",
        )),
    )
    retired = (
        "s0_scour", "s11_bear", "s12_crack", "s13_bearcrack",
        "s14_prof", "s15_track", "s16_all", "s21_scour4",
        "s22_bearcrack4", "s23_all4",
    )
    check(
        "retired ten-rung stage names are absent from the entrypoint",
        not any(stage in source for stage in retired),
    )

    print()
    if FAILURES:
        raise SystemExit(f"CAMPAIGN CONTROLS: {FAILURES} CHECK(S) FAILED")
    print("CAMPAIGN CONTROLS: ALL PASS")


if __name__ == "__main__":
    main()
