"""Mutation checks for authenticated Paper-1 selected-pair HPO.

Run: ``py -3.13 check_paper1_selection_contract.py``
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile

from core.hyperparameter_policy import (
    HYPERPARAMETER_POLICY,
    SELECTED_PAIR_HPO_MODE,
    HyperparameterPolicyError,
    derive_execution_plan,
    validate_run_plan,
)
from core.paper1_dispatch import assigned_training_host, training_manifests
from core.paper1_selection import (
    Paper1SelectionError,
    SELECTION_ARTIFACT_SHA256_ENV,
    build_selection_artifact,
    load_selection_artifact,
    resolve_selection_claim,
    validate_selection_artifact,
)
from core.paper1_training_contract import (
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    PAA_CNN_GAP_BASELINE_ID,
    RAW_CNN_GAP_BASELINE_ID,
    canonical_json_bytes,
    hpo_jobs,
)
from core.protocol import protocol_hash
from training import paper1_executor as executor


FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def rejects(name: str, function) -> None:
    try:
        function()
    except (Paper1SelectionError, HyperparameterPolicyError,
            executor.Paper1ExecutionError):
        check(name, True)
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation was accepted")


def resign(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def selection() -> dict:
    best_raw = next(
        cell.cell_id for cell in FACTORIAL_CELLS
        if cell.representation == "RAW" and cell.cell_id != RAW_CNN_GAP_BASELINE_ID
    )
    best_paa = next(
        cell.cell_id for cell in FACTORIAL_CELLS
        if cell.representation == "PAA" and cell.cell_id != PAA_CNN_GAP_BASELINE_ID
    )
    return build_selection_artifact(
        campaign_run_tag="paper1-qualified-run",
        selected_pair=[1, 3],
        best_raw=best_raw,
        best_paa=best_paa,
        evidence_sha256={
            "factorial_hpo_manifest": "a" * 64,
            "development_adjudication_manifest": "b" * 64,
            "channel_screen_manifest": "c" * 64,
        },
    )


def selected_config(artifact: dict) -> dict:
    architecture = artifact["slot_resolution"]["f40s_best_raw"]
    core = {"fixture": "paper1-selected-pair-policy"}
    full = {
        "core": core,
        "rung": {
            "stage": "L99-M",
            "dataset": "L99-M_fixture",
            "execution_block": "l99m",
            "execution_anchor": "L99-M",
        },
    }
    return {
        "name": "selected-pair-fixture",
        "name_short": architecture,
        "seed": HPO_RESTART_SEEDS[0],
        "dofs": list(artifact["selected_pair"]),
        "protocol_descriptor": full,
        "protocol_hash": protocol_hash(full),
        "protocol_core_hash": protocol_hash(core),
        "hyperparameter_mode": SELECTED_PAIR_HPO_MODE,
        "campaign_run_tag": artifact["campaign_run_tag"],
        "execution_receipt_sha256": "e" * 64,
        "block_reference_manifest_sha256": None,
        "selection_artifact": artifact,
        "selection_artifact_sha256": artifact["artifact_sha256"],
        "selection_slot": "f40s_best_raw",
    }


def main() -> None:
    print("PAPER1 SELECTION / SELECTED-PAIR HPO CHECKS")
    artifact = selection()
    check(
        "canonical selection validates and resolves all four stages",
        validate_selection_artifact(artifact) == artifact
        and all(
            resolve_selection_claim(
                artifact, stage=stage, slot="f40s_best_raw"
            )["selected_pair"] == [1, 3]
            for stage in ("F40-S", "F40-M", "L99-S", "L99-M")
        ),
    )
    check(
        "selection policy registers 100-trial per-block HPO",
        HYPERPARAMETER_POLICY["selected_pair_hpo"] == {
            "applicable_stages": ["F40-S", "F40-M", "L99-S", "L99-M"],
            "retained_pipeline_slots": [
                "f40s_best_raw", "f40s_best_paa",
                "raw_cnn_gap_baseline", "paa_cnn_gap_baseline",
            ],
            "active_dofs": "authenticated F40-S selected_pair",
            "n_trials": 100,
            "use_registered_pruner": True,
        },
    )

    semantic_mutations = []
    mutant = deepcopy(artifact); mutant["selection_stage"] = "F40-M"
    semantic_mutations.append(("wrong selection stage", resign(mutant)))
    mutant = deepcopy(artifact); mutant["applicable_stages"][-1] = "foreign"
    semantic_mutations.append(("wrong applicable stage", resign(mutant)))
    mutant = deepcopy(artifact); mutant["selected_pair"] = [3, 1]
    semantic_mutations.append(("unordered pair", resign(mutant)))
    mutant = deepcopy(artifact); mutant["selected_pair"] = [1, 1]
    semantic_mutations.append(("duplicate channel", resign(mutant)))
    mutant = deepcopy(artifact); mutant["selected_pair"] = [1, 8]
    semantic_mutations.append(("out-of-range channel", resign(mutant)))
    mutant = deepcopy(artifact)
    mutant["slot_resolution"]["f40s_best_raw"] = PAA_CNN_GAP_BASELINE_ID
    semantic_mutations.append(("RAW slot resolved to PAA", resign(mutant)))
    mutant = deepcopy(artifact)
    mutant["slot_resolution"]["raw_cnn_gap_baseline"] = (
        artifact["slot_resolution"]["f40s_best_raw"]
    )
    semantic_mutations.append(("baseline slot changed", resign(mutant)))
    mutant = deepcopy(artifact)
    mutant["canonical_slot"]["raw_cnn_gap_baseline"] = "f40s_best_paa"
    semantic_mutations.append(("deduplication slot forged", resign(mutant)))
    mutant = deepcopy(artifact)
    mutant["evidence_sha256"]["channel_screen_manifest"] = "bad"
    semantic_mutations.append(("evidence digest invalid", resign(mutant)))
    for label, value in semantic_mutations:
        rejects(label, lambda value=value: validate_selection_artifact(value))
    mutant = deepcopy(artifact); mutant["artifact_sha256"] = "d" * 64
    rejects("self-digest forged", lambda: validate_selection_artifact(mutant))

    for label, kwargs in (
        ("foreign downstream stage", {"stage": "foreign"}),
        ("foreign retained slot", {"slot": "foreign"}),
        ("wrong selected pair", {"pair": [0, 2]}),
        ("wrong resolved architecture", {"architecture": PAA_CNN_GAP_BASELINE_ID}),
        ("wrong run tag", {"campaign_run_tag": "other"}),
        ("wrong cited artefact hash", {"artifact_sha256": "f" * 64}),
    ):
        base = {"stage": "L99-M", "slot": "f40s_best_raw"}
        base.update(kwargs)
        rejects(label, lambda base=base: resolve_selection_claim(artifact, **base))

    config = selected_config(artifact)
    plan = derive_execution_plan(
        config,
        dataset_name="L99-M_fixture",
        requested_n_trials=1,
        requested_use_pruner=False,
    )
    check(
        "live policy derives exact selected-pair budget and lineage",
        plan["mode"] == SELECTED_PAIR_HPO_MODE
        and plan["effective_n_trials"] == 100
        and plan["effective_use_pruner"] is True
        and plan["active_dofs"] == [1, 3]
        and plan["selection_artifact_sha256"] == artifact["artifact_sha256"]
        and plan["selection_slot"] == "f40s_best_raw"
        and validate_run_plan(plan) == plan,
    )
    for label, field, value in (
        ("policy rejects wrong pair", "dofs", [0, 2]),
        ("policy rejects wrong slot", "selection_slot", "f40s_best_paa"),
        ("policy rejects wrong selection hash", "selection_artifact_sha256", "f" * 64),
        ("policy rejects wrong run", "campaign_run_tag", "other"),
    ):
        mutant_config = deepcopy(config); mutant_config[field] = value
        rejects(
            label,
            lambda mutant_config=mutant_config: derive_execution_plan(
                mutant_config,
                dataset_name="L99-M_fixture",
                requested_n_trials=100,
                requested_use_pruner=True,
            ),
        )
    for label, field, value in (
        ("run plan rejects reduced selected budget", "effective_n_trials", 1),
        ("run plan rejects disabled selected pruner", "effective_use_pruner", False),
        ("run plan rejects foreign selected slot", "selection_slot", "foreign"),
        ("run plan rejects missing selected digest", "selection_artifact_sha256", None),
    ):
        mutant_plan = deepcopy(plan); mutant_plan[field] = value
        rejects(label, lambda mutant_plan=mutant_plan: validate_run_plan(mutant_plan))

    selected_job = next(
        job for job in hpo_jobs()
        if job["phase"] == "block_selected_pair_hpo"
        and job["stage"] == "L99-M"
        and job["pipeline"] == "f40s_best_raw"
        and job["hpo_restart_seed"] == HPO_RESTART_SEEDS[0]
    )
    host = assigned_training_host(selected_job)
    manifests = training_manifests()
    captured: dict = {}
    original = executor._execute_hpo_job
    saved_path = os.environ.get("TTBI_PAPER1_SELECTION_ARTIFACT")
    saved_sha = os.environ.get(SELECTION_ARTIFACT_SHA256_ENV)
    saved_tag = os.environ.get(executor.RUN_TAG_ENV)
    try:
        with tempfile.TemporaryDirectory(prefix="paper1-selection-") as td:
            path = Path(td).resolve() / "selection.json"
            path.write_bytes(canonical_json_bytes(artifact))
            os.environ["TTBI_PAPER1_SELECTION_ARTIFACT"] = str(path)
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = artifact[
                "artifact_sha256"
            ]
            os.environ[executor.RUN_TAG_ENV] = artifact["campaign_run_tag"]

            def fake_execute(job, manifest, **kwargs):
                captured.update({"job": job, "manifest": manifest, **kwargs})
                return {"schema": "fixture-completion"}

            executor._execute_hpo_job = fake_execute
            result = executor.execute_selected_pair_hpo_job(
                selected_job, manifests[host]
            )
            check(
                "executor resolves manifest slot/pair into live 100-trial adapter",
                result == {"schema": "fixture-completion"}
                and captured["architecture"]["name_short"]
                == artifact["slot_resolution"]["f40s_best_raw"]
                and captured["dofs"] == [1, 3]
                and captured["hyperparameter_mode"] == SELECTED_PAIR_HPO_MODE
                and captured["selection_slot"] == "f40s_best_raw",
            )
            os.environ[executor.RUN_TAG_ENV] = "other-run"
            rejects(
                "executor rejects selection from another run",
                lambda: executor.execute_selected_pair_hpo_job(
                    selected_job, manifests[host]
                ),
            )
            loaded = load_selection_artifact(path)
            check("canonical selection file authenticates", loaded == artifact)
            os.environ[executor.RUN_TAG_ENV] = artifact["campaign_run_tag"]
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = "f" * 64
            rejects(
                "executor rejects externally forged selection digest",
                lambda: executor.execute_selected_pair_hpo_job(
                    selected_job, manifests[host]
                ),
            )
    finally:
        executor._execute_hpo_job = original
        if saved_path is None:
            os.environ.pop("TTBI_PAPER1_SELECTION_ARTIFACT", None)
        else:
            os.environ["TTBI_PAPER1_SELECTION_ARTIFACT"] = saved_path
        if saved_sha is None:
            os.environ.pop(SELECTION_ARTIFACT_SHA256_ENV, None)
        else:
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = saved_sha
        if saved_tag is None:
            os.environ.pop(executor.RUN_TAG_ENV, None)
        else:
            os.environ[executor.RUN_TAG_ENV] = saved_tag

    baseline_artifact = build_selection_artifact(
        campaign_run_tag="paper1-qualified-run",
        selected_pair=[1, 3],
        best_raw=RAW_CNN_GAP_BASELINE_ID,
        best_paa=PAA_CNN_GAP_BASELINE_ID,
        evidence_sha256={
            "factorial_hpo_manifest": "a" * 64,
            "development_adjudication_manifest": "b" * 64,
            "channel_screen_manifest": "c" * 64,
        },
    )
    alias_job = next(
        job for job in hpo_jobs()
        if job["phase"] == "block_selected_pair_hpo"
        and job["stage"] == "L99-M"
        and job["pipeline"] == "raw_cnn_gap_baseline"
        and job["hpo_restart_seed"] == HPO_RESTART_SEEDS[0]
    )
    alias_host = assigned_training_host(alias_job)
    original_alias = executor._complete_selected_alias
    original_hpo = executor._execute_hpo_job
    saved_path = os.environ.get("TTBI_PAPER1_SELECTION_ARTIFACT")
    saved_sha = os.environ.get(SELECTION_ARTIFACT_SHA256_ENV)
    saved_tag = os.environ.get(executor.RUN_TAG_ENV)
    alias_capture: dict = {}
    try:
        with tempfile.TemporaryDirectory(prefix="paper1-selection-alias-") as td:
            path = Path(td).resolve() / "selection.json"
            path.write_bytes(canonical_json_bytes(baseline_artifact))
            os.environ["TTBI_PAPER1_SELECTION_ARTIFACT"] = str(path)
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = baseline_artifact[
                "artifact_sha256"
            ]
            os.environ[executor.RUN_TAG_ENV] = baseline_artifact[
                "campaign_run_tag"
            ]

            def fake_alias(**kwargs):
                alias_capture.update(kwargs)
                return {"schema": "fixture-alias-completion"}

            def forbidden_hpo(*_args, **_kwargs):
                raise AssertionError("deduplicated slot retrained")

            executor._complete_selected_alias = fake_alias
            executor._execute_hpo_job = forbidden_hpo
            result = executor.execute_selected_pair_hpo_job(
                alias_job, manifests[alias_host]
            )
            check(
                "baseline-equal winner is an authenticated canonical-slot alias",
                result == {"schema": "fixture-alias-completion"}
                and alias_capture["claim"]["slot"] == "raw_cnn_gap_baseline"
                and alias_capture["claim"]["canonical_slot"]
                == "f40s_best_raw"
                and alias_capture["selection"]["artifact_sha256"]
                == baseline_artifact["artifact_sha256"],
            )
    finally:
        executor._complete_selected_alias = original_alias
        executor._execute_hpo_job = original_hpo
        if saved_path is None:
            os.environ.pop("TTBI_PAPER1_SELECTION_ARTIFACT", None)
        else:
            os.environ["TTBI_PAPER1_SELECTION_ARTIFACT"] = saved_path
        if saved_sha is None:
            os.environ.pop(SELECTION_ARTIFACT_SHA256_ENV, None)
        else:
            os.environ[SELECTION_ARTIFACT_SHA256_ENV] = saved_sha
        if saved_tag is None:
            os.environ.pop(executor.RUN_TAG_ENV, None)
        else:
            os.environ[executor.RUN_TAG_ENV] = saved_tag

    print()
    if FAILURES:
        raise SystemExit(f"PAPER1 SELECTION: {FAILURES} CHECK(S) FAILED")
    print("PAPER1 SELECTION: ALL PASS")


if __name__ == "__main__":
    main()
