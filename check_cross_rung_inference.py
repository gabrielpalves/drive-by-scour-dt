"""Adversarial check of the registered seven-edge L60 CRN analysis."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from core.campaign_contract import (
    EXPECTED_PROTOCOL_SCHEMA_TAG,
    campaign_stage_contract,
)
from core.execution_environment import execution_environment_sha256
from core.hyperparameter_policy import (
    ARCHITECTURES,
    HYPERPARAMETER_POLICY,
    SEEDS as HYPERPARAMETER_SEEDS,
    STUDY_IDENTITY_SCHEMA,
    build_manifest,
    build_manifest_entry,
    canonical_json_sha256 as hyperparameter_json_sha256,
    policy_sha256 as hyperparameter_policy_sha256,
    write_manifest,
)
import core.cross_rung_inference as cri
from core.protocol import canonical_json, protocol_hash
from core.utils import IDX_TO_DOF_NAME


FAILURES = 0
SEEDS = [42, 1337, 2026]
ARCH = "PAA_CNN"
PAIR = [1, 3]
DOFS = "+".join(IDX_TO_DOF_NAME[index] for index in PAIR)
RECEIPT = ""
RUN_TAG = "fixture-l60-run"
CAPACITY_SHA = "c" * 64
OFFSETS = {
    "s0_scour": 0.0,
    "s11_bear": 1.0,
    "s12_crack": 2.0,
    "s13_bearcrack": 4.0,
    "s14_prof": 5.0,
    "s15_track": 7.0,
    "s16_all": 10.0,
}


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def raises(name: str, function) -> None:
    try:
        function()
    except RuntimeError:
        check(name, True)
    else:
        check(name, False, "mutation was accepted")


def raises_with_message(name: str, function, fragment: str) -> None:
    try:
        function()
    except RuntimeError as exc:
        check(name, fragment in str(exc), str(exc))
    else:
        check(name, False, "mutation was accepted")


def _sha(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity_and_split() -> tuple[dict, dict]:
    anchors = [f"anchor-{index:03d}" for index in range(200)]
    joint = [f"joint-{index:03d}" for index in range(250)]
    row_order = anchors + joint
    inventory = sorted(row_order)
    records = []
    for index, uid in enumerate(inventory):
        if uid.startswith("joint-"):
            stratum = "joint|latentcrack0|scoursev1"
        else:
            stratum = "fixture-anchor"
        records.append({
            "state_uid": uid,
            "state_seed_id": index + 1,
            "stratum": stratum,
            "partition": "",
        })
    for stratum in sorted({record["stratum"] for record in records}):
        members = sorted(
            record["state_uid"]
            for record in records
            if record["stratum"] == stratum
        )
        digest = hashlib.sha256(f"42|{stratum}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        permutation = rng.permutation(len(members))
        partitions = {}
        for position, member_position in enumerate(permutation):
            partitions[members[int(member_position)]] = (
                ("train", "test", "val", "train", "train")[position % 5]
            )
        for record in records:
            if record["stratum"] == stratum:
                record["partition"] = partitions[record["state_uid"]]
    identity = {
        "schema": "ttbi-semantic-state-identity-v2",
        "damage_seed": 1,
        "random_stream_schedule_version": "uid-named-substreams-v2",
        "state_stream_names": [
            "operations", "crack", "profile-state", "track", "profile-phase"
        ],
        "passage_stream_names": ["profile-passage", "oor-passage"],
        "passages_per_state": 50,
        "state_uid_count": len(inventory),
        "state_uid_inventory": inventory,
        "state_uid_inventory_sha256": _sha(inventory),
        "state_uid_row_order": row_order,
        "state_uid_row_order_sha256": _sha(row_order),
        "state_seed_id_by_uid_sha256": _sha([
            [record["state_uid"], record["state_seed_id"]]
            for record in records
        ]),
        "state_named_stream_by_uid_sha256": _sha(
            ["fixture-state-named-streams"]
        ),
        "passage_named_stream_by_uid_sha256": _sha(
            ["fixture-passage-named-streams"]
        ),
        "joint_state_uid_count": len(joint),
        "joint_state_uid_inventory": joint,
        "joint_state_uid_inventory_sha256": _sha(joint),
        "family_counts": {
            "target_healthy": 50,
            "scour_only": 50,
            "bearing_only": 50,
            "nuisance_only": 50,
            "joint": 250,
        },
        "latent_design_root_sha256": _sha(["fixture-latent-design"]),
        "state_identity_root_sha256": _sha(["fixture-active-design"]),
    }
    split = {
        "schema": "ttbi-semantic-split-v1",
        "seed": 42,
        "assignment_by_uid": records,
        "assignment_by_uid_sha256": _sha(records),
        "partition_counts": {
            name: sum(record["partition"] == name for record in records)
            for name in ("train", "val", "test")
        },
    }
    return identity, split


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_csv(path: Path, mutate) -> None:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    mutate(rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hyperparameter_fixture(
    *,
    protocol_core_hash: str,
    anchor_protocol_hash: str,
    anchor_dataset: str,
) -> tuple[dict, str, dict[tuple[str, int], dict], dict, bytes]:
    environment = {
        "schema": "ttbi-execution-environment-v1",
        "host": {
            "hostname": "fixture-host",
            "machine": "AMD64",
            "system": "Windows",
            "platform": "Windows-fixture",
        },
        "accelerator": {
            "backend": "cuda",
            "device_index": 0,
            "name": "Fixture GPU",
            "uuid": "GPU-fixture-a",
            "compute_capability": {"major": 8, "minor": 9},
            "sm_count": 36,
            "total_memory_bytes": 8_000_000_000,
            "driver_version": "fixture-driver",
        },
        "numeric_stack": {
            "torch_version": "fixture-torch",
            "cuda_runtime_version": "12.8",
            "cudnn_version": 90701,
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_enabled": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": True,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    environment_sha = execution_environment_sha256(environment)
    runtime = {
        "schema": "ttbi-execution-runtime-binding-v1",
        "execution_block": "l60",
        "anchor_stage": "s0_scour",
        "execution_environment_sha256": environment_sha,
        "execution_environment_descriptor": environment,
    }
    execution_receipt = {
        "schema": "ttbi-execution-block-receipt-v1",
        "execution_block": "l60",
        "anchor_stage": "s0_scour",
        "protocol_core_hash": protocol_core_hash,
        "run_tag": RUN_TAG,
        "execution_runtime": runtime,
    }
    receipt_payload = canonical_json(execution_receipt).encode("ascii") + b"\n"
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    entries = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for seed in HYPERPARAMETER_SEEDS:
            params = {
                "lr": 0.001,
                "weight_decay": 0.0001,
                "n_conv_layers": 2,
                "n_dense_layers": 1,
                "n_filters_l0": 16,
                "kernel_size_l0": 2,
                "pooling_l0": True,
                "n_filters_l1": 16,
                "kernel_size_l1": 2,
                "pooling_l1": True,
                "n_dense_units_l0": 32,
                "dropout_l0": 0.1,
            }
            if architecture == "PAA_LSTM_NHiTS":
                params.update({
                    "lstm_num_layers": 1,
                    "lstm_hidden_size": 32,
                })
            if architecture != "PAA_CNN":
                params["nhits_pool_rates_key"] = "1_2_4"
            identity = {
                "schema": STUDY_IDENTITY_SCHEMA,
                "execution_block": "l60",
                "anchor_stage": "s0_scour",
                "architecture": architecture,
                "seed": int(seed),
                "active_dofs": list(range(8)),
                "study_name": f"fixture-{architecture}-{seed}",
                "protocol_hash": anchor_protocol_hash,
                "dataset": anchor_dataset,
                "model_name": f"fixture-model-{architecture}",
                "execution_environment_sha256": environment_sha,
                "campaign_run_tag": RUN_TAG,
                "execution_receipt_sha256": receipt_sha,
                "study_protocol_record_sha256": _sha({
                    "architecture": architecture,
                    "seed": int(seed),
                }),
                "effective_n_trials": 100,
                "effective_use_pruner": True,
                "terminal_counts": {
                    "COMPLETE": 100,
                    "PRUNED": 0,
                    "FAIL": 0,
                    "RUNNING": 0,
                    "WAITING": 0,
                    "total": 100,
                },
                "best_trial_number": 0,
                "best_trial_value": 1.0 + architecture_index,
                "best_params_sha256":
                    hyperparameter_json_sha256(params),
            }
            entries.append(build_manifest_entry(
                study_identity=identity,
                params=params,
            ))
    manifest, manifest_sha = build_manifest(
        entries,
        execution_runtime=runtime,
        protocol_core_hash=protocol_core_hash,
        anchor_protocol_hash=anchor_protocol_hash,
        anchor_dataset=anchor_dataset,
        run_tag=RUN_TAG,
        execution_receipt_sha256=receipt_sha,
    )
    sources = {
        (entry["architecture"], int(entry["seed"])): {
            "execution_block": "l60",
            "anchor_stage": "s0_scour",
            "architecture": entry["architecture"],
            "seed": int(entry["seed"]),
            "study_identity_sha256": entry["study_identity_sha256"],
            "params_sha256": entry["params_sha256"],
        }
        for entry in manifest["entries"]
    }
    return (
        manifest,
        manifest_sha,
        sources,
        execution_receipt,
        receipt_payload,
    )


HYPERPARAMETER_PATH_BY_CHAMPION: dict[str, str] = {}
EXECUTION_RECEIPT_PATH_BY_CHAMPION: dict[str, str] = {}
REFERENCE_SHA_BY_CHAMPION: dict[str, str] = {}


def build_fixture(root: Path) -> tuple[dict[str, str], str]:
    global RECEIPT
    identity, split = _identity_and_split()
    core = {
        "protocol_version": 999,
        "code": {"schema_tag": EXPECTED_PROTOCOL_SCHEMA_TAG},
        "optuna": {"seeds": SEEDS},
        "selection": {
            "statistical_inference": {
                "cross_rung_crn": cri.CROSS_RUNG_INFERENCE_POLICY
            }
        },
    }
    core_hash = protocol_hash(core)
    anchor_dataset = campaign_stage_contract("s0_scour")["dataset"]
    anchor_descriptor = {
        "core": core,
        "rung": {
            "stage": "s0_scour",
            "dataset": anchor_dataset,
            "execution_block": "l60",
            "execution_anchor": "s0_scour",
            "dataset_provenance": {
                "state_identity": identity,
                "semantic_split": split,
            },
        },
    }
    anchor_protocol_hash = protocol_hash(anchor_descriptor)
    (
        hyperparameter_manifest,
        hyperparameter_sha,
        sources,
        execution_receipt,
        receipt_payload,
    ) = (
        _hyperparameter_fixture(
            protocol_core_hash=core_hash,
            anchor_protocol_hash=anchor_protocol_hash,
            anchor_dataset=anchor_dataset,
        )
    )
    hyperparameter_path = root / "hyperparameter_manifest.json"
    written_sha = write_manifest(
        hyperparameter_path,
        hyperparameter_manifest,
    )
    assert written_sha == hyperparameter_sha
    receipt_path = (root / "execution_l60.json").resolve()
    receipt_path.write_bytes(receipt_payload)
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    RECEIPT = receipt_sha
    champion = {
        "champion_arch": ARCH,
        "champion_pair": PAIR,
        "selected_at_stage": "s0_scour",
        "schema": EXPECTED_PROTOCOL_SCHEMA_TAG,
        "run_tag": execution_receipt["run_tag"],
        "seeds": SEEDS,
        "n_trials": HYPERPARAMETER_POLICY["anchor_hpo"]["n_trials"],
        "candidate_n_trials":
            HYPERPARAMETER_POLICY["frozen_singleton"]["n_trials"],
        "exhaustive_pairs": True,
        "protocol_core_hash": core_hash,
        "protocol_core": core,
        "protocol_hash": anchor_protocol_hash,
        "execution_receipt_sha256": receipt_sha,
        "capacity_preflight_receipt_sha256": CAPACITY_SHA,
        "hyperparameter_manifest_sha256": hyperparameter_sha,
        "hyperparameter_policy_sha256": hyperparameter_policy_sha256(),
        "execution_runtime": hyperparameter_manifest["execution_runtime"],
        "execution_environment_sha256":
            hyperparameter_manifest["execution_runtime"][
                "execution_environment_sha256"
            ],
        "dataset": anchor_dataset,
        "pair_select_metric": "inner_val_mse",
        "per_arch_median_single_dof_mse": {
            architecture: float(index + 1)
            for index, architecture in enumerate(ARCHITECTURES)
        },
        "frozen_selection_sha256": None,
    }
    champion_path = root / "champion.json"
    stage_dirs: dict[str, str] = {}
    s0_frozen_sha: str | None = None
    row_for_uid = {
        uid: index for index, uid in enumerate(identity["state_uid_row_order"])
    }
    outer_joint = [
        record["state_uid"]
        for record in split["assignment_by_uid"]
        if record["state_uid"].startswith("joint-")
        and record["partition"] == "test"
    ]
    for stage in cri.REGISTERED_L60_STAGES:
        summary = root / stage
        summary.mkdir()
        descriptor = {
            "core": core,
            "rung": {
                "stage": stage,
                "dataset": campaign_stage_contract(stage)["dataset"],
                "execution_block": "l60",
                "execution_anchor": "s0_scour",
                "dataset_provenance": {
                    "state_identity": identity,
                    "semantic_split": split,
                },
            },
        }
        full_hash = protocol_hash(descriptor)
        _write_json(summary / "protocol_descriptor.json", {
            "protocol_hash": full_hash,
            "protocol_core_hash": core_hash,
            "run_tag": RUN_TAG,
            "descriptor": descriptor,
            "execution_runtime": hyperparameter_manifest["execution_runtime"],
            "execution_receipt_sha256": receipt_sha,
            "capacity_preflight_receipt": {
                "receipt_sha256": CAPACITY_SHA,
            },
            "hyperparameter_manifest_sha256": hyperparameter_sha,
        })
        frozen_payload = {
            "stage": stage,
            "architecture": ARCH,
            "dofs": DOFS,
            "selected_pair": PAIR,
            "deployment_selection": False,
            "protocol_hash": full_hash,
            "protocol_core_hash": core_hash,
            "campaign_run_tag": RUN_TAG,
            # The anchor cannot cite the reference object whose digest includes
            # this frozen-selection digest. Followers are pinned after the
            # content-addressed anchor reference exists.
            "block_reference_manifest_sha256": None,
            "execution_runtime": hyperparameter_manifest["execution_runtime"],
            "execution_environment_sha256":
                hyperparameter_manifest["execution_runtime"][
                    "execution_environment_sha256"
                ],
            "execution_receipt_sha256": receipt_sha,
            "capacity_preflight_receipt_sha256": CAPACITY_SHA,
            "hyperparameter_manifest_sha256": hyperparameter_sha,
        }
        _write_json(summary / "frozen_selection.json", frozen_payload)
        if stage == "s0_scour":
            s0_frozen_sha = hyperparameter_json_sha256(frozen_payload)
        rows = []
        for uid_index, uid in enumerate(outer_joint):
            for seed_index, seed in enumerate(SEEDS):
                rows.append({
                    "stage": stage,
                    "protocol_hash": full_hash,
                    "protocol_core_hash": core_hash,
                    "execution_receipt_sha256": receipt_sha,
                    "campaign_run_tag": RUN_TAG,
                    "block_reference_manifest_sha256": None,
                    "hyperparameter_manifest_sha256": hyperparameter_sha,
                    "hyperparameter_source_json": canonical_json(
                        sources[(ARCH, seed)]
                    ),
                    "architecture": ARCH,
                    "dofs": DOFS,
                    "seed": seed,
                    "repeat": 0,
                    "state": row_for_uid[uid],
                    "state_uid": uid,
                    "state_seed_id": next(
                        record["state_seed_id"]
                        for record in split["assignment_by_uid"]
                        if record["state_uid"] == uid
                    ),
                    "family": "joint",
                    "scour_mse": (
                        10.0 + 0.01 * uid_index + 0.1 * seed_index
                        + OFFSETS[stage]
                    ),
                })
        with open(
            summary / "outer_test_state_metrics.csv",
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        stage_dirs[stage] = str(summary)
    assert s0_frozen_sha is not None
    champion["frozen_selection_sha256"] = s0_frozen_sha
    _write_json(champion_path, champion)
    reference_sha = hyperparameter_json_sha256(champion)
    for stage, summary_value in stage_dirs.items():
        protocol_path = Path(summary_value) / "protocol_descriptor.json"
        record = _read_json(protocol_path)
        record["block_reference_manifest_sha256"] = reference_sha
        _write_json(protocol_path, record)
        if stage != "s0_scour":
            frozen_path = Path(summary_value) / "frozen_selection.json"
            frozen = _read_json(frozen_path)
            frozen["block_reference_manifest_sha256"] = reference_sha
            _write_json(frozen_path, frozen)
            _rewrite_csv(
                Path(summary_value) / "outer_test_state_metrics.csv",
                lambda rows: [
                    row.__setitem__(
                        "block_reference_manifest_sha256", reference_sha
                    )
                    for row in rows
                ],
            )
    HYPERPARAMETER_PATH_BY_CHAMPION[str(champion_path)] = str(
        hyperparameter_path.resolve()
    )
    EXECUTION_RECEIPT_PATH_BY_CHAMPION[str(champion_path)] = str(receipt_path)
    REFERENCE_SHA_BY_CHAMPION[str(champion_path)] = reference_sha
    return stage_dirs, str(champion_path)


def analyze_fixture(
    summaries: dict[str, str],
    champion: str,
    output: Path,
) -> dict:
    return cri.analyze_registered_l60_contrasts(
        summaries,
        champion,
        HYPERPARAMETER_PATH_BY_CHAMPION[champion],
        EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion],
        output,
        expected_block_reference_sha256=
            REFERENCE_SHA_BY_CHAMPION[champion],
    )


def repin_reference(
    summaries: dict[str, str],
    champion: str,
    *,
    update_external_root: bool,
) -> str:
    """Update every internal stage pin after an intentional fixture mutation."""

    manifest = _read_json(Path(champion))
    reference_sha = hyperparameter_json_sha256(manifest)
    for summary_value in summaries.values():
        protocol_path = Path(summary_value) / "protocol_descriptor.json"
        record = _read_json(protocol_path)
        record["block_reference_manifest_sha256"] = reference_sha
        _write_json(protocol_path, record)
    if update_external_root:
        REFERENCE_SHA_BY_CHAMPION[champion] = reference_sha
    return reference_sha


def fresh(name: str) -> tuple[Path, dict[str, str], str]:
    path = WORK / name
    path.mkdir()
    stage_dirs, champion = build_fixture(path)
    return path, stage_dirs, champion


def coherently_mutate_rung_descriptor(
    summaries: dict[str, str],
    stage: str,
    mutate,
) -> None:
    """Mutate one rung descriptor and update every dependent full-hash stamp."""

    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    record = _read_json(protocol_path)
    mutate(record["descriptor"]["rung"])
    new_hash = protocol_hash(record["descriptor"])
    record["protocol_hash"] = new_hash
    _write_json(protocol_path, record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["protocol_hash"] = new_hash
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [
            row.__setitem__("protocol_hash", new_hash) for row in rows
        ],
    )


WORK = Path(tempfile.mkdtemp(prefix="crossrungcheck_"))
try:
    rng = np.random.default_rng(91)
    left_fixture = rng.uniform(0.0, 5.0, size=(17, 3))
    right_fixture = left_fixture + rng.normal(0.2, 0.4, size=(17, 3))
    vector_estimate, vector_draws = cri._edge_bootstrap(
        left_fixture,
        right_fixture,
        n_boot=257,
        seed=123,
    )
    reference_rng = np.random.default_rng(123)
    reference_draws = []
    for _ in range(257):
        positions = reference_rng.integers(0, 17, size=17)
        reference_draws.append(
            cri._statistic(right_fixture[positions])
            - cri._statistic(left_fixture[positions])
        )
    check(
        "bounded vectorized bootstrap equals scalar reference implementation",
        abs(
            vector_estimate
            - (cri._statistic(right_fixture) - cri._statistic(left_fixture))
        ) < 1e-15
        and np.allclose(
            vector_draws,
            np.asarray(reference_draws),
            rtol=0.0,
            atol=1e-15,
        ),
    )

    root, summaries, champion = fresh("valid")
    result = analyze_fixture(
        summaries, champion, root / "output"
    )
    primary = [
        row for row in result["summary_rows"]
        if row["analysis_role"] == "primary preregistered edge"
    ]
    expected = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    check(
        "exactly seven immutable primary edges",
        [row["edge"] for row in primary]
        == [f"{left}->{right}" for left, right in cri.REGISTERED_L60_EDGES],
    )
    check(
        "synthetic known edge deltas recovered exactly",
        all(
            abs(float(row["estimate"]) - wanted) < 1e-12
            for row, wanted in zip(primary, expected, strict=True)
        ),
    )
    check(
        "pointwise and Bonferroni-familywise intervals both emitted",
        all(
            abs(float(row["pointwise_ci95_lo"]) - wanted) < 1e-12
            and abs(float(row["pointwise_ci95_hi"]) - wanted) < 1e-12
            and abs(float(row["familywise_bonferroni_ci95_lo"]) - wanted)
            < 1e-12
            and abs(float(row["familywise_bonferroni_ci95_hi"]) - wanted)
            < 1e-12
            for row, wanted in zip(primary, expected, strict=True)
        ),
    )
    interaction = next(
        row for row in result["summary_rows"]
        if row["analysis_role"] == "secondary exploratory interaction"
    )
    check(
        "2x2 bearing×crack DiD recovered and excluded from primary family",
        abs(float(interaction["estimate"]) - 1.0) < 1e-12
        and interaction["familywise_bonferroni_ci95_lo"] == "",
    )
    check(
        "manifest binds inputs, exact UIDs, receipt, and registered seeds",
        result["manifest"]["registered_seeds"] == SEEDS
        and result["manifest"]["execution_receipt_sha256"] == RECEIPT
        and result["manifest"]["execution_receipt"]["file_sha256"] == RECEIPT
        and result["manifest"]["execution_receipt"]["run_tag"]
        == RUN_TAG
        and result["manifest"]["hyperparameter_manifest"][
            "execution_receipt_sha256"
        ] == RECEIPT
        and result["manifest"]["hyperparameter_manifest"]["run_tag"] == RUN_TAG
        and len(result["manifest"]["outer_joint_state_uids"]) == 50
        and set(result["manifest"]["stage_inputs"])
        == set(cri.REGISTERED_L60_STAGES),
    )
    check(
        "analysis manifest preserves the independently supplied trust root",
        result["manifest"]["externally_supplied_block_reference_sha256"]
        == REFERENCE_SHA_BY_CHAMPION[champion]
        == result["manifest"]["reference"][
            "externally_supplied_block_reference_sha256"
        ],
    )
    anchor_frozen = _read_json(
        Path(summaries["s0_scour"]) / "frozen_selection.json"
    )
    follower_frozen = _read_json(
        Path(summaries["s11_bear"]) / "frozen_selection.json"
    )
    with open(
        Path(summaries["s0_scour"]) / "outer_test_state_metrics.csv",
        encoding="utf-8",
        newline="",
    ) as handle:
        anchor_rows = list(csv.DictReader(handle))
    with open(
        Path(summaries["s11_bear"]) / "outer_test_state_metrics.csv",
        encoding="utf-8",
        newline="",
    ) as handle:
        follower_rows = list(csv.DictReader(handle))
    check(
        "valid evidence uses explicit anchor null and exact follower reference pin",
        "block_reference_manifest_sha256" in anchor_frozen
        and anchor_frozen["block_reference_manifest_sha256"] is None
        and anchor_frozen["campaign_run_tag"] == RUN_TAG
        and all(
            row["block_reference_manifest_sha256"] == ""
            and row["campaign_run_tag"] == RUN_TAG
            for row in anchor_rows
        )
        and follower_frozen["block_reference_manifest_sha256"]
        == REFERENCE_SHA_BY_CHAMPION[champion]
        and follower_frozen["campaign_run_tag"] == RUN_TAG
        and all(
            row["block_reference_manifest_sha256"]
            == REFERENCE_SHA_BY_CHAMPION[champion]
            and row["campaign_run_tag"] == RUN_TAG
            for row in follower_rows
        ),
    )

    path, summaries, champion = fresh("single_snapshot")
    expected_input_paths = {
        Path(champion).resolve(),
        Path(HYPERPARAMETER_PATH_BY_CHAMPION[champion]).resolve(),
        Path(EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion]).resolve(),
    }
    for summary_value in summaries.values():
        summary = Path(summary_value)
        expected_input_paths.update({
            (summary / "protocol_descriptor.json").resolve(),
            (summary / "frozen_selection.json").resolve(),
            (summary / "outer_test_state_metrics.csv").resolve(),
        })
    original_safe_read = cri._read_regular_file
    snapshot_reads: dict[Path, int] = {}

    def counted_safe_read(path_value, *args, **kwargs):
        resolved = Path(path_value).resolve()
        snapshot_reads[resolved] = snapshot_reads.get(resolved, 0) + 1
        return original_safe_read(path_value, *args, **kwargs)

    cri._read_regular_file = counted_safe_read
    try:
        analyze_fixture(summaries, champion, path / "out")
    finally:
        cri._read_regular_file = original_safe_read
    check(
        "every registered analysis input is parsed and hashed from one snapshot",
        set(snapshot_reads) == expected_input_paths
        and all(snapshot_reads[path_value] == 1
                for path_value in expected_input_paths),
        str({
            str(path_value): snapshot_reads.get(path_value, 0)
            for path_value in expected_input_paths
            if snapshot_reads.get(path_value, 0) != 1
        }),
    )

    path, summaries, champion = fresh("anchor_missing_explicit_reference")
    frozen_path = Path(summaries["s0_scour"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    del frozen["block_reference_manifest_sha256"]
    _write_json(frozen_path, frozen)
    raises_with_message(
        "anchor frozen selection must carry an explicit null reference field",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "frozen selection provenance mismatch",
    )

    path, summaries, champion = fresh("anchor_frozen_self_reference")
    frozen_path = Path(summaries["s0_scour"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["block_reference_manifest_sha256"] = (
        REFERENCE_SHA_BY_CHAMPION[champion]
    )
    _write_json(frozen_path, frozen)
    raises_with_message(
        "anchor frozen selection cannot self-reference the later manifest",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "frozen selection provenance mismatch",
    )

    path, summaries, champion = fresh("follower_frozen_wrong_run")
    frozen_path = Path(summaries["s11_bear"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["campaign_run_tag"] = "coherent-but-foreign-run"
    _write_json(frozen_path, frozen)
    raises_with_message(
        "follower frozen selection from another campaign run is rejected",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "frozen selection provenance mismatch",
    )

    path, summaries, champion = fresh("follower_frozen_wrong_reference")
    frozen_path = Path(summaries["s12_crack"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["block_reference_manifest_sha256"] = "0" * 64
    _write_json(frozen_path, frozen)
    raises_with_message(
        "follower frozen selection citing another block reference is rejected",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "frozen selection provenance mismatch",
    )

    path, summaries, champion = fresh("anchor_row_nonempty_reference")
    metric_path = Path(summaries["s0_scour"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "block_reference_manifest_sha256",
            REFERENCE_SHA_BY_CHAMPION[champion],
        ),
    )
    raises_with_message(
        "anchor metric row must retain the canonical empty anti-cycle reference",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "anchor-null/follower-pinned block-reference lineage",
    )

    path, summaries, champion = fresh("follower_row_wrong_reference")
    metric_path = Path(summaries["s13_bearcrack"]) / \
        "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "block_reference_manifest_sha256", "0" * 64
        ),
    )
    raises_with_message(
        "follower metric row citing another block reference is rejected",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "anchor-null/follower-pinned block-reference lineage",
    )

    path, summaries, champion = fresh("unselected_row_wrong_run")
    metric_path = Path(summaries["s14_prof"]) / "outer_test_state_metrics.csv"

    def append_unselected_foreign_run(rows):
        foreign = dict(rows[0])
        foreign["architecture"] = "deployment_winner"
        foreign["campaign_run_tag"] = "foreign-run"
        rows.append(foreign)

    _rewrite_csv(metric_path, append_unselected_foreign_run)
    raises_with_message(
        "even an unselected metric row from another campaign run is rejected",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "authenticated campaign run tag",
    )

    path, summaries, champion = fresh("coherent_wrong_stage_reference")
    stage = "s15_track"
    wrong_reference = "0" * 64
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    protocol_record = _read_json(protocol_path)
    protocol_record["block_reference_manifest_sha256"] = wrong_reference
    _write_json(protocol_path, protocol_record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["block_reference_manifest_sha256"] = wrong_reference
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [
            row.__setitem__(
                "block_reference_manifest_sha256", wrong_reference
            )
            for row in rows
        ],
    )
    raises_with_message(
        "coherently substituted protocol/frozen/row reference still fails "
        "the external pin",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "does not pin the supplied canonical block-reference",
    )

    path, summaries, champion = fresh("coherent_wrong_stage_run")
    stage = "s16_all"
    wrong_run = "coherent-but-foreign-run"
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    protocol_record = _read_json(protocol_path)
    protocol_record["run_tag"] = wrong_run
    _write_json(protocol_path, protocol_record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["campaign_run_tag"] = wrong_run
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [
            row.__setitem__("campaign_run_tag", wrong_run)
            for row in rows
        ],
    )
    raises_with_message(
        "coherently substituted protocol/frozen/row campaign run still fails "
        "the authenticated receipt",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "authenticated run/HPO/capacity lineage",
    )

    path, summaries, champion = fresh("anchor_frozen_bool_pair")
    frozen_path = Path(summaries["s0_scour"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["selected_pair"] = [True, 3]
    _write_json(frozen_path, frozen)
    manifest = _read_json(Path(champion))
    manifest["frozen_selection_sha256"] = (
        hyperparameter_json_sha256(frozen)
    )
    _write_json(Path(champion), manifest)
    repin_reference(summaries, champion, update_external_root=True)
    raises_with_message(
        "re-pinned anchor frozen selection rejects boolean DOF coercion",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "strict integer registered DOFs",
    )

    path, summaries, champion = fresh("coherent_reference_substitution")
    frozen_path = Path(summaries["s0_scour"]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["selected_pair"] = [0, 2]
    _write_json(frozen_path, frozen)
    manifest = _read_json(Path(champion))
    manifest["champion_pair"] = [0, 2]
    manifest["frozen_selection_sha256"] = (
        hyperparameter_json_sha256(frozen)
    )
    _write_json(Path(champion), manifest)
    repin_reference(
        summaries, champion, update_external_root=False
    )
    raises_with_message(
        "coherent champion+frozen+seven-pin substitution still fails external root",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "independently supplied canonical SHA-256",
    )

    path, summaries, champion = fresh("reference_extra_field")
    manifest = _read_json(Path(champion))
    manifest["unexpected"] = True
    _write_json(Path(champion), manifest)
    repin_reference(summaries, champion, update_external_root=True)
    raises_with_message(
        "re-pinned reference with an extra field fails the shared contract",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "fields differ from the exact contract",
    )

    path, summaries, champion = fresh("reference_nonexhaustive")
    manifest = _read_json(Path(champion))
    manifest["exhaustive_pairs"] = False
    _write_json(Path(champion), manifest)
    repin_reference(summaries, champion, update_external_root=True)
    raises_with_message(
        "re-pinned non-exhaustive anchor reference is rejected",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "exhaustive anchor sweep",
    )

    path, summaries, champion = fresh("reference_bool_pair")
    manifest = _read_json(Path(champion))
    manifest["champion_pair"] = [True, 3]
    _write_json(Path(champion), manifest)
    repin_reference(summaries, champion, update_external_root=True)
    raises_with_message(
        "re-pinned boolean DOF cannot masquerade as integer DOF 1",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "strict integer registered DOFs",
    )

    path, summaries, champion = fresh("reference_bool_budget")
    manifest = _read_json(Path(champion))
    manifest["candidate_n_trials"] = True
    _write_json(Path(champion), manifest)
    repin_reference(summaries, champion, update_external_root=True)
    raises_with_message(
        "re-pinned boolean candidate budget cannot masquerade as integer one",
        lambda: analyze_fixture(summaries, champion, path / "out"),
        "registered integer budget",
    )

    path, summaries, champion = fresh("wrong_stage")
    stage = "s11_bear"
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    record = _read_json(protocol_path)
    record["descriptor"]["rung"]["stage"] = "s21_scour4"
    new_hash = protocol_hash(record["descriptor"])
    record["protocol_hash"] = new_hash
    _write_json(protocol_path, record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["protocol_hash"] = new_hash
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [row.__setitem__("protocol_hash", new_hash) for row in rows],
    )
    raises(
        "coherently rehashed wrong stage rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("missing_seed")
    metric_path = Path(summaries["s12_crack"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows.__setitem__(
            slice(None), [row for row in rows if row["seed"] != str(SEEDS[-1])]
        ),
    )
    raises(
        "missing registered seed cells rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("champion_wrong_hyperparameter_sha")
    champion_value = _read_json(Path(champion))
    champion_value["hyperparameter_manifest_sha256"] = "0" * 64
    _write_json(Path(champion), champion_value)
    raises(
        "champion citing another hyperparameter manifest rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("row_wrong_hyperparameter_sha")
    metric_path = Path(summaries["s12_crack"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "hyperparameter_manifest_sha256", "0" * 64
        ),
    )
    raises(
        "outer row citing another hyperparameter manifest rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("row_noncanonical_source")
    metric_path = Path(summaries["s13_bearcrack"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "hyperparameter_source_json",
            " " + rows[0]["hyperparameter_source_json"],
        ),
    )
    raises(
        "noncanonical per-row hyperparameter source JSON rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("row_wrong_params_source")
    metric_path = Path(summaries["s14_prof"]) / "outer_test_state_metrics.csv"

    def mutate_params_source(rows):
        source = json.loads(rows[0]["hyperparameter_source_json"])
        source["params_sha256"] = "0" * 64
        rows[0]["hyperparameter_source_json"] = canonical_json(source)

    _rewrite_csv(metric_path, mutate_params_source)
    raises(
        "row citing wrong per-seed parameter hash rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("noncanonical_hyperparameter_file")
    hyperparameter_path = Path(HYPERPARAMETER_PATH_BY_CHAMPION[champion])
    hyperparameter_path.write_bytes(
        hyperparameter_path.read_bytes() + b"\n"
    )
    raises(
        "noncanonical hyperparameter manifest file bytes rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("non_full_array_hpo_identity")
    hyperparameter_path = Path(HYPERPARAMETER_PATH_BY_CHAMPION[champion])
    hyperparameter_value = _read_json(hyperparameter_path)
    hyperparameter_value["entries"][0]["study_identity"]["active_dofs"] = (
        list(range(7))
    )
    hyperparameter_path.write_text(
        canonical_json(hyperparameter_value),
        encoding="ascii",
    )
    raises(
        "HPO identity not calibrated on the full eight-DOF array rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("hpo_wrong_receipt_binding")
    hyperparameter_path = Path(HYPERPARAMETER_PATH_BY_CHAMPION[champion])
    hyperparameter_value = _read_json(hyperparameter_path)
    hyperparameter_value["execution_receipt_sha256"] = "d" * 64
    hyperparameter_path.write_text(
        canonical_json(hyperparameter_value),
        encoding="ascii",
    )
    raises(
        "HPO manifest bound to another execution receipt rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("hpo_wrong_run_tag")
    hyperparameter_path = Path(HYPERPARAMETER_PATH_BY_CHAMPION[champion])
    hyperparameter_value = _read_json(hyperparameter_path)
    hyperparameter_value["run_tag"] = "another-campaign-run"
    hyperparameter_path.write_text(
        canonical_json(hyperparameter_value),
        encoding="ascii",
    )
    raises(
        "HPO manifest bound to another campaign run_tag rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("missing_external_receipt")
    Path(EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion]).unlink()
    raises(
        "missing mandatory external execution receipt rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("noncanonical_external_receipt")
    receipt_path = Path(EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion])
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    raises(
        "noncanonical external execution receipt bytes rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("relative_external_hpo")
    absolute_hpo = HYPERPARAMETER_PATH_BY_CHAMPION[champion]
    HYPERPARAMETER_PATH_BY_CHAMPION[champion] = os.path.relpath(
        absolute_hpo, Path.cwd()
    )
    raises(
        "non-durable relative hyperparameter manifest path rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("relative_external_receipt")
    absolute_receipt = EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion]
    EXECUTION_RECEIPT_PATH_BY_CHAMPION[champion] = os.path.relpath(
        absolute_receipt, Path.cwd()
    )
    raises(
        "non-durable relative execution receipt path rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("duplicate_csv_header")
    metric_path = Path(summaries["s15_track"]) / "outer_test_state_metrics.csv"
    metric_lines = metric_path.read_text(encoding="utf-8").splitlines()
    metric_lines[0] += ",scour_mse"
    metric_lines[1:] = [line + ",999" for line in metric_lines[1:]]
    metric_path.write_text(
        "\n".join(metric_lines) + "\n",
        encoding="utf-8",
    )
    raises(
        "duplicate outer-metric CSV header rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("wrong_arch_pair")
    metric_path = Path(summaries["s13_bearcrack"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: [
            (row.__setitem__("architecture", "deployment_winner"),
             row.__setitem__("dofs", "Wheel1_Vert+Wheel2_Vert"))
            for row in rows
        ],
    )
    raises(
        "architecture/pair or deployment-winner substitution rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("wrong_protocol")
    stage = "s14_prof"
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    record = _read_json(protocol_path)
    record["descriptor"]["core"]["mutated_training_knob"] = True
    new_core = protocol_hash(record["descriptor"]["core"])
    new_full = protocol_hash(record["descriptor"])
    record["protocol_core_hash"] = new_core
    record["protocol_hash"] = new_full
    _write_json(protocol_path, record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["protocol_core_hash"] = new_core
    frozen["protocol_hash"] = new_full
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [
            (row.__setitem__("protocol_core_hash", new_core),
             row.__setitem__("protocol_hash", new_full))
            for row in rows
        ],
    )
    raises(
        "coherent but different protocol rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("wrong_receipt")
    stage = "s15_track"
    bad_receipt = "c" * 64
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    record = _read_json(protocol_path)
    record["execution_receipt_sha256"] = bad_receipt
    _write_json(protocol_path, record)
    frozen_path = Path(summaries[stage]) / "frozen_selection.json"
    frozen = _read_json(frozen_path)
    frozen["execution_receipt_sha256"] = bad_receipt
    _write_json(frozen_path, frozen)
    _rewrite_csv(
        Path(summaries[stage]) / "outer_test_state_metrics.csv",
        lambda rows: [
            row.__setitem__("execution_receipt_sha256", bad_receipt)
            for row in rows
        ],
    )
    raises(
        "different execution receipt rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("different_runtime")
    stage = "s15_track"
    protocol_path = Path(summaries[stage]) / "protocol_descriptor.json"
    record = _read_json(protocol_path)
    foreign_runtime = copy.deepcopy(record["execution_runtime"])
    foreign_runtime["execution_environment_descriptor"]["host"][
        "hostname"
    ] = "coherent-foreign-host"
    foreign_runtime["execution_environment_sha256"] = (
        execution_environment_sha256(
            foreign_runtime["execution_environment_descriptor"]
        )
    )
    record["execution_runtime"] = foreign_runtime
    _write_json(protocol_path, record)
    raises(
        "coherent foreign execution runtime rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("deployment_manifest")
    manifest = _read_json(Path(champion))
    manifest["selected_at_stage"] = "s16_all"
    manifest["champion_arch"] = "deployment_winner"
    _write_json(Path(champion), manifest)
    raises(
        "deployment champion manifest substitution rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("valid_reference_value_substitution")
    manifest = _read_json(Path(champion))
    manifest["champion_arch"] = "PAA_NHiTS"
    manifest["champion_pair"] = [0, 2]
    _write_json(Path(champion), manifest)
    raises(
        "another registered architecture/pair cannot replace the pinned reference",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("renamed_uid")
    metric_path = Path(summaries["s16_all"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "state_uid", rows[0]["state_uid"] + "-renamed"
        ),
    )
    raises(
        "renamed/misaligned outer StateUID rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("wrong_state_stream_root")
    metric_path = Path(summaries["s13_bearcrack"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__(
            "state_seed_id", str(int(rows[0]["state_seed_id"]) + 1)
        ),
    )
    raises(
        "UID-misaligned state-stream root rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("different_named_streams")
    coherently_mutate_rung_descriptor(
        summaries,
        "s14_prof",
        lambda rung: rung["dataset_provenance"]["state_identity"].__setitem__(
            "state_named_stream_by_uid_sha256", "e" * 64
        ),
    )
    raises(
        "coherently rehashed cross-rung named-stream mutation rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("coherent_nonregistered_partition")

    def mutate_partition(rung):
        split = rung["dataset_provenance"]["semantic_split"]
        record = next(
            item for item in split["assignment_by_uid"]
            if item["partition"] == "train"
        )
        record["partition"] = "val"
        split["assignment_by_uid_sha256"] = _sha(
            split["assignment_by_uid"]
        )
        split["partition_counts"] = {
            partition: sum(
                item["partition"] == partition
                for item in split["assignment_by_uid"]
            )
            for partition in ("train", "val", "test")
        }

    coherently_mutate_rung_descriptor(
        summaries,
        "s14_prof",
        mutate_partition,
    )
    raises(
        "coherently rehashed nonregistered UID partition rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("different_latent_design")
    coherently_mutate_rung_descriptor(
        summaries,
        "s15_track",
        lambda rung: rung["dataset_provenance"]["state_identity"].__setitem__(
            "latent_design_root_sha256", "f" * 64
        ),
    )
    raises(
        "coherently rehashed cross-rung latent-design mutation rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("negative_mse")
    metric_path = Path(summaries["s14_prof"]) / "outer_test_state_metrics.csv"
    _rewrite_csv(
        metric_path,
        lambda rows: rows[0].__setitem__("scour_mse", "-0.001"),
    )
    raises(
        "physically impossible negative MSE rejected",
        lambda: analyze_fixture(
            summaries, champion, path / "out"
        ),
    )

    path, summaries, champion = fresh("wrong_edge")
    old_edges = cri.REGISTERED_L60_EDGES
    try:
        cri.REGISTERED_L60_EDGES = old_edges[:-1] + (
            ("s14_prof", "s16_all"),
        )
        raises(
            "runtime primary-edge mutation rejected",
            lambda: analyze_fixture(
                summaries, champion, path / "out"
            ),
        )
    finally:
        cri.REGISTERED_L60_EDGES = old_edges

    path, summaries, champion = fresh("lower_bootstrap_budget")
    old_bootstrap_n = cri.CROSS_RUNG_BOOTSTRAP_N
    try:
        cri.CROSS_RUNG_BOOTSTRAP_N = 2000
        raises(
            "lowered familywise bootstrap budget rejected",
            lambda: analyze_fixture(
                summaries, champion, path / "out"
            ),
        )
    finally:
        cri.CROSS_RUNG_BOOTSTRAP_N = old_bootstrap_n

    path, summaries, champion = fresh("pointwise_as_familywise")
    old_familywise_edge_coverage = cri.CROSS_RUNG_BONFERRONI_EDGE_COVERAGE
    try:
        cri.CROSS_RUNG_BONFERRONI_EDGE_COVERAGE = 0.95
        raises(
            "pointwise alpha substituted for familywise alpha rejected",
            lambda: analyze_fixture(
                summaries, champion, path / "out"
            ),
        )
    finally:
        cri.CROSS_RUNG_BONFERRONI_EDGE_COVERAGE = (
            old_familywise_edge_coverage
        )
finally:
    shutil.rmtree(WORK, ignore_errors=True)


print()
print(
    "CROSS-RUNG INFERENCE: ALL PASS"
    if FAILURES == 0
    else f"CROSS-RUNG INFERENCE: {FAILURES} CHECK(S) FAILED"
)
sys.exit(1 if FAILURES else 0)
