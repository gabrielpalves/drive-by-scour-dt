"""
training/pipeline.py
====================
Two functions that sit at the top of the training call stack:

    execute_ablation_pipeline  - the master loop that drives every step for
                                 every model in the ablation grid: Optuna
                                 optimisation, confusion matrix, DT package
                                 export, stochastic stress-test, and slice plots.

    export_digital_twin_package - bundles the champion weights, scaler, and
                                  architecture metadata into the three files
                                  that drive_by_DT.py loads at startup.

These are the only functions in the training package that the ablation
notebook calls directly.  Everything else is an implementation detail
imported by these two functions.

Imported by:
    ablation.ipynb - execute_ablation_pipeline (multiple phases),
                     export_digital_twin_package (called internally but also
                     available for manual re-export after the fact).
"""

import glob
import hashlib
import json
import os
import shutil

import joblib
import optuna

from core          import task
from core.capacity_preflight import (
    ensure_capacity_preflight,
    validate_capacity_receipt,
)
from core.dataset  import get_or_create_cache, _cache_stem
from core.execution_environment import validate_execution_runtime
from core.hyperparameter_policy import (
    ANCHOR_HPO_MODE,
    FROZEN_SINGLETON_MODE,
    LEGACY_MODE,
    anchor_study_identity,
    build_manifest_entry,
    derive_execution_plan,
    validate_run_plan,
    validate_terminal_study,
)
from core.protocol import OPTUNA_PROTOCOL, protocol_hash
from core.utils    import set_global_seed, DOF_NAME_TO_IDX
from plotting.confusion        import plot_cached_confusion_matrix
from plotting.robustness_plots import generate_optuna_robustness_plots, plot_stochastic_summary
from training.robustness       import evaluate_stochastic_robustness, evaluate_parametric_robustness
from training.trainer          import (
    TRAIN_PROTOCOL,
    Objective,
    print_best_callback,
)

# Silence Optuna's per-trial log spam; callback handles champion announcements
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 used to bind exported weights to their Optuna trial."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str, value: dict) -> None:
    """Publish a JSON sidecar atomically beside the artifact it describes."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4, sort_keys=True)
    os.replace(tmp, path)


def _canonical_json_value(value):
    """The exact value JSON storage preserves (tuples become lists, etc.)."""
    return json.loads(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ))


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        _canonical_json_value(value),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_config_execution_runtime(config: dict) -> dict | None:
    """Return the exact runtime binding required by a protocol-hashed config."""

    runtime = config.get("execution_runtime")
    descriptor = config.get("protocol_descriptor")
    rung = descriptor.get("rung") if isinstance(descriptor, dict) else None
    core = descriptor.get("core") if isinstance(descriptor, dict) else None
    campaign_blocked = (
        (isinstance(core, dict) and "execution_blocking" in core)
        or (
            isinstance(rung, dict)
            and (
                "execution_block" in rung
                or "execution_anchor" in rung
            )
        )
    )
    if not config.get("protocol_hash") or not campaign_blocked:
        return (
            validate_execution_runtime(runtime)
            if runtime is not None else None
        )
    if runtime is None:
        raise RuntimeError(
            f"{config.get('name')}: protocol-hashed config lacks its "
            "execution-runtime attestation."
        )
    runtime = validate_execution_runtime(runtime)
    if not isinstance(rung, dict):
        raise RuntimeError(
            f"{config.get('name')}: execution-blocked protocol lacks its rung."
        )
    expected = (
        rung.get("execution_block"),
        rung.get("execution_anchor"),
    )
    actual = (
        runtime["execution_block"],
        runtime["anchor_stage"],
    )
    if actual != expected:
        raise RuntimeError(
            f"{config.get('name')}: execution runtime {actual!r} does not "
            f"match protocol rung {expected!r}."
        )
    return runtime


def _stamp_study_protocol(
    study: optuna.Study,
    *,
    config: dict,
    dataset_name: str,
    n_trials: int,
    epochs: int,
    sampler_seed: int,
    use_pruner: bool,
    hyperparameter_plan: dict | None = None,
    capacity_receipt: dict | None = None,
) -> None:
    """Persist and enforce the complete protocol record in Optuna storage.

    A protocol-hashed study with existing trials but no record is rejected:
    retroactively attaching metadata would not prove which protocol generated
    those trials. Fresh studies and legacy non-hashed callers remain supported.
    """
    if config.get("protocol_hash"):
        if "seed" not in config:
            raise RuntimeError(
                f"{study.study_name}: protocol-hashed configs must declare the "
                "training seed explicitly.")
        descriptor = config.get("protocol_descriptor")
        if descriptor is None or protocol_hash(descriptor) != config["protocol_hash"]:
            raise RuntimeError(
                f"{study.study_name}: protocol descriptor/hash pair is "
                "missing or internally inconsistent.")
    execution_runtime = _validated_config_execution_runtime(config)
    is_campaign = bool(config.get("protocol_hash"))
    if is_campaign:
        if hyperparameter_plan is None:
            raise RuntimeError(
                f"{study.study_name}: protocol-hashed study lacks its derived "
                "hyperparameter execution plan."
            )
        hyperparameter_plan = validate_run_plan(hyperparameter_plan)
        expected_plan = derive_execution_plan(
            config,
            dataset_name=dataset_name,
            requested_n_trials=hyperparameter_plan["requested_n_trials"],
            requested_use_pruner=(
                hyperparameter_plan["requested_use_pruner"]
            ),
            execution_runtime=execution_runtime,
        )
        if hyperparameter_plan != expected_plan:
            raise RuntimeError(
                f"{study.study_name}: hyperparameter plan does not derive "
                "from this config/runtime."
            )
        if capacity_receipt is None:
            raise RuntimeError(
                f"{study.study_name}: protocol-hashed study lacks its CUDA "
                "capacity-preflight receipt."
            )
        capacity_receipt = validate_capacity_receipt(
            capacity_receipt,
            expected_runtime=execution_runtime,
        )
        if n_trials != hyperparameter_plan["effective_n_trials"]:
            raise RuntimeError(
                f"{study.study_name}: caller passed a non-derived study budget."
            )
        if use_pruner is not hyperparameter_plan["effective_use_pruner"]:
            raise RuntimeError(
                f"{study.study_name}: caller passed a non-derived pruner mode."
            )
    record = {
        "schema": (
            "optuna-study-provenance-v4"
            if is_campaign else "optuna-study-provenance-v2"
        ),
        "protocol_hash": config.get("protocol_hash"),
        "protocol_descriptor": config.get("protocol_descriptor"),
        "execution_environment_sha256": (
            execution_runtime["execution_environment_sha256"]
            if execution_runtime is not None else None
        ),
        "execution_runtime": execution_runtime,
        "dataset": dataset_name,
        "model_name": config.get("name"),
        "seed": int(config.get("seed", sampler_seed)),
        "sampler_seed": int(sampler_seed),
        "n_trials": int(n_trials),
        "epochs": int(epochs),
        "use_pruner": bool(use_pruner),
        "optuna_protocol": OPTUNA_PROTOCOL,
    }
    if is_campaign:
        record.update({
            "hyperparameter_execution_plan": hyperparameter_plan,
            "hyperparameter_policy_sha256":
                hyperparameter_plan["policy_sha256"],
            "hyperparameter_mode": hyperparameter_plan["mode"],
            "hyperparameter_manifest_sha256":
                hyperparameter_plan["hyperparameter_manifest_sha256"],
            "hyperparameter_source":
                hyperparameter_plan["hyperparameter_source"],
            "capacity_preflight_receipt_sha256":
                capacity_receipt["receipt_sha256"],
            "capacity_preflight_policy_sha256":
                capacity_receipt["receipt"]["policy_sha256"],
            "campaign_run_tag":
                hyperparameter_plan["campaign_run_tag"],
            "execution_receipt_sha256":
                hyperparameter_plan["execution_receipt_sha256"],
            "block_reference_manifest_sha256":
                hyperparameter_plan["block_reference_manifest_sha256"],
        })
    # Optuna persists user attributes through JSON.  Canonicalise before both
    # storage and comparison because tuples (notably search-space bounds) return
    # from SQLite as lists.  Comparing the raw Python objects made every genuine
    # restart look like a protocol change even though their canonical JSON was
    # identical.
    record = _canonical_json_value(record)
    key = "ttbi_protocol_record"
    previous = study.user_attrs.get(key)
    if previous is None:
        if config.get("protocol_hash") and study.trials:
            raise RuntimeError(
                f"{study.study_name}: existing protocol-hashed study has "
                f"{len(study.trials)} trial(s) but no {key!r}. Its trials "
                "cannot be certified; start a fresh RUN_TAG/study.")
        study.set_user_attr(key, record)
        if is_campaign:
            study.set_user_attr(
                "ttbi_capacity_preflight_receipt", capacity_receipt
            )
    elif _canonical_json_value(previous) != record:
        raise RuntimeError(
            f"{study.study_name}: stored Optuna protocol record differs from "
            "the requested run. Refusing to mix trials; use a fresh RUN_TAG.")
    elif is_campaign:
        stored_capacity = study.user_attrs.get(
            "ttbi_capacity_preflight_receipt"
        )
        if stored_capacity is None:
            raise RuntimeError(
                f"{study.study_name}: stored campaign study lacks its capacity "
                "receipt; it cannot be certified."
            )
        stored_capacity = validate_capacity_receipt(
            stored_capacity,
            expected_runtime=execution_runtime,
        )
        if stored_capacity != capacity_receipt:
            raise RuntimeError(
                f"{study.study_name}: stored capacity receipt differs from "
                "the current qualification."
            )


def _validated_study_hyperparameter_record(
    study: optuna.Study,
    config: dict,
) -> tuple[dict | None, dict | None]:
    """Validate stored HPO/capacity provenance and return ``(plan, receipt)``."""

    if not config.get("protocol_hash"):
        return None, None
    execution_runtime = _validated_config_execution_runtime(config)
    record = study.user_attrs.get("ttbi_protocol_record")
    expected_keys = {
        "schema",
        "protocol_hash",
        "protocol_descriptor",
        "execution_environment_sha256",
        "execution_runtime",
        "dataset",
        "model_name",
        "seed",
        "sampler_seed",
        "n_trials",
        "epochs",
        "use_pruner",
        "optuna_protocol",
        "hyperparameter_execution_plan",
        "hyperparameter_policy_sha256",
        "hyperparameter_mode",
        "hyperparameter_manifest_sha256",
        "hyperparameter_source",
        "capacity_preflight_receipt_sha256",
        "capacity_preflight_policy_sha256",
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record.get("schema") != "optuna-study-provenance-v4"
    ):
        raise RuntimeError(
            f"{study.study_name}: malformed/missing v4 campaign study record."
        )
    canonical_record = _canonical_json_value(record)
    plan = canonical_record["hyperparameter_execution_plan"]
    requested_n_trials = plan.get("requested_n_trials")
    requested_use_pruner = plan.get("requested_use_pruner")
    expected_plan = derive_execution_plan(
        config,
        dataset_name=canonical_record["dataset"],
        requested_n_trials=requested_n_trials,
        requested_use_pruner=requested_use_pruner,
        execution_runtime=execution_runtime,
    )
    if plan != expected_plan:
        raise RuntimeError(
            f"{study.study_name}: stored hyperparameter plan does not derive "
            "from the current config/policy."
        )
    expected_scalars = {
        "protocol_hash": config["protocol_hash"],
        "protocol_descriptor": _canonical_json_value(
            config["protocol_descriptor"]
        ),
        "execution_environment_sha256":
            execution_runtime["execution_environment_sha256"],
        "execution_runtime": execution_runtime,
        "model_name": config.get("name"),
        "seed": int(config["seed"]),
        "n_trials": plan["effective_n_trials"],
        "use_pruner": plan["effective_use_pruner"],
        "optuna_protocol": _canonical_json_value(OPTUNA_PROTOCOL),
        "hyperparameter_policy_sha256": plan["policy_sha256"],
        "hyperparameter_mode": plan["mode"],
        "hyperparameter_manifest_sha256":
            plan["hyperparameter_manifest_sha256"],
        "hyperparameter_source": plan["hyperparameter_source"],
        "campaign_run_tag": plan["campaign_run_tag"],
        "execution_receipt_sha256": plan["execution_receipt_sha256"],
        "block_reference_manifest_sha256":
            plan["block_reference_manifest_sha256"],
    }
    mismatches = {
        key: (canonical_record.get(key), expected)
        for key, expected in expected_scalars.items()
        if canonical_record.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{study.study_name}: campaign study record mismatch: {mismatches}"
        )
    if (
        isinstance(canonical_record["epochs"], bool)
        or not isinstance(canonical_record["epochs"], int)
        or canonical_record["epochs"] < 1
        or isinstance(canonical_record["sampler_seed"], bool)
        or not isinstance(canonical_record["sampler_seed"], int)
    ):
        raise RuntimeError(
            f"{study.study_name}: invalid epoch/sampler-seed provenance."
        )
    capacity = validate_capacity_receipt(
        study.user_attrs.get("ttbi_capacity_preflight_receipt"),
        expected_runtime=execution_runtime,
    )
    if (
        capacity["receipt_sha256"]
        != canonical_record["capacity_preflight_receipt_sha256"]
        or capacity["receipt"]["policy_sha256"]
        != canonical_record["capacity_preflight_policy_sha256"]
    ):
        raise RuntimeError(
            f"{study.study_name}: capacity receipt hashes disagree with the "
            "study record."
        )
    return plan, capacity


def verify_digital_twin_package(
    study: optuna.Study,
    config: dict,
    output_dir: str,
) -> dict:
    """Verify champion weights against both metadata and the Optuna study."""
    execution_runtime = _validated_config_execution_runtime(config)
    if config.get("protocol_hash"):
        descriptor = config.get("protocol_descriptor")
        if descriptor is None or protocol_hash(descriptor) != config["protocol_hash"]:
            raise RuntimeError(
                f"{config.get('name')}: requested protocol descriptor/hash pair "
                "is missing or internally inconsistent.")
    hyperparameter_plan, capacity_receipt = (
        _validated_study_hyperparameter_record(study, config)
    )
    study_record = study.user_attrs.get("ttbi_protocol_record")
    weights_path = os.path.join(output_dir, "DT_champion_weights.pth")
    metadata_path = os.path.join(output_dir, "DT_metadata.json")
    if not os.path.isfile(weights_path) or not os.path.isfile(metadata_path):
        raise RuntimeError(
            f"{config.get('name')}: incomplete DT package in {output_dir}.")
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)

    actual_sha = _sha256_file(weights_path)
    scaler_name = metadata.get("scaler_filename")
    scaler_path = os.path.join(output_dir, scaler_name) if scaler_name else ""
    if not scaler_name or not os.path.isfile(scaler_path):
        raise RuntimeError(
            f"{config.get('name')}: metadata-linked scaler is missing.")
    scaler_sha = _sha256_file(scaler_path)
    architecture_flags = {
        "use_space2vec": config.get("use_space2vec", False),
        "use_lstm": config.get("use_lstm", False),
        "use_nhits": config.get("use_nhits", False),
        "model_type": config.get("model_type", "1D_MODULAR"),
    }
    expected = {
        "model_name": config.get("name"),
        "preprocessing_method": config.get("method"),
        "active_dofs": list(config.get("dofs", [])),
        "architecture_flags": architecture_flags,
        "discretization": config.get("discretization", 1),
        "n_segments": config.get("n_segments", 512),
        "task": config.get("task", "classification"),
        "target_supports": config.get("target_supports"),
        "bearing_targets": config.get("bearing_targets"),
        "protocol_hash": config.get("protocol_hash"),
        "protocol_descriptor": _canonical_json_value(
            config.get("protocol_descriptor")
        ),
        "execution_environment_sha256": (
            execution_runtime["execution_environment_sha256"]
            if execution_runtime is not None else None
        ),
        "execution_runtime": execution_runtime,
        "dataset": (
            study_record.get("dataset")
            if isinstance(study_record, dict) else None
        ),
        "hyperparameter_mode": (
            hyperparameter_plan["mode"]
            if hyperparameter_plan is not None else None
        ),
        "hyperparameter_policy_sha256": (
            hyperparameter_plan["policy_sha256"]
            if hyperparameter_plan is not None else None
        ),
        "hyperparameter_manifest_sha256": (
            hyperparameter_plan["hyperparameter_manifest_sha256"]
            if hyperparameter_plan is not None else None
        ),
        "hyperparameter_source": (
            hyperparameter_plan["hyperparameter_source"]
            if hyperparameter_plan is not None else None
        ),
        "capacity_preflight_receipt_sha256": (
            capacity_receipt["receipt_sha256"]
            if capacity_receipt is not None else None
        ),
        "capacity_preflight_policy_sha256": (
            capacity_receipt["receipt"]["policy_sha256"]
            if capacity_receipt is not None else None
        ),
        "campaign_run_tag": (
            hyperparameter_plan["campaign_run_tag"]
            if hyperparameter_plan is not None else None
        ),
        "execution_receipt_sha256": (
            hyperparameter_plan["execution_receipt_sha256"]
            if hyperparameter_plan is not None else None
        ),
        "block_reference_manifest_sha256": (
            hyperparameter_plan["block_reference_manifest_sha256"]
            if hyperparameter_plan is not None else None
        ),
        "study_name": study.study_name,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": actual_sha,
        "scaler_sha256": scaler_sha,
    }
    if config.get("protocol_hash"):
        required_campaign_lineage = {
            "campaign_run_tag",
            "execution_receipt_sha256",
            "block_reference_manifest_sha256",
        }
        missing_campaign_lineage = sorted(
            required_campaign_lineage - set(metadata)
        )
        if missing_campaign_lineage:
            raise RuntimeError(
                f"{config.get('name')}: champion package provenance mismatch: "
                "missing campaign lineage field(s): "
                f"{missing_campaign_lineage}"
            )
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if (_canonical_json_value(metadata.get(key))
            != _canonical_json_value(value))
    }
    if (_canonical_json_value(metadata.get("optimal_hyperparameters"))
            != _canonical_json_value(study.best_params)):
        mismatches["optimal_hyperparameters"] = (
            metadata.get("optimal_hyperparameters"), study.best_params)

    if config.get("protocol_hash"):
        canonical_study_record = (
            _canonical_json_value(study_record)
            if study_record is not None else None
        )
        record_sha = _canonical_json_sha256(canonical_study_record)
        if metadata.get("study_protocol_record_sha256") != record_sha:
            mismatches["study_protocol_record_sha256"] = (
                metadata.get("study_protocol_record_sha256"), record_sha)

    expected_artifact = {
        "schema": (
            "champion-artifact-v4"
            if config.get("protocol_hash") else "champion-artifact-v2"
        ),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": actual_sha,
        "scaler_sha256": scaler_sha,
        "protocol_hash": config.get("protocol_hash"),
        "execution_environment_sha256": (
            execution_runtime["execution_environment_sha256"]
            if execution_runtime is not None else None
        ),
    }
    if config.get("protocol_hash"):
        expected_artifact.update({
            "hyperparameter_mode": hyperparameter_plan["mode"],
            "hyperparameter_policy_sha256":
                hyperparameter_plan["policy_sha256"],
            "hyperparameter_manifest_sha256":
                hyperparameter_plan["hyperparameter_manifest_sha256"],
            "hyperparameter_source":
                hyperparameter_plan["hyperparameter_source"],
            "capacity_preflight_receipt_sha256":
                capacity_receipt["receipt_sha256"],
            "capacity_preflight_policy_sha256":
                capacity_receipt["receipt"]["policy_sha256"],
            "campaign_run_tag": hyperparameter_plan["campaign_run_tag"],
            "execution_receipt_sha256":
                hyperparameter_plan["execution_receipt_sha256"],
            "block_reference_manifest_sha256":
                hyperparameter_plan["block_reference_manifest_sha256"],
        })
    artifact = study.user_attrs.get("ttbi_champion_artifact")
    if artifact != expected_artifact:
        mismatches["study.user_attrs.ttbi_champion_artifact"] = (
            artifact, expected_artifact)
    if mismatches:
        raise RuntimeError(
            f"{config.get('name')}: champion package provenance mismatch: "
            f"{mismatches}")
    return metadata


# ──────────────────────────────────────────────────────────────────────────────
def _execute_protocol_study(
    study: optuna.Study,
    objective,
    hyperparameter_plan: dict,
) -> dict:
    """Run exactly the derived campaign budget, without catch/retry semantics."""

    hyperparameter_plan = validate_run_plan(hyperparameter_plan)
    if hyperparameter_plan.get("mode") not in {
        ANCHOR_HPO_MODE,
        FROZEN_SINGLETON_MODE,
    }:
        raise RuntimeError("protocol study helper received a non-campaign plan")
    TS = optuna.trial.TrialState
    states = [trial.state for trial in study.trials]
    counts = {
        state: sum(item == state for item in states)
        for state in (
            TS.COMPLETE,
            TS.PRUNED,
            TS.FAIL,
            TS.RUNNING,
            TS.WAITING,
        )
    }
    useful = counts[TS.COMPLETE] + counts[TS.PRUNED]
    budget = hyperparameter_plan["effective_n_trials"]
    if (
        counts[TS.FAIL]
        or counts[TS.RUNNING]
        or counts[TS.WAITING]
        or useful > budget
        or len(states) != useful
    ):
        raise RuntimeError(
            f"{study.study_name}: existing campaign study has forbidden "
            f"states or exceeds its derived budget: {counts!r}."
        )
    remaining = budget - useful
    if remaining:
        # Deliberately omit catch=: CUDA OOM and every other exception are
        # fatal and persist as evidence rather than triggering another trial.
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[print_best_callback],
            show_progress_bar=True,
        )
    return validate_terminal_study(study, hyperparameter_plan)


# 1. Master ablation loop
# ──────────────────────────────────────────────────────────────────────────────

def execute_ablation_pipeline(
    experiment_path:  list[dict],
    database_name:    str,
    output_dir_name:  str,
    cache_dir_name:   str,
    dataset:          str,
    n_trials:         int  = 50,
    epochs:           int  = 50,
    skip_robustness:  bool = True,
    optuna_seed:      int  = 42,
    use_pruner:       bool = False,
    run_robustness:   bool = True,
) -> list[dict]:
    """
    Run the full ablation pipeline for every model configuration in
    experiment_path and return a list of robustness scorecards for models
    that passed the stochastic gatekeeper.

    Per-model steps
    ---------------
    1. Create or resume the Optuna study and run up to n_trials trials.
    2. Evaluate the champion on the canonical validation set and save its
       confusion matrix.
    3. Bundle the champion weights, scaler, and metadata into a DT package.
    4. Run the 30-seed stochastic stress-test (skipped when the model's
       Optuna score exceeds the physical error tolerance and
       skip_robustness=True).
    5. Generate per-parameter Optuna slice plots.

    Resumability
    ------------
    Both the Optuna study (load_if_exists=True) and the robustness JSON
    checkpoints (written after every seed) survive interruptions.  Re-running
    the pipeline with the same arguments resumes from where it left off with
    no duplicate work.

    Sampler configuration
    ---------------------
    TPESampler with multivariate=True captures correlations between
    hyperparameters (e.g. high lr benefiting from high weight_decay).
    constant_liar=True prevents parallel workers from sampling the same
    region.  n_startup_trials is set to max(10, 25 % of n_trials) to ensure
    enough random exploration before the TPE model is fitted.

    Args:
        experiment_path (list[dict]): Ablation grid - each dict is one model
                                      config with at minimum keys: 'name',
                                      'method', 'dofs', 'discretization',
                                      'use_space2vec', 'use_lstm', 'use_nhits',
                                      'model_type'.
        database_name (str):   Optuna SQLite storage URL.
        output_dir_name (str): Root directory; one sub-folder per model is
                               created inside it.
        cache_dir_name (str):  Directory for preprocessed data caches.
        dataset (str):         Dataset sub-folder name passed to the cache.
        n_trials (int):        Maximum Optuna trials per model.
        epochs (int):          Maximum training epochs per trial.
        skip_robustness (bool): Skip the 30-seed test when the Optuna score
                                exceeds the physical error tolerance threshold.

    Returns:
        list[dict]: One scorecard dict per model that completed the stochastic
                    test.  Keys: 'Model', 'Optuna_Lucky_Score',
                    'Stochastic_Mean_MSE', 'Stochastic_Std_MSE', 'UCB_95_MSE'.
    """
    os.makedirs(cache_dir_name, exist_ok=True)
    set_global_seed(optuna_seed, TRAIN_PROTOCOL["determinism"])

    all_results: list[dict] = []

    for step in experiment_path:
        print(f"\n{'=' * 56}")
        print(f"  {step['name']}")
        print(f"{'=' * 56}")

        output_dir = os.path.join(output_dir_name, step['name'])
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Optuna study ───────────────────────────────────────────────────
        execution_runtime = _validated_config_execution_runtime(step)
        hyperparameter_plan = derive_execution_plan(
            step,
            dataset_name=dataset,
            requested_n_trials=n_trials,
            requested_use_pruner=use_pruner,
            execution_runtime=execution_runtime,
        )
        # This qualification is deliberately before create_study(): an
        # incapable GPU cannot leave a misleading resumable campaign study.
        capacity_receipt = (
            ensure_capacity_preflight(execution_runtime)
            if hyperparameter_plan["mode"] != LEGACY_MODE else None
        )
        effective_n_trials = hyperparameter_plan["effective_n_trials"]
        effective_use_pruner = hyperparameter_plan["effective_use_pruner"]
        study = _create_or_resume_study(
            step['name'], database_name, effective_n_trials,
            sampler_seed=optuna_seed,
            use_pruner=effective_use_pruner,
            force_nop_pruner=(
                hyperparameter_plan["mode"] == FROZEN_SINGLETON_MODE
            ),
        )
        _stamp_study_protocol(
            study,
            config=step,
            dataset_name=dataset,
            n_trials=effective_n_trials,
            epochs=epochs,
            sampler_seed=optuna_seed,
            use_pruner=effective_use_pruner,
            hyperparameter_plan=hyperparameter_plan,
            capacity_receipt=capacity_receipt,
        )

        # Legacy callers retain their historical bounded OOM handling.  The
        # protocol-hashed campaign takes the separate fail-closed helper below:
        # no caught exception, no replacement trial, exact derived terminal
        # budget.
        objective = Objective(
            config=step, dataset_name=dataset, n_epochs=epochs,
            cache_dir=cache_dir_name, output_dir=output_dir,
        )
        TS = optuna.trial.TrialState
        if hyperparameter_plan["mode"] == LEGACY_MODE:
            import torch as _torch
            max_fail_slack = OPTUNA_PROTOCOL["max_fail_slack"]
            oom = tuple(
                error for error in (
                    getattr(_torch.cuda, "OutOfMemoryError", None),
                    getattr(_torch, "OutOfMemoryError", None),
                ) if error
            )
            while True:
                states = [trial.state for trial in study.trials]
                n_useful = sum(
                    state in (TS.COMPLETE, TS.PRUNED) for state in states
                )
                remaining = min(
                    effective_n_trials - n_useful,
                    (effective_n_trials + max_fail_slack) - len(study.trials),
                )
                if remaining <= 0:
                    break
                study.optimize(
                    objective,
                    n_trials=remaining,
                    catch=oom,
                    callbacks=[print_best_callback],
                    show_progress_bar=True,
                )

        # ── FATAL GATE before ANY export / report (audit R7.1 P1/P2) ──────────
        # The study must be FINISHED: useful budget met, >=1 COMPLETE, no in-flight
        # trials. Otherwise refuse to compute best_value / export weights / run
        # robustness on an incomplete study (which selection would reject anyway,
        # but only AFTER the weights were exported).
        if hyperparameter_plan["mode"] == LEGACY_MODE:
            states = [trial.state for trial in study.trials]
            n_complete = sum(state == TS.COMPLETE for state in states)
            n_useful = sum(
                state in (TS.COMPLETE, TS.PRUNED) for state in states
            )
            n_inflight = sum(
                state in (TS.RUNNING, TS.WAITING) for state in states
            )
            if (
                n_inflight
                or n_useful != effective_n_trials
                or n_complete < 1
            ):
                raise RuntimeError(
                    f"{step['name']}: legacy study NOT finished "
                    f"(COMPLETE={n_complete}, useful={n_useful}/"
                    f"{effective_n_trials}, in-flight={n_inflight})."
                )
        else:
            _execute_protocol_study(
                study, objective, hyperparameter_plan
            )
            if hyperparameter_plan["mode"] == FROZEN_SINGLETON_MODE:
                if _canonical_json_value(study.best_params) != (
                    _canonical_json_value(step["frozen_hyperparameters"])
                ):
                    raise RuntimeError(
                        f"{step['name']}: singleton Optuna trial did not "
                        "reproduce its authenticated frozen parameters."
                    )
            elif hyperparameter_plan["mode"] == ANCHOR_HPO_MODE:
                identity = anchor_study_identity(
                    study=study,
                    config=step,
                    plan=hyperparameter_plan,
                    dataset_name=dataset,
                    study_protocol_record=study.user_attrs[
                        "ttbi_protocol_record"
                    ],
                )
                entry = build_manifest_entry(
                    study_identity=identity,
                    params=study.best_params,
                )
                prior = study.user_attrs.get(
                    "ttbi_hyperparameter_manifest_entry"
                )
                if prior is not None and _canonical_json_value(prior) != entry:
                    raise RuntimeError(
                        f"{step['name']}: stored anchor manifest entry differs "
                        "from the completed study."
                    )
                study.set_user_attr(
                    "ttbi_hyperparameter_manifest_entry", entry
                )

        print(f"  Best MSE: {study.best_value:.4f}  (trial {study.best_trial.number})")

        # ── 2. Confusion matrix (classification only) ─────────────────────────
        # A confusion matrix is a class-label artefact; multi-output regression
        # has no class axis, so it is skipped (per-pier MSE / parity plots are
        # the regression diagnostics - produced from the scorecard instead).
        if not task.is_regression(step):
            plot_cached_confusion_matrix(
                study=study, config=step,
                dataset_name=dataset,
                cache_dir=cache_dir_name,
                output_dir=output_dir,
            )

        # ── 3. DT export ──────────────────────────────────────────────────────
        export_digital_twin_package(
            study=study, config=step,
            dataset_name=dataset,
            cache_dir=cache_dir_name,
            output_dir=output_dir,
        )

        # ── 4. Stochastic stress-test ─────────────────────────────────────────
        # run_robustness=False skips the multi-seed Monte-Carlo entirely (the
        # reduced multi-damage grid runs a single seed); the default-True path is
        # unchanged for the single-scour ablation.
        if run_robustness:
            scorecard = evaluate_stochastic_robustness(
                study=study, config=step,
                dataset_name=dataset,
                n_epochs=epochs,
                cache_dir=cache_dir_name,
                output_dir=output_dir,
                skip_robustness=skip_robustness,
            )
            if scorecard:
                all_results.append({'Model': step['name'], **scorecard})

        # ── 5. Optuna slice plots ─────────────────────────────────────────────
        generate_optuna_robustness_plots(
            study=study, config=step, output_dir=output_dir
        )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Digital twin package export
# ──────────────────────────────────────────────────────────────────────────────

def export_digital_twin_package(
    study,
    config:       dict,
    dataset_name: str,
    cache_dir:    str,
    output_dir:   str,
) -> None:
    """
    Bundle the three files that drive_by_DT.py needs at startup:

        DT_metadata.json        - architecture config and Optuna best_params.
        DT_champion_weights.pth - the model weights from the winning trial.
        DT_scaler.pkl           - the fitted scaler (or .pt for PyTorch scalers).

    Also deletes all per-trial weight files (weights_<name>_trial_*.pth) after
    the complete package has passed its provenance verification, reclaiming SSD
    space. Re-running an already completed study is idempotent: when the
    per-trial file was previously retired, the linked champion package is
    verified against the current study and reused.

    Weight file resolution
    ----------------------
    Looks for the per-trial weight file written by train_and_evaluate. If it is
    absent, an existing complete champion package must verify exactly against
    the current study/configuration; otherwise export fails closed.

    Args:
        study:            Completed Optuna study for this model.
        config (dict):    Ablation step config.
        dataset_name:     Dataset sub-folder name (for scaler path resolution).
        cache_dir (str):  Cache directory containing the fitted scaler.
        output_dir (str): Destination directory for the three output files.
    """
    print(f"  --> Exporting DT package: {config['name']}")
    execution_runtime = _validated_config_execution_runtime(config)
    hyperparameter_plan, capacity_receipt = (
        _validated_study_hyperparameter_record(study, config)
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        'model_name':           config['name'],
        'preprocessing_method': config['method'],
        'active_dofs':          config['dofs'],
        'architecture_flags': {
            'use_space2vec': config.get('use_space2vec', False),
            'use_lstm':      config.get('use_lstm',      False),
            'use_nhits':     config.get('use_nhits',     False),
            'model_type':    config.get('model_type',    '1D_MODULAR'),
        },
        'optimal_hyperparameters': study.best_params,
        # Persist the label discretisation so the online DTConfig reconstructs the
        # right number of classes (the ablation uses 1% steps -> 61 classes; a
        # missing value previously defaulted to 5 -> 13 classes, breaking the DT).
        'discretization': config.get('discretization', 1),
        'n_segments':     config.get('n_segments', 512),
        # Task descriptor so the loader/DT rebuilds the right head: classification
        # (default) or multi-output regression over the listed target supports.
        'task':            config.get('task', 'classification'),
        'target_supports': config.get('target_supports'),
        'bearing_targets': config.get('bearing_targets'),
        # Unified protocol hash (2026-07-19): stamps the exported weights with
        # the exact protocol that produced them (None for legacy callers that
        # predate the hash, e.g. the single-scour classification scripts).
        'protocol_hash':   config.get('protocol_hash'),
        'execution_environment_sha256': (
            execution_runtime["execution_environment_sha256"]
            if execution_runtime is not None else None
        ),
        'execution_runtime': execution_runtime,
        'dataset': dataset_name,
        'hyperparameter_mode': (
            hyperparameter_plan["mode"]
            if hyperparameter_plan is not None else None
        ),
        'hyperparameter_policy_sha256': (
            hyperparameter_plan["policy_sha256"]
            if hyperparameter_plan is not None else None
        ),
        'hyperparameter_manifest_sha256': (
            hyperparameter_plan["hyperparameter_manifest_sha256"]
            if hyperparameter_plan is not None else None
        ),
        'hyperparameter_source': (
            hyperparameter_plan["hyperparameter_source"]
            if hyperparameter_plan is not None else None
        ),
        'capacity_preflight_receipt_sha256': (
            capacity_receipt["receipt_sha256"]
            if capacity_receipt is not None else None
        ),
        'capacity_preflight_policy_sha256': (
            capacity_receipt["receipt"]["policy_sha256"]
            if capacity_receipt is not None else None
        ),
        'campaign_run_tag': (
            hyperparameter_plan["campaign_run_tag"]
            if hyperparameter_plan is not None else None
        ),
        'execution_receipt_sha256': (
            hyperparameter_plan["execution_receipt_sha256"]
            if hyperparameter_plan is not None else None
        ),
        'block_reference_manifest_sha256': (
            hyperparameter_plan["block_reference_manifest_sha256"]
            if hyperparameter_plan is not None else None
        ),
    }

    # ── Champion weights ──────────────────────────────────────────────────────
    best_trial_num    = study.best_trial.number
    trial_weight_path = os.path.join(
        output_dir, f"weights_{config['name']}_trial_{best_trial_num}.pth"
    )
    champion_path = os.path.join(output_dir, 'DT_champion_weights.pth')

    if os.path.exists(trial_weight_path):
        source_sha = _sha256_file(trial_weight_path)
        shutil.copy2(trial_weight_path, champion_path)
        champion_sha = _sha256_file(champion_path)
        if champion_sha != source_sha:
            raise RuntimeError(
                "Champion weight copy failed SHA-256 verification: "
                f"{source_sha} != {champion_sha}.")
        print(f"      Champion weights -> {champion_path}")
    else:
        # A successful earlier export deliberately retires per-trial files.
        # Verify that exact package before returning so a campaign restart is
        # resumable without weakening the fail-closed missing-weight contract.
        metadata_path = os.path.join(output_dir, "DT_metadata.json")
        if os.path.isfile(champion_path) and os.path.isfile(metadata_path):
            verify_digital_twin_package(study, config, output_dir)
            print(f"      Verified existing DT package -> {output_dir}")
            return
        raise RuntimeError(
            f"Per-trial weights not found at {trial_weight_path} and no "
            f"verifiable champion package exists for best trial "
            f"{best_trial_num} of study '{config['name']}'. Refusing to "
            f"continue; investigate disk/save-path/interruption state.")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler_path = _copy_scaler(config, dataset_name, cache_dir, output_dir)
    scaler_sha = _sha256_file(scaler_path)
    study_record = study.user_attrs.get("ttbi_protocol_record")
    if config.get("protocol_hash") and study_record is None:
        raise RuntimeError(
            f"{config['name']}: protocol-hashed study has no stored protocol "
            "record; refusing to publish a provenance-incomplete package.")
    study_record_sha = (
        _canonical_json_sha256(study_record) if study_record is not None else None
    )

    artifact = {
        "schema": (
            "champion-artifact-v4"
            if config.get("protocol_hash") else "champion-artifact-v2"
        ),
        "best_trial_number": int(best_trial_num),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": champion_sha,
        "scaler_sha256": scaler_sha,
        "protocol_hash": config.get("protocol_hash"),
        "execution_environment_sha256": (
            execution_runtime["execution_environment_sha256"]
            if execution_runtime is not None else None
        ),
    }
    if config.get("protocol_hash"):
        artifact.update({
            "hyperparameter_mode": hyperparameter_plan["mode"],
            "hyperparameter_policy_sha256":
                hyperparameter_plan["policy_sha256"],
            "hyperparameter_manifest_sha256":
                hyperparameter_plan["hyperparameter_manifest_sha256"],
            "hyperparameter_source":
                hyperparameter_plan["hyperparameter_source"],
            "capacity_preflight_receipt_sha256":
                capacity_receipt["receipt_sha256"],
            "capacity_preflight_policy_sha256":
                capacity_receipt["receipt"]["policy_sha256"],
            "campaign_run_tag": hyperparameter_plan["campaign_run_tag"],
            "execution_receipt_sha256":
                hyperparameter_plan["execution_receipt_sha256"],
            "block_reference_manifest_sha256":
                hyperparameter_plan["block_reference_manifest_sha256"],
        })
    study.set_user_attr("ttbi_champion_artifact", artifact)
    metadata.update({
        "study_name": study.study_name,
        "best_trial_number": int(best_trial_num),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": champion_sha,
        "champion_weights_hash_algorithm": "SHA-256",
        "scaler_filename": os.path.basename(scaler_path),
        "scaler_sha256": scaler_sha,
        "protocol_descriptor": config.get("protocol_descriptor"),
        "study_protocol_record_sha256": study_record_sha,
    })
    _atomic_json(os.path.join(output_dir, "DT_metadata.json"), metadata)
    # Retain the source trial until the complete package and both provenance
    # links have passed verification. A second verification proves cleanup did
    # not disturb the champion package.
    verify_digital_twin_package(study, config, output_dir)
    _delete_trial_weights(output_dir, config['name'])
    verify_digital_twin_package(study, config, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_sqlite_parent_dir(storage: str) -> None:
    """Create the parent directory of a sqlite:///<path> storage URL if missing.

    Optuna/SQLAlchemy will not create intermediate directories, so a fresh
    extract (no database/ folder) raises OperationalError: unable to open
    database file. Only acts on sqlite URLs; other backends are left alone.
    """
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    db_path = storage[len(prefix):]
    parent  = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _create_or_resume_study(
    study_name:    str,
    storage:       str,
    n_trials:      int,
    sampler_seed:  int  = 42,
    use_pruner:    bool = False,
    force_nop_pruner: bool = False,
) -> optuna.Study:
    """
    Create a new TPE study or resume an existing one from storage.

    When `use_pruner=True`, attaches a SuccessiveHalvingPruner. The trainer
    already calls `trial.report(val_mse, epoch)` and `trial.should_prune()`,
    so unpromising trials get killed off after a few epochs - typically
    saves 30-50 % of compute on noisy losses without hurting the picked
    optimum. Default False keeps the pre-existing behaviour (Optuna's
    built-in MedianPruner).

    A SQLite storage ("sqlite:///database/....db") fails with "unable to open
    database file" if the parent folder does not exist - which is the case on a
    fresh checkout/extract where database/ was never created. Create it first.
    """
    _ensure_sqlite_parent_dir(storage)

    # PROTOCOL (2026-07-19): every sampler/pruner value below is read from
    # OPTUNA_PROTOCOL (core/protocol.py), which the unified protocol hash
    # covers — the running configuration and the hashed configuration are the
    # same object, so they cannot drift apart.
    sp = OPTUNA_PROTOCOL["sampler"]
    pp = OPTUNA_PROTOCOL["pruner"]
    assert sp["class"] == "TPESampler" and pp["class"] == "SuccessiveHalvingPruner", \
        "OPTUNA_PROTOCOL names a sampler/pruner class this code does not build"
    n_startup = max(10, n_trials // 4)   # == OPTUNA_PROTOCOL sampler n_startup_rule
    sampler   = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=n_startup,
        multivariate=sp["multivariate"],
        constant_liar=sp["constant_liar"],
        warn_independent_sampling=False,
    )
    if use_pruner and force_nop_pruner:
        raise RuntimeError(
            "cannot request both the registered pruner and NopPruner"
        )
    pruner = (
        optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pp["min_resource"],               # epochs before pruning
            reduction_factor=pp["reduction_factor"],       # keep top 1/N per rung
            min_early_stopping_rate=pp["min_early_stopping_rate"],
        )
        if use_pruner
        else (
            optuna.pruners.NopPruner()
            if force_nop_pruner else None
        )
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=OPTUNA_PROTOCOL["direction"],
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )


def _copy_scaler(
    config:       dict,
    dataset_name: str,
    cache_dir:    str,
    output_dir:   str,
) -> str:
    """
    Locate the fitted scaler in cache_dir and copy it to output_dir as
    DT_scaler.pkl (or DT_scaler.pt for PyTorch scalers).

    The cache filename follows the same naming convention as
    core/dataset._cache_stem so the two modules stay in sync (regression caches
    carry an extra _reg_t<targets> tag - reusing _cache_stem keeps them aligned).
    """
    stem     = f"scaler_{_cache_stem(dataset_name, config)}"

    pkl_src  = os.path.join(cache_dir, f"{stem}.pkl")
    pt_src   = os.path.join(cache_dir, f"{stem}.pt")
    pkl_dst  = os.path.join(output_dir, 'DT_scaler.pkl')
    pt_dst   = os.path.join(output_dir, 'DT_scaler.pt')

    if os.path.exists(pkl_src):
        src, dst, label = pkl_src, pkl_dst, "sklearn"
    elif os.path.exists(pt_src):
        src, dst, label = pt_src, pt_dst, "PyTorch"
    else:
        raise RuntimeError(
            f"Scaler not found in {cache_dir}.\n"
            f"      Expected: {pkl_src}\n"
            "      Refusing to publish an incomplete DT package."
        )
    source_sha = _sha256_file(src)
    if config.get("protocol_hash"):
        prov_path = os.path.join(
            cache_dir, f"cache_{_cache_stem(dataset_name, config)}_prov.json"
        )
        try:
            with open(prov_path, encoding="utf-8") as stream:
                expected_sha = json.load(stream)["artifacts"]["scaler"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot authenticate scaler source: invalid/missing cache "
                f"provenance {prov_path}.") from exc
        if source_sha != expected_sha:
            raise RuntimeError(
                f"Scaler source SHA-256 differs from cache provenance at "
                f"{prov_path}; refusing to auto-sign a corrupted scaler.")
    shutil.copy2(src, dst)
    destination_sha = _sha256_file(dst)
    if destination_sha != source_sha:
        raise RuntimeError(
            f"Scaler copy failed SHA-256 verification: "
            f"{source_sha} != {destination_sha}.")
    print(f"      Scaler ({label}) -> {dst}")
    return dst


def _delete_trial_weights(output_dir: str, model_name: str) -> None:
    """
    Delete all per-trial weight files for model_name in output_dir after
    the champion copy has been safely written.  Reports the count deleted.
    """
    pattern = os.path.join(output_dir, f"weights_{model_name}_trial_*.pth")
    files   = glob.glob(pattern)
    deleted = 0
    for path in files:
        try:
            os.remove(path)
            deleted += 1
        except OSError as e:
            print(f"      [WARNING] Could not delete {path}: {e}")
    if deleted:
        print(f"      Deleted {deleted} sub-optimal trial weight file(s).")
