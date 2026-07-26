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
import torch

from core          import task
from core.dataset  import get_or_create_cache, _cache_stem
from core.protocol import OPTUNA_PROTOCOL, protocol_hash
from core.utils    import set_global_seed, DOF_NAME_TO_IDX
from plotting.confusion        import plot_cached_confusion_matrix
from plotting.robustness_plots import generate_optuna_robustness_plots, plot_stochastic_summary
from training.robustness       import evaluate_stochastic_robustness, evaluate_parametric_robustness
from training.trainer          import (
    DEVICE,
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


def _stamp_study_protocol(
    study: optuna.Study,
    *,
    config: dict,
    dataset_name: str,
    n_trials: int,
    epochs: int,
    sampler_seed: int,
    use_pruner: bool,
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
    record = {
        "schema": "optuna-study-provenance-v1",
        "protocol_hash": config.get("protocol_hash"),
        "protocol_descriptor": config.get("protocol_descriptor"),
        "dataset": dataset_name,
        "model_name": config.get("name"),
        "seed": int(config.get("seed", sampler_seed)),
        "sampler_seed": int(sampler_seed),
        "n_trials": int(n_trials),
        "epochs": int(epochs),
        "use_pruner": bool(use_pruner),
        "optuna_protocol": OPTUNA_PROTOCOL,
    }
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
    elif _canonical_json_value(previous) != record:
        raise RuntimeError(
            f"{study.study_name}: stored Optuna protocol record differs from "
            "the requested run. Refusing to mix trials; use a fresh RUN_TAG.")


def verify_digital_twin_package(
    study: optuna.Study,
    config: dict,
    output_dir: str,
) -> dict:
    """Verify champion weights against both metadata and the Optuna study."""
    if config.get("protocol_hash"):
        descriptor = config.get("protocol_descriptor")
        if descriptor is None or protocol_hash(descriptor) != config["protocol_hash"]:
            raise RuntimeError(
                f"{config.get('name')}: requested protocol descriptor/hash pair "
                "is missing or internally inconsistent.")
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
        "study_name": study.study_name,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": actual_sha,
        "scaler_sha256": scaler_sha,
    }
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

    study_record = study.user_attrs.get("ttbi_protocol_record")
    if config.get("protocol_hash"):
        canonical_study_record = (
            _canonical_json_value(study_record)
            if study_record is not None else None
        )
        expected_record_fields = {
            "schema": "optuna-study-provenance-v1",
            "protocol_hash": config.get("protocol_hash"),
            "protocol_descriptor": _canonical_json_value(
                config.get("protocol_descriptor")
            ),
            "model_name": config.get("name"),
        }
        if "seed" in config:
            expected_record_fields["seed"] = int(config["seed"])
        if (canonical_study_record is None
                or any(canonical_study_record.get(key) != value
                       for key, value in expected_record_fields.items())):
            mismatches["study.user_attrs.ttbi_protocol_record"] = (
                canonical_study_record, expected_record_fields)
        else:
            record_sha = _canonical_json_sha256(canonical_study_record)
            if metadata.get("study_protocol_record_sha256") != record_sha:
                mismatches["study_protocol_record_sha256"] = (
                    metadata.get("study_protocol_record_sha256"), record_sha)

    expected_artifact = {
        "schema": "champion-artifact-v1",
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": actual_sha,
        "scaler_sha256": scaler_sha,
        "protocol_hash": config.get("protocol_hash"),
    }
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
        study = _create_or_resume_study(
            step['name'], database_name, n_trials,
            sampler_seed=optuna_seed,
            use_pruner=use_pruner,
        )
        _stamp_study_protocol(
            study,
            config=step,
            dataset_name=dataset,
            n_trials=n_trials,
            epochs=epochs,
            sampler_seed=optuna_seed,
            use_pruner=use_pruner,
        )

        # Budget = terminal-USEFUL trials (COMPLETE + PRUNED). FAILed trials are
        # RETRIED automatically (audit R7.1 P2): re-evaluate the budget AFTER each
        # optimize() and keep going until the useful budget is met, so a burst of
        # FAILs inside one run does not leave the study short until the whole script
        # is re-run. A deterministically-broken config would retry forever, so cap
        # total trials at n_trials + MAX_FAIL_SLACK; it then stops short of the
        # useful budget and is flagged INCOMPLETE by _study_is_finished (never
        # silently selected). The trainer raises TrialPruned for the pruner but lets
        # deterministic bugs propagate as FAILs, which the cap bounds.
        # PROTOCOL (2026-07-19): the slack value lives in OPTUNA_PROTOCOL so it
        # is part of the unified protocol hash.
        MAX_FAIL_SLACK = OPTUNA_PROTOCOL["max_fail_slack"]
        objective = Objective(
            config=step, dataset_name=dataset, n_epochs=epochs,
            cache_dir=cache_dir_name, output_dir=output_dir,
        )
        # RECOVERABLE trial failures (transient GPU/CPU OOM) are CAUGHT -> marked
        # FAILED -> the loop retries them to the useful budget. EVERY OTHER
        # exception propagates and FAILS FAST (audit R7.1 P2): a deterministic bug
        # must crash immediately, not be silently retried 20 times.
        import optuna.exceptions as _oe                           # noqa: F401
        _oom = tuple(e for e in (getattr(torch.cuda, "OutOfMemoryError", None),
                                 getattr(torch, "OutOfMemoryError", None)) if e)
        TS = optuna.trial.TrialState
        while True:
            states = [t.state for t in study.trials]
            n_useful = sum(s in (TS.COMPLETE, TS.PRUNED) for s in states)
            remaining = min(n_trials - n_useful,
                            (n_trials + MAX_FAIL_SLACK) - len(study.trials))
            if remaining <= 0:
                break
            study.optimize(
                objective,
                n_trials=remaining,
                catch=_oom,                    # retry only transient OOM
                callbacks=[print_best_callback],
                show_progress_bar=True,
            )

        # ── FATAL GATE before ANY export / report (audit R7.1 P1/P2) ──────────
        # The study must be FINISHED: useful budget met, >=1 COMPLETE, no in-flight
        # trials. Otherwise refuse to compute best_value / export weights / run
        # robustness on an incomplete study (which selection would reject anyway,
        # but only AFTER the weights were exported).
        states = [t.state for t in study.trials]
        n_complete = sum(s == TS.COMPLETE for s in states)
        n_useful   = sum(s in (TS.COMPLETE, TS.PRUNED) for s in states)
        n_inflight = sum(s in (TS.RUNNING, TS.WAITING) for s in states)
        if n_inflight or n_useful != n_trials or n_complete < 1:
            raise RuntimeError(
                f"{step['name']}: study NOT finished (COMPLETE={n_complete}, "
                f"useful={n_useful}/{n_trials}, in-flight={n_inflight}) — refusing "
                f"to export weights / report. Investigate the FAILs (or delete the "
                f"study to re-run); an incomplete OR manually extended study "
                f"must not be published.")

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
        "schema": "champion-artifact-v1",
        "best_trial_number": int(best_trial_num),
        "best_trial_value": float(study.best_value),
        "champion_weights_sha256": champion_sha,
        "scaler_sha256": scaler_sha,
        "protocol_hash": config.get("protocol_hash"),
    }
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
    pruner = (
        optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pp["min_resource"],               # epochs before pruning
            reduction_factor=pp["reduction_factor"],       # keep top 1/N per rung
            min_early_stopping_rate=pp["min_early_stopping_rate"],
        )
        if use_pruner else None
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
