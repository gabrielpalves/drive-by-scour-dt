"""
training/trainer.py
===================
Everything that runs inside a single Optuna trial or a single fixed-seed
training run.

Public API
----------
    train_and_evaluate   - one Optuna trial: suggest -> build -> train -> prune.
    run_single_training  - one fixed-seed run with known params (robustness use).
    fit_predict_finalist_fold - one fold-local-scaled finalist CV refit.
    Objective            - callable wrapper so Optuna can call the trial function.
    print_best_callback  - Optuna study callback; prints only on new-best trials.
    DEVICE               - module-level torch.device (cuda if available, else cpu).

Imported by:
    training/robustness.py  - run_single_training
    training/pipeline.py    - Objective, print_best_callback, DEVICE
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import optuna
except ImportError:
    optuna = None   # pipeline.py guards against this; trainer is still importable

from core         import task
from core.dataset import (MemmapDataset, get_or_create_cache,
                          canonical_train_val_split)
from core.models  import build_model
from core.statistical_inference import (
    FoldStandardizedDataset,
    channel_standardization_stats,
    per_state_regression_metrics,
)
from core.utils   import DETERMINISM_POLICY, set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Hardware
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Training + search-space PROTOCOL (unified protocol_hash, 2026-07-19)
# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth: the training code below reads TRAIN_PROTOCOL, and
# _suggest_params reads SEARCH_SPACE. core/protocol.py folds both into the
# protocol hash, so changing any value here changes every study/manifest/
# summary name in lockstep — a re-run can never silently resume studies that
# trained under a different protocol. check_protocol_hash.py additionally
# drives _suggest_params with a recording stub and verifies every suggestion
# it makes matches SEARCH_SPACE exactly (belt-and-braces against drift).

TRAIN_PROTOCOL = {
    "batch_size":  32,       # train + val DataLoader batch size
    "patience":    5,        # early-stop after this many non-improving epochs
    # The Optuna training seed is a required config field. A silent default
    # would collapse nominally independent seed arms onto the same RNG stream.
    "trial_seed": {
        "source":  "config",
        "key":     "seed",
        "missing": "error",
    },
    # Same object consumed by set_global_seed; no duplicated prose contract.
    "determinism": DETERMINISM_POLICY,
    # Executable optimizer/scheduler specs. The factories below consume these
    # exact mappings at every production training call.
    "optimizer": {
        "kind":                "Adam",
        "lr_param":            "lr",
        "weight_decay_param":  "weight_decay",
    },
    "scheduler": {
        "kind":     "CosineAnnealingLR",
        "eta_min":  0.0,
    },
    # Audit r3/r5 (2026-07-22/25): executable, protocol-hashed objective policy.
    # task.objective_value reads this mapping through the trainer call below.
    # Therefore a re-registration changes BOTH running behaviour and the
    # protocol hash; it cannot drift from a prose-only declaration.
    "objective": {
        "regression_with_bearing_heads": "scour_mse",
        "default":                       "mse",
    },
    "loss": {
        "classification": {
            "kind": "cross_entropy",
        },
        "regression_without_bearing_heads": {
            "kind": "mse",
        },
        "regression_with_bearing_heads": {
            "kind": "inverse_range_squared_mse",
            "head_ranges_pct": {
                "scour":   60.0,
                "bearing": 95.0,
            },
        },
    },
}


def resolve_trial_seed(config: dict, policy: dict) -> int:
    """Resolve the Optuna-training seed from the executable protocol policy."""

    required = {"source", "key", "missing"}
    if not isinstance(policy, dict) or set(policy) != required:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            f"trial-seed policy must define exactly {sorted(required)!r}; "
            f"got {got!r}.")
    if policy["source"] != "config" or policy["missing"] != "error":
        raise ValueError(f"unsupported trial-seed policy {policy!r}")
    key = policy["key"]
    if not isinstance(key, str) or not key:
        raise ValueError("trial-seed config key must be a non-empty string")
    if key not in config:
        raise KeyError(
            f"trial-seed policy requires config field {key!r}; "
            "a default seed is forbidden")
    seed = config[key]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError(f"config[{key!r}] must be an integer seed")
    seed = int(seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError(f"config[{key!r}] must lie in [0, 2**32 - 1]")
    return seed


def make_optimizer(parameters, params: dict, policy: dict):
    """Build the optimizer from the executable TRAIN_PROTOCOL specification."""
    required = {"kind", "lr_param", "weight_decay_param"}
    if not isinstance(policy, dict) or set(policy) != required:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            f"optimizer policy must define exactly {sorted(required)!r}; "
            f"got {got!r}.")
    if policy["kind"] != "Adam":
        raise ValueError(f"unsupported optimizer kind {policy['kind']!r}.")
    lr_key = policy["lr_param"]
    wd_key = policy["weight_decay_param"]
    if not isinstance(lr_key, str) or not isinstance(wd_key, str):
        raise ValueError("optimizer parameter keys must be strings.")
    try:
        lr = float(params[lr_key])
        weight_decay = float(params[wd_key])
    except KeyError as exc:
        raise KeyError(
            f"optimizer policy requires hyperparameter {exc.args[0]!r}; "
            f"available keys are {sorted(params)!r}.") from exc
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"optimizer learning rate must be finite and > 0; got {lr}.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(
            "optimizer weight decay must be finite and >= 0; "
            f"got {weight_decay}.")
    return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, max_epochs: int, policy: dict):
    """Build the epoch scheduler from the executable protocol specification."""
    required = {"kind", "eta_min"}
    if not isinstance(policy, dict) or set(policy) != required:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            f"scheduler policy must define exactly {sorted(required)!r}; "
            f"got {got!r}.")
    if policy["kind"] != "CosineAnnealingLR":
        raise ValueError(f"unsupported scheduler kind {policy['kind']!r}.")
    if isinstance(max_epochs, bool) or int(max_epochs) != max_epochs or max_epochs < 1:
        raise ValueError(
            f"scheduler max_epochs must be a positive integer; got {max_epochs!r}.")
    eta_min = float(policy["eta_min"])
    if not np.isfinite(eta_min) or eta_min < 0:
        raise ValueError(
            f"scheduler eta_min must be finite and >= 0; got {eta_min}.")
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(max_epochs), eta_min=eta_min)


# The FULL hyperparameter search space, as data. Spec tuples:
#     ("int",      low, high)          -> suggest_int(low, high)
#     ("int_step", low, high, step)    -> suggest_int(low, high, step=step)
#     ("float",    low, high)          -> suggest_float(low, high)
#     ("logfloat", low, high)          -> suggest_float(low, high, log=True)
#     ("cat",      [choices])          -> suggest_categorical(choices)
# Structure mirrors the conditional shape of the space: per-layer blocks are
# repeated with an _l{i} suffix; 'lstm'/'nhits' blocks are gated on the arch
# flags; lstm_dropout is gated on lstm_num_layers > 1.
SEARCH_SPACE = {
    "base": {
        "n_conv_layers":  ("int", 2, 4),
        "n_dense_layers": ("int", 1, 3),
        "lr":             ("logfloat", 1e-4, 1e-2),
        "weight_decay":   ("logfloat", 1e-5, 1e-3),
    },
    "per_conv_layer": {                       # x n_conv_layers, suffix _l{i}
        "n_filters":   ("int_step", 16, 128, 16),
        "kernel_size": ("cat", [2, 3, 5, 7]),
        "pooling":     ("cat", [True, False]),
    },
    "per_dense_layer": {                      # x n_dense_layers, suffix _l{i}
        "n_dense_units": ("int_step", 32, 256, 16),
        "dropout":       ("float", 0.1, 0.5),
    },
    "lstm": {                                 # iff config['use_lstm']
        "lstm_num_layers":  ("int", 1, 2),
        "lstm_hidden_size": ("int_step", 32, 128, 32),
        "lstm_dropout":     ("float", 0.1, 0.4),   # iff lstm_num_layers > 1
    },
    "nhits": {                                # iff config['use_nhits']
        "nhits_pool_rates_key": ("cat", ["1_2_4", "1_4_8", "1_3_6", "1_2_4_8"]),
    },
}


def _suggest_one(trial, name: str, spec: tuple):
    """Execute ONE search-space spec tuple against an Optuna trial.

    This tiny interpreter is the only place spec tuples are turned into
    suggest_* calls, so SEARCH_SPACE cannot drift from what is sampled."""
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "int_step":
        return trial.suggest_int(name, spec[1], spec[2], step=spec[3])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "logfloat":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "cat":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown search-space spec kind {kind!r} for {name!r}")


def _suggest_frozen_one(
    trial,
    name: str,
    spec: tuple,
    frozen: dict,
):
    """Register one authenticated fixed value as a singleton Optuna domain.

    Fixed candidate fits still use a real one-trial Optuna study so all
    downstream artifact/provenance code sees a complete ``best_params``
    mapping.  The singleton distribution makes renewed tuning impossible by
    construction, while this validator refuses values outside the registered
    search space.
    """

    if name not in frozen:
        raise ValueError(
            f"frozen hyperparameters are missing required key {name!r}"
        )
    value = frozen[name]
    kind = spec[0]
    if kind in {"int", "int_step"}:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise ValueError(
                f"frozen {name!r} must be an integer; got {value!r}"
            )
        value = int(value)
        low, high = int(spec[1]), int(spec[2])
        if not low <= value <= high:
            raise ValueError(
                f"frozen {name!r}={value} lies outside [{low}, {high}]"
            )
        if kind == "int_step" and (value - low) % int(spec[3]) != 0:
            raise ValueError(
                f"frozen {name!r}={value} violates step {spec[3]} from {low}"
            )
        return trial.suggest_int(name, value, value)
    if kind in {"float", "logfloat"}:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"frozen {name!r} must be numeric; got {value!r}"
            )
        value = float(value)
        low, high = float(spec[1]), float(spec[2])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"frozen {name!r}={value!r} lies outside [{low}, {high}]"
            )
        return trial.suggest_float(
            name, value, value, log=(kind == "logfloat")
        )
    if kind == "cat":
        choices = list(spec[1])
        if value not in choices:
            raise ValueError(
                f"frozen {name!r}={value!r} is not in {choices!r}"
            )
        return trial.suggest_categorical(name, [value])
    raise ValueError(f"unknown search-space spec kind {kind!r} for {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Core training & evaluation - one Optuna trial
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    trial,
    config:       dict,
    dataset_name: str,
    n_epochs:     int,
    cache_dir:    str,
    output_dir:   str,
) -> float:
    """
    Load data, suggest hyperparameters, build a model, train it, and return
    the best validation MSE seen during training.

    Integrates two early-stopping mechanisms:
        - Optuna pruning  - kills trials worse than the median at the current epoch.
        - Patience        - kills trials that have plateaued for `patience` epochs.

    The best weights seen during each trial are saved to output_dir so that
    export_digital_twin_package can retrieve the champion later without
    retraining.

    Args:
        trial:        optuna.Trial object (passed automatically by study.optimize).
        config (dict): Ablation step config - method, flags, dofs, discretization.
        dataset_name (str): Dataset sub-folder name (passed to get_or_create_cache).
        n_epochs (int): Maximum training epochs per trial.
        cache_dir (str): Cache directory for preprocessed arrays.
        output_dir (str): Directory to write per-trial weight files.

    Returns:
        float: Best validation MSE (lower is better).
    """
    # AUDIT FIX 2026-07-17: seed from the config, not hardcoded 42. The
    # multi-seed grid claims independent init/shuffle per seed ("Independent
    # Optuna seeds per config"); the hardcoded 42 made every "seed" share the
    # same init stream so only the Optuna sampler varied.
    set_global_seed(
        resolve_trial_seed(config, TRAIN_PROTOCOL["trial_seed"]),
        TRAIN_PROTOCOL["determinism"],
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y, _, groups = get_or_create_cache(config, dataset_name, cache_dir)

    # Canonical split (seed 42 for ALL config-seeds/arms so they compare on
    # identical partitions); grouped by damage state - audit fix 2026-07-17;
    # family-STRATIFIED via the dataset's state table (Feature A 2026-07-19).
    train_idx, val_idx = canonical_train_val_split(len(y), groups,
                                                   dataset_name=dataset_name)

    label_dtype = task.label_dtype(config)
    train_loader = DataLoader(
        MemmapDataset(X, y, train_idx, label_dtype=label_dtype),
        batch_size=TRAIN_PROTOCOL['batch_size'], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MemmapDataset(X, y, val_idx, label_dtype=label_dtype),
        batch_size=TRAIN_PROTOCOL['batch_size'], shuffle=False,
    )

    # ── Hyperparameter suggestion ─────────────────────────────────────────────
    params = _suggest_params(trial, config)

    # ── Model, loss, optimiser, scheduler ────────────────────────────────────
    # Loss follows the task: cross-entropy (classification) or MSE (regression).
    model, _  = build_model(config, params, X.shape, DEVICE)
    criterion = task.make_criterion(
        config, TRAIN_PROTOCOL["loss"]).to(DEVICE)
    optimizer = make_optimizer(
        model.parameters(), params, TRAIN_PROTOCOL["optimizer"])
    scheduler = make_scheduler(
        optimizer, n_epochs, TRAIN_PROTOCOL["scheduler"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mse     = float('inf')
    patience         = TRAIN_PROTOCOL['patience']   # protocol constant (hashed)
    patience_counter = 0
    weights_path     = os.path.join(
        output_dir, f"weights_{config['name']}_trial_{trial.number}.pth"
    )

    for epoch in range(n_epochs):
        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate using the executable, protocol-hashed primary-estimand
        # mapping. Bearing rungs require scour_mse; other tasks require mse.
        val_mse = task.objective_value(
            task.evaluate(model, val_loader, config, DEVICE),
            config,
            TRAIN_PROTOCOL["objective"],
        )

        # Optuna pruning
        trial.report(val_mse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Patience / weight saving
        if val_mse < best_val_mse:
            best_val_mse     = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_mse


# ──────────────────────────────────────────────────────────────────────────────
# 2. Fixed-seed single run (legacy parametric diagnostic)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_training(
    config:   dict,
    params:   dict,
    X:        np.ndarray,
    y:        np.ndarray,
    seed:     int,
    n_epochs: int,
    groups:   np.ndarray | None = None,
    dataset_name: str | None = None,   # Feature A: locates the state table
) -> tuple[float, float, float]:
    """
    Build, train, and evaluate one model with a fixed seed and known params.

    Retained for the legacy one-at-a-time parametric diagnostic. Development
    adjudication and post-freeze stability use ``fit_predict_fixed_group_fold``
    so every split has fold-local scaling.
    Data is passed in directly (already memory-mapped) to avoid redundant
    cache lookups across perturbation runs.

    Args:
        config  (dict):       Ablation step config.
        params  (dict):       Fixed hyperparameter dict (Optuna best_params).
        X       (np.ndarray): Memory-mapped feature array.
        y       (np.ndarray): Memory-mapped label array.
        seed    (int):        Random seed for init/shuffle only; the split stays
                              canonical (seed 42, grouped) so the cached scaler
                              remains valid.
        n_epochs (int):       Training epochs (no early stopping - full run).
        groups  (np.ndarray | None): damage-state id per sample from
                              get_or_create_cache - grouped split when given
                              (audit fix 2026-07-17); None = legacy
                              per-passage split (classification only).

    Returns:
        dict of validation metrics (see core.task.evaluate). Always carries
        'primary' (accuracy | localisation_acc), 'mae', and 'mse'; regression
        adds 'per_head_mse'/'per_head_mae'. Units are class-index (classification)
        or % scour (regression).
    """
    set_global_seed(seed, TRAIN_PROTOCOL["determinism"])

    # AUDIT FIX 2026-07-17: the split stays CANONICAL (seed 42, grouped) and only
    # the init/shuffle varies with `seed`. The scaler in the cache was fit on the
    # seed-42 grouped train partition; varying the split per seed (the old
    # behaviour) let validation groups of some seeds leak into the scaler fit.
    # New multi-split evaluations do not use this path: they call the explicit
    # grouped-fold refit below, which fits a scaler on each fold's train_idx.
    train_idx, val_idx = canonical_train_val_split(len(y), groups,
                                                   dataset_name=dataset_name)

    label_dtype = task.label_dtype(config)
    train_loader = DataLoader(
        MemmapDataset(X, y, train_idx, label_dtype=label_dtype),
        batch_size=TRAIN_PROTOCOL['batch_size'], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MemmapDataset(X, y, val_idx, label_dtype=label_dtype),
        batch_size=TRAIN_PROTOCOL['batch_size'], shuffle=False,
    )

    model, _  = build_model(config, params, X.shape, DEVICE)
    criterion = task.make_criterion(
        config, TRAIN_PROTOCOL["loss"]).to(DEVICE)
    optimizer = make_optimizer(
        model.parameters(), params, TRAIN_PROTOCOL["optimizer"])
    scheduler = make_scheduler(
        optimizer, n_epochs, TRAIN_PROTOCOL["scheduler"])

    model.train()
    for _ in range(n_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(DEVICE)), batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
        scheduler.step()

    return task.evaluate(model, val_loader, config, DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Optuna wrappers
# ──────────────────────────────────────────────────────────────────────────────

def fit_predict_finalist_fold(
    config: dict,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold,
    seed: int,
    n_epochs: int | None = None,
    *,
    max_epochs: int,
    n_scour_heads: int,
) -> dict[str, np.ndarray]:
    """Fixed-parameter refit and inference on one explicit grouped fold.

    This is the single implementation shared by the campaign driver and the
    robustness interfaces. Fold-local scaling is fitted on ``train_idx`` only;
    validation never chooses a checkpoint or changes hyperparameters. RAW and
    PAA use this same implementation and differ only in their input arrays.
    """

    method = str(config.get("method", "")).lower()
    if method not in {"raw", "paa"}:
        raise RuntimeError(
            "fixed grouped-fold refits support RAW and affine-standardised PAA "
            "features only; add an explicit fold-scaler implementation for "
            f"method={config.get('method')!r}"
        )
    if not task.is_regression(config):
        raise ValueError("finalist-CV refit requires a regression config")
    if isinstance(max_epochs, bool) or int(max_epochs) != max_epochs:
        raise ValueError("max_epochs must be a positive integer")
    max_epochs = int(max_epochs)
    if max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    if n_epochs is None:
        n_epochs = max_epochs
    if isinstance(n_epochs, bool) or int(n_epochs) != n_epochs:
        raise ValueError("n_epochs must be an integer")
    n_epochs = int(n_epochs)
    if not 1 <= n_epochs <= max_epochs:
        raise ValueError(
            f"finalist-CV refit epochs must be in [1, {max_epochs}], "
            f"got {n_epochs}"
        )
    if (
        isinstance(n_scour_heads, bool)
        or not isinstance(n_scour_heads, (int, np.integer))
        or int(n_scour_heads) != task.n_scour_outputs(config)
    ):
        raise ValueError(
            "n_scour_heads must equal the regression config's scour outputs")
    n_scour_heads = int(n_scour_heads)

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y and groups must contain the same samples")

    mean, scale = channel_standardization_stats(X, fold.train_idx)
    set_global_seed(seed, TRAIN_PROTOCOL["determinism"])
    train_loader = DataLoader(
        FoldStandardizedDataset(
            X, y, fold.train_idx, mean, scale,
            label_dtype=task.label_dtype(config),
        ),
        batch_size=TRAIN_PROTOCOL["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FoldStandardizedDataset(
            X, y, fold.val_idx, mean, scale,
            label_dtype=task.label_dtype(config),
        ),
        batch_size=TRAIN_PROTOCOL["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    model, _ = build_model(config, params, X.shape, DEVICE)
    criterion = task.make_criterion(
        config, TRAIN_PROTOCOL["loss"]).to(DEVICE)
    optimizer = make_optimizer(
        model.parameters(), params, TRAIN_PROTOCOL["optimizer"])
    # A k-epoch refit reproduces the first k steps of the original max-epoch
    # schedule; it does not compress a fresh cosine cycle into k.
    scheduler = make_scheduler(
        optimizer, max_epochs, TRAIN_PROTOCOL["scheduler"])
    for _ in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    predictions, truth = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions.append(model(batch_x.to(DEVICE)).cpu().numpy())
            truth.append(batch_y.numpy())
    return per_state_regression_metrics(
        np.vstack(predictions),
        np.vstack(truth),
        groups[np.asarray(fold.val_idx, dtype=np.int64)],
        n_scour_heads=n_scour_heads,
    )


# Truthful public name for new callers. Keep the historical finalist-CV name
# above because the campaign driver and retained benchmark import it directly.
fit_predict_fixed_group_fold = fit_predict_finalist_fold


class Objective:
    """
    Callable wrapper that binds a fixed config to train_and_evaluate so that
    Optuna's study.optimize() can call it as objective(trial).

    Args:
        config       (dict): Ablation step config.
        dataset_name (str):  Dataset sub-folder name.
        n_epochs     (int):  Maximum epochs per trial.
        cache_dir    (str):  Cache directory.
        output_dir   (str):  Directory for per-trial weight files.
    """

    def __init__(
        self,
        config:       dict,
        dataset_name: str,
        n_epochs:     int,
        cache_dir:    str,
        output_dir:   str,
    ):
        self.config       = config
        self.dataset_name = dataset_name
        self.n_epochs     = n_epochs
        self.cache_dir    = cache_dir
        self.output_dir   = output_dir

    def __call__(self, trial) -> float:
        return train_and_evaluate(
            trial,
            config=self.config,
            dataset_name=self.dataset_name,
            n_epochs=self.n_epochs,
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
        )


def print_best_callback(study, trial) -> None:
    """
    Optuna callback that prints a summary only when a new best trial is found.

    Uses tqdm.write so the message does not collide with Optuna's own
    progress bar when show_progress_bar=True.
    """
    if study.best_trial.number != trial.number:
        return
    tqdm.write(f"\n[BEST] NEW BEST - Trial {trial.number}  MSE: {trial.value:.4f}")
    params_str = ", ".join(f"{k}: {v}" for k, v in trial.params.items())
    tqdm.write(f"   Params: {params_str}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_params(trial, config: dict) -> dict:
    """
    Ask the Optuna trial to suggest all hyperparameters for the current config.

    Gated suggestions (LSTM params, adaptive pyramid-bin sets under the legacy
    N-HiTS key) are only requested
    when the corresponding architecture flag is active, keeping the search
    space minimal and the Optuna DB schema clean across ablation variants.

    PROTOCOL (2026-07-19): every range/choice comes from SEARCH_SPACE (the
    hashed protocol data) via _suggest_one — no literal here. The CALL ORDER
    is preserved exactly from the pre-refactor code (n_conv, n_dense, lr,
    weight_decay, per-conv blocks, per-dense blocks, lstm_num_layers before
    lstm_hidden_size, nhits) so a seeded sampler reproduces identical trials.
    """
    SS = SEARCH_SPACE
    frozen = config.get("frozen_hyperparameters")
    if frozen is not None and not isinstance(frozen, dict):
        raise ValueError("frozen_hyperparameters must be a mapping when present")

    def suggest(name: str, spec: tuple):
        if frozen is None:
            return _suggest_one(trial, name, spec)
        return _suggest_frozen_one(trial, name, spec, frozen)

    n_conv  = suggest('n_conv_layers',  SS['base']['n_conv_layers'])
    n_dense = suggest('n_dense_layers', SS['base']['n_dense_layers'])

    params = {
        'lr':             suggest('lr',           SS['base']['lr']),
        'weight_decay':   suggest('weight_decay', SS['base']['weight_decay']),
        'n_conv_layers':  n_conv,
        'n_dense_layers': n_dense,
    }

    for i in range(n_conv):        # per-conv-layer block, suffix _l{i}
        for key, spec in SS['per_conv_layer'].items():
            params[f'{key}_l{i}'] = suggest(f'{key}_l{i}', spec)

    for i in range(n_dense):       # per-dense-layer block, suffix _l{i}
        for key, spec in SS['per_dense_layer'].items():
            params[f'{key}_l{i}'] = suggest(f'{key}_l{i}', spec)

    if config.get('use_lstm', False):
        # ORDER MATTERS: lstm_num_layers is suggested BEFORE lstm_hidden_size
        # (as in the original code); lstm_dropout only exists for 2-layer LSTMs.
        n_lstm = suggest('lstm_num_layers', SS['lstm']['lstm_num_layers'])
        params['lstm_hidden_size'] = suggest(
            'lstm_hidden_size', SS['lstm']['lstm_hidden_size'])
        params['lstm_num_layers'] = n_lstm
        if n_lstm > 1:
            params['lstm_dropout'] = suggest(
                'lstm_dropout', SS['lstm']['lstm_dropout'])

    if config.get('use_nhits', False):
        params['nhits_pool_rates_key'] = suggest(
            'nhits_pool_rates_key', SS['nhits']['nhits_pool_rates_key'])

    if frozen is not None and set(params) != set(frozen):
        missing = sorted(set(params) - set(frozen))
        extra = sorted(set(frozen) - set(params))
        raise ValueError(
            "frozen hyperparameter keyset does not match the active "
            f"architecture (missing={missing}, extra={extra})"
        )

    return params
