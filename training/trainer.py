"""
training/trainer.py
===================
Everything that runs inside a single Optuna trial or a single fixed-seed
training run.

Public API
----------
    train_and_evaluate   - one Optuna trial: suggest -> build -> train -> prune.
    run_single_training  - one fixed-seed run with known params (robustness use).
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
from core.utils   import set_global_seed


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
    "optimizer":   "Adam(lr, weight_decay)",
    "scheduler":   "CosineAnnealingLR(T_max=n_epochs)",
    "objective":   "best epoch-wise inner-val aggregate MSE (task.objective_value)",
    "trial_seed":  "config['seed'] (independent init/shuffle per config seed)",
}

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
    set_global_seed(config.get('seed', 42))

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
    criterion = task.make_criterion(config)
    optimizer = optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

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

        # Validate - aggregate MSE (class-index for classification, % scour for
        # regression); the single scalar Optuna minimises for both tasks.
        val_mse = task.objective_value(task.evaluate(model, val_loader, config, DEVICE))

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
# 2. Fixed-seed single run (robustness evaluation)
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

    Used by the stochastic and parametric robustness evaluators to measure
    variance across random initialisations or parameter perturbations.
    Data is passed in directly (already memory-mapped) to avoid redundant
    cache lookups across the 30-seed loop.

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
    set_global_seed(seed)

    # AUDIT FIX 2026-07-17: the split stays CANONICAL (seed 42, grouped) and only
    # the init/shuffle varies with `seed`. The scaler in the cache was fit on the
    # seed-42 grouped train partition; varying the split per seed (the old
    # behaviour) let validation groups of some seeds leak into the scaler fit.
    # Stochastic robustness is meant to measure init/optimisation variance, not
    # split variance, so this is also the correct estimand. (True split/CV
    # resampling would need a per-split scaler — a separate analysis.)
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
    criterion = task.make_criterion(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params.get('weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

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

    Gated suggestions (LSTM params, N-HiTS pool rates) are only requested
    when the corresponding architecture flag is active, keeping the search
    space minimal and the Optuna DB schema clean across ablation variants.

    PROTOCOL (2026-07-19): every range/choice comes from SEARCH_SPACE (the
    hashed protocol data) via _suggest_one — no literal here. The CALL ORDER
    is preserved exactly from the pre-refactor code (n_conv, n_dense, lr,
    weight_decay, per-conv blocks, per-dense blocks, lstm_num_layers before
    lstm_hidden_size, nhits) so a seeded sampler reproduces identical trials.
    """
    SS = SEARCH_SPACE
    n_conv  = _suggest_one(trial, 'n_conv_layers',  SS['base']['n_conv_layers'])
    n_dense = _suggest_one(trial, 'n_dense_layers', SS['base']['n_dense_layers'])

    params = {
        'lr':             _suggest_one(trial, 'lr',           SS['base']['lr']),
        'weight_decay':   _suggest_one(trial, 'weight_decay', SS['base']['weight_decay']),
        'n_conv_layers':  n_conv,
        'n_dense_layers': n_dense,
    }

    for i in range(n_conv):        # per-conv-layer block, suffix _l{i}
        for key, spec in SS['per_conv_layer'].items():
            params[f'{key}_l{i}'] = _suggest_one(trial, f'{key}_l{i}', spec)

    for i in range(n_dense):       # per-dense-layer block, suffix _l{i}
        for key, spec in SS['per_dense_layer'].items():
            params[f'{key}_l{i}'] = _suggest_one(trial, f'{key}_l{i}', spec)

    if config.get('use_lstm', False):
        # ORDER MATTERS: lstm_num_layers is suggested BEFORE lstm_hidden_size
        # (as in the original code); lstm_dropout only exists for 2-layer LSTMs.
        n_lstm = _suggest_one(trial, 'lstm_num_layers', SS['lstm']['lstm_num_layers'])
        params['lstm_hidden_size'] = _suggest_one(
            trial, 'lstm_hidden_size', SS['lstm']['lstm_hidden_size'])
        params['lstm_num_layers'] = n_lstm
        if n_lstm > 1:
            params['lstm_dropout'] = _suggest_one(
                trial, 'lstm_dropout', SS['lstm']['lstm_dropout'])

    if config.get('use_nhits', False):
        params['nhits_pool_rates_key'] = _suggest_one(
            trial, 'nhits_pool_rates_key', SS['nhits']['nhits_pool_rates_key'])

    return params
