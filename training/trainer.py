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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import optuna
except ImportError:
    optuna = None   # pipeline.py guards against this; trainer is still importable

from core         import task
from core.dataset import MemmapDataset, get_or_create_cache
from core.models  import build_model
from core.utils   import set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Hardware
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    set_global_seed(42)

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y, _ = get_or_create_cache(config, dataset_name, cache_dir)

    all_idx            = np.arange(len(y))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.20, random_state=42)

    label_dtype = task.label_dtype(config)
    train_loader = DataLoader(
        MemmapDataset(X, y, train_idx, label_dtype=label_dtype),
        batch_size=32, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MemmapDataset(X, y, val_idx, label_dtype=label_dtype),
        batch_size=32, shuffle=False,
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
    patience         = 5
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
        seed    (int):        Random seed for this run.
        n_epochs (int):       Training epochs (no early stopping - full run).

    Returns:
        dict of validation metrics (see core.task.evaluate). Always carries
        'primary' (accuracy | localisation_acc), 'mae', and 'mse'; regression
        adds 'per_head_mse'/'per_head_mae'. Units are class-index (classification)
        or % scour (regression).
    """
    set_global_seed(seed)

    all_idx            = np.arange(len(y))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.20, random_state=seed)

    label_dtype = task.label_dtype(config)
    train_loader = DataLoader(
        MemmapDataset(X, y, train_idx, label_dtype=label_dtype),
        batch_size=32, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MemmapDataset(X, y, val_idx, label_dtype=label_dtype),
        batch_size=32, shuffle=False,
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
    """
    n_conv  = trial.suggest_int('n_conv_layers',  2, 4)
    n_dense = trial.suggest_int('n_dense_layers', 1, 3)

    params = {
        'lr':             trial.suggest_float('lr',           1e-4, 1e-2, log=True),
        'weight_decay':   trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'n_conv_layers':  n_conv,
        'n_dense_layers': n_dense,
    }

    for i in range(n_conv):
        params[f'n_filters_l{i}']   = trial.suggest_int(
            f'n_filters_l{i}', 16, 128, step=16
        )
        params[f'kernel_size_l{i}'] = trial.suggest_categorical(
            f'kernel_size_l{i}', [2, 3, 5, 7]
        )
        params[f'pooling_l{i}']     = trial.suggest_categorical(
            f'pooling_l{i}', [True, False]
        )

    for i in range(n_dense):
        params[f'n_dense_units_l{i}'] = trial.suggest_int(
            f'n_dense_units_l{i}', 32, 256, step=16
        )
        params[f'dropout_l{i}'] = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)

    if config.get('use_lstm', False):
        n_lstm = trial.suggest_int('lstm_num_layers', 1, 2)
        params['lstm_hidden_size'] = trial.suggest_int(
            'lstm_hidden_size', 32, 128, step=32
        )
        params['lstm_num_layers'] = n_lstm
        if n_lstm > 1:
            params['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.1, 0.4)

    if config.get('use_nhits', False):
        params['nhits_pool_rates_key'] = trial.suggest_categorical(
            'nhits_pool_rates_key', ["1_2_4", "1_4_8", "1_3_6", "1_2_4_8"]
        )

    return params
