"""
training/robustness.py
======================
Two post-optimisation stress tests that measure how trustworthy a champion
model actually is beyond its Optuna lucky-score:

    evaluate_stochastic_robustness  - 30-seed Monte Carlo: does the architecture
                                      consistently reproduce its best result, or
                                      was the Optuna run a fluke?

    evaluate_parametric_robustness  - ±5/10 % perturbation of key hyperparameters
                                      around the Optuna optimum: how sensitive is
                                      the model to small deviations from the chosen
                                      configuration?

Both functions write results to disk after every single run so that a
KeyboardInterrupt or crash loses at most one data point, and resume
automatically from wherever they left off on the next call.

Imported by:
    training/pipeline.py - called from execute_ablation_pipeline after
                           plot_cached_confusion_matrix and export_digital_twin_package.
"""

import copy
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from core.dataset import get_or_create_cache
from training.trainer import run_single_training


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stochastic robustness  (30-seed Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_stochastic_robustness(
    study,
    config:                  dict,
    dataset_name:            str,
    n_epochs:                int   = 50,
    physical_error_tolerance: float = 10.0,
    cache_dir:               str   = '',
    output_dir:              str   = '',
    skip_robustness:         bool  = True,
) -> dict | None:
    """
    Re-train the champion architecture 30 times with different random seeds
    and report the distribution of MSE, MAE, and accuracy.

    Gatekeeper
    ----------
    If the Optuna best MSE already exceeds the physical error tolerance
    threshold and skip_robustness=True, the test is skipped entirely.  The
    threshold is expressed in physical damage-% units and converted to class
    units using the config's discretisation step:

        max_mse_threshold = (physical_error_tolerance / disc) ** 2

    Checkpointing
    -------------
    Results are written to robustness_stochastic.json after every seed.
    On re-entry the function loads existing results and resumes from where
    it left off, so the full 30-run cost is paid only once.

    Args:
        study:                    Optuna study object for the champion.
        config (dict):            Ablation step config.
        dataset_name (str):       Dataset sub-folder name.
        n_epochs (int):           Epochs per seed run.
        physical_error_tolerance: Acceptable physical damage error in %.
        cache_dir (str):          Cache directory.
        output_dir (str):         Directory for JSON checkpoint and boxplot PNG.
        skip_robustness (bool):   If True, skip when Optuna score > threshold.

    Returns:
        dict with keys Optuna_Lucky_Score, Stochastic_Mean_MSE,
        Stochastic_Std_MSE, UCB_95_MSE - or None if skipped.
    """
    print(f"\n--> Stochastic robustness: {config['name']}")

    disc              = config.get('discretization', 1.0)
    max_mse_threshold = (physical_error_tolerance / disc) ** 2

    if study.best_value > max_mse_threshold and skip_robustness:
        print(
            f"  [SKIP] Best MSE ({study.best_value:.2f}) > threshold "
            f"({max_mse_threshold:.2f})  "
            f"(>{physical_error_tolerance}% physical error). Skipping."
        )
        return None

    X, y, _ = get_or_create_cache(config, dataset_name, cache_dir)
    best_params   = study.best_params
    json_path     = os.path.join(output_dir, "robustness_stochastic.json")
    n_seeds       = 30

    # ── Load existing checkpoint ──────────────────────────────────────────────
    accs, maes, mses = _load_stochastic_checkpoint(json_path)
    completed        = len(accs)

    # ── Run remaining seeds ───────────────────────────────────────────────────
    if completed < n_seeds:
        print(f"  Running seeds {completed + 1}-{n_seeds}...")
        pbar = tqdm(range(completed, n_seeds), desc="Monte Carlo seeds", unit="model")
        for run in pbar:
            seed = 42 + run
            m    = run_single_training(
                config, best_params, X, y, seed=seed, n_epochs=n_epochs
            )
            # 'primary' = accuracy (classification) or localisation_acc (regression)
            accs.append(m['primary']); maes.append(m['mae']); mses.append(m['mse'])
            _save_stochastic_checkpoint(json_path, accs, maes, mses)
            pbar.set_postfix({"seed": seed, "MSE": f"{m['mse']:.4f}"})

    # ── Boxplot ───────────────────────────────────────────────────────────────
    _plot_stochastic_boxplot(accs, maes, mses, config['name'], output_dir)

    # ── Scorecard ─────────────────────────────────────────────────────────────
    mu_mse  = float(np.mean(mses))
    std_mse = float(np.std(mses))
    ucb_mse = mu_mse + 2.0 * std_mse

    print(f"  Mean MSE: {mu_mse:.4f}  Std: {std_mse:.4f}  UCB 95%: {ucb_mse:.4f}")

    return {
        'Optuna_Lucky_Score':   study.best_value,
        'Stochastic_Mean_MSE':  mu_mse,
        'Stochastic_Std_MSE':   std_mse,
        'UCB_95_MSE':           ucb_mse,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Parametric robustness  (Todd hyperparameter perturbation test)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_parametric_robustness(
    study,
    config:       dict,
    dataset_name: str,
    baseline_mse: float,
    n_epochs:     int  = 50,
    cache_dir:    str  = '',
    output_dir:   str  = '',
) -> tuple[float, float]:
    """
    Perturb each key hyperparameter by ±5 % and ±10 % around the Optuna
    optimum (one parameter at a time, all others held fixed) and measure how
    much the validation MSE degrades.

    Parameters tested
    -----------------
    Always: lr, weight_decay.
    When present in best_params: dropout_l0, n_filters_l0, lstm_hidden_size.

    Perturbation multipliers: 0.90, 0.95, 1.00 (baseline), 1.05, 1.10.
    Integer params are rounded; the result is clamped to a minimum of 1.

    Checkpointing
    -------------
    Results are written to robustness_sensitivity.json after every single
    perturbation run.  Re-entry resumes from the first uncompleted task.

    Args:
        study:            Optuna study for the champion.
        config (dict):    Ablation step config.
        dataset_name:     Dataset sub-folder name.
        baseline_mse:     The stochastic mean MSE to measure degradation against.
        n_epochs (int):   Epochs per perturbation run.
        cache_dir (str):  Cache directory.
        output_dir (str): Directory for JSON checkpoint.

    Returns:
        (worst_mse, max_degradation): Worst perturbed MSE and the delta above
        baseline_mse.
    """
    print(f"\n--> Parametric robustness: {config['name']}")

    X, y, _      = get_or_create_cache(config, dataset_name, cache_dir)
    best_params  = study.best_params
    json_path    = os.path.join(output_dir, "robustness_sensitivity.json")
    multipliers  = [0.90, 0.95, 1.00, 1.05, 1.10]

    # Parameters to perturb (only those present in best_params)
    candidate_params = ['lr', 'weight_decay', 'dropout_l0', 'n_filters_l0', 'lstm_hidden_size']
    params_to_test   = {p: best_params[p] for p in candidate_params if p in best_params}

    # ── Load existing checkpoint ──────────────────────────────────────────────
    results = _load_sensitivity_checkpoint(json_path)

    # ── Build task list (skip already-completed combinations) ─────────────────
    tasks = []
    for param_name, base_val in params_to_test.items():
        results.setdefault(param_name, {})
        for mult in multipliers:
            key = f"{mult * 100:.0f}%"
            if key not in results[param_name]:
                tasks.append((param_name, base_val, mult, key))

    # ── Run remaining tasks ───────────────────────────────────────────────────
    if tasks:
        print(f"  {len(tasks)} perturbation runs remaining...")
        pbar = tqdm(tasks, desc="Perturbations", unit="run")
        for param_name, base_val, mult, key in pbar:
            pbar.set_postfix({"param": param_name, "mult": key})

            perturbed                = copy.deepcopy(best_params)
            new_val                  = _perturb(base_val, mult)
            perturbed[param_name]    = new_val

            m = run_single_training(
                config, perturbed, X, y, seed=42, n_epochs=n_epochs
            )
            results[param_name][key] = {'mae': m['mae'], 'mse': m['mse']}
            _save_sensitivity_checkpoint(json_path, results)
            pbar.set_postfix({"param": param_name, "mult": key, "MSE": f"{mse:.4f}"})

    # ── Worst-case degradation ────────────────────────────────────────────────
    worst_mse = max(
        v['mse']
        for param_data in results.values()
        for v in param_data.values()
    )
    max_degradation = worst_mse - baseline_mse
    print(f"  Worst perturbed MSE: {worst_mse:.4f}  (+{max_degradation:.4f} vs baseline)")

    return worst_mse, max_degradation


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers - checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def _load_stochastic_checkpoint(
    path: str,
) -> tuple[list, list, list]:
    if not os.path.exists(path):
        return [], [], []
    with open(path) as f:
        data = json.load(f)
    return (
        data.get('accuracies', []),
        data.get('maes',       []),
        data.get('mses',       []),
    )


def _save_stochastic_checkpoint(
    path: str,
    accs: list,
    maes: list,
    mses: list,
) -> None:
    with open(path, 'w') as f:
        json.dump({'accuracies': accs, 'maes': maes, 'mses': mses}, f)


def _load_sensitivity_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_sensitivity_checkpoint(path: str, results: dict) -> None:
    with open(path, 'w') as f:
        json.dump(results, f)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers - perturbation arithmetic
# ──────────────────────────────────────────────────────────────────────────────

def _perturb(base_val: float | int, multiplier: float) -> float | int:
    """
    Scale base_val by multiplier.  Integer params are rounded and clamped
    to a minimum of 1 so the model never receives an invalid architecture dim.
    """
    new_val = base_val * multiplier
    if isinstance(base_val, int):
        return max(1, round(new_val))
    return new_val


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers - plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_stochastic_boxplot(
    accs:       list,
    maes:       list,
    mses:       list,
    study_name: str,
    output_dir: str,
) -> None:
    """
    Three-panel boxplot (MSE, MAE, Accuracy) with individual seed points
    overlaid as a strip plot.  Saved as a PNG; figure is closed immediately
    to prevent RAM accumulation across many models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Stochastic Robustness (30 seeds) - {study_name}",
        fontsize=14, fontweight='bold', y=1.02,
    )

    _panel(axes[0], mses,  'lightcoral',   'darkred',   'Mean Squared Error (MSE)', 'MSE')
    _panel(axes[1], maes,  'lightskyblue', 'darkblue',  'Mean Absolute Error (MAE)', 'MAE')
    # 'primary' = strict accuracy (classification) or localisation accuracy (regression)
    _panel(axes[2], accs,  'lightgreen',   'darkgreen', 'Primary (accuracy / localisation)', 'Primary')

    plt.tight_layout()
    path = os.path.join(output_dir, f"Stochastic_Boxplot_{study_name}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def _panel(
    ax,
    data:       list,
    box_colour: str,
    dot_colour: str,
    title:      str,
    ylabel:     str,
) -> None:
    sns.boxplot(y=data,  ax=ax, color=box_colour, width=0.4, fliersize=0)
    sns.stripplot(y=data, ax=ax, color=dot_colour, alpha=0.6, jitter=True, size=5)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
