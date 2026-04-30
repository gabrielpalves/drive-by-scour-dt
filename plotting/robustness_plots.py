"""
plotting/robustness_plots.py
============================
Three summary-level plotting functions that operate across multiple studies
or multiple seeds rather than on a single model in isolation.

Public API
----------
    generate_optuna_robustness_plots — per-parameter slice plots for one study,
                                       showing how each hyperparameter relates
                                       to validation error across all trials.

    plot_stochastic_summary          — cross-model boxplot comparison of the
                                       30-seed Monte Carlo distributions
                                       (MSE, MAE, Accuracy) for every model
                                       in the ablation grid that passed the
                                       robustness gatekeeper.

    plot_parametric_summary          — per-hyperparameter line plots showing
                                       how the champion's validation MSE,
                                       MAE, and Accuracy change under ±5/10 %
                                       perturbations.

Contrast with plotting/confusion.py
-------------------------------------
confusion.py operates on a single model and a single validation pass.
This module always aggregates: across trials (slice plots), across models
(stochastic summary), or across perturbation multipliers (parametric summary).

Imported by:
    training/pipeline.py — all three functions, at different points in
                           execute_ablation_pipeline.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# 1. Per-parameter Optuna slice plots
# ──────────────────────────────────────────────────────────────────────────────

def generate_optuna_robustness_plots(
    study,
    config:     dict,
    output_dir: str,
) -> None:
    """
    Save one scatter plot per hyperparameter showing how its value across all
    completed Optuna trials correlates with the validation error.

    The best trial is highlighted as a red star.  Log scale is applied to lr
    and weight_decay so the spread of values is legible.  Each plot is saved
    individually into a Slice_Plots/ sub-folder so they do not clutter the
    model's root output directory.

    Args:
        study:            Completed Optuna study.
        config (dict):    Ablation step config (used for the plot subtitle).
        output_dir (str): Root output directory for this model; Slice_Plots/
                          is created inside it.
    """
    print(f"--> Optuna slice plots: {config['name']}")

    df         = study.trials_dataframe()
    df         = df[df['state'] == 'COMPLETE']
    param_cols = [c for c in df.columns if c.startswith('params_')]

    if not param_cols:
        print("  [WARNING] No completed trials with parameters found.")
        return

    slice_dir = os.path.join(output_dir, "Slice_Plots")
    os.makedirs(slice_dir, exist_ok=True)

    best_val = study.best_value

    for col in param_cols:
        param_name = col.removeprefix('params_')

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=df, x=col, y='value',
            ax=ax, color='#3498DB', alpha=0.6, s=80, edgecolor='black',
        )

        best_param = study.best_params.get(param_name)
        if best_param is not None:
            ax.scatter(
                [best_param], [best_val],
                color='#E74C3C', marker='*', s=400,
                edgecolor='black', zorder=5, label='Best config',
            )
            ax.legend()

        ax.set_title(
            f"Sensitivity: {param_name}\n({config['name']})",
            fontweight='bold', fontsize=16, pad=15,
        )
        ax.set_xlabel(f"{param_name}", fontweight='bold', fontsize=14)
        ax.set_ylabel("Validation error (lower is better)", fontweight='bold', fontsize=14)

        if param_name in ('lr', 'weight_decay'):
            ax.set_xscale('log')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(slice_dir, f"Slice_{param_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"    {len(param_cols)} slice plots saved → {slice_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cross-model stochastic summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_stochastic_summary(
    path_list:          list[dict],
    db_storage:         str,
    experiment_root_dir: str,
    summary_output_dir:  str,
    file_prefix:        str = 'stochastic_summary',
) -> None:
    """
    Produce three side-by-side boxplot figures comparing the 30-seed
    robustness distributions of every model in path_list for which a
    robustness_stochastic.json file exists.

    Figure 1 — MSE (log scale) with 95 % UCB diamonds.
    Figure 2 — MAE (linear scale).
    Figure 3 — Accuracy (linear scale, y-axis 0–1).

    The Optuna lucky-score for each model is overlaid as a gold star so the
    gap between the one-shot Optuna result and the true stochastic mean is
    immediately visible.

    Skips any entry in path_list whose study cannot be loaded (e.g. because
    the study was never run) or whose JSON checkpoint does not exist, without
    raising an error, so partial ablation runs still produce useful plots.

    Args:
        path_list (list[dict]):    The ablation grid — each dict must have a
                                   'name' key matching an Optuna study name.
        db_storage (str):          Optuna storage URL for the relevant phase.
        experiment_root_dir (str): Root directory containing one sub-folder per
                                   model (each holding robustness_stochastic.json).
        summary_output_dir (str):  Directory to write the three PNG files.
        file_prefix (str):         Prefix for output filenames.
    """
    print(f"\n--> Stochastic summary plots (prefix: {file_prefix})")

    names                             = []
    mses_list, maes_list, accs_list   = [], [], []
    optuna_mses, optuna_maes, optuna_accs = [], [], []
    ucb_mses                          = []

    for step in path_list:
        study_name = step['name']

        try:
            study   = optuna.load_study(study_name=study_name, storage=db_storage)
            o_mse   = study.best_value
            o_mae   = study.best_trial.user_attrs.get('MAE')
            o_acc   = study.best_trial.user_attrs.get('Accuracy')
        except Exception:
            continue

        json_path = os.path.join(experiment_root_dir, study_name, 'robustness_stochastic.json')
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        mses = data.get('mses',       [])
        maes = data.get('maes',       [])
        accs = data.get('accuracies', [])

        if not (mses and maes and accs):
            continue

        names.append(study_name.replace('_', ' '))
        mses_list.append(mses);  maes_list.append(maes);  accs_list.append(accs)
        optuna_mses.append(o_mse)
        optuna_maes.append(o_mae)
        optuna_accs.append(o_acc)
        ucb_mses.append(float(np.mean(mses)) + 2.0 * float(np.std(mses)))

    if not names:
        print("  [Abort] No valid stochastic data found.")
        return

    os.makedirs(summary_output_dir, exist_ok=True)
    x = np.arange(len(names))

    _summary_boxplot(
        data_list=mses_list, x_coords=x, names=names,
        optuna_scores=optuna_mses, ucb_scores=ucb_mses,
        ylabel='Mean squared error (log scale)',
        title='Architecture robustness: MSE',
        path=os.path.join(summary_output_dir, f"{file_prefix}_MSE.png"),
        is_log=True,
    )
    _summary_boxplot(
        data_list=maes_list, x_coords=x, names=names,
        optuna_scores=optuna_maes, ucb_scores=None,
        ylabel='Mean absolute error (classes off)',
        title='Architecture robustness: MAE',
        path=os.path.join(summary_output_dir, f"{file_prefix}_MAE.png"),
    )
    _summary_boxplot(
        data_list=accs_list, x_coords=x, names=names,
        optuna_scores=optuna_accs, ucb_scores=None,
        ylabel='Validation accuracy',
        title='Architecture robustness: Accuracy',
        path=os.path.join(summary_output_dir, f"{file_prefix}_Accuracy.png"),
        ylim=(0, 1.05),
    )

    print(f"    MSE / MAE / Accuracy summary saved → {summary_output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Champion parametric sensitivity summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_parametric_summary(
    champion_name:       str,
    experiment_root_dir: str,
    summary_output_dir:  str,
) -> None:
    """
    Save one three-panel figure per perturbed hyperparameter showing how
    Accuracy, MAE, and MSE vary across the five perturbation multipliers
    (90 %, 95 %, 100 %, 105 %, 110 %).

    Reads the robustness_sensitivity.json written by
    evaluate_parametric_robustness.  Exits gracefully if the file does not
    exist.

    Args:
        champion_name (str):        Name of the champion model (matches the
                                    study name and its output sub-folder).
        experiment_root_dir (str):  Root directory containing the champion's
                                    sub-folder.
        summary_output_dir (str):   Directory to write the PNG files.
    """
    print(f"\n--> Parametric sensitivity plots: {champion_name}")

    sens_path = os.path.join(experiment_root_dir, champion_name, 'robustness_sensitivity.json')
    if not os.path.exists(sens_path):
        print(f"  [Abort] {sens_path} not found.")
        return

    with open(sens_path) as f:
        results = json.load(f)

    os.makedirs(summary_output_dir, exist_ok=True)

    for param_name, mult_data in results.items():
        x_labels = list(mult_data.keys())
        accs     = [d['acc'] for d in mult_data.values()] if 'acc' in next(iter(mult_data.values()), {}) else None
        maes     = [d['mae'] for d in mult_data.values()]
        mses     = [d['mse'] for d in mult_data.values()]

        # Locate the baseline column (100 %)
        baseline_idx = x_labels.index('100%') if '100%' in x_labels else len(x_labels) // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Parametric sensitivity — {champion_name}: {param_name}",
            fontsize=16, fontweight='bold', y=1.05,
        )

        _sensitivity_axis(axes[0], x_labels, accs or [],  param_name, 'Accuracy',         baseline_idx, ylim=(0, 1.0))
        _sensitivity_axis(axes[1], x_labels, maes,         param_name, 'Classes off (MAE)', baseline_idx)
        _sensitivity_axis(axes[2], x_labels, mses,         param_name, 'Squared error (MSE)', baseline_idx)

        plt.tight_layout()
        save_path = os.path.join(summary_output_dir, f"Parametric_Sensitivity_{param_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"    Parametric plots saved → {summary_output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _summary_boxplot(
    data_list:     list[list],
    x_coords:      np.ndarray,
    names:         list[str],
    optuna_scores: list,
    ucb_scores:    list | None,
    ylabel:        str,
    title:         str,
    path:          str,
    is_log:        bool  = False,
    ylim:          tuple | None = None,
) -> None:
    """
    Render and save one cross-model summary boxplot.

    Optuna lucky-scores are overlaid as gold stars.
    UCB diamonds (only for MSE) are overlaid in red with value annotations.
    Legend is placed outside the right edge of the plot so it never occludes
    the boxes regardless of how many models are compared.
    """
    plt.figure(figsize=(max(12, len(names) * 1.2), 8))

    sns.boxplot(
        data=data_list,
        color='#3498DB', width=0.5, fliersize=4,
        boxprops=dict(edgecolor='black', alpha=0.8),
        medianprops=dict(color='black', linewidth=2),
    )

    if None not in optuna_scores:
        plt.scatter(
            x_coords, optuna_scores,
            color='gold', marker='*', s=400,
            edgecolor='black', zorder=10, label='Optuna score',
        )

    if ucb_scores:
        plt.scatter(
            x_coords, ucb_scores,
            color='#E74C3C', marker='D', s=100,
            edgecolor='black', zorder=10, label='95 % UCB',
        )
        for i, val in enumerate(ucb_scores):
            plt.text(
                i + 0.15, val, f'{val:.2f}',
                ha='left', va='center',
                fontsize=11, fontweight='bold', color='#E74C3C', zorder=15,
            )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(ticks=x_coords, labels=names, rotation=30, ha='right', fontsize=11)

    if is_log:
        plt.yscale('log')
    if ylim:
        plt.ylim(*ylim)

    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12,
               framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def _sensitivity_axis(
    ax,
    x_labels:     list[str],
    y_vals:       list[float],
    param_name:   str,
    ylabel:       str,
    baseline_idx: int,
    ylim:         tuple | None = None,
) -> None:
    """
    Render one panel of the parametric sensitivity figure.

    A dashed vertical line marks the Optuna baseline (100 % multiplier).
    """
    if not y_vals:
        ax.set_visible(False)
        return

    ax.plot(x_labels, y_vals, marker='o', linestyle='-', color='#E74C3C', linewidth=2)
    ax.axvline(x=x_labels[baseline_idx], color='black', linestyle='--',
               label='Optuna baseline (100 %)')
    ax.set_title(f"{ylabel} vs {param_name}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Parameter multiplier", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
