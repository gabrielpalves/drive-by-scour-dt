"""
plotting/confusion.py
=====================
Confusion matrix functions for the ablation pipeline.

Public API
----------
    plot_cached_confusion_matrix   - evaluates the champion model on the
                                     canonical validation set and plots its
                                     true confusion matrix.  Called once per
                                     model inside execute_ablation_pipeline.

    plot_aggregated_confusion_matrix - pure render function: takes pre-computed
                                       prediction arrays and produces a heatmap.
                                       Called by plot_cached_confusion_matrix and
                                       usable standalone for ad-hoc inspection.

Note on plot_best_model_confusion_matrix
----------------------------------------
The original script contained a third function of this name that rebuilt the
model from study.best_params but did NOT load the saved weights, producing a
randomly-initialised model (a silent bug).  That function is not reproduced
here.  plot_cached_confusion_matrix is the correct replacement: it loads the
champion weights written by train_and_evaluate and evaluates a live model.

Imported by:
    training/pipeline.py - plot_cached_confusion_matrix
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.dataset import MemmapDataset, get_or_create_cache
from core.models  import build_model
from training.trainer import DEVICE


# ──────────────────────────────────────────────────────────────────────────────
# 1. Full pipeline: load champion -> infer -> render
# ──────────────────────────────────────────────────────────────────────────────

def plot_cached_confusion_matrix(
    study,
    config:       dict,
    dataset_name: str,
    cache_dir:    str,
    output_dir:   str,
) -> None:
    """
    Evaluate the champion model on the canonical validation set and save its
    confusion matrix as a PNG.

    Steps
    -----
    1. Load the preprocessed data via get_or_create_cache.
    2. Isolate the canonical validation partition (seed 42, 20 %).
    3. Rebuild the champion architecture from study.best_params.
    4. Load the champion weights saved by train_and_evaluate.
    5. Run inference and collect predictions.
    6. Delegate rendering to plot_aggregated_confusion_matrix.

    Weight file resolution
    ----------------------
    Looks for  weights_<name>_trial_<best_trial_number>.pth  first, then
    falls back to  DT_champion_weights.pth  (written by
    export_digital_twin_package).  Raises FileNotFoundError if neither exists
    so the caller is notified immediately rather than silently plotting a
    random-weight model.

    Args:
        study:            Optuna study for this model.
        config (dict):    Ablation step config.
        dataset_name:     Dataset sub-folder name.
        cache_dir (str):  Cache directory.
        output_dir (str): Directory to write the PNG.

    Raises:
        FileNotFoundError: If no weight file can be located.
    """
    print(f"\n--> Confusion matrix: {config['name']}")

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y, _ = get_or_create_cache(config, dataset_name, cache_dir)
    all_idx  = np.arange(len(y))
    _, val_idx = train_test_split(all_idx, test_size=0.20, random_state=42)

    val_loader = DataLoader(
        MemmapDataset(X, y, val_idx),
        batch_size=32, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, n_classes = build_model(config, study.best_params, X.shape, DEVICE)

    weights_path = _resolve_weights_path(output_dir, config['name'], study.best_trial.number)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_trues, all_preds = _collect_predictions(model, val_loader)

    # ── Render ────────────────────────────────────────────────────────────────
    plot_aggregated_confusion_matrix(
        all_trues, all_preds, n_classes,
        study_name=config['name'],
        output_dir=output_dir,
    )
    print(f"    Confusion matrix saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pure render function
# ──────────────────────────────────────────────────────────────────────────────

def plot_aggregated_confusion_matrix(
    all_trues:  list | np.ndarray,
    all_preds:  list | np.ndarray,
    n_classes:  int,
    study_name: str,
    output_dir: str,
) -> None:
    """
    Render and save a confusion matrix heatmap from pre-computed predictions.

    Visual design
    -------------
    - Figure size scales with n_classes so both 13-class and 61-class matrices
      remain legible without manual adjustment.
    - Non-zero cells are annotated with their count; zero cells are blank to
      reduce visual noise.
    - Diagonal cells are outlined in red so correct predictions stand out
      regardless of colour intensity.
    - The raw confusion matrix is also saved as a .npy file alongside the PNG
      for downstream statistical analysis.

    Args:
        all_trues  (array-like): Ground-truth class indices, length N.
        all_preds  (array-like): Predicted class indices, length N.
        n_classes  (int):        Total number of damage classes.
        study_name (str):        Used in the plot title and output filename.
        output_dir (str):        Directory to write PNG and NPY files.
    """
    labels = list(range(n_classes))
    cm     = confusion_matrix(all_trues, all_preds, labels=labels)
    np.save(os.path.join(output_dir, "DT_conf_matrix.npy"), cm)

    # Scale figure: ~0.25 in per class, minimum 8 in
    fig_size = max(8, n_classes * 0.25)
    fig, ax  = plt.subplots(figsize=(fig_size + 2, fig_size))

    # Annotate non-zero cells only
    annot = np.where(cm > 0, cm.astype(str), "")

    sns.heatmap(
        cm,
        ax=ax,
        annot=annot,
        fmt='',
        cmap='Blues',
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 8 if n_classes > 30 else 10, "weight": "bold"},
        linewidths=0.2,
        linecolor='lightgray',
    )

    # Red outline on every diagonal cell
    for i in range(n_classes):
        ax.add_patch(
            patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='#E74C3C', lw=2)
        )

    title = study_name.replace("_", " ")
    ax.set_title(f"Risk categorisation: {title}", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("CNN predicted severity", fontsize=14, fontweight='bold')
    ax.set_ylabel("True physical severity",  fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, rotation=90)
    ax.tick_params(axis='y', labelsize=8, rotation=0)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"Aggregated_CM_{study_name}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_weights_path(output_dir: str, model_name: str, best_trial_num: int) -> str:
    """
    Return the path to the champion weight file, trying the per-trial name
    first and the exported DT name second.

    Raises:
        FileNotFoundError: If neither file exists.
    """
    primary  = os.path.join(output_dir, f"weights_{model_name}_trial_{best_trial_num}.pth")
    fallback = os.path.join(output_dir, "DT_champion_weights.pth")

    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Champion weights not found.\n"
        f"  Tried: {primary}\n"
        f"  Tried: {fallback}\n"
        f"Ensure train_and_evaluate saved weights during the Optuna study, "
        f"or run export_digital_twin_package first."
    )


def _collect_predictions(
    model:  torch.nn.Module,
    loader: DataLoader,
) -> tuple[list, list]:
    """Run inference on loader and return (true_labels, predicted_labels)."""
    all_trues, all_preds = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            _, predicted = torch.max(model(batch_X.to(DEVICE)), 1)
            all_trues.extend(batch_y.numpy())
            all_preds.extend(predicted.cpu().numpy())
    return all_trues, all_preds