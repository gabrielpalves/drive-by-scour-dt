"""
core/task.py
============
The single place that owns the difference between the two learning tasks the
ablation supports:

    'classification'  (default, single-scour) — predict a discretised damage
        CLASS (0..60/disc); loss = cross-entropy; metrics on the class index.
        This is the original, validated single-foundation ablation.

    'regression'      (multi-damage, Stage 0+) — predict a CONTINUOUS per-pier
        scour VECTOR (one output per target support); loss = MSE; metrics =
        per-head MSE, aggregate MSE/MAE, and a localisation accuracy.

The task is selected by `config['task']`. When the key is ABSENT every helper
falls back to classification, so the existing single-scour pipeline, the trained
champions, and the digital twin are unchanged. Multi-output regression is opt-in
by setting config['task']='regression' and config['target_supports']=[...].

Keeping head size, loss, label dtype, and metrics behind these four functions
(`n_outputs`, `make_criterion`, `label_dtype`, `evaluate`) means the trainer,
the model factory, and the dataset cache each carry a single task-aware call
rather than scattered `if task == ...` branches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# A pier counts as "damaged" (for localisation scoring) above this % scour; below
# it the most-damaged-pier argmax is noise and is excluded from the metric.
_LOC_DAMAGE_THRESHOLD_PCT = 5.0


def task_of(config: dict) -> str:
    """Task name; defaults to 'classification' when the key is absent."""
    return config.get("task", "classification")


def is_regression(config: dict) -> bool:
    return task_of(config) == "regression"


def n_outputs(config: dict) -> int:
    """Size of the model's final layer.

    classification -> number of damage classes (int(60/disc)+1).
    regression     -> number of target supports (one continuous head each).
    """
    if is_regression(config):
        targets = config.get("target_supports")
        if not targets:
            raise ValueError("regression task needs config['target_supports'] "
                             "(e.g. [2, 3] for the two internal piers).")
        return len(targets)
    disc = config.get("discretization", 1)
    return int(60 / disc) + 1


def make_criterion(config) -> nn.Module:
    """MSE for regression, cross-entropy for classification."""
    return nn.MSELoss() if is_regression(config) else nn.CrossEntropyLoss()


def label_dtype(config) -> torch.dtype:
    """Float targets for regression, integer class labels for classification."""
    return torch.float32 if is_regression(config) else torch.long


def objective_value(metrics: dict) -> float:
    """Scalar Optuna minimises — the (aggregate) validation MSE for both tasks."""
    return metrics["mse"]


@torch.no_grad()
def evaluate(model: nn.Module, loader, config: dict, device) -> dict:
    """Task-appropriate validation metrics in one pass.

    Returns a dict that always carries 'mse', 'mae', and 'primary' (so callers
    can stay task-agnostic), plus task-specific extras:
        classification : {primary=accuracy, accuracy, mae, mse}        (class-index units)
        regression     : {primary=localisation_acc, localisation_acc, mae, mse,
                          per_head_mse, per_head_mae}                   (% scour units)
    """
    model.eval()
    if not is_regression(config):
        correct = abs_sum = sq_sum = n = 0
        for bx, by in loader:
            by = by.to(device)
            pred = model(bx.to(device)).argmax(dim=1)
            n += by.size(0)
            correct += (pred == by).sum().item()
            d = (pred - by).float()
            abs_sum += d.abs().sum().item()
            sq_sum += (d * d).sum().item()
        acc = correct / max(1, n)
        return {"primary": acc, "accuracy": acc,
                "mae": abs_sum / max(1, n), "mse": sq_sum / max(1, n)}

    # ── regression ────────────────────────────────────────────────────────────
    n = 0
    n_t = None
    abs_sum = sq_sum = 0.0
    ph_sq = ph_abs = None          # per-head accumulators
    loc_correct = loc_n = 0
    for bx, by in loader:
        by = by.to(device).float()
        pred = model(bx.to(device)).float()
        if by.dim() == 1:
            by = by.unsqueeze(1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        n_t = by.shape[1]
        d = pred - by
        n += by.shape[0]
        abs_sum += d.abs().sum().item()
        sq_sum += (d * d).sum().item()
        head_sq = (d * d).sum(dim=0)
        head_abs = d.abs().sum(dim=0)
        ph_sq = head_sq if ph_sq is None else ph_sq + head_sq
        ph_abs = head_abs if ph_abs is None else ph_abs + head_abs
        if n_t > 1:
            true_max, true_arg = by.max(dim=1)
            mask = true_max > _LOC_DAMAGE_THRESHOLD_PCT
            loc_correct += ((pred.argmax(dim=1) == true_arg) & mask).sum().item()
            loc_n += int(mask.sum().item())

    denom = max(1, n * (n_t or 1))
    loc_acc = (loc_correct / loc_n) if loc_n > 0 else float("nan")
    return {
        "primary": loc_acc,
        "localisation_acc": loc_acc,
        "mae": abs_sum / denom,
        "mse": sq_sum / denom,
        "per_head_mse": (ph_sq / max(1, n)).tolist(),
        "per_head_mae": (ph_abs / max(1, n)).tolist(),
    }
