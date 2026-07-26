"""
core/task.py
============
The single place that owns the difference between the two learning tasks the
ablation supports:

    'classification'  (default, single-scour) - predict a discretised damage
        CLASS (0..60/disc); loss = cross-entropy; metrics on the class index.
        This is the original, validated single-foundation ablation.

    'regression'      (multi-damage, Stage 0+) - predict a CONTINUOUS per-pier
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


def bearing_targets_of(config: dict) -> list:
    """Bearing regression heads for Stage 1 (e.g. ['left', 'right']); [] if none.

    When set, the model gains one extra continuous head per bearing target, laid
    out AFTER the scour heads: label = [scour_1..scour_S, bearing_1..bearing_B].
    """
    return list(config.get("bearing_targets") or [])


def n_scour_outputs(config: dict) -> int:
    """Number of scour (target-support) heads."""
    return len(config.get("target_supports") or [])


def n_bearing_outputs(config: dict) -> int:
    """Number of bearing heads (0 for Stage 0)."""
    return len(bearing_targets_of(config))


def n_outputs(config: dict) -> int:
    """Size of the model's final layer.

    classification -> number of damage classes (int(60/disc)+1).
    regression     -> scour heads (target supports) + optional bearing heads.
    """
    if is_regression(config):
        targets = config.get("target_supports")
        if not targets:
            raise ValueError("regression task needs config['target_supports'] "
                             "(e.g. [2, 3] for the two internal piers).")
        return len(targets) + n_bearing_outputs(config)
    disc = config.get("discretization", 1)
    return int(60 / disc) + 1


class WeightedHeadMSE(nn.Module):
    """Range-normalized multi-task MSE (audit r3, 2026-07-22).

    Plain MSELoss over heads lets the wider-range bearing labels (0-95 %)
    dominate the gradient over the scour labels (0-60 %) — the network could
    buy bearing accuracy with scour accuracy even though scour is the primary
    scientific target. Weights are w_h = 1/range_h^2, normalized to mean 1 so
    the loss magnitude (and the tuned LR range) stays comparable to MSELoss.
    """

    def __init__(self, ranges: list):
        super().__init__()
        r = torch.as_tensor(ranges, dtype=torch.float32)
        if r.ndim != 1 or r.numel() == 0:
            raise ValueError("WeightedHeadMSE needs one positive range per head.")
        if not bool(torch.isfinite(r).all()) or bool((r <= 0).any()):
            raise ValueError(
                f"WeightedHeadMSE ranges must be finite and positive; got {ranges!r}.")
        w = 1.0 / (r * r)
        self.register_buffer("w", w / w.mean())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if pred.dim() != 2 or target.dim() != 2 or pred.shape != target.shape:
            raise ValueError(
                "WeightedHeadMSE expects matching (batch, heads) tensors; "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}.")
        if pred.shape[1] != self.w.numel():
            raise ValueError(
                f"WeightedHeadMSE was configured for {self.w.numel()} head(s), "
                f"but received {pred.shape[1]}.")
        return (self.w.to(pred.device) * (pred - target) ** 2).mean()


def make_criterion(config: dict, policy: dict) -> nn.Module:
    """Build the task loss from the executable TRAIN_PROTOCOL specification."""
    required_branches = {
        "classification",
        "regression_without_bearing_heads",
        "regression_with_bearing_heads",
    }
    if not isinstance(policy, dict) or set(policy) != required_branches:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            "loss policy must define exactly "
            f"{sorted(required_branches)!r}; got {got!r}.")

    if not is_regression(config):
        branch = "classification"
    elif n_bearing_outputs(config) > 0:
        branch = "regression_with_bearing_heads"
    else:
        branch = "regression_without_bearing_heads"
    spec = policy[branch]
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(f"loss policy branch {branch!r} needs a mapping with 'kind'.")

    kind = spec["kind"]
    if kind == "cross_entropy":
        if set(spec) != {"kind"}:
            raise ValueError("cross_entropy loss accepts only the 'kind' field.")
        return nn.CrossEntropyLoss()
    if kind == "mse":
        if set(spec) != {"kind"}:
            raise ValueError("mse loss accepts only the 'kind' field.")
        return nn.MSELoss()
    if kind == "inverse_range_squared_mse":
        if set(spec) != {"kind", "head_ranges_pct"}:
            raise ValueError(
                "inverse_range_squared_mse requires exactly kind and "
                "head_ranges_pct.")
        head_ranges = spec["head_ranges_pct"]
        if not isinstance(head_ranges, dict) or set(head_ranges) != {
                "scour", "bearing"}:
            raise ValueError(
                "head_ranges_pct must define exactly 'scour' and 'bearing'.")
        ranges = (
            [head_ranges["scour"]] * n_scour_outputs(config)
            + [head_ranges["bearing"]] * n_bearing_outputs(config)
        )
        return WeightedHeadMSE(ranges)
    raise ValueError(f"unsupported loss kind {kind!r} in branch {branch!r}.")


def label_dtype(config) -> torch.dtype:
    """Float targets for regression, integer class labels for classification."""
    return torch.float32 if is_regression(config) else torch.long


def objective_value(metrics: dict, config: dict, policy: dict) -> float:
    """Return the protocol-selected scalar that Optuna minimises.

    ``policy`` is ``TRAIN_PROTOCOL["objective"]``. It is deliberately passed
    by the trainer rather than imported here (which would create a circular
    import). The policy is executable data: regression tasks with bearing
    heads require their registered metric, while all other tasks require the
    default metric. Missing metrics fail closed instead of silently falling
    back to the all-head aggregate on a malformed bearing evaluation.
    """
    required_branches = {"regression_with_bearing_heads", "default"}
    if not isinstance(policy, dict) or set(policy) != required_branches:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            "objective policy must define exactly "
            f"{sorted(required_branches)!r}; got {got!r}.")
    if not all(isinstance(metric, str) and metric for metric in policy.values()):
        raise ValueError("objective policy metric names must be non-empty strings.")

    branch = (
        "regression_with_bearing_heads"
        if is_regression(config) and n_bearing_outputs(config) > 0
        else "default"
    )
    metric = policy[branch]
    if metric not in metrics:
        raise KeyError(
            f"objective policy requires metric {metric!r} for branch {branch!r}, "
            f"but evaluation returned {sorted(metrics)!r}.")
    return metrics[metric]


@torch.no_grad()
def evaluate(model: nn.Module, loader, config: dict, device) -> dict:
    """Task-appropriate validation metrics in one pass.

    Returns a dict that always carries 'mse', 'mae', and 'primary' (so callers
    can stay task-agnostic), plus task-specific extras:
        classification : {primary=accuracy, accuracy, mae, mse}        (class-index units)
        regression     : {primary=localisation_acc, localisation_acc, mae, mse,
                          per_head_mse, per_head_mae}                   (% units)
                          + {scour_mse, bearing_mse} when bearing heads exist.
    Localisation is scored over the SCOUR heads only; bearing heads (laid out
    after the scour heads) never enter the which-pier argmax.
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
    n_scour = n_scour_outputs(config)      # scour heads come FIRST in the label
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
        # localisation over the SCOUR heads only (bearing heads excluded)
        ns = n_scour if n_scour > 0 else n_t
        if ns > 1:
            by_s, pred_s = by[:, :ns], pred[:, :ns]
            true_max, true_arg = by_s.max(dim=1)
            mask = true_max > _LOC_DAMAGE_THRESHOLD_PCT
            loc_correct += ((pred_s.argmax(dim=1) == true_arg) & mask).sum().item()
            loc_n += int(mask.sum().item())

    denom = max(1, n * (n_t or 1))
    loc_acc = (loc_correct / loc_n) if loc_n > 0 else float("nan")
    per_head_mse = (ph_sq / max(1, n)).tolist()
    out = {
        "primary": loc_acc,
        "localisation_acc": loc_acc,
        "mae": abs_sum / denom,
        "mse": sq_sum / denom,
        "per_head_mse": per_head_mse,
        "per_head_mae": (ph_abs / max(1, n)).tolist(),
    }
    # Scour vs bearing group MSE (Stage 1): the headline of the disentanglement
    # story - is scour still clean once bearing shares the network?
    ns = n_scour if 0 < n_scour < len(per_head_mse) else len(per_head_mse)
    if ns < len(per_head_mse):
        out["scour_mse"]   = sum(per_head_mse[:ns]) / ns
        out["bearing_mse"] = sum(per_head_mse[ns:]) / (len(per_head_mse) - ns)
    return out
