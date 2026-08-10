"""Leak-free statistical inference utilities for the TTBI ablation.

The independent experimental unit is a generated *damage state*.  Passages
within a state share labels and persistent nuisance realisations, so neither
cross-validation nor uncertainty intervals may treat passages as independent.

This module deliberately contains no project-global settings.  Every setting is
passed by the caller and is therefore suitable for inclusion in
``core.protocol``'s hashed descriptor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class GroupedFold:
    """One repeated grouped-CV fold, expressed in sample indices."""

    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_states: np.ndarray
    val_states: np.ndarray


def finalist_cv_strata(strata_by_state: Sequence[str]) -> list[str]:
    """Return the prospectively source-locked coarsening used by finalist CV.

    The canonical R11 outer split uses
    ``joint|latentcrack{0,1}|scoursev{0,1,2}`` strata.  Repeated CV preserves
    family/anchor strata and latent joint crack status while coarsening only the
    joint scour-severity suffix.  Neither active crack nor bearing status may
    enter this key because those mechanisms change across paired campaign rungs.
    """

    result: list[str] = []
    for raw in strata_by_state:
        key = str(raw)
        parts = key.split("|")
        if parts[0] == "joint":
            crack = [
                part for part in parts[1:]
                if part.startswith("latentcrack")
            ]
            if (
                len(crack) != 1
                or crack[0] not in ("latentcrack0", "latentcrack1")
            ):
                raise ValueError(
                    f"malformed joint stratum {key!r}; expected a "
                    "latentcrack0/1 tag"
                )
            key = f"joint|{crack[0]}"
        result.append(key)
    return result


def _stable_seed(seed: int, *parts: object) -> int:
    """Platform-independent RNG seed (unlike Python's salted ``hash``)."""

    msg = "|".join([str(int(seed)), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(msg.encode("utf-8")).digest()[:8], "big")


def repeated_stratified_group_folds(
    groups: np.ndarray,
    development_idx: np.ndarray,
    strata_by_state: Sequence[str],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[GroupedFold]:
    """Repeated stratified grouped folds over the development pool only.

    ``groups`` contains the state id for every passage. ``development_idx`` must
    contain *whole* states; any omitted state is treated as held out (the
    canonical outer test in the campaign).  Within each repeat, states in every
    stratum are deterministically permuted and distributed round-robin over
    folds.  Thus every state is validation exactly once per repeat and no state
    can cross train/validation.

    A stratum must contain at least ``n_splits`` development states.  Silently
    accepting fewer would make some folds lack that family/anchor stratum and
    defeat the stated stratification guarantee.
    """

    groups = np.asarray(groups)
    development_idx = np.asarray(development_idx, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("groups must be a one-dimensional state-id vector")
    if development_idx.ndim != 1 or development_idx.size == 0:
        raise ValueError("development_idx must be a non-empty 1-D index vector")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >=2 and n_repeats must be >=1")
    if development_idx.min() < 0 or development_idx.max() >= len(groups):
        raise IndexError("development_idx contains an out-of-range sample index")
    if len(np.unique(development_idx)) != len(development_idx):
        raise ValueError("development_idx contains duplicate sample indices")

    dev_mask = np.zeros(len(groups), dtype=bool)
    dev_mask[development_idx] = True
    all_states = np.unique(groups)
    if all_states.size == 0 or all_states.min() < 0:
        raise ValueError("groups must contain non-negative state ids")
    if all_states.max() >= len(strata_by_state):
        raise ValueError("strata_by_state does not cover every state id")

    # The outer-test firewall relies on state-level exclusivity.  Reject a
    # development index vector that includes only some passages of a state.
    development_states: list[int] = []
    for state in all_states:
        state_mask = groups == state
        n_in = int(np.count_nonzero(dev_mask & state_mask))
        if n_in not in (0, int(np.count_nonzero(state_mask))):
            raise ValueError(
                f"development_idx contains only part of state {int(state)}"
            )
        if n_in:
            development_states.append(int(state))
    dev_states = np.asarray(development_states, dtype=np.int64)
    if dev_states.size < n_splits:
        raise ValueError(
            f"only {len(dev_states)} development states for {n_splits} folds"
        )

    strata: dict[str, list[int]] = {}
    for state in dev_states:
        strata.setdefault(str(strata_by_state[int(state)]), []).append(int(state))
    too_small = {key: len(states) for key, states in strata.items()
                 if len(states) < n_splits}
    if too_small:
        raise ValueError(
            "each development stratum needs at least n_splits states; "
            f"too small: {too_small}"
        )

    sample_idx = np.arange(len(groups), dtype=np.int64)
    result: list[GroupedFold] = []
    for repeat in range(n_repeats):
        fold_states: list[list[int]] = [[] for _ in range(n_splits)]
        fold_sizes = np.zeros(n_splits, dtype=np.int64)
        for key in sorted(strata):
            members = np.asarray(strata[key], dtype=np.int64)
            rng = np.random.default_rng(_stable_seed(seed, repeat, key))
            members = members[rng.permutation(len(members))]

            # Choose the cyclic offset which best balances cumulative fold
            # sizes.  Stable-seed rotation breaks deterministic ties without
            # depending on dict/set iteration order.
            tie_rotation = _stable_seed(seed, "offset", repeat, key) % n_splits
            candidates = []
            for offset in range(n_splits):
                add = np.bincount(
                    (np.arange(len(members)) + offset) % n_splits,
                    minlength=n_splits,
                )
                projected = fold_sizes + add
                candidates.append((
                    int(projected.max() - projected.min()),
                    int(np.dot(projected, projected)),
                    int((offset - tie_rotation) % n_splits),
                    offset,
                ))
            offset = min(candidates)[-1]
            for pos, state in enumerate(members):
                fold = int((pos + offset) % n_splits)
                fold_states[fold].append(int(state))
                fold_sizes[fold] += 1

        seen: list[int] = []
        for fold, val_state_list in enumerate(fold_states):
            val_states = np.asarray(sorted(val_state_list), dtype=np.int64)
            train_states = np.setdiff1d(dev_states, val_states, assume_unique=True)
            val_mask = dev_mask & np.isin(groups, val_states)
            train_mask = dev_mask & np.isin(groups, train_states)
            train_idx = sample_idx[train_mask]
            val_idx = sample_idx[val_mask]
            if np.intersect1d(train_states, val_states).size:
                raise RuntimeError("group leakage in repeated CV construction")
            if np.intersect1d(train_idx, val_idx).size:
                raise RuntimeError("sample leakage in repeated CV construction")
            seen.extend(val_states.tolist())
            result.append(GroupedFold(
                repeat=repeat,
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                train_states=train_states,
                val_states=val_states,
            ))
        if sorted(seen) != sorted(dev_states.tolist()):
            raise RuntimeError(
                f"repeat {repeat} does not validate every development state once"
            )
    return result


def channel_standardization_stats(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/scale fitted only on a fold's training samples.

    ``X`` may already have been affinely standardized by the canonical cache.
    Standardizing it again using fold-train moments is exactly equivalent to
    inverse-transforming the cache and fitting a new StandardScaler on the same
    fold (up to floating-point round-off), while avoiding another full feature
    cache and never consulting validation or outer-test samples.
    """

    X = np.asarray(X)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if X.ndim < 3:
        raise ValueError("expected X shaped (samples, channels, spatial...)")
    if train_idx.ndim != 1 or train_idx.size == 0:
        raise ValueError("train_idx must be a non-empty 1-D vector")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if train_idx.min() < 0 or train_idx.max() >= len(X):
        raise IndexError("train_idx contains an out-of-range sample index")
    if len(np.unique(train_idx)) != len(train_idx):
        raise ValueError("train_idx contains duplicate sample indices")

    n_channels = X.shape[1]
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)
    count = 0
    reduce_axes = tuple(i for i in range(X.ndim) if i != 1)
    for start in range(0, len(train_idx), chunk_size):
        block = np.asarray(X[train_idx[start:start + chunk_size]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("fold-training features contain non-finite values")
        total += block.sum(axis=reduce_axes)
        total_sq += np.square(block).sum(axis=reduce_axes)
        count += int(block.size // n_channels)
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    scale = np.sqrt(var)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def frozen_checkpoint_epoch_count(
    intermediate_values: Mapping[int, float],
    *,
    max_epochs: int,
) -> int:
    """Freeze the original best-checkpoint epoch for a leakage-free CV refit.

    The Optuna champion was checkpointed whenever its canonical inner-validation
    objective strictly improved.  Reusing the epoch of its minimum recorded
    objective therefore reproduces that selected training duration without
    consulting a repeated-CV validation fold.  The returned value is a count
    (one-based), whereas Optuna's intermediate-value keys are zero-based epochs.
    """

    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if not intermediate_values:
        raise ValueError("best trial has no intermediate validation values")
    clean: list[tuple[int, float]] = []
    for epoch_raw, value_raw in intermediate_values.items():
        epoch = int(epoch_raw)
        value = float(value_raw)
        if epoch != epoch_raw or not (0 <= epoch < max_epochs):
            raise ValueError(
                f"invalid intermediate-value epoch {epoch_raw!r}"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite intermediate value at epoch {epoch}"
            )
        clean.append((epoch, value))
    # Earliest epoch wins an exact tie, matching the strict '<' checkpoint rule.
    best_epoch, _ = min(clean, key=lambda item: (item[1], item[0]))
    return best_epoch + 1


class FoldStandardizedDataset(torch.utils.data.Dataset):
    """Memmap-backed dataset with a fold-specific affine transform."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        *,
        label_dtype: torch.dtype = torch.float32,
    ):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        n_spatial = X.ndim - 2
        self.mean = np.asarray(mean, dtype=np.float32).reshape(
            (X.shape[1],) + (1,) * n_spatial
        )
        self.scale = np.asarray(scale, dtype=np.float32).reshape(
            (X.shape[1],) + (1,) * n_spatial
        )
        self.label_dtype = label_dtype
        if self.mean.size != X.shape[1] or self.scale.size != X.shape[1]:
            raise ValueError("mean/scale must contain one value per channel")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = (np.asarray(self.X[idx], dtype=np.float32) - self.mean) / self.scale
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(np.asarray(self.y[idx]).copy()).to(self.label_dtype),
        )


def per_state_regression_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    states: np.ndarray,
    *,
    n_scour_heads: int,
) -> dict[str, np.ndarray]:
    """State-level regression metrics, one row per sorted state id."""

    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64)
    states = np.asarray(states)
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.ndim == 1:
        true = true[:, None]
    if pred.shape != true.shape or len(states) != len(pred):
        raise ValueError("predictions, truth and states have incompatible shapes")
    if not (1 <= n_scour_heads <= pred.shape[1]):
        raise ValueError("n_scour_heads is outside the prediction head range")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("predictions/truth contain non-finite values")

    uniq = np.unique(states)
    scour_se = np.square(pred[:, :n_scour_heads] - true[:, :n_scour_heads]).mean(axis=1)
    all_se = np.square(pred - true).mean(axis=1)
    out = {
        "state": uniq.astype(np.int64),
        "scour_mse": np.empty(len(uniq), dtype=np.float64),
        "all_head_mse": np.empty(len(uniq), dtype=np.float64),
        "predicted_max_scour_pct": np.empty(len(uniq), dtype=np.float64),
    }
    for i, state in enumerate(uniq):
        mask = states == state
        out["scour_mse"][i] = float(scour_se[mask].mean())
        out["all_head_mse"][i] = float(all_se[mask].mean())
        out["predicted_max_scour_pct"][i] = float(
            np.maximum(pred[mask, :n_scour_heads], 0.0).max(axis=1).mean()
        )
    return out


def assemble_state_repeat_seed_tensor(
    records: Iterable[Mapping[str, object]],
    *,
    seeds: Sequence[int],
    repeats: Sequence[int],
    value_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align long-form records into ``state x repeat x seed``.

    Every Cartesian cell must occur exactly once.  This hard guard prevents a
    missing seed, duplicated fold result or state-order mismatch from silently
    entering a hierarchical interval or paired contrast.
    """

    records = list(records)
    if not records:
        raise ValueError("cannot assemble an empty record set")
    seeds = [int(s) for s in seeds]
    repeats = [int(r) for r in repeats]
    if len(set(seeds)) != len(seeds) or len(set(repeats)) != len(repeats):
        raise ValueError("seeds and repeats must be unique ordered sequences")
    states = np.asarray(
        sorted({int(r["state"]) for r in records}), dtype=np.int64
    )
    if len(states) < 2:
        raise ValueError("need at least two independent states")
    state_pos = {state: i for i, state in enumerate(states)}
    repeat_pos = {repeat: i for i, repeat in enumerate(repeats)}
    seed_pos = {seed: i for i, seed in enumerate(seeds)}
    tensor = np.full(
        (len(states), len(repeats), len(seeds)), np.nan, dtype=np.float64
    )
    for record in records:
        repeat = int(record["repeat"])
        seed = int(record["seed"])
        if repeat not in repeat_pos or seed not in seed_pos:
            raise ValueError(
                f"record has undeclared repeat/seed ({repeat}, {seed})"
            )
        key = (
            state_pos[int(record["state"])], repeat_pos[repeat], seed_pos[seed]
        )
        if np.isfinite(tensor[key]):
            raise ValueError(
                f"duplicate state/repeat/seed cell "
                f"({int(record['state'])}, {repeat}, {seed})"
            )
        tensor[key] = float(record[value_key])
    if not np.isfinite(tensor).all():
        raise ValueError(
            f"state x repeat x seed tensor has "
            f"{int(np.count_nonzero(~np.isfinite(tensor)))} missing/non-finite cells"
        )
    return states, tensor


def hierarchical_state_seed_bootstrap(
    errors: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, float | int]:
    """State-first CI for a cross-seed statistic.

    ``errors`` has shape ``(states, repeats, seeds)``; use a singleton repeat
    axis for the immutable outer test.  In each bootstrap replicate:

    1. states are sampled with replacement;
    2. mean error is computed for every repeat and seed;
    3. the median across seeds is computed *inside* each repeat;
    4. repeated-CV estimates are averaged across repeats.

    This is not the invalid "median of per-seed CI bounds" construction.
    """

    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim == 2:
        errors = errors[:, None, :]
    if errors.ndim != 3:
        raise ValueError("errors must have shape (states, [repeats,] seeds)")
    if not np.isfinite(errors).all():
        raise ValueError("errors contain non-finite values")
    n_states, n_repeats, n_seeds = errors.shape
    if n_states < 2 or n_seeds < 1 or n_repeats < 1:
        raise ValueError("need >=2 states and >=1 repeat/seed")
    if n_boot < 1 or not (0.0 < ci < 1.0):
        raise ValueError("invalid bootstrap settings")

    def statistic(arr: np.ndarray) -> float:
        # arr: states x repeats x seeds
        per_repeat_seed = arr.mean(axis=0)
        return float(np.median(per_repeat_seed, axis=1).mean())

    estimate = statistic(errors)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    # A loop avoids allocating n_boot x n_states x repeats x seeds for the
    # production 2,000-replicate reports.
    for b in range(n_boot):
        idx = rng.integers(0, n_states, size=n_states)
        draws[b] = statistic(errors[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "estimate": estimate,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_states": int(n_states),
        "n_repeats": int(n_repeats),
        "n_seeds": int(n_seeds),
        "n_boot": int(n_boot),
    }


def paired_state_contrast(
    winner_errors: np.ndarray,
    comparator_errors: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, float | int]:
    """Paired state-first contrast, winner minus comparator.

    Negative values favour the selected winner.  Inputs must already be aligned
    on the exact same states, repeats and seeds.
    """

    winner = np.asarray(winner_errors, dtype=np.float64)
    comparator = np.asarray(comparator_errors, dtype=np.float64)
    if winner.shape != comparator.shape:
        raise ValueError("paired contrast inputs must have identical shapes")
    # The paper-facing model statistic is the median of seed-wise state means
    # (then the mean over CV repeats).  Its contrast must therefore be
    #
    #     statistic(winner) - statistic(comparator),
    #
    # NOT median(seed-wise winner-comparator differences): those estimands can
    # differ sharply when the seed rankings differ between algorithms.
    if winner.ndim == 2:
        winner = winner[:, None, :]
        comparator = comparator[:, None, :]
    if winner.ndim != 3:
        raise ValueError(
            "paired contrast inputs must have shape (states, [repeats,] seeds)"
        )
    if (winner.shape[0] < 2 or winner.shape[1] < 1 or winner.shape[2] < 1
            or not np.isfinite(winner).all()
            or not np.isfinite(comparator).all()):
        raise ValueError("paired contrast inputs are incomplete/non-finite")
    if n_boot < 1 or not (0.0 < ci < 1.0):
        raise ValueError("invalid bootstrap settings")

    def statistic(arr: np.ndarray) -> float:
        per_repeat_seed = arr.mean(axis=0)
        return float(np.median(per_repeat_seed, axis=1).mean())

    estimate = statistic(winner) - statistic(comparator)
    rng = np.random.default_rng(seed)
    n_states, n_repeats, n_seeds = winner.shape
    draws = np.empty(n_boot, dtype=np.float64)
    lt_zero = 0
    for b in range(n_boot):
        idx = rng.integers(0, n_states, size=n_states)
        draws[b] = statistic(winner[idx]) - statistic(comparator[idx])
        lt_zero += int(draws[b] < 0.0)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    # This is a descriptive bootstrap fraction, not a multiplicity-corrected
    # hypothesis-test p-value.
    return {
        "estimate": float(estimate),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_states": int(n_states),
        "n_repeats": int(n_repeats),
        "n_seeds": int(n_seeds),
        "n_boot": int(n_boot),
        "descriptive_bootstrap_fraction_winner_better":
            float(lt_zero / n_boot),
    }


def mcse_family_size_recommendations(
    values_by_state: Mapping[int, float],
    family_by_state: Mapping[int, str],
    *,
    target_mcse: float,
    min_evaluation_states: int,
    evaluation_fraction: float,
    current_total_by_family: Mapping[str, int] | None = None,
) -> list[dict[str, float | int | str]]:
    """Pilot-SD recommendation for independent states in each family.

    For a family mean, ``MCSE = sample_SD / sqrt(n)``.  The returned total-state
    recommendation ensures the designated evaluation fraction contains at least
    both ``ceil((SD/target_mcse)^2)`` states and the source-locked design floor.
    It is a planning diagnostic, not a post-hoc change to the current campaign.
    """

    if target_mcse <= 0 or min_evaluation_states < 2:
        raise ValueError("target_mcse must be >0 and min_evaluation_states >=2")
    if not (0.0 < evaluation_fraction < 1.0):
        raise ValueError("evaluation_fraction must lie strictly between 0 and 1")
    grouped: dict[str, list[float]] = {}
    for state, value in values_by_state.items():
        if state not in family_by_state:
            raise ValueError(f"no family supplied for state {state}")
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite planning value for state {state}")
        grouped.setdefault(str(family_by_state[state]), []).append(float(value))

    rows: list[dict[str, float | int | str]] = []
    for family in sorted(grouped):
        vals = np.asarray(grouped[family], dtype=np.float64)
        if len(vals) >= 2:
            sd = float(np.std(vals, ddof=1))
            n_mcse = max(2, int(math.ceil((sd / target_mcse) ** 2)))
            status = "empirical pilot SD"
        else:
            sd = float("nan")
            n_mcse = min_evaluation_states
            status = "insufficient pilot states; design floor used"
        n_eval = max(min_evaluation_states, n_mcse)
        n_total_mcse = int(math.ceil(n_eval / evaluation_fraction))
        current_total = int((current_total_by_family or {}).get(family, 0))
        n_total = max(current_total, n_total_mcse)
        rows.append({
            "family": family.split("|", 1)[0],
            "analysis_stratum": family,
            "pilot_independent_states": int(len(vals)),
            "pilot_between_state_sd_pct": sd,
            "target_mcse_pct": float(target_mcse),
            "mcse_required_evaluation_states": int(n_mcse),
            "design_floor_evaluation_states": int(min_evaluation_states),
            "recommended_evaluation_states": int(n_eval),
            "current_total_family_states": current_total,
            "recommended_total_family_states": n_total,
            "basis": status,
        })
    return rows
