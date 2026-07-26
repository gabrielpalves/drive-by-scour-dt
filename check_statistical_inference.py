"""Adversarial checks for state-grouped CV and statistical inference.

Run:
    python check_statistical_inference.py
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from core.statistical_inference import (
    FoldStandardizedDataset,
    assemble_state_repeat_seed_tensor,
    channel_standardization_stats,
    finalist_cv_strata,
    frozen_checkpoint_epoch_count,
    hierarchical_state_seed_bootstrap,
    mcse_family_size_recommendations,
    paired_state_contrast,
    per_state_regression_metrics,
    repeated_stratified_group_folds,
)
from core.utils import set_global_seed


fails = 0


def check(name, condition):
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def raises(name, fn, exc=Exception):
    try:
        fn()
    except exc:
        check(name, True)
    else:
        check(name, False)


print("\n--- repeated stratified grouped CV ---")
n_states, n_pass = 18, 4
groups = np.repeat(np.arange(n_states), n_pass)
strata = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
# One state per stratum is the immutable outer test.
outer_states = np.array([5, 11, 17])
dev_states = np.setdiff1d(np.arange(n_states), outer_states)
dev_idx = np.flatnonzero(np.isin(groups, dev_states))
folds = repeated_stratified_group_folds(
    groups, dev_idx, strata, n_splits=5, n_repeats=3, seed=271828
)
folds_again = repeated_stratified_group_folds(
    groups, dev_idx, strata, n_splits=5, n_repeats=3, seed=271828
)
check("expected repeat x fold count", len(folds) == 15)
check("deterministic fold state assignments",
      all(np.array_equal(a.val_states, b.val_states)
          for a, b in zip(folds, folds_again)))
check("outer-test states never enter CV",
      all(not np.intersect1d(f.val_states, outer_states).size
          and not np.intersect1d(f.train_states, outer_states).size
          for f in folds))
check("no group crosses train/validation",
      all(not np.intersect1d(f.train_states, f.val_states).size for f in folds))
check("each state validates exactly once per repeat",
      all(sorted(np.concatenate([f.val_states for f in folds
                                 if f.repeat == repeat]).tolist())
          == sorted(dev_states.tolist()) for repeat in range(3)))
check("every fold has every stratum",
      all({strata[s] for s in f.val_states} == {"A", "B", "C"} for f in folds))
folds_other = repeated_stratified_group_folds(
    groups, dev_idx, strata, n_splits=5, n_repeats=3, seed=271829
)
check("changing split seed changes assignments",
      any(not np.array_equal(a.val_states, b.val_states)
          for a, b in zip(folds, folds_other)))
partial = dev_idx[dev_idx != np.flatnonzero(groups == 0)[0]]
raises("partial-state development membership rejected",
       lambda: repeated_stratified_group_folds(
           groups, partial, strata, n_splits=5, n_repeats=1, seed=1
       ), ValueError)
raises("under-populated stratum rejected",
       lambda: repeated_stratified_group_folds(
           groups, dev_idx, ["tiny"] + ["other"] * 17,
           n_splits=5, n_repeats=1, seed=1
       ), ValueError)

# Exact failure mode from the R8 multi-head design: equal-width MAX-severity
# bins can contain only 1-4 joint states. CV coarsens severity but retains crack
# activation, producing viable, pre-registered strata without consulting outer
# outcomes.
sparse_joint = (
    ["target_healthy"] * 12
    + ["nuisance_only"] * 6
    + ["joint|crack0|sev0"] * 4
    + ["joint|crack0|sev1"] * 35
    + ["joint|crack0|sev2"] * 161
    + ["joint|crack1|sev1"] * 12
    + ["joint|crack1|sev2"] * 38
)
coarse_joint = finalist_cv_strata(sparse_joint)
check("CV coarsens sparse joint severity but preserves crack status",
      set(coarse_joint) == {
          "target_healthy", "nuisance_only", "joint|crack0", "joint|crack1"
      }
      and coarse_joint.count("joint|crack0") == 200
      and coarse_joint.count("joint|crack1") == 50)
raises("malformed joint CV stratum rejected",
       lambda: finalist_cv_strata(["joint|sev0"]), ValueError)


print("\n--- driver firewall and fixed-fold integration ---")
driver_path = Path(__file__).with_name("comprehensive_ablation_multidamage.py")
driver_source = driver_path.read_text(encoding="utf-8")
driver_tree = ast.parse(driver_source, filename=str(driver_path))
driver_functions = {
    node.name: node for node in driver_tree.body
    if isinstance(node, ast.FunctionDef)
}
summarize_source = ast.get_source_segment(
    driver_source, driver_functions["summarize"]
)
ordered_tokens = [
    '"frozen_selection.json"',
    "_preflight_comparators(comparators)",
    "_run_finalist_repeated_cv(comparators, winner_key)",
    "evaluate_champion(",
    "_outer_test_hierarchical_inference(",
]
ordered_positions = [summarize_source.index(token) for token in ordered_tokens]
check("freeze -> preflight -> CV -> outer prediction -> inference call order",
      ordered_positions == sorted(ordered_positions))
cv_source = ast.get_source_segment(
    driver_source, driver_functions["_run_finalist_repeated_cv"]
)
check("CV implementation contains no outer prediction/evaluation call",
      all(token not in cv_source for token in (
          "_test_loader(", "evaluate_champion(", "_predictions("
      )))
check("CV records outer membership as firewall metadata only",
      "outer_test_observations_accessed" in cv_source
      and "False" in cv_source
      and "outer_test_membership_used_only_for_disjointness_assertion"
      in cv_source)
check("CV scope is finalist-only pair-search stages",
      "FINALIST_CV_STAGES = set(PAIR_SEARCH_STAGES)" in driver_source)
check("statistical policy flows through unified protocol descriptor",
      "statistical_inference = STATISTICAL_INFERENCE_PROTOCOL"
      in driver_source)

# Compile the shipped function itself from the driver AST, then run a real
# PyTorch optimisation/prediction step on synthetic grouped data. This avoids
# importing the driver (which correctly requires a complete real dataset).
fit_node = driver_functions["_fit_predict_finalist_fold"]
fit_module = ast.Module(body=[fit_node], type_ignores=[])
ast.fix_missing_locations(fit_module)


class TinyRegressor(torch.nn.Module):
    def __init__(self, channels: int, length: int):
        super().__init__()
        self.linear = torch.nn.Linear(channels * length, 1)

    def forward(self, x):
        return self.linear(x.flatten(1))


tiny_task = SimpleNamespace(
    label_dtype=lambda _cfg: torch.float32,
    make_criterion=lambda _cfg: torch.nn.MSELoss(),
)
fit_namespace = {
    "np": np,
    "torch": torch,
    "DataLoader": torch.utils.data.DataLoader,
    "FoldStandardizedDataset": FoldStandardizedDataset,
    "channel_standardization_stats": channel_standardization_stats,
    "per_state_regression_metrics": per_state_regression_metrics,
    "set_global_seed": set_global_seed,
    "task": tiny_task,
    "DEVICE": torch.device("cpu"),
    "TRAIN_PROTOCOL": {"batch_size": 4},
    "EPOCHS": 3,
    "TARGET_SUPPORTS": [2],
}
fit_namespace["build_model"] = lambda _cfg, _params, shape, device: (
    TinyRegressor(shape[1], shape[2]).to(device), None
)
exec(compile(fit_module, str(driver_path), "exec"), fit_namespace)
fit_fold = fit_namespace["_fit_predict_finalist_fold"]
toy_rng = np.random.default_rng(91)
toy_x = toy_rng.normal(size=(12, 2, 5)).astype(np.float32)
toy_y = toy_rng.normal(size=(12, 1)).astype(np.float32)
toy_groups = np.repeat(np.arange(6), 2)
toy_fold = SimpleNamespace(
    train_idx=np.arange(8), val_idx=np.arange(8, 12),
    val_states=np.array([4, 5]),
)
toy_metrics = fit_fold(
    {"method": "PAA"}, {"lr": 1e-3, "weight_decay": 0.0},
    toy_x, toy_y, toy_groups, toy_fold, 42, 1,
)
check("shipped fixed-fold trainer runs and returns state metrics",
      np.array_equal(toy_metrics["state"], [4, 5])
      and np.isfinite(toy_metrics["scour_mse"]).all())
raises("fixed-fold trainer rejects epoch count above protocol maximum",
       lambda: fit_fold(
           {"method": "PAA"}, {"lr": 1e-3, "weight_decay": 0.0},
           toy_x, toy_y, toy_groups, toy_fold, 42, 4,
       ), ValueError)


print("\n--- fold-specific scaling ---")
rng = np.random.default_rng(7)
raw = rng.normal(size=(12, 2, 9)).astype(np.float32)
raw[:, 0] = raw[:, 0] * 3.2 + 11.0
raw[:, 1] = raw[:, 1] * 0.4 - 2.0
canonical_mean = raw[:5].transpose(0, 2, 1).reshape(-1, 2).mean(axis=0)
canonical_scale = raw[:5].transpose(0, 2, 1).reshape(-1, 2).std(axis=0)
cached = ((raw - canonical_mean[None, :, None])
          / canonical_scale[None, :, None]).astype(np.float32)
train_idx = np.array([2, 3, 5, 7, 9])
mean, scale = channel_standardization_stats(cached, train_idx, chunk_size=2)
direct_train = raw[train_idx].transpose(0, 2, 1).reshape(-1, 2)
direct_mean, direct_scale = direct_train.mean(axis=0), direct_train.std(axis=0)
via_cached = (cached - mean[None, :, None]) / scale[None, :, None]
direct = (raw - direct_mean[None, :, None]) / direct_scale[None, :, None]
check("restandardising cached affine features equals direct fold refit",
      np.allclose(via_cached, direct, atol=2e-6))
ds = FoldStandardizedDataset(
    cached, np.arange(12)[:, None].astype(np.float32), np.arange(12),
    mean, scale
)
check("dataset applies the same fold transform",
      np.allclose(ds[6][0].numpy(), via_cached[6], atol=1e-6))
raises("duplicate fold-scaler indices rejected",
       lambda: channel_standardization_stats(cached, np.array([0, 0])),
       ValueError)
raises("negative fold-scaler index rejected",
       lambda: channel_standardization_stats(cached, np.array([-1, 0])),
       IndexError)
check("best checkpoint epoch is frozen without a CV-fold decision",
      frozen_checkpoint_epoch_count(
          {0: 9.0, 1: 5.0, 2: 5.0, 3: 6.0}, max_epochs=5
      ) == 2)
raises("missing checkpoint history rejected",
       lambda: frozen_checkpoint_epoch_count({}, max_epochs=5), ValueError)


print("\n--- state-first, cross-seed inference ---")
# Two passages per state; first two heads are scour, third is bearing.
truth = np.zeros((8, 3))
pred = np.array([
    [1, 3, 9], [1, 1, 7],
    [2, 2, 1], [4, 0, 1],
    [3, 1, 5], [1, 1, 5],
    [0, 2, 8], [2, 2, 6],
], dtype=float)
state_ids = np.repeat(np.arange(4), 2)
metrics = per_state_regression_metrics(
    pred, truth, state_ids, n_scour_heads=2
)
check("state metrics have one row per independent state",
      np.array_equal(metrics["state"], np.arange(4)))
check("scour metric excludes bearing head",
      np.allclose(metrics["scour_mse"], [3.0, 6.0, 3.0, 3.0]))
check("all-head metric includes bearing head",
      metrics["all_head_mse"][0] > metrics["scour_mse"][0])
negative_metrics = per_state_regression_metrics(
    -np.ones((4, 2)), np.zeros((4, 2)), np.repeat([0, 1], 2),
    n_scour_heads=2,
)
check("false-positive planning amplitude clips negative predictions at zero",
      np.array_equal(
          negative_metrics["predicted_max_scour_pct"], np.zeros(2)
      ))

long_records = [
    {"state": state, "repeat": repeat, "seed": seed,
     "loss": 100 * state + 10 * repeat + seed}
    for state in (4, 2, 9)
    for repeat in (0, 1)
    for seed in (3, 7)
]
aligned_states, aligned = assemble_state_repeat_seed_tensor(
    long_records, seeds=[3, 7], repeats=[0, 1], value_key="loss"
)
check("long records align as sorted state x repeat x ordered seed",
      np.array_equal(aligned_states, [2, 4, 9])
      and aligned.shape == (3, 2, 2)
      and aligned[1, 1, 0] == 413)
raises("missing state x repeat x seed cell rejected",
       lambda: assemble_state_repeat_seed_tensor(
           long_records[:-1], seeds=[3, 7], repeats=[0, 1], value_key="loss"
       ), ValueError)
raises("duplicate state x repeat x seed cell rejected",
       lambda: assemble_state_repeat_seed_tensor(
           long_records + [long_records[0]], seeds=[3, 7], repeats=[0, 1],
           value_key="loss"
       ), ValueError)

# state x seed.  The estimator is median of the three seed-wise state means.
err = np.array([[1, 10, 100],
                [3, 14, 104],
                [5, 18, 108],
                [7, 22, 112]], dtype=float)
boot1 = hierarchical_state_seed_bootstrap(
    err, n_boot=500, seed=42, ci=0.95
)
boot2 = hierarchical_state_seed_bootstrap(
    err, n_boot=500, seed=42, ci=0.95
)
check("point estimate computes cross-seed median inside statistic",
      np.isclose(boot1["estimate"], 16.0))
check("bootstrap is exactly deterministic", boot1 == boot2)
check("state resampling creates a non-degenerate interval",
      boot1["ci_lo"] < boot1["estimate"] < boot1["ci_hi"])

err_repeated = np.stack([err, err + 4.0], axis=1)
boot_repeated = hierarchical_state_seed_bootstrap(
    err_repeated, n_boot=200, seed=9, ci=0.95
)
check("repeated-CV statistic is mean of repeat-wise cross-seed medians",
      np.isclose(boot_repeated["estimate"], 18.0)
      and boot_repeated["n_repeats"] == 2)

comp = err + 5.0
contrast = paired_state_contrast(
    err, comp, n_boot=300, seed=123, ci=0.95
)
check("paired contrast preserves exact common-state difference",
      np.isclose(contrast["estimate"], -5.0)
      and np.isclose(contrast["ci_lo"], -5.0)
      and np.isclose(contrast["ci_hi"], -5.0))
check("negative contrast is recorded as winner better",
      contrast["descriptive_bootstrap_fraction_winner_better"] == 1.0)
# Difference of the two published seed medians is 100 - 1 = 99, whereas the
# (incorrect) median of paired seed differences would be median(0,99,1) = 1.
rank_cross_winner = np.tile([0.0, 100.0, 101.0], (4, 1))
rank_cross_comp = np.tile([0.0, 1.0, 100.0], (4, 1))
rank_cross = paired_state_contrast(
    rank_cross_winner, rank_cross_comp, n_boot=30, seed=4, ci=0.95
)
check("contrast is difference of cross-seed medians, not median difference",
      np.isclose(rank_cross["estimate"], 99.0)
      and np.isclose(rank_cross["ci_lo"], 99.0)
      and np.isclose(rank_cross["ci_hi"], 99.0))
raises("unpaired shapes rejected",
       lambda: paired_state_contrast(err, comp[:-1], n_boot=10, seed=1),
       ValueError)


print("\n--- MCSE family-size planning ---")
recs = mcse_family_size_recommendations(
    {0: 0.0, 1: 2.0, 2: 4.0, 3: 7.0},
    {0: "nuisance_only", 1: "nuisance_only",
     2: "scour_only|target2", 3: "scour_only|target2"},
    target_mcse=0.5,
    min_evaluation_states=10,
    evaluation_fraction=0.2,
    current_total_by_family={
        "nuisance_only": 6, "scour_only|target2": 60,
    },
)
by_family = {r["analysis_stratum"]: r for r in recs}
check("design floor gives at least ten independent evaluation states",
      all(r["recommended_evaluation_states"] >= 10 for r in recs))
check("20% evaluation allocation maps ten to fifty total states",
      all(r["recommended_total_family_states"] >= 50 for r in recs))
check("higher between-state SD cannot require fewer states",
      by_family["scour_only|target2"]["mcse_required_evaluation_states"]
      >= by_family["nuisance_only"]["mcse_required_evaluation_states"])
check("recommendation never shrinks an already larger stratum",
      by_family["scour_only|target2"]["recommended_total_family_states"] >= 60)


if fails:
    raise SystemExit(f"\nSTATISTICAL INFERENCE: {fails} FAILURE(S)")
print("\nSTATISTICAL INFERENCE: ALL PASS")
