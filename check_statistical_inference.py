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

from core.models import (
    MultiRatePooling1D,
    Space2Vec,
    SpaceAwareModularNetwork,
    Time2VecPositionEncoding,
)
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
from training import trainer as campaign_trainer
from training.robustness import (
    run_development_adjudication,
    run_post_freeze_stability,
)


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

# Three six-state strata expose the cumulative balancing rule: a fixed cyclic
# offset would put all three "extra" validation states in the same fold and
# produce sizes [6, 3, 3, 3, 3].  The implemented offset search must distribute
# those extras while preserving within-stratum round-robin balance.
balance_groups = np.repeat(np.arange(18), 2)
balance_strata = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
balance_folds = repeated_stratified_group_folds(
    balance_groups, np.arange(len(balance_groups)), balance_strata,
    n_splits=5, n_repeats=3, seed=314159,
)
check("cumulative fold-size balancing is non-trivial and within one state",
      all(
          max(len(f.val_states) for f in balance_folds if f.repeat == repeat)
          - min(len(f.val_states) for f in balance_folds
                if f.repeat == repeat) <= 1
          for repeat in range(3)
      ))
check("round-robin assignment balances every stratum within one state",
      all(
          max(
              sum(balance_strata[int(s)] == key for s in f.val_states)
              for f in balance_folds if f.repeat == repeat
          )
          - min(
              sum(balance_strata[int(s)] == key for s in f.val_states)
              for f in balance_folds if f.repeat == repeat
          ) <= 1
          for repeat in range(3) for key in ("A", "B", "C")
      ))

# Exact failure mode from the R8 multi-head design: equal-width MAX-severity
# bins can contain only 1-4 joint states. CV coarsens scour severity but retains
# the latent (cross-rung invariant) crack draw, producing viable registered
# strata without consulting outer outcomes or active mechanism toggles.
sparse_joint = (
    ["target_healthy"] * 50
    + ["nuisance_only"] * 50
    + ["joint|latentcrack0|scoursev0"] * 4
    + ["joint|latentcrack0|scoursev1"] * 35
    + ["joint|latentcrack0|scoursev2"] * 161
    + ["joint|latentcrack1|scoursev1"] * 12
    + ["joint|latentcrack1|scoursev2"] * 38
)
coarse_joint = finalist_cv_strata(sparse_joint)
check("CV coarsens sparse joint severity but preserves latent crack status",
      set(coarse_joint) == {
          "target_healthy", "nuisance_only",
          "joint|latentcrack0", "joint|latentcrack1"
      }
      and coarse_joint.count("joint|latentcrack0") == 200
      and coarse_joint.count("joint|latentcrack1") == 50)
raises("malformed joint CV stratum rejected",
       lambda: finalist_cv_strata(["joint|scoursev0"]), ValueError)


print("\n--- manifest driver and fixed-fold integration ---")
driver_path = Path(__file__).with_name("comprehensive_ablation_multidamage.py")
driver_source = driver_path.read_text(encoding="utf-8")
driver_tree = ast.parse(driver_source, filename=str(driver_path))
driver_functions = {
    node.name: node for node in driver_tree.body
    if isinstance(node, ast.FunctionDef)
}
retired_driver_functions = {
    "summarize",
    "_run_finalist_repeated_cv",
    "_fit_predict_finalist_fold",
    "_records_to_error_tensor",
    "_outer_test_hierarchical_inference",
}
execute_source = ast.get_source_segment(
    driver_source, driver_functions["execute_registered_job"]
)
check(
    "manifest driver delegates exact registered jobs to the phase executor",
    not retired_driver_functions.intersection(driver_functions)
    and "from training.paper1_executor import execute_manifest_job"
    in execute_source
    and "return execute_manifest_job(job, manifest)" in execute_source,
)
inference_descriptor_source = Path(__file__).with_name("core").joinpath(
    "cross_rung_inference.py"
).read_text(encoding="utf-8")
check(
    "current paired-inference descriptor is external and non-population",
    "STATISTICAL_INFERENCE_PROTOCOL" not in driver_source
    and "MATCHED_BLOCK_INFERENCE_POLICY = {" in inference_descriptor_source
    and '"population_confidence_interval": False'
    in inference_descriptor_source
    and '"automatic_superiority_claim": False'
    in inference_descriptor_source,
)


class TinyRegressor(torch.nn.Module):
    def __init__(self, channels: int, length: int):
        super().__init__()
        self.linear = torch.nn.Linear(channels * length, 1)

    def forward(self, x):
        return self.linear(x.flatten(1))


toy_rng = np.random.default_rng(91)
toy_x = toy_rng.normal(size=(12, 2, 5)).astype(np.float32)
toy_y = toy_rng.normal(size=(12, 1)).astype(np.float32)
toy_groups = np.repeat(np.arange(6), 2)
toy_fold = SimpleNamespace(
    train_idx=np.arange(8), val_idx=np.arange(8, 12),
    train_states=np.array([0, 1, 2, 3]),
    val_states=np.array([4, 5]),
)
toy_config = {
    "method": "PAA",
    "task": "regression",
    "target_supports": [2],
    "bearing_targets": None,
}

# Exercise that shared implementation with a real optimisation/prediction step
# while replacing only the expensive architecture factory.
original_build_model = campaign_trainer.build_model
original_device = campaign_trainer.DEVICE
try:
    campaign_trainer.DEVICE = torch.device("cpu")
    campaign_trainer.build_model = (
        lambda _cfg, _params, shape, device:
        (TinyRegressor(shape[1], shape[2]).to(device), None)
    )
    toy_metrics = campaign_trainer.fit_predict_finalist_fold(
        toy_config,
        {"lr": 1e-3, "weight_decay": 0.0},
        toy_x,
        toy_y,
        toy_groups,
        toy_fold,
        42,
        n_epochs=1,
        max_epochs=3,
        n_scour_heads=1,
    )
    toy_raw_metrics = campaign_trainer.fit_predict_fixed_group_fold(
        {**toy_config, "method": "RAW"},
        {"lr": 1e-3, "weight_decay": 0.0},
        toy_x,
        toy_y,
        toy_groups,
        toy_fold,
        42,
        n_epochs=1,
        max_epochs=3,
        n_scour_heads=1,
    )
finally:
    campaign_trainer.build_model = original_build_model
    campaign_trainer.DEVICE = original_device
check("shared fixed-fold trainer runs and returns state metrics",
      np.array_equal(toy_metrics["state"], [4, 5])
      and np.isfinite(toy_metrics["scour_mse"]).all())
check("RAW and PAA share the fixed grouped-fold trainer",
      np.array_equal(toy_raw_metrics["state"], [4, 5])
      and np.isfinite(toy_raw_metrics["scour_mse"]).all())
raises("fixed-fold trainer rejects epoch count above protocol maximum",
       lambda: campaign_trainer.fit_predict_finalist_fold(
           toy_config, {"lr": 1e-3, "weight_decay": 0.0},
           toy_x, toy_y, toy_groups, toy_fold, 42,
           n_epochs=4, max_epochs=3, n_scour_heads=1,
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

# Median and mean do not commute on this cyclic rank-crossing fixture.  The
# registered statistic first averages sampled states for each seed, then takes
# the cross-seed median (200/3); taking a median within each state first would
# incorrectly return 100.
rank_cross_states = np.array([
    [0.0, 100.0, 100.0],
    [100.0, 0.0, 100.0],
    [100.0, 100.0, 0.0],
])
rank_cross_boot = hierarchical_state_seed_bootstrap(
    rank_cross_states, n_boot=50, seed=17, ci=0.95
)
check("seed median is applied after state aggregation inside the statistic",
      np.isclose(rank_cross_boot["estimate"], 200.0 / 3.0))

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

print("\n--- fixed-width model mechanics ---")
pool = MultiRatePooling1D((1, 2, 4))
short_input = torch.arange(2 * 3 * 17, dtype=torch.float32).reshape(2, 3, 17)
long_input = torch.arange(2 * 3 * 41, dtype=torch.float32).reshape(2, 3, 41)
short_pooled = pool(short_input)
long_pooled = pool(long_input)
expected_short = torch.cat([
    torch.nn.functional.adaptive_max_pool1d(short_input, level).reshape(2, -1)
    for level in (1, 2, 4)
], dim=1)
check("adaptive pyramid width is independent of input length",
      short_pooled.shape == long_pooled.shape == (2, 21))
check("multi-rate representation is the declared temporal pyramid",
      torch.equal(short_pooled, expected_short))
raises("duplicate temporal-pyramid bins rejected",
       lambda: MultiRatePooling1D((1, 2, 2)), ValueError)
raises("empty temporal input rejected",
       lambda: pool(torch.empty(2, 3, 0)), ValueError)

position = Time2VecPositionEncoding(seq_len=17, out_features=5)
position_short = position(torch.linspace(0, 1, 17).reshape(1, 1, -1))
position_long = position(torch.linspace(0, 1, 41).reshape(1, 1, -1))
check("Time2Vec-style coordinate encoding accepts live sequence lengths",
      position_short.shape == (1, 5, 17)
      and position_long.shape == (1, 5, 41))
check("historical Space2Vec import is a compatibility alias only",
      Space2Vec is Time2VecPositionEncoding)

fixed_width_params = {
    "n_conv_layers": 1,
    "n_filters_l0": 4,
    "kernel_size_l0": 3,
    "pooling_l0": False,
    "nhits_pool_rates": (1, 2, 4),
    "n_dense_layers": 1,
    "n_dense_units_l0": 6,
    "dropout_l0": 0.0,
}
network_short = SpaceAwareModularNetwork(
    n_segments=17,
    n_classes=3,
    in_channels=2,
    params=fixed_width_params,
    use_space2vec=True,
    use_nhits=True,
)
network_long = SpaceAwareModularNetwork(
    n_segments=41,
    n_classes=3,
    in_channels=2,
    params=fixed_width_params,
    use_space2vec=True,
    use_nhits=True,
)
parameter_count = lambda model: sum(p.numel() for p in model.parameters())
check("RAW/PAA sequence lengths have equal model parameter count",
      parameter_count(network_short) == parameter_count(network_long))
check("dense input width is independent of configured sequence length",
      network_short.dense_layers[0].in_features
      == network_long.dense_layers[0].in_features == 28)
network_short.eval()
with torch.no_grad():
    logits_short = network_short(torch.randn(2, 2, 17))
    logits_long = network_short(torch.randn(2, 2, 41))
check("one fixed network accepts both sequence lengths",
      logits_short.shape == logits_long.shape == (2, 3))


def _model_mechanics_source_contract(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    if not {"Time2VecPositionEncoding", "MultiRatePooling1D"} <= set(classes):
        return False
    pool_source = ast.get_source_segment(
        source, classes["MultiRatePooling1D"]
    )
    network_source = ast.get_source_segment(
        source, classes["SpaceAwareModularNetwork"]
    )
    return (
        "F.adaptive_max_pool1d" in pool_source
        and "F.max_pool1d(" not in pool_source
        and "self.output_bins = sum(levels)" in pool_source
        and "current_features * self.multi_rate_pool.output_bins"
        in network_source
        and "current_seq_len" not in network_source
        and "steps=x.size(-1)" in network_source
        and "Space2Vec = Time2VecPositionEncoding" in source
    )


models_source = Path(__file__).with_name("core").joinpath("models.py").read_text(
    encoding="utf-8"
)
check("model source satisfies the fixed-width mechanics contract",
      _model_mechanics_source_contract(models_source))
check("mutation: stride pooling cannot masquerade as adaptive pyramid pooling",
      not _model_mechanics_source_contract(models_source.replace(
          "F.adaptive_max_pool1d", "F.max_pool1d", 1
      )))
check("mutation: a sequence-length-sized dense head is rejected",
      not _model_mechanics_source_contract(models_source.replace(
          "current_features * self.multi_rate_pool.output_bins",
          "current_features * self.n_segments",
          1,
      )))
check("mutation: reverting the truthful class name is rejected",
      not _model_mechanics_source_contract(models_source.replace(
          "class Time2VecPositionEncoding", "class LegacySpaceEncoding", 1
      )))


print("\n--- development adjudication and post-freeze stability ---")
robust_groups = np.repeat(np.arange(8), 2)
robust_X = np.zeros((len(robust_groups), 1, 17), dtype=np.float32)
robust_y = np.zeros((len(robust_groups), 1), dtype=np.float32)
robust_outer_states = np.array([6, 7])
robust_outer_idx = np.flatnonzero(np.isin(robust_groups, robust_outer_states))
robust_dev_idx = np.flatnonzero(~np.isin(robust_groups, robust_outer_states))
robust_strata = ["A"] * 4 + ["B"] * 4
development_calls = []


def _fake_fold_evaluator(**kwargs):
    fold = kwargs["fold"]
    development_calls.append((
        kwargs["seed"],
        int(fold.repeat),
        int(fold.fold),
        tuple(fold.train_states.tolist()),
        tuple(fold.val_states.tolist()),
    ))
    return {
        "state": fold.val_states.copy(),
        "scour_mse": np.full(len(fold.val_states), kwargs["seed"], dtype=float),
    }


development_result = run_development_adjudication(
    config={"name": "fixture", "method": "raw"},
    params={"fixture": 1},
    X=robust_X,
    y=robust_y,
    groups=robust_groups,
    development_idx=robust_dev_idx,
    strata_by_state=robust_strata,
    split_seeds=[11, 12],
    initialization_seeds=[21, 22],
    n_splits=2,
    n_repeats=2,
    n_epochs=3,
    max_epochs=5,
    n_scour_heads=1,
    fit_evaluate=_fake_fold_evaluator,
)
check("development adjudication executes every explicit split/fold/init refit",
      development_result["n_completed_refits"]
      == development_result["n_expected_refits"] == 16
      and len(development_calls) == 16)
check("development interface never evaluates sealed outer states",
      development_result["outer_test_observations_accessed"] is False
      and all(
          not set(call[4]).intersection(robust_outer_states.tolist())
          for call in development_calls
      ))
check("development refits use only prospectively supplied initialization seeds",
      {call[0] for call in development_calls} == {21, 22})
raises("empty prospective split-seed list rejected",
       lambda: run_development_adjudication(
           config={"name": "fixture", "method": "raw"},
           params={}, X=robust_X, y=robust_y, groups=robust_groups,
           development_idx=robust_dev_idx, strata_by_state=robust_strata,
           split_seeds=[], initialization_seeds=[21], n_splits=2,
           n_repeats=1, n_epochs=1, max_epochs=1, n_scour_heads=1,
           fit_evaluate=_fake_fold_evaluator,
       ), ValueError)

post_freeze_calls = []


def _fake_post_freeze_evaluator(**kwargs):
    fold = kwargs["fold"]
    post_freeze_calls.append((
        kwargs["seed"],
        tuple(fold.train_states.tolist()),
        tuple(fold.val_states.tolist()),
    ))
    return {
        "state": fold.val_states.copy(),
        "scour_mse": np.full(len(fold.val_states), kwargs["seed"], dtype=float),
    }


post_freeze_result = run_post_freeze_stability(
    config={"name": "fixture", "method": "paa"},
    params={"fixture": 1},
    X=robust_X,
    y=robust_y,
    groups=robust_groups,
    development_idx=robust_dev_idx,
    sealed_outer_test_idx=robust_outer_idx,
    initialization_seeds=[31, 32],
    n_epochs=3,
    max_epochs=5,
    n_scour_heads=1,
    fit_evaluate=_fake_post_freeze_evaluator,
)
check("post-freeze phase is outer-test report-only",
      post_freeze_result["outer_test_observations_accessed"] is True
      and post_freeze_result["selection_permitted"] is False
      and post_freeze_result["n_completed_refits"] == 2)
check("post-freeze refits use the full development/sealed state partition",
      all(set(call[1]) == set(range(6))
          and set(call[2]) == {6, 7} for call in post_freeze_calls))
crossed_dev = np.sort(np.append(robust_dev_idx, robust_outer_idx[0]))
crossed_outer = robust_outer_idx[1:]
raises("state crossing development and sealed outer test rejected",
       lambda: run_post_freeze_stability(
           config={"name": "fixture", "method": "paa"},
           params={}, X=robust_X, y=robust_y, groups=robust_groups,
           development_idx=crossed_dev,
           sealed_outer_test_idx=crossed_outer,
           initialization_seeds=[31], n_epochs=1, max_epochs=1,
           n_scour_heads=1, fit_evaluate=_fake_post_freeze_evaluator,
       ), ValueError)


def _robustness_source_contract(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = {
        "run_development_adjudication",
        "run_post_freeze_stability",
        "evaluate_development_adjudication",
        "evaluate_post_freeze_stability",
    }
    if not required <= set(functions) or "evaluate_stochastic_robustness" in functions:
        return False

    def required_kwonly(function_name, argument_name):
        function = functions[function_name]
        names = [argument.arg for argument in function.args.kwonlyargs]
        if argument_name not in names:
            return False
        return function.args.kw_defaults[names.index(argument_name)] is None

    development_source = ast.get_source_segment(
        source, functions["run_development_adjudication"]
    )
    post_source = ast.get_source_segment(
        source, functions["run_post_freeze_stability"]
    )
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    return (
        required_kwonly("run_development_adjudication", "split_seeds")
        and required_kwonly(
            "run_development_adjudication", "initialization_seeds"
        )
        and required_kwonly(
            "run_post_freeze_stability", "initialization_seeds"
        )
        and "repeated_stratified_group_folds(" in development_source
        and "sealed_outer_test_idx" not in development_source
        and "split_seeds" not in [
            argument.arg
            for argument in functions[
                "run_post_freeze_stability"
            ].args.kwonlyargs
        ]
        and "selection_permitted=False" in post_source
        and "outer_test_observations_accessed=True" in post_source
        and "n_seeds" not in identifiers
        and "seed=42" not in source
        and "42 +" not in source
    )


robustness_source = Path(__file__).with_name("training").joinpath(
    "robustness.py"
).read_text(encoding="utf-8")
check("robustness source enforces phase and seed separation",
      _robustness_source_contract(robustness_source))
check("mutation: replacing grouped folds is rejected",
      not _robustness_source_contract(robustness_source.replace(
          "repeated_stratified_group_folds(", "list(", 1
      )))
check("mutation: a default split seed is rejected",
      not _robustness_source_contract(robustness_source.replace(
          "split_seeds: Sequence[int],",
          "split_seeds: Sequence[int] = (42,),",
          1,
      )))
check("mutation: selection-enabled sealed-test use is rejected",
      not _robustness_source_contract(robustness_source.replace(
          "selection_permitted=False,", "selection_permitted=True,", 1
      )))


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
