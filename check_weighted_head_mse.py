"""Adversarial checks for the range-normalized regression loss.

Run:  python check_weighted_head_mse.py
"""
from __future__ import annotations

import inspect
import math
import sys

import torch

from core import task
from training import trainer


fails = 0


def check(name: str, cond: bool) -> None:
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def rejects(name: str, fn) -> None:
    try:
        fn()
    except ValueError:
        check(name, True)
    except Exception as err:
        print(f"    unexpected {type(err).__name__}: {err}")
        check(name, False)
    else:
        check(name, False)


print("WEIGHTED-HEAD-MSE CHECKS")

# Exact weighting contract: inverse squared range, normalized to mean one.
crit = task.WeightedHeadMSE([60.0, 95.0])
raw = torch.tensor([1 / 60**2, 1 / 95**2], dtype=torch.float32)
expected_w = raw / raw.mean()
check("inverse-range-squared weights", torch.allclose(crit.w, expected_w))
check("weights have mean one", torch.isclose(crit.w.mean(), torch.tensor(1.0)))

pred = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
target = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
loss = crit(pred, target)
manual = (expected_w * (pred - target).square()).mean()
loss.backward()
check("multi-head value matches manual formula", torch.allclose(loss, manual))
check("multi-head backward finite", pred.grad is not None
      and bool(torch.isfinite(pred.grad).all()))

# The pre-fix implementation normalized only when pred itself was 1-D.
# A conventional (B,1) prediction plus (B,) target broadcast to (B,B), silently
# introducing cross-sample error. Both common one-head layouts must be identical.
p1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
t1 = torch.tensor([1.0, 2.0, 3.0])
c1 = task.WeightedHeadMSE([60.0])
check("one-head (B,) layout", c1(p1, t1).item() == 0.0)
check("one-head (B,1) pred + (B,) target does not cross-broadcast",
      c1(p1.unsqueeze(1), t1).item() == 0.0)

rejects("mismatched batch shapes rejected",
        lambda: c1(torch.zeros(3, 1), torch.zeros(2)))
rejects("wrong number of heads rejected",
        lambda: c1(torch.zeros(3, 2), torch.zeros(3, 2)))
for bad in ([], [0.0], [-1.0], [math.inf], [math.nan]):
    rejects(f"invalid ranges rejected: {bad!r}",
            lambda bad=bad: task.WeightedHeadMSE(bad))

# Exercise the actual trainer's device policy and a trainer-style optimizer
# step. Criteria are modules too; both trainer entry points must move their
# registered buffers to DEVICE instead of relying on a per-forward hidden copy.
train_src = inspect.getsource(trainer.train_and_evaluate)
robust_src = inspect.getsource(trainer.run_single_training)
check("Optuna trainer moves criterion to DEVICE",
      'TRAIN_PROTOCOL["loss"]' in train_src
      and "task.make_criterion(" in train_src
      and ".to(DEVICE)" in train_src)
check("fixed-seed trainer moves criterion to DEVICE",
      'TRAIN_PROTOCOL["loss"]' in robust_src
      and "task.make_criterion(" in robust_src
      and ".to(DEVICE)" in robust_src)
check("trainer consumes the protocol-hashed objective mapping",
      "task.objective_value(" in train_src
      and 'TRAIN_PROTOCOL["objective"],' in train_src)

cfg = {
    "task": "regression",
    "target_supports": [2, 3],
    "bearing_targets": ["left", "right"],
}
objective_policy = trainer.TRAIN_PROTOCOL["objective"]
objective_metrics = {"mse": 9.0, "scour_mse": 2.0}
check("objective is SCOUR-primary when bearing heads exist",
      task.objective_value(objective_metrics, cfg, objective_policy) == 2.0)
check("objective defaults to aggregate MSE without bearing heads",
      task.objective_value(
          objective_metrics,
          {"task": "regression", "target_supports": [2, 3]},
          objective_policy,
      ) == 9.0)
try:
    trainer.resolve_trial_seed({}, trainer.TRAIN_PROTOCOL["trial_seed"])
except KeyError:
    check("missing trial seed fails closed", True)
else:
    check("missing trial seed fails closed", False)
check("registered trial seed is returned exactly",
      trainer.resolve_trial_seed(
          {"seed": 2026}, trainer.TRAIN_PROTOCOL["trial_seed"]
      ) == 2026)
device = trainer.DEVICE
model = torch.nn.Linear(5, 4).to(device)
loss_policy = trainer.TRAIN_PROTOCOL["loss"]
criterion = task.make_criterion(cfg, loss_policy).to(device)
optimizer = trainer.make_optimizer(
    model.parameters(),
    {"lr": 1e-3, "weight_decay": 1e-4},
    trainer.TRAIN_PROTOCOL["optimizer"],
)
x = torch.randn(8, 5, device=device)
y = torch.randn(8, 4, device=device)
optimizer.zero_grad()
trainer_style_loss = criterion(model(x), y)
trainer_style_loss.backward()
optimizer.step()
check(f"trainer-style step finite on {device}",
      bool(torch.isfinite(trainer_style_loss)))
check("criterion buffer follows trainer device",
      criterion.w.device.type == device.type)

# The registered branch/ranges are executable, not prose. Exercise every
# branch and prove that mutating only the policy changes the actual weights.
check("classification policy builds cross-entropy",
      isinstance(task.make_criterion(
          {"task": "classification"}, loss_policy), torch.nn.CrossEntropyLoss))
check("scour-only regression policy builds plain MSE",
      isinstance(task.make_criterion(
          {"task": "regression", "target_supports": [2, 3]},
          loss_policy), torch.nn.MSELoss))
registered_ranges = loss_policy[
    "regression_with_bearing_heads"]["head_ranges_pct"]
check("registered scour/bearing ranges remain 60/95 percent",
      registered_ranges == {"scour": 60.0, "bearing": 95.0})
expected_registered = torch.tensor(
    [1 / 60**2, 1 / 60**2, 1 / 95**2, 1 / 95**2],
    dtype=torch.float32,
)
expected_registered /= expected_registered.mean()
check("make_criterion weights are derived from TRAIN_PROTOCOL ranges",
      torch.allclose(criterion.w.cpu(), expected_registered))
mutated_loss_policy = {
    **loss_policy,
    "regression_with_bearing_heads": {
        **loss_policy["regression_with_bearing_heads"],
        "head_ranges_pct": {"scour": 30.0, "bearing": 95.0},
    },
}
mutated_criterion = task.make_criterion(cfg, mutated_loss_policy)
check("changing only loss policy changes executed head weights",
      not torch.allclose(mutated_criterion.w.cpu(), criterion.w.cpu()))
rejects("malformed loss policy rejected",
        lambda: task.make_criterion(
            cfg, {"classification": {"kind": "cross_entropy"}}))

# Optimizer/scheduler are likewise constructed from structured hashed specs.
probe = torch.nn.Parameter(torch.tensor([1.0]))
optimizer_probe = trainer.make_optimizer(
    [probe], {"lr": 2e-3, "weight_decay": 3e-4},
    trainer.TRAIN_PROTOCOL["optimizer"])
check("optimizer policy builds Adam with registered parameter keys",
      isinstance(optimizer_probe, torch.optim.Adam)
      and optimizer_probe.param_groups[0]["lr"] == 2e-3
      and optimizer_probe.param_groups[0]["weight_decay"] == 3e-4)
mutated_optimizer_policy = {
    **trainer.TRAIN_PROTOCOL["optimizer"],
    "lr_param": "alternate_lr",
}
mutated_optimizer = trainer.make_optimizer(
    [torch.nn.Parameter(torch.tensor([1.0]))],
    {"lr": 2e-3, "alternate_lr": 4e-3, "weight_decay": 3e-4},
    mutated_optimizer_policy,
)
check("changing only optimizer policy changes executed learning rate",
      mutated_optimizer.param_groups[0]["lr"] == 4e-3)
scheduler_probe = trainer.make_scheduler(
    optimizer_probe, 7, trainer.TRAIN_PROTOCOL["scheduler"])
check("scheduler policy builds cosine schedule with campaign horizon",
      isinstance(scheduler_probe, torch.optim.lr_scheduler.CosineAnnealingLR)
      and scheduler_probe.T_max == 7
      and scheduler_probe.eta_min == 0.0)
mutated_scheduler_policy = {
    **trainer.TRAIN_PROTOCOL["scheduler"],
    "eta_min": 1e-5,
}
mutated_scheduler = trainer.make_scheduler(
    optimizer_probe, 7, mutated_scheduler_policy)
check("changing only scheduler policy changes executed eta_min",
      mutated_scheduler.eta_min == 1e-5)
rejects("unsupported optimizer kind rejected",
        lambda: trainer.make_optimizer(
            [torch.nn.Parameter(torch.tensor([1.0]))],
            {"lr": 1e-3, "weight_decay": 0.0},
            {**trainer.TRAIN_PROTOCOL["optimizer"], "kind": "SGD"}))
rejects("malformed scheduler policy rejected",
        lambda: trainer.make_scheduler(
            optimizer_probe, 7, {"kind": "CosineAnnealingLR"}))

# The campaign trainer currently has no autocast/GradScaler path. Still verify
# half-precision predictions on CUDA because future AMP would exercise this
# mixed-dtype multiplication; the loss intentionally accumulates in float32.
if torch.cuda.is_available():
    hp = torch.randn(8, 4, device="cuda", dtype=torch.float16,
                     requires_grad=True)
    ht = torch.randn_like(hp)
    hc = task.WeightedHeadMSE([60, 60, 95, 95]).to("cuda")
    hl = hc(hp, ht)
    hl.backward()
    check("CUDA float16 forward/backward finite", bool(torch.isfinite(hl))
          and hp.grad is not None and bool(torch.isfinite(hp.grad).all()))
    check("mixed precision loss accumulates float32", hl.dtype == torch.float32)
else:
    print("  [SKIP] CUDA float16 check (CUDA unavailable)")

print()
print("WEIGHTED HEAD MSE: ALL PASS" if fails == 0
      else f"WEIGHTED HEAD MSE: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
