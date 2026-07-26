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
      "task.make_criterion(config).to(DEVICE)" in train_src)
check("fixed-seed trainer moves criterion to DEVICE",
      "task.make_criterion(config).to(DEVICE)" in robust_src)

cfg = {
    "task": "regression",
    "target_supports": [2, 3],
    "bearing_targets": ["left", "right"],
}
device = trainer.DEVICE
model = torch.nn.Linear(5, 4).to(device)
criterion = task.make_criterion(cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
