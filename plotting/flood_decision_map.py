"""
plotting/flood_decision_map.py
==============================
Conceptual figures for the flood-decision branch (Paper 2, Step 2) — the slides
for Prof. Todd's group.

Two artifacts:

  1. flood_trigger_cdf.png — the probabilistic major-flood trigger
     P(classify major | severity) = Φ((ln s − ln s_major)/β_gauge). A river gauge
     measures stage precisely but infers DISCHARGE through a rating curve whose
     error blows up (50-200 %) during the extreme floods that matter, so the
     hard design-flood threshold (β→0 step) becomes a SMOOTH classifier. The
     curve is the object we refine with the Todd group.

  2. flood_decision_map.png — the action the FloodResponsePlanner takes as a
     function of the observed gauge severity (x) and the pre-flood damage belief
     (y), under three conditions: risk-neutral, risk-averse (CVaR), and degraded
     sensors (low probe reliability). The four regions —
     do_nothing / restrict_operations(probe) / inspect / interrupt — fall out of
     a risk-adjusted one-step Value-of-Information calculation; risk aversion and
     sensor degradation visibly shift the escalation boundaries (risk → close
     sooner; degraded probe → skip the probe, inspect).

Decision granularity is the readable 13-class (5 %) grid; the production champion
is 61-class but the decision STRUCTURE is identical and far clearer at 5 %.

Usage
-----
    .venv/Scripts/python plotting/flood_decision_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from digital_twin.costs import CostModel
from digital_twin.flood import FloodTrigger, widen_belief
from digital_twin.harness import banded_like
from digital_twin.planner import FloodResponsePlanner
from digital_twin.risk import RiskModel

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
DISC = 5.0                       # % per damage label (13-class, readable map)
MAX_DAMAGE = 60.0
N = int(MAX_DAMAGE / DISC) + 1   # 13
DPI = 200
OUT_DIR = Path("results/flood_decision")

SEVERITY_MAJOR = 1.5             # major/minor boundary on the severity scale
GAUGE_BETA = 0.6                 # discharge-reading dispersion
PROBE_SIGMA = 1.0                # assumed wider (low-SNR) probe confusion
INSPECT_SIGMA = 0.25            # sharp inspection confusion
PRIOR_SIGMA_LABELS = 0.6         # spread of the (realistic) pre-flood belief

SEV_GRID = np.linspace(0.4, 4.5, 130)        # observed gauge severity (x-axis)
# Realistic pre-flood damage range: routine maintenance keeps scour low, so an
# asset rarely ENTERS a major flood already badly scoured. Showing 0-30% keeps
# the map on the operationally-relevant region (above ~30% you would have
# repaired) and makes all four decision regions legible.
PRIOR_MAX_LABEL = 6                            # 30 % scour
PRIOR_LABELS = np.arange(PRIOR_MAX_LABEL + 1)  # pre-flood damage label (y-axis)

# Action order = escalation order; colours from calm green to alarm red.
ACTIONS = ["do_nothing", "restrict_operations", "inspect", "interrupt"]
ACTION_LABEL = {"do_nothing": "do nothing",
                "restrict_operations": "restrict ops (probe)",
                "inspect": "inspect",
                "interrupt": "interrupt (close)"}
ACTION_COLORS = ["#2E7D32", "#F9C74F", "#F3722C", "#C1121F"]
A_IDX = {a: i for i, a in enumerate(ACTIONS)}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def prior_belief(label: int) -> np.ndarray:
    """Realistic pre-flood belief: a narrow Gaussian bump around `label`."""
    x = np.arange(N, dtype=float)
    b = np.exp(-0.5 * ((x - label) / PRIOR_SIGMA_LABELS) ** 2)
    return b / b.sum()


def make_planner(attitude="neutral", aversion=1.0) -> FloodResponsePlanner:
    cm = CostModel()
    risk = RiskModel(attitude=attitude, level=aversion,
                     cost_scale=cm.c_failure + cm.c_downtime_day * cm.downtime_days_failure)
    return FloodResponsePlanner(
        [str(i) for i in range(N)], cm,
        banded_like(N, PROBE_SIGMA), banded_like(N, INSPECT_SIGMA),
        max_damage=MAX_DAMAGE, discretization=DISC, risk_model=risk)


def decision_grid(planner: FloodResponsePlanner, probe_reliability=1.0) -> np.ndarray:
    """Action index over (prior_label, severity); rows=prior, cols=severity."""
    grid = np.zeros((len(PRIOR_LABELS), len(SEV_GRID)), dtype=int)
    for i, lbl in enumerate(PRIOR_LABELS):
        base = prior_belief(lbl)
        for j, sev in enumerate(SEV_GRID):
            b = widen_belief(base, sev, DISC)
            J = planner.score(b, probe_reliability)
            grid[i, j] = A_IDX[min(J, key=J.get)]
    return grid


# --------------------------------------------------------------------------- #
#  Figure 1 — the probabilistic trigger CDF
# --------------------------------------------------------------------------- #
def plot_trigger_cdf(path: Path) -> None:
    s = np.linspace(0.0, 5.0, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for beta, ls, lab in [(1e-6, "--", "hard threshold (β→0)"),
                          (0.35, "-", "sharp gauge (β=0.35)"),
                          (0.60, "-", "default gauge (β=0.60)"),
                          (0.90, "-", "very uncertain (β=0.90)")]:
        ax.plot(s, FloodTrigger(SEVERITY_MAJOR, beta).p_major(s), ls, lw=2.2,
                label=lab, color=("0.4" if beta < 1e-3 else None))
    ax.axvline(SEVERITY_MAJOR, color="0.6", lw=1, ls=":")
    ax.text(SEVERITY_MAJOR + 0.05, 0.04, "design-flood\nseverity $s_{major}$",
            fontsize=9, color="0.4")
    ax.set_xlabel("flood severity  $s$  (scour jump / mean jump)")
    ax.set_ylabel("P(classify as major flood)")
    ax.set_title("Probabilistic major-flood trigger\n"
                 r"$P(\mathrm{major}\,|\,s)=\Phi\!\left(\frac{\ln s-\ln s_{major}}{\beta_{gauge}}\right)$"
                 "  — rating-curve uncertainty smooths the threshold", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved -> {path}")


# --------------------------------------------------------------------------- #
#  Figure 2 — the decision maps
# --------------------------------------------------------------------------- #
def _draw_map(ax, grid, title):
    cmap = ListedColormap(ACTION_COLORS)
    y_top = (len(PRIOR_LABELS) - 0.5) * DISC
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=3,
              extent=[SEV_GRID[0], SEV_GRID[-1], -0.5 * DISC, y_top])
    ax.axvline(SEVERITY_MAJOR, color="white", lw=1.2, ls=":")
    ax.set_xlabel("observed gauge severity  $s$")
    ax.set_title(title, fontsize=10)


def plot_decision_maps(path: Path) -> None:
    neutral = make_planner("neutral")
    averse = make_planner("cvar", 0.1)
    grids = [
        (decision_grid(neutral), "risk-neutral\n(healthy sensors)"),
        (decision_grid(averse), "risk-averse  CVaR$_{0.1}$\n(healthy sensors)"),
        (decision_grid(neutral, probe_reliability=0.15),
         "degraded sensors\n(probe reliability 0.15)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (grid, title) in zip(axes, grids):
        _draw_map(ax, grid, title)
    axes[0].set_ylabel("pre-flood damage belief  (% scour)")
    handles = [Patch(facecolor=c, label=ACTION_LABEL[a])
               for a, c in zip(ACTIONS, ACTION_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Flood-response decision map — action vs observed severity × pre-flood belief\n"
                 "(risk aversion closes sooner; a degraded probe is skipped for inspection)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_trigger_cdf(OUT_DIR / "flood_trigger_cdf.png")
    plot_decision_maps(OUT_DIR / "flood_decision_map.png")
    print("done.")


if __name__ == "__main__":
    main()
