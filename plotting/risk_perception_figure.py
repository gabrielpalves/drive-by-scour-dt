"""
plotting/risk_perception_figure.py
==================================
The risk-perception finding as one slide for Prof. Todd's group.

With a CHEAP, SHARP inspection available, making the decision maker more risk-
averse (CVaR_alpha, alpha shrinking from 1=neutral to 0.05=worst-5% tail) does
NOT buy fewer failures — it buys more MONITORING:

  * inspections per trajectory climb steeply and monotonically;
  * repairs stay flat, and so does the mean final damage;
  * the catastrophic € tail (CVaR_10 across trajectories) does NOT shrink — it
    actually rises with the extra inspection spend.

Mechanism (the talking point): a cheap sharp inspection re-collapses the belief
to a tight spike BEFORE the repair decision, and a risk measure acts on the
SPREAD of the belief (a spike has CVaR ≈ mean). So risk aversion routes into
inspection, while the catastrophic tail is dominated by sudden flood-shock jumps
that outrun the inspect/repair loop and are largely irreducible by maintenance
policy. Cutting that tail needs faster/continuous monitoring or a shock-
anticipating policy (the flood branch), not merely a more risk-averse objective.

Usage
-----
    .venv/Scripts/python plotting/risk_perception_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digital_twin.harness import sweep_risk

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
PLANNER = "pomdp"        # belief-based; hybrid gives the same qualitative story
ATTITUDE = "cvar"
N_STEPS = 360            # 30 years monthly
N_SEEDS = 80
P_ADVANCE = 0.03
DPI = 200
OUT_DIR = Path("results/risk_perception")

C_INSPECT = "#1f77b4"
C_REPAIR = "#d62728"
C_MEAN = "#555555"
C_TAIL = "#C1121F"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = sweep_risk(PLANNER, ATTITUDE, n_steps=N_STEPS, n_seeds=N_SEEDS,
                    p_advance=P_ADVANCE)
    df.to_csv(OUT_DIR / "risk_perception_sweep.csv", index=False)
    print(df[["risk_label", "n_inspect", "n_repair",
              "lifecycle_eur", "lifecycle_eur_cvar10"]].to_string(index=False))

    x = range(len(df))
    labels = list(df["risk_label"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ── Panel A: monitoring (inspections) up, repairs flat ────────────────────
    axL.plot(x, df["n_inspect"], "o-", color=C_INSPECT, lw=2.2, ms=7,
             label="inspections / trajectory")
    axL.set_ylabel("mean inspections per trajectory", color=C_INSPECT)
    axL.tick_params(axis="y", labelcolor=C_INSPECT)
    axL.set_ylim(bottom=0)
    a2 = axL.twinx()
    a2.plot(x, df["n_repair"], "s--", color=C_REPAIR, lw=2.0, ms=7,
            label="repairs / trajectory")
    a2.set_ylabel("mean repairs per trajectory", color=C_REPAIR)
    a2.tick_params(axis="y", labelcolor=C_REPAIR)
    a2.set_ylim(0, max(1.0, float(df["n_repair"].max()) * 4))
    axL.set_title("Risk aversion buys MONITORING\n(inspections ↑, repairs flat)",
                  fontsize=11)
    axL.set_xticks(list(x))
    axL.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axL.set_xlabel("risk attitude   (neutral  →  more risk-averse)")
    axL.grid(alpha=0.25)

    # ── Panel B: the € tail does not shrink ───────────────────────────────────
    axR.plot(x, df["lifecycle_eur"] / 1e6, "o-", color=C_MEAN, lw=2.2, ms=7,
             label="mean life-cycle €")
    axR.plot(x, df["lifecycle_eur_cvar10"] / 1e6, "^-", color=C_TAIL, lw=2.2, ms=8,
             label=r"CVaR$_{10}$ catastrophic tail €")
    axR.set_ylabel("life-cycle cost  (€ millions)")
    axR.set_title("…not a smaller failure tail\n(the flood-shock tail is "
                  "policy-irreducible)", fontsize=11)
    axR.set_xticks(list(x))
    axR.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axR.set_xlabel("risk attitude   (neutral  →  more risk-averse)")
    axR.legend(fontsize=10, loc="upper left", framealpha=0.9)
    axR.grid(alpha=0.25)
    axR.set_ylim(bottom=0)

    fig.suptitle("Risk-sensitive drive-by SHM (POMDP, CVaR): conservatism is spent "
                 "on inspection, not on cutting the catastrophic tail",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "risk_perception.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
