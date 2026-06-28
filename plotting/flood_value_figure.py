"""
plotting/flood_value_figure.py
==============================
The QUANTITATIVE value of the flood-decision branch, for the Todd-group talk —
the number that pairs with the decision-map logic (`flood_decision_map.py`).

Story (and its honest limits): in our framework routine drive-by monitoring is
reliable and monthly, so it already catches a flood's scour jump at the next
passage. The dedicated flood branch therefore adds only a *one-month head start*
plus targeted closures — a MODEST effect on the mean cost. Where it matters is
the **catastrophic tail**: it trims the CVaR_10 life-cycle cost by a few-to-~10 %,
and the effect GROWS as the scour fragility becomes conservative (i.e. when flood-
shock scour actually threatens failure). This is exactly the part of the tail the
risk-perception result showed risk-aversion *cannot* reach (the flood-shock tail
is policy-irreducible by a routine objective) — so the flood-ANTICIPATING protocol
is the policy that addresses it. Fragility is a swept assumption throughout.

Output: results/flood_value/flood_protocol_value.png + a CSV of the raw numbers.

Usage
-----
    .venv/Scripts/python plotting/flood_value_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from digital_twin.costs import CostModel
from digital_twin.harness import run_comparison

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
PLANNERS = ("pomdp", "hybrid")          # the realistic operating planners
FRAGILITIES = [("base", 0.80), ("conservative", 0.40)]   # median (damage fraction)
N_STEPS = 360
N_SEEDS = 150                            # high, for a stable CVaR_10 tail
P_ADVANCE = 0.007
DPI = 200
OUT_DIR = Path("results/flood_value")

C_BASE = "#6BAED6"
C_CONS = "#08519C"


def run_pair(planner, frag_median):
    """Return (off_row, on_row) for one planner at one fragility."""
    cm = CostModel(fragility_median=frag_median, fragility_beta=0.3)
    common = dict(planner_names=(planner,), n_steps=N_STEPS, n_seeds=N_SEEDS,
                  enable_shock=True, p_advance=P_ADVANCE, cost_model=cm)
    off = run_comparison(flood_branch=False, **common).iloc[0]
    on = run_comparison(flood_branch=True, **common).iloc[0]
    return off, on


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for planner in PLANNERS:
        for fname, fmed in FRAGILITIES:
            off, on = run_pair(planner, fmed)
            rows.append(dict(
                planner=planner, fragility=fname, fragility_median=fmed,
                mean_off=off["lifecycle_eur"], mean_on=on["lifecycle_eur"],
                cvar10_off=off["lifecycle_eur_cvar10"],
                cvar10_on=on["lifecycle_eur_cvar10"],
                mean_red_pct=100 * (off["lifecycle_eur"] - on["lifecycle_eur"]) / off["lifecycle_eur"],
                tail_red_pct=100 * (off["lifecycle_eur_cvar10"] - on["lifecycle_eur_cvar10"]) / off["lifecycle_eur_cvar10"],
                n_flood_inspect=on.get("n_flood_inspect", 0.0),
                n_interrupt=on.get("n_interrupt", 0.0),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "flood_protocol_value.csv", index=False)
    print(df[["planner", "fragility", "mean_red_pct", "tail_red_pct",
              "cvar10_off", "cvar10_on"]].to_string(index=False))

    # ── grouped bar: % CVaR_10 tail reduction, planner × fragility ────────────
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(PLANNERS))
    w = 0.36
    for k, (fname, _) in enumerate(FRAGILITIES):
        vals = [df[(df.planner == p) & (df.fragility == fname)]["tail_red_pct"].iloc[0]
                for p in PLANNERS]
        offs = [df[(df.planner == p) & (df.fragility == fname)] for p in PLANNERS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w,
                      color=(C_BASE if fname == "base" else C_CONS),
                      label=f"{fname} fragility (θ={FRAGILITIES[k][1]:g})")
        for xi, p, b in zip(x, PLANNERS, bars):
            r = df[(df.planner == p) & (df.fragility == fname)].iloc[0]
            ax.annotate(f"{r.cvar10_off/1e6:.1f}→{r.cvar10_on/1e6:.1f} M€",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in PLANNERS])
    ax.set_ylabel("reduction in CVaR$_{10}$ catastrophic tail  (%)")
    ax.set_xlabel("routine planner")
    ax.set_title("Value of the flood-decision branch — it trims the catastrophic flood tail\n"
                 "(the part risk-aversion alone cannot reach; larger under a conservative fragility)",
                 fontsize=10.5)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylim(0, max(df["tail_red_pct"].max() * 1.25, 1.0))
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "flood_protocol_value.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
