"""
run_dt.py
=========
Entry point for the drive-by Digital-Twin Value-of-SHM study.

Edit the CONFIGURATION block, then run:

    .venv/Scripts/python run_dt.py

It runs every selected planner methodology over the simulation horizon and
prints the expected discounted € life-cycle cost of each (the Value-of-SHM
comparison), saving the table to ablation_analysis/vosh_comparison.csv.

Modes
-----
mode='mock' : fast — synthetic truth + a banded confusion matrix. Use for
              developing the framework and sanity-checking policies.
mode='live' : real — live TTBI passages + the champion classifier. Minutes per
              passage; keep n_steps/n_seeds small, or switch the observation
              source to a held-out signal library for production speed.
"""

from pathlib import Path

from digital_twin.costs import CostModel
from digital_twin.harness import run_comparison

# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
MODE = "mock"            # "mock" (fast) | "live" (TTBI + champion classifier)

# Champion model folder (used only in live mode).
CHAMPION_DIR = "models/champion_PAA_NHiTS_full8dof"

PLANNERS = ["do_nothing", "cost_vi", "heuristic", "pomdp", "hybrid"]

# Simulation clock — floods/deterioration evolve on this fine grid.
DT_YEARS = 1.0 / 12.0    # monthly steps
N_STEPS  = 360           # 360 months = 30 years
N_SEEDS  = 30            # Monte-Carlo trajectories per planner (mock)

ENABLE_SHOCK = True      # Kamariotis flood (Compound Poisson) term
P_ADVANCE    = 0.03      # belief-filter deterioration prior per step (monthly)

# Cost model (verified Kamariotis € figures; tune fragility/downtime as needed).
COST = CostModel()

# =========================================================================== #
def main() -> None:
    if MODE == "live":
        N = 24            # keep tiny in live mode (minutes per passage)
        seeds = 1
        print(f"LIVE mode — champion: {CHAMPION_DIR}  ({N} steps, {seeds} seed)")
    else:
        N = N_STEPS
        seeds = N_SEEDS

    df = run_comparison(
        planner_names=PLANNERS,
        n_steps=N,
        dt_years=DT_YEARS,
        n_seeds=seeds,
        enable_shock=ENABLE_SHOCK,
        p_advance=P_ADVANCE,
        cost_model=COST,
        mode=MODE,
        champion_dir=CHAMPION_DIR,
    )

    print(f"\n=== Value-of-SHM comparison ({MODE}, "
          f"{N} steps x {DT_YEARS:.3f} yr, shock={ENABLE_SHOCK}) ===")
    show = df.copy()
    show["lifecycle_eur"] = show["lifecycle_eur"].map(lambda v: f"{v:.3e}")
    show["lifecycle_eur_std"] = show["lifecycle_eur_std"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    out = Path("ablation_analysis"); out.mkdir(exist_ok=True)
    df.to_csv(out / "vosh_comparison.csv", index=False)
    print(f"\nsaved -> {out / 'vosh_comparison.csv'}")


if __name__ == "__main__":
    main()
