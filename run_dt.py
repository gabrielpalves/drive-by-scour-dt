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
from digital_twin.harness import run_comparison, sweep_risk

# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
MODE = "mock"            # "mock" (fast) | "library" (champion + held-out signals) | "live" (TTBI)

# Champion model folder (live/library modes). The production champion is the
# 2-sensor pair RearBogie_Vert + CarBody_Pitch (DOFs 2,5), PAA + N-HiTS, 61-class
# — best 2-sensor pair (median MSE 0.41 ~ full-8DOF 0.39) with robust fallbacks.
# 'models/champion_PAA_NHiTS_full8dof' (all 8 DOFs) is kept as a reference.
CHAMPION_DIR = "models/champion_PAA_NHiTS_2sensor_RBvert_CBpitch"

# Single-sensor fallback champions, used by the sensor-health layer when one of
# the two sensors fails (key = the surviving DOF index). Both are PAA + N-HiTS,
# 61-class, trained on the SAME data variant as the champion.
FALLBACK_DIRS = {
    2: "models/fallback_PAA_NHiTS_1sensor_RBvert",    # CarBody_Pitch (5) failed
    5: "models/fallback_PAA_NHiTS_1sensor_CBpitch",   # RearBogie_Vert (2) failed
}

# Held-out signal library folder (library mode) — generate this with the MATLAB
# pipeline (per-state .mat passages, unseen seeds). 'data/data_all_variabilities'
# is only a dev stand-in (it overlaps the training set -> optimistic).
LIBRARY_DIR = "data/held_out"

PLANNERS = ["do_nothing", "cost_vi", "heuristic", "pomdp", "hybrid"]

# Simulation clock — floods/deterioration evolve on this fine grid.
DT_YEARS = 1.0 / 12.0    # monthly steps
N_STEPS  = 360           # 360 months = 30 years
N_SEEDS  = 30            # Monte-Carlo trajectories per planner (mock)

ENABLE_SHOCK = True      # Kamariotis flood (Compound Poisson) term
P_ADVANCE    = 0.007     # belief-filter deterioration prior per step (monthly);
                         # calibrated from the true ScourModel — see digital_twin/calibrate.py

# Cost model (verified Kamariotis € figures; tune fragility/downtime as needed).
COST = CostModel()

# Risk perception (digital_twin/risk.py) — makes the belief-based planners
# (pomdp, hybrid) risk-sensitive; the others ignore it. Chadha 2023 / Ames 2025.
#   RISK_ATTITUDE = "neutral"  -> risk-neutral baseline (expected cost)
#                 = "cvar"     -> CVaR_alpha; RISK_AVERSION = alpha in (0,1],
#                                 SMALLER = more averse (e.g. 0.1 = worst-10%)
#                 = "entropic" -> exponential utility; RISK_AVERSION = theta >= 0,
#                                 LARGER = more averse
RISK_ATTITUDE = "neutral"
RISK_AVERSION = 1.0

# Sensor health (digital_twin/sensor_health.py) — makes the drive-by sensors
# fallible. Faults corrupt the signal; a dead sensor falls back to the surviving
# single-DOF champion in FALLBACK_DIRS (and its wider confusion matrix, so the
# belief distrusts it); sensor maintenance is costed. Placeholder fault rates
# live in SensorHealthParams (sweep/override there once real MTBF data arrives).
SENSOR_HEALTH = False

# Flood-decision branch (digital_twin/flood.py + planner.FloodResponsePlanner) —
# Step 2. On an observed MAJOR flood (a probabilistic river-gauge trigger, not a
# hard threshold — discharge is 50-200% uncertain during extreme floods) the
# belief is WIDENED by the uncertain scour shock and the planner picks
# restrict_operations(probe) / inspect / interrupt by risk-adjusted VoI,
# escalating to inspect if the probe is sensor-corrupted. Applies to every
# planner, so flood-branch ON vs OFF is the value of the flood protocol.
FLOOD_BRANCH         = False
FLOOD_SEVERITY_MAJOR = 1.5    # severity at the major/minor boundary (~design flood)
FLOOD_GAUGE_BETA     = 0.6    # discharge-reading dispersion (rating-curve error)
PROBE_SIGMA          = 1.0    # assumed wider probe confusion (until a measured
                              # low-mass/low-speed MATLAB batch -> probe_dir)
N_PROBE              = 2      # restricted-ops passages fused per major flood

# SWEEP_RISK: when True, sweep the risk-aversion knob for SWEEP_PLANNER and report
# how lifecycle/tail € cost and inspection/repair rates move with risk attitude
# (the risk-perception-vs-Value-of-SHM curve) instead of the planner comparison.
SWEEP_RISK     = False
SWEEP_PLANNER  = "pomdp"      # belief-based planner to sweep ('pomdp' | 'hybrid')
SWEEP_ATTITUDE = "cvar"       # 'cvar' | 'entropic'

# =========================================================================== #
def main() -> None:
    # library/live modes use the real 61-class champion (disc=1); mock uses 13.
    disc = 1.0 if MODE in ("library", "live") else 5.0
    if MODE == "live":
        N, seeds = 24, 1          # minutes per passage — keep tiny
        print(f"LIVE mode — champion: {CHAMPION_DIR}  ({N} steps, {seeds} seed)")
    elif MODE == "library":
        N, seeds = N_STEPS, max(3, N_SEEDS // 5)
        print(f"LIBRARY mode — champion: {CHAMPION_DIR} | library: {LIBRARY_DIR}")
    else:
        N, seeds = N_STEPS, N_SEEDS

    out = Path("ablation_analysis"); out.mkdir(exist_ok=True)
    common = dict(n_steps=N, dt_years=DT_YEARS, n_seeds=seeds,
                  enable_shock=ENABLE_SHOCK, p_advance=P_ADVANCE,
                  discretization=disc, cost_model=COST, mode=MODE,
                  champion_dir=CHAMPION_DIR, library_dir=LIBRARY_DIR,
                  sensor_health=SENSOR_HEALTH, fallback_dirs=FALLBACK_DIRS,
                  flood_branch=FLOOD_BRANCH, flood_severity_major=FLOOD_SEVERITY_MAJOR,
                  flood_gauge_beta=FLOOD_GAUGE_BETA, probe_sigma=PROBE_SIGMA,
                  n_probe=N_PROBE)

    if SWEEP_RISK:
        df = sweep_risk(SWEEP_PLANNER, SWEEP_ATTITUDE, **common)
        print(f"\n=== Risk perception vs Value-of-SHM "
              f"({SWEEP_PLANNER}, {SWEEP_ATTITUDE} sweep, {MODE}) ===")
        show = df.copy()
        for c in ("lifecycle_eur", "lifecycle_eur_cvar10", "lifecycle_eur_p90"):
            show[c] = show[c].map(lambda v: f"{v:.3e}")
        print(show.to_string(index=False))
        df.to_csv(out / "vosh_risk_sweep.csv", index=False)
        print(f"\nsaved -> {out / 'vosh_risk_sweep.csv'}")
        return

    df = run_comparison(planner_names=PLANNERS, risk_attitude=RISK_ATTITUDE,
                        risk_aversion=RISK_AVERSION, **common)

    risk_tag = "neutral" if RISK_ATTITUDE == "neutral" else f"{RISK_ATTITUDE}/{RISK_AVERSION:g}"
    print(f"\n=== Value-of-SHM comparison ({MODE}, "
          f"{N} steps x {DT_YEARS:.3f} yr, shock={ENABLE_SHOCK}, risk={risk_tag}) ===")
    show = df.copy()
    for c in ("lifecycle_eur", "lifecycle_eur_cvar10", "lifecycle_eur_p90"):
        show[c] = show[c].map(lambda v: f"{v:.3e}")
    show["lifecycle_eur_std"] = show["lifecycle_eur_std"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    df.to_csv(out / "vosh_comparison.csv", index=False)
    print(f"\nsaved -> {out / 'vosh_comparison.csv'}")


if __name__ == "__main__":
    main()
