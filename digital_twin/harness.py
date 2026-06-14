"""
digital_twin/harness.py
=======================
Value-of-SHM comparison harness for the drive-by Digital Twin.

Runs the DTSimulator under several planner methodologies (do-nothing /
cost-VI / heuristic / POMDP / hybrid) and reports the expected discounted €
life-cycle cost of each — the Value-of-SHM comparison. Two observation modes:

    mode='mock'  — fast: true scour from MultiScourModel, drive-by observations
                   sampled from a banded confusion matrix. For developing /
                   debugging the framework without the heavy stack.
    mode='live'  — real: PhysicalAsset (live TTBI scour passages) + the champion
                   classifier (DigitalAsset); drive-by likelihood = the champion
                   confusion matrix. Minutes per passage — use the held-out
                   signal library (swap driveby_observe) for production speed.

Only the observation source changes between modes; the loop, planners, and cost
accounting are identical (see digital_twin.simulation.DTSimulator).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from digital_twin.costs import CostModel
from digital_twin.planner import (CostBenefitPlanner, HeuristicPlanner,
                                   HybridPlanner, POMDPPlanner)
from digital_twin.scour_multi import MultiScourModel
from digital_twin.simulation import DTSimulator, row_normalise


# --------------------------------------------------------------------------- #
#  Observation likelihoods
# --------------------------------------------------------------------------- #
def banded_like(n: int, sigma: float) -> np.ndarray:
    """Row-normalised Gaussian-banded confusion matrix L[true, pred]."""
    L = np.array([[np.exp(-0.5 * ((t - p) / sigma) ** 2) for p in range(n)]
                  for t in range(n)])
    return row_normalise(L)


# --------------------------------------------------------------------------- #
#  Planner factory
# --------------------------------------------------------------------------- #
class _DoNothing:
    def __init__(self, *_): pass
    def decide(self, belief, context=None): return "do_nothing"


def build_planner(name, states, cost_model, p_advance):
    """Return (planner, actions_list, uses_inspect)."""
    a3 = ["do_nothing", "inspect", "repair"]
    a2 = ["do_nothing", "repair"]
    if name == "do_nothing":
        return _DoNothing(), a2, False
    if name == "cost_vi":
        return (CostBenefitPlanner(states, a2, cost_model, p_advance=p_advance),
                a2, False)
    if name == "heuristic":
        return (HeuristicPlanner(states, a3, repair_threshold=4, inspect_threshold=3,
                                 inspect_after_flood=True, entropy_frac=0.9),
                a3, True)
    if name == "pomdp":
        return POMDPPlanner(states, a3, cost_model), a3, True
    if name == "hybrid":
        return HybridPlanner(states, a3, cost_model, p_advance=p_advance), a3, True
    raise ValueError(f"unknown planner '{name}'")


# --------------------------------------------------------------------------- #
#  Mock physical asset (fast)
# --------------------------------------------------------------------------- #
class MockPhysical:
    """True scour from MultiScourModel (single support) — no TTBI, no torch."""

    def __init__(self, dt_years, enable_shock, seed, disc, n):
        self.m = MultiScourModel([0.0], enable_shock=enable_shock,
                                 rng=np.random.default_rng(seed))
        self.dt = dt_years
        self.disc = disc
        self.n = n
        self.state_continuous = 0.0
        self.flood_this_step = False

    def update_physical_state(self, action):
        if action in ("repair", "perfect_repair"):
            self.m.repair()
        self.m.evolve(self.dt)
        self.state_continuous = float(self.m.current_X[0])
        self.flood_this_step = self.m.flood_occurred_last_step()

    def get_true_mapped_label(self):
        return int(np.clip(round(self.state_continuous / self.disc), 0, self.n - 1))


# --------------------------------------------------------------------------- #
#  Comparison runner
# --------------------------------------------------------------------------- #
def run_comparison(
    planner_names=("do_nothing", "cost_vi", "heuristic", "pomdp", "hybrid"),
    n_steps: int = 360,
    dt_years: float = 1.0 / 12.0,        # monthly
    n_seeds: int = 30,
    enable_shock: bool = True,
    p_advance: float = 0.03,
    max_damage: float = 60.0,
    discretization: float = 5.0,
    cost_model: CostModel | None = None,
    mode: str = "mock",
    champion_dir: str | None = None,
    driveby_sigma: float = 0.7,
    inspect_sigma: float = 0.25,
) -> pd.DataFrame:
    """Run every planner over n_seeds trajectories; return a comparison table."""
    cm = cost_model or CostModel()
    n = int(max_damage / discretization) + 1
    states = [str(i) for i in range(n)]

    if mode == "live":
        L_drive, L_insp, make_phys, make_obs = _live_setup(
            champion_dir, n, dt_years, enable_shock, inspect_sigma)
    else:
        L_drive = banded_like(n, driveby_sigma)
        L_insp = banded_like(n, inspect_sigma)
        def make_phys(seed):
            return MockPhysical(dt_years, enable_shock, seed, discretization, n)
        def make_obs(L):
            return lambda phys: int(np.random.choice(n, p=L[phys.get_true_mapped_label()]))

    rows = []
    for name in planner_names:
        costs, ninsp, nrep, finals = [], [], [], []
        for s in range(n_seeds):
            np.random.seed(s)
            planner, actions, use_insp = build_planner(name, states, cm, p_advance)
            sim = DTSimulator(
                make_phys(s), planner, cm, states, actions,
                driveby_observe=make_obs(L_drive), driveby_like=L_drive,
                inspect_observe=make_obs(L_insp) if use_insp else None,
                inspect_like=L_insp if use_insp else None,
                p_advance=p_advance, dt_years=dt_years,
                discretization=discretization, max_damage=max_damage,
            )
            df = sim.run(n_steps)
            costs.append(sim.lifecycle_cost())
            ninsp.append(int((df.action == "inspect").sum() + df.inspected.sum()))
            nrep.append(int(df.action.isin(["repair", "perfect_repair"]).sum()))
            finals.append(float(df.true_pct.iloc[-1]))
        rows.append(dict(
            planner=name,
            lifecycle_eur=float(np.mean(costs)),
            lifecycle_eur_std=float(np.std(costs)),
            n_inspect=float(np.mean(ninsp)),
            n_repair=float(np.mean(nrep)),
            mean_final_pct=float(np.mean(finals)),
        ))
    return pd.DataFrame(rows).sort_values("lifecycle_eur").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Live setup (champion classifier + TTBI)
# --------------------------------------------------------------------------- #
def _live_setup(champion_dir, n, dt_years, enable_shock, inspect_sigma):
    """Wire the champion model + PhysicalAsset for live TTBI observations."""
    import sys
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "TTBI_2D"))    # TTBI flat-imports need this
    from digital_twin.assets import DigitalAsset, PhysicalAsset
    from digital_twin.config import DTConfig

    cdir = Path(champion_dir)
    conf = np.load(cdir / "DT_conf_matrix.npy").astype(float)
    L_drive = row_normalise(conf)                # rows=true, cols=pred
    L_insp = banded_like(n, inspect_sigma)

    config = DTConfig.from_metadata(str(cdir / "DT_metadata.json"),
                                    monitoring_interval=dt_years)
    config.enable_shock = enable_shock
    digital = DigitalAsset(str(cdir / "DT_champion_weights.pth"),
                           str(cdir / "DT_metadata.json"),
                           str(cdir / "DT_scaler.pkl"), config)

    def make_phys(seed):
        np.random.seed(seed)
        return PhysicalAsset(config, monitoring_interval=dt_years)

    def make_obs(L):
        if L is L_drive:
            return lambda phys: digital.estimate_state(phys.get_observation_signal())
        return lambda phys: int(np.random.choice(n, p=L[phys.get_true_mapped_label()]))

    return L_drive, L_insp, make_phys, make_obs
