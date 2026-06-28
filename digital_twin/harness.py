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
from digital_twin.risk import RiskModel
from digital_twin.scour_multi import MultiScourModel
from digital_twin.sensor_health import (HealthAwareClassifier,
                                         MockHealthClassifier, SensorHealthModel,
                                         SensorHealthParams, health_observe,
                                         mock_health_observe)
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


def make_risk_model(cost_model, risk_attitude="neutral", risk_aversion=1.0):
    """RiskModel with cost_scale fixed to a representative catastrophic € cost.

    The scale (failure cost + failure close-down) keeps the entropic θ knob
    dimensionless and numerically stable; CVaR ignores it. Returns a neutral
    model when risk_attitude='neutral'.
    """
    scale = (cost_model.c_failure
             + cost_model.c_downtime_day * cost_model.downtime_days_failure)
    return RiskModel(attitude=risk_attitude, level=risk_aversion, cost_scale=scale)


def build_planner(name, states, cost_model, p_advance, discretization=5.0,
                  max_damage=60.0, risk_attitude="neutral", risk_aversion=1.0):
    """Return (planner, actions_list, uses_inspect).

    risk_attitude/risk_aversion configure a RiskModel (digital_twin.risk) that
    makes the belief-based planners (pomdp, hybrid) risk-sensitive; the other
    planners ignore it. Default neutral reproduces the risk-neutral baseline.
    """
    a3 = ["do_nothing", "inspect", "repair"]
    a2 = ["do_nothing", "repair"]
    risk = make_risk_model(cost_model, risk_attitude, risk_aversion)
    if name == "do_nothing":
        return _DoNothing(), a2, False
    if name == "cost_vi":
        return (CostBenefitPlanner(states, a2, cost_model, max_damage=max_damage,
                                   discretization=discretization, p_advance=p_advance),
                a2, False)
    if name == "heuristic":
        # Thresholds in % -> labels, so they hold at any discretisation
        # (the cost/POMDP planners use damage fractions and adapt automatically).
        rep = max(1, round(20.0 / discretization))   # repair ~20% scour
        ins = max(1, round(15.0 / discretization))   # watch from ~15%
        return (HeuristicPlanner(states, a3, repair_threshold=rep, inspect_threshold=ins,
                                 inspect_after_flood=True, entropy_frac=0.9),
                a3, True)
    if name == "pomdp":
        return (POMDPPlanner(states, a3, cost_model, max_damage=max_damage,
                             discretization=discretization, risk_model=risk),
                a3, True)
    if name == "hybrid":
        return (HybridPlanner(states, a3, cost_model, max_damage=max_damage,
                              discretization=discretization, p_advance=p_advance,
                              risk_model=risk),
                a3, True)
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
        self.flood_severity = 0.0

    def update_physical_state(self, action):
        if action in ("repair", "perfect_repair"):
            self.m.repair()
        self.m.evolve(self.dt)
        self.state_continuous = float(self.m.current_X[0])
        self.flood_this_step = self.m.flood_occurred_last_step()
        self.flood_severity = self.m.flood_severity()

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
    library_dir: str | None = None,
    n_passages: int = 8,
    driveby_sigma: float = 0.7,
    inspect_sigma: float = 0.25,
    risk_attitude: str = "neutral",
    risk_aversion: float = 1.0,
    sensor_health: bool = False,
    sensor_params: SensorHealthParams | None = None,
    fallback_dirs: dict | None = None,
    champion_dofs=(2, 5),
    fallback_sigma: float = 1.6,
    flood_branch: bool = False,
    flood_severity_major: float = 1.5,
    flood_gauge_beta: float = 0.6,
    probe_sigma: float = 1.0,
    probe_dir: str | None = None,
    n_probe: int = 2,
    flood_exposure_frac: float = 0.02,
    flood_jump_pct_mean: float = 5.0,
    flood_jump_beta: float = 0.7,
    probe_corruption_prob: float = 0.0,
) -> pd.DataFrame:
    """Run every planner over n_seeds trajectories; return a comparison table.

    risk_attitude/risk_aversion (digital_twin.risk) make the belief-based
    planners risk-sensitive; default neutral is the risk-neutral baseline.

    sensor_health (digital_twin.sensor_health) makes the drive-by sensors
    fallible: faults corrupt the signal, a dead sensor triggers fall-back to the
    surviving single-DOF champion (and its wider confusion matrix, so the belief
    distrusts it), and sensor maintenance is costed. fallback_dirs={dof: path}
    are the single-DOF champions (library/live); champion_dofs/fallback_sigma
    parametrise the torch-free mock equivalent.

    flood_branch (digital_twin.flood + planner.FloodResponsePlanner) adds the
    Step-2 major-flood decision branch to EVERY planner: an observed major flood
    (FloodTrigger probabilistic classification of a noisy river gauge) widens the
    belief and chooses restrict_operations(probe) / inspect / interrupt by risk-
    adjusted VoI, escalating to inspect if the probe is sensor-corrupted. The
    probe observation model is a WIDER banded matrix (probe_sigma) — or a measured
    one loaded from probe_dir (the low-mass/low-speed MATLAB batch, §6.5) when
    available. probe_corruption_prob injects per-passage probe loss (mock realised
    escalation); flood_* expose the trigger/widening knobs for the decision map.
    """
    cm = cost_model or CostModel()
    n = int(max_damage / discretization) + 1
    states = [str(i) for i in range(n)]

    extra = {}
    if mode == "live":
        L_drive, L_insp, make_phys, make_obs, extra = _live_setup(
            champion_dir, n, dt_years, enable_shock, inspect_sigma)
    elif mode == "library":
        L_drive, L_insp, make_phys, make_obs, extra = _library_setup(
            champion_dir, library_dir, n, dt_years, enable_shock,
            inspect_sigma, n_passages, discretization)
    else:
        L_drive = banded_like(n, driveby_sigma)
        L_insp = banded_like(n, inspect_sigma)
        def make_phys(seed):
            return MockPhysical(dt_years, enable_shock, seed, discretization, n)
        def make_obs(L):
            return lambda phys: int(np.random.choice(n, p=L[phys.get_true_mapped_label()]))

    # ── sensor-health observer factory (per-seed: the health model is stateful) ──
    sh_dofs = tuple(extra["digital"].active_dofs) if (sensor_health and "digital" in extra) \
        else tuple(champion_dofs)
    sh_classifier = None
    if sensor_health:
        if mode in ("library", "live"):
            sh_classifier = _build_health_classifier(extra, fallback_dirs, L_drive, n)
        else:
            sh_classifier = MockHealthClassifier(sh_dofs, n, driveby_sigma, fallback_sigma)

    def make_driveby(seed):
        """Return (driveby_observe, sensor_health_obj_or_None) for this seed."""
        if not sensor_health:
            return make_obs(L_drive), None
        health = SensorHealthModel(sh_dofs, sensor_params,
                                   rng=np.random.default_rng(10_000 + seed))
        if mode in ("library", "live"):
            obs = health_observe(extra["lib"], sh_classifier, health,
                                 np.random.default_rng(20_000 + seed))
        else:
            obs = mock_health_observe(sh_classifier, health)
        return obs, health

    # ── flood-decision branch (Step 2) ───────────────────────────────────────────
    # Built once (independent of the routine planner): the probabilistic trigger
    # and the VoI flood-response planner. The probe observation model is the
    # assumed wider matrix (probe_sigma) unless a measured one is supplied.
    flood_trigger = flood_response = L_probe = None
    if flood_branch:
        from digital_twin.flood import FloodTrigger
        from digital_twin.planner import FloodResponsePlanner
        if probe_dir:
            L_probe = row_normalise(np.load(Path(probe_dir) / "DT_conf_matrix.npy").astype(float))
        else:
            L_probe = banded_like(n, probe_sigma)        # assumed lower-SNR probe
        flood_trigger = FloodTrigger(severity_major=flood_severity_major,
                                     gauge_beta=flood_gauge_beta)
        flood_response = FloodResponsePlanner(
            states, cm, L_probe, L_insp, max_damage=max_damage,
            discretization=discretization,
            risk_model=make_risk_model(cm, risk_attitude, risk_aversion),
            exposure_frac=flood_exposure_frac)

    def make_probe_obs():
        """Restricted-ops probe sampler; returns None on a corrupted passage so the
        simulator escalates to a sharp inspection."""
        def obs(phys):
            if probe_corruption_prob > 0 and np.random.random() < probe_corruption_prob:
                return None
            return int(np.random.choice(n, p=L_probe[phys.get_true_mapped_label()]))
        return obs

    rows = []
    for name in planner_names:
        costs, ninsp, nrep, finals, nrepl, faultsteps = [], [], [], [], [], []
        nmaj, nrestr, nintr, nfinsp, nesc = [], [], [], [], []
        for s in range(n_seeds):
            np.random.seed(s)
            planner, actions, use_insp = build_planner(
                name, states, cm, p_advance, discretization, max_damage,
                risk_attitude=risk_attitude, risk_aversion=risk_aversion)
            driveby, health = make_driveby(s)
            want_inspect = use_insp or flood_branch     # flood inspect/interrupt need it
            sim = DTSimulator(
                make_phys(s), planner, cm, states, actions,
                driveby_observe=driveby, driveby_like=L_drive,
                inspect_observe=make_obs(L_insp) if want_inspect else None,
                inspect_like=L_insp if want_inspect else None,
                p_advance=p_advance, dt_years=dt_years,
                discretization=discretization, max_damage=max_damage,
                sensor_health=health,
                flood_response=flood_response, flood_trigger=flood_trigger,
                probe_observe=make_probe_obs() if flood_branch else None,
                probe_like=L_probe, n_probe=n_probe,
                probe_reliability=1.0 - probe_corruption_prob,
                flood_jump_pct_mean=flood_jump_pct_mean, flood_jump_beta=flood_jump_beta,
                flood_rng=np.random.default_rng(30_000 + s),
            )
            df = sim.run(n_steps)
            costs.append(sim.lifecycle_cost())
            ninsp.append(int((df.action == "inspect").sum() + df.inspected.sum()))
            nrep.append(int(df.action.isin(["repair", "perfect_repair"]).sum()))
            finals.append(float(df.true_pct.iloc[-1]))
            nrepl.append(health.n_replacements if health is not None else 0)
            faultsteps.append(int((df.driveby_pred.isna()).sum()) if "driveby_pred" in df else 0)
            if flood_branch:
                fa = df.flood_action
                nmaj.append(int(df.major_flood.sum()))
                nrestr.append(int((fa == "restrict_operations").sum()))
                nintr.append(int((fa == "interrupt").sum()))
                nfinsp.append(int((fa == "inspect").sum()))
                nesc.append(int(df.escalated.sum()))
        row = dict(
            planner=name,
            lifecycle_eur=float(np.mean(costs)),
            lifecycle_eur_std=float(np.std(costs)),
            # Tail cost across seeds: where risk aversion is meant to pay off. A
            # risk-averse policy should trade a higher mean for a lower CVaR/P90.
            lifecycle_eur_cvar10=_tail_mean(costs, 0.10),
            lifecycle_eur_p90=float(np.percentile(costs, 90)),
            n_inspect=float(np.mean(ninsp)),
            n_repair=float(np.mean(nrep)),
            mean_final_pct=float(np.mean(finals)),
        )
        if sensor_health:
            row["n_sensor_replace"] = float(np.mean(nrepl))
            row["n_blind_steps"] = float(np.mean(faultsteps))
        if flood_branch:
            row["n_major_flood"] = float(np.mean(nmaj))
            row["n_restrict"] = float(np.mean(nrestr))
            row["n_flood_inspect"] = float(np.mean(nfinsp))
            row["n_interrupt"] = float(np.mean(nintr))
            row["n_probe_escalated"] = float(np.mean(nesc))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lifecycle_eur").reset_index(drop=True)


def _build_health_classifier(extra, fallback_dirs, champion_like, n):
    """Load single-DOF fallback champions and wrap them with the champion into a
    HealthAwareClassifier (library/live sensor-health mode)."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "TTBI_2D"))
    from digital_twin.assets import DigitalAsset
    from digital_twin.config import DTConfig

    if not fallback_dirs:
        raise ValueError("sensor_health needs fallback_dirs={dof: path} in library/live mode")
    fallbacks, fb_like = {}, {}
    for dof, fdir in fallback_dirs.items():
        cfg = DTConfig.from_metadata(str(Path(fdir) / "DT_metadata.json"))
        fallbacks[int(dof)] = DigitalAsset(str(Path(fdir) / "DT_champion_weights.pth"),
                                           str(Path(fdir) / "DT_metadata.json"),
                                           str(Path(fdir) / "DT_scaler.pkl"), cfg)
        fb_like[int(dof)] = row_normalise(np.load(Path(fdir) / "DT_conf_matrix.npy").astype(float))
    return HealthAwareClassifier(extra["digital"], champion_like, fallbacks, fb_like)


def _tail_mean(costs, alpha: float) -> float:
    """Empirical CVaR_α: mean of the worst (highest) α-fraction of `costs`."""
    c = np.sort(np.asarray(costs, dtype=float))[::-1]
    k = max(1, int(np.ceil(alpha * len(c))))
    return float(c[:k].mean())


# --------------------------------------------------------------------------- #
#  Risk-attitude sweep (risk perception vs Value-of-SHM)
# --------------------------------------------------------------------------- #
# Default sweep grids per attitude. CVaR: α from 1.0 (neutral) down to a sharp
# 5 % tail (more risk-averse). Entropic: θ from 0 (neutral) up (more averse).
RISK_GRIDS = {
    "cvar":     [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
    "entropic": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
}


def sweep_risk(
    planner: str = "pomdp",
    attitude: str = "cvar",
    levels=None,
    **run_kwargs,
) -> pd.DataFrame:
    """Sweep the risk-aversion knob for one belief-based planner.

    For each risk level it runs the full Monte-Carlo comparison and pulls out the
    selected planner's row, so the result table shows how mean € life-cycle cost,
    tail (CVaR/P90) cost, and the inspection/repair rates move as the decision
    maker becomes more risk-averse — the risk-perception-vs-Value-of-SHM curve.

    Args:
        planner:   belief-based planner to sweep ('pomdp' or 'hybrid'); other
                   planners ignore risk, so the curve would be flat.
        attitude:  'cvar' or 'entropic'.
        levels:    risk levels to sweep; default RISK_GRIDS[attitude].
        run_kwargs: forwarded to run_comparison (n_steps, n_seeds, p_advance, ...).

    Returns:
        DataFrame, one row per risk level, sorted from neutral to most averse.
    """
    if attitude not in RISK_GRIDS:
        raise ValueError(f"sweep attitude must be one of {list(RISK_GRIDS)}")
    levels = RISK_GRIDS[attitude] if levels is None else list(levels)
    run_kwargs.pop("planner_names", None)            # we fix it to `planner`

    out = []
    for lvl in levels:
        df = run_comparison(planner_names=(planner,), risk_attitude=attitude,
                            risk_aversion=lvl, **run_kwargs)
        r = df.iloc[0].to_dict()
        rm = RiskModel(attitude=attitude, level=lvl)
        r["risk_attitude"] = attitude
        r["risk_level"] = lvl
        r["risk_label"] = rm.describe()
        out.append(r)
    cols = (["risk_attitude", "risk_level", "risk_label", "planner",
             "lifecycle_eur", "lifecycle_eur_cvar10", "lifecycle_eur_p90",
             "n_inspect", "n_repair", "mean_final_pct"])
    return pd.DataFrame(out)[cols]


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

    # extra: handles the sensor-health layer needs (champion + signal source).
    # Live mode samples signals on the fly, so 'lib' is None (health mode falls
    # back to live observation only if a library is provided).
    return L_drive, L_insp, make_phys, make_obs, {"digital": digital, "lib": None}


def _library_setup(champion_dir, library_dir, n, dt_years, enable_shock,
                   inspect_sigma, n_passages, discretization):
    """Wire the champion classifier + a held-out SignalLibrary (no live TTBI).

    Drive-by observations are real signals sampled from the held-out library and
    classified by the champion — the production path, free of train/serve skew.
    """
    import sys
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "TTBI_2D"))
    from digital_twin.assets import DigitalAsset
    from digital_twin.config import DTConfig
    from digital_twin.signal_library import SignalLibrary, library_observe

    cdir = Path(champion_dir)
    L_drive = row_normalise(np.load(cdir / "DT_conf_matrix.npy").astype(float))
    L_insp = banded_like(n, inspect_sigma)

    config = DTConfig.from_metadata(str(cdir / "DT_metadata.json"),
                                    monitoring_interval=dt_years)
    config.enable_shock = enable_shock
    digital = DigitalAsset(str(cdir / "DT_champion_weights.pth"),
                           str(cdir / "DT_metadata.json"),
                           str(cdir / "DT_scaler.pkl"), config)
    # n_states=None -> load EVERY NNNN.mat (the dataset may have a far finer scour
    # grid than the 61 classifier classes, e.g. 600 states; true damage is read
    # per-file from data.Dano). n_passages caps memory.
    lib = SignalLibrary.from_mat_folder(library_dir, n_passages=n_passages)
    obs_rng = np.random.default_rng(123)

    def make_phys(seed):
        return MockPhysical(dt_years, enable_shock, seed, discretization, n)

    def make_obs(L):
        if L is L_drive:
            return library_observe(lib, digital, obs_rng)
        return lambda phys: int(np.random.choice(n, p=L[phys.get_true_mapped_label()]))

    # extra: champion + the SignalLibrary, reused by the sensor-health layer.
    return L_drive, L_insp, make_phys, make_obs, {"digital": digital, "lib": lib}
