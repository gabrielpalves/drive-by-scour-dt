"""
digital_twin/calibrate.py
=========================
Calibration helpers for the drive-by Digital Twin.

Two free parameters are not directly observable and benefit from calibration:

  1. p_advance — the belief-filter's per-step deterioration prior. We estimate it
     from the TRUE scour model (Monte-Carlo) so it matches the simulated reality,
     removing it as a hand-tuned knob.

  2. the fragility curve P(failure | scour) — genuinely uncertain and the one we
     deliberately keep reliability-light. Rather than pin a single curve, we run
     a SENSITIVITY sweep and report how the cost-optimal repair threshold and the
     planner ranking change, so results can be stated as robust to the fragility
     assumption.

Run:
    .venv/Scripts/python -m digital_twin.calibrate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.costs import CostModel
from digital_twin.planner import CostBenefitPlanner
from digital_twin.scour_multi import MultiScourModel
from digital_twin.harness import run_comparison


def estimate_p_advance(
    dt_years: float = 1.0 / 12.0,
    n_steps: int = 360,
    n_mc: int = 2000,
    discretization: float = 5.0,
    max_damage: float = 60.0,
    enable_shock: bool = True,
    seed0: int = 0,
) -> dict:
    """Empirical per-step class-advance statistics from the true ScourModel.

    Returns p_advance (fraction of steps where the discrete class increased),
    the mean per-step class increment, and the fraction of steps that jumped
    more than one class (driven by floods — these the simple +1 transition
    model under-represents, so treat p_advance as a first-order prior).
    """
    n = int(max_damage / discretization) + 1
    inc_steps = multi = total = 0
    increments = []
    for s in range(n_mc):
        m = MultiScourModel([0.0], enable_shock=enable_shock,
                            rng=np.random.default_rng(seed0 + s))
        prev = 0
        for _ in range(n_steps):
            m.evolve(dt_years)
            lab = int(np.clip(round(m.current_X[0] / discretization), 0, n - 1))
            d = lab - prev
            inc_steps += d > 0
            multi += d > 1
            increments.append(max(d, 0))
            total += 1
            prev = lab
    return dict(
        p_advance=inc_steps / total,
        mean_increment=float(np.mean(increments)),
        frac_multiclass_jumps=multi / total,
    )


def fragility_sensitivity(
    medians=(0.6, 0.7, 0.8, 0.9),
    betas=(0.05, 0.3, 0.4),
    dt_years: float = 1.0 / 12.0,
    n_steps: int = 360,
    n_seeds: int = 12,
    p_advance: float = 0.03,
    discretization: float = 5.0,
    max_damage: float = 60.0,
) -> pd.DataFrame:
    """Sweep the lognormal fragility (median θ × dispersion β); report the
    cost-optimal repair threshold + planner ranking. β grid brackets the
    literature: 0.05 brittle (masonry arch) … 0.3–0.4 deep piers."""
    n = int(max_damage / discretization) + 1
    states = [str(i) for i in range(n)]
    rows = []
    for med in medians:
        for beta in betas:
            cm = CostModel(fragility_median=med, fragility_beta=beta)
            cvi = CostBenefitPlanner(states, ["do_nothing", "repair"], cm,
                                     max_damage=max_damage,
                                     discretization=discretization,
                                     p_advance=p_advance)
            df = run_comparison(
                planner_names=("do_nothing", "cost_vi", "heuristic", "pomdp", "hybrid"),
                n_steps=n_steps, dt_years=dt_years, n_seeds=n_seeds,
                p_advance=p_advance, cost_model=cm, mode="mock",
                discretization=discretization, max_damage=max_damage,
            )
            ranking = " < ".join(df.planner.tolist())
            thr = cvi.threshold_label
            rows.append(dict(
                fragility_median=med,
                fragility_beta=beta,
                repair_threshold_label=thr,
                repair_threshold_pct=None if thr is None else thr * discretization,
                best_planner=df.iloc[0].planner,
                ranking=ranking,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    print("=== p_advance calibration (true ScourModel, monthly, shock on) ===")
    pa = estimate_p_advance()
    for k, v in pa.items():
        print(f"  {k:24s} {v:.4f}")
    print(f"  -> use p_advance ~ {pa['p_advance']:.3f} in the belief filter "
          f"(note {pa['frac_multiclass_jumps']*100:.1f}% of steps are flood multi-class jumps)")

    print("\n=== fragility sensitivity (cost-optimal repair threshold + ranking) ===")
    fs = fragility_sensitivity()
    print(fs.to_string(index=False))
    print("\nRobust if best_planner / ranking are stable across rows.")


if __name__ == "__main__":
    main()
