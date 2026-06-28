"""
digital_twin/costs.py
=====================
Monetary cost model for the drive-by digital twin (Value-of-SHM analysis).

Every decision methodology (threshold, cost value-iteration, active inference)
and every monitoring strategy (drive-by, periodic inspection, …) shares this
single CostModel, so their expected life-cycle costs are directly comparable.

Verified figures (Kamariotis et al. 2024, "Quantifying the value of SHM"):
    c_failure      = 5e7  €   (structural failure)
    c_inspect      = 2e4  €   (scour visual inspection)
    c_repair       = 6e5  €   (major scour repair)
    c_downtime_day = 1.5e5 €/day   (bridge close-down; UK railway scour ≤1.65e5)
    discount_rate  = 0.02
    horizon        = 50 yr
See memory: damage-evolution-and-costs.

Action semantics (see also the Torzoni 2024 reward, which we reproduce in
dimensionless form for cross-checking):
    do_nothing — free; the structure keeps deteriorating.
    inspect    — pay c_inspect for a sharp (low-noise) observation; the
                 structural state is unchanged. Only valuable through the
                 information it provides (POMDP / heuristic schedule).
    repair     — pay c_repair (+ close-down during the works); perfect reset
                 to the healthy state.

Flood-branch actions (Step 2; see digital_twin.flood / planner.FloodResponsePlanner):
    restrict_operations — a reduced-risk PROBE: run a few low-mass / low-speed
                 instrumented passages under reduced service. Costs a small
                 reduced-capacity penalty (a fraction of a close-down day) and
                 yields a WIDER (lower-SNR) drive-by observation. The middle
                 option between full drive-by and full closure.
    interrupt  — precautionary CLOSURE: stop traffic now (so this step carries no
                 failure-under-load risk) and dispatch a sharp inspection. Costs
                 the inspection plus a short close-down pending it. Chosen when
                 the train must not be risked (near-threshold belief + severe
                 flood) or when the probe sensors are untrustworthy.

Failure risk is charged as an expected cost  P_fail(damage) · (c_failure +
close-down) every step, so a do-nothing policy on a deteriorating bridge
accrues rising risk cost until repair becomes worthwhile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CostModel:
    # --- verified € figures (Kamariotis 2024) ---
    c_failure:      float = 5.0e7
    c_inspect:      float = 2.0e4
    c_repair:       float = 6.0e5
    c_downtime_day: float = 1.5e5
    discount_rate:  float = 0.02

    # --- close-down durations [days] (placeholders — set from project data) ---
    downtime_days_repair:  float = 30.0
    downtime_days_failure: float = 365.0
    # Flood-branch close-downs (full-closure-equivalent days): a restrict-ops
    # probe just runs a couple of slow/light passages → a SMALL reduced-capacity
    # penalty (≈3e3 €, far below an inspection so the cheap probe has a real
    # niche); an interrupt closes the line ~2 days pending the survey (≈3e5 €).
    # Placeholders — set from project data.
    downtime_days_restrict:  float = 0.02
    downtime_days_interrupt: float = 2.0

    # --- fragility P(failure | damage_fraction): LOGNORMAL CDF, configurable ---
    # damage_fraction = damage% / max_damage%  (0 healthy → 1 fully scoured).
    # The scour-fragility literature uses a lognormal CDF (not logistic): justified
    # by the non-negativity of scour and multiplicative uncertainty. P_fail =
    # Φ((ln s − ln θ)/β), with θ = median capacity (damage fraction at P_fail=0.5)
    # and β = lognormal dispersion. Reported as a SWEPT assumption — bracket β by
    # the literature: ≈0.05 brittle (masonry arch) … 0.3–0.4 deep piers. (Refs:
    # Maroni/Tubaldi; Lamb; "scour fragility for masonry arch bridges"; E3S 2025.)
    fragility_median: float = 0.80   # θ: damage fraction at which P_fail = 0.5
    fragility_beta:   float = 0.30   # β: lognormal dispersion (smaller = sharper)

    # action -> base monetary cost [€]; close-down added separately
    _action_cost: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._action_cost:
            self._action_cost = {
                "do_nothing":     0.0,
                "inspect":        self.c_inspect,
                "repair":         self.c_repair + self.c_downtime_day * self.downtime_days_repair,
                "perfect_repair": self.c_repair + self.c_downtime_day * self.downtime_days_repair,
                # Flood-branch actions (Step 2). Ordering by default figures:
                # do_nothing(0) < restrict(3e3) < inspect(2e4) < interrupt(3.2e5)
                # < repair(5.1e6) — so the planner escalates only as risk warrants.
                "restrict_operations": self.c_downtime_day * self.downtime_days_restrict,
                "interrupt":           self.c_inspect
                                       + self.c_downtime_day * self.downtime_days_interrupt,
            }

    # ── Building blocks ─────────────────────────────────────────────────────────

    def action_cost(self, action: str) -> float:
        """Direct monetary cost of an action [€] (repair includes close-down)."""
        return float(self._action_cost.get(action, 0.0))

    def failure_prob(self, damage_fraction: float) -> float:
        """P(failure | damage) via a lognormal fragility CDF in [0, 1].

        P_fail = Φ((ln s − ln θ)/β); s=0 (healthy) → 0. θ=fragility_median,
        β=fragility_beta. Φ via the error function (no scipy dependency).
        """
        s = float(damage_fraction)
        if s <= 0.0:
            return 0.0
        z = (math.log(s) - math.log(self.fragility_median)) / self.fragility_beta
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    def expected_failure_cost(self, damage_fraction: float) -> float:
        """Expected risk cost charged for being in a damage state [€]."""
        p = self.failure_prob(damage_fraction)
        return p * (self.c_failure + self.c_downtime_day * self.downtime_days_failure)

    def step_cost(self, damage_fraction: float, action: str) -> float:
        """Total expected cost of one step: action + failure risk [€]."""
        return self.action_cost(action) + self.expected_failure_cost(damage_fraction)

    def discount(self, year: float) -> float:
        """Discount factor 1/(1+r)^year."""
        return 1.0 / (1.0 + self.discount_rate) ** year

    # ── Trajectory accounting ───────────────────────────────────────────────────

    def lifecycle_cost(self, damage_fractions, actions, dt: float = 1.0) -> float:
        """Discounted total cost of a simulated trajectory [€].

        damage_fractions : per-step damage fraction in [0,1] (true state).
        actions          : per-step action label.
        dt               : years per step (for the discount clock).
        """
        total = 0.0
        for k, (d, a) in enumerate(zip(damage_fractions, actions)):
            total += self.discount(k * dt) * self.step_cost(float(d), a)
        return float(total)
