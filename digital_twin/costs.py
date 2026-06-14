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

Failure risk is charged as an expected cost  P_fail(damage) · (c_failure +
close-down) every step, so a do-nothing policy on a deteriorating bridge
accrues rising risk cost until repair becomes worthwhile.
"""

from __future__ import annotations

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

    # --- fragility P(failure | damage_fraction): logistic, configurable ---
    # damage_fraction = damage% / max_damage%  (0 healthy → 1 fully scoured).
    fragility_midpoint:  float = 0.80   # fraction at which P_fail = 0.5
    fragility_steepness: float = 12.0   # larger = sharper transition

    # action -> base monetary cost [€]; close-down added separately
    _action_cost: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._action_cost:
            self._action_cost = {
                "do_nothing":     0.0,
                "inspect":        self.c_inspect,
                "repair":         self.c_repair + self.c_downtime_day * self.downtime_days_repair,
                "perfect_repair": self.c_repair + self.c_downtime_day * self.downtime_days_repair,
            }

    # ── Building blocks ─────────────────────────────────────────────────────────

    def action_cost(self, action: str) -> float:
        """Direct monetary cost of an action [€] (repair includes close-down)."""
        return float(self._action_cost.get(action, 0.0))

    def failure_prob(self, damage_fraction: float) -> float:
        """P(failure | damage) via a logistic fragility curve in [0, 1]."""
        z = self.fragility_steepness * (damage_fraction - self.fragility_midpoint)
        return float(1.0 / (1.0 + np.exp(-z)))

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
