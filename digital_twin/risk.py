"""
digital_twin/risk.py
====================
Risk-perception layer for the drive-by Digital Twin.

The cost-based planners (POMDP, Hybrid) score a decision by the EXPECTED cost it
incurs under the current belief. A risk-NEUTRAL decision maker minimises that
expectation. A risk-AVERSE decision maker — the realistic asset owner facing a
small chance of a catastrophic scour failure — weights the dangerous tail of the
distribution more heavily, and so inspects and repairs more conservatively
(Chadha et al. 2023; Ames et al. 2025; Todd group, UCSD).

`RiskModel` is that one knob. It maps a cost distribution (per-outcome costs `c`
with probabilities `p`, e.g. the failure-risk cost across a belief over damage
states) to a single PERCEIVED cost via a coherent risk measure:

    attitude='neutral'  : E[c]              = Σ pᵢ cᵢ              (risk-neutral)
    attitude='cvar'     : CVaR_α[c]         = mean of the worst α-tail of c
    attitude='entropic' : (1/λ) log E[e^{λc}]  (exponential-utility CE)

All three coincide with the mean in the risk-neutral limit (α→1, λ→0), so the
default attitude='neutral' reproduces the existing planners exactly: the risk
layer is a strict, opt-in extension.

`level` is the single risk-aversion knob, with an attitude-specific meaning:
    cvar     : level = α ∈ (0, 1]   tail fraction; 1.0 neutral → 0.05 worst-5 %
                                    (SMALLER α = MORE risk-averse).
    entropic : level = θ ≥ 0        dimensionless aversion; 0 neutral, larger =
                                    more risk-averse. Internally λ = θ/cost_scale
                                    so θ is comparable regardless of the € scale.

This keeps risk perception orthogonal to the planner: any planner that scores a
belief through `RiskModel.perceived_cost` becomes risk-sensitive with no further
change. Sweeping `level` against life-cycle € / inspection rate / tail cost is
the risk-attitude-vs-Value-of-SHM experiment (digital_twin.harness.sweep_risk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskModel:
    """Coherent risk measure used to turn a cost distribution into a scalar.

    Args:
        attitude:   'neutral' | 'cvar' | 'entropic'.
        level:      risk-aversion knob — α for cvar (∈(0,1], smaller = more
                    averse), θ for entropic (≥0, larger = more averse). Ignored
                    when attitude='neutral'.
        cost_scale: € reference magnitude used to make the entropic λ = θ/cost_scale
                    dimensionless and numerically stable (set to a representative
                    catastrophic cost, e.g. c_failure + failure downtime).
    """

    attitude:   str   = "neutral"
    level:      float = 1.0
    cost_scale: float = 1.0e8

    def __post_init__(self) -> None:
        self.attitude = str(self.attitude).lower()
        if self.attitude not in ("neutral", "cvar", "entropic"):
            raise ValueError(f"unknown risk attitude '{self.attitude}' "
                             "(expected neutral | cvar | entropic)")

    @property
    def is_neutral(self) -> bool:
        """True when the model is risk-neutral (so perceived_cost == E[c])."""
        return (self.attitude == "neutral"
                or (self.attitude == "cvar" and self.level >= 1.0)
                or (self.attitude == "entropic" and self.level <= 0.0))

    def describe(self) -> str:
        # ASCII only (printed to the Windows cp1252 console): a=alpha, th=theta.
        if self.is_neutral:
            return "neutral"
        param = "a" if self.attitude == "cvar" else "th"
        return f"{self.attitude}({param}={self.level:g})"

    # ── public interface ─────────────────────────────────────────────────────────

    def perceived_cost(self, costs, probs) -> float:
        """Perceived (risk-adjusted) cost of a cost distribution [€].

        Args:
            costs: per-outcome cost, shape (k,).
            probs: outcome probabilities, shape (k,); renormalised internally.

        Returns:
            Scalar perceived cost. With a risk-averse attitude the tail of
            `costs` is up-weighted, so the result rises above the mean E[c].
        """
        c = np.asarray(costs, dtype=float)
        p = np.asarray(probs, dtype=float)
        tot = p.sum()
        if tot <= 0:
            return 0.0
        p = p / tot
        if self.is_neutral:
            return float(c @ p)
        if self.attitude == "cvar":
            return self._cvar(c, p)
        return self._entropic(c, p)

    # ── risk measures ────────────────────────────────────────────────────────────

    def _cvar(self, c: np.ndarray, p: np.ndarray) -> float:
        """CVaR_α: expected cost over the worst (highest-cost) α-probability tail."""
        alpha = float(self.level)
        order = np.argsort(c)[::-1]                 # worst (highest) cost first
        c_s, p_s = c[order], p[order]
        cum_before = np.cumsum(p_s) - p_s           # mass strictly above each atom
        take = np.clip(alpha - cum_before, 0.0, p_s)  # mass of each atom inside the α-tail
        w = take.sum()
        if w <= 0:                                  # α smaller than the worst atom
            return float(c_s[0])
        return float((c_s * take).sum() / w)

    def _entropic(self, c: np.ndarray, p: np.ndarray) -> float:
        """Exponential-utility certainty equivalent (1/λ) log Σ pᵢ e^{λcᵢ}."""
        lam = float(self.level) / float(self.cost_scale)
        if lam <= 0:
            return float(c @ p)
        # log-sum-exp for stability: ρ = (1/λ) logsumexp(log pᵢ + λcᵢ)
        z = lam * c + np.log(np.clip(p, 1e-300, None))
        m = float(z.max())
        return float((m + np.log(np.sum(np.exp(z - m)))) / lam)
