"""
digital_twin/flood.py
======================
The flood-decision layer of the drive-by Digital Twin (Paper 2, Step 2).

Scour is flood-driven (Kamariotis Compound-Poisson shock, digital_twin.assets /
scour_multi). When a flood occurs the asset owner does NOT yet know how much
scour it added; they only have a noisy hydrological reading from a river-stage
gauge. This module turns that reading into a decision input:

    1. FloodTrigger  — the major-vs-minor classifier. A river gauge measures
       stage precisely but DISCHARGE is inferred through a stage-discharge
       rating curve whose error explodes (50-200 %) exactly during the extreme
       floods that matter (rating-curve extrapolation + hysteresis). So "is this
       a major flood?" is itself uncertain. We model it as a hard design-flood
       threshold applied to a *noisy* gauge reading, which marginalises to a
       SMOOTH probabilistic trigger  P(classify major | true severity) — a
       lognormal CDF. The hard-threshold special case is the gauge_beta -> 0
       limit. This probabilistic trigger is the object we refine with the Todd
       group (FHWA HEC-18 / AREMA design-flood thresholds; Maroni et al. gauge-
       linked Plan-of-Action triggers; the rating-curve uncertainty literature).

    2. widen_belief — an observed major flood injects an uncertain scour shock,
       so the belief over damage must be widened (shifted up AND spread) before
       any maintenance decision. The widening kernel is parameterised by the
       observed severity: a bigger reading shifts more mass higher and fattens
       the dangerous tail, which is what makes the downstream planner escalate
       from a cheap probe to a full inspection.

Severity convention
-------------------
Severity is dimensionless: severity = (scour jump this flood) / (mean jump),
so severity ~ 1 is a typical flood and the upper tail (severity >~ 1.5) are the
hydrologically significant events. The major/minor boundary `severity_major`
sits on this scale (the design-flood return period mapped to severity); it and
`gauge_beta` are documented placeholders, swept and refined with the Todd group.

NumPy only — no torch / TTBI — so the trigger and the decision map are testable
without the heavy stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf, otypes=[float])   # math.erf is scalar; otypes keeps size-0 safe


def _lognormal_cdf(x: np.ndarray, median: float, beta: float) -> np.ndarray:
    """Φ((ln x − ln median)/β), vectorised; x<=0 -> 0. No scipy dependency."""
    x = np.asarray(x, dtype=float)
    if median <= 0.0 or beta <= 0.0:
        # Degenerate: a hard step at `median` (the β->0 deterministic threshold).
        return (x >= median).astype(float)
    pos = x > 0.0
    # Evaluate erf over the whole array (log fed a safe 1.0 where x<=0), then mask.
    z = (np.log(np.where(pos, x, 1.0)) - math.log(median)) / beta
    return np.where(pos, 0.5 * (1.0 + _erf(z / _SQRT2)), 0.0)


@dataclass
class FloodTrigger:
    """Probabilistic major-flood classifier from a noisy river gauge.

    The operator declares a *major* flood when the gauge-inferred discharge
    crosses a design-flood threshold (`severity_major`). Because the discharge
    inference is uncertain (rating-curve extrapolation), the gauge reading is a
    noisy multiple of the true severity: gauge = severity · ε, with ε lognormal
    of dispersion `gauge_beta`. Marginalising over ε turns the hard threshold
    into the smooth trigger probability

        P(classify major | severity) = Φ((ln severity − ln severity_major)/gauge_beta)

    so `gauge_beta` is literally the discharge-estimation uncertainty, and
    `gauge_beta -> 0` recovers the deterministic design-flood threshold.

    Args:
        severity_major: severity at the major/minor boundary (P(major)=0.5).
                        ~1.5 ⇒ roughly the upper third of floods are "major".
        gauge_beta:     lognormal dispersion of the discharge reading (≈0.5-0.7
                        reflects the 50-200 % extreme-flood rating error).
    """

    severity_major: float = 1.5
    gauge_beta:     float = 0.6

    def p_major(self, severity) -> np.ndarray | float:
        """Marginal probability the event is classified major (the trigger CDF)."""
        p = _lognormal_cdf(np.asarray(severity, dtype=float),
                           self.severity_major, self.gauge_beta)
        return float(p) if np.ndim(severity) == 0 else p

    def observe_gauge(self, severity_true: float, rng: np.random.Generator) -> float:
        """Noisy gauge reading of the true severity (gauge = severity · lognormal).

        The operator acts on THIS value, not the (latent) true severity. Its
        dispersion is `gauge_beta`, so classify(observe_gauge(s)) reproduces
        p_major(s) in expectation.
        """
        if severity_true <= 0.0:
            return 0.0
        # Median-unbiased multiplicative error (the rating curve is the median
        # stage->discharge relation): median(gauge)=severity, so a hard threshold
        # on the gauge marginalises EXACTLY to p_major(severity).
        eps = rng.lognormal(mean=0.0, sigma=self.gauge_beta)
        return float(severity_true * eps)

    def classify(self, gauge: float) -> bool:
        """Hard design-flood threshold on the observed gauge reading."""
        return bool(gauge >= self.severity_major)


# --------------------------------------------------------------------------- #
#  Belief widening on an observed major flood
# --------------------------------------------------------------------------- #
def flood_transition(
    n: int,
    severity: float,
    discretization: float,
    jump_pct_mean: float = 5.0,
    jump_beta: float = 0.7,
) -> np.ndarray:
    """Upward-diffusion kernel W[s, s'] for a flood of the given (observed) severity.

    The scour added by the flood is uncertain: a lognormal increment in % whose
    median scales with the observed severity (median ≈ severity · jump_pct_mean)
    and whose dispersion `jump_beta` folds in both the natural jump variability
    (Kamariotis JUMP_COV) and the gauge/discharge uncertainty. Discretising that
    increment into label steps gives a per-state transition that both SHIFTS the
    belief up and SPREADS it — the "shock of uncertain size" the planner reacts
    to. severity<=0 returns the identity (no widening).

    Returns:
        (n, n) row-stochastic matrix; widened_belief = belief @ W.
    """
    W = np.zeros((n, n))
    if severity <= 0.0 or jump_pct_mean <= 0.0:
        np.fill_diagonal(W, 1.0)
        return W

    median_pct = float(severity) * float(jump_pct_mean)
    # pmf over label-increments k = 0,1,... by integrating the lognormal increment
    # over each label-width bin [k·disc, (k+1)·disc) %; the final bin absorbs the
    # upper tail so the row stays normalised.
    edges = np.arange(n) * float(discretization)          # lower edge of each k-bin, %
    cdf = _lognormal_cdf(edges, median_pct, jump_beta)     # F(lower edge)
    cdf = np.append(cdf, 1.0)                              # tail -> last bin
    pmf_k = np.diff(cdf)                                   # P(increment in bin k)
    pmf_k = np.clip(pmf_k, 0.0, None)
    if pmf_k.sum() <= 0:
        np.fill_diagonal(W, 1.0)
        return W
    pmf_k /= pmf_k.sum()

    for s in range(n):
        for k in range(n):
            if pmf_k[k] <= 0:
                continue
            W[s, min(s + k, n - 1)] += pmf_k[k]
    return W


def widen_belief(
    belief: np.ndarray,
    severity: float,
    discretization: float,
    jump_pct_mean: float = 5.0,
    jump_beta: float = 0.7,
) -> np.ndarray:
    """Belief after an observed major flood of `severity` (belief @ flood_transition)."""
    belief = np.asarray(belief, dtype=float)
    W = flood_transition(len(belief), severity, discretization, jump_pct_mean, jump_beta)
    out = belief @ W
    s = out.sum()
    return out / s if s > 0 else belief
