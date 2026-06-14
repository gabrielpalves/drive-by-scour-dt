"""
digital_twin/scour_multi.py
===========================
Correlated multi-foundation scour evolution (Kamariotis gradual + flood shock)
with a Gaussian-copula dependence structure across supports.

Why a copula
------------
Foundations on the same river do not scour independently. Two physical sources
of dependence (per the project notes / Adnan 2026):

  1. River-flow (global): every pier sees the same flood/flow forcing, so their
     scour parameters are strongly — but not perfectly — correlated (the flow is
     not exactly uniform across the whole section).
  2. Proximity (local): closely-spaced piers interact hydraulically, so their
     scour is extra-correlated; far-apart piers are independent in this term.

We encode both in a single correlation matrix and sample the Kamariotis
marginals (A, B, the inter-annual noise, and the flood-jump magnitudes) through
a Gaussian copula with that correlation. Floods are a *shared* arrival process
(one river → one Poisson clock); only the per-pier jump magnitudes are
correlated-but-not-identical.

This model is self-contained (NumPy only) so it is testable without the torch /
TTBI stack. It produces a per-support scour-rate vector to feed straight into
the TTBI engine via Damage.scour_rates (see TTBI_2D/damage_config).

Single-support note: with one support it reduces to the scalar ScourModel.
Marginal constants mirror digital_twin.assets.ScourModel — keep in sync.
"""

from __future__ import annotations

import numpy as np


class MultiScourModel:
    DAMAGE_MAX: float = 60.0   # cap on scour damage [%], per support

    # --- Kamariotis marginals (mirror ScourModel) ---
    _A_MEAN: float = 1.94e-4;  _A_COV: float = 0.40
    _B_MEAN: float = 2.0;      _B_COV: float = 0.10
    _OMEGA_MEAN: float = -0.005; _OMEGA_COV: float = 0.10
    # --- shared flood (Compound Poisson) ---
    _LAMBDA_FLOOD: float = 0.10   # flood arrivals per year (one river clock)
    _JUMP_MEAN:    float = 5.0    # mean scour increment per flood [%]
    _JUMP_COV:     float = 0.60

    def __init__(
        self,
        support_locs_m,
        rho_river:    float = 0.70,
        corr_length:  float = 20.0,
        enable_shock: bool  = False,
        rng:          np.random.Generator | None = None,
    ):
        """
        support_locs_m : support positions in METRES (e.g. [0, 20, 40]).
        rho_river      : baseline (global) correlation shared by all supports.
        corr_length    : proximity decay length [m]; closer supports -> more
                         correlated via exp(-d_ij / corr_length).
        enable_shock   : include the flood (Compound Poisson) term.
        """
        self.x = np.atleast_1d(np.asarray(support_locs_m, dtype=float))
        self.n = len(self.x)
        self.rho_river = float(rho_river)
        self.corr_length = float(corr_length)
        self.enable_shock = enable_shock
        self.rng = rng if rng is not None else np.random.default_rng()

        self.R = self._corr_matrix()
        self._L = np.linalg.cholesky(self.R)   # for correlated sampling

        self.time = 0.0
        self.current_X = np.zeros(self.n)      # scour [%] per support
        self.last_flood_count = 0
        self.last_shock = np.zeros(self.n)
        self._sample_parameters()

    # ── correlation structure ──────────────────────────────────────────────────

    def _corr_matrix(self) -> np.ndarray:
        """R_ij = rho_river + (1-rho_river)·exp(-d_ij/corr_length); diag = 1.

        Convex combination of a rank-1 PSD (all-ones) and an exponential-decay
        kernel (PSD) ⇒ a valid correlation matrix.
        """
        d = np.abs(self.x[:, None] - self.x[None, :])
        R = self.rho_river + (1.0 - self.rho_river) * np.exp(-d / self.corr_length)
        np.fill_diagonal(R, 1.0)
        return R

    def _correlated_normals(self) -> np.ndarray:
        """One draw of N(0, R) standard normals across supports."""
        return self._L @ self.rng.standard_normal(self.n)

    # ── parameter sampling (Gaussian copula on the marginals) ───────────────────

    def _sample_parameters(self) -> None:
        a_sigma = np.sqrt(np.log(self._A_COV ** 2 + 1.0))
        a_mu    = np.log(self._A_MEAN) - 0.5 * a_sigma ** 2
        self._A = np.exp(a_mu + a_sigma * self._correlated_normals())          # lognormal
        self._B = self._B_MEAN + (self._B_MEAN * self._B_COV) * self._correlated_normals()  # normal

    # ── evolution ───────────────────────────────────────────────────────────────

    def evolve(self, delta_t: float) -> np.ndarray:
        """Advance all supports by delta_t years; return scour fractions [0,1]."""
        t_mid = self.time + delta_t / 2.0
        gradual_rate = self._A * self._B * (t_mid ** (self._B - 1.0))
        omega_sigma  = abs(self._OMEGA_MEAN * self._OMEGA_COV)
        omega = self._OMEGA_MEAN + omega_sigma * self._correlated_normals()    # correlated weather
        self.current_X += gradual_rate * delta_t * np.exp(omega)

        self.last_flood_count = 0
        self.last_shock = np.zeros(self.n)
        if self.enable_shock:
            n_floods = int(self.rng.poisson(self._LAMBDA_FLOOD * delta_t))     # shared arrival
            if n_floods > 0:
                j_sigma = np.sqrt(np.log(self._JUMP_COV ** 2 + 1.0))
                j_mu    = np.log(self._JUMP_MEAN) - 0.5 * j_sigma ** 2
                shock = np.zeros(self.n)
                for _ in range(n_floods):
                    shock += np.exp(j_mu + j_sigma * self._correlated_normals())  # correlated magnitudes
                self.current_X += shock
                self.last_flood_count = n_floods
                self.last_shock = shock

        self.time += delta_t
        return self.get_scour_fractions()

    def flood_occurred_last_step(self) -> bool:
        return self.last_flood_count > 0

    def get_scour_fractions(self) -> np.ndarray:
        """Per-support scour fraction in [0, 1] (= damage% / DAMAGE_MAX, capped)."""
        return np.clip(self.current_X, 0.0, self.DAMAGE_MAX) / self.DAMAGE_MAX

    def get_scour_rates(self) -> np.ndarray:
        """Alias for the TTBI Damage.scour_rates vector (vertical-stiffness loss)."""
        return self.get_scour_fractions()

    def repair(self, support: int | None = None) -> None:
        """Reset one support (index) or all supports to healthy; resample params."""
        if support is None:
            self.current_X[:] = 0.0
            self.time = 0.0
            self._sample_parameters()
        else:
            self.current_X[support] = 0.0
