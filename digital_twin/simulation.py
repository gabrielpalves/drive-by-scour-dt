"""
digital_twin/simulation.py
==========================
Toggleable drive-by Digital-Twin simulation loop for the Value-of-SHM study.

This is the assembly layer that ties together the pieces built earlier:
    ScourModel / MultiScourModel  — true deterioration (gradual + flood shock)
    an observation source         — drive-by (live TTBI→classifier) or inspection
    a transparent Bayes filter    — the belief over damage state
    a planner                     — threshold / heuristic / cost-VI / POMDP
    CostModel                     — € accounting -> discounted life-cycle cost

Design goals
------------
* "DBN does belief filtering, the planner decides": the belief update here is a
  plain discrete Bayes filter (predict with the action transition, correct with
  the observation likelihood) — functionally the same posterior the pgmpy DBN
  produces, but transparent and able to switch observation models and action
  sets. The pgmpy Graph remains available for Torzoni-style replication.
* Observation source is INJECTED (`driveby_observe`, `inspect_observe`): today
  these wrap live TTBI + the champion classifier; later, swap to sampling a
  held-out signal library with no change to this file.
* `inspect` is a real action: choosing it triggers a sharp observation that
  re-sharpens the belief within the step, then a follow-up commit
  (do_nothing / repair). This is where inspection's value is realised.

Monitoring strategy and planner are independent toggles, so a single run config
selects e.g. (monitoring='drive_by', planner=POMDPPlanner) and the harness can
compare lifecycle € across combinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def row_normalise(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return M / np.clip(M.sum(axis=1, keepdims=True), 1e-12, None)


class DTSimulator:
    """
    Args:
        physical:        object with update_physical_state(action),
                         state_continuous (true damage %), get_true_mapped_label(),
                         and flood_this_step (bool).
        planner:         object with decide(belief, context) -> action label.
        cost_model:      digital_twin.costs.CostModel.
        states_list:     ['0', …, 'N-1'] (label 0 = healthy).
        actions_list:    must include 'do_nothing', 'repair' (or 'perfect_repair');
                         'inspect' optional.
        driveby_observe: callable(physical) -> predicted label int (live TTBI →
                         classifier, dataset sample, or a mock).
        driveby_like:    (N, N) observation likelihood L[true, pred] for drive-by
                         (row-normalised confusion matrix of the classifier).
        inspect_observe: callable(physical) -> predicted label for an inspection
                         (sharp). Optional; required only if 'inspect' is used.
        inspect_like:    (N, N) likelihood for the inspection observation (sharp).
        p_advance:       per-step deterioration prob (transition prior for the filter).
        dt_years:        years per step (discount clock).
        monitoring:      label for bookkeeping ('drive_by', 'periodic', ...).
    """

    def __init__(
        self,
        physical,
        planner,
        cost_model,
        states_list,
        actions_list,
        driveby_observe,
        driveby_like,
        inspect_observe=None,
        inspect_like=None,
        p_advance: float = 0.10,
        dt_years: float = 1.0,
        discretization: float = 5.0,
        max_damage: float = 60.0,
        monitoring: str = "drive_by",
    ):
        self.phys = physical
        self.planner = planner
        self.cm = cost_model
        self.states = states_list
        self.actions = actions_list
        self.n = len(states_list)
        self.driveby_observe = driveby_observe
        self.inspect_observe = inspect_observe
        self.L_drive = row_normalise(driveby_like)
        self.L_inspect = None if inspect_like is None else row_normalise(inspect_like)
        self.p_advance = p_advance
        self.dt = dt_years
        self.disc = discretization
        self.max_damage = max_damage
        self.monitoring = monitoring
        self._repair = next(a for a in ("repair", "perfect_repair") if a in actions_list)

        self.belief = np.full(self.n, 1.0 / self.n)   # uninformative prior
        self.last_action = "do_nothing"
        self.t = 0
        self.history: list[dict] = []

    # ── Bayes filter ────────────────────────────────────────────────────────────

    def _transition(self, action: str) -> np.ndarray:
        T = np.zeros((self.n, self.n))
        if action == self._repair:
            T[:, 0] = 1.0                                  # repair -> healthy
        else:                                              # do_nothing / inspect
            for s in range(self.n):
                if s == self.n - 1:
                    T[s, s] = 1.0
                else:
                    T[s, s] = 1.0 - self.p_advance
                    T[s, s + 1] = self.p_advance
        return T

    def _predict(self, action: str) -> None:
        self.belief = self.belief @ self._transition(action)

    def _update(self, like: np.ndarray, pred: int) -> None:
        b = like[:, pred] * self.belief
        s = b.sum()
        self.belief = b / s if s > 0 else self.belief

    # ── cost ────────────────────────────────────────────────────────────────────

    def _true_frac(self) -> float:
        return min(self.phys.state_continuous, self.max_damage) / self.max_damage

    # ── one step ────────────────────────────────────────────────────────────────

    def step(self) -> None:
        # 1. physical evolves under the action chosen last step
        self.phys.update_physical_state(self.last_action)
        flood = bool(getattr(self.phys, "flood_this_step", False))

        # 2. belief predict under that action's transition
        self._predict(self.last_action)

        # 3. drive-by observation + correction
        pred = int(self.driveby_observe(self.phys))
        self._update(self.L_drive, pred)

        # 4. decide
        ctx = {"flood": flood, "step": self.t}
        action = self.planner.decide(self.belief, ctx)
        step_cost = self.cm.action_cost(action)

        # 5. inspect: sharp observation re-sharpens belief, then commit
        inspected = False
        if action == "inspect" and self.inspect_observe is not None and self.L_inspect is not None:
            inspected = True
            ipred = int(self.inspect_observe(self.phys))
            self._update(self.L_inspect, ipred)
            action = self.planner.decide(self.belief, {**ctx, "post_inspect": True})
            if action == "inspect":      # never inspect twice in one step
                action = "do_nothing"
            step_cost += self.cm.action_cost(action)

        # 6. realised risk cost for the current TRUE state. expected_failure_cost
        #    is a per-YEAR hazard, so scale by dt → invariant to step size (a
        #    monthly clock must not inflate risk 12x vs a yearly one). Action
        #    costs are per-event and are NOT scaled.
        risk = self.cm.expected_failure_cost(self._true_frac()) * self.dt
        total = (step_cost + risk) * self.cm.discount(self.t * self.dt)

        # 7. record + carry action forward
        self.history.append(dict(
            t=self.t,
            true_pct=self.phys.state_continuous,
            true_label=self.phys.get_true_mapped_label(),
            belief_label=int(np.argmax(self.belief)),
            driveby_pred=pred,
            flood=flood,
            inspected=inspected,
            action=action,
            step_cost_eur=step_cost,
            risk_cost_eur=risk,
            discounted_eur=total,
        ))
        self.last_action = action
        self.t += 1

    def run(self, n_steps: int) -> pd.DataFrame:
        for _ in range(n_steps):
            self.step()
        return pd.DataFrame(self.history)

    def lifecycle_cost(self) -> float:
        return float(sum(h["discounted_eur"] for h in self.history))
