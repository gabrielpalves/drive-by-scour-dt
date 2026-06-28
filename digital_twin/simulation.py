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

from digital_twin.flood import widen_belief


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
        sensor_health=None,
        flood_response=None,
        flood_trigger=None,
        probe_observe=None,
        probe_like=None,
        n_probe: int = 2,
        probe_reliability: float = 1.0,
        flood_jump_pct_mean: float = 5.0,
        flood_jump_beta: float = 0.7,
        flood_rng=None,
    ):
        self.phys = physical
        self.planner = planner
        self.cm = cost_model
        self.sensor_health = sensor_health
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

        # ── flood-decision branch (Step 2; optional) ──────────────────────────────
        # When a flood_response planner + trigger are supplied AND a sharp
        # inspection source exists, an observed MAJOR flood diverts the step into
        # _major_flood_step (widen belief -> choose probe/inspect/interrupt ->
        # observe -> commit). With flood_response=None the loop is byte-for-byte
        # the original routine path, so the planner comparison is unchanged.
        self.flood_response = flood_response
        self.flood_trigger = flood_trigger
        self.probe_observe = probe_observe
        self.L_probe = None if probe_like is None else row_normalise(probe_like)
        self.n_probe = int(n_probe)
        self.probe_reliability = float(probe_reliability)
        self.flood_jump_pct_mean = float(flood_jump_pct_mean)
        self.flood_jump_beta = float(flood_jump_beta)
        self.frng = flood_rng if flood_rng is not None else np.random.default_rng()
        self._flood_on = (flood_response is not None and flood_trigger is not None
                          and self.inspect_observe is not None and self.L_inspect is not None)

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

    def _sensor_cost(self) -> float:
        return (self.sensor_health.flush_cost()
                if self.sensor_health is not None else 0.0)

    def _sensor_status(self) -> str:
        return (self.sensor_health.status_code()
                if self.sensor_health is not None else "")

    def _record(self, **kw) -> None:
        """Append a history row, filling the full column set with defaults so the
        routine and flood paths produce an aligned DataFrame."""
        row = dict(
            t=self.t, true_pct=self.phys.state_continuous,
            true_label=self.phys.get_true_mapped_label(),
            belief_label=int(np.argmax(self.belief)),
            driveby_pred=None, flood=False, major_flood=False, severity=0.0,
            flood_action="", probe_used=0, escalated=False, inspected=False,
            action="do_nothing", step_cost_eur=0.0, sensor_cost_eur=0.0,
            sensor_status=self._sensor_status(), risk_cost_eur=0.0,
            discounted_eur=0.0,
        )
        row.update(kw)
        self.history.append(row)

    # ── one step ────────────────────────────────────────────────────────────────

    def step(self) -> None:
        # 1. physical evolves under the action chosen last step
        self.phys.update_physical_state(self.last_action)
        flood = bool(getattr(self.phys, "flood_this_step", False))
        severity_true = float(getattr(self.phys, "flood_severity", 0.0))

        # 2. belief predict under that action's transition
        self._predict(self.last_action)

        # 2b. FLOOD BRANCH: an observed MAJOR flood diverts the whole step. The
        #     gauge reads the (latent) severity with rating-curve noise; a hard
        #     design-flood threshold on that noisy reading is the probabilistic
        #     trigger (digital_twin.flood). Minor / unsignalled floods fall
        #     through to the routine drive-by (revealed in later routine data).
        if self._flood_on and flood:
            gauge = self.flood_trigger.observe_gauge(severity_true, self.frng)
            if self.flood_trigger.classify(gauge):
                self._major_flood_step(gauge)
                return

        self._routine_step(flood)

    def _routine_step(self, flood: bool) -> None:
        # drive-by observation + correction. The observation source may return a
        # label (uses L_drive) or a (label, likelihood) pair — the sensor-health
        # layer returns its effective likelihood (champion / fallback) so a
        # degraded reading is distrusted; label=None means no usable observation.
        obs = self.driveby_observe(self.phys)
        pred, like = obs if isinstance(obs, tuple) else (obs, None)
        if pred is not None:
            pred = int(pred)
            self._update(self.L_drive if like is None else row_normalise(like), pred)

        sensor_cost = self._sensor_cost()
        ctx = {"flood": flood, "step": self.t}
        action = self.planner.decide(self.belief, ctx)
        step_cost = self.cm.action_cost(action) + sensor_cost

        # inspect: sharp observation re-sharpens belief, then commit
        inspected = False
        if action == "inspect" and self.inspect_observe is not None and self.L_inspect is not None:
            inspected = True
            ipred = int(self.inspect_observe(self.phys))
            self._update(self.L_inspect, ipred)
            action = self.planner.decide(self.belief, {**ctx, "post_inspect": True})
            if action == "inspect":      # never inspect twice in one step
                action = "do_nothing"
            step_cost += self.cm.action_cost(action)

        # realised risk for the current TRUE state. expected_failure_cost is a
        # per-YEAR hazard, scaled by dt so a monthly clock does not inflate risk
        # 12x vs a yearly one. Action costs are per-event and NOT scaled.
        risk = self.cm.expected_failure_cost(self._true_frac()) * self.dt
        total = (step_cost + risk) * self.cm.discount(self.t * self.dt)

        self._record(driveby_pred=pred, flood=flood, inspected=inspected,
                     action=action, step_cost_eur=step_cost,
                     sensor_cost_eur=sensor_cost, risk_cost_eur=risk,
                     discounted_eur=total)
        self.last_action = action
        self.t += 1

    def _probe_update(self) -> int:
        """Run up to n_probe restricted-ops passages, Bayesian-fusing each usable
        one into the belief. Returns the number of passages that yielded data;
        0 means the probe was fully corrupted (sensor-health) -> escalate."""
        used = 0
        for _ in range(self.n_probe):
            obs = self.probe_observe(self.phys)
            pred, like = obs if isinstance(obs, tuple) else (obs, None)
            if pred is None:
                continue
            L = self.L_probe if like is None else row_normalise(like)
            self._update(L if L is not None else self.L_drive, int(pred))
            used += 1
        return used

    def _major_flood_step(self, gauge: float) -> None:
        # widen the belief: an uncertain scour shock of the OBSERVED severity
        self.belief = widen_belief(self.belief, gauge, self.disc,
                                   self.flood_jump_pct_mean, self.flood_jump_beta)
        sensor_cost = self._sensor_cost()
        ctx = {"flood": True, "major_flood": True, "severity": float(gauge),
               "step": self.t, "probe_reliability": self.probe_reliability}
        faction = self.flood_response.decide(self.belief, ctx)
        cost = self.cm.action_cost(faction) + sensor_cost

        probe_used = 0
        escalated = False
        inspected = False
        if faction == "restrict_operations" and self.probe_observe is not None:
            probe_used = self._probe_update()
            if probe_used == 0:                       # probe corrupted -> escalate
                escalated = True
                inspected = True
                cost += self.cm.action_cost("inspect")
                self._update(self.L_inspect, int(self.inspect_observe(self.phys)))
        elif faction in ("inspect", "interrupt"):
            inspected = True
            self._update(self.L_inspect, int(self.inspect_observe(self.phys)))

        # maintenance commit on the post-information belief (base planner). Only
        # a repair is honoured here; a second inspect is collapsed to do_nothing.
        commit = self.planner.decide(self.belief, {**ctx, "post_flood": True})
        if commit == self._repair:
            cost += self.cm.action_cost(self._repair)
        else:
            commit = "do_nothing"

        # interrupt closes the line this step -> no under-load failure risk.
        interrupted = (faction == "interrupt")
        risk = (0.0 if interrupted
                else self.cm.expected_failure_cost(self._true_frac()) * self.dt)
        total = (cost + risk) * self.cm.discount(self.t * self.dt)

        self._record(driveby_pred=None, flood=True, major_flood=True,
                     severity=float(gauge), flood_action=faction,
                     probe_used=probe_used, escalated=escalated,
                     inspected=inspected, action=commit, step_cost_eur=cost,
                     sensor_cost_eur=sensor_cost, risk_cost_eur=risk,
                     discounted_eur=total)
        self.last_action = commit
        self.t += 1

    def run(self, n_steps: int) -> pd.DataFrame:
        for _ in range(n_steps):
            self.step()
        return pd.DataFrame(self.history)

    def lifecycle_cost(self) -> float:
        return float(sum(h["discounted_eur"] for h in self.history))
