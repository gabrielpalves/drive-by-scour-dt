"""
digital_twin/planner.py
=======================
Maps a belief distribution over damage states to a maintenance action.

The single class, Planner, implements a deterministic threshold policy:
repair if the MAP damage label is at or above a critical threshold,
do nothing otherwise.  This is the simplest policy that is coherent with
the DBN structure and serves as the baseline for the drive-by digital twin.

The policy is stored as a (n_states, n_actions) matrix so that the DBN
can treat it as a CPD — p(U | D) — and marginalise over the belief
distribution in one matrix multiply rather than a loop.

Swapping policy
---------------
The planner is the natural extension point for more sophisticated decision
rules.  To replace the threshold policy with, say, a POMDP-optimal policy
or a cost-benefit rule, subclass Planner and override compute_policy():

    class CostBenefitPlanner(Planner):
        def compute_policy(self) -> np.ndarray:
            ...  # return (n_states, n_actions) matrix

Graph.__init__ accepts any object with a .policy attribute, so no other
code needs to change.

Imported by:
    drive_by_DT.py — Planner (instantiated once, passed to Graph)
    digital_twin/dbn.py — uses planner.policy as cpd_d_to_u
"""

import numpy as np

from digital_twin.risk import RiskModel


class Planner:
    """
    Deterministic threshold policy for scour damage management.

    Decision rule
    -------------
    Label 0 is healthy (0 % damage); label n_classes-1 is maximum damage.
    Damage worsens as label increases.

    For each state s:
        label(s) >= threshold  →  action = 'perfect_repair'   (too damaged)
        label(s) <  threshold  →  action = 'do_nothing'        (acceptable)

    The threshold is expressed as a damage class label rather than a physical
    percentage so it can be compared directly against classifier outputs
    without unit conversion.

    Args:
        states_list (list[str]):  Ordered state labels, e.g. ['0', …, '12'].
                                  Must match the DBN state ordering.
        actions_list (list[str]): Ordered action labels.
                                  Index 0 must be 'do_nothing',
                                  index 1 must be 'perfect_repair'.
        threshold_label (int):    Damage labels at or above this value trigger
                                  repair.  E.g. threshold_label=6 repairs
                                  states labelled 6–12 (≥ 30 % damage when
                                  discretization=5) and ignores states 0–5.

    Attributes:
        policy (np.ndarray): Shape (n_states, n_actions).  Row s gives the
                             action probability distribution for state s.
                             Deterministic, so each row has exactly one 1.0.
    """

    def __init__(
        self,
        states_list:     list[str],
        actions_list:    list[str],
        threshold_label: int,
    ):
        self.states_list     = states_list
        self.actions_list    = actions_list
        self.threshold_label = threshold_label
        self.policy          = self._compute_policy()

    # ── Public interface ──────────────────────────────────────────────────────

    def decide(self, belief: np.ndarray, context: dict | None = None) -> str:
        """
        Return the MAP action given a belief distribution over states.

        Computes the expected action probability under the belief and
        returns the action with the highest expected probability.  For a
        deterministic policy this is equivalent to acting on the MAP state.

        Args:
            belief (np.ndarray): Shape (n_states,).  Must sum to 1.
            context (dict): Unused here; accepted so all planners share the
                            decide(belief, context) interface (HeuristicPlanner,
                            POMDPPlanner use it).

        Returns:
            str: The chosen action label, e.g. 'do_nothing'.
        """
        action_probs = self.policy.T @ belief
        return self.actions_list[int(np.argmax(action_probs))]

    def action_probs(self, belief: np.ndarray) -> np.ndarray:
        """
        Return the full action probability vector under the belief.

        Used by Graph to populate hist_actions_prob with a distribution
        rather than a one-hot vector, which is more informative when the
        belief is spread across the threshold boundary.

        Args:
            belief (np.ndarray): Shape (n_states,).

        Returns:
            np.ndarray: Shape (n_actions,).
        """
        return self.policy.T @ belief

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_policy(self) -> np.ndarray:
        """
        Build the deterministic policy matrix from the threshold rule.

        Returns:
            np.ndarray: Shape (n_states, n_actions), dtype float64.
                        policy[s, 0] = 1  if state s is below threshold (do nothing).
                        policy[s, 1] = 1  if state s is at or above threshold (repair).
        """
        n_states  = len(self.states_list)
        n_actions = len(self.actions_list)
        policy    = np.zeros((n_states, n_actions))

        for s, label_str in enumerate(self.states_list):
            label = int(label_str)
            if label >= self.threshold_label:
                policy[s, 1] = 1.0   # 'perfect_repair'
            else:
                policy[s, 0] = 1.0   # 'do_nothing'

        return policy


class HeuristicPlanner:
    """
    Kamariotis-style heuristic decision rule over a belief distribution.

    Belief-based (acts on the full posterior, not a per-state policy matrix), so
    it can use context such as "a flood just happened". Interface: decide(belief,
    context) -> action label. This is the achievable baseline that the POMDP and
    active-inference planners are compared against.

    Rule (label 0 = healthy, higher = worse), on the belief-expected label:
        expected_label >= repair_threshold        -> 'repair'
        else if any inspect trigger fires          -> 'inspect'
        else                                       -> 'do_nothing'
    Inspect triggers: a flood (if inspect_after_flood), a scheduled interval, a
    "watch zone" (expected_label >= inspect_threshold), or high belief entropy.

    Args mirror the project conventions; thresholds are damage-class labels.
    """

    def __init__(
        self,
        states_list:        list[str],
        actions_list:       list[str],
        repair_threshold:   float = 6.0,
        inspect_threshold:  float | None = None,
        inspection_interval: int | None = None,
        inspect_after_flood: bool = True,
        entropy_frac:       float = 0.75,
    ):
        self.states_list = states_list
        self.actions_list = actions_list
        self.repair_threshold = float(repair_threshold)
        self.inspect_threshold = (repair_threshold - 2.0
                                  if inspect_threshold is None else float(inspect_threshold))
        self.inspection_interval = inspection_interval
        self.inspect_after_flood = inspect_after_flood
        self.entropy_frac = entropy_frac
        self._labels = np.array([int(s) for s in states_list], dtype=float)
        self._max_entropy = np.log(len(states_list))

    def decide(self, belief: np.ndarray, context: dict | None = None) -> str:
        ctx = context or {}
        belief = np.asarray(belief, dtype=float)
        exp_label = float(belief @ self._labels)

        if exp_label >= self.repair_threshold and "repair" in self.actions_list:
            return "repair"

        if "inspect" in self.actions_list:
            flood = bool(ctx.get("flood", False))
            step = int(ctx.get("step", -1))
            ent = -np.sum(belief * np.log(belief + 1e-12)) / self._max_entropy
            scheduled = (self.inspection_interval and step >= 0
                         and step % self.inspection_interval == 0)
            if ((flood and self.inspect_after_flood)
                    or scheduled
                    or exp_label >= self.inspect_threshold
                    or ent >= self.entropy_frac):
                return "inspect"

        return "do_nothing"


class POMDPPlanner:
    """
    Myopic (one-step) Value-of-Information POMDP planner.

    Captures *why* one would inspect: a fully-observed cost-MDP never inspects
    (see CostBenefitPlanner), because inspection's only value is reduced
    uncertainty. Here we make that value explicit and monetary:

        VoI = E[best decision cost | current belief]
              − E_over_observations[ best decision cost | belief after inspecting ]

    Inspect iff VoI exceeds the inspection cost; otherwise commit to the cheaper
    of repair / do_nothing under the current belief. This is the standard myopic
    POMDP approximation — interpretable and € quantifiable — and can later be
    upgraded to a point-based POMDP solver.

    Risk perception (digital_twin.risk.RiskModel): the perceived cost of *not*
    acting is `risk.perceived_cost(failure_costs, belief)` — a risk measure over
    the belief, not a plain expectation. A risk-averse model up-weights the
    dangerous tail of the belief, so the planner both inspects more (resolving a
    tail that a risk-neutral mean would ignore) and repairs earlier (perceiving
    that tail as costlier). Default attitude='neutral' reproduces the plain
    expected-cost POMDP exactly.

    Args:
        states_list, actions_list : must include 'do_nothing', 'inspect', and a
                                    repair action ('repair'/'perfect_repair').
        cost_model                : digital_twin.costs.CostModel.
        max_damage, discretization: damage-fraction mapping.
        inspect_conf              : (n, n) observation model p(obs | true state)
                                    for an inspection; default = sharp (accurate)
                                    so inspecting yields near-ground-truth.
        risk_model                : digital_twin.risk.RiskModel; default neutral
                                    (= expected-cost POMDP).
    """

    def __init__(
        self,
        states_list:    list[str],
        actions_list:   list[str],
        cost_model,
        max_damage:     float = 60.0,
        discretization: float = 5.0,
        inspect_conf:   np.ndarray | None = None,
        risk_model:     RiskModel | None = None,
    ):
        self.states_list = states_list
        self.actions_list = actions_list
        self.cm = cost_model
        self.n = len(states_list)
        self.fracs = np.array([(int(s) * discretization) / max_damage
                               for s in states_list])
        self.risk = risk_model or RiskModel()
        self._failcosts = np.array([self.cm.expected_failure_cost(f)
                                    for f in self.fracs])
        self._repair = next(a for a in ("repair", "perfect_repair")
                            if a in actions_list)
        self.inspect_conf = (self._default_inspect_conf()
                             if inspect_conf is None else np.asarray(inspect_conf))

    def _default_inspect_conf(self, acc: float = 0.9) -> np.ndarray:
        """Sharp observation model: prob `acc` on the true class, rest to neighbours."""
        C = np.zeros((self.n, self.n))
        for s in range(self.n):
            C[s, s] = acc
            if s > 0:        C[s - 1, s] += (1 - acc) / (2 if 0 < s < self.n - 1 else 1)
            if s < self.n-1: C[s + 1, s] += (1 - acc) / (2 if 0 < s < self.n - 1 else 1)
        C /= C.sum(axis=0, keepdims=True)
        return C

    # ── decision-cost building blocks ───────────────────────────────────────────

    def _cost_do_nothing(self, belief: np.ndarray) -> float:
        # Risk-adjusted perceived cost of the failure-risk distribution under the
        # belief (plain expectation when the risk model is neutral).
        return self.risk.perceived_cost(self._failcosts, belief)

    def _cost_repair(self, belief: np.ndarray) -> float:
        return self.cm.action_cost(self._repair)

    def _best_decision_cost(self, belief: np.ndarray) -> float:
        return min(self._cost_do_nothing(belief), self._cost_repair(belief))

    def value_of_inspection(self, belief: np.ndarray) -> float:
        """Expected € reduction in decision cost from one inspection."""
        belief = np.asarray(belief, dtype=float)
        base = self._best_decision_cost(belief)
        p_obs = self.inspect_conf @ belief                      # p(obs | belief)
        exp_after = 0.0
        for o in range(self.n):
            if p_obs[o] <= 0:
                continue
            post = self.inspect_conf[o, :] * belief
            post = post / post.sum()
            exp_after += p_obs[o] * self._best_decision_cost(post)
        return float(base - exp_after)

    def decide(self, belief: np.ndarray, context: dict | None = None) -> str:
        belief = np.asarray(belief, dtype=float)
        if "inspect" in self.actions_list:
            voi = self.value_of_inspection(belief)
            if voi > self.cm.action_cost("inspect"):
                return "inspect"
        return self._repair if self._cost_repair(belief) < self._cost_do_nothing(belief) \
            else "do_nothing"


class CostBenefitPlanner(Planner):
    """
    Cost-optimal maintenance policy via value iteration with monetary costs.

    Drop-in replacement for Planner: exposes the same (n_states, n_actions)
    `.policy` matrix, so Graph and the DBN use it unchanged. Instead of a fixed
    threshold, the repair point is chosen to minimise the expected discounted
    life-cycle cost under a CostModel (digital_twin.costs).

    Convention: label 0 = healthy, higher label = more damage; repair resets to
    label 0. (Note: dbn.py's _restart_transition currently maps to label N-1 —
    flag/verify that against this convention before a full pgmpy run.)

    Note on 'inspect': value iteration over the fully-observed damage state will
    never select 'inspect' — it costs money without changing the structural
    state, so its only value is informational (a POMDP / active-inference effect,
    or a heuristic monitoring schedule). Including it here makes that explicit:
    the cost-MDP optimum uses only do_nothing / repair.

    Args:
        states_list:    Ordered state labels ['0', …, 'N-1'] (0 = healthy).
        actions_list:   Ordered action labels; must contain 'do_nothing' and a
                        repair action ('repair' or 'perfect_repair'). 'inspect'
                        is allowed but will not be selected.
        cost_model:     digital_twin.costs.CostModel instance.
        max_damage:     Maximum damage percent (for damage-fraction mapping).
        discretization: Damage step percent per label.
        p_advance:      Per-step probability of advancing one damage level under
                        do_nothing (deterioration rate prior).
        gamma:          Discount factor for value iteration (≈ 1/(1+r)).
    """

    def __init__(
        self,
        states_list:    list[str],
        actions_list:   list[str],
        cost_model,
        max_damage:     float = 60.0,
        discretization: float = 5.0,
        p_advance:      float = 0.10,
        gamma:          float = 0.95,
        n_iter:         int   = 5000,
        tol:            float = 1e-9,
    ):
        self.states_list     = states_list
        self.actions_list    = actions_list
        self.cost_model      = cost_model
        self.max_damage      = float(max_damage)
        self.discretization  = float(discretization)
        self.p_advance       = float(p_advance)
        self.gamma           = float(gamma)
        self.n_iter          = int(n_iter)
        self.tol             = float(tol)

        self._repair_idx = self._find_action(("repair", "perfect_repair"))
        self._nothing_idx = self._find_action(("do_nothing",))
        self.threshold_label = None       # set after value iteration, for logging
        self.policy = self._compute_policy()

    # ── Private ───────────────────────────────────────────────────────────────

    def _find_action(self, names: tuple) -> int:
        for nm in names:
            if nm in self.actions_list:
                return self.actions_list.index(nm)
        raise ValueError(f"actions_list must contain one of {names}; got {self.actions_list}")

    def _damage_fraction(self, label: int) -> float:
        return (label * self.discretization) / self.max_damage

    def _transition(self, action_idx: int) -> np.ndarray:
        """P(s'|s, action): row = from-state, col = to-state."""
        n = len(self.states_list)
        T = np.zeros((n, n))
        if action_idx == self._repair_idx:
            T[:, 0] = 1.0                       # repair -> healthy (label 0)
        else:
            # do_nothing AND inspect: the structure keeps deteriorating either
            # way (inspecting does not halt scour — its only value is the
            # information it yields, which a fully-observed MDP cannot see). So
            # inspect shares the do_nothing transition and is strictly dominated
            # here; it can only be selected by a POMDP / active-inference layer.
            for s in range(n):
                if s == n - 1:
                    T[s, s] = 1.0               # absorbing at max damage
                else:
                    T[s, s]     = 1.0 - self.p_advance
                    T[s, s + 1] = self.p_advance
        return T

    def _compute_policy(self) -> np.ndarray:
        n  = len(self.states_list)
        na = len(self.actions_list)

        # Reward R[s, a] = -(action cost + expected failure risk) [€, negative].
        R = np.zeros((n, na))
        for s in range(n):
            frac = self._damage_fraction(s)
            for a, name in enumerate(self.actions_list):
                R[s, a] = -self.cost_model.step_cost(frac, name)

        T = [self._transition(a) for a in range(na)]

        # Value iteration.
        V = np.zeros(n)
        for _ in range(self.n_iter):
            Q = np.column_stack([R[:, a] + self.gamma * (T[a] @ V) for a in range(na)])
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < self.tol:
                V = V_new
                break
            V = V_new

        best = Q.argmax(axis=1)
        policy = np.zeros((n, na))
        policy[np.arange(n), best] = 1.0

        # Smallest damaged label at which repair becomes optimal (for logging).
        repaired = np.where(best == self._repair_idx)[0]
        self.threshold_label = int(repaired.min()) if repaired.size else None
        return policy

class FloodResponsePlanner:
    """Decision branch for an observed major flood (Step 2).

    A river gauge has just declared a major flood and the belief over damage has
    been WIDENED to represent the uncertain scour shock (digital_twin.flood). The
    owner must now choose how hard to intervene, among four actions of rising
    cost and protection:

        do_nothing          — accept the widened belief, run the line normally.
        restrict_operations — a reduced-risk PROBE: a few low-mass / low-speed
                              instrumented passages → a WIDER (lower-SNR) drive-by
                              observation. Cheap, but its information is degraded
                              and can be lost if the sensors are faulty.
        inspect             — a sharp human/diver survey (sensor-independent).
        interrupt           — close the line now (so it carries NO failure risk
                              this step) AND inspect.

    Choice rule — risk-adjusted one-step lookahead (the same Value-of-Information
    logic as POMDPPlanner, generalised to several information actions with
    different cost, observation model, and protection):

        R(b)        = risk.perceived_cost(failure_costs, b)         # carried risk
        commit(b)   = min(R(b), c_repair)                          # best follow-up
        E_commit(b,L) = Σ_o p(o|b) · commit(posterior(b, L, o))    # after observing

        J(do_nothing) = expo·R(b) + commit(b)
        J(restrict)   = c_restrict + expo·R(b)
                        + reliab·E_commit(b, L_probe) + (1−reliab)·commit(b)
        J(inspect)    = c_inspect  + expo·R(b) + E_commit(b, L_inspect)
        J(interrupt)  = c_interrupt              + E_commit(b, L_inspect)

    and the action with the smallest J is taken. The `expo·R(b)` term is the
    failure risk the OPEN line carries during the acute post-flood window; it
    cancels among the three open-line actions (so they are ranked by cost-vs-VoI,
    exactly the POMDP) but is removed by `interrupt`, which is therefore chosen
    precisely when R(widened) is large — a near-threshold belief hit by a severe
    flood. The `reliab` factor is the probability the probe yields usable data
    (from the sensor-health layer); a low value collapses the probe's VoI and
    makes the planner escalate to inspect / interrupt — the sensor-corruption
    escalation, expressed at the decision level. Risk aversion (RiskModel) up-
    weights the dangerous tail of the widened belief, lowering the bar for
    inspect / interrupt; neutral reproduces expected-cost behaviour.

    Args:
        states_list:    ordered damage labels (0 = healthy).
        cost_model:     digital_twin.costs.CostModel (uses restrict_operations,
                        inspect, interrupt, repair action costs).
        probe_conf:     (n,n) observation model L[true,pred] for the restricted-ops
                        probe (WIDER than drive-by; assumed now, measurable from a
                        MATLAB low-mass/low-speed batch later — §6.5).
        inspect_conf:   (n,n) sharp observation model for the inspection/interrupt.
        risk_model:     digital_twin.risk.RiskModel; default neutral.
        exposure_frac:  fraction of the post-flood window the OPEN line carries
                        the elevated failure risk before the survey result is in
                        (≈ inspection latency / step; ~0.02 ≈ same-day survey on a
                        monthly clock). Small: interrupt is reserved for the
                        genuinely severe corner where R(widened) is large; raising
                        it makes the owner close more readily.
    """

    FLOOD_ACTIONS = ("do_nothing", "restrict_operations", "inspect", "interrupt")

    def __init__(
        self,
        states_list:    list[str],
        cost_model,
        probe_conf:     np.ndarray,
        inspect_conf:   np.ndarray,
        max_damage:     float = 60.0,
        discretization: float = 5.0,
        risk_model:     RiskModel | None = None,
        exposure_frac:  float = 0.02,
    ):
        self.states_list = states_list
        self.cm = cost_model
        self.n = len(states_list)
        self.fracs = np.array([(int(s) * discretization) / max_damage
                               for s in states_list])
        self.risk = risk_model or RiskModel()
        self._failcosts = np.array([self.cm.expected_failure_cost(f)
                                    for f in self.fracs])
        self.exposure_frac = float(exposure_frac)
        self.L_probe = self._colnorm(probe_conf)
        self.L_inspect = self._colnorm(inspect_conf)
        self._c_restrict = self.cm.action_cost("restrict_operations")
        self._c_inspect = self.cm.action_cost("inspect")
        self._c_interrupt = self.cm.action_cost("interrupt")
        self._c_repair = self.cm.action_cost(
            next(a for a in ("repair", "perfect_repair") if a in self.cm._action_cost))

    @staticmethod
    def _colnorm(M: np.ndarray) -> np.ndarray:
        """Row-normalise L[true, pred] over predictions (matches the convention in
        DTSimulator.row_normalise: row=true state, column=predicted label)."""
        M = np.asarray(M, dtype=float)
        return M / np.clip(M.sum(axis=1, keepdims=True), 1e-12, None)

    # ── building blocks ─────────────────────────────────────────────────────────

    def _R(self, belief: np.ndarray) -> float:
        return self.risk.perceived_cost(self._failcosts, belief)

    def _commit(self, belief: np.ndarray) -> float:
        return min(self._R(belief), self._c_repair)

    def _exp_commit(self, belief: np.ndarray, L: np.ndarray) -> float:
        """E over observations of commit(posterior) after observing through L."""
        p_obs = belief @ L                                   # p(pred=o | belief)
        acc = 0.0
        for o in range(self.n):
            if p_obs[o] <= 0:
                continue
            post = L[:, o] * belief
            tot = post.sum()
            if tot <= 0:
                continue
            acc += p_obs[o] * self._commit(post / tot)
        return float(acc)

    # ── decision ────────────────────────────────────────────────────────────────

    def score(self, belief: np.ndarray, probe_reliability: float = 1.0) -> dict:
        """Risk-adjusted expected € cost J of each flood action (smaller = better)."""
        belief = np.asarray(belief, dtype=float)
        R0 = self._R(belief)
        commit0 = min(R0, self._c_repair)
        expo = self.exposure_frac * R0
        rel = float(np.clip(probe_reliability, 0.0, 1.0))
        ec_probe = self._exp_commit(belief, self.L_probe)
        ec_sharp = self._exp_commit(belief, self.L_inspect)
        return {
            "do_nothing":          expo + commit0,
            "restrict_operations": self._c_restrict + expo
                                   + rel * ec_probe + (1.0 - rel) * commit0,
            "inspect":             self._c_inspect + expo + ec_sharp,
            "interrupt":           self._c_interrupt + ec_sharp,
        }

    def decide(self, belief, context: dict | None = None) -> str:
        """Return the lowest-cost flood action. context may carry
        'probe_reliability' (default 1.0) from the sensor-health layer."""
        ctx = context or {}
        J = self.score(belief, ctx.get("probe_reliability", 1.0))
        return min(J, key=J.get)


class HybridPlanner:
    """
    Hybrid planner: POMDP Value-of-Information for the *inspect* decision and a
    cost-optimal value-iteration policy for the *repair* timing.

    Rationale (from the mock comparison): the myopic POMDP inspects sensibly but
    under-repairs (no lookahead), while cost-VI repairs at the cost-optimal
    threshold but never inspects. Combining them gives selective inspection AND
    well-timed repair. Same decide(belief, context) interface as the others.

    Decision order each step:
        1. if the cost-VI policy says repair  -> repair
        2. elif inspecting is worth it (VoI > c_inspect) -> inspect
        3. else -> do_nothing

    Risk perception (RiskModel) is applied to the inspection decision through the
    embedded POMDP planner; the repair timing comes from the (risk-neutral)
    cost-VI lookahead. Default neutral reproduces the original behaviour.
    """

    def __init__(
        self,
        states_list,
        actions_list,
        cost_model,
        max_damage: float = 60.0,
        discretization: float = 5.0,
        p_advance: float = 0.10,
        gamma: float = 0.95,
        risk_model: RiskModel | None = None,
    ):
        self.actions = actions_list
        self.cm = cost_model
        self._repair = next(a for a in ("repair", "perfect_repair") if a in actions_list)
        self._repair_planner = CostBenefitPlanner(
            states_list, [a for a in actions_list if a != "inspect"],
            cost_model, max_damage, discretization, p_advance, gamma)
        self._inspect_planner = POMDPPlanner(
            states_list, actions_list, cost_model, max_damage, discretization,
            risk_model=risk_model)

    def decide(self, belief, context=None) -> str:
        if self._repair_planner.decide(belief) == self._repair:
            return self._repair
        if ("inspect" in self.actions
                and self._inspect_planner.value_of_inspection(belief)
                > self.cm.action_cost("inspect")):
            return "inspect"
        return "do_nothing"
