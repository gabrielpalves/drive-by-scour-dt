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

    Args:
        states_list, actions_list : must include 'do_nothing', 'inspect', and a
                                    repair action ('repair'/'perfect_repair').
        cost_model                : digital_twin.costs.CostModel.
        max_damage, discretization: damage-fraction mapping.
        inspect_conf              : (n, n) observation model p(obs | true state)
                                    for an inspection; default = sharp (accurate)
                                    so inspecting yields near-ground-truth.
    """

    def __init__(
        self,
        states_list:    list[str],
        actions_list:   list[str],
        cost_model,
        max_damage:     float = 60.0,
        discretization: float = 5.0,
        inspect_conf:   np.ndarray | None = None,
    ):
        self.states_list = states_list
        self.actions_list = actions_list
        self.cm = cost_model
        self.n = len(states_list)
        self.fracs = np.array([(int(s) * discretization) / max_damage
                               for s in states_list])
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
        return float(np.sum(belief * np.array(
            [self.cm.expected_failure_cost(f) for f in self.fracs])))

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
    ):
        self.actions = actions_list
        self.cm = cost_model
        self._repair = next(a for a in ("repair", "perfect_repair") if a in actions_list)
        self._repair_planner = CostBenefitPlanner(
            states_list, [a for a in actions_list if a != "inspect"],
            cost_model, max_damage, discretization, p_advance, gamma)
        self._inspect_planner = POMDPPlanner(
            states_list, actions_list, cost_model, max_damage, discretization)

    def decide(self, belief, context=None) -> str:
        if self._repair_planner.decide(belief) == self._repair:
            return self._repair
        if ("inspect" in self.actions
                and self._inspect_planner.value_of_inspection(belief)
                > self.cm.action_cost("inspect")):
            return "inspect"
        return "do_nothing"
