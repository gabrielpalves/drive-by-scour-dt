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

    def decide(self, belief: np.ndarray) -> str:
        """
        Return the MAP action given a belief distribution over states.

        Computes the expected action probability under the belief and
        returns the action with the highest expected probability.  For a
        deterministic policy this is equivalent to acting on the MAP state.

        Args:
            belief (np.ndarray): Shape (n_states,).  Must sum to 1.

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