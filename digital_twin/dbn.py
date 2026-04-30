"""
digital_twin/dbn.py
===================
Dynamic Bayesian Network infrastructure for the drive-by digital twin.

Public API
----------
    normalize_cpd  — column-wise normalisation of a CPD matrix.
    GetSetup       — builds the DBN graph structure and all CPD tables.
    Graph          — assembles the DBN, runs the simulation loop, and
                     exposes a forward-prediction method.

The DBN models two coupled time slices.  At each monitoring step the
physical bridge transitions under a stochastic degradation law; the
classifier observes a noisy measurement D_NN; and the DBN fuses the
prior belief, the degradation transition, and the new observation into
a posterior over the true damage state D.

Graph topology (main DBN)
--------------------------

    U^{A}_{-1} ──┐
                 ▼
    D_{NN}  ────► D_t ──► U^{P}_t
                  │
                  ▼
    D_{NN+1} ───► D_{t+1} ──► U^{P}_{t+1}
                 ▲
    U^{A}_{t} ───┘

Prediction sub-graph (digital-only, no physical observations)
--------------------------------------------------------------

    D_t ──► U_t ──► D_{t+1} ──► U_{t+1}

Imported by:
    drive_by_DT.py — GetSetup, Graph, normalize_cpd
"""

import numpy as np
import pandas as pd

from pgmpy.models          import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference        import DBNInference


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def normalize_cpd(matrix: np.ndarray) -> np.ndarray:
    """
    Column-wise normalisation of a CPD matrix.

    Zero-sum columns are left as-is (no division by zero) so that
    partially-specified CPDs do not raise errors during graph assembly.

    Args:
        matrix (np.ndarray): Shape (n_states, n_parent_configs) or (n_states,).

    Returns:
        np.ndarray: Same shape, each column summing to 1 (or 0 if the input
                    column was all zeros).
    """
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return matrix / col_sums


# ──────────────────────────────────────────────────────────────────────────────
# 1. DBN builder
# ──────────────────────────────────────────────────────────────────────────────

class GetSetup:
    """
    Constructs the full CPD tables and graph structure for the DBN.

    The damage state is indexed from 0 (healthy, 0 % damage) to N-1
    (maximum damage), so "deterioration" means transitioning toward
    higher indices.

    Args:
        states_list (list[str]):  Ordered list of state labels, e.g.
                                  ['0', '1', …, '12'].  Index 0 is
                                  the healthy state.
        actions_list (list[str]): Ordered list of action labels, e.g.
                                  ['do_nothing', 'perfect_repair'].
        conf_mat_dt (np.ndarray): Confusion matrix of the trained classifier,
                                  shape (n_states, n_states).  Column j gives
                                  the distribution over predicted labels given
                                  true label j, i.e. p(D_NN | D).
    """

    def __init__(
        self,
        states_list:  list[str],
        actions_list: list[str],
        conf_mat_dt:  np.ndarray,
    ):
        self.states_list  = states_list
        self.actions_list = actions_list
        self.n_states     = len(states_list)
        self.conf_mat_dt  = conf_mat_dt

    # ── Transition matrices ───────────────────────────────────────────────────

    def _low_diag_transition(self, prob_adv: float) -> np.ndarray:
        """
        Upper-triangular transition matrix for monotonic damage progression.

        Label 0 is healthy; label n-1 is maximum damage.  The state can
        only stay the same or advance one step toward higher labels
        (more damage).  Label n-1 is absorbing — maximum damage cannot
        worsen further.

        Args:
            prob_adv: Probability of advancing one damage level per time step.

        Returns:
            np.ndarray: (n_states, n_states) column-stochastic matrix.
        """
        n = self.n_states
        T = np.zeros((n, n))
        for i in range(n):
            if i == n - 1:              # absorbing at maximum damage
                T[i, i]     = 1.0
            else:
                T[i,   i]   = 1.0 - prob_adv
                T[i+1, i]   = prob_adv  # advance toward higher label (more damage)
        return T

    def _restart_transition(self) -> np.ndarray:
        """
        Transition matrix for perfect repair: all states map to healthy (N-1).

        Returns:
            np.ndarray: (n_states, n_states).
        """
        T = np.zeros((self.n_states, self.n_states))
        T[self.n_states - 1, :] = 1.0
        return T

    def _fuse_transitions(
        self,
        phys_transition: np.ndarray,
        obs_transition:  np.ndarray,
    ) -> np.ndarray:
        """
        Combine p(D_t | D_{t-1}) and p(D_t | D_NN_t) into a joint CPD.

        The product p(D_t | D_{t-1}, D_NN_t) ∝ p(D_t | D_{t-1}) × p(D_t | D_NN_t)
        is formed column-by-column for every (D_{t-1}, D_NN_t) combination
        and then normalised.

        Args:
            phys_transition: (n_states, n_states) — p(D_t | D_{t-1}).
            obs_transition:  (n_states, n_states) — p(D_t | D_NN_t)
                             (typically the confusion matrix).

        Returns:
            np.ndarray: (n_states, n_states²) — CPD ready for TabularCPD.
        """
        n    = self.n_states
        fused = np.zeros((n, n * n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    fused[i, j * n + k] = phys_transition[i, j] * obs_transition[i, k]
        return normalize_cpd(fused)

    def get_transitions(self, p_nothing: float) -> np.ndarray:
        """
        Return the stacked transition tensor for both actions.

        Args:
            p_nothing: Annual probability of advancing one damage level
                       under the 'do_nothing' action.

        Returns:
            np.ndarray: (2, n_states, n_states) — axis 0 indexes the action
                        (0 = do_nothing, 1 = perfect_repair).
        """
        T_nothing = self._low_diag_transition(prob_adv=p_nothing)
        T_repair  = self._restart_transition()
        return np.stack([T_nothing, T_repair], axis=0)

    # ── Graph assembly ────────────────────────────────────────────────────────

    def get_graph(
        self,
        cpd_d_to_u:  np.ndarray,
        transitions: np.ndarray,
    ) -> tuple:
        """
        Build the full edge lists and CPD objects for both DBNs.

        Args:
            cpd_d_to_u (np.ndarray):  Policy matrix, shape (n_actions, n_states).
                                       Column s gives p(U | D=s).
            transitions (np.ndarray): Output of get_transitions — shape
                                      (n_actions, n_states, n_states).

        Returns:
            (graph_structure, digital_subgraph,
             list_cpd_graph, list_cpd_subgraph)

            graph_structure:    Edge list for the main filtering DBN.
            digital_subgraph:   Edge list for the prediction sub-graph.
            list_cpd_graph:     CPD objects for the main DBN.
            list_cpd_subgraph:  CPD objects for the prediction sub-graph.
        """
        Ns      = self.n_states
        Na      = len(self.actions_list)
        states  = self.states_list
        actions = self.actions_list
        C       = self.conf_mat_dt

        # ── Edge lists ────────────────────────────────────────────────────────
        graph_structure = [
            (('U^{A}_{-1}', 0), ('D', 0)),
            (('D_{NN}',     0), ('D', 0)),
            (('D',          0), ('U^{P}', 0)),
            (('D',          0), ('D', 1)),
            (('U^{A}_{-1}', 1), ('D', 1)),
            (('D_{NN}',     1), ('D', 1)),
            (('D',          1), ('U^{P}', 1)),
        ]

        digital_subgraph = [
            (('D', 0), ('U', 0)),
            (('D', 0), ('D', 1)),
            (('U', 0), ('D', 1)),
            (('D', 1), ('U', 1)),
        ]

        # ── Fused transition CPDs for each action ─────────────────────────────
        cond_d_nothing = self._fuse_transitions(C, transitions[0])
        cond_d_repair  = self._fuse_transitions(C, transitions[1])
        cond_d_full    = np.concatenate([cond_d_nothing, cond_d_repair], axis=1)

        # ── CPD objects — main DBN ────────────────────────────────────────────
        Ua0 = TabularCPD(
            ('U^{A}_{-1}', 0), Na, np.ones((Na, 1)) / Na,
            state_names={('U^{A}_{-1}', 0): actions},
        )
        DNN0 = TabularCPD(
            ('D_{NN}', 0), Ns, np.ones((Ns, 1)) / Ns,
            state_names={('D_{NN}', 0): states},
        )
        # t=0: D_0 has no D_{-1} parent, so the physical prior is just C tiled
        D0 = TabularCPD(
            ('D', 0), Ns, np.tile(C, (1, Na)),
            evidence=[('U^{A}_{-1}', 0), ('D_{NN}', 0)],
            evidence_card=[Na, Ns],
            state_names={
                ('D', 0):          states,
                ('U^{A}_{-1}', 0): actions,
                ('D_{NN}', 0):     states,
            },
        )
        Up0 = TabularCPD(
            ('U^{P}', 0), Na, cpd_d_to_u,
            evidence=[('D', 0)], evidence_card=[Ns],
            state_names={('U^{P}', 0): actions, ('D', 0): states},
        )
        Ua1 = TabularCPD(
            ('U^{A}_{-1}', 1), Na, np.ones((Na, 1)) / Na,
            state_names={('U^{A}_{-1}', 1): actions},
        )
        DNN1 = TabularCPD(
            ('D_{NN}', 1), Ns, np.ones((Ns, 1)) / Ns,
            state_names={('D_{NN}', 1): states},
        )
        D1 = TabularCPD(
            ('D', 1), Ns, cond_d_full,
            evidence=[('U^{A}_{-1}', 1), ('D_{NN}', 1), ('D', 0)],
            evidence_card=[Na, Ns, Ns],
            state_names={
                ('D', 1):          states,
                ('U^{A}_{-1}', 1): actions,
                ('D_{NN}', 1):     states,
                ('D', 0):          states,
            },
        )
        Up1 = TabularCPD(
            ('U^{P}', 1), Na, cpd_d_to_u,
            evidence=[('D', 1)], evidence_card=[Ns],
            state_names={('U^{P}', 1): actions, ('D', 1): states},
        )

        # ── CPD objects — prediction sub-graph ────────────────────────────────
        D0_sub = TabularCPD(
            ('D', 0), Ns, np.ones((Ns, 1)) / Ns,
            state_names={('D', 0): states},
        )
        U0_sub = TabularCPD(
            ('U', 0), Na, cpd_d_to_u,
            evidence=[('D', 0)], evidence_card=[Ns],
            state_names={('U', 0): actions, ('D', 0): states},
        )
        D1_sub = TabularCPD(
            ('D', 1), Ns,
            np.concatenate([transitions[i] for i in range(Na)], axis=1),
            evidence=[('U', 0), ('D', 0)],
            evidence_card=[Na, Ns],
            state_names={
                ('D', 1): states,
                ('U', 0): actions,
                ('D', 0): states,
            },
        )
        U1_sub = TabularCPD(
            ('U', 1), Na, cpd_d_to_u,
            evidence=[('D', 1)], evidence_card=[Ns],
            state_names={('U', 1): actions, ('D', 1): states},
        )

        list_cpd_graph    = [Ua0, DNN0, D0, Up0, Ua1, DNN1, D1, Up1]
        list_cpd_subgraph = [D0_sub, U0_sub, D1_sub, U1_sub]

        return graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph


# ──────────────────────────────────────────────────────────────────────────────
# 2. Simulation manager
# ──────────────────────────────────────────────────────────────────────────────

class Graph:
    """
    Assembles the two DBNs, drives the simulation loop, and accumulates
    history for downstream plotting and analysis.

    History dataframes
    ------------------
        hist_var           — one row per step: continuous physical state,
                             discrete true label, MAP belief label, classifier
                             prediction, and chosen action.
        hist_d_state_prob  — one row per step: full belief distribution
                             p(D_t | observations 1…t).
        hist_actions_prob  — one row per step: marginal action probabilities.

    Args:
        physical_asset:  PhysicalAsset instance.
        digital_asset:   DigitalAsset instance.
        states_list:     Ordered state label list (matches GetSetup).
        actions_list:    Ordered action label list (matches GetSetup).
        planner:         Planner instance (supplies the policy matrix).
    """

    def __init__(
        self,
        physical_asset,
        digital_asset,
        states_list:  list[str],
        actions_list: list[str],
        planner,
    ):
        self.physical_asset = physical_asset
        self.digital_asset  = digital_asset
        self.states_list    = states_list
        self.actions_list   = actions_list
        self.planner        = planner

        self.dbn         = DBN()
        self.dbn_predict = DBN()

        self.hist_var:          pd.DataFrame | None = None
        self.hist_d_state_prob: pd.DataFrame | None = None
        self.hist_actions_prob: pd.DataFrame | None = None
        self.time_index: int = -1

    # ── Setup ─────────────────────────────────────────────────────────────────

    def assemble_graph(
        self,
        graph_structure:     list,
        cpd_list:            list,
        digital_subgraph:    list,
        list_cpd_subgraph:   list,
    ) -> None:
        """
        Add edges and CPDs to both DBNs and call check_model().

        Args:
            graph_structure:   Main DBN edge list (from GetSetup.get_graph).
            cpd_list:          Main DBN CPD objects.
            digital_subgraph:  Prediction sub-graph edge list.
            list_cpd_subgraph: Prediction sub-graph CPD objects.
        """
        self.dbn.add_edges_from(graph_structure)
        for cpd in cpd_list:
            cpd.normalize()
            self.dbn.add_cpds(cpd)
        self.dbn.check_model()
        print("Main DBN assembled and verified.")

        self.dbn_predict.add_edges_from(digital_subgraph)
        for cpd in list_cpd_subgraph:
            cpd.normalize()
            self.dbn_predict.add_cpds(cpd)
        self.dbn_predict.check_model()
        print("Prediction DBN assembled and verified.")

        self.time_index = -1

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self, n_steps: int, n_samples: int) -> None:
        """
        Run the filtering simulation for n_steps monitoring intervals.

        At each step:
        1. The physical asset evolves under the previous action.
        2. The physics engine generates a sensor observation.
        3. The classifier predicts a damage label (D_NN).
        4. DBN forward inference fuses the prior belief with D_NN and
           the previous action to produce a posterior over D_t.
        5. The planner maps the posterior to an action distribution.
        6. History is appended.

        Initialisation (time_index == -1)
        ----------------------------------
        The first call initialises the history at t=0 by running one
        physical evolution step, one observation, and one DBN inference
        before entering the main loop.  Subsequent calls to simulate()
        resume from the current time_index, allowing the simulation to
        be extended without re-running earlier steps.

        Args:
            n_steps   (int): Number of monitoring steps to run.
            n_samples (int): Number of Monte Carlo samples drawn from the
                             previous belief distribution for the DBN
                             inference approximation.
        """
        if self.time_index == -1:
            self._initialise(n_samples)

        infer = DBNInference(self.dbn)

        for i in range(n_steps):
            self.time_index += 1
            print(f"\nStep {i + 1}/{n_steps}  (t = {self.time_index})")

            last_action       = self.hist_var['current_action'].iloc[-1]
            last_belief       = self.hist_d_state_prob.iloc[-1].to_numpy()

            # 1. Physical evolution under previous action
            self.physical_asset.update_physical_state(action_str=last_action)

            # 2. Observation
            raw_signal        = self.physical_asset.get_observation_signal()

            # 3. Classifier prediction
            pred_label        = self.digital_asset.estimate_state(raw_signal)

            # 4. DBN inference (particle approximation over the prior)
            new_belief, new_action_prob = self._infer_step(
                infer, last_action, pred_label, last_belief, n_samples
            )

            # 5. MAP action
            current_action = self.actions_list[int(np.argmax(new_action_prob))]

            # 6. Append history
            true_label = self.physical_asset.get_true_mapped_label()
            self._append_history(
                true_label, pred_label, new_belief, new_action_prob, current_action
            )
            self._print_step(true_label, pred_label, new_belief, current_action)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        n_steps:   int,
        n_samples: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Project the current belief n_steps into the future using the
        digital-only prediction sub-graph (no new physical observations).

        Uses pgmpy's DBN.simulate() seeded with the current posterior
        as a virtual-evidence prior on D_0.

        Args:
            n_steps   (int): Horizon length in monitoring intervals.
            n_samples (int): Monte Carlo samples for the belief projection.

        Returns:
            (result_D, result_U):
                result_D — DataFrame, shape (n_steps+1, n_states).
                result_U — DataFrame, shape (n_steps+1, n_actions).
        """
        print(f"\nForward prediction from t = {self.time_index} "
              f"({n_steps} steps, {n_samples} samples)")

        current_belief = self.hist_d_state_prob.iloc[self.time_index].to_list()
        virtual_prior  = [[p] for p in current_belief]

        sim = self.dbn_predict.simulate(
            n_time_slices=n_steps + 1,
            n_samples=n_samples,
            virtual_evidence=[
                TabularCPD(('D', 0), len(self.states_list), virtual_prior)
            ],
            show_progress=False,
        ).to_numpy()

        prob_D = np.zeros((n_steps + 1, len(self.states_list)))
        prob_U = np.zeros((n_steps + 1, len(self.actions_list)))

        for i in range(n_steps + 1):
            for j in range(n_samples):
                prob_D[i, sim[j, 2 * i]]     += 1
                prob_U[i, sim[j, 2 * i + 1]] += 1

        prob_D = normalize_cpd(prob_D.T).T
        prob_U = normalize_cpd(prob_U.T).T

        return (
            pd.DataFrame(prob_D, columns=self.states_list),
            pd.DataFrame(prob_U, columns=self.actions_list),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initialise(self, n_samples: int) -> None:
        """Run t=0 setup: first physical evolution, observation, and inference."""
        self.time_index = 0

        self.physical_asset.update_physical_state(action_str='do_nothing')
        raw_signal  = self.physical_asset.get_observation_signal()
        pred_label  = self.digital_asset.estimate_state(raw_signal)

        # DBN inference at t=0 uses D_0_cpd (no D_{t-1} parent)
        infer_init  = DBNInference(self.dbn)
        evid_init   = {
            ('D_{NN}',     0): self.states_list[pred_label],
            ('U^{A}_{-1}', 0): 'do_nothing',
        }
        result = infer_init.forward_inference([('D', 0)], evidence=evid_init)
        belief = result[('D', 0)].values

        action_prob    = np.matmul(self.planner.policy.T, belief)
        current_action = self.actions_list[int(np.argmax(action_prob))]

        true_label = self.physical_asset.get_true_mapped_label()
        self.hist_var = pd.DataFrame([{
            'p_state':          self.physical_asset.state_continuous,
            'p_state_discrete': self.states_list[true_label],
            'd_state':          self.states_list[int(np.argmax(belief))],
            'd_state_nn':       self.states_list[pred_label],
            'current_action':   current_action,
        }])
        self.hist_d_state_prob = pd.DataFrame(
            [belief.tolist()], columns=self.states_list
        )
        self.hist_actions_prob = pd.DataFrame(
            [action_prob.tolist()], columns=self.actions_list
        )

    def _infer_step(
        self,
        infer:       DBNInference,
        last_action: str,
        pred_label:  int,
        last_belief: np.ndarray,
        n_samples:   int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Particle approximation of DBN forward inference for one time slice.

        Samples n_samples states from the previous belief, runs DBN inference
        conditioned on each, and averages the resulting posteriors.

        Returns:
            (normalised belief, normalised action probabilities)
        """
        sampled        = np.random.choice(len(self.states_list), size=n_samples, p=last_belief)
        new_belief     = np.zeros(len(self.states_list))
        new_action_prob = np.zeros(len(self.actions_list))

        evid_base = {
            ('U^{A}_{-1}', 1): last_action,
            ('D_{NN}',     1): self.states_list[pred_label],
        }

        for s in sampled:
            evid = {**evid_base, ('D', 0): self.states_list[s]}
            res  = infer.forward_inference([('D', 1), ('U^{P}', 1)], evidence=evid)
            new_belief      += res[('D',    1)].values
            new_action_prob += res[('U^{P}', 1)].values

        return normalize_cpd(new_belief), normalize_cpd(new_action_prob)

    def _append_history(
        self,
        true_label:     int,
        pred_label:     int,
        belief:         np.ndarray,
        action_prob:    np.ndarray,
        current_action: str,
    ) -> None:
        """Append one row to each of the three history dataframes."""
        row = pd.DataFrame([{
            'p_state':          self.physical_asset.state_continuous,
            'p_state_discrete': self.states_list[true_label],
            'd_state':          self.states_list[int(np.argmax(belief))],
            'd_state_nn':       self.states_list[pred_label],
            'current_action':   current_action,
        }])
        self.hist_var = pd.concat(
            [self.hist_var, row], ignore_index=True
        )
        self.hist_d_state_prob = pd.concat(
            [self.hist_d_state_prob,
             pd.DataFrame([belief.tolist()], columns=self.states_list)],
            ignore_index=True,
        )
        self.hist_actions_prob = pd.concat(
            [self.hist_actions_prob,
             pd.DataFrame([action_prob.tolist()], columns=self.actions_list)],
            ignore_index=True,
        )

    def _print_step(
        self,
        true_label:     int,
        pred_label:     int,
        belief:         np.ndarray,
        current_action: str,
    ) -> None:
        """Print a concise summary line for the current step."""
        disc         = self.physical_asset.config.discretization
        true_dmg     = self.physical_asset.state_continuous
        pred_dmg     = pred_label  * disc
        belief_label = int(np.argmax(belief))
        belief_dmg   = belief_label * disc

        print(f"  True:    {true_dmg:5.1f} %  (label {true_label})")
        print(f"  CNN:     {pred_dmg:5.1f} %  (label {pred_label})")
        print(f"  Belief:  {belief_dmg:5.1f} %  (label {belief_label})")
        print(f"  Action:  {current_action}")