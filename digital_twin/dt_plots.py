"""
digital_twin/dt_plots.py
========================
Visualisation of the drive-by digital twin simulation results.

The Plot class accepts a completed Graph instance and saves two multi-panel
figures:

    plot_history_all_together      — four-panel view of what happened during
                                     the simulation: physical damage trajectory,
                                     classifier accuracy, DBN belief heatmap,
                                     and action history.

    plot_prediction_all_together   — two-panel view of where the DBN expects
                                     the bridge to go from the current moment:
                                     future belief distribution and the implied
                                     action probability.

Design notes
------------
All figures are saved to disk and closed immediately (plt.close()) to
prevent RAM accumulation when this class is called in a loop or notebook.
No figure is held open — the caller is responsible for calling plt.show()
if interactive display is needed.

Imported by:
    drive_by_DT.py — Plot (instantiated after graph_frame.simulate() returns)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


class Plot:
    """
    Plotting helper for one completed Graph simulation.

    Args:
        graph (Graph):    The completed Graph instance, with hist_var,
                          hist_d_state_prob, and hist_actions_prob populated.
        output_dir (str): Directory where PNG files are written.
                          Created if it does not exist.
    """

    # Colour palette — consistent across all panels
    _C_TRUE    = '#2C3E50'   # dark slate  — physical ground truth
    _C_CNN     = '#E67E22'   # amber       — raw classifier output
    _C_BELIEF  = '#2980B9'   # steel blue  — DBN MAP belief
    _C_REPAIR  = '#E74C3C'   # red         — repair action
    _C_NOTHING = '#95A5A6'   # light grey  — do-nothing action

    def __init__(self, graph, output_dir: str):
        self.graph      = graph
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Convenience aliases
        self.hist        = graph.hist_var
        self.hist_belief = graph.hist_d_state_prob
        self.hist_action = graph.hist_actions_prob
        self.states      = graph.states_list
        self.actions     = graph.actions_list
        self.config      = graph.physical_asset.config

        self.n_steps  = len(self.hist)
        self.timesteps = np.arange(self.n_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. History figure
    # ──────────────────────────────────────────────────────────────────────────

    def plot_history_all_together(self, filename: str = 'dt_history.png') -> str:
        """
        Four-panel figure covering the full simulation history.

        Panel 1 — Damage trajectory
            Physical true damage (continuous %), DBN MAP estimate, and raw
            CNN label, all plotted in physical damage-% units.  Repair
            events are marked as vertical dashed lines.

        Panel 2 — Classifier error
            Absolute difference in damage-% between the CNN prediction and
            the true state at each step.  Helps identify monitoring steps
            where the sensor was particularly unreliable.

        Panel 3 — Belief heatmap
            Full posterior distribution p(D_t | y_{1:t}) over all damage
            classes at every time step.  Colour intensity encodes probability
            mass.  The MAP label is overlaid as a white line.

        Panel 4 — Action history
            Stacked bar showing the do-nothing (grey) and repair (red)
            action probabilities at each step.  Deterministic policy produces
            bars of 0 or 1; a probabilistic policy produces gradients.

        Args:
            filename (str): Output filename inside output_dir.

        Returns:
            str: Absolute path of the saved figure.
        """
        disc = self.config.discretization

        # Convert label columns to physical damage %
        p_dmg    = self.hist['p_state'].to_numpy()
        cnn_dmg  = self._labels_to_damage(self.hist['d_state_nn'])
        belief_map_dmg = self._labels_to_damage(self.hist['d_state'])

        # Repair step indices
        repair_steps = self.hist.index[
            self.hist['current_action'] == 'perfect_repair'
        ].tolist()

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        fig.suptitle('Drive-by Digital Twin — Simulation History',
                     fontsize=16, fontweight='bold', y=1.01)

        # ── Panel 1: Damage trajectory ────────────────────────────────────────
        ax = axes[0]
        ax.plot(self.timesteps, p_dmg,        color=self._C_TRUE,   lw=2.0,
                label='True damage (physical)')
        ax.plot(self.timesteps, belief_map_dmg, color=self._C_BELIEF, lw=1.8,
                linestyle='--', label='DBN MAP estimate')
        ax.plot(self.timesteps, cnn_dmg,       color=self._C_CNN,    lw=1.2,
                linestyle=':',  alpha=0.8, label='CNN prediction')

        for s in repair_steps:
            ax.axvline(s, color=self._C_REPAIR, lw=1.2, linestyle='--', alpha=0.6)

        ax.set_ylabel('Damage (%)', fontweight='bold')
        ax.set_ylim(-2, self.config.max_damage + 5)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('Damage trajectory', fontsize=12, pad=6)

        # ── Panel 2: Classifier error ─────────────────────────────────────────
        ax = axes[1]
        abs_err = np.abs(cnn_dmg - p_dmg)
        ax.fill_between(self.timesteps, abs_err, color=self._C_CNN, alpha=0.4)
        ax.plot(self.timesteps, abs_err, color=self._C_CNN, lw=1.2)

        ax.axhline(np.mean(abs_err), color='black', lw=1.0, linestyle='--',
                   label=f'Mean error: {np.mean(abs_err):.1f} %')
        ax.set_ylabel('|CNN error| (%)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('Classifier absolute error', fontsize=12, pad=6)

        # ── Panel 3: Belief heatmap ────────────────────────────────────────────
        ax = axes[2]
        belief_matrix = self.hist_belief.to_numpy().T   # (n_states, n_steps)

        sns.heatmap(
            belief_matrix,
            ax=ax,
            cmap='Blues',
            xticklabels=False,
            yticklabels=self.states,
            cbar_kws={'label': 'Probability', 'shrink': 0.8},
            vmin=0, vmax=1,
        )

        # Overlay MAP label as a white line
        map_labels = belief_matrix.argmax(axis=0)
        ax.plot(self.timesteps + 0.5, map_labels + 0.5,
                color='white', lw=1.5, linestyle='--', label='MAP label')

        ax.set_ylabel('Damage label', fontweight='bold')
        ax.set_title('Posterior belief distribution p(D_t | observations)',
                     fontsize=12, pad=6)
        ax.invert_yaxis()

        # ── Panel 4: Action history ────────────────────────────────────────────
        ax = axes[3]
        do_nothing_prob = self.hist_action[self.actions[0]].to_numpy()
        repair_prob     = self.hist_action[self.actions[1]].to_numpy()

        ax.bar(self.timesteps, do_nothing_prob, color=self._C_NOTHING,
               label='Do nothing', width=0.8)
        ax.bar(self.timesteps, repair_prob, bottom=do_nothing_prob,
               color=self._C_REPAIR, label='Perfect repair', width=0.8)

        ax.set_ylabel('Action probability', fontweight='bold')
        ax.set_xlabel('Monitoring step (years)', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('Maintenance action history', fontsize=12, pad=6)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"History figure saved → {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Prediction figure
    # ──────────────────────────────────────────────────────────────────────────

    def plot_prediction_all_together(
        self,
        n_steps:   int = 20,
        n_samples: int = 1_000,
        filename:  str = 'dt_prediction.png',
    ) -> str:
        """
        Two-panel figure projecting the DBN n_steps into the future from
        the current belief.

        Panel 1 — Future damage belief heatmap
            p(D_{t+k} | y_{1:t}) for k = 0, …, n_steps.  The probability
            mass is expected to spread and drift toward higher-damage states
            as uncertainty accumulates and the degradation law advances the
            state.

        Panel 2 — Future action probabilities
            The marginal probability of each action at each future step
            under the threshold policy.  Shows when, in expectation, the
            bridge is predicted to require repair.

        Args:
            n_steps (int):   Prediction horizon in monitoring steps.
            n_samples (int): Monte Carlo samples for the DBN simulation.
            filename (str):  Output filename inside output_dir.

        Returns:
            str: Absolute path of the saved figure.
        """
        print(f"\nRunning {n_steps}-step forward prediction ({n_samples} samples)...")
        result_D, result_U = self.graph.predict(n_steps=n_steps, n_samples=n_samples)

        horizon = np.arange(n_steps + 1)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f'Drive-by Digital Twin — {n_steps}-step Forward Prediction\n'
            f'(from monitoring step {self.graph.time_index})',
            fontsize=15, fontweight='bold', y=1.01,
        )

        # ── Panel 1: Future belief heatmap ────────────────────────────────────
        ax = axes[0]
        pred_matrix = result_D.to_numpy().T     # (n_states, n_steps+1)

        sns.heatmap(
            pred_matrix,
            ax=ax,
            cmap='Oranges',
            xticklabels=False,
            yticklabels=self.states,
            cbar_kws={'label': 'Probability', 'shrink': 0.8},
            vmin=0, vmax=1,
        )

        # MAP label trajectory
        map_labels = pred_matrix.argmax(axis=0)
        ax.plot(horizon + 0.5, map_labels + 0.5,
                color='white', lw=2.0, linestyle='--', label='MAP label')

        ax.set_ylabel('Damage label', fontweight='bold')
        ax.set_title('Predicted future damage belief', fontsize=12, pad=6)
        ax.invert_yaxis()

        # ── Panel 2: Future action probabilities ──────────────────────────────
        ax = axes[1]
        do_nothing_pred = result_U[self.actions[0]].to_numpy()
        repair_pred     = result_U[self.actions[1]].to_numpy()

        ax.bar(horizon, do_nothing_pred, color=self._C_NOTHING,
               label='Do nothing', width=0.8)
        ax.bar(horizon, repair_pred, bottom=do_nothing_pred,
               color=self._C_REPAIR, label='Perfect repair', width=0.8)

        ax.set_ylabel('Action probability', fontweight='bold')
        ax.set_xlabel('Steps ahead (years)', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('Predicted maintenance actions', fontsize=12, pad=6)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Prediction figure saved → {path}")
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience export
    # ──────────────────────────────────────────────────────────────────────────

    def save_history_csv(self) -> str:
        """
        Write the three history dataframes to CSV as a fallback when
        plotting is unavailable or as a companion to the figures.

        Files written:
            dt_history.csv       — scalar state and action columns.
            dt_belief_history.csv — full belief distribution per step.
            dt_action_history.csv — full action distribution per step.

        Returns:
            str: The output directory path.
        """
        self.hist.to_csv(os.path.join(self.output_dir, 'dt_history.csv'),      index=False)
        self.hist_belief.to_csv(os.path.join(self.output_dir, 'dt_belief_history.csv'), index=False)
        self.hist_action.to_csv(os.path.join(self.output_dir, 'dt_action_history.csv'), index=False)
        print(f"History CSVs saved → {self.output_dir}")
        return self.output_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _labels_to_damage(self, label_series: pd.Series) -> np.ndarray:
        """
        Convert a Series of string damage labels to physical damage percentages.

        Uses config.label_to_damage so the conversion stays consistent with
        the config's discretisation step rather than hardcoding the formula.

        Args:
            label_series: Series of string labels, e.g. ['0', '6', '12', …].

        Returns:
            np.ndarray: float, same length, values in [0, max_damage].
        """
        return np.array([
            self.config.label_to_damage(int(lbl))
            for lbl in label_series
        ])
