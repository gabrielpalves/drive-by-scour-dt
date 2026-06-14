"""
digital_twin/signal_library.py
==============================
Held-out drive-by signal library for the online Digital Twin.

Running one TTBI passage live takes minutes, and re-simulating in Python also
introduces train/serve skew (the classifier was trained on the MATLAB pipeline).
The fix is to pre-generate a HELD-OUT library of passages — produced by the same
(MATLAB) pipeline as the training data but with UNSEEN random seeds — and, during
the DT loop, simply SAMPLE a signal for the current damage state. This removes
TTBI from the online loop entirely and makes fine (monthly) time-stepping cheap.

Expected dataset format (same as the training data, so it plugs straight in)
---------------------------------------------------------------------------
A folder of MATLAB files, one per damage state, each holding several passages:

    held_out/
        0001.mat, 0002.mat, ...        # one file per damage state
    mat['data'][0,0] has fields AcelPrimVag, AcelRodaPrimVag, PitchPrimVag,
    each shaped (1, n_passages) of cells; cell p is (rows, L) with the DOF on
    the row (see _DOF_SOURCE below).

Damage values: integer 0..60 % by file index by default (file k -> k %), OR pass
`damage_values` for a continuous grid (e.g. [0.0, 0.67, 1.39, ...]) — continuous
states are more realistic for the DT, and only a few passages per state are
needed (this is a sampling pool, not a training set).

Usage
-----
    lib = SignalLibrary.from_mat_folder("held_out", n_passages=10)
    signal = lib.sample(damage_pct=23.4)      # nearest state, random passage → (8, L)
"""

from __future__ import annotations

import os

import numpy as np
import scipy.io as sio

# DOF index -> (MATLAB field, row) — mirrors core.dataset.load_ttbi_dataset.
_DOF_SOURCE = {
    0: ("AcelPrimVag", 0), 1: ("AcelPrimVag", 1), 2: ("AcelPrimVag", 2),
    3: ("AcelRodaPrimVag", 0), 4: ("AcelRodaPrimVag", 1),
    5: ("PitchPrimVag", 0), 6: ("PitchPrimVag", 1), 7: ("PitchPrimVag", 2),
}


class SignalLibrary:
    """A pool of drive-by passages indexed by true damage state (%).

    Stores full 8-channel signals so DigitalAsset can select its active DOFs.
    Signals may have different lengths (speed varies) — kept as a list per state.
    """

    def __init__(self, states_pct: np.ndarray, signals: list[np.ndarray]):
        self.states = np.asarray(states_pct, dtype=float)   # (N,)
        self.signals = signals                              # list of (8, L)
        self.unique_states = np.unique(self.states)

    # ── constructors ────────────────────────────────────────────────────────────

    @classmethod
    def from_mat_folder(
        cls,
        folder: str,
        n_passages: int = 10,
        damage_values: list[float] | None = None,
        n_states: int = 61,
    ) -> "SignalLibrary":
        """Load a folder of NNNN.mat files (same format as the training data).

        folder        : path to the held-out dataset folder.
        n_passages    : passages to load per state (a small pool is enough).
        damage_values : damage % for each file in order; default 0..n_states-1.
        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Signal library folder not found: {folder}")
        states_pct, signals = [], []
        dofs = list(range(8))
        idx = 0
        while True:
            fp = os.path.join(folder, f"{idx + 1:04d}.mat")
            if not os.path.exists(fp):
                break
            d = sio.loadmat(fp)["data"][0, 0]
            names = d.dtype.names or ()
            # Prefer the true scour value saved in the file (A00 stores data.Dano
            # as a fraction 0-0.60 -> %); else explicit damage_values; else index.
            if damage_values is not None:
                dmg = float(damage_values[idx])
            elif "Dano" in names:
                dmg = float(np.ravel(d["Dano"])[0]) * 100.0
            else:
                dmg = float(idx)
            avail = d["AcelPrimVag"].shape[1]
            for p in range(min(n_passages, avail)):
                chans = [d[_DOF_SOURCE[k][0]][0, p][_DOF_SOURCE[k][1], :] for k in dofs]
                signals.append(np.vstack(chans).astype(np.float32))   # (8, L)
                states_pct.append(dmg)
            idx += 1
            if damage_values is None and idx >= n_states:
                break
        if not signals:
            raise RuntimeError(f"No passages loaded from {folder}")
        print(f"SignalLibrary: {len(signals)} passages across "
              f"{len(set(states_pct))} states from {folder}")
        return cls(np.array(states_pct), signals)

    # ── sampling ────────────────────────────────────────────────────────────────

    def sample(self, damage_pct: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return one random passage (8, L) from the state nearest to damage_pct."""
        rng = rng or np.random.default_rng()
        nearest = self.unique_states[np.argmin(np.abs(self.unique_states - damage_pct))]
        pool = np.where(self.states == nearest)[0]
        return self.signals[int(rng.choice(pool))]

    def states_available(self) -> np.ndarray:
        return self.unique_states


def library_observe(library: SignalLibrary, digital, rng=None):
    """Drive-by observation adapter for DTSimulator: sample a held-out signal for
    the asset's current true damage and classify it. Same signature the simulator
    expects (physical -> predicted label)."""
    rng = rng or np.random.default_rng()
    def observe(phys):
        return int(digital.estimate_state(library.sample(phys.state_continuous, rng)))
    return observe
