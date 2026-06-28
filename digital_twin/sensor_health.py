"""
digital_twin/sensor_health.py
=============================
Sensor-health layer for the drive-by Digital Twin.

The drive-by sensors are themselves fallible components. This layer models that,
and — crucially — models how the twin COPES with it, which is the real version of
the "the DBN protects us against bad data" claim (it must be modelled, not
assumed). Three pieces (see memory: sensor-health-model):

1. A per-sensor health process (`SensorHealthModel`): each active sensor is
   WORKING / DEGRADED / DEAD. Failure modes mapped to reality —
     * transient / intermittent : a noise BURST in the middle of one passage
                                   (loose connection, EMI, debris);
     * degraded                 : the WHOLE signal is noisier (aging / drift);
     * dead                     : registers nothing -> the channel is zeros.
   Fault probability and noise scale by mounting exposure: car body (sprung,
   enclosed) < bogie < wheel/axle-box (unsprung, exposed). Transient faults are
   a per-passage Bernoulli; permanent failure is a slow per-step aging hazard.

2. How the twin copes (`HealthAwareClassifier`):
     * both sensors alive  -> the 2-sensor champion + its confusion matrix;
     * one sensor dead     -> FALL BACK to the surviving single-DOF champion and
                              ITS (wider) confusion matrix, so the belief update
                              automatically distrusts the lower-quality estimate;
     * no sensor alive     -> no observation this step (the belief just predicts
                              forward until a sensor is replaced).
   A cheap inspect-/replace-sensor action (<< a bridge repair) restores health.

3. A tiny € accounting for sensor maintenance, fed back into the life-cycle cost.

ALL probabilities / noise scales / costs below are PLACEHOLDERS — reasonable
order-of-magnitude defaults to be replaced with vendor MTBF / field data. They
are deliberately collected in `SensorHealthParams` so they can be swept.

Pure NumPy: runs in the analysis env. The classifier routing needs the torch
DigitalAssets (library/live mode); `MockHealthClassifier` mirrors the routing
with banded matrices for fast, torch-free testing of the whole loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.utils import IDX_TO_DOF_NAME

# Health states.
WORKING  = "working"
DEGRADED = "degraded"
DEAD     = "dead"


def sensor_location(dof: int) -> str:
    """Mounting location of a DOF — drives the fault/noise scaling.

    car body (sprung, enclosed, protected) -> lowest fault prob + noise;
    bogie intermediate; wheel/axle-box (unsprung, direct wheel-rail impacts,
    exposed) -> highest.
    """
    name = IDX_TO_DOF_NAME[dof].lower()
    if "carbody" in name:
        return "carbody"
    if "wheel" in name:
        return "wheel"
    if "bogie" in name:
        return "bogie"
    return "bogie"


def _is_gyro(dof: int) -> bool:
    """Pitch-rate DOFs are gyroscopes (optionally a touch more fault/drift)."""
    return "pitch" in IDX_TO_DOF_NAME[dof].lower()


@dataclass
class SensorHealthParams:
    """Tunable sensor-health knobs (PLACEHOLDERS — replace with real data).

    Per-location dicts scale by mounting exposure (carbody < bogie < wheel).
    Illustrative defaults; the whole struct is meant to be swept / overridden.
    """
    # Per-PASSAGE transient (intermittent) fault probability.
    p_transient: dict = field(default_factory=lambda:
                              {"carbody": 0.010, "bogie": 0.020, "wheel": 0.040})
    # Per-STEP permanent aging hazards.
    p_degrade:   dict = field(default_factory=lambda:
                              {"carbody": 0.002, "bogie": 0.004, "wheel": 0.008})
    p_dead:      dict = field(default_factory=lambda:
                              {"carbody": 0.001, "bogie": 0.003, "wheel": 0.006})
    # Injected noise as a multiple of the channel's own std.
    noise_degraded:  dict = field(default_factory=lambda:
                                  {"carbody": 1.0, "bogie": 2.0, "wheel": 3.0})
    noise_transient: dict = field(default_factory=lambda:
                                  {"carbody": 2.0, "bogie": 3.0, "wheel": 5.0})
    # Degraded-mode PERSISTENT errors (drawn once when a sensor degrades; EN 61373
    # / VRE / thermal-drift literature). bias_degraded = std of the DC bias offset
    # as a fraction of the channel std (VRE on the bogie, thermal bias on the car
    # body); scale_factor_degraded = std of the multiplicative gain error.
    # NOTE: the classifier preprocessor standardises each channel (subtract mean,
    # divide by std), which largely REMOVES a constant bias and a constant scale
    # factor — so for our PAA pipeline the degraded *noise* term dominates the
    # classifier impact; bias/scale matter for raw-signal pipelines and physical
    # fidelity, and are kept so a re-user with a different preprocessor is served.
    bias_degraded:   dict = field(default_factory=lambda:
                                  {"carbody": 0.5, "bogie": 1.0, "wheel": 2.0})
    scale_factor_degraded: dict = field(default_factory=lambda:
                                  {"carbody": 0.05, "bogie": 0.10, "wheel": 0.20})
    transient_frac: float = 0.25     # fraction of the passage hit by a burst
    gyro_factor:    float = 1.0      # extra fault/noise factor for gyro (pitch) DOFs
    # Sensor maintenance costs [€] — CHEAP vs a bridge repair (~6e5 €).
    c_inspect_sensor: float = 5.0e2
    c_replace_sensor: float = 5.0e3
    # Replacement policy: "on_dead" (swap a sensor once it dies),
    # "on_faulty" (swap as soon as degraded), or "never".
    replace_policy: str = "on_dead"


class SensorHealthModel:
    """Per-sensor health state machine + signal fault injector.

    Tracks the health of each active drive-by DOF, advances a slow aging hazard
    each monitoring step, injects the three failure modes into a raw (8, L)
    signal, and applies a replacement policy with € accounting.
    """

    def __init__(self, active_dofs, params: SensorHealthParams | None = None,
                 rng: np.random.Generator | None = None):
        self.active_dofs = list(active_dofs)
        self.p = params or SensorHealthParams()
        self.rng = rng or np.random.default_rng()
        self.state = {d: WORKING for d in self.active_dofs}
        self._pending_cost = 0.0
        self.n_replacements = 0
        # Persistent degraded-mode errors per sensor (drawn lazily on first use
        # after a sensor degrades; a degraded sensor carries a stable bias / gain
        # error, not a fresh one each passage). Cleared on replace().
        self._deg_bias: dict = {}
        self._deg_sf: dict = {}

    # ── aging ─────────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance permanent aging by one monitoring step (call once per step)."""
        for d in self.active_dofs:
            if self.state[d] == DEAD:
                continue
            loc = sensor_location(d)
            g = self.p.gyro_factor if _is_gyro(d) else 1.0
            if self.rng.random() < min(1.0, self.p.p_dead[loc] * g):
                self.state[d] = DEAD
            elif self.state[d] == WORKING and \
                    self.rng.random() < min(1.0, self.p.p_degrade[loc] * g):
                self.state[d] = DEGRADED

    # ── fault injection ─────────────────────────────────────────────────────────

    def corrupt(self, signal: np.ndarray) -> np.ndarray:
        """Return a copy of the (8, L) signal with faults injected into the
        ACTIVE channels per their current health and a fresh transient draw."""
        sig = np.array(signal, dtype=np.float32, copy=True)
        L = sig.shape[1]
        for d in self.active_dofs:
            loc = sensor_location(d)
            g = self.p.gyro_factor if _is_gyro(d) else 1.0
            std = float(np.std(sig[d])) or 1.0
            if self.state[d] == DEAD:
                sig[d] = 0.0
                continue
            if self.state[d] == DEGRADED:
                # persistent gain error + DC bias (drawn once), + fresh noise.
                if d not in self._deg_sf:
                    self._deg_sf[d] = 1.0 + self.rng.normal(
                        0.0, self.p.scale_factor_degraded[loc] * g)
                    self._deg_bias[d] = self.rng.normal(
                        0.0, self.p.bias_degraded[loc] * g * std)
                noise = self.rng.normal(0.0, self.p.noise_degraded[loc] * g * std, size=L)
                sig[d] = self._deg_sf[d] * sig[d] + self._deg_bias[d] + noise
            # transient burst — independent of the degraded state, alive only
            if self.rng.random() < min(1.0, self.p.p_transient[loc] * g):
                w = max(1, int(self.p.transient_frac * L))
                start = int(self.rng.integers(0, max(1, L - w)))
                sig[d, start:start + w] += self.rng.normal(
                    0.0, self.p.noise_transient[loc] * g * std, size=w)
        return sig

    # ── maintenance (replacement policy + costs) ────────────────────────────────

    def maintain(self) -> None:
        """Apply the replacement policy and accrue its € cost (call after the
        observation so this step's fallback/likelihood reflects the fault)."""
        policy = self.p.replace_policy
        for d in self.active_dofs:
            need = (policy == "on_dead" and self.state[d] == DEAD) or \
                   (policy == "on_faulty" and self.state[d] != WORKING)
            if need:
                self.replace(d)

    def replace(self, dof: int) -> None:
        """Swap a sensor back to WORKING and charge the replacement cost."""
        self.state[dof] = WORKING
        self._deg_bias.pop(dof, None)        # a new unit has no degraded errors
        self._deg_sf.pop(dof, None)
        self._pending_cost += self.p.c_replace_sensor
        self.n_replacements += 1

    def inspect(self) -> None:
        """Charge a sensor-inspection cost (a health check; no state change)."""
        self._pending_cost += self.p.c_inspect_sensor

    def flush_cost(self) -> float:
        """Return the sensor-maintenance € accrued since the last call, reset to 0."""
        c, self._pending_cost = self._pending_cost, 0.0
        return c

    # ── queries ─────────────────────────────────────────────────────────────────

    def alive(self) -> list[int]:
        return [d for d in self.active_dofs if self.state[d] != DEAD]

    def dead(self) -> list[int]:
        return [d for d in self.active_dofs if self.state[d] == DEAD]

    def any_faulty(self) -> bool:
        return any(s != WORKING for s in self.state.values())

    def status(self) -> dict:
        return {IDX_TO_DOF_NAME[d]: self.state[d] for d in self.active_dofs}

    def status_code(self) -> str:
        """Compact per-sensor code for logging, e.g. 'RearBogie_Vert:working'."""
        return ",".join(f"{IDX_TO_DOF_NAME[d]}:{self.state[d]}" for d in self.active_dofs)


class HealthAwareClassifier:
    """Route a (corrupted) 8-channel signal to the right classifier given which
    sensors are alive; return (label, likelihood_matrix).

    Args:
        champion      : DigitalAsset for the full sensor set.
        champion_like : row-normalised confusion matrix of the champion.
        fallbacks     : {dof: DigitalAsset} single-DOF champions.
        fallback_like : {dof: row-normalised confusion matrix}.
    """

    def __init__(self, champion, champion_like, fallbacks, fallback_like):
        self.champion = champion
        self.champion_set = set(champion.active_dofs)
        self.champion_like = champion_like
        self.fallbacks = dict(fallbacks)
        self.fallback_like = dict(fallback_like)

    def classify(self, signal8: np.ndarray, alive_dofs):
        """Return (predicted_label, likelihood) or (None, None) if unobservable."""
        alive = set(alive_dofs)
        if not alive:
            return None, None
        if alive == self.champion_set:
            return self.champion.estimate_state(signal8), self.champion_like
        # one (or more) sensor down -> use a surviving single-DOF model we have
        for d in alive_dofs:
            if d in self.fallbacks:
                return self.fallbacks[d].estimate_state(signal8), self.fallback_like[d]
        return None, None      # no model for the surviving set -> no usable obs


class MockHealthClassifier:
    """Torch-free stand-in for HealthAwareClassifier (mock mode / testing).

    Mirrors the routing using banded confusion matrices: a sharp one for the full
    set, a wider one per single-DOF fallback (worse sensor -> wider -> the belief
    update distrusts it). Samples a predicted label from the matching row.
    """

    def __init__(self, champion_dofs, n, champion_sigma=0.7, fallback_sigma=1.6,
                 rng=None):
        from digital_twin.harness import banded_like
        self.champion_set = set(champion_dofs)
        self.n = n
        self.rng = rng or np.random.default_rng()
        self.champion_like = banded_like(n, champion_sigma)
        self.fallback_like = banded_like(n, fallback_sigma)

    def classify(self, true_label: int, alive_dofs):
        alive = set(alive_dofs)
        if not alive:
            return None, None
        L = self.champion_like if alive == self.champion_set else self.fallback_like
        pred = int(self.rng.choice(self.n, p=L[int(true_label)]))
        return pred, L


def health_observe(library, classifier, health: SensorHealthModel, rng):
    """Drive-by observation adapter WITH sensor health, for DTSimulator.

    Each call: age the sensors, sample + corrupt a held-out signal, classify with
    the health-aware router, then apply the replacement policy. Returns
    (label, likelihood) — the simulator uses the returned likelihood so a fallback
    or degraded reading is automatically distrusted; (None, None) means no usable
    observation this step.
    """
    def observe(phys):
        health.step()
        sig = health.corrupt(library.sample(phys.state_continuous, rng))
        label, like = classifier.classify(sig, health.alive())
        health.maintain()
        return label, like
    return observe


def mock_health_observe(classifier: MockHealthClassifier, health: SensorHealthModel):
    """health_observe equivalent for mock mode (true label -> banded sampling)."""
    def observe(phys):
        health.step()
        label, like = classifier.classify(phys.get_true_mapped_label(), health.alive())
        health.maintain()
        return label, like
    return observe
