"""
digital_twin/assets.py
======================
The three asset classes that make up the digital twin framework:

    ScourModel    — stochastic support-stiffness-loss scenario model.

    PhysicalAsset — wrapper around ScourModel that drives the physics engine
                    and tracks the true continuous damage state.

    DigitalAsset  — loads the offline-trained classifier and performs the
                    full online inference pipeline: channel selection →
                    preprocessing → scaling → prediction.

Bug fixes relative to the original drive_by_DT.py
---------------------------------------------------
1. PhysicalAsset.__init__ received `degradation_law` as an argument but
   never stored it, so update_physical_state() raised AttributeError on
   the `self.degradation_law` reference.  Fixed by storing it.

2. DigitalAsset.estimate_state referenced `signal_fft` after renaming the
   variable to `signal_processed`, causing an immediate NameError on every
   inference call.  Fixed by using the correct name throughout.

3. DigitalAsset used the retired fft_preprocess() function.  Replaced by
   TTBIPreprocessor, which handles all preprocessing methods (raw, PAA, FFT,
   CWT) consistently with the ablation training pipeline.

4. DigitalAsset received raw_signal of shape (8, L) but passed it to the
   scaler as a flat 1-D vector, which breaks for multi-channel models.
   Fixed: the active channel indices (legacy metadata key ``active_dofs``) are
   selected first, then transform() is called with the correct
   (1, n_channels, L) shape.

Imported by:
    drive_by_DT.py — all three classes
"""

import numpy as np
import torch

from core.artifact_provenance import verify_standalone_dt_package
from core.models        import build_model
from core.preprocessing import TTBIPreprocessor
from core.utils         import dof_label
from digital_twin.physics import run_single_passage


# Re-export DEVICE from trainer so drive_by_DT.py has one import location.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monitoring cadence — years between drive-by passages.
# Matches the default in DTConfig and the DBN transition prior.
_DEFAULT_MONITORING_INTERVAL = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stochastic degradation model
# ──────────────────────────────────────────────────────────────────────────────

class ScourModel:
    """
    Gradual support-stiffness-loss scenario with an optional flood shock.

    State variable X(t) evolves as a power-law process with log-normal
    amplitude A and normally-distributed exponent B, both sampled fresh
    after each repair.  A multiplicative noise term omega_k is drawn at
    every time step to capture inter-annual variability.

    The continuous state X(t) is mapped to the [0, 60] percentage-point design
    scale used by the estimator via get_current_damage_case(). It is a modeled
    loss of vertical foundation-support stiffness, not scour depth. The flood
    rate and jump parameters below are author-chosen placeholders pending
    primary-source or site-specific calibration; this class must not be
    described as a validated deterioration law.

    The model is intentionally self-contained: it carries no reference to
    the classifier, the DBN, or any plotting utility.
    """

    DAMAGE_MAX: float = 60.0   # campaign design ceiling, not a physical limit

    # Generic structural-deterioration prior from Kamariotis et al. (2023),
    # Table 1. These A/B/omega values reproduce that paper's illustrative
    # prior; they were not fitted to hydraulic scour or support-stiffness loss.
    # The central-difference update below follows its Eqs. (17) and (19).
    _A_MEAN: float = 1.94e-4;  _A_COV: float  = 0.40
    _B_MEAN: float = 2.0;      _B_COV: float  = 0.10
    _OMEGA_MEAN: float = -0.005; _OMEGA_COV: float = 0.10

    # SHOCK family: homogeneous Compound Poisson Process, as in Kamariotis
    # Eq. (17). The numerical values below are deliberately project placeholders,
    # not that paper's Table 1 values (lambda=0.04/year, jump mean=3.75,
    # jump COV=0.25) and not a fitted flood/scour model. They are environmental
    # scenario properties and are therefore not resampled on repair.
    _LAMBDA_FLOOD: float = 0.10   # flood occurrence rate [events/year]
    _JUMP_MEAN:    float = 5.0    # mean scour increment per flood [%]
    _JUMP_COV:     float = 0.60   # COV of the jump magnitude

    def __init__(self, enable_shock: bool = False):
        self.enable_shock: bool = enable_shock
        self.current_X: float = 0.0
        self.time:      float = 0.0
        # Diagnostics for the decision layer (e.g. "a flood just happened").
        self.last_flood_count:     int   = 0
        self.last_shock_magnitude: float = 0.0
        # Dimensionless event severity = shock magnitude / mean jump (the latent
        # driver the river gauge observes; see digital_twin.flood.FloodTrigger).
        self.last_severity:        float = 0.0
        self._sample_parameters()

    # ── Public interface ──────────────────────────────────────────────────────

    def evolve(self, delta_t: float) -> float:
        """
        Advance the scour state by one time step of length delta_t (years).

        Uses the mid-point of the interval for the rate evaluation to reduce
        numerical integration error over coarse yearly steps.

        Returns:
            float: Current damage in [0, DAMAGE_MAX] %.
        """
        # ── Generic gradual-process discretization (Kamariotis Eqs. 17, 19) ──
        t_mid          = self.time + delta_t / 2.0
        gradual_rate   = self._A * self._B * (t_mid ** (self._B - 1.0))

        omega_sigma    = abs(self._OMEGA_MEAN * self._OMEGA_COV)
        omega_k        = np.random.normal(loc=self._OMEGA_MEAN, scale=omega_sigma)

        self.current_X += gradual_rate * delta_t * np.exp(omega_k)

        # ── Shock deterioration (Eq. 17, second term: Compound Poisson) ────────
        # Number of floods in this interval ~ Poisson(λ·Δt); each adds an i.i.d.
        # lognormal scour jump. Captures the "flood event" the DT must react to.
        self.last_flood_count     = 0
        self.last_shock_magnitude = 0.0
        self.last_severity        = 0.0
        if self.enable_shock:
            n_floods = int(np.random.poisson(self._LAMBDA_FLOOD * delta_t))
            if n_floods > 0:
                j_sigma = np.sqrt(np.log(self._JUMP_COV ** 2 + 1.0))
                j_mu    = np.log(self._JUMP_MEAN) - 0.5 * j_sigma ** 2
                jumps   = np.random.lognormal(mean=j_mu, sigma=j_sigma, size=n_floods)
                shock   = float(jumps.sum())
                self.current_X           += shock
                self.last_flood_count     = n_floods
                self.last_shock_magnitude = shock
                self.last_severity        = shock / self._JUMP_MEAN

        self.time += delta_t

        return self.get_current_damage_case()

    def flood_occurred_last_step(self) -> bool:
        """True if one or more floods (shocks) occurred in the last evolve()."""
        return self.last_flood_count > 0

    def flood_severity(self) -> float:
        """Dimensionless severity of the last flood (0 if none); gauge driver."""
        return self.last_severity

    def get_current_damage_case(self) -> float:
        """Map internal state X(t) to damage % capped at DAMAGE_MAX."""
        return min(self.current_X, self.DAMAGE_MAX)

    def repair(self) -> None:
        """Reset to 'as-new' condition and resample stochastic parameters."""
        self.current_X = 0.0
        self.time      = 0.0
        self._sample_parameters()
        print("  -> PHYSICAL ACTION: bridge repaired. Damage reset to 0 %.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _sample_parameters(self) -> None:
        """Draw A (log-normal) and B (normal) from their prior distributions."""
        A_sigma  = np.sqrt(np.log(self._A_COV ** 2 + 1.0))
        A_mu     = np.log(self._A_MEAN) - 0.5 * A_sigma ** 2
        self._A  = np.random.lognormal(mean=A_mu, sigma=A_sigma)

        self._B  = np.random.normal(
            loc=self._B_MEAN,
            scale=self._B_MEAN * self._B_COV,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Physical asset
# ──────────────────────────────────────────────────────────────────────────────

class PhysicalAsset:
    """
    Represents the physical bridge and its true damage state.

    Wraps ScourModel for degradation and calls run_single_passage for each
    drive-by observation.  Variability in speed, temperature, and vehicle
    properties is sampled here so the physics engine itself remains
    deterministic given its inputs.

    Args:
        config:           DTConfig instance — supplies n_veh, n_prop,
                          discretization, and n_classes.
        degradation_law:  String tag for the degradation model.
                          Currently only 'scour_gradual' is implemented;
                          the argument is stored for future extensibility.
        monitoring_interval (float): Years between observations.
    """

    def __init__(
        self,
        config,
        degradation_law:      str   = 'scour_gradual',
        monitoring_interval:  float = _DEFAULT_MONITORING_INTERVAL,
    ):
        self.config               = config
        self.degradation_law      = degradation_law      # bug-fix: was never stored
        self.monitoring_interval  = monitoring_interval
        self.scour_model          = ScourModel(
            enable_shock=getattr(config, 'enable_shock', False)
        )
        self.state_continuous:    float = 0.0
        self.time:                float = 0.0
        # Set True whenever a flood (shock) occurred in the last state update —
        # the trigger for the "inspect now vs wait for next train" decision.
        self.flood_this_step:     bool  = False
        # Severity of that flood (0 when none); the river-gauge observes a noisy
        # version of it (digital_twin.flood).
        self.flood_severity:      float = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def get_observation_signal(self) -> np.ndarray:
        """
        Run a single TTBI passage at the current damage state and return all
        eight response channels (six vehicle, two virtual moving-rail).

        Variability sources (speed, temperature, vehicle properties) are
        sampled fresh on every call to reflect real-world measurement noise.

        Returns:
            np.ndarray: float32, shape (8, sequence_length).
        """
        # state_continuous is stored in percentage points. TTBI consumes the
        # dimensionless loss fraction d in k_v=(1-d)k_v0, so convert exactly once.
        scour_rate      = self.state_continuous / 100.0
        temp            = np.random.uniform(3.0, 33.0)
        speed           = np.random.uniform(70.0, 90.0)
        veh_props       = np.random.randn(self.config.n_veh, self.config.n_prop)

        print(
            f"  -> Passage: damage={self.state_continuous:.2f} % | "
            f"T={temp:.1f} °C | v={speed:.1f} km/h"
        )

        return run_single_passage(
            damage_percent=scour_rate,
            speed_kmh=speed,
            temp_celsius=temp,
            add_signal_noise=True,
            n_veh=self.config.n_veh,
            n_prop=self.config.n_prop,
            vehicle_props=veh_props,
        )

    def update_physical_state(self, action_str: str) -> None:
        """
        Apply the maintenance action, then evolve the degradation model by
        one monitoring interval.

        Args:
            action_str: 'perfect_repair' resets the bridge to new condition
                        before evolving; any other value (e.g. 'do_nothing')
                        leaves the current state unchanged before evolving.
        """
        if action_str == 'perfect_repair':
            self.scour_model.repair()

        if self.degradation_law == 'scour_gradual':
            self.state_continuous = self.scour_model.evolve(
                delta_t=self.monitoring_interval
            )
            self.flood_this_step = self.scour_model.flood_occurred_last_step()
            self.flood_severity  = self.scour_model.flood_severity()
        else:
            # Placeholder for future degradation laws
            self.state_continuous += 0.5
            self.flood_this_step = False
            self.flood_severity  = 0.0

        self.time += self.monitoring_interval

    def get_true_mapped_label(self) -> int:
        """
        Map the continuous damage state to the nearest discrete class label.

        Label 0 = healthy (0 %); label (n_classes − 1) = maximum damage.
        The label is clamped to [0, n_classes − 1] so floating-point
        rounding never produces an out-of-range index.

        Returns:
            int: Discrete damage label in [0, n_classes − 1].
        """
        disc      = self.config.discretization
        n_classes = self.config.n_classes
        label     = round(self.state_continuous / disc)
        return int(np.clip(label, 0, n_classes - 1))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Digital asset
# ──────────────────────────────────────────────────────────────────────────────

class DigitalAsset:
    """
    Loads the offline-trained classifier and performs online state estimation.

    The inference pipeline mirrors the training pipeline exactly:
        raw signal (8, L)
            → DOF selection  (n_dofs, L)
            → TTBIPreprocessor.transform  (1, n_dofs, n_segments)
            → model forward pass
            → predicted label

    The scaler is loaded via TTBIPreprocessor.load_scaler() so that the
    same per-channel scaling statistics used during training are applied
    at inference time without re-fitting.

    Args:
        model_path (str):   Path to DT_champion_weights.pth.
        metadata_path (str): Path to DT_metadata.json (written by
                             export_digital_twin_package).
        scaler_path (str):  Path to DT_scaler.pkl or DT_scaler.pt.
        config:             DTConfig instance.
    """

    def __init__(
        self,
        model_path:    str,
        metadata_path: str,
        scaler_path:   str,
        config,
    ):
        self.config = config

        # ── Load architecture metadata ────────────────────────────────────────
        self.metadata = verify_standalone_dt_package(
            model_path, metadata_path, scaler_path
        )
        if int(self.metadata.get("n_segments", config.n_segments)) \
                != int(config.n_segments):
            raise RuntimeError(
                "DTConfig.n_segments differs from the authenticated package.")

        arch_flags   = self.metadata['architecture_flags']
        best_params  = self.metadata['optimal_hyperparameters']
        self.active_dofs: list[int] = self.metadata['active_dofs']

        # ── Rebuild model and load weights ────────────────────────────────────
        # Shape tuple that build_model needs: (N, n_dofs, n_segments)
        # We pass a dummy N=1 — only the channel and length dims are used.
        n_dofs        = len(self.active_dofs)
        input_shape   = (1, n_dofs, config.n_segments)

        self.model, _ = build_model(arch_flags, best_params, input_shape, DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.model.eval()

        # ── Load preprocessor with pre-fitted scaler ──────────────────────────
        method = self.metadata['preprocessing_method'].lower()
        self.preprocessor = TTBIPreprocessor(
            method=method,
            n_segments=config.n_segments,
        )
        self.preprocessor.load_scaler(scaler_path)

        print(
            f"DigitalAsset ready | method={method} | "
            f"channels={[dof_label(d) for d in self.active_dofs]}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def estimate_state(self, raw_signal: np.ndarray) -> int:
        """
        Preprocess a raw eight-channel simulated response and return a predicted
        damage class label.

        Args:
            raw_signal (np.ndarray): float32, shape (8, sequence_length).
                                     All eight response channels returned by
                                     run_single_passage; channel selection is
                                     applied internally. Channels 3/4 are
                                     virtual moving-rail samples, not sensors.

        Returns:
            int: Predicted damage label in [0, n_classes − 1].
                 Label 0 = healthy (0 %); label (n_classes − 1) = maximum damage.
        """
        # 1. Select active DOFs → (n_dofs, L)
        signal_selected = raw_signal[self.active_dofs, :]

        # 2. Add batch dimension → (1, n_dofs, L)
        X = signal_selected[np.newaxis, ...]

        # 3. Preprocess and scale using the pre-fitted scaler
        #    (fit_scaler=False — scaler was loaded from disk in __init__)
        X_scaled = self.preprocessor.transform(X, fit_scaler=False)

        # 4. Convert to tensor and run forward pass
        tensor = torch.tensor(X_scaled).float().to(DEVICE)
        with torch.no_grad():
            logits           = self.model(tensor)
            _, predicted_label = torch.max(logits, dim=1)

        return int(predicted_label.item())
