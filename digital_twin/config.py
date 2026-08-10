"""
digital_twin/config.py
======================
Single source of truth for every tuneable parameter in the drive-by
digital twin.

DTConfig holds three categories of parameters:

    Physics      — vehicle and bridge simulation settings fed to
                   run_single_passage via PhysicalAsset.
    Classifier   — signal processing settings that must match the
                   offline training pipeline exactly.
    Framework    — simulation horizon, monitoring cadence, and the
                   damage threshold that triggers a maintenance action.

Consistency guarantee
---------------------
The classifier parameters (n_segments, discretization, n_classes) must
match those used during the ablation study that produced the loaded model.
The classmethod DTConfig.from_metadata() reads these values directly from
DT_metadata.json so they cannot drift from the trained model.  Manual
construction is available for experimentation but requires the caller to
ensure consistency.

Derived fields
--------------
n_classes and damage_to_label are computed from discretization and
max_damage at construction time and are read-only properties, so they
cannot be accidentally overwritten after construction.

Imported by:
    drive_by_DT.py      — DTConfig.from_metadata() for production runs,
                          DTConfig() for experimental / debug runs.
    digital_twin/assets.py — PhysicalAsset, DigitalAsset consume config fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class DTConfig:
    """
    Configuration container for the drive-by digital twin.

    All parameters have documented defaults that match the baseline
    experiment.  Override any field at construction time.

    Physics parameters
    ------------------
    n_veh (int):
        Number of vehicle axles in the TTBI_2D simulation.  Must match
        the dimensionality of the vehicle property matrix passed to
        run_single_passage.  Default: 5.

    n_prop (int):
        Number of property dimensions per axle (e.g. mass, stiffness,
        damping).  Default: 3.

    max_damage (float):
        Maximum scour damage in percent.  The ScourModel caps its output
        at this value.  Default: 60.0.

    Classifier parameters
    ---------------------
    discretization (int):
        Damage discretisation step in percent.  Determines the number of
        classes: n_classes = int(max_damage / discretization) + 1.
        Must match the ablation study that produced the loaded model.
        Default: 5  →  13 classes spanning [0 %, 5 %, …, 60 %].

    n_segments (int):
        Target sequence length after preprocessing (PAA / FFT bins or
        CWT time-axis width).  Must match the trained model's input size.
        Default: 512.

    Framework parameters
    --------------------
    sim_years (int):
        Total simulation horizon in years.  Passed as n_steps to
        Graph.simulate().  Default: 60.

    monitoring_interval (float):
        Years between consecutive drive-by passages.  Used by
        ScourModel.evolve(delta_t=...) and the DBN transition prior.
        Default: 1.0.

    repair_threshold_label (int):
        Damage class labels at or above this value trigger repair.
        Label 0 = healthy (0 %); label n_classes-1 = maximum damage.
        Default: 6  →  repairs states representing ≥ 30 % damage
        when discretization=5.
    """

    # ── Physics ───────────────────────────────────────────────────────────────
    n_veh:      int   = 5
    n_prop:     int   = 3
    max_damage: float = 60.0

    # ── Classifier ────────────────────────────────────────────────────────────
    discretization: int = 5
    n_segments:     int = 512

    # ── Framework ─────────────────────────────────────────────────────────────
    sim_years:             int   = 60
    monitoring_interval:   float = 1.0
    repair_threshold_label: int  = 6
    # Enable the project flood scenario, which uses the generic Compound-Poisson
    # family in Kamariotis et al. (2023) but author-chosen rate/jump values. Off
    # by default; this is not a calibrated hydraulic scour process. See ScourModel.
    enable_shock:          bool  = False

    # ── Derived (computed in __post_init__; do not pass at construction) ──────
    n_classes:      int  = field(init=False, repr=True)
    damage_to_label: dict = field(init=False, repr=False)

    # Class-level constant — not a field
    _NOISE_STD: ClassVar[float] = 0.05

    def __post_init__(self) -> None:
        self._recompute_derived()
        self._validate()

    # ── Derived-field computation ─────────────────────────────────────────────

    def _recompute_derived(self) -> None:
        """
        Compute n_classes and damage_to_label from discretization and
        max_damage.

        damage_to_label maps each physical damage value (0, 5, 10, …, 60
        when discretization=5) to its integer class label.  Convention:
        label 0 = healthy (0 %), label n_classes-1 = maximum damage (60 %):

            0 % → label 0,  5 % → label 1,  …,  60 % → label 12
        """
        damage_cases     = list(range(0, int(self.max_damage) + 1, self.discretization))
        self.n_classes   = len(damage_cases)
        self.damage_to_label = {
            dmg: i
            for i, dmg in enumerate(sorted(damage_cases))   # 0%→label 0, 60%→label 12
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Catch obviously invalid configurations at construction time."""
        errors = []

        if self.discretization <= 0:
            errors.append(f"discretization must be > 0, got {self.discretization}.")
        if self.max_damage % self.discretization != 0:
            errors.append(
                f"max_damage ({self.max_damage}) must be divisible by "
                f"discretization ({self.discretization})."
            )
        if not (0 <= self.repair_threshold_label < self.n_classes):
            errors.append(
                f"repair_threshold_label ({self.repair_threshold_label}) must be "
                f"in [0, {self.n_classes - 1}]."
            )
        if self.n_segments <= 0:
            errors.append(f"n_segments must be > 0, got {self.n_segments}.")
        if self.monitoring_interval <= 0:
            errors.append(f"monitoring_interval must be > 0, got {self.monitoring_interval}.")
        if self.sim_years <= 0:
            errors.append(f"sim_years must be > 0, got {self.sim_years}.")

        if errors:
            raise ValueError("DTConfig validation failed:\n  " + "\n  ".join(errors))

    # ── Alternative constructors ──────────────────────────────────────────────

    @classmethod
    def from_metadata(
        cls,
        metadata_path:         str,
        repair_threshold_label: int   = 6,
        sim_years:             int   = 60,
        monitoring_interval:   float = 1.0,
        n_veh:                 int   = 5,
        n_prop:                int   = 3,
    ) -> DTConfig:
        """
        Build a DTConfig whose classifier parameters are read directly from
        DT_metadata.json, guaranteeing consistency with the loaded model.

        The metadata file is written by export_digital_twin_package and
        contains the preprocessing method, active DOFs, discretization, and
        Optuna best_params — everything needed to reconstruct the model.
        n_segments is inferred from the 'n_segments' key if present, or
        defaults to 512 (the value used throughout the baseline study).

        Parameters not stored in the metadata (simulation horizon, monitoring
        cadence, repair threshold, physics engine settings) are taken from the
        keyword arguments, which carry the same defaults as DTConfig().

        Args:
            metadata_path (str):           Path to DT_metadata.json.
            repair_threshold_label (int):  Damage label that triggers repair.
            sim_years (int):               Simulation horizon in years.
            monitoring_interval (float):   Years between observations.
            n_veh (int):                   Vehicle axle count.
            n_prop (int):                  Vehicle property dimensions.

        Returns:
            DTConfig: Fully initialised configuration consistent with the
                      loaded model.

        Raises:
            FileNotFoundError: If metadata_path does not exist.
            KeyError:          If required keys are absent from the metadata.
        """
        with open(metadata_path) as f:
            meta = json.load(f)

        # The ablation uses 1% damage steps (61 classes); default to 1 (not 5)
        # when older metadata omits the field, matching the trained champions.
        discretization = int(meta.get('discretization', 1))
        n_segments     = int(meta.get('n_segments', 512))

        # Infer n_segments from best_params if not stored explicitly.
        # The preprocessor always uses 512 in the baseline study, but
        # future experiments may vary it.
        if 'n_segments' not in meta:
            best = meta.get('optimal_hyperparameters', {})
            n_segments = int(best.get('n_segments', 512))

        config = cls(
            n_veh=n_veh,
            n_prop=n_prop,
            discretization=discretization,
            n_segments=n_segments,
            repair_threshold_label=repair_threshold_label,
            sim_years=sim_years,
            monitoring_interval=monitoring_interval,
        )

        # Surface the method and DOFs for logging without storing them —
        # DigitalAsset reads these fields from the metadata directly.
        print(
            f"DTConfig loaded from metadata:\n"
            f"  method={meta.get('preprocessing_method')}  "
            f"discretization={discretization}  "
            f"n_classes={config.n_classes}  "
            f"n_segments={n_segments}\n"
            f"  active_dofs={meta.get('active_dofs')}"
        )

        return config

    # ── Convenience ───────────────────────────────────────────────────────────

    def label_to_damage(self, label: int) -> float:
        """
        Convert a class label back to a physical damage percentage.

        Inverse of damage_to_label.  Label 0 → 0 % (healthy),
        label n_classes-1 → max_damage (most damaged).

        Args:
            label (int): Damage class label in [0, n_classes-1].

        Returns:
            float: Physical damage in [0, max_damage] %.
        """
        return float(label * self.discretization)

    def __str__(self) -> str:
        return (
            f"DTConfig("
            f"disc={self.discretization}, "
            f"n_classes={self.n_classes}, "
            f"n_segments={self.n_segments}, "
            f"threshold={self.repair_threshold_label}, "
            f"horizon={self.sim_years} yr, "
            f"interval={self.monitoring_interval} yr"
            f")"
        )
