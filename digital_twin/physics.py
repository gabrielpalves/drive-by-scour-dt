"""
digital_twin/physics.py
=======================
Thin Python wrapper around the TTBI_2D physics engine.

The single public function, run_single_passage, executes one full
vehicle-bridge interaction simulation and returns the eight sensor
signals in the canonical DOF order used throughout the project.

All variability sources (speed, temperature, vehicle properties,
signal noise) are explicit arguments so the caller — PhysicalAsset
in digital_twin/assets.py — controls randomness and can reproduce
any passage exactly given the same inputs.

Imported by:
    digital_twin/assets.py — PhysicalAsset.get_observation_signal

Nothing in the ablation pipeline imports this module.
"""

import numpy as np

from TTBI_2D.a01_train         import a01_train
from TTBI_2D.a02_track         import a02_track
from TTBI_2D.a03_bridge        import a03_bridge
from TTBI_2D.a04_options       import a04_options
from TTBI_2D.b00_calculations  import b00_calculations
from TTBI_2D.d01_data_processing import d01_data_processing


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

class EmptyObj:
    """
    Attribute container that mimics MATLAB struct behaviour.

    TTBI_2D functions assign fields dynamically (e.g. Beam.Prop.E = ...).
    Python dataclasses would work here too, but EmptyObj keeps the bridge
    code unmodified and avoids declaring every field upfront.
    """
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Physics engine
# ──────────────────────────────────────────────────────────────────────────────

# Canonical DOF order shared with core/utils.DOF_NAMES.
# Documented here so the stacking logic below is self-explanatory.
#
#   Index  Name               Source field        Row
#   ─────  ─────────────────  ──────────────────  ───
#     0    CarBody_Vert       AcelPrimVag          0
#     1    FrontBogie_Vert    AcelPrimVag          1
#     2    RearBogie_Vert     AcelPrimVag          2
#     3    Wheel1_Vert        AcelRodaPrimVag      0
#     4    Wheel2_Vert        AcelRodaPrimVag      1
#     5    CarBody_Pitch      PitchPrimVag         0
#     6    FrontBogie_Pitch   PitchPrimVag         1
#     7    RearBogie_Pitch    PitchPrimVag         2

_N_DOFS = 8


def run_single_passage(
    damage_percent:   float,
    speed_kmh:        float             = 80.0,
    temp_celsius:     float             = 25.0,
    add_signal_noise: bool              = True,
    noise_std:        float             = 0.05,
    n_veh:            int               = 5,
    n_prop:           int               = 3,
    vehicle_props:    np.ndarray | None = None,
) -> np.ndarray:
    """
    Run one TTBI vehicle-bridge passage and return all eight DOF signals.

    Pipeline
    --------
    1. Configure damage, bridge stiffness, and vehicle variability structs.
    2. Build the train, track, and bridge models via the TTBI_2D functions.
    3. Apply temperature degradation to Young's modulus (0.3 % per °C above 15 °C).
    4. Run the numerical integration (b00_calculations).
    5. Spatial-interpolate the raw solver output (d01_data_processing).
    6. Extract and stack the eight DOF rows in canonical order.

    Temperature model
    -----------------
    E is reduced linearly from the reference value at 15 °C:
        E(T) = E_ref × (1 − 0.003 × (T − 15))
    This is applied after a03_bridge sets E_ref so that the bridge object
    contains the temperature-adjusted stiffness before the solver runs.

    Args:
        damage_percent (float):     Scour damage as a percentage of the
                                    reference stiffness, in [0.0, 100.0].
                                    Passed directly to the Damage struct;
                                    the TTBI_2D functions interpret it as
                                    a stiffness reduction rate.
        speed_kmh (float):          Vehicle crossing speed in km/h.
                                    Converted to m/s internally.
        temp_celsius (float):       Ambient temperature in °C.
                                    Used for the Young's modulus correction.
        add_signal_noise (bool):    If True, Gaussian noise with std=noise_std
                                    is added to the output signals via the
                                    Damage.desvio field.
        noise_std (float):          Standard deviation of the additive noise.
                                    Ignored when add_signal_noise=False.
        n_veh (int):                Number of vehicles (axles) in the model.
        n_prop (int):               Number of vehicle property dimensions.
        vehicle_props (np.ndarray): Shape (n_veh, n_prop).  Vehicle property
                                    perturbations passed to a01_train.
                                    If None, a zero matrix is used (no
                                    vehicle variability).

    Returns:
        np.ndarray: float32, shape (8, sequence_length).
                    Rows follow the canonical DOF order defined above.
                    sequence_length is determined by the solver and depends
                    on speed and bridge discretisation.

    Notes:
        • The function is stateless — repeated calls with the same arguments
          produce the same output (given the same TTBI_2D RNG state).
        • Caller is responsible for randomising speed, temperature, and
          vehicle_props when variability is required (see PhysicalAsset).
    """
    # ── 1. Structs ────────────────────────────────────────────────────────────
    damage            = EmptyObj()
    damage.desvio     = noise_std if add_signal_noise else 0.0
    damage.DOFStiff_ROT_value  = 0
    damage.DOF_ChangeRate_value = damage_percent

    beam            = EmptyObj()
    beam.Prop       = EmptyObj()
    beam.Prop.E_mod = 1
    beam.Prop.n_mod = 100

    speed_ms = speed_kmh / 3.6

    if vehicle_props is None:
        vehicle_props = np.zeros((n_veh, n_prop))
    else:
        vehicle_props = np.asarray(vehicle_props, dtype=float).reshape(n_veh, n_prop)

    # ── 2. Build models ───────────────────────────────────────────────────────
    train        = a01_train(speed_ms, vehicle_props)
    track        = a02_track()
    beam         = a03_bridge(beam)

    # ── 3. Temperature degradation of stiffness ───────────────────────────────
    beam.Prop.E  = beam.Prop.E - beam.Prop.E * 0.003 * (temp_celsius - 15.0)

    # ── 4. Numerical integration ──────────────────────────────────────────────
    calc, beam, track           = a04_options(beam, track)
    sol, calc, train, beam, track = b00_calculations(calc, train, track, beam, damage)

    # ── 5. Spatial interpolation ──────────────────────────────────────────────
    data = EmptyObj()
    data = d01_data_processing(0, 0, sol, train, calc, damage, data)

    # ── 6. Extract DOFs ───────────────────────────────────────────────────────
    acel_bogie  = data.AcelPrimVag[0, 0]      # (3, L): CarBody, FrontBogie, RearBogie vert
    acel_wheel  = data.AcelRodaPrimVag[0, 0]  # (4, L): Wheels 1–4 vert
    pitch_bogie = data.PitchPrimVag[0, 0]     # (3, L): CarBody, FrontBogie, RearBogie pitch

    signal = np.vstack((
        acel_bogie[0, :],   # 0: CarBody_Vert
        acel_bogie[1, :],   # 1: FrontBogie_Vert
        acel_bogie[2, :],   # 2: RearBogie_Vert
        acel_wheel[0, :],   # 3: Wheel1_Vert
        acel_wheel[1, :],   # 4: Wheel2_Vert
        pitch_bogie[0, :],  # 5: CarBody_Pitch
        pitch_bogie[1, :],  # 6: FrontBogie_Pitch
        pitch_bogie[2, :],  # 7: RearBogie_Pitch
    ))

    return signal.astype(np.float32)
