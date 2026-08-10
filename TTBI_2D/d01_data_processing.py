import numpy as np
from scipy.interpolate import interp1d

def d01_data_processing(i, j, Sol, Train, Calc, Damage, data):
    """
    Extract the legacy response groups and map time samples to space.

    ``AcelRodaPrimVag`` is retained as a compatibility key. Its rows are
    ``N(x_w).T @ A_rail``: the Eulerian/partial rail FE acceleration field at
    the instantaneous wheel coordinates. They are neither wheelset/axle-box
    accelerations nor total moving-contact accelerations; B66 adds the
    convective terms separately.
    """
    
    # 1. Initialize dictionary structures dynamically if they don't exist
    required_attrs = ['AcelPrimVag', 'AcelRodaPrimVag', 'PitchPrimVag', 'Velocidade', 'Posicao']
    for attr in required_attrs:
        if not hasattr(data, attr):
            setattr(data, attr, {})

    # 2. Extract base signals for the first vehicle (0-based indexing)
    #    MATLAB D01 uses A(1:3) for vertical accel and V(4:6) for the PITCH RATES
    #    (the V field stores vertical velocities in rows 1-3 and pitch/rotational
    #    rates in rows 4-6). In 0-based Python that is A[0:3] and V[3:6]. Using
    #    V[0:3] here fed vertical velocities mislabelled as pitch -> a ~10x scale
    #    mismatch on the pitch channels and the train/serve skew. Keep in sync
    #    with scour_MATLAB/D01_DataProcessing.m.
    acel_bogie = Sol.Veh[0].A[0:3, :]
    rail_eulerian_acc_at_wheels = Sol.Veh[0].acc_under[0:4, :]
    pitch_bogie = Sol.Veh[0].V[3:6, :]

    # 3. Apply artificial noise (proportional to signal magnitude).
    #    PARITY: MATLAB D01 adds noise only to the legacy AcelRodaPrimVag group;
    #    this is a compatibility perturbation on moving rail-response channels,
    #    not a physical wheel-sensor noise model. The training data has clean
    #    body/bogie and pitch
    #    channels. The live DT observation path must match that distribution, so
    #    noise here is limited to that legacy group too (it previously hit all
    #    three groups — a train/serve skew on the champion's clean channels).
    #    Sensor faults on
    #    the other channels are modelled separately (digital_twin/sensor_health).
    num_t = Calc.Solver.num_t
    noise_level = Damage.desvio

    if noise_level > 0:
        rail_eulerian_acc_at_wheels += (
            noise_level
            * rail_eulerian_acc_at_wheels
            * np.random.randn(4, num_t)
        )

    # 4. Store velocity and calculate positional bounds
    data.Velocidade[i, j] = Train.vel
    data.Posicao[i, j] = Train.vel * (num_t / 1000.0)

    # 5. Spatial Domain Discretization Setup
    dim_acel = acel_bogie.shape[1]
    dim_space = int(round(data.Posicao[i, j] * 100))

    xi = np.arange(1, dim_space + 1)
    xx = np.linspace(1, dim_space, dim_acel)

    # LEGACY spatial crop window (space domain is 100 samples/m), kept
    # byte-identical to reproduce the original 40 m dataset window
    # (cols 1000:6831 = 5831 samples). Window = ~10 m approach skip + bridge
    # span + 1831 samples of vehicle crossing/after; the 1831 samples span
    # 1830 grid intervals = 18.30 m of travel (the leading vehicle's
    # first-to-last-axle span), and the window opens ~0.20 m before exact
    # deck entry — see the registered-crop truth statement in the MATLAB
    # D01_DataProcessing.m header. NOT the registered R11 crop: this legacy
    # path rounds L_bridge to the nearest metre (L99.6 -> 100 m = 10000
    # samples), whereas the registered MATLAB crop uses round(100*L_bridge)
    # (L99.6 -> 9960 samples). Do not use this path for R11 campaign data.
    crop_start = 1000                                          # ~10 m approach skip
    bridge_samp = int(round(Calc.Profile.L_bridge)) * 100      # legacy metre-rounded span [samples]
    crop_end = min(crop_start + bridge_samp + 1831, dim_space)  # + crossing/after

    # Helper function to interpolate and slice in one step
    def interpolate_and_slice(signal):
        f_interp = interp1d(xx, signal, axis=1, kind='linear', fill_value="extrapolate")
        return f_interp(xi)[:, crop_start:crop_end]

    # 6. Apply interpolation and assign to the data object
    data.AcelPrimVag[i, j] = interpolate_and_slice(acel_bogie)
    data.AcelRodaPrimVag[i, j] = interpolate_and_slice(
        rail_eulerian_acc_at_wheels
    )
    data.PitchPrimVag[i, j] = interpolate_and_slice(pitch_bogie)

    return data
