import numpy as np
from scipy.interpolate import interp1d

def d01_data_processing(i, j, Sol, Train, Calc, Damage, data):
    """
    Data processing: Extracts accelerations, applies noise, and maps them 
    from the time domain to the spatial domain using interpolation.
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
    acel_wheel = Sol.Veh[0].acc_under[0:4, :]
    pitch_bogie = Sol.Veh[0].V[3:6, :]

    # 3. Apply artificial noise (proportional to signal magnitude)
    num_t = Calc.Solver.num_t
    noise_level = Damage.desvio
    
    if noise_level > 0:
        acel_bogie += noise_level * acel_bogie * np.random.randn(3, num_t)
        acel_wheel += noise_level * acel_wheel * np.random.randn(4, num_t)
        pitch_bogie += noise_level * pitch_bogie * np.random.randn(3, num_t)

    # 4. Store velocity and calculate positional bounds
    data.Velocidade[i, j] = Train.vel
    data.Posicao[i, j] = Train.vel * (num_t / 1000.0)

    # 5. Spatial Domain Discretization Setup
    dim_acel = acel_bogie.shape[1]
    dim_space = int(round(data.Posicao[i, j] * 100))

    xi = np.arange(1, dim_space + 1)
    xx = np.linspace(1, dim_space, dim_acel)

    # Spatial crop window, generalised for any bridge length (space domain is
    # 100 samples/m). Window = 10 m approach skip + bridge span + 18.31 m of
    # vehicle crossing/after. Rounding L_bridge to the nearest metre reproduces
    # the original 40 m window exactly (cols 1000:6831 = 5831 samples), while a
    # longer bridge now captures its whole passage. Keep this in sync with the
    # MATLAB D01_DataProcessing crop.
    crop_start = 1000                                          # ~10 m approach skip
    bridge_samp = int(round(Calc.Profile.L_bridge)) * 100      # bridge span [samples]
    crop_end = min(crop_start + bridge_samp + 1831, dim_space)  # + crossing/after

    # Helper function to interpolate and slice in one step
    def interpolate_and_slice(signal):
        f_interp = interp1d(xx, signal, axis=1, kind='linear', fill_value="extrapolate")
        return f_interp(xi)[:, crop_start:crop_end]

    # 6. Apply interpolation and assign to the data object
    data.AcelPrimVag[i, j] = interpolate_and_slice(acel_bogie)
    data.AcelRodaPrimVag[i, j] = interpolate_and_slice(acel_wheel)
    data.PitchPrimVag[i, j] = interpolate_and_slice(pitch_bogie)

    return data