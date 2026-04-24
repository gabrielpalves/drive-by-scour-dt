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
    acel_bogie = Sol.Veh[0].A[0:3, :]
    acel_wheel = Sol.Veh[0].acc_under[0:4, :]
    pitch_bogie = Sol.Veh[0].V[0:3, :]

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

    # Helper function to interpolate and slice in one step
    def interpolate_and_slice(signal):
        f_interp = interp1d(xx, signal, axis=1, kind='linear', fill_value="extrapolate")
        # Slice assignments (Python 0-based index: 1000 to 6831 covers MATLAB's 1001 to 6831)
        return f_interp(xi)[:, 1000:6831]

    # 6. Apply interpolation and assign to the data object
    data.AcelPrimVag[i, j] = interpolate_and_slice(acel_bogie)
    data.AcelRodaPrimVag[i, j] = interpolate_and_slice(acel_wheel)
    data.PitchPrimVag[i, j] = interpolate_and_slice(pitch_bogie)

    return data