import numpy as np
from scipy.interpolate import interp1d

def d01_data_processing(i, j, Sol, Train, Calc, data):
    """
    Data processing: Extracts accelerations and maps them from the time domain 
    to the spatial domain using interpolation.
    """
    
    # Initialize dictionary structures if they don't exist yet
    if not hasattr(data, 'AceleracaoPrimVag'):
        data.AceleracaoPrimVag = {}
        data.AceleracaoIntVag = {}
        data.AceleracaoUltVag = {}
        data.AcelRodaPrimVag = {}
        data.AcelRodaIntVag = {}
        data.AcelRodaUltVag = {}
        data.Velocidade = {}
        data.Posicao = {}

    # Spacial domain extraction (Note: 0-based indexing for vehicles 1, 3, and 5)
    data.AceleracaoPrimVag[i, j] = Sol.Veh[0].A[0:3, :]  # Primeiro vagao
    data.AceleracaoIntVag[i, j]  = Sol.Veh[2].A[0:3, :]  # Terceiro vagao
    data.AceleracaoUltVag[i, j]  = Sol.Veh[4].A[0:3, :]  # Último vagao
    
    data.AcelRodaPrimVag[i, j]   = Sol.Veh[0].acc_under[0:4, :]
    data.AcelRodaIntVag[i, j]    = Sol.Veh[2].acc_under[0:4, :]
    data.AcelRodaUltVag[i, j]    = Sol.Veh[4].acc_under[0:4, :]
    
    data.Velocidade[i, j] = Train.vel
    data.Posicao[i, j] = Train.vel * (Calc.Solver.num_t / 1000.0)

    # Discretizacao no dominio do espaco
    DimAcel = Sol.Veh[0].A.shape[1]
    DimSpace = int(round(data.Posicao[i, j] * 100))

    xi = np.arange(1, DimSpace + 1)
    xx = np.linspace(1, DimSpace, DimAcel)

    # Transformacao para o dominio do espaco (Vectorized Interpolation)
    # Using axis=1 interpolates all rows (DOFs) simultaneously, eliminating the 'for ii' loop!
    f_prim = interp1d(xx, data.AceleracaoPrimVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelEspacoPrimVag = f_prim(xi)
    
    f_int = interp1d(xx, data.AceleracaoIntVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelEspacoIntVag = f_int(xi)
    
    f_ult = interp1d(xx, data.AceleracaoUltVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelEspacoUltVag = f_ult(xi)

    f_r_prim = interp1d(xx, data.AcelRodaPrimVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelRodaPrimVag = f_r_prim(xi)
    
    f_r_int = interp1d(xx, data.AcelRodaIntVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelRodaIntVag = f_r_int(xi)
    
    f_r_ult = interp1d(xx, data.AcelRodaUltVag[i, j], axis=1, kind='linear', fill_value="extrapolate")
    AcelRodaUltVag = f_r_ult(xi)

    # Slice assignments (Python 0-based index: 1000 to 6831 covers MATLAB's 1001 to 6831)
    data.AceleracaoPrimVag[i, j] = AcelEspacoPrimVag[:, 1000:6831]
    data.AcelRodaPrimVag[i, j]   = AcelRodaPrimVag[:, 1000:6831]
    
    # Note: AcelEspacoIntVag, AcelEspacoUltVag, AcelRodaIntVag, and AcelRodaUltVag are calculated 
    # but dropped here to perfectly match the original MATLAB script's exact output behavior.

    return data