import numpy as np
import scipy.linalg as la
from types import SimpleNamespace

def b08_veh_freq(Veh, Calc):
    """
    Calculates all vehicle frequencies given their system matrices.
    """

    # Check if calculation options are activated
    if getattr(Calc.Options, 'calc_veh_frq', 0) == 1:
        
        for v in range(len(Veh)):
            
            # ---- Eigenvalue analysis ----
            # eigh calculates eigenvalues for complex Hermitian or real symmetric matrices
            eigvals = la.eigh(Veh[v].SysM.K, Veh[v].SysM.M, eigvals_only=True)
            
            # Prevent NaN warnings from floating-point noise on zero-frequency (rigid) modes
            eigvals = np.maximum(eigvals, 0.0)
            
            # Initialize Modal namespace if it doesn't exist
            if not hasattr(Veh[v], 'Modal'):
                Veh[v].Modal = SimpleNamespace()
                
            Veh[v].Modal.w = np.sqrt(eigvals)                     # Vehicle circular frequencies
            Veh[v].Modal.f = Veh[v].Modal.w / (2.0 * np.pi)       # Vehicle frequencies (Hz)

    return Veh

# ---- End of script ----