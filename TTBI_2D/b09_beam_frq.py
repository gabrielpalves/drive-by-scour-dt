import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math

# Dummy class for structure initialization if needed
class EmptyObj:
    pass

def b09_beam_frq(Beam, Calc):
    """
    Calculates the beam frequencies given its system matrices
    """
    
    # Initialize Modal if it doesn't exist
    if not hasattr(Beam, 'Modal'):
        Beam.Modal = EmptyObj()

    # Convert sparse matrices to dense arrays for the eigensolver
    # (Matches MATLAB's full() command)
    Kg_dense = Beam.Mesh.Kg.toarray()
    Mg_dense = Beam.Mesh.Mg.toarray()

    # Only natural frequencies calculation
    if Calc.Options.calc_beam_frq == 1 and Calc.Options.calc_beam_modes == 0:
        
        # eigh returns eigenvalues (eigvals_only=True skips eigenvectors)
        # Note: eigh assumes matrices are symmetric/Hermitian. 
        lambda_vals = la.eigh(Kg_dense, Mg_dense, eigvals_only=True)
        
        Beam.Modal.w = np.sqrt(lambda_vals)
        Beam.Modal.f = Beam.Modal.w / (2 * np.pi)

        # Removing values associated to BC
        Beam.Modal.w = Beam.Modal.w[Beam.BC.num_DOF_fixed:]
        Beam.Modal.f = Beam.Modal.f[Beam.BC.num_DOF_fixed:]

    # Natural frequencies and Modes of vibration calculation
    elif Calc.Options.calc_beam_frq == 1 and Calc.Options.calc_beam_modes == 1:
        
        # eigh returns eigenvalues and eigenvectors
        lambda_vals, V = la.eigh(Kg_dense, Mg_dense)
        
        # Sort eigenvalues and eigenvectors (eigh usually sorts them, but doing it to be safe)
        idx = np.argsort(lambda_vals)
        lambda_vals = lambda_vals[idx]
        V = V[:, idx]
        
        # Normalization of eigenvectors
        # V.T @ Mg_dense @ V is the matrix equivalent of V'*Mg*V in MATLAB
        Factor = np.diag(V.T @ Mg_dense @ V)
        Beam.Modal.modes = V / np.sqrt(Factor)
        
        # EigenValues to Natural frequencies
        Beam.Modal.w = np.sqrt(lambda_vals)
        Beam.Modal.f = Beam.Modal.w / (2 * np.pi)
        
        # Removing values associated to BC
        Beam.Modal.w = Beam.Modal.w[Beam.BC.num_DOF_fixed:]
        Beam.Modal.f = Beam.Modal.f[Beam.BC.num_DOF_fixed:]
        # Remove the first 'num_DOF_fixed' columns
        Beam.Modal.modes = Beam.Modal.modes[:, Beam.BC.num_DOF_fixed:]

    # ------------------------------ Plotting ---------------------------------
    
    if not hasattr(Calc.Plot, 'P1_Beam_frq'): Calc.Plot.P1_Beam_frq = 0
    if not hasattr(Calc.Plot, 'P2_Beam_modes'): Calc.Plot.P2_Beam_modes = 0

    # -- Plotting of calculated Natural frequencies --
    if Calc.Plot.P1_Beam_frq == 1:
        plt.figure()
        # range generates integers starting at 1 for the x-axis
        modes_range = list(range(1, len(Beam.Modal.f) + 1))
        plt.plot(modes_range, Beam.Modal.f, '.')
        plt.axis('tight')
        plt.xlabel('Mode number')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"Beam Only (1st frq: {round(Beam.Modal.f[0], 2)} Hz;  Last frq: {round(Beam.Modal.f[-1], 2)} Hz)")
        plt.pause(0.25)

    # -- Plotting Mode shapes --
    
    if Calc.Plot.P2_Beam_modes >= 1:
        num_modes_to_plot = int(Calc.Plot.P2_Beam_modes)
        
        # Calculate rows for subplots
        aux1 = math.ceil(num_modes_to_plot / 2)
        plt.figure()
        
        for k in range(num_modes_to_plot):
            # Extract vertical displacements: 0-indexed, step of 2 -> 0::2
            vert_modes = Beam.Modal.modes[0::2, k]
            aux2 = np.max(np.abs(vert_modes))
            
            plt.subplot(aux1, 2, k + 1) # Subplot index is 1-based in matplotlib
            Xdata = Beam.Mesh.Nodes.acum
            
            plt.plot(Xdata, vert_modes / aux2)
            plt.xlim([0, Beam.Prop.L])
            plt.title(f"Mode {k + 1} ({round(Beam.Modal.f[k], 3)} Hz)")
            
        plt.tight_layout() # Prevents subplot titles from overlapping
        plt.pause(0.25)

    return Beam

# ---- End of function ----