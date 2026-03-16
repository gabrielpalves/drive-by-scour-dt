import numpy as np
import scipy.linalg as la

# Dummy class for structure initialization
class EmptyObj:
    pass

def b56_model_frq(Model, Calc):
    """
    Adapted version of b09_beam_frq
    Calculates the model modes and frequencies given the system matrices
    """

    # Initialize Modal if it doesn't exist
    if not hasattr(Model, 'Modal'):
        Model.Modal = EmptyObj()

    # Check if calculation options are activated
    calc_frq = (Calc.Options.calc_model_frq == 1)
    calc_modes = (Calc.Options.calc_model_modes == 1)

    if calc_frq:
        # Convert sparse matrices to dense arrays for the eigensolver
        # (Matches MATLAB's full() command)
        Kg_dense = Model.Mesh.Kg.toarray()
        Mg_dense = Model.Mesh.Mg.toarray()

        # Only natural frequencies calculation
        if not calc_modes:
            print('Calculating model frequencies ...')
            
            # eigh returns eigenvalues (eigvals_only=True skips eigenvectors)
            lambda_vals = la.eigh(Kg_dense, Mg_dense, eigvals_only=True)
            
            Model.Modal.w = np.sqrt(lambda_vals)
            Model.Modal.f = Model.Modal.w / (2 * np.pi)

            # Removing values associated to BC
            # 0-based indexing: we skip the first 'num_DOF_fixed' values
            Model.Modal.w = Model.Modal.w[Model.BC.num_DOF_fixed:]
            Model.Modal.f = Model.Modal.f[Model.BC.num_DOF_fixed:]
            
        # Natural frequencies and Modes of vibration calculation
        elif calc_modes:
            print('Calculating model modes and frequencies ...')
            
            # eigh returns eigenvalues and eigenvectors natively
            lambda_vals, V = la.eigh(Kg_dense, Mg_dense)
            
            # Sort eigenvalues and eigenvectors 
            # (eigh typically returns them sorted, but this ensures absolute safety)
            idx = np.argsort(lambda_vals)
            lambda_vals = lambda_vals[idx]
            V = V[:, idx]
            
            # Normalization of eigenvectors
            # V.T @ Mg_dense @ V is the equivalent of V'*Mg*V in MATLAB
            Factor = np.diag(V.T @ Mg_dense @ V)
            Model.Modal.modes = V / np.sqrt(Factor)
            
            # EigenValues to Natural frequencies
            Model.Modal.w = np.sqrt(lambda_vals)
            Model.Modal.f = Model.Modal.w / (2 * np.pi)
            
            # Removing values associated to BC
            Model.Modal.w = Model.Modal.w[Model.BC.num_DOF_fixed:]
            Model.Modal.f = Model.Modal.f[Model.BC.num_DOF_fixed:]
            
            # Remove the first 'num_DOF_fixed' columns from the mode shape matrix
            Model.Modal.modes = Model.Modal.modes[:, Model.BC.num_DOF_fixed:]

    return Model

# ---- End of function ----