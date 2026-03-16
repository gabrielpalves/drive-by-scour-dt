import numpy as np
import matplotlib.pyplot as plt

def b24_beam_damping(Beam):
    """
    Calculates the Beam damping matrix 
       Rayleigh damping is adopted 
       1st and 2nd beam frequencies are taken as reference (excluding rigid modes)
    """

    if Beam.Damping.per > 0:
        
        # Reference frequencies (0-based indexing)
        # We skip the rigid body modes and grab the next two frequencies
        start_idx = int(Beam.Modal.num_rigid_modes)
        wr = Beam.Modal.w[start_idx : start_idx + 2]
        
        # Rayleigh's coefficients 'alpha' and 'beta'
        # Setting up the 2x2 matrix A and vector b to solve: A * x = b
        A = 0.5 * np.array([
            [1.0 / wr[0], wr[0]],
            [1.0 / wr[1], wr[1]]
        ])
        b = np.array([1.0, 1.0]) * (Beam.Damping.per / 100.0)
        
        # Solve the linear system (equivalent to MATLAB's backslash operator '\')
        aux1 = np.linalg.solve(A, b)

        # Damping matrix assembly
        # If Mg and Kg are scipy.sparse matrices (from B03), this will natively preserve sparsity
        Beam.Mesh.Cg = aux1[0] * Beam.Mesh.Mg + aux1[1] * Beam.Mesh.Kg
        
        # # Graphical check (Translated to Python/Matplotlib)
        # w = np.linspace(wr[0] / 2.0, wr[1] * 1.5, 100)
        # plt.figure()
        # plt.plot(w, aux1[0] / (2 * w) + aux1[1] * w / 2, label='Rayleigh Curve')
        # plt.plot(wr, np.array([1, 1]) * (Beam.Damping.per / 100.0), 'r.', label='Reference Frequencies')
        # plt.xlabel('Circular frequency')
        # plt.ylabel('Damping ratio')
        # plt.legend()
        # plt.show()

    else:
        
        # No Damping case
        # Multiplying the sparse matrix by 0 creates a zero-filled matrix of the exact same size/sparsity
        Beam.Mesh.Cg = Beam.Mesh.Kg * 0.0
        
    return Beam

# ---- End of script ----