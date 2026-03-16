import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def b03_beam_matrices(Beam):
    """
    Generates the FEM system matrices for the beam model.
    """

    Tnum = Beam.Mesh.DOF.Tnum

    # Initialize matrices using LIL format for efficient assembly
    Beam.Mesh.Kg = lil_matrix((Tnum, Tnum), dtype=float)
    CMM = lil_matrix((Tnum, Tnum), dtype=float)

    # Elemental matrices (lambda functions) (more efficient alternative to subfunctions)
    b04_beam_ele_m = lambda r, A, L: (r * A * L / 420.0) * np.array([
        [156,      22 * L,     54,      -13 * L],
        [22 * L,   4 * L**2,   13 * L,  -3 * L**2],
        [54,       13 * L,     156,     -22 * L],
        [-13 * L,  -3 * L**2,  -22 * L, 4 * L**2]
    ], dtype=float)

    b05_beam_ele_k = lambda EI, L: (EI / L**3) * np.array([
        [12,       6 * L,      -12,     6 * L],
        [6 * L,    4 * L**2,   -6 * L,  2 * L**2],
        [-12,      -6 * L,     12,      -6 * L],
        [6 * L,    2 * L**2,   -6 * L,  4 * L**2]
    ], dtype=float)

    # ---- Beam element Shape function ----
    Beam.Mesh.shape_fun = lambda x, a: np.array([
        (a + 2*x) * (a - x)**2 / a**3, 
        x * (a - x)**2 / a**2, 
        x**2 * (3*a - 2*x) / a**3, 
        -x**2 * (a - x) / a**2
    ], dtype=float)

    Beam.Mesh.shape_fun_p = lambda x, a: np.array([
        -(6*x * (a - x)) / a**3, 
        1 - (x * (4*a - 3*x)) / a**2, 
        (6*x * (a - x)) / a**3, 
        -(x * (2*a - 3*x)) / a**2
    ], dtype=float)

    Beam.Mesh.shape_fun_pp = lambda x, a: np.array([
        (12*x) / a**3 - 6 / a**2, 
        (6*x) / a**2 - 4 / a, 
        6 / a**2 - (12*x) / a**3, 
        (6*x) / a**2 - 2 / a
    ], dtype=float)

    # ---- Stiffness and Consistent Mass Matrices ----
    for ele_num in range(Beam.Mesh.Ele.Tnum):

        # Element matrices
        Me = b04_beam_ele_m(Beam.Prop.rho_n[ele_num], Beam.Prop.A_n[ele_num], Beam.Mesh.Ele.a[ele_num])
        Ke = b05_beam_ele_k(Beam.Prop.E_n[ele_num] * Beam.Prop.I_n[ele_num], Beam.Mesh.Ele.a[ele_num])
        
        # Element DOFs (0-based from B01)
        dof = Beam.Mesh.Ele.DOF[ele_num]
        
        # np.ix_ creates the open mesh to index the submatrix block correctly
        idx = np.ix_(dof, dof)
        
        # Assembly of matrices
        Beam.Mesh.Kg[idx] += Ke
        CMM[idx] += Me

    # Additional mass assignment block
    if hasattr(Beam.Prop, 'm2'):
        # 0-based indices for the first and last element
        end_elements = [0, Beam.Mesh.Ele.Tnum - 1]
        for ele_num in end_elements:
            # Element matrices
            Me = b04_beam_ele_m(Beam.Prop.m2 / Beam.Mesh.Ele.a[ele_num], 1, Beam.Mesh.Ele.a[ele_num])
            
            dof = Beam.Mesh.Ele.DOF[ele_num]
            idx = np.ix_(dof, dof)
            
            # Assembly of matrices
            CMM[idx] += Me

    # ---- Checking Mass Matrix consistency option ----
    if Beam.Options.k_Mconsist != 1:

        # ---- Lumped Mass Matrix (LMM) ----
        # Initialize matrix with the same sparsity structure as Kg but empty
        LMM = lil_matrix((Tnum, Tnum), dtype=float)
        
        for ele_num in range(Beam.Mesh.Ele.Tnum):

            # Element matrix
            scalar_val = Beam.Prop.rho_n[ele_num] * Beam.Prop.A_n[ele_num] * Beam.Mesh.Ele.a[ele_num]
            Me = scalar_val * np.diag([0.5, 0, 0.5, 0])
            
            dof = Beam.Mesh.Ele.DOF[ele_num]
            idx = np.ix_(dof, dof)
            
            # LMM assembly
            LMM[idx] += Me

        # Mass Matrix
        Beam.Mesh.Mg = Beam.Options.k_Mconsist * CMM + (1 - Beam.Options.k_Mconsist) * LMM
        
    else:
        Beam.Mesh.Mg = CMM

    # ---- Application of boundary conditions ----
    # Diagonal elements equal 1, and columns and rows equal zero for bc DOF

    # Support DOF with values
    for i in range(Beam.BC.num_DOF_with_values):
        dof = Beam.BC.DOF_with_values[i]
        Beam.Mesh.Kg[dof, dof] += Beam.BC.DOF_stiff_values[i]

    # Fixed DOF
    fixed_dofs = Beam.BC.DOF_fixed
    
    # Zero out rows and columns for fixed DOFs
    Beam.Mesh.Kg[fixed_dofs, :] = 0
    Beam.Mesh.Kg[:, fixed_dofs] = 0
    Beam.Mesh.Mg[fixed_dofs, :] = 0
    Beam.Mesh.Mg[:, fixed_dofs] = 0
    
    # Set the diagonal elements for the fixed DOFs to the fixed value constraint
    for dof in fixed_dofs:
        Beam.Mesh.Kg[dof, dof] = Beam.BC.DOF_fixed_value
        Beam.Mesh.Mg[dof, dof] = Beam.BC.DOF_fixed_value

    # Optional but highly recommended: Convert LIL matrices to CSR format for faster math operations later
    Beam.Mesh.Kg = Beam.Mesh.Kg.tocsr()
    Beam.Mesh.Mg = Beam.Mesh.Mg.tocsr()

    return Beam

# ---- End of function ----