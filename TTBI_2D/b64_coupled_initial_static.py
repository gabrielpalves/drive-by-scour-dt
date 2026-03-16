import numpy as np
from scipy.sparse import lil_matrix, issparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Dummy class for structure initialization
class EmptyObj:
    pass

def b64_coupled_initial_static(Veh, Model, Calc, Track):
    """
    Calculation of initial static deformation of vehicle and track.
    The coupled stiffness matrix is defined and solved for the external forces.
    """

    Sol = EmptyObj()
    Sol.Veh = [EmptyObj() for _ in range(len(Veh))]
    Sol.Model = EmptyObj()
    Sol.Model.Nodal = EmptyObj()

    Coup = EmptyObj()
    Coup.DOF = EmptyObj()
    Coup.BC = EmptyObj()
    
    # ---- Initialize variables ----
    # 0-based indexing adjustments for global_ind
    veh_end_global_ind = Veh[-1].global_ind[-1]
    Coup.DOF.Tnum = veh_end_global_ind + 1 + Model.Mesh.DOF.Tnum
    
    # Use LIL sparse matrix for efficient incremental block assignment
    Coup.Kg = lil_matrix((Coup.DOF.Tnum, Coup.DOF.Tnum), dtype=float)
    
    # We optionally can define Mg and Cg to mirror MATLAB, though only Kg is solved here
    Coup.Mg = lil_matrix((Coup.DOF.Tnum, Coup.DOF.Tnum), dtype=float)
    Coup.Cg = lil_matrix((Coup.DOF.Tnum, Coup.DOF.Tnum), dtype=float)

    # Force vector is dense
    Coup.F = np.zeros((Coup.DOF.Tnum, 1), dtype=float)

    # If not redux model
    if getattr(Calc.Options, 'redux', 0) == 0:

        # **** With VBI ***
        if getattr(Calc.Options, 'VBI', 1) == 1:

            # Vehicles contributions
            for v in range(len(Veh)):

                # Vehicle's Diagonal block matrices
                veh_idx = np.ix_(Veh[v].global_ind, Veh[v].global_ind)
                Coup.Kg[veh_idx] = Veh[v].SysM.K

                for wheel in range(Veh[v].Wheels.num):

                    # Element to which each wheel belongs to
                    ele_num = int(Calc.Veh[v].elexj[wheel, 0])
                    
                    if ele_num < 0:
                        continue # Wheel is out of bounds, skip force/stiffness mapping

                    # Distance from each x_path to its left node
                    x = Calc.Veh[v].xj[wheel, 0]
                    # Element dimension
                    a = Track.Rail.Mesh.Ele.a[ele_num]
                    
                    # Element shape functions at x (reshape to column vector)
                    shape_fun_at_x = Track.Rail.Mesh.shape_fun(x, a).reshape(-1, 1)

                    # Addition primary suspension stiffness to Track
                    # Shift indices by the total vehicle DOFs to place it in the track section of the global matrix
                    track_offset = veh_end_global_ind + 1
                    eq_num = track_offset + Track.Rail.Mesh.Ele.DOF[ele_num]
                    
                    NN = shape_fun_at_x @ shape_fun_at_x.T
                    eq_idx = np.ix_(eq_num, eq_num)
                    Coup.Kg[eq_idx] += NN * Veh[v].Susp.Prim.k[wheel]

                    # Off-diagonal block matrices
                    # Vehicle's Node to wheel displacements (ensure 2D array for matrix mult)
                    N2w = Veh[v].Wheels.N2w[wheel, :].reshape(1, -1)
                    
                    # Off diagonal block matrix
                    OffDiagBlockMat = -(shape_fun_at_x @ N2w) * Veh[v].Susp.Prim.k[wheel]
                    
                    # Addition to Coupled stiffness matrix
                    rows = Veh[v].global_ind
                    cols = eq_num
                    
                    idx_rc = np.ix_(rows, cols)
                    idx_cr = np.ix_(cols, rows)
                    
                    Coup.Kg[idx_rc] += OffDiagBlockMat.T
                    Coup.Kg[idx_cr] += OffDiagBlockMat

                    # Force vector
                    Coup.F[cols, 0] += (Veh[v].Wheels.m[wheel] * shape_fun_at_x.flatten() * Calc.Cte.grav)

                # Force vector for vehicle body
                veh_forces = Veh[v].SysM.M @ (Veh[v].DOF.vert * Calc.Cte.grav)
                Coup.F[Veh[v].global_ind, 0] += veh_forces.flatten()

            # Track contribution
            track_dofs = np.arange(track_offset, Coup.DOF.Tnum)
            track_idx = np.ix_(track_dofs, track_dofs)
            Coup.Kg[track_idx] += Model.Mesh.Kg

            # Re-apply the boundary conditions
            Coup.BC.DOF_fixed = track_offset + Model.BC.DOF_fixed
            Coup.BC.num_DOF_fixed = len(Coup.BC.DOF_fixed)
            fixed_dofs = Coup.BC.DOF_fixed
            
            # Zero out rows and columns for fixed DOFs
            Coup.Mg[fixed_dofs, :] = 0; Coup.Mg[:, fixed_dofs] = 0
            Coup.Cg[fixed_dofs, :] = 0; Coup.Cg[:, fixed_dofs] = 0
            Coup.Kg[fixed_dofs, :] = 0; Coup.Kg[:, fixed_dofs] = 0
            
            for dof in fixed_dofs:
                Coup.Mg[dof, dof] = Model.BC.DOF_fixed_value
                Coup.Kg[dof, dof] = Model.BC.DOF_fixed_value
                
            Coup.F[fixed_dofs, 0] = 0

        # **** Moving Force ****
        elif getattr(Calc.Options, 'VBI', 1) == 0:

            # Vehicles contributions
            for v in range(len(Veh)):
                # Vehicle's Diagonal block matrices
                veh_idx = np.ix_(Veh[v].global_ind, Veh[v].global_ind)
                Coup.Kg[veh_idx] = Veh[v].SysM.K

                # Force vector
                veh_forces = Veh[v].SysM.M @ (Veh[v].DOF.vert * Calc.Cte.grav)
                Coup.F[Veh[v].global_ind, 0] += veh_forces

            # Track contribution
            track_offset = veh_end_global_ind + 1
            track_dofs = np.arange(track_offset, Coup.DOF.Tnum)
            track_idx = np.ix_(track_dofs, track_dofs)
            Coup.Kg[track_idx] += Model.Mesh.Kg

        # Coupled system static solution
        # Convert LIL to CSR format for the fast sparse solver
        Coup.Kg_csr = Coup.Kg.tocsr()
        
        # spsolve acts as MATLAB's backslash '\' for sparse matrices
        # It returns a 1D array, so we reshape it back to a column vector
        Coup.U0 = spsolve(Coup.Kg_csr, Coup.F).reshape(-1, 1)

    elif getattr(Calc.Options, 'redux', 0) == 1:

        Coup.U0 = np.zeros((Coup.DOF.Tnum, 1))

        # Vehicles initial deformations
        for v in range(len(Veh)):
            # Assuming Veh[v].U0 is a 1D or column array
            Coup.U0[Veh[v].global_ind, 0] = Veh[v].U0.flatten()
            
    # Dividing the results into Sol.Veh and Sol.Model
    track_offset = veh_end_global_ind + 1
    
    for v in range(len(Veh)):
        Sol.Veh[v].U0 = Coup.U0[Veh[v].global_ind, 0]
        Sol.Veh[v].V0 = Sol.Veh[v].U0 * 0.0
        Sol.Veh[v].A0 = Sol.Veh[v].U0 * 0.0
        
    Sol.Model.Nodal.U0 = Coup.U0[track_offset:, 0]
    Sol.Model.Nodal.V0 = Sol.Model.Nodal.U0 * 0.0
    Sol.Model.Nodal.A0 = Sol.Model.Nodal.U0 * 0.0

    # # Graphical Check
    # plt.figure()
    # 
    # rows = track_offset + Model.Mesh.DOF.rail_vert
    # 
    # plt.subplot(2, 1, 1)
    # plt.box(True)
    # for v in range(len(Veh)):
    #     # Ensure proper matrix multiplication
    #     veh_u0 = Coup.U0[Veh[v].global_ind].reshape(-1, 1)
    #     veh_wheels_u = Veh[v].Wheels.N2w @ veh_u0
    #     plt.plot(Calc.Veh[v].x_path[:, 0], veh_wheels_u.flatten() * 1000, 'r.', markersize=10)
    # 
    # plt.xlim([Calc.Veh[-1].x_path[-1, 0], Model.Mesh.XLoc.rail_vert[-1]])
    # plt.xlabel('Distance (m)')
    # plt.ylabel('Vehicles Vert. Disp. (mm)')
    # 
    # plt.subplot(2, 1, 2)
    # plt.plot(Model.Mesh.XLoc.rail_vert, Coup.U0[rows].flatten() * 1000)
    # plt.xlim([Calc.Veh[-1].x_path[-1, 0], Model.Mesh.XLoc.rail_vert[-1]])
    # plt.xlabel('Distance (m)')
    # plt.ylabel('Rail Vert. Disp. (mm)')
    # 
    # plt.tight_layout()
    # plt.show()

    return Sol

# ---- End of script ----