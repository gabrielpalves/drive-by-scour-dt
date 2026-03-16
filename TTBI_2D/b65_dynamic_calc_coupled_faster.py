import time
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def b65_dynamic_calc_coupled_faster(Veh, Model, Calc, Track, Sol, Damage):
    """
    Faster alternative version of script B65. The main difference is that the
    system matrices are assembled using the sparse() command.
    """

    # ---- Initialize variables ----
    veh_end_global_ind = Veh[-1].global_ind[-1]
    Coup_DOF_Tnum = veh_end_global_ind + 1 + Model.Mesh.DOF.Tnum
    num_t = Calc.Solver.num_t
    dt = Calc.Solver.dt

    # We use LIL format for initial uncoupled matrix building
    UnCoup_Kg = lil_matrix((Coup_DOF_Tnum, Coup_DOF_Tnum), dtype=float)
    UnCoup_Cg = lil_matrix((Coup_DOF_Tnum, Coup_DOF_Tnum), dtype=float)
    UnCoup_Mg = lil_matrix((Coup_DOF_Tnum, Coup_DOF_Tnum), dtype=float)
    UnCoup_F = np.zeros((Coup_DOF_Tnum, 1), dtype=float) # Dense vector

    Coup_U = np.zeros((Coup_DOF_Tnum, num_t), dtype=float)
    Coup_V = np.zeros((Coup_DOF_Tnum, num_t), dtype=float)
    Coup_A = np.zeros((Coup_DOF_Tnum, num_t), dtype=float)

    # ---- Coupled equations BC ----
    Coup_BC_DOF_fixed = veh_end_global_ind + 1 + Model.BC.DOF_fixed
    Coup_BC_num_DOF_fixed = len(Coup_BC_DOF_fixed)

    # ---- Auxiliary variables ----
    disp_every_t = getattr(Calc.Options, 'disp_every', 2.0)
    start_time = time.time()
    last_display_time = disp_every_t
    
    vel = Veh[0].vel
    vel2 = vel**2
    ele_DOF = Track.Rail.Mesh.Ele.DOF
    grav = Calc.Cte.grav

    beta = Calc.Solver.NewMark_beta
    delta = Calc.Solver.NewMark_delta

    # Newmark-Beta constants
    NB_cte = np.array([
        1.0 / (beta * dt**2),
        delta / (beta * dt),
        1.0 / (beta * dt),
        (1.0 / (2 * beta)) - 1.0,
        1.0 - (delta / beta),
        (1.0 - delta / (2 * beta)) * dt
    ])

    # ---- Initial deformation ----
    for v in range(len(Veh)):
        Coup_U[Veh[v].global_ind, 0] = Sol.Veh[v].U0
        Coup_V[Veh[v].global_ind, 0] = Sol.Veh[v].V0
        Coup_A[Veh[v].global_ind, 0] = Sol.Veh[v].A0
        
    track_offset = veh_end_global_ind + 1
    Coup_U[track_offset:, 0] = Sol.Model.Nodal.U0
    Coup_V[track_offset:, 0] = Sol.Model.Nodal.V0
    Coup_A[track_offset:, 0] = Sol.Model.Nodal.A0

    # ---- Uncoupled System Matrices ----
    # Vehicles contributions
    for v in range(len(Veh)):
        idx = np.ix_(Veh[v].global_ind, Veh[v].global_ind)
        UnCoup_Kg[idx] = Veh[v].SysM.K
        UnCoup_Cg[idx] = Veh[v].SysM.C
        UnCoup_Mg[idx] = Veh[v].SysM.M
        # Force vector
        vector = Veh[v].SysM.M @ (Veh[v].DOF.vert * grav)
        UnCoup_F[Veh[v].global_ind, 0] = vector.flatten()

    # Track contribution
    track_idx = np.ix_(np.arange(track_offset, Coup_DOF_Tnum), np.arange(track_offset, Coup_DOF_Tnum))
    UnCoup_Kg[track_idx] = Model.Mesh.Kg
    UnCoup_Cg[track_idx] = Model.Mesh.Cg
    UnCoup_Mg[track_idx] = Model.Mesh.Mg

    # Convert to CSR format for fast mathematical addition in the loop
    UnCoup_Kg = UnCoup_Kg.tocsr()
    UnCoup_Cg = UnCoup_Cg.tocsr()
    UnCoup_Mg = UnCoup_Mg.tocsr()

    # **** With VBI ***
    if getattr(Calc.Options, 'VBI', 1) == 1:

        # --------------------------- Time Step Loop ------------------------------
        for t in range(num_t - 1): # t goes from 0 to num_t-2

            # Progress display inline logic
            elapsed_time = time.time() - start_time
            if elapsed_time > last_display_time:
                pct = round(((t) / num_t) * 100, 2)
                print(f"Time step {t} of {num_t} ({pct}%)")
                last_display_time += disp_every_t

            # ---- Time dependent system matrices ----
            # We will use coordinate lists to build sparse matrices quickly, bypassing LIL entirely
            eq_num1, eq_num2 = [], []
            vals_Kg, vals_Cg, vals_Mg = [], [], []
            
            eq_num1_off, eq_num2_off = [], []
            vals_Kg_off, vals_Cg_off = [], []

            Coup_F = UnCoup_F.copy()

            # Vehicles contributions
            for v in range(len(Veh)):
                ks = Veh[v].Susp.Prim.k
                cs = Veh[v].Susp.Prim.c
                ms = Veh[v].Wheels.m
                
                ele_num_t_1 = Calc.Veh[v].elexj[:, t + 1]
                N2w_wheels = Veh[v].Wheels.N2w
                
                h_path_t_1 = Calc.Veh[v].h_path[:, t + 1]
                hd_path_t_1 = Calc.Veh[v].hd_path[:, t + 1]
                hdd_path_t_1 = Calc.Veh[v].hdd_path[:, t + 1]
                rows = Veh[v].global_ind 

                for wheel in range(Veh[v].Wheels.num):
                    ele_num = int(ele_num_t_1[wheel])

                    if ele_num >= 0: # 0-based valid element check

                        x = Calc.Veh[v].xj[wheel, t + 1]
                        a = Track.Rail.Mesh.Ele.a[ele_num]

                        # Element shape functions at x (reshape to column vectors)
                        shape_at_x = Track.Rail.Mesh.shape_fun(x, a).reshape(-1, 1)
                        shape_at_x_p = Track.Rail.Mesh.shape_fun_p(x, a).reshape(-1, 1)
                        shape_at_x_pp = Track.Rail.Mesh.shape_fun_pp(x, a).reshape(-1, 1)

                        # Auxiliary matrices
                        NN = shape_at_x @ shape_at_x.T
                        NNp = shape_at_x @ shape_at_x_p.T
                        NNpp = shape_at_x @ shape_at_x_pp.T

                        # Addition primary suspension properties to Track
                        cols = track_offset + ele_DOF[ele_num]
                        
                        eq_num1.extend(np.tile(cols, 4))
                        eq_num2.extend(np.repeat(cols, 4))
                        
                        vals_Kg_aux = NN * ks[wheel] + cs[wheel] * vel * NNp + ms[wheel] * vel2 * NNpp
                        vals_Cg_aux = NN * cs[wheel] + 2 * ms[wheel] * vel * NNp
                        vals_Mg_aux = NN * ms[wheel]

                        # Fortran flattening ensures columns map to np.tile/repeat coordinate logic
                        vals_Kg.extend(vals_Kg_aux.flatten('F'))
                        vals_Cg.extend(vals_Cg_aux.flatten('F'))
                        vals_Mg.extend(vals_Mg_aux.flatten('F'))

                        # Off-diagonal block matrices
                        N2w = N2w_wheels[wheel, :].reshape(1, -1)

                        OffDiagBlockMat = -(shape_at_x @ N2w)
                        OffDiagBlockMat_d = -(shape_at_x_p @ N2w) * vel

                        # Addition to Coupled stiffness matrix
                        Tnum_DOF = len(rows)
                        eq_num1_off.extend(np.tile(rows, 4))
                        eq_num1_off.extend(np.repeat(cols, Tnum_DOF))
                        
                        eq_num2_off.extend(np.repeat(cols, Tnum_DOF))
                        eq_num2_off.extend(np.tile(rows, 4))
                        
                        vals_Kg_aux1 = (OffDiagBlockMat * ks[wheel] + OffDiagBlockMat_d * cs[wheel]).T
                        vals_Kg_aux2 = (OffDiagBlockMat * ks[wheel]).T
                        
                        vals_Cg_aux_off = (OffDiagBlockMat * cs[wheel]).T

                        vals_Kg_off.extend(vals_Kg_aux1.flatten('F'))
                        vals_Kg_off.extend(vals_Kg_aux2.flatten('F'))
                        
                        vals_Cg_off.extend(vals_Cg_aux_off.flatten('F'))
                        vals_Cg_off.extend(vals_Cg_aux_off.flatten('F'))

                        # Force vector
                        force_part1 = (ms[wheel] * grav - ms[wheel] * hdd_path_t_1[wheel]) * shape_at_x
                        force_part2 = (ks[wheel] * h_path_t_1[wheel] + cs[wheel] * hd_path_t_1[wheel]) * np.concatenate((N2w.T, -shape_at_x))
                        
                        # Add dense vector directly
                        Coup_F[cols, 0] += force_part1.flatten()
                        Coup_F[rows, 0] += force_part2[:Tnum_DOF].flatten()
                        Coup_F[cols, 0] += force_part2[Tnum_DOF:].flatten()

            # Adding the coupling terms as sparse elements
            shape = (Coup_DOF_Tnum, Coup_DOF_Tnum)
            
            delta_Kg_1 = coo_matrix((vals_Kg, (eq_num1, eq_num2)), shape=shape)
            delta_Kg_2 = coo_matrix((vals_Kg_off, (eq_num1_off, eq_num2_off)), shape=shape)
            Coup_Kg = UnCoup_Kg + (delta_Kg_1 + delta_Kg_2).tocsr()
            
            delta_Cg_1 = coo_matrix((vals_Cg, (eq_num1, eq_num2)), shape=shape)
            delta_Cg_2 = coo_matrix((vals_Cg_off, (eq_num1_off, eq_num2_off)), shape=shape)
            Coup_Cg = UnCoup_Cg + (delta_Cg_1 + delta_Cg_2).tocsr()
            
            delta_Mg = coo_matrix((vals_Mg, (eq_num1, eq_num2)), shape=shape)
            Coup_Mg = UnCoup_Mg + delta_Mg.tocsr()

            # Re-apply the boundary conditions using LIL conversion (safest way to zero out rows/cols)
            Coup_Kg = Coup_Kg.tolil()
            Coup_Cg = Coup_Cg.tolil()
            Coup_Mg = Coup_Mg.tolil()
            
            Coup_Mg[Coup_BC_DOF_fixed, :] = 0; Coup_Mg[:, Coup_BC_DOF_fixed] = 0
            Coup_Cg[Coup_BC_DOF_fixed, :] = 0; Coup_Cg[:, Coup_BC_DOF_fixed] = 0
            Coup_Kg[Coup_BC_DOF_fixed, :] = 0; Coup_Kg[:, Coup_BC_DOF_fixed] = 0
            
            for dof in Coup_BC_DOF_fixed:
                Coup_Mg[dof, dof] = Model.BC.DOF_fixed_value
                Coup_Kg[dof, dof] = Model.BC.DOF_fixed_value
                
            Coup_F[Coup_BC_DOF_fixed, 0] = 0

            # Convert back to CSR for fast solving
            Coup_Kg = Coup_Kg.tocsr()
            Coup_Cg = Coup_Cg.tocsr()
            Coup_Mg = Coup_Mg.tocsr()

            # ---- Direct integration ----
            # -- Newmark-Beta --
            effKg = Coup_Kg + NB_cte[0] * Coup_Mg + NB_cte[1] * Coup_Cg
            
            # Formulate the state vectors
            U_t = Coup_U[:, t].reshape(-1, 1)
            V_t = Coup_V[:, t].reshape(-1, 1)
            A_t = Coup_A[:, t].reshape(-1, 1)
            
            A = U_t * NB_cte[0] + V_t * NB_cte[2] + A_t * NB_cte[3]
            B = U_t * NB_cte[1] - V_t * NB_cte[4] - A_t * NB_cte[5]
            
            # Solve system
            rhs = Coup_F + Coup_Mg @ A + Coup_Cg @ B
            Coup_U[:, t + 1] = spsolve(effKg, rhs)
            
            # Update kinematics
            U_next = Coup_U[:, t + 1].reshape(-1, 1)
            Coup_V[:, t + 1] = (NB_cte[1] * U_next - B).flatten()
            Coup_A[:, t + 1] = (U_next * NB_cte[0] - A).flatten()

    # **** Moving Forces ***
    elif getattr(Calc.Options, 'VBI', 1) == 0:
        print('No VBI!!!')

        Coup_Kg = UnCoup_Kg
        Coup_Cg = UnCoup_Cg
        Coup_Mg = UnCoup_Mg

        effKg = Coup_Kg + Coup_Mg * NB_cte[0] + Coup_Cg * NB_cte[1]
        
        # effKg is constant in Moving Force, so we can pre-factorize it for extreme speed, 
        # but to match MATLAB perfectly, we just use spsolve inside the loop.

        for t in range(num_t - 1):
            
            elapsed_time = time.time() - start_time
            if elapsed_time > last_display_time:
                pct = round(((t) / num_t) * 100, 2)
                print(f"Time step {t} of {num_t} ({pct}%)")
                last_display_time += disp_every_t

            Coup_F = UnCoup_F.copy()

            for v in range(len(Veh)):
                for wheel in range(Veh[v].Wheels.num):
                    ele_num = int(Calc.Veh[v].elexj[wheel, t + 1])
                    if ele_num >= 0:
                        x = Calc.Veh[v].xj[wheel, t + 1]
                        a = Track.Rail.Mesh.Ele.a[ele_num]
                        shape_at_x = Track.Rail.Mesh.shape_fun(x, a)
                        
                        cols = track_offset + ele_DOF[ele_num]
                        Coup_F[cols, 0] += Veh[v].sta_loads[wheel] * shape_at_x

            Coup_F[Coup_BC_DOF_fixed, 0] = 0

            # ---- Direct integration ----
            U_t = Coup_U[:, t].reshape(-1, 1)
            V_t = Coup_V[:, t].reshape(-1, 1)
            A_t = Coup_A[:, t].reshape(-1, 1)

            A = U_t * NB_cte[0] + V_t * NB_cte[2] + A_t * NB_cte[3]
            B = U_t * NB_cte[1] - V_t * NB_cte[4] - A_t * NB_cte[5]

            rhs = Coup_F + Coup_Mg @ A + Coup_Cg @ B
            Coup_U[:, t + 1] = spsolve(effKg, rhs)

            U_next = Coup_U[:, t + 1].reshape(-1, 1)
            Coup_V[:, t + 1] = (NB_cte[1] * U_next - B).flatten()
            Coup_A[:, t + 1] = (U_next * NB_cte[0] - A).flatten()

    # ---- Output generation ----
    for v in range(len(Veh)):
        Sol.Veh[v].U = Coup_U[Veh[v].global_ind, :]
        Sol.Veh[v].V = Coup_V[Veh[v].global_ind, :]
        Sol.Veh[v].A = Coup_A[Veh[v].global_ind, :]
        
        # Generates measurement noise on the first 3 DOFs (simulating sensors)
        noise = Damage.desvio * Sol.Veh[v].A[0:3, :] * np.random.randn(3, num_t)
        Sol.Veh[v].A[0:3, :] += noise

    Sol.Model.Nodal.U = Coup_U[track_offset:, :]
    Sol.Model.Nodal.V = Coup_V[track_offset:, :]
    Sol.Model.Nodal.A = Coup_A[track_offset:, :]

    return Sol

# ---- End of script ----