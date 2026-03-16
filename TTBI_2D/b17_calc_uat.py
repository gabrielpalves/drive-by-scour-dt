import numpy as np
import matplotlib.pyplot as plt

def b17_calc_uat(Sol, Track, Calc, Veh):
    """
    Function to calculate the vertical displacement of the model under the
    wheels of each vehicle in the Train.
    """

    num_t = int(Calc.Solver.num_t)

    # Vehicle number loop
    for v in range(len(Veh)):
        
        num_wheels = Veh[v].Wheels.num

        # Initialize output arrays
        Sol.Veh[v].def_under = np.zeros((num_wheels, num_t))
        Sol.Veh[v].vel_under = np.zeros((num_wheels, num_t))
        Sol.Veh[v].acc_under = np.zeros((num_wheels, num_t))
        Sol.Veh[v].def_under_p = np.zeros((num_wheels, num_t))
        Sol.Veh[v].def_under_pp = np.zeros((num_wheels, num_t))
        Sol.Veh[v].vel_under_p = np.zeros((num_wheels, num_t))

        # Calculation of deformation using shape functions and nodal displacements
        for wheel in range(num_wheels):
            
            # Extract the tracked element indices
            ele_num = Calc.Veh[v].elexj[wheel, :].astype(int)
            
            # Find the time indices where the wheel is actually on the bridge (valid elements >= 0)
            valid_t = np.where(ele_num >= 0)[0]

            if len(valid_t) == 0:
                continue # Skip if the wheel never enters the model in this time window

            # Extract the valid elements, local coordinates, and lengths
            elex = ele_num[valid_t]
            x = Calc.Veh[v].xj[wheel, valid_t]
            a = Track.Rail.Mesh.Ele.a[elex]

            # Evaluate shape functions in a fully vectorized manner
            # Track.Rail.Mesh.shape_fun returns an array of shape (4, len(valid_t))
            sf = Track.Rail.Mesh.shape_fun(x, a)
            sfp = Track.Rail.Mesh.shape_fun_p(x, a)
            sfpp = Track.Rail.Mesh.shape_fun_pp(x, a)

            # Get the 4 DOFs for the active elements. 
            # Transpose to shape (4, len(valid_t)) to match the shape functions
            dofs_T = Track.Rail.Mesh.Ele.DOF[elex, :].T

            # Extract nodal kinematics using advanced NumPy broadcasting
            # We use valid_t[np.newaxis, :] to broadcast the 1D time array across the 4 DOF rows
            time_idx = valid_t[np.newaxis, :]
            
            U_val = Sol.Model.Nodal.U[dofs_T, time_idx]
            V_val = Sol.Model.Nodal.V[dofs_T, time_idx]
            A_val = Sol.Model.Nodal.A[dofs_T, time_idx]

            # Compute the dot products instantaneously by summing down the rows (axis=0)
            Sol.Veh[v].def_under[wheel, valid_t] = np.sum(sf * U_val, axis=0)
            Sol.Veh[v].vel_under[wheel, valid_t] = np.sum(sf * V_val, axis=0)
            Sol.Veh[v].acc_under[wheel, valid_t] = np.sum(sf * A_val, axis=0)
            
            Sol.Veh[v].def_under_p[wheel, valid_t] = np.sum(sfp * U_val, axis=0)
            Sol.Veh[v].def_under_pp[wheel, valid_t] = np.sum(sfpp * U_val, axis=0)
            Sol.Veh[v].vel_under_p[wheel, valid_t] = np.sum(sfp * V_val, axis=0)

    # # -- Graphical Check --
    # v = 0 # Vehicle index (0-based)
    # 
    # plt.figure(figsize=(10, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].def_under.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Deformation (m)')
    # plt.title(f'Deformation under wheels of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.subplot(3, 1, 2)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].vel_under.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.title(f'Velocity under wheels of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.subplot(3, 1, 3)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].acc_under.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Acceleration (m/s^2)')
    # plt.title(f'Acceleration under wheels of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.tight_layout()
    # plt.show()
    # 
    # plt.figure(figsize=(10, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].def_under_p.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Deformation (m)')
    # plt.title(f'Deformation under wheels (1st derivative) of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.subplot(3, 1, 2)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].def_under_pp.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Deformation (m)')
    # plt.title(f'Deformation under wheels (2nd derivative) of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.subplot(3, 1, 3)
    # plt.plot(Calc.Solver.t, Sol.Veh[v].vel_under_p.T)
    # plt.xlabel('Solver time (s)')
    # plt.ylabel('Velocity (m/s^2)')
    # plt.title(f'Velocity under wheels (1st derivative) of vehicle {v + 1}')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # 
    # plt.tight_layout()
    # plt.show()

    return Sol

# ---- End of function ----