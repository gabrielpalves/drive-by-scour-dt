import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt

# Dummy class for structure initialization
class EmptyObj:
    pass

def b25_wheel_profiles(Calc, Veh):
    """
    Calculates the profile under each wheel
    """

    # Ensure Calc.Veh exists as a list of objects
    if not hasattr(Calc, 'Veh'):
        Calc.Veh = [EmptyObj() for _ in range(len(Veh))]

    num_t = Calc.Solver.num_t
    dt = Calc.Solver.dt

    for v in range(len(Veh)):
        
        num_wheels = Veh[v].Wheels.num
        
        # Initialize variables
        Calc.Veh[v].x_path = np.zeros((num_wheels, num_t))
        Calc.Veh[v].h_path = np.zeros((num_wheels, num_t))

        # Slicing time array 
        # Note: 0-based indexing. Adding +1 because Python slicing [start:stop] is exclusive of 'stop'
        t0 = int(Calc.Time.t_0_ind)
        t_end = int(Calc.Time.t_end_ind)
        x_time_slice = Calc.Position.x[t0 : t_end + 1]

        # Profile for each wheel
        for wheel in range(num_wheels):
            Calc.Veh[v].x_path[wheel, :] = x_time_slice - Veh[v].Ax_dist[wheel] - Veh[v].First_wheel_dist
            
            # Piecewise cubic interpolation (pchip)
            # extrapolate=False ensures out-of-bounds values return NaN, matching MATLAB's interp1 default behavior
            Calc.Veh[v].h_path[wheel, :] = pchip_interpolate(
                Calc.Profile.x, 
                Calc.Profile.h, 
                Calc.Veh[v].x_path[wheel, :]
            )

        # Removing NaN values (replacing with the first/last valid heights)
        Calc.Veh[v].h_path[Calc.Veh[v].x_path < Calc.Profile.x[0]] = Calc.Profile.h[0]
        Calc.Veh[v].h_path[Calc.Veh[v].x_path > Calc.Profile.x[-1]] = Calc.Profile.h[-1]

        # 1st derivative in time 
        # np.diff reduces the array dimension by 1, so we divide by dt and then prepend the first column
        hd_diff = np.diff(Calc.Veh[v].h_path, n=1, axis=1) / dt
        Calc.Veh[v].hd_path = np.column_stack((hd_diff[:, 0], hd_diff))

        # 2nd derivative in time
        hdd_diff = np.diff(Calc.Veh[v].hd_path, n=1, axis=1) / dt
        Calc.Veh[v].hdd_path = np.column_stack((hdd_diff[:, 0], hdd_diff))

        # First point of profile for each wheel at level zero
        # Using [:, 0:1] keeps it as a 2D column vector (N x 1), allowing NumPy to broadcast it cleanly across all time steps
        Calc.Veh[v].h_path = Calc.Veh[v].h_path - Calc.Veh[v].h_path[:, 0:1]

        # # Graphical Check
        # plt.figure()
        # 
        # plt.subplot(3, 1, 1)
        # plt.plot(Calc.Veh[v].x_path.T, Calc.Veh[v].h_path.T)
        # plt.plot(Calc.Veh[v].x_path[:, [0, -1]].T, Calc.Veh[v].h_path[:, [0, -1]].T, '.', markersize=20)
        # plt.ylabel('Profile')
        # 
        # plt.subplot(3, 1, 2)
        # plt.plot(Calc.Veh[v].x_path.T, Calc.Veh[v].hd_path.T)
        # plt.plot(Calc.Veh[v].x_path[:, [0, -1]].T, Calc.Veh[v].hd_path[:, [0, -1]].T, '.', markersize=20)
        # plt.ylabel('1st Derivative')
        # 
        # plt.subplot(3, 1, 3)
        # plt.plot(Calc.Veh[v].x_path.T, Calc.Veh[v].hdd_path.T)
        # plt.plot(Calc.Veh[v].x_path[:, [0, -1]].T, Calc.Veh[v].hdd_path[:, [0, -1]].T, '.', markersize=20)
        # plt.ylabel('2nd Derivative')
        # 
        # plt.tight_layout()
        # plt.show()

    return Calc

# ---- End of script ----