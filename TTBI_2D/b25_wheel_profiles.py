import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt

# Dummy class for structure initialization
class EmptyObj:
    pass

def b25_wheel_profiles(Calc, Veh, Damage=None):
    """
    Calculates the profile under each wheel.

    Optional Damage argument may carry per-passage WHEEL damage descriptors
    (Stage 3 EOVs; NOT labels) — exact mirror of scour_MATLAB/B25_WheelProfiles.m,
    literature-anchored (docs/stage3_alldamage_spec.md):
        Damage.oor_flats  : rows [veh(1-based), wheel(1-based), flat_length_m,
                                  depth_m, phase_rad] -> periodic haversine dip,
                            period 2*pi*R. Depth PRE-COMPUTED by the caller
                            (fresh d=L^2/8R, run-in d=L^2/16R).
        Damage.oor_poly   : rows [veh, wheel, order_n, amp_m, phase_rad]
                            -> continuous polygonization amp*cos(n*x/R + phase).
        Damage.oor_radius : wheel radius R [m] (default 0.46)
    Both are added BEFORE the derivatives so hd/hdd carry them automatically.
    """

    # Ensure Calc.Veh exists as a list of objects
    if not hasattr(Calc, 'Veh'):
        Calc.Veh = [EmptyObj() for _ in range(len(Veh))]

    def _rows(name):
        v = getattr(Damage, name, None) if Damage is not None else None
        return np.atleast_2d(np.asarray(v, dtype=float)) if (v is not None and np.size(v)) else None
    flats = _rows('oor_flats')
    poly  = _rows('oor_poly')
    oor_R = float(getattr(Damage, 'oor_radius', 0.46) or 0.46) if Damage is not None else 0.46

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

        # ---- Wheel flats + polygonization (per-passage EOVs; NOT labels) ----
        # Added BEFORE the derivatives so hd/hdd carry them automatically.
        # Descriptor veh/wheel indices are 1-BASED (MATLAB convention).
        if flats is not None:
            for row in flats[flats[:, 0].astype(int) == v + 1]:
                wheel = int(row[1]) - 1
                if wheel < 0 or wheel >= num_wheels:
                    continue
                lf = row[2]                       # flat length [m]
                depth = row[3]                    # pre-computed depth [m]
                phase = row[4]                    # phase [rad]
                circ = 2.0 * np.pi * oor_R        # wheel circumference [m]
                s = np.mod(Calc.Veh[v].x_path[wheel, :] + phase / (2.0 * np.pi) * circ, circ)
                dip = -0.5 * depth * (1.0 - np.cos(2.0 * np.pi * s / lf)) * (s < lf)
                Calc.Veh[v].h_path[wheel, :] += dip
        if poly is not None:
            for row in poly[poly[:, 0].astype(int) == v + 1]:
                wheel = int(row[1]) - 1
                if wheel < 0 or wheel >= num_wheels:
                    continue
                n_ord, amp, phase = row[2], row[3], row[4]
                Calc.Veh[v].h_path[wheel, :] += amp * np.cos(
                    n_ord * Calc.Veh[v].x_path[wheel, :] / oor_R + phase)

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