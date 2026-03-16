import numpy as np
import math
import matplotlib.pyplot as plt

def b11_time_space_discretization(Calc):
    """
    Generates the uniformly spaced time discretization, 
    and corresponding space discretization.
    """
    
    # Calculates the appropriate time step length for the problem considering
    # the maximum frequency of interest
    Calc.Solver.t_steps_per_second = Calc.Solver.max_accurate_frq * 2
    
    # Time solver array
    num_points = math.ceil(Calc.Solver.t_steps_per_second * Calc.Time.t_end) + 1
    Calc.Solver.t = np.linspace(0, Calc.Time.t_end, num_points)
    
    # Solver sampling time 
    Calc.Solver.dt = Calc.Solver.t[1] - Calc.Solver.t[0]
    
    # Number of solver steps
    Calc.Solver.num_t = len(Calc.Solver.t)
    
    # Based on start and end position of vehicle 
    # calculation of corresponding start/end time indices
    aux1 = Calc.Position.v_0 * Calc.Solver.t + Calc.Position.a_0 * (Calc.Solver.t ** Calc.Position.aa)
    
    dx_start_arr = np.abs(np.diff(aux1))
    dx_end = dx_start_arr[-1]
    dx_start = dx_start_arr[0]
    
    # Python 0-based Indexing
    Calc.Time.t_0_ind = 0
    
    if Calc.Position.x_0 > 0:
        aux2 = np.linspace(0, Calc.Position.x_0, int(round(Calc.Position.x_0 / dx_start)) + 1)
        # 0-based index targeting the last element of aux2
        Calc.Time.t_0_ind = len(aux2) - 1 
        aux1 = np.concatenate((aux2[:-1], aux1 + Calc.Position.x_0))
        
    if Calc.Position.x_end < Calc.Profile.L:
        aux2 = np.linspace(Calc.Position.x_end, Calc.Profile.L, int(round((Calc.Profile.L - Calc.Position.x_end) / dx_end)))
        aux1 = np.concatenate((aux1[:-1], aux2))
        
    Calc.Position.x = aux1
    Calc.Position.num_x = len(Calc.Position.x)
    
    # Final index mapped for Python 0-based slicing
    Calc.Time.t_end_ind = Calc.Time.t_0_ind + Calc.Solver.num_t - 1

    # ---- Plotting Results ----
    if getattr(Calc.Plot, 'P3_VehPos', 0) == 1:
        plt.figure(figsize=(8, 10))
        
        # 1) Load position in time
        plt.subplot(3, 1, 1)
        # Note: +1 in slice to include the t_end_ind bounds
        plt.plot(Calc.Position.x[Calc.Time.t_0_ind : Calc.Time.t_end_ind + 1], Calc.Solver.t)
        plt.xlabel('Vehicle Position (m) with respect to the Profile')
        plt.ylabel('Time (s)')
        plt.xlim([0, Calc.Position.x_end])
        plt.ylim([0, Calc.Time.t_end])
        plt.axvline(x=Calc.Profile.L, color='k', linestyle='--')
        plt.text(Calc.Profile.L, plt.ylim()[1], 'End of profile', horizontalalignment='right', verticalalignment='top')
        plt.title('FRONT wheel position, velocity and Acceleration')
        
        # 2) Load velocity in time
        plt.subplot(3, 1, 2)
        v_t = Calc.Position.v_0 + Calc.Position.aa * Calc.Position.a_0 * (Calc.Solver.t ** (Calc.Position.aa - 1))
        plt.plot(Calc.Solver.t, v_t)
        plt.xlabel('Time(s)')
        plt.ylabel('Velocity (m/s)')
        plt.xlim([0, Calc.Time.t_end])
        plt.ylim([0, Calc.Position.v_max])
        if Calc.Position.a_min == 0 and Calc.Position.a_max == 0:
            plt.ylim([0, Calc.Position.v_max * 1.2])
            
        # 3) Load acceleration in time
        plt.subplot(3, 1, 3)
        a_t = Calc.Position.aa * (Calc.Position.aa - 1) * Calc.Position.a_0 * (Calc.Solver.t ** (Calc.Position.aa - 2))
        plt.plot(Calc.Solver.t, a_t)
        plt.xlabel('Time(s)')
        plt.ylabel('Acceleration (m^2/s)')
        plt.xlim([0, Calc.Time.t_end])
        try: # In case both a_min and a_max are zero
            plt.ylim([min(0, Calc.Position.a_min) * 1.2, max(0, Calc.Position.a_max) * 1.2])
        except Exception:
            pass
            
        plt.tight_layout()
        plt.show()

    return Calc