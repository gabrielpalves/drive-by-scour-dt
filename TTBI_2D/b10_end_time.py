import numpy as np

def b10_end_time(Calc):
    """
    Calculates the time t_end, when the load reaches the end of the path
    and generates other position, velocities and acceleration variables.
    """
    
    dt = 1.0  # Starting time incremental step
    
    # Auxiliary variables
    L = Calc.Position.x_end - Calc.Position.x_0
    k_cont = 1
    counter = 0
    max_counter = 100 + L / dt
    t1 = 0.0

    v0 = Calc.Position.v_0
    a0 = Calc.Position.a_0
    aa = Calc.Position.aa
    tol = Calc.Cte.tol

    # Solution search
    while k_cont == 1:
        t = np.array([t1, t1 + dt])
        counter += 1
        
        # Calculate position at bounds
        Lt = v0 * t + a0 * (t ** aa)
        
        # np.sign returns -1, 0, or 1. np.sum checks if L crossed the boundary
        sign_sum = np.sum(np.sign(L - Lt))
        
        if sign_sum == 0:
            # The target L is between Lt[0] and Lt[1]. Reduce time step and refine.
            if dt > tol:
                dt = dt / 2.0
            else:
                k_cont = 0
        elif sign_sum == 1:
            # Reached exact boundary limit
            t = np.array([t[1], t[1]])
            k_cont = 0
        else:
            # Target L not reached yet. Move window forward.
            t1 = t1 + dt
            
        if counter >= max_counter:
            print('Initial position / Velocity / Acceleration are WRONG!!!')
            raise ValueError('Initial position / Velocity / Acceleration are WRONG!!!')

    # -- Output calculation and generation --
    Calc.Time.t_end = np.mean(t)
    
    Calc.Position.v_end = v0 + aa * a0 * (Calc.Time.t_end ** (aa - 1))
    Calc.Position.a_end = aa * (aa - 1) * a0 * (Calc.Time.t_end ** (aa - 2))
    
    Calc.Position.v_max = max(v0, Calc.Position.v_end)
    Calc.Position.v_min = min(v0, Calc.Position.v_end)
    Calc.Position.a_max = max(a0, Calc.Position.a_end)
    Calc.Position.a_min = min(a0, Calc.Position.a_end)

    return Calc