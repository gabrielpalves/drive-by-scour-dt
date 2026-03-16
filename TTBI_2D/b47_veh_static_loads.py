import numpy as np

def b47_veh_static_loads(Veh, Calc):
    """
    Calculates the static loads of the vehicles using their system matrices.
    """

    for v in range(len(Veh)):
        
        # External forces (Mass matrix * gravity vector)
        # Using @ for matrix multiplication
        Fext = Veh[v].SysM.M @ (Veh[v].DOF.vert * Calc.Cte.grav)
        
        # System displacements (Solving K * U0 = Fext)
        Veh[v].U0 = np.linalg.solve(Veh[v].SysM.K, Fext)
        
        # Wheels displacements
        wheel_disp = Veh[v].Wheels.N2w @ Veh[v].U0
        
        # Reshape primary suspension stiffness and wheel mass arrays to column vectors
        k = np.array(Veh[v].Susp.Prim.k).reshape(-1, 1)
        m = np.array(Veh[v].Wheels.m).reshape(-1, 1)
        
        # Static load
        Veh[v].sta_loads = (k * wheel_disp) + (m * Calc.Cte.grav)
        
        # ---- Check ----
        # Calculate total mass of the vehicle (Body + Bogies + Wheels)
        total_mass = (
            Veh[v].Body.m + 
            np.sum(Veh[v].Bogie.m) + 
            np.sum(Veh[v].Wheels.m)
        )
        
        check = np.sum(Veh[v].sta_loads) - (total_mass * Calc.Cte.grav)
        
        if abs(check) > Calc.Cte.tol:
            print(f"Static weight of vehicle {v + 1} is not correct")

    return Veh

# ---- End of script ----