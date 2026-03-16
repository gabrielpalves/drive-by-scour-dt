import numpy as np

def b18_train_veh_eq(Veh):
    """
    Generates the train vehicle equations for all the vehicles.
    """

    # Vehicle loop
    for v in range(len(Veh)):

        # Variables definition
        m = Veh[v].Body.m
        I = Veh[v].Body.I
        mB1 = Veh[v].Bogie.m[0]
        mB2 = Veh[v].Bogie.m[1]
        IB1 = Veh[v].Bogie.I[0]
        IB2 = Veh[v].Bogie.I[1]
        
        k1 = Veh[v].Susp.Prim.k[0]
        k2 = Veh[v].Susp.Prim.k[1]
        k3 = Veh[v].Susp.Prim.k[2]
        k4 = Veh[v].Susp.Prim.k[3]
        
        c1 = Veh[v].Susp.Prim.c[0]
        c2 = Veh[v].Susp.Prim.c[1]
        c3 = Veh[v].Susp.Prim.c[2]
        c4 = Veh[v].Susp.Prim.c[3]
        
        ks1 = Veh[v].Susp.Sec.k[0]
        ks2 = Veh[v].Susp.Sec.k[1]
        
        cs1 = Veh[v].Susp.Sec.c[0]
        cs2 = Veh[v].Susp.Sec.c[1]
        
        dB1_F = Veh[v].Bogie.L[0] / 2.0
        dB1_B = Veh[v].Bogie.L[1] / 2.0
        dB2_F = Veh[v].Bogie.L[0] / 2.0
        dB2_B = Veh[v].Bogie.L[1] / 2.0
        
        # Check for specific Body length fields
        if hasattr(Veh[v].Body, 'L_F') and Veh[v].Body.L_F is not None:
            d_F = Veh[v].Body.L_F
            d_B = Veh[v].Body.L_B
        else:
            d_F = Veh[v].Body.L / 2.0
            d_B = Veh[v].Body.L / 2.0

        # Create SysM structure if it doesn't exist
        if not hasattr(Veh[v], 'SysM'):
            class EmptyObj: pass
            Veh[v].SysM = EmptyObj()

        # Mass matrix
        Veh[v].SysM.M = np.array([
            [  m,   0,   0,   0,   0,   0],
            [  0, mB1,   0,   0,   0,   0],
            [  0,   0, mB2,   0,   0,   0],
            [  0,   0,   0,   I,   0,   0],
            [  0,   0,   0,   0, IB1,   0],
            [  0,   0,   0,   0,   0, IB2]
        ], dtype=float)

        # Damping matrix
        Veh[v].SysM.C = np.array([
            [cs1+cs2, -cs1, -cs2, cs1*d_F-cs2*d_B, 0, 0],
            [-cs1, c1+c2+cs1, 0, -cs1*d_F, c1*dB1_F-c2*dB1_B, 0],
            [-cs2, 0, c3+c4+cs2, cs2*d_B, 0, c3*dB2_F-c4*dB2_B],
            [d_F*cs1-cs2*d_B, -d_F*cs1, cs2*d_B, cs1*d_F**2+cs2*d_B**2, 0, 0],
            [0, dB1_F*c1-dB1_B*c2, 0, 0, c1*dB1_F**2+c2*dB1_B**2, 0],
            [0, 0, dB2_F*c3-dB2_B*c4, 0, 0, c3*dB2_F**2+c4*dB2_B**2]
        ], dtype=float)

        # Stiffness matrix
        Veh[v].SysM.K = np.array([
            [ks1+ks2, -ks1, -ks2, d_F*ks1-d_B*ks2, 0, 0],
            [-ks1, k1+k2+ks1, 0, -d_F*ks1, dB1_F*k1-dB1_B*k2, 0],
            [-ks2, 0, k3+k4+ks2, d_B*ks2, 0, dB2_F*k3-dB2_B*k4],
            [d_F*ks1-d_B*ks2, -d_F*ks1, d_B*ks2, d_F**2*ks1+d_B**2*ks2, 0, 0],
            [0, dB1_F*k1-dB1_B*k2, 0, 0, k1*dB1_F**2+k2*dB1_B**2, 0],
            [0, 0, dB2_F*k3-dB2_B*k4, 0, 0, k3*dB2_F**2+k4*dB2_B**2]
        ], dtype=float)

        # Total number of vehicle DOF
        Veh[v].Tnum_DOF = Veh[v].SysM.M.shape[0]

        # ---- Wheel displacements relation ----
        # Nodal displacements to wheel displacements
        Veh[v].Wheels.N2w = np.array([
            [0, 1, 0, 0, dB1_F, 0],
            [0, 1, 0, 0, -dB1_B, 0],
            [0, 0, 1, 0, 0, dB2_F],
            [0, 0, 1, 0, 0, -dB2_B]
        ], dtype=float)

        # Wheel mechanical properties
        Veh[v].ktn = Veh[v].Susp.Prim.k
        Veh[v].ctn = Veh[v].Susp.Prim.c
        Veh[v].mtn = Veh[v].Wheels.m

        # Local and global indices for the DOF (0-based for Python)
        Veh[v].local_ind = np.arange(Veh[v].Tnum_DOF)
        if v == 0:
            Veh[v].global_ind = Veh[v].local_ind
        else:
            Veh[v].global_ind = Veh[v - 1].global_ind[-1] + 1 + Veh[v].local_ind

        # Description of the DOF
        if not hasattr(Veh[v], 'DOF'):
            Veh[v].DOF = EmptyObj()
            
        # Reshaping to (6, 1) to ensure they act as column vectors like MATLAB
        Veh[v].DOF.vert = np.array([1, 1, 1, 0, 0, 0]).reshape(-1, 1)
        Veh[v].DOF.rot = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

    return Veh

# ---- End of script ----