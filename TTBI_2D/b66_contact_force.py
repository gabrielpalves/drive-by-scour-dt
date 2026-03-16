import numpy as np
from b17_calc_uat import b17_calc_uat

def b66_contact_force(Sol, Track, Calc, Train):
    """
    Calculation of vertical contact forces for each vehicle.
    Note: Contact force calculation looks good only if Calc.Options.redux = 0
       When the redux model option is used, then the vehicle starts the simulation
       on a flat rigid surface. As the simulation progresses, each wheel enters
       the track and produces large vertical deformations on the track, which
       at the same time produces large oscillations in the contact force.
    """

    # Note: 'ones_1_x_num_t' is not needed in Python due to native broadcasting.

    # **** VBI ****
    if getattr(Calc.Options, 'VBI', 1) == 1:
        
        # Deformation under each wheel (Assuming b17_calc_uat is translated and imported)
        # Note: Pass Train.Veh, but the function returns Sol
        Sol = b17_calc_uat(Sol, Track, Calc, Train.Veh)

        # Force on Beam (Contact Force)
        for v in range(len(Train.Veh)):

            # Boolean mask converted to float for mathematical multiplication
            ind = (Calc.Veh[v].x_path >= 0).astype(float)
            
            # Extract variables for clean equations
            N2w = Train.Veh[v].Wheels.N2w
            U = Sol.Veh[v].U
            V = Sol.Veh[v].V
            vel = Train.vel
            grav = Calc.Cte.grav
            
            # Reshape parameters to column vectors (N x 1) for automatic NumPy broadcasting
            k = np.array(Train.Veh[v].Susp.Prim.k).reshape(-1, 1)
            c = np.array(Train.Veh[v].Susp.Prim.c).reshape(-1, 1)
            m = np.array(Train.Veh[v].Wheels.m).reshape(-1, 1)

            # Stiffness term
            term_k = ( (N2w @ U) - Sol.Veh[v].def_under - (Calc.Veh[v].h_path * ind) ) * k
            
            # Damping term
            term_c = ( (N2w @ V) - Sol.Veh[v].vel_under - (Sol.Veh[v].def_under_p * vel) - (Calc.Veh[v].hd_path * ind) ) * c
            
            # Inertial term
            term_m1 = -(Sol.Veh[v].acc_under + (Sol.Veh[v].def_under_pp * vel**2) + (2 * Sol.Veh[v].vel_under_p * vel)) * m
            
            # Static gravity term
            term_m2 = m * grav

            # Total Contact Force
            Sol.Veh[v].F_onBeam = term_k + term_c + term_m1 + term_m2

            Sol.Veh[v].F_onBeam_max = np.max(Sol.Veh[v].F_onBeam)
            Sol.Veh[v].F_onBeam_min = np.min(Sol.Veh[v].F_onBeam)

            # Checking contact (while on the bridge)
            L1 = Calc.Profile.L_Approach + Calc.Position.x_0 * Calc.Options.redux_factor
            L2 = Calc.Profile.L_Approach + Calc.Profile.L_bridge + Calc.Position.x_0 * Calc.Options.redux_factor
            
            # Boolean mask for positions on the bridge
            ind_bridge = (Calc.Veh[v].x_path >= L1) & (Calc.Veh[v].x_path <= L2)
            
            # Extract forces only when the wheel is on the bridge
            valid_forces = Sol.Veh[v].F_onBeam[ind_bridge]
            
            # Safety check: ensure the array isn't empty (e.g., train didn't reach bridge)
            if valid_forces.size > 0:
                Sol.Veh[v].F_onBridge_max = np.max(valid_forces)
                Sol.Veh[v].F_onBridge_min = np.min(valid_forces)
                Sol.Veh[v].contactLost = int(Sol.Veh[v].F_onBridge_max > 0)
            else:
                Sol.Veh[v].F_onBridge_max = 0
                Sol.Veh[v].F_onBridge_min = 0
                Sol.Veh[v].contactLost = 0

    # **** Moving Force ****
    elif getattr(Calc.Options, 'VBI', 1) == 0:

        # Force on Beam (Contact Force)
        for v in range(len(Train.Veh)):

            N2w = Train.Veh[v].Wheels.N2w
            U = Sol.Veh[v].U
            V = Sol.Veh[v].V
            grav = Calc.Cte.grav
            
            k = np.array(Train.Veh[v].Susp.Prim.k).reshape(-1, 1)
            c = np.array(Train.Veh[v].Susp.Prim.c).reshape(-1, 1)
            m = np.array(Train.Veh[v].Wheels.m).reshape(-1, 1)

            Sol.Veh[v].F_onBeam = ((N2w @ U) * k) + ((N2w @ V) * c) + (m * grav)
            
            Sol.Veh[v].F_onBeam_max = np.max(Sol.Veh[v].F_onBeam)
            Sol.Veh[v].F_onBeam_min = np.min(Sol.Veh[v].F_onBeam)

            # Checking contact (while on the bridge)
            L1 = Calc.Profile.L_Approach
            L2 = Calc.Profile.L_Approach + Calc.Profile.L_bridge
            
            ind_bridge = (Calc.Veh[v].x_path >= L1) & (Calc.Veh[v].x_path <= L2)
            
            valid_forces = Sol.Veh[v].F_onBeam[ind_bridge]
            
            if valid_forces.size > 0:
                Sol.Veh[v].contactLost = int(np.max(valid_forces) > 0)
            else:
                Sol.Veh[v].contactLost = 0

    # ---- Checking contact ----
    # Evaluate if any vehicle lost contact using a list comprehension
    contact_lost_flags = [getattr(veh, 'contactLost', 0) for veh in Sol.Veh]
    
    if max(contact_lost_flags) > 0:
        Sol.contactLost = 1
        print('There is no permanent contact between wheels and rail')
    else:
        Sol.contactLost = 0

    return Sol

# ---- End of script ----