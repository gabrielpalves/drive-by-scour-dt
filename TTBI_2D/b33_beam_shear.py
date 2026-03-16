import numpy as np

# Dummy class for structure initialization
class EmptyObj:
    pass

def b33_beam_shear(Sol, Model, Beam, Calc, calc_type):
    """
    Calculates the Shear of the beam using the nodal displacements
    """

    # Input processing
    if calc_type == 0:
        out_field = 'StaticShear'
        in_field = 'StaticU'
    elif calc_type == 1:
        out_field = 'Shear'
        in_field = 'U'

    # Ensure the target nested structure exists
    if not hasattr(Sol.Beam, out_field):
        setattr(Sol.Beam, out_field, EmptyObj())
        
    shear_obj = getattr(Sol.Beam, out_field)
    
    # Extract the full nodal displacements
    u_matrix = getattr(Sol.Model.Nodal, in_field)

    # Initialize variables
    shear_obj.xt = np.zeros((Beam.Mesh.Nodes.Tnum, Calc.Solver.num_t))

    # In-line functions (more efficient alternative to subfunctions)
    B32_Beam_ele_HS = lambda L, E, I: (E * I) * np.array([
        [12 / L**3, 6 / L**2, -12 / L**3, 6 / L**2],
        [12 / L**3, 6 / L**2, -12 / L**3, 6 / L**2]
    ], dtype=float)

    # ---- NO average nodal values ----
    if Calc.Options.Shear_calc_mode == 0:
        
        for ele in range(Beam.Mesh.Ele.Tnum):
            aux1 = B32_Beam_ele_HS(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
            dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
            
            # Note: There was a typo in your original MATLAB code here where it called `Sol.Model.Nodal.U` 
            # instead of `in_field`. I have corrected it below to use `u_matrix`.
            shear_obj.xt[ele, :] = aux1[0, :] @ u_matrix[dofs, :]

        # Final node
        ele = Beam.Mesh.Ele.Tnum - 1
        last_node = Beam.Mesh.Nodes.Tnum - 1
        
        aux1 = B32_Beam_ele_HS(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
        dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
        shear_obj.xt[last_node, :] = aux1[1, :] @ u_matrix[dofs, :]
        
    # ---- AVERAGE nodal values ----
    elif Calc.Options.Shear_calc_mode == 1:
        
        for ele in range(Beam.Mesh.Ele.Tnum):
            aux1 = B32_Beam_ele_HS(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
            dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
            
            shear_obj.xt[ele : ele + 2, :] += aux1 @ u_matrix[dofs, :]

        # Average of interior nodes with multiple calculations
        shear_obj.xt[1:-1, :] = shear_obj.xt[1:-1, :] / 2.0

    # ---- Additional Outputs ----

    # Maximum Shear Force
    max_cols = np.max(shear_obj.xt, axis=0)
    aux1 = np.argmax(shear_obj.xt, axis=0)
    
    aux2 = np.argmax(max_cols)
    shear_obj.max = max_cols[aux2]
    
    shear_obj.max_node = aux1[aux2]
    shear_obj.max_COP = Beam.Mesh.Nodes.acum[shear_obj.max_node]
    shear_obj.max_pCOP = (shear_obj.max_COP / Beam.Prop.L) * 100.0
    shear_obj.max_t_crit = Calc.Solver.t[aux2]
    
    if shear_obj.max_pCOP < 50.0:
        shear_obj.max_supp = np.max(shear_obj.xt[0, :])
    else:
        shear_obj.max_supp = np.max(shear_obj.xt[-1, :])

    # Minimum Shear Force
    min_cols = np.min(shear_obj.xt, axis=0)
    aux1 = np.argmin(shear_obj.xt, axis=0)
    
    aux2 = np.argmin(min_cols)
    shear_obj.min = min_cols[aux2]
    
    shear_obj.min_node = aux1[aux2]
    shear_obj.min_COP = Beam.Mesh.Nodes.acum[shear_obj.min_node]
    shear_obj.min_pCOP = (shear_obj.min_COP / Beam.Prop.L) * 100.0
    shear_obj.min_t_crit = Calc.Solver.t[aux2]
    
    if shear_obj.min_pCOP < 50.0:
        shear_obj.min_supp = np.min(shear_obj.xt[0, :])
    else:
        shear_obj.min_supp = np.min(shear_obj.xt[-1, :])

    return Sol

# ---- End of function ----