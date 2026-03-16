import numpy as np

# Dummy class for structure initialization
class EmptyObj:
    pass

def b31_beam_bm(Sol, Model, Beam, Calc, calc_type):
    """
    Calculates the Bending Moment (BM) of the beam using the nodal displacements
    """

    # Input processing
    if calc_type == 0:
        out_field = 'StaticBM'
        in_field = 'StaticU'
    elif calc_type == 1:
        out_field = 'BM'
        in_field = 'U'

    # Ensure the target nested structure exists
    if not hasattr(Sol.Beam, out_field):
        setattr(Sol.Beam, out_field, EmptyObj())
        
    bm_obj = getattr(Sol.Beam, out_field)
    
    # Extract the full nodal displacements from Sol.Model.Nodal
    if isinstance(getattr(Sol.Model.Nodal, in_field), dict):
        # Fallback if someone implemented Sol as a dict
        u_matrix = getattr(Sol.Model.Nodal, in_field) 
    else:
        u_matrix = getattr(Sol.Model.Nodal, in_field)

    # Initialize variables
    bm_obj.xt = np.zeros((Beam.Mesh.Nodes.Tnum, Calc.Solver.num_t))

    # In-line functions (more efficient alternative to subfunctions)
    B30_Beam_ele_H = lambda L, E, I: (E * I) * np.array([
        [-6 / L**2, -4 / L, 6 / L**2, -2 / L],
        [6 / L**2,   2 / L, -6 / L**2,  4 / L]
    ], dtype=float)

    # ---- NO average nodal values ----
    if Calc.Options.BM_calc_mode == 0:
        
        for ele in range(Beam.Mesh.Ele.Tnum):
            aux1 = B30_Beam_ele_H(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
            
            # Grab the 4 DOFs for this element
            dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
            
            # Assign BM for the left node of the element using dot product (@)
            bm_obj.xt[ele, :] = aux1[0, :] @ u_matrix[dofs, :]

        # Explicitly handle the very last node on the right side
        ele = Beam.Mesh.Ele.Tnum - 1
        last_node = Beam.Mesh.Nodes.Tnum - 1
        
        aux1 = B30_Beam_ele_H(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
        dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
        bm_obj.xt[last_node, :] = aux1[1, :] @ u_matrix[dofs, :]
        
    # ---- AVERAGE nodal values ----
    elif Calc.Options.BM_calc_mode == 1:
        
        for ele in range(Beam.Mesh.Ele.Tnum):
            aux1 = B30_Beam_ele_H(Beam.Mesh.Ele.a[ele], Beam.Prop.E_n[ele], Beam.Prop.I_n[ele])
            dofs = Model.Mesh.DOF.beam[Beam.Mesh.Ele.DOF[ele, :]]
            
            # Add contributions to both left (ele) and right (ele+1) nodes natively
            bm_obj.xt[ele : ele + 2, :] += aux1 @ u_matrix[dofs, :]

        # Average of interior nodes (those with multiple calculations)
        # Slicing [1:-1] drops the first and last node, acting on everything in between
        bm_obj.xt[1:-1, :] = bm_obj.xt[1:-1, :] / 2.0

    # ---- Additional Outputs ----

    # Maximum Bending Moment
    max_cols = np.max(bm_obj.xt, axis=0)
    aux1 = np.argmax(bm_obj.xt, axis=0)  # Row index of the max in each column
    
    aux2 = np.argmax(max_cols)           # Column index of the global max
    bm_obj.max = max_cols[aux2]
    
    bm_obj.COP = Beam.Mesh.Nodes.acum[aux1[aux2]]
    bm_obj.pCOP = (bm_obj.COP / Beam.Prop.L) * 100.0
    bm_obj.t_crit = Calc.Solver.t[aux2]

    # Mid-span Bending Moment
    if getattr(Beam.Mesh.Nodes.Mid, 'exists', 0) == 1:
        node_idx = int(Beam.Mesh.Nodes.Mid.node)
        bm_obj.max05 = np.max(bm_obj.xt[node_idx, :])
    else:
        row_maxs = np.max(bm_obj.xt, axis=1)
        bm_obj.max05 = np.interp(Beam.Prop.L / 2.0, Beam.Mesh.Nodes.acum, row_maxs)

    # Additional calculation for BM minima over specific halves
    Tnum = Beam.Mesh.Nodes.Tnum
    for k in [1, 2]:
        if k == 1:
            field = 'LeftSup'
            ind_add = 0
            ind_end = int(np.floor(Tnum / 2))
        elif k == 2:
            field = 'RightSup'
            ind_add = int(np.ceil(Tnum / 2)) - 1
            ind_end = Tnum

        # Initialize the sub-field object
        setattr(bm_obj, field, EmptyObj())
        sub_obj = getattr(bm_obj, field)

        # Slice the segment of interest
        sub_xt = bm_obj.xt[ind_add:ind_end, :]

        # Find minima inside this segment
        min_cols_sub = np.min(sub_xt, axis=0)
        aux1_sub = np.argmin(sub_xt, axis=0)
        aux2_sub = np.argmin(min_cols_sub)

        sub_obj.min = min_cols_sub[aux2_sub]
        
        # Map the local relative index (aux1_sub) back to the global index using ind_add
        global_node_idx = aux1_sub[aux2_sub] + ind_add
        sub_obj.COP = Beam.Mesh.Nodes.acum[global_node_idx]
        sub_obj.pCOP = (sub_obj.COP / Beam.Prop.L) * 100.0
        
        # Note: MATLAB takes min(xt(1,:)), which corresponds to index 0 in Python
        sub_obj.min05 = np.min(bm_obj.xt[0, :])

    return Sol

# ---- End of function ----