import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from b14_eq_vert_nodal_force import b14_eq_vert_nodal_force

# Dummy class for structure initialization
class EmptyObj:
    pass

def b49_beam_deformation(Sol, Model, Beam, Calc, Train, calc_type):
    """
    Calculates beam deformation minimum from the nodal displacements.
    """

    # Optional safety checks to initialize structures if they don't exist
    if not hasattr(Sol, 'Model'): Sol.Model = EmptyObj()
    if not hasattr(Sol.Model, 'Nodal'): Sol.Model.Nodal = EmptyObj()
    if not hasattr(Sol, 'Beam'): Sol.Beam = EmptyObj()

    if calc_type == 0:
        usefield = 'StaticU'

        # Vehicle loop
        for v in range(len(Train.Veh)):
            # Definition of static force in time
            # Using reshape(-1, 1) and np.tile to broadcast the static loads across all time steps
            sta_loads = np.array(Train.Veh[v].sta_loads).reshape(-1, 1)
            Calc.Veh[v].F_onBeam = np.tile(sta_loads, (1, Calc.Solver.num_t))

        # Nodal forces calculation
        Model.Mesh.shape_fun = Beam.Mesh.shape_fun
        
        # Assuming B14_EqVertNodalForce is translated and returns a matrix F
        F = b14_eq_vert_nodal_force(Model, Calc)
        
        # Ensure F is dense for the solver, mimicking MATLAB's full(F)
        F_dense = F.toarray() if issparse(F) else np.array(F)

        # Nodal displacements (Solving the static system Kg * U = F)
        if issparse(Model.Mesh.Kg):
            Kg_csr = Model.Mesh.Kg.tocsr()
            static_u = spsolve(Kg_csr, F_dense)
        else:
            static_u = np.linalg.solve(Model.Mesh.Kg, F_dense)
            
        # Dynamically assign Sol.Model.Nodal.StaticU
        setattr(Sol.Model.Nodal, usefield, static_u)

    elif calc_type == 1:
        usefield = 'U'

    # Ensure the target nested structure exists (e.g., Sol.Beam.StaticU or Sol.Beam.U)
    if not hasattr(Sol.Beam, usefield):
        setattr(Sol.Beam, usefield, EmptyObj())

    # Get references to the dynamic fields
    beam_field = getattr(Sol.Beam, usefield)
    model_nodal_field = getattr(Sol.Model.Nodal, usefield)

    # Displacements
    # Slice the global displacements to only extract the vertical beam DOFs
    beam_field.xt = model_nodal_field[Model.Mesh.DOF.beam_vert, :]

    # Note: Normal traffic vertical loading gives negative displacement. 
    # Thus the minimum displacement (most negative) is of interest.

    # ---- Additional Outputs ----
    # Minimum Displacements
    # np.min/np.argmin with axis=0 operates column-by-column (across time for each node)
    min_cols = np.min(beam_field.xt, axis=0)
    aux1 = np.argmin(beam_field.xt, axis=0) # Row index of the minimum in each column
    
    # Global minimum across the entire time history
    aux2 = np.argmin(min_cols) # Column index of the global minimum
    beam_field.min = min_cols[aux2]
    
    # Critical observation point
    beam_field.COP = Beam.Mesh.Nodes.acum[aux1[aux2]]
    beam_field.pCOP = (beam_field.COP / Beam.Prop.L) * 100.0
    beam_field.t_crit = Calc.Solver.t[aux2]

    # Mid-span Minimum Displacement
    if getattr(Beam.Mesh.Nodes.Mid, 'exists', 0) == 1:
        # Use the 0-based node index exactly as identified in B01_ElementsAndCoordinates
        node_idx = int(Beam.Mesh.Nodes.Mid.node)
        beam_field.min05 = np.min(beam_field.xt[node_idx, :])
    else:
        # Interpolate if a perfect mid-span node doesn't exist
        row_mins = np.min(beam_field.xt, axis=1)
        beam_field.min05 = np.interp(Beam.Prop.L / 2.0, Beam.Mesh.Nodes.acum, row_mins)

    return Sol

# ---- End of function ----