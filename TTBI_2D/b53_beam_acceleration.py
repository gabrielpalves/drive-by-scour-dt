import numpy as np

# Dummy class for structure initialization
class EmptyObj:
    pass

def b53_beam_acceleration(Sol, Model, Beam, Calc):
    """
    Extracts the Beam acceleration from the nodal displacements.
    """

    # Optional safety checks to initialize structures if they don't exist
    if not hasattr(Sol, 'Beam'):
        Sol.Beam = EmptyObj()
    if not hasattr(Sol.Beam, 'Acc'):
        Sol.Beam.Acc = EmptyObj()

    # Displacements to Accelerations extraction
    # Slicing the global acceleration matrix to grab only vertical beam DOFs
    Sol.Beam.Acc.xt = Sol.Model.Nodal.A[Model.Mesh.DOF.beam_vert, :]

    # Note: The absolute value of the acceleration is analyzed
    abs_acc = np.abs(Sol.Beam.Acc.xt)

    # ---- Additional Outputs ----
    # Maximum Acceleration
    # np.max/np.argmax with axis=0 operates column-by-column (across time for each node)
    max_cols = np.max(abs_acc, axis=0)
    aux1 = np.argmax(abs_acc, axis=0)  # Row index of the max absolute value in each column
    
    # Global max across the entire time history
    aux2 = np.argmax(max_cols)         # Column index of the global max
    Sol.Beam.Acc.max = max_cols[aux2]
    
    # Critical Observation Point mapping
    Sol.Beam.Acc.COP = Beam.Mesh.Nodes.acum[aux1[aux2]]
    Sol.Beam.Acc.pCOP = (Sol.Beam.Acc.COP / Beam.Prop.L) * 100.0
    Sol.Beam.Acc.t_crit = Calc.Solver.t[aux2]

    # Mid-span Acceleration
    if getattr(Beam.Mesh.Nodes.Mid, 'exists', 0) == 1:
        # Extract max absolute acceleration exactly at the mid-span node
        node_idx = int(Beam.Mesh.Nodes.Mid.node)
        Sol.Beam.Acc.max05 = np.max(np.abs(Sol.Beam.Acc.xt[node_idx, :]))
    else:
        # Interpolate the maximums if a perfect mid-span node doesn't exist
        # axis=1 extracts the maximum absolute acceleration over time for *every* node
        row_maxs = np.max(abs_acc, axis=1)
        Sol.Beam.Acc.max05 = np.interp(Beam.Prop.L / 2.0, Beam.Mesh.Nodes.acum, row_maxs)

    return Sol

# ---- End of function ----