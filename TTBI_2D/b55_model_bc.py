import numpy as np
from scipy.sparse import issparse

# A lightweight dummy class to serve as a structure for missing parent fields
class EmptyObj:
    pass

def b55_model_bc(Model, Beam, Track):
    """
    Applies the boundary conditions to the coupled model.
    """

    # Optional safety check: Ensure Model.BC exists
    if not hasattr(Model, 'BC'):
        Model.BC = EmptyObj()

    # From Beam DOF to Model DOF
    # Using previous 0-based fixed DOFs to index into the global DOF map
    beam_fixed = Model.Mesh.DOF.beam[Beam.BC.DOF_fixed]

    # From Rail DOF to Model DOF
    rail_fixed = Model.Mesh.DOF.rail[Track.Rail.BC.DOF_fixed]

    # Combine and sort array
    Model.BC.DOF_fixed = np.sort(np.concatenate((beam_fixed, rail_fixed))).astype(int)

    # Number of fixed DOF
    Model.BC.num_DOF_fixed = len(Model.BC.DOF_fixed)

    # Value to use in the diagonal element when the DOF is fixed
    Model.BC.DOF_fixed_value = Beam.BC.DOF_fixed_value

    # --- Matrix Modifications ---
    fixed_dofs = Model.BC.DOF_fixed

    # If matrices are sparse (e.g., CSR), convert to LIL format for fast row/col zeroing
    matrices_were_sparse = False
    if issparse(Model.Mesh.Mg):
        matrices_were_sparse = True
        Model.Mesh.Mg = Model.Mesh.Mg.tolil()
        Model.Mesh.Cg = Model.Mesh.Cg.tolil()
        Model.Mesh.Kg = Model.Mesh.Kg.tolil()

    # Zero out rows for Fixed DOF
    Model.Mesh.Mg[fixed_dofs, :] = 0
    Model.Mesh.Cg[fixed_dofs, :] = 0
    Model.Mesh.Kg[fixed_dofs, :] = 0

    # Zero out columns for Fixed DOF
    Model.Mesh.Mg[:, fixed_dofs] = 0
    Model.Mesh.Cg[:, fixed_dofs] = 0
    Model.Mesh.Kg[:, fixed_dofs] = 0

    # Set the diagonal elements to the fixed stiff value
    for dof in fixed_dofs:
        Model.Mesh.Mg[dof, dof] = Model.BC.DOF_fixed_value
        Model.Mesh.Kg[dof, dof] = Model.BC.DOF_fixed_value
        # Note: Original MATLAB script doesn't assign this to Cg, preserving that logic

    # Convert back to CSR format for fast mathematical operations during the dynamic solver phase
    if matrices_were_sparse:
        Model.Mesh.Mg = Model.Mesh.Mg.tocsr()
        Model.Mesh.Cg = Model.Mesh.Cg.tocsr()
        Model.Mesh.Kg = Model.Mesh.Kg.tocsr()

    return Model

# ---- End of script ----