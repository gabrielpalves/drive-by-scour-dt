import numpy as np

def b02_boundary_conditions(Beam, Damage):
    """
    Definition of DOF with boundary conditions for different configurations.
    """

    # Convert inputs to NumPy arrays for logical indexing and operations
    loc = np.atleast_1d(Beam.BC.loc)
    vert_stiff = np.atleast_1d(Beam.BC.vert_stiff)
    rot_stiff = np.atleast_1d(Beam.BC.rot_stiff)
    acum = np.array(Beam.Mesh.Nodes.acum)

    # Number of supports
    Beam.BC.supp_num = len(loc) 

    # Supports location index
    if Beam.BC.supp_num > 0:
        # Broadcasting: loc[:, np.newaxis] is (N, 1), acum[np.newaxis, :] is (1, M)
        # This creates an NxM matrix of differences cleanly without explicitly multiplying by ones()
        diffs = np.abs(acum[np.newaxis, :] - loc[:, np.newaxis])
        # argmin across axis 1 (rows) gives the 0-based node index closest to the support
        Beam.BC.loc_ind = np.argmin(diffs, axis=1)
    else:
        Beam.BC.loc_ind = np.array([], dtype=int)

    # Fixed vertical displacement DOF (0-based indexing: node * 2)
    fixed_vert = Beam.BC.loc_ind[vert_stiff == -1] * 2

    # Fixed rotational DOF (0-based indexing: node * 2 + 1)
    fixed_rot = Beam.BC.loc_ind[rot_stiff == -1] * 2 + 1

    # Sorting fixed DOF
    Beam.BC.DOF_fixed = np.sort(np.concatenate((fixed_vert, fixed_rot))).astype(int)

    # -- Vertical displacement DOF with stiffness values --
    vert_with_values = Beam.BC.loc_ind[vert_stiff > 0] * 2

    # I included
    DOF_Original_value = 344e6

    # Array with the same length as the number of supports with different stiffness
    # Note: Hardcoded for 3 supports as per the original MATLAB script
    Beam.BC.DOF_stiff_values = np.array([
        DOF_Original_value,                                # Vertical Support 1
        Damage.DOF_ChangeRate_value * DOF_Original_value,  # Vertical Support 2
        DOF_Original_value                                 # Vertical Support 3
    ])

    # -- Fixed rotational DOF with stiffness values --
    rot_with_values = Beam.BC.loc_ind[rot_stiff > 0] * 2 + 1

    # Combine vertical and rotational DOFs with values
    Beam.BC.DOF_with_values = np.concatenate((vert_with_values, rot_with_values)).astype(int)

    # Sorting DOFs with values and aligning the stiffness array to match
    aux2 = np.argsort(Beam.BC.DOF_with_values)
    Beam.BC.DOF_with_values = Beam.BC.DOF_with_values[aux2]
    
    # Ensure DOF_stiff_values is only sorted if its length matches (preventing broadcast errors 
    # if there are rot_stiff values added but not accounted for in the hardcoded array above)
    if len(Beam.BC.DOF_stiff_values) == len(aux2):
        Beam.BC.DOF_stiff_values = Beam.BC.DOF_stiff_values[aux2]

    # Auxiliary variables
    Beam.BC.num_DOF_fixed = len(Beam.BC.DOF_fixed)
    Beam.BC.num_DOF_with_values = len(Beam.BC.DOF_with_values)
    
    # Initialize Modal if it doesn't exist
    if not hasattr(Beam, 'Modal'):
        class EmptyObj: pass
        Beam.Modal = EmptyObj()
        
    Beam.Modal.num_rigid_modes = max([0, 2 - Beam.BC.num_DOF_fixed])

    # Value to use in the diagonal element when the DOF is fixed
    # Usually the value 1 was used, but this gave some ill-conditioning in some cases
    Beam.BC.DOF_fixed_value = DOF_Original_value

    return Beam, Damage