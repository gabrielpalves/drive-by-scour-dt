import numpy as np


def _per_support(Damage, name, n):
    """Return (length-n float array, was_provided) for a per-support damage spec.

    Pads with zeros / truncates to n. Returns (zeros, False) when the field is
    absent so callers can fall back to legacy behaviour.
    """
    val = getattr(Damage, name, None)
    out = np.zeros(n, dtype=float)
    if val is None:
        return out, False
    v = np.atleast_1d(np.asarray(val, dtype=float))
    m = min(n, len(v))
    out[:m] = v[:m]
    return out, True


def b02_boundary_conditions(Beam, Damage):
    """
    Definition of DOF with boundary conditions — generalised to any number of
    supports, with per-support scour (vertical) and bearing (rotational) damage.

    Damage interface (all optional, default = healthy):
      Damage.scour_rates       per-support fraction in [0,1]; retained vertical
                               stiffness = (1 - rate) * DOF_Original_value.
      Damage.bearing_rot_stiff per-support rotational stiffness [Nm/rad]; only
                               used where Beam.BC.rot_stiff > 0 (abutments).
                               0 = free rotation (healthy bearing).
    Legacy fallback: if scour_rates is absent, Damage.DOF_ChangeRate_value is
    applied to the middle support only (reproduces the original 3-support model).

    Note: this runs for the bridge *and* the rail. With redux=0 the rail has no
    supports (loc=[]), so the per-support loops below simply do nothing for it.
    """

    # Convert inputs to NumPy arrays for logical indexing and operations
    loc = np.atleast_1d(Beam.BC.loc)
    vert_stiff = np.atleast_1d(Beam.BC.vert_stiff)
    rot_stiff = np.atleast_1d(Beam.BC.rot_stiff)
    acum = np.array(Beam.Mesh.Nodes.acum)

    # Number of supports
    Beam.BC.supp_num = len(loc)
    n = Beam.BC.supp_num

    # Supports location index (nearest mesh node to each support)
    if n > 0:
        diffs = np.abs(acum[np.newaxis, :] - loc[:, np.newaxis])
        Beam.BC.loc_ind = np.argmin(diffs, axis=1)
    else:
        Beam.BC.loc_ind = np.array([], dtype=int)

    # Fixed DOFs (perfectly stiff supports, flag == -1)
    fixed_vert = Beam.BC.loc_ind[vert_stiff == -1] * 2
    fixed_rot = Beam.BC.loc_ind[rot_stiff == -1] * 2 + 1
    Beam.BC.DOF_fixed = np.sort(np.concatenate((fixed_vert, fixed_rot))).astype(int)

    DOF_Original_value = 344e6

    # ---- Per-support damage specifications ----
    scour_rates, scour_given = _per_support(Damage, 'scour_rates', n)
    if not scour_given:
        # Legacy single-foundation behaviour: scour on the middle support only.
        legacy = float(getattr(Damage, 'DOF_ChangeRate_value', 0.0))
        if n > 0:
            scour_rates[n // 2] = legacy
    bearing_rot, _ = _per_support(Damage, 'bearing_rot_stiff', n)

    # ---- Vertical (scour) support springs ----
    vert_with_values, vert_values = [], []
    for i in range(n):
        if vert_stiff[i] > 0:
            vert_with_values.append(Beam.BC.loc_ind[i] * 2)
            vert_values.append((1.0 - scour_rates[i]) * DOF_Original_value)

    # ---- Rotational (bearing) support springs ----
    # value 0 (healthy/free) adds nothing to the diagonal, so the healthy model
    # is identical whether or not the abutments are marked bearing-capable.
    rot_with_values, rot_values = [], []
    for i in range(n):
        if rot_stiff[i] > 0:
            rot_with_values.append(Beam.BC.loc_ind[i] * 2 + 1)
            rot_values.append(bearing_rot[i])

    # Combine vertical and rotational DOFs with values, and align the values
    Beam.BC.DOF_with_values = np.array(vert_with_values + rot_with_values, dtype=int)
    stiff_values = np.array(vert_values + rot_values, dtype=float)

    if Beam.BC.DOF_with_values.size:
        aux2 = np.argsort(Beam.BC.DOF_with_values)
        Beam.BC.DOF_with_values = Beam.BC.DOF_with_values[aux2]
        Beam.BC.DOF_stiff_values = stiff_values[aux2]
    else:
        Beam.BC.DOF_stiff_values = stiff_values

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