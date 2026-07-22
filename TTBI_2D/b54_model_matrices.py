import numpy as np
import math
from scipy.sparse import lil_matrix, diags

# Dummy class for structure initialization
class EmptyObj:
    pass

def _track_vectors(Track, Model, Calc, Damage):
    """Per-sleeper track-layer property vectors from Damage.track descriptors.

    Exact mirror of scour_MATLAB/B54_ModelMatrices.m::local_track_vectors.
    COORDINATE FRAME (audit fix 2026-07-17): descriptors carrying an
    x_bridge_local field are in A00's BRIDGE-LOCAL window frame (deck starts
    at x = x_bridge_local); the sleeper axis here is GLOBAL and under redux=0
    the deck really starts at num_app*spacing = L_Approach + max_TL, so the
    axis is shifted into the descriptor frame before any selection. Legacy
    descriptors without x_bridge_local are read as global coordinates.
    Optional fields of Damage.track (attribute container or dict):
        ballast_patches : rows [x_start, x_end, eta_k, eta_c]
        hanging_groups  : rows [x_start, n_consec]  (ballast support -> ~0)
        pad_stiff_mult  : scalar chi_pad (global pad aging)
        pad_damp_mult   : scalar beta_pad
        pad_failures    : x-positions of failed pads (k -> ~0)
        x_bridge_local  : deck-start position in the descriptor frame [m]
    Healthy (Damage/track absent or empty) -> uniform vectors; assembled
    matrices are numerically identical to the legacy scalar path.
    """
    n = Track.Sleeper.Tnum
    x = np.asarray(Model.Mesh.XLoc.sleepers, dtype=float)
    mult_bal_k = np.ones(n); mult_bal_c = np.ones(n)
    mult_pad_k = np.ones(n); mult_pad_c = np.ones(n)
    KILL = 1e-6      # "removed" support: not exactly 0 to keep Kg well-posed

    T = getattr(Damage, 'track', None) if Damage is not None else None
    if isinstance(T, dict):           # allow dict descriptors too
        class _D: pass
        d = _D(); [setattr(d, k, v) for k, v in T.items()]; T = d
    if T is not None:
        xbl = getattr(T, 'x_bridge_local', None)
        if xbl is not None and np.size(xbl):
            x_deck0 = Track.Sleeper.num_app * Track.Sleeper.spacing
            x = x - (x_deck0 - float(np.ravel(xbl)[0]))
    if T is not None:
        patches = np.atleast_2d(np.asarray(getattr(T, 'ballast_patches', []) if
                                getattr(T, 'ballast_patches', None) is not None else [], dtype=float))
        if patches.size:
            for p in patches:
                sel = (x >= p[0]) & (x <= p[1])
                # Overlapping patches: largest |log eta_k| deviation governs
                # and supplies BOTH eta_k and eta_c — exact mirror of the
                # B54 fix 2026-07-22 (audit r3); the old stacked product
                # could leave the documented per-patch bands, and mixing
                # k/c across patches would be an unphysical hybrid state.
                upd = sel & (abs(np.log(p[2])) > np.abs(np.log(mult_bal_k)))
                mult_bal_k[upd] = p[2]
                mult_bal_c[upd] = p[3]
        groups = np.atleast_2d(np.asarray(getattr(T, 'hanging_groups', []) if
                               getattr(T, 'hanging_groups', None) is not None else [], dtype=float))
        if groups.size:
            for g in groups:
                cand = np.where(x >= g[0] - Calc.Cte.tol)[0]
                if cand.size == 0:
                    continue
                i0 = int(cand[0])
                idx = np.arange(i0, min(i0 + int(g[1]), n))
                mult_bal_k[idx] = KILL
                mult_bal_c[idx] = KILL
        chi = getattr(T, 'pad_stiff_mult', None)
        if chi is not None:
            mult_pad_k *= float(chi)
        beta = getattr(T, 'pad_damp_mult', None)
        if beta is not None:
            mult_pad_c *= float(beta)
        fails = np.atleast_1d(np.asarray(getattr(T, 'pad_failures', []) if
                              getattr(T, 'pad_failures', None) is not None else [], dtype=float))
        for fx in fails:
            i0 = int(np.argmin(np.abs(x - fx)))
            mult_pad_k[i0] = KILL
            mult_pad_c[i0] = KILL

    i_app = np.arange(Track.Sleeper.num_app)
    i_on  = Track.Sleeper.num_app + np.arange(Track.Sleeper.num_onbeam)
    i_aft = (Track.Sleeper.num_app + Track.Sleeper.num_onbeam +
             np.arange(Track.Sleeper.num_aft))

    class _V: pass
    V = _V()
    V.pad_k  = Track.Pad.Prop.k           * mult_pad_k          # full track
    V.pad_c  = Track.Pad.Prop.c           * mult_pad_c
    V.balA_k = Track.Ballast.Prop.k       * mult_bal_k[i_app]   # approach
    V.balA_c = Track.Ballast.Prop.c       * mult_bal_c[i_app]
    V.balB_k = Track.BallastOnBeam.Prop.k * mult_bal_k[i_on]    # on bridge
    V.balB_c = Track.BallastOnBeam.Prop.c * mult_bal_c[i_on]
    V.balF_k = Track.Ballast.Prop.k       * mult_bal_k[i_aft]   # after
    V.balF_c = Track.Ballast.Prop.c       * mult_bal_c[i_aft]
    return V


def b54_model_matrices(Beam, Track, Calc, Damage=None):
    """
    Assembles the coupled model (Track+Beam) system matrices.

    Optional Damage argument may carry per-passage TRACK-LAYER damage
    descriptors in Damage.track — see _track_vectors and
    docs/stage3_alldamage_spec.md. Mirrors scour_MATLAB/B54_ModelMatrices.m.
    """

    if not hasattr(Calc, 'Model'):
        Model = EmptyObj()
        Model.Mesh = EmptyObj()
        Model.Mesh.DOF = EmptyObj()
        Model.Mesh.XLoc = EmptyObj()
        Model.Mesh.Ele = EmptyObj()
    else:
        Model = Calc.Model # Assuming it might be passed inside Calc or exists globally

    # ---- Counting ----
    # Total
    Track.Sleeper.Tnum = int(round(Calc.Profile.L / Track.Sleeper.spacing)) + 1
    
    # Approach
    Track.Sleeper.num_app = int(round(
        (Calc.Profile.max_TL * Calc.Options.redux_factor + Calc.Profile.L_Approach) / Track.Sleeper.spacing
    ))
    
    if Calc.Profile.extra_L < Calc.Cte.tol:
        Track.Sleeper.num_onbeam = int(round(Beam.Prop.L / Track.Sleeper.spacing)) + 1
        Track.Sleeper.num_aft = int(round(
            (Calc.Profile.L - (Calc.Profile.max_TL * Calc.Options.redux_factor + 
             Calc.Profile.L_Approach + Beam.Prop.L + Calc.Profile.extra_L)) / Track.Sleeper.spacing
        ))
    else:
        Track.Sleeper.num_onbeam = math.floor(Beam.Prop.L / Track.Sleeper.spacing) + 1
        Track.Sleeper.num_aft = int(round(
            (Calc.Profile.L - (Calc.Profile.max_TL * Calc.Options.redux_factor + 
             Calc.Profile.L_Approach + Beam.Prop.L + Calc.Profile.extra_L)) / Track.Sleeper.spacing
        )) + 1

    # ---- DOF indices (Converted to 0-based Python indexing) ----
    Model.Mesh.DOF.rail = np.arange(Track.Rail.Mesh.DOF.Tnum)
    Model.Mesh.DOF.rail_vert = Model.Mesh.DOF.rail[0::2]
    
    # Nodes at sleepers converted to 0-based index mapped to vertical DOF
    Model.Mesh.DOF.rail_vert_at_sleepers = np.arange(
        0, Track.Rail.Mesh.Nodes.Tnum, Track.Rail.Mesh.Ele.num_per_spacing
    ) * 2
    
    Model.Mesh.DOF.sleepers = np.arange(Track.Sleeper.Tnum) + Track.Rail.Mesh.DOF.Tnum
    
    Model.Mesh.DOF.sleepers_app = Model.Mesh.DOF.sleepers[:Track.Sleeper.num_app]
    Model.Mesh.DOF.sleepers_onbeam = Model.Mesh.DOF.sleepers[Track.Sleeper.num_app : Track.Sleeper.num_app + Track.Sleeper.num_onbeam]
    Model.Mesh.DOF.sleepers_aft = Model.Mesh.DOF.sleepers[Track.Sleeper.num_app + Track.Sleeper.num_onbeam : Track.Sleeper.num_app + Track.Sleeper.num_onbeam + Track.Sleeper.num_aft]
    
    # Safely building continuous DOF blocks regardless of empty arrays
    last_dof = Model.Mesh.DOF.sleepers[-1]
    
    Model.Mesh.DOF.ballast_app = last_dof + 1 + np.arange(Track.Sleeper.num_app)
    if len(Model.Mesh.DOF.ballast_app) > 0: last_dof = Model.Mesh.DOF.ballast_app[-1]
    
    Model.Mesh.DOF.beam = last_dof + 1 + np.arange(Beam.Mesh.DOF.Tnum)
    last_dof = Model.Mesh.DOF.beam[-1]
    
    Model.Mesh.DOF.beam_vert = Model.Mesh.DOF.beam[0::2]
    Model.Mesh.DOF.beam_vert_under_sleeper = Model.Mesh.DOF.beam_vert[0::Beam.Mesh.Ele.num_per_spacing]
    
    Model.Mesh.DOF.ballast_aft = last_dof + 1 + np.arange(Track.Sleeper.num_aft)

    if Track.Sleeper.num_aft == 0:
        Model.Mesh.DOF.Tnum = int(Model.Mesh.DOF.beam[-1] + 1)
    else:
        Model.Mesh.DOF.Tnum = int(Model.Mesh.DOF.ballast_aft[-1] + 1)

    # ---- X location of some DOF ---- (Useful for plotting)
    Model.Mesh.XLoc.rail_vert = Track.Rail.Mesh.Nodes.acum
    Model.Mesh.XLoc.sleepers = np.arange(len(Model.Mesh.DOF.sleepers)) * Track.Sleeper.spacing
    Model.Mesh.XLoc.ballast_app = np.arange(len(Model.Mesh.DOF.ballast_app)) * Track.Sleeper.spacing
    Model.Mesh.XLoc.beam_vert = Beam.Mesh.Nodes.acum + Calc.Profile.L_Approach + Calc.Profile.max_TL * Calc.Options.redux_factor
    Model.Mesh.XLoc.ballast_aft = (np.arange(len(Model.Mesh.DOF.ballast_aft)) + 1) * Track.Sleeper.spacing + \
                                  Calc.Profile.L_Approach + Beam.Prop.L + \
                                  Calc.Profile.max_TL * Calc.Options.redux_factor + Calc.Profile.extra_L

    # Temporary variable name change
    if getattr(Track.PadUnderSleeperOnBeam, 'included', 0) == 1:
        Track.BallastOnBeam.Prop.m = 0
        Track.BallastOnBeam.Prop.c = Track.PadUnderSleeperOnBeam.Prop.c
        Track.BallastOnBeam.Prop.k = Track.PadUnderSleeperOnBeam.Prop.k

    # ---- Per-sleeper track-layer property vectors (damage-aware EOVs) ----
    # Mirrors the MATLAB B54 local_track_vectors call (Stage 3 track damage).
    TrkV = _track_vectors(Track, Model, Calc, Damage)

    # ---------------------- Building Global matrices ------------------------- 
    
    # Initialize matrices as LIL format for efficient sparse block assignment
    Tnum = Model.Mesh.DOF.Tnum
    Model.Mesh.Mg = lil_matrix((Tnum, Tnum), dtype=float)
    Model.Mesh.Cg = lil_matrix((Tnum, Tnum), dtype=float)
    Model.Mesh.Kg = lil_matrix((Tnum, Tnum), dtype=float)

    # -------------------------- Subfunctions ---------------------------------
    def funDiag(size, value):
        # Generates a sparse diagonal matrix to save memory
        return diags([value], [0], shape=(size, size), format='lil')

    def funAdd1(InM, ind1, AddM):
        if len(ind1) == 0: return InM
        idx = np.ix_(ind1, ind1)
        InM[idx] += AddM
        return InM

    def funAdd2(InM, ind1, ind2, AddM):
        if len(ind1) == 0 or len(ind2) == 0: return InM
        idx1 = np.ix_(ind1, ind2)
        idx2 = np.ix_(ind2, ind1)
        InM[idx1] += AddM
        InM[idx2] += AddM
        return InM

    # ---- Diagonal Elements ----

    # Track
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.rail, Track.Rail.Mesh.Mg)
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.rail, Track.Rail.Mesh.Cg)
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.rail, Track.Rail.Mesh.Kg)

    # Pads to rail DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.rail_vert_at_sleepers, funDiag(Track.Sleeper.Tnum, TrkV.pad_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.rail_vert_at_sleepers, funDiag(Track.Sleeper.Tnum, TrkV.pad_k))

    # Pads to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, TrkV.pad_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, TrkV.pad_k))

    # Sleepers
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, Track.Sleeper.Prop.m))

    # Ballast on approach to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_app, funDiag(Track.Sleeper.num_app, TrkV.balA_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_app, funDiag(Track.Sleeper.num_app, TrkV.balA_k))

    # Ballast on bridge to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_onbeam, funDiag(Track.Sleeper.num_onbeam, TrkV.balB_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_onbeam, funDiag(Track.Sleeper.num_onbeam, TrkV.balB_k))

    # Ballast after bridge to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_aft, funDiag(Track.Sleeper.num_aft, TrkV.balF_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_aft, funDiag(Track.Sleeper.num_aft, TrkV.balF_k))

    # Ballast on approach to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, TrkV.balA_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, TrkV.balA_k))

    # Ballast on bridge to Bridge DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.beam_vert_under_sleeper, funDiag(Track.Sleeper.num_onbeam, TrkV.balB_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.beam_vert_under_sleeper, funDiag(Track.Sleeper.num_onbeam, TrkV.balB_k))

    # Ballast after bridge to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, TrkV.balF_c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, TrkV.balF_k))

    # Ballast on approach
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.m))

    # Ballast on bridge (Distributed to all Beam's vertical DOF)
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.beam_vert, funDiag(Beam.Mesh.Nodes.Tnum, Track.BallastOnBeam.Prop.m / Beam.Mesh.Ele.num_per_spacing))

    # Ballast after approach (Note: MATLAB script says 'after approach' but assigns to 'ballast_aft')
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.m))

    # Beam
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.beam, Beam.Mesh.Mg)
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.beam, Beam.Mesh.Cg)
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.beam, Beam.Mesh.Kg)

    # Sub-Ballast on approach to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, Track.SubBallast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, Track.SubBallast.Prop.k))

    # Sub-Ballast after bridge to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, Track.SubBallast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, Track.SubBallast.Prop.k))

    # ---- Off-Diagonal Elements ----

    # Rail and Sleepers
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.rail_vert_at_sleepers, Model.Mesh.DOF.sleepers, -funDiag(Track.Sleeper.Tnum, TrkV.pad_c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.rail_vert_at_sleepers, Model.Mesh.DOF.sleepers, -funDiag(Track.Sleeper.Tnum, TrkV.pad_k))

    # Sleepers and Ballast on approach
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_app, Model.Mesh.DOF.ballast_app, -funDiag(Track.Sleeper.num_app, TrkV.balA_c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_app, Model.Mesh.DOF.ballast_app, -funDiag(Track.Sleeper.num_app, TrkV.balA_k))

    # Sleepers and Beam
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_onbeam, Model.Mesh.DOF.beam_vert_under_sleeper, -funDiag(Track.Sleeper.num_onbeam, TrkV.balB_c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_onbeam, Model.Mesh.DOF.beam_vert_under_sleeper, -funDiag(Track.Sleeper.num_onbeam, TrkV.balB_k))

    # Sleepers and Ballast after bridge
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_aft, Model.Mesh.DOF.ballast_aft, -funDiag(Track.Sleeper.num_aft, TrkV.balF_c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_aft, Model.Mesh.DOF.ballast_aft, -funDiag(Track.Sleeper.num_aft, TrkV.balF_k))

    # Convert matrices to highly optimized CSR format
    Model.Mesh.Mg = Model.Mesh.Mg.tocsr()
    Model.Mesh.Cg = Model.Mesh.Cg.tocsr()
    Model.Mesh.Kg = Model.Mesh.Kg.tocsr()

    # Checking symmetry
    # Max absolute difference between the matrix and its transpose should be ~0
    checksum = max([
        abs(Model.Mesh.Mg - Model.Mesh.Mg.T).max(),
        abs(Model.Mesh.Cg - Model.Mesh.Cg.T).max(),
        abs(Model.Mesh.Kg - Model.Mesh.Kg.T).max()
    ])
    
    if checksum > Calc.Cte.tol:
        print('System matrices are not symmetric')
        raise ValueError('System matrices are not symmetric')

    # Auxiliary variables
    Model.Mesh.Ele.DOF = Track.Rail.Mesh.Ele.DOF
    Model.Mesh.Ele.a = Track.Rail.Mesh.Ele.a

    return Model

# ---- End of script ----