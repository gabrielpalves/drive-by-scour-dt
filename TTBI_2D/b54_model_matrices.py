import numpy as np
import math
from scipy.sparse import lil_matrix, diags

# Dummy class for structure initialization
class EmptyObj:
    pass

def b54_model_matrices(Beam, Track, Calc):
    """
    Assembles the coupled model (Track+Beam) system matrices
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
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.rail_vert_at_sleepers, funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.rail_vert_at_sleepers, funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.k))

    # Pads to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.k))

    # Sleepers
    Model.Mesh.Mg = funAdd1(Model.Mesh.Mg, Model.Mesh.DOF.sleepers, funDiag(Track.Sleeper.Tnum, Track.Sleeper.Prop.m))

    # Ballast on approach to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_app, funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_app, funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.k))

    # Ballast on bridge to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_onbeam, funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_onbeam, funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.k))

    # Ballast after bridge to sleepers DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_aft, funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_aft, funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.k))

    # Ballast on approach to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_app, funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.k))

    # Ballast on bridge to Bridge DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.beam_vert_under_sleeper, funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.beam_vert_under_sleeper, funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.k))

    # Ballast after bridge to Ballast DOF
    Model.Mesh.Cg = funAdd1(Model.Mesh.Cg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd1(Model.Mesh.Kg, Model.Mesh.DOF.ballast_aft, funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.k))

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
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.rail_vert_at_sleepers, Model.Mesh.DOF.sleepers, -funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.rail_vert_at_sleepers, Model.Mesh.DOF.sleepers, -funDiag(Track.Sleeper.Tnum, Track.Pad.Prop.k))

    # Sleepers and Ballast on approach
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_app, Model.Mesh.DOF.ballast_app, -funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_app, Model.Mesh.DOF.ballast_app, -funDiag(Track.Sleeper.num_app, Track.Ballast.Prop.k))

    # Sleepers and Beam
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_onbeam, Model.Mesh.DOF.beam_vert_under_sleeper, -funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_onbeam, Model.Mesh.DOF.beam_vert_under_sleeper, -funDiag(Track.Sleeper.num_onbeam, Track.BallastOnBeam.Prop.k))

    # Sleepers and Ballast after bridge
    Model.Mesh.Cg = funAdd2(Model.Mesh.Cg, Model.Mesh.DOF.sleepers_aft, Model.Mesh.DOF.ballast_aft, -funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.c))
    Model.Mesh.Kg = funAdd2(Model.Mesh.Kg, Model.Mesh.DOF.sleepers_aft, Model.Mesh.DOF.ballast_aft, -funDiag(Track.Sleeper.num_aft, Track.Ballast.Prop.k))

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