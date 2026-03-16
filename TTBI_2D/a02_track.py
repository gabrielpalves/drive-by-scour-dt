class EmptyObj:
    pass

def a02_track():
    """
    Definition of Track Properties (Zhai et al. with Ballast on Bridge)
    """
    Track = EmptyObj()
    
    # ---- Rail ----
    Track.Rail = EmptyObj()
    Track.Rail.Prop = EmptyObj()
    Track.Rail.Prop.E = 2.059e11
    Track.Rail.Prop.I = (3.217e-5) * 2
    Track.Rail.Prop.rho = 60.64 * 2
    Track.Rail.Damping = EmptyObj()
    Track.Rail.Damping.per = 0.1
    Track.Rail.Options = EmptyObj()
    
    # ---- Pad ----
    Track.Pad = EmptyObj()
    Track.Pad.Prop = EmptyObj()
    Track.Pad.Prop.k = 6.5e7
    Track.Pad.Prop.c = 7.5e4
    
    # ---- Sleeper ----
    Track.Sleeper = EmptyObj()
    Track.Sleeper.spacing = 0.6
    Track.Sleeper.Prop = EmptyObj()
    Track.Sleeper.Prop.m = 125.5 * 2
    
    # ---- Ballast ----
    Track.Ballast = EmptyObj()
    Track.Ballast.Prop = EmptyObj()
    Track.Ballast.Prop.m = 531.4
    Track.Ballast.Prop.k = 137.75e6
    Track.Ballast.Prop.c = 5.88e4
    
    # ---- SubBallast ----
    Track.SubBallast = EmptyObj()
    Track.SubBallast.Prop = EmptyObj()
    Track.SubBallast.Prop.k = 77.5e6
    Track.SubBallast.Prop.c = 3.115e4
    
    # ---- Ballast on Bridge ----
    Track.BallastOnBeam = EmptyObj()
    Track.BallastOnBeam.Prop = EmptyObj()
    Track.BallastOnBeam.Prop.m = Track.Ballast.Prop.m
    Track.BallastOnBeam.Prop.k = Track.Ballast.Prop.k
    Track.BallastOnBeam.Prop.c = Track.Ballast.Prop.c
    
    return Track
