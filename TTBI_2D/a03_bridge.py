class EmptyObj:
    pass

def a03_bridge(Beam):
    """
    Bridge model definition. A 40m bridge (Adjusted from source 50m to script 40m).
    """
    if Beam is None:
        Beam = EmptyObj()
        
    if not hasattr(Beam, 'Prop'): Beam.Prop = EmptyObj()
    if not hasattr(Beam, 'Damping'): Beam.Damping = EmptyObj()
    if not hasattr(Beam, 'BC'): Beam.BC = EmptyObj()

    Beam.Prop.L = 40.0
    Beam.Prop.E = 35e9
    Beam.Prop.I = 0.33
    Beam.Damping.per = 3.0
    Beam.Prop.rho = 9.6
    
    # Boundary conditions
    Beam.BC.text = 'UD'
    
    return Beam