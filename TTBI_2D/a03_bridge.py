class EmptyObj:
    pass

def a03_bridge(Beam):
    """
    Bridge model definition.

    Geometry is toggleable: set ``Beam.Prop.L`` and ``Beam.Prop.num_spans``
    before calling (e.g. via damage_config.configure_bridge) to change the span
    count / length. Defaults reproduce the legacy 40 m, 2-span (3-support)
    bridge. Material properties are fixed.
    """
    if Beam is None:
        Beam = EmptyObj()

    if not hasattr(Beam, 'Prop'): Beam.Prop = EmptyObj()
    if not hasattr(Beam, 'Damping'): Beam.Damping = EmptyObj()
    if not hasattr(Beam, 'BC'): Beam.BC = EmptyObj()

    # Geometry — respect values pre-set by the caller, else fall back to legacy.
    if not hasattr(Beam.Prop, 'L'):         Beam.Prop.L = 40.0
    if not hasattr(Beam.Prop, 'num_spans'): Beam.Prop.num_spans = 2

    Beam.Prop.E = 35e9
    Beam.Prop.I = 0.33
    Beam.Damping.per = 3.0
    Beam.Prop.rho = 9.6

    # Boundary conditions (expanded from num_spans in b07_options_processing)
    Beam.BC.text = 'UD'

    return Beam