class EmptyObj:
    pass

def a04_options(Beam, Track):
    """
    Definition of addition relevant aspects of the model and calculation options.
    """
    Calc = EmptyObj()
    Calc.Profile = EmptyObj()
    Calc.Options = EmptyObj()
    Calc.Solver = EmptyObj()
    Calc.Plot = EmptyObj()
    Calc.Position = EmptyObj()
    Calc.Time = EmptyObj()
    
    # ------------------------- Track irregularity ----------------------------
    Calc.Profile.Type = 2
    
    # --------------------------- Mesh Properties -----------------------------
    if not hasattr(Beam, 'Mesh'): Beam.Mesh = EmptyObj()
    if not hasattr(Beam.Mesh, 'Ele'): Beam.Mesh.Ele = EmptyObj()
    if not hasattr(Beam.Mesh, 'Nodes'): Beam.Mesh.Nodes = EmptyObj()
    if not hasattr(Beam.Mesh, 'DOF'): Beam.Mesh.DOF = EmptyObj()
    if not hasattr(Beam, 'Prop'): Beam.Prop = EmptyObj()
    Beam.Mesh.Ele.num_per_spacing = 2
    
    if not hasattr(Track, 'Mesh'): Track.Mesh = EmptyObj()
    if not hasattr(Track.Rail, 'Mesh'): Track.Rail.Mesh = EmptyObj()
    if not hasattr(Track.Rail.Mesh, 'Ele'): Track.Rail.Mesh.Ele = EmptyObj()
    if not hasattr(Track.Rail.Mesh, 'Nodes'): Track.Rail.Mesh.Nodes = EmptyObj()
    if not hasattr(Track.Rail.Mesh, 'DOF'): Track.Rail.Mesh.DOF = EmptyObj()
    if not hasattr(Track.Rail, 'Prop'): Track.Rail.Prop = EmptyObj()
    Track.Rail.Mesh.Ele.num_per_spacing = 2
    
    Calc.Profile.minL_Approach = 16.0
    Calc.Profile.minL_After = 10.0
    
    # ----------------- Calculation options and variables ---------------------
    Calc.Options.calc_beam_sections = [Beam.Prop.L / 2.0]
    Calc.Solver.max_accurate_frq = 500
    Calc.Options.VBI = 1
    Calc.Options.redux = 0
    Calc.Options.calc_beam_modes = 0
    
    # ------------------------- Plotting Options ------------------------------
    Calc.Plot.Model = EmptyObj()
    Calc.Plot.Model.P00_ModelVisualization = 1
    Calc.Plot.Model.P01_ModelDef = 50
    
    Calc.Plot.Beam = EmptyObj()
    Calc.Plot.Beam.P01_DispContour = 1
    Calc.Plot.Beam.P03_BMContour = 1
    Calc.Plot.Beam.P04_StaticBMContour = 1
    Calc.Plot.Beam.P09_Sections_BeamBM = 1
    
    return Calc, Beam, Track