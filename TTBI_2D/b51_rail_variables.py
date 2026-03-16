import numpy as np

# A lightweight dummy class to serve as a structure for missing parent fields
class EmptyObj:
    pass

def b51_rail_variables(Track, Calc):
    """
    Definition of variables related to the Track
    """

    # Optional safety check: Ensure nested structures exist before assigning
    if not hasattr(Track, 'Rail'): Track.Rail = EmptyObj()
    if not hasattr(Track.Rail, 'Prop'): Track.Rail.Prop = EmptyObj()
    if not hasattr(Track.Rail, 'Mesh'): Track.Rail.Mesh = EmptyObj()
    if not hasattr(Track.Rail.Mesh, 'Ele'): Track.Rail.Mesh.Ele = EmptyObj()
    if not hasattr(Track.Rail, 'Options'): Track.Rail.Options = EmptyObj()
    if not hasattr(Track.Rail, 'BC'): Track.Rail.BC = EmptyObj()

    Track.Rail.Prop.L = Calc.Profile.L
    Track.Rail.Prop.A = 1
    
    # Calculate number of elements and explicitly cast to integer for future mesh indexing
    Track.Rail.Mesh.Ele.num = int(round(
        (Track.Rail.Prop.L / Track.Sleeper.spacing) * Track.Rail.Mesh.Ele.num_per_spacing
    ))
    
    Track.Rail.Options.k_Mconsist = 1

    if Calc.Options.redux == 0:
        # Boundary conditions - None
        # Using empty lists, which map perfectly to empty arrays in subsequent NumPy logic
        Track.Rail.BC.loc = []
        Track.Rail.BC.vert_stiff = []
        Track.Rail.BC.rot_stiff = []
        
    elif Calc.Options.redux == 1:
        # Boundary conditions - Simply supported
        Track.Rail.BC.loc = [0, Track.Rail.Prop.L]
        Track.Rail.BC.vert_stiff = [-1, -1]
        Track.Rail.BC.rot_stiff = [0, 0]

    return Track

# ---- End of script ----