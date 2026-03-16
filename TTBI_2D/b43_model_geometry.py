import numpy as np
import math

def b43_model_geometry(Calc, Train, Track, Beam):
    """
    Auxiliary calculations related to the model geometry
    """

    # -- Track length --

    # Model redux option default value
    if not hasattr(Calc.Options, 'redux'):
        Calc.Options.redux = 1

    # Redux_factor (Is 1 if redux is Off. Gives 0 if redux is ON)
    Calc.Options.redux_factor = int(Calc.Options.redux == 0)

    # Approach distance (distance from first front wheel to bridge start)
    Calc.Profile.L_Approach = (
        round(Calc.Profile.minL_Approach / Track.Sleeper.spacing) * Track.Sleeper.spacing
    )

    # After crossing default value
    if not hasattr(Calc.Profile, 'minL_After'):
        Calc.Profile.minL_After = 10

    # After crossing distance
    Calc.Profile.L_After = (
        round(Calc.Profile.minL_After / Track.Sleeper.spacing) * Track.Sleeper.spacing
    )

    # -- Beam model --

    # Beam elements number and size
    Beam.Mesh.Ele.L = Track.Sleeper.spacing / Beam.Mesh.Ele.num_per_spacing
    Beam.Mesh.Ele.num = round(Beam.Prop.L / Beam.Mesh.Ele.L)

    # Modified beam length
    Beam.Prop.L = Beam.Mesh.Ele.L * Beam.Mesh.Ele.num
    Calc.Profile.L_bridge = Beam.Prop.L

    # Cross-section Area (Not needed, because mass is defined in terms of rho)
    Beam.Prop.A = 1

    # -- Vehicle variables --

    # Total number of vehicles (assuming Train.Veh is a list of objects)
    num_vehicles = len(Train.Veh)
    Train.Veh[0].Tnum = num_vehicles

    # Total train length and approach distances for each vehicle
    # Note: MATLAB's Train.Veh(1) maps to Python's Train.Veh[0], etc.
    Train.Veh[0].TL = Train.Veh[0].Body.L + Train.Veh[0].Bogie.L[0] / 2
    Train.Veh[0].L_app = Calc.Profile.L_Approach
    
    for v in range(1, num_vehicles): # Loops from 1 to num_vehicles-1
        # Total length of vehicle
        Train.Veh[0].TL += (Train.Veh[v-1].Body.Le[1] + 
                            Train.Veh[v].Body.Le[0] + 
                            Train.Veh[v].Body.L)
        
        # Approach distances
        Train.Veh[v].L_app = (Train.Veh[v-1].L_app + 
                              Train.Veh[v-1].Bogie.L[0] / 2 + 
                              Train.Veh[v-1].Body.L + 
                              Train.Veh[v-1].Body.Le[1] + 
                              Train.Veh[v].Body.Le[0] - 
                              Train.Veh[v].Bogie.L[0] / 2)
                              
    Train.Veh[0].TL += Train.Veh[-1].Bogie.L[1] / 2

    # Initialize Calc.Train.Veh if it doesn't exist yet to avoid attribute errors
    if not hasattr(Calc, 'Train'):
        class EmptyTrainObj: pass
        Calc.Train = EmptyTrainObj()
        # Create a list of dummy objects matching the number of vehicles
        Calc.Train.Veh = [type('VehObj', (object,), {})() for _ in range(num_vehicles)]

    # Geometric properties for each vehicle
    for v in range(num_vehicles):
        # Axle distances
        # Translating MATLAB array concatenation: [A, B, C + [-1, 1]*D]
        part1 = [0, Train.Veh[v].Bogie.L[0]]
        base_val = Train.Veh[v].Bogie.L[0] / 2 + Train.Veh[v].Body.L
        part2 = base_val + np.array([-1, 1]) * (Train.Veh[v].Bogie.L[1] / 2)
        
        Train.Veh[v].Ax_dist = np.concatenate((part1, part2))
        
        # Wheelbase
        Train.Veh[v].wheelbase = Train.Veh[v].Ax_dist[-1]
        
        # First wheel distance to Locomotive's first wheel
        Train.Veh[v].First_wheel_dist = Train.Veh[v].L_app - Train.Veh[0].L_app
        
        # Vehicle contact force flag
        Calc.Train.Veh[v].Positive_ContactForce = 0

    # Vehicle total length rounded up to closest number of sleeper spacings
    Train.Veh[0].max_TL = (
        math.ceil(Train.Veh[0].TL / Track.Sleeper.spacing) * Track.Sleeper.spacing
    )
    Calc.Profile.max_TL = Train.Veh[0].max_TL

    # Additional model length to have start and end the model with a sleeper.
    Calc.Profile.extra_L = Beam.Prop.L % Track.Sleeper.spacing
    if Calc.Profile.extra_L > 0:
        Calc.Profile.extra_L = Track.Sleeper.spacing - Calc.Profile.extra_L

    # Additional number of sleepers behind the vehicle at start and end
    Track.Rail.Options.num_add_sleepers = 10

    # Additional length not needed when working with the redux model
    if Calc.Options.redux == 1:
        Track.Rail.Options.num_add_sleepers = 0

    # Additional length to remove the influence of the track ends
    Calc.Profile.extra_L2 = Track.Rail.Options.num_add_sleepers * Track.Sleeper.spacing

    # Minimum total length of profile
    Calc.Profile.L = (2 * Calc.Profile.max_TL * Calc.Options.redux_factor + 
                      Train.Veh[0].L_app + 
                      Beam.Prop.L + 
                      Calc.Profile.extra_L + 
                      2 * Calc.Profile.extra_L2 * Calc.Options.redux_factor + 
                      Calc.Profile.L_After)
                      
    Calc.Profile.L = round(Calc.Profile.L / Track.Sleeper.spacing) * Track.Sleeper.spacing

    # Vehicle's initial position
    # Using a list to represent the 2-element array
    Calc.Position.x_start_end = [
        (Train.Veh[0].max_TL + Calc.Profile.extra_L2) * Calc.Options.redux_factor,
        Calc.Profile.L - Calc.Profile.extra_L2 * Calc.Options.redux_factor + 
        Train.Veh[0].max_TL * (1 - Calc.Options.redux_factor)
    ]

    # Length of approach + Train total length (Used for plotting)
    Calc.Profile.L_Aw = Calc.Profile.L_Approach + Calc.Profile.max_TL * Calc.Options.redux_factor

    return Calc, Train, Beam