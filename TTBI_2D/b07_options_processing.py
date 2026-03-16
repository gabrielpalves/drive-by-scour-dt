import numpy as np

# A lightweight dummy class to serve as a structure for missing parent fields
class EmptyObj:
    pass

def b07_options_processing(Calc, Train, Track, Beam):
    """
    Processing of input variables, and generating new or changing auxiliary 
    variables accordingly.
    """

    # Ensure Cte exists
    if not hasattr(Calc, 'Cte'):
        Calc.Cte = EmptyObj()

    # Constants
    Calc.Cte.tol = 1e-6             # Numerical tolerance
    Calc.Cte.grav = -9.81           # Gravity [m/s^2]

    # ------------------------- Type of calculation ---------------------------
    # Note: Only coupled system solution implemented in TTB-2D

    # Type of calculation text label
    if not hasattr(Calc, 'Type'):
        Calc.Type = EmptyObj()
        Calc.Type.short_text = 'COUP'

    # Other variables
    if Calc.Type.short_text == 'COUP':
        Calc.Type.id = 1
        Calc.Type.long_text = 'Coupled system calculation'

    # ------------------------------- Options ---------------------------------

    # Starting Options field
    if not hasattr(Calc, 'Options'):
        Calc.Options = EmptyObj()

    # Beam sections to calculate results
    if not hasattr(Calc.Options, 'calc_beam_sections') or Calc.Options.calc_beam_sections is None:
        Calc.Options.calc_beam_sections = []
        Calc.Options.num_calc_beam_sections = 0
    else:
        Calc.Options.num_calc_beam_sections = len(Calc.Options.calc_beam_sections)

    # Calculate model frequencies
    if not hasattr(Calc.Options, 'calc_model_frq'):
        Calc.Options.calc_model_frq = 0

    # Calculated model modes
    if not hasattr(Calc.Options, 'calc_model_modes'):
        Calc.Options.calc_model_modes = 0

    # VBI on/off option
    if not hasattr(Calc.Options, 'VBI'):
        Calc.Options.VBI = 1

    # Frequency of progress display (in seconds)
    if not hasattr(Calc.Options, 'disp_every_t'):  # Note: MATLAB variable named disp_every_t but assigned to disp_every
        Calc.Options.disp_every = 2

    # -------------------------------- Solver ---------------------------------

    # Starting Solver field
    if not hasattr(Calc, 'Solver'):
        Calc.Solver = EmptyObj()
        Calc.Solver.was_empty = 1

    # Newmark-Beta scheme
    if not hasattr(Calc.Solver, 'NewMark_damp'):
        Calc.Solver.NewMark_damp = 0 # Average acceleration method (Normal Newmark-beta)

    # Newmark scheme constants
    if Calc.Solver.NewMark_damp == 0:
        # Default = Average (or constant) acceleration method
        Calc.Solver.NewMark_delta = 0.5 
        Calc.Solver.NewMark_beta = 0.25
    elif Calc.Solver.NewMark_damp == 1:
        # Damped Newmark-Beta scheme
        Calc.Solver.NewMark_delta = 0.6 
        Calc.Solver.NewMark_beta = 0.3025

    # -------------------------------- Profile --------------------------------

    # Starting Profile field
    if not hasattr(Calc, 'Profile'):
        Calc.Profile = EmptyObj()

    # Default Smooth profile
    if not hasattr(Calc.Profile, 'Type'):
        Calc.Profile.Type = 0

    # Minimum Profile sampling period (m)
    if not hasattr(Calc.Profile, 'min_dx'):
        # Replicating MATLAB's min([abs(diff([Train.Veh(:).Ax_dist])), 0.01])
        # First, flatten and concatenate all Ax_dist arrays across all vehicles
        all_ax_dist = np.concatenate([veh.Ax_dist for veh in Train.Veh])
        # Compute differences, take absolute value, find min
        if len(all_ax_dist) > 1:
            min_diff = np.min(np.abs(np.diff(all_ax_dist)))
            Calc.Profile.min_dx = min(min_diff, 0.01)
        else:
            Calc.Profile.min_dx = 0.01

    # ------------------------------- Plotting --------------------------------

    # If no plots are selected
    if not hasattr(Calc, 'Plot'):
        Calc.Plot = EmptyObj()
        Calc.Plot.NoPlot = 1
    
    if not hasattr(Calc.Plot, 'Veh'):
        Calc.Plot.Veh = EmptyObj()
        Calc.Plot.Veh.NoPlot = 1
        
    if not hasattr(Calc.Plot, 'Model'):
        Calc.Plot.Model = EmptyObj()
        Calc.Plot.Model.NoPlot = 1
        
    if not hasattr(Calc.Plot, 'Beam'):
        Calc.Plot.Beam = EmptyObj()
        Calc.Plot.Beam.NoPlot = 1

    # Change Settings to generate P1 plot
    if hasattr(Calc.Plot, 'P1_Beam_frq'):
        if Calc.Plot.P1_Beam_frq == 1:
            Calc.Options.calc_beam_frq = 1
    else:
        Calc.Plot.P1_Beam_frq = 0

    if hasattr(Calc.Plot, 'P2_Beam_modes'):
        if Calc.Plot.P2_Beam_modes >= 1:
            Calc.Options.calc_beam_frq = 1
            Calc.Options.calc_beam_modes = 1
    else:
        Calc.Plot.P2_Beam_modes = 0

    if not hasattr(Calc.Plot, 'P3_VehPos'):
        Calc.Plot.P3_VehPos = 0

    if not hasattr(Calc.Plot, 'Profile_original'):
        Calc.Plot.Profile_original = 0

    if not hasattr(Calc.Plot, 'Model_modes'):
        Calc.Plot.Model_modes = []

    # ---- Vehicle ----
    if not hasattr(Calc.Plot.Veh, 'P01_VertDisp'): Calc.Plot.Veh.P01_VertDisp = 0
    if not hasattr(Calc.Plot.Veh, 'P02_VertVel'): Calc.Plot.Veh.P02_VertVel = 0
    if not hasattr(Calc.Plot.Veh, 'P03_VertAcc'): Calc.Plot.Veh.P03_VertAcc = 0
    if not hasattr(Calc.Plot.Veh, 'P04_ContactForce_t'): Calc.Plot.Veh.P04_ContactForce_t = 0
    if not hasattr(Calc.Plot.Veh, 'P05_ContactForce_x'): Calc.Plot.Veh.P05_ContactForce_x = 0

    # ---- Model ----
    if not hasattr(Calc.Plot.Model, 'P00_ModelVisualization'): Calc.Plot.Model.P00_ModelVisualization = 0
    if not hasattr(Calc.Plot.Model, 'P01_ModelDef'): Calc.Plot.Model.P01_ModelDef = -1
    if not hasattr(Calc.Plot.Model, 'P02_ModelRot'): Calc.Plot.Model.P02_ModelRot = -1

    # ---- Beam ----
    if not hasattr(Calc.Plot.Beam, 'P01_DispContour'): Calc.Plot.Beam.P01_DispContour = 0
    if not hasattr(Calc.Plot.Beam, 'P02_StaticDispContour'): Calc.Plot.Beam.P02_StaticDispContour = 0
    if not hasattr(Calc.Plot.Beam, 'P03_BMContour'): Calc.Plot.Beam.P03_BMContour = 0
    if not hasattr(Calc.Plot.Beam, 'P04_StaticBMContour'): Calc.Plot.Beam.P04_StaticBMContour = 0
    if not hasattr(Calc.Plot.Beam, 'P05_ShearContour'): Calc.Plot.Beam.P05_ShearContour = 0
    if not hasattr(Calc.Plot.Beam, 'P06_StaticShearContour'): Calc.Plot.Beam.P06_StaticShearContour = 0
    if not hasattr(Calc.Plot.Beam, 'P07_VertAccContour'): Calc.Plot.Beam.P07_VertAccContour = 0
    if not hasattr(Calc.Plot.Beam, 'P08_Sections_BeamVertDisp'): Calc.Plot.Beam.P08_Sections_BeamVertDisp = 0
    if not hasattr(Calc.Plot.Beam, 'P09_Sections_BeamBM'): Calc.Plot.Beam.P09_Sections_BeamBM = 0
    if not hasattr(Calc.Plot.Beam, 'P10_Sections_BeamShear'): Calc.Plot.Beam.P10_Sections_BeamShear = 0
    if not hasattr(Calc.Plot.Beam, 'P11_Sections_BeamAcc'): Calc.Plot.Beam.P11_Sections_BeamAcc = 0

    # ------------------------------- Vehicle ---------------------------------

    # Velocity and acceleration
    Train.Veh[0].VelAcc = [Train.vel, 0, 2]     # [v0, a0, aa]

    # Default calculation of Vehicle natural frequencies
    if not hasattr(Calc.Options, 'calc_veh_frq'):
        Calc.Options.calc_veh_frq = 1

    # Ensure Calc.Position exists
    if not hasattr(Calc, 'Position'):
        Calc.Position = EmptyObj()

    # Load position, velocity and acceleration
    Calc.Position.x_0 = Calc.Position.x_start_end[0]
    Calc.Position.x_end = Calc.Position.x_start_end[1]
    Calc.Position.v_0 = Train.Veh[0].VelAcc[0]
    Calc.Position.a_0 = Train.Veh[0].VelAcc[1]
    Calc.Position.aa = Train.Veh[0].VelAcc[2]
    
    if abs(Calc.Position.a_0) > 0 and Calc.Position.aa < 2:
        print('The acceleration exponent aa should be greater or equal to 2!!!')

    # -------------------------------- Beam -----------------------------------
    
    if not hasattr(Beam, 'Options'):
        Beam.Options = EmptyObj()

    Beam.Options.k_Mconsist = 1        # Consistent Mass Matrix (CMM) = 1; 0 = LMM

    # Ensure Beam.BC exists to prevent AttributeErrors
    if not hasattr(Beam, 'BC'):
        Beam.BC = EmptyObj()
        Beam.BC.text = ''

    # Boundary conditions variables
    # Note: Python compares full strings natively, no need for all() like MATLAB
    if Beam.BC.text == 'SP':
        Beam.BC.loc = [0, Beam.Prop.L] 
        Beam.BC.vert_stiff = [-1, -1] 
        Beam.BC.rot_stiff = [0, 0]
        Beam.BC.text_long = 'Simply Supported'
    elif Beam.BC.text == 'FF':
        Beam.BC.loc = [0, Beam.Prop.L] 
        Beam.BC.vert_stiff = [-1, -1] 
        Beam.BC.rot_stiff = [-1, -1]
        Beam.BC.text_long = 'Fixed-Fixed'
    elif Beam.BC.text == 'UD':
        Beam.BC.loc = [0, Beam.Prop.L / 2, Beam.Prop.L]  # 2*(Beam.Prop.L)/2 simplified
        Beam.BC.vert_stiff = [1, 1, 1]              # Vertical stiffness of supports (-1 perfectly stiff, >0 has stiffness)
        Beam.BC.rot_stiff = [0, 0, 0]               # Rotational stiffness (-1 perfectly stiff, 0 Free rotation)
        Beam.BC.text = 'User-defined'
        Beam.BC.text_long = '2-span bridge'

    # Default calculation of Beam natural frequencies
    if not hasattr(Calc.Options, 'calc_beam_frq'):
        Calc.Options.calc_beam_frq = 1

    # Default calculation of Beam modes of vibration
    if not hasattr(Calc.Options, 'calc_beam_modes'):
        Calc.Options.calc_beam_frq = 1
        Calc.Options.calc_beam_modes = 1

    # Beam Damping
    if not hasattr(Beam, 'Damping'):
        Beam.Damping = EmptyObj()
        Beam.Damping.per = 0

    # Default BM calculation method
    if not hasattr(Calc.Options, 'BM_calc_mode'):
        Calc.Options.BM_calc_mode = 1  # The average nodal result is considered

    # Default Shear calculation method
    if not hasattr(Calc.Options, 'Shear_calc_mode'):
        Calc.Options.Shear_calc_mode = 1  # The average nodal result is considered

    # -------------------------------- Track ----------------------------------

    # Rail damping
    if not hasattr(Track.Rail, 'Damping'):
        Track.Rail.Damping = EmptyObj()
        Track.Rail.Damping.per = 0

    # -------------------------------- Checks ---------------------------------

    if Calc.Options.num_calc_beam_sections > 0:
        # Check if any section is out of bounds
        out_of_bounds = any(x < 0 or x > Beam.Prop.L for x in Calc.Options.calc_beam_sections)
        if out_of_bounds:
            print('Beam calculation section not on beam')
            raise ValueError('Beam calculation section not on beam')

    if not hasattr(Track, 'BallastOnBeam'):
        Track.BallastOnBeam = EmptyObj()
        Track.BallastOnBeam.included = 0
    else:
        if not hasattr(Track.BallastOnBeam, 'included'):
            Track.BallastOnBeam.included = 1

    if not hasattr(Track, 'PadUnderSleeperOnBeam'):
        Track.PadUnderSleeperOnBeam = EmptyObj()
        Track.PadUnderSleeperOnBeam.included = 0
    else:
        if not hasattr(Track.PadUnderSleeperOnBeam, 'included'):
            Track.PadUnderSleeperOnBeam.included = 1

    check = Track.BallastOnBeam.included + Track.PadUnderSleeperOnBeam.included
    
    if check == 0:
        print('Include model properties of either:')
        print(" " * 8 + 'Ballast on Beam')
        print(" " * 8 + 'Pad under Sleeper on Beam')
        raise ValueError('Missing information for "Ballast on Beam" or "Pad under Sleeper on Beam"')
    elif check == 2:
        print('Too many model properties have been defined.')
        print('Remove the properties of one of the following:')
        print(" " * 8 + 'Ballast on Beam')
        print(" " * 8 + 'Pad under Sleeper on Beam')
        raise ValueError('Properties defined simultaneously for "Ballast on Beam" and "Pad under Sleeper on Beam"')

    # Calculate how many sleepers fit into the profile length
    ratio = Calc.Profile.L / Track.Sleeper.spacing

    # Check if the difference between the exact ratio and the nearest whole number exceeds the tolerance
    if abs(ratio - round(ratio)) * Track.Sleeper.spacing > Calc.Cte.tol:
        print('Profile length is wrong')
        raise ValueError('Profile length is wrong')

    # Redux model
    if Calc.Options.redux == 0:
        print('No redux model is used! Are you sure about this?')

    return Calc, Train, Track, Beam