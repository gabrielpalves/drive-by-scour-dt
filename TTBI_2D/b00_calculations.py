import time
import numpy as np

# Import block: from file_name import function_name
from b43_model_geometry import b43_model_geometry
from b07_options_processing import b07_options_processing
from b01_elements_and_coordinates import b01_elements_and_coordinates
from b02_boundary_conditions import b02_boundary_conditions
from b03_beam_matrices import b03_beam_matrices
from b09_beam_frq import b09_beam_frq
from b24_beam_damping import b24_beam_damping
from b51_rail_variables import b51_rail_variables
from b54_model_matrices import b54_model_matrices
from b55_model_bc import b55_model_bc
from b56_model_frq import b56_model_frq
from b18_train_veh_eq import b18_train_veh_eq
from b08_veh_freq import b08_veh_freq
from b10_end_time import b10_end_time
from b11_time_space_discretization import b11_time_space_discretization
from b47_veh_static_loads import b47_veh_static_loads
from b19_generate_profile import b19_generate_profile
from b25_wheel_profiles import b25_wheel_profiles
from b50_element_num_of_force import b50_element_num_of_force
from b64_coupled_initial_static import b64_coupled_initial_static
from b65_dynamic_calc_coupled_faster import b65_dynamic_calc_coupled_faster
from b66_contact_force import b66_contact_force
from b49_beam_deformation import b49_beam_deformation
from b31_beam_bm import b31_beam_bm
from b33_beam_shear import b33_beam_shear
from b53_beam_acceleration import b53_beam_acceleration
from b58_results_beam_sections import b58_results_beam_sections

def b00_calculations(Calc, Train, Track, Beam, Damage):
    """
    Core calculations of TTB-2D simulation.
    Basically this script does the following:
       - Inputs processing and auxiliary variables generation
       - Model generation (system matrices and load sequence)
       - Numerical integration of solution
       - Generation of additional results based on the solution
       
    Uses and modifies the variables: Train, Track, Beam and Calc.
       Generates the simulation results and saves them in variable Sol.
    
    *************************************************************************
    *** Script part of TTB-2D tool for Python environment.                ***
    *** Original Author: Daniel Cantero (daniel.cantero@ntnu.no)          ***
    *************************************************************************
    """

    # -- Model geometry --
    Calc, Train, Beam = b43_model_geometry(Calc, Train, Track, Beam)

    # -- Options Processing --
    Calc, Train, Track, Beam = b07_options_processing(Calc, Train, Track, Beam)

    # ---- Beam Model ----
    # Elements and coordinates
    Beam = b01_elements_and_coordinates(Beam, Calc)

    # MATLAB uses 1-based indexing. If n_mod contains indices, ensure they are 
    # 0-based in Python before this step, or subtract 1: Beam.Prop.n_mod - 1
    Beam.Prop.E_n[Beam.Prop.n_mod] = Beam.Prop.E * Beam.Prop.E_mod

    # Boundary conditions 
    Beam, Damage = b02_boundary_conditions(Beam, Damage)
    # Beam system matrices (Mass and Stiffness)
    Beam = b03_beam_matrices(Beam)
    # Beam frequencies 
    Beam = b09_beam_frq(Beam, Calc)
    # Beam Damping Matrix
    Beam = b24_beam_damping(Beam)

    # ---- Track Model ----
    # Definition of auxiliary variables
    Track = b51_rail_variables(Track, Calc)
    # Elements and coordinates
    Track.Rail = b01_elements_and_coordinates(Track.Rail)
    # Boundary conditions 
    Track.Rail, _ = b02_boundary_conditions(Track.Rail, Damage)
    # Rail system matrices (Mass and stiffness)
    Track.Rail = b03_beam_matrices(Track.Rail)
    # Rail frequencies
    Track.Rail = b09_beam_frq(Track.Rail, Calc)
    
    # Reference frequencies for damping definition
    zeros_array = np.zeros(int(Track.Rail.Modal.num_rigid_modes))
    Track.Rail.Modal.w = np.concatenate((zeros_array, Beam.Modal.w[0:2]))
    
    # Beam Damping Matrix
    Track.Rail = b24_beam_damping(Track.Rail)

    # ---- Complete Model ----
    # System matrices
    print('Building model system matrices ...', end='', flush=True)
    Model = b54_model_matrices(Beam, Track, Calc)
    # Model boundary conditions
    Model = b55_model_bc(Model, Beam, Track)
    # Modal analysis of Model
    Model = b56_model_frq(Model, Calc)
    print(' DONE') 

    # ---- Vehicle Model ----
    # Vehicles system matrices
    Train.Veh = b18_train_veh_eq(Train.Veh)
    # Vehicle frequencies
    Train.Veh = b08_veh_freq(Train.Veh, Calc)
    # Vehicle position and Solver time array
    Calc = b10_end_time(Calc)
    Calc = b11_time_space_discretization(Calc)
    # Vehicle static loads
    Train.Veh = b47_veh_static_loads(Train.Veh, Calc)

    # ---- Irregularity profile ----
    # Generation
    Calc = b19_generate_profile(Calc)
    # Assigning profile to each wheel
    Calc = b25_wheel_profiles(Calc, Train.Veh)

    # ---- Coupled Equations Solver (COUP) ----
    # Element number of vertical forces in time
    Calc = b50_element_num_of_force(Track.Rail, Calc)
    # Initial coupled system's static deformation
    Sol = b64_coupled_initial_static(Train.Veh, Model, Calc, Track)
    
    print('Performing Dynamic Calculations (Coupled System) ...')
    start_time = time.time() 
    
    Sol = b65_dynamic_calc_coupled_faster(Train.Veh, Model, Calc, Track, Sol, Damage)
    
    elapsed_time = time.time() - start_time 
    print(f"Calculation time: {elapsed_time:.2f}s")

    # ---- Additional results from solution ----
    # Contact Force
    Sol = b66_contact_force(Sol, Track, Calc, Train)
    # Beam deformation
    Sol = b49_beam_deformation(Sol, Model, Beam, Calc, Train, 1)
    # Static beam deformation
    Sol = b49_beam_deformation(Sol, Model, Beam, Calc, Train, 0)
    # Beam Bending Moment
    Sol = b31_beam_bm(Sol, Model, Beam, Calc, 1)
    # Static Beam Bending Moment
    Sol = b31_beam_bm(Sol, Model, Beam, Calc, 0)
    # Beam Shear Force
    Sol = b33_beam_shear(Sol, Model, Beam, Calc, 1)
    # Static Beam Shear Force
    Sol = b33_beam_shear(Sol, Model, Beam, Calc, 0)
    # Beam Acceleration
    Sol = b53_beam_acceleration(Sol, Model, Beam, Calc)
    # Results at selected Beam sections
    Sol = b58_results_beam_sections(Sol, Beam, Calc)

    print('All calculations finished successfully')

    return Sol, Calc, Train, Beam, Track

# ---- End of script ----