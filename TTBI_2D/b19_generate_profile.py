#!/usr/bin/env python3

"""
Calculates the irregularity profile for the rail
*************************************************************************
*** Script part of TTB-2D tool for Python environment.                 ***
*** Licensed under the GNU General Public License v3.0                 ***
*** Author: Daniel Cantero (daniel.cantero@ntnu.no)                    ***
*** For help, modifications, and collaboration contact the author.     ***
*************************************************************************
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io  # <-- Import for .mat file loading
import os
from types import SimpleNamespace

# -------------------------------------------------------------------------
# === Helper functions to convert .mat structs to Python Objects (Robust) ===
# -------------------------------------------------------------------------

def _mat_to_obj(mat_data):
    """
    Recursively converts a MATLAB object (loaded by scipy.io)
    into Python SimpleNamespace objects, lists, and native types.
    """
    
    # If it's a tuple, it was likely a 1xN cell array. Recurse on items.
    if isinstance(mat_data, tuple):
        return [_mat_to_obj(item) for item in mat_data]

    # If it's a numpy array
    if isinstance(mat_data, np.ndarray):
        
        # Case 1: Struct array (e.g., Veh(1), Veh(2))
        if mat_data.dtype.names:
            if mat_data.size == 1:
                # Scalar struct, unpack and process its fields
                return _mat_scalar_struct_to_obj(mat_data.item())
            else:
                # Struct array, return a list of objects
                return [_mat_scalar_struct_to_obj(s) for s in mat_data]
        
        # Case 2: Cell array (loaded as object array)
        elif mat_data.dtype == 'object':
            if mat_data.size == 1:
                # 1x1 cell array, unpack and recurse
                return _mat_to_obj(mat_data.item())
            else:
                # N-D cell array, convert to list and recurse
                return [_mat_to_obj(item) for item in mat_data.flat]
        
        # Case 3: Regular numeric/logical array
        else:
            if mat_data.size == 1:
                return mat_data.item()  # Convert 1x1 array to scalar
            return mat_data  # Keep as numpy array

    # Case 4: Scalar struct (not in an array)
    elif isinstance(mat_data, np.void) and mat_data.dtype.names:
        return _mat_scalar_struct_to_obj(mat_data)
    
    # Case 5: Primitive type (int, float, str, bool)
    elif isinstance(mat_data, (str, int, float, bool)):
        return mat_data
        
    # Case 6: Other types (like None, etc.)
    else:
        return mat_data

def _mat_scalar_struct_to_obj(mat_struct_scalar):
    """Converts a single (scalar) struct (np.void) into a Python SimpleNamespace."""
    py_obj = SimpleNamespace()
    if mat_struct_scalar.dtype.names:
        for name in mat_struct_scalar.dtype.names:
            # Recurse on the field's value and set as object attribute
            setattr(py_obj, name, _mat_to_obj(mat_struct_scalar[name]))
    return py_obj

# -------------------------------------------------------------------------
# === Main Profile Generation Function ===
# -------------------------------------------------------------------------

def _psd_to_profile(PSD_Y, N, x_target):
    """
    Generates a random profile from a 1-sided PSD using IFFT.
    This is a helper function translating the MATLAB subfunction.
    """
    
    # --- Auxiliary variables ---
    dx = x_target[1] - x_target[0]
    nx = len(x_target)
    N = int(N) # Ensure N is an integer
    
    # --- Two-sided PSD ---
    PSD_Y = PSD_Y.copy() # Avoid modifying the original
    PSD2 = PSD_Y / 2.0
    
    if len(PSD2) > 0:
        PSD2[0] = 0
    if len(PSD2) > 1:
        PSD2[-1] = 0
    
    PSD2 = np.concatenate((PSD2, np.flip(PSD2[1:-1])))
    
    # --- Amplitude ---
    Amplitude = np.sqrt(PSD2 * N / dx)
    
    # --- Random Phase values ---
    rand_phases = 2 * np.pi * np.random.rand(int(N / 2) - 1)
    phase = np.concatenate(([0], rand_phases, [0], -np.flip(rand_phases)))
    
    # --- Fourier transform coefficients ---
    FFT_y = Amplitude * np.exp(1j * phase)
    
    # --- Inverse Fourier Transform ---
    iFFT_y = fft.ifft(FFT_y, N)
    
    y = np.real(iFFT_y[:nx])
    
    return y

def b19_generate_profile(Calc):
    """
    Calculates the irregularity profile for the rail.
    
    Args:
        Calc (SimpleNamespace): The Calculation options object.
        
    Returns:
        SimpleNamespace: The updated Calc object with profile data.
    """

    # Ensure Profile namespace exists
    if not hasattr(Calc, 'Profile'):
        Calc.Profile = SimpleNamespace()

    # --- Profile sampling rate ---
    try:
        min_dx_pos = np.min(np.abs(np.diff(Calc.Position.x)))
    except (ValueError, AttributeError):
        min_dx_pos = np.inf
        
    Calc.Profile.dx = min(min_dx_pos, getattr(Calc.Profile, 'min_dx', min_dx_pos))
    dx = Calc.Profile.dx

    # --- Create new profile x-axis ---
    x_start = Calc.Position.x[0]
    x_end = Calc.Profile.L
    
    Calc.Profile.x = np.arange(x_start, x_end + dx, dx)
    Calc.Profile.nx = len(Calc.Profile.x)

    # ---- Type of Road Profile ----
    profile_type = getattr(Calc.Profile, 'Type', 0)

    # --- Smooth ---
    if profile_type == 0:
        Calc.Profile.h = np.zeros(Calc.Profile.nx)
        Calc.Profile.text = 'Smooth'
        
    # --- From PSD ---
    elif profile_type == 1:
        min_spaf = 1 / Calc.Profile.max_WaveLength
        max_spaf = 1 / Calc.Profile.min_WaveLength
        
        N = 1 << int(np.ceil(np.log2(Calc.Profile.nx)))
        
        fs = 1 / dx
        PSD_X = (fs / N) * np.arange(int(N / 2) + 1)
        
        PSD_Y = Calc.Profile.PSD_Y_fun(PSD_X, Calc.Profile.inputs)
        
        PSD_Y[PSD_X < min_spaf] = 0
        PSD_Y[PSD_X > max_spaf] = 0

        Calc.Profile.h = _psd_to_profile(PSD_Y, N, Calc.Profile.x)
        Calc.Profile.PSD_X = PSD_X
        Calc.Profile.PSD_Y = PSD_Y
        
    # --- Profile from a dataset ---
    elif profile_type == 2:
        print("B19_GenerateProfile: Bypassing calculations and loading hardcoded debug values.")

        try:
            # --- 1. Hardcode the scalar values ---
            Calc.Profile.Type = 1
            
            # Translate MATLAB anonymous function to Python lambda
            Calc.Profile.PSD_Y_fun = lambda spaf, inputs: \
                inputs[0] * inputs[2]**2 * (spaf**2 + inputs[1]**2) / \
                (spaf**4 * (spaf**2 + inputs[2]**2))
                
            Calc.Profile.inputs = np.array([2.75e-08, 0.0233, 0.131])
            Calc.Profile.min_WaveLength = 1.5240
            Calc.Profile.max_WaveLength = 304.80
            Calc.Profile.text = 'FRA'
            Calc.Profile.minL_Approach = 30.0
            Calc.Profile.minL_After = 30.0
            
            # These are set by B43, but we set them here to match the debug state
            Calc.Profile.L_Approach = 30.0
            Calc.Profile.L_After = 30.0
            Calc.Profile.L_bridge = 39.90
            Calc.Profile.max_TL = 106.80
            Calc.Profile.extra_L = 0.30
            Calc.Profile.extra_L2 = 6
            Calc.Profile.L = 325.80
            Calc.Profile.L_Aw = 136.80
            Calc.Profile.min_dx = 0.010
            Calc.Profile.dx = 0.010
            Calc.Profile.nx = 32581

            # --- 2. Load the 4 arrays from separate .mat files ---
            # Note: scipy.io.loadmat inherently returns dictionaries, which is correct here
            # for extracting individual arrays by their string keys.
            
            x_data = scipy.io.loadmat('x.mat', squeeze_me=True)
            Calc.Profile.x = x_data['x']
            
            h_data = scipy.io.loadmat('h.mat', squeeze_me=True)
            Calc.Profile.h = h_data['h']

            psd_x_data = scipy.io.loadmat('PSD_X.mat', squeeze_me=True)
            Calc.Profile.PSD_X = psd_x_data['PSD_X']

            psd_y_data = scipy.io.loadmat('PSD_Y.mat', squeeze_me=True)
            Calc.Profile.PSD_Y = psd_y_data['PSD_Y']
            
            print("B19: Successfully loaded x, h, PSD_X, and PSD_Y from .mat files.")

        except FileNotFoundError as e:
            print(f"Error in B19: Could not find required .mat file: {e.filename}")
            print("This debug script requires x.mat, h.mat, PSD_X.mat, and PSD_Y.mat to be present.")
            raise # Stop the simulation
        except KeyError as e:
            print(f"Error in B19: .mat file is missing expected variable: {e}")
            raise # Stop the simulation
        except Exception as e:
            print(f"Error loading .mat files in B19: {e}")
            raise # Stop the simulation

    # ---- Rail-irregularity intensity toggle ----
    # Scale the irregularity amplitude (proxy for track degradation severity).
    # Default 1.0 leaves the profile unchanged; 0.0 -> perfectly smooth.
    intensity = float(getattr(Calc.Profile, 'intensity', 1.0))
    if intensity != 1.0:
        Calc.Profile.h = Calc.Profile.h * intensity

    # ---- Plotting profile ----
    if hasattr(Calc, 'Plot') and getattr(Calc.Plot, 'Profile_original', 0) == 1:
        plt.figure(figsize=(10, 3))
        plt.plot(Calc.Profile.x, Calc.Profile.h * 1000)
        plt.xlim(Calc.Profile.x[0], Calc.Profile.x[-1])
        plt.xlabel('Distance (m)')
        plt.ylabel('Profile Elevation (mm)')
        plt.title(f"Generated Profile: {getattr(Calc.Profile, 'text', 'N/A')}")
        plt.grid(True, linestyle=':')
        plt.show(block=False)
        plt.pause(0.25)
        
    return Calc