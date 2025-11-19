import scipy.io
import numpy as np
import os
import random
import warnings

def load_data(damage_cases, dof=1, Npass=200, file_path='.'):
    """
    Loads and processes scour data from .mat files.
    
    New Data Structure:
    - Files: 0001.mat ... 0061.mat
    - Inside: 'data' (1x200 cell)
    - Inside Cell: (3xN matrix)
    
    New Labeling:
    - 0001.mat (index 0) -> 60% Damage
    - 0002.mat (index 1) -> 59% Damage
    - ...
    - 0061.mat (index 60) -> 0% Damage

    Args:
        damage_cases (list or range): A list of 0-indexed file indices to load
                                     (e.g., [0, 1, 2, ... 60]).
        dof (int, optional): The 0-indexed row (Degree of Freedom) to extract
                             from the 3xN matrix. Defaults to 1 (the 2nd row).
        Npass (int, optional): The number of random passages (cells) to sample
                             from each file. Defaults to 200.
        file_path (str, optional): Path to the directory containing .mat files.
                                   Defaults to '.'.

    Returns:
        X_data (np.ndarray): 
            3D array, shape: (n_scenarios, Npass, n_cols)
            
        y_labels (np.ndarray): 
            2D array, shape: (n_scenarios, Npass)
            Contains the damage % (60, 59, ..., 0)
            
        X_flat (np.ndarray):
            2D array, shape: (n_scenarios * Npass, n_cols)
            
        y_flat (np.ndarray):
            1D array, shape: (n_scenarios * Npass,)
            
        Returns (None, None, None, None) if a critical error occurs.
    """
    
    # --- 1. Create the File Index to Damage % Label Map ---
    # map[0] = 60, map[1] = 59, ..., map[60] = 0
    label_map = dict(zip(range(61), range(60, -1, -1)))

    print(f"--- Loading Data ---")
    print(f"Target DOF (row): {dof}")
    print(f"Passages per case: {Npass}")

    all_cases_data = []
    all_cases_labels = []
    n_cols_expected = -1  # To store the length of the first time series

    # --- 2. Loop through each requested damage case index ---
    for i, dc_index in enumerate(damage_cases):
        
        # File name is 1-indexed (e.g., index 0 is file 0001.mat)
        file_name = f"{(dc_index + 1):04d}.mat"
        full_path = os.path.join(file_path, file_name)
        
        # Get the correct damage label (60, 59, ...)
        try:
            dc_label = label_map[dc_index]
        except KeyError:
            warnings.warn(f"Invalid damage case index: {dc_index}. Must be 0-60. Skipping.")
            continue

        try:
            # Load the .mat file
            mat = scipy.io.loadmat(full_path)
            
            # Access the 'data' variable (1x200 cell)
            # This loads as a (1, 200) numpy object array
            variable_data = mat['data']
            
            n_total_pass_available = variable_data.shape[1]
            
            if n_total_pass_available == 0:
                warnings.warn(f"File {file_name} contains 0 passages. Skipping.")
                continue

            # --- 3. Get random passage indices ---
            passage_indices_all = list(range(n_total_pass_available))
            indices_to_load = []
            
            if Npass > n_total_pass_available:
                warnings.warn(f"Warning: Requested {Npass} passages from {file_name}, "
                              f"but only {n_total_pass_available} exist. "
                              f"Sampling WITH replacement.")
                indices_to_load = random.choices(passage_indices_all, k=Npass)
            else:
                indices_to_load = random.sample(passage_indices_all, Npass)
            
            passages_for_this_case = []
            labels_for_this_case = []

            # --- 4. Extract the data for each passage ---
            for passage_index in indices_to_load:
                # Access the cell element (3xN matrix)
                passage_array = variable_data[0, passage_index]
                
                if not isinstance(passage_array, np.ndarray):
                    warnings.warn(f"Data in {file_name}, passage {passage_index} is not a valid matrix. Skipping.")
                    continue
                    
                if dof >= passage_array.shape[0]:
                    warnings.warn(f"DOF {dof} not available in {file_name}, passage {passage_index}. "
                                  f"Matrix shape is {passage_array.shape}. Skipping.")
                    continue
                        
                # Get the specified DOF row
                time_series = passage_array[dof, :].astype(np.float32)
                
                # --- 5. Ensure consistent time series length ---
                if n_cols_expected == -1:
                    # This is the first time series we've loaded.
                    n_cols_expected = len(time_series)
                
                current_len = len(time_series)
                if current_len != n_cols_expected:
                    warnings.warn(f"Inconsistent length in {file_name}, passage {passage_index}. "
                                  f"Was {current_len}, expected {n_cols_expected}. Padding/Truncating.")
                    if current_len > n_cols_expected:
                        time_series = time_series[:n_cols_expected] # Truncate
                    else:
                        # Pad with zeros
                        padding = n_cols_expected - current_len
                        time_series = np.pad(time_series, (0, padding), 'constant')
                
                passages_for_this_case.append(time_series)
                labels_for_this_case.append(dc_label)
            
            if passages_for_this_case:
                all_cases_data.append(passages_for_this_case)
                all_cases_labels.append(labels_for_this_case)

        except FileNotFoundError:
            print(f"Error: File not found: {full_path}")
        except KeyError:
            print(f"Error: Variable 'data' not found in {full_path}")
        except Exception as e:
            print(f"An unknown error occurred with {full_path}: {e}")

    # --- 6. Convert to Numpy arrays and return ---
    if not all_cases_data:
        print("\nError: No data was loaded at all.")
        return None, None, None, None

    try:
        X_data = np.array(all_cases_data)
        y_labels = np.array(all_cases_labels, dtype=int)
        
        n_scenarios, n_pass, n_cols = X_data.shape
        X_flat = X_data.reshape((n_scenarios * n_pass, n_cols))
        y_flat = y_labels.reshape((n_scenarios * n_pass))
        
        print("\n--- Data Loading Finished ---")
        return X_data, y_labels, X_flat, y_flat
        
    except ValueError as e:
        print("\nError: Could not create final numpy array. ")
        print("This is likely due to inconsistent time series lengths that weren't fixed.")
        print(f"Details: {e}")
        return None, None, None, None

# --- Example of how to use the function ---
if __name__ == "__main__":
    
    # --- Your Configuration ---
    
    # Load all 61 damage cases (file indices 0 to 60)
    DAMAGE_CASES_TO_LOAD = list(range(61)) 
    
    # Use the 2nd row (Python index 1)
    DOF_TO_USE = 1
    
    # Load 50 random passages from each
    PASSAGES_TO_LOAD = 50
    
    # Path to your files
    FILE_DIRECTORY = 'data_only_noise' # Assumes files are in the same directory
    
    # --- Run the function ---
    X_data, y_labels, X_flat, y_flat = load_data(
        damage_cases=DAMAGE_CASES_TO_LOAD,
        dof=DOF_TO_USE,
        Npass=PASSAGES_TO_LOAD,
        file_path=FILE_DIRECTORY
    )
    
    if X_data is not None:
        print("\n--- Example Verification ---")
        print(f"Shape of X_data (3D): {X_data.shape}")
        print(f"Shape of y_labels (2D): {y_labels.shape}")
        print(f"Shape of X_flat (2D): {X_flat.shape}")
        print(f"Shape of y_flat (1D): {y_flat.shape}")

        print("\n--- Label Check ---")
        print(f"First label (from file 0001.mat): {y_labels[0, 0]} (Expected 60)")
        print(f"Last label (from file 0061.mat): {y_labels[-1, -1]} (Expected 0)")
        print(f"Total unique labels: {len(np.unique(y_flat))}")