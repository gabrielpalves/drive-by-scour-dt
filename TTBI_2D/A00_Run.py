import os
import re
import math
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import scipy.io as sio

# Assuming all your translated modules are imported here:
from a01_train import a01_train
from a02_track import a02_track
from a03_bridge import a03_bridge
from a04_options import a04_options
from b00_calculations import b00_calculations
from d01_data_processing import d01_data_processing

class EmptyObj:
    pass

def save_progress(data, dc_idx, run_folder):
    """
    Helper function to serialize and save the progress of a damage case (DC)
    using Python's native pickle format.
    """
    # Saves as 0001.pkl, 0002.pkl, etc.
    filename = os.path.join(run_folder, f"{dc_idx:04d}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    print("\n" + "="*73)
    print("*** Script part of TTB-2D tool for Python environment.                ***")
    print("*** Licensed under the GNU General Public License v3.0                ***")
    print("*** For help, modifications, and collaboration contact the author.    ***")
    print("="*73 + "\n")

    results_folder = 'Results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Get list of existing run folders (subdirectories)
    # Exclude '.' and '..' natively by using os.listdir and isdir
    run_dirs_names = [d for d in os.listdir(results_folder) 
                      if os.path.isdir(os.path.join(results_folder, d))]

    # Sort runs by newest first by parsing the datetime folder names
    # Assuming format: dd_MM_yyyy_HH_mm_ss
    date_format = "%d_%m_%Y_%H_%M_%S"
    
    run_dirs = []
    for name in run_dirs_names:
        try:
            dt_obj = datetime.strptime(name, date_format)
            run_dirs.append({'name': name, 'date': dt_obj})
        except ValueError:
            # If a folder doesn't match the timestamp format, put it at the end
            run_dirs.append({'name': name, 'date': datetime.min})

    run_dirs.sort(key=lambda x: x['date'], reverse=True)

    # =========================================================================
    # Show run status + disk usage + completion + sort by date
    # =========================================================================
    print('\nRun Status (sorted by date):')
    print('-' * 90)
    print(f"{'#':<3} {'Run Name':<22} {'Status':<12} {'Completion':<20} {'Size(MB)':<10}")
    print('-' * 90)

    for i, run_info in enumerate(run_dirs):
        run_name = run_info['name']
        run_path = os.path.join(results_folder, run_name)
        
        ini_file = os.path.join(run_path, 'tempo_inicial.pkl')
        end_file = os.path.join(run_path, 'tempo_final.pkl')

        # ------------------------------------------------------------
        # Determine run status
        # ------------------------------------------------------------
        if os.path.exists(ini_file) and os.path.exists(end_file):
            status = 'COMPLETE'
        elif os.path.exists(ini_file) and not os.path.exists(end_file):
            status = 'INCOMPLETE'
        else:
            status = 'UNAVAILABLE'

        # ------------------------------------------------------------
        # AUTO-DETECT TOTAL_DC (4-digit filenames)
        # ------------------------------------------------------------
        max_idx = 0
        completed_dc = 0
        
        # Searching for files like 0001.pkl, 0002.pkl
        for f_name in os.listdir(run_path):
            match = re.match(r'^(\d{4})\.pkl$', f_name)
            if match:
                dc_num = int(match.group(1))
                completed_dc += 1
                if dc_num > max_idx:
                    max_idx = dc_num

        TOTAL_DC = max_idx if max_idx > 0 else float('nan')

        # ------------------------------------------------------------
        # Compute completion %
        # ------------------------------------------------------------
        if math.isnan(TOTAL_DC):
            completion_str = '---'
        else:
            pct = 100.0 * completed_dc / TOTAL_DC
            completion_str = f"{completed_dc}/{TOTAL_DC} ({pct:5.1f}%)"

        # ------------------------------------------------------------
        # Disk usage
        # ------------------------------------------------------------
        # rglob('*') recursively finds all files
        total_bytes = sum(f.stat().st_size for f in Path(run_path).rglob('*') if f.is_file())
        size_MB = total_bytes / (1024**2)

        # ------------------------------------------------------------
        # Print run information
        # ------------------------------------------------------------
        print(f"{i+1:<3} {run_name:<22} {status:<12} {completion_str:<20} {size_MB:8.2f}")

    print('-' * 90 + '\n')

    # =========================================================================
    # Ask user to start new or resume run
    # =========================================================================
    print('Available runs:')
    for i, run_info in enumerate(run_dirs):
        print(f"{i+1} - {run_info['name']}")

    choice = 0
    if not run_dirs:
        print('Starting a NEW run')
    else:
        print('0 - Start a NEW run')
        try:
            choice = int(input('Enter your choice: '))
        except ValueError:
            choice = 0 # Default to 0 if input is invalid

    if choice == 0:
        # Start a new run (create timestamped folder)
        tempo_inicial = datetime.now()
        tempo_inicial_str = tempo_inicial.strftime(date_format)
        print(f'Starting NEW run: {tempo_inicial_str}')
        
        run_path = os.path.join(results_folder, tempo_inicial_str)
        os.makedirs(run_path, exist_ok=True)
        
        # Save tempo_inicial inside the folder for traceability
        with open(os.path.join(run_path, 'tempo_inicial.pkl'), 'wb') as f:
            pickle.dump(tempo_inicial, f)
    else:
        # Resume an existing run (convert 1-based choice to 0-based index)
        idx = choice - 1
        tempo_inicial_str = run_dirs[idx]['name']
        run_path = os.path.join(results_folder, tempo_inicial_str)
        print(f'Resuming run: {tempo_inicial_str}')
        
        tempo_file = os.path.join(run_path, 'tempo_inicial.pkl')
        if os.path.exists(tempo_file):
            with open(tempo_file, 'rb') as f:
                tempo_inicial = pickle.load(f)
        else:
            tempo_inicial = datetime.strptime(tempo_inicial_str, date_format)

    # =========================================================================
    # Simulation setup
    # =========================================================================
    # User Configuration - Set these to True or False
    use_signal_noise = False           # Toggle artificial signal noise
    use_vehicle_variability = False   # Toggle vehicle property variability
    use_speed_variability = False     # Toggle train speed variability
    use_temp_variability = False      # Toggle temperature variability
    
    Dano = np.linspace(0.40, 1.00, 301)
    Npass = 5         # number of passages per damage case
    Nveh = 5          # number of vehicles
    Nprop = 3         # how many vehicle properties will be varied
    Desvio = 0.05     # standard deviation of the noise of the signal (artificial noise)
    
    temp_min, temp_max = 3, 33        # min and max temperature
    vel_min, vel_max = 70, 90         # min and max velocity (km/h)
    
    run_folder = os.path.join('Results', tempo_inicial_str)

    # Identify which DCs have already been processed (for resume mode)
    # Note: Using 1-based indexing for DC to match your Dano length mapping perfectly
    completed = np.zeros(len(Dano) + 1, dtype=bool) 
    
    for f_name in os.listdir(run_folder):
        match = re.match(r'^(\d{4})\.pkl$', f_name)
        if match:
            dc_idx = int(match.group(1))
            if 1 <= dc_idx <= len(Dano):
                completed[dc_idx] = True

    # =========================================================================
    #  Parallel / Main loop
    # =========================================================================
    for DC in range(1, len(Dano) + 1): # 1-based loop to match MATLAB logic
        if completed[DC]:
            print(f'Skipping DC {DC} — result already exists.')
            continue
            
        Damage = EmptyObj()
        data = EmptyObj()
        Beam = EmptyObj()
        Track = EmptyObj()
        
        Damage.desvio = Desvio * int(use_signal_noise)
        
        # Latin Hypercube Sampling (LHS)
        # scipy generates a (samples x variables) matrix. 
        # Transpose it to (variables x samples) to match MATLAB's lhs(1, t) indexing.
        sampler = qmc.LatinHypercube(d=2)
        lhs = sampler.random(n=Npass).T 
        
        Damage.DOFStiff_ROT_value = 0
        Beam.Prop = EmptyObj()
        Beam.Prop.E_mod = 1
        
        # Python arrays are 0-indexed, so we access Dano[DC - 1]
        Damage.DOF_ChangeRate_variab = [Dano[DC - 1]] 
        Beam.Prop.n_mod = 100
        
        data.Aceleracao = [[None] * Npass for _ in range(len(Damage.DOF_ChangeRate_variab))]
        data.Velocidade = [[None] * Npass for _ in range(len(Damage.DOF_ChangeRate_variab))]
        data.Posicao = [[None] * Npass for _ in range(len(Damage.DOF_ChangeRate_variab))]
        
        for i in range(len(Damage.DOF_ChangeRate_variab)):
            Damage.DOF_ChangeRate_value = Damage.DOF_ChangeRate_variab[i]
            
            # Operational variability - train velocity
            Velocidade = np.ones(Npass) * (80.0 / 3.6)
            Temperatura = np.ones(Npass) * 25.0
            
            # Vehicle Variability
            if use_vehicle_variability:
                x_veh = np.ones((Nveh, Nprop, Npass))
                for t in range(Nveh):
                    for j in range(Nprop):
                        variability_term = np.random.randn(1, 1, Npass)
                        x_veh[t, j, :] = x_veh[t, j, :] * variability_term
            else:
                x_veh = np.zeros((Nveh, Nprop, Npass))
                
            vel_avg = (vel_max + vel_min) / 2.0
            temp_avg = (temp_max + temp_min) / 2.0
            
            for t in range(Npass):
                if use_speed_variability:
                    # lhs[0, t] corresponds to MATLAB's lhs(1, t)
                    Velocidade[t] = round(vel_min + (vel_max - vel_min) * lhs[0, t]) / 3.6
                else:
                    Velocidade[t] = round(vel_avg) / 3.6
                
                # Temperature Variability
                if use_temp_variability:
                    Temperatura[t] = round(temp_min + (temp_max - temp_min) * lhs[1, t])
                else:
                    Temperatura[t] = 25.0
                    
            for j in range(Npass):
                Train = a01_train(Velocidade[j], x_veh[:, :, j])
                Track = a02_track()
                Beam  = a03_bridge(Beam)
                
                Beam.Prop.E = Beam.Prop.E - Beam.Prop.E * 0.003 * (Temperatura[j] - 15)
                
                Calc, Beam, Track = a04_options(Beam, Track)
                Sol, Calc, Train, Beam, Track = b00_calculations(Calc, Train, Track, Beam, Damage)
                data = d01_data_processing(i, j, Sol, Train, Calc, data)
                
                # --- Validation Plotting ---
                # fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
                # fig.canvas.manager.set_window_title('TTB-2D Validation')
                # fig.patch.set_facecolor('white')

                # # 1. Bridge Mid-span Displacement (Time Domain)
                # # 0-based indexing for the middle node
                # mid_node = int(np.floor(Beam.Mesh.Nodes.Tnum / 2)) 
                # axes[0].plot(Calc.Solver.t, Sol.Beam.U.xt[mid_node, :] * 1000, color='blue', linewidth=1.5, linestyle='-')
                # axes[0].set_title('Bridge Mid-Span Vertical Displacement', fontsize=12, fontname='Arial')
                # axes[0].set_xlabel('Time (s)', fontsize=11, fontname='Arial')
                # axes[0].set_ylabel('Displacement (mm)', fontsize=11, fontname='Arial')
                # axes[0].grid(True, alpha=0.3)
                # axes[0].tick_params(axis='both', labelsize=10)
                # axes[0].autoscale(enable=True, axis='x', tight=True)

                # # 2. Vehicle 1 Body Vertical Acceleration (Spatial Domain)
                # # Extract the first DOF (index 0) of the processed spatial acceleration
                # spatial_accel = data.AceleracaoPrimVag[i, j][0, :] 
                # space_axis = np.linspace(0, data.Posicao[i, j], len(spatial_accel))

                # axes[1].plot(space_axis, spatial_accel, color='red', linewidth=1.5, linestyle='-')
                # axes[1].set_title('Vehicle 1 Body Vertical Acceleration', fontsize=12, fontname='Arial')
                # axes[1].set_xlabel('Distance (m)', fontsize=11, fontname='Arial')
                # axes[1].set_ylabel('Acceleration (m/s^2)', fontsize=11, fontname='Arial')
                # axes[1].grid(True, alpha=0.3)
                # axes[1].tick_params(axis='both', labelsize=10)
                # axes[1].autoscale(enable=True, axis='x', tight=True)

                # plt.tight_layout()
                # plt.show(block=True) # Pauses the loop so you can inspect the plot
                
                # --- Automated Numerical Validation ---
                # mid_node = int(np.floor(Beam.Mesh.Nodes.Tnum / 2))
                # U_mid_py = Sol.Beam.U.xt[mid_node, :]
                # spatial_accel = data.AceleracaoPrimVag[i, j][0, :]
                
                # # Match MATLAB's 1-based naming convention (j + 1)
                # val_filename = f'ValidationData_DC{DC}_Pass{j + 1}.mat'
                
                # try:
                #     mat_baseline = sio.loadmat(val_filename, squeeze_me=True)
                #     U_mid_matlab = mat_baseline['U_mid_matlab']
                #     Acc_veh_matlab = mat_baseline['Acc_veh_matlab']
                    
                #     # Calculate Errors
                #     mse_U = np.mean((U_mid_py - U_mid_matlab)**2)
                #     max_err_U = np.max(np.abs(U_mid_py - U_mid_matlab))
                    
                #     mse_A = np.mean((spatial_accel - Acc_veh_matlab)**2)
                #     max_err_A = np.max(np.abs(spatial_accel - Acc_veh_matlab))
                    
                #     # Print Report
                #     print(f"\n  >> [{val_filename}] VALIDATION REPORT")
                #     print(f"     Bridge Disp - MSE: {mse_U:.4e} | Max Err: {max_err_U:.4e}")
                #     print(f"     Veh 1 Accel - MSE: {mse_A:.4e} | Max Err: {max_err_A:.4e}\n")
                    
                # except FileNotFoundError:
                #     print(f"\n  [!] {val_filename} not found. Skipping validation for this pass.")
                # except ValueError as e:
                #     print(f"\n  [!] Dimension mismatch during validation for {val_filename}: {e}")
                # ------------------------------
                
                print(f'--- DC {DC}: Damage = {Damage.DOF_ChangeRate_variab[i]:.2f}, Pass {j + 1} of {Npass} ---')
                
            data.Temperatura = Temperatura
            data.Velocidade = Velocidade
            data.VehiclesProps = x_veh
            
        save_progress(data, DC, run_folder)

    # Mark end time of the run
    tempo_final = datetime.now()
    with open(os.path.join(run_path, 'tempo_final.pkl'), 'wb') as f:
        pickle.dump(tempo_final, f)

    # Compute total time
    tempo_total = tempo_final - tempo_inicial
    print(f'Tempo total: {tempo_total}')
    
    with open(os.path.join(run_path, 'tempo_total.pkl'), 'wb') as f:
        pickle.dump(tempo_total, f)

if __name__ == "__main__":
    main()