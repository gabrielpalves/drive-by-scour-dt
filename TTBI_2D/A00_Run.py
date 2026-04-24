import os
import re
import math
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.stats import qmc

# Shared modules
from a01_train import a01_train
from a02_track import a02_track
from a03_bridge import a03_bridge
from a04_options import a04_options
from b00_calculations import b00_calculations
from d01_data_processing import d01_data_processing

class EmptyObj:
    """Simple container for dynamic attribute assignment."""
    pass

def save_progress(data, dc_idx, run_folder):
    """Serializes and saves the progress of a damage case (DC)."""
    filename = os.path.join(run_folder, f"{dc_idx:04d}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def manage_run_directories(results_folder):
    """
    Scans the results folder, displays the status of previous runs, 
    and prompts the user to either resume a run or start a new one.
    Returns the path to the active run folder and its start time.
    """
    os.makedirs(results_folder, exist_ok=True)
    date_format = "%d_%m_%Y_%H_%M_%S"
    
    # 1. Gather existing runs
    run_dirs_names = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    run_dirs = []
    
    for name in run_dirs_names:
        try:
            dt_obj = datetime.strptime(name, date_format)
            run_dirs.append({'name': name, 'date': dt_obj})
        except ValueError:
            run_dirs.append({'name': name, 'date': datetime.min})
            
    run_dirs.sort(key=lambda x: x['date'], reverse=True)

    # 2. Display Status Menu
    print('\nRun Status (sorted by date):')
    print('-' * 90)
    print(f"{'#':<3} {'Run Name':<22} {'Status':<12} {'Completion':<20} {'Size(MB)':<10}")
    print('-' * 90)

    for i, run_info in enumerate(run_dirs):
        run_name = run_info['name']
        run_path = os.path.join(results_folder, run_name)
        
        ini_file = os.path.join(run_path, 'tempo_inicial.pkl')
        end_file = os.path.join(run_path, 'tempo_final.pkl')

        status = 'COMPLETE' if (os.path.exists(ini_file) and os.path.exists(end_file)) else \
                 'INCOMPLETE' if os.path.exists(ini_file) else 'UNAVAILABLE'

        # Auto-detect total damage cases completed
        completed_dc = sum(1 for f in os.listdir(run_path) if re.match(r'^\d{4}\.pkl$', f))
        max_idx = max([int(m.group(1)) for f in os.listdir(run_path) if (m := re.match(r'^(\d{4})\.pkl$', f))], default=0)
        
        total_dc = max_idx if max_idx > 0 else float('nan')
        completion_str = '---' if math.isnan(total_dc) else f"{completed_dc}/{total_dc} ({(100.0 * completed_dc / total_dc):5.1f}%)"
        
        size_mb = sum(f.stat().st_size for f in Path(run_path).rglob('*') if f.is_file()) / (1024**2)
        print(f"{i+1:<3} {run_name:<22} {status:<12} {completion_str:<20} {size_mb:8.2f}")

    print('-' * 90 + '\n')

    # 3. User Selection
    print('0 - Start a NEW run')
    for i, run_info in enumerate(run_dirs):
        print(f"{i+1} - Resume: {run_info['name']}")

    try:
        choice = int(input('\nEnter your choice: '))
    except ValueError:
        choice = 0

    # 4. Initialize Folder
    if choice == 0 or not run_dirs:
        tempo_inicial = datetime.now()
        folder_name = tempo_inicial.strftime(date_format)
        print(f'\nStarting NEW run: {folder_name}')
        run_path = os.path.join(results_folder, folder_name)
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, 'tempo_inicial.pkl'), 'wb') as f:
            pickle.dump(tempo_inicial, f)
    else:
        folder_name = run_dirs[choice - 1]['name']
        run_path = os.path.join(results_folder, folder_name)
        print(f'\nResuming run: {folder_name}')
        
        tempo_file = os.path.join(run_path, 'tempo_inicial.pkl')
        if os.path.exists(tempo_file):
            with open(tempo_file, 'rb') as f: tempo_inicial = pickle.load(f)
        else:
            tempo_inicial = datetime.strptime(folder_name, date_format)

    return run_path, tempo_inicial


def generate_dataset(run_folder):
    """
    Executes the TTBI Physics simulation across all damage cases and passages.
    """
    # --- Configuration ---
    use_signal_noise = False
    use_vehicle_variability = False
    use_speed_variability = False
    use_temp_variability = False
    
    Dano = np.linspace(0.00, 0.60, 61)
    Npass = 5
    Nveh = 5
    Nprop = 3
    Desvio = 0.05
    
    temp_min, temp_max = 3, 33
    vel_min, vel_max = 70, 90
    vel_avg, temp_avg = (vel_max + vel_min) / 2.0, (temp_max + temp_min) / 2.0

    # Identify completed cases for fast-resume
    completed = {int(m.group(1)) for f in os.listdir(run_folder) if (m := re.match(r'^(\d{4})\.pkl$', f))}

    # --- Main Simulation Loop ---
    for DC in range(1, len(Dano) + 1):
        if DC in completed:
            print(f'Skipping DC {DC} — result already exists.')
            continue
            
        # 1. Initialize Objects
        Damage, data, Beam, Track = EmptyObj(), EmptyObj(), EmptyObj(), EmptyObj()
        Damage.desvio = Desvio if use_signal_noise else 0.0
        Damage.DOFStiff_ROT_value = 0
        Damage.DOF_ChangeRate_value = Dano[DC - 1]
        
        Beam.Prop = EmptyObj()
        Beam.Prop.E_mod = 1
        Beam.Prop.n_mod = 100
        
        data.Aceleracao = [[None] * Npass]
        data.Velocidade = [[None] * Npass]
        data.Posicao = [[None] * Npass]
        
        # 2. Latin Hypercube Sampling (LHS)
        lhs = qmc.LatinHypercube(d=2).random(n=Npass).T 
        
        Velocidade = np.ones(Npass) * (vel_avg / 3.6)
        Temperatura = np.ones(Npass) * temp_avg
        x_veh = np.zeros((Nveh, Nprop, Npass))
        
        # 3. Apply Variability
        if use_vehicle_variability:
            x_veh = np.ones((Nveh, Nprop, Npass)) * np.random.randn(Nveh, Nprop, Npass)
            
        for t in range(Npass):
            if use_speed_variability:
                Velocidade[t] = round(vel_min + (vel_max - vel_min) * lhs[0, t]) / 3.6
            if use_temp_variability:
                Temperatura[t] = round(temp_min + (temp_max - temp_min) * lhs[1, t])

        # 4. Run Physics for all passages in this Damage Case
        for j in range(Npass):
            Train = a01_train(Velocidade[j], x_veh[:, :, j])
            Track = a02_track()
            Beam  = a03_bridge(Beam)
            
            # Temperature effect on stiffness
            Beam.Prop.E = Beam.Prop.E - Beam.Prop.E * 0.003 * (Temperatura[j] - 15)
            
            # Execute Physics Engine
            Calc, Beam, Track = a04_options(Beam, Track)
            Sol, Calc, Train, Beam, Track = b00_calculations(Calc, Train, Track, Beam, Damage)
            
            # Process Data (i is hardcoded to 0 since DOF_ChangeRate_variab is always length 1)
            data = d01_data_processing(0, j, Sol, Train, Calc, Damage, data)
            
            print(f'--- DC {DC}: Damage = {Damage.DOF_ChangeRate_value:.2f}, Pass {j + 1} of {Npass} ---')
            
        # 5. Store Metadata and Save
        data.Temperatura = Temperatura
        data.Velocidade = Velocidade
        data.VehiclesProps = x_veh
        
        save_progress(data, DC, run_folder)


def main():
    print("\n" + "="*73)
    print("*** TTBI-2D Offline Dataset Generator                              ***")
    print("="*73 + "\n")

    results_folder = 'Results'
    
    # Setup directories and get the active run path
    run_path, tempo_inicial = manage_run_directories(results_folder)
    
    # Generate the data
    generate_dataset(run_path)

    # Finalize the run
    tempo_final = datetime.now()
    with open(os.path.join(run_path, 'tempo_final.pkl'), 'wb') as f:
        pickle.dump(tempo_final, f)
        
    with open(os.path.join(run_path, 'tempo_total.pkl'), 'wb') as f:
        pickle.dump(tempo_final - tempo_inicial, f)
        
    print(f'\nTempo total: {tempo_final - tempo_inicial}')

if __name__ == "__main__":
    main()