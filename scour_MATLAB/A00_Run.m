% Main script to run TTB-2D model

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *************************************************************************
clear; clc; close all
% process_results;

% =========================================================================
% Simulation setup
% =========================================================================
% ===== Bridge geometry — MUST match the trained classifier =====
% The drive-by champion was trained on the ablation bridge (40 m, 2 spans,
% 3 supports), scour-only. The held-out DT dataset must use the SAME geometry,
% or the classifier will mispredict. For a future multi-damage dataset, set
% L_bridge = 100, num_spans = 4 (5 supports).
L_bridge  = 40;     % [m]
num_spans = 2;      % 2 spans = 3 supports

% ===== Operational variability (keep ON to match the training distribution) ===
use_signal_noise = true;        % Toggle artificial signal noise
use_vehicle_variability = true; % Toggle vehicle property variability
use_speed_variability = true;   % Toggle train speed variability
use_temp_variability = true;    % Toggle temperature variability

% ===== Held-out DT observation dataset =====
% Continuous scour states over the FULL 0-60% range (more realistic for the DT
% than the integer training grid), with only a FEW passages per state — this is
% a sampling pool for the online twin, not a training set. File k -> Dano(k).
Dano = linspace(0.0, 0.60, 91);   % ~0.67% steps
Npass = 15;                        % passages per state (a few)

% ===== Bearing damage intensity (0 = healthy) =====
% Champion is scour-only -> keep 0 for the single-scour held-out set. Use
% 1e9 Nm/rad (Fernandes et al.) for a multi-damage dataset.
Bearing_Intensity = 0.0;

Nveh = 5;     % number of vehicles
Nprop = 3;    % how many vehicle properties will be varied
Desvio = 0.05; % standard deviation of the noise of the signal
temp_min = 3; temp_max = 33; % min and max temperature [°C]
vel_min = 70; vel_max = 90; % min and max velocity [km/h]

tempo_inicial = datetime('now');
tempo_inicial_str = datestr(tempo_inicial, 'dd_MM_yyyy_HH_mm_ss');
run_folder = fullfile('Results', tempo_inicial_str);
if ~exist(run_folder, 'dir'), mkdir(run_folder); end

% Identify which DCs have already been processed (for resume mode)
saved_files = dir(fullfile(run_folder, '*.mat'));
completed = false(length(Dano), 1);

for k = 1:length(saved_files)
    tokens = regexp(saved_files(k).name, '(\d{4})\.mat', 'tokens');
    if ~isempty(tokens)
        dc_idx = str2double(tokens{1}{1});
        if dc_idx >= 1 && dc_idx <= length(Dano)
            completed(dc_idx) = true;
        end
    end
end

% =========================================================================
%  Parallel loop (PARFOR)
% =========================================================================
parfor DC = 1:length(Dano)
    if completed(DC)
        fprintf('Skipping DC %d — result already exists.\n', DC);
        continue;
    end

    % 1. Local initializations for PARFOR transparency
    Damage = struct();
    data = struct();
    data2save = struct();

    % Noise definition
    Damage.desvio = Desvio * use_signal_noise;

    % ---------------------------------------------------------------------
    % 2. CONFIGURE DAMAGE  (per-support; consumed by B02_BoundaryConditions)
    % ---------------------------------------------------------------------
    % Scour Dano(DC) on the central pier of an (num_spans+1)-support bridge;
    % all other supports healthy. For num_spans=2 this is [0, Dano, 0], which
    % matches the ablation. Bearing at the entrance abutment (0 = healthy).
    n_supp  = num_spans + 1;
    central = floor(n_supp / 2) + 1;          % middle support index
    Damage.scour_rates          = zeros(1, n_supp);
    Damage.scour_rates(central) = Dano(DC);
    Damage.bearing_left  = Bearing_Intensity; % 0 = healthy (scour-only set)
    Damage.bearing_right = 0.0;
    % ---------------------------------------------------------------------

    % Pre-allocate LHS for this specific damage case
    lhs_matrix = lhsdesign(2, Npass);

    % Variabilidade operacional - velocidade do trem
    Velocidade = ones(1, Npass) * 80/3.6;
    Temperatura = ones(1, Npass) * 25;

    % Vehicle Variability
    if use_vehicle_variability
        x_veh = ones(Nveh, Nprop, Npass);
        for t = 1:Nveh
            for j = 1:Nprop
                variability_term = randn(1, 1, Npass);
                x_veh(t, j, :) = x_veh(t, j, :) .* variability_term;
            end
        end
    else
        x_veh = zeros(Nveh, Nprop, Npass);
    end

    vel_avg = (vel_max + vel_min) / 2;
    temp_avg = (temp_max + temp_min) / 2;
    
    for t = 1:Npass
        if use_speed_variability
            Velocidade(t) = (round(vel_min + (vel_max-vel_min)*lhs_matrix(1,t)))/3.6;
        else
            Velocidade(t) = round(vel_avg)/3.6;
        end

        % Temperature Variability
        if use_temp_variability
            Temperatura(t) = round(temp_min + (temp_max - temp_min)*lhs_matrix(2,t));
        else
            Temperatura(t) = 25;
        end
    end

    % Inner loop for each passage
    for j_pass = 1:Npass
        % Create localized struct versions to avoid parfor conflicts
        Train_local = A01_Train(Velocidade(j_pass), x_veh(:, :, j_pass));
        Track_local = A02_Track();
        Beam_seed = struct();                      % drive geometry from the config above
        Beam_seed.Prop.L = L_bridge;
        Beam_seed.Prop.num_spans = num_spans;
        Beam_local  = A03_Bridge(Beam_seed);
        
        % Modulus of elasticity temperature adjustment
        Beam_local.Prop.E = Beam_local.Prop.E - Beam_local.Prop.E * 0.003 * (Temperatura(j_pass)-15);
        
        % Processing and Calculations
        [Calc_local, Beam_local, Track_local] = A04_Options(Beam_local, Track_local);
        [Sol_local, Calc_local, Train_local, Beam_local, Track_local] = B00_Calculations(Calc_local, Train_local, Track_local, Beam_local, Damage);
        
        % Data Processing
        data = D01_DataProcessing(1, j_pass, Sol_local, Train_local, Calc_local, Damage, data);

        fprintf('--- DC %d: Scour = %.2f, Pass %d of %d ---\n', DC, Dano(DC), j_pass, Npass);
    end
    
    % Store final data for this damage case
    data.Temperatura = Temperatura;
    data.Velocidade = Velocidade;
    data.VehiclesProps = x_veh;

    data2save.AcelPrimVag = data.AceleracaoPrimVag;
    data2save.AcelRodaPrimVag = data.AcelRodaPrimVag;
    data2save.PitchPrimVag = data.PitchPrimVag;
    data2save.Dano = Dano(DC);
    data2save.Temperatura = Temperatura;
    data2save.Velocidade = Velocidade;
    data2save.VehiclesProps = x_veh;

    save_progress(data2save, DC, run_folder);
end

% Mark end time of the run
tempo_final = datetime('now', 'Format', 'dd_MM_yyyy_HH_mm_ss');

% Save tempo_final inside the same run folder
save(fullfile(run_folder, 'tempo_final.mat'), 'tempo_final');

% Compute total time
tempo_total = tempo_final - tempo_inicial;
fprintf('Tempo total: %s\n', char(tempo_total));

save(fullfile(run_folder, 'tempo_total.mat'), 'tempo_total');