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
% ===== Damage scenario mode =====
% 'single_scour'  legacy: scour swept on ONE (central) pier; label = scalar Dano.
%                 Reproduces the ablation / held-out single-scour datasets.
% 'multi_scour'   Stage 0 (multi-damage): INDEPENDENT scour at several piers;
%                 label = the per-support scour VECTOR (data.scour_vector).
%                 The classifier must localise (which pier) AND quantify (how much).
damage_mode = 'multi_scour';

% ===== Bridge geometry =====
% single_scour champion was trained on 40 m / 2-span / 3-support. For multi_scour
% Stage 0 we need >=2 INTERNAL piers -> a 3-span (4-support) bridge. The scoured
% supports are chosen by scour_supports below (must be internal piers, i.e. NOT
% the first/last support). Verify the support positions land on mesh nodes for
% your L/num_spans (B07/B43) before a long run.
if strcmp(damage_mode, 'single_scour')
    L_bridge  = 40;     num_spans = 2;      % 3 supports
else
    L_bridge  = 60;     num_spans = 3;      % 4 supports -> internal piers 2 and 3
end

% ===== Operational variability (keep ON to match the training distribution) ===
use_signal_noise = true;        % Toggle artificial signal noise
use_vehicle_variability = true; % Toggle vehicle property variability
use_speed_variability = true;   % Toggle train speed variability
use_temp_variability = true;    % Toggle temperature variability

% ===== Damage states =====
% --- single_scour: continuous scour grid on the central pier (file k -> Dano(k)).
Dano  = linspace(0.0, 0.60, 61);  % used only when damage_mode == 'single_scour'

% --- multi_scour (Stage 0): independent scour at the chosen supports.
scour_supports   = [2 3];   % 1-based support indices that carry INDEPENDENT scour
                            % (for num_spans=3 -> supports {1,2,3,4}; [2 3] = the
                            % two internal piers). Extend for Stage 2 (e.g. [2 3 4]).
n_states_multi   = 200;     % number of independent joint damage states (LHS)
dano_max         = 0.60;    % per-support max scour fraction
include_anchors  = true;    % prepend healthy + single-pier sweeps (localisation extremes)
n_anchor_levels  = 4;       % single-pier anchor levels per scoured support
damage_seed      = 1;       % RNG seed for a REPRODUCIBLE damage-state grid

Npass = 50;                 % passages per damage state (operational variability)

% ===== Bearing damage intensity (0 = healthy) =====
% Stage 0 is scour-only -> keep 0. Stage 1 adds bearing at the abutments
% (1e9 Nm/rad, Fernandes et al.) as an extra, separately-labelled target.
Bearing_Intensity = 0.0;

Nveh = 5;     % number of vehicles
Nprop = 3;    % how many vehicle properties will be varied
Desvio = 0.05; % standard deviation of the noise of the signal
temp_min = 3; temp_max = 33; % min and max temperature [°C]
vel_min = 70; vel_max = 90; % min and max velocity [km/h]

% =========================================================================
%  Build the damage-state matrix  (n_states x n_supports), one row per file
% =========================================================================
% Every mode produces a DamageStates matrix whose row DC is the full per-support
% scour-rate vector for file DC. This unifies the parfor body and the labelling:
% the saved label is always data.scour_vector = DamageStates(DC,:).
n_supp = num_spans + 1;
if strcmp(damage_mode, 'single_scour')
    central = floor(n_supp / 2) + 1;
    DamageStates = zeros(numel(Dano), n_supp);
    DamageStates(:, central) = Dano(:);
    scour_supports = central;                 % the single (central) target
else  % multi_scour
    rng(damage_seed);                         % reproducible joint grid
    n_tgt = numel(scour_supports);
    anchors = zeros(1, n_supp);               % healthy (0,...,0)
    if include_anchors
        for ti = 1:n_tgt
            levels = linspace(dano_max / n_anchor_levels, dano_max, n_anchor_levels)';
            blk = zeros(numel(levels), n_supp);
            blk(:, scour_supports(ti)) = levels;   % ONE pier damaged, others 0
            anchors = [anchors; blk]; %#ok<AGROW>
        end
    end
    joint = zeros(n_states_multi, n_supp);
    joint(:, scour_supports) = lhsdesign(n_states_multi, n_tgt) * dano_max;  % independent LHS
    DamageStates = [anchors; joint];
end
n_states = size(DamageStates, 1);

tempo_inicial = datetime('now');
tempo_inicial_str = datestr(tempo_inicial, 'dd_MM_yyyy_HH_mm_ss');

% =========================================================================
%  Self-identifying run folder + case manifest
% =========================================================================
% The output folder is named after the CASE (bridge / damage / variability), not
% a timestamp: different cases never share a folder, and re-running the SAME case
% resumes into it (the parfor skips states whose NNNN.mat already exists). Move
% the finished folder under an 'observations/' tree and point the loader at it.
var_tag = [repmat('N',1,double(use_signal_noise)), ...
           repmat('V',1,double(use_vehicle_variability)), ...
           repmat('S',1,double(use_speed_variability)), ...
           repmat('T',1,double(use_temp_variability))];
if isempty(var_tag), var_tag = 'none'; end
if Bearing_Intensity > 0, bear_tag = 'ON'; else, bear_tag = 'OFF'; end
supp_tag = strjoin(string(scour_supports), '-');
case_name = sprintf('L%g_%dspan_%s_scourS%s_bear%s_dano0-%gpct_states%d_Npass%d_var%s', ...
    L_bridge, num_spans, damage_mode, supp_tag, bear_tag, ...
    dano_max*100, n_states, Npass, var_tag);
run_folder = fullfile('Results', case_name);
if ~exist(run_folder, 'dir'), mkdir(run_folder); end

% --- Manifest: machine-readable (.mat) + human-readable (.txt) -----------
case_info = struct( ...
    'case_name', case_name, 'timestamp', tempo_inicial_str, ...
    'damage_mode', damage_mode, ...
    'L_bridge_m', L_bridge, 'num_spans', num_spans, ...
    'num_supports', n_supp, 'scour_supports', mat2str(scour_supports), ...
    'n_states', n_states, 'passages_per_state', Npass, ...
    'scour_dano_max_frac', dano_max, ...
    'bearing_intensity_Nm_rad', Bearing_Intensity, ...
    'use_signal_noise', use_signal_noise, 'noise_std', Desvio, ...
    'use_vehicle_variability', use_vehicle_variability, ...
    'use_speed_variability', use_speed_variability, ...
    'use_temp_variability', use_temp_variability, ...
    'n_vehicles', Nveh, 'n_props_varied', Nprop, ...
    'temp_min_C', temp_min, 'temp_max_C', temp_max, ...
    'vel_min_kmh', vel_min, 'vel_max_kmh', vel_max);
save(fullfile(run_folder, 'case_info.mat'), 'case_info');
% Also store the full damage-state matrix so the dataset is self-describing.
save(fullfile(run_folder, 'damage_states.mat'), 'DamageStates', 'scour_supports');
fid = fopen(fullfile(run_folder, 'case_info.txt'), 'w');
fn = fieldnames(case_info);
fprintf(fid, '%% TTBI dataset — case manifest\n');
for i = 1:numel(fn)
    v = case_info.(fn{i});
    if ischar(v), fprintf(fid, '%-26s : %s\n', fn{i}, v);
    else,         fprintf(fid, '%-26s : %g\n', fn{i}, v); end
end
fclose(fid);
fprintf('Run folder: %s  (%d states x %d passages)\n', run_folder, n_states, Npass);

% Identify which states have already been processed (for resume mode)
saved_files = dir(fullfile(run_folder, '*.mat'));
completed = false(n_states, 1);
for k = 1:length(saved_files)
    tokens = regexp(saved_files(k).name, '^(\d{4})\.mat$', 'tokens');
    if ~isempty(tokens)
        dc_idx = str2double(tokens{1}{1});
        if dc_idx >= 1 && dc_idx <= n_states
            completed(dc_idx) = true;
        end
    end
end

% =========================================================================
%  Parallel loop (PARFOR)  — one file per damage state
% =========================================================================
for DC = 1:n_states
    if completed(DC)
        fprintf('Skipping state %d — result already exists.\n', DC);
        continue;
    end

    % 1. Local initializations for PARFOR transparency
    Damage = struct();
    data = struct();
    data2save = struct();

    % Noise definition
    Damage.desvio = Desvio * use_signal_noise;

    % ---------------------------------------------------------------------
    % 2. CONFIGURE DAMAGE  (per-support scour vector; consumed by B02)
    % ---------------------------------------------------------------------
    scour_vec = DamageStates(DC, :);          % full per-support scour-rate row
    Damage.scour_rates   = scour_vec;
    Damage.bearing_left  = Bearing_Intensity; % 0 = healthy (scour-only set)
    Damage.bearing_right = 0.0;
    % ---------------------------------------------------------------------

    % Pre-allocate LHS for this specific damage case (speed/temperature)
    lhs_matrix = lhsdesign(2, Npass);

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
    for t = 1:Npass
        if use_speed_variability
            Velocidade(t) = (round(vel_min + (vel_max-vel_min)*lhs_matrix(1,t)))/3.6;
        else
            Velocidade(t) = round(vel_avg)/3.6;
        end
        if use_temp_variability
            Temperatura(t) = round(temp_min + (temp_max - temp_min)*lhs_matrix(2,t));
        else
            Temperatura(t) = 25;
        end
    end

    % Inner loop for each passage
    for j_pass = 1:Npass
        Train_local = A01_Train(Velocidade(j_pass), x_veh(:, :, j_pass));
        Track_local = A02_Track();
        Beam_seed = struct();                      % drive geometry from the config above
        
        % Placeholder for crack
        Beam_seed.Prop.n_mod = 100;
        Beam_seed.Prop.E_mod = 1;

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

        fprintf('--- State %d/%d: scour=[%s], Pass %d/%d ---\n', ...
            DC, n_states, num2str(scour_vec, '%.2f '), j_pass, Npass);
    end

    % Store final data for this damage case
    data.Temperatura = Temperatura;
    data.Velocidade = Velocidade;
    data.VehiclesProps = x_veh;

    data2save.AcelPrimVag = data.AceleracaoPrimVag;
    data2save.AcelRodaPrimVag = data.AcelRodaPrimVag;
    data2save.PitchPrimVag = data.PitchPrimVag;
    % --- LABELS ---
    % Multi-output label = the full per-support scour-rate vector + which support
    % indices are the regression targets. Dano kept as the scalar AGGREGATE
    % (max scour) for backward compatibility with the single-scour loaders / DT.
    data2save.scour_vector = scour_vec;
    data2save.scour_supports = scour_supports;
    data2save.Dano = max(scour_vec);
    data2save.Temperatura = Temperatura;
    data2save.Velocidade = Velocidade;
    data2save.VehiclesProps = x_veh;

    save_progress(data2save, DC, run_folder);
end

% Mark end time of the run
tempo_final = datetime('now', 'Format', 'dd_MM_yyyy_HH_mm_ss');
save(fullfile(run_folder, 'tempo_final.mat'), 'tempo_final');
tempo_total = tempo_final - tempo_inicial;
fprintf('Tempo total: %s\n', char(tempo_total));
save(fullfile(run_folder, 'tempo_total.mat'), 'tempo_total');
