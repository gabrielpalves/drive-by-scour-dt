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
% ===== STAGE PRESET — the paper-case selector ============================
% Pick ONE. Each preset maps to a CASE presented in the paper and sets the
% damage-defining toggles below (geometry, which damages are TARGETS vs EOV
% nuisances). Everything else (passage counts, ranges, variability) is a
% stage-independent knob kept at its default further down. This is the single
% place to switch between paper cases.
%
%   'stage0_multiscour' — 2 independent scours, 3-span/60 m, NO other damage.
%                         The localisation gate. == the dataset already run.
%   'stage1_bearing'    — + abutment bearing as a LABELLED TARGET (scour-vs-
%                         bearing disentanglement). Adds data.bearing_vector.
%   'stage1_eov'        — scour target only, but crack + rail-profile turned on
%                         as randomized EOV NUISANCES (confounder-robustness /
%                         "when does scour detection break?" study).
%   'stage1_full'       — bearing target AND crack+profile EOV nuisances.
%   'stage2_4span'      — scale-up: 4-span/100 m, 3 internal piers, bearing
%                         target + EOV nuisances.
%   'stage3_alldamage'  — "kitchen-sink" robustness arm (ablation paper only):
%                         Stage-1 geometry/targets + crack + psd_fra + TRACK-
%                         LAYER damage (ballast patches, hanging sleepers,
%                         rail pads) + wheel OOR/flats. All nuisances, logged.
%                         Spec: docs/stage3_alldamage_spec.md.
STAGE = 'stage1_crack';

% Track-layer / wheel-OOR EOV toggles (only stage3 turns them on)
use_track_eov = false;
use_oor_eov   = false;

switch STAGE
    case 'stage0_multiscour'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='off';    use_crack_eov=false; profile_mode='fixed';
    case 'stage1_bearing'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=false; profile_mode='fixed';
    case 'stage1_crack'
        % BRIDGE-damage EOV stage (Fernandes-comparable: scour + bearing +
        % crack; rail-side EOVs enter at stage1_full). Crack is drawn per
        % STATE (persistent condition; 2026-07-12 EOV design review).
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='fixed';
    case 'stage1_eov'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='off';    use_crack_eov=true;  profile_mode='psd_fra';
    case 'stage1_full'
        % stage1_crack + rail-profile EOV (per-STATE FRA-4 realization +
        % per-passage jitter): the Stage-2 collapse ATTRIBUTION run on the
        % known-good L60 geometry; also gates Stage 3.
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
    case 'stage2_4span'
        % L=99.6 (not 100): spans of 24.9 m = 83 elements of 0.3 m EXACTLY, so
        % all 5 supports land exactly on mesh nodes (B43 mesh / B02 snapping).
        % L=100 would give a 99.9 m mesh with snapped spans 24.9/24.9/25.2/24.9
        % and a floating-point knife-edge tie at the mid support. Verified
        % 2026-07-09; see docs/framework_rationale.md.
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
    case 'stage3_alldamage'
        % Same bridge/targets as Stage 1 -> clean marginal-effect chain
        % (Stage 0 clean -> Stage 1 +bearing -> Stage 3 +everything).
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
        use_track_eov=true;    use_oor_eov=true;
    otherwise
        error('A00: unknown STAGE preset "%s"', STAGE);
end
% Verify the support positions land on mesh nodes for your L/num_spans
% (B07/B43) before a long run. num_spans=N -> supports {1..N+1}; internal
% piers (valid scour targets) are 2..N.

% ===== Operational variability (keep ON to match the training distribution) ===
% NOISE POLICY 2026-07-12 (user decision): generation is NOISE-FREE from
% stage1_crack onward. Measurement noise is injected at LOAD time instead
% (core/dataset.py 'sensor_noise' config), where per-channel levels stay
% configurable per experiment - sensor grade depends on mounting position
% (EN 61373 vibration severity: carbody < bogie < axle; see
% papers/'Confiabilidade Sensores MEMS Ferroviários'). Set true ONLY to
% reproduce the legacy Stage-0/1 pipeline (wheel-only multiplicative noise
% applied in D01; adds 'N' to the folder var_tag).
use_signal_noise = false;       % legacy toggle - keep FALSE (noise at load time)
use_vehicle_variability = true; % Toggle vehicle property variability
use_speed_variability = true;   % Toggle train speed variability
use_temp_variability = true;    % Toggle temperature variability

% ===== Damage states =====
% --- single_scour: continuous scour grid on the central pier (file k -> Dano(k)).
Dano  = linspace(0.0, 0.60, 61);  % used only when damage_mode == 'single_scour'

% --- multi_scour: independent scour at scour_supports (set by the STAGE preset).
n_states_multi   = 250;     % number of independent joint damage states (LHS)
                            % (250 + 9 anchors = 259, matching the Stage-0 run)
dano_max         = 0.60;    % per-support max scour fraction
include_anchors  = true;    % prepend healthy + single-pier sweeps (localisation extremes)
n_anchor_levels  = 4;       % single-pier anchor levels per scoured support
damage_seed      = 1;       % RNG seed for a REPRODUCIBLE damage-state grid

Npass = 50;                 % passages per damage state (operational variability)

% ===== Bearing damage (abutment rotational stiffness, Nm/rad) =====
% MODE is set by the STAGE preset; these are the numeric knobs it uses.
% 'off'    healthy bearings everywhere.
% 'fixed'  the SAME Bearing_Intensity at the LEFT abutment for every state.
% 'target' bearing state [left,right] becomes a LABELLED TARGET — sampled
%          JOINTLY with scour by one LHS and saved as data.bearing_vector.
Bearing_Intensity = 0.0;    % 'fixed' mode value [Nm/rad]
bearing_max       = 1e9;    % 'target' mode upper bound [Nm/rad] (1e9 = seized, Fernandes)

% ===== Confounder damages — EOV nuisance augmentation =====
% use_crack_eov is set by the STAGE preset; these are its numeric knobs.
% Cracks are NOT labels: the network never estimates them. EOV DESIGN REVIEW
% 2026-07-12 (deep research, papers/'Drive-By Scour ML Literature Design'):
% a crack is a PERSISTENT condition -> drawn once per damage STATE and held
% for all its passages (crack_draw='per_state'), with prevalence ~0.25 (a
% p=1.0 per-passage redraw deprives the model of any healthy-deck baseline
% and is physically indefensible). Semantics mirror TTBI_2D/damage_config.py
% (local EI reduction, Sinha et al.).
crack_draw       = 'per_state';   % 'per_state' (persistent damage) | 'per_passage' (DEPRECATED)
crack_p          = 0.25;          % P(state carries a crack); report: 20-30% [VERIFY vs sources]
crack_frac_range = [0.10 0.90];   % crack location as a fraction of L
crack_int_range  = [0.05 0.30];   % EI-loss fraction (Fernandes 2025: 0.14/0.22)
crack_lc         = 0.0;           % half-length [m] each side; <=0 -> single element

% ===== Rail profile mode (EOV) =====
% profile_mode is set by the STAGE preset; these are its numeric knobs.
% 'fixed'        legacy measured profile — byte-identical baseline (Type 2).
% 'fixed_scaled' measured profile x amplitude factor drawn per passage.
% 'psd_fra'      profile REGENERATED from the FRA PSD; severity = FRA class.
% EOV DESIGN REVIEW 2026-07-12: track geometry evolves over MGT, not between
% trains (EN 13848-2 pass-to-pass repeatability <=0.5 mm @95%; Sato/Shenton) ->
% ONE class + ONE phase realization drawn per damage STATE and held for its
% passages (profile_draw='per_state'; B19 seeds the phase draw per state), plus
% a small additive per-passage jitter (metrological repeatability / wander).
% Class set FIXED at FRA class 4 = roughest geometry permissible at 70-90 km/h
% (classes 5/6 are premium track, too smooth for a scour-prone regional line).
% The old per-passage class+phase redraw over {4,5,6} is DEPRECATED (it is what
% collapsed the sprung channels in the L100 Stage-2 pilot).
profile_draw         = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
profile_jitter_sd_mm = 0.5;          % per-passage additive white noise [mm] (EN 13848-2)
profile_int_range    = [0.5 2.0];    % 'fixed_scaled' amplitude range
profile_fra_classes  = 4;            % 'psd_fra' class set (scalar = fixed class)

% ===== Track-layer damage EOVs (Stage 3; use_track_eov) =====
% Verified sampling spec: docs/track_eov_sampling_spec.md (deep-research,
% quote-checked). Randomized NUISANCES — logged, NOT labels. EOV DESIGN REVIEW
% 2026-07-12: ballast patches / hanging sleepers / pad aging are PERSISTENT
% infrastructure conditions (same argument as crack/profile) -> drawn once per
% damage STATE (track_draw='per_state'). Wheel OOR stays per-PASSAGE: each
% passage plausibly is a different train of the fleet (vehicle variability).
% Track x-coords: the bridge occupies [track_L_app, track_L_app+L_bridge].
track_draw        = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
track_L_app       = 30;          % = A04 minL_Approach (fixed, 50 sleepers)
ballast_n_patches = [1 2];       % patches per passage, discrete uniform
ballast_patch_len = [5 20];      % patch length U(5,20) m (cited)
ballast_p_wet     = 0.5;         % P(wet/saturated) vs dry-fouled state
ballast_eta_k_dry = [1.2 2.0]; ballast_eta_c_dry = [0.4 0.8];   % (cited)
ballast_eta_k_wet = [0.7 0.9]; ballast_eta_c_wet = [1.5 4.0];   % (cited)
hang_n_groups     = [1 3];       % hanging-sleeper groups per passage
hang_group_size   = [1 5];       % consecutive sleepers, DU(1,5) (cited)
hang_p_transition = 0.6;         % P(group in a transition zone) [assumption]
hang_trans_margin = 15;          % density-spike zone +-15 m of abutments (cited)
pad_chi_range     = [1.0 3.5];   % pad aging stiffness multiplier bounds
pad_weibull       = [1.8 2.2];   % chi_pad ~ Weibull(lambda,k), clipped to range
pad_beta_range    = [0.8 1.2];   % pad damping multiplier
pad_p_fail        = 0.005;       % per-pad failure prob (~1-yr snapshot)

% ===== Wheel flats + polygonization (Stage 3; use_oor_eov) =====
% LITERATURE-VERIFIED (deep research 2026-07-09; docs/stage3_alldamage_spec.md):
% ~12% of in-service wheels carry a flat; braking couples axles within a bogie.
% Generative model: independent per-bogie slide events (q), leading axle always
% flats, trailing w.p. oor_p_trailing -> wheel marginal 0.12, P(bogie2|bogie1)~q.
% (Left/right 0.85 same-axle correlation is N/A in this 2D single-rail model.)
oor_q_bogie     = 0.171;         % P(slide event) per bogie
oor_p_trailing  = 0.40;          % P(trailing axle also flats | event)
oor_p_fresh     = 0.125;         % P(flat is FRESH | flat) (1.5% vs 10.5% split)
oor_len_fresh   = [0.010 0.035]; % fresh flat length U(10,35) mm (cited)
oor_len_runin   = [0.030 0.060]; % run-in flat length U(30,60) mm (cited)
oor_radius      = 0.46;          % wheel radius R [m]
% depth: FRESH d=L^2/(8R) (chord sagitta) | RUN-IN d=L^2/(16R) (filtered) — cited
% ---- low-order polygonization: separate CONTINUOUS nuisance (cited verdict) ----
poly_p_wheel    = 0.30;          % P(a wheel is polygonized)
poly_orders     = [1 5];         % harmonic order n ~ DU(1,5)
poly_amp_lnorm  = [-10.0 0.5];   % ln(amp [m]) ~ N(mu,sigma) (median ~45 um)
poly_amp_bounds = [1e-5 1.2e-4]; % clip to cited service range 0.01-0.12 mm

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
n_bear = 2 * strcmp(bearing_mode, 'target');  % bearing label dims (left,right)
if strcmp(damage_mode, 'single_scour')
    central = floor(n_supp / 2) + 1;
    DamageStates = zeros(numel(Dano), n_supp);
    DamageStates(:, central) = Dano(:);
    scour_supports = central;                 % the single (central) target
else  % multi_scour
    rng(damage_seed);                         % reproducible joint grid
    n_tgt = numel(scour_supports);
    anchors_s = zeros(1, n_supp);             % healthy (0,...,0)
    anchors_b = zeros(1, 2);
    if include_anchors
        for ti = 1:n_tgt                      % single-pier scour anchors
            levels = linspace(dano_max / n_anchor_levels, dano_max, n_anchor_levels)';
            blk = zeros(numel(levels), n_supp);
            blk(:, scour_supports(ti)) = levels;   % ONE pier damaged, others 0
            anchors_s = [anchors_s; blk]; %#ok<AGROW>
            anchors_b = [anchors_b; zeros(numel(levels), 2)]; %#ok<AGROW>
        end
        if n_bear > 0                         % single-bearing anchors (scour 0)
            for bi = 1:2
                levels = linspace(bearing_max / n_anchor_levels, bearing_max, n_anchor_levels)';
                blk = zeros(numel(levels), 2);
                blk(:, bi) = levels;
                anchors_b = [anchors_b; blk]; %#ok<AGROW>
                anchors_s = [anchors_s; zeros(numel(levels), n_supp)]; %#ok<AGROW>
            end
        end
    end
    % ONE joint LHS over scour + bearing targets = broad joint coverage
    lhs = lhsdesign(n_states_multi, n_tgt + n_bear);
    joint_s = zeros(n_states_multi, n_supp);
    joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;
    joint_b = zeros(n_states_multi, 2);
    if n_bear > 0
        joint_b = lhs(:, n_tgt+1:end) * bearing_max;
    end
    DamageStates      = [anchors_s; joint_s];
    BearingStatesMulti = [anchors_b; joint_b];
end
n_states = size(DamageStates, 1);

% Bearing state per file (n_states x 2 = [left,right]), any damage_mode:
switch bearing_mode
    case 'off',    BearingStates = zeros(n_states, 2);
    case 'fixed',  BearingStates = repmat([Bearing_Intensity, 0.0], n_states, 1);
    case 'target'
        if strcmp(damage_mode, 'single_scour')
            error('A00: bearing_mode=''target'' requires damage_mode=''multi_scour''.');
        end
        BearingStates = BearingStatesMulti;
    otherwise, error('A00: unknown bearing_mode "%s"', bearing_mode);
end

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
% bear_tag keeps the legacy ON/OFF naming so existing run folders still match
% (resume-by-folder-name); new modes/EOVs only ADD tags when actually active.
switch bearing_mode
    case 'off',    bear_tag = 'OFF';
    case 'fixed',  if Bearing_Intensity > 0, bear_tag = 'ON'; else, bear_tag = 'OFF'; end
    case 'target', bear_tag = 'TGT';
end
% EOV tags: the ...ST suffix marks a PER-STATE draw (2026-07-12 design); the
% legacy tags (crackON / prof-psd_fra / trackEOV) meant per-passage redraw, so
% old folders remain distinguishable from new ones by name alone.
eov_tag = '';
if use_crack_eov
    if strcmp(crack_draw, 'per_state'), eov_tag = [eov_tag, '_crackST'];
    else,                               eov_tag = [eov_tag, '_crackON']; end
end
if ~strcmp(profile_mode, 'fixed')
    eov_tag = [eov_tag, '_prof-', profile_mode];
    if strcmp(profile_mode, 'psd_fra') && strcmp(profile_draw, 'per_state')
        eov_tag = [eov_tag, 'ST'];
    end
end
if use_track_eov
    if strcmp(track_draw, 'per_state'), eov_tag = [eov_tag, '_trackEOVST'];
    else,                               eov_tag = [eov_tag, '_trackEOV']; end
end
if use_oor_eov,                    eov_tag = [eov_tag, '_oorON']; end
supp_tag = strjoin(string(scour_supports), '-');
case_name = sprintf('L%g_%dspan_%s_scourS%s_bear%s%s_dano0-%gpct_states%d_Npass%d_var%s', ...
    L_bridge, num_spans, damage_mode, supp_tag, bear_tag, eov_tag, ...
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
    'bearing_mode', bearing_mode, ...
    'bearing_intensity_Nm_rad', Bearing_Intensity, ...
    'bearing_max_Nm_rad', bearing_max, ...
    'use_crack_eov', use_crack_eov, 'crack_draw', crack_draw, ...
    'crack_p', crack_p, ...
    'crack_frac_range', mat2str(crack_frac_range), ...
    'crack_int_range', mat2str(crack_int_range), 'crack_lc', crack_lc, ...
    'profile_mode', profile_mode, 'profile_draw', profile_draw, ...
    'profile_jitter_sd_mm', profile_jitter_sd_mm, ...
    'profile_int_range', mat2str(profile_int_range), ...
    'profile_fra_classes', mat2str(profile_fra_classes), ...
    'use_track_eov', use_track_eov, 'track_draw', track_draw, ...
    'ballast_patch_len', mat2str(ballast_patch_len), ...
    'hang_group_size', mat2str(hang_group_size), ...
    'pad_p_fail', pad_p_fail, ...
    'use_oor_eov', use_oor_eov, ...
    'oor_q_bogie', oor_q_bogie, ...
    'oor_p_trailing', oor_p_trailing, ...
    'oor_p_fresh', oor_p_fresh, ...
    'oor_len_fresh', mat2str(oor_len_fresh), ...
    'oor_len_runin', mat2str(oor_len_runin), ...
    'oor_radius', oor_radius, ...
    'poly_p_wheel', poly_p_wheel, ...
    'poly_orders', mat2str(poly_orders), ...
    'poly_amp_lnorm', mat2str(poly_amp_lnorm), ...
    'use_signal_noise', use_signal_noise, 'noise_std', Desvio, ...
    'use_vehicle_variability', use_vehicle_variability, ...
    'use_speed_variability', use_speed_variability, ...
    'use_temp_variability', use_temp_variability, ...
    'n_vehicles', Nveh, 'n_props_varied', Nprop, ...
    'temp_min_C', temp_min, 'temp_max_C', temp_max, ...
    'vel_min_kmh', vel_min, 'vel_max_kmh', vel_max);
save(fullfile(run_folder, 'case_info.mat'), 'case_info');
% Also store the full damage-state matrices so the dataset is self-describing.
save(fullfile(run_folder, 'damage_states.mat'), 'DamageStates', 'BearingStates', 'scour_supports');
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
parfor DC = 1:n_states
    if completed(DC)
        fprintf('Skipping state %d — result already exists.\n', DC);
        continue;
    end

    % 1. Local initializations for PARFOR transparency
    Damage = struct();
    data = struct();
    data2save = struct();

    % Reproducible per-state RNG stream (speeds, temps, vehicle props and the
    % nuisance-EOV draws below all depend only on damage_seed + state index).
    rng(damage_seed * 100000 + DC);

    % Noise definition
    Damage.desvio = Desvio * use_signal_noise;

    % ---------------------------------------------------------------------
    % 2. CONFIGURE DAMAGE  (per-support scour vector; consumed by B02)
    % ---------------------------------------------------------------------
    scour_vec = DamageStates(DC, :);          % full per-support scour-rate row
    bear_vec  = BearingStates(DC, :);         % [left,right] abutment Nm/rad
    Damage.scour_rates   = scour_vec;
    Damage.bearing_left  = bear_vec(1);
    Damage.bearing_right = bear_vec(2);
    % ---------------------------------------------------------------------

    % Nuisance-EOV logs (saved with the file for traceability; NOT labels).
    % Kept per-passage-shaped even for per-STATE draws (rows then repeat), so
    % downstream loaders read old and new datasets identically.
    CrackLog   = zeros(Npass, 3);             % [loc_m, EI-loss frac, lc_m]
    ProfileLog = ones(Npass, 1);              % intensity or FRA class (mode-dep.)
    TrackLog   = cell(Npass, 1);              % Damage.track struct per passage
    OORLog     = cell(Npass, 1);              % Damage.oor rows per passage

    % Per-STATE EOV state (2026-07-12 design review: persistent conditions -
    % crack, profile realization, track layer - are drawn ONCE per damage
    % state at j_pass==1 and HELD; wheel OOR stays per-passage = a different
    % train of the fleet each passage). Pre-initialised for parfor analysis.
    state_fra_class = profile_fra_classes(1);
    Tk = struct();

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
        Beam_seed.Prop.L = L_bridge;
        Beam_seed.Prop.num_spans = num_spans;
        Beam_local  = A03_Bridge(Beam_seed);

        % Modulus of elasticity temperature adjustment
        Beam_local.Prop.E = Beam_local.Prop.E - Beam_local.Prop.E * 0.003 * (Temperatura(j_pass)-15);

        % --- Nuisance-EOV draws (domain randomization) --------------------
        % PERSISTENT conditions (crack, profile realization, track layer) are
        % drawn once per STATE at j_pass==1 and HELD for all its passages
        % ('per_state'; 2026-07-12 design review). The 'per_passage' branches
        % (redraw every passage) are DEPRECATED - kept only to reproduce
        % legacy datasets such as the L100 Stage-2 pilot.
        % Crack: local EI reduction applied in B00; NOT a label.
        if use_crack_eov && (strcmp(crack_draw, 'per_passage') || j_pass == 1)
            if rand() <= crack_p
                c_loc = (crack_frac_range(1) + diff(crack_frac_range)*rand()) * L_bridge;
                c_int = crack_int_range(1) + diff(crack_int_range)*rand();
                Damage.crack_locs      = c_loc;
                Damage.crack_intensity = c_int;
                Damage.crack_lc        = crack_lc;
            else
                Damage.crack_locs      = [];
                Damage.crack_intensity = [];
                Damage.crack_lc        = 0;
            end
        elseif ~use_crack_eov
            Damage.crack_locs      = [];
            Damage.crack_intensity = [];
            Damage.crack_lc        = 0;
        end   % per_state & j_pass>1: Damage.crack_* persists from j_pass==1
        if ~isempty(Damage.crack_locs)
            CrackLog(j_pass, :) = [Damage.crack_locs, Damage.crack_intensity, ...
                                   Damage.crack_lc];
        end
        % Rail profile: amplitude factor or FRA class + phase realization.
        Profile_cfg = struct('mode', profile_mode);
        if strcmp(profile_mode, 'fixed_scaled')
            Damage.profile_intensity = profile_int_range(1) + diff(profile_int_range)*rand();
            ProfileLog(j_pass) = Damage.profile_intensity;
        elseif strcmp(profile_mode, 'psd_fra')
            if strcmp(profile_draw, 'per_state')
                % ONE class + ONE phase realization per state: class drawn
                % from the state stream at j_pass==1; phases locked via a
                % per-state seed consumed inside B19 (which saves/restores
                % the passage stream); 0.5 mm additive jitter re-drawn per
                % passage in B19 (EN 13848-2 repeatability).
                if j_pass == 1
                    state_fra_class = profile_fra_classes(randi(numel(profile_fra_classes)));
                end
                Profile_cfg.fra_class   = state_fra_class;
                Profile_cfg.phase_seed  = 1e9 + damage_seed*100000 + DC;
                Profile_cfg.jitter_sd_m = profile_jitter_sd_mm / 1000;
            else   % DEPRECATED per-passage class + phase redraw
                Profile_cfg.fra_class = profile_fra_classes(randi(numel(profile_fra_classes)));
            end
            ProfileLog(j_pass) = Profile_cfg.fra_class;
        end
        % Track-layer damage descriptors (consumed by B54; see
        % docs/track_eov_sampling_spec.md). x-coords: bridge at
        % [track_L_app, track_L_app+L_bridge]; abutments = the transitions.
        % Persistent infrastructure condition -> drawn at j_pass==1 and HELD
        % when track_draw='per_state' (the Tk struct persists across the
        % passage loop); 'per_passage' redraw is DEPRECATED.
        if use_track_eov
          if strcmp(track_draw, 'per_passage') || j_pass == 1
            Tk = struct();
            % -- ballast fouling/degradation patches --
            np_ = randi(ballast_n_patches);
            P_ = zeros(np_, 4);
            x_lo = track_L_app - hang_trans_margin;
            x_hi = track_L_app + L_bridge + hang_trans_margin;
            for ip = 1:np_
                plen = ballast_patch_len(1) + diff(ballast_patch_len)*rand();
                x0 = x_lo + (x_hi - x_lo - plen)*rand();
                if rand() < ballast_p_wet
                    ek = ballast_eta_k_wet(1) + diff(ballast_eta_k_wet)*rand();
                    ec = ballast_eta_c_wet(1) + diff(ballast_eta_c_wet)*rand();
                else
                    ek = ballast_eta_k_dry(1) + diff(ballast_eta_k_dry)*rand();
                    ec = ballast_eta_c_dry(1) + diff(ballast_eta_c_dry)*rand();
                end
                P_(ip,:) = [x0, x0 + plen, ek, ec];
            end
            % -- hanging/unsupported sleeper groups --
            ng_ = randi(hang_n_groups);
            H_ = zeros(ng_, 2);
            for ig = 1:ng_
                gsz = randi(hang_group_size);
                if rand() < hang_p_transition   % density spike at transitions
                    trans_x = track_L_app + (rand() < 0.5)*L_bridge;
                    gx = trans_x - hang_trans_margin + 2*hang_trans_margin*rand();
                else                             % anywhere on the bridge
                    gx = track_L_app + rand()*L_bridge;
                end
                H_(ig,:) = [max(gx, 0), gsz];
            end
            % -- rail pads: global aging + sparse failures --
            chi_ = min(max(wblrnd(pad_weibull(1), pad_weibull(2)), ...
                pad_chi_range(1)), pad_chi_range(2));
            beta_ = pad_beta_range(1) + diff(pad_beta_range)*rand();
            win_ = track_L_app + L_bridge + 30;          % app+bridge+after [m]
            nf_ = sum(rand(1, round(win_/0.6)) < pad_p_fail);
            fx_ = sort(rand(1, nf_))*win_;
            Tk.ballast_patches = P_;   Tk.hanging_groups = H_;
            Tk.pad_stiff_mult  = chi_; Tk.pad_damp_mult  = beta_;
            Tk.pad_failures    = fx_;
          end   % per_state & j_pass>1: Tk persists from j_pass==1
            Damage.track = Tk;
            TrackLog{j_pass} = Tk;
        else
            Damage.track = [];
        end
        % Wheel flats + polygonization: literature-anchored draws (consumed by
        % B25). Wheels 1-2 = leading bogie (axles 1,2); 3-4 = trailing bogie.
        if use_oor_eov
            Fl_ = zeros(0, 5);          % [veh wheel length_m depth_m phase]
            Po_ = zeros(0, 5);          % [veh wheel order amp_m phase]
            for v_ = 1:Nveh
                for b_ = 0:1            % per-bogie slide events
                    if rand() < oor_q_bogie
                        ax_ = 2*b_ + 1;                    % leading axle flats
                        if rand() < oor_p_trailing
                            ax_(end+1) = 2*b_ + 2;         %#ok<AGROW> trailing too
                        end
                        for w_ = ax_
                            if rand() < oor_p_fresh        % fresh: sharp chord
                                lf_ = oor_len_fresh(1) + diff(oor_len_fresh)*rand();
                                df_ = lf_^2/(8*oor_radius);
                            else                            % run-in: filtered
                                lf_ = oor_len_runin(1) + diff(oor_len_runin)*rand();
                                df_ = lf_^2/(16*oor_radius);
                            end
                            Fl_(end+1,:) = [v_, w_, lf_, df_, 2*pi*rand()]; %#ok<AGROW>
                        end
                    end
                end
                for w_ = 1:4            % polygonization: independent per wheel
                    if rand() < poly_p_wheel
                        n_ = randi(poly_orders);
                        a_ = exp(poly_amp_lnorm(1) + poly_amp_lnorm(2)*randn());
                        a_ = min(max(a_, poly_amp_bounds(1)), poly_amp_bounds(2));
                        Po_(end+1,:) = [v_, w_, n_, a_, 2*pi*rand()]; %#ok<AGROW>
                    end
                end
            end
            Damage.oor_flats  = Fl_;
            Damage.oor_poly   = Po_;
            Damage.oor_radius = oor_radius;
            OORLog{j_pass} = struct('flats', Fl_, 'poly', Po_);
        else
            Damage.oor_flats = []; Damage.oor_poly = [];
        end
        % ------------------------------------------------------------------

        % Processing and Calculations
        [Calc_local, Beam_local, Track_local] = A04_Options(Beam_local, Track_local, Profile_cfg);
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
    % Bearing label ([left,right] Nm/rad; zeros unless bearing_mode='target'/'fixed')
    data2save.bearing_vector = bear_vec;
    % --- NUISANCE-EOV TRACEABILITY (not labels) ---
    data2save.crack_log    = CrackLog;      % per passage: [loc_m, EI-loss, lc_m]
    data2save.profile_mode = profile_mode;
    data2save.profile_log  = ProfileLog;    % per passage: intensity or FRA class
    data2save.track_log    = TrackLog;      % per passage: Damage.track struct ([] if off)
    data2save.oor_log      = OORLog;        % per passage: [veh wheel len_m phase] rows
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
