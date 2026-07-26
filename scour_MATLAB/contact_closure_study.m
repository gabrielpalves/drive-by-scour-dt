function report = contact_closure_study(dataset_dir, state_index, passage_selector, varargin)
%CONTACT_CLOSURE_STUDY Time-step refinement for the bilateral-contact audit.
%
% report = contact_closure_study(DATASET_DIR, STATE_INDEX, PASSAGE_SELECTOR)
% re-runs one saved MATLAB state/passage at nominal time steps
% [1, 0.5, 0.25] ms. PASSAGE_SELECTOR is a positive integer or "worst",
% which selects the row with the largest saved contact_log(:,4).
%
% The saved state is the source of every passage descriptor: operational
% variables, scour/bearing labels, crack, rail profile, track EOVs and wheel
% OOR. Nothing is re-sampled. The production contact gate is not read or
% modified; 0/12/24 kN are evaluated post hoc as DIAGNOSTIC classifications.
%
% Name-value options:
%   DtMs          positive vector, default [1 0.5 0.25]
%   GatesN        nonnegative vector, default [0 12000 24000]
%   FractionGate  nonnegative scalar, default 0.002
%   CommonDxM     common spatial comparison step, default 0.01 m
%   DamageSeed    profile phase-seed root; default case_info.damage_seed
%                 when present, otherwise the A00 campaign default 1
%   VerifyIntegrity require completion marker + per-file manifest, default true
%   DryRun        validate/describe the case without invoking the solver
%   OutputDir     optional report destination; default "" writes nothing
%   Overwrite     allow replacement of an existing report, default false
%
% When OutputDir is supplied, a small .md appendix report and a .mat record
% are written. OutputDir must not be the source dataset directory.
%
% Example (after the regenerated datasets are available):
%   contact_closure_study("D:\runs\s23_all4_L99.6_st308", 24, "worst", ...
%       "OutputDir", "D:\verification\contact");
%   contact_closure_study("D:\runs\s15_track_L60_st308", 244, "worst", ...
%       "OutputDir", "D:\verification\contact");

arguments
    dataset_dir {mustBeTextScalar}
    state_index (1,1) double {mustBeInteger,mustBePositive}
    passage_selector = "worst"
end
arguments (Repeating)
    varargin
end

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'DtMs', [1, 0.5, 0.25], @local_positive_vector);
addParameter(parser, 'GatesN', [0, 12000, 24000], @local_nonnegative_vector);
addParameter(parser, 'FractionGate', 0.002, @local_nonnegative_scalar);
addParameter(parser, 'CommonDxM', 0.01, @local_positive_scalar);
addParameter(parser, 'DamageSeed', NaN, @local_nan_or_nonnegative_integer);
addParameter(parser, 'VerifyIntegrity', true, @local_logical_scalar);
addParameter(parser, 'DryRun', false, @local_logical_scalar);
addParameter(parser, 'OutputDir', "", @local_text_scalar);
addParameter(parser, 'Overwrite', false, @local_logical_scalar);
parse(parser, varargin{:});
opt = parser.Results;

dataset_dir = char(dataset_dir);
if ~isfolder(dataset_dir)
    error('contact_closure:MissingDataset', ...
        'Dataset directory does not exist: %s', dataset_dir);
end
dataset_dir = local_absolute_path(dataset_dir);

case_path = fullfile(dataset_dir, 'case_info.mat');
state_path = fullfile(dataset_dir, sprintf('%04d.mat', state_index));
if ~isfile(case_path)
    error('contact_closure:MissingCaseInfo', ...
        'Missing case_info.mat in %s.', dataset_dir);
end
if ~isfile(state_path)
    error('contact_closure:MissingState', ...
        ['Missing %04d.mat in %s. The closure harness needs the completed ' ...
         'state file because it reuses its exact passage descriptors; it ' ...
         'never re-samples an aborted state.'], state_index, dataset_dir);
end

case_blob = load(case_path);
if ~isfield(case_blob, 'case_info') || ~isstruct(case_blob.case_info)
    error('contact_closure:BadCaseInfo', ...
        'case_info.mat does not contain a case_info struct.');
end
case_info = case_blob.case_info;
state_blob = load(state_path);
if isfield(state_blob, 'data') && isstruct(state_blob.data)
    state_data = state_blob.data;
else
    state_data = state_blob;
end

local_validate_case(case_info, state_data, state_index);
n_passages = numel(state_data.Velocidade);
passage_index = local_select_passage( ...
    passage_selector, state_data.contact_log, n_passages);

dt_ms = unique(double(opt.DtMs(:)'), 'stable');
gates_n = unique(double(opt.GatesN(:)'), 'stable');
fraction_gate = double(opt.FractionGate);
common_dx_m = double(opt.CommonDxM);
damage_seed = double(opt.DamageSeed);
if isnan(damage_seed)
    if isfield(case_info, 'damage_seed')
        damage_seed = double(case_info.damage_seed);
    else
        damage_seed = 1;
    end
end

saved_contact = double(state_data.contact_log(passage_index, 1:4));
saved_tension_n = max(0, saved_contact(4));
saved_gate_pass = local_gate_pass( ...
    saved_tension_n, saved_contact(3), gates_n, fraction_gate);

report = struct();
report.study_schema = 'contact-closure-v1';
report.created_utc = char(datetime('now', ...
    'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
report.matlab_release = version('-release');
report.dataset_dir = dataset_dir;
report.state_file = state_path;
report.state_file_sha256 = local_file_sha256(state_path);
report.dataset_integrity = local_verify_dataset_integrity( ...
    dataset_dir, state_path, report.state_file_sha256, case_info, ...
    state_data, state_index, logical(opt.VerifyIntegrity));
report.stage = local_case_text(case_info, 'stage', 'unknown');
report.case_name = local_case_text(case_info, 'case_name', 'unknown');
report.gen_schema = local_case_text(case_info, 'gen_schema', 'unknown');
report.gen_fingerprint = local_case_text( ...
    case_info, 'gen_fingerprint', 'unknown');
report.state_index = state_index;
report.passage_index = passage_index;
report.passage_selector = char(string(passage_selector));
report.damage_seed = damage_seed;
report.dt_requested_ms = dt_ms;
report.gates_n = gates_n;
report.fraction_gate = fraction_gate;
report.common_dx_m = common_dx_m;
report.saved_contact_log = saved_contact;
report.saved_gate_pass = saved_gate_pass;
report.harness_sha256 = local_file_sha256(which(mfilename));
report.b66_sha256 = local_file_sha256(which('B66_ContactForce'));
report.solver_source_sha256 = local_solver_source_manifest();
report.dry_run = logical(opt.DryRun);
report.numeric_hash_selfcheck = local_numeric_sha256([0; 1]);

if opt.DryRun
    % Exercise the late-stage long-table transformation even when no target
    % dataset is locally available. This guards against discovering a shape
    % error only after an expensive solver run.
    selfcheck_table = local_channel_metric_table([1, 0.5], {'a'; 'b'}, ...
        zeros(2), zeros(2), ones(2));
    assert(height(selfcheck_table) == 4, ...
        'contact_closure:InternalSelfCheck', ...
        'Internal channel-table self-check failed.');
    report.status = 'DRY_RUN_VALIDATED';
    report.descriptor = local_descriptor_summary( ...
        case_info, state_data, state_index, passage_index, damage_seed);
    report.run_table = table();
    report.channel_table = table();
    report.saved_baseline_table = table();
    local_maybe_write_report(report, opt.OutputDir, logical(opt.Overwrite));
    return
end

rng_before = rng;
rng_cleanup = onCleanup(@() rng(rng_before));
n_dt = numel(dt_ms);
run_cells = cell(1, n_dt);
for k = 1:n_dt
    run_cells{k} = local_run_one(case_info, state_data, state_index, ...
        passage_index, damage_seed, dt_ms(k));
end
runs = [run_cells{:}];

% The finest requested step is the numerical reference. Signals are compared
% on the production-style spatial window (10 m skip + bridge + 18.31 m) at a
% common, explicitly recorded dx. This avoids comparing unequal time grids.
[~, ref_idx] = min([runs.actual_dt_ms]);
x_lo = 10;
x_hi_requested = 10 + double(case_info.L_bridge_m) + 18.31;
x_hi_available = min(arrayfun(@(r) r.x_rel_m(end), runs));
x_hi = min(x_hi_requested, x_hi_available);
if x_hi <= x_lo
    error('contact_closure:ShortSignal', ...
        'Passage ends at %.3g m, before the comparison window.', x_hi);
end
x_common = x_lo:common_dx_m:x_hi;
n_channels = size(runs(1).signal, 1);
signal_common = zeros(n_channels, numel(x_common), n_dt);
for k = 1:n_dt
    signal_common(:, :, k) = interp1(runs(k).x_rel_m, ...
        runs(k).signal', x_common, 'linear')';
end

ref_signal = signal_common(:, :, ref_idx);
nrmse = zeros(n_dt, n_channels);
nmax = zeros(n_dt, n_channels);
corrcoef_ch = zeros(n_dt, n_channels);
for k = 1:n_dt
    delta = signal_common(:, :, k) - ref_signal;
    for ch = 1:n_channels
        ref_rms = sqrt(mean(ref_signal(ch, :).^2));
        ref_peak = max(abs(ref_signal(ch, :)));
        nrmse(k, ch) = sqrt(mean(delta(ch, :).^2)) / max(ref_rms, eps);
        nmax(k, ch) = max(abs(delta(ch, :))) / max(ref_peak, eps);
        if std(signal_common(ch, :, k)) == 0 || std(ref_signal(ch, :)) == 0
            corrcoef_ch(k, ch) = double(all(delta(ch, :) == 0));
        else
            c_ = corrcoef(signal_common(ch, :, k), ref_signal(ch, :));
            corrcoef_ch(k, ch) = c_(1, 2);
        end
    end
    runs(k).signal_common_sha256 = local_numeric_sha256( ...
        signal_common(:, :, k));
end

peak_signed_n = [runs.contact_peak_signed_n]';
peak_tension_n = max(0, peak_signed_n);
tension_fraction = [runs.tension_fraction]';
contact_lost_track = logical([runs.contact_lost_track]');
gate_pass = false(n_dt, numel(gates_n));
for k = 1:n_dt
    gate_pass(k, :) = local_gate_pass(peak_tension_n(k), ...
        tension_fraction(k), gates_n, fraction_gate);
end

run_table = table(dt_ms(:), [runs.actual_dt_ms]', [runs.n_samples]', ...
    peak_signed_n, peak_tension_n, tension_fraction, contact_lost_track, ...
    'VariableNames', {'requested_dt_ms', 'actual_dt_ms', 'n_samples', ...
    'peak_contact_signed_N', 'peak_tension_N', 'tension_fraction', ...
    'contact_lost_track'});
for g = 1:numel(gates_n)
    gate_name = matlab.lang.makeValidName( ...
        sprintf('pass_gate_%g_N', gates_n(g)));
    run_table.(gate_name) = gate_pass(:, g);
end

channel_names = runs(1).channel_names(:);
channel_table = local_channel_metric_table( ...
    dt_ms, channel_names, nrmse, nmax, corrcoef_ch);

baseline_idx = local_nearest_dt_index(runs, 1);
[saved_table, saved_note] = local_saved_baseline_comparison( ...
    state_data, passage_index, x_common, ...
    signal_common(:, :, baseline_idx), channel_names, ...
    runs(baseline_idx).signal, runs(baseline_idx).x_rel_m);

report.status = 'COMPLETED';
report.reference_dt_ms = runs(ref_idx).actual_dt_ms;
report.comparison_window_m = [x_common(1), x_common(end)];
report.descriptor = local_descriptor_summary( ...
    case_info, state_data, state_index, passage_index, damage_seed);
report.run_table = run_table;
report.channel_table = channel_table;
report.saved_baseline_table = saved_table;
report.saved_baseline_note = saved_note;
report.signal_common_sha256 = {runs.signal_common_sha256};
report.contact_peak_delta_vs_finest_N = ...
    peak_tension_n - peak_tension_n(ref_idx);
report.tension_fraction_delta_vs_finest = ...
    tension_fraction - tension_fraction(ref_idx);

local_maybe_write_report(report, opt.OutputDir, logical(opt.Overwrite));
end

% -------------------------------------------------------------------------
function channel_table = local_channel_metric_table( ...
        dt_ms, channel_names, nrmse, nmax, correlation)
n_dt = numel(dt_ms);
n_channels = numel(channel_names);
expected_size = [n_dt, n_channels];
if ~isequal(size(nrmse), expected_size) || ...
        ~isequal(size(nmax), expected_size) || ...
        ~isequal(size(correlation), expected_size)
    error('contact_closure:MetricShape', ...
        'Channel metric arrays must be n_dt x n_channels.');
end
row_count = n_dt * n_channels;
dt_column = repelem(dt_ms(:), n_channels, 1);
channel_column = repmat(channel_names(:), n_dt, 1);
channel_table = table(dt_column, channel_column, ...
    reshape(nrmse', row_count, 1), reshape(nmax', row_count, 1), ...
    reshape(correlation', row_count, 1), ...
    'VariableNames', {'requested_dt_ms', 'channel', 'nrmse_vs_finest', ...
    'nmax_vs_finest', 'correlation_vs_finest'});
end

% -------------------------------------------------------------------------
function run = local_run_one(case_info, data, state_index, passage_index, ...
        damage_seed, requested_dt_ms)
velocity = double(data.Velocidade(passage_index));
temperature = double(data.Temperatura(passage_index));
vehicle_draw = double(data.VehiclesProps(:, :, passage_index));

Train = A01_Train(velocity, vehicle_draw);
Track = A02_Track();
Beam_seed = struct();
Beam_seed.Prop.L = double(case_info.L_bridge_m);
Beam_seed.Prop.num_spans = double(case_info.num_spans);
Beam = A03_Bridge(Beam_seed);
Beam.Prop.E = Beam.Prop.E - Beam.Prop.E * 0.003 * (temperature - 15);

Damage = local_damage_descriptor( ...
    case_info, data, state_index, passage_index, damage_seed);
Profile_cfg = local_profile_descriptor( ...
    case_info, data, state_index, passage_index, damage_seed);
[Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);

% B11 derives samples/s as 2*max_accurate_frq. This override changes only
% integration resolution; no production constant or contact gate is touched.
Calc.Solver.max_accurate_frq = 1 / (2 * requested_dt_ms * 1e-3);
[Sol, Calc, Train] = B00_Calculations(Calc, Train, Track, Beam, Damage);

run = struct();
run.requested_dt_ms = requested_dt_ms;
run.actual_dt_ms = Calc.Solver.dt * 1000;
run.n_samples = Calc.Solver.num_t;
run.contact_peak_signed_n = double(Sol.F_tension_max);
run.tension_fraction = double(Sol.tension_frac_max);
run.contact_lost_track = logical(Sol.contactLost_track);
run.contact_lost_bridge = logical(Sol.contactLost);
run.x_rel_m = double(Train.vel * Calc.Solver.t);
run.signal = double([Sol.Veh(1).A(1:3, :); ...
    Sol.Veh(1).acc_under(1:4, :); Sol.Veh(1).V(4:6, :)]);
run.channel_names = { ...
    'carbody_vertical_acc'; ...
    'front_bogie_vertical_acc'; ...
    'rear_bogie_vertical_acc'; ...
    'wheel_1_vertical_acc'; ...
    'wheel_2_vertical_acc'; ...
    'wheel_3_vertical_acc'; ...
    'wheel_4_vertical_acc'; ...
    'carbody_pitch_rate'; ...
    'front_bogie_pitch_rate'; ...
    'rear_bogie_pitch_rate'};
end

% -------------------------------------------------------------------------
function Damage = local_damage_descriptor(case_info, data, state_index, ...
        passage_index, damage_seed)
Damage = struct();
Damage.desvio = 0;
Damage.scour_rates = double(data.scour_vector(:)');
if isfield(data, 'bearing_vector') && numel(data.bearing_vector) >= 2
    bearing = double(data.bearing_vector(:)');
else
    bearing = [0, 0];
end
Damage.bearing_left = bearing(1);
Damage.bearing_right = bearing(2);

crack_row = double(data.crack_log(passage_index, :));
if numel(crack_row) >= 2 && crack_row(2) > 0
    Damage.crack_locs = crack_row(1);
    Damage.crack_intensity = crack_row(2);
    if numel(crack_row) >= 3
        Damage.crack_lc = crack_row(3);
    else
        Damage.crack_lc = 0;
    end
else
    Damage.crack_locs = [];
    Damage.crack_intensity = [];
    Damage.crack_lc = 0;
end

profile_mode = local_state_text(data, 'profile_mode', ...
    local_case_text(case_info, 'profile_mode', 'fixed'));
if strcmp(profile_mode, 'fixed_scaled')
    Damage.profile_intensity = double(data.profile_log(passage_index));
end

track_value = local_indexed_value(data.track_log, passage_index);
if isempty(track_value)
    Damage.track = [];
else
    Damage.track = track_value;
end

oor_value = local_indexed_value(data.oor_log, passage_index);
Damage.oor_flats = [];
Damage.oor_poly = [];
if isstruct(oor_value)
    if isfield(oor_value, 'flats'), Damage.oor_flats = oor_value.flats; end
    if isfield(oor_value, 'poly'), Damage.oor_poly = oor_value.poly; end
end
if isfield(case_info, 'oor_radius')
    Damage.oor_radius = double(case_info.oor_radius);
else
    Damage.oor_radius = 0.46;
end

% Retain these in the descriptor for report/debug provenance. B00 ignores
% them; the phase seed is consumed via Profile_cfg below.
Damage.contact_closure_state = state_index;
Damage.contact_closure_passage = passage_index;
Damage.contact_closure_damage_seed = damage_seed;
end

% -------------------------------------------------------------------------
function cfg = local_profile_descriptor(case_info, data, state_index, ...
        passage_index, damage_seed)
profile_mode = local_state_text(data, 'profile_mode', ...
    local_case_text(case_info, 'profile_mode', 'fixed'));
cfg = struct('mode', profile_mode);
switch profile_mode
    case 'fixed'
        % No additional descriptor.
    case 'fixed_scaled'
        cfg.intensity = double(data.profile_log(passage_index));
    case 'psd_fra'
        cfg.fra_class = double(data.profile_log(passage_index));
        cfg.phase_seed = 1e9 + damage_seed * 100000 + state_index;
        jitter_mm = 0;
        if isfield(case_info, 'profile_jitter_sd_mm')
            jitter_mm = double(case_info.profile_jitter_sd_mm);
        end
        if jitter_mm ~= 0
            error('contact_closure:UnreproducibleProfileJitter', ...
                ['profile_jitter_sd_mm=%g is nonzero, but the realized jitter ' ...
                 'was not persisted. Exact passage reconstruction is impossible.'], ...
                jitter_mm);
        end
        cfg.jitter_sd_m = 0;
    otherwise
        error('contact_closure:UnknownProfile', ...
            'Unsupported profile_mode "%s".', profile_mode);
end
end

% -------------------------------------------------------------------------
function local_validate_case(case_info, data, state_index)
required_case = {'L_bridge_m', 'num_spans'};
for k = 1:numel(required_case)
    if ~isfield(case_info, required_case{k})
        error('contact_closure:BadCaseInfo', ...
            'case_info is missing "%s".', required_case{k});
    end
end
if isfield(case_info, 'n_states') && state_index > double(case_info.n_states)
    error('contact_closure:StateOutOfRange', ...
        'State %d exceeds case_info.n_states=%d.', ...
        state_index, double(case_info.n_states));
end
required_data = {'Velocidade', 'Temperatura', 'VehiclesProps', ...
    'scour_vector', 'crack_log', 'profile_mode', 'profile_log', ...
    'track_log', 'oor_log', 'contact_log'};
for k = 1:numel(required_data)
    if ~isfield(data, required_data{k})
        error('contact_closure:BadState', ...
            'State payload is missing "%s".', required_data{k});
    end
end
n_passages = numel(data.Velocidade);
if size(data.contact_log, 1) ~= n_passages || size(data.contact_log, 2) < 4
    error('contact_closure:BadContactLog', ...
        'contact_log must have one row per passage and at least four columns.');
end
if size(data.VehiclesProps, 3) ~= n_passages
    error('contact_closure:BadVehicleProps', ...
        'VehiclesProps third dimension must equal the passage count.');
end
if any(~isfinite(double(data.contact_log(:))))
    error('contact_closure:BadContactLog', ...
        'contact_log contains NaN/Inf.');
end
end

% -------------------------------------------------------------------------
function passage_index = local_select_passage(selector, contact_log, n_passages)
if isnumeric(selector) && isscalar(selector) && isfinite(selector) && ...
        selector == round(selector)
    passage_index = double(selector);
elseif ischar(selector) || (isstring(selector) && isscalar(selector))
    selector = lower(strtrim(char(selector)));
    if strcmp(selector, 'worst')
        [~, passage_index] = max(double(contact_log(:, 4)));
    else
        parsed = str2double(selector);
        if ~isfinite(parsed) || parsed ~= round(parsed)
            error('contact_closure:BadPassage', ...
                'Passage selector must be a positive integer or "worst".');
        end
        passage_index = parsed;
    end
else
    error('contact_closure:BadPassage', ...
        'Passage selector must be a positive integer or "worst".');
end
if passage_index < 1 || passage_index > n_passages
    error('contact_closure:BadPassage', ...
        'Passage %d is outside 1..%d.', passage_index, n_passages);
end
end

% -------------------------------------------------------------------------
function pass = local_gate_pass(peak_tension_n, tension_fraction, gates_n, ...
        fraction_gate)
if ~isfinite(peak_tension_n) || ~isfinite(tension_fraction)
    pass = false(size(gates_n));
else
    pass = (peak_tension_n <= gates_n) & ...
        (tension_fraction <= fraction_gate);
end
end

% -------------------------------------------------------------------------
function [comparison, note] = local_saved_baseline_comparison( ...
        data, passage_index, x_common, rerun_signal_common, channel_names, ...
        rerun_signal_raw, rerun_x_raw)
comparison = table();
note = 'Saved raw channels unavailable.';
needed = {'AcelPrimVag', 'AcelRodaPrimVag', 'PitchPrimVag', 'DimSpace'};
if ~all(isfield(data, needed))
    return
end
saved_signal = double([data.AcelPrimVag{1, passage_index}; ...
    data.AcelRodaPrimVag{1, passage_index}; ...
    data.PitchPrimVag{1, passage_index}]);
if size(saved_signal, 1) ~= numel(channel_names)
    note = 'Saved raw channel count does not match the closure channel map.';
    return
end

% Prefer a direct raw-sample comparison when the current 1 ms run has the
% same grid. This is the exact reconstruction check. Only fall back to the
% historical D01 spatial map when lengths differ.
if isequal(size(saved_signal), size(rerun_signal_raw))
    saved_compare = saved_signal;
    rerun_compare = rerun_signal_raw;
    comparison_mode = 'direct raw samples';
else
    saved_distance_m = double(data.DimSpace(passage_index)) / 100;
    x_saved = linspace(0, saved_distance_m, size(saved_signal, 2));
    if x_common(end) > x_saved(end) || x_common(end) > rerun_x_raw(end)
        note = sprintf(['Saved/current signal ends before comparison window ' ...
            'ends at %.3f m.'], x_common(end));
        return
    end
    saved_compare = interp1( ...
        x_saved, saved_signal', x_common, 'linear')';
    rerun_compare = rerun_signal_common;
    comparison_mode = 'common spatial grid';
end
delta = rerun_compare - saved_compare;
n_channels = size(saved_compare, 1);
nrmse = zeros(n_channels, 1);
nmax = zeros(n_channels, 1);
correlation = zeros(n_channels, 1);
for ch = 1:n_channels
    nrmse(ch) = sqrt(mean(delta(ch, :).^2)) / ...
        max(sqrt(mean(saved_compare(ch, :).^2)), eps);
    nmax(ch) = max(abs(delta(ch, :))) / ...
        max(max(abs(saved_compare(ch, :))), eps);
    if std(rerun_compare(ch, :)) == 0 || std(saved_compare(ch, :)) == 0
        correlation(ch) = double(all(delta(ch, :) == 0));
    else
        c_ = corrcoef(rerun_compare(ch, :), saved_compare(ch, :));
        correlation(ch) = c_(1, 2);
    end
end
comparison = table(channel_names(:), nrmse, nmax, correlation, ...
    'VariableNames', {'channel', 'nrmse_rerun_vs_saved', ...
    'nmax_rerun_vs_saved', 'correlation_rerun_vs_saved'});
note = sprintf(['Comparison mode: %s. This is a diagnostic reproduction ' ...
    'check, not a gate; nonzero differences are expected when the saved ' ...
    'state predates current physics fixes.'], comparison_mode);
end

% -------------------------------------------------------------------------
function idx = local_nearest_dt_index(runs, target_ms)
[~, idx] = min(abs([runs.actual_dt_ms] - target_ms));
end

% -------------------------------------------------------------------------
function descriptor = local_descriptor_summary(case_info, data, state_index, ...
        passage_index, damage_seed)
descriptor = struct();
descriptor.L_bridge_m = double(case_info.L_bridge_m);
descriptor.num_spans = double(case_info.num_spans);
descriptor.velocity_kmh = double(data.Velocidade(passage_index)) * 3.6;
descriptor.temperature_C = double(data.Temperatura(passage_index));
descriptor.scour_vector = double(data.scour_vector(:)');
if isfield(data, 'bearing_vector')
    descriptor.bearing_vector_Nm_rad = double(data.bearing_vector(:)');
else
    descriptor.bearing_vector_Nm_rad = [0, 0];
end
descriptor.crack_row = double(data.crack_log(passage_index, :));
descriptor.profile_mode = local_state_text(data, 'profile_mode', ...
    local_case_text(case_info, 'profile_mode', 'fixed'));
descriptor.profile_value = double(data.profile_log(passage_index));
descriptor.profile_phase_seed = ...
    1e9 + damage_seed * 100000 + state_index;
descriptor.has_track_eov = ...
    ~isempty(local_indexed_value(data.track_log, passage_index));
oor = local_indexed_value(data.oor_log, passage_index);
descriptor.n_flats = 0;
descriptor.n_polygonization = 0;
if isstruct(oor)
    if isfield(oor, 'flats'), descriptor.n_flats = size(oor.flats, 1); end
    if isfield(oor, 'poly')
        descriptor.n_polygonization = size(oor.poly, 1);
    end
end
end

% -------------------------------------------------------------------------
function value = local_indexed_value(container, index)
if iscell(container)
    value = container{index};
elseif isstruct(container) && numel(container) >= index
    value = container(index);
elseif isempty(container)
    value = [];
else
    error('contact_closure:BadDescriptor', ...
        'Passage descriptor has an unsupported container type/size.');
end
end

% -------------------------------------------------------------------------
function text_value = local_case_text(case_info, field, fallback)
if isfield(case_info, field)
    text_value = char(string(case_info.(field)));
else
    text_value = fallback;
end
end

% -------------------------------------------------------------------------
function text_value = local_state_text(data, field, fallback)
if isfield(data, field)
    value = data.(field);
    if iscell(value)
        value = value{1};
    end
    text_value = char(string(value));
else
    text_value = fallback;
end
end

% -------------------------------------------------------------------------
function local_maybe_write_report(report, output_dir, overwrite)
if strlength(string(output_dir)) == 0
    return
end
output_dir = local_absolute_path(char(output_dir));
dataset_dir = local_absolute_path(report.dataset_dir);
if local_is_same_or_child(output_dir, dataset_dir)
    error('contact_closure:UnsafeOutput', ...
        'OutputDir must not be the source dataset directory or its child.');
end
if ~isfolder(output_dir)
    [ok, msg] = mkdir(output_dir);
    if ~ok
        error('contact_closure:OutputCreateFailed', ...
            'Could not create OutputDir: %s', msg);
    end
end
safe_stage = regexprep(report.stage, '[^A-Za-z0-9_-]', '_');
stem = sprintf('%s_state%04d_pass%02d_contact_closure', ...
    safe_stage, report.state_index, report.passage_index);
mat_path = fullfile(output_dir, [stem, '.mat']);
md_path = fullfile(output_dir, [stem, '.md']);
if ~overwrite && (isfile(mat_path) || isfile(md_path))
    error('contact_closure:OutputExists', ...
        'Report already exists; pass Overwrite=true to replace it: %s', stem);
end

tmp_mat = [mat_path, '.tmp'];
tmp_md = [md_path, '.tmp'];
save(tmp_mat, 'report');
local_write_markdown(tmp_md, report);
movefile(tmp_mat, mat_path, 'f');
movefile(tmp_md, md_path, 'f');
fprintf('Contact-closure report: %s\n', md_path);
end

% -------------------------------------------------------------------------
function local_write_markdown(path, report)
fid = fopen(path, 'w');
if fid < 0
    error('contact_closure:OutputOpenFailed', ...
        'Could not open report for writing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Wheel-rail contact closure: %s state %d, passage %d\n\n', ...
    report.stage, report.state_index, report.passage_index);
fprintf(fid, '- Status: `%s`\n', report.status);
fprintf(fid, '- Generated UTC: `%s`\n', report.created_utc);
fprintf(fid, '- MATLAB: `%s`\n', report.matlab_release);
fprintf(fid, '- Generator schema: `%s`\n', report.gen_schema);
fprintf(fid, '- Generator fingerprint: `%s`\n', report.gen_fingerprint);
fprintf(fid, '- State-file SHA-256: `%s`\n', report.state_file_sha256);
fprintf(fid, '- Dataset integrity: `%s`\n', report.dataset_integrity.status);
if strcmp(report.dataset_integrity.status, 'VERIFIED')
    fprintf(fid, '- Dataset manifest root: `%s`\n', ...
        report.dataset_integrity.manifest_root);
end
fprintf(fid, '- Harness SHA-256: `%s`\n', report.harness_sha256);
fprintf(fid, '- B66 SHA-256: `%s`\n\n', report.b66_sha256);
fprintf(fid, '## Reconstructed passage\n\n');
fprintf(fid, '- Velocity: `%.8g km/h`\n', ...
    report.descriptor.velocity_kmh);
fprintf(fid, '- Temperature: `%.8g C`\n', ...
    report.descriptor.temperature_C);
fprintf(fid, '- Scour vector: `%s`\n', ...
    mat2str(report.descriptor.scour_vector, 8));
fprintf(fid, '- Bearing vector: `%s Nm/rad`\n', ...
    mat2str(report.descriptor.bearing_vector_Nm_rad, 8));
fprintf(fid, '- Crack row: `%s`\n', ...
    mat2str(report.descriptor.crack_row, 8));
fprintf(fid, '- Profile: `%s`, value `%.8g`, phase seed `%.0f`\n', ...
    report.descriptor.profile_mode, report.descriptor.profile_value, ...
    report.descriptor.profile_phase_seed);
fprintf(fid, '- Track EOV present: `%s`\n', ...
    string(report.descriptor.has_track_eov));
fprintf(fid, '- Wheel flats/polygonization rows: `%d/%d`\n\n', ...
    report.descriptor.n_flats, report.descriptor.n_polygonization);
fprintf(fid, '## Solver source manifest\n\n');
fprintf(fid, '| module | SHA-256 |\n|---|---|\n');
for source_index = 1:height(report.solver_source_sha256)
    fprintf(fid, '| %s | `%s` |\n', ...
        report.solver_source_sha256.module{source_index}, ...
        report.solver_source_sha256.sha256{source_index});
end
fprintf(fid, '\n');
fprintf(fid, ['Diagnostic acceptance requires peak tension <= the indicated ' ...
    'gate and tension fraction <= %.6g. These gates do not alter the solver.\n\n'], ...
    report.fraction_gate);
fprintf(fid, ['Saved contact row `[bridge_lost, track_lost, fraction, ' ...
    'signed_peak_N]`: `%s`.\n\n'], mat2str(report.saved_contact_log, 8));

if report.dry_run
    fprintf(fid, 'Dry run only: descriptors validated; no solver was invoked.\n');
    return
end

fprintf(fid, '| requested dt (ms) | actual dt (ms) | samples | peak tension (N) | fraction |');
for g = report.gates_n
    fprintf(fid, ' gate %.0f N |', g);
end
fprintf(fid, '\n|---:|---:|---:|---:|---:|');
fprintf(fid, '%s', repmat('---:|', 1, numel(report.gates_n)));
fprintf(fid, '\n');
for r = 1:height(report.run_table)
    fprintf(fid, '| %.6g | %.6g | %d | %.8g | %.8g |', ...
        report.run_table.requested_dt_ms(r), ...
        report.run_table.actual_dt_ms(r), report.run_table.n_samples(r), ...
        report.run_table.peak_tension_N(r), ...
        report.run_table.tension_fraction(r));
    for g = 1:numel(report.gates_n)
        gate_name = matlab.lang.makeValidName( ...
            sprintf('pass_gate_%g_N', report.gates_n(g)));
        fprintf(fid, ' %s |', string(report.run_table.(gate_name)(r)));
    end
    fprintf(fid, '\n');
end

fprintf(fid, '\n## Signal convergence against finest dt\n\n');
fprintf(fid, '| dt (ms) | channel | NRMSE | normalized max error | correlation |\n');
fprintf(fid, '|---:|---|---:|---:|---:|\n');
for r = 1:height(report.channel_table)
    fprintf(fid, '| %.6g | %s | %.8g | %.8g | %.8g |\n', ...
        report.channel_table.requested_dt_ms(r), ...
        report.channel_table.channel{r}, ...
        report.channel_table.nrmse_vs_finest(r), ...
        report.channel_table.nmax_vs_finest(r), ...
        report.channel_table.correlation_vs_finest(r));
end

if ~isempty(report.saved_baseline_table)
    fprintf(fid, '\n## Current 1 ms rerun versus saved raw passage\n\n');
    fprintf(fid, '| channel | NRMSE | normalized max error | correlation |\n');
    fprintf(fid, '|---|---:|---:|---:|\n');
    for r = 1:height(report.saved_baseline_table)
        fprintf(fid, '| %s | %.8g | %.8g | %.8g |\n', ...
            report.saved_baseline_table.channel{r}, ...
            report.saved_baseline_table.nrmse_rerun_vs_saved(r), ...
            report.saved_baseline_table.nmax_rerun_vs_saved(r), ...
            report.saved_baseline_table.correlation_rerun_vs_saved(r));
    end
    fprintf(fid, '\n%s\n', report.saved_baseline_note);
end
end

% -------------------------------------------------------------------------
function integrity = local_verify_dataset_integrity(dataset_dir, state_path, ...
        state_sha, case_info, state_data, state_index, verify_integrity)
integrity = struct('status', 'SKIPPED', 'manifest_root', '', ...
    'state_digest_match', false, 'marker_match', false, ...
    'state_table_match', false);
if ~verify_integrity
    return
end

marker_path = fullfile(dataset_dir, '_GENERATION_COMPLETE');
digests_path = fullfile(dataset_dir, 'file_digests.mat');
states_path = fullfile(dataset_dir, 'damage_states.mat');
if ~isfile(marker_path) || ~isfile(digests_path) || ~isfile(states_path)
    error('contact_closure:IncompleteDataset', ...
        ['Integrity verification requires _GENERATION_COMPLETE, ' ...
         'file_digests.mat and damage_states.mat.']);
end

digest_blob = load(digests_path);
if ~isfield(digest_blob, 'file_digests') || ...
        ~isstruct(digest_blob.file_digests)
    error('contact_closure:BadDigestManifest', ...
        'file_digests.mat does not contain file_digests.');
end
manifest = digest_blob.file_digests;
if ~isfield(manifest, 'digest_lines') || ~isfield(manifest, 'root')
    error('contact_closure:BadDigestManifest', ...
        'file_digests is missing digest_lines/root.');
end
digest_lines = char(manifest.digest_lines);
computed_root = local_text_sha256(digest_lines);
manifest_root = lower(char(manifest.root));
if ~strcmp(computed_root, manifest_root)
    error('contact_closure:BadDigestRoot', ...
        'file_digests.root does not match digest_lines.');
end

[~, state_name, state_ext] = fileparts(state_path);
state_filename = [state_name, state_ext];
lines = regexp(digest_lines, '\r?\n', 'split');
prefix = [state_filename, ':'];
matching = lines(startsWith(lines, prefix));
if ~isscalar(matching)
    error('contact_closure:MissingStateDigest', ...
        'Manifest must contain exactly one digest for %s.', state_filename);
end
expected_sha = extractAfter(matching{1}, strlength(prefix));
if ~strcmpi(expected_sha, state_sha)
    error('contact_closure:StateDigestMismatch', ...
        'State bytes do not match file_digests.mat: %s.', state_filename);
end

marker_lines = regexp(strtrim(fileread(marker_path)), '\r?\n', 'split');
if numel(marker_lines) < 3
    error('contact_closure:BadCompletionMarker', ...
        '_GENERATION_COMPLETE must contain schema, fingerprint and root.');
end
schema = local_case_text(case_info, 'gen_schema', '');
fingerprint = local_case_text(case_info, 'gen_fingerprint', '');
if ~strcmp(marker_lines{1}, schema) || ...
        ~strcmp(marker_lines{2}, fingerprint) || ...
        ~strcmpi(marker_lines{3}, manifest_root)
    error('contact_closure:CompletionMarkerMismatch', ...
        '_GENERATION_COMPLETE disagrees with case_info/file_digests.');
end

state_table = load(states_path);
if ~isfield(state_table, 'DamageStates') || ...
        size(state_table.DamageStates, 1) < state_index
    error('contact_closure:BadStateTable', ...
        'damage_states.mat has no DamageStates row %d.', state_index);
end
if ~isequal(double(state_table.DamageStates(state_index, :)), ...
        double(state_data.scour_vector(:)'))
    error('contact_closure:StateTableMismatch', ...
        'Saved scour_vector differs from DamageStates row %d.', state_index);
end
if isfield(state_table, 'BearingStates') && ...
        isfield(state_data, 'bearing_vector') && ...
        ~isequal(double(state_table.BearingStates(state_index, :)), ...
            double(state_data.bearing_vector(:)'))
    error('contact_closure:StateTableMismatch', ...
        'Saved bearing_vector differs from BearingStates row %d.', state_index);
end

integrity.status = 'VERIFIED';
integrity.manifest_root = manifest_root;
integrity.state_digest_match = true;
integrity.marker_match = true;
integrity.state_table_match = true;
end

% -------------------------------------------------------------------------
function manifest = local_solver_source_manifest()
names = { ...
    'A01_Train'; ...
    'A02_Track'; ...
    'A03_Bridge'; ...
    'A04_Options'; ...
    'B00_Calculations'; ...
    'B54_ModelMatrices'; ...
    'B65_DynamicCalcCoupledFaster'; ...
    'B66_ContactForce'};
paths = cell(size(names));
sha256 = cell(size(names));
for k = 1:numel(names)
    paths{k} = which(names{k});
    if isempty(paths{k})
        error('contact_closure:MissingSolverSource', ...
            'Could not locate %s.', names{k});
    end
    sha256{k} = local_file_sha256(paths{k});
end
manifest = table(names, paths, sha256, ...
    'VariableNames', {'module', 'path', 'sha256'});
end

% -------------------------------------------------------------------------
function absolute = local_absolute_path(path)
path = char(path);
if isempty(path)
    absolute = path;
    return
end
file_obj = java.io.File(path);
absolute = char(file_obj.getCanonicalPath());
end

% -------------------------------------------------------------------------
function tf = local_is_same_or_child(candidate, parent)
candidate = [lower(strrep(candidate, '/', '\')), '\'];
parent = [lower(strrep(parent, '/', '\')), '\'];
tf = startsWith(candidate, parent);
end

% -------------------------------------------------------------------------
function digest = local_file_sha256(path)
fid = fopen(path, 'r');
if fid < 0
    error('contact_closure:HashOpenFailed', ...
        'Could not open file for SHA-256: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
digest = local_bytes_sha256(bytes);
end

% -------------------------------------------------------------------------
function digest = local_numeric_sha256(values)
bytes = typecast(values(:), 'uint8');
digest = local_bytes_sha256(bytes);
end

% -------------------------------------------------------------------------
function digest = local_text_sha256(value)
bytes = unicode2native(char(value), 'UTF-8');
digest = local_bytes_sha256(bytes);
end

% -------------------------------------------------------------------------
function digest = local_bytes_sha256(bytes)
engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(bytes);
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end

% -------------------------------------------------------------------------
function tf = local_positive_vector(value)
tf = isnumeric(value) && isvector(value) && ~isempty(value) && ...
    all(isfinite(value)) && all(value > 0);
end

% -------------------------------------------------------------------------
function tf = local_nonnegative_vector(value)
tf = isnumeric(value) && isvector(value) && ~isempty(value) && ...
    all(isfinite(value)) && all(value >= 0);
end

% -------------------------------------------------------------------------
function tf = local_positive_scalar(value)
tf = isnumeric(value) && isscalar(value) && isfinite(value) && value > 0;
end

% -------------------------------------------------------------------------
function tf = local_nonnegative_scalar(value)
tf = isnumeric(value) && isscalar(value) && isfinite(value) && value >= 0;
end

% -------------------------------------------------------------------------
function tf = local_nan_or_nonnegative_integer(value)
tf = isnumeric(value) && isscalar(value) && ...
    (isnan(value) || (isfinite(value) && value >= 0 && value == round(value)));
end

% -------------------------------------------------------------------------
function tf = local_logical_scalar(value)
tf = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
    isfinite(value) && any(value == [0, 1]);
end

% -------------------------------------------------------------------------
function tf = local_text_scalar(value)
tf = ischar(value) || (isstring(value) && isscalar(value));
end
