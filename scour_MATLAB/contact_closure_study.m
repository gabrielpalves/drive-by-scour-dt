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
%   ReconstructionRtol raw-sample relative tolerance, default 1e-10
%   ReconstructionAtol raw-sample absolute tolerance, default 1e-12
%   VerifyIntegrity require completion marker + per-file manifest, default true
%   DryRun        validate/describe the case without invoking the solver
%   OutputDir     optional report destination; default "" writes nothing
%   Overwrite     allow replacement of an existing report, default false
%
% When OutputDir is supplied, a small .md appendix report and a .mat record
% are written. OutputDir must not be the source dataset directory.
%
% Paper-1 qualification must use integer passage selectors supplied by
% contact_closure_gate.  "worst" remains a non-qualifying interactive
% diagnostic only; no historical state number is part of the gate.
%
% MODULE LAYOUT: this entry file is a thin orchestrator.  Cohesive
% responsibilities live in dedicated files, each returning a struct of
% function handles (pattern documented in contact_closure_common.m):
%   contact_load_study_dataset_snapshots
%                                 stable byte capture + MAT parsing of the
%                                 authenticated case/state/state table
%   contact_study_reconstruction  saved-state/current validation, passage and
%                                 Damage/Profile descriptor rebuild (owns
%                                 the single profile-phase seed assignment)
%   contact_study_solver          production-chain execution + solver
%                                 source manifest
%   contact_study_metrics         convergence/QOI tables, diagnostic gate,
%                                 saved-baseline comparison, descriptor
%                                 summary
%   contact_study_report          durable .mat/.md publication
%   contact_closure_common        shared hashing/path/closeness utilities
%   contact_solver_modules        frozen 37-module solver inventory
% The comparison-window construction stays HERE so the acceptance support
% policy is visible at the orchestration level.
%
% HARNESS IDENTITY: report.harness_sha256 is the SHA-256 root of the
% complete study executable set listed above (sorted "name:sha256" lines,
% LF-joined, no terminal LF -- the generator digest-root grammar), NOT the
% hash of this single file.  See contact_closure_common.m.

arguments
    dataset_dir {mustBeTextScalar}
    state_index (1,1) double {mustBeInteger,mustBePositive}
    passage_selector = "worst"
end
arguments (Repeating)
    varargin
end

% Validate the root-building chain before any shadowable project helper is
% executed. The bootstrap itself must resolve beside this reviewed entry.
reviewed_dir = fileparts(mfilename('fullpath'));
bootstrap_path = which('contact_assert_reviewed_bootstrap');
bootstrap_expected = fullfile( ...
    reviewed_dir, 'contact_assert_reviewed_bootstrap.m');
bootstrap_resolved_file = java.io.File(bootstrap_path);
bootstrap_expected_file = java.io.File(bootstrap_expected);
bootstrap_resolved_absolute = strrep( ...
    char(bootstrap_resolved_file.getAbsolutePath()), '\', '/');
bootstrap_expected_absolute = strrep( ...
    char(bootstrap_expected_file.getAbsolutePath()), '\', '/');
bootstrap_resolved_canonical = strrep( ...
    char(bootstrap_resolved_file.getCanonicalPath()), '\', '/');
bootstrap_expected_canonical = strrep( ...
    char(bootstrap_expected_file.getCanonicalPath()), '\', '/');
if ispc
    bootstrap_resolved_absolute = lower(bootstrap_resolved_absolute);
    bootstrap_expected_absolute = lower(bootstrap_expected_absolute);
    bootstrap_resolved_canonical = lower(bootstrap_resolved_canonical);
    bootstrap_expected_canonical = lower(bootstrap_expected_canonical);
end
if isempty(bootstrap_path) || ...
        ~strcmp(bootstrap_resolved_absolute, bootstrap_expected_absolute) || ...
        ~strcmp(bootstrap_resolved_absolute, bootstrap_resolved_canonical) || ...
        ~strcmp(bootstrap_expected_absolute, bootstrap_expected_canonical)
    error('contact_closure:HarnessShadowed', ...
        'Study bootstrap helper is missing or shadowed.');
end
contact_assert_reviewed_bootstrap(reviewed_dir, { ...
    'contact_absolute_path'; ...
    'contact_bytes_sha256'; ...
    'contact_closure_common'; ...
    'contact_comparison_path'; ...
    'contact_file_bytes'; ...
    'contact_file_observation'; ...
    'contact_file_sha256'; ...
    'contact_filesystem_identity'; ...
    'contact_java_boolean_value'; ...
    'contact_path_component_is_link_alias'; ...
    'contact_regular_nonsymlink'; ...
    'contact_regular_nonsymlink_directory'; ...
    'contact_resolved_module_root'; ...
    'contact_run_small_process'; ...
    'contact_stable_file_bytes'; ...
    'contact_study_harness_files'; ...
    'contact_study_harness_root'; ...
    'contact_text_sha256'; ...
    'contact_unlinked_path_identity'; ...
    'contact_windows_file_identity'}, ...
    'contact_closure:HarnessShadowed', 'Study bootstrap module');

common = contact_closure_common();
common.study_harness_root();
recon = contact_study_reconstruction();
metrics = contact_study_metrics();
solver = contact_study_solver();
writer = contact_study_report();

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'DtMs', [1, 0.5, 0.25], @contact_positive_vector);
addParameter(parser, 'GatesN', [0, 12000, 24000], ...
    @contact_nonnegative_vector);
addParameter(parser, 'FractionGate', 0.002, @contact_nonnegative_scalar);
addParameter(parser, 'CommonDxM', 0.01, @contact_positive_scalar);
addParameter(parser, 'ReconstructionRtol', 1e-10, ...
    @contact_nonnegative_scalar);
addParameter(parser, 'ReconstructionAtol', 1e-12, ...
    @contact_nonnegative_scalar);
addParameter(parser, 'VerifyIntegrity', true, common.logical_scalar);
addParameter(parser, 'DryRun', false, common.logical_scalar);
addParameter(parser, 'OutputDir', "", common.text_scalar);
addParameter(parser, 'Overwrite', false, common.logical_scalar);
parse(parser, varargin{:});
opt = parser.Results;

dataset_dir = char(dataset_dir);
try
    dataset_identity = contact_unlinked_path_identity(dataset_dir);
catch dataset_path_error
    error('contact_closure:LinkedDataset', ...
        ['Dataset directory path is linked or cannot be authenticated: ' ...
         '%s (%s)'], dataset_dir, dataset_path_error.message);
end
if ~dataset_identity.exists || ~dataset_identity.is_directory
    error('contact_closure:MissingDataset', ...
        'Dataset directory does not exist: %s', dataset_dir);
end
dataset_dir = dataset_identity.canonical_path;

dataset_snapshot = recon.load_dataset_snapshots( ...
    dataset_dir, state_index, logical(opt.VerifyIntegrity));
case_info = dataset_snapshot.case_info;
state_data = dataset_snapshot.state_data;
state_table = dataset_snapshot.state_table;
state_path = dataset_snapshot.state_path;

recon.validate_case(case_info, state_data, state_index);
descriptor_contract = recon.validate_r11_descriptor( ...
    case_info, state_data, state_table, state_index, ...
    logical(opt.VerifyIntegrity));
n_passages = numel(state_data.Velocidade);
passage_index = recon.select_passage( ...
    passage_selector, state_data.contact_log, n_passages);

dt_ms = unique(double(opt.DtMs(:)'), 'stable');
gates_n = unique(double(opt.GatesN(:)'), 'stable');
fraction_gate = double(opt.FractionGate);
common_dx_m = double(opt.CommonDxM);
reconstruction_rtol = double(opt.ReconstructionRtol);
reconstruction_atol = double(opt.ReconstructionAtol);

saved_contact = double(state_data.contact_log(passage_index, 1:4));
saved_tension_n = max(0, saved_contact(4));
saved_gate_pass = metrics.gate_pass( ...
    saved_tension_n, saved_contact(3), gates_n, fraction_gate);

report = struct();
report.study_schema = 'contact-closure-v3';
report.created_utc = char(datetime('now', ...
    'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
report.matlab_release = version('-release');
report.dataset_dir = dataset_dir;
report.state_file = state_path;
report.state_file_sha256 = dataset_snapshot.state_snapshot.sha256;
[report.dataset_integrity, integrity_snapshots] = ...
    recon.verify_dataset_integrity( ...
        dataset_snapshot, descriptor_contract, ...
        logical(opt.VerifyIntegrity));
publication_snapshots = [dataset_snapshot.snapshots; integrity_snapshots];
report.stage = recon.case_text(case_info, 'stage', 'unknown');
report.case_name = recon.case_text(case_info, 'case_name', 'unknown');
report.gen_schema = recon.case_text(case_info, 'gen_schema', 'unknown');
report.generation_behavior_version = recon.case_text( ...
    case_info, 'generation_behavior_version', 'unknown');
report.channel_schema_id = recon.case_text( ...
    case_info, 'channel_schema_id', 'unknown');
report.gen_fingerprint = recon.case_text( ...
    case_info, 'gen_fingerprint', 'unknown');
report.state_index = state_index;
report.passage_index = passage_index;
report.passage_selector = char(string(passage_selector));
report.state_uid = descriptor_contract.state_uid;
report.state_family = descriptor_contract.state_family;
report.profile_phase_stream_index = ...
    descriptor_contract.profile_phase_stream_index;
report.profile_phase_seed = descriptor_contract.profile_phase_seed;
report.dt_requested_ms = dt_ms;
report.gates_n = gates_n;
report.fraction_gate = fraction_gate;
report.common_dx_m = common_dx_m;
report.reconstruction_rtol = reconstruction_rtol;
report.reconstruction_atol = reconstruction_atol;
report.saved_contact_log = saved_contact;
report.saved_gate_pass = saved_gate_pass;
% harness_sha256: SHA-256 root of the complete study executable set (see
% file header and contact_closure_common.m).  The gate and the Python
% checker recompute this identical multi-file root.
report.harness_sha256 = common.study_harness_root();
report.b66_sha256 = common.file_sha256(which('B66_ContactForce'));
[report.solver_source_sha256, report.solver_execution_root_sha256] = ...
    solver.solver_source_manifest( ...
        descriptor_contract.current_generator_source_digest_lines);
report.current_generator_source_root_sha256 = ...
    descriptor_contract.current_generator_source_root_sha256;
report.current_matlab_environment_sha256 = ...
    descriptor_contract.current_matlab_environment_sha256;
report.dry_run = logical(opt.DryRun);
report.numeric_hash_selfcheck = common.numeric_sha256([0; 1]);

if opt.DryRun
    % Exercise the late-stage long-table transformation even when no target
    % dataset is locally available. This guards against discovering a shape
    % error only after an expensive solver run.
    selfcheck_table = metrics.channel_metric_table([1, 0.5], {'a'; 'b'}, ...
        zeros(2), zeros(2), ones(2));
    assert(height(selfcheck_table) == 4, ...
        'contact_closure:InternalSelfCheck', ...
        'Internal channel-table self-check failed.');
    report.status = 'DRY_RUN_VALIDATED';
    report.descriptor = metrics.descriptor_summary( ...
        case_info, state_data, passage_index, descriptor_contract);
    report.run_table = table();
    report.channel_table = table();
    report.channel_qoi_table = table();
    report.saved_baseline_table = table();
    report.direct_reconstruction_pass = false;
    report.saved_contact_reconstruction_pass = false;
    contact_assert_snapshot_set_unchanged(publication_snapshots);
    writer.maybe_write_report(report, opt.OutputDir, logical(opt.Overwrite));
    contact_assert_snapshot_set_unchanged(publication_snapshots);
    return
end

rng_before = rng;
rng_cleanup = onCleanup(@() rng(rng_before));
n_dt = numel(dt_ms);
run_cells = cell(1, n_dt);
for k = 1:n_dt
    run_cells{k} = solver.run_one(case_info, state_data, state_index, ...
        passage_index, descriptor_contract, dt_ms(k));
end
runs = [run_cells{:}];

% The finest requested step is the numerical reference. Signals are compared
% on the registered production crop span (10 m skip + bridge + 18.30 m,
% where 18.30 m is the D01 crop's 1830 post-deck grid intervals) at a
% common, explicitly recorded dx. This avoids comparing unequal time grids.
% Fail-closed: a signal that ends before this window is an error; the
% acceptance support is never silently shrunk.
[~, ref_idx] = min([runs.actual_dt_ms]);
x_lo = 10;
x_hi_requested = 10 + double(case_info.L_bridge_m) + 18.30;
x_hi_available = min(arrayfun(@(r) r.x_rel_m(end), runs));
if x_hi_available < x_hi_requested
    error('contact_closure:ShortSignal', ...
        ['Signal ends at %.6g m, before the registered comparison ', ...
        'window end %.6g m (10 m skip + bridge + 18.30 m).'], ...
        x_hi_available, x_hi_requested);
end
x_hi = x_hi_requested;
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
    runs(k).signal_common_sha256 = common.numeric_sha256( ...
        signal_common(:, :, k));
end

peak_signed_n = [runs.contact_peak_signed_n]';
peak_tension_n = max(0, peak_signed_n);
tension_fraction = [runs.tension_fraction]';
contact_lost_track = logical([runs.contact_lost_track]');
contact_lost_bridge = logical([runs.contact_lost_bridge]');
gate_pass = false(n_dt, numel(gates_n));
for k = 1:n_dt
    gate_pass(k, :) = metrics.gate_pass(peak_tension_n(k), ...
        tension_fraction(k), gates_n, fraction_gate);
end

run_table = table(dt_ms(:), [runs.actual_dt_ms]', [runs.t_end_s]', ...
    [runs.n_samples]', ...
    peak_signed_n, peak_tension_n, tension_fraction, contact_lost_track, ...
    contact_lost_bridge, ...
    'VariableNames', {'requested_dt_ms', 'actual_dt_ms', 't_end_s', ...
    'n_samples', 'peak_contact_signed_N', 'peak_tension_N', ...
    'tension_fraction', 'contact_lost_track', 'contact_lost_bridge'});
for g = 1:numel(gates_n)
    gate_name = matlab.lang.makeValidName( ...
        sprintf('pass_gate_%g_N', gates_n(g)));
    run_table.(gate_name) = gate_pass(:, g);
end

channel_names = runs(1).channel_names(:);
channel_table = metrics.channel_metric_table( ...
    dt_ms, channel_names, nrmse, nmax, corrcoef_ch);
channel_qoi_table = metrics.channel_qoi_table( ...
    dt_ms, channel_names, signal_common);

baseline_idx = metrics.nearest_dt_index(runs, 1);
[saved_table, saved_note, saved_mode, saved_signal_pass] = ...
    metrics.saved_baseline_comparison( ...
    state_data, passage_index, x_common, ...
    signal_common(:, :, baseline_idx), channel_names, ...
    runs(baseline_idx).signal, runs(baseline_idx).x_rel_m, ...
    reconstruction_rtol, reconstruction_atol);
rerun_contact = [double(runs(baseline_idx).contact_lost_bridge), ...
    double(runs(baseline_idx).contact_lost_track), ...
    runs(baseline_idx).tension_fraction, ...
    runs(baseline_idx).contact_peak_signed_n];
saved_contact_pass = isequal(saved_contact(1:2), rerun_contact(1:2)) && ...
    common.allclose(saved_contact(3:4), rerun_contact(3:4), ...
        reconstruction_rtol, reconstruction_atol);

report.status = 'COMPLETED';
report.reference_dt_ms = runs(ref_idx).actual_dt_ms;
report.comparison_window_m = [x_common(1), x_common(end)];
report.descriptor = metrics.descriptor_summary( ...
    case_info, state_data, passage_index, descriptor_contract);
report.run_table = run_table;
report.channel_table = channel_table;
report.channel_qoi_table = channel_qoi_table;
report.saved_baseline_table = saved_table;
report.saved_baseline_note = saved_note;
report.saved_baseline_mode = saved_mode;
report.direct_reconstruction_pass = ...
    strcmp(saved_mode, 'direct_raw_samples') && saved_signal_pass;
report.rerun_contact_log_1ms = rerun_contact;
report.saved_contact_reconstruction_pass = saved_contact_pass;
report.signal_common_sha256 = {runs.signal_common_sha256};
report.contact_peak_delta_vs_finest_N = ...
    peak_tension_n - peak_tension_n(ref_idx);
report.tension_fraction_delta_vs_finest = ...
    tension_fraction - tension_fraction(ref_idx);

contact_assert_snapshot_set_unchanged(publication_snapshots);
writer.maybe_write_report(report, opt.OutputDir, logical(opt.Overwrite));
contact_assert_snapshot_set_unchanged(publication_snapshots);
end
