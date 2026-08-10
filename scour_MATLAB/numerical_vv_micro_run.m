function result = numerical_vv_micro_run(output_dir, varargin)
%NUMERICAL_VV_MICRO_RUN Execute a bounded, explicitly nonqualifying V&V micro.
%
% The micro exercises registered mesh/time metadata, simply-supported static
% and modal fixtures, actual B11 time-grid realization, artifact hashing, and
% fail-closed publication.  It does not run the coupled dynamic solver and
% cannot support a validation or production-resolution qualification claim.

parser = inputParser;
addRequired(parser, 'output_dir', @local_text_scalar);
addParameter(parser, 'BridgeLengthsM', [60.0, 99.6], @isnumeric);
addParameter(parser, 'MeshLevelIDs', {'M0', 'M1', 'M2'}, @iscell);
addParameter(parser, 'TimeLevelIDs', {'dt1', 'dt0p5', 'dt0p25'}, @iscell);
addParameter(parser, 'PointLoadN', 123456.7, @local_positive_scalar);
addParameter(parser, 'SyntheticTimeWindowS', 4.1234567, @local_positive_scalar);
addParameter(parser, 'MaxMeshCases', 8, @local_positive_integer);
addParameter(parser, 'Command', '', @local_text_scalar);
parse(parser, output_dir, varargin{:});
opt = parser.Results;
output_dir = local_absolute_path(char(opt.output_dir));

if exist(output_dir, 'file') || exist(output_dir, 'dir')
    error('numerical_vv:OutputAlreadyExists', ...
        'Micro output must be a new path; overwrite/resume is forbidden: %s', ...
        output_dir);
end
[made, message] = mkdir(output_dir);
if ~made
    error('numerical_vv:OutputCreate', ...
        'Could not create output directory: %s', message);
end
incomplete_path = fullfile(output_dir, '_RUN_INCOMPLETE');
local_write_text(incomplete_path, sprintf( ...
    'schema=numerical-vv-incomplete-v1\nstatus=INCOMPLETE\n'));

P = numerical_vv_protocol_definition();
lengths = unique(double(opt.BridgeLengthsM(:)'), 'stable');
if isempty(lengths) || any(~isfinite(lengths)) || ...
        any(~ismember(lengths, [P.geometries.bridge_length_m]))
    error('numerical_vv:UnregisteredGeometry', ...
        'BridgeLengthsM must be a nonempty subset of registered geometries.');
end
mesh_ids = local_registered_ids(opt.MeshLevelIDs, ...
    {P.mesh_levels.id}, 'mesh level');
time_ids = local_registered_ids(opt.TimeLevelIDs, ...
    {P.time_levels.id}, 'time level');
n_mesh_cases = numel(lengths)*numel(mesh_ids);
if n_mesh_cases > opt.MaxMeshCases
    error('numerical_vv:MicroCaseLimit', ...
        'Requested %d mesh cases exceeds bounded micro limit %d.', ...
        n_mesh_cases, opt.MaxMeshCases);
end

source = local_source_identity();
input_descriptor = struct( ...
    'schema', 'numerical-vv-micro-input-v1', ...
    'bridge_lengths_m', lengths, ...
    'mesh_level_ids', {mesh_ids}, ...
    'time_level_ids', {time_ids}, ...
    'point_load_N', opt.PointLoadN, ...
    'synthetic_time_window_s', opt.SyntheticTimeWindowS, ...
    'dynamic_solver_executed', false, ...
    'qualification_requested', false);
input_json = jsonencode(input_descriptor);
input_hash = numerical_vv_sha256_bytes( ...
    unicode2native(input_json, 'UTF-8'));

start_utc = local_utc_now();
clock_start = tic;
run_cells = cell(n_mesh_cases, 1);
case_index = 0;
for g = 1:numel(lengths)
    for m = 1:numel(mesh_ids)
        case_index = case_index+1;
        one_run = numerical_vv_bridge_fixture( ...
            lengths(g), mesh_ids{m}, opt.PointLoadN);
        one_run.case_id = sprintf('%s_%s_ss_micro', ...
            one_run.geometry_id, one_run.mesh_level);
        run_cells{case_index} = one_run;
    end
end
runs = vertcat(run_cells{:});
time_rows = local_time_grid_rows(P, time_ids, ...
    opt.SyntheticTimeWindowS, source.commit, input_hash);

[case_table, descriptor_table, static_table, modal_table, scalar_table] = ...
    local_tables(runs, source.commit, input_hash);
support_table = struct2table(P.registered_support_alignment);
tolerance_table = local_tolerance_table(P, source.commit, input_hash);

writetable(case_table, fullfile(output_dir, 'case_table.csv'));
writetable(descriptor_table, fullfile(output_dir, 'descriptor_hashes.csv'));
writetable(static_table, ...
    fullfile(output_dir, 'static_equilibrium_checks.csv'));
writetable(modal_table, fullfile(output_dir, 'modal_matching.csv'));
writetable(scalar_table, fullfile(output_dir, 'mesh_scalar_qoi.csv'));
writetable(time_rows, fullfile(output_dir, 'time_grid_realization.csv'));
writetable(support_table, fullfile(output_dir, 'support_alignment.csv'));
writetable(tolerance_table, ...
    fullfile(output_dir, 'tolerance_rationale.csv'));
save(fullfile(output_dir, 'raw_bridge_fixture.mat'), 'runs', '-v7.3');
local_write_json(fullfile(output_dir, 'protocol_snapshot.json'), P);
local_write_json(fullfile(output_dir, 'input_descriptor.json'), input_descriptor);

present_before_verdict = local_file_names(output_dir, {'_RUN_INCOMPLETE'});
required = P.required_artifacts(:);
present_for_contract = [present_before_verdict; ...
    {'vv_verdict.json'; 'manifest.json'}];
missing_qualification_artifacts = setdiff(required, present_for_contract, 'stable');
alignment_failures = P.registered_support_alignment( ...
    ~[P.registered_support_alignment.exactly_aligned_for_qualification]);
verdict = struct();
verdict.schema = 'numerical-vv-verdict-v1';
verdict.run_kind = P.micro_run_kind;
verdict.overall_status = 'UNVERIFIED';
verdict.integrity_status = 'REQUIRES_INDEPENDENT_RECEIPT';
verdict.numerical_verification_claim_authorized = false;
verdict.physical_validation_claim_authorized = false;
verdict.production_resolution_qualified = false;
verdict.dynamic_solver_executed = false;
verdict.completed_mesh_case_ids = {runs.case_id};
verdict.completed_time_grid_level_ids = time_ids;
verdict.missing_qualification_artifacts = missing_qualification_artifacts;
verdict.unresolved_contract_items = P.unresolved_contract_items;
verdict.support_alignment_failure_count = numel(alignment_failures);
verdict.claims_can_support = { ...
    'harness execution and artifact-integrity plumbing'; ...
    'nonqualifying simply-supported static/modal trend inspection'; ...
    'requested-versus-actual B11 time-grid recording'};
verdict.claims_cannot_support = { ...
    'coupled TTBI mesh convergence'; ...
    'bridge-response time convergence'; ...
    'production-resolution qualification'; ...
    'damage-model validation'; ...
    'field validation'};
local_write_json(fullfile(output_dir, 'vv_verdict.json'), verdict);

artifact_names = local_file_names(output_dir, {'_RUN_INCOMPLETE'});
artifact_inventory = local_artifact_inventory(output_dir, artifact_names);
elapsed_s = toc(clock_start);
end_utc = local_utc_now();
if isempty(opt.Command)
    command = sprintf([ ...
        'numerical_vv_micro_run("%s", "BridgeLengthsM", [%s], ' ...
        '"MeshLevelIDs", {%s}, "TimeLevelIDs", {%s})'], ...
        output_dir, strjoin(compose('%.17g', lengths), ' '), ...
        strjoin(compose('"%s"', string(mesh_ids)), ','), ...
        strjoin(compose('"%s"', string(time_ids)), ','));
else
    command = char(opt.Command);
end

manifest = struct();
manifest.schema = 'numerical-vv-manifest-v1';
manifest.protocol_schema = P.schema;
manifest.run_kind = P.micro_run_kind;
manifest.status = 'NON_QUALIFYING_COMPLETE';
manifest.numerical_verification_claim_authorized = false;
manifest.physical_validation_claim_authorized = false;
manifest.source_commit = source.commit;
manifest.source_clean = source.clean;
manifest.source_identity_status = source.status;
manifest.source_identity_scope = ...
    'development snapshot at run start; not a publication attestation';
manifest.source_files = source.files;
manifest.environment = local_environment_identity();
manifest.host = getenv('COMPUTERNAME');
manifest.command = command;
manifest.start_utc = start_utc;
manifest.end_utc = end_utc;
manifest.elapsed_s = elapsed_s;
manifest.input_hash = input_hash;
manifest.input_descriptor = input_descriptor;
manifest.completed_mesh_case_count = n_mesh_cases;
manifest.completed_time_grid_count = height(time_rows);
manifest.artifacts = artifact_inventory;
manifest.required_qualification_artifacts = P.required_artifacts;
manifest.qualifying_contract_ready = P.qualification_ready;
manifest.support_alignment_policy_status = ...
    P.support_alignment_policy.current_policy_status;
local_write_json(fullfile(output_dir, 'manifest.json'), manifest);

manifest_sha = numerical_vv_file_sha256( ...
    fullfile(output_dir, 'manifest.json'));
delete(incomplete_path);
complete_path = fullfile(output_dir, '_RUN_COMPLETE');
local_write_text(complete_path, sprintf([ ...
    'schema=numerical-vv-completion-v1\n' ...
    'run_kind=%s\nstatus=NON_QUALIFYING_COMPLETE\nmanifest_sha256=%s\n'], ...
    P.micro_run_kind, manifest_sha));

result = numerical_vv_validate_package(output_dir, ...
    'AllowNonqualifyingMicro', true);
end

function rows = local_time_grid_rows(P, ids, time_window_s, source_commit, input_hash)
n = numel(ids);
case_id = strings(n, 1);
geometry_id = repmat("SYNTHETIC_TIME_GRID", n, 1);
mesh_level = repmat("NOT_APPLICABLE", n, 1);
requested_dt_ms = zeros(n, 1);
actual_dt_ms = zeros(n, 1);
n_samples = zeros(n, 1);
t_end_s = zeros(n, 1);
response_solved = false(n, 1);
status = repmat("GRID_ONLY_NONQUALIFYING", n, 1);
for k = 1:n
    index = strcmp({P.time_levels.id}, ids{k});
    requested_dt_ms(k) = P.time_levels(index).requested_dt_ms;
    Calc = struct();
    Calc.Solver.max_accurate_frq = ...
        1/(2*requested_dt_ms(k)*1e-3);
    Calc.Time.t_end = time_window_s;
    Calc.Position = struct('v_0', 80/3.6, 'a_0', 0, 'aa', 2, ...
        'x_0', 0, 'x_end', (80/3.6)*time_window_s, ...
        'v_max', 80/3.6, 'a_min', 0, 'a_max', 0);
    Calc.Profile.L = Calc.Position.x_end;
    Calc.Plot.P3_VehPos = 0;
    Calc.Cte.tol = 1e-6;
    Calc = B11_TimeSpaceDiscretization(Calc);
    case_id(k) = "synthetic_"+string(ids{k});
    actual_dt_ms(k) = Calc.Solver.dt*1000;
    n_samples(k) = Calc.Solver.num_t;
    t_end_s(k) = Calc.Solver.t(end);
end
source_commit = repmat(string(source_commit), n, 1);
input_hash = repmat(string(input_hash), n, 1);
units = repmat("ms,samples,s", n, 1);
rows = table(case_id, geometry_id, mesh_level, requested_dt_ms, ...
    actual_dt_ms, n_samples, t_end_s, response_solved, status, ...
    source_commit, input_hash, units);
end

function [case_table, descriptor_table, static_table, modal_table, scalar_table] = ...
        local_tables(runs, source_commit_value, input_hash_value)
n = numel(runs);
case_id = strings(n, 1);
geometry_id = strings(n, 1);
study_kind = repmat("simply_supported_bridge_fixture", n, 1);
mesh_level = strings(n, 1);
bridge_length_m = zeros(n, 1);
bridge_elements_per_sleeper_bay = zeros(n, 1);
rail_elements_per_sleeper_bay = zeros(n, 1);
bridge_nominal_h_m = zeros(n, 1);
rail_nominal_h_m = zeros(n, 1);
bridge_actual_h_m = zeros(n, 1);
rail_actual_h_m = nan(n, 1);
bridge_mesh_executed = true(n, 1);
rail_mesh_executed = false(n, 1);
bridge_n_elements = zeros(n, 1);
bridge_n_nodes = zeros(n, 1);
max_support_offset_m = zeros(n, 1);
support_alignment_pass = false(n, 1);
source_commit = repmat(string(source_commit_value), n, 1);
input_hash = repmat(string(input_hash_value), n, 1);
units = repmat("m,count", n, 1);
descriptor_hash = strings(n, 1);
descriptor_json = strings(n, 1);
descriptor_schema = repmat("numerical-vv-bridge-fixture-v1", n, 1);
for k = 1:n
    case_id(k) = string(runs(k).case_id);
    geometry_id(k) = string(runs(k).geometry_id);
    mesh_level(k) = string(runs(k).mesh_level);
    bridge_length_m(k) = runs(k).bridge_length_m;
    bridge_elements_per_sleeper_bay(k) = ...
        runs(k).bridge_elements_per_sleeper_bay;
    rail_elements_per_sleeper_bay(k) = ...
        runs(k).rail_elements_per_sleeper_bay;
    bridge_nominal_h_m(k) = runs(k).bridge_nominal_h_m;
    rail_nominal_h_m(k) = runs(k).rail_nominal_h_m;
    bridge_actual_h_m(k) = runs(k).bridge_actual_h_m;
    rail_actual_h_m(k) = runs(k).rail_actual_h_m;
    bridge_mesh_executed(k) = runs(k).bridge_mesh_executed;
    rail_mesh_executed(k) = runs(k).rail_mesh_executed;
    bridge_n_elements(k) = runs(k).bridge_n_elements;
    bridge_n_nodes(k) = runs(k).bridge_n_nodes;
    max_support_offset_m(k) = max(abs(runs(k).support_signed_offsets_m));
    support_alignment_pass(k) = runs(k).support_alignment_pass;
    descriptor = struct('case_id', runs(k).case_id, ...
        'geometry_id', runs(k).geometry_id, ...
        'mesh_level', runs(k).mesh_level, ...
        'bridge_length_m', runs(k).bridge_length_m, ...
        'bridge_elements_per_sleeper_bay', ...
            runs(k).bridge_elements_per_sleeper_bay, ...
        'rail_elements_per_sleeper_bay', ...
            runs(k).rail_elements_per_sleeper_bay, ...
        'bridge_nominal_h_m', runs(k).bridge_nominal_h_m, ...
        'rail_nominal_h_m', runs(k).rail_nominal_h_m, ...
        'rail_mesh_executed', runs(k).rail_mesh_executed, ...
        'point_load_N', runs(k).point_load_N, ...
        'scope', runs(k).scope);
    descriptor_json(k) = string(jsonencode(descriptor));
    descriptor_hash(k) = string(numerical_vv_sha256_bytes( ...
        unicode2native(char(descriptor_json(k)), 'UTF-8')));
end
case_table = table(case_id, geometry_id, study_kind, mesh_level, ...
    bridge_length_m, bridge_elements_per_sleeper_bay, ...
    rail_elements_per_sleeper_bay, bridge_nominal_h_m, rail_nominal_h_m, ...
    bridge_actual_h_m, rail_actual_h_m, bridge_mesh_executed, ...
    rail_mesh_executed, bridge_n_elements, bridge_n_nodes, ...
    max_support_offset_m, support_alignment_pass, source_commit, ...
    input_hash, units);
descriptor_table = table(case_id, geometry_id, mesh_level, descriptor_schema, ...
    descriptor_json, descriptor_hash, source_commit, input_hash, ...
    repmat("SHA-256", n, 1), 'VariableNames', { ...
    'case_id', 'geometry_id', 'mesh_level', 'descriptor_schema', ...
    'descriptor_json', 'descriptor_hash', 'source_commit', 'input_hash', ...
    'units'});

static_table = table(case_id, geometry_id, mesh_level, ...
    [runs.bridge_length_m]', [runs.point_load_N]', ...
    [runs.midspan_displacement_m]', ...
    [runs.analytic_midspan_displacement_m]', [runs.left_reaction_N]', ...
    [runs.right_reaction_N]', [runs.analytic_reaction_N]', ...
    [runs.force_balance_residual_N]', [runs.static_residual_norm_N]', ...
    [runs.strain_energy_J]', [runs.external_work_J]', ...
    [runs.energy_residual_J]', ...
    repmat("REPORTED_NO_ACCEPTANCE_GATE", n, 1), source_commit, ...
    input_hash, repmat("m,N,J", n, 1), 'VariableNames', { ...
    'case_id', 'geometry_id', 'mesh_level', 'bridge_length_m', ...
    'point_load_N', 'midspan_displacement_m', ...
    'analytic_midspan_displacement_m', 'left_reaction_N', ...
    'right_reaction_N', 'analytic_reaction_N', ...
    'force_balance_residual_N', 'static_residual_norm_N', ...
    'strain_energy_J', 'external_work_J', 'energy_residual_J', ...
    'status', 'source_commit', 'input_hash', 'units'});

n_modal = n*5;
modal_case = strings(n_modal, 1);
modal_geometry = strings(n_modal, 1);
modal_mesh = strings(n_modal, 1);
mode_number = zeros(n_modal, 1);
frequency_hz = zeros(n_modal, 1);
analytic_frequency_hz = zeros(n_modal, 1);
relative_frequency_error = zeros(n_modal, 1);
mac_vs_analytic = zeros(n_modal, 1);
row = 0;
for k = 1:n
    for mode = 1:5
        row = row+1;
        modal_case(row) = case_id(k);
        modal_geometry(row) = geometry_id(k);
        modal_mesh(row) = mesh_level(k);
        mode_number(row) = mode;
        frequency_hz(row) = runs(k).frequency_hz(mode);
        analytic_frequency_hz(row) = runs(k).analytic_frequency_hz(mode);
        relative_frequency_error(row) = abs(frequency_hz(row)- ...
            analytic_frequency_hz(row))/analytic_frequency_hz(row);
        mac_vs_analytic(row) = runs(k).mac_vs_analytic(mode);
    end
end
modal_table = table(modal_case, modal_geometry, modal_mesh, mode_number, ...
    frequency_hz, analytic_frequency_hz, relative_frequency_error, ...
    mac_vs_analytic, repmat("REPORTED_NO_ACCEPTANCE_GATE", n_modal, 1), ...
    repmat(string(source_commit_value), n_modal, 1), ...
    repmat(string(input_hash_value), n_modal, 1), ...
    repmat("Hz,1", n_modal, 1), 'VariableNames', { ...
    'case_id', 'geometry_id', 'mesh_level', 'mode_number', ...
    'frequency_hz', 'analytic_frequency_hz', ...
    'relative_frequency_error', 'mac_vs_analytic', 'status', ...
    'source_commit', 'input_hash', 'units'});

qoi_ids = {'midspan_displacement'; 'left_support_reaction'; ...
    'right_support_reaction'; 'strain_energy'};
qoi_units = {'m'; 'N'; 'N'; 'J'};
qoi_count = n*numel(qoi_ids);
scalar_case = strings(qoi_count, 1);
scalar_geometry = strings(qoi_count, 1);
scalar_mesh = strings(qoi_count, 1);
qoi_id = strings(qoi_count, 1);
value = zeros(qoi_count, 1);
reference_value = zeros(qoi_count, 1);
unit = strings(qoi_count, 1);
row = 0;
for k = 1:n
    values = [runs(k).midspan_displacement_m, runs(k).left_reaction_N, ...
        runs(k).right_reaction_N, runs(k).strain_energy_J];
    references = [runs(k).analytic_midspan_displacement_m, ...
        runs(k).analytic_reaction_N, runs(k).analytic_reaction_N, ...
        runs(k).external_work_J];
    for q = 1:numel(qoi_ids)
        row = row+1;
        scalar_case(row) = case_id(k);
        scalar_geometry(row) = geometry_id(k);
        scalar_mesh(row) = mesh_level(k);
        qoi_id(row) = string(qoi_ids{q});
        value(row) = values(q);
        reference_value(row) = references(q);
        unit(row) = string(qoi_units{q});
    end
end
relative_error = abs(value-reference_value) ./ ...
    max(abs(reference_value), realmin('double'));
scalar_table = table(scalar_case, scalar_geometry, scalar_mesh, qoi_id, ...
    value, reference_value, relative_error, unit, ...
    repmat("REPORTED_NO_ACCEPTANCE_GATE", qoi_count, 1), ...
    repmat(string(source_commit_value), qoi_count, 1), ...
    repmat(string(input_hash_value), qoi_count, 1), 'VariableNames', { ...
    'case_id', 'geometry_id', 'mesh_level', 'qoi_id', 'value', ...
    'reference_value', 'relative_error', 'units', 'status', ...
    'source_commit', 'input_hash'});
end

function T = local_tolerance_table(P, source_commit, input_hash)
check_id = [ ...
    "gci_safety_factor"; ...
    "contact_coarse_to_fine_nrmse"; ...
    "contact_coarse_to_fine_nmax"; ...
    "contact_coarse_to_fine_correlation"; ...
    "contact_medium_to_fine_nrmse"; ...
    "contact_medium_to_fine_nmax"; ...
    "contact_medium_to_fine_correlation"; ...
    "contact_positive_tensile_peak"; ...
    "contact_on_track_tensile_fraction"];
numeric_value = [P.gci_safety_factor; ...
    P.contact_policy.coarse_to_fine_nrmse; ...
    P.contact_policy.coarse_to_fine_nmax; ...
    P.contact_policy.coarse_to_fine_correlation; ...
    P.contact_policy.medium_to_fine_nrmse; ...
    P.contact_policy.medium_to_fine_nmax; ...
    P.contact_policy.medium_to_fine_correlation; ...
    P.contact_policy.positive_tensile_peak_N; ...
    P.contact_policy.on_track_tensile_fraction];
units = ["1"; "1"; "1"; "1"; "1"; "1"; "1"; "N"; "1"];
evidence_class = ["author-chosen-convention"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"; ...
    "author-chosen-engineering-closure"];
applied_in_this_micro = false(size(numeric_value));
basis = repmat("source-locked protocol; not exercised by this micro", ...
    size(numeric_value));
source_commit = repmat(string(source_commit), size(numeric_value));
input_hash = repmat(string(input_hash), size(numeric_value));
case_id = repmat("PROTOCOL", size(numeric_value));
geometry_id = repmat("ALL", size(numeric_value));
T = table(case_id, geometry_id, check_id, numeric_value, units, ...
    evidence_class, basis, applied_in_this_micro, source_commit, input_hash);
end

function source = local_source_identity()
matlab_root = fileparts(mfilename('fullpath'));
repo_root = fileparts(matlab_root);
[commit_status, commit_text] = system(sprintf( ...
    'git -C "%s" rev-parse HEAD', repo_root));
[dirty_status, dirty_text] = system(sprintf( ...
    'git -C "%s" status --porcelain --untracked-files=normal', repo_root));
if commit_status == 0
    commit = strtrim(commit_text);
else
    commit = 'UNAVAILABLE';
end
clean = commit_status == 0 && dirty_status == 0 && isempty(strtrim(dirty_text));
if commit_status == 0 && dirty_status == 0
    status = 'RECORDED';
else
    status = 'UNAVAILABLE';
end
names = { ...
    'numerical_vv_protocol_definition.m'; ...
    'numerical_vv_support_alignment.m'; ...
    'numerical_vv_bridge_fixture.m'; ...
    'numerical_vv_micro_run.m'; ...
    'numerical_vv_validate_package.m'; ...
    'numerical_vv_sha256_bytes.m'; ...
    'numerical_vv_file_sha256.m'; ...
    'A03_Bridge.m'; ...
    'B01_ElementsAndCoordinates.m'; ...
    'B02_BoundaryConditions.m'; ...
    'B03_BeamMatrices.m'; ...
    'B11_TimeSpaceDiscretization.m'};
files = repmat(struct('path', '', 'sha256', ''), numel(names), 1);
for k = 1:numel(names)
    path = fullfile(matlab_root, names{k});
    files(k).path = strrep(fullfile('scour_MATLAB', names{k}), '\', '/');
    files(k).sha256 = numerical_vv_file_sha256(path);
end
source = struct('commit', commit, 'clean', clean, ...
    'status', status, 'files', files);
end

function environment = local_environment_identity()
toolboxes = ver;
toolbox_rows = repmat(struct('name', '', 'version', ''), numel(toolboxes), 1);
for k = 1:numel(toolboxes)
    toolbox_rows(k).name = toolboxes(k).Name;
    toolbox_rows(k).version = toolboxes(k).Version;
end
environment = struct('matlab_version', version, ...
    'matlab_release', version('-release'), 'architecture', computer('arch'), ...
    'operating_system', system_dependent('getos'), 'toolboxes', toolbox_rows);
end

function inventory = local_artifact_inventory(root, names)
inventory = repmat(struct('path', '', 'sha256', '', 'bytes', 0), ...
    numel(names), 1);
for k = 1:numel(names)
    path = fullfile(root, names{k});
    info = dir(path);
    inventory(k).path = names{k};
    inventory(k).sha256 = numerical_vv_file_sha256(path);
    inventory(k).bytes = info.bytes;
end
end

function names = local_file_names(root, excluded)
entries = dir(root);
entries = entries(~[entries.isdir]);
names = sort(reshape({entries.name}, [], 1));
names = setdiff(names, excluded, 'stable');
end

function ids = local_registered_ids(values, registered, label)
if isempty(values) || any(~cellfun(@local_text_scalar, values))
    error('numerical_vv:BadLevelSelection', ...
        '%s selection must be a nonempty cell array of text scalars.', label);
end
ids = cellfun(@char, values, 'UniformOutput', false);
if numel(unique(ids, 'stable')) ~= numel(ids) || ...
        any(~ismember(ids, registered))
    error('numerical_vv:UnregisteredLevelSelection', ...
        '%s selection contains duplicates or unregistered IDs.', label);
end
end

function local_write_json(path, value)
local_write_text(path, [jsonencode(value, 'PrettyPrint', true), newline]);
end

function local_write_text(path, value)
bytes = unicode2native(char(value), 'UTF-8');
fid = fopen(path, 'wb');
if fid < 0
    error('numerical_vv:FileWrite', 'Could not open %s for writing.', path);
end
cleanup = onCleanup(@() fclose(fid));
written = fwrite(fid, bytes, 'uint8');
if written ~= numel(bytes)
    error('numerical_vv:FileWrite', 'Short write to %s.', path);
end
end

function path = local_absolute_path(path)
path = char(java.io.File(path).getCanonicalPath());
end

function text_value = local_utc_now()
text_value = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSXXX'));
end

function tf = local_text_scalar(value)
tf = ischar(value) || (isstring(value) && isscalar(value));
end

function tf = local_positive_scalar(value)
tf = isnumeric(value) && isreal(value) && isscalar(value) && ...
    isfinite(value) && value > 0;
end

function tf = local_positive_integer(value)
tf = local_positive_scalar(value) && value == floor(value);
end
