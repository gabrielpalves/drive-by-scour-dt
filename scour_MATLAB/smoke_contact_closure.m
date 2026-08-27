% smoke_contact_closure.m
% Fast, self-contained checks for the Paper-1 contact-closure harness:
%   1. exact named-stream/state identity and diagnostic gate logic;
%   2. fail-closed mutations of the current identity/vehicle contracts;
%   3. B66's whole-track mask excludes pre-entry and post-exit samples.
% No production dataset/result/bundle is read or modified.
clear; clc;

fixture_dir = tempname;
mkdir(fixture_dir);
fixture_cleanup = onCleanup(@() rmdir(fixture_dir, 's'));
report_dir = tempname;
mkdir(report_dir);
report_cleanup = onCleanup(@() rmdir(report_dir, 's'));

[source_root, ~, ~] = generator_source_root();
[environment_sha, environment_descriptor] = matlab_environment_identity( ...
    current_matlab_environment());
policy_api = contact_gate_policy();
policy_api.validate_locked_matlab_environment( ...
    environment_sha, environment_descriptor);
fprintf('MATLAB capabilities/reference integrity and live provenance: PASS\n');
stream_names = {'operations', 'crack', 'profile-state', 'track', ...
    'profile-phase'};
state_uid = 'fixture-joint-state-0001';
state_family = 'joint';
state_seed_id = uint32(314159);
named_seed_row = uint32([101, 202, 303, 404, 505]);

case_info = struct( ...
    'case_name', 'contact_closure_fixture', ...
    'stage', 'smoke', ...
    'gen_schema', 'audit-2026-08-09-r12', ...
    'generation_behavior_version', 'generation-rules-v8', ...
    'channel_schema_id', 'physical8_v1', ...
    'gen_fingerprint', repmat('0', 1, 64), ...
    'L_bridge_m', 60, ...
    'num_spans', 3, ...
    'n_states', 1, ...
    'passages_per_state', 2, ...
    'n_vehicles', 5, ...
    'n_props_varied', 3, ...
    'profile_mode', 'fixed', ...
    'profile_jitter_sd_mm', 0, ...
    'use_signal_noise', false, ...
    'random_stream_schedule_version', 'uid-named-substreams-v2', ...
    'state_stream_names', strjoin(stream_names, ','), ...
    'generator_source_root_sha256', source_root, ...
    'actual_matlab_environment_sha256', environment_sha, ...
    'oor_radius', 0.46);
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');

DamageStates = [0, 0.2, 0, 0];
BearingStates = [0, 0];
StateUID = {state_uid};
StateFamily = {state_family};
StateSeedID = state_seed_id;
StateNamedStreamSeedID = named_seed_row;
state_stream_names = stream_names;
random_stream_schedule_version = 'uid-named-substreams-v2';
save(fullfile(fixture_dir, 'damage_states.mat'), ...
    'DamageStates', 'BearingStates', 'StateUID', 'StateFamily', ...
    'StateSeedID', 'StateNamedStreamSeedID', 'state_stream_names', ...
    'random_stream_schedule_version');

n_passages = 2;
data = struct();
data.gen_schema = case_info.gen_schema;
data.generation_behavior_version = case_info.generation_behavior_version;
data.channel_schema_id = case_info.channel_schema_id;
data.state_uid = state_uid;
data.state_family = state_family;
data.state_seed_id = state_seed_id;
data.random_stream_schedule_version = ...
    case_info.random_stream_schedule_version;
data.state_named_stream_seed_id = named_seed_row;
data.Velocidade = [80, 81] / 3.6;
data.Temperatura = [20, 21];
data.VehiclesProps = zeros(5, 3, n_passages);
data.scour_vector = [0, 0.2, 0, 0];
data.bearing_vector = [0, 0];
data.crack_log = zeros(n_passages, 3);
data.profile_mode = 'fixed';
data.profile_log = ones(n_passages, 1);
data.track_log = cell(n_passages, 1);
data.oor_log = cell(n_passages, 1);
data.contact_log = [0, 1, 0.001, 13000; 0, 0, 0, -100000];
state_path = fullfile(fixture_dir, '0001.mat');
save(state_path, 'data');

digest_names = {'0001.mat'; 'case_info.mat'; 'damage_states.mat'};
digest_lines_cell = cell(size(digest_names));
for digest_index = 1:numel(digest_names)
    digest_lines_cell{digest_index} = sprintf('%s:%s', ...
        digest_names{digest_index}, ...
        local_file_sha256(fullfile(fixture_dir, ...
            digest_names{digest_index})));
end
digest_lines = strjoin(sort(digest_lines_cell), newline);
file_digests = struct( ...
    'schema', 'source-digests-v2', ...
    'scope', 'NNNN.mat+case_info.mat+damage_states.mat', ...
    'digest_lines', digest_lines, ...
    'root', local_text_sha256(digest_lines));
digest_path = fullfile(fixture_dir, 'file_digests.mat');
save(digest_path, 'file_digests');
marker_path = fullfile(fixture_dir, '_GENERATION_COMPLETE');
fid = fopen(marker_path, 'wb');
assert(fid >= 0);
marker_text = sprintf('%s\n%s\n%s\n', ...
    case_info.gen_schema, case_info.gen_fingerprint, file_digests.root);
assert(fwrite(fid, unicode2native(marker_text, 'UTF-8'), 'uint8') == ...
    numel(unicode2native(marker_text, 'UTF-8')));
assert(fclose(fid) == 0);

strict_manifest = validate_dataset_digest_manifest( ...
    fixture_dir, 1, 'RetainSnapshots', true);
assert(strcmp(strict_manifest.root, file_digests.root));
assert(isequal(strict_manifest.verified_state_indices, 1));
assert(numel(strict_manifest.retained_snapshots) == 4);
case_snapshot_probe = contact_named_file_snapshot( ...
    strict_manifest.retained_snapshots, 'case_info.mat');
state_snapshot_probe = contact_named_file_snapshot( ...
    strict_manifest.retained_snapshots, '0001.mat');
states_snapshot_probe = contact_named_file_snapshot( ...
    strict_manifest.retained_snapshots, 'damage_states.mat');
digests_snapshot_probe = contact_named_file_snapshot( ...
    strict_manifest.retained_snapshots, 'file_digests.mat');
case_snapshot_blob = contact_load_mat_bytes(case_snapshot_probe.bytes);
state_snapshot_blob = contact_load_mat_bytes(state_snapshot_probe.bytes);
states_snapshot_blob = contact_load_mat_bytes(states_snapshot_probe.bytes);
assert(strcmp(case_snapshot_blob.case_info.case_name, case_info.case_name));
assert(strcmp(state_snapshot_blob.data.state_uid, state_uid));
assert(isequal(states_snapshot_blob.DamageStates, DamageStates));
assert(strcmp(digests_snapshot_probe.sha256, ...
    strict_manifest.file_digests_sha256));
assert(isequal(digests_snapshot_probe, ...
    strict_manifest.file_digests_snapshot));
contact_assert_snapshot_set_unchanged( ...
    strict_manifest.retained_snapshots);
fprintf('strict v2 digest manifest: PASS\n');

% The original dataset spelling is a trust-boundary input. A symlink/junction
% must fail before contact_absolute_path can erase that evidence.
fixture_alias = [fixture_dir, '_alias'];
ttbi.create_directory_alias(fixture_alias, fixture_dir);
fixture_alias_cleanup = onCleanup( ...
    @() ttbi.delete_file_entry_if_present(fixture_alias));
local_assert_throws( ...
    @() validate_dataset_digest_manifest(fixture_alias, 1), ...
    'dataset_digest_manifest:LinkedDataset');
local_assert_throws(@() contact_closure_study( ...
    fixture_alias, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:LinkedDataset');
clear fixture_alias_cleanup
ttbi.delete_file_entry_if_present(fixture_alias);
fprintf('dataset-root alias guards: PASS\n');

% Manifest parsing consumes the authenticated byte snapshot. Mutating the live
% path afterwards cannot change the parsed snapshot, and final reassertion
% detects that the path no longer represents the observed file.
[manifest_bytes, manifest_observation] = ...
    contact_stable_file_bytes(digest_path);
manifest_sha256 = contact_bytes_sha256(manifest_bytes);
file_digests_base = file_digests;
file_digests.root = repmat('f', 1, 64);
save(digest_path, 'file_digests');
snapshot_blob = contact_load_mat_bytes(manifest_bytes);
assert(strcmp(snapshot_blob.file_digests.root, file_digests_base.root));
local_assert_throws(@() contact_assert_file_snapshot_unchanged( ...
    digest_path, manifest_observation, manifest_sha256), ...
    'contact_snapshot:FileRace');
file_digests = file_digests_base;
% Restore the exact authenticated bytes. Re-saving an equivalent MAT variable
% changes the MAT-file header timestamp and makes this TOCTOU smoke depend on
% whether both saves happened within the same wall-clock second.
local_write_file_bytes(digest_path, manifest_bytes);
fprintf('stable manifest snapshot/reassertion: PASS\n');

report = contact_closure_study(fixture_dir, 1, 1, ...
    'VerifyIntegrity', true, 'DryRun', true, 'OutputDir', report_dir);
assert(strcmp(report.status, 'DRY_RUN_VALIDATED'));
assert(strcmp(report.dataset_integrity.status, 'VERIFIED'));
assert(strcmp(report.state_file_sha256, state_snapshot_probe.sha256));
assert(strcmp(report.dataset_integrity.case_info_sha256, ...
    case_snapshot_probe.sha256));
assert(strcmp(report.dataset_integrity.damage_states_sha256, ...
    states_snapshot_probe.sha256));
assert(strcmp(report.dataset_integrity.file_digests_sha256, ...
    digests_snapshot_probe.sha256));
assert(isempty( ...
    report.dataset_integrity.qualification_host_receipt_sha256));
assert(report.passage_index == 1);
assert(isequal(report.saved_gate_pass, [false, false, true]));
assert(report.descriptor.velocity_kmh == 80);
assert(isequal(report.descriptor.scour_vector, [0, 0.2, 0, 0]));
assert(report.profile_phase_stream_index == 5);
assert(report.profile_phase_seed == double(named_seed_row(5)));
assert(isscalar(dir(fullfile(report_dir, '*.md'))));
assert(isscalar(dir(fullfile(report_dir, '*.mat'))));
fprintf('R11 descriptor/named seed/gates: PASS\n');

% The manifest must authenticate sidecar bytes, not merely its own root.
case_info_base = case_info;
case_info_path = fullfile(fixture_dir, 'case_info.mat');
case_info_backup_path = fullfile(fixture_dir, 'case_info.bytes.bak');
assert(copyfile(case_info_path, case_info_backup_path));
case_info_mutated = case_info;
case_info_mutated.case_name = 'corrupted-after-digest';
case_info = case_info_mutated;
save(case_info_path, 'case_info');
local_assert_throws(@() validate_dataset_digest_manifest(fixture_dir, 1), ...
    'dataset_digest_manifest:DigestMismatch');
assert(copyfile(case_info_backup_path, case_info_path, 'f'));
case_info = case_info_base;

digest_parts = strsplit( ...
    file_digests.digest_lines, newline, 'CollapseDelimiters', false);
file_digests.digest_lines = strjoin(digest_parts(2:end), newline);
file_digests.root = local_text_sha256(file_digests.digest_lines);
save(digest_path, 'file_digests');
local_assert_throws(@() validate_dataset_digest_manifest(fixture_dir, 1), ...
    'dataset_digest_manifest:BadInventory');
file_digests = file_digests_base;
file_digests.digest_lines = sprintf('%s\nextra.mat:%s', ...
    file_digests.digest_lines, repmat('0', 1, 64));
file_digests.root = local_text_sha256(file_digests.digest_lines);
save(digest_path, 'file_digests');
local_assert_throws(@() validate_dataset_digest_manifest(fixture_dir, 1), ...
    'dataset_digest_manifest:BadInventory');
file_digests = file_digests_base;
local_write_file_bytes(digest_path, manifest_bytes);
fprintf('v2 sidecar/inventory mutation guards: PASS\n');

% Named streams are semantic identities: duplication or omission must fail.
case_info.state_stream_names = ...
    'operations,crack,profile-state,track,profile-phase,profile-phase';
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');
local_assert_throws(@() contact_closure_study( ...
    fixture_dir, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:BadNamedStreams');
case_info = case_info_base;
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');

case_info.state_stream_names = 'operations,crack,profile-state,track';
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');
local_assert_throws(@() contact_closure_study( ...
    fixture_dir, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:BadNamedStreams');
case_info = case_info_base;
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');
fprintf('named-stream mutation guards: PASS\n');

% The state payload must match the independently persisted identity table.
data_base = data;
data.state_named_stream_seed_id(5) = uint32(506);
save(state_path, 'data');
local_assert_throws(@() contact_closure_study( ...
    fixture_dir, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:StateIdentityMismatch');
data = data_base;
save(state_path, 'data');
fprintf('state-identity mutation guard: PASS\n');

% Shape checks are exact, not merely based on element count, and NaN/Inf
% vehicle draws are rejected before any solver work.
data.VehiclesProps = zeros(1, 15, n_passages);
save(state_path, 'data');
local_assert_throws(@() contact_closure_study( ...
    fixture_dir, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:BadVehicleProps');
data = data_base;
data.VehiclesProps(1, 1, 1) = NaN;
save(state_path, 'data');
local_assert_throws(@() contact_closure_study( ...
    fixture_dir, 1, 1, 'VerifyIntegrity', false, 'DryRun', true), ...
    'contact_closure:BadVehicleProps');
data = data_base;
save(state_path, 'data');
fprintf('vehicle-tensor mutation guards: PASS\n');

% The semantic named seed must be the exact value handed to B19; row/state
% arithmetic is forbidden. After the study split the assignment lives in
% contact_profile_descriptor.m and must appear exactly once across the
% WHOLE study executable set. The occurrence counts are taken over
% EXECUTABLE STATEMENTS ONLY (local_matlab_statements below mirrors
% _matlab_statements in check_contact_closure_gate.py), because the module's
% own rationale header quotes the assignment verbatim: counting raw text
% would make this smoke disagree with the authoritative Python guard, which
% strips comments before evaluating the same pin. Exercise B19 itself
% without running the coupled bridge solver.
profile_descriptor_source = local_matlab_statements( ...
    fileread(which('contact_profile_descriptor')));
% Read the STUDY EXECUTABLE SET from its single source of truth rather than
% relisting it here: a hand-copied list silently stops covering the set the
% moment a member is added (which is exactly how the transitive-closure gap
% in P1-2 survived).
common_api = contact_closure_common();
study_set_names = common_api.study_harness_files();
study_parts = cell(1, numel(study_set_names));
for k = 1:numel(study_set_names)
    [~, base_] = fileparts(study_set_names{k});
    study_parts{k} = local_matlab_statements(fileread(which(base_)));
end
study_set_sources = strjoin(study_parts, newline);
phase_assignment = ...
    'cfg.phase_seed = descriptor_contract.profile_phase_seed;';
assert(isscalar(strfind(profile_descriptor_source, phase_assignment)));
assert(isscalar(strfind(study_set_sources, phase_assignment)));
assert(~contains(study_set_sources, ...
    '1e9 + damage_seed * 100000 + state_index'));
% EXECUTED-MODULE COMPLETENESS. Ask MATLAB itself which reviewed files the
% study and gate entries depend on, and require every one to be a member of a
% DECLARED inventory. This is what turns the R11 inventories from a hand-kept
% list into a checked one: an ordinary dependency added anywhere in the chain
% now fails this smoke instead of silently executing unhashed.
%
% All production dependencies are ordinary function calls. Dynamic dispatch is
% forbidden by the independent source contract, so requiredFilesAndProducts
% must now discover the complete executable closure without a manual blind spot.
declared = [contact_solver_modules(); ...
    strrep(common_api.study_harness_files(), '.m', ''); ...
    strrep(common_api.gate_module_files(), '.m', '')];
% Data assets are not executable modules; this one is reachable only from the
% retired Type-2 branch of B19 (A04_Options unconditionally sets Type 1 for
% every registered rung), so it is allowlisted rather than hashed as a module.
allowed_assets = {'Calc.ProfileData15_05.mat'};
reviewed_dir = fileparts(mfilename('fullpath'));
closure_entries = {'contact_closure_study.m', 'contact_closure_gate.m'};
closure_parts = cell(1, numel(closure_entries));
for k = 1:numel(closure_entries)
    closure_parts{k} = ...
        matlab.codetools.requiredFilesAndProducts(closure_entries{k});
end
static_closure = [closure_parts{:}];
undeclared = cell(1, numel(static_closure));
n_undeclared = 0;
for k = 1:numel(static_closure)
    [dep_dir, dep_name, dep_ext] = fileparts(static_closure{k});
    [dep_qualified, inside] = local_reviewed_member_name( ...
        dep_dir, dep_name, reviewed_dir);
    if ~inside
        continue  % outside the reviewed boundary: covered by MATLAB itself
    end
    if any(strcmp([dep_name, dep_ext], allowed_assets))
        continue
    end
    if ~any(strcmp(dep_qualified, declared))
        n_undeclared = n_undeclared + 1;
        undeclared{n_undeclared} = [dep_qualified, dep_ext];
    end
end
undeclared = undeclared(1:n_undeclared);
assert(isempty(undeclared), 'smoke_contact_closure:UndeclaredDependency', ...
    ['Executed reviewed files absent from every declared inventory: %s. ' ...
     'Add them to contact_solver_modules / study_harness_files / the gate ' ...
     'module list (and their Python mirrors) so they are path-checked and ' ...
     'hashed.'], strjoin(unique(undeclared), ', '));
fprintf('executed-module completeness (%d declared): PASS\n', ...
    numel(unique(declared)));

% SHADOW RESOLUTION. Declaring a module only helps if a same-named file
% earlier on the MATLAB path makes the run fail instead of executing while the
% evidence keeps quoting the reviewed blob. Drive the two identity guards
% DIRECTLY rather than through a full study run: only `which` is consulted, so
% no stub behaviour can mask the path check by erroring or succeeding first.
%
% One probe per resolution class, which is what the classes are for: a
% classic solver module, both directly-called property additions, an
% always-executed provenance helper, and a study orchestrator. Every member of
% a given class is resolved by the identical loop over its inventory, so
% per-member repetition would only re-test the same resolution rule.
% The probe MUST reproduce the production working-directory configuration.
% MATLAB resolves the current folder BEFORE the search path, so while this
% smoke runs inside the reviewed directory no path entry can shadow anything
% and every probe would pass vacuously. Production is different: the study and
% gate are invoked with scour_MATLAB on the path from whatever folder the
% operator happens to be in, and there path order decides. So each probe runs
% from a neutral temporary folder with the reviewed directory on the path and
% the impostor ahead of it - the only configuration in which shadowing is
% actually reachable, and therefore the only one worth asserting against.
solver_api = contact_study_solver();
[~, generator_digest_lines_probe, ~] = generator_source_root();
probe_home = pwd;
probe_path = path;
neutral_dir = tempname;
mkdir(neutral_dir);
shadow_cleanup = onCleanup(@() local_restore_probe_env( ...
    probe_home, probe_path, neutral_dir));
shadow_probes = { ...
    'B66_ContactForce',                           'solver'; ...
    'TrainProp_ObrienCalibrate',                   'solver'; ...
    'TrackProp_Zhai_et_al_WithBallastOnBridge',    'solver'; ...
    'validate_dataset_digest_manifest',            'harness'; ...
    'contact_study_metrics',                       'harness'; ...
    'contact_gate_acceptance',                     'gate'; ...
    'contact_assert_reviewed_bootstrap',            'gate_bootstrap'; ...
    'contact_assert_reviewed_bootstrap',            'study_bootstrap'};
for k = 1:size(shadow_probes, 1)
    probe_name = shadow_probes{k, 1};
    probe_class = shadow_probes{k, 2};
    shadow_dir = tempname;
    mkdir(shadow_dir);
    fid = fopen(fullfile(shadow_dir, [probe_name, '.m']), 'w');
    assert(fid >= 0);
    fprintf(fid, 'function varargout = %s(varargin)\n', probe_name);
    fprintf(fid, '%% Shadowing impostor written by smoke_contact_closure.\n');
    fprintf(fid, 'varargout = cell(1, nargout);\n');
    fprintf(fid, 'end\n');
    fclose(fid);
    cd(neutral_dir);
    addpath(reviewed_dir);
    addpath(shadow_dir, '-begin');
    % Confirm the probe is not vacuous: the impostor must actually be what
    % MATLAB now resolves, otherwise a passing assertion proves nothing.
    assert(strcmpi(local_absolute_path_smoke(fileparts(which(probe_name))), ...
        local_absolute_path_smoke(shadow_dir)), ...
        'smoke_contact_closure:VacuousShadowProbe', ...
        'Impostor %s did not win name resolution; probe proves nothing.', ...
        probe_name);
    try
        switch probe_class
            case 'solver'
                local_assert_throws( ...
                    @() solver_api.solver_source_manifest( ...
                        generator_digest_lines_probe), ...
                    'contact_closure:MissingSolverSource');
            case 'harness'
                local_assert_throws(@() common_api.study_harness_root(), ...
                    'contact_closure:HarnessShadowed');
            case 'gate'
                local_assert_throws(@() common_api.gate_execution_root(), ...
                    'contact_closure:GateModuleShadowed');
            case 'gate_bootstrap'
                local_assert_throws(@() contact_closure_gate( ...
                    "", "", "", "", "", ...
                    'SourceCommit', repmat('a', 1, 40)), ...
                    'contact_closure:GateModuleShadowed');
            case 'study_bootstrap'
                local_assert_throws(@() contact_closure_study( ...
                    "", 1, 1), ...
                    'contact_closure:HarnessShadowed');
            otherwise
                error('smoke_contact_closure:BadProbeClass', ...
                    'Unknown shadow probe class %s.', probe_class);
        end
    catch probe_error
        rmpath(shadow_dir);
        cd(probe_home);
        rmdir(shadow_dir, 's');
        rethrow(probe_error);
    end
    rmpath(shadow_dir);
    cd(probe_home);
    rmdir(shadow_dir, 's');
end
path(probe_path);
cd(probe_home);
% Both guards must be green again once the impostors are off the path.
assert(numel(common_api.study_harness_root()) == 64);
restored_manifest = solver_api.solver_source_manifest( ...
    generator_digest_lines_probe);
assert(height(restored_manifest) == numel(contact_solver_modules()));
fprintf('shadowed-module rejection (%d classes): PASS\n', ...
    size(shadow_probes, 1));

profile_calc = struct();
profile_calc.Position.x = 0:0.01:2;
profile_calc.Profile = struct( ...
    'Type', 1, 'min_dx', 0.01, 'L', 2, ...
    'max_WaveLength', 1, 'min_WaveLength', 0.04, ...
    'PSD_Y_fun', @(frequency, inputs) ...
        inputs(1) * ones(size(frequency)), ...
    'inputs', 1e-8, 'phase_seed', double(named_seed_row(5)));
profile_calc.Plot.Profile_original = 0;
rng_before_profile = rng;
generated_a = B19_GenerateProfile(profile_calc);
rng(999, 'twister');
generated_b = B19_GenerateProfile(profile_calc);
profile_calc.Profile.phase_seed = double(named_seed_row(5)) + 1;
generated_c = B19_GenerateProfile(profile_calc);
rng(rng_before_profile);
assert(isequal(generated_a.Profile.h, generated_b.Profile.h));
assert(~isequal(generated_a.Profile.h, generated_c.Profile.h));
fprintf('exact named phase-seed -> B19 path: PASS\n');

% The independent authorization contract relies on a plain, canonical MAT
% projection, current-host attestation at both ends of the run, and an
% explicit QOI-GCI gate.  Keep these source-level guards beside executable
% numerical checks so refactors cannot silently remove one side of the
% MATLAB/Python contract.  After the gate split each token is pinned to
% the specific module file that owns it.
gate_entry_source = fileread(which('contact_closure_gate'));
gate_acceptance_source = fileread(which('contact_gate_accept_report'));
gate_contraction_source = fileread(which('contact_contracts_to_finest'));
gate_summary_source = fileread(which('contact_gate_summary_skeleton'));
gate_plain_report_source = fileread(which('contact_gate_plain_report'));
gate_selection_source = fileread(which('contact_gate_build_selection'));
required_gate_tokens = { ...
    'contact_closure_gate', gate_entry_source, ...
        'closure_host_start = selection_api.closure_host_attestation'; ...
    'contact_closure_gate', gate_entry_source, ...
        'closure_host_end = selection_api.closure_host_attestation'; ...
    'contact_closure_gate', gate_entry_source, ...
        'selection_api.assert_closure_host_matches_datasets'; ...
    'contact_closure_gate', gate_entry_source, ...
        '~isequaln(closure_host_end, closure_host_start)'; ...
    'contact_gate_summary_skeleton', gate_summary_source, ...
        'summary.closure_host_attestation = closure_host;'; ...
    'contact_closure_gate', gate_entry_source, ...
        '''canonical_case'''; ...
    'contact_gate_plain_report', gate_plain_report_source, ...
        'contact_gate_plain_table'; ...
    'contact_gate_build_selection', gate_selection_source, ...
        'contact_load_mat_bytes(case_snapshot.bytes)'; ...
    'contact_gate_build_selection', gate_selection_source, ...
        'contact_load_mat_bytes(state_snapshot.bytes)'; ...
    'contact_gate_build_selection', gate_selection_source, ...
        'contact_validate_completion_marker_snapshot'; ...
    'contact_gate_build_selection', gate_selection_source, ...
        'all_dataset_snapshots{stage_index}'; ...
    'contact_closure_gate', gate_entry_source, ...
        '''selection_records'''; ...
    'contact_closure_gate', gate_entry_source, ...
        '''canonical_policy'''; ...
    'contact_gate_accept_report', gate_acceptance_source, ...
        'qoi_gci_pass'; ...
    'contact_gate_accept_report', gate_acceptance_source, ...
        'acceptance.channel_qoi_gci_all_pass = qoi_gci_all_pass;'; ...
    'contact_gate_accept_report', gate_acceptance_source, ...
        'channel QOI GCI convergence gate failed'; ...
    'contact_contracts_to_finest', gate_contraction_source, ...
        'tol = atol + rtol * max([1; abs(phi)]);'};
for token_index = 1:size(required_gate_tokens, 1)
    assert(contains(required_gate_tokens{token_index, 2}, ...
        required_gate_tokens{token_index, 3}), ...
        'smoke_contact_closure:MissingGateContract', ...
        '%s is missing contract token: %s', ...
        required_gate_tokens{token_index, 1}, ...
        required_gate_tokens{token_index, 3});
end

% Publication is an independently reviewable responsibility. Its focused
% smoke checks the filesystem invariants without enlarging this numerical
% fixture or invoking a production coupled solve.
smoke_contact_gate_publication();

critical_phi = [0; 2e-11; 0];
critical_rtol = 1e-10;
critical_atol = 1e-12;
critical_coarse_error = abs(critical_phi(1) - critical_phi(3));
critical_medium_error = abs(critical_phi(2) - critical_phi(3));
critical_tol = critical_atol + critical_rtol * ...
    max([1; abs(critical_phi)]);
legacy_tol = critical_atol + critical_rtol * max(abs(critical_phi));
assert(critical_medium_error <= critical_coarse_error + critical_tol);
assert(critical_medium_error > critical_coarse_error + legacy_tol);
fprintf('host/canonical-MAT/QOI-GCI/contraction contracts: PASS\n');

% Direct B66 regression: only positions x=0 and x=1 are on track.
% One of those two samples is tensile -> fraction 1/2. The obsolete
% lower-bound-only mask would include two post-exit zeros and return 1/4.
Calc = struct();
Calc.Options.VBI = 0;
Calc.Solver.num_t = 5;
Calc.Profile.L = 1;
Calc.Profile.L_Aw = 0;
Calc.Profile.L_bridge = 1;
Calc.Cte.grav = -9.81;
Calc.Veh(1).x_path = [-1, 0, 1, 2, 3];

Train = struct();
Train.Veh(1).Tnum = 1;
Train.Veh(1).Wheels.N2w = 1;
Train.Veh(1).Wheels.m = 0;
Train.Veh(1).Susp.Prim.k = 1;
Train.Veh(1).Susp.Prim.c = 0;

Sol = struct();
Sol.Veh(1).U = [0, 1, -1, 0, 0];
Sol.Veh(1).V = zeros(1, 5);
[Sol] = B66_ContactForce(Sol, struct(), Calc, Train, struct());
assert(Sol.F_tension_max == 1);
assert(Sol.contactLost_track == 1);
assert(Sol.tension_frac_max == 0.5);
fprintf('B66 bounded on-track mask: PASS\n');
fprintf('SMOKE CONTACT CLOSURE: ALL PASS\n');

function local_restore_probe_env(home, saved_path, scratch_dir)
% Restore the working directory and MATLAB path even if a shadow probe throws,
% so one failing probe cannot leave an impostor on the path for the rest of
% the smoke (or the session).
path(saved_path);
cd(home);
if isfolder(scratch_dir)
    rmdir(scratch_dir, 's');
end
end

function [qualified, inside] = local_reviewed_member_name( ...
        dep_dir, dep_name, reviewed_dir)
% Map one dependency path to its DECLARED inventory name, and say whether it
% lies inside the reviewed boundary.
%
% The boundary is scour_MATLAB AND its MATLAB package folders (+pkg). Without
% the package arm, splitting code into +ttbi/ would move those files to a
% different directory and this guard would skip them as "outside" - silently
% un-hashed and un-path-checked, which is precisely the blind spot the R11
% executed-module inventories were built to remove. A package member is
% declared by its QUALIFIED name (local_state_uid), the same name callers write
% and `which` resolves, so the inventory reads the way the code reads.
reviewed = local_absolute_path_smoke(reviewed_dir);
here = local_absolute_path_smoke(dep_dir);
if strcmpi(here, reviewed)
    qualified = dep_name;
    inside = true;
    return
end
[parent, leaf] = fileparts(here);
if strcmpi(local_absolute_path_smoke(parent), reviewed) && ...
        startsWith(leaf, '+')
    qualified = [leaf(2:end), '.', dep_name];
    inside = true;
    return
end
qualified = dep_name;
inside = false;
end

function absolute = local_absolute_path_smoke(path)
% Canonical absolute form for directory comparison in this smoke only.
if isempty(path)
    absolute = '';
    return
end
info = dir(path);
if isempty(info)
    absolute = char(path);
    return
end
absolute = char(info(1).folder);
if strcmp(info(1).name, '.')
    absolute = char(info(1).folder);
end
end

function statements = local_matlab_statements(source)
% Drop whole-line MATLAB comments and %{ ... %} blocks, leaving executable
% statements joined by newline. Deliberately identical in behaviour to
% _matlab_statements in check_contact_closure_gate.py: only whole-line
% comments are removed, because the apostrophe is both a string delimiter
% and the transpose operator, so trailing-comment stripping is ambiguous.
lines = regexp(source, '\r\n|\n|\r', 'split');
if ~isempty(lines) && isempty(lines{end})
    % A source ending in a line terminator yields a final EMPTY element
    % here, which Python's str.splitlines() does not produce. Drop it so
    % both implementations emit byte-identical statement text (verified by
    % SHA-256 over the complete study executable set).
    lines = lines(1:end-1);
end
kept = cell(1, numel(lines));
n_kept = 0;
in_block = false;
for k = 1:numel(lines)
    stripped = strtrim(lines{k});
    if in_block
        if strcmp(stripped, '%}')
            in_block = false;
        end
        continue
    end
    if strcmp(stripped, '%{')
        in_block = true;
        continue
    end
    if startsWith(stripped, '%')
        continue
    end
    n_kept = n_kept + 1;
    kept{n_kept} = lines{k};
end
statements = strjoin(kept(1:n_kept), newline);
end

function local_assert_throws(action, expected_identifier)
try
    action();
catch ME
    assert(strcmp(ME.identifier, expected_identifier), ...
        'smoke_contact_closure:WrongError', ...
        'Expected %s, got %s: %s', ...
        expected_identifier, ME.identifier, ME.message);
    return
end
error('smoke_contact_closure:MissingError', ...
    'Expected %s, but the mutation was accepted.', expected_identifier);
end

function digest = local_file_sha256(path)
fid = fopen(path, 'rb');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
digest = local_bytes_sha256(fread(fid, Inf, '*uint8'));
end

function local_write_file_bytes(path, bytes)
fid = fopen(path, 'wb');
assert(fid >= 0, 'Could not open %s for byte-exact restoration.', path);
cleanup = onCleanup(@() fclose(fid));
written = fwrite(fid, uint8(bytes), 'uint8');
assert(written == numel(bytes), ...
    'Short byte-exact restoration for %s.', path);
end

function digest = local_text_sha256(text)
digest = local_bytes_sha256(unicode2native(char(text), 'UTF-8'));
end

function digest = local_bytes_sha256(bytes)
engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(bytes);
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end
