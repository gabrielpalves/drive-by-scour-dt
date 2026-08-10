function F25_Run(experiment_name, max_workers)
%F25_RUN Generate the shared Fernandes-2025 reconstruction dataset.
%
% Usage from the reviewed MATLAB source directory:
%   F25_Run('F25-R', 4)
%
% F25-R and F25-X intentionally resolve to the same dataset/fingerprint.  The
% experiment selector verifies that shared contract; it does not generate a
% second copy.  Training manifests, caches, and results remain isolated.

if nargin < 1 || isempty(experiment_name)
    experiment_name = 'F25-R';
end
if nargin < 2 || isempty(max_workers)
    max_workers = 4;
end
validateattributes(max_workers, {'numeric'}, ...
    {'real','finite','scalar','integer','>=',1}, mfilename, 'max_workers');

script_dir = fileparts(mfilename('fullpath'));
if ~strcmp(ttbi.canonical_execution_path(pwd), ...
        ttbi.canonical_execution_path(script_dir))
    error('F25_Run:WorkingDirectory', ...
        ['F25_Run must execute with the current folder equal to ' ...
         'scour_MATLAB because the authenticated Type-2 loader uses its ' ...
         'reviewed local asset.']);
end
repository_root = fileparts(script_dir);
config = ttbi.f25_experiment_config(experiment_name);
if max_workers > config.max_parfor_workers
    error('F25_Run:WorkerCap', ...
        'F25 production permits at most %d process workers.', ...
        config.max_parfor_workers);
end
state = ttbi.f25_state_design(config);

% Prove that selecting either downstream experiment cannot alter the shared
% physical data or named random streams.
other_name = 'F25-X';
if strcmp(config.experiment_name, 'F25-X')
    other_name = 'F25-R';
end
other_config = ttbi.f25_experiment_config(other_name);
other_state = ttbi.f25_state_design(other_config);
if ~strcmp(config.dataset_id, other_config.dataset_id) || ...
        ~isequal(state.StateUID, other_state.StateUID) || ...
        ~isequal(state.StateNamedStreamSeedID, ...
            other_state.StateNamedStreamSeedID) || ...
        ~isequal(state.PassageNamedStreamSeedID, ...
            other_state.PassageNamedStreamSeedID)
    error('F25_Run:SharedDataDrift', ...
        'F25-R and F25-X no longer resolve to one shared physical dataset.');
end

environment_lock_path = fullfile(repository_root, 'environment', ...
    'campaign-py313-cu128.json');
environment_lock = jsondecode(fileread(environment_lock_path));
if ~isstruct(environment_lock) || ~isscalar(environment_lock) || ...
        ~isfield(environment_lock, 'schema') || ...
        ~strcmp(environment_lock.schema, 'ttbi-campaign-environment-v2') || ...
        ~isfield(environment_lock, 'matlab_environment') || ...
        ~isfield(environment_lock, 'matlab_environment_sha256')
    error('F25_Run:EnvironmentLock', ...
        'Campaign environment lock is missing its reviewed MATLAB identity.');
end
campaign_environment = environment_lock.matlab_environment;
[campaign_environment_sha256, campaign_environment_descriptor] = ...
    matlab_environment_identity(campaign_environment);
if ~strcmp(campaign_environment_sha256, ...
        environment_lock.matlab_environment_sha256)
    error('F25_Run:EnvironmentLockDigest', ...
        'Campaign MATLAB environment descriptor does not reproduce its hash.');
end
actual_environment = current_matlab_environment();
[actual_environment_sha256, actual_environment_descriptor] = ...
    matlab_environment_identity(actual_environment);
if ~strcmp(actual_environment_sha256, campaign_environment_sha256)
    error('F25_Run:CampaignMATLABEnvironment', ...
        ['F25 production data require exact campaign MATLAB environment %s ' ...
         '(%s); this host is %s (%s).'], ...
        campaign_environment_sha256, campaign_environment.release, ...
        actual_environment_sha256, actual_environment.release);
end

[source_root_sha256, source_digest_lines, source_file_count] = ...
    generator_source_root();
provenance = struct();
provenance.matlab_release = actual_environment.release;
provenance.campaign_matlab_release = campaign_environment.release;
provenance.actual_matlab_environment_descriptor = ...
    actual_environment_descriptor;
provenance.actual_matlab_environment_sha256 = ...
    actual_environment_sha256;
provenance.campaign_matlab_environment_descriptor = ...
    campaign_environment_descriptor;
provenance.campaign_matlab_environment_sha256 = ...
    campaign_environment_sha256;
provenance.generator_source_root_sha256 = source_root_sha256;
provenance.generator_source_digest_lines = source_digest_lines;
provenance.generator_source_file_count = source_file_count;
provenance.qualification_source_sha256 = 'PRODUCTION';
provenance.release_qualification_run = false;
identity = ttbi.f25_generation_identity(config, state, provenance);
other_identity = ttbi.f25_generation_identity( ...
    other_config, other_state, provenance);
if ~strcmp(identity.gen_fingerprint, other_identity.gen_fingerprint)
    error('F25_Run:SharedGenerationIdentity', ...
        'F25-R and F25-X produce different shared generation identities.');
end
provenance.gen_schema = identity.gen_schema;
provenance.gen_fingerprint = identity.gen_fingerprint;

relative_run_folder = [config.generation_results_root '/' config.dataset_id];
run_folder_observation = local_ensure_repository_directory( ...
    repository_root, relative_run_folder);
run_folder = run_folder_observation.canonical_path;
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);

case_info = local_case_info(config, state, identity, provenance);
local_publish_or_validate_case_info( ...
    run_folder, run_folder_observation, case_info, state);
completed = local_resume_inventory( ...
    run_folder, run_folder_observation, config, state, identity);

context = struct();
context.config = config;
context.state = state;
context.provenance = provenance;
context.identity = identity;
context.run_folder = run_folder;
context.run_folder_observation = run_folder_observation;
missing = find(~completed);
if ~isempty(missing)
    ttbi.assert_generator_source_unchanged(provenance);
    existing_pool = gcp('nocreate');
    if ~isempty(existing_pool)
        delete(existing_pool);
    end
    pool = parpool('Processes', min(max_workers, numel(missing)));
    pool_cleanup = onCleanup(@() local_delete_pool(pool));
    worker_attestation = parallel.pool.Constant( ...
        @() ttbi.authenticate_generation_worker(provenance));
    attestation_cleanup = onCleanup( ...
        @() delete(worker_attestation));
    parfor missing_index = 1:numel(missing)
        state_index = missing(missing_index);
        ttbi.f25_execute_generation_state( ...
            state_index, context, worker_attestation.Value);
    end
    delete(worker_attestation);
    clear attestation_cleanup
    delete(pool);
    clear pool_cleanup
    ttbi.assert_generator_source_unchanged(provenance);
end

completed = local_resume_inventory( ...
    run_folder, run_folder_observation, config, state, identity);
if ~all(completed)
    error('F25_Run:Incomplete', ...
        'F25 generation returned with %d/%d states authenticated.', ...
        sum(completed), state.n_states);
end
ttbi.assert_generator_source_unchanged(provenance);
local_publish_completion( ...
    run_folder, run_folder_observation, state, identity);
fprintf(['[F25] COMPLETE %s: %d classes x %d passages; contract %s; ' ...
    'generation %s.\n'], config.dataset_id, state.n_states, config.Npass, ...
    config.python_contract_sha256, identity.gen_fingerprint);
end

function observation = local_ensure_repository_directory(root, relative_path)
parts = regexp(strrep(relative_path, '\', '/'), '/', 'split');
if isempty(parts) || any(cellfun(@isempty, parts)) || ...
        any(ismember(parts, {'.','..'}))
    error('F25_Run:OutputPath', ...
        'F25 output path must be a normalized repository-relative path.');
end
cursor = root;
cursor_observation = ttbi.directory_observation(cursor);
for index = 1:numel(parts)
    ttbi.assert_generation_output_directory(cursor, cursor_observation);
    child = fullfile(cursor, parts{index});
    if ~isfolder(child)
        [made, message] = mkdir(child);
        if ~made
            error('F25_Run:OutputCreate', ...
                'Could not create F25 output directory: %s', message);
        end
    end
    child_observation = ttbi.directory_observation(child);
    cursor = child;
    cursor_observation = child_observation;
end
observation = cursor_observation;
end

function case_info = local_case_info(config, state, identity, provenance)
case_info = struct();
case_info.schema = 'f25-generation-case-info-v1';
case_info.dataset_id = config.dataset_id;
case_info.shared_data_contract_id = config.shared_data_contract_id;
case_info.python_contract_schema = config.python_contract_schema;
case_info.python_contract_sha256 = config.python_contract_sha256;
case_info.generation_schema = identity.gen_schema;
case_info.gen_fingerprint = identity.gen_fingerprint;
case_info.generation_config_json = identity.generation_config_json;
case_info.channel_schema_id = identity.channel_schema_id;
case_info.n_states = state.n_states;
case_info.passages_per_state = config.Npass;
case_info.state_design_kind = state.state_design_kind;
case_info.profile_asset_sha256 = config.profile_asset_sha256;
case_info.monitoring_window = config.monitoring_window;
case_info.partition_seed = config.partition_seed;
case_info.noise_master_seed = config.noise_master_seed;
case_info.matlab_release = provenance.matlab_release;
case_info.campaign_matlab_release = ...
    provenance.campaign_matlab_release;
case_info.actual_matlab_environment_descriptor = ...
    provenance.actual_matlab_environment_descriptor;
case_info.actual_matlab_environment_sha256 = ...
    provenance.actual_matlab_environment_sha256;
case_info.campaign_matlab_environment_descriptor = ...
    provenance.campaign_matlab_environment_descriptor;
case_info.campaign_matlab_environment_sha256 = ...
    provenance.campaign_matlab_environment_sha256;
case_info.generator_source_root_sha256 = ...
    provenance.generator_source_root_sha256;
case_info.generator_source_digest_lines = ...
    provenance.generator_source_digest_lines;
case_info.generator_source_file_count = ...
    provenance.generator_source_file_count;
case_info.release_qualification_run = false;
end

function local_publish_or_validate_case_info( ...
        run_folder, observation, case_info, state)
case_path = fullfile(run_folder, 'case_info.mat');
damage_path = fullfile(run_folder, 'damage_states.mat');
if isfile(case_path) || isfile(damage_path)
    if ~(isfile(case_path) && isfile(damage_path))
        error('F25_Run:PartialManifest', ...
            'F25 output has only one of its two required manifests.');
    end
    previous = load(case_path, 'case_info');
    damage = load(damage_path, 'StateUID', 'StateSeedID');
    if ~isfield(previous, 'case_info') || ...
            ~strcmp(previous.case_info.gen_fingerprint, ...
                case_info.gen_fingerprint) || ...
            ~strcmp(previous.case_info.python_contract_sha256, ...
                case_info.python_contract_sha256) || ...
            ~isequal(damage.StateUID, state.StateUID) || ...
            ~isequal(damage.StateSeedID, state.StateSeedID)
        error('F25_Run:ForeignManifest', ...
            'Existing F25 output belongs to another source/configuration.');
    end
    return
end
case_tmp = fullfile(run_folder, '.case_info.mat.tmp');
damage_tmp = fullfile(run_folder, '.damage_states.mat.tmp');
if isfile(case_tmp) || isfile(damage_tmp)
    error('F25_Run:ManifestTemporary', ...
        'Stale F25 manifest temporary file exists.');
end
StateUID = state.StateUID;
StateSeedID = state.StateSeedID;
DamageStates = state.DamageStates;
BearingStates = state.BearingStates;
CrackOn = state.CrackOn;
CrackLocation = state.CrackLocation;
CrackIntensity = state.CrackIntensity;
CrackHalfLength = state.CrackHalfLength;
StateNamedStreamSeedID = state.StateNamedStreamSeedID;
PassageNamedStreamSeedID = state.PassageNamedStreamSeedID;
ttbi.assert_generation_output_directory(run_folder, observation);
save(case_tmp, 'case_info', '-v7');
save(damage_tmp, 'StateUID', 'StateSeedID', 'DamageStates', ...
    'BearingStates', 'CrackOn', 'CrackLocation', 'CrackIntensity', ...
    'CrackHalfLength', 'StateNamedStreamSeedID', ...
    'PassageNamedStreamSeedID', '-v7');
[ok, message] = movefile(case_tmp, case_path);
if ~ok
    error('F25_Run:CaseManifestMove', '%s', message);
end
[ok, message] = movefile(damage_tmp, damage_path);
if ~ok
    error('F25_Run:DamageManifestMove', '%s', message);
end
ttbi.assert_generation_output_directory(run_folder, observation);
end

function completed = local_resume_inventory( ...
        run_folder, observation, config, state, identity)
entries = dir(run_folder);
numbered = {entries(~[entries.isdir]).name};
numbered = numbered(~cellfun(@isempty, ...
    regexp(numbered, '^\d{4}\.mat$', 'once')));
expected_names = arrayfun(@(i) sprintf('%04d.mat', i), ...
    1:state.n_states, 'UniformOutput', false);
if any(~ismember(numbered, expected_names))
    error('F25_Run:ForeignState', ...
        'F25 output contains a numbered state outside 0001..%04d.', ...
        state.n_states);
end
completed = false(state.n_states,1);
for state_index = 1:state.n_states
    path = fullfile(run_folder, expected_names{state_index});
    if ~isfile(path)
        continue
    end
    stamps = load(path, 'file_f25_schema', 'file_gen_fingerprint', ...
        'file_python_contract_sha256', 'file_state_uid', ...
        'file_state_seed_id', 'file_clean_trimmed_shape');
    required = {'file_f25_schema','file_gen_fingerprint', ...
        'file_python_contract_sha256','file_state_uid', ...
        'file_state_seed_id','file_clean_trimmed_shape'};
    if ~all(isfield(stamps, required)) || ...
            ~strcmp(stamps.file_f25_schema, 'f25-saved-state-v1') || ...
            ~strcmp(stamps.file_gen_fingerprint, identity.gen_fingerprint) || ...
            ~strcmp(stamps.file_python_contract_sha256, ...
                config.python_contract_sha256) || ...
            ~strcmp(stamps.file_state_uid, state.StateUID{state_index}) || ...
            ~isequal(stamps.file_state_seed_id, ...
                state.StateSeedID(state_index)) || ...
            ~isequal(double(stamps.file_clean_trimmed_shape), ...
                [config.Npass 8 config.trimmed_window_samples])
        error('F25_Run:ResumeState', ...
            'Existing state %04d fails F25 resume authentication.', ...
            state_index);
    end
    completed(state_index) = true;
end
ttbi.assert_generation_output_directory(run_folder, observation);
end

function local_publish_completion(run_folder, observation, state, identity)
names = arrayfun(@(i) sprintf('%04d.mat', i), 1:state.n_states, ...
    'UniformOutput', false);
digests = cell(state.n_states,1);
for index = 1:state.n_states
    digests{index} = ttbi.file_sha256(fullfile(run_folder, names{index}));
end
receipt = struct();
receipt.schema = 'f25-generation-artifact-digests-v1';
receipt.gen_fingerprint = identity.gen_fingerprint;
receipt.python_contract_sha256 = identity.python_contract_sha256;
receipt.files = struct('name', names(:), 'sha256', digests);
receipt.digest_root_sha256 = ttbi.sha256(jsonencode(receipt.files));
json_text = jsonencode(receipt, 'PrettyPrint', true);
digest_path = fullfile(run_folder, 'file_digests.json');
marker_path = fullfile(run_folder, '_F25_GENERATION_COMPLETE');
if isfile(digest_path) || isfile(marker_path)
    if ~(isfile(digest_path) && isfile(marker_path))
        error('F25_Run:PartialCompletion', ...
            'F25 completion publication is partial.');
    end
    previous = jsondecode(fileread(digest_path));
    if ~strcmp(previous.gen_fingerprint, identity.gen_fingerprint) || ...
            ~strcmp(previous.digest_root_sha256, ...
                receipt.digest_root_sha256)
        error('F25_Run:CompletionMismatch', ...
            'Existing F25 completion receipt does not match live state bytes.');
    end
    return
end
local_write_text_atomic(digest_path, json_text, run_folder, observation);
marker = sprintf(['schema=f25-generation-complete-v1\n' ...
    'gen_fingerprint=%s\npython_contract_sha256=%s\n' ...
    'digest_root_sha256=%s\n'], identity.gen_fingerprint, ...
    identity.python_contract_sha256, receipt.digest_root_sha256);
local_write_text_atomic(marker_path, marker, run_folder, observation);
end

function local_write_text_atomic(path, text, run_folder, observation)
tmp = [path '.tmp'];
if isfile(tmp)
    error('F25_Run:TextTemporary', ...
        'Stale F25 publication temporary exists: %s', tmp);
end
ttbi.assert_generation_output_directory(run_folder, observation);
fid = fopen(tmp, 'w', 'n', 'UTF-8');
if fid < 0
    error('F25_Run:TextOpen', 'Could not open %s.', tmp);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', text);
fclose(fid);
clear cleanup
[ok, message] = movefile(tmp, path);
if ~ok
    error('F25_Run:TextMove', '%s', message);
end
ttbi.assert_generation_output_directory(run_folder, observation);
end

function local_delete_pool(pool)
if ~isempty(pool) && isvalid(pool)
    delete(pool);
end
end
