function config = f25_experiment_config(experiment_name)
%F25_EXPERIMENT_CONFIG Isolated F25-R/F25-X configuration of record.
%
% F25 is an experiment configuration on the qualified production source tree,
% not a second solver lineage.  The generated 2,000-passage dataset is shared
% by F25-R and F25-X; manifests, caches and training results remain isolated
% from the four Paper-1 blocks and from each other.

if nargin < 1 || isempty(experiment_name)
    experiment_name = 'F25-R';
end
experiment_name = char(string(experiment_name));
if ~any(strcmp(experiment_name, {'F25-R', 'F25-X'}))
    error('ttbi:f25_experiment_config:Experiment', ...
        'experiment_name must be exactly F25-R or F25-X.');
end

package_dir = fileparts(mfilename('fullpath'));
matlab_root = fileparts(package_dir);
profile_asset_name = 'Calc.ProfileData15_05.mat';
profile_asset_path = fullfile(matlab_root, profile_asset_name);
profile_asset_sha256 = ...
    '71c69d9923bdc184a2c8448e0e0e6debb1670302908e093b758f57c36147465d';
if ~ttbi.regular_nonsymlink_file(profile_asset_path)
    error('ttbi:f25_experiment_config:ProfileAsset', ...
        'The reviewed Fernandes Type-2 profile asset is missing or linked.');
end
actual_profile_sha256 = ttbi.file_sha256(profile_asset_path);
if ~strcmp(actual_profile_sha256, profile_asset_sha256)
    error('ttbi:f25_experiment_config:ProfileDigest', ...
        ['Calc.ProfileData15_05.mat no longer matches the source-locked ' ...
         'Fernandes profile digest (%s ~= %s).'], ...
        actual_profile_sha256, profile_asset_sha256);
end

catalog = ttbi.f25_scenario_catalog();
if strcmp(experiment_name, 'F25-R')
    experiment_id = 'fernandes-2025-f25-r-v1';
else
    experiment_id = 'fernandes-2025-f25-x-v1';
end

profile_config = struct();
profile_config.mode = 'f25_stored_type2';
profile_config.profile_contract = 'fernandes-2025-stored-type2-v1';
profile_config.asset_name = profile_asset_name;
profile_config.asset_sha256 = profile_asset_sha256;
profile_config.intensity = 1;

config = struct();
config.schema = 'f25-experiment-config-v1';
config.experiment_name = experiment_name;
config.experiment_id = experiment_id;
config.dataset_id = 'fernandes-2025-f25-data-v1';
config.python_contract_schema = 'f25-experiment-contract-v1';
config.python_contract_sha256 = ...
    'a80b9c754f911737b1ce6d841bfb837d6ac3200c334508abd3f901f9167d48ce';
config.shared_data_contract_id = 'f25-shared-data-and-partition-v1';
config.generation_schema = 'f25-generation-v1';
config.state_design_kind = 'f25-ten-scenario-v1';
config.channel_schema_id = 'physical8_v1';

% Dedicated roots.  All are repository-relative by design; a bundle may map
% them to different disks without changing the scientific identifiers.
config.generation_results_root = 'f25_artifacts/shared_data/generation';
config.generation_manifest_root = 'f25_artifacts/shared_data/manifests';
config.cache_root = sprintf('f25_artifacts/%s/cache', experiment_name);
config.training_manifest_root = sprintf( ...
    'f25_artifacts/%s/manifests', experiment_name);
config.training_results_root = sprintf( ...
    'f25_artifacts/%s/results', experiment_name);

% Source geometry and its support-aligned uniform refinement.
config.L_bridge = 39.9;
config.num_spans = 2;
config.span_lengths_m = [19.95 19.95];
config.support_locations_m = [0 19.95 39.9];
config.scour_supports = 2;
config.deck_element_length_m = 0.15;
config.deck_element_count = 266;
config.deck_mass_per_length_kg_per_m = 9600;
config.deck_E_Pa = 35e9;
config.deck_I_m4 = 0.33;
config.deck_damping_percent = 3;

config.profile_mode = profile_config.mode;
config.profile_config = profile_config;
config.profile_asset_path = profile_asset_path;
config.profile_asset_sha256 = profile_asset_sha256;

config.scenarios = catalog;
config.Npass = catalog.passages_per_scenario;
config.n_states = catalog.n_scenarios;
config.total_passages = config.Npass * config.n_states;
% These decimal integers mirror core.f25_experiment_contract exactly.  The
% EOV master drives immutable state/named-stream identities; the partition
% and load-time noise seeds are consumed downstream but are repeated here so
% every generated manifest binds the complete shared-data contract.
config.eov_master_seed = 2025080902;
config.damage_seed = config.eov_master_seed;
config.partition_seed = 2025080901;
config.noise_master_seed = 2025080903;

% Paper-matched operational/environmental envelope.  The established
% generator owns the actual LHS/vehicle-property draw implementation.
config.Nveh = 5;
config.Nprop = 3;
config.Desvio = 0.05;
config.vel_min = 70;
config.vel_max = 90;
config.temp_min = 3;
config.temp_max = 33;
config.primary_suspension_kN_per_m = [2640 2920];
config.secondary_suspension_kN_per_m = [942 1042];
config.carbody_mass_kg = [33000 40000];
config.use_vehicle_variability = true;
config.use_speed_variability = true;
config.use_temp_variability = true;
config.use_signal_noise = false;
config.load_time_measurement_noise_fraction = 0.05;
config.noise_standard_deviation_ddof = 0;

% Production solver controls.  F25 reuses the qualified coupled solver and
% its registered bilateral-contact admissibility envelope, but publishes to
% a dedicated shared-data root.
config.max_parfor_workers = 4;
config.contact_max_tension_N = 24000;
config.contact_max_tension_fraction = 0.002;

% Monitoring and preprocessing contract.  The full RAW passage is re-cropped
% under the source's 58.30 m convention by ttbi.f25_monitoring_window; this is
% intentionally separate from D01's corrected main-campaign crop. F25 then
% trims the final sample, preserving the opening coordinate, before the exact
% 583 by 10-point PAA reduction in Python.
config.raw_window_samples = 5831;
config.trimmed_window_samples = 5830;
config.tail_samples_trimmed = 1;
config.paa_segments = 583;
config.paa_samples_per_segment = 10;

config.split_train = 80;
config.split_validation = 20;
config.split_test = 100;
config.hpo_trials = 100;
config.hpo_executions_per_trial = 5;
config.reported_runs = 20;
config.max_epochs = 1000;
config.early_stopping_patience = 50;
config.lr_plateau_patience = 30;
config.lr_plateau_factor = 0.5;
config.min_learning_rate = 1e-6;

config.monitoring_window = ttbi.f25_monitoring_window(config);

if config.total_passages ~= 2000 || ...
        config.trimmed_window_samples ~= ...
            config.paa_segments*config.paa_samples_per_segment || ...
        config.monitoring_window.untrimmed_sample_count ~= ...
            config.raw_window_samples || ...
        config.monitoring_window.trimmed_sample_count ~= ...
            config.trimmed_window_samples || ...
        config.split_train + config.split_validation + config.split_test ~= ...
            config.Npass || ...
        config.deck_element_count*config.deck_element_length_m ~= ...
            config.L_bridge || ...
        any(config.span_lengths_m ~= config.L_bridge/config.num_spans)
    error('ttbi:f25_experiment_config:Arithmetic', ...
        'The frozen F25 geometry, passage, split, or PAA arithmetic drifted.');
end
end
