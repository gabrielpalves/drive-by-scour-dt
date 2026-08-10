function identity = f25_generation_identity(config, state, provenance)
%F25_GENERATION_IDENTITY Bind shared F25 data to config, source, and runtime.
%
% Experiment-specific cache/result roots are deliberately excluded: F25-R
% and F25-X consume one physical dataset.  Every datum capable of changing
% those passages, including the complete realized RNG catalogue, is included.

required_provenance = {'campaign_matlab_environment_sha256', ...
    'generator_source_root_sha256','qualification_source_sha256'};
if ~isstruct(config) || ~isscalar(config) || ...
        ~strcmp(config.schema, 'f25-experiment-config-v1') || ...
        ~isstruct(state) || ~isscalar(state) || ...
        ~strcmp(state.schema, 'f25-state-design-v1') || ...
        ~isstruct(provenance) || ~isscalar(provenance) || ...
        ~all(isfield(provenance, required_provenance))
    error('ttbi:f25_generation_identity:Inputs', ...
        'F25 identity requires reviewed config, state, and provenance.');
end

body = struct();
body.schema = config.generation_schema;
body.python_contract_schema = config.python_contract_schema;
body.python_contract_sha256 = config.python_contract_sha256;
body.shared_data_contract_id = config.shared_data_contract_id;
body.dataset_id = config.dataset_id;
body.channel_schema_id = config.channel_schema_id;
body.campaign_matlab_environment_sha256 = ...
    provenance.campaign_matlab_environment_sha256;
body.generator_source_root_sha256 = ...
    provenance.generator_source_root_sha256;
body.qualification_source_sha256 = ...
    provenance.qualification_source_sha256;
body.profile_mode = config.profile_mode;
body.profile_asset_sha256 = config.profile_asset_sha256;
body.L_bridge = config.L_bridge;
body.num_spans = config.num_spans;
body.span_lengths_m = config.span_lengths_m;
body.support_locations_m = config.support_locations_m;
body.deck_element_length_m = config.deck_element_length_m;
body.deck_element_count = config.deck_element_count;
body.deck_mass_per_length_kg_per_m = ...
    config.deck_mass_per_length_kg_per_m;
body.deck_E_Pa = config.deck_E_Pa;
body.deck_I_m4 = config.deck_I_m4;
body.deck_damping_percent = config.deck_damping_percent;
body.n_states = state.n_states;
body.Npass = config.Npass;
body.state_design_kind = state.state_design_kind;
body.state_identity_version = state.state_identity_version;
body.random_stream_schedule_version = ...
    state.random_stream_schedule_version;
body.state_stream_names = state.state_stream_names;
body.passage_stream_names = state.passage_stream_names;
body.StateUID = state.StateUID;
body.StateSeedID = state.StateSeedID;
body.StateNamedStreamSeedID = state.StateNamedStreamSeedID;
body.PassageNamedStreamSeedIDFlat = ...
    state.PassageNamedStreamSeedIDFlat;
body.DamageStates = state.DamageStates;
body.BearingStates = state.BearingStates;
body.CrackOn = state.CrackOn;
body.CrackLocation = state.CrackLocation;
body.CrackIntensity = state.CrackIntensity;
body.CrackHalfLength = state.CrackHalfLength;
body.axis_codes = state.axis_codes;
body.eov_master_seed = config.eov_master_seed;
body.partition_seed = config.partition_seed;
body.noise_master_seed = config.noise_master_seed;
body.Nveh = config.Nveh;
body.Nprop = config.Nprop;
body.vel_km_h = [config.vel_min config.vel_max];
body.temp_C = [config.temp_min config.temp_max];
body.primary_suspension_kN_per_m = ...
    config.primary_suspension_kN_per_m;
body.secondary_suspension_kN_per_m = ...
    config.secondary_suspension_kN_per_m;
body.carbody_mass_kg = config.carbody_mass_kg;
body.monitoring_window = config.monitoring_window;
body.load_time_measurement_noise_fraction = ...
    config.load_time_measurement_noise_fraction;
body.noise_standard_deviation_ddof = ...
    config.noise_standard_deviation_ddof;
body.contact_max_tension_N = config.contact_max_tension_N;
body.contact_max_tension_fraction = ...
    config.contact_max_tension_fraction;

generation_config_json = jsonencode(body);
identity = struct();
identity.gen_schema = config.generation_schema;
identity.gen_fingerprint = ttbi.sha256(generation_config_json);
identity.generation_config_json = generation_config_json;
identity.python_contract_sha256 = config.python_contract_sha256;
identity.dataset_id = config.dataset_id;
identity.channel_schema_id = config.channel_schema_id;
end
