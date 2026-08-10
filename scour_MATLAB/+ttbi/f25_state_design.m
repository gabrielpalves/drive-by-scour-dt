function design = f25_state_design(config)
%F25_STATE_DESIGN Build the ten immutable Fernandes scenario states.
%
% Each structural scenario is one persistent state with 200 operational/EOV
% passages.  F25-R and F25-X share these UIDs and named stream seeds exactly;
% only their downstream sensors/architectures differ.

if nargin < 1 || isempty(config)
    config = ttbi.f25_experiment_config('F25-R');
end
if ~isstruct(config) || ~isscalar(config) || ...
        ~isfield(config, 'schema') || ...
        ~strcmp(config.schema, 'f25-experiment-config-v1')
    error('ttbi:f25_state_design:Config', ...
        'config must be one f25-experiment-config-v1 scalar struct.');
end

catalog = config.scenarios;
n_states = catalog.n_scenarios;
StateUID = cell(n_states, 1);
for scenario_index = 1:n_states
    StateUID{scenario_index} = sprintf( ...
        'f25-state-v1|scenario=%s', catalog.labels{scenario_index});
end

StateSeedID = ttbi.state_seed_ids(StateUID, config.damage_seed);
random_stream_schedule_version = 'uid-named-substreams-v2';
state_stream_names = {'operations','crack','profile-state','track','profile-phase'};
passage_stream_names = {'profile-passage','oor-passage'};
[StateNamedStreamSeedID, PassageNamedStreamSeedID] = ...
    ttbi.named_stream_seed_ids(StateSeedID, StateUID, config.Npass, ...
        random_stream_schedule_version, state_stream_names, ...
        passage_stream_names);

Beam_probe = A03_Bridge(struct( ...
    'Prop', struct('L', config.L_bridge, 'num_spans', config.num_spans)));
k_ref_bear = 4 * Beam_probe.Prop.E * Beam_probe.Prop.I / ...
    (config.L_bridge/config.num_spans);
BearingStates = catalog.bearing_vectors_Nm_per_rad;
BearingFixity = BearingStates ./ (BearingStates + k_ref_bear);

CrackLocation = nan(n_states, 1);
CrackLocation(catalog.crack_active) = catalog.crack_block_centre_m;
CrackHalfLength = zeros(n_states, 1);
CrackHalfLength(catalog.crack_active) = ...
    catalog.crack_block_half_length_m;

if numel(unique(StateUID)) ~= n_states || ...
        ~isequal(size(StateNamedStreamSeedID), ...
            [n_states numel(state_stream_names)]) || ...
        ~isequal(size(PassageNamedStreamSeedID), ...
            [n_states config.Npass numel(passage_stream_names)])
    error('ttbi:f25_state_design:Identity', ...
        'F25 semantic state or named-stream identity is malformed.');
end

design = struct();
design.schema = 'f25-state-design-v1';
design.state_design_kind = config.state_design_kind;
design.state_identity_version = 'f25-state-v1';
design.random_stream_schedule_version = random_stream_schedule_version;
design.state_stream_names = state_stream_names;
design.passage_stream_names = passage_stream_names;
design.n_states = n_states;
design.n_supp = config.num_spans + 1;
design.n_latent_bear = 2;
design.k_ref_bear = k_ref_bear;
design.scenario_labels = catalog.labels;
design.class_index_zero_based = catalog.class_index_zero_based;
design.axis_names = catalog.axis_names;
design.axis_codes = catalog.axis_codes;
design.DamageStates = catalog.scour_vectors;
design.BearingStates = BearingStates;
design.BearingFixity = BearingFixity;
design.LatentBearingFixity = BearingFixity;
design.CrackOn = catalog.crack_active;
design.LatentCrackOn = catalog.crack_active;
design.CrackLocation = CrackLocation;
design.CrackIntensity = catalog.crack_ei_loss;
design.CrackHalfLength = CrackHalfLength;
design.StateFamily = repmat({'f25_scenario'}, n_states, 1);
design.AnchorTarget = zeros(n_states, 1);
design.AnchorLevel = zeros(n_states, 1);
design.StateUID = StateUID;
design.StateSeedID = StateSeedID;
design.StateNamedStreamSeedID = StateNamedStreamSeedID;
design.PassageNamedStreamSeedID = PassageNamedStreamSeedID;
design.PassageNamedStreamSeedIDFlat = reshape( ...
    PassageNamedStreamSeedID, n_states, []);
design.Npass = config.Npass;
design.channel_schema_id = config.channel_schema_id;
end
