function [profile_config, profile_log_value, state_fra_class] = ...
        build_profile_config(config, state_fra_class, state_profile_seed_id, ...
            state_phase_seed_id, passage_profile_seed_id, passage_index)
%BUILD_PROFILE_CONFIG Build the registered rail-profile input for one passage.
%
% Fixed and state-random profiles share the same FRA-v2 spectrum. Only their
% phase/intensity rules differ. RNG resets below preserve the named-stream
% schedule used by the pre-refactor generator.

expected_clearance_m = 6;
expected_decision_id = 'paper1-rail-domain-clearance-c06-v1';
if ~isfield(config,'rail_end_clearance_m') || ...
        ~isnumeric(config.rail_end_clearance_m) || ...
        ~isreal(config.rail_end_clearance_m) || ...
        ~isscalar(config.rail_end_clearance_m) || ...
        ~isfinite(config.rail_end_clearance_m) || ...
        config.rail_end_clearance_m ~= expected_clearance_m
    error('ttbi:MainCampaignRailEndClearance', ...
        'Main-campaign profile config must request exactly 6 m clearance.');
end
if ~isfield(config,'rail_end_clearance_decision_id') || ...
        ~(ischar(config.rail_end_clearance_decision_id) || ...
          (isstring(config.rail_end_clearance_decision_id) && ...
           isscalar(config.rail_end_clearance_decision_id))) || ...
        ~strcmp(char(config.rail_end_clearance_decision_id),expected_decision_id)
    error('ttbi:MainCampaignRailEndClearanceDecision', ...
        'Main-campaign clearance decision identity is missing or unreviewed.');
end

profile_config = struct( ...
    'mode', config.profile_mode, ...
    'spectrum_contract', config.profile_spectrum_contract, ...
    'rail_end_clearance_m', config.rail_end_clearance_m, ...
    'rail_end_clearance_decision_id', ...
        char(config.rail_end_clearance_decision_id));
profile_log_value = 1;

if any(strcmp(config.profile_mode, {'fixed', 'fixed_scaled'}))
    profile_config.fra_class = config.profile_fra_classes;
    profile_config.phase_seed = config.profile_fixed_phase_seed;
    profile_config.jitter_sd_m = config.profile_jitter_sd_mm / 1000;
end

if strcmp(config.profile_mode, 'fixed_scaled')
    rng(double(passage_profile_seed_id), 'twister');
    profile_intensity = config.profile_int_range(1) + ...
        diff(config.profile_int_range) * rand();
    profile_config.intensity = profile_intensity;
    profile_log_value = profile_intensity;
elseif strcmp(config.profile_mode, 'fixed')
    profile_log_value = config.profile_fra_classes;
elseif strcmp(config.profile_mode, 'psd_fra')
    if strcmp(config.profile_draw, 'per_state')
        if passage_index == 1
            rng(double(state_profile_seed_id), 'twister');
            state_fra_class = config.profile_fra_classes( ...
                randi(numel(config.profile_fra_classes)));
        end
        profile_config.fra_class = state_fra_class;
        profile_config.phase_seed = double(state_phase_seed_id);
        profile_config.jitter_sd_m = config.profile_jitter_sd_mm / 1000;
    else
        % Deprecated legacy behavior: consume the current stream, exactly as
        % the former inline branch did.
        profile_config.fra_class = config.profile_fra_classes( ...
            randi(numel(config.profile_fra_classes)));
    end
    profile_log_value = profile_config.fra_class;
end
end
