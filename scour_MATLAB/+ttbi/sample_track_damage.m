function [Damage, track_state, track_log_value] = sample_track_damage( ...
        Damage, track_state, Track, config, track_seed_id, passage_index)
%SAMPLE_TRACK_DAMAGE Draw persistent ballast, sleeper, and pad descriptors.
%
% Coordinates use the bridge-local modeled-track frame. In production
% per-state mode, the first passage draws the infrastructure condition and all
% later passages reuse it.

track_log_value = [];
if ~config.use_track_eov
    Damage.track = [];
    return;
end

% Resolve the optional sensitivity arm before touching the RNG.  The ordinary
% campaign omits this field and therefore retains the reviewed [1.2, 2.0]
% dry-stiffness law byte-for-byte.  The reciprocal arm is a matched-magnitude
% sign sensitivity, not a field-calibrated prior.
dry_stiffness_arm = ttbi.dry_ballast_stiffness_arm(config);

draw_now = strcmp(config.track_draw, 'per_passage') || passage_index == 1;
if ~draw_now
    config_has_arm = isfield(config, 'ballast_dry_stiffness_arm');
    state_has_arm = isfield(track_state, 'ballast_dry_stiffness_arm');
    if config_has_arm ~= state_has_arm || ...
            (config_has_arm && ~strcmp( ...
                dry_stiffness_arm, track_state.ballast_dry_stiffness_arm))
        error('ttbi:sample_track_damage:SensitivityArmMismatch', ...
            ['Persistent track state was created under a different ' ...
             'dry-ballast stiffness sensitivity configuration.']);
    end
end
if draw_now
    rng(double(track_seed_id), 'twister');
    track_state = struct();
    track_window = config.track_L_app + config.L_bridge + ...
        config.track_L_after;
    sleeper_spacing = Track.Sleeper.spacing;
    sleeper_ratio = track_window / sleeper_spacing;
    last_sleeper_location = ...
        floor(sleeper_ratio + 10 * eps(sleeper_ratio)) * sleeper_spacing;

    % Ballast patches: Poisson count over modeled length, with an enriched
    % centre density close to either bridge transition.
    patch_count = poissrnd(config.ballast_rate_100m * track_window / 100);
    ballast_patches = zeros(patch_count, 4);
    abutments = [config.track_L_app, ...
        config.track_L_app + config.L_bridge];
    for patch_index = 1:patch_count
        patch_length = config.ballast_patch_len(1) + ...
            diff(config.ballast_patch_len) * rand();
        proposal_accepted = false;
        for attempt = 1:50
            patch_start = (track_window - patch_length) * rand();
            near_transition = any(abs( ...
                (patch_start + patch_length/2) - abutments) <= ...
                config.ballast_trans_margin);
            weight = 1;
            if near_transition
                weight = config.ballast_trans_mult;
            end
            if rand() <= weight / config.ballast_trans_mult
                proposal_accepted = true;
                break;
            end
        end
        if ~proposal_accepted
            error('ttbi:sample_track_damage:BallastRejectionLimit', ...
                ['Ballast location rejection sampler exhausted 50 proposals. ' ...
                 'Refusing to retain the last rejected proposal.']);
        end

        if rand() < config.ballast_p_wet
            eta_k = config.ballast_eta_k_wet(1) + ...
                diff(config.ballast_eta_k_wet) * rand();
            eta_c = config.ballast_eta_c_wet(1) + ...
                diff(config.ballast_eta_c_wet) * rand();
        else
            eta_k_base = config.ballast_eta_k_dry(1) + ...
                diff(config.ballast_eta_k_dry) * rand();
            eta_c = config.ballast_eta_c_dry(1) + ...
                diff(config.ballast_eta_c_dry) * rand();
            if strcmp(dry_stiffness_arm, 'reciprocal-softening')
                % eta_soft=1/eta_base gives log(eta_soft)=-log(eta_base):
                % equal absolute log-distance from the healthy value 1, with
                % the stiffness direction reversed.  No extra RNG draw occurs.
                if ~isfinite(eta_k_base) || eta_k_base <= 1
                    error('ttbi:sample_track_damage:DryStiffnessBase', ...
                        ['Reciprocal sensitivity requires each retained-arm ' ...
                         'dry stiffness multiplier to be finite and > 1.']);
                end
                eta_k = 1 / eta_k_base;
            else
                eta_k = eta_k_base;
            end
        end
        ballast_patches(patch_index, :) = [patch_start, ...
            patch_start + patch_length, eta_k, eta_c];
    end

    % Hanging-sleeper groups: the registered transition mixture followed by
    % a fouled-patch odds enrichment.
    group_count = poissrnd(config.hang_rate_100m * track_window / 100);
    hanging_groups = zeros(group_count, 2);
    for group_index = 1:group_count
        group_size = randi(config.hang_group_size);
        % The descriptor count is exact and the sampled damage window is its
        % domain. Reject starts whose full sleeper group would leave it (and
        % could otherwise be truncated at a coincident model boundary).
        latest_group_start = last_sleeper_location - ...
            (group_size - 1) * Track.Sleeper.spacing;
        proposal_accepted = false;
        for attempt = 1:50
            if rand() < config.hang_p_transition
                transition = config.track_L_app + ...
                    (rand() < 0.5) * config.L_bridge;
                group_location = transition - config.hang_trans_margin + ...
                    2 * config.hang_trans_margin * rand();
            else
                group_location = rand() * track_window;
            end
            group_location = max(group_location, 0);
            if group_location > latest_group_start
                continue;
            end

            inside_fouled_patch = false;
            for patch_index = 1:size(ballast_patches, 1)
                if group_location >= ballast_patches(patch_index, 1) && ...
                        group_location <= ballast_patches(patch_index, 2)
                    inside_fouled_patch = true;
                    break;
                end
            end
            weight = 1;
            if inside_fouled_patch
                weight = config.hang_foul_mult;
            end
            if rand() <= weight / config.hang_foul_mult
                proposal_accepted = true;
                break;
            end
        end
        if ~proposal_accepted
            error('ttbi:sample_track_damage:HangingRejectionLimit', ...
                ['Unsupported-sleeper location rejection sampler exhausted ' ...
                 '50 proposals. Refusing to retain the last rejected proposal.']);
        end
        hanging_groups(group_index, :) = [group_location, group_size];
    end

    % One global pad service-condition draw plus independent Bernoulli failures at the
    % actual sleeper lattice.
    pad_stiffness = min(max(wblrnd(config.pad_weibull(1), ...
        config.pad_weibull(2)), config.pad_chi_range(1)), ...
        config.pad_chi_range(2));
    pad_damping = config.pad_beta_range(1) + ...
        diff(config.pad_beta_range) * rand();
    pad_failures = sample_pad_failures( ...
        track_window, Track.Sleeper.spacing, config.pad_p_fail);

    track_state.ballast_patches = ballast_patches;
    track_state.hanging_groups = hanging_groups;
    track_state.pad_stiff_mult = pad_stiffness;
    track_state.pad_damp_mult = pad_damping;
    track_state.pad_failures = pad_failures;
    track_state.pad_failure_rule = config.pad_failure_rule;
    track_state.x_bridge_local = config.track_L_app;
    if isfield(config, 'ballast_dry_stiffness_arm')
        % The sensitivity selector is persisted beside every descriptor; the
        % ordinary campaign omits it and keeps its historical payload shape.
        track_state.ballast_dry_stiffness_arm = dry_stiffness_arm;
    end
end

Damage.track = track_state;
track_log_value = track_state;
end
