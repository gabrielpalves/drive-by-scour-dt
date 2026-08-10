function [Damage, crack_row] = sample_crack_damage( ...
        Damage, config, crack_seed_id, crack_is_active, passage_index)
%SAMPLE_CRACK_DAMAGE Apply the registered crack nuisance draw for one passage.
%
% In the production per-state mode, location and intensity are drawn on the
% first passage and retained in Damage thereafter. The per-passage branch is
% kept only for reproducibility of legacy datasets.

draw_now = config.use_crack_eov && ...
    (strcmp(config.crack_draw, 'per_passage') || passage_index == 1);

if draw_now
    rng(double(crack_seed_id), 'twister');
    if strcmp(config.crack_draw, 'per_passage')
        crack_now = rand() <= config.crack_p;
    else
        crack_now = crack_is_active;
    end

    if crack_now
        support_positions = linspace(0, config.L_bridge, ...
            config.num_spans + 1);
        internal_supports = support_positions(2:end-1);
        span_length = config.L_bridge / config.num_spans;

        hogging_probability = config.crack_hog_ratio / ...
            (config.crack_hog_ratio + 1);
        if ~isempty(internal_supports) && rand() < hogging_probability
            crack_centre = internal_supports( ...
                randi(numel(internal_supports)));
        else
            span_index = randi(config.num_spans);
            crack_centre = (support_positions(span_index) + ...
                support_positions(span_index + 1)) / 2;
        end

        crack_location = crack_centre + (2*rand() - 1) * ...
            config.crack_hog_margin * span_length;
        crack_location = min(max(crack_location, ...
            config.crack_frac_range(1) * config.L_bridge), ...
            config.crack_frac_range(2) * config.L_bridge);
        crack_intensity = config.crack_int_range(1) + ...
            diff(config.crack_int_range) * rand();

        Damage.crack_locs = crack_location;
        Damage.crack_intensity = crack_intensity;
        Damage.crack_lc = config.crack_lc;
    else
        Damage.crack_locs = [];
        Damage.crack_intensity = [];
        Damage.crack_lc = 0;
    end
elseif ~config.use_crack_eov
    Damage.crack_locs = [];
    Damage.crack_intensity = [];
    Damage.crack_lc = 0;
end

crack_row = zeros(1, 3);
if ~isempty(Damage.crack_locs)
    crack_row = [Damage.crack_locs, Damage.crack_intensity, Damage.crack_lc];
end
end
