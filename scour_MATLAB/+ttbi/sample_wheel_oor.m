function [Damage, oor_log_value] = sample_wheel_oor( ...
        Damage, config, passage_seed_id)
%SAMPLE_WHEEL_OOR Draw wheel polygonization and optional flat descriptors.
%
% This nuisance condition is passage-specific: each passage represents a
% different train from the fleet and therefore uses its named OOR seed.

oor_log_value = [];
if ~config.use_oor_eov
    Damage.oor_flats = [];
    Damage.oor_poly = [];
    return;
end

rng(double(passage_seed_id), 'twister');
flats = zeros(0, 5);       % [vehicle wheel length_m depth_m phase]
polygons = zeros(0, 5);    % [vehicle wheel order amplitude_m phase]

for vehicle_index = 1:config.Nveh
    if config.oor_flats_enabled
        for bogie_index = 0:1
            if rand() < config.oor_q_bogie
                wheels = 2*bogie_index + 1;
                if rand() < config.oor_p_trailing
                    wheels(end+1) = 2*bogie_index + 2; %#ok<AGROW>
                end
                for wheel_index = wheels
                    if rand() < config.oor_p_fresh
                        flat_length = config.oor_len_fresh(1) + ...
                            diff(config.oor_len_fresh) * rand();
                        flat_depth = flat_length^2 / (8*config.oor_radius);
                    else
                        flat_length = config.oor_len_runin(1) + ...
                            diff(config.oor_len_runin) * rand();
                        flat_depth = flat_length^2 / (16*config.oor_radius);
                    end
                    flats(end+1, :) = [vehicle_index, wheel_index, ...
                        flat_length, flat_depth, 2*pi*rand()]; %#ok<AGROW>
                end
            end
        end
    end

    for wheel_index = 1:4
        if rand() < config.poly_p_wheel
            order = randi(config.poly_orders);
            amplitude = exp(config.poly_amp_lnorm(1) + ...
                config.poly_amp_lnorm(2) * randn());
            amplitude = min(max(amplitude, ...
                config.poly_amp_bounds(1)), config.poly_amp_bounds(2));
            polygons(end+1, :) = [vehicle_index, wheel_index, ...
                order, amplitude, 2*pi*rand()]; %#ok<AGROW>
        end
    end
end

Damage.oor_flats = flats;
Damage.oor_poly = polygons;
Damage.oor_radius = config.oor_radius;
oor_log_value = struct('flats', flats, 'poly', polygons);
end
