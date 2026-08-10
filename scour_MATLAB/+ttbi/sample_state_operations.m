function operations = sample_state_operations(config, operations_seed_id)
%SAMPLE_STATE_OPERATIONS Draw one state's speed, temperature, and fleet EOVs.
%
% The caller supplies the semantic state's named "operations" seed. Keeping
% this draw in one function makes the LHS orientation and the ordering of the
% vehicle-normal draws explicit and independently reviewable.

rng(double(operations_seed_id), 'twister');

Npass = config.Npass;
Nveh = config.Nveh;
Nprop = config.Nprop;

% lhsdesign(n,p) returns n samples by p variables. The transpose produces the
% established 2 x Npass layout used by the generator.
lhs_matrix = lhsdesign(Npass, 2)';

% A transposed lhsdesign call previously forced speed and temperature into
% opposite halves. Verify deterministic defining properties instead of
% rejecting statistically unusual but valid LHS draws based on correlation or
% quadrant occupancy.
if ~isequal(size(lhs_matrix), [2, Npass])
    error('ttbi:OperationsLHSShape', ...
        'Speed/temperature LHS must have size 2 x Npass.');
end
observed_strata = sort(floor(lhs_matrix * Npass), 2);
expected_strata = repmat(0:Npass-1, 2, 1);
if ~isequal(observed_strata, expected_strata)
    error('ttbi:OperationsLHSStrata', ...
        'Speed/temperature LHS lost exact marginal stratification.');
end

% Preserve the established draw order: fleet-normal draws follow the LHS.
if config.use_vehicle_variability
    vehicle_properties = ones(Nveh, Nprop, Npass);
    for vehicle_index = 1:Nveh
        for property_index = 1:Nprop
            variability = randn(1, 1, Npass);
            vehicle_properties(vehicle_index, property_index, :) = ...
                vehicle_properties(vehicle_index, property_index, :) .* ...
                variability;
        end
    end
else
    vehicle_properties = zeros(Nveh, Nprop, Npass);
end

speed_mps = ones(1, Npass) * 80/3.6;
temperature_C = ones(1, Npass) * 25;
mean_speed_kmh = (config.vel_max + config.vel_min) / 2;
for passage_index = 1:Npass
    if config.use_speed_variability
        speed_mps(passage_index) = round(config.vel_min + ...
            (config.vel_max-config.vel_min) * ...
            lhs_matrix(1, passage_index)) / 3.6;
    else
        speed_mps(passage_index) = round(mean_speed_kmh) / 3.6;
    end
    if config.use_temp_variability
        temperature_C(passage_index) = round(config.temp_min + ...
            (config.temp_max-config.temp_min) * ...
            lhs_matrix(2, passage_index));
    else
        temperature_C(passage_index) = 25;
    end
end

operations = struct( ...
    'speed_mps', speed_mps, ...
    'temperature_C', temperature_C, ...
    'vehicle_properties', vehicle_properties);
end
