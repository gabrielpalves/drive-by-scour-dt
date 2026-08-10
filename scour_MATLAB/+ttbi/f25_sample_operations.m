function operations = f25_sample_operations(config, operations_seed_id)
%F25_SAMPLE_OPERATIONS Draw the source-backed F25 operational envelope.
%
% One seeded Latin hypercube covers speed, temperature, and the three varied
% properties of each of five vehicles.  A01_Train's public input is expressed
% as standardised offsets around its Obrien-calibration defaults, so this
% helper maps the frozen Fernandes physical ranges back to those offsets and
% records both representations.  No Gaussian tail can therefore escape the
% published min/max envelope.

if ~isstruct(config) || ~isscalar(config) || ...
        ~isfield(config, 'schema') || ...
        ~strcmp(config.schema, 'f25-experiment-config-v1')
    error('ttbi:f25_sample_operations:Config', ...
        'config must be one f25-experiment-config-v1 scalar struct.');
end
validateattributes(operations_seed_id, {'numeric'}, ...
    {'real','finite','scalar','integer','positive'}, mfilename, ...
    'operations_seed_id');

n_passages = config.Npass;
n_vehicle_dimensions = config.Nveh*config.Nprop;
n_dimensions = 2+n_vehicle_dimensions;
rng(double(operations_seed_id), 'twister');
unit = lhsdesign(n_passages, n_dimensions, 'criterion', 'maximin', ...
    'iterations', 10);
if ~isequal(size(unit), [n_passages n_dimensions]) || ...
        any(~isfinite(unit), 'all') || any(unit <= 0 | unit >= 1, 'all')
    error('ttbi:f25_sample_operations:LHS', ...
        'F25 Latin-hypercube construction returned an invalid design.');
end
expected_strata = repmat((0:n_passages-1)', 1, n_dimensions);
observed_strata = sort(floor(unit*n_passages), 1);
if ~isequal(observed_strata, expected_strata)
    error('ttbi:f25_sample_operations:Strata', ...
        'F25 operational design lost exact marginal stratification.');
end

speed_km_h = config.vel_min + ...
    (config.vel_max-config.vel_min)*unit(:,1);
temperature_C = config.temp_min + ...
    (config.temp_max-config.temp_min)*unit(:,2);

vehicle_unit = reshape(unit(:,3:end)', ...
    config.Nprop, config.Nveh, n_passages);
vehicle_unit = permute(vehicle_unit, [2 1 3]);
body_mass_kg = config.carbody_mass_kg(1) + ...
    diff(config.carbody_mass_kg)*vehicle_unit(:,1,:);
primary_one_side_N_per_m = 1e3*( ...
    config.primary_suspension_kN_per_m(1) + ...
    diff(config.primary_suspension_kN_per_m)*vehicle_unit(:,2,:));
secondary_one_side_N_per_m = 1e3*( ...
    config.secondary_suspension_kN_per_m(1) + ...
    diff(config.secondary_suspension_kN_per_m)*vehicle_unit(:,3,:));

% In the 2-D reduction each suspension stiffness is doubled.  A01's offsets
% are parameterised relative to those doubled baseline values.
body_reference = 36852;
primary_reference = 2779e3*2;
secondary_reference = 1000e3*2;
a01_inputs = zeros(config.Nveh, config.Nprop, n_passages);
a01_inputs(:,1,:) = (body_mass_kg-body_reference)/(0.10*body_reference);
a01_inputs(:,2,:) = (2*primary_one_side_N_per_m-primary_reference) / ...
    (0.05*primary_reference);
a01_inputs(:,3,:) = (2*secondary_one_side_N_per_m-secondary_reference) / ...
    (0.05*secondary_reference);

if any(~isfinite(a01_inputs), 'all')
    error('ttbi:f25_sample_operations:VehicleMap', ...
        'F25 physical-to-A01 vehicle mapping returned nonfinite values.');
end

operations = struct();
operations.schema = 'f25-operations-v1';
operations.seed_id = uint32(operations_seed_id);
operations.lhs_unit = unit;
operations.speed_km_h = speed_km_h';
operations.speed_mps = operations.speed_km_h/3.6;
operations.temperature_C = temperature_C';
operations.a01_vehicle_inputs = a01_inputs;
operations.body_mass_kg = body_mass_kg;
operations.primary_suspension_N_per_m = ...
    2*primary_one_side_N_per_m;
operations.secondary_suspension_N_per_m = ...
    2*secondary_one_side_N_per_m;
end
