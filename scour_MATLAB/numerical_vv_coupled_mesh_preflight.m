function result = numerical_vv_coupled_mesh_preflight( ...
        bridge_length_m, num_spans, bridge_elements_per_bay, ...
        rail_elements_per_bay, varargin)
%NUMERICAL_VV_COUPLED_MESH_PREFLIGHT Exercise mixed bridge/rail mesh assembly.
%
% This bounded preflight executes production geometry and B54 assembly, but no
% dynamic solve.  It verifies that bridge and rail may use different integer
% element counts per 0.6 m sleeper bay while both retain exact sleeper coupling.
% It does not constitute mesh/time convergence evidence.

parser = inputParser;
addParameter(parser, 'Assemble', true, @local_logical_scalar);
addParameter(parser, 'Redux', 0, @local_logical_scalar);
parse(parser, varargin{:});
assemble = logical(parser.Results.Assemble);
redux = double(parser.Results.Redux);

P = numerical_vv_protocol_definition();
geometry_index = find(abs([P.geometries.bridge_length_m]-bridge_length_m) < 1e-12);
if numel(geometry_index) ~= 1 || ...
        P.geometries(geometry_index).num_spans ~= num_spans
    error('numerical_vv:UnregisteredGeometry', ...
        'Length/span count must identify one registered geometry.');
end
local_positive_integer(bridge_elements_per_bay, 'bridge_elements_per_bay');
local_positive_integer(rail_elements_per_bay, 'rail_elements_per_bay');

sleeper_spacing_m = P.sleeper_spacing_m;
Track = A02_Track();
ballast_lump_mass_kg = Track.BallastOnBeam.Prop.m;
bridge_element_count_real = bridge_length_m/sleeper_spacing_m * ...
    bridge_elements_per_bay;
bridge_element_count = round(bridge_element_count_real);
if abs(bridge_element_count_real-bridge_element_count) > ...
        256*eps(max(bridge_element_count_real, 1))
    error('numerical_vv:BridgeLengthNotNested', ...
        'Bridge mesh does not contain an integer element count.');
end
nominal_supports = linspace(0, bridge_length_m, num_spans+1);
bridge_h_m = bridge_length_m/bridge_element_count;
bridge_nodes = [0, cumsum(ones(1, bridge_element_count)*bridge_h_m)];
realized_supports = zeros(size(nominal_supports));
for k = 1:numel(nominal_supports)
    [~, index] = min(abs(bridge_nodes-nominal_supports(k)));
    realized_supports(k) = bridge_nodes(index);
end
support_offsets = realized_supports-nominal_supports;
summation_roundoff_factor = max(256, 2*bridge_element_count);
alignment_tolerance_m = summation_roundoff_factor * ...
    eps(max(bridge_length_m, 1));

result = struct();
result.schema = 'numerical-vv-coupled-mesh-preflight-v1';
result.scope = 'nonqualifying-geometry-and-B54-assembly-preflight';
result.geometry_id = P.geometries(geometry_index).id;
result.bridge_length_m = bridge_length_m;
result.num_spans = num_spans;
result.bridge_elements_per_bay = bridge_elements_per_bay;
result.rail_elements_per_bay = rail_elements_per_bay;
result.bridge_h_m = bridge_h_m;
result.rail_h_m = sleeper_spacing_m/rail_elements_per_bay;
result.nominal_support_coordinates_m = nominal_supports;
result.realized_support_coordinates_m = realized_supports;
result.support_signed_offsets_m = support_offsets;
result.support_alignment_pass = ...
    all(abs(support_offsets) <= alignment_tolerance_m);
result.assembly_executed = assemble;
result.redux = redux;
result.assembly_pass = false;
result.model_dof_count = NaN;
result.bridge_sleeper_coupling_count = NaN;
result.rail_sleeper_coupling_count = NaN;
result.ballast_mass_assembly_max_abs_error_kg = NaN;
result.ballast_mass_assembly_actual_total_kg = NaN;
result.on_bridge_sleeper_count = bridge_length_m/sleeper_spacing_m+1;
result.bridge_vertical_node_count = bridge_element_count+1;
result.ballast_nominal_lump_mass_kg = ballast_lump_mass_kg;
result.ballast_mass_per_bridge_sleeper_node_kg = ballast_lump_mass_kg;
result.ballast_B54_lumped_total_kg = ...
    result.on_bridge_sleeper_count*ballast_lump_mass_kg;
result.ballast_continuous_bay_total_kg = ...
    (bridge_length_m/sleeper_spacing_m)*ballast_lump_mass_kg;
result.ballast_full_sleeper_total_kg = ...
    result.on_bridge_sleeper_count*ballast_lump_mass_kg;
result.ballast_delta_vs_continuous_bay_kg = ...
    result.ballast_B54_lumped_total_kg- ...
    result.ballast_continuous_bay_total_kg;
result.ballast_delta_vs_full_sleeper_kg = ...
    result.ballast_B54_lumped_total_kg- ...
    result.ballast_full_sleeper_total_kg;
old_L60_n2_node_count = round(60/sleeper_spacing_m)*2+1;
result.reference_old_L60_n2_total_kg = ...
    old_L60_n2_node_count*ballast_lump_mass_kg/2;
result.ballast_delta_vs_old_L60_n2_total_kg = ...
    result.ballast_B54_lumped_total_kg- ...
    result.reference_old_L60_n2_total_kg;
result.ballast_endpoint_mass_convention_status = ...
    ['PROXY_INFORMED_SUPPORT_POINT_LUMPS_' ...
     'AUTHOR_CHOSEN_ENDPOINT_OWNERSHIP'];

if ~assemble
    return
end

vehicle_draw = zeros(5, 3);
Train = A01_Train(80/3.6, vehicle_draw);
Beam = A03_Bridge(struct('Prop', struct( ...
    'L', bridge_length_m, 'num_spans', num_spans)));
[Calc, Beam, Track] = A04_Options(Beam, Track, struct('mode', 'fixed'));
% Default to the production redux=0 geometry. Redux=1 remains available for
% additional bounded refinement checks, but cannot substitute for at least
% one production-path assembly oracle.
Calc.Options.redux = redux;
Beam.Mesh.Ele.num_per_spacing = bridge_elements_per_bay;
Track.Rail.Mesh.Ele.num_per_spacing = rail_elements_per_bay;
[Calc, Train, Beam] = B43_ModelGeometry(Calc, Train, Track, Beam);
[Calc, ~, Track, Beam] = B07_OptionsProcessing(Calc, Train, Track, Beam);

Damage = struct('scour_rates', zeros(1, num_spans+1), ...
    'bearing_left', 0, 'bearing_right', 0);
Beam = B01_ElementsAndCoordinates(Beam, Calc);
[Beam, Damage] = B02_BoundaryConditions(Beam, Damage);
Beam = B03_BeamMatrices(Beam);
Beam.Mesh.Cg = sparse(size(Beam.Mesh.Kg, 1), size(Beam.Mesh.Kg, 2));

Track = B51_RailVariables(Track, Calc);
Track.Rail = B01_ElementsAndCoordinates(Track.Rail);
Track.Rail = B02_BoundaryConditions(Track.Rail, Damage);
Track.Rail = B03_BeamMatrices(Track.Rail);
Track.Rail.Mesh.Cg = sparse( ...
    size(Track.Rail.Mesh.Kg, 1), size(Track.Rail.Mesh.Kg, 2));

Model = B54_ModelMatrices(Beam, Track, Calc, Damage);
expected_on_bridge = round(bridge_length_m/sleeper_spacing_m)+1;
beam_coupling_count = numel(Model.Mesh.DOF.beam_vert_under_sleeper);
rail_coupling_count = numel(Model.Mesh.DOF.rail_vert_at_sleepers);
if beam_coupling_count ~= expected_on_bridge
    error('numerical_vv:BridgeSleeperCouplingCount', ...
        'Mixed mesh produced the wrong bridge/sleeper coupling count.');
end
if rail_coupling_count ~= numel(Model.Mesh.DOF.sleepers)
    error('numerical_vv:RailSleeperCouplingCount', ...
        'Mixed mesh produced the wrong rail/sleeper coupling count.');
end
beam_under_sleeper_x = Beam.Mesh.Nodes.acum( ...
    1:bridge_elements_per_bay:end);
expected_bridge_x = (0:expected_on_bridge-1)*sleeper_spacing_m;
if numel(beam_under_sleeper_x) ~= numel(expected_bridge_x) || ...
        max(abs(beam_under_sleeper_x-expected_bridge_x)) > ...
        alignment_tolerance_m
    error('numerical_vv:BridgeSleeperCoordinateMismatch', ...
        'Bridge/sleeper coupling coordinates are not exact.');
end

% Isolate the only mass contribution added to the bridge block by B54.
% The production contract is one inherited 531.4 kg deck-attached ballast
% lump at every support-aligned on-bridge sleeper node, independent of bridge
% element density.  This checks the declared condensed inventory only; it is
% not validation of Zhai's independent-ballast/shear topology.
model_bridge_mass = Model.Mesh.Mg( ...
    Model.Mesh.DOF.beam, Model.Mesh.DOF.beam);
assembled_ballast_mass = model_bridge_mass-Beam.Mesh.Mg;
beam_dof_count = numel(Model.Mesh.DOF.beam);
beam_under_sleeper_local = Model.Mesh.DOF.beam_vert_under_sleeper- ...
    Model.Mesh.DOF.beam(1)+1;
expected_ballast_mass = sparse(beam_dof_count, beam_dof_count);
expected_ballast_mass(beam_under_sleeper_local, ...
    beam_under_sleeper_local) = ...
    speye(expected_on_bridge)*ballast_lump_mass_kg;
ballast_mass_error = assembled_ballast_mass-expected_ballast_mass;
ballast_mass_max_abs_error_kg = local_max_abs(ballast_mass_error);
ballast_mass_tolerance_kg = 1024*eps(max( ...
    result.ballast_B54_lumped_total_kg, 1));
if ballast_mass_max_abs_error_kg > ballast_mass_tolerance_kg
    error('numerical_vv:BallastMassAssemblyMismatch', ...
        ['B54 did not add exactly one ballast lump at each on-bridge ' ...
         'sleeper node.']);
end

result.assembly_pass = true;
result.model_dof_count = Model.Mesh.DOF.Tnum;
result.bridge_sleeper_coupling_count = beam_coupling_count;
result.rail_sleeper_coupling_count = rail_coupling_count;
result.bridge_sleeper_coordinate_max_error_m = ...
    max(abs(beam_under_sleeper_x-expected_bridge_x));
result.ballast_mass_assembly_max_abs_error_kg = ...
    ballast_mass_max_abs_error_kg;
result.ballast_mass_assembly_actual_total_kg = ...
    full(sum(diag(assembled_ballast_mass)));
result.matrix_symmetry_max_abs = max([ ...
    local_max_abs(Model.Mesh.Mg-Model.Mesh.Mg'), ...
    local_max_abs(Model.Mesh.Cg-Model.Mesh.Cg'), ...
    local_max_abs(Model.Mesh.Kg-Model.Mesh.Kg')]);
end

function local_positive_integer(value, name)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
        ~isfinite(value) || value < 1 || value ~= floor(value)
    error('numerical_vv:BadElementsPerBay', ...
        '%s must be one finite positive integer.', name);
end
end

function tf = local_logical_scalar(value)
tf = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
    isfinite(double(value)) && any(double(value) == [0, 1]);
end

function value = local_max_abs(array)
values = nonzeros(array);
if isempty(values)
    value = 0;
else
    value = full(max(abs(values)));
end
end
