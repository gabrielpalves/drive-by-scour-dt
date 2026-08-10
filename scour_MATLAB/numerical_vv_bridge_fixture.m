function run = numerical_vv_bridge_fixture( ...
        bridge_length_m, mesh_level_id, point_load_N)
%NUMERICAL_VV_BRIDGE_FIXTURE Bounded simply-supported mesh fixture.
%
% This fixture calls production B01/B02/B03 for matrix construction, then
% applies exact free-DOF elimination and compares static/modal QoIs with
% independent simply-supported Euler--Bernoulli formulas.  It is useful for
% exercising a refinement harness, but it is NOT the independent assembly/BC
% oracle required for qualification and does not exercise the coupled solver.

P = numerical_vv_protocol_definition();
if ~isscalar(bridge_length_m) || ~isfinite(bridge_length_m) || ...
        ~any(abs(bridge_length_m-[P.geometries.bridge_length_m]) < 1e-12)
    error('numerical_vv:UnregisteredGeometry', ...
        'Bridge length must be one registered geometry.');
end
if ~(ischar(mesh_level_id) || ...
        (isstring(mesh_level_id) && isscalar(mesh_level_id)))
    error('numerical_vv:BadMeshLevel', 'Mesh level must be one text scalar.');
end
mesh_level_id = char(mesh_level_id);
mesh_index = strcmp({P.mesh_levels.id}, mesh_level_id);
if nnz(mesh_index) ~= 1
    error('numerical_vv:UnregisteredMeshLevel', ...
        'Unknown registered mesh level %s.', mesh_level_id);
end
if ~isscalar(point_load_N) || ~isfinite(point_load_N) || point_load_N <= 0
    error('numerical_vv:BadPointLoad', ...
        'Point load magnitude must be one finite positive scalar [N].');
end

geometry_index = abs([P.geometries.bridge_length_m]- ...
    bridge_length_m) < 1e-12;
geometry_id = P.geometries(geometry_index).id;
sequence_index = strcmp({P.geometry_mesh_sequences.geometry_id}, ...
    geometry_id) & strcmp({P.geometry_mesh_sequences.level_id}, ...
    mesh_level_id);
if nnz(sequence_index) ~= 1
    error('numerical_vv:MissingGeometryMeshSequence', ...
        'No unique mesh sequence exists for %s/%s.', ...
        geometry_id, mesh_level_id);
end
sequence = P.geometry_mesh_sequences(sequence_index);
h_requested = sequence.bridge_nominal_h_m;
n_elements_real = bridge_length_m / h_requested;
n_elements = round(n_elements_real);
if abs(n_elements_real-n_elements) > 256*eps(max(n_elements_real, 1))
    error('numerical_vv:BridgeLengthNotNested', ...
        'Registered bridge length is not divisible by the mesh spacing.');
end

seed = struct('Prop', struct('L', double(bridge_length_m), ...
    'num_spans', 1));
Beam = A03_Bridge(seed);
Beam.Prop.A = 1;
Beam.Mesh.Ele.num = n_elements;
Beam.Options.k_Mconsist = 1;
Beam.BC.loc = [0, bridge_length_m];
Beam.BC.vert_stiff = [-1, -1];
Beam.BC.rot_stiff = [0, 0];
Beam = B01_ElementsAndCoordinates(Beam);
Damage = struct('scour_rates', [0, 0], ...
    'bearing_left', 0, 'bearing_right', 0);
[Beam, ~] = B02_BoundaryConditions(Beam, Damage);

% Obtain the unmodified assembled matrices by disabling only B03's artificial
% fixed-row replacement.  The production constrained form is built separately
% and checked to have an identical free block.
RawBeam = Beam;
RawBeam.BC.DOF_fixed = [];
RawBeam.BC.num_DOF_fixed = 0;
RawBeam = B03_BeamMatrices(RawBeam);
ProductionBeam = B03_BeamMatrices(Beam);

fixed = Beam.BC.DOF_fixed(:);
all_dofs = (1:Beam.Mesh.DOF.Tnum)';
free = setdiff(all_dofs, fixed, 'stable');
K = RawBeam.Mesh.Kg;
M = RawBeam.Mesh.Mg;
Kff = K(free, free);
Mff = M(free, free);

mid_coordinate_error = abs(Beam.Mesh.Nodes.acum-bridge_length_m/2);
[mid_error_m, mid_node] = min(mid_coordinate_error);
% B01 forms coordinates by cumulative summation.  Use the same source-locked
% element-count-scaled roundoff allowance as B02 and the protocol alignment
% table, including at the conditional M3 resolution.
geometry_tol = max(256, 2*n_elements) * ...
    eps(max(bridge_length_m, 1));
if mid_error_m > geometry_tol
    error('numerical_vv:MissingMidspanNode', ...
        'The bridge fixture requires an exact midspan node.');
end
mid_dof = 2*mid_node-1;
mid_free_index = find(free == mid_dof);
if numel(mid_free_index) ~= 1
    error('numerical_vv:ConstrainedMidspan', ...
        'Midspan vertical DOF must be one free DOF.');
end

force = zeros(Beam.Mesh.DOF.Tnum, 1);
force(mid_dof) = -double(point_load_N);
u = zeros(size(force));
u(free) = Kff \ force(free);
residual = K*u-force;
reactions = residual(fixed);

EI = Beam.Prop.E * Beam.Prop.I;
rhoA = Beam.Prop.rho * Beam.Prop.A;
analytic_midspan_m = -point_load_N*bridge_length_m^3/(48*EI);
analytic_reaction_N = point_load_N/2;
strain_energy_J = 0.5 * full(u'*(K*u));
external_work_J = 0.5 * full(force'*u);

n_modes = 5;
opts = struct('tol', 1e-11, ...
    'maxit', max(1000, 5*size(Kff, 1)), 'disp', 0);
try
    [mode_free, lambda_matrix] = eigs(Kff, Mff, n_modes, ...
        'smallestabs', opts);
catch first_error
    try
        [mode_free, lambda_matrix] = eigs(Kff, Mff, n_modes, ...
            'smallestreal', opts);
    catch
        rethrow(first_error)
    end
end
lambda = real(diag(lambda_matrix));
[lambda, order] = sort(lambda, 'ascend');
mode_free = real(mode_free(:, order));
if any(~isfinite(lambda)) || any(lambda <= 0)
    error('numerical_vv:InvalidFixtureEigenvalue', ...
        'The bridge fixture returned a nonpositive/nonfinite elastic eigenvalue.');
end
frequency_hz = sqrt(lambda)/(2*pi);

x_nodes = Beam.Mesh.Nodes.acum(:);
analytic_frequency_hz = zeros(n_modes, 1);
mac_vs_analytic = zeros(n_modes, 1);
mode_full = zeros(Beam.Mesh.DOF.Tnum, n_modes);
for mode_number = 1:n_modes
    phi = mode_free(:, mode_number);
    phi = phi / sqrt(real(phi'*Mff*phi));
    mode_free(:, mode_number) = phi;
    mode_full(free, mode_number) = phi;

    analytic_full = zeros(Beam.Mesh.DOF.Tnum, 1);
    analytic_full(1:2:end) = sin(mode_number*pi*x_nodes/bridge_length_m);
    analytic_full(2:2:end) = (mode_number*pi/bridge_length_m) * ...
        cos(mode_number*pi*x_nodes/bridge_length_m);
    analytic_free = analytic_full(free);
    analytic_norm = sqrt(real(analytic_free'*Mff*analytic_free));
    if analytic_norm <= 0 || ~isfinite(analytic_norm)
        error('numerical_vv:InvalidAnalyticMode', ...
            'Analytic mode has invalid discrete mass norm.');
    end
    analytic_free = analytic_free / analytic_norm;
    mac_vs_analytic(mode_number) = abs(analytic_free'*Mff*phi)^2;
    analytic_frequency_hz(mode_number) = ...
        mode_number^2*pi/(2*bridge_length_m^2) * sqrt(EI/rhoA);
end

free_k_delta = local_max_abs( ...
    ProductionBeam.Mesh.Kg(free, free)-Kff);
free_m_delta = local_max_abs( ...
    ProductionBeam.Mesh.Mg(free, free)-Mff);
fixed_k_offdiag = ProductionBeam.Mesh.Kg(fixed, :);
fixed_m_offdiag = ProductionBeam.Mesh.Mg(fixed, :);
for k = 1:numel(fixed)
    fixed_k_offdiag(k, fixed(k)) = 0;
    fixed_m_offdiag(k, fixed(k)) = 0;
end

run = struct();
run.schema = 'numerical-vv-bridge-fixture-v1';
run.scope = 'nonqualifying-simply-supported-production-matrix-micro';
run.geometry_id = geometry_id;
run.mesh_level = mesh_level_id;
run.bridge_length_m = double(bridge_length_m);
run.bridge_elements_per_sleeper_bay = ...
    sequence.bridge_elements_per_sleeper_bay;
run.rail_elements_per_sleeper_bay = ...
    sequence.rail_elements_per_sleeper_bay;
run.bridge_nominal_h_m = sequence.bridge_nominal_h_m;
run.rail_nominal_h_m = sequence.rail_nominal_h_m;
run.bridge_actual_h_m = Beam.Prop.L/Beam.Mesh.Ele.Tnum;
run.rail_actual_h_m = NaN;
run.bridge_mesh_executed = true;
run.rail_mesh_executed = false;
run.bridge_n_elements = Beam.Mesh.Ele.Tnum;
run.bridge_n_nodes = Beam.Mesh.Nodes.Tnum;
run.node_coordinates_m = Beam.Mesh.Nodes.acum;
run.nominal_support_coordinates_m = Beam.BC.loc;
run.realized_support_coordinates_m = Beam.Mesh.Nodes.acum(Beam.BC.loc_ind);
run.support_signed_offsets_m = ...
    run.realized_support_coordinates_m-run.nominal_support_coordinates_m;
run.support_alignment_pass = ...
    all(abs(run.support_signed_offsets_m) <= geometry_tol);
run.point_load_N = double(point_load_N);
run.midspan_displacement_m = u(mid_dof);
run.analytic_midspan_displacement_m = analytic_midspan_m;
run.left_reaction_N = reactions(1);
run.right_reaction_N = reactions(2);
run.analytic_reaction_N = analytic_reaction_N;
run.force_balance_residual_N = sum(reactions)+sum(force);
run.static_residual_norm_N = norm(residual(free));
run.strain_energy_J = strain_energy_J;
run.external_work_J = external_work_J;
run.energy_residual_J = strain_energy_J-external_work_J;
run.frequency_hz = frequency_hz;
run.analytic_frequency_hz = analytic_frequency_hz;
run.mac_vs_analytic = mac_vs_analytic;
run.free_block_K_max_abs_delta = free_k_delta;
run.free_block_M_max_abs_delta = free_m_delta;
run.fixed_K_offdiag_max_abs = local_max_abs(fixed_k_offdiag);
run.fixed_M_offdiag_max_abs = local_max_abs(fixed_m_offdiag);
run.fixed_K_diagonal = full(diag(ProductionBeam.Mesh.Kg(fixed, fixed)));
run.fixed_M_diagonal = full(diag(ProductionBeam.Mesh.Mg(fixed, fixed)));
run.fixed_artificial_diagonal_expected = Beam.BC.DOF_fixed_value;
run.free_dofs = free;
run.displacement = u;
run.force = force;
run.mode_shapes = mode_full;
end

function value = local_max_abs(array)
if isempty(array)
    value = 0;
    return
end
values = nonzeros(array);
if isempty(values)
    value = 0;
else
    value = full(max(abs(values)));
end
end
