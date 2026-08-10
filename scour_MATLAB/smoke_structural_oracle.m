function report = smoke_structural_oracle
%SMOKE_STRUCTURAL_ORACLE Independent bridge/rail FEM and BC verification.
%
% This test compares production B02/B03/B09/B24 and the B54 structural-block
% embedding with vv_euler_bernoulli_reference, whose expected matrices come
% from exact polynomial integration rather than the production coefficients.
% It is code-verification evidence, not physical model validation.

fprintf('STRUCTURAL ORACLE: independent Euler--Bernoulli verification\n');
mutation_count = 0;
mutation_count = mutation_count + local_check_elements_and_units();
mutation_count = mutation_count + local_check_small_assemblies();
support_report = local_check_registered_support_mapping();
geometry_report = local_compare_l60_historical_and_aligned_meshes();
mutation_count = mutation_count + local_check_elastic_supports();
[modal_report,rigid_mutations] = local_check_rigid_static_modal_damping();
mutation_count = mutation_count + rigid_mutations;
mutation_count = mutation_count + local_check_free_and_fixed_cases();
mutation_count = mutation_count + local_check_b54_structural_embedding();

report = struct();
report.status = 'PASS';
report.mutations_rejected = mutation_count;
report.support_mapping = support_report;
report.L60_historical_vs_aligned = geometry_report;
report.simply_supported_modal = modal_report;
fprintf(['STRUCTURAL ORACLE: PASS (element, assembly, free/elastic/rigid BC, ' ...
    'static, modal, damping, B54; %d plausible mutations rejected)\n'], ...
    mutation_count);
end

function comparison = local_compare_l60_historical_and_aligned_meshes
% Diagnostic requested for the registered three-span L60 geometry.  The
% h=0.20 m candidate has three bridge elements per 0.60 m sleeper bay, so all
% requested support coordinates are nodes; h=0.30 m has two per bay and snaps
% the internal supports by 0.10 m.  No acceptance claim is made here.

L = 60;
n_spans = 3;
E = 35e9; I = 0.33; rho = 9600; A = 1;
P = 137.3e3;
mesh_h = [0.30,0.20];
labels = {'historical_snapped_h0p30','aligned_h0p20'};
result = repmat(struct('label','','h_m',0,'support_x_m',[], ...
    'frequency_hz',[],'load_x_m',0,'load_deflection_m',0, ...
    'spring_reaction_N',[],'force_balance_N',0, ...
    'strain_energy_J',0),1,2);

for item = 1:2
    h = mesh_h(item);
    n_elements = round(L/h);
    lengths = repmat(L/n_elements,1,n_elements);
    requested_supports = linspace(0,L,n_spans+1);
    if item == 1
        % B02 rejects requested L60 coordinates on h=.30.  These explicit
        % coordinates reproduce the historical nearest-node model solely for
        % this diagnostic; they cannot arise through silent production snap.
        support_locations = [0,20.1,39.9,60];
    else
        support_locations = requested_supports;
    end
    bc = struct('loc',support_locations, ...
        'vert_stiff',ones(1,n_spans+1), ...
        'rot_stiff',[1,0,0,1]);
    Damage = struct('scour_rates',zeros(1,n_spans+1), ...
        'bearing_left',0,'bearing_right',0);
    Beam = local_production_beam(lengths,E,I,rho,A,bc,Damage);
    Beam = B09_BeamFrq(Beam,local_modal_calc());
    support_x = Beam.Mesh.Nodes.acum(Beam.BC.loc_ind);
    [~,load_node] = min(abs(Beam.Mesh.Nodes.acum-L/2));
    load_dof = 2*load_node-1;
    force = zeros(Beam.Mesh.DOF.Tnum,1);
    force(load_dof) = -P;
    displacement = Beam.Mesh.Kg\force;
    support_dof = 2*Beam.BC.loc_ind-1;
    spring_reaction = -344e6*displacement(support_dof);

    result(item).label = labels{item};
    result(item).h_m = h;
    result(item).support_x_m = support_x;
    result(item).frequency_hz = Beam.Modal.f(1:5).';
    result(item).load_x_m = Beam.Mesh.Nodes.acum(load_node);
    result(item).load_deflection_m = displacement(load_dof);
    result(item).spring_reaction_N = spring_reaction.';
    result(item).force_balance_N = sum(spring_reaction)-P;
    result(item).strain_energy_J = ...
        0.5*displacement.'*Beam.Mesh.Kg*displacement;
    assert(abs(result(item).force_balance_N) <= 1e-9*P, ...
        'smoke_structural_oracle:StaticForceBalance', ...
        '%s spring-force residual %.6g N exceeds 1e-9 of the load.', ...
        labels{item},result(item).force_balance_N);
end

comparison = struct();
comparison.historical_snapped = result(1);
comparison.aligned = result(2);
comparison.relative_frequency_change = ...
    (result(2).frequency_hz-result(1).frequency_hz) ./ ...
    result(1).frequency_hz;
comparison.relative_deflection_magnitude_change = ...
    (abs(result(2).load_deflection_m)-abs(result(1).load_deflection_m)) / ...
    abs(result(1).load_deflection_m);
comparison.relative_energy_change = ...
    (result(2).strain_energy_J-result(1).strain_energy_J) / ...
    result(1).strain_energy_J;

fprintf('  L60 geometry diagnostic (historical snapped h=.30 -> aligned h=.20):\n');
fprintf('    supports: [%s] -> [%s] m\n', ...
    local_number_list(result(1).support_x_m), ...
    local_number_list(result(2).support_x_m));
fprintf('    f1..f5 historical snapped [Hz]: [%s]\n', ...
    local_number_list(result(1).frequency_hz));
fprintf('    f1..f5 aligned [Hz]: [%s]\n', ...
    local_number_list(result(2).frequency_hz));
fprintf('    relative aligned-minus-historical frequency change: [%s]\n', ...
    local_number_list(comparison.relative_frequency_change));
fprintf('    w(30 m): %.9e -> %.9e m; relative magnitude change %.3e\n', ...
    result(1).load_deflection_m,result(2).load_deflection_m, ...
    comparison.relative_deflection_magnitude_change);
fprintf('    spring reactions historical snapped [N]: [%s]\n', ...
    local_number_list(result(1).spring_reaction_N));
fprintf('    spring reactions aligned [N]: [%s]\n', ...
    local_number_list(result(2).spring_reaction_N));
end

function mutation_count = local_check_elements_and_units
% Two deliberately non-round property sets challenge every element entry.

cases = [31.73e9,0.2847,2647.3,1.173,2.371; ...
         42.19e9,0.1937,7823.4,0.0837,0.713];
for item = 1:size(cases,1)
    E = cases(item,1); I = cases(item,2); rho = cases(item,3);
    A = cases(item,4); L = cases(item,5);
    Ref = vv_euler_bernoulli_reference(E,I,rho,A,L);
    Beam = local_production_beam(L,E,I,rho,A,local_free_bc(),struct());
    local_assert_close(full(Beam.Mesh.Mg),Ref.M, ...
        sprintf('element %d consistent mass',item));
    local_assert_close(full(Beam.Mesh.Kg),Ref.K, ...
        sprintf('element %d bending stiffness',item));
    local_assert_symmetric_positive_mass(Ref.M, ...
        sprintf('element %d mass',item));
    local_assert_close(Ref.K,Ref.K.',sprintf('element %d K symmetry',item));
    local_assert_small(Ref.K*Ref.translation_mode,Ref.K, ...
        sprintf('element %d translation null mode',item));
    local_assert_small(Ref.K*Ref.rotation_mode,Ref.K, ...
        sprintf('element %d rotation null mode',item));

    q = [2.13e-3;-8.27e-4;-1.19e-3;4.41e-4];
    qdot = [-0.371;0.083;0.247;-0.061];
    [strain_energy,kinetic_energy] = ...
        local_gauss_field_energies(q,qdot,E,I,rho,A,L);
    local_assert_scalar_close(0.5*q.'*Beam.Mesh.Kg*q,strain_energy, ...
        sprintf('element %d strain energy [J]',item));
    local_assert_scalar_close(0.5*qdot.'*Beam.Mesh.Mg*qdot,kinetic_energy, ...
        sprintf('element %d kinetic energy [J]',item));
end

% SI dependency and length exponents for mixed displacement/rotation DOFs.
E = cases(1,1); I = cases(1,2); rho = cases(1,3);
A = cases(1,4); L = cases(1,5);
Ref = vv_euler_bernoulli_reference(E,I,rho,A,L);
Ref2L = vv_euler_bernoulli_reference(E,I,rho,A,2*L);
is_rotation = [0,1,0,1];
mass_exponent = 1 + is_rotation.' + is_rotation;
stiffness_exponent = -3 + is_rotation.' + is_rotation;
local_assert_close(Ref2L.M,Ref.M.*(2.^mass_exponent), ...
    'element length scaling of generalized mass units');
local_assert_close(Ref2L.K,Ref.K.*(2.^stiffness_exponent), ...
    'element length scaling of generalized stiffness units');
Ref2E = vv_euler_bernoulli_reference(2*E,I,rho,A,L);
Ref2I = vv_euler_bernoulli_reference(E,2*I,rho,A,L);
Ref2rho = vv_euler_bernoulli_reference(E,I,2*rho,A,L);
Ref2A = vv_euler_bernoulli_reference(E,I,rho,2*A,L);
local_assert_close(Ref2E.K,2*Ref.K,'K linear in E [N/m^2]');
local_assert_close(Ref2I.K,2*Ref.K,'K linear in I [m^4]');
local_assert_close(Ref2rho.M,2*Ref.M,'M linear in rho [kg/m^3]');
local_assert_close(Ref2A.M,2*Ref.M,'M linear in A [m^2]');
local_assert_close(Ref2E.M,Ref.M,'mass independent of E');
local_assert_close(Ref2rho.K,Ref.K,'stiffness independent of rho');

mutation_count = 0;
mutated = Ref.K;
mutated([1,3],[1,3]) = -mutated([1,3],[1,3]);
local_expect_mismatch(mutated,Ref.K,'mutated stiffness signs');
mutation_count = mutation_count + 1;
mutated = diag(sum(Ref.M,2));
local_expect_mismatch(mutated,Ref.M,'lumped-for-consistent mass mutation');
mutation_count = mutation_count + 1;
fprintf('  element/units: 2 non-round fixtures, exact energies and scaling PASS\n');
end

function mutation_count = local_check_small_assemblies
fixtures = { ...
    struct('L',[1.171,0.829], ...
        'E',[30.17e9,36.29e9],'I',[0.217,0.193], ...
        'rho',[2513.7,2681.9],'A',[1.071,0.943]), ...
    struct('L',[0.713,1.127,0.887], ...
        'E',[28.31e9,35.73e9,32.11e9], ...
        'I',[0.183,0.241,0.207], ...
        'rho',[2437.1,2719.3,2591.7], ...
        'A',[0.917,1.113,0.981])};
mutation_count = 0;
for item = 1:numel(fixtures)
    F = fixtures{item};
    Ref = vv_euler_bernoulli_reference(F.E,F.I,F.rho,F.A,F.L);
    Beam = local_production_beam( ...
        F.L,F.E,F.I,F.rho,F.A,local_free_bc(),struct());
    local_assert_close(full(Beam.Mesh.Mg),Ref.M, ...
        sprintf('%d-element assembled M',numel(F.L)));
    local_assert_close(full(Beam.Mesh.Kg),Ref.K, ...
        sprintf('%d-element assembled K',numel(F.L)));
    local_assert_symmetric_positive_mass(Ref.M, ...
        sprintf('%d-element global mass',numel(F.L)));
    local_assert_small(Ref.K*Ref.translation_mode,Ref.K, ...
        sprintf('%d-element global translation mode',numel(F.L)));
    local_assert_small(Ref.K*Ref.rotation_mode,Ref.K, ...
        sprintf('%d-element global rotation mode',numel(F.L)));
    singular_values = svd(Ref.K);
    nullity = sum(singular_values <= 1e-10*singular_values(1));
    assert(nullity == 2, ...
        'smoke_structural_oracle:RigidNullityMismatch', ...
        '%d-element free beam has %d rather than 2 rigid modes.', ...
        numel(F.L),nullity);
    local_assert_scalar_close( ...
        Ref.translation_mode.'*Ref.M*Ref.translation_mode, ...
        Ref.total_mass_kg, ...
        sprintf('%d-element total translational mass [kg]',numel(F.L)));
end

F = fixtures{2};
Ref = vv_euler_bernoulli_reference(F.E,F.I,F.rho,F.A,F.L);
mutated = Ref.K;
shared = 3:4;
mutated(shared,shared) = mutated(shared,shared) - ...
    Ref.element(2).K(1:2,1:2);
local_expect_mismatch(mutated,Ref.K,'missing shared-node accumulation');
mutation_count = mutation_count + 1;
fprintf('  assembly: independent Boolean transform for 2/3 elements PASS\n');
end

function support_report = local_check_registered_support_mapping
% Exercise the alignment gate on the historical universal h=0.30 m grid.
% Current production selects h=0.20 m for L60/3 and retains h=0.30 m for
% L99.6/4; this fixture proves why the old L60 grid is now rejected.
designs = [60.0,3;99.6,4];
h = 0.30;
support_report = repmat(struct( ...
    'length_m',0,'num_spans',0,'requested_m',[], ...
    'realized_m',[],'max_error_m',0,'production_accepts',false), ...
    1,size(designs,1));
for item = 1:size(designs,1)
    L = designs(item,1);
    n_spans = designs(item,2);
    n_elements = round(L/h);
    nodes = (0:n_elements)*h;
    bc = struct('loc',linspace(0,L,n_spans+1), ...
        'vert_stiff',ones(1,n_spans+1), ...
        'rot_stiff',[1,zeros(1,n_spans-1),1]);
    Beam.Mesh.Nodes = struct('acum',nodes,'Tnum',numel(nodes));
    Beam.Mesh.Ele = struct('Tnum',n_elements);
    Beam.Modal = struct();
    Damage = struct('scour_rates',zeros(1,n_spans+1), ...
        'bearing_left',0,'bearing_right',0);
    expected_index = round(bc.loc/h)+1;
    realized = nodes(expected_index);
    error_m = abs(realized-bc.loc);
    assert(numel(unique(expected_index)) == n_spans+1, ...
        'smoke_structural_oracle:DuplicateSupportNode', ...
        'L=%.1f mapped distinct supports onto one node.',L);
    assert(all(error_m <= h/2 + 64*eps(L)), ...
        'smoke_structural_oracle:SupportMappingTooFar', ...
        'L=%.1f support mapping exceeds half an element.',L);
    support_report(item).length_m = L;
    support_report(item).num_spans = n_spans;
    support_report(item).requested_m = bc.loc;
    support_report(item).realized_m = realized;
    support_report(item).max_error_m = max(error_m);
    Beam.BC = bc;
    alignment_tolerance = 256*eps(max([abs(nodes),abs(bc.loc),1]));
    if max(error_m) <= alignment_tolerance
        [Beam,~] = B02_BoundaryConditions(Beam,Damage);
        support_report(item).production_accepts = true;
        assert(isequal(Beam.BC.loc_ind,expected_index), ...
            'smoke_structural_oracle:SupportIndexMismatch', ...
            'L=%.1f production support indices differ from the oracle.',L);
    else
        local_expect_error(@() B02_BoundaryConditions(Beam,Damage), ...
            'B02:SupportNotOnNode','misaligned positive-spring support');
    end
    fprintf('  support map: L=%4.1f m/%d spans, max |x_node-x_support|=%.3f m\n', ...
        L,n_spans,max(error_m));
end
local_assert_scalar_close(support_report(1).max_error_m,0.10, ...
    'registered L60 support-coordinate discretization');
assert(support_report(2).max_error_m <= 1e-12, ...
    'smoke_structural_oracle:L99SupportMappingChanged', ...
    'Registered L99.6 supports no longer land exactly on the h=0.30 m mesh.');
end

function mutation_count = local_check_elastic_supports
lengths = [3.173,2.827];
E = 34.71e9; I = 0.317; rho = 9237.5; A = 1.037;
Ref = vv_euler_bernoulli_reference(E,I,rho,A,lengths);
locations = [0,lengths(1),sum(lengths)];
bc = struct('loc',locations,'vert_stiff',[1,1,1], ...
    'rot_stiff',[1,1,1]);
base_damage = struct('scour_rates',[0,0,0], ...
    'bearing_left',0,'bearing_right',0);
damage30 = base_damage; damage30.scour_rates(2) = 0.30;
damage60 = base_damage; damage60.scour_rates(2) = 0.60;

Beam0 = local_production_beam( ...
    lengths,E,I,rho,A,bc,base_damage);
Beam30 = local_production_beam( ...
    lengths,E,I,rho,A,bc,damage30);
Beam60 = local_production_beam( ...
    lengths,E,I,rho,A,bc,damage60);
vertical_dof = [1,3,5];
k_vertical = 344e6;
expected0 = Ref.K;
expected0(sub2ind(size(expected0),vertical_dof,vertical_dof)) = ...
    expected0(sub2ind(size(expected0),vertical_dof,vertical_dof)) + k_vertical;
expected30 = expected0;
expected30(3,3) = expected30(3,3) - 0.30*k_vertical;
expected60 = expected0;
expected60(3,3) = expected60(3,3) - 0.60*k_vertical;
local_assert_close(full(Beam0.Mesh.Kg),expected0,'zero-scour elastic K');
local_assert_close(full(Beam30.Mesh.Kg),expected30,'30-percent scour elastic K');
local_assert_close(full(Beam60.Mesh.Kg),expected60,'60-percent scour elastic K');
local_assert_close(full(Beam30.Mesh.Kg-Beam0.Mesh.Kg), ...
    expected30-expected0,'30-percent single-support delta K');
local_assert_close(full(Beam60.Mesh.Kg-Beam0.Mesh.Kg), ...
    expected60-expected0,'60-percent single-support delta K');
local_assert_close(full(Beam0.Mesh.Mg),Ref.M,'elastic support mass invariance');
local_assert_close(full(Beam30.Mesh.Mg),Ref.M,'scour mass invariance');
assert(Beam0.Modal.num_rigid_modes == 0 && ...
    Beam30.Modal.num_rigid_modes == 0 && Beam60.Modal.num_rigid_modes == 0, ...
    'smoke_structural_oracle:ElasticRigidModeCount', ...
    'Three positive vertical springs must remove both rigid modes.');

bearing_damage = base_damage;
bearing_damage.bearing_left = 1.237e9;
bearing_damage.bearing_right = 0.863e9;
Bearing = local_production_beam( ...
    lengths,E,I,rho,A,bc,bearing_damage);
expected_bearing = expected0;
expected_bearing(2,2) = expected_bearing(2,2) + ...
    bearing_damage.bearing_left;
expected_bearing(6,6) = expected_bearing(6,6) + ...
    bearing_damage.bearing_right;
local_assert_close(full(Bearing.Mesh.Kg),expected_bearing, ...
    'abutment-only rotational spring K');
assert(Bearing.Mesh.Kg(4,4) == Beam0.Mesh.Kg(4,4), ...
    'smoke_structural_oracle:IntermediateBearingChanged', ...
    'Intermediate support rotation changed under abutment fixity.');

mutated = expected30;
mutated(2,2) = mutated(2,2) - 0.30*k_vertical;
mutated(3,3) = mutated(3,3) + 0.30*k_vertical;
local_expect_mismatch(mutated,expected30,'scour applied at rotational DOF');
mutation_count = 1;
fprintf('  elastic BC: exact 0/30/60%% scour and abutment-only k_r PASS\n');
end

function [modal_report,mutation_count] = ...
        local_check_rigid_static_modal_damping
L = 7.4; E = 31.73e9; I = 0.2847; rho = 8713.2; A = 1.173;
lengths = [L/2,L/2];
Ref = vv_euler_bernoulli_reference(E,I,rho,A,lengths);
bc = struct('loc',[0,L],'vert_stiff',[-1,-1],'rot_stiff',[0,0]);
Beam = local_production_beam(lengths,E,I,rho,A,bc,struct());
fixed = [1,5];
local_check_artificial_constraints(Beam,Ref,fixed,'simply supported');

% Exact nodal midspan point-load solution and independent reactions.
P = 137.3e3;
force = zeros(6,1); force(3) = -P;
free = setdiff(1:6,fixed);
u = zeros(6,1);
u(free) = Ref.K(free,free)\force(free);
reaction = Ref.K*u-force;
analytical_deflection = -P*L^3/(48*E*I);
local_assert_scalar_close(u(3),analytical_deflection, ...
    'simply-supported midspan deflection [m]');
local_assert_scalar_close(reaction(1),P/2,'left vertical reaction [N]');
local_assert_scalar_close(reaction(5),P/2,'right vertical reaction [N]');
local_assert_scalar_close(reaction(1)+reaction(5)+sum(force(1:2:end)), ...
    0,'discrete vertical equilibrium [N]');
local_assert_scalar_close(u.'*Ref.K*u,u.'*force, ...
    'static virtual work equality [J]');
local_assert_scalar_close(0.5*u.'*Ref.K*u,0.5*u.'*force, ...
    'static strain energy equality [J]');

% Mesh-error table for analytical simply-supported frequencies.
n_elements = [2,4,8];
relative_error = zeros(numel(n_elements),2);
minimum_mac = zeros(1,numel(n_elements));
for level = 1:numel(n_elements)
    n = n_elements(level);
    level_lengths = repmat(L/n,1,n);
    LevelRef = vv_euler_bernoulli_reference(E,I,rho,A,level_lengths);
    LevelBeam = local_production_beam( ...
        level_lengths,E,I,rho,A,bc,struct());
    LevelBeam = B09_BeamFrq(LevelBeam,local_modal_calc());
    analytical_f = (1:2).^2*pi/(2*L^2)*sqrt(E*I/(rho*A));
    relative_error(level,:) = abs(LevelBeam.Modal.f(1:2).'-analytical_f) ...
        ./analytical_f;
    minimum_mac(level) = local_assert_modes_match( ...
        LevelBeam,LevelRef,[1,2*n+1],2,'simply-supported modes');
end
assert(all(diff(relative_error(:,1)) < 0) && ...
    all(diff(relative_error(:,2)) < 0), ...
    'smoke_structural_oracle:ModalConvergenceLost', ...
    'Analytical frequency error must decrease from 2 to 4 to 8 elements.');
fprintf('  simply-supported frequency relative errors (n=2/4/8):\n');
for level = 1:numel(n_elements)
    fprintf('    %2d elements: f1 %.3e, f2 %.3e, min MAC %.12f\n', ...
        n_elements(level),relative_error(level,1), ...
        relative_error(level,2),minimum_mac(level));
end

% Recheck production Rayleigh calibration on independently matched modes.
n = n_elements(end);
level_lengths = repmat(L/n,1,n);
LevelRef = vv_euler_bernoulli_reference(E,I,rho,A,level_lengths);
LevelBeam = local_production_beam( ...
    level_lengths,E,I,rho,A,bc,struct());
LevelBeam = B09_BeamFrq(LevelBeam,local_modal_calc());
LevelBeam.Damping.per = 2.73;
LevelBeam = B24_BeamDamping(LevelBeam);
zeta_target = LevelBeam.Damping.per/100;
wr = LevelBeam.Modal.w(1:2);
independent_coefficients = [1./(2*wr(:)),wr(:)/2] \ ...
    [zeta_target;zeta_target];
local_assert_scalar_close(LevelBeam.Damping.rayleigh_alpha, ...
    independent_coefficients(1),'Rayleigh alpha [1/s]');
local_assert_scalar_close(LevelBeam.Damping.rayleigh_beta, ...
    independent_coefficients(2),'Rayleigh beta [s]');
[reference_modes,reference_w] = ...
    local_reference_elastic_modes(LevelRef,[1,2*n+1]);
for mode = 1:2
    phi = reference_modes(:,mode);
    modal_zeta = (phi.'*LevelBeam.Mesh.Cg*phi) / ...
        (2*reference_w(mode)*(phi.'*LevelBeam.Mesh.Mg*phi));
    local_assert_scalar_close(modal_zeta,zeta_target, ...
        sprintf('Rayleigh target mode %d',mode));
end

mutation_count = 0;
mutated_mass = full(Beam.Mesh.Mg);
mutated_mass(fixed(1),fixed(1)) = 1;
local_expect_mismatch(mutated_mass,full(Beam.Mesh.Mg), ...
    'fixed-DOF artificial mass diagonal');
mutation_count = mutation_count + 1;
corrupt = LevelBeam;
corrupt.Modal.modes(:,1) = LevelBeam.Modal.modes(:,1) + ...
    0.05*LevelBeam.Modal.modes(:,2);
local_expect_mode_mismatch(corrupt,LevelRef,[1,2*n+1],2, ...
    'contaminated elastic mode shape');
mutation_count = mutation_count + 1;

modal_report = struct('element_counts',n_elements, ...
    'relative_frequency_error',relative_error, ...
    'minimum_mac',minimum_mac,'damping_target',zeta_target);
fprintf('  rigid/static/modal/damping: reactions, energy, MAC and zeta PASS\n');
end

function mutation_count = local_check_free_and_fixed_cases
% Free-free is the rail BC when redux=0; fixed-fixed exercises all rigid BCs.
lengths = [0.713,1.127,0.887];
E = 30.91e9; I = 0.2037; rho = 2631.4; A = 0.973;
Ref = vv_euler_bernoulli_reference(E,I,rho,A,lengths);
Free = local_production_beam( ...
    lengths,E,I,rho,A,local_free_bc(),struct());
Free = B09_BeamFrq(Free,local_modal_calc());
assert(Free.Modal.num_rigid_modes == 2 && ...
    numel(Free.Modal.w) >= 4 && all(Free.Modal.w(1:2) == 0), ...
    'smoke_structural_oracle:FreeRigidModes', ...
    'Free-free production beam/rail must retain exactly two zero modes.');
minimum_mac = local_assert_modes_match(Free,Ref,[],2,'free-free elastic modes');
assert(minimum_mac > 1-1e-10, ...
    'smoke_structural_oracle:FreeModeMAC', ...
    'Free-free elastic mode MAC is below contract.');
Free.Damping.per = 1.91;
Free = B24_BeamDamping(Free);
assert(isequal(Free.Damping.reference_mode_indices,[3,4]), ...
    'smoke_structural_oracle:FreeDampingIndices', ...
    'Free-free damping must skip its two rigid modes.');
zeta = Free.Damping.rayleigh_alpha./(2*Free.Modal.w(3:4)) + ...
    Free.Damping.rayleigh_beta.*Free.Modal.w(3:4)/2;
local_assert_close(zeta,[0.0191;0.0191], ...
    'free-free Rayleigh elastic-mode targets');

L = sum(lengths);
fixed_bc = struct('loc',[0,L],'vert_stiff',[-1,-1], ...
    'rot_stiff',[-1,-1]);
Fixed = local_production_beam( ...
    lengths,E,I,rho,A,fixed_bc,struct());
fixed_dof = [1,2,7,8];
local_check_artificial_constraints(Fixed,Ref,fixed_dof,'fixed-fixed');
Fixed = B09_BeamFrq(Fixed,local_modal_calc());
local_assert_modes_match(Fixed,Ref,fixed_dof,2,'fixed-fixed modes');
fprintf('  free/fixed BC: 2 rail-like rigid modes and exact elimination PASS\n');
mutation_count = 0;
end

function mutation_count = local_check_b54_structural_embedding
% Isolate B54's rail/beam blocks by zeroing every inter-layer property.
Beam = local_production_beam( ...
    [1,1],33.17e9,0.217,2673.4,1.031,local_free_bc(),struct());
Beam.Mesh.Cg = sparse(size(Beam.Mesh.Mg,1),size(Beam.Mesh.Mg,2));
Beam.Mesh.Ele.num_per_spacing = 1;
Rail = local_production_beam( ...
    [1,1,1,1],210.3e9,3.07e-5,7851.7,7.69e-3, ...
    local_free_bc(),struct());
Rail.Mesh.Cg = sparse(size(Rail.Mesh.Mg,1),size(Rail.Mesh.Mg,2));
Rail.Mesh.Ele.num_per_spacing = 1;

Track.Sleeper.spacing = 1;
Track.Sleeper.Prop.m = 0;
Track.Pad.Prop = struct('k',0,'c',0);
Track.Ballast.Prop = struct('k',0,'c',0,'m',0);
Track.BallastOnBeam.Prop = struct('k',0,'c',0,'m',0);
Track.SubBallast.Prop = struct('k',0,'c',0);
Track.PadUnderSleeperOnBeam.included = 0;
Track.Rail = Rail;
Calc.Profile = struct('L',4,'max_TL',0,'L_Approach',1,'extra_L',0);
Calc.Options.redux_factor = 0;
Calc.Cte.tol = 1e-10;
Model = B54_ModelMatrices(Beam,Track,Calc,struct());

expected_mass = zeros(Model.Mesh.DOF.Tnum);
expected_stiffness = zeros(Model.Mesh.DOF.Tnum);
expected_mass(Model.Mesh.DOF.rail,Model.Mesh.DOF.rail) = full(Rail.Mesh.Mg);
expected_mass(Model.Mesh.DOF.beam,Model.Mesh.DOF.beam) = full(Beam.Mesh.Mg);
expected_stiffness(Model.Mesh.DOF.rail,Model.Mesh.DOF.rail) = ...
    full(Rail.Mesh.Kg);
expected_stiffness(Model.Mesh.DOF.beam,Model.Mesh.DOF.beam) = ...
    full(Beam.Mesh.Kg);
local_assert_close(full(Model.Mesh.Mg),expected_mass, ...
    'B54 isolated rail/beam mass embedding');
local_assert_close(full(Model.Mesh.Kg),expected_stiffness, ...
    'B54 isolated rail/beam stiffness embedding');
local_assert_close(full(Model.Mesh.Cg),zeros(size(expected_mass)), ...
    'B54 isolated zero damping embedding');

mutated = expected_stiffness;
rail_first = Model.Mesh.DOF.rail(1);
beam_first = Model.Mesh.DOF.beam(1);
mutated(rail_first,rail_first) = mutated(rail_first,rail_first) - ...
    Rail.Mesh.Kg(1,1);
mutated(beam_first,beam_first) = mutated(beam_first,beam_first) + ...
    Rail.Mesh.Kg(1,1);
local_expect_mismatch(mutated,expected_stiffness, ...
    'B54 rail stiffness inserted into beam block');
mutation_count = 1;
fprintf('  B54 embedding: isolated production rail and bridge blocks PASS\n');
end

function Beam = local_production_beam( ...
        lengths,E,I,rho,A,bc,Damage)
lengths = double(lengths(:).');
n_elements = numel(lengths);
Beam.Prop.L = sum(lengths);
Beam.Prop.E_n = local_expand(E,n_elements);
Beam.Prop.I_n = local_expand(I,n_elements);
Beam.Prop.rho_n = local_expand(rho,n_elements);
Beam.Prop.A_n = local_expand(A,n_elements);
Beam.Mesh.Ele.a = lengths;
Beam.Mesh.Ele.num = n_elements;
Beam.Mesh.Ele.Tnum = n_elements;
Beam.Mesh.Ele.num_nodes = 2;
Beam.Mesh.Ele.nodes = [(1:n_elements).',(2:n_elements+1).'];
first_dof = (1:2:2*n_elements).';
Beam.Mesh.Ele.DOF = [first_dof,first_dof+1,first_dof+2,first_dof+3];
Beam.Mesh.Nodes.acum = [0,cumsum(lengths)];
Beam.Mesh.Nodes.coord = Beam.Mesh.Nodes.acum.';
Beam.Mesh.Nodes.Tnum = n_elements+1;
Beam.Mesh.Nodes.num_DOF = 2;
Beam.Mesh.DOF.Tnum = 2*(n_elements+1);
Beam.Options.k_Mconsist = 1;
Beam.BC = bc;
Beam.Modal = struct();
[Beam,~] = B02_BoundaryConditions(Beam,Damage);
Beam = B03_BeamMatrices(Beam);
end

function values = local_expand(values,n_elements)
values = double(values(:));
if isscalar(values)
    values = repmat(values,n_elements,1);
else
    assert(numel(values) == n_elements, ...
        'smoke_structural_oracle:FixturePropertyCount', ...
        'Fixture property count does not match its element count.');
end
end

function bc = local_free_bc
bc = struct('loc',[],'vert_stiff',[],'rot_stiff',[]);
end

function Calc = local_modal_calc
Calc.Options.calc_beam_frq = 1;
Calc.Options.calc_beam_modes = 1;
Calc.Plot.P1_Beam_frq = 0;
Calc.Plot.P2_Beam_modes = 0;
end

function local_check_artificial_constraints(Beam,Ref,fixed_dof,label)
expected_K = Ref.K;
expected_M = Ref.M;
expected_K(fixed_dof,:) = 0; expected_K(:,fixed_dof) = 0;
expected_M(fixed_dof,:) = 0; expected_M(:,fixed_dof) = 0;
fixed_value = 344e6;
expected_K(sub2ind(size(expected_K),fixed_dof,fixed_dof)) = fixed_value;
expected_M(sub2ind(size(expected_M),fixed_dof,fixed_dof)) = fixed_value;
local_assert_close(full(Beam.Mesh.Kg),expected_K, ...
    [label ' constrained K versus exact elimination']);
local_assert_close(full(Beam.Mesh.Mg),expected_M, ...
    [label ' constrained M versus exact elimination']);

[vectors,lambda] = eig(full(Beam.Mesh.Kg),full(Beam.Mesh.Mg),'vector');
participation = sum(abs(vectors(fixed_dof,:)).^2,1) ./ ...
    sum(abs(vectors).^2,1);
artificial = participation > 1-1e-12;
assert(sum(artificial) == numel(fixed_dof), ...
    'smoke_structural_oracle:ArtificialModeCount', ...
    '%s has %d rather than %d artificial constraint modes.', ...
    label,sum(artificial),numel(fixed_dof));
artificial_error = max(abs(lambda(artificial)-1));
assert(artificial_error <= 1e-8, ...
    'smoke_structural_oracle:ArtificialEigenvalueMismatch', ...
    '%s artificial eigenvalues differ from 1 (rad/s)^2 by %.6g.', ...
    label,artificial_error);
end

function minimum_mac = local_assert_modes_match( ...
        Beam,Ref,fixed_dof,n_modes,label)
[reference_modes,reference_w] = ...
    local_reference_elastic_modes(Ref,fixed_dof);
first_elastic = Beam.Modal.num_rigid_modes+1;
elastic_indices = first_elastic:(first_elastic+n_modes-1);
actual_modes = Beam.Modal.modes(:,elastic_indices);
actual_w = Beam.Modal.w(elastic_indices);
reference_modes = reference_modes(:,1:n_modes);
reference_w = reference_w(1:n_modes);
mass = Ref.M;
mac = zeros(n_modes);
for actual = 1:n_modes
    for reference = 1:n_modes
        numerator = abs(actual_modes(:,actual).'*mass* ...
            reference_modes(:,reference))^2;
        denominator = (actual_modes(:,actual).'*mass*actual_modes(:,actual)) * ...
            (reference_modes(:,reference).'*mass*reference_modes(:,reference));
        mac(actual,reference) = real(numerator/denominator);
    end
end

used = false(1,n_modes);
matched_mac = zeros(1,n_modes);
matched_w = zeros(1,n_modes);
for reference = 1:n_modes
    candidates = find(~used);
    [matched_mac(reference),position] = max(mac(candidates,reference));
    actual = candidates(position);
    used(actual) = true;
    matched_w(reference) = actual_w(actual);
end
minimum_mac = min(matched_mac);
if minimum_mac < 1-1e-10
    error('smoke_structural_oracle:ModeMismatch', ...
        '%s minimum mass-normalized MAC %.12g is below contract.', ...
        label,minimum_mac);
end
local_assert_close(matched_w(:),reference_w(:),[label ' matched frequencies']);
end

function [modes,w] = local_reference_elastic_modes(Ref,fixed_dof)
n_dof = size(Ref.M,1);
free_dof = setdiff(1:n_dof,fixed_dof);
[vectors,lambda] = eig(Ref.K(free_dof,free_dof), ...
    Ref.M(free_dof,free_dof),'vector');
[lambda,order] = sort(real(lambda),'ascend');
vectors = real(vectors(:,order));
scale = max(1,max(abs(lambda)));
positive = lambda > 128*eps(scale);
lambda = lambda(positive);
vectors = vectors(:,positive);
modes = zeros(n_dof,numel(lambda));
modes(free_dof,:) = vectors;
for mode = 1:size(modes,2)
    modes(:,mode) = modes(:,mode) / ...
        sqrt(modes(:,mode).'*Ref.M*modes(:,mode));
end
w = sqrt(lambda);
end

function local_expect_mode_mismatch(Beam,Ref,fixed_dof,n_modes,label)
try
    local_assert_modes_match(Beam,Ref,fixed_dof,n_modes,label);
catch caught
    assert(strcmp(caught.identifier,'smoke_structural_oracle:ModeMismatch'), ...
        'smoke_structural_oracle:WrongMutationFailure', ...
        '%s raised %s rather than the MAC mismatch.',label,caught.identifier);
    return;
end
error('smoke_structural_oracle:MutationEscaped', ...
    '%s escaped the independent modal oracle.',label);
end

function [strain_energy,kinetic_energy] = ...
        local_gauss_field_energies(q,qdot,E,I,rho,A,L)
% Eight-point Gauss integration is exact for the degree-six mass integrand.
abscissa = [-0.9602898564975363,-0.7966664774136267, ...
    -0.5255324099163290,-0.1834346424956498, ...
     0.1834346424956498, 0.5255324099163290, ...
     0.7966664774136267, 0.9602898564975363];
weights = [0.1012285362903763,0.2223810344533745, ...
    0.3137066458778873,0.3626837833783620, ...
    0.3626837833783620,0.3137066458778873, ...
    0.2223810344533745,0.1012285362903763]/2;
xi = (abscissa+1)/2;
strain_integral = 0;
kinetic_integral = 0;
for point = 1:numel(xi)
    s = xi(point);
    shape = [1-3*s^2+2*s^3;L*(s-2*s^2+s^3); ...
        3*s^2-2*s^3;L*(-s^2+s^3)];
    curvature = [(-6+12*s)/L^2;(-4+6*s)/L; ...
        (6-12*s)/L^2;(-2+6*s)/L];
    strain_integral = strain_integral + ...
        weights(point)*(curvature.'*q)^2;
    kinetic_integral = kinetic_integral + ...
        weights(point)*(shape.'*qdot)^2;
end
strain_energy = 0.5*E*I*L*strain_integral;
kinetic_energy = 0.5*rho*A*L*kinetic_integral;
end

function local_assert_symmetric_positive_mass(matrix,label)
local_assert_close(matrix,matrix.',[label ' symmetry']);
[~,flag] = chol((matrix+matrix.')/2);
assert(flag == 0,'smoke_structural_oracle:MassNotPositiveDefinite', ...
    '%s is not positive definite.',label);
end

function local_assert_small(residual,reference,label)
limit = 2e-12*max(1,norm(reference,'fro'));
if norm(residual,inf) > limit
    error('smoke_structural_oracle:ResidualMismatch', ...
        '%s residual %.6g exceeds %.6g.',label,norm(residual,inf),limit);
end
end

function local_assert_scalar_close(actual,expected,label)
error_value = abs(actual-expected);
limit = 5e-12*max([1,abs(actual),abs(expected)]);
if error_value > limit
    error('smoke_structural_oracle:ScalarMismatch', ...
        '%s differs: actual %.16g, expected %.16g, error %.6g.', ...
        label,actual,expected,error_value);
end
end

function local_assert_close(actual,expected,label)
if ~isequal(size(actual),size(expected))
    error('smoke_structural_oracle:MatrixMismatch', ...
        '%s size differs from the independent reference.',label);
end
error_value = norm(double(actual(:))-double(expected(:)),inf);
limit = 5e-12*max(1,norm(double(expected(:)),inf));
if error_value > limit
    error('smoke_structural_oracle:MatrixMismatch', ...
        '%s max error %.6g exceeds %.6g.',label,error_value,limit);
end
end

function local_expect_mismatch(mutated,reference,label)
try
    local_assert_close(mutated,reference,label);
catch caught
    assert(strcmp(caught.identifier,'smoke_structural_oracle:MatrixMismatch'), ...
        'smoke_structural_oracle:WrongMutationFailure', ...
        '%s raised %s rather than a matrix mismatch.',label,caught.identifier);
    return;
end
error('smoke_structural_oracle:MutationEscaped', ...
    '%s escaped the independent matrix oracle.',label);
end

function local_expect_error(call,expected_identifier,label)
try
    call();
catch caught
    assert(strcmp(caught.identifier,expected_identifier), ...
        'smoke_structural_oracle:WrongExpectedError', ...
        '%s raised %s rather than %s.', ...
        label,caught.identifier,expected_identifier);
    return;
end
error('smoke_structural_oracle:MissingExpectedError', ...
    '%s did not raise %s.',label,expected_identifier);
end

function value = local_number_list(numbers)
value = strtrim(sprintf('%.9g ',numbers));
end
