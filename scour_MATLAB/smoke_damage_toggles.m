% _smoke_damage_toggles.m — temporary smoke test for the new damage/EOV toggles
% (crack local-EI, profile intensity, psd_fra profile mode). Run from scour_MATLAB.

%% Test 1: B19 Type-2 (loaded profile) intensity capture & scaling
Calc1 = struct();
Calc1.Profile.Type = 2;
Calc1.Profile.min_dx = 0.01;
Calc1.Profile.L = 10;
Calc1.Position.x = [0 1];
Calc1.Plot.Profile_original = 0;
C_ref = B19_GenerateProfile(Calc1);

Calc2 = Calc1;
Calc2.Profile.intensity = 2.0;
C_scl = B19_GenerateProfile(Calc2);

assert(max(abs(C_scl.Profile.h - 2*C_ref.Profile.h)) < 1e-14, ...
    'T1 FAIL: Type-2 intensity scaling wrong');
fprintf('T1 OK: Type-2 profile x2 intensity (max|h| %.3e -> %.3e m)\n', ...
    max(abs(C_ref.Profile.h)), max(abs(C_scl.Profile.h)));

%% Test 2: full passage — scour + crack + psd_fra profile through B00
Damage = struct();
Damage.desvio        = 0;
Damage.scour_rates   = [0 0.30 0];   % 30% scour at the central pier
Damage.bearing_left  = 0;
Damage.bearing_right = 0;
Damage.crack_locs      = 30.0;       % crack at 30 m
Damage.crack_intensity = 0.22;       % 22% EI loss (Fernandes upper case)
Damage.crack_lc        = 0;          % single element

Train = A01_Train(80/3.6, zeros(5,3));
Track = A02_Track();
Beam_seed = struct();
Beam_seed.Prop.L = 40;
Beam_seed.Prop.num_spans = 2;
Beam = A03_Bridge(Beam_seed);

Profile_cfg = struct('mode','psd_fra','fra_class',5);
[Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);
[Sol, Calc, Train, Beam, Track] = B00_Calculations(Calc, Train, Track, Beam, Damage);

% crack: exactly ONE element with I reduced to 0.78*I
n_red = sum(abs(Beam.Prop.I_n - Beam.Prop.I) > 1e-12);
ratio = min(Beam.Prop.I_n) / Beam.Prop.I;
fprintf('T2: elements with reduced I = %d (expect 1); ratio = %.4f (expect 0.7800)\n', n_red, ratio);
assert(n_red == 1, 'T2 FAIL: crack did not hit exactly one element');
assert(abs(ratio - 0.78) < 1e-9, 'T2 FAIL: crack EI ratio wrong');

% profile: regenerated, non-trivial
assert(std(Calc.Profile.h) > 0, 'T2 FAIL: psd_fra profile is empty');
fprintf('T2: FRA class-5 profile std = %.4g mm over %.0f m\n', ...
    1000*std(Calc.Profile.h), Calc.Profile.L);

% solve produced a solution
assert(isfield(Sol,'Veh') && ~isempty(Sol.Veh), 'T2 FAIL: no vehicle solution');
w12 = Beam.Modal.w(1:2);
zeta12 = Beam.Damping.rayleigh_alpha ./ (2*w12) + ...
    Beam.Damping.rayleigh_beta .* w12 / 2;
assert(issorted(Beam.Modal.w), ...
    'T2 FAIL: production no-mode eigenfrequencies are not sorted');
assert(isequal(Beam.Damping.reference_mode_indices, [1 2]), ...
    'T2 FAIL: elastically supported bridge damping did not use modes 1 and 2');
assert(max(abs(zeta12 - Beam.Damping.per/100)) < 1e-10, ...
    'T2 FAIL: Rayleigh damping is not calibrated on elastic modes 1 and 2');
fprintf('T2: elastic-mode damping ratios = %.6f, %.6f (target %.6f)\n', ...
    zeta12(1), zeta12(2), Beam.Damping.per/100);
fprintf('T2 OK: dynamic solve completed with crack + FRA-PSD profile\n');

%% Test 3: omitted mode = the registered fixed-phase FRA-v2 baseline
[Calc3, ~, ~] = A04_Options(A03_Bridge(struct()), A02_Track());
assert(Calc3.Profile.Type == 1, 'T3 FAIL: default must be generated from PSD');
assert(Calc3.Profile.phase_seed == 20260728, ...
    'T3 FAIL: default fixed-phase seed changed');
assert(strcmp(Calc3.Profile.spectrum_contract, ...
    'fra-v2-class4-cycles-per-m-v1'), ...
    'T3 FAIL: default FRA-v2 spectrum contract changed');
fprintf('T3 OK: omitted mode selects the fixed-phase FRA-v2 class-4 baseline\n');

%% Test 4: analytic scour/support-fixity stiffness insertion through B02
Beam4 = struct();
Beam4.BC.loc = [0 20 40 60];
Beam4.BC.vert_stiff = ones(1, 4);
Beam4.BC.rot_stiff = [1 0 0 1];
Beam4.Mesh.Nodes.acum = [0 20 40 60];
Beam4.Mesh.Nodes.Tnum = 4;
Beam4.Modal = struct();
Damage4 = struct( ...
    'scour_rates', [0 0.30 0.60 0], ...
    'bearing_left', 1.25e9, ...
    'bearing_right', 3.75e9);
[Beam4, Damage4] = B02_BoundaryConditions(Beam4, Damage4);
expected_dofs4 = [1 2 3 5 7 8];
expected_k4 = [344e6 1.25e9 0.70*344e6 0.40*344e6 344e6 3.75e9];
assert(isequal(Beam4.BC.DOF_with_values, expected_dofs4), ...
    'T4 FAIL: B02 inserted stiffness at the wrong structural DOF(s)');
assert(max(abs(Beam4.BC.DOF_stiff_values - expected_k4)) < 1e-6, ...
    'T4 FAIL: B02 scour/bearing stiffness mapping is not analytic expectation');
assert(Beam4.BC.DOF_fixed_value == 344e6, ...
    'T4 FAIL: nominal vertical support stiffness changed');
assert(Beam4.Modal.num_rigid_modes == 0, ...
    'T4 FAIL: elastically supported bridge was assigned rigid modes');
fprintf(['T4 OK: k_v=(1-d)*344e6 at every support and supplied k_r at ' ...
    'the two abutment rotational DOFs\n']);

%% Test 5: malformed physical inputs fail closed
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'scour_rates', [0 0.2 0]), ... %#ok<SFLD>
    'B02:ScourSupportCountMismatch');
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'scour_rates', [0 -0.1 0.2 0]), ... %#ok<SFLD>
    'B02:InvalidScourRates');
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'scour_rates', [0 0.1 1.01 0]), ... %#ok<SFLD>
    'B02:InvalidScourRates');
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'scour_rates', [0 0.1 NaN 0]), ... %#ok<SFLD>
    'B02:InvalidScourRates');
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'bearing_left', -1), ... %#ok<SFLD>
    'B02:InvalidBearingStiffness');
local_assert_b02_rejects(Beam4, ...
    setfield(Damage4, 'bearing_right', Inf), ... %#ok<SFLD>
    'B02:InvalidBearingStiffness');
Beam4_bad = Beam4;
Beam4_bad.BC.vert_stiff = [1 1 NaN 1];
local_assert_b02_rejects(Beam4_bad, Damage4, 'B02:InvalidVerticalBC');
Beam4_bad = Beam4;
Beam4_bad.BC.rot_stiff = [1 0 -0.5 1];
local_assert_b02_rejects(Beam4_bad, Damage4, 'B02:InvalidRotationalBC');

% B00 intentionally reuses the bridge Damage struct for a redux=0 rail with
% no boundary springs. That no-op path must remain valid.
Rail4 = struct();
Rail4.BC = struct('loc', [], 'vert_stiff', [], 'rot_stiff', []);
Rail4.Mesh.Nodes = struct('acum', [0 1], 'Tnum', 2);
Rail4.Modal = struct();
Rail4 = B02_BoundaryConditions(Rail4, Damage4);
assert(isempty(Rail4.BC.DOF_with_values), ...
    'T5 FAIL: no-support rail path unexpectedly acquired a spring');
assert(Rail4.Modal.num_rigid_modes == 2, ...
    'T5 FAIL: free-free rail must retain two beam rigid modes');

OneSpring4 = struct();
OneSpring4.BC = struct( ...
    'loc', 0, 'vert_stiff', 1, 'rot_stiff', 0);
OneSpring4.Mesh.Nodes = struct('acum', [0 1], 'Tnum', 2);
OneSpring4.Modal = struct();
OneSpringDamage4 = struct( ...
    'scour_rates', 0, 'bearing_left', 0, 'bearing_right', 0);
OneSpring4 = B02_BoundaryConditions(OneSpring4, OneSpringDamage4);
assert(OneSpring4.Modal.num_rigid_modes == 1, ...
    'T5 FAIL: one vertical spring must leave one rigid rotation');
fprintf('T5 OK: invalid scour/support-fixity inputs are rejected fail-closed\n');

disp('ALL SMOKE TESTS PASSED');

function local_assert_b02_rejects(Beam, Damage, expected_id)
try
    B02_BoundaryConditions(Beam, Damage);
catch ME
    assert(strcmp(ME.identifier, expected_id), ...
        'T5 FAIL: expected %s, received %s', expected_id, ME.identifier);
    return;
end
error('T5 FAIL: malformed input escaped B02 guard (%s)', expected_id);
end
