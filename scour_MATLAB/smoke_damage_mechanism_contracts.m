function smoke_damage_mechanism_contracts
%SMOKE_DAMAGE_MECHANISM_CONTRACTS Contracts for retained track/wheel mechanisms.
%
% This fast smoke test deliberately avoids a coupled passage solve.  It
% verifies, one mechanism at a time, that the production B54 and B25 entry
% points implement the equations and units documented in
% docs/damage_model_reference.md:
%   * ballast patches alter only the selected ballast spring k/c pair;
%   * hanging sleepers replace only selected ballast k/c by 1e-6 times
%     their nominal values;
%   * pad service multipliers and discrete failures alter only rail-pad
%     spring pairs;
%   * simultaneous descriptors compose according to the documented
%     override rules; and
%   * polygonization adds amp_m*cos(order*x/radius + phase), in metres,
%     before the production finite-difference derivatives are evaluated.

[Beam,Track,Calc] = local_track_fixture();
Healthy = B54_ModelMatrices(Beam,Track,Calc,struct());
n = Track.Sleeper.Tnum;
ones_n = ones(1,n);
kill = 1e-6;

% Healthy descriptors are exact identity multipliers and reproduce nominal
% SI-valued spring vectors.
[V0,D0] = B54_TrackVectors(Track,Healthy,Calc,struct());
local_assert_equal(D0.mult_bal_k,ones_n,'healthy ballast-k multiplier');
local_assert_equal(D0.mult_bal_c,ones_n,'healthy ballast-c multiplier');
local_assert_equal(D0.mult_pad_k,ones_n,'healthy pad-k multiplier');
local_assert_equal(D0.mult_pad_c,ones_n,'healthy pad-c multiplier');
local_assert_equal(V0.pad_k,Track.Pad.Prop.k*ones_n, ...
    'healthy pad stiffness [N/m]');
local_assert_equal(V0.pad_c,Track.Pad.Prop.c*ones_n, ...
    'healthy pad damping [N s/m]');

% One ballast patch at bridge sleeper x=2 m: eta_k and eta_c are paired,
% and the dimensionless multipliers scale the on-bridge ballast properties.
D = struct('track',struct('ballast_patches',[2 2 0.70 1.50]));
bal_k = [1 1 0.70 1 1];
bal_c = [1 1 1.50 1 1];
pad_k = ones_n; pad_c = ones_n;
local_check_track_case('ballast patch',D,Beam,Track,Calc,Healthy, ...
    bal_k,bal_c,pad_k,pad_c);

% A two-sleeper hanging group begins at x=2 m.  This is a near-zero linear
% support model, not a settlement/gap model.
D = struct('track',struct('hanging_groups',[2 2]));
bal_k = [1 1 kill kill 1];
bal_c = bal_k;
local_check_track_case('hanging sleeper group',D,Beam,Track,Calc,Healthy, ...
    bal_k,bal_c,ones_n,ones_n);

% State-global pad service multipliers act on every rail-pad spring and do
% not alter any sleeper-ballast spring.
D = struct('track',struct( ...
    'pad_stiff_mult',1.80,'pad_damp_mult',0.60));
pad_k = 1.80*ones_n;
pad_c = 0.60*ones_n;
local_check_track_case('pad service multipliers',D,Beam,Track,Calc,Healthy, ...
    ones_n,ones_n,pad_k,pad_c);

% A discrete failure matches an exact sleeper-lattice location and overrides
% both pad service properties at that sleeper only; off-lattice values fail.
D = struct('track',struct('pad_failures',2.00));
pad_k = [1 1 kill 1 1];
pad_c = pad_k;
local_check_track_case('single pad failure',D,Beam,Track,Calc,Healthy, ...
    ones_n,ones_n,pad_k,pad_c);

% Cross-mechanism composition: hanging support overrides a ballast patch;
% pad failure overrides the global pad state at its own sleeper.  The two
% physical layers remain independent in the assembled matrices.
T.ballast_patches = [2 3 0.70 1.50];
T.hanging_groups = [3 1];
T.pad_stiff_mult = 1.80;
T.pad_damp_mult = 0.60;
T.pad_failures = 2.00;
D = struct('track',T);
bal_k = [1 1 0.70 kill 1];
bal_c = [1 1 1.50 kill 1];
pad_k = [1.80 1.80 kill 1.80 1.80];
pad_c = [0.60 0.60 kill 0.60 0.60];
local_check_track_case('composed track mechanisms',D,Beam,Track,Calc,Healthy, ...
    bal_k,bal_c,pad_k,pad_c);

local_check_track_descriptor_validation(Track,Healthy,Calc);
local_check_production_track_sampler();
local_check_polygonization();

fprintf(['DAMAGE MECHANISM CONTRACTS: PASS ' ...
    '(ballast, hanging sleepers, pads, polygonization, descriptors)\n']);
end

function local_check_track_case(label,Damage,Beam,Track,Calc,Healthy, ...
        bal_k,bal_c,pad_k,pad_c)
% Check the resolved property vectors and every entry of the assembled delta.

[V,Dbg] = B54_TrackVectors(Track,Healthy,Calc,Damage);
local_assert_equal(Dbg.mult_bal_k,bal_k,[label ' ballast-k multipliers']);
local_assert_equal(Dbg.mult_bal_c,bal_c,[label ' ballast-c multipliers']);
local_assert_equal(Dbg.mult_pad_k,pad_k,[label ' pad-k multipliers']);
local_assert_equal(Dbg.mult_pad_c,pad_c,[label ' pad-c multipliers']);

% These checks pin units as nominal SI spring properties times dimensionless
% multipliers, including the approach/on-bridge/aft property split.
local_assert_equal(V.pad_k,Track.Pad.Prop.k*pad_k, ...
    [label ' pad stiffness [N/m]']);
local_assert_equal(V.pad_c,Track.Pad.Prop.c*pad_c, ...
    [label ' pad damping [N s/m]']);
local_assert_equal(V.balA_k,Track.Ballast.Prop.k*bal_k(1), ...
    [label ' approach ballast stiffness [N/m]']);
local_assert_equal(V.balA_c,Track.Ballast.Prop.c*bal_c(1), ...
    [label ' approach ballast damping [N s/m]']);
local_assert_equal(V.balB_k,Track.BallastOnBeam.Prop.k*bal_k(2:4), ...
    [label ' bridge ballast stiffness [N/m]']);
local_assert_equal(V.balB_c,Track.BallastOnBeam.Prop.c*bal_c(2:4), ...
    [label ' bridge ballast damping [N s/m]']);
local_assert_equal(V.balF_k,Track.Ballast.Prop.k*bal_k(5), ...
    [label ' aft ballast stiffness [N/m]']);
local_assert_equal(V.balF_c,Track.Ballast.Prop.c*bal_c(5), ...
    [label ' aft ballast damping [N s/m]']);

Damaged = B54_ModelMatrices(Beam,Track,Calc,Damage);
[expected_k,expected_c] = local_expected_matrix_delta( ...
    Healthy,Track,bal_k,bal_c,pad_k,pad_c);
local_assert_close(full(Damaged.Mesh.Kg - Healthy.Mesh.Kg), ...
    full(expected_k),[label ' full assembled delta-K']);
local_assert_close(full(Damaged.Mesh.Cg - Healthy.Mesh.Cg), ...
    full(expected_c),[label ' full assembled delta-C']);
local_assert_equal(full(Damaged.Mesh.Mg),full(Healthy.Mesh.Mg), ...
    [label ' mass matrix invariance']);
local_assert_equal(Damaged.Mesh.Kg,Damaged.Mesh.Kg.', ...
    [label ' stiffness symmetry']);
local_assert_equal(Damaged.Mesh.Cg,Damaged.Mesh.Cg.', ...
    [label ' damping symmetry']);
end

function [delta_k,delta_c] = local_expected_matrix_delta( ...
        Model,Track,bal_k,bal_c,pad_k,pad_c)
% Independently assemble the analytic two-node spring changes.

n_dof = Model.Mesh.DOF.Tnum;
delta_k = sparse(n_dof,n_dof);
delta_c = sparse(n_dof,n_dof);
n = Track.Sleeper.Tnum;
for sleeper = 1:n
    rail_dof = Model.Mesh.DOF.rail_vert_at_sleepers(sleeper);
    sleeper_dof = Model.Mesh.DOF.sleepers(sleeper);
    delta_k = local_add_spring(delta_k,rail_dof,sleeper_dof, ...
        Track.Pad.Prop.k*(pad_k(sleeper) - 1));
    delta_c = local_add_spring(delta_c,rail_dof,sleeper_dof, ...
        Track.Pad.Prop.c*(pad_c(sleeper) - 1));

    if sleeper <= Track.Sleeper.num_app
        support_dof = Model.Mesh.DOF.ballast_app(sleeper);
        base_k = Track.Ballast.Prop.k;
        base_c = Track.Ballast.Prop.c;
    elseif sleeper <= Track.Sleeper.num_app + Track.Sleeper.num_onbeam
        bridge_sleeper = sleeper - Track.Sleeper.num_app;
        support_dof = Model.Mesh.DOF.beam_vert_under_sleeper(bridge_sleeper);
        base_k = Track.BallastOnBeam.Prop.k;
        base_c = Track.BallastOnBeam.Prop.c;
    else
        aft_sleeper = sleeper - Track.Sleeper.num_app - ...
            Track.Sleeper.num_onbeam;
        support_dof = Model.Mesh.DOF.ballast_aft(aft_sleeper);
        base_k = Track.Ballast.Prop.k;
        base_c = Track.Ballast.Prop.c;
    end
    delta_k = local_add_spring(delta_k,sleeper_dof,support_dof, ...
        base_k*(bal_k(sleeper) - 1));
    delta_c = local_add_spring(delta_c,sleeper_dof,support_dof, ...
        base_c*(bal_c(sleeper) - 1));
end
end

function matrix = local_add_spring(matrix,dof_1,dof_2,value)
% Add value*[1 -1; -1 1] at a physical two-node spring pair.

matrix(dof_1,dof_1) = matrix(dof_1,dof_1) + value;
matrix(dof_2,dof_2) = matrix(dof_2,dof_2) + value;
matrix(dof_1,dof_2) = matrix(dof_1,dof_2) - value;
matrix(dof_2,dof_1) = matrix(dof_2,dof_1) - value;
end

function local_check_polygonization
% Analytic B25 check with a flat nominal profile and two independent wheels.

Calc.Solver.num_t = 11;
Calc.Solver.dt = 0.01;
Calc.Time.t_0_ind = 1;
Calc.Time.t_end_ind = Calc.Solver.num_t;
Calc.Position.x = 0:0.01:0.10;
Calc.Profile.x = -1:0.01:1;
Calc.Profile.h = zeros(size(Calc.Profile.x));

Veh(1).Tnum = 1;
Veh(1).Wheels.num = 2;
Veh(1).Ax_dist = [0 0];
Veh(1).First_wheel_dist = 0;

Baseline = B25_WheelProfiles(Calc,Veh,struct());
local_assert_equal(Baseline.Veh(1).h_path,zeros(2,11), ...
    'zero-profile B25 baseline');

radius_m = 0.50;
order = 3;
amplitude_m = 80e-6;
phase_rad = pi/7;
Damage.oor_radius = radius_m;
Damage.oor_poly = [1 1 order amplitude_m phase_rad];
Single = B25_WheelProfiles(Calc,Veh,Damage);
x_m = Calc.Position.x;
raw_m = amplitude_m*cos(order*x_m/radius_m + phase_rad);
[expected_h,expected_hd,expected_hdd] = ...
    local_profile_and_derivatives(raw_m,Calc.Solver.dt);
local_assert_close(Single.Veh(1).h_path(1,:),expected_h, ...
    'polygon elevation amp/order/phase law [m]');
local_assert_close(Single.Veh(1).hd_path(1,:),expected_hd, ...
    'polygon first time derivative [m/s]');
local_assert_close(Single.Veh(1).hdd_path(1,:),expected_hdd, ...
    'polygon second time derivative [m/s^2]');
local_assert_equal(Single.Veh(1).h_path(2,:),Baseline.Veh(1).h_path(2,:), ...
    'polygon unrelated-wheel elevation invariance');
local_assert_equal(Single.Veh(1).hd_path(2,:),Baseline.Veh(1).hd_path(2,:), ...
    'polygon unrelated-wheel velocity invariance');
local_assert_equal(Single.Veh(1).hdd_path(2,:),Baseline.Veh(1).hdd_path(2,:), ...
    'polygon unrelated-wheel acceleration invariance');
local_assert_equal(Single.Veh(1).x_path,Baseline.Veh(1).x_path, ...
    'polygon wheel-coordinate invariance');

% Two rows for one wheel superpose linearly; the second row has a distinct
% order, amplitude, and phase so column swaps cannot satisfy the contract.
order_2 = 5;
amplitude_2_m = 20e-6;
phase_2_rad = -pi/4;
Damage.oor_poly = [Damage.oor_poly; ...
    1 1 order_2 amplitude_2_m phase_2_rad];
Composed = B25_WheelProfiles(Calc,Veh,Damage);
raw_composed_m = raw_m + amplitude_2_m*cos( ...
    order_2*x_m/radius_m + phase_2_rad);
[expected_h,expected_hd,expected_hdd] = ...
    local_profile_and_derivatives(raw_composed_m,Calc.Solver.dt);
local_assert_close(Composed.Veh(1).h_path(1,:),expected_h, ...
    'superposed polygon elevation [m]');
local_assert_close(Composed.Veh(1).hd_path(1,:),expected_hd, ...
    'superposed polygon velocity [m/s]');
local_assert_close(Composed.Veh(1).hdd_path(1,:),expected_hdd, ...
    'superposed polygon acceleration [m/s^2]');
local_assert_equal(Composed.Veh(1).h_path(2,:),Baseline.Veh(1).h_path(2,:), ...
    'superposed polygon unrelated-wheel invariance');

% Malformed rows must fail at the production boundary rather than be
% ignored, silently indexed, or allowed to contaminate a profile with NaN.
valid = struct('oor_radius',radius_m, ...
    'oor_poly',[1 1 order amplitude_m phase_rad]);
local_assert_throws(@() B25_WheelProfiles(Calc,Veh, ...
    struct('oor_poly',[1 1 2 1e-5])), ...
    'B25_WheelProfiles:InvalidPolygonization','polygon wrong shape');
bad = valid; bad.oor_poly(5) = NaN;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidPolygonization','polygon nonfinite value');
bad = valid; bad.oor_poly(1) = 0;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidPolygonVehicleIndex','polygon vehicle index');
bad = valid; bad.oor_poly(2) = 3;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidPolygonWheelIndex','polygon wheel index');
bad = valid; bad.oor_poly(3) = 1.5;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidPolygonOrder','polygon noninteger order');
bad = valid; bad.oor_poly(4) = 0;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidPolygonAmplitude','polygon zero amplitude');
bad = valid; bad.oor_radius = 0;
local_assert_throws(@() B25_WheelProfiles(Calc,Veh,bad), ...
    'B25_WheelProfiles:InvalidOorRadius','polygon invalid radius');
end

function local_check_track_descriptor_validation(Track,Model,Calc)
% Invalid track rows must fail before selection, snapping, or truncation.

call = @(track) B54_TrackVectors(Track,Model,Calc,struct('track',track));
local_assert_throws(@() call(42), ...
    'B54_TrackVectors:InvalidDamageTrack','track descriptor type');
local_assert_throws(@() call(struct('x_bridge_local',NaN)), ...
    'B54_TrackVectors:InvalidBridgeLocalCoordinate', ...
    'bridge-local coordinate');
local_assert_throws(@() call(struct('ballast_patches',[1 2 1])), ...
    'B54_TrackVectors:InvalidBallastPatches','ballast wrong shape');
local_assert_throws(@() call(struct( ...
    'ballast_patches',[1 2 1 NaN])), ...
    'B54_TrackVectors:InvalidBallastPatches','ballast nonfinite value');
local_assert_throws(@() call(struct( ...
    'ballast_patches',[2 1 1 1])), ...
    'B54_TrackVectors:ReversedBallastPatch','ballast reversed interval');
local_assert_throws(@() call(struct( ...
    'ballast_patches',[1 2 0 1])), ...
    'B54_TrackVectors:InvalidBallastMultiplier','ballast multiplier');
local_assert_throws(@() call(struct( ...
    'ballast_patches',[-1 1 1 1])), ...
    'B54_TrackVectors:BallastPatchOutsideDomain','ballast coordinate domain');
local_assert_throws(@() call(struct( ...
    'ballast_patches',[0.2 0.3 1 1])), ...
    'B54_TrackVectors:BallastPatchSelectsNoSleeper', ...
    'ballast empty selection');
local_assert_throws(@() call(struct('hanging_groups',[1 2 3])), ...
    'B54_TrackVectors:InvalidHangingGroups','hanging wrong shape');
local_assert_throws(@() call(struct('hanging_groups',[1 1.5])), ...
    'B54_TrackVectors:InvalidHangingCount','hanging noninteger count');
local_assert_throws(@() call(struct('hanging_groups',[-1 1])), ...
    'B54_TrackVectors:HangingGroupOutsideDomain','hanging coordinate domain');
local_assert_throws(@() call(struct('hanging_groups',[4 2])), ...
    'B54_TrackVectors:HangingGroupExceedsDomain','hanging truncation');
% Production samples a continuous group-start coordinate. The documented
% discretization (first sleeper at or to the right) remains valid.
[~,continuous_group] = B54_TrackVectors(Track,Model,Calc,struct( ...
    'track',struct('hanging_groups',[2.1 2])));
local_assert_equal(continuous_group.mult_bal_k,[1 1 1 1e-6 1e-6], ...
    'continuous hanging-group start mapping');
local_assert_throws(@() call(struct('pad_stiff_mult',[1 2])), ...
    'B54_TrackVectors:InvalidPadStiffnessMultiplier', ...
    'pad stiffness scalar shape');
local_assert_throws(@() call(struct('pad_damp_mult',0)), ...
    'B54_TrackVectors:InvalidPadDampingMultiplier', ...
    'pad damping multiplier');
local_assert_throws(@() call(struct('pad_failures',[1 2; 3 4])), ...
    'B54_TrackVectors:InvalidPadFailures','pad-failure shape');
local_assert_throws(@() call(struct('pad_failures',-1)), ...
    'B54_TrackVectors:PadFailureOutsideDomain','pad-failure coordinate domain');
local_assert_throws(@() call(struct('pad_failures',2.1)), ...
    'B54_TrackVectors:PadFailureOffLattice','pad-failure lattice');
local_assert_throws(@() call(struct('pad_failures',[2 2])), ...
    'B54_TrackVectors:DuplicatePadFailure','duplicate pad failure');
end

function local_check_production_track_sampler
% A deterministic property regression for the active local-window sampler.
%
% Before the boundary fix, 68 of these 5000 seeds emitted a hanging group
% whose stored count extended beyond the sampled sleeper window. B54 could
% then truncate the count at a coincident model boundary. Every descriptor
% must now represent exactly the count it stores.

inputs = struct('STAGE','F40-M','n_states_multi',1,'Npass',1, ...
    'n_healthy_states',1,'n_anchor_levels',1,'n_anchor_reps',1, ...
    'n_nuisance_states',1,'qualification_run',true);
config = ttbi.campaign_setup(inputs);
assert(~config.use_track_eov, ...
    'Paper-1 production must keep the deferred track EOV disabled.');
sampler_config = config;
sampler_config.use_track_eov = true;
sample_track.Sleeper.spacing = 0.6;
window_m = config.track_L_app + config.L_bridge + config.track_L_after;
sleeper_x_m = 0:sample_track.Sleeper.spacing:window_m;
tol = 1e-10;

for seed = 1:5000
    [sampled,~,~] = ttbi.sample_track_damage( ...
        struct(),[],sample_track,sampler_config,seed,1);
    descriptor = sampled.track;

    patches = descriptor.ballast_patches;
    assert(size(patches,2) == 4 && all(isfinite(patches(:))) && ...
        all(patches(:,1) >= 0) && all(patches(:,2) <= window_m) && ...
        all(patches(:,1) <= patches(:,2)) && ...
        all(patches(:,3) > 0) && all(patches(:,4) > 0), ...
        'production sampler emitted an invalid ballast descriptor');

    groups = descriptor.hanging_groups;
    assert(size(groups,2) == 2 && all(isfinite(groups(:))) && ...
        all(groups(:,1) >= 0) && all(groups(:,1) <= window_m) && ...
        all(groups(:,2) >= 1) && all(groups(:,2) == fix(groups(:,2))), ...
        'production sampler emitted an invalid hanging-group descriptor');
    for row = 1:size(groups,1)
        first = find(sleeper_x_m >= groups(row,1) - tol,1,'first');
        assert(~isempty(first) && ...
            first + groups(row,2) - 1 <= numel(sleeper_x_m), ...
            ['production hanging-group seed %d row %d cannot realize its ' ...
             'stored count within the sampled sleeper domain'],seed,row);
    end

    failures = descriptor.pad_failures;
    assert(isscalar(descriptor.pad_stiff_mult) && ...
        isfinite(descriptor.pad_stiff_mult) && descriptor.pad_stiff_mult > 0 && ...
        isscalar(descriptor.pad_damp_mult) && ...
        isfinite(descriptor.pad_damp_mult) && descriptor.pad_damp_mult > 0, ...
        'production sampler emitted an invalid pad service multiplier');
    for item = 1:numel(failures)
        assert(min(abs(sleeper_x_m - failures(item))) <= tol, ...
            'production sampler emitted an off-lattice pad failure');
    end
    assert(numel(unique(failures)) == numel(failures), ...
        'production sampler emitted duplicate pad failures');
end
end

function [h,hd,hdd] = local_profile_and_derivatives(raw_h,dt)
% Reproduce B25's documented ordering: differences before zero referencing.

hd = diff(raw_h,1,2)/dt;
hd = [hd(1),hd];
hdd = diff(hd,1,2)/dt;
hdd = [hdd(1),hdd];
h = raw_h - raw_h(1);
end

function [Beam,Track,Calc] = local_track_fixture
% Five sleepers: one approach, three on a 2 m bridge, and one aft.

Track.Sleeper.spacing = 1;
Track.Sleeper.Tnum = 5;
Track.Sleeper.num_app = 1;
Track.Sleeper.num_onbeam = 3;
Track.Sleeper.num_aft = 1;
Track.Sleeper.Prop.m = 50;
Track.Pad.Prop.k = 100;
Track.Pad.Prop.c = 10;
Track.Ballast.Prop.k = 200;
Track.Ballast.Prop.c = 20;
Track.Ballast.Prop.m = 40;
Track.BallastOnBeam.Prop.k = 300;
Track.BallastOnBeam.Prop.c = 30;
Track.BallastOnBeam.Prop.m = 40;
Track.SubBallast.Prop.k = 400;
Track.SubBallast.Prop.c = 40;
Track.PadUnderSleeperOnBeam.included = 0;

Track.Rail.Mesh.Nodes.Tnum = 5;
Track.Rail.Mesh.Nodes.acum = 0:4;
Track.Rail.Mesh.DOF.Tnum = 10;
Track.Rail.Mesh.Ele.num_per_spacing = 1;
Track.Rail.Mesh.Ele.DOF = [];
Track.Rail.Mesh.Ele.a = [];
Track.Rail.Mesh.Mg = sparse(10,10);
Track.Rail.Mesh.Cg = sparse(10,10);
Track.Rail.Mesh.Kg = sparse(10,10);

Beam.Prop.L = 2;
Beam.Mesh.Nodes.Tnum = 3;
Beam.Mesh.Nodes.acum = 0:2;
Beam.Mesh.DOF.Tnum = 6;
Beam.Mesh.Ele.num_per_spacing = 1;
Beam.Mesh.Mg = sparse(6,6);
Beam.Mesh.Cg = sparse(6,6);
Beam.Mesh.Kg = sparse(6,6);

Calc.Profile.L = 4;
Calc.Profile.max_TL = 0;
Calc.Profile.L_Approach = 1;
Calc.Profile.extra_L = 0;
Calc.Options.redux_factor = 0;
Calc.Cte.tol = 1e-10;
end

function local_assert_equal(actual,expected,label)
assert(isequal(actual,expected),'%s differs from its exact contract',label);
end

function local_assert_close(actual,expected,label)
error_norm = norm(actual(:) - expected(:),inf);
scale = max(1,norm(expected(:),inf));
assert(error_norm <= 1e-12*scale, ...
    '%s differs from contract: max error %.3e',label,error_norm);
end

function local_assert_throws(call,expected_identifier,label)
try
    call();
catch caught
    assert(strcmp(caught.identifier,expected_identifier), ...
        '%s raised %s instead of %s',label,caught.identifier,expected_identifier);
    return;
end
error('smoke_damage_mechanism_contracts:MissingExpectedError', ...
    '%s did not raise %s',label,expected_identifier);
end
