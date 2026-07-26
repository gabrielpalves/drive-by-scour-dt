% smoke_contact_closure.m
% Fast, self-contained checks for the contact-closure harness:
%   1. descriptor loading, "worst" passage choice and diagnostic gate logic;
%   2. B66's whole-track mask excludes both pre-entry and post-exit samples.
% No production dataset/result/bundle is read or modified.
clear; clc;

fixture_dir = tempname;
mkdir(fixture_dir);
fixture_cleanup = onCleanup(@() rmdir(fixture_dir, 's'));
report_dir = tempname;
mkdir(report_dir);
report_cleanup = onCleanup(@() rmdir(report_dir, 's'));

case_info = struct( ...
    'case_name', 'contact_closure_fixture', ...
    'stage', 'smoke', ...
    'gen_schema', 'smoke', ...
    'gen_fingerprint', repmat('0', 1, 64), ...
    'L_bridge_m', 60, ...
    'num_spans', 3, ...
    'n_states', 24, ...
    'damage_seed', 1, ...
    'profile_mode', 'fixed', ...
    'profile_jitter_sd_mm', 0, ...
    'oor_radius', 0.46);
save(fullfile(fixture_dir, 'case_info.mat'), 'case_info');

n_passages = 2;
data = struct();
data.Velocidade = [80, 81] / 3.6;
data.Temperatura = [20, 21];
data.VehiclesProps = zeros(5, 3, n_passages);
data.scour_vector = [0, 0.2, 0, 0];
data.bearing_vector = [0, 0];
data.crack_log = zeros(n_passages, 3);
data.profile_mode = 'fixed';
data.profile_log = ones(n_passages, 1);
data.track_log = cell(n_passages, 1);
data.oor_log = cell(n_passages, 1);
data.contact_log = [0, 1, 0.001, 13000; 0, 0, 0, -100000];
save(fullfile(fixture_dir, '0024.mat'), 'data');

report = contact_closure_study(fixture_dir, 24, "worst", ...
    'VerifyIntegrity', false, 'DryRun', true, 'OutputDir', report_dir);
assert(strcmp(report.status, 'DRY_RUN_VALIDATED'));
assert(report.passage_index == 1);
assert(isequal(report.saved_gate_pass, [false, false, true]));
assert(report.descriptor.velocity_kmh == 80);
assert(isequal(report.descriptor.scour_vector, [0, 0.2, 0, 0]));
assert(isscalar(dir(fullfile(report_dir, '*.md'))));
assert(isscalar(dir(fullfile(report_dir, '*.mat'))));
fprintf('descriptor/gates: PASS\n');

% Direct B66 regression: only positions x=0 and x=1 are on track.
% One of those two samples is tensile -> fraction 1/2. The obsolete
% lower-bound-only mask would include two post-exit zeros and return 1/4.
Calc = struct();
Calc.Options.VBI = 0;
Calc.Solver.num_t = 5;
Calc.Profile.L = 1;
Calc.Profile.L_Aw = 0;
Calc.Profile.L_bridge = 1;
Calc.Cte.grav = -9.81;
Calc.Veh(1).x_path = [-1, 0, 1, 2, 3];

Train = struct();
Train.Veh(1).Tnum = 1;
Train.Veh(1).Wheels.N2w = 1;
Train.Veh(1).Wheels.m = 0;
Train.Veh(1).Susp.Prim.k = 1;
Train.Veh(1).Susp.Prim.c = 0;

Sol = struct();
Sol.Veh(1).U = [0, 1, -1, 0, 0];
Sol.Veh(1).V = zeros(1, 5);
[Sol] = B66_ContactForce(Sol, struct(), Calc, Train, struct());
assert(Sol.F_tension_max == 1);
assert(Sol.contactLost_track == 1);
assert(Sol.tension_frac_max == 0.5);
fprintf('B66 bounded on-track mask: PASS\n');
fprintf('SMOKE CONTACT CLOSURE: ALL PASS\n');
