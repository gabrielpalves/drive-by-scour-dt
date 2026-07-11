% smoke_stage3.m — Stage-3 track-layer damage + wheel-OOR smoke test.
% Run from scour_MATLAB/ in MATLAB. Mirrors the Python _stage3_smoke.py:
%   1. HEALTHY PARITY: one full passage with Damage.track=[] / Damage.oor=[]
%      must produce signals IDENTICAL to the legacy path (norm diff == 0).
%   2. DAMAGED: ballast patch + hanging sleepers + pad aging/failure + two
%      wheel flats must run, stay finite, and change the signal.
% Takes ~2 solver runs (a few minutes).
clear; clc;

L_bridge = 60; num_spans = 3;

run_one = @(Damage) local_run_passage(Damage, L_bridge, num_spans);

% ---- Passage A: legacy healthy -----------------------------------------
Damage = struct();
Damage.desvio = 0;
Damage.scour_rates = zeros(1, num_spans+1);
Damage.bearing_left = 0; Damage.bearing_right = 0;
Damage.crack_locs = []; Damage.crack_intensity = []; Damage.crack_lc = 0;
A = run_one(Damage);
fprintf('A legacy-healthy done (%d samples)\n', size(A,2));

% ---- Passage B: healthy with EMPTY new-damage fields (parity check) ----
Damage.track     = [];
Damage.oor_flats = [];
Damage.oor_poly  = [];
B = run_one(Damage);
d = max(abs(A(:) - B(:)));
fprintf('B empty-descriptors done | healthy parity max|A-B| = %.3e (%s)\n', ...
    d, string(d == 0));

% ---- Passage C: track damage + wheel flats ------------------------------
% Bridge occupies x = [30, 30+L_bridge] in track coordinates.
Tk = struct();
Tk.ballast_patches = [40.0, 52.0, 0.8, 2.0];   % wet patch mid-bridge
Tk.hanging_groups  = [30.5, 3];                % 3 sleepers at entry abutment
Tk.pad_stiff_mult  = 1.8;
Tk.pad_damp_mult   = 1.1;
Tk.pad_failures    = 45.0;
Damage.track = Tk;
R_ = 0.46;
Damage.oor_flats = [1, 1, 0.020, 0.020^2/(8*R_),  0.0;   % fresh 20 mm flat
                    2, 3, 0.050, 0.050^2/(16*R_), 1.5];  % run-in 50 mm flat
Damage.oor_poly  = [1, 2, 3, 5e-5, 0.7];                 % order-3 poly, 50 um
Damage.oor_radius = R_;
C = run_one(Damage);
assert(all(isfinite(C(:))), 'damaged signal has NaN/Inf');
rel = max(abs(C - A), [], 2) ./ (max(abs(A), [], 2) + 1e-12);
fprintf('C damaged done | per-DOF max|C-A|/max|A|:\n');
disp(rel');
assert(max(rel) > 1e-3, 'damage did not change the signal');
if d == 0
    fprintf('SMOKE STAGE-3: ALL PASS\n');
else
    fprintf('SMOKE STAGE-3: damaged path OK but HEALTHY PARITY BROKEN — investigate\n');
end

% -------------------------------------------------------------------------
function sig = local_run_passage(Damage, L_bridge, num_spans)
    Train = A01_Train(80/3.6, zeros(5,3));
    Track = A02_Track();
    Beam_seed = struct();
    Beam_seed.Prop.L = L_bridge;
    Beam_seed.Prop.num_spans = num_spans;
    Beam = A03_Bridge(Beam_seed);
    Profile_cfg = struct('mode', 'fixed');     % deterministic baseline profile
    [Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);
    [Sol, Calc, Train, ~, ~] = B00_Calculations(Calc, Train, Track, Beam, Damage);
    data = struct();
    data = D01_DataProcessing(1, 1, Sol, Train, Calc, Damage, data);
    sig = [data.AceleracaoPrimVag{1,1}; data.AcelRodaPrimVag{1,1}; ...
           data.PitchPrimVag{1,1}];
end
