% smoke_stage3.m — Stage-3 track-layer damage + wheel-OOR smoke test.
% Run from scour_MATLAB/ in MATLAB. Mirrors the Python _stage3_smoke.py:
%   1. HEALTHY PARITY: one full passage with Damage.track=[] / Damage.oor=[]
%      must produce signals IDENTICAL to the legacy path (norm diff == 0),
%      and show NO wheel-rail tension anywhere on the track.
%   2. DAMAGED: ballast patch + hanging sleepers + pad condition/failure +
%      polygonized wheel must run, stay finite, change the signal, land the
%      defects ON/AROUND THE REAL DECK (audit fix 2026-07-17: descriptors are
%      bridge-local; B54 anchors them to the deck at L_Aw ~ 123 m, not 30 m),
%      and keep permanent wheel-rail contact (bilateral-solver validity).
% NOTE: wheel FLATS are deliberately NOT exercised - they are disabled in the
% campaign (A00 oor_flats_enabled=false) because they violate the permanent-
% contact assumption; see the A00 comment block and framework_rationale.md.
% Takes ~3 solver runs (a few minutes).
clear; clc;

L_bridge = 60; num_spans = 3;

run_one = @(Damage) local_run_passage(Damage, L_bridge, num_spans);

% ---- Passage A: legacy healthy -----------------------------------------
Damage = struct();
Damage.desvio = 0;
Damage.scour_rates = zeros(1, num_spans+1);
Damage.bearing_left = 0; Damage.bearing_right = 0;
Damage.crack_locs = []; Damage.crack_intensity = []; Damage.crack_lc = 0;
[A, SolA] = run_one(Damage);
assert(SolA.contactLost_track == 0, ...
    'healthy passage shows wheel-rail tension - solver/contact regression');
fprintf('A legacy-healthy done (%d samples) | contact OK\n', size(A,2));

% ---- Passage B: healthy with EMPTY new-damage fields (parity check) ----
Damage.track     = [];
Damage.oor_flats = [];
Damage.oor_poly  = [];
B = run_one(Damage);
d = max(abs(A(:) - B(:)));
fprintf('B empty-descriptors done | healthy parity max|A-B| = %.3e (%s)\n', ...
    d, string(d == 0));

% ---- Passage C: track damage + polygonized wheel ------------------------
% BRIDGE-LOCAL frame: deck occupies [30, 30+L_bridge]; B54 maps it onto the
% real deck via Tk.x_bridge_local (global deck start = L_Aw ~ 123 m).
Tk = struct();
Tk.ballast_patches = [40.0, 52.0, 0.8, 2.0];   % wet patch mid-deck (local)
Tk.hanging_groups  = [30.5, 3];                % 3 sleepers at entry abutment
Tk.pad_stiff_mult  = 1.8;
Tk.pad_damp_mult   = 1.1;
Tk.pad_failures    = 45.0;
Tk.x_bridge_local  = 30.0;                     % deck start in this frame
Damage.track = Tk;
R_ = 0.46;
Damage.oor_flats = zeros(0,5);                 % flats DISABLED (see header)
Damage.oor_poly  = [1, 2, 3, 5e-5, 0.7];       % order-3 poly, 50 um
Damage.oor_radius = R_;
[C, SolC, Model] = run_one(Damage);
assert(all(isfinite(C(:))), 'damaged signal has NaN/Inf');
rel = max(abs(C - A), [], 2) ./ (max(abs(A), [], 2) + 1e-12);
fprintf('C damaged done | per-DOF max|C-A|/max|A|:\n');
disp(rel');
assert(max(rel) > 1e-3, 'damage did not change the signal');

% ---- Placement assertions (audit 2026-07-17) ----------------------------
Dbg = Model.TrackDmgDbg;
deck0 = Dbg.x_deck_global(1); deck1 = Dbg.x_deck_global(2);
fprintf('deck (global sleeper axis) = [%.1f, %.1f] m | frame offset = %.1f m\n', ...
    deck0, deck1, Dbg.frame_offset);
assert(deck0 > 100, 'deck start <= 100 m - expected ~123 m under redux=0');
% patch [40,52] local -> global [deck0+10, deck0+22]: strictly ON the deck
assert(~isempty(Dbg.bal_x_global) && ...
    all(Dbg.bal_x_global >= deck0 & Dbg.bal_x_global <= deck1), ...
    'ballast patch did not land on the deck');
% hanging group at 30.5 local -> global ~deck0+0.5 (entry abutment)
assert(~isempty(Dbg.hang_x_global) && ...
    all(abs(Dbg.hang_x_global - (deck0 + 0.5)) <= 3*0.6 + 1e-9), ...
    'hanging group did not land at the entry abutment');
% pad failure at 45 local -> global ~deck0+15 (on deck)
assert(~isempty(Dbg.padfail_x_global) && ...
    all(abs(Dbg.padfail_x_global - (deck0 + 15)) <= 0.6 + 1e-9), ...
    'pad failure did not land mid-deck');
fprintf('placement OK: patch %s | hang %s | padfail %s (global m)\n', ...
    mat2str(round([min(Dbg.bal_x_global), max(Dbg.bal_x_global)],1)), ...
    mat2str(round(Dbg.hang_x_global,1)), mat2str(round(Dbg.padfail_x_global,1)));

% ---- Contact assertion (audit 2026-07-17) -------------------------------
fprintf('C contact: lost_track=%d | tension_frac_max=%.2e | F_tension_max=%.3e N\n', ...
    SolC.contactLost_track, SolC.tension_frac_max, SolC.F_tension_max);
assert(SolC.contactLost_track == 0, ...
    ['damaged passage shows wheel-rail TENSION - bilateral solver invalid ', ...
     'for this severity; reduce the defect or treat as rejected']);

if d == 0
    fprintf('SMOKE STAGE-3: ALL PASS\n');
else
    fprintf('SMOKE STAGE-3: damaged path OK but HEALTHY PARITY BROKEN — investigate\n');
end

% -------------------------------------------------------------------------
function [sig, Sol, Model] = local_run_passage(Damage, L_bridge, num_spans)
    Train = A01_Train(80/3.6, zeros(5,3));
    Track = A02_Track();
    Beam_seed = struct();
    Beam_seed.Prop.L = L_bridge;
    Beam_seed.Prop.num_spans = num_spans;
    Beam = A03_Bridge(Beam_seed);
    Profile_cfg = struct('mode', 'fixed');     % deterministic baseline profile
    [Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);
    [Sol, Calc, Train, ~, ~, Model] = B00_Calculations(Calc, Train, Track, Beam, Damage);
    data = struct();
    data = D01_DataProcessing(1, 1, Sol, Train, Calc, Damage, data);
    sig = [data.AceleracaoPrimVag{1,1}; data.AcelRodaPrimVag{1,1}; ...
           data.PitchPrimVag{1,1}];
end
