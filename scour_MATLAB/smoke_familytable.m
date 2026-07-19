function smoke_familytable()
% smoke_familytable — Feature A (2026-07-19) MATLAB<->Python contract smoke.
%
% Validates, with REAL MATLAB save/load (not scipy approximations), every type
% assumption the new family machinery relies on:
%   1. damage_states.mat family arrays (cellstr / double / logical) survive a
%      save/load round trip with the exact types the A00 code writes;
%   2. the resume-scan equality checks (isequal on scour_vector /
%      bearing_fixity rows, strcmp on char(state_family)) hold after a real
%      save/load of a data struct — i.e. a LEGITIMATE resume can never
%      false-positive abort on type/orientation drift;
%   3. jsonencode of an fp_cfg carrying cell-of-char + logical arrays is
%      deterministic and well-formed (the fingerprint's stability).
% It then leaves Results/_smoke_familytable/damage_states.mat on disk for the
% PYTHON side of the contract: run
%     python check_familytable_roundtrip.py
% afterwards — it reads this file with core.dataset.read_state_table and
% asserts the REAL MATLAB cell encoding parses (the Python check fixtures use
% scipy-written cells, which encode differently).
%
% Prints PASS lines and hard-errors on any failure (same style as smoke_audit).

out_dir = fullfile('Results', '_smoke_familytable');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- 1. Family table with EXACTLY A00's construction types ---------------
n_supp = 4;
StateFamily  = [repmat({'target_healthy'}, 2, 1);
                repmat({'scour_only'},     2, 1);
                repmat({'bearing_only'},   1, 1);
                repmat({'nuisance_only'},  1, 1);
                repmat({'joint'},          2, 1)];
AnchorTarget = [0; 0; 2; 3; 1; 0; 0; 0];
AnchorLevel  = [0; 0; 1; 2; 1; 0; 0; 0];
CrackOn      = logical([0; 0; 0; 0; 0; 1; 1; 0]);
n_states     = numel(StateFamily);
DamageStates  = zeros(n_states, n_supp);
DamageStates(3, 2) = 0.15;  DamageStates(4, 3) = 0.30;   % scour_only rows
DamageStates(7, 2:3) = [0.10 0.40];                      % a joint row
BearingFixity = zeros(n_states, 2);
BearingFixity(5, 1) = 0.2375;                            % bearing_only row
BearingStates = BearingFixity * 1e9;                     % k_r stand-in
k_ref_bear = 2.31e9; scour_supports = [2 3];             %#ok<NASGU>
save(fullfile(out_dir, 'damage_states.mat'), 'DamageStates', 'BearingStates', ...
    'BearingFixity', 'k_ref_bear', 'scour_supports', ...
    'StateFamily', 'AnchorTarget', 'AnchorLevel', 'CrackOn');

% Round trip in MATLAB itself: the resume scan loads these same arrays.
S = load(fullfile(out_dir, 'damage_states.mat'));
assert(iscellstr(S.StateFamily) && isequal(S.StateFamily, StateFamily), ...
    'StateFamily did not survive save/load as an identical cellstr');   %#ok<ISCLSTR>
assert(islogical(S.CrackOn) && isequal(S.CrackOn, CrackOn), ...
    'CrackOn did not survive save/load as an identical logical vector');
assert(isequal(S.DamageStates, DamageStates) && isequal(S.BearingFixity, BearingFixity), ...
    'damage matrices changed in the round trip');
fprintf('[PASS] damage_states.mat family arrays round-trip exactly\n');

% ---- 2. Resume-check equalities on a REAL saved data struct --------------
% Mirrors save_progress: `data` struct + top-level provenance vars, then the
% exact comparisons the A00 resume scan performs.
dc_idx = 3;                                   % pretend this is state 0003.mat
data = struct();
data.scour_vector   = DamageStates(dc_idx, :);   % 1 x n_supp row, as in A00
data.bearing_fixity = BearingFixity(dc_idx, :);  % 1 x 2 row
data.bearing_vector = BearingStates(dc_idx, :);
data.scour_supports = scour_supports;
data.state_family   = StateFamily{dc_idx};       % char row vector
fstate = fullfile(out_dir, 'state_roundtrip.mat');
save(fstate, 'data');
D = load(fstate); d_ = D.data;
assert(isequal(d_.scour_vector,   DamageStates(dc_idx, :)),  'scour row isequal failed');
assert(isequal(d_.bearing_fixity, BearingFixity(dc_idx, :)), 'fixity row isequal failed');
assert(isequal(d_.bearing_vector, BearingStates(dc_idx, :)), 'k_r row isequal failed');
assert(isequal(d_.scour_supports, scour_supports),           'scour_supports isequal failed');
assert(strcmp(char(d_.state_family), StateFamily{dc_idx}),   'state_family strcmp failed');
fprintf('[PASS] resume-scan equality checks hold after a real save/load\n');

% ---- 3. jsonencode determinism with cell + logical fingerprint fields ----
fp_cfg = struct('schema', 'smoke', 'n_states', n_states);
fp_cfg.StateFamily  = StateFamily;
fp_cfg.AnchorTarget = AnchorTarget;
fp_cfg.AnchorLevel  = AnchorLevel;
fp_cfg.CrackOn      = CrackOn;
j1 = jsonencode(fp_cfg); j2 = jsonencode(fp_cfg);
assert(ischar(j1) || isstring(j1), 'jsonencode did not return text');
assert(strcmp(j1, j2), 'jsonencode is not deterministic for the same struct');
assert(contains(j1, '"target_healthy"') && contains(j1, 'true'), ...
    'jsonencode dropped the cellstr or logical content');
fprintf('[PASS] jsonencode(fp_cfg with cellstr+logical) deterministic + well-formed\n');

fprintf('\nsmoke_familytable: ALL PASS\n');
fprintf('Now run:  python check_familytable_roundtrip.py  (Python side of the contract)\n');
end
