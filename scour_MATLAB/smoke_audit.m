% smoke_audit.m — fast post-audit invariant checks (2026-07-17), no solver.
% Run from scour_MATLAB/ in MATLAB (~seconds). Asserts the three cheap
% invariants the audit fixed; the geometry/contact invariants need a full
% solve and live in smoke_stage3.m.
%   1. LHS orientation: speed/temp draw is a true 2-variable LHS
%      (|corr| small, all four quadrants populated). The transposed
%      lhsdesign(2,Npass) fails BOTH (corr = -0.75, two empty quadrants).
%   2. FRA PSD units: A04's psd_fra inputs evaluated on the cycles/m axis
%      must match the reference rad/m formula converted consistently
%      (S_cyc(n) = 2*pi*S_rad(2*pi*n), all in m^2/(cycle/m)). The old
%      unconverted corner gave ~16x at the 1.524 m cutoff.
%   3. Fixed and state-varying profiles share exactly one FRA-v2 class-4
%      spectral contract; only their phase-realization rule differs.
%   4. The former 0.5 mm per-passage white profile perturbation is OFF in A00
%      (it was an EN 13848-2 metrology misreading; observation noise belongs
%      to the load-time model).
%   5. Pad failures are sampled once per unique 0.6-m sleeper-lattice
%      position and resolve one-to-one in B54_TrackVectors.
%   6. The deployed physical8_v1 wheelset channel is the manufactured
%      four-term moving-coordinate acceleration with the solver-active mask.
clear; clc;
fails = 0;

% ---- 1. LHS orientation --------------------------------------------------
Npass = 50; nrep = 200;
corrs = zeros(1, nrep); quad_ok = true;
for r = 1:nrep
    lhs_ = lhsdesign(Npass, 2)';           % as in A00 (audit fix)
    corrs(r) = corr(lhs_(1,:)', lhs_(2,:)');
    occ_ = [any(lhs_(1,:) <  .5 & lhs_(2,:) <  .5), ...
            any(lhs_(1,:) <  .5 & lhs_(2,:) >= .5), ...
            any(lhs_(1,:) >= .5 & lhs_(2,:) <  .5), ...
            any(lhs_(1,:) >= .5 & lhs_(2,:) >= .5)];
    quad_ok = quad_ok && all(occ_);
end
fprintf('[1] LHS: mean corr = %+.3f (sd %.3f), quadrants %s\n', ...
    mean(corrs), std(corrs), string(quad_ok));
if abs(mean(corrs)) > 0.15 || ~quad_ok
    fprintf('    FAIL: degenerate LHS (transposed call gives mean corr ~ -0.75)\n');
    fails = fails + 1;
end

% ---- 2. FRA PSD unit consistency ----------------------------------------
% Implemented form (A04 psd_fra, class 4) on the cycles/m axis:
k_   = 0.25;
A_v  = 0.5376;                 % class 4 [cm^2 rad/m]
n_c  = 0.8245/(2*pi);          % corner in cycles/m (the audit fix)
conv = 1e-4/(2*pi);            % cm^2 -> m^2 and rad -> cycles
S_impl = @(n) (k_*A_v*n_c^2)./((n.^2).*(n.^2 + n_c^2))*conv;
% Reference: rad/m formula S_rad(W) = k*A_v*Wc^2/(W^2*(W^2+Wc^2)) [cm^2/(rad/m)]
% converted to m^2/(cycle/m): S_ref(n) = 2*pi*1e-4*S_rad(2*pi*n).
Wc = 0.8245;
S_ref = @(n) 2*pi*1e-4 * (k_*A_v*Wc^2)./(((2*pi*n).^2).*((2*pi*n).^2 + Wc^2));
n_test = 1./[304.8, 30, 7.43, 3, 1.524];      % cycles/m across the FRA band
ratio  = S_impl(n_test)./S_ref(n_test);
fprintf('[2] PSD implemented/reference across band: %s\n', mat2str(round(ratio,4)));
if any(abs(ratio - 1) > 1e-9)
    fprintf('    FAIL: unit mismatch (old rad/m corner gave ratio ~16 at 1.524 m)\n');
    fails = fails + 1;
end
% Cross-check against the values A00/A04 actually configure:
Trk_ = A02_Track(); Beam_seed = struct(); Beam_seed.Prop.L = 60;
Beam_seed.Prop.num_spans = 3; Bm_ = A03_Bridge(Beam_seed);
cfg_ = struct('mode', 'psd_fra', 'fra_class', 4);
[Calc_, ~, ~] = A04_Options(Bm_, Trk_, cfg_);
S_live = Calc_.Profile.PSD_Y_fun(n_test, Calc_.Profile.inputs);
ratio_live = S_live./S_ref(n_test);
fprintf('    live A04 inputs vs reference: %s\n', mat2str(round(ratio_live,4)));
if any(abs(ratio_live - 1) > 1e-9)
    fprintf('    FAIL: A04 psd_fra inputs are unit-inconsistent\n');
    fails = fails + 1;
end

% ---- 3. One spectral contract; phase is the only registered contrast -----
contract_ = 'fra-v2-class4-cycles-per-m-v1';
fixed_cfg_ = struct('mode', 'fixed', 'fra_class', 4, ...
    'phase_seed', 20260728, 'spectrum_contract', contract_);
same_cfg_ = struct('mode', 'psd_fra', 'fra_class', 4, ...
    'phase_seed', 20260728, 'spectrum_contract', contract_);
other_cfg_ = same_cfg_;
other_cfg_.phase_seed = 314159265;
[Fixed_, ~, ~] = A04_Options(Bm_, Trk_, fixed_cfg_);
[Same_, ~, ~] = A04_Options(Bm_, Trk_, same_cfg_);
[Other_, ~, ~] = A04_Options(Bm_, Trk_, other_cfg_);
spectral_equal_ = Fixed_.Profile.Type == 1 && Same_.Profile.Type == 1 && ...
    isequal(Fixed_.Profile.inputs, Same_.Profile.inputs) && ...
    Fixed_.Profile.min_WaveLength == Same_.Profile.min_WaveLength && ...
    Fixed_.Profile.max_WaveLength == Same_.Profile.max_WaveLength && ...
    strcmp(Fixed_.Profile.spectrum_contract, contract_) && ...
    strcmp(Same_.Profile.spectrum_contract, contract_) && ...
    Fixed_.Profile.fra_class == 4 && Same_.Profile.fra_class == 4 && ...
    isequal(Fixed_.Profile.PSD_Y_fun(n_test, Fixed_.Profile.inputs), ...
            Same_.Profile.PSD_Y_fun(n_test, Same_.Profile.inputs));

% Exercise the production generator, including seed isolation.
Fixed_.Profile.min_dx = 0.02;
Fixed_.Profile.L = 20;
Fixed_.Position.x = 0:0.02:20;
Fixed_.Plot.Profile_original = 0;
Same_.Profile.min_dx = 0.02;
Same_.Profile.L = 20;
Same_.Position.x = 0:0.02:20;
Same_.Plot.Profile_original = 0;
Other_.Profile.min_dx = 0.02;
Other_.Profile.L = 20;
Other_.Position.x = 0:0.02:20;
Other_.Plot.Profile_original = 0;
rng(991, 'twister');
expected_after_ = rand(1, 4);
rng(991, 'twister');
FixedA_ = B19_GenerateProfile(Fixed_);
observed_after_ = rand(1, 4);
rng(123, 'twister');
FixedB_ = B19_GenerateProfile(Fixed_);
Same_ = B19_GenerateProfile(Same_);
Other_ = B19_GenerateProfile(Other_);
phase_rule_ok_ = isequal(FixedA_.Profile.h, FixedB_.Profile.h) && ...
    isequal(FixedA_.Profile.h, Same_.Profile.h) && ...
    ~isequal(FixedA_.Profile.h, Other_.Profile.h) && ...
    isequal(FixedA_.Profile.PSD_Y, Other_.Profile.PSD_Y) && ...
    isequal(expected_after_, observed_after_);

% Exercise the optional state-random phase namespace directly. Two passages of
% one StateUID must regenerate the exact same profile, while a different
% StateUID receives a different phase realization under the same FRA-4 PSD.
% Paper-1 production uses fixed mode; this keeps psd_fra executable for a
% separately authorized future sensitivity study.
state_uids_ = { ...
    ttbi.state_uid(40, 2, 2, 'scour_only', 2, 20, 1); ...
    ttbi.state_uid(40, 2, 2, 'scour_only', 2, 20, 2)};
state_roots_ = ttbi.state_seed_ids(state_uids_, 1);
state_names_ = {'operations','crack','profile-state','track','profile-phase'};
passage_names_ = {'profile-passage','oor-passage'};
[state_streams_, ~] = ttbi.named_stream_seed_ids( ...
    state_roots_, state_uids_, 2, 'uid-named-substreams-v2', ...
    state_names_, passage_names_);
state1_cfg_ = struct('mode', 'psd_fra', 'fra_class', 4, ...
    'phase_seed', double(state_streams_(1, 5)), ...
    'spectrum_contract', contract_);
state2_cfg_ = state1_cfg_;
state2_cfg_.phase_seed = double(state_streams_(2, 5));
[State1_, ~, ~] = A04_Options(Bm_, Trk_, state1_cfg_);
[State2_, ~, ~] = A04_Options(Bm_, Trk_, state2_cfg_);
State1_.Profile.min_dx = 0.02;
State1_.Profile.L = 20;
State1_.Position.x = 0:0.02:20;
State1_.Plot.Profile_original = 0;
State2_.Profile.min_dx = 0.02;
State2_.Profile.L = 20;
State2_.Position.x = 0:0.02:20;
State2_.Plot.Profile_original = 0;
rng(17, 'twister');
State1Pass1_ = B19_GenerateProfile(State1_);
rng(29, 'twister');
State1Pass2_ = B19_GenerateProfile(State1_);
State2Pass1_ = B19_GenerateProfile(State2_);
state_persistence_ok_ = ...
    state_streams_(1, 5) ~= state_streams_(2, 5) && ...
    isequal(State1Pass1_.Profile.h, State1Pass2_.Profile.h) && ...
    ~isequal(State1Pass1_.Profile.h, State2Pass1_.Profile.h) && ...
    isequal(State1Pass1_.Profile.PSD_Y, State2Pass1_.Profile.PSD_Y);
fprintf(['[3] shared FRA-v2 spectrum %s; phase-only contrast %s; ' ...
    'per-StateUID persistence %s\n'], string(spectral_equal_), ...
    string(phase_rule_ok_), string(state_persistence_ok_));
if ~spectral_equal_ || ~phase_rule_ok_ || ~state_persistence_ok_
    fprintf(['    FAIL: fixed and psd_fra must share one spectrum; fixed must ' ...
        'repeat, while psd_fra must persist within and differ across StateUIDs\n']);
    fails = fails + 1;
end

% ---- 4. Jitter disabled ---------------------------------------------------
driver_source_ = fileread('A00_Run.m');
config_source_ = fileread(fullfile('+ttbi', 'campaign_setup.m'));
tokens_ = regexp(config_source_, ...
    'profile_jitter_sd_mm\s*=\s*([0-9.]+)', 'tokens');
if numel(tokens_) ~= 1
    error('smoke_audit:CampaignSetupDrift', ...
        'Expected one live profile_jitter_sd_mm assignment, found %d.', ...
        numel(tokens_));
end
jit_ = str2double(tokens_{1}{1});
fprintf('[4] campaign_setup profile_jitter_sd_mm = %g\n', jit_);
if jit_ ~= 0
    fprintf('    FAIL: per-passage white profile perturbation must stay 0 (audit 2026-07-17)\n');
    fails = fails + 1;
end

% ---- 5. One Bernoulli draw per sleeper-lattice pad position --------------
track_damage_source_ = fileread( ...
    fullfile('+ttbi', 'sample_track_damage.m'));
pad_tokens_ = { ...
    'pad_failures = sample_pad_failures( ...', ...
    'track_window, Track.Sleeper.spacing, config.pad_p_fail);'};
pad_source_ok_ = all(cellfun( ...
    @(s) contains(track_damage_source_, s), pad_tokens_));
pad_window_ = 12;
pad_spacing_ = 0.6;
pad_probability_ = 0.2;
rng(77123, 'twister');
[sampled_, lattice_, failed_] = sample_pad_failures( ...
    pad_window_, pad_spacing_, pad_probability_);
rng(77123, 'twister');
expected_lattice_ = 0:pad_spacing_:pad_window_;
expected_failed_ = rand(size(expected_lattice_)) < pad_probability_;
sampler_ok_ = isequal(lattice_, expected_lattice_) && ...
    isequal(failed_, expected_failed_) && ...
    isequal(sampled_, expected_lattice_(expected_failed_)) && ...
    numel(sampled_) == numel(unique(sampled_)) && ...
    all(ismember(sampled_, expected_lattice_));
TrackP_.Sleeper.Tnum = 9;
TrackP_.Sleeper.num_app = 2;
TrackP_.Sleeper.num_onbeam = 5;
TrackP_.Sleeper.num_aft = 2;
TrackP_.Sleeper.spacing = 0.6;
TrackP_.Pad.Prop.k = 11; TrackP_.Pad.Prop.c = 12;
TrackP_.Ballast.Prop.k = 21; TrackP_.Ballast.Prop.c = 22;
TrackP_.BallastOnBeam.Prop.k = 31; TrackP_.BallastOnBeam.Prop.c = 32;
ModelP_.Mesh.XLoc.sleepers = 0:0.6:4.8;
CalcP_.Cte.tol = 1e-12;
% Non-zero frame offset: global deck starts at 1.2 m but its descriptor-frame
% coordinate is 2.4 m, so descriptor x = global x + 1.2 m.
TP_.x_bridge_local = 2.4;
TP_.pad_stiff_mult = 2.0;
TP_.pad_damp_mult = 0.9;
TP_.pad_failures = [1.2, 2.4, 6.0];
DamageP_.track = TP_;
[~, DbgP_] = B54_TrackVectors(TrackP_, ModelP_, CalcP_, DamageP_);
kill_ = 1e-6;
expected_global_failures_ = [0, 1.2, 4.8];
failure_mask_ = ismember( ...
    ModelP_.Mesh.XLoc.sleepers, expected_global_failures_);
pad_mapping_ok_ = DbgP_.frame_offset == -1.2 && ...
    isequal(DbgP_.padfail_x_global, expected_global_failures_) && ...
    all(DbgP_.mult_pad_k(failure_mask_) == kill_) && ...
    all(DbgP_.mult_pad_c(failure_mask_) == kill_) && ...
    all(DbgP_.mult_pad_k(~failure_mask_) == 2.0) && ...
    all(DbgP_.mult_pad_c(~failure_mask_) == 0.9);
fprintf(['[5] pad lattice source %s; sampler %s; ' ...
    'offset B54 mapping %s\n'], string(pad_source_ok_), ...
    string(sampler_ok_), string(pad_mapping_ok_));
if ~pad_source_ok_ || ~sampler_ok_ || ~pad_mapping_ok_
    fprintf('    FAIL: pad failures are not unique Bernoulli sleeper-lattice draws\n');
    fails = fails + 1;
end

% ---- 6. Manufactured physical8_v1 four-term wheelset response ------------
% Deliberately make every term non-zero, and retain non-zero hdd_path values at
% inactive samples.  The fixture therefore rejects omission of any chain-rule
% term as well as the former unmasked profile-inertia behavior without running
% the coupled solver.
SolW_.acc_under   = reshape(1:20, 4, 5) / 10;
SolW_.vel_under_p = reshape(21:40, 4, 5) / 100;
SolW_.def_under_pp = reshape(41:60, 4, 5) / 1000;
CalcW_.hdd_path = reshape(61:80, 4, 5) / 10;
CalcW_.elexj = [1 1 0 2 0; 1 0 3 0 1; 0 2 1 1 0; 4 0 0 2 1];
velW_ = 3.25;
activeW_expected_ = CalcW_.elexj > 0;
term1_ = SolW_.acc_under;
term2_ = 2 * velW_ * SolW_.vel_under_p;
term3_ = velW_^2 * SolW_.def_under_pp;
term4_ = CalcW_.hdd_path .* activeW_expected_;
expectedW_ = term1_ + term2_ + term3_ + term4_;
[actualW_, activeW_] = ttbi.wheel_contact_kinematics( ...
    SolW_, CalcW_, velW_);
tolW_ = 32 * eps(max(1, max(abs(expectedW_), [], 'all')));
terms_exercised_ = all([any(term1_(:) ~= 0), any(term2_(:) ~= 0), ...
    any(term3_(:) ~= 0), any(term4_(:) ~= 0)]) && ...
    any(CalcW_.hdd_path(~activeW_expected_) ~= 0);
wheelset_ok_ = terms_exercised_ && isequal(activeW_, activeW_expected_) && ...
    max(abs(actualW_ - expectedW_), [], 'all') <= tolW_;
fprintf('[6] physical8_v1 manufactured four-term + active mask %s\n', ...
    string(wheelset_ok_));
if ~wheelset_ok_
    fprintf(['    FAIL: wheelset response must equal u_tt + 2*v*u_xt + ' ...
        'v^2*u_xx + h_tt*(elexj>0)\n']);
    fails = fails + 1;
end

% ---- Verdict --------------------------------------------------------------
if fails == 0
    fprintf('SMOKE AUDIT: ALL PASS\n');
else
    error('SMOKE AUDIT: %d CHECK(S) FAILED', fails);
end
