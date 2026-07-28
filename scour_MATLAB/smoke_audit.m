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
%   3. The former 0.5 mm per-passage white profile perturbation is OFF in A00
%      (it was an EN 13848-2 metrology misreading; observation noise belongs
%      to the load-time model).
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

% ---- 3. Jitter disabled ---------------------------------------------------
txt_ = fileread('A00_Run.m');
tok_ = regexp(txt_, 'profile_jitter_sd_mm\s*=\s*([0-9.]+)', 'tokens', 'once');
jit_ = str2double(tok_{1});
fprintf('[3] A00 profile_jitter_sd_mm = %g\n', jit_);
if jit_ ~= 0
    fprintf('    FAIL: per-passage white profile perturbation must stay 0 (audit 2026-07-17)\n');
    fails = fails + 1;
end

% ---- Verdict --------------------------------------------------------------
if fails == 0
    fprintf('SMOKE AUDIT: ALL PASS\n');
else
    error('SMOKE AUDIT: %d CHECK(S) FAILED', fails);
end
