% smoke_geometry.m — bridge-geometry / crop / CONTACT regression (audit 2026-07-17).
% Runs a full healthy passage for L60/L99.6 x fixed/psd_fra (and L99.6/fixed at
% 70/80/90 km/h) and asserts that (a) B19 keeps the LIVE bridge geometry,
% (b) D01 crops the whole deck at round(100*L_bridge) samples, (c) the profile
% covers the whole track and is finite, and (d) there is NO wheel-rail tension
% (contactLost_track == 0). (d) is the check that a previous version lacked, so
% a discontinuous profile-tiling seam that spiked accel to ~176 m/s^2 slipped
% through. Ends with error() on any failure (nonzero exit). ~6 full solves.
clear; clc;
fails = 0;
% {mode, L_bridge, num_spans, speed_kmh}
cfgs = {{'fixed', 60, 3, 80}, ...
        {'fixed', 99.6, 4, 70}, {'fixed', 99.6, 4, 80}, {'fixed', 99.6, 4, 90}, ...
        {'psd_fra', 60, 3, 80}, {'psd_fra', 99.6, 4, 80}};

for i = 1:numel(cfgs)
    mode = cfgs{i}{1}; Lb = cfgs{i}{2}; ns = cfgs{i}{3}; vk = cfgs{i}{4};
    fprintf('\n=== %s  L=%.1f  (%d spans)  %d km/h ===\n', mode, Lb, ns, vk);
    [sig, Calc, data, Sol] = local_geo_passage(mode, Lb, ns, vk);

    % 1. B19 kept the LIVE bridge length (not the stored 39.9 m)
    Lb_live = Calc.Profile.L_bridge;
    fprintf('  Calc.Profile.L_bridge = %.4g  (expect %.4g)\n', Lb_live, Lb);
    if abs(Lb_live - Lb) > 1e-6
        fprintf('  FAIL: L_bridge reverted (fixed-profile overwrite regression)\n');
        fails = fails + 1;
    end

    % 2. D01 crop spans the whole deck: bridge_samp = round(100*L_bridge)
    Lb_eff = data.L_bridge_eff(1,1);
    bs_expect = round(100 * Lb);
    fprintf('  D01 L_bridge_eff = %.4g | bridge_samp = %d (expect %d)\n', ...
        Lb_eff, data.bridge_samp(1,1), bs_expect);
    if abs(Lb_eff - Lb) > 1e-6 || data.bridge_samp(1,1) ~= bs_expect
        fprintf('  FAIL: crop window does not span the live deck\n');
        fails = fails + 1;
    end

    % 3. Profile covers the whole live track and is finite (no seam NaN)
    nx_x = numel(Calc.Profile.x); nx_h = numel(Calc.Profile.h);
    fprintf('  profile: x %d pts to %.1f m | h %d pts | finite %s\n', ...
        nx_x, max(Calc.Profile.x), nx_h, string(all(isfinite(Calc.Profile.h))));
    if nx_x ~= nx_h || ~all(isfinite(Calc.Profile.h)) || ...
            max(Calc.Profile.x) < Calc.Profile.L - 1
        fprintf('  FAIL: profile does not cover the live track / has NaN\n');
        fails = fails + 1;
    end

    % 4. NO wheel-rail tension anywhere on the track (bilateral-solver validity)
    fprintf('  contact: lost_track=%d | tension_frac_max=%.2e | F_tension_max=%.3e N\n', ...
        Sol.contactLost_track, Sol.tension_frac_max, Sol.F_tension_max);
    if Sol.contactLost_track ~= 0
        fprintf('  FAIL: wheel-rail TENSION (profile seam / roughness too harsh)\n');
        fails = fails + 1;
    end

    % 5. Signal finite
    if ~all(isfinite(sig(:)))
        fprintf('  FAIL: signal has NaN/Inf\n'); fails = fails + 1;
    end
end

fprintf('\n');
if fails == 0
    fprintf('SMOKE GEOMETRY: ALL PASS\n');
else
    error('SMOKE GEOMETRY: %d CHECK(S) FAILED', fails);
end

% -------------------------------------------------------------------------
function [sig, Calc, data, Sol] = local_geo_passage(mode, L_bridge, num_spans, vk)
    Train = A01_Train(vk/3.6, zeros(5,3));
    Track = A02_Track();
    Beam_seed = struct();
    Beam_seed.Prop.L = L_bridge;
    Beam_seed.Prop.num_spans = num_spans;
    Beam = A03_Bridge(Beam_seed);
    Profile_cfg = struct('mode', mode);
    if strcmp(mode, 'psd_fra'), Profile_cfg.fra_class = 4; end
    [Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);
    Damage = struct();
    Damage.desvio = 0;
    Damage.scour_rates = zeros(1, num_spans+1);
    Damage.bearing_left = 0; Damage.bearing_right = 0;
    Damage.crack_locs = []; Damage.crack_intensity = []; Damage.crack_lc = 0;
    [Sol, Calc, Train, ~, ~] = B00_Calculations(Calc, Train, Track, Beam, Damage);
    data = struct();
    data = D01_DataProcessing(1, 1, Sol, Train, Calc, Damage, data);
    sig = [data.AceleracaoPrimVag{1,1}; data.AcelRodaPrimVag{1,1}; ...
           data.PitchPrimVag{1,1}];
end
