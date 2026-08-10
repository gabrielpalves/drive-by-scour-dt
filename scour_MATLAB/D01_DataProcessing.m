function [data] = D01_DataProcessing(i, j, Sol, Train, Calc, Damage, data)
%% RAW (time-domain) save — "Option B", 2026-07-14
% Stores the vehicle responses EXACTLY as the solver produced them (TIME domain,
% un-interpolated, noise-free) plus every parameter Python needs to reproduce the
% space-domain transform and the bridge crop at LOAD time.
%
% WHY: the time->space transform is a LINEAR interpolation, so noise added BEFORE
% it is band-limited/coloured and speed-dependent, while noise added AFTER it is
% white — the two are NOT equivalent (verified: ~0.67x variance, +0.77 lag-1
% autocorrelation, ~1.46x the energy surviving PAA). Saving the raw signal keeps
% BOTH options open forever: the measurement model lives entirely at load time and
% can be changed without ever regenerating the data.
%
% STORAGE is ~neutral: the space grid is 100 samples/m and DimSpace ~= 2.2*DimAcel,
% so the raw time series is about the size of the old cropped space window — but it
% covers the WHOLE passage (approach + bridge + exit) instead of just the crop.
%
% Python mirror: core/dataset._raw_to_space_crop  (np.interp == interp1 'linear').

% ---- Raw time-domain channels (leading vehicle) -------------------------
% Rows: A(1:3) = car-body / front-bogie / rear-bogie VERTICAL acceleration;
%       V(4:6) = car-body / front-bogie / rear-bogie PITCH RATES (V(1:3) are
%                vertical velocities — do NOT use them for pitch);
%       acc_under(1:4) = N(x_w)'*A_rail, the Eulerian (partial-time) rail FE
%                vertical-acceleration field interpolated at each instantaneous
%                wheel coordinate. It is NOT wheelset/axle-box acceleration and
%                NOT the total acceleration following the moving contact point;
%                B66 adds v^2*u_xx + 2*v*(u_t)_x separately. AcelRodaPrimVag is
%                retained below solely as the legacy on-disk field identifier.
%       AcelWheelsetPrimVag(1:4) = channel schema `physical8_v1` (2026-08-06):
%                idealized model-predicted constrained-wheelset vertical
%                acceleration along the moving contact coordinate, used as an
%                axle-box response proxy. AcelRodaPrimVag is NOT that quantity.
data.AceleracaoPrimVag{i,j} = Sol.Veh(1).A(1:3,:);
data.PitchPrimVag{i,j}      = Sol.Veh(1).V(4:6,:);
data.AcelRodaPrimVag{i,j}   = Sol.Veh(1).acc_under(1:4,:);

% ---- physical8_v1: TOTAL wheelset vertical acceleration (2026-08-06) -----
% AcelRodaPrimVag above is the EULERIAN partial-time rail field only. The
% wheelset is constrained to follow the rail/profile, so this idealized model
% proxy is the total derivative along the moving contact coordinate:
%
%     z_w,tt = u_tt + 2*v*u_xt + v^2*u_xx + h_tt
%              \___/   \______/   \______/   \___/
%           acc_under  vel_under_p def_under_pp hdd_path
%
% All four terms already exist at this point: B17_CalcUat returns the three
% rail-field terms (Eulerian acceleration, mixed space-time derivative, second
% spatial derivative) and B25_WheelProfiles returns the profile's second time
% derivative. Constant speed per passage, so v is scalar.
%
% AUDIT 2026-08-09: computed by ttbi.wheel_contact_kinematics, the SAME helper
% B66_ContactForce uses, so the saved channel and the reported contact force
% cannot drift apart. Profile inertia is masked with the solver's own active
% mask (elexj > 0), not x_path >= 0 — see the helper's header.
%
% This is an ADDITIVE field: AcelRodaPrimVag keeps its legacy name, rows and
% meaning as the virtual rail-field diagnostic for solver V&V. Four rows are
% stored (not just the two deployed channels) so the new field stays
% structurally parallel to that diagnostic. Neither field is an instrument
% model; mounting dynamics, contact compliance, bandwidth and filtering are
% outside this response definition.
wheelset_acc_all = ttbi.wheel_contact_kinematics( ...
    Sol.Veh(1), Calc.Veh(1), Train.vel);
data.AcelWheelsetPrimVag{i,j} = wheelset_acc_all(1:4,:);

% ---- Legacy AcelRoda-group noise: OFF by policy (desvio=0) -------------
% Kept CONDITIONAL so the legacy pipeline stays reproducible, and so no random
% numbers are drawn when it is off (an unconditional randn would perturb the RNG
% stream). When enabled, this is a compatibility perturbation on the saved
% virtual moving-rail response. It must not be described as physical
% wheel-sensor noise; Python preserves its interpolation behavior.
if Damage.desvio > 0
    data.AcelRodaPrimVag{i,j} = data.AcelRodaPrimVag{i,j} + Damage.desvio * ...
        data.AcelRodaPrimVag{i,j} .* randn(size(data.AcelRodaPrimVag{i,j}));
end

data.Velocidade{i,j} = Train.vel;
data.Posicao{i,j}    = Train.vel*(Calc.Solver.num_t/1000);   % distance travelled [m]

% ---- Everything Python needs to rebuild space domain + bridge crop -------
% Space domain is 100 samples/m. The time->space map is UNIFORM (constant speed
% per passage), exactly as the legacy code did it:
%     xx = linspace(1, DimSpace, DimAcel);  xi = 1:DimSpace;  interp1(xx, y, xi)
% Crop window = 10 m approach skip + bridge span + 18.30 m crossing/after
% (1831 retained post-deck samples = 1830 grid intervals; exact semantics in
% the registered-crop-constants block below).
% Calc.Profile.L_bridge is the LIVE bridge length for every mode (60 / 99.6),
% so the crop spans the whole deck. (Until the 2026-07-17 audit, profile_mode=
% 'fixed' let B19 overwrite Calc.Profile wholesale and L_bridge reverted to the
% stored 39.9 m — a ~40 m crop on a 60/99.6 m bridge; B19 now preserves the live
% geometry.) L_bridge_eff is still recorded per passage so the crop is
% self-describing. NOTE: the RAW full-length signal is saved, so a wrong crop
% could always be repaired in Python without regenerating.
DimAcel  = size(Sol.Veh(1).A, 2);
DimSpace = round(data.Posicao{i,j}*100);

% ---- Registered crop constants (values UNCHANGED; semantics documented) ---
% Exact frame trace (B43_ModelGeometry; campaign redux=0, sleeper 0.6 m):
% L_Approach = 16.2 m, extra_L2 = 6.0 m, max_TL = 106.8 m, so the leading
% wheel starts at x_0 = 112.8 m and the deck starts at L_Aw = 123.0 m; the
% travel from t=0 to deck entry is therefore 10.2 m (R11 re-audit 2026-07-28).
% crop_start = 1001 spans 1000 grid intervals = 10.00 m of travel, so the
% retained window opens ~0.20 m BEFORE the deck entry. It still contains the
% whole deck (R2/R3 crop audits).
% post_deck_samp = 1831 spans 1830 intervals = 18.30 m, numerically equal to
% the leading vehicle's first-to-last axle span (B43 axle offsets
% [0, 2.3, 16.0, 18.3] m => wheelbase 18.3 m); combined with the early start,
% the window ends ~0.20 m before exact last-axle deck clearance. These ~0.20 m
% offsets are DOCUMENTED OBSERVED BEHAVIOR of the registered constants, not a
% claimed original design intent. Both values are registered generation
% identity mirrored by core/dataset._raw_to_space_crop, and the RAW
% full-length signal is saved, so the crop is always recoverable. Do NOT
% change either value without a generation-behavior version bump.
crop_start     = 1001;                                % ~10 m approach margin [samples]
post_deck_samp = 1831;                                % 18.30 m post-deck window [samples]
% AUDIT R3 2026-07-17: round AFTER scaling by 100 samples/m. round(L)*100 gave
% round(99.6)*100 = 10000 (a 0.4 m / 40-sample over-crop at L99.6); the correct
% span is round(100*99.6) = 9960 samples. Identical at L60 (both 6000).
bridge_samp = round(100 * Calc.Profile.L_bridge);     % bridge span [samples]
crop_end    = min(crop_start - 1 + bridge_samp + post_deck_samp, DimSpace);

data.DimAcel(i,j)      = DimAcel;
data.DimSpace(i,j)     = DimSpace;
data.crop_start(i,j)   = crop_start;
data.crop_end(i,j)     = crop_end;
data.bridge_samp(i,j)  = bridge_samp;
data.L_bridge_eff(i,j) = Calc.Profile.L_bridge;

end
