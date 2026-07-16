% Main script to run TTB-2D model

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *************************************************************************
clear; clc; close all
% process_results;

% =========================================================================
% Simulation setup
% =========================================================================
% ===== STAGE PRESET — the paper-case selector ============================
% Pick ONE. Each preset maps to a CASE presented in the paper and sets the
% damage-defining toggles below (geometry, which damages are TARGETS vs EOV
% nuisances). Everything else (passage counts, ranges, variability) is a
% stage-independent knob kept at its default further down. This is the single
% place to switch between paper cases.
%
%   'stage0_multiscour' — 2 independent scours, 3-span/60 m, NO other damage.
%                         The localisation gate. == the dataset already run.
%   'stage1_bearing'    — + abutment bearing as a LABELLED TARGET (scour-vs-
%                         bearing disentanglement). Adds data.bearing_vector.
%   'stage1_eov'        — scour target only, but crack + rail-profile turned on
%                         as randomized EOV NUISANCES (confounder-robustness /
%                         "when does scour detection break?" study).
%   'stage1_full'       — bearing target AND crack+profile EOV nuisances.
%   'stage2_4span'      — scale-up: 4-span/100 m, 3 internal piers, bearing
%                         target + EOV nuisances.
%   'stage3_alldamage'  — "kitchen-sink" robustness arm (ablation paper only):
%                         Stage-1 geometry/targets + crack + psd_fra + TRACK-
%                         LAYER damage (ballast patches, hanging sleepers,
%                         rail pads) + wheel OOR/flats. All nuisances, logged.
%                         Spec: docs/stage3_alldamage_spec.md.
% ===== THE ABLATION LADDER (2026-07-14 design; user-approved) ============
% ONE factor changes per rung, so any degradation is ATTRIBUTABLE. The bridge
% maintainer's questions in order: can I localise scour? does a bearing fool
% me? does a crack fool me? do BOTH? does rail roughness? does track damage?
% does the train itself? Then repeat the key rungs on the bigger bridge.
%
%   L60 / 3-span ladder (scour piers 2,3)
%     s0_scour       scour only .................... baseline + architecture selection
%     s11_bear       + bearing (HEAD)
%     s12_crack      + crack (nuisance, no bearing)
%     s13_bearcrack  + bearing + crack ............. all BRIDGE damages
%     s14_prof       + rail profile (FRA-4, per-state) ...... the ROUGHNESS rung
%     s15_track      + track-layer damage (ballast / hanging sleepers / pads)
%     s16_all        + wheel OOR + flats ........... ALL damages
%   L99.6 / 4-span scale-up (scour piers 2,3,4)
%     s21_scour4     scour only
%     s22_bearcrack4 + bearing + crack ............. all BRIDGE damages
%     s23_all4       all damages
%
% HEADS = scour (per pier) + bearing (per abutment) ONLY. Crack, rail profile,
% track-layer and wheel damage are NUISANCES: randomized, logged, never
% estimated - the network must be INVARIANT to them, not report them. (Cracks
% are visually inspectable; submerged scour is not - that is where drive-by
% earns its keep.) Rationale: docs/framework_rationale.md.
STAGE = 's0_scour';

% Defaults; each rung below only switches ON what it adds.
use_track_eov = false;
use_oor_eov   = false;

switch STAGE
    % ---------------- L60 / 3-span bridge-damage ladder ------------------
    case 's0_scour'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='off';    use_crack_eov=false; profile_mode='fixed';
    case 's11_bear'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=false; profile_mode='fixed';
    case 's12_crack'
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='off';    use_crack_eov=true;  profile_mode='fixed';
    case 's13_bearcrack'
        % All BRIDGE damages (Fernandes-comparable set: scour + bearing + crack).
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='fixed';
    case 's14_prof'
        % THE ROUGHNESS RUNG: adds ONLY the rail profile (per-STATE FRA-4
        % realization + per-passage jitter) on top of s13. Isolates the
        % roughness effect that collapsed the sprung channels in the deprecated
        % L100 pilot - the single most important open question in the chain.
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
    case 's15_track'
        % + TRACK-layer damage (ballast patches, hanging sleepers, rail pads).
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
        use_track_eov=true;
    case 's16_all'
        % + TRAIN (wheel OOR / flats) = ALL damages. Split from s15 so a drop
        % is attributable to the rail vs the train.
        damage_mode='multi_scour'; L_bridge=60;  num_spans=3; scour_supports=[2 3];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
        use_track_eov=true;    use_oor_eov=true;

    % ---------------- L99.6 / 4-span scale-up ----------------------------
    % L=99.6 (not 100): spans of 24.9 m = 83 elements of 0.3 m EXACTLY, so all 5
    % supports land on mesh nodes (B43 mesh / B02 snapping). L=100 re-meshes to
    % 99.9 m with snapped spans 24.9/24.9/25.2/24.9 AND a floating-point tie at
    % the mid support - that is what the deprecated first Stage-2 run used.
    case 's21_scour4'
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        bearing_mode='off';    use_crack_eov=false; profile_mode='fixed';
    case 's22_bearcrack4'
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='fixed';
    case 's23_all4'
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        bearing_mode='target'; use_crack_eov=true;  profile_mode='psd_fra';
        use_track_eov=true;    use_oor_eov=true;
    otherwise
        error('A00: unknown STAGE preset "%s"', STAGE);
end
% Verify the support positions land on mesh nodes for your L/num_spans
% (B07/B43) before a long run. num_spans=N -> supports {1..N+1}; internal
% piers (valid scour targets) are 2..N.

% ===== Operational variability (keep ON to match the training distribution) ===
% NOISE POLICY 2026-07-12 (user decision): generation is NOISE-FREE from
% stage1_crack onward. Measurement noise is injected at LOAD time instead
% (core/dataset.py 'sensor_noise' config), where per-channel levels stay
% configurable per experiment - levels from sensor DATASHEETS (noise floor);
% EN 61373 position severities describe the vibration ENVIRONMENT for
% equipment qualification (range/reliability), NOT the acquisition noise.
% Set true ONLY to reproduce the legacy Stage-0/1 pipeline (wheel-only
% multiplicative noise applied in D01; adds 'N' to the folder var_tag).
use_signal_noise = false;       % legacy toggle - keep FALSE (noise at load time)
use_vehicle_variability = true; % Toggle vehicle property variability
use_speed_variability = true;   % Toggle train speed variability
use_temp_variability = true;    % Toggle temperature variability

% ===== Damage states =====
% --- single_scour: continuous scour grid on the central pier (file k -> Dano(k)).
Dano  = linspace(0.0, 0.60, 61);  % used only when damage_mode == 'single_scour'

% --- multi_scour: independent scour at scour_supports (set by the STAGE preset).
n_states_multi   = 250;     % number of independent joint damage states (LHS)
                            % (250 + 9 anchors = 259, matching the Stage-0 run)
dano_max         = 0.60;    % per-support max scour fraction
include_anchors  = true;    % prepend healthy + single-pier sweeps (localisation extremes)
n_anchor_levels  = 4;       % single-pier anchor levels per scoured support
damage_seed      = 1;       % RNG seed for a REPRODUCIBLE damage-state grid

Npass = 50;                 % passages per damage state (operational variability)

% ===== Bearing damage (abutment rotational stiffness, Nm/rad) =====
% MODE is set by the STAGE preset; these are the numeric knobs it uses.
% 'off'    healthy bearings everywhere.
% 'fixed'  the SAME Bearing_Intensity at the LEFT abutment for every state.
% 'target' bearing state [left,right] becomes a LABELLED TARGET — sampled
%          JOINTLY with scour by one LHS and saved as data.bearing_vector.
Bearing_Intensity = 0.0;    % 'fixed' mode value [Nm/rad]
% ===== RANGE EXTENDED 2026-07-15: sample FIXITY, not raw k_r =============
% The old design sampled k_r ~ U(0, 1e9) and labelled seized-% = k_r/1e9.
% Problem: Fernandes's 1e9 is the MINIMUM drive-by-DETECTABLE seized bearing
% (a detectability FLOOR from the Feng/O'Brien line) — using it as our MAXIMUM
% meant the whole label range sat between "healthy" and "barely detectable".
% Khan (2022) sweeps k_r continuously to 1e12 and reports near-full fixity, so
% real seized bearings live far above 1e9.
% The response is governed by the ANALYTIC FIXITY RATIO  phi = k/(k + 4EI/L_end)
% (verified by our own k_r sweep, results/bearing_sensitivity/). For our deck
% 4EI/L_end ~ 2.3e9, so k_r = 1e9 is only phi ~ 0.30 — the lower third of the
% available response. But a LINEAR draw over a big k_r range is also wrong: it
% would pile almost all the response into the first 1% of the label.
% FIX: draw phi ~ U(0, bearing_fixity_max) and invert  k_r = phi/(1-phi)*4EI/L.
% phi is bounded [0,1], near-LINEAR in response (a good regression target), and
% GEOMETRY-NORMALISED — so the bearing label means the same thing on the L60 and
% L99.6 bridges, which the raw k_r never did. Fernandes's 1e9 survives as a
% reportable landmark (phi ~ 0.30), not as a cap.
bearing_fixity_max = 0.95;  % max fixity ratio sampled (0 = free, ->1 = fully fixed)

% ===== Confounder damages — EOV nuisance augmentation =====
% use_crack_eov is set by the STAGE preset; these are its numeric knobs.
% Cracks are NOT labels: the network never estimates them. EOV DESIGN REVIEW
% 2026-07-12 (deep research, papers/'Drive-By Scour ML Literature Design'):
% a crack is a PERSISTENT condition -> drawn once per damage STATE and held
% for all its passages (crack_draw='per_state'), with prevalence ~0.25 (a
% p=1.0 per-passage redraw deprives the model of any healthy-deck baseline
% and is physically indefensible). Semantics mirror TTBI_2D/damage_config.py
% (local EI reduction, Sinha et al.).
crack_draw       = 'per_state';   % 'per_state' (persistent damage) | 'per_passage' (DEPRECATED)
crack_p          = 0.25;          % P(state carries a crack) — CONFIRMED 2026-07-15:
% deep research puts spans with a MACROSCOPIC EI loss in our Sinha 5-30% band at
% 20-30% of an aged concrete/composite inventory (marginal non-structural cracks
% reach 70-80%, but those do not produce a modelled EI drop). 0.25 stands.
%
% ===== Crack LOCATION: HOGGING-WEIGHTED (corrected 2026-07-15) ===========
% We previously drew U(0.10, 0.90)*L and justified it from the moving-load |M|
% ENVELOPE (broad; peaks mid-span 1.00/0.84 vs over-pier 0.51/0.42, i.e. ~2:1
% FAVOURING mid-span; only ~2-4% of the range ever sees |M|<35% of max).
% The deep research OVERTURNS that conclusion — and the moment envelope was the
% wrong lens. Crack PREVALENCE is set by the MATERIAL and the ENVIRONMENT, not
% by moment magnitude alone: over an internal support the deck is in HOGGING, so
% the TOP fibre is in tension (concrete's weakest mode, f_ct ~ few MPa) and that
% same top surface takes deck runoff / chlorides. At mid-span the tension is on
% the soffit and early thermal/shrinkage cracks there tend to close under
% compression, rarely giving a real EI loss. Reported ratio: hogging cracking
% exceeds sagging by >4:1 to 5:1 in occurrence AND inertial impact. Eurocode 4
% goes further and MANDATES analysing a "cracked" section over 15% of the span
% each side of internal supports. So: sample cracks preferentially in the
% over-pier zones. (Our earlier caveat — "hogging may crack more than its moment
% share" — was right; this quantifies it.)
crack_hog_ratio  = 4.0;           % hogging:sagging occurrence ratio (report >4:1..5:1)
crack_hog_margin = 0.175;         % zone half-width as a fraction of SPAN (EC4: 15%; report 15-20%)
crack_frac_range = [0.10 0.90];   % LEGACY uniform bounds — used only as a clamp now
crack_int_range  = [0.05 0.30];   % EI-loss fraction (Fernandes 2025: 0.14/0.22)
crack_lc         = 0.0;           % half-length [m] each side; <=0 -> single element

% ===== Rail profile mode (EOV) =====
% profile_mode is set by the STAGE preset; these are its numeric knobs.
% 'fixed'        legacy measured profile — byte-identical baseline (Type 2).
% 'fixed_scaled' measured profile x amplitude factor drawn per passage.
% 'psd_fra'      profile REGENERATED from the FRA PSD; severity = FRA class.
% EOV DESIGN REVIEW 2026-07-12: track geometry evolves over MGT, not between
% trains (EN 13848-2 pass-to-pass repeatability <=0.5 mm @95%; Sato/Shenton) ->
% ONE class + ONE phase realization drawn per damage STATE and held for its
% passages (profile_draw='per_state'; B19 seeds the phase draw per state), plus
% a small additive per-passage jitter (metrological repeatability / wander).
% Class set FIXED at FRA class 4 = roughest geometry permissible at 70-90 km/h
% (classes 5/6 are premium track, too smooth for a scour-prone regional line).
% The old per-passage class+phase redraw over {4,5,6} is DEPRECATED (it is what
% collapsed the sprung channels in the L100 Stage-2 pilot).
profile_draw         = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
profile_jitter_sd_mm = 0.5;          % per-passage additive white noise [mm] (EN 13848-2)
profile_int_range    = [0.5 2.0];    % 'fixed_scaled' amplitude range
profile_fra_classes  = 4;            % 'psd_fra' class set (scalar = fixed class)

% ===== Track-layer damage EOVs (Stage 3; use_track_eov) =====
% Verified sampling spec: docs/track_eov_sampling_spec.md (deep-research,
% quote-checked). Randomized NUISANCES — logged, NOT labels. EOV DESIGN REVIEW
% 2026-07-12: ballast patches / hanging sleepers / pad aging are PERSISTENT
% infrastructure conditions (same argument as crack/profile) -> drawn once per
% damage STATE (track_draw='per_state'). Wheel OOR stays per-PASSAGE: each
% passage plausibly is a different train of the fleet (vehicle variability).
% Track x-coords: the bridge occupies [track_L_app, track_L_app+L_bridge].
track_draw        = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
track_L_app       = 30;          % = A04 minL_Approach (fixed, 50 sleepers)
track_L_after     = 30;          % modelled track beyond the bridge [m]
% ===== COUNTS: now ANCHORED (deep research 2026-07-15) ===================
% Source: papers/'Track Defect Prevalence Data Search' (Gemini Deep Research;
% derivations flagged INFERENCE but each built on a cited anchor). It RESOLVED
% the prevalence paradox that blocked us: the cited "~50% of concrete sleepers
% show some voiding" and our ~9% mechanical model are BOTH right, because most
% voids are SUB-THRESHOLD. Void taxonomy: <1.0 mm = accommodation (rail/fastener
% elasticity absorbs it); 1.0-2.5 mm = DYNAMIC IMPACT threshold (a 1-2 mm void
% raises adjacent sleeper-ballast contact force up to 70% and wheel loads ~85%);
% >2.5 mm = critical. Only 10-20% of visibly-settled sleepers exceed 1.5-2.0 mm
% => IMPACTFULLY unsupported = 5-10% of sleepers (mid 7.5%).
% Derived counts PER 100 m: 0.075*167 = 12.5 sleepers / ~3 per group = 4.2
% groups; post-tamping (3-5%) = 1.6-2.7 => the report recommends POISSON with
% lambda ~ 2.0-3.0 for the intermediate degradation cycle. Ballast: GPR surveys
% put FI>30 ("highly fouled") at 10-20% of route length; 15% / 12.5 m mean patch
% = 1.2 patches per 100 m.
% RATES are per 100 m and are SCALED BY THE MODELLED WINDOW below (our track is
% 120 m at L60 / 159.6 m at L99.6 - drawing a fixed count regardless of length
% was itself an error).
% ⚠ Report-internal inconsistency, resolved in our favour of its own derivation:
% its summary table claims P(no hanging group)~0.25 while its text recommends
% lambda 2-3 (which gives P(0)=0.05-0.14). We follow the DERIVATION (lambda=2.5).
% ⚠ VERIFY: the pivotal "FI>30 on 10-20% of length" cites a Slideshare deck -
% re-check against the peer-reviewed GPR sources in the same report before the
% paper (Tandfonline 2022 / MDPI Sensors GPR condition-index papers).
% lambda = 3.0 RECONCILES the report's two self-inconsistent statements: it
% recommends Poisson lambda 2.0-3.0 ("intermediate degradation cycle"), but its
% own table demands 5-10% of sleepers impactfully unsupported. lambda*3 sleepers
% per group / 167 per 100 m => lambda must be >= 2.78 for >=5%, and <= 3.0 to
% stay in the recommended range. lambda = 3.0 is the only value satisfying BOTH
% (gives 5.4%). MC-verified.
hang_rate_100m    = 3.0;         % unsupported-sleeper GROUPS per 100 m (Poisson)
ballast_rate_100m = 1.2;         % fouled PATCHES per 100 m (Poisson; = 15% length / 12.5 m)
ballast_patch_len = [5 20];      % patch length U(5,20) m (cited)
% Fouling <-> voiding are STRONGLY correlated (bidirectional: fouling -> mud
% pumping -> lubrication -> void; void -> impact -> pulverisation -> fouling).
% Report: weight hanging-group placement x3 inside a fouled patch (and ~1/3
% outside). Drawing them independently was a documented modelling vulnerability.
hang_foul_mult    = 3.0;         % relative density of hanging groups inside a fouled patch
% Ballast fouling is ALSO elevated at bridge transitions (impact + deck runoff
% into the expansion joint); corrective work there is 4-8x more frequent and the
% settlement/pulverisation rate 3-4x. Report: multiply fouled-patch probability
% by ~3 within 15-24 m of the abutment.
ballast_trans_mult   = 3.0;      % fouled-patch density multiplier near the abutments
ballast_trans_margin = 20;       % [m] extent of that zone (report: 15-24 m)
ballast_p_wet     = 0.5;         % P(wet/saturated) vs dry-fouled state
ballast_eta_k_dry = [1.2 2.0]; ballast_eta_c_dry = [0.4 0.8];   % (cited)
ballast_eta_k_wet = [0.7 0.9]; ballast_eta_c_wet = [1.5 4.0];   % (cited)
% (group COUNT is now Poisson(hang_rate_100m * track_win/100) - see above)
hang_group_size   = [1 5];       % consecutive sleepers, DU(1,5) (cited)
hang_p_transition = 0.6;         % P(group in a transition zone) [assumption]
hang_trans_margin = 15;          % density-spike zone +-15 m of abutments (cited)
pad_chi_range     = [1.0 3.5];   % pad aging stiffness multiplier bounds
pad_weibull       = [1.8 2.2];   % chi_pad ~ Weibull(lambda,k), clipped to range
pad_beta_range    = [0.8 1.2];   % pad damping multiplier
% Pad failures: the cited 0.5% is an ANNUAL INCIDENCE RATE, not a snapshot
% prevalence (2026-07-15 report). Secondary lines renew small components on a
% 3-5 yr cadence, so failures ACCUMULATE: observed snapshot prevalence = 1.5-3.0%
% of fastening positions. Our old 0.005 (=> ~0.83 pads/100 m) was ~4x
% SUB-representative of real regional-line noise; the report asks for ~2-5 failed
% positions per 100 m, which 0.02 delivers (0.02 * 167 = 3.3 per 100 m).
pad_p_fail        = 0.02;        % per-pad SNAPSHOT failure prevalence (report 1.5-3.0%)

% ===== Wheel flats + polygonization (Stage 3; use_oor_eov) =====
% LITERATURE-VERIFIED (deep research 2026-07-09; docs/stage3_alldamage_spec.md):
% ~12% of in-service wheels carry a flat; braking couples axles within a bogie.
% Generative model: independent per-bogie slide events (q), leading axle always
% flats, trailing w.p. oor_p_trailing -> wheel marginal 0.12, P(bogie2|bogie1)~q.
% (Left/right 0.85 same-axle correlation is N/A in this 2D single-rail model.)
oor_q_bogie     = 0.171;         % P(slide event) per bogie
oor_p_trailing  = 0.40;          % P(trailing axle also flats | event)
oor_p_fresh     = 0.125;         % P(flat is FRESH | flat) (1.5% vs 10.5% split)
oor_len_fresh   = [0.010 0.035]; % fresh flat length U(10,35) mm (cited)
oor_len_runin   = [0.030 0.060]; % run-in flat length U(30,60) mm (cited)
oor_radius      = 0.46;          % wheel radius R [m]
% depth: FRESH d=L^2/(8R) (chord sagitta) | RUN-IN d=L^2/(16R) (filtered) — cited
% ---- low-order polygonization: separate CONTINUOUS nuisance (cited verdict) ----
poly_p_wheel    = 0.30;          % P(a wheel is polygonized)
poly_orders     = [1 5];         % harmonic order n ~ DU(1,5)
poly_amp_lnorm  = [-10.0 0.5];   % ln(amp [m]) ~ N(mu,sigma) (median ~45 um)
poly_amp_bounds = [1e-5 1.2e-4]; % clip to cited service range 0.01-0.12 mm

Nveh = 5;     % number of vehicles
Nprop = 3;    % how many vehicle properties will be varied
Desvio = 0.05; % standard deviation of the noise of the signal
temp_min = 3; temp_max = 33; % min and max temperature [°C]
vel_min = 70; vel_max = 90; % min and max velocity [km/h]

% =========================================================================
%  Build the damage-state matrix  (n_states x n_supports), one row per file
% =========================================================================
% Every mode produces a DamageStates matrix whose row DC is the full per-support
% scour-rate vector for file DC. This unifies the parfor body and the labelling:
% the saved label is always data.scour_vector = DamageStates(DC,:).
n_supp = num_spans + 1;
n_bear = 2 * strcmp(bearing_mode, 'target');  % bearing label dims (left,right)
% Fixity->k_r conversion reference (see bearing_fixity_max above). A03 is a pure
% function, so probe it for the beam properties actually used.
Beam_probe  = A03_Bridge(struct('Prop', struct('L', L_bridge, 'num_spans', num_spans)));
k_ref_bear  = 4 * Beam_probe.Prop.E * Beam_probe.Prop.I / (L_bridge / num_spans);
fix2k       = @(phi) k_ref_bear .* phi ./ (1 - phi);   % k_r [Nm/rad] from fixity
fprintf('Bearing: 4EI/L_end = %.3g Nm/rad; fixity %.2f -> k_r = %.3g (Fernandes 1e9 = fixity %.2f)\n', ...
    k_ref_bear, bearing_fixity_max, fix2k(bearing_fixity_max), 1e9/(1e9 + k_ref_bear));
if strcmp(damage_mode, 'single_scour')
    central = floor(n_supp / 2) + 1;
    DamageStates = zeros(numel(Dano), n_supp);
    DamageStates(:, central) = Dano(:);
    scour_supports = central;                 % the single (central) target
else  % multi_scour
    rng(damage_seed);                         % reproducible joint grid
    n_tgt = numel(scour_supports);
    anchors_s  = zeros(1, n_supp);            % healthy (0,...,0)
    anchors_bf = zeros(1, 2);                 % bearing FIXITY (the label); k_r derived
    if include_anchors
        for ti = 1:n_tgt                      % single-pier scour anchors
            levels = linspace(dano_max / n_anchor_levels, dano_max, n_anchor_levels)';
            blk = zeros(numel(levels), n_supp);
            blk(:, scour_supports(ti)) = levels;   % ONE pier damaged, others 0
            anchors_s  = [anchors_s; blk]; %#ok<AGROW>
            anchors_bf = [anchors_bf; zeros(numel(levels), 2)]; %#ok<AGROW>
        end
        if n_bear > 0                         % single-bearing anchors (scour 0)
            for bi = 1:2
                levels = linspace(bearing_fixity_max / n_anchor_levels, ...
                                  bearing_fixity_max, n_anchor_levels)';
                blk = zeros(numel(levels), 2);
                blk(:, bi) = levels;
                anchors_bf = [anchors_bf; blk]; %#ok<AGROW>
                anchors_s  = [anchors_s; zeros(numel(levels), n_supp)]; %#ok<AGROW>
            end
        end
    end
    % ONE joint LHS over scour + bearing targets = broad joint coverage.
    % Bearing dims are sampled in FIXITY space (bounded, near-linear in response,
    % geometry-normalised) and inverted to k_r for the physics.
    lhs = lhsdesign(n_states_multi, n_tgt + n_bear);
    joint_s = zeros(n_states_multi, n_supp);
    joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;
    joint_bf = zeros(n_states_multi, 2);
    if n_bear > 0
        joint_bf = lhs(:, n_tgt+1:end) * bearing_fixity_max;
    end
    DamageStates       = [anchors_s; joint_s];
    BearingFixityMulti = [anchors_bf; joint_bf];
    BearingStatesMulti = fix2k(BearingFixityMulti);      % k_r [Nm/rad]
end
n_states = size(DamageStates, 1);

% Bearing state per file (n_states x 2 = [left,right]), any damage_mode:
switch bearing_mode
    case 'off'
        BearingStates = zeros(n_states, 2);
        BearingFixity = zeros(n_states, 2);
    case 'fixed'
        BearingStates = repmat([Bearing_Intensity, 0.0], n_states, 1);
        BearingFixity = BearingStates ./ (BearingStates + k_ref_bear);
    case 'target'
        if strcmp(damage_mode, 'single_scour')
            error('A00: bearing_mode=''target'' requires damage_mode=''multi_scour''.');
        end
        BearingStates = BearingStatesMulti;
        BearingFixity = BearingFixityMulti;
    otherwise, error('A00: unknown bearing_mode "%s"', bearing_mode);
end

tempo_inicial = datetime('now');
tempo_inicial_str = datestr(tempo_inicial, 'dd_MM_yyyy_HH_mm_ss');

% =========================================================================
%  Self-identifying run folder + case manifest
% =========================================================================
% The output folder is named after the CASE (bridge / damage / variability), not
% a timestamp: different cases never share a folder, and re-running the SAME case
% resumes into it (the parfor skips states whose NNNN.mat already exists). Move
% the finished folder under an 'observations/' tree and point the loader at it.
var_tag = [repmat('N',1,double(use_signal_noise)), ...
           repmat('V',1,double(use_vehicle_variability)), ...
           repmat('S',1,double(use_speed_variability)), ...
           repmat('T',1,double(use_temp_variability))];
if isempty(var_tag), var_tag = 'none'; end
% bear_tag keeps the legacy ON/OFF naming so existing run folders still match
% (resume-by-folder-name); new modes/EOVs only ADD tags when actually active.
switch bearing_mode
    case 'off',    bear_tag = 'OFF';
    case 'fixed',  if Bearing_Intensity > 0, bear_tag = 'ON'; else, bear_tag = 'OFF'; end
    case 'target', bear_tag = 'TGT';
end
% EOV tags: the ...ST suffix marks a PER-STATE draw (2026-07-12 design); the
% legacy tags (crackON / prof-psd_fra / trackEOV) meant per-passage redraw, so
% old folders remain distinguishable from new ones by name alone.
eov_tag = '';
if use_crack_eov
    if strcmp(crack_draw, 'per_state'), eov_tag = [eov_tag, '_crackST'];
    else,                               eov_tag = [eov_tag, '_crackON']; end
end
if ~strcmp(profile_mode, 'fixed')
    eov_tag = [eov_tag, '_prof-', profile_mode];
    if strcmp(profile_mode, 'psd_fra') && strcmp(profile_draw, 'per_state')
        eov_tag = [eov_tag, 'ST'];
    end
end
if use_track_eov
    if strcmp(track_draw, 'per_state'), eov_tag = [eov_tag, '_trackEOVST'];
    else,                               eov_tag = [eov_tag, '_trackEOV']; end
end
if use_oor_eov,                    eov_tag = [eov_tag, '_oorON']; end
supp_tag = strjoin(string(scour_supports), '-');
% ---- Folder name: SHORT (2026-07-14) -----------------------------------
% The old self-describing name ran to ~110 characters, which blew past
% Windows' 260-char MAX_PATH once nested under the repo + results tree. The
% folder is now just <stage>_L<len>_st<nstates> (~20 chars); the FULL
% descriptor is preserved as case_info.case_desc (+ case_info.txt) so the
% dataset stays self-describing without the path pain.
case_name = sprintf('%s_L%g_st%d', STAGE, L_bridge, n_states);
case_desc = sprintf('L%g_%dspan_%s_scourS%s_bear%s%s_dano0-%gpct_states%d_Npass%d_var%s', ...
    L_bridge, num_spans, damage_mode, supp_tag, bear_tag, eov_tag, ...
    dano_max*100, n_states, Npass, var_tag);
run_folder = fullfile('Results', case_name);
if ~exist(run_folder, 'dir'), mkdir(run_folder); end

% --- Manifest: machine-readable (.mat) + human-readable (.txt) -----------
case_info = struct( ...
    'case_name', case_name, 'case_desc', case_desc, 'stage', STAGE, ...
    'timestamp', tempo_inicial_str, ...
    'damage_mode', damage_mode, ...
    'L_bridge_m', L_bridge, 'num_spans', num_spans, ...
    'num_supports', n_supp, 'scour_supports', mat2str(scour_supports), ...
    'n_states', n_states, 'passages_per_state', Npass, ...
    'scour_dano_max_frac', dano_max, ...
    'bearing_mode', bearing_mode, ...
    'bearing_intensity_Nm_rad', Bearing_Intensity, ...
    'bearing_fixity_max', bearing_fixity_max, ...
    'bearing_k_ref_Nm_rad', k_ref_bear, ...   % 4EI/L_end; k_r = phi/(1-phi)*k_ref
    'bearing_label', 'fixity_ratio', ...      % label = phi in [0,1], NOT k_r/bearing_max
    'use_crack_eov', use_crack_eov, 'crack_draw', crack_draw, ...
    'crack_p', crack_p, ...
    'crack_hog_ratio', crack_hog_ratio, 'crack_hog_margin', crack_hog_margin, ...
    'crack_frac_range', mat2str(crack_frac_range), ...
    'crack_int_range', mat2str(crack_int_range), 'crack_lc', crack_lc, ...
    'profile_mode', profile_mode, 'profile_draw', profile_draw, ...
    'profile_jitter_sd_mm', profile_jitter_sd_mm, ...
    'profile_int_range', mat2str(profile_int_range), ...
    'profile_fra_classes', mat2str(profile_fra_classes), ...
    'use_track_eov', use_track_eov, 'track_draw', track_draw, ...
    'track_L_after', track_L_after, ...
    'ballast_rate_100m', ballast_rate_100m, ...
    'ballast_patch_len', mat2str(ballast_patch_len), ...
    'ballast_trans_mult', ballast_trans_mult, ...
    'ballast_trans_margin', ballast_trans_margin, ...
    'hang_rate_100m', hang_rate_100m, ...
    'hang_group_size', mat2str(hang_group_size), ...
    'hang_foul_mult', hang_foul_mult, ...
    'pad_p_fail', pad_p_fail, ...
    'use_oor_eov', use_oor_eov, ...
    'oor_q_bogie', oor_q_bogie, ...
    'oor_p_trailing', oor_p_trailing, ...
    'oor_p_fresh', oor_p_fresh, ...
    'oor_len_fresh', mat2str(oor_len_fresh), ...
    'oor_len_runin', mat2str(oor_len_runin), ...
    'oor_radius', oor_radius, ...
    'poly_p_wheel', poly_p_wheel, ...
    'poly_orders', mat2str(poly_orders), ...
    'poly_amp_lnorm', mat2str(poly_amp_lnorm), ...
    'use_signal_noise', use_signal_noise, 'noise_std', Desvio, ...
    'use_vehicle_variability', use_vehicle_variability, ...
    'use_speed_variability', use_speed_variability, ...
    'use_temp_variability', use_temp_variability, ...
    'n_vehicles', Nveh, 'n_props_varied', Nprop, ...
    'temp_min_C', temp_min, 'temp_max_C', temp_max, ...
    'vel_min_kmh', vel_min, 'vel_max_kmh', vel_max);
save(fullfile(run_folder, 'case_info.mat'), 'case_info');
% Also store the full damage-state matrices so the dataset is self-describing.
save(fullfile(run_folder, 'damage_states.mat'), 'DamageStates', 'BearingStates', ...
    'BearingFixity', 'k_ref_bear', 'scour_supports');
fid = fopen(fullfile(run_folder, 'case_info.txt'), 'w');
fn = fieldnames(case_info);
fprintf(fid, '%% TTBI dataset — case manifest\n');
for i = 1:numel(fn)
    v = case_info.(fn{i});
    if ischar(v), fprintf(fid, '%-26s : %s\n', fn{i}, v);
    else,         fprintf(fid, '%-26s : %g\n', fn{i}, v); end
end
fclose(fid);
fprintf('Run folder: %s  (%d states x %d passages)\n', run_folder, n_states, Npass);

% Identify which states have already been processed (for resume mode)
saved_files = dir(fullfile(run_folder, '*.mat'));
completed = false(n_states, 1);
for k = 1:length(saved_files)
    tokens = regexp(saved_files(k).name, '^(\d{4})\.mat$', 'tokens');
    if ~isempty(tokens)
        dc_idx = str2double(tokens{1}{1});
        if dc_idx >= 1 && dc_idx <= n_states
            completed(dc_idx) = true;
        end
    end
end

% =========================================================================
%  Parallel loop (PARFOR)  — one file per damage state
% =========================================================================
parfor DC = 1:n_states
    if completed(DC)
        fprintf('Skipping state %d — result already exists.\n', DC);
        continue;
    end

    % 1. Local initializations for PARFOR transparency
    Damage = struct();
    data = struct();
    data2save = struct();

    % Reproducible per-state RNG stream (speeds, temps, vehicle props and the
    % nuisance-EOV draws below all depend only on damage_seed + state index).
    rng(damage_seed * 100000 + DC);

    % Noise definition
    Damage.desvio = Desvio * use_signal_noise;

    % ---------------------------------------------------------------------
    % 2. CONFIGURE DAMAGE  (per-support scour vector; consumed by B02)
    % ---------------------------------------------------------------------
    scour_vec = DamageStates(DC, :);          % full per-support scour-rate row
    bear_vec  = BearingStates(DC, :);         % [left,right] abutment k_r [Nm/rad]
    bear_fix  = BearingFixity(DC, :);         % [left,right] FIXITY ratio = the LABEL
    Damage.scour_rates   = scour_vec;
    Damage.bearing_left  = bear_vec(1);
    Damage.bearing_right = bear_vec(2);
    % ---------------------------------------------------------------------

    % Nuisance-EOV logs (saved with the file for traceability; NOT labels).
    % Kept per-passage-shaped even for per-STATE draws (rows then repeat), so
    % downstream loaders read old and new datasets identically.
    CrackLog   = zeros(Npass, 3);             % [loc_m, EI-loss frac, lc_m]
    ProfileLog = ones(Npass, 1);              % intensity or FRA class (mode-dep.)
    TrackLog   = cell(Npass, 1);              % Damage.track struct per passage
    OORLog     = cell(Npass, 1);              % Damage.oor rows per passage

    % Per-STATE EOV state (2026-07-12 design review: persistent conditions -
    % crack, profile realization, track layer - are drawn ONCE per damage
    % state at j_pass==1 and HELD; wheel OOR stays per-passage = a different
    % train of the fleet each passage). Pre-initialised for parfor analysis.
    state_fra_class = profile_fra_classes(1);
    Tk = struct();

    % Pre-allocate LHS for this specific damage case (speed/temperature)
    lhs_matrix = lhsdesign(2, Npass);

    Velocidade = ones(1, Npass) * 80/3.6;
    Temperatura = ones(1, Npass) * 25;

    % Vehicle Variability
    if use_vehicle_variability
        x_veh = ones(Nveh, Nprop, Npass);
        for t = 1:Nveh
            for j = 1:Nprop
                variability_term = randn(1, 1, Npass);
                x_veh(t, j, :) = x_veh(t, j, :) .* variability_term;
            end
        end
    else
        x_veh = zeros(Nveh, Nprop, Npass);
    end

    vel_avg = (vel_max + vel_min) / 2;
    for t = 1:Npass
        if use_speed_variability
            Velocidade(t) = (round(vel_min + (vel_max-vel_min)*lhs_matrix(1,t)))/3.6;
        else
            Velocidade(t) = round(vel_avg)/3.6;
        end
        if use_temp_variability
            Temperatura(t) = round(temp_min + (temp_max - temp_min)*lhs_matrix(2,t));
        else
            Temperatura(t) = 25;
        end
    end

    % Inner loop for each passage
    for j_pass = 1:Npass
        Train_local = A01_Train(Velocidade(j_pass), x_veh(:, :, j_pass));
        Track_local = A02_Track();
        Beam_seed = struct();                      % drive geometry from the config above
        Beam_seed.Prop.L = L_bridge;
        Beam_seed.Prop.num_spans = num_spans;
        Beam_local  = A03_Bridge(Beam_seed);

        % Modulus of elasticity temperature adjustment
        Beam_local.Prop.E = Beam_local.Prop.E - Beam_local.Prop.E * 0.003 * (Temperatura(j_pass)-15);

        % --- Nuisance-EOV draws (domain randomization) --------------------
        % PERSISTENT conditions (crack, profile realization, track layer) are
        % drawn once per STATE at j_pass==1 and HELD for all its passages
        % ('per_state'; 2026-07-12 design review). The 'per_passage' branches
        % (redraw every passage) are DEPRECATED - kept only to reproduce
        % legacy datasets such as the L100 Stage-2 pilot.
        % Crack: local EI reduction applied in B00; NOT a label.
        if use_crack_eov && (strcmp(crack_draw, 'per_passage') || j_pass == 1)
            if rand() <= crack_p
                % HOGGING-WEIGHTED placement (2026-07-15): pick an over-pier zone
                % with probability ratio crack_hog_ratio:1 against a mid-span
                % zone, then jitter within +-crack_hog_margin of a SPAN length.
                % Nominal support positions (B02 snaps them a few cm - immaterial
                % for a nuisance prior).
                supp_nom = linspace(0, L_bridge, num_spans + 1);
                int_supp = supp_nom(2:end-1);            % internal supports = piers
                span_len = L_bridge / num_spans;
                if ~isempty(int_supp) && rand() < crack_hog_ratio/(crack_hog_ratio + 1)
                    s_ = int_supp(randi(numel(int_supp)));               % HOGGING
                else
                    k_ = randi(num_spans);                               % SAGGING
                    s_ = (supp_nom(k_) + supp_nom(k_+1))/2;
                end
                c_loc = s_ + (2*rand() - 1) * crack_hog_margin * span_len;
                c_loc = min(max(c_loc, crack_frac_range(1)*L_bridge), ...
                                       crack_frac_range(2)*L_bridge);
                c_int = crack_int_range(1) + diff(crack_int_range)*rand();
                Damage.crack_locs      = c_loc;
                Damage.crack_intensity = c_int;
                Damage.crack_lc        = crack_lc;
            else
                Damage.crack_locs      = [];
                Damage.crack_intensity = [];
                Damage.crack_lc        = 0;
            end
        elseif ~use_crack_eov
            Damage.crack_locs      = [];
            Damage.crack_intensity = [];
            Damage.crack_lc        = 0;
        end   % per_state & j_pass>1: Damage.crack_* persists from j_pass==1
        if ~isempty(Damage.crack_locs)
            CrackLog(j_pass, :) = [Damage.crack_locs, Damage.crack_intensity, ...
                                   Damage.crack_lc];
        end
        % Rail profile: amplitude factor or FRA class + phase realization.
        Profile_cfg = struct('mode', profile_mode);
        if strcmp(profile_mode, 'fixed_scaled')
            Damage.profile_intensity = profile_int_range(1) + diff(profile_int_range)*rand();
            ProfileLog(j_pass) = Damage.profile_intensity;
        elseif strcmp(profile_mode, 'psd_fra')
            if strcmp(profile_draw, 'per_state')
                % ONE class + ONE phase realization per state: class drawn
                % from the state stream at j_pass==1; phases locked via a
                % per-state seed consumed inside B19 (which saves/restores
                % the passage stream); 0.5 mm additive jitter re-drawn per
                % passage in B19 (EN 13848-2 repeatability).
                if j_pass == 1
                    state_fra_class = profile_fra_classes(randi(numel(profile_fra_classes)));
                end
                Profile_cfg.fra_class   = state_fra_class;
                Profile_cfg.phase_seed  = 1e9 + damage_seed*100000 + DC;
                Profile_cfg.jitter_sd_m = profile_jitter_sd_mm / 1000;
            else   % DEPRECATED per-passage class + phase redraw
                Profile_cfg.fra_class = profile_fra_classes(randi(numel(profile_fra_classes)));
            end
            ProfileLog(j_pass) = Profile_cfg.fra_class;
        end
        % Track-layer damage descriptors (consumed by B54; see
        % docs/track_eov_sampling_spec.md). x-coords: bridge at
        % [track_L_app, track_L_app+L_bridge]; abutments = the transitions.
        % Persistent infrastructure condition -> drawn at j_pass==1 and HELD
        % when track_draw='per_state' (the Tk struct persists across the
        % passage loop); 'per_passage' redraw is DEPRECATED.
        if use_track_eov
          if strcmp(track_draw, 'per_passage') || j_pass == 1
            Tk = struct();
            % -- ballast fouling/degradation patches --
            % Sampled over the WHOLE modelled track, not just the bridge span
            % (fix 2026-07-14, user): the track continues before/after the
            % bridge and degrades there too — the approach/exit is exactly
            % where a drive-by vehicle is excited before it reaches the deck.
            track_win = track_L_app + L_bridge + track_L_after;   % modelled track [0, track_win] m
            % COUNT ~ Poisson(rate_per_100m * window/100): scales with the
            % modelled length (120 m at L60, 159.6 m at L99.6) instead of a
            % fixed draw, and admits the sound-track case naturally.
            np_ = poissrnd(ballast_rate_100m * track_win / 100);
            P_ = zeros(np_, 4);
            x_lo = 0;
            x_hi = track_win;
            % Fouled patches cluster at the bridge transitions (deck runoff into
            % the joint + impact at the stiffness discontinuity): density x
            % ballast_trans_mult within ballast_trans_margin of either abutment.
            % Rejection-sample the patch START against that density.
            abut_x = [track_L_app, track_L_app + L_bridge];
            for ip = 1:np_
                plen = ballast_patch_len(1) + diff(ballast_patch_len)*rand();
                % rejection-sample x0 against the transition-enriched density
                for try_ = 1:50
                    x0 = x_lo + (x_hi - x_lo - plen)*rand();
                    near_ab = any(abs((x0 + plen/2) - abut_x) <= ballast_trans_margin);
                    w_ = 1; if near_ab, w_ = ballast_trans_mult; end
                    if rand() <= w_ / ballast_trans_mult, break; end
                end
                if rand() < ballast_p_wet
                    ek = ballast_eta_k_wet(1) + diff(ballast_eta_k_wet)*rand();
                    ec = ballast_eta_c_wet(1) + diff(ballast_eta_c_wet)*rand();
                else
                    ek = ballast_eta_k_dry(1) + diff(ballast_eta_k_dry)*rand();
                    ec = ballast_eta_c_dry(1) + diff(ballast_eta_c_dry)*rand();
                end
                P_(ip,:) = [x0, x0 + plen, ek, ec];
            end
            % -- hanging/unsupported sleeper groups --
            % COUNT ~ Poisson(2.5 per 100 m * window/100) — anchored via the
            % impact-threshold derivation (7.5% of sleepers impactfully voided /
            % ~3 per group). Group SIZE stays DU(1,5) (cited: 1-4 m of contiguous
            % differential settlement at 0.6 m spacing = 2-6 sleepers).
            ng_ = poissrnd(hang_rate_100m * track_win / 100);
            H_ = zeros(ng_, 2);
            for ig = 1:ng_
                gsz = randi(hang_group_size);
                % Location: TWO composed priors — the cited transition spike, then
                % a rejection step enriching placement inside fouled patches
                % (fouling -> mud pumping -> loss of support; report x3 inside).
                for try_ = 1:50
                    if rand() < hang_p_transition   % density spike at transitions (cited)
                        trans_x = track_L_app + (rand() < 0.5)*L_bridge;
                        gx = trans_x - hang_trans_margin + 2*hang_trans_margin*rand();
                    else                             % anywhere on the MODELLED TRACK
                        % (fix 2026-07-14, user): was bridge-span only. Hanging
                        % sleepers are a TRACK defect - they occur on the
                        % approach/exit too, not just over the deck.
                        gx = rand()*track_win;
                    end
                    gx = max(gx, 0);
                    in_foul = false;
                    for ip = 1:size(P_, 1)
                        if gx >= P_(ip,1) && gx <= P_(ip,2), in_foul = true; break; end
                    end
                    w_ = 1/hang_foul_mult; if in_foul, w_ = hang_foul_mult; end
                    if rand() <= w_ / hang_foul_mult, break; end
                end
                H_(ig,:) = [gx, gsz];
            end
            % -- rail pads: global aging + sparse failures --
            chi_ = min(max(wblrnd(pad_weibull(1), pad_weibull(2)), ...
                pad_chi_range(1)), pad_chi_range(2));
            beta_ = pad_beta_range(1) + diff(pad_beta_range)*rand();
            win_ = track_win;                            % app+bridge+after [m]
            nf_ = sum(rand(1, round(win_/0.6)) < pad_p_fail);
            fx_ = sort(rand(1, nf_))*win_;
            Tk.ballast_patches = P_;   Tk.hanging_groups = H_;
            Tk.pad_stiff_mult  = chi_; Tk.pad_damp_mult  = beta_;
            Tk.pad_failures    = fx_;
          end   % per_state & j_pass>1: Tk persists from j_pass==1
            Damage.track = Tk;
            TrackLog{j_pass} = Tk;
        else
            Damage.track = [];
        end
        % Wheel flats + polygonization: literature-anchored draws (consumed by
        % B25). Wheels 1-2 = leading bogie (axles 1,2); 3-4 = trailing bogie.
        if use_oor_eov
            Fl_ = zeros(0, 5);          % [veh wheel length_m depth_m phase]
            Po_ = zeros(0, 5);          % [veh wheel order amp_m phase]
            for v_ = 1:Nveh
                for b_ = 0:1            % per-bogie slide events
                    if rand() < oor_q_bogie
                        ax_ = 2*b_ + 1;                    % leading axle flats
                        if rand() < oor_p_trailing
                            ax_(end+1) = 2*b_ + 2;         %#ok<AGROW> trailing too
                        end
                        for w_ = ax_
                            if rand() < oor_p_fresh        % fresh: sharp chord
                                lf_ = oor_len_fresh(1) + diff(oor_len_fresh)*rand();
                                df_ = lf_^2/(8*oor_radius);
                            else                            % run-in: filtered
                                lf_ = oor_len_runin(1) + diff(oor_len_runin)*rand();
                                df_ = lf_^2/(16*oor_radius);
                            end
                            Fl_(end+1,:) = [v_, w_, lf_, df_, 2*pi*rand()]; %#ok<AGROW>
                        end
                    end
                end
                for w_ = 1:4            % polygonization: independent per wheel
                    if rand() < poly_p_wheel
                        n_ = randi(poly_orders);
                        a_ = exp(poly_amp_lnorm(1) + poly_amp_lnorm(2)*randn());
                        a_ = min(max(a_, poly_amp_bounds(1)), poly_amp_bounds(2));
                        Po_(end+1,:) = [v_, w_, n_, a_, 2*pi*rand()]; %#ok<AGROW>
                    end
                end
            end
            Damage.oor_flats  = Fl_;
            Damage.oor_poly   = Po_;
            Damage.oor_radius = oor_radius;
            OORLog{j_pass} = struct('flats', Fl_, 'poly', Po_);
        else
            Damage.oor_flats = []; Damage.oor_poly = [];
        end
        % ------------------------------------------------------------------

        % Processing and Calculations
        [Calc_local, Beam_local, Track_local] = A04_Options(Beam_local, Track_local, Profile_cfg);
        [Sol_local, Calc_local, Train_local, Beam_local, Track_local] = B00_Calculations(Calc_local, Train_local, Track_local, Beam_local, Damage);

        % Data Processing
        data = D01_DataProcessing(1, j_pass, Sol_local, Train_local, Calc_local, Damage, data);

        fprintf('--- State %d/%d: scour=[%s], Pass %d/%d ---\n', ...
            DC, n_states, num2str(scour_vec, '%.2f '), j_pass, Npass);
    end

    % Store final data for this damage case
    data.Temperatura = Temperatura;
    data.Velocidade = Velocidade;
    data.VehiclesProps = x_veh;

    % RAW time-domain channels (Option B, 2026-07-14) — NOT interpolated, NOT
    % cropped, noise-free. Python rebuilds the space domain + bridge crop at load
    % time from the parameters saved just below (core/dataset._raw_to_space_crop).
    data2save.AcelPrimVag = data.AceleracaoPrimVag;
    data2save.AcelRodaPrimVag = data.AcelRodaPrimVag;
    data2save.PitchPrimVag = data.PitchPrimVag;
    % Space-transform + crop parameters, per passage (1 x Npass each). The
    % presence of DimSpace is what marks a file as RAW format for the loader.
    data2save.DimAcel      = data.DimAcel;
    data2save.DimSpace     = data.DimSpace;
    data2save.crop_start   = data.crop_start;
    data2save.crop_end     = data.crop_end;
    data2save.bridge_samp  = data.bridge_samp;
    data2save.L_bridge_eff = data.L_bridge_eff;
    % --- LABELS ---
    % Multi-output label = the full per-support scour-rate vector + which support
    % indices are the regression targets. Dano kept as the scalar AGGREGATE
    % (max scour) for backward compatibility with the single-scour loaders / DT.
    data2save.scour_vector = scour_vec;
    data2save.scour_supports = scour_supports;
    data2save.Dano = max(scour_vec);
    % Bearing label ([left,right] Nm/rad; zeros unless bearing_mode='target'/'fixed')
    data2save.bearing_vector = bear_vec;      % k_r [Nm/rad] (the physics)
    data2save.bearing_fixity = bear_fix;      % fixity in [0,1] (the regression LABEL)
    % --- NUISANCE-EOV TRACEABILITY (not labels) ---
    data2save.crack_log    = CrackLog;      % per passage: [loc_m, EI-loss, lc_m]
    data2save.profile_mode = profile_mode;
    data2save.profile_log  = ProfileLog;    % per passage: intensity or FRA class
    data2save.track_log    = TrackLog;      % per passage: Damage.track struct ([] if off)
    data2save.oor_log      = OORLog;        % per passage: [veh wheel len_m phase] rows
    data2save.Temperatura = Temperatura;
    data2save.Velocidade = Velocidade;
    data2save.VehiclesProps = x_veh;

    save_progress(data2save, DC, run_folder);
end

% Mark end time of the run
tempo_final = datetime('now', 'Format', 'dd_MM_yyyy_HH_mm_ss');
save(fullfile(run_folder, 'tempo_final.mat'), 'tempo_final');
tempo_total = tempo_final - tempo_inicial;
fprintf('Tempo total: %s\n', char(tempo_total));
save(fullfile(run_folder, 'tempo_total.mat'), 'tempo_total');
