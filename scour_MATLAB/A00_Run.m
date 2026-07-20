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
        % + TRAIN (wheel OOR) = ALL damages. Split from s15 so a drop is
        % attributable to the rail vs the train. NOTE (audit 2026-07-17):
        % use_oor_eov now delivers POLYGONIZATION ONLY - flats are disabled
        % (oor_flats_enabled=false below) until the solver supports the
        % separation/impact they cause.
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
dano_max         = 0.60;    % per-support max scour fraction
include_anchors  = true;    % prepend healthy + single-pier sweeps (localisation extremes)
n_anchor_levels  = 4;       % single-pier anchor levels per scoured support
damage_seed      = 1;       % RNG seed for a REPRODUCIBLE damage-state grid

% ===== Feature A (2026-07-19): explicit state FAMILIES =====================
% Every state carries a `state_family` identity tag written by the GENERATOR
% (not derived from labels downstream — with per-STATE persistent nuisances a
% zero-head state still carries a crack/profile/track fingerprint, so `y`
% alone cannot identify families; that was the flaw in the pre-redesign
% threshold-based probe). Families:
%   target_healthy  all heads zero; nuisances follow the rung's normal policy
%                   EXCEPT crack, which is forced OFF (clean baseline states).
%   scour_only      exactly ONE pier scoured (4 levels x n_anchor_reps);
%                   bearing zero, crack forced OFF (controlled probe).
%   bearing_only    exactly ONE abutment seized (4 levels x n_anchor_reps);
%                   scour zero, crack forced OFF (controlled probe).
%   nuisance_only   all heads zero, crack forced ON (the false-positive probe:
%                   does the model report damage when only a nuisance exists?).
%                   Only generated at rungs with use_crack_eov (crack is the
%                   only per-state nuisance worth controlling: the FRA profile
%                   class is FIXED at 4 — only its phase varies — and track
%                   damage is present w.p. ~1 under its Poisson rates).
%   joint           the LHS block; crack drawn naturally (p = crack_p).
% WHY: the stratified split (core/dataset.py) guarantees every family lands in
% train / inner-val / outer-test, so the disentanglement and false-positive
% probes can never be silently empty (audit R6 C8C loud-guard now a guarantee),
% and 1 healthy state becomes 12 independent ones (50 correlated passages of a
% single state are NOT 50 healthy states — Codex R7.1 correction).
n_healthy_states  = 12;     % target_healthy states (were: 1)
n_anchor_reps     = 2;      % replicas per (target, level) anchor — so every
                            % (family, target) stratum spans all 3 partitions
n_nuisance_states = 6;      % nuisance_only states (0 when use_crack_eov=false)

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
% (damaged-element EI reduction, cf. Fernandes et al. 2025; NOT Sinha's
% tapered model - mis-citation corrected in the 2026-07-17 audit).
crack_draw       = 'per_state';   % 'per_state' (persistent damage) | 'per_passage' (DEPRECATED)
crack_p          = 0.25;          % P(state carries a crack) — CONFIRMED 2026-07-15:
% deep research puts spans with a MACROSCOPIC EI loss in our 5-30% band at
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
% passages (profile_draw='per_state'; B19 seeds the phase draw per state).
% Class set FIXED at FRA class 4 = roughest geometry permissible at 70-90 km/h
% (classes 5/6 are premium track, too smooth for a scour-prone regional line).
% The old per-passage class+phase redraw over {4,5,6} is DEPRECATED (it is what
% collapsed the sprung channels in the L100 Stage-2 pilot).
profile_draw         = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
% AUDIT FIX 2026-07-17: physical profile jitter DISABLED (was 0.5). The 0.5 mm
% white noise misread EN 13848-2 - that figure is the MEASURING SYSTEM's
% repeatability, not physical track change between trains - and, being
% spectrally white on a profile band-limited at 1.524 m, it contributed ~zero
% in-band but injected 100% fictitious sub-1.5 m roughness (~125 g-equivalent
% axle forcing up to the solver Nyquist, loading exactly the wheel channels).
% Noise belongs to the OBSERVATION model: load-time sensor noise in
% core/dataset.py (docs/framework_rationale.md). If a physical pass-to-pass
% perturbation is ever wanted, it must be BAND-LIMITED (EN 13848 D1: 3-25 m,
% sigma ~ 0.18 mm per run), not white.
profile_jitter_sd_mm = 0;            % per-passage additive white noise [mm] - keep 0 (see above)
profile_int_range    = [0.5 2.0];    % 'fixed_scaled' amplitude range
profile_fra_classes  = 4;            % 'psd_fra' class set (scalar = fixed class)

% ===== Track-layer damage EOVs (Stage 3; use_track_eov) =====
% Verified sampling spec: docs/track_eov_sampling_spec.md (deep-research,
% quote-checked). Randomized NUISANCES — logged, NOT labels. EOV DESIGN REVIEW
% 2026-07-12: ballast patches / hanging sleepers / pad aging are PERSISTENT
% infrastructure conditions (same argument as crack/profile) -> drawn once per
% damage STATE (track_draw='per_state'). Wheel OOR stays per-PASSAGE: each
% passage plausibly is a different train of the fleet (vehicle variability).
% Track x-coords: descriptors are sampled in a BRIDGE-LOCAL WINDOW FRAME -
% x = 0 is the window start and the deck occupies [track_L_app,
% track_L_app + L_bridge]. B54 anchors this frame to the real bridge position
% via Tk.x_bridge_local (AUDIT FIX 2026-07-17: these coordinates were being
% consumed as GLOBAL track positions, but under redux=0 the global frame
% includes the max_TL ~ 106.8 m run-in, so the L60 deck actually starts at
% 123 m, NOT 30 m - every ballast patch / hanging group / pad failure landed
% on the approach and NONE could ever touch the deck or its transitions).
track_draw        = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
track_L_app       = 30;          % window margin BEFORE the deck [m] (local frame)
track_L_after     = 30;          % window margin AFTER the deck [m] (local frame)
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
% lambda ~ 2.0-3.0 for the intermediate degradation cycle.
% Ballast rate RE-ANCHORED 2026-07-19 (NotebookLM verification pass; full
% record in docs/research_brief_gpr_fouling.md): the old "FI>30 on 10-20% of
% route" was Slideshare-only and does NOT exist as a citable network constant.
% Citable envelope instead: Norway national defect-rate proxy ~9.4%
% (Husoy et al. 2024, Civil-Comp, DOI 10.4203/ccc.7.24.3); the only continuous
% regional-line GPR survey shows 66% at Selig FI>=20 and 12% HIGHLY fouled
% FI>=40 (Sadeghi et al. 2018, J. Appl. Geophys., DOI
% 10.1016/j.jappgeo.2018.02.020); US heavy-haul "majority fouled" (FRA 2017
% slide deck, flagged non-peer-reviewed). Our 15% of length / 12.5 m mean
% patch = 1.2 patches per 100 m sits INSIDE that envelope, close to the 12%
% highly-fouled fraction of the regional survey. Threshold semantics: patches
% represent ballast at/beyond the FI~30 functional DRAINAGE-LOSS limit =
% "highly fouled" in the Zetica/FRA GPR schema (FRA DOT/FRA/ORD-22/01 Table 5);
% Selig's own "highly fouled" boundary is FI>=40 (fouled = 20-40). Value KEPT
% at 1.2 (protocol unchanged); the sensitivity sweep lambda in {0.6, 1.2, 2.4}
% (= 7.5-30% of length) spans the citable envelope's plausible regional band.
% RATES are per 100 m and are SCALED BY THE MODELLED WINDOW below (our track is
% 120 m at L60 / 159.6 m at L99.6 - drawing a fixed count regardless of length
% was itself an error).
% ⚠ Report-internal inconsistency, resolved in our favour of its own derivation:
% its summary table claims P(no hanging group)~0.25 while its text recommends
% lambda 2-3 (which gives P(0)=0.05-0.14). We follow the DERIVATION (lambda=2.5).
% ✅ VERIFIED 2026-07-19: the "FI>30 on 10-20%" Slideshare anchor was replaced
% by the citable envelope above (see docs/research_brief_gpr_fouling.md).
% Patch length U(5,20) m also now CITED: FRA empirical fouled-section lengths
% 1.5-33.5 m (DOT/FRA/ORD-22/01) + 2.4 m GPR minimum feature (Guo et al. 2023,
% Int. J. Rail Transp., DOI 10.1080/23248378.2022.2064346).
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
% ---- Wheel FLATS disabled pending a valid impact model (audit 2026-07-17) --
% Every flat in the sampled range exceeds the wheel-uplift threshold by
% 12-38x: kinematic |hdd| at the haversine dip = pi^2*v^2/(4R) (length-
% independent under d = L^2/8R) = 2000-3350 m/s^2 at 70-90 km/h, vs the
% separation threshold F_stat/m_wheel ~ 88 m/s^2. The TTB-2D solver couples
% wheel and rail BILATERALLY (no separation state), so it rides TENSION
% through the dip; and at dt = 1 ms the train advances 19-25 mm per step vs
% 10-60 mm flats = 0.4-2.4 samples per flat (impact content 324-2500 Hz vs
% the 500 Hz solver accuracy) - the impulse is aliased on top of being
% bilateral-invalid. Do NOT re-enable until the solver has: unilateral (or
% Hertzian) wheel-rail contact with gap opening + re-contact, a timestep
% refined enough for the impact, and anti-aliased downsampling of the
% result (design sketch: docs/framework_rationale.md, 2026-07-17 entry).
% POLYGONIZATION STAYS ON: its worst case (~0.9 g) is ~10x BELOW the uplift
% threshold - a valid permanent-contact excitation.
oor_flats_enabled = false;       % KEEP FALSE (see above); poly unaffected
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
    % Family table for the legacy single-scour mode (kept well-defined so the
    % r8 artifacts are uniform; the ladder never uses this mode). File 1 is the
    % zero-scour state; the rest are single-pier sweeps of the central pier.
    StateFamily  = [{'target_healthy'}; repmat({'scour_only'}, numel(Dano) - 1, 1)];
    AnchorTarget = [0; repmat(central, numel(Dano) - 1, 1)];
    AnchorLevel  = [0; (1:numel(Dano) - 1)'];
else  % multi_scour
    rng(damage_seed);                         % reproducible joint grid
    n_tgt = numel(scour_supports);
    % Feature A (2026-07-19): anchors are built WITH their family identity —
    % four aligned accumulators (scour rows, bearing-fixity rows, family tag,
    % anchor target + level). See the family definitions at the config block.
    anchors_s  = zeros(0, n_supp);            % per-support scour rows
    anchors_bf = zeros(0, 2);                 % bearing FIXITY rows (the label)
    fam_  = cell(0, 1);                       % state_family tag per anchor row
    atgt_ = zeros(0, 1);                      % scour: pier idx | bearing: 1=L,2=R
    alvl_ = zeros(0, 1);                      % severity level idx (1..n_anchor_levels)
    n_nuis_here = n_nuisance_states * double(use_crack_eov);   % crack rungs only
    if include_anchors
        % (1) target_healthy — 12 independent all-zero-head states. Each gets
        % its OWN per-state EOV draws (speed/temp/vehicle + profile/track), so
        % "healthy" spans train/val/test as genuinely distinct states.
        anchors_s  = [anchors_s;  zeros(n_healthy_states, n_supp)]; %#ok<AGROW>
        anchors_bf = [anchors_bf; zeros(n_healthy_states, 2)];      %#ok<AGROW>
        fam_  = [fam_;  repmat({'target_healthy'}, n_healthy_states, 1)]; %#ok<AGROW>
        atgt_ = [atgt_; zeros(n_healthy_states, 1)];                %#ok<AGROW>
        alvl_ = [alvl_; zeros(n_healthy_states, 1)];                %#ok<AGROW>
        % (2) scour_only — per target pier: n_anchor_levels severities x
        % n_anchor_reps replicas (replication is what lets the stratified split
        % place every (family, pier) stratum in all three partitions).
        levels_s = linspace(dano_max / n_anchor_levels, dano_max, n_anchor_levels)';
        for ti = 1:n_tgt
            for rep_ = 1:n_anchor_reps
                blk = zeros(n_anchor_levels, n_supp);
                blk(:, scour_supports(ti)) = levels_s;   % ONE pier damaged
                anchors_s  = [anchors_s; blk];                         %#ok<AGROW>
                anchors_bf = [anchors_bf; zeros(n_anchor_levels, 2)];  %#ok<AGROW>
                fam_  = [fam_;  repmat({'scour_only'}, n_anchor_levels, 1)]; %#ok<AGROW>
                atgt_ = [atgt_; repmat(scour_supports(ti), n_anchor_levels, 1)]; %#ok<AGROW>
                alvl_ = [alvl_; (1:n_anchor_levels)'];                 %#ok<AGROW>
            end
        end
        % (3) bearing_only — same structure per abutment (1=left, 2=right).
        if n_bear > 0
            levels_b = linspace(bearing_fixity_max / n_anchor_levels, ...
                                bearing_fixity_max, n_anchor_levels)';
            for bi = 1:2
                for rep_ = 1:n_anchor_reps
                    blk = zeros(n_anchor_levels, 2);
                    blk(:, bi) = levels_b;
                    anchors_bf = [anchors_bf; blk];                        %#ok<AGROW>
                    anchors_s  = [anchors_s; zeros(n_anchor_levels, n_supp)]; %#ok<AGROW>
                    fam_  = [fam_;  repmat({'bearing_only'}, n_anchor_levels, 1)]; %#ok<AGROW>
                    atgt_ = [atgt_; repmat(bi, n_anchor_levels, 1)];       %#ok<AGROW>
                    alvl_ = [alvl_; (1:n_anchor_levels)'];                 %#ok<AGROW>
                end
            end
        end
        % (4) nuisance_only — all heads zero, crack FORCED ON (see the CrackOn
        % pre-draw below). Zero states at rungs without the crack EOV.
        if n_nuis_here > 0
            anchors_s  = [anchors_s;  zeros(n_nuis_here, n_supp)]; %#ok<AGROW>
            anchors_bf = [anchors_bf; zeros(n_nuis_here, 2)];      %#ok<AGROW>
            fam_  = [fam_;  repmat({'nuisance_only'}, n_nuis_here, 1)]; %#ok<AGROW>
            atgt_ = [atgt_; zeros(n_nuis_here, 1)];                %#ok<AGROW>
            alvl_ = [alvl_; zeros(n_nuis_here, 1)];                %#ok<AGROW>
        end
    end
    % (5) ONE joint LHS over scour + bearing targets = broad joint coverage.
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
    StateFamily  = [fam_;  repmat({'joint'}, n_states_multi, 1)];
    AnchorTarget = [atgt_; zeros(n_states_multi, 1)];
    AnchorLevel  = [alvl_; zeros(n_states_multi, 1)];
end
n_states = size(DamageStates, 1);
% Family table must line up with the damage matrix ROW FOR ROW — a mismatch
% here would silently mislabel every downstream probe, so hard-assert it.
if numel(StateFamily) ~= n_states || numel(AnchorTarget) ~= n_states || ...
        numel(AnchorLevel) ~= n_states
    error('A00: state-family table (%d/%d/%d) does not match n_states=%d.', ...
        numel(StateFamily), numel(AnchorTarget), numel(AnchorLevel), n_states);
end

% ---- Per-state crack ACTIVATION, pre-drawn (Feature A, 2026-07-19) --------
% Drawn OUTSIDE the parfor from its own seeded stream so that (a) the family
% design can FORCE it (OFF for target_healthy / scour_only / bearing_only —
% they are controlled probes; ON for nuisance_only — the false-positive probe),
% (b) it is part of the fingerprint + the Python stratification table, and
% (c) the joint block keeps the natural prevalence p = crack_p. The crack's
% LOCATION / intensity are still drawn from the per-state stream inside the
% parfor (only the on/off decision moved out).
rng(damage_seed + 424243);              % independent stream: activation only
CrackOn = false(n_states, 1);
if use_crack_eov && strcmp(crack_draw, 'per_state')
    is_joint_ = strcmp(StateFamily, 'joint');
    CrackOn(is_joint_) = rand(sum(is_joint_), 1) <= crack_p;
    CrackOn(strcmp(StateFamily, 'nuisance_only')) = true;
elseif use_crack_eov && any(strcmp(StateFamily, 'nuisance_only'))
    % Guard (2026-07-19 self-review): the DEPRECATED per-passage crack redraw
    % ignores CrackOn, so nuisance_only states would NOT get their forced crack
    % and the family semantics would silently break. Refuse the combination.
    error(['A00: nuisance_only anchor states require crack_draw=''per_state'' ' ...
        '(the family design forces per-state crack activation); ' ...
        '''per_passage'' is deprecated and incompatible with Feature A.']);
end

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

% ---- Generator schema tag (audit 2026-07-17) ----------------------------
% Bump this string whenever the generator physics or the saved format changes.
% It is written to the manifest AND into every NNNN.mat, and the resume guard
% below REFUSES to add states to a folder written by a different schema — so
% extracting a new bundle over an old Results/ folder aborts loudly instead of
% silently mixing code versions (the pre-audit 2026-07-16 data is a different
% schema and must not be resumed).
% Bumped R5 (2026-07-17): fixed-profile mapping is now REFLECT (no wavelength
% rescale) and the Python side added a 3-way split; R2/R3/R4 data is a
% DIFFERENT physical realization / protocol and must be rejected by the R5
% loader. _EXPECTED_GEN_SCHEMA in core/dataset.py mirrors this exact string.
% Bumped R6 (2026-07-18, reviewer-hardening pass): fingerprint now folds in the
% rail-profile asset bytes (C7), A00 hard-fails on wheel-rail tension loss to
% match the Python gate (C6), resume validates the payload not just provenance
% (C5), and the Python loader enforces exact passage counts + label/signal
% finiteness (C4).
% Bumped R7 (2026-07-18, Codex re-review of R6): PER-PASSAGE contact abort with a
% text diagnostic (fixes the parfor `save '-struct'` PFSV error and adds a
% non-finite gate), stricter resume payload + marker validation, dano_max in the
% manifest, plus Python-side strict test-once selection, all-arch completeness,
% and cache-artifact digests. No ladder data exists yet, so the bump is free.
% Bumped R8 (2026-07-19, close-out of the R7.1 queue — Codex retired, in-house
% adversarial review): source per-state SHA + root digest in the marker, FULL
% resume payload validation (shapes/finiteness/ranges/labels==DamageStates row),
% Feature A state FAMILIES (target_healthy x12 / scour_only / bearing_only /
% nuisance_only / joint + pre-drawn CrackOn; anchors 4 levels x 2 reps) with the
% family table in damage_states.mat + per-file state_family, and the Python-side
% unified protocol hash + family-stratified split + deployment selection.
% State COUNTS changed (s0=278 ... s22/s23=308) — r7 data is a different design.
% _EXPECTED_GEN_SCHEMA in core/dataset.py mirrors this string.
gen_schema = 'audit-2026-07-19-r8';

% ---- Config fingerprint (audit R5 2026-07-17: CANONICAL HASH) -----------
% A schema match alone does not prove the CONFIG matches. Instead of a
% hand-maintained sprintf (which silently drifts as knobs are added), build a
% struct holding EVERY run-defining knob PLUS the actual damage matrices, and
% SHA-256 the canonical JSON. Any change to any field changes the hash. Stored
% in the manifest AND in every NNNN.mat; the resume guard and the Python loader
% both compare against it. Fields listed explicitly so the JSON order is
% deterministic across MATLAB versions.
fp_cfg = struct( ...
    'schema', gen_schema, 'STAGE', STAGE, 'damage_mode', damage_mode, ...
    'L_bridge', L_bridge, 'num_spans', num_spans, ...
    'scour_supports', scour_supports, 'n_states', n_states, 'Npass', Npass, ...
    'dano_max', dano_max, 'include_anchors', include_anchors, ...
    'n_anchor_levels', n_anchor_levels, 'damage_seed', damage_seed, ...
    'n_healthy_states', n_healthy_states, 'n_anchor_reps', n_anchor_reps, ...
    'n_nuisance_states', n_nuisance_states, ...   % Feature A family knobs
    'bearing_mode', bearing_mode, 'Bearing_Intensity', Bearing_Intensity, ...
    'bearing_fixity_max', bearing_fixity_max, ...
    'use_crack_eov', use_crack_eov, 'crack_draw', crack_draw, ...
    'crack_p', crack_p, 'crack_hog_ratio', crack_hog_ratio, ...
    'crack_hog_margin', crack_hog_margin, 'crack_frac_range', crack_frac_range, ...
    'crack_int_range', crack_int_range, 'crack_lc', crack_lc, ...
    'profile_mode', profile_mode, 'profile_draw', profile_draw, ...
    'profile_jitter_sd_mm', profile_jitter_sd_mm, ...
    'profile_int_range', profile_int_range, ...
    'profile_fra_classes', profile_fra_classes, ...
    'use_track_eov', use_track_eov, 'track_draw', track_draw, ...
    'track_L_app', track_L_app, 'track_L_after', track_L_after, ...
    'hang_rate_100m', hang_rate_100m, 'hang_group_size', hang_group_size, ...
    'hang_p_transition', hang_p_transition, 'hang_trans_margin', hang_trans_margin, ...
    'hang_foul_mult', hang_foul_mult, 'ballast_rate_100m', ballast_rate_100m, ...
    'ballast_patch_len', ballast_patch_len, 'ballast_trans_mult', ballast_trans_mult, ...
    'ballast_trans_margin', ballast_trans_margin, 'ballast_p_wet', ballast_p_wet, ...
    'ballast_eta_k_dry', ballast_eta_k_dry, 'ballast_eta_c_dry', ballast_eta_c_dry, ...
    'ballast_eta_k_wet', ballast_eta_k_wet, 'ballast_eta_c_wet', ballast_eta_c_wet, ...
    'pad_p_fail', pad_p_fail, 'pad_chi_range', pad_chi_range, ...
    'pad_weibull', pad_weibull, 'pad_beta_range', pad_beta_range, ...
    'use_oor_eov', use_oor_eov, 'oor_flats_enabled', oor_flats_enabled, ...
    'oor_q_bogie', oor_q_bogie, 'oor_p_trailing', oor_p_trailing, ...
    'oor_p_fresh', oor_p_fresh, 'oor_len_fresh', oor_len_fresh, ...
    'oor_len_runin', oor_len_runin, 'oor_radius', oor_radius, ...
    'poly_p_wheel', poly_p_wheel, 'poly_orders', poly_orders, ...
    'poly_amp_lnorm', poly_amp_lnorm, 'poly_amp_bounds', poly_amp_bounds, ...
    'use_signal_noise', use_signal_noise, 'Desvio', Desvio, ...
    'use_vehicle_variability', use_vehicle_variability, ...
    'use_speed_variability', use_speed_variability, ...
    'use_temp_variability', use_temp_variability, ...
    'Nveh', Nveh, 'Nprop', Nprop, 'temp_min', temp_min, 'temp_max', temp_max, ...
    'vel_min', vel_min, 'vel_max', vel_max);
% Include the actual damage realization matrices (captures the exact per-state
% draws, not just the knobs that seed them).
fp_cfg.DamageStates  = DamageStates;
fp_cfg.BearingStates = BearingStates;
fp_cfg.BearingFixity = BearingFixity;
% Feature A (2026-07-19): the family table + pre-drawn crack activation are
% run-defining too — a changed family layout or crack draw must change the
% fingerprint (the Python stratified split consumes exactly these arrays).
fp_cfg.StateFamily   = StateFamily;
fp_cfg.AnchorTarget  = AnchorTarget;
fp_cfg.AnchorLevel   = AnchorLevel;
fp_cfg.CrackOn       = CrackOn;
% Profile ASSET provenance (audit R6 C7): the fixed-profile stages (Type==2,
% profile_mode 'fixed'/'fixed_scaled') read the rail irregularity from an
% external .mat (B19_GenerateProfile.m:118, load('Calc.ProfileData15_05.mat')).
% Only the string knob profile_mode enters fp_cfg above, NOT the asset's bytes,
% so swapping the realization under the same filename would change every
% generated signal while leaving the fingerprint bit-identical. Fold a SHA-256
% of the asset bytes in to close that hole (one ~350 KB read, once per run).
% Guarded by exist() so a psd_fra-only checkout without the asset does not error;
% 'ABSENT' is itself recorded so a fixed-profile run that lost the asset differs
% from one that has it. The B19 load uses a bare filename (cwd), so we resolve it
% the same way, preferring the scour_MATLAB copy next to this script.
profile_asset_ = 'Calc.ProfileData15_05.mat';
if ~exist(profile_asset_, 'file')
    profile_asset_ = fullfile(fileparts(mfilename('fullpath')), 'Calc.ProfileData15_05.mat');
end
if exist(profile_asset_, 'file')
    fp_cfg.profile_asset_sha256 = local_file_sha256(profile_asset_);
    fprintf('[fingerprint] profile asset %s -> %s\n', profile_asset_, ...
        fp_cfg.profile_asset_sha256);
else
    fp_cfg.profile_asset_sha256 = 'ABSENT';
end
gen_fingerprint = local_sha256(jsonencode(fp_cfg));

% ---- Resume-safety guard: never mix schemas OR configs in one folder -----
existing_ci_ = fullfile(run_folder, 'case_info.mat');
if exist(existing_ci_, 'file')
    prev_ = load(existing_ci_); prev_ci_ = prev_.case_info;
    prev_schema_ = ''; prev_fp_ = '';
    if isfield(prev_ci_, 'gen_schema'),      prev_schema_ = prev_ci_.gen_schema; end
    if isfield(prev_ci_, 'gen_fingerprint'), prev_fp_ = prev_ci_.gen_fingerprint; end
    if ~strcmp(prev_schema_, gen_schema)
        error(['A00 RESUME ABORTED: folder "%s" holds data from generator ' ...
            'schema "%s", but this code is "%s". Move or delete the old ' ...
            'folder (do NOT extract a new bundle over an old run). See the ' ...
            '2026-07-17 audit entry in docs/framework_rationale.md.'], ...
            run_folder, prev_schema_, gen_schema);
    end
    if ~strcmp(prev_fp_, gen_fingerprint)
        error(['A00 RESUME ABORTED: folder "%s" was generated with a DIFFERENT ' ...
            'configuration.\n  existing: %s\n  current : %s\nUse a fresh ' ...
            'folder.'], run_folder, prev_fp_, gen_fingerprint);
    end
end

% --- Manifest: machine-readable (.mat) + human-readable (.txt) -----------
case_info = struct( ...
    'case_name', case_name, 'case_desc', case_desc, 'stage', STAGE, ...
    'gen_schema', gen_schema, ...
    'gen_fingerprint', gen_fingerprint, ...
    'oor_flats_enabled', oor_flats_enabled, ...
    'timestamp', tempo_inicial_str, ...
    'damage_mode', damage_mode, ...
    'L_bridge_m', L_bridge, 'num_spans', num_spans, ...
    'num_supports', n_supp, 'scour_supports', mat2str(scour_supports), ...
    'n_states', n_states, 'passages_per_state', Npass, ...
    'scour_dano_max_frac', dano_max, ...
    'n_target_healthy', sum(strcmp(StateFamily, 'target_healthy')), ...  % Feature A
    'n_scour_only',     sum(strcmp(StateFamily, 'scour_only')), ...
    'n_bearing_only',   sum(strcmp(StateFamily, 'bearing_only')), ...
    'n_nuisance_only',  sum(strcmp(StateFamily, 'nuisance_only')), ...
    'n_joint',          sum(strcmp(StateFamily, 'joint')), ...
    'n_anchor_reps',    n_anchor_reps, ...
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
% Feature A (2026-07-19): the family table + crack activation ship with the
% damage matrices — core/dataset.read_state_table() consumes them for the
% stratified split and the family-identity disentanglement probe.
save(fullfile(run_folder, 'damage_states.mat'), 'DamageStates', 'BearingStates', ...
    'BearingFixity', 'k_ref_bear', 'scour_supports', ...
    'StateFamily', 'AnchorTarget', 'AnchorLevel', 'CrackOn');
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

% Identify which states are already DONE (resume mode). AUDIT R5: do not trust
% the filename alone — a state counts as complete ONLY if its file carries THIS
% run's schema AND fingerprint (validated cheaply via the top-level provenance
% variables, no need to load the big signal arrays). A file with a DIFFERENT
% schema/fingerprint (foreign/old/copied) aborts the run; a file lacking the
% provenance vars (pre-R5 format) also aborts. This is what makes resume safe.
%
% Clear any stale completion marker FIRST (audit R7 P5): delete it BEFORE the scan
% so that even if the scan aborts on a bad file, an old "complete" marker can
% never survive. It is (re-)written only if THIS run reaches full completion.
marker_path = fullfile(run_folder, '_GENERATION_COMPLETE');
if exist(marker_path, 'file'), delete(marker_path); end
% Contact hard-gate tolerances, == core/dataset.py _CONTACT_F_TOL_N /
% _CONTACT_FRAC_TOL. Defined at TOP LEVEL so BOTH gates read one value: the
% resume validation below and the per-passage gate inside the parfor.
% RECALIBRATED 2026-07-19 (first campaign dispatch): the zero-tolerance gate
% (1000 N) was calibrated on HEALTHY-bridge smokes and aborted s23_all4 at
% state 24 (scour_only anchor, 60% at the middle pier, L99.6 + FRA-4 + track
% damage + poly OOR) on a 6.4 kN tension spike lasting 0.042% of the path —
% brief wheel unloading past zero, which the BILATERAL solver renders as small
% spurious tension instead of a few-ms separation + re-contact. That is a
% physically plausible superposition tail on the severe-EOV rungs, and
% censoring/resampling it would MNAR-bias exactly the severe states the paper
% studies. TWO-TIER rule now:
%   TOLERATED + logged: peak tension <= 10% of the ~118 kN static wheel load
%     AND tension on <= 0.2% of path samples (brief micro-unloading).
%   FATAL (physics regression): beyond either bound, or non-finite. The known
%     true regressions are far above: R3 profile-seam bug 170 kN (144% of
%     static); wheel flats exceed the uplift threshold 12-38x.
% NOT in fp_cfg (deliberate): a stricter-gate dataset is a strict subset of
% what this gate accepts, so pre-recalibration states remain valid and a
% partial run RESUMES seamlessly under the patched generator.
F_CONTACT_TOL_N   = 12000;   % FATAL above this peak tension [N] (10% static)
F_CONTACT_FRACTOL = 0.002;   % FATAL above this fraction of path samples in tension
saved_files = dir(fullfile(run_folder, '*.mat'));
completed = false(n_states, 1);
for k = 1:length(saved_files)
    tokens = regexp(saved_files(k).name, '^(\d{4})\.mat$', 'tokens');
    if isempty(tokens), continue; end
    dc_idx = str2double(tokens{1}{1});
    if dc_idx < 1 || dc_idx > n_states, continue; end
    fpath_ = fullfile(run_folder, saved_files(k).name);
    vinfo_ = whos('-file', fpath_);            % variable list only (cheap)
    vnames_ = {vinfo_.name};
    if ~all(ismember({'file_gen_schema','file_gen_fingerprint'}, vnames_))
        error(['A00 RESUME ABORTED: state %d file "%s" lacks provenance vars ' ...
            '(pre-R5 / foreign format). Use a FRESH folder.'], dc_idx, saved_files(k).name);
    end
    S_ = load(fpath_, 'file_gen_schema', 'file_gen_fingerprint');
    if ~strcmp(S_.file_gen_schema, gen_schema) || ...
            ~strcmp(S_.file_gen_fingerprint, gen_fingerprint)
        error(['A00 RESUME ABORTED: state %d file "%s" has a DIFFERENT ' ...
            'schema/fingerprint than this run (foreign/old/mixed). Use a ' ...
            'FRESH folder — never merge states from different runs.'], ...
            dc_idx, saved_files(k).name);
    end
    % Payload validation (audit R6 C5): matching provenance strings do NOT prove
    % the file holds real signals. A correctly-stamped but corrupt/garbage `data`
    % would otherwise be silently SKIPPED on every resume (its provenance matches
    % so it is never regenerated), the completion marker would still be written,
    % and the failure would only surface — unrepairably — at Python load time.
    % Verify the payload struct and its essential fields NOW so a bad state is
    % caught at generation time (delete it or use a fresh folder). Runs only on
    % resume (per already-present file), so the load cost is a one-off at startup.
    if ~ismember('data', vnames_)
        error(['A00 RESUME ABORTED: state %d file "%s" carries provenance vars ' ...
            'but NO `data` payload (corrupt/foreign). Delete it or use a FRESH folder.'], ...
            dc_idx, saved_files(k).name);
    end
    D_ = load(fpath_, 'data'); d_ = D_.data;
    % Required payload fields (audit R7.1 item 2, 2026-07-19): now includes the
    % bearing k_r vector and ALL SIX RAW-format fields (Option B) — a file
    % missing any of them would be rejected by the Python loader anyway, so
    % catch it HERE where it can still be regenerated cheaply.
    req_ = {'AcelPrimVag','AcelRodaPrimVag','PitchPrimVag','scour_vector', ...
            'bearing_fixity','bearing_vector','scour_supports','state_family', ...
            'contact_log','gen_schema','gen_fingerprint', ...
            'DimAcel','DimSpace','crop_start','crop_end','bridge_samp','L_bridge_eff'};
    if ~isstruct(d_) || ~all(isfield(d_, req_))
        miss_ = req_(~isfield(d_, req_));
        error(['A00 RESUME ABORTED: state %d file "%s" is missing payload field(s) ' ...
            '{%s} — corrupt/partial write. Delete it or use a FRESH folder.'], ...
            dc_idx, saved_files(k).name, strjoin(miss_, ', '));
    end
    % Internal provenance must match too (audit R7 P5): the payload's OWN
    % gen_schema/gen_fingerprint (not just the cheap top-level vars) must equal
    % this run's — catches a file whose top-level stamp was doctored.
    if ~strcmp(char(d_.gen_schema), gen_schema) || ...
            ~strcmp(char(d_.gen_fingerprint), gen_fingerprint)
        error(['A00 RESUME ABORTED: state %d file "%s" data.gen_schema/fingerprint ' ...
            'disagree with this run (doctored/mixed). Use a FRESH folder.'], ...
            dc_idx, saved_files(k).name);
    end
    % EVERY signal channel + contact_log must have exactly Npass passages (audit
    % R7 P5) — not just AcelPrimVag; a short wheel/pitch/contact array is a partial
    % write the Python loader would later reject.
    if size(d_.AcelPrimVag,2) ~= Npass || size(d_.AcelRodaPrimVag,2) ~= Npass || ...
            size(d_.PitchPrimVag,2) ~= Npass || size(d_.contact_log,1) ~= Npass
        error(['A00 RESUME ABORTED: state %d file "%s" channel/contact counts ' ...
            '(Acel %d, Roda %d, Pitch %d, contact rows %d) != Npass=%d — partial ' ...
            'write. Delete it or use a FRESH folder.'], dc_idx, saved_files(k).name, ...
            size(d_.AcelPrimVag,2), size(d_.AcelRodaPrimVag,2), ...
            size(d_.PitchPrimVag,2), size(d_.contact_log,1), Npass);
    end
    % =====================================================================
    % FULL payload validation (audit R7.1 item 2, 2026-07-19). Counts alone
    % do not prove the payload is REAL: a file could carry Npass-sized arrays
    % of garbage (NaN signals, wrong-state labels after a manual copy, a
    % tension-laden contact_log) and be skipped forever on every resume while
    % the completion marker certifies the folder. Everything below runs ONCE
    % per already-present file at startup (data is already fully loaded), so
    % the cost is negligible next to regenerating even one state.
    %
    % (a) contact_log must be EXACTLY Npass x 4, all finite, and within the
    %     same hard gate the generation loop enforces (col 4 tension <= tol) —
    %     a resumed file must meet the same physics bar as a fresh one.
    if ~isequal(size(d_.contact_log), [Npass, 4]) || ~all(isfinite(d_.contact_log(:)))
        error(['A00 RESUME ABORTED: state %d file "%s" contact_log is not a ' ...
            'finite %dx4 matrix (got %s) — corrupt. Delete it or use a FRESH ' ...
            'folder.'], dc_idx, saved_files(k).name, Npass, mat2str(size(d_.contact_log)));
    end
    if any(d_.contact_log(:,4) > F_CONTACT_TOL_N) || ...
            any(d_.contact_log(:,3) > F_CONTACT_FRACTOL)
        error(['A00 RESUME ABORTED: state %d file "%s" contact_log exceeds the ' ...
            'two-tier gate (peak tension %.4g N vs tol %g N; worst path ' ...
            'fraction %.3g vs tol %g) — this state would be hard-rejected by ' ...
            'the loader. Regenerate it (delete the file).'], dc_idx, ...
            saved_files(k).name, max(d_.contact_log(:,4)), F_CONTACT_TOL_N, ...
            max(d_.contact_log(:,3)), F_CONTACT_FRACTOL);
    end
    % (b) LABELS must equal THIS run's damage-state row bit-for-bit. The
    %     fingerprint already guarantees the run CONFIG matches (DamageStates/
    %     BearingFixity are inside fp_cfg), so the stored labels must equal the
    %     recomputed rows exactly; any difference means the label block of the
    %     file is corrupt or belongs to another state (e.g. a renamed NNNN.mat).
    %     NOTE isequal() is false for NaN, so non-finite labels also abort here.
    if ~isequal(d_.scour_vector, DamageStates(dc_idx, :))
        error(['A00 RESUME ABORTED: state %d file "%s" scour_vector %s != this ' ...
            'run''s DamageStates row %s — mislabelled/corrupt state file.'], ...
            dc_idx, saved_files(k).name, mat2str(d_.scour_vector, 6), ...
            mat2str(DamageStates(dc_idx, :), 6));
    end
    if ~isequal(d_.bearing_fixity, BearingFixity(dc_idx, :)) || ...
            ~isequal(d_.bearing_vector, BearingStates(dc_idx, :))
        error(['A00 RESUME ABORTED: state %d file "%s" bearing label/physics ' ...
            '(fixity %s, k_r %s) != this run''s rows (fixity %s, k_r %s) — ' ...
            'mislabelled/corrupt state file.'], dc_idx, saved_files(k).name, ...
            mat2str(d_.bearing_fixity, 6), mat2str(d_.bearing_vector, 6), ...
            mat2str(BearingFixity(dc_idx, :), 6), mat2str(BearingStates(dc_idx, :), 6));
    end
    if ~isequal(d_.scour_supports, scour_supports)
        error(['A00 RESUME ABORTED: state %d file "%s" scour_supports %s != ' ...
            'this run''s %s — foreign target layout.'], dc_idx, ...
            saved_files(k).name, mat2str(d_.scour_supports), mat2str(scour_supports));
    end
    % Family identity must match this run's table (Feature A, 2026-07-19): a
    % renamed NNNN.mat carrying another state's payload matches provenance but
    % NOT its family/labels — both checks catch it independently.
    if ~strcmp(char(d_.state_family), StateFamily{dc_idx})
        error(['A00 RESUME ABORTED: state %d file "%s" state_family "%s" != ' ...
            'this run''s "%s" — mislabelled/foreign state file.'], dc_idx, ...
            saved_files(k).name, char(d_.state_family), StateFamily{dc_idx});
    end
    % (c) Range sanity (defence-in-depth; equality above already implies this
    %     when DamageStates itself is sane — this also guards THAT assumption):
    %     scour is a fraction in [0, dano_max]; fixity is in [0, 1) by
    %     construction (phi = k/(k+4EI/L) < 1 for any finite k).
    if any(d_.scour_vector < 0) || any(d_.scour_vector > dano_max + 1e-9)
        error(['A00 RESUME ABORTED: state %d file "%s" scour_vector %s outside ' ...
            '[0, dano_max=%g].'], dc_idx, saved_files(k).name, ...
            mat2str(d_.scour_vector, 6), dano_max);
    end
    if any(d_.bearing_fixity < 0) || any(d_.bearing_fixity >= 1)
        error(['A00 RESUME ABORTED: state %d file "%s" bearing_fixity %s outside ' ...
            '[0, 1).'], dc_idx, saved_files(k).name, mat2str(d_.bearing_fixity, 6));
    end
    % (d) RAW space-transform metadata (Option B): each of the six fields must
    %     be a finite 1 x Npass row; the dim/crop fields must be INTEGRAL and
    %     the crop window must satisfy 1 <= crop_start <= crop_end <= DimSpace
    %     per passage (the loader enforces the same contract, audit R7.1 P6).
    rawf_ = {'DimAcel','DimSpace','crop_start','crop_end','bridge_samp','L_bridge_eff'};
    for rf_ = 1:numel(rawf_)
        v_ = d_.(rawf_{rf_});
        if ~isequal(size(v_), [1, Npass]) || ~all(isfinite(v_))
            error(['A00 RESUME ABORTED: state %d file "%s" RAW field %s is not ' ...
                'a finite 1x%d row (got %s) — partial/corrupt write.'], dc_idx, ...
                saved_files(k).name, rawf_{rf_}, Npass, mat2str(size(v_)));
        end
        if ~strcmp(rawf_{rf_}, 'L_bridge_eff') && any(v_ ~= round(v_))
            error(['A00 RESUME ABORTED: state %d file "%s" RAW field %s has ' ...
                'non-integer entries — corrupt space/crop metadata.'], dc_idx, ...
                saved_files(k).name, rawf_{rf_});
        end
    end
    if any(d_.crop_start < 1) || any(d_.crop_start > d_.crop_end) || ...
            any(d_.crop_end > d_.DimSpace)
        error(['A00 RESUME ABORTED: state %d file "%s" crop window violates ' ...
            '1 <= crop_start <= crop_end <= DimSpace for some passage.'], ...
            dc_idx, saved_files(k).name);
    end
    % (e) SIGNALS: every passage's cell must be a finite 2-D matrix whose time
    %     axis matches DimAcel(j) and whose row count is constant across the
    %     state (rows are vehicle-model-specific: 3 carbody / 4 wheel / 3 pitch
    %     today, but the invariant checked is consistency + the loader's needs).
    sigf_ = {'AcelPrimVag','AcelRodaPrimVag','PitchPrimVag'};
    minrows_ = [3, 2, 3];      % highest channel row the Python loader indexes
    for sf_ = 1:numel(sigf_)
        c_ = d_.(sigf_{sf_});
        rows0_ = size(c_{1, 1}, 1);
        if rows0_ < minrows_(sf_)
            error(['A00 RESUME ABORTED: state %d file "%s" %s has %d rows; the ' ...
                'loader needs >= %d channels.'], dc_idx, saved_files(k).name, ...
                sigf_{sf_}, rows0_, minrows_(sf_));
        end
        for jp_ = 1:Npass
            a_ = c_{1, jp_};
            if ~ismatrix(a_) || size(a_, 1) ~= rows0_ || ...
                    size(a_, 2) ~= d_.DimAcel(1, jp_) || ~all(isfinite(a_(:)))
                error(['A00 RESUME ABORTED: state %d file "%s" %s passage %d is ' ...
                    'not a finite %dxDimAcel(=%d) matrix (got %s) — corrupt ' ...
                    'signal payload. Delete the file or use a FRESH folder.'], ...
                    dc_idx, saved_files(k).name, sigf_{sf_}, jp_, rows0_, ...
                    d_.DimAcel(1, jp_), mat2str(size(a_)));
            end
        end
    end
    completed(dc_idx) = true;                   % valid + matching + payload OK -> skip
end
if any(completed)
    fprintf('Resume: %d/%d states already complete and provenance-validated.\n', ...
        sum(completed), n_states);
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
    % Contact-validity log (audit 2026-07-17): the solver couples wheel and
    % rail BILATERALLY, so wheel-rail TENSION (= separation it cannot
    % represent) makes a passage physically invalid. B66 now measures it over
    % the whole on-track path; columns:
    %   [contactLost_bridge, contactLost_track, tension_frac_max, F_tension_max_N]
    % Nothing is auto-resampled (that would censor the severest cases) - the
    % log lets Python filter and the smoke tests assert.
    ContactLog = zeros(Npass, 4);

    % Per-STATE EOV state (2026-07-12 design review: persistent conditions -
    % crack, profile realization, track layer - are drawn ONCE per damage
    % state at j_pass==1 and HELD; wheel OOR stays per-passage = a different
    % train of the fleet each passage). Pre-initialised for parfor analysis.
    state_fra_class = profile_fra_classes(1);
    Tk = struct();
    beam_f1 = NaN;   % deck 1st natural frequency [Hz] - see the capture below

    % Pre-allocate LHS for this specific damage case (speed/temperature).
    % lhsdesign(n,p) = n SAMPLES x p VARIABLES: draw Npass samples of the 2
    % EOVs, then transpose to the 2 x Npass layout consumed below.
    % AUDIT FIX 2026-07-17: this was lhsdesign(2, Npass) - 2 "samples" in
    % Npass dimensions. Each LHS column stratifies its 2 values into opposite
    % halves of [0,1], so speed and temperature were forced anti-correlated
    % (corr = -0.75 exactly; -0.78 with the default maximin criterion) and the
    % (slow,cold) / (fast,hot) quadrants were NEVER generated. Marginals stay
    % uniform, so histograms could not show it.
    lhs_matrix = lhsdesign(Npass, 2)';
    % Guard: proper 2-var LHS has |corr| ~ 0 (sd ~ 1/sqrt(Npass)) and all four
    % quadrants occupied (P(fail) < 1e-5 at Npass = 50). A transposed call
    % fails BOTH checks, so this error would have caught the bug.
    corr_st_ = corr(lhs_matrix(1,:)', lhs_matrix(2,:)');
    occ_ = [any(lhs_matrix(1,:) <  0.5 & lhs_matrix(2,:) <  0.5), ...
            any(lhs_matrix(1,:) <  0.5 & lhs_matrix(2,:) >= 0.5), ...
            any(lhs_matrix(1,:) >= 0.5 & lhs_matrix(2,:) <  0.5), ...
            any(lhs_matrix(1,:) >= 0.5 & lhs_matrix(2,:) >= 0.5)];
    if abs(corr_st_) > 0.6 || ~all(occ_)
        error('A00: degenerate speed/temp LHS (corr=%.2f, quadrants=%s) - check lhsdesign orientation.', ...
            corr_st_, mat2str(occ_));
    end

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
        % Feature A (2026-07-19): in 'per_state' mode the ON/OFF decision is the
        % PRE-DRAWN CrackOn(DC) (forced OFF for controlled-probe families, ON
        % for nuisance_only, natural p=crack_p for joint — see the pre-parfor
        % block). Only the deprecated 'per_passage' branch still rolls a rand()
        % here. Location/intensity stay drawn from the per-state stream below.
        if use_crack_eov && (strcmp(crack_draw, 'per_passage') || j_pass == 1)
            if strcmp(crack_draw, 'per_passage')
                crack_now_ = rand() <= crack_p;   % DEPRECATED legacy behaviour
            else
                crack_now_ = CrackOn(DC);         % family-controlled activation
            end
            if crack_now_
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
                % the passage stream). Physical jitter is DISABLED (audit
                % 2026-07-17; see profile_jitter_sd_mm above) - measurement
                % noise is injected at LOAD time in core/dataset.py.
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
        % docs/track_eov_sampling_spec.md). x-coords: BRIDGE-LOCAL window
        % frame - deck at [track_L_app, track_L_app+L_bridge]; abutments =
        % the transitions. B54 maps this frame onto the real deck position
        % via Tk.x_bridge_local (audit fix 2026-07-17).
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
                    % NB: name must NOT collide with a for-loop variable used
                    % anywhere else in this parfor body (w_ is the wheel index
                    % in the OOR block below -> parfor rejects assigning to it).
                    wgt_ = 1; if near_ab, wgt_ = ballast_trans_mult; end
                    if rand() <= wgt_ / ballast_trans_mult, break; end
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
                    wgt_ = 1/hang_foul_mult; if in_foul, wgt_ = hang_foul_mult; end
                    if rand() <= wgt_ / hang_foul_mult, break; end
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
            % Frame anchor (audit fix 2026-07-17): position of the deck START
            % in the local window frame above. B54 shifts its global sleeper
            % axis by (real bridge start - x_bridge_local) so the window
            % [0, track_win] lands 30 m before / 30 m after the REAL deck.
            % Legacy descriptors without this field are read as global coords.
            Tk.x_bridge_local  = track_L_app;
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
                if oor_flats_enabled    % DISABLED (audit 2026-07-17; see toggle)
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
                end % if oor_flats_enabled
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
        % ---- Deck fundamental frequency: SELF-DOCUMENT every dataset ----
        % A 1000x deck-mass error (rho 9.6 vs 9600) hid for months precisely
        % because the ML still trained: it put f1 at ~131 Hz instead of ~4 Hz,
        % making the deck rigid to the vehicle. Recording f1 in every file makes
        % that class of bug impossible to miss. Expect ~4.1 Hz (L60/L40) or
        % ~2.8 Hz (L99.6). If it reads tens of Hz, STOP - check A03 rho / B43 A.
        if j_pass == 1 && isfield(Beam_local, 'Modal') && ...
                isfield(Beam_local.Modal, 'f') && ~isempty(Beam_local.Modal.f)
            beam_f1 = Beam_local.Modal.f(1);
            if DC == 1
                fprintf('[CHECK] deck f1 = %.2f Hz (healthy state; expect a few Hz)\n', beam_f1);
            end
        end

        % Contact-validity metrics for this passage (see ContactLog above)
        ContactLog(j_pass, :) = [double(Sol_local.contactLost), ...
            Sol_local.contactLost_track, Sol_local.tension_frac_max, ...
            Sol_local.F_tension_max];
        % HARD GATE per PASSAGE (audit R7 P5; TWO-TIER since 2026-07-19 — see
        % the F_CONTACT_TOL_N block at top level): abort at the FIRST passage
        % beyond the tolerated micro-unloading tier, saving the rest of the
        % state's compute. Non-finite is fatal in EITHER metric (NaN>tol is
        % false, so ~isfinite must be tested explicitly). The diagnostic is
        % written with fprintf, NOT `save` — `save` is illegal inside a parfor
        % loop without '-fromstruct' (checkcode PFSV). Both tolerances are the
        % TOP-LEVEL constants (broadcast into the parfor) so the resume
        % validation, the Python loader and this gate can never disagree.
        Ften_ = Sol_local.F_tension_max;
        Tfrac_ = Sol_local.tension_frac_max;
        if ~isfinite(Ften_) || ~isfinite(Tfrac_) || ...
                Ften_ > F_CONTACT_TOL_N || Tfrac_ > F_CONTACT_FRACTOL
            df_ = fopen(fullfile(run_folder, ...
                sprintf('contact_violation_state%04d.txt', DC)), 'w');
            if df_ >= 0
                fprintf(df_, ['state %d  passage %d/%d\nscour_vec %s\n' ...
                    'tol_N %g  tol_frac %g\n'], DC, j_pass, Npass, ...
                    mat2str(scour_vec), F_CONTACT_TOL_N, F_CONTACT_FRACTOL);
                fprintf(df_, ['F_tension_max_N %g  contactLost_track %g  ' ...
                    'tension_frac_max %g\n'], Ften_, Sol_local.contactLost_track, ...
                    Tfrac_);
                fclose(df_);
            end
            error(['A00 ABORT (contact): state %d passage %d exceeds the ' ...
                'two-tier contact gate (tension %.4g N vs tol %g N; path ' ...
                'fraction %.3g vs tol %g; non-finite also fatal). Beyond brief ' ...
                'micro-unloading = physics regression. Fix the generator. ' ...
                'See contact_violation_state%04d.txt'], ...
                DC, j_pass, Ften_, F_CONTACT_TOL_N, Tfrac_, F_CONTACT_FRACTOL, DC);
        end

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
    % Feature A (2026-07-19): the state's FAMILY identity, self-describing per
    % file (the Python loader cross-checks it against damage_states.mat).
    data2save.state_family = StateFamily{DC};
    % Bearing label ([left,right] Nm/rad; zeros unless bearing_mode='target'/'fixed')
    data2save.beam_f1_Hz = beam_f1;           % deck 1st nat. freq [Hz] - sanity anchor
    data2save.bearing_vector = bear_vec;      % k_r [Nm/rad] (the physics)
    data2save.bearing_fixity = bear_fix;      % fixity in [0,1] (the regression LABEL)
    % --- NUISANCE-EOV TRACEABILITY (not labels) ---
    data2save.crack_log    = CrackLog;      % per passage: [loc_m, EI-loss, lc_m]
    data2save.profile_mode = profile_mode;
    data2save.profile_log  = ProfileLog;    % per passage: intensity or FRA class
    data2save.track_log    = TrackLog;      % per passage: Damage.track struct ([] if off)
    data2save.oor_log      = OORLog;        % per passage: [veh wheel len_m phase] rows
    % Contact validity per passage (audit 2026-07-17):
    % [contactLost_bridge, contactLost_track, tension_frac_max, F_tension_max_N]
    data2save.contact_log  = ContactLog;
    data2save.gen_schema   = gen_schema;      % generator schema (resume/loader guard)
    data2save.gen_fingerprint = gen_fingerprint;  % full config fingerprint (per-file provenance)
    % Contact was hard-gated PER PASSAGE inside the loop above (audit R7 P5), so
    % by here every passage is within tolerance and finite. NOTE on Codex's R6
    % framing: col 2 = (F_onTrack_max>0) and col 4 = F_onTrack_max derive from the
    % SAME quantity, so col4>tol always implies col2=1 — MATLAB was never "silent"
    % on the wrong column; the real defect was warn-not-error (now: abort per
    % passage, on col 4, byte-for-byte the Python _CONTACT_F_TOL_N gate).
    % Tolerated-tier note (0 < tension <= gate): brief micro-unloading events
    % the two-tier gate accepts (2026-07-19). Surfaced per state so the run log
    % carries the incidence; the Python loader aggregates the same statistic
    % dataset-wide for the paper.
    grey_ = (ContactLog(:,4) > 0) & (ContactLog(:,4) <= F_CONTACT_TOL_N);
    if any(grey_)
        fprintf(['[NOTE] state %d: %d/%d passages in the tolerated micro-' ...
            'tension tier (0 < F <= %g N, frac <= %g; worst F %.4g N).\n'], ...
            DC, sum(grey_), Npass, F_CONTACT_TOL_N, F_CONTACT_FRACTOL, ...
            max(ContactLog(:,4)));
    end
    data2save.Temperatura = Temperatura;
    data2save.Velocidade = Velocidade;
    data2save.VehiclesProps = x_veh;

    save_progress(data2save, DC, run_folder, gen_schema, gen_fingerprint);
end

% Mark end time of the run
tempo_final = datetime('now', 'Format', 'dd_MM_yyyy_HH_mm_ss');
save(fullfile(run_folder, 'tempo_final.mat'), 'tempo_final');
tempo_total = tempo_final - tempo_inicial;
fprintf('Tempo total: %s\n', char(tempo_total));
save(fullfile(run_folder, 'tempo_total.mat'), 'tempo_total');

% ---- Completion marker (audit R5; atomic write R6 C5) -------------------
% Written ONLY when all n_states state files exist. The Python loader REQUIRES
% this marker, so a partial/interrupted generation can never be trained on
% (even if the files present happen to look self-consistent). Written
% atomically (temp + movefile) so a crash mid-write cannot leave a truncated
% marker that the loader treats as "complete".
present_ = 0;
for dc_ = 1:n_states
    if exist(fullfile(run_folder, sprintf('%04d.mat', dc_)), 'file'), present_ = present_ + 1; end
end
if present_ == n_states
    % Source content integrity (audit R7.1 P4): SHA-256 EVERY state file, build a
    % 'digest_lines' blob ("NNNN.mat:<sha>\n...") + a ROOT = SHA-256 of it, save
    % file_digests.mat, and put the root as LINE 3 of the marker. The Python loader
    % verifies each file's bytes against this, catching bit-flip / disk-transport
    % corruption / same-size overwrite (none preserve a SHA).
    fnames_ = arrayfun(@(k) sprintf('%04d.mat', k), (1:n_states)', 'UniformOutput', false);
    lines_ = cell(n_states, 1);
    for k_ = 1:n_states
        sha_ = local_file_sha256(fullfile(run_folder, fnames_{k_}));
        lines_{k_} = sprintf('%s:%s', fnames_{k_}, sha_);
    end
    digest_lines = strjoin(sort(lines_), sprintf('\n'));   % sorted, '\n'-joined
    root_ = local_sha256(digest_lines);
    file_digests = struct('digest_lines', digest_lines, 'root', root_); %#ok<NASGU>
    save(fullfile(run_folder, 'file_digests.mat'), 'file_digests');

    tmp_marker_ = fullfile(run_folder, '._GENERATION_COMPLETE.tmp');
    fid_ = fopen(tmp_marker_, 'w');
    fprintf(fid_, '%s\n%s\n%s\n', gen_schema, gen_fingerprint, root_);
    fclose(fid_);
    movefile(tmp_marker_, marker_path, 'f');
    fprintf('Generation COMPLETE: %d/%d states -> wrote _GENERATION_COMPLETE + file_digests.\n', ...
        present_, n_states);
else
    fprintf(['Generation INCOMPLETE: %d/%d states present — NO completion ' ...
        'marker written. Re-run A00 in this folder to finish.\n'], present_, n_states);
end

% =========================================================================
%  Local functions
% =========================================================================
function h = local_sha256(str)
    % SHA-256 hex digest of a char row vector (audit R5 canonical fingerprint).
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(uint8(str));               % Java byte[] -> MATLAB int8
    h = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end

function h = local_file_sha256(fpath)
    % SHA-256 hex digest of a file's raw BYTES (audit R6 C7: profile-asset hash).
    fid = fopen(fpath, 'r');
    if fid < 0, error('local_file_sha256: cannot open %s', fpath); end
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(bytes);                     % Java byte[] -> MATLAB int8
    h = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end
