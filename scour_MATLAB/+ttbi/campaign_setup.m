function config = campaign_setup(inputs)
%CAMPAIGN_SETUP Define one reviewed Paper-1 campaign block and its priors.
%
% This function owns configuration only: stage geometry/activation, state-count
% constants supplied by the driver, operational variability, and the registered
% nuisance priors. It performs no state sampling, filesystem IO, or solver work.
% The long scientific comments stay beside the values they justify.

required = {'STAGE'; 'n_states_multi'; 'Npass'; 'n_healthy_states'; ...
    'n_anchor_levels'; 'n_anchor_reps'; 'n_nuisance_states'};
allowed = [required; {'qualification_run'}];
if ~isstruct(inputs) || ~isscalar(inputs) || ...
        ~(isequal(sort(fieldnames(inputs)), sort(required)) || ...
          isequal(sort(fieldnames(inputs)), sort(allowed)))
    error('ttbi:campaign_setup:Inputs', ...
        ['inputs must contain the seven reviewed driver fields and only the ' ...
         'optional explicit qualification_run marker.']);
end
qualification_run = false;
if isfield(inputs, 'qualification_run')
    qualification_run = inputs.qualification_run;
end
if ~islogical(qualification_run) || ~isscalar(qualification_run)
    error('ttbi:campaign_setup:QualificationMarker', ...
        'qualification_run must be one scalar logical.');
end
for field = required(2:end)'
    value = inputs.(field{1});
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || ...
            value < 0 || value ~= floor(value) || ...
            (strcmp(field{1}, 'Npass') && value < 1) || ...
            (strcmp(field{1}, 'n_anchor_reps') && value < 1)
        error('ttbi:campaign_setup:Count', ...
            ['%s must be a nonnegative integer scalar; Npass and ' ...
             'n_anchor_reps must be positive.'], field{1});
    end
end

% =========================================================================
% Simulation setup
% =========================================================================
% ===== STAGE PRESET — the paper-case selector ============================
% Pick ONE active rung. Each preset sets the geometry, labelled bridge targets,
% and randomized/logged environmental or operational variability (EOV).
%
% Paper 1 has four self-contained blocks.  Track damage, rail-pad failure,
% hanging sleepers, ballast patches and wheel polygonization remain present in
% the source tree but are disabled by every block below.
%
%   F40-S  2 x 20 m; central scour; dense 0..60 percent grid (61 x 5)
%   F40-M  same bridge; scour + nominal bearing fixity + local EI loss
%   L99-S  4 x 24.9 m; supports 2/3/4; scour-only physics
%   L99-M  same bridge; scour + nominal bearing fixity + local EI loss
%
% HEADS = scour (per pier) + bearing (per abutment) ONLY. Crack, rail profile,
% track-layer and wheel damage are NUISANCES: randomized, logged, never
% estimated. They support an empirical robustness evaluation under confounding
% variability; perfect invariance is neither assumed nor required. Rationale:
% docs/framework_rationale.md.
STAGE = char(string(inputs.STAGE));

% Defaults; each rung below only switches ON what it adds.
use_track_eov = false;
use_oor_eov   = false;

switch STAGE
    case 'F40-S'
        damage_mode='multi_scour'; L_bridge=40; num_spans=2; scour_supports=2;
        state_design_kind='dense-scour-61x5-v1';
        bearing_mode='off'; use_crack_eov=false; profile_mode='fixed';
    case 'F40-M'
        damage_mode='multi_scour'; L_bridge=40; num_spans=2; scour_supports=2;
        state_design_kind='five-family-multidamage-v2';
        bearing_mode='target'; use_crack_eov=true; profile_mode='fixed';
    case 'L99-S'
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        state_design_kind='five-family-multidamage-v2';
        bearing_mode='off';    use_crack_eov=false; profile_mode='fixed';
    case 'L99-M'
        damage_mode='multi_scour'; L_bridge=99.6; num_spans=4; scour_supports=[2 3 4];
        state_design_kind='five-family-multidamage-v2';
        bearing_mode='target'; use_crack_eov=true; profile_mode='fixed';
    otherwise
        error('A00: unknown STAGE preset "%s"', STAGE);
end
if strcmp(STAGE, 'F40-S')
    expected_counts = [0 5 60 5 0];
else
    expected_counts = [250 50 5 5 50];
end
actual_counts = [inputs.n_states_multi inputs.n_healthy_states ...
    inputs.n_anchor_levels inputs.n_anchor_reps inputs.n_nuisance_states];
if ~qualification_run && ~isequal(actual_counts, expected_counts)
    error('A00:StageCountContract', ...
        ['Stage %s requires [joint healthy levels replicas nuisance] = %s; ' ...
         'the driver/bundle supplied %s.'], STAGE, mat2str(expected_counts), ...
        mat2str(actual_counts));
end
if qualification_run
    if inputs.Npass > 5 || any(actual_counts > [16 8 8 8 16])
        error('A00:QualificationCountContract', ...
            ['Qualification count tuple/passages exceeds the reviewed micro ' ...
             'ceiling: tuple=%s, Npass=%d.'], mat2str(actual_counts), ...
            inputs.Npass);
    end
    % F40-S production uses the exact dense 61x5 catalogue.  Its reduced
    % qualification fixture deliberately exercises the common five-family
    % machinery instead; qualification output is segregated and loader-
    % inadmissible, so it cannot impersonate the scientific dense design.
    state_design_kind = 'qualification-five-family-v1';
end
% Verify the support positions land on mesh nodes for your L/num_spans
% (B07/B43) before a long run. num_spans=N -> supports {1..N+1}; internal
% piers (valid support-stiffness-loss targets) are 2..N.

% ===== Operational variability (keep ON to match the training distribution) ===
% NOISE POLICY 2026-07-12 (user decision): generation is NOISE-FREE from
% stage1_crack onward. Measurement noise is injected at LOAD time instead
% (core/dataset.py 'sensor_noise' config), where per-channel levels stay
% configurable per experiment - levels from sensor DATASHEETS (noise floor);
% EN 61373 position severities describe the vibration ENVIRONMENT for
% equipment qualification (range/reliability), NOT the acquisition noise.
% Set true ONLY to reproduce the legacy Stage-0/1 pipeline (multiplicative
% noise on the AcelRoda moving-rail group in D01; adds 'N' to var_tag).
use_signal_noise = false;       % legacy toggle - keep FALSE (noise at load time)
use_vehicle_variability = true; % Toggle vehicle property variability
use_speed_variability = true;   % Toggle train speed variability
use_temp_variability = true;    % Toggle temperature variability

% ===== Registered target/design states =====
% Dense F40-S severities.  The multi-damage blocks use their controlled anchor
% and joint designs instead; all values remain fractions of k_v0.
Dano  = (0:60)/100;

% --- multi_scour: independent scour at scour_supports (set by the STAGE preset).
n_states_multi = inputs.n_states_multi; % independent joint LHS states
dano_max         = 0.60;    % max modeled support-stiffness-loss fraction
include_anchors  = true;    % prepend zero-target + single-pier sweeps (localisation extremes)
n_anchor_levels = inputs.n_anchor_levels; % anchor severities
damage_seed      = 1;       % RNG seed for a REPRODUCIBLE damage-state grid

% ===== Feature A (2026-07-19): explicit state FAMILIES =====================
% Every state carries a `state_family` identity tag written by the GENERATOR.
% The five-family INVENTORY is fixed within a geometry in every rung; a
% bearing_only/nuisance_only row remains a latent design stratum when its
% mechanism is OFF. Only active physics changes across paired rungs, never
% sample size, row order, UID inventory, or scour realization.
% (not derived from labels downstream — with per-STATE persistent nuisances a
% zero-head state still carries a crack/profile/track fingerprint, so `y`
% alone cannot identify families; that was the flaw in the pre-redesign
% threshold-based probe). Families:
%   target_healthy  LEGACY ID: all heads zero; nuisances follow the rung's
%                   active policy EXCEPT crack, which is forced OFF. This is a
%                   zero-target baseline, not a physically clean condition.
%   scour_only      exactly ONE pier scoured (5 levels x n_anchor_reps);
%                   bearing zero, crack forced OFF (controlled probe).
%   bearing_only    exactly ONE abutment has nonzero nominal rotational fixity
%                   (5 levels x n_anchor_reps);
%                   scour zero, crack forced OFF (controlled probe).
%   nuisance_only   all heads zero, crack forced ON (the false-positive probe:
%                   does the model report damage when only a nuisance exists?).
%                   Always present as a latent stratum; crack activates only
%                   when use_crack_eov=true. The FRA profile
%                   class is FIXED at 4 — only its phase varies — and track
%                   damage is present w.p. ~1 under its Poisson rates).
%   joint           the LHS block; crack drawn naturally (p = crack_p).
% WHY: the stratified split (core/dataset.py) guarantees every family lands in
% train / inner-val / outer-test, so the disentanglement and false-positive
% probes can never be silently empty (audit R6 C8C loud-guard now a guarantee),
% and 50 zero-target rows are independent states (50 correlated passages of
% one state are NOT 50 independent zero-target states).
n_healthy_states = inputs.n_healthy_states; % zero-target diagnostics
n_anchor_reps = inputs.n_anchor_reps; % replicas per family/target/level
n_nuisance_states = inputs.n_nuisance_states; % nuisance diagnostics

Npass = inputs.Npass; % passages per semantic state
% PROSPECTIVE PASSAGE BUDGET: 50 is a fixed, balanced, compute-feasible inner
% integration grid for the 50 speed-temperature LHS cells generated per state.
% Passages are correlated repeated observations, not independent inferential
% units. This count is not a power calculation or universal passage-level MCSE
% guarantee. It is identical across states/rungs and must not be changed after
% outcomes are seen; Python reports seed/repeated-CV stability and uses its
% state-level MCSE diagnostic only to plan a future regeneration.
% Operational memory guard (R11): each independent TTBI solve has a large
% process-private matrix footprint. MATLAB's automatic 16-worker pool exhausted
% 32 GB even for the 17-state micro qualification run. Keep the process pool
% explicitly bounded; every RNG namespace is keyed by semantic StateUID below, so
% scheduling cannot change the generated draws. This value is fingerprinted and
% recorded, making any deliberate retuning a fresh, auditable run.
max_parfor_workers = 4;

% ===== Bearing nominal rotational fixity (abutment spring, Nm/rad) =====
% MODE is set by the STAGE preset; these are the numeric knobs it uses.
% 'off'    zero nominal rotational fixity at both abutments.
% 'fixed'  the SAME Bearing_Intensity at the LEFT abutment for every state.
% 'target' bearing state [left,right] becomes a LABELLED TARGET — sampled
%          JOINTLY with scour by one LHS and saved as data.bearing_vector.
Bearing_Intensity = 0.0;    % 'fixed' mode value [Nm/rad]
% ===== RANGE EXTENDED 2026-07-15: sample FIXITY, not raw k_r =============
% The old design sampled k_r ~ U(0, 1e9) and mislabeled k_r/1e9 as seized-%.
% Problem: Fernandes's 1e9 is a studied drive-by detectability landmark, not
% a universal physical maximum. Using it as our cap compressed the modeled
% response into the low-fixity portion of this particular end-spring model.
% Khan (2022) sweeps k_r continuously to 1e12 and reaches the model's
% near-fixed regime; this does not establish a population distribution for
% field bearing-condition populations.
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
% for all its passages (crack_draw='per_state'), with author-chosen activation
% probability 0.25 (a
% p=1.0 per-passage redraw deprives the model of any zero-crack deck baseline
% and is physically indefensible). Semantics mirror TTBI_2D/damage_config.py
% (damaged-element EI reduction, cf. Fernandes et al. 2025; NOT Sinha's
% tapered model - mis-citation corrected in the 2026-07-17 audit).
crack_draw       = 'per_state';   % 'per_state' (persistent damage) | 'per_passage' (DEPRECATED)
crack_p          = 0.25;          % registered crack-state modeling prior
% A historical research report proposed a 20-30% context range, but no directly
% audited primary calibrates that range to this campaign population or to this
% Bernoulli law. The value 0.25 is therefore wholly author-chosen and is not an
% estimated population prevalence.
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
% compression, rarely giving a real EI loss. The registered 4:1
% hogging:sagging ratio is a MODELING PRIOR, not a measured population
% prevalence. Literature on support-region cracking and Eurocode 4
% cracked-region guidance motivates preferential support-zone sampling but
% does not identify these exact odds. Eurocode 4 mandates analysing a
% "cracked" section over 15% of the span each side of internal supports.
crack_hog_ratio  = 4.0;           % hogging:sagging location-weight design prior
crack_hog_margin = 0.175;         % zone half-width as a fraction of SPAN (EC4: 15%; report 15-20%)
crack_frac_range = [0.10 0.90];   % LEGACY uniform bounds — used only as a clamp now
crack_int_range  = [0.05 0.30];   % EI-loss fraction (Fernandes 2025: 0.14/0.22)
crack_lc         = 0.0;           % half-length [m] each side; <=0 -> single element

% ===== Rail profile mode (EOV) =====
% profile_mode is set by the STAGE preset; these are its numeric knobs.
% 'fixed'        one fixed-phase FRA-v2 class-4 realization shared by states.
% 'fixed_scaled' that same realization x an amplitude factor (non-registered).
% 'psd_fra'      the SAME FRA-v2 class-4 spectrum with per-state phases.
% Therefore s13->s14 and s22->s23 do not silently change the spectral law or
% amplitude: the profile contrast is fixed realization vs a distribution of
% phase realizations under one executable spectrum.
% EOV DESIGN REVIEW 2026-07-12, wording corrected 2026-07-27: ONE class + ONE
% phase realization is drawn per damage STATE and held for its passages
% (profile_draw='per_state'; B19 seeds the phase draw per state). This is an
% explicit scenario-persistence assumption, not an inference from EN 13848-2
% measurement-system repeatability. Any physical persistence claim in the paper
% requires its own track-evolution evidence.
% Class set FIXED at FRA class 4 by the prospectively source-locked design.
% It is not claimed to be the legal or physically worst permissible condition;
% any service/speed interpretation must be conditioned on the applicable rule.
% The old per-passage class+phase redraw over {4,5,6} is DEPRECATED (it is what
% collapsed the sprung channels in the L100 Stage-2 pilot).
profile_draw         = 'per_state';  % 'per_state' | 'per_passage' (DEPRECATED)
% AUDIT FIX 2026-07-17: former 0.5 mm per-passage white profile perturbation
% DISABLED. The 0.5 mm
% white noise misread EN 13848-2 - that figure is the MEASURING SYSTEM's
% repeatability, not physical track change between trains - and, being
% spectrally white on a profile band-limited at 1.524 m, it contributed ~zero
% in-band but injected 100% fictitious sub-1.5 m roughness (~125 g-equivalent
% axle forcing up to the solver Nyquist, strongly loading the legacy AcelRoda
% moving-coordinate rail channels).
% Noise belongs to the OBSERVATION model: load-time sensor noise in
% core/dataset.py (docs/framework_rationale.md). If a physical pass-to-pass
% perturbation is ever wanted, it must be separately justified, band-limited,
% versioned and validated against an appropriate physical evolution model.
profile_jitter_sd_mm = 0;            % per-passage additive white noise [mm] - keep 0 (see above)
profile_int_range    = [0.5 2.0];    % 'fixed_scaled' amplitude range
profile_fra_classes  = 4;            % 'psd_fra' class set (scalar = fixed class)
profile_fixed_phase_seed = 20260728; % shared fixed FRA-v2 realization
profile_spectrum_contract = 'fra-v2-class4-cycles-per-m-v1';
% Reviewed finite-rail-domain decision.  The source-locked 18-case 6/15/30 m
% coupled study selected 6 m; production must request it explicitly and must
% never inherit B43's nonproduction compatibility fallback.
rail_end_clearance_m = 6;
rail_end_clearance_decision_id = ...
    'paper1-rail-domain-clearance-c06-v1';

% ===== Track-layer damage EOVs (Stage 3; use_track_eov) =====
% Evidence-boundary specification: docs/track_eov_sampling_spec.md. It
% separates primary-source anchors and proxies from inferred or author-chosen
% design priors; the exact active numbers are not claimed as measured network
% distributions. Randomized NUISANCES — logged, NOT labels. EOV DESIGN REVIEW
% 2026-07-12: ballast patches / hanging sleepers / pad service-condition
% scalars are PERSISTENT
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
% ===== MIXED-EVIDENCE MODELING PRIORS (reviewed 2026-07-31) ===============
% These parameters mix primary-source anchors, engineering proxies, inferred
% values, and author-chosen stress scenarios. None is a measured joint
% population distribution. The primary-PDF verdict and exact boundary for each
% value are recorded in docs/track_eov_sampling_spec.md. The primaries do not
% identify an impact-threshold taxonomy or a population count law. Narrow
% anchors only: widespread poor support is reported; one modeled 1 mm gap
% increased the adjacent sleeper-ballast force by 70%; a separate review
% summarizes wheel-rail-force increases up to 80%. These facts cannot be
% combined into a prevalence estimate. The Poisson rate below is therefore an
% author-chosen stress prior. Its arithmetic 3*3/167 = 5.4% share assumes a
% mean group size of three and is not field prevalence.
% Direct-source correction 2026-07-31: published ballast surveys use
% incompatible thresholds and denominators, so they do not fit the exact
% Poisson rate below. The rate is retained only as an author-chosen design
% point. Published fouled-zone lengths contextualize the 5-20 m bounds, but do
% not establish the exact uniform law. The proposed lambda sensitivity sweep
% {0.6, 1.2, 2.4} is not implemented and must not be claimed as a result.
% RATES are per 100 m and are SCALED BY THE MODELLED WINDOW below (our track is
% 120 m at L60 / 159.6 m at L99.6 - drawing a fixed count regardless of length
% was itself an error).
% A historical report selected lambda=3.0 by intersecting its proposed
% 2.0-3.0 range with an unsupported 5-10% prevalence assertion. The direct-PDF
% audit invalidated that calibration. Keep the frozen value only as an
% author-chosen stress prior; 3*3/167=5.4% is conditional arithmetic, not data.
hang_rate_100m    = 3.0;         % author-chosen Poisson stress prior
ballast_rate_100m = 1.2;         % author-chosen Poisson design rate
ballast_patch_len = [5 20];      % author-chosen U over source-contextualized bounds
% Literature motivates coupling fouling and loss of support, but does not fit
% the exact placement odds. The 3:1 rule is an author-chosen quantification.
hang_foul_mult    = 3.0;         % author-chosen 3:1 coupling odds, evidence-motivated
% Literature motivates transition-zone vulnerability. It does not establish
% the exact x3 density or 20 m window used here; both are design choices.
ballast_trans_mult   = 3.0;      % evidence-motivated design multiplier, not measured density
ballast_trans_margin = 20;       % author-chosen [m] within 15-24 m context
ballast_p_wet     = 0.5;         % author-chosen wet/dry scenario balance
% Dry stiffness is an author-chosen high-stiffness scenario (the audited
% dry-sand primary instead shows mild softening). The damping band overlaps
% the observed dry-fouling reduction but is still a design range. The opt-in,
% arm-paired sign check is specified in
% docs/dry_ballast_stiffness_sign_sensitivity.md; it is not a campaign rung.
ballast_eta_k_dry = [1.2 2.0]; ballast_eta_c_dry = [0.4 0.8];
% Wet multipliers use flooded clean ballast as an engineering proxy; they are
% not a direct calibration of a jointly wet-and-fouled field condition.
ballast_eta_k_wet = [0.7 0.9]; ballast_eta_c_wet = [1.5 4.0];
% (group COUNT is now Poisson(hang_rate_100m * track_win/100) - see above)
hang_group_size   = [1 5];       % author-chosen DU; sources do not fit this law
hang_p_transition = 0.6;         % P(group in a transition zone) [assumption]
hang_trans_margin = 15;          % evidence-motivated transition-zone extent
% The pad law is an author-chosen service-condition stress scenario. The
% audited pad papers do not fit this Weibull law and report mixed aging
% directions; do not describe these parameters as an empirical aging model.
pad_chi_range     = [1.0 3.5];   % service-condition stiffness multiplier bounds
pad_weibull       = [1.8 2.2];   % author-chosen Weibull(lambda,k), clipped
pad_beta_range    = [0.8 1.2];   % author-chosen damping multiplier range
% No audited primary in the local library establishes the previously claimed
% 0.5%/year pad-failure incidence (Williams et al. 2014 was a misattribution).
% Therefore p=0.02 is wholly an author-chosen per-position snapshot stress
% prior, not a converted field rate or measured network prevalence.
pad_p_fail        = 0.02;        % author-chosen per-pad snapshot stress prior
pad_failure_rule  = 'independent-bernoulli-sleeper-lattice-v1';

% ===== Wheel out-of-roundness EOV (polygonization active; flats disabled) ===
% Mechanism/source scope: docs/damage_model_reference.md. Literature supports
% flat/polygonization physics and fleet-specific examples; it does not validate
% the exact author-chosen polygon occurrence/order/amplitude prior below.
% ~12% of one reported in-service wheel population carried a flat; braking
% couples axles within a bogie.
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
% ---- low-order polygonization: separate CONTINUOUS nuisance -------------
% Primaries establish the mechanism and fleet-specific low-order examples,
% not this occurrence/order/amplitude triplet. All three form one
% author-chosen maintained-wheel design prior.
poly_p_wheel    = 0.30;          % author-chosen P(any modeled polygonization)
poly_orders     = [1 5];         % author-chosen n ~ DU(1,5)
poly_amp_lnorm  = [-10.0 0.5];   % author-chosen ln(amp [m]) law
poly_amp_bounds = [1e-5 1.2e-4]; % clip 0.01-0.12 mm: author-chosen design band (literature supports the polygonization physics, not this exact band)

Nveh = 5;     % number of vehicles
Nprop = 3;    % how many vehicle properties will be varied
Desvio = 0.05; % standard deviation of the noise of the signal
temp_min = 3; temp_max = 33; % min and max temperature [°C]
vel_min = 70; vel_max = 90; % min and max velocity [km/h]


% Explicit output assignment is deliberate: callers and reviewers can see the
% complete configuration API, and cell-valued fields cannot expand the struct.
config = struct();
config.STAGE = STAGE;
config.qualification_run = qualification_run;
config.state_design_kind = state_design_kind;
config.damage_mode = damage_mode;
config.L_bridge = L_bridge;
config.num_spans = num_spans;
config.scour_supports = scour_supports;
config.bearing_mode = bearing_mode;
config.use_crack_eov = use_crack_eov;
config.profile_mode = profile_mode;
config.use_track_eov = use_track_eov;
config.use_oor_eov = use_oor_eov;
config.use_signal_noise = use_signal_noise;
config.use_vehicle_variability = use_vehicle_variability;
config.use_speed_variability = use_speed_variability;
config.use_temp_variability = use_temp_variability;
config.Dano = Dano;
config.n_states_multi = n_states_multi;
config.dano_max = dano_max;
config.include_anchors = include_anchors;
config.n_anchor_levels = n_anchor_levels;
config.damage_seed = damage_seed;
config.n_healthy_states = n_healthy_states;
config.n_anchor_reps = n_anchor_reps;
config.n_nuisance_states = n_nuisance_states;
config.Npass = Npass;
config.max_parfor_workers = max_parfor_workers;
config.Bearing_Intensity = Bearing_Intensity;
config.bearing_fixity_max = bearing_fixity_max;
config.crack_draw = crack_draw;
config.crack_p = crack_p;
config.crack_hog_ratio = crack_hog_ratio;
config.crack_hog_margin = crack_hog_margin;
config.crack_frac_range = crack_frac_range;
config.crack_int_range = crack_int_range;
config.crack_lc = crack_lc;
config.profile_draw = profile_draw;
config.profile_jitter_sd_mm = profile_jitter_sd_mm;
config.profile_int_range = profile_int_range;
config.profile_fra_classes = profile_fra_classes;
config.profile_fixed_phase_seed = profile_fixed_phase_seed;
config.profile_spectrum_contract = profile_spectrum_contract;
config.rail_end_clearance_m = rail_end_clearance_m;
config.rail_end_clearance_decision_id = rail_end_clearance_decision_id;
config.track_draw = track_draw;
config.track_L_app = track_L_app;
config.track_L_after = track_L_after;
config.hang_rate_100m = hang_rate_100m;
config.ballast_rate_100m = ballast_rate_100m;
config.ballast_patch_len = ballast_patch_len;
config.hang_foul_mult = hang_foul_mult;
config.ballast_trans_mult = ballast_trans_mult;
config.ballast_trans_margin = ballast_trans_margin;
config.ballast_p_wet = ballast_p_wet;
config.ballast_eta_k_dry = ballast_eta_k_dry;
config.ballast_eta_c_dry = ballast_eta_c_dry;
config.ballast_eta_k_wet = ballast_eta_k_wet;
config.ballast_eta_c_wet = ballast_eta_c_wet;
config.hang_group_size = hang_group_size;
config.hang_p_transition = hang_p_transition;
config.hang_trans_margin = hang_trans_margin;
config.pad_chi_range = pad_chi_range;
config.pad_weibull = pad_weibull;
config.pad_beta_range = pad_beta_range;
config.pad_p_fail = pad_p_fail;
config.pad_failure_rule = pad_failure_rule;
config.oor_flats_enabled = oor_flats_enabled;
config.oor_q_bogie = oor_q_bogie;
config.oor_p_trailing = oor_p_trailing;
config.oor_p_fresh = oor_p_fresh;
config.oor_len_fresh = oor_len_fresh;
config.oor_len_runin = oor_len_runin;
config.oor_radius = oor_radius;
config.poly_p_wheel = poly_p_wheel;
config.poly_orders = poly_orders;
config.poly_amp_lnorm = poly_amp_lnorm;
config.poly_amp_bounds = poly_amp_bounds;
config.Nveh = Nveh;
config.Nprop = Nprop;
config.Desvio = Desvio;
config.temp_min = temp_min;
config.temp_max = temp_max;
config.vel_min = vel_min;
config.vel_max = vel_max;
end
