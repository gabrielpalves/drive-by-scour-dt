function design = build_state_design(config)
%BUILD_STATE_DESIGN Build the immutable R11 state and RNG design.
%
% This function owns the scientific state catalogue: the five state families,
% semantic UIDs, latent bearing/crack variables, and named RNG stream IDs.
% Keeping it separate from filesystem publication and solver execution makes
% the campaign design reviewable without stepping through the full driver.
%
% Input is one scalar struct whose fields are listed explicitly below. Output
% is a scalar struct; every field is assigned explicitly at the end so MATLAB
% cannot accidentally expand a cell-valued field into a struct array.

required = { ...
    'L_bridge', 'num_spans', 'damage_mode', 'state_design_kind', ...
    'include_anchors', 'Dano', ...
    'scour_supports', 'n_states_multi', 'dano_max', 'damage_seed', ...
    'n_healthy_states', 'n_anchor_levels', 'n_anchor_reps', ...
    'n_nuisance_states', 'bearing_fixity_max', 'Npass', ...
    'use_crack_eov', 'crack_draw', 'crack_p', ...
    'profile_mode', 'profile_draw', 'use_track_eov', 'track_draw', ...
    'bearing_mode', 'Bearing_Intensity'};
if ~isstruct(config) || ~isscalar(config)
    error('ttbi:build_state_design:ConfigType', ...
        'config must be one scalar struct.');
end
missing = setdiff(required, fieldnames(config));
if ~isempty(missing)
    error('ttbi:build_state_design:ConfigFields', ...
        'config is missing required fields: %s', strjoin(missing, ', '));
end

% Local aliases keep the equations readable and make the input boundary clear.
L_bridge = config.L_bridge;
num_spans = config.num_spans;
damage_mode = config.damage_mode;
state_design_kind = config.state_design_kind;
include_anchors = config.include_anchors;
Dano = config.Dano;
scour_supports = config.scour_supports;
n_states_multi = config.n_states_multi;
dano_max = config.dano_max;
damage_seed = config.damage_seed;
n_healthy_states = config.n_healthy_states;
n_anchor_levels = config.n_anchor_levels;
n_anchor_reps = config.n_anchor_reps;
n_nuisance_states = config.n_nuisance_states;
bearing_fixity_max = config.bearing_fixity_max;
Npass = config.Npass;
use_crack_eov = config.use_crack_eov;
crack_draw = config.crack_draw;
crack_p = config.crack_p;
profile_mode = config.profile_mode;
profile_draw = config.profile_draw;
use_track_eov = config.use_track_eov;
track_draw = config.track_draw;
bearing_mode = config.bearing_mode;
Bearing_Intensity = config.Bearing_Intensity;

% Every row of DamageStates is the complete per-support scour vector saved in
% one state file. The latent bearing dimensions are drawn in every rung so the
% state inventory and common-random-number design do not depend on activation.
n_supp = num_spans + 1;
n_latent_bear = 2;
state_identity_version = 'semantic-state-v2';
if strcmp(state_design_kind, 'dense-scour-61x5-v1')
    joint_lhs_design = 'not-applicable-dense-scour';
elseif strcmp(state_design_kind, 'five-family-multidamage-v2')
    joint_lhs_design = 'master-scour-plus-two-bearing-v2';
elseif strcmp(state_design_kind, 'qualification-five-family-v1')
    if ~isfield(config, 'qualification_run') || ...
            ~isequal(config.qualification_run, true)
        error('A00:QualificationStateDesignMarker', ...
            ['qualification-five-family-v1 requires the explicit true ' ...
             'qualification marker.']);
    end
    joint_lhs_design = 'qualification-master-lhs-v1';
else
    error('A00:UnknownStateDesign', ...
        'Unknown state_design_kind "%s".', state_design_kind);
end
random_stream_schedule_version = 'uid-named-substreams-v2';
state_stream_names = {'operations','crack','profile-state','track','profile-phase'};
passage_stream_names = {'profile-passage','oor-passage'};
if ~strcmp(damage_mode, 'multi_scour') || ~include_anchors
    error(['A00: Paper-1 campaign stages require multi_scour with the ' ...
        'registered replicated-anchor inventory.']);
end

% Convert the geometry-normalised bearing fixity ratio to rotational stiffness
% using the exact beam properties used by the solver.
Beam_probe = A03_Bridge(struct( ...
    'Prop', struct('L', L_bridge, 'num_spans', num_spans)));
k_ref_bear  = 4 * Beam_probe.Prop.E * Beam_probe.Prop.I / (L_bridge / num_spans);
fix2k       = @(phi) k_ref_bear .* phi ./ (1 - phi);
fprintf(['Bearing: 4EI/L_end = %.3g Nm/rad; fixity %.2f -> ' ...
    'k_r = %.3g (Fernandes 1e9 = fixity %.2f)\n'], ...
    k_ref_bear, bearing_fixity_max, fix2k(bearing_fixity_max), ...
    1e9 / (1e9 + k_ref_bear));

% Physical percent is encoded in AnchorLevel/StateUID.  This makes the
% F40-M controlled anchors byte-identical in semantic identity (and therefore
% in every named EOV stream) to their F40-S counterparts.
n_tgt = numel(scour_supports);
if strcmp(state_design_kind, 'dense-scour-61x5-v1')
    if n_tgt ~= 1 || numel(Dano) ~= 61 || ...
            ~isequal(reshape(Dano, 1, []), (0:60)/100) || ...
            n_states_multi ~= 0 || n_healthy_states ~= 5 || ...
            n_anchor_levels ~= 60 || n_anchor_reps ~= 5 || ...
            n_nuisance_states ~= 0
        error('A00:DenseScourContract', ...
            ['dense-scour-61x5-v1 requires one target, Dano=61, and count ' ...
             'tuple [joint healthy levels replicas nuisance]=[0 5 60 5 0].']);
    end
    n_states = numel(Dano) * n_anchor_reps;
    DamageStates = zeros(n_states, n_supp);
    LatentBearingFixity = zeros(n_states, n_latent_bear);
    StateFamily = cell(n_states, 1);
    AnchorTarget = zeros(n_states, 1);
    AnchorLevel = zeros(n_states, 1);
    StateUID = cell(n_states, 1);
    row_ = 0;
    target_ = scour_supports(1);
    for severity_pct_ = 0:60
        severity_frac_ = severity_pct_/100;
        for rep_ = 1:n_anchor_reps
            row_ = row_ + 1;
            DamageStates(row_, target_) = severity_frac_;
            if severity_pct_ == 0
                StateFamily{row_} = 'target_healthy';
                StateUID{row_} = ttbi.state_uid( ...
                    L_bridge, num_spans, scour_supports, ...
                    'target_healthy', 0, 0, rep_);
            else
                StateFamily{row_} = 'scour_only';
                AnchorTarget(row_) = target_;
                AnchorLevel(row_) = severity_pct_;
                StateUID{row_} = ttbi.state_uid( ...
                    L_bridge, num_spans, scour_supports, ...
                    'scour_only', target_, severity_pct_, rep_);
            end
        end
    end
else
    rng(damage_seed, 'twister');

    % Anchor accumulators remain row-aligned by construction.
    anchors_s = zeros(0, n_supp);
    anchors_bf = zeros(0, 2);
    fam_ = cell(0, 1);
    atgt_ = zeros(0, 1);
    alvl_ = zeros(0, 1);
    uid_ = cell(0, 1);
    n_nuis_here = n_nuisance_states;

    if include_anchors
        % (1) Independent zero-target diagnostic states.
        anchors_s = [anchors_s; zeros(n_healthy_states, n_supp)];
        anchors_bf = [anchors_bf; zeros(n_healthy_states, 2)];
        fam_ = [fam_; repmat({'target_healthy'}, n_healthy_states, 1)];
        atgt_ = [atgt_; zeros(n_healthy_states, 1)];
        alvl_ = [alvl_; zeros(n_healthy_states, 1)];
        for rep_ = 1:n_healthy_states
            uid_{end + 1, 1} = ttbi.state_uid( ...
                L_bridge, num_spans, scour_supports, ...
                'target_healthy', 0, 0, rep_); %#ok<AGROW>
        end

        % (2) Controlled single-pier scour anchors.
        levels_s = linspace( ...
            dano_max / n_anchor_levels, dano_max, n_anchor_levels)';
        level_codes_s = round(100*levels_s);
        for ti = 1:n_tgt
            for rep_ = 1:n_anchor_reps
                block = zeros(n_anchor_levels, n_supp);
                block(:, scour_supports(ti)) = levels_s;
                anchors_s = [anchors_s; block]; %#ok<AGROW>
                anchors_bf = [ ...
                    anchors_bf; zeros(n_anchor_levels, 2)]; %#ok<AGROW>
                fam_ = [ ...
                    fam_; repmat({'scour_only'}, n_anchor_levels, 1)]; %#ok<AGROW>
                atgt_ = [ ...
                    atgt_; repmat(scour_supports(ti), n_anchor_levels, 1)]; %#ok<AGROW>
                alvl_ = [alvl_; level_codes_s]; %#ok<AGROW>
                for level_ = 1:n_anchor_levels
                    uid_{end + 1, 1} = ttbi.state_uid( ...
                        L_bridge, num_spans, scour_supports, ...
                        'scour_only', scour_supports(ti), ...
                        level_codes_s(level_), rep_); %#ok<AGROW>
                end
            end
        end

        % (3) Controlled one-abutment nominal-fixity anchors.
        levels_b = linspace(bearing_fixity_max / n_anchor_levels, ...
            bearing_fixity_max, n_anchor_levels)';
        for bi = 1:2
            for rep_ = 1:n_anchor_reps
                block = zeros(n_anchor_levels, 2);
                block(:, bi) = levels_b;
                anchors_bf = [anchors_bf; block]; %#ok<AGROW>
                anchors_s = [ ...
                    anchors_s; zeros(n_anchor_levels, n_supp)]; %#ok<AGROW>
                fam_ = [ ...
                    fam_; repmat({'bearing_only'}, n_anchor_levels, 1)]; %#ok<AGROW>
                atgt_ = [ ...
                    atgt_; repmat(bi, n_anchor_levels, 1)]; %#ok<AGROW>
                alvl_ = [alvl_; (1:n_anchor_levels)']; %#ok<AGROW>
                for level_ = 1:n_anchor_levels
                    uid_{end + 1, 1} = ttbi.state_uid( ...
                        L_bridge, num_spans, scour_supports, ...
                        'bearing_only', bi, level_, rep_); %#ok<AGROW>
                end
            end
        end

        % (4) Nuisance-only false-positive diagnostic states.
        if n_nuis_here > 0
            anchors_s = [ ...
                anchors_s; zeros(n_nuis_here, n_supp)];
            anchors_bf = [ ...
                anchors_bf; zeros(n_nuis_here, 2)];
            fam_ = [ ...
                fam_; repmat({'nuisance_only'}, n_nuis_here, 1)];
            atgt_ = [atgt_; zeros(n_nuis_here, 1)];
            alvl_ = [alvl_; zeros(n_nuis_here, 1)];
            for rep_ = 1:n_nuis_here
                uid_{end + 1, 1} = ttbi.state_uid( ...
                    L_bridge, num_spans, scour_supports, ...
                    'nuisance_only', 0, 0, rep_); %#ok<AGROW>
            end
        end
    end

    % (5) One geometry-level master LHS over scour and both latent bearings.
    lhs = lhsdesign(n_states_multi, n_tgt + n_latent_bear);
    joint_s = zeros(n_states_multi, n_supp);
    joint_s(:, scour_supports) = lhs(:, 1:n_tgt) * dano_max;
    joint_bf = lhs(:, n_tgt+1:n_tgt+n_latent_bear) * bearing_fixity_max;
    joint_uid_ = cell(n_states_multi, 1);
    for joint_ = 1:n_states_multi
        joint_uid_{joint_} = ttbi.state_uid( ...
            L_bridge, num_spans, scour_supports, ...
            'joint', 0, joint_, 1);
    end

    DamageStates = [anchors_s; joint_s];
    LatentBearingFixity = [anchors_bf; joint_bf];
    StateFamily = [fam_; repmat({'joint'}, n_states_multi, 1)];
    AnchorTarget = [atgt_; zeros(n_states_multi, 1)];
    AnchorLevel = [alvl_; zeros(n_states_multi, 1)];
    StateUID     = [uid_; joint_uid_];
end

n_states = size(DamageStates, 1);
if strcmp(state_design_kind, 'dense-scour-61x5-v1')
    expected_states_ = 61*n_anchor_reps;
else
    expected_states_ = n_healthy_states + ...
        n_tgt * n_anchor_levels * n_anchor_reps + ...
        n_latent_bear * n_anchor_levels * n_anchor_reps + ...
        n_nuisance_states + n_states_multi;
end
if n_states ~= expected_states_
    error('A00: fixed five-family inventory has %d states; expected %d.', ...
        n_states, expected_states_);
end
if numel(StateFamily) ~= n_states || ...
        numel(AnchorTarget) ~= n_states || numel(AnchorLevel) ~= n_states
    error('A00: state-family table (%d/%d/%d) does not match n_states=%d.', ...
        numel(StateFamily), numel(AnchorTarget), ...
        numel(AnchorLevel), n_states);
end
if numel(StateUID) ~= n_states || ...
        ~isequal(size(LatentBearingFixity), [n_states, n_latent_bear]) || ...
        numel(unique(StateUID)) ~= n_states
    error(['A00: semantic state identity/latent-bearing table is malformed ' ...
        '(%d UIDs, %d unique, latent size %s, n_states=%d).'], ...
        numel(StateUID), numel(unique(StateUID)), ...
        mat2str(size(LatentBearingFixity)), n_states);
end

StateSeedID = ttbi.state_seed_ids(StateUID, damage_seed);
[StateNamedStreamSeedID, PassageNamedStreamSeedID] = ...
    ttbi.named_stream_seed_ids(StateSeedID, StateUID, Npass, ...
        random_stream_schedule_version, ...
        state_stream_names, passage_stream_names);
PassageNamedStreamSeedIDFlat = reshape( ...
    PassageNamedStreamSeedID, n_states, []);

% ---- Per-state crack ACTIVATION, keyed by stable semantic UID ------------
% Controlled anchors are crack-off, nuisance-only is crack-on, and joint rows
% use the stable semantic UID. The rung toggle activates this fixed latent draw.
LatentCrackOn = false(n_states, 1);
joint_indices_ = find(strcmp(StateFamily, 'joint'));
for state_ = reshape(joint_indices_, 1, [])
    LatentCrackOn(state_) = ttbi.state_uniform(StateUID{state_}, ...
        damage_seed, 'latent-crack-v1') <= crack_p;
end
LatentCrackOn(strcmp(StateFamily, 'nuisance_only')) = true;
CrackOn = logical(use_crack_eov) & LatentCrackOn;
if use_crack_eov && ~strcmp(crack_draw, 'per_state')
    error(['A00: nuisance_only anchor states require crack_draw=''per_state'' ' ...
        '(the family design forces per-state crack activation); ' ...
        '''per_passage'' is deprecated and incompatible with Feature A.']);
end
if strcmp(profile_mode, 'psd_fra') && ~strcmp(profile_draw, 'per_state')
    error(['A00: generation-rules-v8 requires profile_draw=''per_state''; ' ...
        'the deprecated per-passage profile branch is outside the CRN design.']);
end
if use_track_eov && ~strcmp(track_draw, 'per_state')
    error(['A00: generation-rules-v8 requires track_draw=''per_state''; ' ...
        'the deprecated per-passage track branch is outside the CRN design.']);
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
            error(['A00: bearing_mode=''target'' requires ' ...
                'damage_mode=''multi_scour''.']);
        end
        BearingFixity = LatentBearingFixity;
        BearingStates = fix2k(BearingFixity);
    otherwise
        error('A00: unknown bearing_mode "%s"', bearing_mode);
end

design = struct();
design.n_supp = n_supp;
design.n_latent_bear = n_latent_bear;
design.k_ref_bear = k_ref_bear;
design.state_design_kind = state_design_kind;
design.state_identity_version = state_identity_version;
design.joint_lhs_design = joint_lhs_design;
design.random_stream_schedule_version = random_stream_schedule_version;
design.state_stream_names = state_stream_names;
design.passage_stream_names = passage_stream_names;
design.DamageStates = DamageStates;
design.LatentBearingFixity = LatentBearingFixity;
design.StateFamily = StateFamily;
design.AnchorTarget = AnchorTarget;
design.AnchorLevel = AnchorLevel;
design.StateUID = StateUID;
design.n_states = n_states;
design.StateSeedID = StateSeedID;
design.StateNamedStreamSeedID = StateNamedStreamSeedID;
design.PassageNamedStreamSeedID = PassageNamedStreamSeedID;
design.PassageNamedStreamSeedIDFlat = PassageNamedStreamSeedIDFlat;
design.LatentCrackOn = LatentCrackOn;
design.CrackOn = CrackOn;
design.BearingStates = BearingStates;
design.BearingFixity = BearingFixity;
end
