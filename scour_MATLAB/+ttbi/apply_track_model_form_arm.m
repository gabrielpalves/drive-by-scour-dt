function Track = apply_track_model_form_arm(Track,arm_id)
%APPLY_TRACK_MODEL_FORM_ARM Apply one registered parameter-only track arm.
%
% The input must be a fresh inherited A02_Track baseline.  Requiring that
% boundary prevents accidental arm compounding and makes each comparison a
% one-factor intervention against the same declared production properties.

if nargin ~= 2 || ~isstruct(Track) || ~isscalar(Track) || ...
        ~(ischar(arm_id) || (isstring(arm_id) && isscalar(arm_id)))
    error('ttbi:apply_track_model_form_arm:Input', ...
        'Expected one scalar Track struct and one scalar arm identifier.');
end
arm_id = char(arm_id);
arms = ttbi.track_model_form_arms();
index = find(strcmp({arms.id},arm_id));
if numel(index) ~= 1
    error('ttbi:apply_track_model_form_arm:UnknownArm', ...
        'Unknown arm "%s"; expected one of: %s.', ...
        arm_id,strjoin({arms.id},', '));
end

local_assert_inherited_baseline(Track);
source_0545 = ttbi.zhai_ballast_properties(0.545);
realized_spacing = source_0545;

switch arm_id
    case 'baseline-hybrid-v1'
        % Explicit no-op: inherited production baseline.

    case 'consistent-one-seat-v1'
        Track.Rail.Prop.I = Track.Rail.Prop.I/2;
        Track.Rail.Prop.rho = Track.Rail.Prop.rho/2;
        Track.Sleeper.Prop.m = Track.Sleeper.Prop.m/2;

    case 'consistent-two-rail-v1'
        Track.Pad.Prop.k = 2*Track.Pad.Prop.k;
        Track.Pad.Prop.c = 2*Track.Pad.Prop.c;
        Track.Ballast.Prop.m = 2*Track.Ballast.Prop.m;
        Track.Ballast.Prop.k = 2*Track.Ballast.Prop.k;
        Track.Ballast.Prop.c = 2*Track.Ballast.Prop.c;
        Track.SubBallast.Prop.k = 2*Track.SubBallast.Prop.k;
        Track.SubBallast.Prop.c = 2*Track.SubBallast.Prop.c;
        Track.BallastOnBeam.Prop.m = 2*Track.BallastOnBeam.Prop.m;
        Track.BallastOnBeam.Prop.k = 2*Track.BallastOnBeam.Prop.k;
        Track.BallastOnBeam.Prop.c = 2*Track.BallastOnBeam.Prop.c;

    case 'spacing-consistent-0p600-v1'
        realized_spacing = ttbi.zhai_ballast_properties(0.600);
        Track.Ballast.Prop.m = ...
            realized_spacing.effective_ballast_mass_kg;
        Track.Ballast.Prop.k = ...
            realized_spacing.effective_ballast_stiffness_n_m;
        Track.SubBallast.Prop.k = ...
            realized_spacing.effective_subgrade_stiffness_n_m;
        Track.BallastOnBeam.Prop.m = Track.Ballast.Prop.m;
        Track.BallastOnBeam.Prop.k = Track.Ballast.Prop.k;

    case 'rail-damping-0p050pct-v1'
        Track.Rail.Damping.per = 0.050;

    case 'rail-damping-0p200pct-v1'
        Track.Rail.Damping.per = 0.200;
end

Track.ModelForm = struct( ...
    'schema','paper1-track-model-form-arm-v1', ...
    'arm_id',arm_id, ...
    'family',arms(index).family, ...
    'description',arms(index).description, ...
    'production_candidate',logical(arms(index).production_candidate), ...
    'production_baseline_arm_id','baseline-hybrid-v1', ...
    'source_spacing_properties',source_0545, ...
    'realized_spacing_properties',realized_spacing);
end

function local_assert_inherited_baseline(Track)
required = { ...
    {'Rail','Prop','I',2*3.217e-5}; ...
    {'Rail','Prop','rho',2*60.64}; ...
    {'Rail','Damping','per',0.1}; ...
    {'Pad','Prop','k',6.5e7}; ...
    {'Pad','Prop','c',7.5e4}; ...
    {'Sleeper','spacing','',0.6}; ...
    {'Sleeper','Prop','m',2*125.5}; ...
    {'Ballast','Prop','m',531.4}; ...
    {'Ballast','Prop','k',137.75e6}; ...
    {'Ballast','Prop','c',5.88e4}; ...
    {'SubBallast','Prop','k',77.5e6}; ...
    {'SubBallast','Prop','c',3.115e4}};
for k = 1:size(required,1)
    path = required{k};
    if isempty(path{3})
        value = Track.(path{1}).(path{2});
    else
        value = Track.(path{1}).(path{2}).(path{3});
    end
    expected = path{4};
    if ~isscalar(value) || ~isfinite(value) || value ~= expected
        error('ttbi:apply_track_model_form_arm:NotFreshBaseline', ...
            'Track.%s is not the registered inherited baseline.', ...
            strjoin(path(1:3),'.'));
    end
end
if ~isequal(Track.BallastOnBeam.Prop,Track.Ballast.Prop)
    error('ttbi:apply_track_model_form_arm:NotFreshBaseline', ...
        'Ballast-on-beam properties do not match the inherited baseline.');
end
end
