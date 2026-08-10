function arm = dry_ballast_stiffness_arm(value)
%DRY_BALLAST_STIFFNESS_ARM Resolve and validate the sign-sensitivity arm.
%
% ARM = ttbi.dry_ballast_stiffness_arm(CONFIG) reads the optional
% CONFIG.ballast_dry_stiffness_arm field.  Its absence means the unchanged
% campaign default, 'retained-stiffening'.  Passing scalar text validates an
% explicit sensitivity-arm request.  No prefix, abbreviation, or case-folded
% alias is accepted: a typo must fail before any random draw is consumed.

if isstruct(value)
    if ~isscalar(value)
        error('ttbi:dry_ballast_stiffness_arm:InvalidConfig', ...
            'Track configuration must be a scalar struct.');
    end
    if ~isfield(value, 'ballast_dry_stiffness_arm')
        arm = 'retained-stiffening';
        return;
    end
    value = value.ballast_dry_stiffness_arm;
end

if isstring(value)
    if ~isscalar(value) || ismissing(value)
        error('ttbi:dry_ballast_stiffness_arm:InvalidType', ...
            'Sensitivity arm must be one nonmissing text scalar.');
    end
    value = char(value);
end
if ~ischar(value) || ~isrow(value) || isempty(value)
    error('ttbi:dry_ballast_stiffness_arm:InvalidType', ...
        'Sensitivity arm must be one nonempty text scalar.');
end

allowed = {'retained-stiffening', 'reciprocal-softening'};
if ~any(strcmp(value, allowed))
    error('ttbi:dry_ballast_stiffness_arm:UnsupportedArm', ...
        ['Unknown dry-ballast stiffness arm "%s". Allowed values are ' ...
         'retained-stiffening and reciprocal-softening.'], value);
end
arm = value;
end
