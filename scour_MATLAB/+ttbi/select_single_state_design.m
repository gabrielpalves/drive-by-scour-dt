function selected = select_single_state_design(design, state_index)
%SELECT_SINGLE_STATE_DESIGN Extract one row while retaining design metadata.
%
% This is a focused-smoke utility. The returned design can be passed through
% the production identity/context/sidecar builders, so the one-state fixture
% has one internally coherent configuration identity instead of inherited
% full-campaign stamps.

validateattributes(state_index, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', '>=', 1, ...
     '<=', design.n_states}, mfilename, 'state_index');

selected = design;
selected.n_states = 1;
row_fields = { ...
    'DamageStates'; 'LatentBearingFixity'; 'StateFamily'; ...
    'AnchorTarget'; 'AnchorLevel'; 'StateUID'; 'StateSeedID'; ...
    'StateNamedStreamSeedID'; 'PassageNamedStreamSeedIDFlat'; ...
    'LatentCrackOn'; 'CrackOn'; 'BearingStates'; 'BearingFixity'};
for field_index = 1:numel(row_fields)
    field_name = row_fields{field_index};
    selected.(field_name) = design.(field_name)(state_index, :);
end
selected.PassageNamedStreamSeedID = ...
    design.PassageNamedStreamSeedID(state_index, :, :);
end
