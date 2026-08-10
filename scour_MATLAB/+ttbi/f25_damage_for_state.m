function Damage = f25_damage_for_state(design, state_index)
%F25_DAMAGE_FOR_STATE Materialize one exact F25 scenario for the solver.

if ~isstruct(design) || ~isscalar(design) || ...
        ~isfield(design, 'schema') || ...
        ~strcmp(design.schema, 'f25-state-design-v1')
    error('ttbi:f25_damage_for_state:Design', ...
        'design must be one f25-state-design-v1 scalar struct.');
end
if ~isnumeric(state_index) || ~isscalar(state_index) || ...
        ~isfinite(state_index) || state_index ~= fix(state_index) || ...
        state_index < 1 || state_index > design.n_states
    error('ttbi:f25_damage_for_state:Index', ...
        'state_index must be an integer in 1..%d.', design.n_states);
end

Damage = struct();
Damage.scour_rates = design.DamageStates(state_index, :);
Damage.bearing_left = design.BearingStates(state_index, 1);
Damage.bearing_right = design.BearingStates(state_index, 2);
Damage.desvio = 0;
if design.CrackOn(state_index)
    Damage.crack_locs = design.CrackLocation(state_index);
    Damage.crack_intensity = design.CrackIntensity(state_index);
    Damage.crack_lc = design.CrackHalfLength(state_index);
else
    Damage.crack_locs = [];
    Damage.crack_intensity = [];
    Damage.crack_lc = 0;
end
end
