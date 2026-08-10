function Damage = contact_damage_descriptor( ...
        case_info, data, state_index, passage_index, descriptor_contract)
%CONTACT_DAMAGE_DESCRIPTOR Rebuild a production Damage struct from saved data.

Damage = struct();
Damage.desvio = 0;
Damage.scour_rates = double(data.scour_vector(:)');
if isfield(data, 'bearing_vector') && numel(data.bearing_vector) >= 2
    bearing = double(data.bearing_vector(:)');
else
    bearing = [0, 0];
end
Damage.bearing_left = bearing(1);
Damage.bearing_right = bearing(2);

crack_row = double(data.crack_log(passage_index, :));
if numel(crack_row) >= 2 && crack_row(2) > 0
    Damage.crack_locs = crack_row(1);
    Damage.crack_intensity = crack_row(2);
    if numel(crack_row) >= 3
        Damage.crack_lc = crack_row(3);
    else
        Damage.crack_lc = 0;
    end
else
    Damage.crack_locs = [];
    Damage.crack_intensity = [];
    Damage.crack_lc = 0;
end

profile_mode = contact_state_text(data, 'profile_mode', ...
    contact_case_text(case_info, 'profile_mode', 'fixed'));
if strcmp(profile_mode, 'fixed_scaled')
    Damage.profile_intensity = double(data.profile_log(passage_index));
end

track_value = contact_indexed_value(data.track_log, passage_index);
if isempty(track_value)
    Damage.track = [];
else
    Damage.track = track_value;
end

oor_value = contact_indexed_value(data.oor_log, passage_index);
Damage.oor_flats = [];
Damage.oor_poly = [];
if isstruct(oor_value)
    if isfield(oor_value, 'flats'), Damage.oor_flats = oor_value.flats; end
    if isfield(oor_value, 'poly'), Damage.oor_poly = oor_value.poly; end
end
if isfield(case_info, 'oor_radius')
    Damage.oor_radius = double(case_info.oor_radius);
else
    Damage.oor_radius = 0.46;
end

% Retain these for report/debug provenance; B00 ignores them.
Damage.contact_closure_state = state_index;
Damage.contact_closure_passage = passage_index;
Damage.contact_closure_state_uid = descriptor_contract.state_uid;
end
