function descriptor = contact_descriptor_summary( ...
        case_info, data, passage_index, descriptor_contract)
%CONTACT_DESCRIPTOR_SUMMARY Summarize the reconstructed passage provenance.

recon = contact_study_reconstruction();
descriptor = struct();
descriptor.L_bridge_m = double(case_info.L_bridge_m);
descriptor.num_spans = double(case_info.num_spans);
descriptor.velocity_kmh = double(data.Velocidade(passage_index)) * 3.6;
descriptor.temperature_C = double(data.Temperatura(passage_index));
descriptor.scour_vector = double(data.scour_vector(:)');
if isfield(data, 'bearing_vector')
    descriptor.bearing_vector_Nm_rad = double(data.bearing_vector(:)');
else
    descriptor.bearing_vector_Nm_rad = [0, 0];
end
descriptor.crack_row = double(data.crack_log(passage_index, :));
descriptor.profile_mode = recon.state_text(data, 'profile_mode', ...
    recon.case_text(case_info, 'profile_mode', 'fixed'));
descriptor.profile_value = double(data.profile_log(passage_index));
descriptor.profile_phase_seed = descriptor_contract.profile_phase_seed;
descriptor.profile_phase_stream_index = ...
    descriptor_contract.profile_phase_stream_index;
descriptor.state_uid = descriptor_contract.state_uid;
descriptor.state_family = descriptor_contract.state_family;
descriptor.has_track_eov = ...
    ~isempty(recon.indexed_value(data.track_log, passage_index));
oor = recon.indexed_value(data.oor_log, passage_index);
descriptor.n_flats = 0;
descriptor.n_polygonization = 0;
if isstruct(oor)
    if isfield(oor, 'flats'), descriptor.n_flats = size(oor.flats, 1); end
    if isfield(oor, 'poly')
        descriptor.n_polygonization = size(oor.poly, 1);
    end
end
end
