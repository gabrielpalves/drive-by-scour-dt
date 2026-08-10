function validate_resumed_state_payload( ...
        data, state_index, file_name, context)
%VALIDATE_RESUMED_STATE_PAYLOAD Authenticate one complete state payload.
%
% The orchestration stays intentionally short. Each scientific responsibility
% lives in a separate validator that is also exercised by the focused worker
% smoke: provenance, semantic identity, RAW geometry, signals, named-stream
% metadata, and the signed bilateral-contact diagnostic.

if ~isstruct(data) || ~isscalar(data)
    error('ttbi:ResumePayloadStruct', ...
        ['A00 RESUME ABORTED: state %d file "%s" does not contain one ' ...
         'scalar data struct. Delete it or use a FRESH folder.'], ...
        state_index, file_name);
end

expected_fields = sort(ttbi.state_payload_fields());
observed_fields = sort(fieldnames(data));
if ~isequal(observed_fields, expected_fields)
    missing = setdiff(expected_fields, observed_fields);
    unexpected = setdiff(observed_fields, expected_fields);
    error('ttbi:ResumePayloadFields', ...
        ['A00 RESUME ABORTED: state %d file "%s" has a noncanonical ' ...
         'payload inventory (missing={%s}; unexpected={%s}).'], ...
        state_index, file_name, strjoin(missing, ', '), ...
        strjoin(unexpected, ', '));
end

ttbi.validate_state_provenance(data, state_index, file_name, context);
ttbi.validate_state_identity(data, state_index, file_name, context);
ttbi.validate_state_metadata(data, state_index, file_name, context);
ttbi.validate_state_raw_metadata(data, state_index, file_name, context);
ttbi.validate_state_signals(data, state_index, file_name, context.Npass);
ttbi.validate_state_contact(data, state_index, file_name, context);
end
