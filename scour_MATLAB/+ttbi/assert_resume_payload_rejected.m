function assert_resume_payload_rejected( ...
        data, state_index, label, context)
%ASSERT_RESUME_PAYLOAD_REJECTED Require one payload mutation to fail closed.

rejected = false;
try
    ttbi.validate_resumed_state_payload( ...
        data, state_index, ['mutation:' label], context);
catch
    rejected = true;
end
assert(rejected, ...
    'Resume payload validator accepted mutation: %s', label);
end
