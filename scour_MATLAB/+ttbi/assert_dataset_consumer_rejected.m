function assert_dataset_consumer_rejected( ...
        folder, n_states, description, expected_identifier)
%ASSERT_DATASET_CONSUMER_REJECTED Require the MATLAB digest consumer to fail.

if nargin < 4
    expected_identifier = '';
end

rejected = false;
try
    validate_dataset_digest_manifest(folder, n_states);
catch consumer_error
    if ~isempty(expected_identifier)
        assert(strcmp(consumer_error.identifier, expected_identifier), ...
            'ttbi:SmokeWrongConsumerError', ...
            'Expected %s, got %s: %s', expected_identifier, ...
            consumer_error.identifier, consumer_error.message);
    end
    rejected = true;
end
assert(rejected, ...
    'MATLAB dataset consumer accepted %s.', description);
end
