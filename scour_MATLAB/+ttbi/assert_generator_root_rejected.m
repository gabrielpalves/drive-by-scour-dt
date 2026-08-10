function assert_generator_root_rejected(description, expected_identifier)
%ASSERT_GENERATOR_ROOT_REJECTED Require one source-boundary mutation to fail.

if nargin < 2
    expected_identifier = '';
end

rejected = false;
try
    generator_source_root();
catch root_error
    if ~isempty(expected_identifier)
        assert(strcmp(root_error.identifier, expected_identifier), ...
            'ttbi:SmokeWrongGeneratorRootError', ...
            'Expected %s, got %s: %s', expected_identifier, ...
            root_error.identifier, root_error.message);
    end
    rejected = true;
end
assert(rejected, ...
    'Generator source authentication accepted %s.', description);
end
