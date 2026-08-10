function assert_generation_publication_rejected( ...
        folder, context, description)
%ASSERT_GENERATION_PUBLICATION_REJECTED Exercise one negative publication case.

rejected = false;
try
    ttbi.publish_generation_completion(folder, context);
catch
    rejected = true;
end
assert(rejected, ...
    'Generation publication accepted %s.', description);
ttbi.assert_publication_credentials_absent( ...
    folder, ['Rejected ' description]);
end
