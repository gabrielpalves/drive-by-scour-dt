function assert_publication_credentials_absent(folder, scenario)
%ASSERT_PUBLICATION_CREDENTIALS_ABSENT Check every final/temporary path entry.

credential_names = ttbi.generation_publication_credential_names();
for name_index = 1:numel(credential_names)
    name = credential_names{name_index};
    assert(~ttbi.path_entry_exists(fullfile(folder, name)), ...
        'ttbi:SmokeStaleCredential', ...
        '%s left publication credential %s behind.', scenario, name);
end
end
