function smoke_credential_revocation_failure(scratch)
%SMOKE_CREDENTIAL_REVOCATION_FAILURE Prove best-effort, nonrecursive cleanup.

folder = fullfile(scratch, 'credential_directory_rejection');
mkdir(folder);
ttbi.seed_stale_generation_credentials(folder);
names = ttbi.generation_publication_credential_names();
folder_observation = ttbi.directory_observation(folder);

blocked_path = fullfile(folder, names{1});
ttbi.delete_file_entry_if_present(blocked_path);
mkdir(blocked_path);

rejected = false;
try
    ttbi.revoke_generation_publication(folder, folder_observation);
catch cleanup_error
    assert(strcmp(cleanup_error.identifier, ...
        'ttbi:CredentialPathDirectory'), ...
        'ttbi:SmokeWrongCredentialError', ...
        'Expected directory rejection, got %s: %s', ...
        cleanup_error.identifier, cleanup_error.message);
    rejected = true;
end
assert(rejected, 'ttbi:SmokeCredentialDirectoryAccepted', ...
    'Credential revocation accepted a directory path.');
assert(isfolder(blocked_path), ...
    'Credential revocation recursively removed the blocking directory.');
for name_index = 2:numel(names)
    assert(~ttbi.path_entry_exists(fullfile(folder, names{name_index})), ...
        'ttbi:SmokeCredentialCleanupAborted', ...
        'Credential %s survived an independent directory error.', ...
        names{name_index});
end

rmdir(blocked_path);
ttbi.assert_publication_credentials_absent( ...
    folder, 'Best-effort credential revocation smoke');
end
