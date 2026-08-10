function revoke_generation_publication(run_folder, run_folder_observation)
%REVOKE_GENERATION_PUBLICATION Remove all old and temporary credentials.
%
% A caller invokes this before any inventory/semantic check. Consequently an
% absent or corrupt state/sidecar can never leave a stale marker or stale
% digest table that appears to describe the current folder.

credential_names = ttbi.generation_publication_credential_names();
first_error = cell(0, 1);
for name_index = 1:numel(credential_names)
    try
        ttbi.assert_generation_output_directory( ...
            run_folder, run_folder_observation);
        ttbi.delete_file_entry_if_present( ...
            fullfile(run_folder, credential_names{name_index}));
        ttbi.assert_generation_output_directory( ...
            run_folder, run_folder_observation);
    catch cleanup_error
        % Keep trying every independent credential. A directory at one path
        % is never removed recursively, but it must not preserve other stale
        % files by aborting the cleanup loop early.
        if isempty(first_error)
            first_error = {cleanup_error};
        end
    end
end
if ~isempty(first_error)
    rethrow(first_error{1});
end
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
end
