function smoke_generation_output_boundary(scratch, context)
%SMOKE_GENERATION_OUTPUT_BOUNDARY Reject a linked publication root pre-delete.
%
% The target contains all four publication credentials. Calling the real
% publisher through a junction/symlink must fail on directory authentication
% before the revoker can delete any target entry or write a new artifact.

target_folder = fullfile(scratch, 'linked_output_target');
mkdir(target_folder);
ttbi.seed_stale_generation_credentials(target_folder);

alias_folder = fullfile(scratch, 'linked_output_alias');
ttbi.create_directory_alias(alias_folder, target_folder);
alias_cleanup = onCleanup( ...
    @() ttbi.delete_file_entry_if_present(alias_folder));

linked_context = context;
linked_context.run_folder = alias_folder;
linked_context.run_folder_observation = ...
    ttbi.directory_observation(target_folder);

rejected = false;
try
    ttbi.publish_generation_completion(alias_folder, linked_context);
catch boundary_error
    assert(strcmp(boundary_error.identifier, ...
        'ttbi:DirectoryObservationLinked'), ...
        'ttbi:SmokeWrongOutputBoundaryError', ...
        'Expected linked-root rejection, got %s: %s', ...
        boundary_error.identifier, boundary_error.message);
    rejected = true;
end
assert(rejected, 'ttbi:SmokeLinkedOutputAccepted', ...
    'Generation publisher accepted a linked output directory.');

credential_names = ttbi.generation_publication_credential_names();
for name_index = 1:numel(credential_names)
    assert(ttbi.path_entry_exists( ...
        fullfile(target_folder, credential_names{name_index})), ...
        'ttbi:SmokeLinkedOutputDeletedTarget', ...
        'Linked-root rejection deleted target credential %s.', ...
        credential_names{name_index});
end

clear alias_cleanup
ttbi.delete_file_entry_if_present(alias_folder);
end
