function smoke_rejected_publication( ...
        valid_folder, scratch, publication_context)
%SMOKE_REJECTED_PUBLICATION Prove stale credentials are always revoked.

ttbi.smoke_credential_revocation_failure(scratch);

% The validator must inspect the caller's original directory spelling before
% canonicalization can erase a symlink/junction. The alias is deleted as one
% no-follow entry, never recursively through its target.
dataset_alias = fullfile(scratch, 'published_dataset_alias');
ttbi.create_directory_alias(dataset_alias, valid_folder);
dataset_alias_cleanup = onCleanup( ...
    @() ttbi.delete_file_entry_if_present(dataset_alias));
ttbi.assert_dataset_consumer_rejected( ...
    dataset_alias, 1, 'a linked dataset root', ...
    'dataset_digest_manifest:LinkedDataset');
clear dataset_alias_cleanup
ttbi.delete_file_entry_if_present(dataset_alias);

% The downstream MATLAB consumer shares the hard-link guard. Temporarily add
% an alias to an otherwise valid published state and require rejection.
consumer_alias = fullfile(scratch, 'consumer_state_hardlink_alias.mat');
ttbi.create_hardlink( ...
    consumer_alias, fullfile(valid_folder, '0001.mat'));
consumer_alias_cleanup = onCleanup(@() delete(consumer_alias));
ttbi.assert_dataset_consumer_rejected( ...
    valid_folder, 1, 'a hard-linked authenticated state', ...
    'dataset_digest_manifest:BadEntry');
clear consumer_alias_cleanup

% An incomplete folder returns normally, but must first lose both old
% credentials. This covers the direct-helper path that A00_Run does not own.
missing_folder = fullfile(scratch, 'missing_state_publication');
mkdir(missing_folder);
ttbi.copy_generation_publication_inputs( ...
    valid_folder, missing_folder, false);
ttbi.seed_stale_generation_credentials(missing_folder);
publication_context.run_folder = missing_folder;
publication_context.run_folder_observation = ...
    ttbi.directory_observation(missing_folder);
ttbi.publish_generation_completion(missing_folder, publication_context);
ttbi.assert_publication_credentials_absent( ...
    missing_folder, 'Missing-state publication');

corrupt_folder = fullfile(scratch, 'corrupt_state_publication');
mkdir(corrupt_folder);
ttbi.copy_generation_publication_inputs( ...
    valid_folder, corrupt_folder, true);
container = load(fullfile(corrupt_folder, '0001.mat'));
container.data.crop_start(1) = 1002;
save(fullfile(corrupt_folder, '0001.mat'), '-struct', 'container');
ttbi.seed_stale_generation_credentials(corrupt_folder);
publication_context.run_folder = corrupt_folder;
publication_context.run_folder_observation = ...
    ttbi.directory_observation(corrupt_folder);
ttbi.assert_generation_publication_rejected( ...
    corrupt_folder, publication_context, ...
    'a stamped payload with a wrong crop');

sidecar_folder = fullfile(scratch, 'corrupt_sidecar_publication');
mkdir(sidecar_folder);
ttbi.copy_generation_publication_inputs( ...
    valid_folder, sidecar_folder, true);
catalogue = load(fullfile(sidecar_folder, 'damage_states.mat'));
catalogue.DamageStates(1, 1) = catalogue.DamageStates(1, 1) + 1;
save(fullfile(sidecar_folder, 'damage_states.mat'), ...
    '-struct', 'catalogue');
ttbi.seed_stale_generation_credentials(sidecar_folder);
publication_context.run_folder = sidecar_folder;
publication_context.run_folder_observation = ...
    ttbi.directory_observation(sidecar_folder);
ttbi.assert_generation_publication_rejected( ...
    sidecar_folder, publication_context, 'a foreign state catalogue');

% A second directory entry to the same state bytes is a mutable alias and
% therefore outside the publication contract on both Windows and Linux.
hardlink_folder = fullfile(scratch, 'hardlink_state_publication');
mkdir(hardlink_folder);
ttbi.copy_generation_publication_inputs( ...
    valid_folder, hardlink_folder, true);
state_path = fullfile(hardlink_folder, '0001.mat');
ttbi.create_hardlink( ...
    fullfile(scratch, 'state_hardlink_alias.mat'), state_path);
assert(~ttbi.regular_nonsymlink_file(state_path), ...
    'Hard-linked state unexpectedly passed the regular-file boundary.');
ttbi.seed_stale_generation_credentials(hardlink_folder);
publication_context.run_folder = hardlink_folder;
publication_context.run_folder_observation = ...
    ttbi.directory_observation(hardlink_folder);
ttbi.assert_generation_publication_rejected( ...
    hardlink_folder, publication_context, 'a hard-linked state file');
end
