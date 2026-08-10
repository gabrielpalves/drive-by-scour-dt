function publish_generation_completion(run_folder, context)
%PUBLISH_GENERATION_COMPLETION Authenticate and atomically publish a dataset.
%
% Publication is fail-closed. Exact state inventory, full state semantics,
% source identity, and artifact bytes are checked across both sides of digest
% publication. Consumers still verify the published digest root, closing any
% later mutation after the marker appears.

n_states = context.n_states;
provenance = context.provenance;
run_folder_observation = context.run_folder_observation;
marker_path = fullfile(run_folder, '_GENERATION_COMPLETE');
digest_path = fullfile(run_folder, 'file_digests.mat');
temporary_digest = fullfile(run_folder, '.file_digests.mat.tmp');
temporary_marker = fullfile(run_folder, '._GENERATION_COMPLETE.tmp');

% Revoke every old publication credential before the first operation that can
% return or throw. This makes the helper fail-closed even when called directly,
% outside A00_Run's normal marker revocation path.
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
ttbi.revoke_generation_publication( ...
    run_folder, run_folder_observation);

state_paths = ttbi.inspect_numbered_state_inventory(run_folder, n_states);
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
present = sum(~cellfun('isempty', state_paths));
if present ~= n_states
    fprintf(['Generation INCOMPLETE: %d/%d states present - NO completion ' ...
        'marker written. Re-run A00 in this folder to finish.\n'], ...
        present, n_states);
    return;
end

% Snapshot bytes before semantic validation, then require the same snapshot
% afterwards. A concurrent state/sidecar edit cannot hide inside validation.
ttbi.assert_generator_source_unchanged(provenance);
ttbi.validate_generation_sidecars(run_folder, context);
[digest_lines_before, digest_root_before] = ...
    ttbi.generation_artifact_digests(run_folder, n_states);
completed = ttbi.validate_resume_states(run_folder, context);
if ~all(completed)
    error('ttbi:GenerationSemanticInventory', ...
        'Full semantic validation did not classify all %d states.', n_states);
end
[digest_lines, digest_root] = ...
    ttbi.generation_artifact_digests(run_folder, n_states);
if ~strcmp(digest_lines_before, digest_lines) || ...
        ~strcmp(digest_root_before, digest_root)
    error('ttbi:GenerationArtifactRace', ...
        'Generation artifacts changed during semantic validation.');
end

file_digests = struct( ...
    'schema', 'source-digests-v2', ...
    'scope', 'NNNN.mat+case_info.mat+damage_states.mat', ...
    'digest_lines', digest_lines, ...
    'root', digest_root);
try
    ttbi.assert_generation_output_directory( ...
        run_folder, run_folder_observation);
    save(temporary_digest, 'file_digests');
    ttbi.assert_generation_output_directory( ...
        run_folder, run_folder_observation);
    [moved, move_message] = movefile(temporary_digest, digest_path, 'f');
    if ~moved
        error('ttbi:GenerationDigestPublish', ...
            'Could not atomically publish file_digests.mat: %s', move_message);
    end
    % Close both source and artifact boundaries once more after serializing
    % the digest table, immediately before creating the completion marker.
    ttbi.assert_generator_source_unchanged(provenance);
    ttbi.validate_generation_sidecars(run_folder, context);
    completed_after = ttbi.validate_resume_states(run_folder, context);
    if ~all(completed_after)
        error('ttbi:GenerationSemanticRevalidation', ...
            'A state disappeared during publication revalidation.');
    end
    [digest_lines_after, digest_root_after] = ...
        ttbi.generation_artifact_digests(run_folder, n_states);
    if ~strcmp(digest_lines_after, digest_lines) || ...
            ~strcmp(digest_root_after, digest_root)
        error('ttbi:GenerationArtifactRace', ...
            'Generation artifacts changed while the digest table was published.');
    end

    ttbi.write_generation_marker_temp( ...
        temporary_marker, provenance, digest_root);

    % This is intentionally the last expensive operation before the atomic
    % marker rename. Earlier fences bracket semantic/digest work; this one
    % closes the source window after that work and after marker bytes are
    % flushed, leaving only the single same-directory rename.
    ttbi.assert_generator_source_unchanged(provenance);
    ttbi.assert_generation_output_directory( ...
        run_folder, run_folder_observation);
    [moved, move_message] = movefile(temporary_marker, marker_path, 'f');
    if ~moved
        error('ttbi:CompletionMarkerPublish', ...
            'Could not publish the atomic completion marker: %s', move_message);
    end
    ttbi.assert_generation_output_directory( ...
        run_folder, run_folder_observation);
catch publication_error
    try
        ttbi.revoke_generation_publication( ...
            run_folder, run_folder_observation);
    catch cleanup_error
        publication_error = addCause(publication_error, cleanup_error);
    end
    rethrow(publication_error);
end

fprintf(['Generation COMPLETE: %d/%d states -> wrote ' ...
    '_GENERATION_COMPLETE + file_digests.\n'], present, n_states);
end
