function [digest_lines, digest_root] = ...
        generation_artifact_digests(run_folder, n_states)
%GENERATION_ARTIFACT_DIGESTS Hash the exact generated-dataset publication set.

state_names = arrayfun(@(index) sprintf('%04d.mat', index), ...
    (1:n_states)', 'UniformOutput', false);
artifact_names = [state_names; {'case_info.mat'; 'damage_states.mat'}];
digest_entries = cell(numel(artifact_names), 1);
for file_index = 1:numel(artifact_names)
    artifact_path = fullfile(run_folder, artifact_names{file_index});
    if ~ttbi.regular_nonsymlink_file(artifact_path)
        error('ttbi:GenerationArtifactMissing', ...
            ['Required generation artifact is missing, nonregular, or ' ...
             'has a symlink/junction/hard-link alias: %s'], artifact_path);
    end
    digest_entries{file_index} = sprintf('%s:%s', ...
        artifact_names{file_index}, ...
        ttbi.stable_file_sha256(artifact_path));
end
digest_lines = strjoin(sort(digest_entries), newline);
digest_root = ttbi.sha256(digest_lines);
end
