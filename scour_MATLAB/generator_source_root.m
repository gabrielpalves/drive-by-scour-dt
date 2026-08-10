function [root_sha256, digest_lines, file_count] = generator_source_root()
%GENERATOR_SOURCE_ROOT Authenticate the complete reviewed MATLAB source set.
%
% The tracked repository manifest defines reviewed files. A separate disk
% inventory rejects unmanifested executable shadows, and MATLAB's resolver
% must map every reviewed function/script to that exact file. Two independently
% read manifest/inventory/source snapshots must agree before a root is returned.

[repository_root, entries] = ttbi.reviewed_source_entries();
selected = sort(entries(startsWith(entries, 'scour_MATLAB/')));
if isempty(selected)
    error('generator_source_root:Empty', ...
        'Manifest contains no scour_MATLAB/ source or asset.');
end

ttbi.assert_no_shadow_matlab_sources( ...
    repository_root, selected);
ttbi.assert_reviewed_matlab_resolution( ...
    repository_root, selected);
first_lines = ttbi.hash_reviewed_source_entries( ...
    repository_root, selected);

% Re-open every boundary rather than trusting path names captured above.
% File-key guarded reads plus this independent inventory close replacement
% and ordinary ABA races compatible with the modular MATLAB layout.
[confirmed_root, confirmed_entries] = ttbi.reviewed_source_entries();
if ~strcmp(ttbi.canonical_execution_path(repository_root), ...
        ttbi.canonical_execution_path(confirmed_root)) || ...
        ~isequal(entries, confirmed_entries)
    error('generator_source_root:ManifestRace', ...
        'Reviewed source manifest changed during source authentication.');
end
ttbi.assert_no_shadow_matlab_sources( ...
    confirmed_root, selected);
ttbi.assert_reviewed_matlab_resolution( ...
    confirmed_root, selected);
confirmed_lines = ttbi.hash_reviewed_source_entries( ...
    confirmed_root, selected);
if ~strcmp(first_lines, confirmed_lines)
    error('generator_source_root:SourceRace', ...
        'Reviewed MATLAB source bytes changed during authentication.');
end

digest_lines = confirmed_lines;
root_sha256 = ttbi.sha256(digest_lines);
file_count = numel(selected);
end
