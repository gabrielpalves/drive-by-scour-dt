function validate_generation_completion_marker(dataset_dir, digest_root)
%VALIDATE_GENERATION_COMPLETION_MARKER Bind marker, case identity, and digests.

marker_path = fullfile(dataset_dir, '_GENERATION_COMPLETE');
case_path = fullfile(dataset_dir, 'case_info.mat');
if ~ttbi.regular_nonsymlink_file(marker_path) || ...
        ~ttbi.regular_nonsymlink_file(case_path)
    error('ttbi:CompletionCredentialPath', ...
        'Completion marker and case_info.mat must be regular unlinked files.');
end

before_sha = ttbi.stable_file_sha256(marker_path);
marker_text = fileread(marker_path);
after_sha = ttbi.stable_file_sha256(marker_path);
if ~strcmp(before_sha, after_sha)
    error('ttbi:CompletionMarkerRace', ...
        'Completion marker changed while it was parsed.');
end
lines = regexp(marker_text, '\r\n|\n|\r', 'split');
if numel(lines) ~= 4 || ~isempty(lines{4}) || ...
        any(cellfun(@isempty, lines(1:3))) || ...
        any(cellfun(@(value) contains(value, char(0)), lines(1:3)))
    error('ttbi:CompletionMarkerGrammar', ...
        'Completion marker must contain exactly three nonempty text lines.');
end

variables = whos('-file', case_path);
if ~isequal({variables.name}', {'case_info'})
    error('ttbi:CompletionCaseInventory', ...
        'case_info.mat must contain exactly one case_info variable.');
end
loaded = load(case_path, 'case_info');
case_info = loaded.case_info;
if ~isstruct(case_info) || ~isscalar(case_info) || ...
        ~all(isfield(case_info, {'gen_schema', 'gen_fingerprint'}))
    error('ttbi:CompletionCaseIdentity', ...
        'case_info.mat lacks one scalar generation identity.');
end
if ~strcmp(lines{1}, case_info.gen_schema) || ...
        ~strcmp(lines{2}, case_info.gen_fingerprint) || ...
        ~strcmp(lines{3}, digest_root)
    error('ttbi:CompletionMarkerIdentity', ...
        'Completion marker does not bind case_info and file_digests.');
end
end
