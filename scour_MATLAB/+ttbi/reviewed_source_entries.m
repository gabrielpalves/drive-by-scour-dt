function [repository_root, entries] = reviewed_source_entries()
%REVIEWED_SOURCE_ENTRIES Parse and authenticate bundle_source_files.txt.

package_dir = fileparts(mfilename('fullpath'));
matlab_root = fileparts(package_dir);
repository_root = fileparts(matlab_root);
manifest_path = fullfile(repository_root, 'bundle_source_files.txt');
if ~ttbi.regular_nonsymlink_file(manifest_path)
    error('generator_source_root:ManifestMissing', ...
        'Tracked source manifest is missing or linked: %s', manifest_path);
end

before_sha = ttbi.stable_file_sha256(manifest_path);
manifest_text = fileread(manifest_path);
after_sha = ttbi.stable_file_sha256(manifest_path);
if ~strcmp(before_sha, after_sha)
    error('generator_source_root:ManifestRace', ...
        'Tracked source manifest changed while it was parsed.');
end

raw_lines = regexp(manifest_text, '\r\n|\n|\r', 'split');
entries = cell(0, 1);
for line_index = 1:numel(raw_lines)
    entry = raw_lines{line_index};
    if isempty(entry) || startsWith(entry, '#')
        continue
    end
    if ~strcmp(entry, strtrim(entry))
        error('generator_source_root:Whitespace', ...
            'Manifest path has leading/trailing whitespace: "%s".', entry);
    end
    ttbi.validate_repository_relative_path(entry);
    native_parts = strsplit(entry, '/');
    absolute_path = fullfile(repository_root, native_parts{:});
    if ~ttbi.regular_nonsymlink_file(absolute_path)
        error('generator_source_root:FileMissing', ...
            'Manifest entry is missing, linked, or nonregular: %s', entry);
    end
    entries{end + 1, 1} = entry; %#ok<AGROW>
end

lower_entries = cellfun(@lower, entries, 'UniformOutput', false);
if numel(unique(entries)) ~= numel(entries) || ...
        numel(unique(lower_entries)) ~= numel(entries)
    error('generator_source_root:Duplicate', ...
        'Manifest contains duplicate or case-colliding paths.');
end
if ~isequal(entries, sort(entries))
    error('generator_source_root:ManifestOrder', ...
        'Manifest entries must be in canonical lexical order.');
end
end
