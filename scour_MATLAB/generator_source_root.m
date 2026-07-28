function [root_sha256, digest_lines, file_count] = generator_source_root()
%GENERATOR_SOURCE_ROOT Hash every reviewed MATLAB generator source/asset.
%   Reads the tracked repository manifest bundle_source_files.txt, validates
%   every non-comment entry, then selects every regular file below
%   scour_MATLAB/ regardless of extension. Exact file bytes are SHA-256 hashed.
%   DIGEST_LINES is the lexicographically sorted, LF-joined sequence
%       repository/relative/name:sha256
%   with no terminal LF. ROOT_SHA256 is the UTF-8 SHA-256 of DIGEST_LINES.
%
%   Missing files, duplicate/case-colliding entries, unsafe paths, or an empty
%   MATLAB selection fail closed. Thus the dataset identifies the complete
%   reviewed generator source set, including .m files and binary .mat assets.

    helper_dir = fileparts(mfilename('fullpath'));
    repository_root = fileparts(helper_dir);
    manifest_path = fullfile(repository_root, 'bundle_source_files.txt');
    if ~isfile(manifest_path)
        error('generator_source_root:ManifestMissing', ...
            'Tracked source manifest is missing: %s', manifest_path);
    end

    manifest_text = fileread(manifest_path);
    raw_lines = regexp(manifest_text, '\r\n|\n|\r', 'split');
    entries = cell(0, 1);
    for k = 1:numel(raw_lines)
        entry = raw_lines{k};
        if isempty(entry) || startsWith(entry, '#')
            continue;
        end
        if ~strcmp(entry, strtrim(entry))
            error('generator_source_root:Whitespace', ...
                'Manifest path has leading/trailing whitespace: "%s".', entry);
        end
        local_validate_manifest_path(entry);
        entry_parts = strsplit(entry, '/');
        entry_path = fullfile(repository_root, entry_parts{:});
        if ~isfile(entry_path)
            error('generator_source_root:FileMissing', ...
                'Manifest entry is not a regular file: %s', entry);
        end
        entries{end + 1, 1} = entry; %#ok<AGROW>
    end
    lower_entries = cellfun(@lower, entries, 'UniformOutput', false);
    if numel(unique(entries)) ~= numel(entries) || ...
            numel(unique(lower_entries)) ~= numel(entries)
        error('generator_source_root:Duplicate', ...
            'Manifest contains duplicate or case-colliding paths.');
    end

    selected = entries(startsWith(entries, 'scour_MATLAB/'));
    if isempty(selected)
        error('generator_source_root:Empty', ...
            'Manifest contains no scour_MATLAB/ source or asset.');
    end
    selected = sort(selected);
    lines = cell(numel(selected), 1);
    for k = 1:numel(selected)
        relative_name = selected{k};
        native_parts = strsplit(relative_name, '/');
        absolute_path = fullfile(repository_root, native_parts{:});
        if ~isfile(absolute_path)
            error('generator_source_root:FileMissing', ...
                'Manifest MATLAB entry is not a regular file: %s', relative_name);
        end
        lines{k} = sprintf('%s:%s', relative_name, ...
            local_file_sha256(absolute_path));
    end

    digest_lines = strjoin(sort(lines), newline);
    root_sha256 = local_text_sha256(digest_lines);
    file_count = numel(lines);
end

function local_validate_manifest_path(entry)
    if contains(entry, '\') || startsWith(entry, '/') || ...
            ~isempty(regexp(entry, '^[A-Za-z]:', 'once')) || ...
            contains(entry, '//')
        error('generator_source_root:UnsafePath', ...
            'Manifest path is not repository-relative POSIX form: "%s".', entry);
    end
    components = strsplit(entry, '/');
    if any(cellfun(@isempty, components)) || ...
            any(strcmp(components, '.')) || any(strcmp(components, '..'))
        error('generator_source_root:UnsafePath', ...
            'Manifest path contains an unsafe component: "%s".', entry);
    end
end

function sha256 = local_file_sha256(path)
    fid = fopen(path, 'rb');
    if fid < 0
        error('generator_source_root:Open', 'Cannot open source file: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(bytes);
    sha256 = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end

function sha256 = local_text_sha256(text)
    bytes = unicode2native(text, 'UTF-8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(bytes);
    sha256 = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end
