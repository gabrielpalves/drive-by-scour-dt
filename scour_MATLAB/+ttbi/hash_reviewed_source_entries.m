function digest_lines = hash_reviewed_source_entries( ...
        repository_root, selected)
%HASH_REVIEWED_SOURCE_ENTRIES Hash each reviewed MATLAB file from stable bytes.

lines = cell(numel(selected), 1);
for file_index = 1:numel(selected)
    relative_name = selected{file_index};
    native_parts = strsplit(relative_name, '/');
    absolute_path = fullfile(repository_root, native_parts{:});
    % One read is guarded by file identity/metadata on both sides. The caller
    % repeats the complete manifest/inventory/hash snapshot independently.
    digest = ttbi.stable_file_sha256(absolute_path);
    lines{file_index} = sprintf('%s:%s', relative_name, digest);
end
digest_lines = strjoin(sort(lines), newline);
end
