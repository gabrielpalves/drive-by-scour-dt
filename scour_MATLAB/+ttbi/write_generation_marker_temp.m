function expected_sha256 = write_generation_marker_temp( ...
        marker_path, provenance, digest_root)
%WRITE_GENERATION_MARKER_TEMP Persist and verify canonical marker bytes.
%
% The completion marker is an operational credential. Binary mode plus
% explicit UTF-8/LF bytes makes it identical on Windows and Linux. Publication
% proceeds only after exact write/close status, path safety, and byte digest all
% agree; the caller still performs the final source fence and atomic rename.

schema = provenance.gen_schema;
fingerprint = provenance.gen_fingerprint;
if ~(ischar(schema) && isrow(schema)) || ...
        isempty(regexp(schema, '^[A-Za-z0-9._-]+$', 'once')) || ...
        ~(ischar(fingerprint) && isrow(fingerprint)) || ...
        isempty(regexp(fingerprint, '^[0-9a-f]{64}$', 'once')) || ...
        ~(ischar(digest_root) && isrow(digest_root)) || ...
        isempty(regexp(digest_root, '^[0-9a-f]{64}$', 'once'))
    error('ttbi:CompletionMarkerIdentity', ...
        'Completion marker identity fields are malformed.');
end

marker_text = [schema newline fingerprint newline digest_root newline];
expected_bytes = reshape(unicode2native(marker_text, 'UTF-8'), 1, []);
expected_sha256 = ttbi.sha256_bytes(expected_bytes);
ttbi.delete_file_entry_if_present(marker_path);

file_id = fopen(marker_path, 'wb');
if file_id < 0
    error('ttbi:CompletionMarkerOpen', ...
        'Could not open temporary generation marker: %s', marker_path);
end
try
    wrote = fwrite(file_id, expected_bytes, 'uint8');
    close_status = fclose(file_id);
    file_id = -1;
catch write_error
    if file_id >= 0
        fclose(file_id);
    end
    ttbi.delete_file_entry_if_present(marker_path);
    rethrow(write_error);
end
if wrote ~= numel(expected_bytes) || close_status ~= 0
    ttbi.delete_file_entry_if_present(marker_path);
    error('ttbi:CompletionMarkerWrite', ...
        'Temporary completion marker did not persist every expected byte.');
end
if ~ttbi.regular_nonsymlink_file(marker_path) || ...
        ~strcmp(ttbi.stable_file_sha256(marker_path), expected_sha256)
    ttbi.delete_file_entry_if_present(marker_path);
    error('ttbi:CompletionMarkerVerification', ...
        'Temporary completion marker failed path/byte verification.');
end
end
