function snapshot = contact_validate_completion_marker_snapshot( ...
        marker_path, schema, fingerprint, manifest_root)
%CONTACT_VALIDATE_COMPLETION_MARKER_SNAPSHOT Validate exact marker bytes.

snapshot = contact_capture_file_snapshot(marker_path);
expected_text = sprintf('%s\n%s\n%s\n', ...
    char(schema), char(fingerprint), char(manifest_root));
expected_bytes = reshape(unicode2native(expected_text, 'UTF-8'), 1, []);
if ~isequal(snapshot.bytes, expected_bytes)
    error('contact_closure:CompletionMarkerMismatch', ...
        ['_GENERATION_COMPLETE is not the exact canonical three-line ' ...
         'schema/fingerprint/content-root marker.']);
end
end
