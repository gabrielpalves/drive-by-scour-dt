function contact_assert_file_snapshot_unchanged( ...
        path, expected_observation, expected_sha256)
%CONTACT_ASSERT_FILE_SNAPSHOT_UNCHANGED Re-read and compare an earlier snapshot.

[bytes, observation] = contact_stable_file_bytes(path);
actual_sha256 = contact_bytes_sha256(bytes);
if ~isequal(observation, expected_observation) || ...
        ~strcmp(actual_sha256, expected_sha256)
    error('contact_snapshot:FileRace', ...
        'File identity, metadata, or bytes changed during validation: %s', path);
end
end
