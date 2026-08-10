function blob = contact_load_mat_bytes(bytes)
%CONTACT_LOAD_MAT_BYTES Parse an authenticated byte snapshot, not a live path.
%
% MATLAB's public load API is path-based. A private create-once temporary copy
% decouples parsing from subsequent mutations of the dataset path. Its bytes and
% identity are checked on both sides of load before the parsed value is used.

if ~isa(bytes, 'uint8') || ~isvector(bytes)
    error('contact_snapshot:BadMatBytes', ...
        'MAT snapshot must be a uint8 vector.');
end
bytes = reshape(bytes, 1, []);
snapshot_path = [tempname, '.mat'];
cleanup = onCleanup( ...
    @() contact_delete_file_entry_if_present(snapshot_path));

fid = fopen(snapshot_path, 'wb');
if fid < 0
    error('contact_snapshot:TempOpen', ...
        'Could not create a private MAT snapshot.');
end
count = fwrite(fid, bytes, 'uint8');
close_status = fclose(fid);
if count ~= numel(bytes) || close_status ~= 0
    error('contact_snapshot:TempWrite', ...
        'Could not flush the complete private MAT snapshot.');
end

[written_bytes, observation] = contact_stable_file_bytes(snapshot_path);
expected_sha256 = contact_bytes_sha256(bytes);
if ~isequal(written_bytes, bytes)
    error('contact_snapshot:TempMismatch', ...
        'Private MAT snapshot bytes differ before parsing.');
end
blob = load(snapshot_path);
contact_assert_file_snapshot_unchanged( ...
    snapshot_path, observation, expected_sha256);
clear cleanup
contact_delete_file_entry_if_present(snapshot_path);
end
