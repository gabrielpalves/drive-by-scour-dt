function contact_assert_snapshot_set_unchanged(snapshots)
%CONTACT_ASSERT_SNAPSHOT_SET_UNCHANGED Re-assert captured files as one set.

if ~iscell(snapshots)
    error('contact_snapshot:BadSnapshotSet', ...
        'Snapshot set must be a cell vector of scalar snapshot structs.');
end
for k = 1:numel(snapshots)
    snapshot = snapshots{k};
    required = {'path', 'observation', 'sha256'};
    if ~isstruct(snapshot) || ~isscalar(snapshot) || ...
            ~all(isfield(snapshot, required))
        error('contact_snapshot:BadSnapshotSet', ...
            'Snapshot %d is missing path, observation, or SHA-256.', k);
    end
    contact_assert_file_snapshot_unchanged( ...
        snapshot.path, snapshot.observation, snapshot.sha256);
end
end
