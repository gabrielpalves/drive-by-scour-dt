function snapshot = contact_named_file_snapshot(snapshots, name)
%CONTACT_NAMED_FILE_SNAPSHOT Select one uniquely named captured file.

if ~iscell(snapshots)
    error('contact_snapshot:BadSnapshotSet', ...
        'Snapshot set must be a cell vector.');
end
matches = false(size(snapshots));
for k = 1:numel(snapshots)
    value = snapshots{k};
    matches(k) = isstruct(value) && isscalar(value) && ...
        isfield(value, 'name') && strcmp(value.name, char(name));
end
indices = find(matches);
if ~isscalar(indices)
    error('contact_snapshot:MissingNamedSnapshot', ...
        'Snapshot set must contain exactly one entry named %s.', char(name));
end
snapshot = snapshots{indices};
end
