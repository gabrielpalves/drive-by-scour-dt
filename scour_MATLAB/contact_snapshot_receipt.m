function receipt = contact_snapshot_receipt(snapshot)
%CONTACT_SNAPSHOT_RECEIPT Drop payload bytes but retain reassertion evidence.

if ~isstruct(snapshot) || ~isscalar(snapshot) || ...
        ~all(isfield(snapshot, ...
            {'name', 'path', 'bytes', 'observation', 'sha256'}))
    error('contact_snapshot:BadSnapshot', ...
        'A complete scalar file snapshot is required.');
end
receipt = snapshot;
receipt.bytes = uint8([]);
end
