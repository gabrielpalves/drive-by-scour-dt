function digest = stable_file_sha256(path)
%STABLE_FILE_SHA256 Hash bytes only if the same file remains at the path.
%
% The pre/post fileKey and metadata observations close the practical path-ABA
% window around fopen/fread. Callers that protect executable source take two
% independent stable snapshots as an additional byte-level confirmation.

before = ttbi.file_observation(path);
digest = ttbi.file_sha256(path);
after = ttbi.file_observation(path);
if ~isequal(before, after)
    error('ttbi:StableFileRace', ...
        'File identity or metadata changed while hashing: %s', path);
end
end
