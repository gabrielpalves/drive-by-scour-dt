function digest = numerical_vv_file_sha256(path)
%NUMERICAL_VV_FILE_SHA256 SHA-256 of one file's exact bytes.

if ~(ischar(path) || (isstring(path) && isscalar(path)))
    error('numerical_vv:BadFilePath', 'File path must be one text scalar.');
end
path = char(path);
info = dir(path);
if numel(info) ~= 1 || info.isdir
    error('numerical_vv:MissingRegularFile', ...
        'Expected one regular file: %s', path);
end
fid = fopen(path, 'rb');
if fid < 0
    error('numerical_vv:FileRead', 'Could not open %s.', path);
end
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
digest = numerical_vv_sha256_bytes(bytes);
end
