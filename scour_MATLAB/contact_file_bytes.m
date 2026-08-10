function bytes = contact_file_bytes(path)
%CONTACT_FILE_BYTES Read a file as one row of uint8 bytes.

fid = fopen(path, 'rb');
if fid < 0
    error('contact_closure:FileOpen', ...
        'Could not open file: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
bytes = reshape(fread(fid, Inf, '*uint8'), 1, []);
end
