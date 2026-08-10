function h = file_sha256(fpath)
%FILE_SHA256 SHA-256 of a file's exact bytes.
    % SHA-256 hex digest of a file's raw BYTES (audit R6 C7: profile-asset hash).
    fid = fopen(fpath, 'rb');
    if fid < 0, error('local_file_sha256: cannot open %s', fpath); end
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
    h = ttbi.sha256_bytes(bytes);
end
