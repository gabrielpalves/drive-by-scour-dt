function h = sha256_bytes(bytes)
%SHA256_BYTES SHA-256 of a byte vector, lowercase hex.
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(reshape(uint8(bytes), [], 1)); % Java byte[] -> MATLAB int8
    h = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end
