function h = sha256(str)
%SHA256 SHA-256 of text, lowercase hex.
    % SHA-256 hex digest of a char row vector (audit R5 canonical fingerprint).
    h = ttbi.sha256_bytes(uint8(str));
end
