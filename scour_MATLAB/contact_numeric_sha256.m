function digest = contact_numeric_sha256(values)
%CONTACT_NUMERIC_SHA256 SHA-256 of MATLAB's in-memory numeric bytes.

digest = contact_bytes_sha256(typecast(values(:), 'uint8'));
end
