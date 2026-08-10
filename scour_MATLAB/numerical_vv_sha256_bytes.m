function digest = numerical_vv_sha256_bytes(bytes)
%NUMERICAL_VV_SHA256_BYTES Lowercase SHA-256 for exact uint8 bytes.

if ~isa(bytes, 'uint8')
    error('numerical_vv:BadHashInput', 'Hash input must be uint8 bytes.');
end
engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(bytes(:));
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end
