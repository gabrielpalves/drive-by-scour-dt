function digest = contact_bytes_sha256(bytes)
%CONTACT_BYTES_SHA256 Lowercase SHA-256 for a uint8 byte sequence.

engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(bytes);
raw = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(raw, 2)', 1, []));
end
