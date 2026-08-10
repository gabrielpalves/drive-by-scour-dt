function digest = contact_text_sha256(value)
%CONTACT_TEXT_SHA256 SHA-256 of canonical UTF-8 text bytes.

digest = contact_bytes_sha256(unicode2native(char(value), 'UTF-8'));
end
