function digest = contact_file_sha256(path)
%CONTACT_FILE_SHA256 SHA-256 of one stable file snapshot.

[bytes, ~] = contact_stable_file_bytes(path);
digest = contact_bytes_sha256(bytes);
end
