function tf = contact_regular_nonsymlink_directory(path)
%CONTACT_REGULAR_NONSYMLINK_DIRECTORY Accept one real, unaliased directory.

tf = ischar(path) && isrow(path) && ~isempty(path);
if ~tf
    return
end
try
    identity = contact_unlinked_path_identity(path);
    tf = identity.exists && identity.is_directory && ~isempty(identity.file_key);
catch
    % Filesystem identity is a trust boundary. An ambiguous query rejects the
    % directory instead of silently canonicalizing away the evidence.
    tf = false;
end
end
