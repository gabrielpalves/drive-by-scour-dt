function absolute = contact_absolute_path(path)
%CONTACT_ABSOLUTE_PATH Return an absolute path only when it has no link alias.

path = char(path);
if isempty(path)
    absolute = path;
    return
end
identity = contact_unlinked_path_identity(path);
absolute = identity.canonical_path;
end
