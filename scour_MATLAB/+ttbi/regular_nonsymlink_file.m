function tf = regular_nonsymlink_file(path)
%REGULAR_NONSYMLINK_FILE Accept one regular file with exactly one hard link.
%
% Java's absolute path preserves a symlink/junction component while its
% canonical path resolves it. Comparing the two therefore rejects both a
% linked file and a file reached through a linked parent directory. The
% platform helper also rejects hard links: otherwise bytes could be changed
% through an alias after this path had passed validation.

tf = ischar(path) && isrow(path) && isfile(path);
if ~tf
    return
end
file = javaObject('java.io.File', path);
absolute = char(file.getAbsolutePath());
canonical = char(file.getCanonicalPath());
absolute = strrep(absolute, '\', '/');
canonical = strrep(canonical, '\', '/');
if ispc
    absolute = lower(absolute);
    canonical = lower(canonical);
end
if ~strcmp(absolute, canonical)
    tf = false;
    return
end

try
    tf = ttbi.hardlink_count(path) == 1;
catch
    % Link-count attestation is a security boundary. An unavailable or
    % ambiguous platform query must reject the file instead of weakening it.
    tf = false;
end
end
