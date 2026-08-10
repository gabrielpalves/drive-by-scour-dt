function tf = contact_regular_nonsymlink(path)
%CONTACT_REGULAR_NONSYMLINK Accept one regular file with no link aliases.
%
% This primitive is intentionally self-contained because its exact bytes are
% already part of both frozen contact executable roots. Java canonical paths
% reject symlinks/junctions. The native link count then rejects hard links.

tf = ischar(path) && isrow(path) && isfile(path);
if ~tf
    return
end
file = javaObject('java.io.File', path);
absolute = strrep(char(file.getAbsolutePath()), '\', '/');
canonical = strrep(char(file.getCanonicalPath()), '\', '/');
if ispc
    absolute = lower(absolute);
    canonical = lower(canonical);
end
if ~strcmp(absolute, canonical)
    tf = false;
    return
end

try
    if ispc
        system_directory = char(System.Environment.SystemDirectory);
        executable = fullfile(system_directory, 'fsutil.exe');
        [raw_lines, exit_code] = contact_run_small_process( ...
            {executable, 'hardlink', 'list', ...
            char(file.getAbsolutePath())}, 10);
        lines = cellfun(@(line) strtrim(char(line)), raw_lines, ...
            'UniformOutput', false);
        link_count = sum(~cellfun(@isempty, lines));
        tf = exit_code == 0 && link_count == 1;
    elseif isunix
        options = javaArray('java.nio.file.LinkOption', 1);
        options(1) = javaMethod( ...
            'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
        value = javaMethod('getAttribute', 'java.nio.file.Files', ...
            file.toPath(), 'unix:nlink', options);
        tf = double(value) == 1;
    else
        tf = false;
    end
catch
    tf = false;
end
end
