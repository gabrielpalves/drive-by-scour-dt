function count = hardlink_count(path)
%HARDLINK_COUNT Return the exact filesystem link count for one file.
%
% Windows exposes hard-link aliases through fsutil. Linux and other Unix Java
% providers expose the native nlink attribute. Both queries avoid a command
% shell, so an unusual but valid filename cannot inject command syntax.

if ~(ischar(path) && isrow(path) && isfile(path))
    error('ttbi:HardlinkPath', ...
        'Hard-link count requires one existing character-row file path.');
end

if ispc
    count = ttbi.windows_hardlink_count(path);
elseif isunix
    file = javaObject('java.io.File', path);
    options = ttbi.nofollow_link_options();
    try
        value = javaMethod('getAttribute', 'java.nio.file.Files', ...
            file.toPath(), 'unix:nlink', options);
        count = double(value);
    catch query_error
        error('ttbi:HardlinkQuery', ...
            'Cannot attest the Unix hard-link count for %s: %s', ...
            path, query_error.message);
    end
else
    error('ttbi:HardlinkPlatform', ...
        'Hard-link attestation is implemented only for Windows and Unix.');
end

if ~isscalar(count) || ~isfinite(count) || ...
        count < 1 || count ~= round(count)
    error('ttbi:HardlinkCount', ...
        'Filesystem returned an invalid hard-link count for %s.', path);
end
end
