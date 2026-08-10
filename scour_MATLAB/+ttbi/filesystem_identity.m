function identity = filesystem_identity(path)
%FILESYSTEM_IDENTITY Return a stable file/directory identity when available.
%
% Java NIO normally exposes BasicFileAttributes.fileKey. Some Windows cloud
% filesystem providers legally return an empty key, so Windows falls back to
% the native file ID queried by fsutil through a shell-free ProcessBuilder.
% Unix falls back to its native device/inode pair. Callers remain responsible
% for rejecting linked path components before trusting this identity.

if ~(ischar(path) && isrow(path) && ~isempty(path) && ...
        (isfile(path) || isfolder(path)))
    error('ttbi:FilesystemIdentityPath', ...
        'Filesystem identity requires one existing character-row path.');
end

file = javaObject('java.io.File', path);
options = ttbi.nofollow_link_options();
java_diagnostic = 'Java NIO returned no fileKey';
try
    attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
        file.toPath(), 'basic:fileKey', options);
    file_key = attributes.get('fileKey');
    if ~isempty(file_key)
        key_text = strtrim(char(file_key.toString()));
        if ~isempty(key_text)
            identity = ['nio|' key_text];
            return
        end
    end
catch java_error
    java_diagnostic = java_error.message;
end

if ispc
    try
        identity = ['windows|' ttbi.windows_file_identity(path)];
        return
    catch native_error
        error('ttbi:FilesystemIdentityUnavailable', ...
            ['Cannot obtain a stable Windows file ID for %s. ' ...
             'Java: %s Native: %s'], ...
            path, java_diagnostic, native_error.message);
    end
end

if isunix
    try
        device = javaMethod('getAttribute', 'java.nio.file.Files', ...
            file.toPath(), 'unix:dev', options);
        inode = javaMethod('getAttribute', 'java.nio.file.Files', ...
            file.toPath(), 'unix:ino', options);
        if isempty(device) || isempty(inode)
            error('ttbi:FilesystemIdentityUnixEmpty', ...
                'Unix device/inode attributes are empty.');
        end
        identity = ['unix|dev=' char(device.toString()) ...
            '|ino=' char(inode.toString())];
        return
    catch native_error
        error('ttbi:FilesystemIdentityUnavailable', ...
            ['Cannot obtain a stable Unix file ID for %s. ' ...
             'Java: %s Native: %s'], ...
            path, java_diagnostic, native_error.message);
    end
end

error('ttbi:FilesystemIdentityPlatform', ...
    'Stable filesystem identity is unsupported on this platform.');
end
