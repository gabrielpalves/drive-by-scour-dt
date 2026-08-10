function identity = contact_filesystem_identity(path)
%CONTACT_FILESYSTEM_IDENTITY Return a stable file/directory identity.
%
% Java NIO supplies fileKey on most filesystems. Windows cloud providers may
% return an empty key, so the contact evidence chain falls back to a native,
% volume-bound file ID. Unix falls back to its device/inode pair. Callers must
% reject linked path components before trusting the returned identity.

if ~(ischar(path) && isrow(path) && ~isempty(path) && ...
        (isfile(path) || isfolder(path)))
    error('contact_path:FilesystemIdentityPath', ...
        'Filesystem identity requires one existing character-row path.');
end

file = javaObject('java.io.File', path);
options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
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
        identity = ['windows|' contact_windows_file_identity(path)];
        return
    catch native_error
        error('contact_path:FilesystemIdentityUnavailable', ...
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
            error('contact_path:FilesystemIdentityUnixEmpty', ...
                'Unix device/inode attributes are empty.');
        end
        identity = ['unix|dev=' char(device.toString()) ...
            '|ino=' char(inode.toString())];
        return
    catch native_error
        error('contact_path:FilesystemIdentityUnavailable', ...
            ['Cannot obtain a stable Unix file ID for %s. ' ...
             'Java: %s Native: %s'], ...
            path, java_diagnostic, native_error.message);
    end
end

error('contact_path:FilesystemIdentityPlatform', ...
    'Stable filesystem identity is unsupported on this platform.');
end
