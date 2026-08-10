function observation = contact_file_observation(path)
%CONTACT_FILE_OBSERVATION Capture stable identity around one file read.

if ~contact_regular_nonsymlink(path)
    error('contact_snapshot:UnsafeFile', ...
        ['Snapshot requires one regular file reached without symbolic, ' ...
         'junction, or hard-link aliases: %s'], path);
end
identity = contact_unlinked_path_identity(path);
if ~identity.exists || identity.is_directory || isempty(identity.file_key)
    error('contact_snapshot:UnsafeFile', ...
        'Snapshot path has no stable regular-file identity: %s', path);
end

options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
file_obj = javaObject('java.io.File', identity.absolute_path);
try
    attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
        file_obj.toPath(), 'basic:size,lastModifiedTime', options);
    byte_count = attributes.get('size');
    modified = attributes.get('lastModifiedTime');
    if isempty(byte_count) || isempty(modified)
        error('contact_snapshot:MissingIdentity', ...
            'Filesystem returned incomplete metadata for %s.', path);
    end
catch observation_error
    if startsWith(observation_error.identifier, 'contact_snapshot:')
        rethrow(observation_error)
    end
    error('contact_snapshot:Metadata', ...
        'Could not read stable file metadata for %s: %s', ...
        path, observation_error.message);
end

confirmed_identity = contact_unlinked_path_identity(path);
if ~isequal(confirmed_identity, identity) || ...
        ~contact_regular_nonsymlink(path)
    error('contact_snapshot:IdentityChanged', ...
        'File identity changed during metadata capture for %s.', path);
end

observation = struct( ...
    'canonical_path', identity.comparison_path, ...
    'file_key', identity.file_key, ...
    'byte_count', double(byte_count), ...
    'modified_ms', double(modified.toMillis()), ...
    'hardlink_count', 1);
end
