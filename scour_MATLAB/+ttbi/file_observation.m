function observation = file_observation(path)
%FILE_OBSERVATION Capture path identity and metadata around a byte read.
%
% A native filesystem identity (Java fileKey, Windows volume/file ID, or Unix
% device/inode) identifies the underlying object. Combining it with canonical
% path, byte length, modification time, and a one-link requirement detects
% replacement races without assuming every Java provider supplies fileKey.

if ~ttbi.regular_nonsymlink_file(path)
    error('ttbi:FileObservationPath', ...
        ['File observation requires one regular file reached without ' ...
         'symlink, junction, or hard-link aliases: %s'], path);
end

file = javaObject('java.io.File', path);
options = ttbi.nofollow_link_options();
try
    file_identity = ttbi.filesystem_identity(path);
    attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
        file.toPath(), 'basic:size,lastModifiedTime', options);
    byte_count = double(attributes.get('size'));
    modified_ms = double( ...
        attributes.get('lastModifiedTime').toMillis());
    confirmed_identity = ttbi.filesystem_identity(path);
    if ~strcmp(file_identity, confirmed_identity)
        error('ttbi:FileObservationChanged', ...
            'File identity changed during metadata capture for %s.', path);
    end
catch observation_error
    error('ttbi:FileObservationMetadata', ...
        'Cannot capture stable file metadata for %s: %s', ...
        path, observation_error.message);
end

observation = struct( ...
    'canonical_path', ttbi.canonical_execution_path(path), ...
    'file_key', file_identity, ...
    'byte_count', byte_count, ...
    'modified_ms', modified_ms, ...
    'hardlink_count', 1);
end
