function observation = directory_observation(path)
%DIRECTORY_OBSERVATION Authenticate one directory path without following links.
%
% The normalized absolute spelling must equal Java's canonical spelling. This
% rejects a directory reached through a symlink or junction. Every path
% component is also checked with NOFOLLOW semantics. A native filesystem
% identity identifies the underlying directory even when a Java cloud-provider
% implementation omits fileKey, so callers can detect aliases/traversal loops.

if ~(ischar(path) && isrow(path) && ~isempty(path))
    error('ttbi:DirectoryObservationPath', ...
        'Directory observation requires one nonempty character path: %s', ...
        char(string(path)));
end

file = javaObject('java.io.File', path);
absolute_nio = file.toPath().toAbsolutePath().normalize();
cursor = javaObject('java.io.File', char(absolute_nio.toString()));
while ~isempty(cursor)
    if ttbi.path_component_is_link_alias(char(cursor.getPath()))
        error('ttbi:DirectoryObservationLinked', ...
            ['Directory path contains a symbolic-link, junction, or ' ...
             'reparse-like component: %s'], path);
    end
    cursor = cursor.getParentFile();
end

absolute = ttbi.comparison_path(char(absolute_nio.toString()));
canonical = ttbi.comparison_path(char(file.getCanonicalPath()));
if ~strcmp(absolute, canonical)
    error('ttbi:DirectoryObservationLinked', ...
        ['Directory path contains a symlink, junction, reparse alias, or ' ...
         'other canonical-path indirection: %s'], path);
end

options = ttbi.nofollow_link_options();
try
    file_identity = ttbi.filesystem_identity(path);
    attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
        absolute_nio, 'basic:isDirectory', options);
    is_directory = attributes.get('isDirectory');
    if isempty(is_directory) || ...
            ~ttbi.java_boolean_value(is_directory) || ...
            isempty(file_identity)
        error('ttbi:DirectoryObservationMetadata', ...
            'Filesystem returned no stable directory identity for %s.', path);
    end

    confirmed_identity = ttbi.filesystem_identity(path);
    confirmed_nio = file.toPath().toAbsolutePath().normalize();
    confirmed_absolute = ttbi.comparison_path( ...
        char(confirmed_nio.toString()));
    confirmed_canonical = ttbi.comparison_path( ...
        char(file.getCanonicalPath()));
    confirmed_cursor = javaObject( ...
        'java.io.File', char(confirmed_nio.toString()));
    while ~isempty(confirmed_cursor)
        if ttbi.path_component_is_link_alias( ...
                char(confirmed_cursor.getPath()))
            error('ttbi:DirectoryObservationLinked', ...
                ['Directory path acquired a symbolic-link, junction, or ' ...
                 'reparse-like component during observation: %s'], path);
        end
        confirmed_cursor = confirmed_cursor.getParentFile();
    end
    if ~strcmp(file_identity, confirmed_identity) || ...
            ~strcmp(absolute, confirmed_absolute) || ...
            ~strcmp(absolute, confirmed_canonical)
        error('ttbi:DirectoryObservationChanged', ...
            'Directory identity changed during observation for %s.', path);
    end
catch observation_error
    if startsWith(observation_error.identifier, ...
            'ttbi:DirectoryObservationMetadata')
        rethrow(observation_error)
    end
    error('ttbi:DirectoryObservationMetadata', ...
        'Cannot capture stable directory metadata for %s: %s', ...
        path, observation_error.message);
end

observation = struct( ...
    'canonical_path', canonical, ...
    'file_key', file_identity);
end
