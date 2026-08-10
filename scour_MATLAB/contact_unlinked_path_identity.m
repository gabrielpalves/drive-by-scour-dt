function identity = contact_unlinked_path_identity(path)
%CONTACT_UNLINKED_PATH_IDENTITY Normalize a path without accepting aliases.
%
% toAbsolutePath().normalize() removes lexical "." and ".." components but
% does not resolve links. Its spelling must therefore equal getCanonicalPath(),
% which does resolve symlinks and junctions. An explicit per-component
% symbolic/reparse-like check supplies the corresponding NOFOLLOW assertion.

if ~(ischar(path) && isrow(path) && ~isempty(path))
    error('contact_path:InvalidPath', ...
        'Path must be one nonempty character row vector.');
end

file_obj = javaObject('java.io.File', path);
absolute_nio = file_obj.toPath().toAbsolutePath().normalize();
absolute_native = char(absolute_nio.toString());
cursor = javaObject('java.io.File', absolute_native);
while ~isempty(cursor)
    if contact_path_component_is_link_alias(char(cursor.getPath()))
        error('contact_path:LinkedPath', ...
            ['Path contains a symbolic-link, junction, or reparse-like ' ...
             'component: %s'], path);
    end
    cursor = cursor.getParentFile();
end

canonical_native = char(file_obj.getCanonicalPath());
absolute_comparison = contact_comparison_path(absolute_native);
canonical_comparison = contact_comparison_path(canonical_native);
if ~strcmp(absolute_comparison, canonical_comparison)
    error('contact_path:LinkedPath', ...
        ['Path contains a symlink, junction, reparse alias, or other ' ...
         'canonical-path indirection: %s'], path);
end

options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
exists = javaMethod('exists', 'java.nio.file.Files', absolute_nio, options);
file_key = '';
is_directory = false;
if exists
    try
        file_key = contact_filesystem_identity(absolute_native);
        attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
            absolute_nio, 'basic:isDirectory', options);
        directory_value = attributes.get('isDirectory');
        if isempty(file_key) || isempty(directory_value)
            error('contact_path:MissingIdentity', ...
                'Filesystem returned no stable identity for %s.', path);
        end
        is_directory = contact_java_boolean_value(directory_value);

        confirmed_file_key = contact_filesystem_identity(absolute_native);
        confirmed_nio = file_obj.toPath().toAbsolutePath().normalize();
        confirmed_absolute = contact_comparison_path( ...
            char(confirmed_nio.toString()));
        confirmed_canonical = contact_comparison_path( ...
            char(file_obj.getCanonicalPath()));
        confirmed_cursor = javaObject( ...
            'java.io.File', char(confirmed_nio.toString()));
        while ~isempty(confirmed_cursor)
            if contact_path_component_is_link_alias( ...
                    char(confirmed_cursor.getPath()))
                error('contact_path:LinkedPath', ...
                    ['Path acquired a symbolic-link, junction, or ' ...
                     'reparse-like component during observation: %s'], path);
            end
            confirmed_cursor = confirmed_cursor.getParentFile();
        end
        if ~strcmp(file_key, confirmed_file_key) || ...
                ~strcmp(absolute_comparison, confirmed_absolute) || ...
                ~strcmp(absolute_comparison, confirmed_canonical)
            error('contact_path:IdentityChanged', ...
                'Path identity changed during observation for %s.', path);
        end
    catch identity_error
        if startsWith(identity_error.identifier, 'contact_path:')
            rethrow(identity_error)
        end
        error('contact_path:IdentityQuery', ...
            'Could not read no-follow path identity for %s: %s', ...
            path, identity_error.message);
    end
end

identity = struct( ...
    'absolute_path', absolute_native, ...
    'canonical_path', canonical_native, ...
    'comparison_path', canonical_comparison, ...
    'exists', logical(exists), ...
    'is_directory', logical(is_directory), ...
    'file_key', file_key);
end
