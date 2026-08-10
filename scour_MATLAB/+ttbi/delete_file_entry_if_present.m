function delete_file_entry_if_present(path)
%DELETE_FILE_ENTRY_IF_PRESENT Remove one exact file/link, never a real tree.
%
% Java NIO deletes a symlink/junction itself rather than its target. Real
% directories are rejected, so malformed credentials cannot trigger recursion.

file = javaObject('java.io.File', path);
nio_path = file.toPath();
options = ttbi.nofollow_link_options();
exists = javaMethod('exists', 'java.nio.file.Files', nio_path, options);
if ~exists
    return
end
is_directory = javaMethod( ...
    'isDirectory', 'java.nio.file.Files', nio_path, options);
if is_directory
    absolute = ttbi.comparison_path(char( ...
        file.toPath().toAbsolutePath().normalize().toString()));
    canonical = ttbi.comparison_path(char(file.getCanonicalPath()));
    is_alias = ttbi.path_component_is_link_alias(path) || ...
        ~strcmp(absolute, canonical);
    if ~is_alias
        error('ttbi:CredentialPathDirectory', ...
            'Refusing to delete a directory at publication path: %s', path);
    end
end
javaMethod('delete', 'java.nio.file.Files', nio_path);
end
