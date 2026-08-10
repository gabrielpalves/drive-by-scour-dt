function contact_delete_file_entry_if_present(path)
%CONTACT_DELETE_FILE_ENTRY_IF_PRESENT Delete one file/link, never a real tree.

file_obj = javaObject('java.io.File', path);
nio_path = file_obj.toPath();
options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
if ~javaMethod('exists', 'java.nio.file.Files', nio_path, options)
    return
end
is_directory = javaMethod( ...
    'isDirectory', 'java.nio.file.Files', nio_path, options);
if is_directory
    absolute = contact_comparison_path(char( ...
        nio_path.toAbsolutePath().normalize().toString()));
    canonical = contact_comparison_path(char(file_obj.getCanonicalPath()));
    is_alias = contact_path_component_is_link_alias(path) || ...
        ~strcmp(absolute, canonical);
    if ~is_alias
        error('contact_snapshot:DeleteDirectory', ...
            'Refusing to recursively remove a snapshot directory: %s', path);
    end
end
javaMethod('delete', 'java.nio.file.Files', nio_path);
end
