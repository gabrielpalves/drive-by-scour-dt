function create_directory_alias(alias_path, target_path)
%CREATE_DIRECTORY_ALIAS Create a smoke-only directory symlink or junction.

if ~ttbi.regular_nonsymlink_directory(target_path)
    error('ttbi:DirectoryAliasTarget', ...
        'Directory-alias target is missing, linked, or ambiguous: %s', ...
        target_path);
end
if ttbi.path_entry_exists(alias_path)
    error('ttbi:DirectoryAliasExists', ...
        'Directory-alias path already exists: %s', alias_path);
end

alias_nio = javaObject('java.io.File', alias_path).toPath();
target_nio = javaObject('java.io.File', target_path).toPath().toAbsolutePath();
attributes = javaArray('java.nio.file.attribute.FileAttribute', 0);
try
    javaMethod('createSymbolicLink', 'java.nio.file.Files', ...
        alias_nio, target_nio, attributes);
catch symbolic_link_error
    if ~ispc
        error('ttbi:DirectoryAliasCreate', ...
            'Could not create directory symlink: %s', ...
            symbolic_link_error.message);
    end
    ttbi.create_windows_junction(alias_path, target_path);
end

if ~isfolder(alias_path) || ttbi.regular_nonsymlink_directory(alias_path)
    ttbi.delete_file_entry_if_present(alias_path);
    error('ttbi:DirectoryAliasVerification', ...
        'Created directory alias did not expose the expected linked boundary.');
end
end
