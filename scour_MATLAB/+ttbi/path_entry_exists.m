function tf = path_entry_exists(path)
%PATH_ENTRY_EXISTS Test one exact filesystem entry without following links.

file_obj = javaObject('java.io.File', path);
tf = javaMethod('exists', 'java.nio.file.Files', ...
    file_obj.toPath(), ttbi.nofollow_link_options());
end
