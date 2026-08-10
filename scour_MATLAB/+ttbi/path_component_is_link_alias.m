function tf = path_component_is_link_alias(path)
%PATH_COMPONENT_IS_LINK_ALIAS Detect symbolic/reparse-like path components.

nio_path = javaObject('java.io.File', path).toPath();
options = ttbi.nofollow_link_options();
if ~javaMethod('exists', 'java.nio.file.Files', nio_path, options)
    tf = false;
    return
end
attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
    nio_path, 'basic:isSymbolicLink,isOther', options);
symbolic = attributes.get('isSymbolicLink');
other = attributes.get('isOther');
if isempty(symbolic) || isempty(other)
    error('ttbi:PathComponentMetadata', ...
        'Filesystem returned incomplete no-follow metadata for %s.', path);
end
tf = ttbi.java_boolean_value(symbolic) || ...
    ttbi.java_boolean_value(other);
end
