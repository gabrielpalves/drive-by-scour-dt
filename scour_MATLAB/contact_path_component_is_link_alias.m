function tf = contact_path_component_is_link_alias(path)
%CONTACT_PATH_COMPONENT_IS_LINK_ALIAS Detect symbolic/reparse-like entries.

nio_path = javaObject('java.io.File', path).toPath();
options = javaArray('java.nio.file.LinkOption', 1);
options(1) = javaMethod( ...
    'valueOf', 'java.nio.file.LinkOption', 'NOFOLLOW_LINKS');
if ~javaMethod('exists', 'java.nio.file.Files', nio_path, options)
    tf = false;
    return
end
attributes = javaMethod('readAttributes', 'java.nio.file.Files', ...
    nio_path, 'basic:isSymbolicLink,isOther', options);
symbolic = attributes.get('isSymbolicLink');
other = attributes.get('isOther');
if isempty(symbolic) || isempty(other)
    error('contact_path:ComponentMetadata', ...
        'Filesystem returned incomplete no-follow metadata for %s.', path);
end
tf = contact_java_boolean_value(symbolic) || ...
    contact_java_boolean_value(other);
end
