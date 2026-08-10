function tf = path_is_same_or_child(candidate, parent)
%PATH_IS_SAME_OR_CHILD Compare canonical paths on a component boundary.

candidate = ttbi.canonical_execution_path(candidate);
parent = ttbi.canonical_execution_path(parent);
tf = strcmp(candidate, parent) || startsWith(candidate, [parent '/']);
end
