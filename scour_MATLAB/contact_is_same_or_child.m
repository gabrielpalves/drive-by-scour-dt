function tf = contact_is_same_or_child(candidate, parent)
%CONTACT_IS_SAME_OR_CHILD Test a canonical Windows path containment relation.

candidate = [lower(strrep(candidate, '/', '\')), '\'];
parent = [lower(strrep(parent, '/', '\')), '\'];
tf = startsWith(candidate, parent);
end
