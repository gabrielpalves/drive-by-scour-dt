function normalized = contact_comparison_path(path)
%CONTACT_COMPARISON_PATH Normalize path text without resolving links.

normalized = strrep(char(path), '\', '/');
while numel(normalized) > 1 && endsWith(normalized, '/') && ...
        isempty(regexp(normalized, '^[A-Za-z]:/$', 'once'))
    normalized(end) = [];
end
if ispc
    normalized = lower(normalized);
end
end
