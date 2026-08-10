function normalized = comparison_path(path)
%COMPARISON_PATH Normalize separators/case without resolving filesystem links.

normalized = strrep(char(path), '\', '/');
while numel(normalized) > 1 && endsWith(normalized, '/') && ...
        isempty(regexp(normalized, '^[A-Za-z]:/$', 'once'))
    normalized(end) = [];
end
if ispc
    normalized = lower(normalized);
end
end
