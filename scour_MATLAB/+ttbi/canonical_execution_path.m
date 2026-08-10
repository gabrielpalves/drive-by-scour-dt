function normalized = canonical_execution_path(raw_path)
%CANONICAL_EXECUTION_PATH Canonical absolute path of an executed file.
    % Resolve dot/parent components and normalize case/separators before the
    % fail-closed execution-root comparison. Do not mutate MATLAB's cwd here:
    % an implicit cd would make unreviewed path shadowing hard to diagnose.
    file_ = javaObject('java.io.File', char(raw_path));
    normalized = char(file_.getCanonicalPath());
    normalized = strrep(normalized, '\', '/');
    is_drive_root_ = ~isempty(regexp(normalized, '^[A-Za-z]:/$', 'once'));
    while numel(normalized) > 1 && endsWith(normalized, '/') && ...
            ~is_drive_root_
        normalized(end) = [];
    end
    if ispc
        normalized = lower(normalized);
    end
end
