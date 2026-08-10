function contact_assert_reviewed_bootstrap( ...
        reviewed_dir, module_names, error_id, label)
%CONTACT_ASSERT_REVIEWED_BOOTSTRAP Validate root-building functions by path.
%
% Executable-set roots normally validate every resolved module. This small
% bootstrap closes the circular edge: it checks the functions that build the
% root before any of them executes, using only MATLAB/Java built-ins.

for k = 1:numel(module_names)
    name = module_names{k};
    expected = fullfile(reviewed_dir, [name, '.m']);
    resolved = which(name);
    if isempty(resolved)
        error(error_id, '%s %s is missing.', label, name);
    end
    expected_file = java.io.File(expected);
    resolved_file = java.io.File(resolved);
    expected_absolute = strrep(char(expected_file.getAbsolutePath()), '\', '/');
    resolved_absolute = strrep(char(resolved_file.getAbsolutePath()), '\', '/');
    expected_canonical = strrep( ...
        char(expected_file.getCanonicalPath()), '\', '/');
    resolved_canonical = strrep( ...
        char(resolved_file.getCanonicalPath()), '\', '/');
    if ispc
        expected_absolute = lower(expected_absolute);
        resolved_absolute = lower(resolved_absolute);
        expected_canonical = lower(expected_canonical);
        resolved_canonical = lower(resolved_canonical);
    end
    if ~strcmp(expected_absolute, resolved_absolute) || ...
            ~strcmp(expected_absolute, expected_canonical) || ...
            ~strcmp(resolved_absolute, resolved_canonical)
        error(error_id, ...
            ['%s %s is shadowed outside the reviewed scour_MATLAB ' ...
             'directory.'], label, name);
    end
end
end
