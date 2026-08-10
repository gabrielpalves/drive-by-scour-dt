function assert_reviewed_matlab_resolution(repository_root, selected)
%ASSERT_REVIEWED_MATLAB_RESOLUTION Bind MATLAB lookup to manifest paths.

matlab_entries = selected(endsWith(selected, '.m', 'IgnoreCase', true));
for entry_index = 1:numel(matlab_entries)
    relative_name = matlab_entries{entry_index};
    inside = char(extractAfter(relative_name, 'scour_MATLAB/'));
    [folder, base_name] = fileparts(inside);
    if isempty(folder)
        symbol = base_name;
    elseif strcmp(folder, '+ttbi')
        symbol = ['ttbi.' base_name];
    else
        error('generator_source_root:UnsupportedSourceLayout', ...
            'Reviewed MATLAB source uses an unsupported lookup path: %s', ...
            relative_name);
    end
    resolved = which(symbol);
    if isempty(resolved)
        error('generator_source_root:UnresolvedSource', ...
            'MATLAB cannot resolve reviewed source symbol %s.', symbol);
    end
    native_parts = strsplit(relative_name, '/');
    expected = fullfile(repository_root, native_parts{:});
    resolved_nio = javaObject( ...
        'java.io.File', resolved).toPath().toAbsolutePath().normalize();
    expected_nio = javaObject( ...
        'java.io.File', expected).toPath().toAbsolutePath().normalize();
    resolved_absolute = ttbi.comparison_path(char(resolved_nio.toString()));
    expected_absolute = ttbi.comparison_path(char(expected_nio.toString()));
    if ~strcmp(resolved_absolute, expected_absolute) || ...
            ~ttbi.regular_nonsymlink_file(resolved)
        error('generator_source_root:ShadowResolution', ...
            ['MATLAB resolves %s to an unreviewed path (%s) instead of ' ...
             '%s. Remove the shadowing path before generation.'], ...
            symbol, resolved, expected);
    end
end
end
