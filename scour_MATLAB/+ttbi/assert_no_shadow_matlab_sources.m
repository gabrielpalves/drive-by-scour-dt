function assert_no_shadow_matlab_sources(repository_root, selected)
%ASSERT_NO_SHADOW_MATLAB_SOURCES Reject unreviewed executable .m files.
%
% Results/ and Results_sensitivity/ contain generated evidence and are never
% on the generator search path. Root-level micro_A00_*.m files are generated
% qualification drivers;
% their self-identity is checked separately and their names cannot shadow a
% production function. Every other executable MATLAB source below
% scour_MATLAB (.m, .p, .mlx, or platform MEX) must be manifest-bound.

matlab_root = fullfile(repository_root, 'scour_MATLAB');
ttbi.assert_results_not_on_matlab_path(matlab_root);
listed = ttbi.list_matlab_executable_files(matlab_root);
actual = cell(0, 1);
root_prefix = [strrep(char(javaObject( ...
    'java.io.File', matlab_root).getAbsolutePath()), '\', '/'), '/'];
for entry_index = 1:numel(listed)
    absolute_path = listed{entry_index};
    [~, name] = fileparts(absolute_path);
    absolute_posix = strrep(char(javaObject( ...
        'java.io.File', absolute_path).getAbsolutePath()), '\', '/');
    comparison_path = absolute_posix;
    comparison_prefix = root_prefix;
    if ispc
        comparison_path = lower(comparison_path);
        comparison_prefix = lower(comparison_prefix);
    end
    if ~startsWith(comparison_path, comparison_prefix)
        error('generator_source_root:InventoryEscape', ...
            'MATLAB inventory escaped its reviewed root: %s', absolute_path);
    end
    relative_inside = absolute_posix(numel(root_prefix) + 1:end);
    relative_name = ['scour_MATLAB/' relative_inside];
    if ~contains(relative_inside, '/') && ...
            startsWith(name, 'micro_A00_')
        continue
    end
    if ~ttbi.regular_nonsymlink_file(absolute_path)
        error('generator_source_root:ShadowLinked', ...
            'MATLAB source is linked or nonregular: %s', relative_name);
    end
    actual{end + 1, 1} = relative_name; %#ok<AGROW>
end

manifest_executable_mask = ~cellfun(@isempty, regexpi( ...
    selected, '\.(m|p|mlx|mex[^/]*)$', 'once'));
manifest_matlab = selected(manifest_executable_mask);
actual_folded = cellfun(@lower, actual, 'UniformOutput', false);
manifest_folded = cellfun(@lower, manifest_matlab, 'UniformOutput', false);
unexpected = actual(~ismember(actual_folded, manifest_folded));
if ~isempty(unexpected)
    error('generator_source_root:UnmanifestedMatlab', ...
        ['Unmanifested MATLAB source could shadow reviewed execution: %s. ' ...
         'Review it and add it to bundle_source_files.txt, or remove it.'], ...
        strjoin(sort(unexpected), ', '));
end
end
