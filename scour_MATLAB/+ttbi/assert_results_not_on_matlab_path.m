function assert_results_not_on_matlab_path(matlab_root)
%ASSERT_RESULTS_NOT_ON_MATLAB_PATH Keep generated code outside source lookup.

generated_root_names = {'Results', 'Results_sensitivity'};
path_entries = strsplit(path, pathsep);
for root_index = 1:numel(generated_root_names)
    results_root = fullfile(matlab_root, generated_root_names{root_index});
    if ~isfolder(results_root)
        continue
    end
    for entry_index = 1:numel(path_entries)
        entry = path_entries{entry_index};
        if isempty(entry) || ~isfolder(entry)
            continue
        end
        if ttbi.path_is_same_or_child(entry, results_root)
            error('generator_source_root:ResultsOnPath', ...
                ['Generated result content is present on the MATLAB path: ' ...
                 '%s. Remove it before source authentication.'], entry);
        end
    end
end
end
