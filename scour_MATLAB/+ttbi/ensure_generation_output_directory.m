function observation = ensure_generation_output_directory( ...
        run_folder, results_root, results_root_observation)
%ENSURE_GENERATION_OUTPUT_DIRECTORY Create and authenticate one child chain.
%
% Each missing component is created only after its existing parent has been
% reauthenticated. Existing symlink/junction/reparse components are rejected
% before MATLAB can traverse them. RUN_FOLDER must be a relative descendant of
% RESULTS_ROOT and may not contain dot, parent, or empty components.

if ~(ischar(run_folder) && isrow(run_folder) && ~isempty(run_folder)) || ...
        javaObject('java.io.File', run_folder).isAbsolute()
    error('ttbi:GenerationOutputPath', ...
        'Generation output path must be one nonempty relative character row.');
end
if ~(ischar(results_root) && isrow(results_root) && ~isempty(results_root))
    error('ttbi:GenerationOutputRoot', ...
        'Results root must be one nonempty relative character row.');
end

normalized = strrep(run_folder, '\', '/');
parts = regexp(normalized, '/', 'split');
if numel(parts) < 2 || ~strcmp(parts{1}, results_root) || ...
        any(cellfun(@isempty, parts)) || ...
        any(ismember(parts, {'.', '..'}))
    error('ttbi:GenerationOutputPath', ...
        'Generation output must be a direct component chain below %s.', ...
        results_root);
end

ttbi.assert_generation_output_directory( ...
    results_root, results_root_observation);
cursor = results_root;
cursor_observation = results_root_observation;
for part_index = 2:numel(parts)
    ttbi.assert_generation_output_directory(cursor, cursor_observation);
    child = fullfile(cursor, parts{part_index});
    if ttbi.path_entry_exists(child)
        child_observation = ttbi.directory_observation(child);
    else
        [made, message] = mkdir(child);
        if ~made
            error('ttbi:GenerationOutputCreate', ...
                'Could not create generation output component %s: %s', ...
                child, message);
        end
        ttbi.assert_generation_output_directory(cursor, cursor_observation);
        child_observation = ttbi.directory_observation(child);
    end
    cursor = child;
    cursor_observation = child_observation;
end

observation = cursor_observation;
ttbi.assert_generation_output_directory(run_folder, observation);
end
