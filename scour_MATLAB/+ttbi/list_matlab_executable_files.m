function paths = list_matlab_executable_files(matlab_root)
%LIST_MATLAB_EXECUTABLE_FILES Walk source without scanning generated results.
%
% MATLAB's recursive ** wildcard descends into potentially huge generated
% datasets before a caller can filter them. This explicit breadth-first walk
% skips only the top-level Results and Results_sensitivity trees, rejects linked
% source directories, and returns every .m/.p/.mlx/platform-MEX file elsewhere
% under scour_MATLAB.

root_observation = ttbi.directory_observation(matlab_root);
pending_paths = {matlab_root};
pending_observations = {root_observation};
visited_paths = {root_observation.canonical_path};
visited_keys = {root_observation.file_key};
paths = cell(0, 1);
next_folder = 1;
while next_folder <= numel(pending_paths)
    folder = pending_paths{next_folder};
    expected_folder_observation = pending_observations{next_folder};
    next_folder = next_folder + 1;
    folder_observation = ttbi.directory_observation(folder);
    if ~isequal(folder_observation, expected_folder_observation)
        error('generator_source_root:SourceDirectoryRace', ...
            'MATLAB source directory identity changed before traversal: %s', ...
            folder);
    end
    is_root_folder = strcmp(folder_observation.canonical_path, ...
        root_observation.canonical_path);
    entries = dir(folder);
    for entry_index = 1:numel(entries)
        entry = entries(entry_index);
        if any(strcmp(entry.name, {'.', '..'}))
            continue
        end
        absolute_path = fullfile(folder, entry.name);
        if entry.isdir
            try
                observation = ttbi.directory_observation(absolute_path);
            catch directory_error
                error('generator_source_root:LinkedSourceDirectory', ...
                    ['MATLAB source tree contains an unauthenticated ' ...
                     'directory (%s): %s'], absolute_path, ...
                    directory_error.message);
            end
            if is_root_folder && any(strcmpi(entry.name, ...
                    {'Results', 'Results_sensitivity'}))
                continue
            end
            if any(strcmp(visited_paths, observation.canonical_path)) || ...
                    any(strcmp(visited_keys, observation.file_key))
                error('generator_source_root:SourceDirectoryAlias', ...
                    ['MATLAB source traversal encountered a repeated ' ...
                     'directory identity or loop: %s'], absolute_path);
            end
            visited_paths{end + 1, 1} = observation.canonical_path; %#ok<AGROW>
            visited_keys{end + 1, 1} = observation.file_key; %#ok<AGROW>
            pending_paths{end + 1} = absolute_path; %#ok<AGROW>
            pending_observations{end + 1} = observation; %#ok<AGROW>
            continue
        end
        [~, ~, extension] = fileparts(entry.name);
        is_executable = strcmpi(extension, '.m') || ...
            strcmpi(extension, '.p') || strcmpi(extension, '.mlx') || ...
            startsWith(extension, '.mex', 'IgnoreCase', true);
        if is_executable
            paths{end + 1, 1} = absolute_path; %#ok<AGROW>
        end
    end
    if ~isequal(ttbi.directory_observation(folder), folder_observation)
        error('generator_source_root:SourceDirectoryRace', ...
            'MATLAB source directory identity changed during traversal: %s', ...
            folder);
    end
end
end
