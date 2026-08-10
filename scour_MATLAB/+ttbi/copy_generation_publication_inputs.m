function copy_generation_publication_inputs( ...
        source_folder, destination_folder, include_state)
%COPY_GENERATION_PUBLICATION_INPUTS Copy authenticated inputs for a smoke.

names = {'case_info.mat'; 'damage_states.mat'};
if include_state
    names = [{'0001.mat'}; names];
end
for name_index = 1:numel(names)
    [copied, message] = copyfile( ...
        fullfile(source_folder, names{name_index}), ...
        fullfile(destination_folder, names{name_index}));
    if ~copied
        error('ttbi:SmokeFixtureCopy', ...
            'Could not copy %s into the publication smoke: %s', ...
            names{name_index}, message);
    end
end
end
