function state_paths = inspect_numbered_state_inventory(run_folder, n_states)
%INSPECT_NUMBERED_STATE_INVENTORY Classify every numeric-looking MAT artifact.
%
% Valid state names are exactly 0001.mat ... NNNN.mat. Numeric aliases such as
% 1.mat, 00001.mat, 0000.mat, or an out-of-range 9999.mat are fatal instead of
% being silently ignored during resume.

if ~(ischar(run_folder) && isrow(run_folder) && isfolder(run_folder))
    error('ttbi:StateInventoryFolder', ...
        'run_folder must be an existing character-row directory.');
end
validateattributes(n_states, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
    mfilename, 'n_states');

state_paths = cell(n_states, 1);
entries = dir(run_folder);
for entry_index = 1:numel(entries)
    name = entries(entry_index).name;
    tokens = regexp(name, '^(\d+)\.[mM][aA][tT]$', 'tokens', 'once');
    if isempty(tokens)
        continue;
    end

    state_index = str2double(tokens{1});
    canonical_name = '';
    if isfinite(state_index) && state_index >= 1 && ...
            state_index <= n_states && state_index == round(state_index)
        canonical_name = sprintf('%04d.mat', state_index);
    end
    if entries(entry_index).isdir || ~strcmp(name, canonical_name)
        error('ttbi:StateInventoryName', ...
            ['A00 RESUME ABORTED: numeric MAT artifact "%s" is not one ' ...
             'canonical in-range state name (0001.mat ... %04d.mat).'], ...
            name, n_states);
    end
    state_path = fullfile(run_folder, name);
    if ~ttbi.regular_nonsymlink_file(state_path)
        error('ttbi:StateInventoryPath', ...
            ['A00 RESUME ABORTED: state "%s" is not one regular file ' ...
             'without symlink, junction, or hard-link aliases.'], name);
    end
    if ~isempty(state_paths{state_index})
        error('ttbi:StateInventoryDuplicate', ...
            'A00 RESUME ABORTED: duplicate state index %d.', state_index);
    end
    state_paths{state_index} = state_path;
end
end
