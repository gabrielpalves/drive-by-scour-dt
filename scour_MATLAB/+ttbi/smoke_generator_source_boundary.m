function [root, lines, count] = smoke_generator_source_boundary()
%SMOKE_GENERATOR_SOURCE_BOUNDARY Reject shadows and hard-linked source aliases.

package_dir = fileparts(mfilename('fullpath'));
matlab_root = fileparts(package_dir);
shadow_path = fullfile(matlab_root, ...
    'unmanifested_generator_shadow_probe.m');
if isfile(shadow_path)
    error('ttbi:SmokeShadowPreexisting', ...
        'Refusing to overwrite pre-existing source probe: %s', shadow_path);
end

file_id = fopen(shadow_path, 'w');
if file_id < 0
    error('ttbi:SmokeShadowOpen', ...
        'Could not create source-shadow probe: %s', shadow_path);
end
file_cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, 'function unmanifested_generator_shadow_probe\nend\n');
clear file_cleanup
shadow_cleanup = onCleanup(@() delete(shadow_path));
ttbi.assert_generator_root_rejected( ...
    'an unmanifested MATLAB source', ...
    'generator_source_root:UnmanifestedMatlab');
clear shadow_cleanup

reviewed_path = fullfile(matlab_root, 'A01_Train.m');
alias_path = fullfile(matlab_root, ...
    'unmanifested_reviewed_hardlink_probe.m');
if isfile(alias_path)
    error('ttbi:SmokeHardlinkPreexisting', ...
        'Refusing to overwrite pre-existing hard-link probe: %s', alias_path);
end
ttbi.create_hardlink(alias_path, reviewed_path);
alias_cleanup = onCleanup(@() delete(alias_path));
assert(~ttbi.regular_nonsymlink_file(reviewed_path), ...
    'Reviewed source hard-link alias was not detected.');
% The manifest parser checks every reviewed entry before the later shadow
% inventory.  A second hard link therefore reaches its fail-closed
% "missing, linked, or nonregular" boundary first.
ttbi.assert_generator_root_rejected( ...
    'a hard-linked reviewed source', ...
    'generator_source_root:FileMissing');
clear alias_cleanup

directory_alias = fullfile(matlab_root, ...
    'unmanifested_source_directory_alias');
if ttbi.path_entry_exists(directory_alias)
    error('ttbi:SmokeDirectoryAliasPreexisting', ...
        'Refusing to overwrite pre-existing directory probe: %s', ...
        directory_alias);
end
ttbi.create_directory_alias(directory_alias, package_dir);
directory_alias_cleanup = onCleanup( ...
    @() ttbi.delete_file_entry_if_present(directory_alias));
ttbi.assert_generator_root_rejected( ...
    'a linked source directory', ...
    'generator_source_root:LinkedSourceDirectory');
clear directory_alias_cleanup
ttbi.delete_file_entry_if_present(directory_alias);

% The authentic root must be usable again after every probe is removed.
[root, lines, count] = generator_source_root();
assert(~isempty(regexp(root, '^[0-9a-f]{64}$', 'once')) && ...
    ~isempty(lines) && count > 0, ...
    'Generator source root did not recover after negative probes.');
end
