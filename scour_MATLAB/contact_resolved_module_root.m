function root = contact_resolved_module_root(names, error_id, label)
%CONTACT_RESOLVED_MODULE_ROOT Hash an unshadowed executable-file inventory.
%
% Each name must resolve to the reviewed scour_MATLAB directory. The root is
% SHA-256 of sorted "<name>:<sha256>" lines joined by LF without terminal LF.

module_dir = fileparts(mfilename('fullpath'));
lines = cell(size(names));
for k = 1:numel(names)
    expected = fullfile(module_dir, names{k});
    [~, base_name] = fileparts(names{k});
    resolved = which(base_name);
    if isempty(resolved) || ~strcmpi(contact_absolute_path(resolved), ...
            contact_absolute_path(expected))
        error(error_id, ...
            ['%s %s is missing or shadowed outside the reviewed ' ...
             'scour_MATLAB directory.'], label, names{k});
    end
    lines{k} = sprintf('%s:%s', names{k}, contact_file_sha256(expected));
end
root = contact_text_sha256(strjoin(sort(lines), newline));
end
