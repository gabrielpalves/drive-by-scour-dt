function [manifest, execution_root] = ...
        contact_solver_source_manifest(generator_digest_lines)
%CONTACT_SOLVER_SOURCE_MANIFEST Bind every executed production solver module.

common = contact_closure_common();
names = contact_solver_modules();
paths = cell(size(names));
sha256 = cell(size(names));
root_dir = fileparts(mfilename('fullpath'));
digest_lines = strsplit(char(generator_digest_lines), newline);
for k = 1:numel(names)
    resolved = which(names{k});
    [expected, source_relative] = local_solver_source(root_dir, names{k});
    if isempty(resolved) || ...
            ~strcmpi(common.absolute_path(resolved), ...
                common.absolute_path(expected))
        error('contact_closure:MissingSolverSource', ...
            ['Executed solver module %s is missing or shadowed outside the ' ...
             'reviewed scour_MATLAB directory.'], names{k});
    end
    paths{k} = common.absolute_path(resolved);
    sha256{k} = common.file_sha256(paths{k});
    source_line = sprintf('scour_MATLAB/%s:%s', ...
        source_relative, sha256{k});
    if sum(strcmp(digest_lines, source_line)) ~= 1
        error('contact_closure:SolverSourceMismatch', ...
            'Executed solver module %s is absent/different in source root.', ...
            names{k});
    end
end
manifest = table(names, paths, sha256, ...
    'VariableNames', {'module', 'path', 'sha256'});
execution_lines = cellfun(@(name, sha) sprintf('%s:%s', name, sha), ...
    names, sha256, 'UniformOutput', false);
execution_root = common.text_sha256(strjoin(execution_lines, newline));
end

function [expected, relative] = local_solver_source(root_dir, name)
parts = strsplit(name, '.');
if numel(parts) == 1
    relative = [name, '.m'];
    expected = fullfile(root_dir, relative);
    return
end
path_parts = cell(1, numel(parts));
for k = 1:(numel(parts) - 1)
    path_parts{k} = ['+', parts{k}];
end
path_parts{end} = [parts{end}, '.m'];
expected = fullfile(root_dir, path_parts{:});
relative = strjoin(path_parts, '/');
end
