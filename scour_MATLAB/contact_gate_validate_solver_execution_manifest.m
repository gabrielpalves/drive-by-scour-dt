function root = contact_gate_validate_solver_execution_manifest(manifest)
%CONTACT_GATE_VALIDATE_SOLVER_EXECUTION_MANIFEST Revalidate executed solver bytes.
%
% The exact module order and SHA-256 reduction grammar are shared with the
% study report. Every resolved path must point to this reviewed directory.

common = contact_closure_common();
names = contact_solver_modules();
if ~istable(manifest) || ...
        ~isequal(manifest.Properties.VariableNames, ...
            {'module', 'path', 'sha256'}) || ...
        ~isequal(manifest.module, names)
    error('contact_closure_gate:SolverExecution', ...
        'Solver execution manifest has the wrong exact module inventory.');
end
source_dir = fileparts(mfilename('fullpath'));
lines = cell(numel(names), 1);
for k = 1:numel(names)
    expected_path = common.absolute_path( ...
        local_solver_source(source_dir, names{k}));
    resolved_path = common.absolute_path(which(names{k}));
    if ~strcmpi(resolved_path, expected_path) || ...
            ~strcmpi(common.absolute_path(manifest.path{k}), expected_path) || ...
            ~strcmp(manifest.sha256{k}, common.file_sha256(expected_path))
        error('contact_closure_gate:SolverExecution', ...
            'Solver module %s is shadowed or byte-different.', names{k});
    end
    lines{k} = sprintf('%s:%s', names{k}, manifest.sha256{k});
end
root = common.text_sha256(strjoin(lines, newline));
end

function expected = local_solver_source(root_dir, name)
parts = strsplit(name, '.');
if numel(parts) == 1
    expected = fullfile(root_dir, [name, '.m']);
    return
end
path_parts = cell(1, numel(parts));
for k = 1:(numel(parts) - 1)
    path_parts{k} = ['+', parts{k}];
end
path_parts{end} = [parts{end}, '.m'];
expected = fullfile(root_dir, path_parts{:});
end
