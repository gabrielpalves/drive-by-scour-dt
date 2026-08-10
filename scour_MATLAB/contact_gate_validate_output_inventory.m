function contact_gate_validate_output_inventory( ...
        output_dir, expected_cases, require_final)
%CONTACT_GATE_VALIDATE_OUTPUT_INVENTORY Enforce the output whitelist.
%
% No unexplained file, directory, temporary artifact or partial summary is
% accepted in a gate evidence directory.

common = contact_closure_common();
entries = dir(output_dir);
entries = entries(~ismember({entries.name}, {'.', '..'}));
observed = {entries.name};
common_names = {'closure_policy.mat', 'closure_policy.txt', ...
    'selection_manifest.mat', 'selection_manifest.tsv', 'cases'};
final_names = {'gate_summary.mat', 'gate_summary.json', 'GATE_STATUS.txt'};
optional = {'PLAN_ONLY_NONQUALIFYING.json'};
allowed = [common_names, final_names, optional];
extra = setdiff(observed, allowed);
missing_common = setdiff(common_names, observed);
if ~isempty(extra) || ~isempty(missing_common)
    if ~isempty(missing_common)
        error('contact_closure_gate:PartialPlan', ...
            ['Frozen-plan publication is incomplete (missing=%s). No solver ' ...
             'case may be trusted in this directory; use a fresh empty ' ...
             'OutputDir.'], strjoin(missing_common, ','));
    end
    error('contact_closure_gate:OutputInventory', ...
        'Output inventory has unexplained extras=%s.', strjoin(extra, ','));
end

if require_final && ~isempty(setdiff(final_names, observed))
    error('contact_closure_gate:OutputInventory', ...
        'Final output is missing one or more summary artifacts.');
end
for k = 1:numel(common_names)
    name = common_names{k};
    entry = entries(strcmp(observed, name));
    if ~isscalar(entry) || ...
            (strcmp(name, 'cases') && ~entry.isdir) || ...
            (~strcmp(name, 'cases') && ...
                (entry.isdir || ~common.regular_nonsymlink( ...
                    fullfile(output_dir, name))))
        error('contact_closure_gate:OutputInventory', ...
            'Output entry %s has the wrong type.', name);
    end
end
for name = [final_names, optional]
    if ismember(name{1}, observed) && ...
            ~common.regular_nonsymlink(fullfile(output_dir, name{1}))
        error('contact_closure_gate:OutputInventory', ...
            'Output artifact %s is not regular/non-symlink.', name{1});
    end
end
if ~ismember('gate_summary.mat', observed) && ...
        (ismember('gate_summary.json', observed) || ...
         ismember('GATE_STATUS.txt', observed))
    error('contact_closure_gate:PartialSummary', ...
        'Summary sidecar exists without gate_summary.mat.');
end
contact_gate_validate_case_inventory( ...
    fullfile(output_dir, 'cases'), expected_cases, require_final);
end
