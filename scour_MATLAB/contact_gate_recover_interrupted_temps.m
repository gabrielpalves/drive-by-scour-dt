function contact_gate_recover_interrupted_temps(output_dir, expected_cases)
%CONTACT_GATE_RECOVER_INTERRUPTED_TEMPS Remove known non-evidence temp files.
%
% Recovery is intentionally narrow. A temp is removable only when its final
% target is absent; final-and-temp coexistence is rejected as ambiguous.

root_finals = {'closure_policy.mat', 'closure_policy.txt', ...
    'selection_manifest.mat', 'selection_manifest.tsv', ...
    'PLAN_ONLY_NONQUALIFYING.json', 'gate_summary.mat', ...
    'gate_summary.json', 'GATE_STATUS.txt'};
for k = 1:numel(root_finals)
    final_path = fullfile(output_dir, root_finals{k});
    tmp_path = [final_path, '.tmp'];
    contact_gate_recover_one_temp(final_path, tmp_path);
end
cases_dir = fullfile(output_dir, 'cases');
if ~isfolder(cases_dir)
    return
end
for ordinal = 1:expected_cases
    for extension = {'.mat', '.json'}
        final_path = fullfile(cases_dir, sprintf( ...
            '%04d_case%s', ordinal, extension{1}));
        contact_gate_recover_one_temp(final_path, [final_path, '.tmp']);
    end
end
end
