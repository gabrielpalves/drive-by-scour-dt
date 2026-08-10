function root = contact_gate_case_artifact_root(cases_dir, expected_cases)
%CONTACT_GATE_CASE_ARTIFACT_ROOT Hash the exact completed case inventory.
%
% The reduction is SHA-256 over LF-joined "name:sha256" lines in ordinal
% order, with the MAT source preceding its JSON projection.

common = contact_closure_common();
contact_gate_validate_case_inventory(cases_dir, expected_cases, true);
lines = cell(expected_cases * 2, 1);
line_index = 0;
for ordinal = 1:expected_cases
    for extension = {'.mat', '.json'}
        name = sprintf('%04d_case%s', ordinal, extension{1});
        path = fullfile(cases_dir, name);
        if ~isfile(path)
            error('contact_closure_gate:MissingCaseArtifact', ...
                'Missing completed case artifact: %s', path);
        end
        line_index = line_index + 1;
        lines{line_index} = sprintf('%s:%s', name, ...
            common.file_sha256(path));
    end
end
root = common.text_sha256(strjoin(lines, newline));
end
