function contact_gate_publish_or_validate_summary_sidecars(output_dir, summary)
%CONTACT_GATE_PUBLISH_OR_VALIDATE_SUMMARY_SIDECARS Write deterministic views.
%
% Existing sidecars are compared byte-for-byte and are never overwritten.

json_path = fullfile(output_dir, 'gate_summary.json');
status_path = fullfile(output_dir, 'GATE_STATUS.txt');
json_text = [jsonencode(summary, PrettyPrint=true), newline];
status_text = sprintf( ...
    '%s\nexpected=%d\ncompleted=%d\npass=%d\nfail=%d\nerror=%d\n', ...
    summary.status, summary.expected_cases, summary.completed_cases, ...
    summary.pass_cases, summary.fail_cases, summary.error_cases);
contact_gate_write_or_verify_text(json_path, json_text);
contact_gate_write_or_verify_text(status_path, status_text);
end
