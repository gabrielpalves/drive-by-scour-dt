function summary = contact_gate_summary_skeleton( ...
        policy, policy_sha, selection_sha, source_root, source_lines, ...
        source_count, environment_sha, environment_descriptor, host_id, ...
        closure_host, datasets)
%CONTACT_GATE_SUMMARY_SKELETON Start the canonical gate summary.

common = contact_closure_common();
summary = struct();
summary.schema = 'contact-closure-gate-summary-v2';
summary.status = 'RUNNING';
summary.source_commit = policy.source_commit;
summary.policy_sha256 = policy_sha;
summary.selection_sha256 = selection_sha;
summary.expected_cases = policy.expected_cases;
summary.completed_cases = 0;
summary.pass_cases = 0;
summary.fail_cases = 0;
summary.error_cases = 0;
summary.case_artifact_root_sha256 = '';
summary.declared_host_id = host_id;
summary.closure_host_attestation = closure_host;
summary.generator_source_root_sha256 = source_root;
summary.generator_source_digest_lines = source_lines;
summary.generator_source_file_count = source_count;
summary.matlab_environment_sha256 = environment_sha;
summary.matlab_environment_descriptor = environment_descriptor;
summary.matlab_release = version('-release');
% Bind the files MATLAB actually resolves for every gate decision.
summary.gate_execution_root_sha256 = common.gate_execution_root();
summary.datasets = datasets;
summary.started_utc = common.utc_now();
summary.completed_utc = '';
end
