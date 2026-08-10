function result = contact_gate_case_result_skeleton( ...
        row, policy_sha, selection_sha, source_root, environment_sha)
%CONTACT_GATE_CASE_RESULT_SKELETON Start one canonical case record.
%
% The gate writes this identity before attaching a study report. Keeping the
% constructor in its own file makes the persistent schema easy to review.

common = contact_closure_common();
result = struct();
result.schema = 'contact-closure-case-v1';
result.status = 'RUNNING';
result.ordinal = row.ordinal;
result.stage = row.stage{1};
result.dataset_dir = row.dataset_dir{1};
result.state_index = row.state_index;
result.passage_index = row.passage_index;
result.state_uid = row.state_uid{1};
result.state_file_sha256 = row.state_file_sha256{1};
result.policy_sha256 = policy_sha;
result.selection_sha256 = selection_sha;
result.generator_source_root_sha256 = source_root;
result.matlab_environment_sha256 = environment_sha;
result.started_utc = common.utc_now();
result.completed_utc = '';
result.report = struct();
result.acceptance = struct();
result.failure_reasons = cell(0, 1);
result.error_identifier = '';
result.error_message = '';
end
