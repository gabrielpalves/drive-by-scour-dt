function public = contact_gate_public_case(result)
%CONTACT_GATE_PUBLIC_CASE Build the durable projection of one case result.
%
% Large internal solver objects stay out of the canonical case artifact.
% Tables are converted to scalar structs so MAT and JSON projections agree.

common = contact_closure_common();
public = struct();
identity_fields = {'schema', 'status', 'ordinal', 'stage', 'state_index', ...
    'passage_index', 'state_uid', 'state_file_sha256', 'policy_sha256', ...
    'selection_sha256', 'generator_source_root_sha256', ...
    'matlab_environment_sha256', 'started_utc', 'completed_utc', ...
    'failure_reasons', 'error_identifier', 'error_message'};
for k = 1:numel(identity_fields)
    public.(identity_fields{k}) = result.(identity_fields{k});
end
if any(strcmp(result.status, {'PASS', 'FAIL'}))
    report = result.report;
    public.study_schema = report.study_schema;
    public.report_status = report.status;
    public.report_stage = report.stage;
    public.report_state_index = report.state_index;
    public.report_passage_index = report.passage_index;
    public.report_state_uid = report.state_uid;
    public.report_state_family = report.state_family;
    public.report_state_file_sha256 = report.state_file_sha256;
    public.report_gen_fingerprint = report.gen_fingerprint;
    public.report_dataset_dir_sha256 = ...
        common.text_sha256(report.dataset_dir);
    public.report_generator_source_root_sha256 = ...
        report.current_generator_source_root_sha256;
    public.report_matlab_environment_sha256 = ...
        report.current_matlab_environment_sha256;
    public.report_harness_sha256 = report.harness_sha256;
    public.report_b66_sha256 = report.b66_sha256;
    public.report_solver_execution_root_sha256 = ...
        report.solver_execution_root_sha256;
    public.report_dataset_manifest_root = ...
        report.dataset_integrity.manifest_root;
    public.report_case_info_sha256 = ...
        report.dataset_integrity.case_info_sha256;
    public.report_damage_states_sha256 = ...
        report.dataset_integrity.damage_states_sha256;
    public.report_file_digests_sha256 = ...
        report.dataset_integrity.file_digests_sha256;
    public.report_completion_marker_sha256 = ...
        report.dataset_integrity.completion_marker_sha256;
    public.report_host_receipt_sha256 = ...
        report.dataset_integrity.qualification_host_receipt_sha256;
    public.profile_phase_stream_index = ...
        report.profile_phase_stream_index;
    public.profile_phase_seed = report.profile_phase_seed;
    public.dt_requested_ms = report.dt_requested_ms;
    public.gates_n = report.gates_n;
    public.fraction_gate = report.fraction_gate;
    public.common_dx_m = report.common_dx_m;
    public.reconstruction_rtol = report.reconstruction_rtol;
    public.reconstruction_atol = report.reconstruction_atol;
    public.saved_baseline_mode = report.saved_baseline_mode;
    public.direct_reconstruction_pass = ...
        report.direct_reconstruction_pass;
    public.saved_contact_reconstruction_pass = ...
        report.saved_contact_reconstruction_pass;
    public.requested_dt_ms = report.run_table.requested_dt_ms';
    public.actual_dt_ms = report.run_table.actual_dt_ms';
    public.t_end_s = report.run_table.t_end_s';
    public.n_samples = report.run_table.n_samples';
    public.peak_contact_signed_N = ...
        report.run_table.peak_contact_signed_N';
    public.peak_tension_N = report.run_table.peak_tension_N';
    public.tension_fraction = report.run_table.tension_fraction';
    public.contact_lost_track = ...
        double(report.run_table.contact_lost_track');
    public.contact_lost_bridge = ...
        double(report.run_table.contact_lost_bridge');
    public.saved_contact_log = report.saved_contact_log;
    public.rerun_contact_log_1ms = report.rerun_contact_log_1ms;
    public.channel_metrics = table2struct( ...
        report.channel_table, 'ToScalar', true);
    public.channel_qoi = table2struct( ...
        report.channel_qoi_table, 'ToScalar', true);
    public.saved_reconstruction = table2struct( ...
        report.saved_baseline_table, 'ToScalar', true);
    public.report_plain = contact_gate_plain_report(report);
    public.acceptance = result.acceptance;
end
end
