function M = contact_study_metrics()
%CONTACT_STUDY_METRICS Numerical-evidence functions for the closure study.
%
% The stable handle fields delegate to one-function modules. This keeps
% metric definitions separately readable without changing callers.

M = struct();
M.channel_metric_table = @contact_channel_metric_table;
M.channel_qoi_table = @contact_channel_qoi_table;
M.gate_pass = @contact_gate_pass;
M.saved_baseline_comparison = @contact_saved_baseline_comparison;
M.nearest_dt_index = @contact_nearest_dt_index;
M.descriptor_summary = @contact_descriptor_summary;
end
