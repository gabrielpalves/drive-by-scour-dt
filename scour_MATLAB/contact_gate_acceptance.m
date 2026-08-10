function A = contact_gate_acceptance()
%CONTACT_GATE_ACCEPTANCE Per-case acceptance metrics for the closure gate.
%
% CONTRACT: this file owns every pass/fail decision applied to one study
% report: schema/policy echo, 1-ms direct raw reconstruction, the B11
% time-grid contract, signed-peak/flag consistency, registered contact
% limits, 0/12/24-kN classification stability across dt, contraction to
% the finest grid, actual-step generalized Richardson/GCI bounds on peak
% and tension fraction, waveform convergence, finest-grid identity, and
% the per-channel QOI GCI gate.  It reads no files and writes no files;
% every input arrives in the report/policy arguments and every output is
% the acceptance struct plus explicit failure reasons.
%
% RATIONALE: acceptance mathematics is the scientific core the independent
% Python checker recomputes bit-for-bit; isolating it from orchestration,
% provenance and publication keeps that surface small and auditable.
%
% The handle struct preserves the public API while each implementation lives
% in a separately reviewable one-function module.

A = struct();
A.accept_report = @contact_gate_accept_report;
end
