function W = contact_study_report()
%CONTACT_STUDY_REPORT Durable report-publication functions.
%
% Publication and Markdown rendering live in separate one-function modules.

W = struct();
W.maybe_write_report = @contact_maybe_write_report;
end
