function V = contact_study_solver()
%CONTACT_STUDY_SOLVER Production-chain execution functions.
%
% The stable handle fields delegate to one-function modules.

V = struct();
V.run_one = @contact_run_one;
V.solver_source_manifest = @contact_solver_source_manifest;
end
