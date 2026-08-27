function P = contact_gate_policy()
%CONTACT_GATE_POLICY Immutable closure-gate policy and environment lock.
%
% CONTRACT: this file owns (a) the frozen contact-closure-gate-v2 policy
% struct (stages, expected case inventory, refinement steps, diagnostic
% gates, tolerances, GCI parameters, channel inventory), (b) its canonical
% key=value descriptor projection used for the policy SHA-256, and (c) the
% MATLAB reference/live-descriptor validation used for provenance. Exact
% release equality is not a gate; local capabilities and closure numerics are.
% Nothing here touches solver state.
%
% RATIONALE: the policy literals (notably policy.expected_cases = 420 and
% policy.stages) are scientific commitments verified byte-for-byte by the
% independent Python checker.  Isolating them from orchestration and
% mutation-test fixtures keeps every policy edit a small, reviewable diff.
%
% Handle-struct pattern: implementations live in one-function contact_*
% modules while this factory preserves the compact public API.

P = struct();
P.policy = @contact_gate_policy_definition;
P.policy_descriptor = @contact_gate_policy_descriptor;
P.validate_locked_matlab_environment = ...
    @contact_validate_locked_matlab_environment;
end
