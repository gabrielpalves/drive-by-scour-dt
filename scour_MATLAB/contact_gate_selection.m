function S = contact_gate_selection()
%CONTACT_GATE_SELECTION Dataset/selection reconstruction and host identity.
%
% The public fields are stable; each implementation is a separately named
% one-function module so provenance checks remain readable and auditable.

S = struct();
S.build_selection = @contact_gate_build_selection;
S.closure_host_attestation = @contact_closure_host_attestation;
S.assert_closure_host_matches_datasets = ...
    @contact_assert_closure_host_matches_datasets;
S.selection_descriptor = @contact_selection_descriptor;
end
