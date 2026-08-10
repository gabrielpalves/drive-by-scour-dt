function result = smoke_rail_domain_clearance(output_dir)
%SMOKE_RAIL_DOMAIN_CLEARANCE One coupled C15 geometry/profile plumbing case.
%
% The smoke intentionally executes only F40/V70/T3/C15. Its package verdict
% is UNVERIFIED and cannot select a production clearance. The full study is
% the default 18-solve rail_domain_clearance_study call.

if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(tempdir,sprintf( ...
        'rail_domain_clearance_smoke_%s',char(java.util.UUID.randomUUID)));
end
result = rail_domain_clearance_study(output_dir,'SmokeOnly',true);
assert(strcmp(result.verdict.run_kind,'rail_domain_clearance_smoke'));
assert(strcmp(result.verdict.overall_status,'UNVERIFIED'));
assert(strcmp(result.verdict.production_selection_status,'NOT_EVALUATED'));
assert(result.manifest.case_count == 1);
fprintf(['RAIL DOMAIN CLEARANCE SMOKE: PASS ' ...
    '(one C15 coupled solve; selection not evaluated)\n']);
end
