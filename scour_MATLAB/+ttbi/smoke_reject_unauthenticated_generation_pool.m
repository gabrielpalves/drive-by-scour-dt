function smoke_reject_unauthenticated_generation_pool(context)
%SMOKE_REJECT_UNAUTHENTICATED_GENERATION_POOL Prove worker failure tears down.

foreign = context;
foreign.provenance.generator_source_root_sha256 = repmat('0', 1, 64);
rejected = false;
try
    ttbi.run_generation_states(foreign, false, 1);
catch
    rejected = true;
end
assert(rejected, ...
    'Generation runner accepted a worker/source provenance mismatch.');
assert(isempty(gcp('nocreate')), ...
    'Rejected worker attestation leaked its temporary process pool.');
assert(~isfile(fullfile(context.run_folder, '0001.mat')), ...
    'Rejected worker attestation entered the state solver.');
end
