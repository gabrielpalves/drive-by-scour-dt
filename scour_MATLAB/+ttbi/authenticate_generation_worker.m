function attestation = authenticate_generation_worker(provenance)
%AUTHENTICATE_GENERATION_WORKER Build one source attestation on a pool worker.
%
% parallel.pool.Constant invokes this constructor at most once on each worker
% whose Value is used. Requiring a current task prevents accidental client-side
% construction from masquerading as worker authentication.

task = getCurrentTask();
if isempty(task)
    error('ttbi:WorkerAttestationContext', ...
        'Generation worker attestation must execute inside a pool task.');
end
attestation = ttbi.build_generator_source_attestation(provenance);
attestation.worker_context_authenticated = true;
end
