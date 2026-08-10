function run_generation_states(context, completed, max_workers)
%RUN_GENERATION_STATES Execute missing states on fresh authenticated workers.
%
% When work is missing, a pre-existing pool can contain cached definitions from
% before the source root was measured and is always destroyed. A fresh
% process-only pool is created for this call and destroyed on every normal/error
% exit. Fully resumed runs do not touch unrelated client pool state. Each used
% worker hashes and resolves the reviewed source once through a Constant; every
% parfor iteration validates that cached proof before any solver execution.

n_states = context.n_states;
if ~isempty(getCurrentTask())
    error('ttbi:GenerationRunnerContext', ...
        'Generation pool orchestration must execute on the MATLAB client.');
end
validateattributes(max_workers, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
    mfilename, 'max_workers');
if ~islogical(completed) || ~isvector(completed) || ...
        numel(completed) ~= n_states
    error('ttbi:GenerationCompletedShape', ...
        'completed must be one logical entry per semantic state.');
end
completed = completed(:);
ttbi.assert_generation_output_directory( ...
    context.run_folder, context.run_folder_observation);

if all(completed)
    fprintf(['Resume: all %d states are already valid; existing parallel ' ...
        'state is untouched and no generation pool was created.\n'], n_states);
    return
end

existing_pool = gcp('nocreate');
if ~isempty(existing_pool)
    fprintf(['Deleting pre-existing %s (%d workers); generation never ' ...
        'reuses workers with cached source.\n'], ...
        class(existing_pool), existing_pool.NumWorkers);
    delete(existing_pool);
end
if ~isempty(gcp('nocreate'))
    error('ttbi:GenerationPoolTeardown', ...
        'Pre-existing parallel pool did not terminate completely.');
end

cluster = parcluster('Processes');
pool_workers = min(max_workers, cluster.NumWorkers);
if pool_workers < 1
    error('ttbi:GenerationPoolCapacity', ...
        'Processes cluster exposes no usable generation worker.');
end
pool = parpool(cluster, pool_workers);
pool_cleanup = onCleanup(@() ttbi.delete_generation_pool(pool));
if ~isa(pool, 'parallel.ProcessPool') || pool.NumWorkers ~= pool_workers
    error('ttbi:GenerationPool', ...
        ['Generation requires the newly created parallel.ProcessPool with ' ...
         'exactly %d workers; got %s with %d workers.'], ...
        pool_workers, class(pool), pool.NumWorkers);
end

provenance = context.provenance;
worker_source = parallel.pool.Constant( ...
    @() ttbi.authenticate_generation_worker(provenance));
attestation_cleanup = onCleanup(@() delete(worker_source));
fprintf(['Generation pool: %d fresh process workers (configured maximum %d); ' ...
    'source attestation is cached once per used worker.\n'], ...
    pool_workers, max_workers);

parfor (state_index = 1:n_states, pool_workers)
    worker_attestation = worker_source.Value;
    ttbi.require_generation_worker_attestation( ...
        worker_attestation, provenance);
    if completed(state_index)
        fprintf('Skipping state %d - result already exists.\n', state_index);
        continue
    end
    ttbi.execute_generation_state( ...
        state_index, context, worker_attestation);
end

clear attestation_cleanup
clear pool_cleanup
if ~isempty(gcp('nocreate'))
    error('ttbi:GenerationPoolLeak', ...
        'Fresh generation pool remained active after worker completion.');
end
ttbi.assert_generation_output_directory( ...
    context.run_folder, context.run_folder_observation);
end
