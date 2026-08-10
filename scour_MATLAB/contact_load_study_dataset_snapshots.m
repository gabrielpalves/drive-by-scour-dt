function dataset = contact_load_study_dataset_snapshots( ...
        dataset_dir, state_index, verify_integrity)
%CONTACT_LOAD_STUDY_DATASET_SNAPSHOTS Parse one coherent saved-state snapshot.
%
% Every MAT value returned here is parsed from the same stable byte buffer
% whose SHA and file identity are retained for the final publication-time
% reassertion. In strict mode the final buffers are also the buffers accepted
% by the canonical source-digests-v2 manifest.

case_path = fullfile(dataset_dir, 'case_info.mat');
state_name = sprintf('%04d.mat', state_index);
state_path = fullfile(dataset_dir, state_name);
states_path = fullfile(dataset_dir, 'damage_states.mat');
if ~isfile(case_path)
    error('contact_closure:MissingCaseInfo', ...
        'Missing case_info.mat in %s.', dataset_dir);
end
if ~isfile(state_path)
    error('contact_closure:MissingState', ...
        ['Missing %04d.mat in %s. The closure harness needs the completed ' ...
         'state file because it reuses its exact passage descriptors; it ' ...
         'never re-samples an aborted state.'], state_index, dataset_dir);
end
if ~isfile(states_path)
    error('contact_closure:MissingStateTable', ...
        'Paper-1 closure requires damage_states.mat.');
end

% A first exact case snapshot supplies n_states, which is required to parse
% the fixed manifest inventory. Strict mode then replaces it with the case
% snapshot authenticated by that manifest and checks that n_states did not
% change across the boundary.
case_snapshot = contact_capture_file_snapshot(case_path);
case_blob = contact_load_mat_bytes(case_snapshot.bytes);
case_info = contact_case_info_from_snapshot(case_blob);
n_states = contact_case_state_count(case_info);
if state_index > n_states
    error('contact_closure:StateOutOfRange', ...
        'State %d exceeds case_info.n_states=%d.', state_index, n_states);
end

strict_manifest = struct();
if verify_integrity
    strict_manifest = validate_dataset_digest_manifest( ...
        dataset_dir, n_states, 'StateIndices', state_index, ...
        'RetainSnapshots', true);
    snapshots = strict_manifest.retained_snapshots;
    case_snapshot = contact_named_file_snapshot( ...
        snapshots, 'case_info.mat');
    state_snapshot = contact_named_file_snapshot(snapshots, state_name);
    states_snapshot = contact_named_file_snapshot( ...
        snapshots, 'damage_states.mat');

    case_blob = contact_load_mat_bytes(case_snapshot.bytes);
    case_info = contact_case_info_from_snapshot(case_blob);
    if contact_case_state_count(case_info) ~= n_states
        error('contact_closure:DatasetRace', ...
            'case_info.n_states changed across strict manifest validation.');
    end
else
    state_snapshot = contact_capture_file_snapshot(state_path);
    states_snapshot = contact_capture_file_snapshot(states_path);
    snapshots = {case_snapshot; state_snapshot; states_snapshot};
end

state_blob = contact_load_mat_bytes(state_snapshot.bytes);
if isfield(state_blob, 'data') && isstruct(state_blob.data)
    state_data = state_blob.data;
else
    state_data = state_blob;
end
state_table = contact_load_mat_bytes(states_snapshot.bytes);
contact_assert_snapshot_set_unchanged(snapshots);

dataset = struct();
dataset.dataset_dir = dataset_dir;
dataset.state_index = state_index;
dataset.state_path = state_path;
dataset.case_info = case_info;
dataset.state_data = state_data;
dataset.state_table = state_table;
dataset.strict_manifest = strict_manifest;
dataset.snapshots = snapshots;
dataset.case_snapshot = case_snapshot;
dataset.state_snapshot = state_snapshot;
dataset.states_snapshot = states_snapshot;
end
