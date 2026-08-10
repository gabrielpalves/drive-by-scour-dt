function manifest = validate_dataset_digest_manifest( ...
        dataset_dir, n_states, varargin)
%VALIDATE_DATASET_DIGEST_MANIFEST Strict R11 dataset-content authentication.
%
% manifest = validate_dataset_digest_manifest(DATASET_DIR, N_STATES)
% validates the exact source-digests-v2 grammar and hashes every numbered
% state plus case_info.mat and damage_states.mat.
%
% manifest = validate_dataset_digest_manifest(..., 'StateIndices', K)
% still validates the complete canonical filename/digest inventory and both
% sidecars, but hashes only state K. This mode is for a per-state consumer
% after an enclosing gate has authenticated the complete dataset up front.
%
% 'RetainSnapshots', true returns the exact manifest/member byte snapshots
% in manifest.retained_snapshots. Consumers must parse those buffers rather
% than reopening authenticated paths. The default is false so a whole-dataset
% gate does not retain every potentially large state payload in memory.

arguments
    dataset_dir {mustBeTextScalar}
    n_states (1,1) double {mustBeInteger,mustBePositive}
end
arguments (Repeating)
    varargin
end

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'StateIndices', 1:n_states, ...
    @(v) isnumeric(v) && isvector(v) && all(isfinite(v)) && ...
    all(v == round(v)) && ...
    all(v >= 1) && all(v <= n_states) && ...
    numel(unique(v)) == numel(v));
addParameter(parser, 'RetainSnapshots', false, ...
    @(v) (islogical(v) || isnumeric(v)) && isscalar(v) && ...
    isfinite(v) && any(v == [0, 1]));
parse(parser, varargin{:});
state_indices = double(parser.Results.StateIndices(:)');
retain_snapshots = logical(parser.Results.RetainSnapshots);

dataset_dir = char(dataset_dir);
try
    dataset_identity = contact_unlinked_path_identity(dataset_dir);
catch dataset_path_error
    error('dataset_digest_manifest:LinkedDataset', ...
        ['Dataset directory path is linked or cannot be authenticated: ' ...
         '%s (%s)'], dataset_dir, dataset_path_error.message);
end
if ~dataset_identity.exists || ~dataset_identity.is_directory
    error('dataset_digest_manifest:MissingDataset', ...
        'Dataset directory does not exist: %s', dataset_dir);
end
dataset_dir = dataset_identity.canonical_path;
manifest_path = fullfile(dataset_dir, 'file_digests.mat');
if ~contact_regular_nonsymlink(manifest_path)
    error('dataset_digest_manifest:MissingManifest', ...
        'file_digests.mat must be one regular unlinked file.');
end

manifest_snapshot = contact_capture_file_snapshot(manifest_path);
manifest_sha256 = manifest_snapshot.sha256;
blob = contact_load_mat_bytes(manifest_snapshot.bytes);
if ~isequal(fieldnames(blob), {'file_digests'}) || ...
        ~isstruct(blob.file_digests) || ~isscalar(blob.file_digests)
    error('dataset_digest_manifest:BadManifest', ...
        'file_digests.mat must contain only one scalar file_digests struct.');
end
value = blob.file_digests;
expected_fields = sort({'schema'; 'scope'; 'digest_lines'; 'root'});
if ~isequal(sort(fieldnames(value)), expected_fields)
    error('dataset_digest_manifest:BadManifest', ...
        'file_digests has a missing or extra field.');
end
if ~strcmp(contact_exact_manifest_text(value.schema), ...
        'source-digests-v2') || ...
        ~strcmp(contact_exact_manifest_text(value.scope), ...
            'NNNN.mat+case_info.mat+damage_states.mat')
    error('dataset_digest_manifest:BadManifest', ...
        'file_digests schema/scope is not the exact R11 v2 contract.');
end
digest_lines = contact_exact_manifest_text(value.digest_lines);
root = contact_exact_manifest_text(value.root);
if isempty(regexp(root, '^[0-9a-f]{64}$', 'once')) || ...
        ~strcmp(contact_text_sha256(digest_lines), root)
    error('dataset_digest_manifest:BadRoot', ...
        'file_digests root is malformed or does not hash digest_lines.');
end
if contains(digest_lines, char(13)) || ...
        startsWith(digest_lines, newline) || endsWith(digest_lines, newline)
    error('dataset_digest_manifest:BadManifest', ...
        'digest_lines must be canonical LF text with no terminal newline.');
end

state_names = arrayfun(@(k) sprintf('%04d.mat', k), (1:n_states)', ...
    'UniformOutput', false);
expected_names = sort([state_names; {'case_info.mat'; 'damage_states.mat'}]);
lines = strsplit(digest_lines, newline)';
if numel(lines) ~= numel(expected_names)
    error('dataset_digest_manifest:BadInventory', ...
        'Digest inventory has %d entries; expected %d.', ...
        numel(lines), numel(expected_names));
end
observed_names = cell(size(lines));
observed_sha = cell(size(lines));
for k = 1:numel(lines)
    tokens = regexp(lines{k}, '^([^:]+):([0-9a-f]{64})$', ...
        'tokens', 'once');
    if isempty(tokens)
        error('dataset_digest_manifest:BadManifest', ...
            'Malformed canonical digest line %d.', k);
    end
    observed_names{k} = tokens{1};
    observed_sha{k} = tokens{2};
end
if ~isequal(observed_names, expected_names) || ...
        numel(unique(observed_names)) ~= numel(observed_names)
    error('dataset_digest_manifest:BadInventory', ...
        'Digest filenames are missing, extra, duplicated or out of order.');
end

listed = dir(dataset_dir);
listed = listed(~[listed.isdir]);
numbered_mask = ~cellfun(@isempty, ...
    regexpi({listed.name}, '^\d{4}\.mat$', 'once'));
numbered_names = sort({listed(numbered_mask).name})';
if ~isequal(numbered_names, state_names)
    error('dataset_digest_manifest:BadInventory', ...
        ['Dataset numbered-state inventory must be exactly ' ...
         '0001.mat..%04d.mat with canonical case.'], n_states);
end

verify_names = [{'case_info.mat'; 'damage_states.mat'}; ...
    state_names(state_indices)];
snapshot_paths = {manifest_path};
snapshot_observations = {manifest_snapshot.observation};
snapshot_sha256 = {manifest_sha256};
retained_snapshots = {};
if retain_snapshots
    retained_snapshots = {manifest_snapshot};
end
for k = 1:numel(verify_names)
    name = verify_names{k};
    index = find(strcmp(observed_names, name));
    if ~isscalar(index)
        error('dataset_digest_manifest:BadInventory', ...
            'No unique digest exists for %s.', name);
    end
    path = fullfile(dataset_dir, name);
    if ~contact_regular_nonsymlink(path)
        error('dataset_digest_manifest:BadEntry', ...
            ['Digested entry is missing, non-regular, or has a link ' ...
             'alias: %s'], name);
    end
    artifact_snapshot = contact_capture_file_snapshot(path);
    actual = artifact_snapshot.sha256;
    if ~strcmp(actual, observed_sha{index})
        error('dataset_digest_manifest:DigestMismatch', ...
            'Actual SHA-256 differs for %s.', name);
    end
    snapshot_paths{end + 1, 1} = path; %#ok<AGROW>
    snapshot_observations{end + 1, 1} = ...
        artifact_snapshot.observation; %#ok<AGROW>
    snapshot_sha256{end + 1, 1} = actual; %#ok<AGROW>
    if retain_snapshots
        retained_snapshots{end + 1, 1} = ...
            artifact_snapshot; %#ok<AGROW>
    end
    clear artifact_snapshot
end

% Re-assert the complete set after the final member read. These independent
% stable snapshots prevent a mixed pre/post validation result from being
% returned after a persistent replacement or ordinary concurrent mutation.
listed_after = dir(dataset_dir);
listed_after = listed_after(~[listed_after.isdir]);
numbered_after_mask = ~cellfun(@isempty, ...
    regexpi({listed_after.name}, '^\d{4}\.mat$', 'once'));
numbered_names_after = sort({listed_after(numbered_after_mask).name})';
if ~isequal(numbered_names_after, numbered_names)
    error('dataset_digest_manifest:InventoryRace', ...
        'Dataset numbered-state inventory changed during validation.');
end
for k = 1:numel(snapshot_paths)
    contact_assert_file_snapshot_unchanged( ...
        snapshot_paths{k}, snapshot_observations{k}, snapshot_sha256{k});
end
dataset_identity_after = contact_unlinked_path_identity(dataset_dir);
if ~isequal(dataset_identity_after, dataset_identity)
    error('dataset_digest_manifest:DatasetRace', ...
        'Dataset directory identity changed during validation.');
end

manifest = struct();
manifest.schema = 'source-digests-v2';
manifest.scope = 'NNNN.mat+case_info.mat+damage_states.mat';
manifest.digest_lines = digest_lines;
manifest.root = root;
manifest.names = observed_names;
manifest.sha256 = observed_sha;
manifest.verified_state_indices = state_indices;
manifest.file_digests_sha256 = manifest_sha256;
manifest.file_digests_snapshot = manifest_snapshot;
manifest.retained_snapshots = retained_snapshots;
end
