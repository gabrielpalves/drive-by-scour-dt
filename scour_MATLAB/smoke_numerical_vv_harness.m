% smoke_numerical_vv_harness.m
% Fast foundation checks only; no coupled dynamic or campaign execution.
clear; clc;

P = numerical_vv_protocol_definition();
assert(strcmp(P.schema, 'numerical-vv-protocol-v1'));
assert(isequal({P.mesh_levels.id}, {'M0', 'M1', 'M2', 'M3'}));
assert(isequal({P.mesh_levels.role}, { ...
    'current-production', 'primary-finer-comparison', ...
    'second-primary-finer-comparison', 'conditional-resolution'}));
assert(isequal([P.time_levels.requested_dt_ms], [1, 0.5, 0.25, 0.125]));
assert(~P.qualification_ready);
assert(~P.qualification_verifier_implemented);

alignment = struct2table(P.registered_support_alignment);
L60 = alignment(alignment.geometry_id == "L60_3span", :);
L99 = alignment(alignment.geometry_id == "L99p6_4span", :);
for level = ["M0", "M1", "M2", "M3"]
    local_assert_supports(L60, level, [0, 20, 40, 60]);
    local_assert_supports(L99, level, [0, 24.9, 49.8, 74.7, 99.6]);
end
assert(all(alignment.exactly_aligned_for_qualification));
L60_sequence = P.geometry_mesh_sequences(strcmp( ...
    {P.geometry_mesh_sequences.geometry_id}, 'L60_3span'));
L99_sequence = P.geometry_mesh_sequences(strcmp( ...
    {P.geometry_mesh_sequences.geometry_id}, 'L99p6_4span'));
assert(isequal([L60_sequence.bridge_elements_per_sleeper_bay], ...
    [3, 6, 12, 24]));
assert(isequal([L60_sequence.rail_elements_per_sleeper_bay], ...
    [2, 4, 8, 16]));
assert(isequal([L99_sequence.bridge_elements_per_sleeper_bay], ...
    [2, 4, 8, 16]));
assert(isequal([L99_sequence.rail_elements_per_sleeper_bay], ...
    [2, 4, 8, 16]));

legacy = struct2table(P.rejected_legacy_support_alignment);
legacy_L60 = legacy(legacy.geometry_id == "L60_3span", :);
legacy_L99 = legacy(legacy.geometry_id == "L99p6_4span", :);
local_assert_supports(legacy_L60, 'M0', [0, 19.8, 40.2, 60]);
local_assert_supports(legacy_L60, 'M1', [0, 20.1, 39.9, 60]);
local_assert_supports(legacy_L60, 'M2', [0, 19.95, 40.05, 60]);
local_assert_supports(legacy_L60, 'M3', [0, 20.025, 39.975, 60]);
local_assert_supports(legacy_L99, 'M0', [0, 24.6, 49.8, 74.4, 99.6]);
assert(any(~legacy_L60.exactly_aligned_for_qualification));
assert(any(~legacy_L99.exactly_aligned_for_qualification));
fprintf('source-locked aligned grids and rejected legacy snapping: PASS\n');

gci = numerical_vv_scalar_convergence( ...
    [0.6, 0.3, 0.15], 1+[0.6, 0.3, 0.15].^2, 1e-12);
assert(gci.gci_available);
assert(abs(gci.observed_order-2) < 1e-12);
nonmonotone = numerical_vv_scalar_convergence( ...
    [0.6, 0.3, 0.15], [1.2, 0.9, 1.0], 1e-12);
assert(~nonmonotone.gci_available);
assert(strcmp(nonmonotone.reason, 'nonmonotone-sequence'));
switched = numerical_vv_scalar_convergence( ...
    [0.6, 0.3, 0.15], [1.2, 1.1, 1.05], 1e-12, ...
    'ModeConsistent', false);
assert(~switched.gci_available);
assert(strcmp(switched.reason, 'mode-matching-changed'));
unreasonable_order = numerical_vv_scalar_convergence( ...
    [4, 2, 1], [2^20+2, 2, 1], 1e-12);
assert(~unreasonable_order.gci_available);
assert(strcmp(unreasonable_order.reason, ...
    'nonfinite-nonpositive-or-unreasonable-observed-order'));
local_assert_throws(@() numerical_vv_scalar_convergence( ...
    [4, 2, 1], [2^20+2, 2, 1], 1e-12, ...
    'MaxObservedOrder', 100), ...
    'numerical_vv:ObservedOrderCeilingEscalation');
fprintf('scalar convergence fail-closed paths: PASS\n');

x = 0:0.01:2;
y = [sin(2*pi*x); cos(2*pi*x)];
metrics = numerical_vv_waveform_metrics(x, y, x, y, x, ...
    'ChannelNames', {'sin', 'cos'});
assert(all(metrics.nrmse == 0));
assert(all(metrics.normalized_max_error == 0));
assert(all(abs(metrics.correlation-1) < 1e-14));
local_assert_throws(@() numerical_vv_waveform_metrics( ...
    x, y, x, y, -0.01:0.01:2), 'numerical_vv:CommonGridOutsideDomain');
fprintf('explicit-grid waveform metrics: PASS\n');

old_L60 = numerical_vv_coupled_mesh_preflight(60, 3, 2, 2, ...
    'Assemble', false);
assert(~old_L60.support_alignment_pass);
aligned_L60 = numerical_vv_coupled_mesh_preflight(60, 3, 3, 2, ...
    'Assemble', true);
assert(aligned_L60.support_alignment_pass && aligned_L60.assembly_pass);
assert(aligned_L60.redux == 0);
assert(aligned_L60.bridge_sleeper_coupling_count == 101);
assert(aligned_L60.bridge_elements_per_bay ~= ...
    aligned_L60.rail_elements_per_bay);
assert(abs(aligned_L60.ballast_B54_lumped_total_kg-53671.4) < 1e-10);
assert(abs(aligned_L60.ballast_continuous_bay_total_kg-53140.0) < 1e-10);
assert(abs(aligned_L60.ballast_full_sleeper_total_kg-53671.4) < 1e-10);
assert(abs(aligned_L60.ballast_delta_vs_old_L60_n2_total_kg-265.7) < 1e-10);
assert(aligned_L60.ballast_mass_assembly_max_abs_error_kg < 1e-8);
assert(abs(aligned_L60.ballast_mass_assembly_actual_total_kg-53671.4) < 1e-8);
fine_L60 = numerical_vv_coupled_mesh_preflight(60, 3, 6, 4, ...
    'Assemble', true, 'Redux', 1);
assert(fine_L60.assembly_pass);
assert(fine_L60.ballast_mass_assembly_max_abs_error_kg < 1e-8);
assert(abs(fine_L60.ballast_mass_assembly_actual_total_kg- ...
    aligned_L60.ballast_mass_assembly_actual_total_kg) < 1e-8);
for density = [12, 8; 24, 16]'
    refined_L60 = numerical_vv_coupled_mesh_preflight(60, 3, ...
        density(1), density(2), 'Assemble', true, 'Redux', 1);
    assert(refined_L60.assembly_pass);
    assert(refined_L60.ballast_mass_assembly_max_abs_error_kg < 1e-8);
    assert(abs(refined_L60.ballast_mass_assembly_actual_total_kg- ...
        aligned_L60.ballast_mass_assembly_actual_total_kg) < 1e-8);
end
aligned_L99 = numerical_vv_coupled_mesh_preflight(99.6, 4, 2, 2, ...
    'Assemble', true);
assert(aligned_L99.support_alignment_pass && aligned_L99.assembly_pass);
assert(aligned_L99.redux == 0);
assert(aligned_L99.bridge_sleeper_coupling_count == 167);
assert(abs(aligned_L99.ballast_B54_lumped_total_kg-88743.8) < 1e-10);
assert(abs(aligned_L99.ballast_continuous_bay_total_kg-88212.4) < 1e-10);
assert(abs(aligned_L99.ballast_full_sleeper_total_kg-88743.8) < 1e-10);
assert(aligned_L99.ballast_mass_assembly_max_abs_error_kg < 1e-8);
assert(abs(aligned_L99.ballast_mass_assembly_actual_total_kg-88743.8) < 1e-8);
assert(strcmp(aligned_L99.ballast_endpoint_mass_convention_status, ...
    ['PROXY_INFORMED_SUPPORT_POINT_LUMPS_' ...
     'AUTHOR_CHOSEN_ENDPOINT_OWNERSHIP']));
fine_L99 = numerical_vv_coupled_mesh_preflight(99.6, 4, 4, 4, ...
    'Assemble', true, 'Redux', 1);
assert(fine_L99.assembly_pass);
assert(fine_L99.ballast_mass_assembly_max_abs_error_kg < 1e-8);
assert(abs(fine_L99.ballast_mass_assembly_actual_total_kg- ...
    aligned_L99.ballast_mass_assembly_actual_total_kg) < 1e-8);
for density = [8, 8; 16, 16]'
    refined_L99 = numerical_vv_coupled_mesh_preflight(99.6, 4, ...
        density(1), density(2), 'Assemble', true, 'Redux', 1);
    assert(refined_L99.assembly_pass);
    assert(refined_L99.ballast_mass_assembly_max_abs_error_kg < 1e-8);
    assert(abs(refined_L99.ballast_mass_assembly_actual_total_kg- ...
        aligned_L99.ballast_mass_assembly_actual_total_kg) < 1e-8);
end
fprintf('geometry-specific mixed-mesh B54 mass/assembly preflight: PASS\n');

output_dir = tempname;
cleanup = onCleanup(@() local_cleanup(output_dir));
receipt = numerical_vv_micro_run(output_dir, ...
    'BridgeLengthsM', 60, 'MeshLevelIDs', {'M0'}, ...
    'TimeLevelIDs', {'dt1'}, 'MaxMeshCases', 1);
assert(receipt.integrity_pass);
assert(strcmp(receipt.status, 'NONQUALIFYING_INTEGRITY_PASS'));
assert(~receipt.numerical_verification_claim_authorized);
assert(~receipt.physical_validation_claim_authorized);
local_assert_throws(@() numerical_vv_validate_package(output_dir), ...
    'numerical_vv:NonqualifyingMicro');
local_assert_throws(@() numerical_vv_validate_package(output_dir, ...
    'AllowNonqualifyingMicro', true, 'RequireQualification', true), ...
    'numerical_vv:QualificationVerifierNotImplemented');

forged_dir = [tempname, '_forged_qualification'];
forged_cleanup = onCleanup(@() local_cleanup(forged_dir));
assert(copyfile(output_dir, forged_dir));
forged_manifest_path = fullfile(forged_dir, 'manifest.json');
forged_manifest = jsondecode(fileread(forged_manifest_path));
forged_manifest.run_kind = 'qualification';
local_write_text(forged_manifest_path, ...
    [jsonencode(forged_manifest, 'PrettyPrint', true), newline]);
forged_sha = numerical_vv_file_sha256(forged_manifest_path);
local_write_text(fullfile(forged_dir, '_RUN_COMPLETE'), sprintf([ ...
    'schema=numerical-vv-completion-v1\nrun_kind=qualification\n' ...
    'status=COMPLETE\nmanifest_sha256=%s\n'], forged_sha));
local_assert_throws(@() numerical_vv_validate_package(forged_dir, ...
    'AllowNonqualifyingMicro', true), 'numerical_vv:UnsupportedRunKind');
fprintf('qualification self-attestation refused: PASS\n');

semantic_dir = [tempname, '_forged_mesh_semantics'];
semantic_cleanup = onCleanup(@() local_cleanup(semantic_dir));
assert(copyfile(output_dir, semantic_dir));
semantic_case_path = fullfile(semantic_dir, 'case_table.csv');
semantic_case = readtable(semantic_case_path, 'Delimiter', ',', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
semantic_case.bridge_elements_per_sleeper_bay(1) = 2;
semantic_case.bridge_nominal_h_m(1) = 0.3;
semantic_case.bridge_actual_h_m(1) = 0.3;
semantic_case.bridge_n_elements(1) = 200;
semantic_case.bridge_n_nodes(1) = 201;
semantic_case.max_support_offset_m(1) = 0.1;
semantic_case.support_alignment_pass(1) = true;
writetable(semantic_case, semantic_case_path);
semantic_descriptor_path = fullfile(semantic_dir, 'descriptor_hashes.csv');
semantic_descriptor = readtable(semantic_descriptor_path, ...
    'Delimiter', ',', 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
descriptor = jsondecode(char(semantic_descriptor.descriptor_json(1)));
descriptor.bridge_elements_per_sleeper_bay = 2;
descriptor.bridge_nominal_h_m = 0.3;
descriptor_json = jsonencode(descriptor);
semantic_descriptor.descriptor_json(1) = string(descriptor_json);
semantic_descriptor.descriptor_hash(1) = string( ...
    numerical_vv_sha256_bytes(unicode2native(descriptor_json, 'UTF-8')));
writetable(semantic_descriptor, semantic_descriptor_path);
local_refresh_manifest(semantic_dir);
local_assert_throws(@() numerical_vv_validate_package(semantic_dir, ...
    'AllowNonqualifyingMicro', true), ...
    'numerical_vv:SourceLockedMeshMismatch');
repo_root = fileparts(fileparts(mfilename('fullpath')));
python_checker = fullfile(repo_root, 'check_numerical_vv_package.py');
python_command = sprintf('python "%s" "%s" --allow-nonqualifying-micro', ...
    python_checker, semantic_dir);
[python_status, python_output] = system(python_command);
assert(python_status ~= 0 && contains(python_output, ...
    'bridge count differs from source lock'), ...
    ['Python validator did not reject the source-locked mesh forgery as ' ...
     'expected (status=%d): %s'], python_status, python_output);
fprintf('rehashed bridge/rail semantic forgery refused by MATLAB/Python: PASS\n');

duplicate_dir = [tempname, '_duplicate_descriptor'];
duplicate_cleanup = onCleanup(@() local_cleanup(duplicate_dir));
assert(copyfile(output_dir, duplicate_dir));
duplicate_descriptor_path = fullfile(duplicate_dir, ...
    'descriptor_hashes.csv');
duplicate_descriptor = readtable(duplicate_descriptor_path, ...
    'Delimiter', ',', 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
duplicate_descriptor = [duplicate_descriptor; duplicate_descriptor(1, :)];
writetable(duplicate_descriptor, duplicate_descriptor_path);
local_refresh_manifest(duplicate_dir);
local_assert_throws(@() numerical_vv_validate_package(duplicate_dir, ...
    'AllowNonqualifyingMicro', true), ...
    'numerical_vv:DescriptorInventoryMismatch');
duplicate_command = sprintf( ...
    'python "%s" "%s" --allow-nonqualifying-micro', ...
    python_checker, duplicate_dir);
[duplicate_status, duplicate_output] = system(duplicate_command);
assert(duplicate_status ~= 0 && contains(duplicate_output, ...
    'descriptor/case identity mismatch'), ...
    ['Python validator did not reject the duplicate descriptor row as ' ...
     'expected (status=%d): %s'], duplicate_status, duplicate_output);
fprintf('Python exact descriptor inventory enforcement: PASS\n');

protocol_dir = [tempname, '_forged_protocol_snapshot'];
protocol_cleanup = onCleanup(@() local_cleanup(protocol_dir));
assert(copyfile(output_dir, protocol_dir));
protocol_path = fullfile(protocol_dir, 'protocol_snapshot.json');
protocol_snapshot = jsondecode(fileread(protocol_path));
protocol_snapshot.geometry_mesh_sequences(1).bridge_elements_per_sleeper_bay = 2;
protocol_snapshot.geometry_mesh_sequences(1).bridge_nominal_h_m = 0.3;
local_write_text(protocol_path, ...
    [jsonencode(protocol_snapshot, 'PrettyPrint', true), newline]);
local_refresh_manifest(protocol_dir);
local_assert_throws(@() numerical_vv_validate_package(protocol_dir, ...
    'AllowNonqualifyingMicro', true), ...
    'numerical_vv:ProtocolSnapshotMismatch');
protocol_command = sprintf( ...
    'python "%s" "%s" --allow-nonqualifying-micro', ...
    python_checker, protocol_dir);
[protocol_status, protocol_output] = system(protocol_command);
assert(protocol_status ~= 0 && contains(protocol_output, ...
    'protocol bridge count differs from source lock'), ...
    ['Python validator did not reject the forged protocol snapshot as ' ...
     'expected (status=%d): %s'], protocol_status, protocol_output);
fprintf('source-locked protocol snapshot mutation guards: PASS\n');

raw_dir = [tempname, '_forged_raw_fixture'];
raw_cleanup = onCleanup(@() local_cleanup(raw_dir));
assert(copyfile(output_dir, raw_dir));
raw_path = fullfile(raw_dir, 'raw_bridge_fixture.mat');
raw_fixture = load(raw_path, 'runs');
runs = raw_fixture.runs;
runs(1).bridge_n_elements = runs(1).bridge_n_elements-1;
save(raw_path, 'runs', '-v7.3');
local_refresh_manifest(raw_dir);
local_assert_throws(@() numerical_vv_validate_package(raw_dir, ...
    'AllowNonqualifyingMicro', true), ...
    'numerical_vv:RawFixtureMismatch');
fprintf('MATLAB raw-fixture semantic binding guard: PASS\n');

case_path = fullfile(output_dir, 'case_table.csv');
fid = fopen(case_path, 'ab');
assert(fid >= 0);
assert(fwrite(fid, uint8('tamper'), 'uint8') == 6);
assert(fclose(fid) == 0);
local_assert_throws(@() numerical_vv_validate_package(output_dir, ...
    'AllowNonqualifyingMicro', true), ...
    'numerical_vv:ArtifactDigestMismatch');
fprintf('hashed nonqualifying package and mutation guard: PASS\n');

fprintf('NUMERICAL V&V HARNESS FOUNDATION: PASS\n');

function local_assert_supports(table_value, level, expected)
rows = table_value(table_value.mesh_level == string(level), :);
[~, order] = sort(rows.support_number);
actual = rows.realized_coordinate_m(order)';
assert(numel(actual) == numel(expected));
assert(max(abs(actual-expected)) < 1e-10);
end

function local_assert_throws(fun, expected_id)
did_throw = false;
try
    fun();
catch ME
    did_throw = strcmp(ME.identifier, expected_id);
    if ~did_throw
        rethrow(ME)
    end
end
assert(did_throw, 'Expected error %s.', expected_id);
end

function local_cleanup(path)
if exist(path, 'dir') == 7
    rmdir(path, 's');
end
end

function local_write_text(path, value)
bytes = unicode2native(char(value), 'UTF-8');
fid = fopen(path, 'wb');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
assert(fwrite(fid, bytes, 'uint8') == numel(bytes));
end

function local_refresh_manifest(package_dir)
manifest_path = fullfile(package_dir, 'manifest.json');
manifest = jsondecode(fileread(manifest_path));
for k = 1:numel(manifest.artifacts)
    artifact_path = fullfile(package_dir, manifest.artifacts(k).path);
    info = dir(artifact_path);
    assert(isscalar(info) && ~info.isdir);
    manifest.artifacts(k).bytes = info.bytes;
    manifest.artifacts(k).sha256 = ...
        numerical_vv_file_sha256(artifact_path);
end
local_write_text(manifest_path, ...
    [jsonencode(manifest, 'PrettyPrint', true), newline]);
manifest_sha = numerical_vv_file_sha256(manifest_path);
local_write_text(fullfile(package_dir, '_RUN_COMPLETE'), sprintf([ ...
    'schema=numerical-vv-completion-v1\n' ...
    'run_kind=nonqualifying_micro\n' ...
    'status=NON_QUALIFYING_COMPLETE\nmanifest_sha256=%s\n'], ...
    manifest_sha));
end
