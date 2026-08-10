function smoke_contact_gate_publication()
%SMOKE_CONTACT_GATE_PUBLICATION Exercise durable gate-publication primitives.
%
% This smoke stays separate from the numerical closure fixture because it
% checks a different responsibility: immutable writes, deterministic sidecar
% verification, interrupted-temp recovery, exact case inventories and solver
% execution binding. It writes only beneath MATLAB's temporary directory.

work_dir = tempname;
mkdir(work_dir);
cleanup = onCleanup(@() rmdir(work_dir, 's'));

% MAT artifacts are create-once.
mat_path = fullfile(work_dir, 'value.mat');
contact_gate_save_atomic(mat_path, 'value', 42);
loaded = load(mat_path);
assert(isequal(fieldnames(loaded), {'value'}) && loaded.value == 42);
try
    contact_gate_save_atomic(mat_path, 'value', 43);
    error('smoke_contact_gate_publication:MissingError', ...
        'An existing immutable MAT artifact was accepted.');
catch ME
    assert(strcmp(ME.identifier, ...
        'contact_closure_gate:ImmutableArtifact'));
end

% Text projections are verified byte-for-byte, never overwritten.
text_path = fullfile(work_dir, 'note.txt');
contact_gate_write_text_atomic(text_path, sprintf('alpha\n'));
contact_gate_write_or_verify_text(text_path, sprintf('alpha\n'));
try
    contact_gate_write_or_verify_text(text_path, sprintf('beta\n'));
    error('smoke_contact_gate_publication:MissingError', ...
        'A changed deterministic sidecar was accepted.');
catch ME
    assert(strcmp(ME.identifier, 'contact_closure_gate:ImmutableSummary'));
end

% An orphan temp is non-evidence and recoverable. Final+temp is ambiguous.
orphan_path = fullfile(work_dir, 'orphan.mat');
orphan_tmp = [orphan_path, '.tmp'];
fid = fopen(orphan_tmp, 'wb');
assert(fid >= 0);
assert(fwrite(fid, uint8(1), 'uint8') == 1);
assert(fclose(fid) == 0);
contact_gate_recover_one_temp(orphan_path, orphan_tmp);
assert(~isfile(orphan_tmp));

ambiguous_path = fullfile(work_dir, 'ambiguous.mat');
fid = fopen(ambiguous_path, 'wb');
assert(fid >= 0);
assert(fwrite(fid, uint8(1), 'uint8') == 1);
assert(fclose(fid) == 0);
fid = fopen([ambiguous_path, '.tmp'], 'wb');
assert(fid >= 0);
assert(fwrite(fid, uint8(2), 'uint8') == 1);
assert(fclose(fid) == 0);
try
    contact_gate_recover_one_temp( ...
        ambiguous_path, [ambiguous_path, '.tmp']);
    error('smoke_contact_gate_publication:MissingError', ...
        'A final artifact plus temp ambiguity was accepted.');
catch ME
    assert(strcmp(ME.identifier, 'contact_closure_gate:AmbiguousTemp'));
end

% A completed case inventory is exactly one MAT/JSON pair per ordinal.
cases_dir = fullfile(work_dir, 'cases');
mkdir(cases_dir);
canonical_case = struct('status', 'PASS');
save(fullfile(cases_dir, '0001_case.mat'), 'canonical_case');
fid = fopen(fullfile(cases_dir, '0001_case.json'), 'wb');
assert(fid >= 0);
json_bytes = unicode2native(sprintf('{"status":"PASS"}\n'), 'UTF-8');
assert(fwrite(fid, json_bytes, 'uint8') == numel(json_bytes));
assert(fclose(fid) == 0);
contact_gate_validate_case_inventory(cases_dir, 1, true);
assert(numel(contact_gate_case_artifact_root(cases_dir, 1)) == 64);
fid = fopen(fullfile(cases_dir, '0001_case.json.tmp'), 'wb');
assert(fid >= 0);
assert(fwrite(fid, uint8(1), 'uint8') == 1);
assert(fclose(fid) == 0);
try
    contact_gate_validate_case_inventory(cases_dir, 1, true);
    error('smoke_contact_gate_publication:MissingError', ...
        'A foreign case temp was accepted.');
catch ME
    assert(strcmp(ME.identifier, 'contact_closure_gate:CaseInventory'));
end

% The gate independently re-resolves and re-hashes all production modules.
[~, generator_digest_lines, ~] = generator_source_root();
solver = contact_study_solver();
manifest = solver.solver_source_manifest(generator_digest_lines);
solver_root = contact_gate_validate_solver_execution_manifest(manifest);
assert(ischar(solver_root) && numel(solver_root) == 64);
mutated_manifest = manifest;
mutated_manifest.sha256{1} = repmat('0', 1, 64);
try
    contact_gate_validate_solver_execution_manifest(mutated_manifest);
    error('smoke_contact_gate_publication:MissingError', ...
        'A changed solver-module digest was accepted.');
catch ME
    assert(strcmp(ME.identifier, 'contact_closure_gate:SolverExecution'));
end

% Direct table conversion is part of both frozen-plan and report projection.
plain = contact_gate_plain_table(table((1:2)', 'VariableNames', {'ordinal'}));
assert(isequal(plain.ordinal, (1:2)'));

fprintf('gate publication atomic/resume/inventory binding: PASS\n');
end
