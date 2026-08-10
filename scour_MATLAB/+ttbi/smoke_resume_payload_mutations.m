function smoke_resume_payload_mutations( ...
        valid_path, state_index, scratch, context)
%SMOKE_RESUME_PAYLOAD_MUTATIONS Exercise fail-closed resume mutations.

saved = load(valid_path);
baseline = saved.data;
ttbi.validate_resumed_state_payload( ...
    baseline, state_index, 'baseline mutation fixture', context);

mutated = baseline;
mutated.unexpected_payload_field = true;
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'unexpected payload field', context);

mutated = rmfield(baseline, 'Dano');
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'missing Dano', context);

mutated = baseline;
mutated.Dano = mutated.Dano + 1;
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'wrong scalar damage target', context);

mutated = baseline;
mutated.Temperatura(1) = mutated.Temperatura(1) + 1;
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'wrong named-stream operation', context);

mutated = baseline;
mutated.AcelRodaPrimVag{1} = mutated.AcelRodaPrimVag{1}(1:3, :);
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'three-row wheel channel', context);

mutated = baseline;
mutated.crop_start(1) = 1002;
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'changed registered crop start', context);

mutated = baseline;
mutated.track_log{1}.unexpected = true;
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'changed track descriptor', context);

mutated = baseline;
mutated.contact_log(1, :) = [0, 0, 0, 1];
ttbi.assert_resume_payload_rejected( ...
    mutated, state_index, 'contact flag/force disagreement', context);

% Compression is negative under the signed solver convention and must remain
% admissible. This acceptance assertion guards against a future sign mistake.
compressive = baseline;
compressive.contact_log(1, :) = [0, 0, 0, -1];
ttbi.validate_resumed_state_payload( ...
    compressive, state_index, 'valid signed-compression fixture', context);

inventory_root = fullfile(scratch, 'resume_inventory_mutations');
mkdir(inventory_root);

out_of_range = fullfile(inventory_root, 'out_of_range');
mkdir(out_of_range);
copyfile(valid_path, fullfile(out_of_range, '9999.mat'));
ttbi.assert_resume_folder_rejected( ...
    out_of_range, 'out-of-range 9999.mat', context);

noncanonical = fullfile(inventory_root, 'noncanonical');
mkdir(noncanonical);
copyfile(valid_path, fullfile(noncanonical, '1.mat'));
ttbi.assert_resume_folder_rejected( ...
    noncanonical, 'noncanonical 1.mat', context);

extra_top = fullfile(inventory_root, 'extra_top');
mkdir(extra_top);
canonical_name = sprintf('%04d.mat', state_index);
extra_top_path = fullfile(extra_top, canonical_name);
copyfile(valid_path, extra_top_path);
unexpected_top_level = true; %#ok<NASGU>
save(extra_top_path, 'unexpected_top_level', '-append');
ttbi.assert_resume_folder_rejected( ...
    extra_top, 'unexpected top-level variable', context);

corrupt_payload = fullfile(inventory_root, 'corrupt_payload');
mkdir(corrupt_payload);
corrupt_path = fullfile(corrupt_payload, canonical_name);
copyfile(valid_path, corrupt_path);
container = load(corrupt_path);
container.data.crop_start(1) = 1002;
save(corrupt_path, '-struct', 'container');
ttbi.assert_resume_folder_rejected( ...
    corrupt_payload, 'stamped payload with wrong crop', context);
end
