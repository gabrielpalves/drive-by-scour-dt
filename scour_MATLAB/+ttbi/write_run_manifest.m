function write_run_manifest( ...
        run_folder, case_info, campaign, state, run_folder_observation)
%WRITE_RUN_MANIFEST Persist case metadata and the semantic state catalogue.
%
% The MAT sidecars are consumed by Python.  case_info.txt mirrors scalar and
% text fields for a reviewer inspecting a run without MATLAB. Every file is
% completed under a temporary name and then installed in the same directory;
% A00 has already revoked the completion marker before calling this function.

ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
case_path = fullfile(run_folder, 'case_info.mat');
case_temp = fullfile(run_folder, '.case_info.mat.tmp');
if isfile(case_temp)
    delete(case_temp);
end
save(case_temp, 'case_info', '-mat');
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
[case_moved, case_message] = movefile(case_temp, case_path, 'f');
if ~case_moved
    error('ttbi:RunManifestCasePublish', ...
        'Could not atomically publish case_info.mat: %s', case_message);
end
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);

DamageStates = state.DamageStates;
BearingStates = state.BearingStates;
BearingFixity = state.BearingFixity;
LatentBearingFixity = state.LatentBearingFixity;
k_ref_bear = state.k_ref_bear;
scour_supports = campaign.scour_supports;
StateFamily = state.StateFamily;
AnchorTarget = state.AnchorTarget;
AnchorLevel = state.AnchorLevel;
StateUID = state.StateUID;
StateSeedID = state.StateSeedID;
StateNamedStreamSeedID = state.StateNamedStreamSeedID;
PassageNamedStreamSeedID = state.PassageNamedStreamSeedID;
PassageNamedStreamSeedIDFlat = state.PassageNamedStreamSeedIDFlat;
random_stream_schedule_version = state.random_stream_schedule_version;
state_stream_names = state.state_stream_names;
passage_stream_names = state.passage_stream_names;
LatentCrackOn = state.LatentCrackOn;
CrackOn = state.CrackOn;
damage_path = fullfile(run_folder, 'damage_states.mat');
damage_temp = fullfile(run_folder, '.damage_states.mat.tmp');
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
if isfile(damage_temp)
    delete(damage_temp);
end
save(damage_temp, ...
    'DamageStates', 'BearingStates', 'BearingFixity', ...
    'LatentBearingFixity', 'k_ref_bear', 'scour_supports', ...
    'StateFamily', 'AnchorTarget', 'AnchorLevel', 'StateUID', ...
    'StateSeedID', 'StateNamedStreamSeedID', ...
    'PassageNamedStreamSeedID', 'PassageNamedStreamSeedIDFlat', ...
    'random_stream_schedule_version', 'state_stream_names', ...
    'passage_stream_names', 'LatentCrackOn', 'CrackOn', '-mat');
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
[damage_moved, damage_message] = movefile( ...
    damage_temp, damage_path, 'f');
if ~damage_moved
    error('ttbi:RunManifestDamagePublish', ...
        'Could not atomically publish damage_states.mat: %s', ...
        damage_message);
end

text_path = fullfile(run_folder, 'case_info.txt');
text_temp = fullfile(run_folder, '.case_info.txt.tmp');
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
if isfile(text_temp)
    delete(text_temp);
end
file_id = fopen(text_temp, 'w');
if file_id < 0
    error('A00:CaseInfoTextOpen', ...
        'Could not open temporary case_info text: %s', text_temp);
end
cleanup = onCleanup(@() fclose(file_id));
field_names = fieldnames(case_info);
fprintf(file_id, '%% TTBI dataset — case manifest\n');
for field_index = 1:numel(field_names)
    field_name = field_names{field_index};
    value = case_info.(field_name);
    if ischar(value)
        fprintf(file_id, '%-26s : %s\n', field_name, value);
    else
        fprintf(file_id, '%-26s : %g\n', field_name, value);
    end
end
clear cleanup  % close before the same-directory atomic rename
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
[text_moved, text_message] = movefile(text_temp, text_path, 'f');
if ~text_moved
    error('ttbi:RunManifestTextPublish', ...
        'Could not atomically publish case_info.txt: %s', text_message);
end
ttbi.assert_generation_output_directory( ...
    run_folder, run_folder_observation);
end
