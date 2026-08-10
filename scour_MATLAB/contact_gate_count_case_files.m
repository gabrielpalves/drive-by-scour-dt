function count = contact_gate_count_case_files(cases_dir)
%CONTACT_GATE_COUNT_CASE_FILES Count published case MAT artifacts.

files = dir(fullfile(cases_dir, '*_case.mat'));
count = numel(files);
end
