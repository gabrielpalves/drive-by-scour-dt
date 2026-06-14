function list_all_runs()
% Folder containing all runs
results_folder = 'Results';

% Get list of subfolders (each run)
run_dirs = dir(results_folder);
run_dirs = run_dirs([run_dirs.isdir] & ~ismember({run_dirs.name}, {'.', '..'}));

fprintf('\n=== Available Runs and Durations ===\n');

if isempty(run_dirs)
    fprintf('No previous runs found.\n');
else
    for i = 1:length(run_dirs)
        run_name = run_dirs(i).name;
        run_path = fullfile(results_folder, run_name);

        tempo_inicial_file = fullfile(run_path, 'tempo_inicial.mat');
        tempo_final_file   = fullfile(run_path, 'tempo_final.mat');

        if exist(tempo_inicial_file, 'file')
            load(tempo_inicial_file, 'tempo_inicial');
        else
            tempo_inicial = [];
        end

        if exist(tempo_final_file, 'file')
            load(tempo_final_file, 'tempo_final');
        else
            tempo_final = [];
        end

        if ~isempty(tempo_inicial) && ~isempty(tempo_final)
            tempo_total = tempo_final - tempo_inicial;
            fprintf('%2d) %s | Duration: %s\n', i, run_name, char(tempo_total));
        elseif ~isempty(tempo_inicial)
            fprintf('%2d) %s | Incomplete (still running or interrupted)\n', i, run_name);
        else
            fprintf('%2d) %s | Invalid (missing tempo_inicial.mat)\n', i, run_name);
        end
    end
end
fprintf('====================================\n\n');
end
