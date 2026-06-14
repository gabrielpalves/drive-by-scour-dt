results_folder = 'Results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Get list of existing run folders (subdirectories)
run_dirs = dir(results_folder);
run_dirs = run_dirs([run_dirs.isdir] & ~ismember({run_dirs.name}, {'.', '..'}));

% =========================================================================
% Show run status + disk usage + completion + sort by date
% =========================================================================

if ~isempty(run_dirs)
    % Sort runs by newest first
    [~, idx] = sort([run_dirs.datenum], 'descend');
    run_dirs = run_dirs(idx);
end

fprintf('\nRun Status (sorted by date):\n');
fprintf('------------------------------------------------------------------------------------------\n');
fprintf('%-3s %-22s %-12s %-20s %-10s\n', '#', 'Run Name', 'Status', 'Completion', 'Size(MB)');
fprintf('------------------------------------------------------------------------------------------\n');

for i = 1:length(run_dirs)

    run_name = run_dirs(i).name;
    run_path = fullfile(results_folder, run_name);

    ini_file = fullfile(run_path, 'tempo_inicial.mat');
    end_file = fullfile(run_path, 'tempo_final.mat');

    % ------------------------------------------------------------
    % Determine run status
    % ------------------------------------------------------------
    if exist(ini_file, 'file') && exist(end_file, 'file')
        status = 'COMPLETE';
    elseif exist(ini_file, 'file') && ~exist(end_file, 'file')
        status = 'INCOMPLETE';
    else
        status = 'UNAVAILABLE';
    end

    % ------------------------------------------------------------
    % AUTO-DETECT TOTAL_DC (4-digit filenames)
    % ------------------------------------------------------------
    mat_files = dir(fullfile(run_path, '*.mat'));

    max_idx = 0;
    completed_dc = 0;

    for k = 1:length(mat_files)
        tok = regexp(mat_files(k).name, '(\d{4})\.mat', 'tokens');  % <-- 4-digit filenames
        if ~isempty(tok)
            dc_num = str2double(tok{1}{1});
            completed_dc = completed_dc + 1;
            if dc_num > max_idx
                max_idx = dc_num;
            end
        end
    end

    if max_idx > 0
        TOTAL_DC = max_idx;   % auto detected
    else
        TOTAL_DC = NaN;       % none found
    end

    % ------------------------------------------------------------
    % Compute completion %
    % ------------------------------------------------------------
    if isnan(TOTAL_DC)
        completion_str = '---';
    else
        pct = 100 * completed_dc / TOTAL_DC;
        completion_str = sprintf('%d/%d (%5.1f%%)', completed_dc, TOTAL_DC, pct);
    end

    % ------------------------------------------------------------
    % Disk usage
    % ------------------------------------------------------------
    all_files = dir(fullfile(run_path, '**', '*'));
    total_bytes = sum([all_files.bytes]);
    size_MB = total_bytes / (1024^2);

    % ------------------------------------------------------------
    % Print run information
    % ------------------------------------------------------------
    fprintf('%-3d %-22s %-12s %-20s %8.2f\n', ...
            i, run_name, status, completion_str, size_MB);

end

fprintf('------------------------------------------------------------------------------------------\n\n');


% =========================================================================
% Ask user to start new or resume run
% =========================================================================
fprintf('Available runs:\n');
for i = 1:length(run_dirs)
    fprintf('%d - %s\n', i, run_dirs(i).name);
end

choice = 0;
if isempty(run_dirs)
    fprintf('Starting a NEW run\n');
else
    fprintf('0 - Start a NEW run\n');

    choice = input('Enter your choice: ');
end

if choice == 0
    % Start a new run (create timestamped folder)
    tempo_inicial = datetime('now', 'Format', 'dd_MM_yyyy_HH_mm_ss');
    tempo_inicial_str = char(tempo_inicial);
    fprintf('Starting NEW run: %s\n', tempo_inicial_str);

    run_path = fullfile(results_folder, tempo_inicial_str);
    mkdir(run_path);

    % Save tempo_inicial inside the folder for traceability
    save(fullfile(run_path, 'tempo_inicial.mat'), 'tempo_inicial');

else
    % Resume an existing run
    tempo_inicial_str = run_dirs(choice).name;
    run_path = fullfile(results_folder, tempo_inicial_str);
    fprintf('Resuming run: %s\n', tempo_inicial_str);

    tempo_file = fullfile(run_path, 'tempo_inicial.mat');
    if exist(tempo_file, 'file')
        % Load from saved .mat (recommended)
        load(tempo_file, 'tempo_inicial');
    else
        % Fallback: reconstruct from folder name if .mat doesn’t exist
        tempo_inicial = datetime(tempo_inicial_str, 'InputFormat', 'dd_MM_yyyy_HH_mm_ss');
    end
end