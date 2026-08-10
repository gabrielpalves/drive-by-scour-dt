function count = windows_hardlink_count(path)
%WINDOWS_HARDLINK_COUNT Query fsutil without invoking cmd.exe.

try
    system_directory = char(System.Environment.SystemDirectory);
catch system_error
    error('ttbi:HardlinkWindowsRoot', ...
        'Cannot obtain the native Windows system directory: %s', ...
        system_error.message);
end
executable = fullfile(system_directory, 'fsutil.exe');
if ~isfile(executable)
    error('ttbi:HardlinkWindowsTool', ...
        'Windows hard-link query tool is missing: %s', executable);
end

[raw_lines, exit_code] = ttbi.run_small_process( ...
    {executable, 'hardlink', 'list', ...
    char(javaObject('java.io.File', path).getAbsolutePath())}, 10);
lines = cellfun(@(line) strtrim(char(line)), raw_lines, ...
    'UniformOutput', false);
lines = lines(~cellfun(@isempty, lines));
if exit_code ~= 0 || isempty(lines)
    error('ttbi:HardlinkWindowsQuery', ...
        'fsutil could not attest hard links for %s.', path);
end
count = numel(lines);
end
