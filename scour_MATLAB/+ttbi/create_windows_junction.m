function create_windows_junction(alias_path, target_path)
%CREATE_WINDOWS_JUNCTION Create one junction without a visible shell window.

if ~ispc
    error('ttbi:WindowsJunctionPlatform', ...
        'Windows junction creation is available only on Windows.');
end
unsafe_pattern = '["&|<>^%!\r\n]';
if ~isempty(regexp(alias_path, unsafe_pattern, 'once')) || ...
        ~isempty(regexp(target_path, unsafe_pattern, 'once'))
    error('ttbi:WindowsJunctionPath', ...
        'Junction smoke paths contain unsupported command characters.');
end

command = sprintf('mklink /J "%s" "%s"', alias_path, target_path);
arguments = javaObject('java.util.ArrayList');
arguments.add(fullfile(getenv('SystemRoot'), 'System32', 'cmd.exe'));
arguments.add('/d');
arguments.add('/s');
arguments.add('/c');
arguments.add(command);
builder = javaObject('java.lang.ProcessBuilder', arguments);
builder.redirectErrorStream(true);
process = builder.start();
reader = javaObject('java.io.BufferedReader', javaObject( ...
    'java.io.InputStreamReader', process.getInputStream(), 'UTF-8'));
reader_cleanup = onCleanup(@() reader.close());
output_lines = cell(0, 1);
while true
    line = reader.readLine();
    if isempty(line)
        break
    end
    output_lines{end + 1, 1} = char(line); %#ok<AGROW>
end
exit_code = process.waitFor();
clear reader_cleanup
if exit_code ~= 0
    error('ttbi:WindowsJunctionCreate', ...
        'Could not create directory junction: %s', ...
        strjoin(output_lines, newline));
end
end
