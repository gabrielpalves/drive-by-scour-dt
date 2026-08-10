function [lines, exit_code] = run_small_process(arguments, timeout_seconds)
%RUN_SMALL_PROCESS Run one bounded, low-output native command without a shell.
%
% The generator uses this only for fsutil metadata queries. Waiting with a
% finite timeout prevents an unavailable filesystem provider from hanging a
% campaign indefinitely. Output is read only after the process exits; an
% unexpectedly large output therefore reaches the timeout and fails closed.

if ~iscell(arguments) || isempty(arguments) || ...
        ~all(cellfun(@(item) ischar(item) && isrow(item) && ...
        ~isempty(item), arguments))
    error('ttbi:SmallProcessArguments', ...
        'Native-process arguments must be nonempty character rows.');
end
if ~(isnumeric(timeout_seconds) && isscalar(timeout_seconds) && ...
        isfinite(timeout_seconds) && timeout_seconds >= 1 && ...
        timeout_seconds <= 60 && timeout_seconds == fix(timeout_seconds))
    error('ttbi:SmallProcessTimeout', ...
        'Native-process timeout must be an integer from 1 to 60 seconds.');
end

native_arguments = javaObject('java.util.ArrayList');
for argument_index = 1:numel(arguments)
    native_arguments.add(arguments{argument_index});
end
builder = javaObject('java.lang.ProcessBuilder', native_arguments);
builder.redirectErrorStream(true);
process = builder.start();
process_cleanup = onCleanup( ...
    @() javaMethod('destroyForcibly', process));

seconds = javaMethod( ...
    'valueOf', 'java.util.concurrent.TimeUnit', 'SECONDS');
finished = javaMethod( ...
    'waitFor', process, int64(timeout_seconds), seconds);
if ~finished
    error('ttbi:SmallProcessTimedOut', ...
        'Native metadata query exceeded %d seconds.', timeout_seconds);
end
exit_code = double(javaMethod('exitValue', process));

reader = javaObject('java.io.BufferedReader', javaObject( ...
    'java.io.InputStreamReader', process.getInputStream(), 'UTF-8'));
reader_cleanup = onCleanup(@() reader.close());
lines = cell(0, 1);
while true
    line = reader.readLine();
    if isempty(line)
        break
    end
    lines{end + 1, 1} = char(line); %#ok<AGROW>
end
clear reader_cleanup process_cleanup
end

