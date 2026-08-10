function identity = contact_windows_file_identity(path)
%CONTACT_WINDOWS_FILE_IDENTITY Query a volume-bound ID without a shell.
%
% ProcessBuilder passes the path as one literal argument to the system fsutil
% executable. The parser accepts one exact 64- or 128-bit hexadecimal file ID,
% normalizes it to 128 bits, and binds it to the Windows volume serial number.

if ~(ischar(path) && isrow(path) && ~isempty(path) && ...
        (isfile(path) || isfolder(path)))
    error('contact_path:WindowsFileIdentityPath', ...
        'Windows file identity requires one existing character-row path.');
end

try
    system_directory = char(System.Environment.SystemDirectory);
catch system_error
    error('contact_path:WindowsFileIdentityRoot', ...
        'Cannot obtain the native Windows system directory: %s', ...
        system_error.message);
end
executable = fullfile(system_directory, 'fsutil.exe');
if ~isfile(executable)
    error('contact_path:WindowsFileIdentityTool', ...
        'Windows file-ID query tool is missing: %s', executable);
end

file = javaObject('java.io.File', path);
absolute_nio = file.toPath().toAbsolutePath().normalize();
try
    store = javaMethod( ...
        'getFileStore', 'java.nio.file.Files', absolute_nio);
    volume_before_value = javaMethod( ...
        'getAttribute', store, 'volume:vsn');
    volume_before_value = double(volume_before_value);
    if ~isscalar(volume_before_value) || ~isfinite(volume_before_value) || ...
            volume_before_value ~= fix(volume_before_value) || ...
            volume_before_value < -2147483648 || ...
            volume_before_value > 4294967295
        error('contact_path:WindowsFileIdentityVolumeEmpty', ...
            'Windows returned an invalid volume serial number.');
    end
    volume_before = sprintf('%.0f', volume_before_value);
catch volume_error
    error('contact_path:WindowsFileIdentityVolume', ...
        'Cannot obtain the Windows volume identity for %s: %s', ...
        path, volume_error.message);
end

[lines, exit_code] = contact_run_small_process( ...
    {executable, 'file', 'queryfileid', char(absolute_nio.toString())}, 10);

output = strjoin(lines, newline);
matches = regexp(output, ...
    '(?i)(?<![0-9a-f])0x(?:[0-9a-f]{32}|[0-9a-f]{16})(?![0-9a-f])', ...
    'match');
if exit_code ~= 0 || numel(matches) ~= 1
    error('contact_path:WindowsFileIdentityQuery', ...
        'fsutil did not return exactly one file ID for %s.', path);
end
file_id = lower(matches{1}(3:end));
if all(file_id == '0') || all(file_id == 'f')
    error('contact_path:WindowsFileIdentitySentinel', ...
        'Windows returned an invalid all-zero/all-ones file ID for %s.', ...
        path);
end
if numel(file_id) == 16
    file_id = [repmat('0', 1, 16) file_id];
end

try
    store_after = javaMethod( ...
        'getFileStore', 'java.nio.file.Files', absolute_nio);
    volume_after_value = javaMethod( ...
        'getAttribute', store_after, 'volume:vsn');
    volume_after_value = double(volume_after_value);
    if ~isscalar(volume_after_value) || ~isfinite(volume_after_value) || ...
            volume_after_value ~= fix(volume_after_value) || ...
            volume_after_value < -2147483648 || ...
            volume_after_value > 4294967295
        error('contact_path:WindowsFileIdentityVolumeEmpty', ...
            'Windows returned an invalid volume serial number.');
    end
    volume_after = sprintf('%.0f', volume_after_value);
catch volume_error
    error('contact_path:WindowsFileIdentityVolume', ...
        'Cannot confirm the Windows volume identity for %s: %s', ...
        path, volume_error.message);
end
if ~strcmp(volume_before, volume_after)
    error('contact_path:WindowsFileIdentityVolumeChanged', ...
        'Windows volume identity changed during the query for %s.', path);
end
identity = ['volume-vsn=' volume_before '|file-id=' file_id];
end
