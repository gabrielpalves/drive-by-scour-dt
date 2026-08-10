function [bytes, observation] = contact_stable_file_bytes(path)
%CONTACT_STABLE_FILE_BYTES Read bytes only while path identity stays fixed.

before = contact_file_observation(path);
bytes = contact_file_bytes(path);
after = contact_file_observation(path);
if ~isequal(before, after) || numel(bytes) ~= before.byte_count
    error('contact_snapshot:FileRace', ...
        'File identity or metadata changed during byte capture: %s', path);
end
observation = before;
end
