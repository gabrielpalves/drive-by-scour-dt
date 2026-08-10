function output = ...
        contact_replace_unique_bytes(input, needle, replacement, label)
%CONTACT_REPLACE_UNIQUE_BYTES Replace exactly one byte-token occurrence.

matches = strfind(input, needle);
if numel(matches) ~= 1
    error('contact_closure_gate:QualificationExecutable', ...
        'Expected exactly one %s; found %d.', label, numel(matches));
end
first = matches(1);
output = [input(1:first - 1), replacement, ...
    input(first + numel(needle):end)];
end
