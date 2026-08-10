function output = replace_unique_bytes(input, needle, replacement, label)
%REPLACE_UNIQUE_BYTES Replace exactly one occurrence in a byte vector.
    matches = strfind(input, needle);
    if numel(matches) ~= 1
        error('A00:QualificationCanonicalization', ...
            'Expected exactly one %s in executable bytes; found %d.', ...
            label, numel(matches));
    end
    first = matches(1);
    output = [input(1:first-1), replacement, ...
        input(first+numel(needle):end)];
end
