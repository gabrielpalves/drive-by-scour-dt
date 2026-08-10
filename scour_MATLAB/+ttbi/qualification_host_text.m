function value = qualification_host_text(value, label)
%QUALIFICATION_HOST_TEXT Validated scalar text field for a receipt.
    value = strtrim(char(value));
    if isempty(value) || numel(value) > 1024 || ...
            contains(value, newline) || contains(value, char(13)) || ...
            contains(value, char(0))
        error('A00:QualificationHostDiagnostic', ...
            'Qualification host diagnostic %s is empty or noncanonical.', label);
    end
end
