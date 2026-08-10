function value = contact_closure_host_text(value, label)
%CONTACT_CLOSURE_HOST_TEXT Validate one bounded canonical host field.

value = strtrim(char(value));
if isempty(value) || numel(value) > 1024 || ...
        contains(value, newline) || contains(value, char(13)) || ...
        contains(value, char(0))
    error('contact_closure_gate:HostAttestation', ...
        'Closure host diagnostic %s is empty or noncanonical.', label);
end
end
