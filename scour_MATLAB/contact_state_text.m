function text_value = contact_state_text(data, field, fallback)
%CONTACT_STATE_TEXT Read one state field as scalar text with a fallback.

if isfield(data, field)
    value = data.(field);
    if iscell(value)
        value = value{1};
    end
    text_value = char(string(value));
else
    text_value = fallback;
end
end
