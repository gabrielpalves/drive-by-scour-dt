function text_value = contact_case_text(case_info, field, fallback)
%CONTACT_CASE_TEXT Read one case-info field as text with a fallback.

if isfield(case_info, field)
    text_value = char(string(case_info.(field)));
else
    text_value = fallback;
end
end
