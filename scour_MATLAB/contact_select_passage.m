function passage_index = ...
        contact_select_passage(selector, contact_log, n_passages)
%CONTACT_SELECT_PASSAGE Resolve a numeric or interactive "worst" selector.

if isnumeric(selector) && isscalar(selector) && isfinite(selector) && ...
        selector == round(selector)
    passage_index = double(selector);
elseif ischar(selector) || (isstring(selector) && isscalar(selector))
    selector = lower(strtrim(char(selector)));
    if strcmp(selector, 'worst')
        [~, passage_index] = max(double(contact_log(:, 4)));
    else
        parsed = str2double(selector);
        if ~isfinite(parsed) || parsed ~= round(parsed)
            error('contact_closure:BadPassage', ...
                'Passage selector must be a positive integer or "worst".');
        end
        passage_index = parsed;
    end
else
    error('contact_closure:BadPassage', ...
        'Passage selector must be a positive integer or "worst".');
end
if passage_index < 1 || passage_index > n_passages
    error('contact_closure:BadPassage', ...
        'Passage %d is outside 1..%d.', passage_index, n_passages);
end
end
