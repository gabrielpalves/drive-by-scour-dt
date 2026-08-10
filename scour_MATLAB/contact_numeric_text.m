function text = contact_numeric_text(values)
%CONTACT_NUMERIC_TEXT Canonical comma-separated numeric vector.

parts = arrayfun(@(v) sprintf('%.17g', v), double(values(:)'), ...
    'UniformOutput', false);
text = strjoin(parts, ',');
end
