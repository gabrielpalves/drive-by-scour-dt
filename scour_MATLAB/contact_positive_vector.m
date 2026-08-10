function tf = contact_positive_vector(value)
%CONTACT_POSITIVE_VECTOR Validate a finite, nonempty, positive vector.

tf = isnumeric(value) && isvector(value) && ~isempty(value) && ...
    all(isfinite(value)) && all(value > 0);
end
