function tf = contact_nonnegative_vector(value)
%CONTACT_NONNEGATIVE_VECTOR Validate a finite nonnegative vector.

tf = isnumeric(value) && isvector(value) && ~isempty(value) && ...
    all(isfinite(value)) && all(value >= 0);
end
