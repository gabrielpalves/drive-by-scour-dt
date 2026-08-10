function tf = contact_logical_scalar(value)
%CONTACT_LOGICAL_SCALAR Validate a scalar logical-or-binary value.

tf = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
    isfinite(value) && any(value == [0, 1]);
end
