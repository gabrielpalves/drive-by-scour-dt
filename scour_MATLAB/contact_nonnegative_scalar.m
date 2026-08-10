function tf = contact_nonnegative_scalar(value)
%CONTACT_NONNEGATIVE_SCALAR Validate one finite nonnegative numeric scalar.

tf = isnumeric(value) && isscalar(value) && isfinite(value) && value >= 0;
end
