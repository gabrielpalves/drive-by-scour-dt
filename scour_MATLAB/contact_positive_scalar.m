function tf = contact_positive_scalar(value)
%CONTACT_POSITIVE_SCALAR Validate one finite positive numeric scalar.

tf = isnumeric(value) && isscalar(value) && isfinite(value) && value > 0;
end
