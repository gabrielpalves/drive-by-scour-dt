function tf = contact_text_scalar(value)
%CONTACT_TEXT_SCALAR Validate one MATLAB text scalar.

tf = ischar(value) || (isstring(value) && isscalar(value));
end
