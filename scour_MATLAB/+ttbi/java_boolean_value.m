function value = java_boolean_value(raw)
%JAVA_BOOLEAN_VALUE Normalize a Java Boolean or MATLAB logical scalar.
%
% MATLAB releases differ in whether values returned from a Java attribute map
% remain java.lang.Boolean objects or are converted automatically to logical.
% This small boundary helper accepts both representations and nothing else.

if islogical(raw) && isscalar(raw)
    value = raw;
    return
end
if isnumeric(raw) && isscalar(raw) && isfinite(raw) && ...
        (raw == 0 || raw == 1)
    value = logical(raw);
    return
end
try
    converted = javaMethod('booleanValue', raw);
catch conversion_error
    error('ttbi:JavaBooleanValue', ...
        'Java metadata is not a Boolean scalar: %s', ...
        conversion_error.message);
end
if ~(islogical(converted) && isscalar(converted))
    error('ttbi:JavaBooleanValue', ...
        'Java metadata did not convert to a logical scalar.');
end
value = converted;
end

