function value = contact_java_boolean_value(raw)
%CONTACT_JAVA_BOOLEAN_VALUE Normalize Java/MATLAB Boolean metadata.
%
% Attribute-map Booleans are Java objects in some MATLAB updates and native
% logicals in others. The contact evidence chain accepts either exact scalar
% representation and rejects ambiguous values.

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
    error('contact_path:JavaBooleanValue', ...
        'Java metadata is not a Boolean scalar: %s', ...
        conversion_error.message);
end
if ~(islogical(converted) && isscalar(converted))
    error('contact_path:JavaBooleanValue', ...
        'Java metadata did not convert to a logical scalar.');
end
value = converted;
end

