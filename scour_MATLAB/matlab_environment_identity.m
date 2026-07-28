function [sha256, descriptor] = matlab_environment_identity(environment)
%MATLAB_ENVIRONMENT_IDENTITY Canonical identity of a MATLAB numerical stack.
%   [SHA256, DESCRIPTOR] = MATLAB_ENVIRONMENT_IDENTITY(ENVIRONMENT) requires
%   exactly these eight nonempty character-row fields:
%
%     arch, blas, lapack, matlab_product_version,
%     parallel_toolbox_version, release,
%     statistics_toolbox_version, version
%
%   DESCRIPTOR is the lexicographically field-sorted sequence
%       field=value
%   joined by LF with NO terminal LF. Values containing CR/LF/NUL are rejected.
%   SHA256 is the lowercase SHA-256 of DESCRIPTOR encoded as UTF-8. This format
%   is shared verbatim with core.environment.matlab_environment_descriptor.

    required = sort({ ...
        'release', 'version', 'arch', 'blas', 'lapack', ...
        'matlab_product_version', 'statistics_toolbox_version', ...
        'parallel_toolbox_version'});

    if ~isstruct(environment) || ~isscalar(environment)
        error('matlab_environment_identity:Type', ...
            'Environment must be a scalar struct.');
    end
    actual = sort(fieldnames(environment))';
    if ~isequal(actual, required)
        missing = setdiff(required, actual);
        extra = setdiff(actual, required);
        error('matlab_environment_identity:Fields', ...
            'Environment field set differs (missing={%s}, extra={%s}).', ...
            strjoin(missing, ', '), strjoin(extra, ', '));
    end

    lines = cell(numel(required), 1);
    for k = 1:numel(required)
        name = required{k};
        value = environment.(name);
        if ~ischar(value) || ~isrow(value) || isempty(value)
            error('matlab_environment_identity:FieldType', ...
                'Environment field "%s" must be a nonempty character row.', name);
        end
        if any(value == char(13)) || any(value == newline) || ...
                any(value == char(0))
            error('matlab_environment_identity:UnsafeValue', ...
                'Environment field "%s" contains a forbidden control byte.', name);
        end
        lines{k} = sprintf('%s=%s', name, value);
    end
    descriptor = strjoin(lines, newline);
    bytes = unicode2native(descriptor, 'UTF-8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    raw = md.digest(bytes);
    sha256 = lower(sprintf('%02x', typecast(int8(raw), 'uint8')));
end
