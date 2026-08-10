function elements_per_bay = bridge_mesh_elements_per_sleeper( ...
    bridge_length_m, num_spans, sleeper_spacing_m, minimum_elements_per_bay)
%BRIDGE_MESH_ELEMENTS_PER_SLEEPER Choose a support-aligned bridge mesh.
%
% The TTBI coupling requires bridge nodes at every sleeper.  Equal-span
% bridge supports must also be nodes; otherwise B02 would move a support to
% the nearest node and silently change the physical geometry.  Starting at
% the requested minimum density, this function returns the first integer
% number of elements per sleeper bay that satisfies both constraints.
%
% Example: a 60 m, three-span bridge cannot use 0.30 m elements because
% 20/0.30 is not an integer.  Three elements per 0.60 m bay gives h=0.20 m
% and places the supports exactly at 0, 20, 40, and 60 m.  A 99.6 m,
% four-span bridge is already aligned with two elements per bay (h=0.30 m).

if nargin < 4
    minimum_elements_per_bay = 2;
end
local_positive_scalar(bridge_length_m, 'bridge_length_m');
local_positive_scalar(sleeper_spacing_m, 'sleeper_spacing_m');
local_positive_integer(num_spans, 'num_spans');
local_positive_integer(minimum_elements_per_bay, ...
    'minimum_elements_per_bay');

maximum_elements_per_bay = 64;
span_length_m = bridge_length_m / num_spans;
for candidate = minimum_elements_per_bay:maximum_elements_per_bay
    element_length_m = sleeper_spacing_m / candidate;
    bridge_element_count = bridge_length_m / element_length_m;
    span_element_count = span_length_m / element_length_m;
    scale = max([abs(bridge_element_count), abs(span_element_count), 1]);
    tolerance = 256 * eps(scale);
    if abs(bridge_element_count-round(bridge_element_count)) <= tolerance && ...
            abs(span_element_count-round(span_element_count)) <= tolerance
        elements_per_bay = candidate;
        return
    end
end

error('bridge_mesh:NoSupportAlignedDensity', ...
    ['No support-aligned mesh was found between %d and %d elements per ' ...
     'sleeper bay for L=%.17g m, %d equal spans, and sleeper spacing ' ...
     '%.17g m.'], minimum_elements_per_bay, maximum_elements_per_bay, ...
    bridge_length_m, num_spans, sleeper_spacing_m);
end

function local_positive_scalar(value, name)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
        ~isfinite(value) || value <= 0
    error('bridge_mesh:InvalidPositiveScalar', ...
        '%s must be one finite positive real scalar.', name);
end
end

function local_positive_integer(value, name)
local_positive_scalar(value, name);
if value ~= fix(value)
    error('bridge_mesh:InvalidPositiveInteger', ...
        '%s must be a positive integer.', name);
end
end
