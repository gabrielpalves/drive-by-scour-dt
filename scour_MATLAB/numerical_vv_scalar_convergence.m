function result = numerical_vv_scalar_convergence(h, q, q_floor, varargin)
%NUMERICAL_VV_SCALAR_CONVERGENCE Fail-closed scalar refinement diagnostics.
%
% RESULT = NUMERICAL_VV_SCALAR_CONVERGENCE(H,Q,Q_FLOOR) consumes levels in
% coarse-to-fine order.  It always reports consecutive relative differences
% and a finest-pair bound.  It reports Richardson/GCI only for a monotone,
% mode/event-consistent, equal-ratio three-level tail with finite positive
% observed order.  It never decides a scientific acceptance threshold.

protocol = numerical_vv_protocol_definition();
source_locked_max_order = double(protocol.max_gci_observed_order);
parser = inputParser;
addParameter(parser, 'ModeConsistent', true, @local_logical_scalar);
addParameter(parser, 'EventConsistent', true, @local_logical_scalar);
addParameter(parser, 'MaxObservedOrder', source_locked_max_order, ...
    @local_positive_scalar);
parse(parser, varargin{:});

% The protocol ceiling is a source-locked fail-closed policy, not a caller
% tuning parameter.  Retain MaxObservedOrder only so callers may choose a
% stricter ceiling; attempts to raise the registered value are errors.
if parser.Results.MaxObservedOrder > source_locked_max_order
    error('numerical_vv:ObservedOrderCeilingEscalation', ...
        ['MaxObservedOrder may tighten but may not raise the source-locked ' ...
         'ceiling of %.17g.'], source_locked_max_order);
end

h = double(h(:)');
q = double(q(:)');
if numel(h) ~= numel(q) || numel(h) < 2
    error('numerical_vv:BadScalarSequence', ...
        'H and Q must have equal length with at least two levels.');
end
if any(~isfinite(h)) || any(h <= 0) || any(diff(h) >= 0)
    error('numerical_vv:BadMeshSpacing', ...
        'H must be finite, positive, and strictly coarse-to-fine decreasing.');
end
if any(~isfinite(q)) || ~isscalar(q_floor) || ~isfinite(q_floor) || q_floor <= 0
    error('numerical_vv:BadScalarQoi', ...
        'Q must be finite and Q_FLOOR must be one finite positive scalar.');
end

result = struct();
result.schema = 'numerical-vv-scalar-convergence-v1';
result.h = h;
result.q = q;
result.q_floor = double(q_floor);
result.consecutive_relative_change = abs(diff(q)) ./ ...
    max(abs(q(2:end)), q_floor);
result.finest_pair_bound = result.consecutive_relative_change(end);
result.mode_consistent = logical(parser.Results.ModeConsistent);
result.event_consistent = logical(parser.Results.EventConsistent);
result.max_observed_order = min(double(parser.Results.MaxObservedOrder), ...
    source_locked_max_order);
result.gci_available = false;
result.observed_order = NaN;
result.richardson_extrapolate = NaN;
result.gci_fine = NaN;
result.reason = 'at-least-three-levels-required';
result.status = 'BOUND_ONLY_UNVERIFIED';

if numel(h) < 3
    return
end

ht = h(end-2:end);
qt = q(end-2:end);
r12 = ht(1) / ht(2);
r23 = ht(2) / ht(3);
ratio_tol = 64 * eps(max([r12, r23, 1]));
if abs(r12-r23) > ratio_tol
    result.reason = 'unequal-refinement-ratios-require-generalized-estimator';
    return
end
if ~result.mode_consistent
    result.reason = 'mode-matching-changed';
    return
end
if ~result.event_consistent
    result.reason = 'event-or-classification-changed';
    return
end
d_coarse = qt(1) - qt(2);
d_fine = qt(2) - qt(3);
scale = max([abs(qt), q_floor]);
zero_tol = 64 * eps(scale);
if abs(d_coarse) <= zero_tol || abs(d_fine) <= zero_tol
    result.reason = 'difference-numerically-zero';
    return
end
if sign(d_coarse) ~= sign(d_fine)
    result.reason = 'nonmonotone-sequence';
    return
end

p = log(abs(d_coarse/d_fine)) / log(r23);
if ~isfinite(p) || p <= 0 || p > result.max_observed_order
    result.reason = 'nonfinite-nonpositive-or-unreasonable-observed-order';
    return
end
denominator = r23^p - 1;
if ~isfinite(denominator) || denominator <= 0 || abs(qt(3)) <= q_floor
    result.reason = 'invalid-gci-denominator-or-near-zero-fine-qoi';
    return
end

result.gci_available = true;
result.observed_order = p;
result.richardson_extrapolate = qt(3) + (qt(3)-qt(2))/denominator;
result.gci_fine = 1.25 * abs(qt(2)-qt(3)) / ...
    (abs(qt(3))*denominator);
result.reason = 'monotone-equal-ratio-tail';
result.status = 'GCI_AVAILABLE_NO_ACCEPTANCE_DECISION';
end

function tf = local_logical_scalar(value)
tf = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
    isfinite(double(value)) && any(double(value) == [0, 1]);
end

function tf = local_positive_scalar(value)
tf = isnumeric(value) && isreal(value) && isscalar(value) && ...
    isfinite(value) && value > 0;
end
