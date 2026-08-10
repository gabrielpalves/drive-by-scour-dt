function [pass, result] = contact_gci_bound( ...
        phi, actual_step_s, limit, safety_factor, ...
        equivalence_rtol, equivalence_atol, p_min, p_max)
%CONTACT_GCI_BOUND Generalized three-grid Richardson/GCI upper bound.
%
% The step vector contains actual solver steps in coarse/medium/fine order.
% B11 uses ceil before linspace, so nominal refinement ratios are invalid.

phi = double(phi(:));
actual_step_s = double(actual_step_s(:));
result = struct('status', 'INVALID', 'observed_order', NaN, ...
    'extrapolated', NaN, 'fine_uncertainty', NaN, 'upper_bound', NaN, ...
    'actual_step_s', actual_step_s(:)', ...
    'coarse_medium_ratio', NaN, 'medium_fine_ratio', NaN);
pass = false;
if numel(phi) ~= 3 || numel(actual_step_s) ~= 3 || ...
        any(~isfinite(phi)) || any(~isfinite(actual_step_s)) || ...
        any(actual_step_s <= 0) || any(diff(actual_step_s) >= 0)
    return
end
result.coarse_medium_ratio = actual_step_s(1) / actual_step_s(2);
result.medium_fine_ratio = actual_step_s(2) / actual_step_s(3);
scale = max([1; abs(phi)]);
tol = equivalence_atol + equivalence_rtol * scale;
e32 = phi(1) - phi(2);
e21 = phi(2) - phi(3);
if abs(e32) <= tol && abs(e21) <= tol
    result.status = 'EQUIVALENT';
    result.observed_order = Inf;
    result.extrapolated = phi(3);
    result.fine_uncertainty = 0;
    result.upper_bound = max(0, phi(3));
    pass = result.upper_bound <= limit;
    return
end
if abs(e21) <= tol && abs(e32) > tol
    result.status = 'FINE_PAIR_EQUIVALENT';
    result.observed_order = Inf;
    result.extrapolated = phi(3);
    result.fine_uncertainty = 0;
    result.upper_bound = max(0, phi(3));
    pass = result.upper_bound <= limit;
    return
end
if abs(e32) <= tol || e32 * e21 <= 0
    result.status = 'OSCILLATORY_OR_STALLED';
    return
end
observed_ratio = abs(e32 / e21);
r31 = actual_step_s(1) / actual_step_s(3);
r21 = actual_step_s(2) / actual_step_s(3);
model_ratio = @(order) ...
    (r31.^order - r21.^order) ./ (r21.^order - 1);
p_lo = p_min;
p_hi = p_max;
f_lo = model_ratio(p_lo) - observed_ratio;
f_hi = model_ratio(p_hi) - observed_ratio;
if ~isfinite(f_lo) || ~isfinite(f_hi) || f_lo > 0 || f_hi < 0
    result.status = 'NO_POSITIVE_ORDER';
    return
end
try
    p = fzero(@(order) model_ratio(order) - observed_ratio, ...
        [p_lo, p_hi]);
catch
    result.status = 'ORDER_SOLVE_FAILED';
    return
end
if ~isfinite(p) || p <= 0 || p > p_hi
    result.status = 'NO_POSITIVE_ORDER';
    return
end
denominator = r21^p - 1;
extrapolated = (r21^p * phi(3) - phi(2)) / denominator;
uncertainty = safety_factor * abs(phi(3) - phi(2)) / denominator;
upper = max(0, extrapolated) + uncertainty;
result.status = 'MONOTONIC';
result.observed_order = p;
result.extrapolated = extrapolated;
result.fine_uncertainty = uncertainty;
result.upper_bound = upper;
pass = upper <= limit;
end
