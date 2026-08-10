function metrics = numerical_vv_waveform_metrics( ...
        x_reference, y_reference, x_test, y_test, x_common, varargin)
%NUMERICAL_VV_WAVEFORM_METRICS Compare histories on an explicit common grid.
%
% Arrays are channel-by-sample.  X_COMMON is mandatory: this function never
% extrapolates or silently selects a comparison window.  The caller therefore
% owns the protocol's fixed 0.01 m grid and window.

parser = inputParser;
addParameter(parser, 'ChannelNames', {}, @iscell);
addParameter(parser, 'RmsFloor', 1e-15, @local_positive_scalar);
addParameter(parser, 'PeakFloor', 1e-15, @local_positive_scalar);
parse(parser, varargin{:});

x_reference = local_grid(x_reference, 'x_reference');
x_test = local_grid(x_test, 'x_test');
x_common = local_grid(x_common, 'x_common');
y_reference = local_signals(y_reference, numel(x_reference), 'y_reference');
y_test = local_signals(y_test, numel(x_test), 'y_test');
if size(y_reference, 1) ~= size(y_test, 1)
    error('numerical_vv:ChannelCountMismatch', ...
        'Reference and test histories must have the same channel count.');
end
if x_common(1) < x_reference(1) || x_common(end) > x_reference(end) || ...
        x_common(1) < x_test(1) || x_common(end) > x_test(end)
    error('numerical_vv:CommonGridOutsideDomain', ...
        'X_COMMON must lie inside both signal domains; extrapolation is forbidden.');
end

n_channels = size(y_reference, 1);
names = parser.Results.ChannelNames;
if isempty(names)
    names = arrayfun(@(k) sprintf('channel_%d', k), 1:n_channels, ...
        'UniformOutput', false);
elseif numel(names) ~= n_channels || ...
        any(~cellfun(@(v) ischar(v) || (isstring(v) && isscalar(v)), names))
    error('numerical_vv:BadChannelNames', ...
        'ChannelNames must contain one text scalar per channel.');
end

reference_common = zeros(n_channels, numel(x_common));
test_common = zeros(n_channels, numel(x_common));
for k = 1:n_channels
    reference_common(k, :) = interp1( ...
        x_reference, y_reference(k, :), x_common, 'linear');
    test_common(k, :) = interp1(x_test, y_test(k, :), x_common, 'linear');
end
if any(~isfinite(reference_common(:))) || any(~isfinite(test_common(:)))
    error('numerical_vv:NonfiniteInterpolation', ...
        'Common-grid interpolation produced a non-finite value.');
end

nrmse = zeros(n_channels, 1);
nmax = zeros(n_channels, 1);
correlation = zeros(n_channels, 1);
reference_rms = zeros(n_channels, 1);
test_rms = zeros(n_channels, 1);
reference_abs_peak = zeros(n_channels, 1);
test_abs_peak = zeros(n_channels, 1);
peak_amplitude_error = zeros(n_channels, 1);
peak_position_error_m = zeros(n_channels, 1);
for k = 1:n_channels
    yr = reference_common(k, :);
    yt = test_common(k, :);
    delta = yt - yr;
    reference_rms(k) = sqrt(mean(yr.^2));
    test_rms(k) = sqrt(mean(yt.^2));
    [reference_abs_peak(k), ir] = max(abs(yr));
    [test_abs_peak(k), it] = max(abs(yt));
    nrmse(k) = sqrt(mean(delta.^2)) / ...
        max(reference_rms(k), parser.Results.RmsFloor);
    nmax(k) = max(abs(delta)) / ...
        max(reference_abs_peak(k), parser.Results.PeakFloor);
    peak_amplitude_error(k) = test_abs_peak(k) - reference_abs_peak(k);
    peak_position_error_m(k) = x_common(it) - x_common(ir);
    if std(yr) == 0 || std(yt) == 0
        correlation(k) = double(all(yr == yt));
    else
        C = corrcoef(yr, yt);
        correlation(k) = C(1, 2);
    end
end

metrics = table(string(names(:)), nrmse, nmax, correlation, ...
    reference_rms, test_rms, reference_abs_peak, test_abs_peak, ...
    peak_amplitude_error, peak_position_error_m, ...
    'VariableNames', {'channel_id', 'nrmse', 'normalized_max_error', ...
    'correlation', 'reference_rms', 'test_rms', 'reference_abs_peak', ...
    'test_abs_peak', 'peak_amplitude_error', 'peak_position_error_m'});
end

function value = local_grid(value, name)
value = double(value(:)');
if numel(value) < 2 || any(~isfinite(value)) || any(diff(value) <= 0)
    error('numerical_vv:BadGrid', ...
        '%s must be finite and strictly increasing.', name);
end
end

function value = local_signals(value, n_samples, name)
value = double(value);
if isvector(value)
    value = value(:)';
end
if size(value, 2) ~= n_samples || any(~isfinite(value(:)))
    error('numerical_vv:BadWaveform', ...
        '%s must be finite channel-by-sample data.', name);
end
end

function tf = local_positive_scalar(value)
tf = isnumeric(value) && isreal(value) && isscalar(value) && ...
    isfinite(value) && value > 0;
end
