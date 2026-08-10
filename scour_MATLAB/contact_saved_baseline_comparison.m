function [comparison, note, mode, all_channels_pass] = ...
        contact_saved_baseline_comparison( ...
        data, passage_index, x_common, rerun_signal_common, channel_names, ...
        rerun_signal_raw, rerun_x_raw, reconstruction_rtol, ...
        reconstruction_atol)
%CONTACT_SAVED_BASELINE_COMPARISON Compare a rerun to persisted raw channels.

comparison = table();
note = 'Saved raw channels unavailable.';
mode = 'unavailable';
all_channels_pass = false;
needed = {'AcelPrimVag', 'AcelWheelsetPrimVag', 'PitchPrimVag', ...
    'DimSpace', 'channel_schema_id'};
if ~all(isfield(data, needed))
    return
end
if ~strcmp(char(string(data.channel_schema_id)), 'physical8_v1')
    note = 'Saved payload is not physical8_v1.';
    return
end
saved_wheelset = double(data.AcelWheelsetPrimVag{1, passage_index});
if size(saved_wheelset, 1) < 2
    note = 'Saved wheelset proxy payload lacks the two deployed rows.';
    return
end
saved_signal = double([data.AcelPrimVag{1, passage_index}; ...
    saved_wheelset(1:2, :); ...
    data.PitchPrimVag{1, passage_index}]);
if size(saved_signal, 1) ~= numel(channel_names)
    note = 'Saved raw channel count does not match the closure channel map.';
    return
end

% Prefer an exact raw-grid comparison. Fall back to the historical D01
% spatial map only when the saved and rerun sample counts differ.
if isequal(size(saved_signal), size(rerun_signal_raw))
    saved_compare = saved_signal;
    rerun_compare = rerun_signal_raw;
    comparison_mode = 'direct_raw_samples';
else
    saved_distance_m = double(data.DimSpace(passage_index)) / 100;
    x_saved = linspace(0, saved_distance_m, size(saved_signal, 2));
    if x_common(end) > x_saved(end) || x_common(end) > rerun_x_raw(end)
        note = sprintf(['Saved/current signal ends before comparison window ' ...
            'ends at %.3f m.'], x_common(end));
        return
    end
    saved_compare = interp1( ...
        x_saved, saved_signal', x_common, 'linear')';
    rerun_compare = rerun_signal_common;
    comparison_mode = 'common_spatial_grid_fallback';
end
delta = rerun_compare - saved_compare;
n_channels = size(saved_compare, 1);
nrmse = zeros(n_channels, 1);
nmax = zeros(n_channels, 1);
correlation = zeros(n_channels, 1);
max_absolute = zeros(n_channels, 1);
max_relative = zeros(n_channels, 1);
max_tolerance_ratio = zeros(n_channels, 1);
within_tolerance = false(n_channels, 1);
for ch = 1:n_channels
    nrmse(ch) = sqrt(mean(delta(ch, :).^2)) / ...
        max(sqrt(mean(saved_compare(ch, :).^2)), eps);
    nmax(ch) = max(abs(delta(ch, :))) / ...
        max(max(abs(saved_compare(ch, :))), eps);
    if std(rerun_compare(ch, :)) == 0 || std(saved_compare(ch, :)) == 0
        correlation(ch) = double(all(delta(ch, :) == 0));
    else
        c_ = corrcoef(rerun_compare(ch, :), saved_compare(ch, :));
        correlation(ch) = c_(1, 2);
    end
    absolute_delta = abs(delta(ch, :));
    scale = max(abs(saved_compare(ch, :)), abs(rerun_compare(ch, :)));
    allowed = reconstruction_atol + reconstruction_rtol * scale;
    max_absolute(ch) = max(absolute_delta);
    max_tolerance_ratio(ch) = max(absolute_delta ./ allowed);
    nonzero = scale > 0;
    if any(nonzero)
        max_relative(ch) = max(absolute_delta(nonzero) ./ scale(nonzero));
    else
        max_relative(ch) = 0;
    end
    within_tolerance(ch) = all(absolute_delta <= allowed);
end
comparison = table(channel_names(:), nrmse, nmax, correlation, ...
    max_absolute, max_relative, max_tolerance_ratio, within_tolerance, ...
    'VariableNames', {'channel', 'nrmse_rerun_vs_saved', ...
    'nmax_rerun_vs_saved', 'correlation_rerun_vs_saved', ...
    'max_abs_rerun_vs_saved', 'max_rel_rerun_vs_saved', ...
    'max_tolerance_ratio', 'within_tolerance'});
note = sprintf(['Comparison mode: %s. This is a diagnostic reproduction ' ...
    'check. Paper-1 qualification accepts only direct raw comparison and ' ...
    'requires every value within rtol=%.17g, atol=%.17g.'], ...
    comparison_mode, reconstruction_rtol, reconstruction_atol);
mode = comparison_mode;
all_channels_pass = all(within_tolerance);
end
