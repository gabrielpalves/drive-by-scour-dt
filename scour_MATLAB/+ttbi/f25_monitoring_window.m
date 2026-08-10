function window = f25_monitoring_window(config)
%F25_MONITORING_WINDOW Reconstruct the source's 58.30 m signal convention.
%
% The live F25 deck is 39.9 m.  The corrected main-campaign crop therefore
% uses 3,990 bridge samples and would retain 5,821 points.  Fernandes reports
% a 58.30 m monitoring length and 5,830 acceleration values; the accepted F25
% protocol starts from this repository's 5,831-point inclusive space grid and
% trims its tail to 5,830 before PAA.  Reproducing that convention requires
% the source-era round-before-scale bridge term, round(39.9)*100 = 4,000.
%
% This helper operates on the full RAW passage saved by D01.  It does not
% change D01 or the four-block campaign crop.

if ~isstruct(config) || ~isscalar(config) || ...
        ~isfield(config, 'schema') || ...
        ~strcmp(config.schema, 'f25-experiment-config-v1')
    error('ttbi:f25_monitoring_window:Config', ...
        'config must be one f25-experiment-config-v1 scalar struct.');
end

samples_per_m = 100;
crop_start = 1001;
post_deck_samples = 1831;
physical_bridge_samples = round(samples_per_m*config.L_bridge);
source_convention_bridge_samples = ...
    round(config.L_bridge)*samples_per_m;
crop_end_untrimmed = crop_start - 1 + ...
    source_convention_bridge_samples + post_deck_samples;
untrimmed_sample_count = crop_end_untrimmed - crop_start + 1;
crop_end_trimmed = crop_end_untrimmed - 1;
trimmed_sample_count = crop_end_trimmed - crop_start + 1;

if physical_bridge_samples ~= 3990 || ...
        source_convention_bridge_samples ~= 4000 || ...
        untrimmed_sample_count ~= 5831 || trimmed_sample_count ~= 5830
    error('ttbi:f25_monitoring_window:Arithmetic', ...
        'The frozen F25 5,831-to-5,830 reconstruction arithmetic drifted.');
end

window = struct();
window.schema = 'f25-monitoring-window-v1';
window.source = 'full_raw_passage_reconstruction';
window.samples_per_m = samples_per_m;
window.crop_start = crop_start;
window.crop_end_untrimmed = crop_end_untrimmed;
window.crop_end_trimmed = crop_end_trimmed;
window.post_deck_samples = post_deck_samples;
window.physical_bridge_samples = physical_bridge_samples;
window.source_convention_bridge_samples = ...
    source_convention_bridge_samples;
window.nominal_length_m = 58.30;
window.untrimmed_sample_count = untrimmed_sample_count;
window.trimmed_sample_count = trimmed_sample_count;
window.tail_samples_trimmed = 1;
window.extra_beyond_physical_bridge_samples = ...
    source_convention_bridge_samples-physical_bridge_samples;
end
