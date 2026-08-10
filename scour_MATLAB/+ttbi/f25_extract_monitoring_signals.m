function output = f25_extract_monitoring_signals(data, config)
%F25_EXTRACT_MONITORING_SIGNALS Build and tail-trim the F25 space window.
%
% D01 keeps the coupled solver's complete time-domain passage.  F25 alone
% maps that response to the established 100 samples/m space grid, extracts
% samples 1001..6831 inclusive, retains sample 6831 separately as an audit
% tail, and publishes samples 1001..6830 as the clean 5,830-point learning
% signal.  The ordinary Paper-1 crop metadata and loader are untouched.

if ~isstruct(config) || ~isscalar(config) || ...
        ~isfield(config, 'schema') || ...
        ~strcmp(config.schema, 'f25-experiment-config-v1')
    error('ttbi:f25_extract_monitoring_signals:Config', ...
        'config must be one f25-experiment-config-v1 scalar struct.');
end
required = {'AceleracaoPrimVag','AcelWheelsetPrimVag','PitchPrimVag', ...
    'DimAcel','DimSpace'};
missing = required(~isfield(data, required));
if ~isempty(missing)
    error('ttbi:f25_extract_monitoring_signals:Fields', ...
        'D01 output lacks field(s): %s', strjoin(missing, ', '));
end

n_passages = config.Npass;
n_channels = 8;
window = config.monitoring_window;
clean_trimmed = zeros(n_passages, n_channels, ...
    window.trimmed_sample_count);
tail_sample = zeros(n_passages, n_channels);

for passage_index = 1:n_passages
    groups = { ...
        data.AceleracaoPrimVag{1, passage_index}(1:3,:); ...
        data.AcelWheelsetPrimVag{1, passage_index}(1:2,:); ...
        data.PitchPrimVag{1, passage_index}(1:3,:)};
    response = vertcat(groups{:});
    dim_acel = data.DimAcel(1, passage_index);
    dim_space = data.DimSpace(1, passage_index);
    if ~isscalar(dim_acel) || ~isscalar(dim_space) || ...
            ~isfinite(dim_acel) || ~isfinite(dim_space) || ...
            dim_acel ~= fix(dim_acel) || dim_space ~= fix(dim_space) || ...
            dim_acel < 2 || dim_space < window.crop_end_untrimmed || ...
            ~isequal(size(response), [n_channels dim_acel]) || ...
            any(~isfinite(response), 'all')
        error('ttbi:f25_extract_monitoring_signals:Passage', ...
            'Passage %d has invalid RAW response geometry.', passage_index);
    end

    raw_coordinate = linspace(1, dim_space, dim_acel);
    query = window.crop_start:window.crop_end_untrimmed;
    source_window = interp1(raw_coordinate, response', query, 'linear')';
    if ~isequal(size(source_window), ...
            [n_channels window.untrimmed_sample_count]) || ...
            any(~isfinite(source_window), 'all')
        error('ttbi:f25_extract_monitoring_signals:Interpolation', ...
            'Passage %d did not yield one finite 8-by-5831 window.', ...
            passage_index);
    end
    clean_trimmed(passage_index,:,:) = ...
        source_window(:,1:window.trimmed_sample_count);
    tail_sample(passage_index,:) = source_window(:,end);
end

output = struct();
output.schema = 'f25-saved-monitoring-window-v1';
output.channel_schema_id = config.channel_schema_id;
output.clean_trimmed = clean_trimmed;
output.monitoring_tail_sample = tail_sample;
output.source_window_samples = window.untrimmed_sample_count;
output.trimmed_window_samples = window.trimmed_sample_count;
output.tail_samples_trimmed = window.tail_samples_trimmed;
output.crop_start_one_based = window.crop_start;
output.crop_end_untrimmed_one_based = window.crop_end_untrimmed;
output.crop_end_trimmed_one_based = window.crop_end_trimmed;
output.samples_per_m = window.samples_per_m;
end
