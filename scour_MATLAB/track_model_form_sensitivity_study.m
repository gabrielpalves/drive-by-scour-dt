function report = track_model_form_sensitivity_study(output_file,design)
%TRACK_MODEL_FORM_SENSITIVITY_STUDY Run the predeclared paired track arms.
%
% report = track_model_form_sensitivity_study()
% report = track_model_form_sensitivity_study(output_file,design)
%
% Six deterministic healthy F40 passages are solved: the inherited production
% baseline plus consistent one-seat, consistent two-rail, spacing-corrected,
% and low/high rail-damping arms.  All operational, profile, vehicle, geometry,
% crop, and solver controls are exact common random numbers.  The study reports
% signed response changes; it does not impose a post-hoc direction gate.

if nargin < 1 || isempty(output_file)
    output_file = '';
end
if nargin < 2
    design = struct();
end
if ~(ischar(output_file) || (isstring(output_file) && isscalar(output_file)))
    error('track_model_form_sensitivity:Output', ...
        'output_file must be empty or one scalar path.');
end
output_file = char(output_file);
if ~isempty(output_file) && (exist(output_file,'file') || exist(output_file,'dir'))
    error('track_model_form_sensitivity:OutputExists', ...
        'Refusing to overwrite existing output: %s',output_file);
end

% Frozen representative Paper-1 case; callers may override values explicitly.
defaults = struct( ...
    'L_bridge_m',40, ...
    'num_spans',2, ...
    'speed_kmh',80, ...
    'temperature_c',18, ...
    'case_label','F40_HEALTHY_V80_T18', ...
    'vehicle_draw',zeros(5,3), ...
    'profile_phase_seed',20260728, ...
    'profile_fra_class',4, ...
    'scour_support',2);
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(design,names{k})
        design.(names{k}) = defaults.(names{k});
    end
end
[healthy_damage,~,design] = response_signature_damage_catalog(design);
arms = ttbi.track_model_form_arms();
n_arms = numel(arms);
runs = cell(n_arms,1);

for k = 1:n_arms
    one_design = design;
    one_design.track_model_form_arm = arms(k).id;
    fprintf('TRACK MODEL-FORM [%d/%d] %s\n',k,n_arms,arms(k).id);
    runs{k} = response_signature_run_one(healthy_damage,one_design);
end

baseline = runs{1};
channel_names = baseline.physical8.channel_names;
n_channels = numel(channel_names);
arm_id = strings(n_arms,1);
family = strings(n_arms,1);
channel_rms = zeros(n_arms,n_channels);
channel_peak_abs = zeros(n_arms,n_channels);
bridge_displacement_rms_m = zeros(n_arms,1);
bridge_acceleration_rms_m_s2 = zeros(n_arms,1);
contact_tensile_peak_n = zeros(n_arms,1);
contact_positive_fraction = zeros(n_arms,1);
first_elastic_bridge_frequency_hz = zeros(n_arms,1);
crn_controls_exact = false(n_arms,1);
track_properties = cell(n_arms,1);

baseline_controls = rmfield(baseline.controls,'track_model_form');
for k = 1:n_arms
    one = runs{k};
    if ~isequal(size(one.physical8.signal),size(baseline.physical8.signal)) || ...
            ~isequal(size(one.bridge.signal),size(baseline.bridge.signal))
        error('track_model_form_sensitivity:ResponseShape', ...
            'Arm %s changed a registered response shape.',arms(k).id);
    end
    arm_id(k) = string(arms(k).id);
    family(k) = string(arms(k).family);
    channel_rms(k,:) = sqrt(mean(one.physical8.signal.^2,2));
    channel_peak_abs(k,:) = max(abs(one.physical8.signal),[],2);
    bridge_displacement_rms_m(k) = sqrt(mean(one.bridge.signal(1,:).^2));
    bridge_acceleration_rms_m_s2(k) = sqrt(mean(one.bridge.signal(2,:).^2));
    contact_tensile_peak_n(k) = one.contact.aggregate_tensile_peak_n;
    contact_positive_fraction(k) = one.contact.aggregate_positive_fraction;
    first_elastic_bridge_frequency_hz(k) = one.modal.frequency_hz(1);
    crn_controls_exact(k) = isequal( ...
        rmfield(one.controls,'track_model_form'),baseline_controls);
    track_properties{k} = one.controls.track_model_form;
end
if ~all(crn_controls_exact)
    error('track_model_form_sensitivity:CrnMismatch', ...
        'At least one arm changed a non-intervention control.');
end

channel_rms_fractional_change = channel_rms./channel_rms(1,:)-1;
channel_peak_fractional_change = ...
    channel_peak_abs./channel_peak_abs(1,:)-1;
summary = table(arm_id,family,bridge_displacement_rms_m, ...
    bridge_acceleration_rms_m_s2,contact_tensile_peak_n, ...
    contact_positive_fraction,first_elastic_bridge_frequency_hz, ...
    crn_controls_exact);

report = struct();
report.schema = 'paper1-track-model-form-sensitivity-v1';
report.status = 'EXPLORATORY_PARAMETER_SENSITIVITY_COMPLETE';
report.production_baseline_arm_id = 'baseline-hybrid-v1';
report.interpretation = [ ...
    'Paired deterministic parameter-level sensitivity. Report signed ' ...
    'changes; do not describe these six passages as population inference.'];
report.design = design;
report.arms = arms;
report.channel_schema_id = 'physical8_v1';
report.channel_names = channel_names;
report.summary = summary;
report.channel_rms = channel_rms;
report.channel_peak_abs = channel_peak_abs;
report.channel_rms_fractional_change = channel_rms_fractional_change;
report.channel_peak_fractional_change = channel_peak_fractional_change;
report.track_properties = track_properties;

if ~isempty(output_file)
    save(output_file,'report','-v7.3');
end
fprintf('TRACK MODEL-FORM COMPLETE: %d paired healthy F40 arms\n',n_arms);
end
