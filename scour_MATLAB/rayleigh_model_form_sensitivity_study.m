function report = rayleigh_model_form_sensitivity_study(output_file,design)
%RAYLEIGH_MODEL_FORM_SENSITIVITY_STUDY Compare two damping closures.
%
% report = rayleigh_model_form_sensitivity_study()
% report = rayleigh_model_form_sensitivity_study(output_file,design)
%
% One healthy F40 passage establishes the bridge and rail Rayleigh
% coefficients. Scour, bearing-fixity, and crack interventions are then each
% solved twice: (1) the production closure recalibrated from the damaged
% state's modes and (2) the prospectively fixed healthy coefficients. M and K
% must be bit-identical within each pair; only C and the resulting response may
% differ. This is a deterministic model-form sensitivity, not population
% inference and not a physical-validation claim.

if nargin < 1 || isempty(output_file)
    output_file = '';
end
if nargin < 2 || isempty(design)
    design = struct();
end
if ~(ischar(output_file) || (isstring(output_file) && isscalar(output_file)))
    error('rayleigh_sensitivity:Output', ...
        'output_file must be empty or one scalar path.');
end
output_file = char(output_file);
if ~isempty(output_file) && (exist(output_file,'file') || exist(output_file,'dir'))
    error('rayleigh_sensitivity:OutputExists', ...
        'Refusing to overwrite existing output: %s',output_file);
end

defaults = struct( ...
    'L_bridge_m',40, ...
    'num_spans',2, ...
    'speed_kmh',80, ...
    'temperature_c',18, ...
    'case_label','F40_HEALTHY_V80_T18', ...
    'vehicle_draw',zeros(5,3), ...
    'profile_phase_seed',20260728, ...
    'profile_fra_class',4, ...
    'scour_support',2, ...
    'crack_location_m',20);
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(design,names{k})
        design.(names{k}) = defaults.(names{k});
    end
end

[healthy_damage,catalog,design] = ...
    response_signature_damage_catalog(design);
selected = strcmp({catalog.first_changed_quantity}, ...
    'bridge_stiffness_state_specific_rayleigh_damping');
catalog = catalog(selected);
if ~isequal({catalog.name},{'scour','bearing_fixity','crack'})
    error('rayleigh_sensitivity:DamageInventory', ...
        'The registered structural intervention inventory drifted.');
end

fprintf('RAYLEIGH SENSITIVITY: healthy coefficient source\n');
healthy = response_signature_run_one(healthy_damage,design);
source_id = sprintf('healthy-F40-V80-T18-rayleigh-v1');
fixed = struct( ...
    'source_id',source_id, ...
    'bridge_alpha',healthy.damping.bridge.alpha, ...
    'bridge_beta',healthy.damping.bridge.beta, ...
    'rail_alpha',healthy.damping.rail.alpha, ...
    'rail_beta',healthy.damping.rail.beta);

n = numel(catalog);
mechanism = strings(n,1);
production_bridge_alpha = zeros(n,1);
production_bridge_beta = zeros(n,1);
production_rail_alpha = zeros(n,1);
production_rail_beta = zeros(n,1);
beam_c_delta_fro = zeros(n,1);
model_c_delta_fro = zeros(n,1);
bridge_acceleration_rms_fractional_change = zeros(n,1);
contact_tensile_peak_fractional_change = zeros(n,1);
physical8_rms_fractional_change = zeros(n,8);
physical8_peak_fractional_change = zeros(n,8);
fixed_closure_changes_only_damping = false(n,1);
crn_controls_exact = false(n,1);
production_runs = cell(n,1);
fixed_runs = cell(n,1);

for k = 1:n
    item = catalog(k);
    fprintf('  [%d/%d] %s: recalibrated closure\n',k,n,item.name);
    production = response_signature_run_one(item.damage,design);
    fixed_design = design;
    fixed_design.fixed_rayleigh = fixed;
    fprintf('  [%d/%d] %s: fixed healthy closure\n',k,n,item.name);
    fixed_run = response_signature_run_one(item.damage,fixed_design);

    same_mk = isequal(production.physics.beam_M,fixed_run.physics.beam_M) && ...
        isequal(production.physics.beam_K,fixed_run.physics.beam_K) && ...
        isequal(production.physics.model_M,fixed_run.physics.model_M) && ...
        isequal(production.physics.model_K,fixed_run.physics.model_K);
    same_modes = isequal(production.modal.frequency_hz, ...
        fixed_run.modal.frequency_hz) && ...
        isequal(production.modal.mode_shapes,fixed_run.modal.mode_shapes);
    c_changed = ~isequal(production.physics.beam_C,fixed_run.physics.beam_C) && ...
        ~isequal(production.physics.model_C,fixed_run.physics.model_C);
    fixed_exact = ...
        strcmp(fixed_run.damping.bridge.policy, ...
            'fixed-coefficients-sensitivity-v1') && ...
        strcmp(fixed_run.damping.rail.policy, ...
            'fixed-coefficients-sensitivity-v1') && ...
        fixed_run.damping.bridge.alpha == fixed.bridge_alpha && ...
        fixed_run.damping.bridge.beta == fixed.bridge_beta && ...
        fixed_run.damping.rail.alpha == fixed.rail_alpha && ...
        fixed_run.damping.rail.beta == fixed.rail_beta;
    if ~(same_mk && same_modes && c_changed && fixed_exact)
        error('rayleigh_sensitivity:MechanismBoundary', ...
            ['%s did not preserve exact M/K/modes while changing only the ' ...
             'declared damping closure.'],item.name);
    end

    crn_controls_exact(k) = isequal(healthy.controls,production.controls) && ...
        isequal(production.controls,fixed_run.controls);
    if ~crn_controls_exact(k)
        error('rayleigh_sensitivity:CrnMismatch', ...
            '%s changed a non-intervention passage control.',item.name);
    end

    mechanism(k) = string(item.name);
    production_bridge_alpha(k) = production.damping.bridge.alpha;
    production_bridge_beta(k) = production.damping.bridge.beta;
    production_rail_alpha(k) = production.damping.rail.alpha;
    production_rail_beta(k) = production.damping.rail.beta;
    beam_c_delta_fro(k) = norm( ...
        fixed_run.physics.beam_C-production.physics.beam_C,'fro');
    model_c_delta_fro(k) = norm( ...
        fixed_run.physics.model_C-production.physics.model_C,'fro');
    production_bridge_rms = sqrt(mean(production.bridge.signal(2,:).^2));
    fixed_bridge_rms = sqrt(mean(fixed_run.bridge.signal(2,:).^2));
    bridge_acceleration_rms_fractional_change(k) = ...
        local_fractional_change(fixed_bridge_rms,production_bridge_rms);
    contact_tensile_peak_fractional_change(k) = local_fractional_change( ...
        fixed_run.contact.aggregate_tensile_peak_n, ...
        production.contact.aggregate_tensile_peak_n);
    production_rms = sqrt(mean(production.physical8.signal.^2,2));
    fixed_rms = sqrt(mean(fixed_run.physical8.signal.^2,2));
    production_peak = max(abs(production.physical8.signal),[],2);
    fixed_peak = max(abs(fixed_run.physical8.signal),[],2);
    physical8_rms_fractional_change(k,:) = ...
        local_fractional_change(fixed_rms,production_rms)';
    physical8_peak_fractional_change(k,:) = ...
        local_fractional_change(fixed_peak,production_peak)';
    fixed_closure_changes_only_damping(k) = true;
    production_runs{k} = production;
    fixed_runs{k} = fixed_run;
end

summary = table(mechanism,production_bridge_alpha, ...
    production_bridge_beta,production_rail_alpha,production_rail_beta, ...
    beam_c_delta_fro,model_c_delta_fro, ...
    bridge_acceleration_rms_fractional_change, ...
    contact_tensile_peak_fractional_change, ...
    fixed_closure_changes_only_damping,crn_controls_exact);

report = struct();
report.schema = 'paper1-rayleigh-model-form-sensitivity-v1';
report.status = 'EXPLORATORY_PARAMETER_SENSITIVITY_COMPLETE';
report.interpretation = [ ...
    'Paired deterministic fixed-healthy-Rayleigh sensitivity. Signed changes ' ...
    'are reported without a direction gate and are not population inference.'];
report.production_policy = 'recalibrated-current-state-grid-v1';
report.sensitivity_policy = 'fixed-coefficients-sensitivity-v1';
report.design = design;
report.fixed_healthy_coefficients = fixed;
report.healthy_damping = healthy.damping;
report.channel_schema_id = 'physical8_v1';
report.channel_names = healthy.physical8.channel_names;
report.summary = summary;
report.physical8_rms_fractional_change = ...
    physical8_rms_fractional_change;
report.physical8_peak_fractional_change = ...
    physical8_peak_fractional_change;
report.production_runs = production_runs;
report.fixed_runs = fixed_runs;

if ~isempty(output_file)
    save(output_file,'report','-v7.3');
end
fprintf('RAYLEIGH SENSITIVITY COMPLETE: %d structural pairs\n',n);
end

function change = local_fractional_change(test,reference)
if any(~isfinite(test(:))) || any(~isfinite(reference(:))) || ...
        any(reference(:) == 0)
    error('rayleigh_sensitivity:NonfiniteMetric', ...
        'Fractional-change metrics require finite nonzero references.');
end
change = test./reference-1;
end
