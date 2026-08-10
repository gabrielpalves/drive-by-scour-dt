function smoke_rayleigh_model_form_sensitivity()
%SMOKE_RAYLEIGH_MODEL_FORM_SENSITIVITY Solver-free damping-policy fixture.

base = struct();
base.Damping = struct('per',3);
base.Modal = struct('num_rigid_modes',0,'w',[10;20]);
base.Mesh = struct('Mg',eye(2),'Kg',diag([100 400]));
calibrated = B24_BeamDamping(base);
assert(strcmp(calibrated.Damping.rayleigh_policy, ...
    'recalibrated-current-state-grid-v1'));
assert(max(abs(calibrated.Damping.achieved_reference_damping_ratios-0.03)) ...
    < 1e-14);

fixed = base;
fixed.Modal.w = [12;24];
fixed.Damping.fixed_rayleigh_coefficients = [ ...
    calibrated.Damping.rayleigh_alpha, ...
    calibrated.Damping.rayleigh_beta];
fixed.Damping.fixed_rayleigh_source_id = 'manufactured-healthy-M0';
fixed = B24_BeamDamping(fixed);
assert(strcmp(fixed.Damping.rayleigh_policy, ...
    'fixed-coefficients-sensitivity-v1'));
assert(strcmp(fixed.Damping.rayleigh_coefficient_source_id, ...
    'manufactured-healthy-M0'));
expected = calibrated.Damping.rayleigh_alpha*fixed.Mesh.Mg + ...
    calibrated.Damping.rayleigh_beta*fixed.Mesh.Kg;
assert(isequal(fixed.Mesh.Cg,expected));
assert(any(abs(fixed.Damping.achieved_reference_damping_ratios-0.03) ...
    > 1e-6));

bad = base;
bad.Damping.fixed_rayleigh_coefficients = [1 NaN];
bad.Damping.fixed_rayleigh_source_id = 'bad';
try
    B24_BeamDamping(bad);
    error('smoke_rayleigh:AcceptedInvalid', ...
        'Invalid fixed coefficients were accepted.');
catch ME
    assert(strcmp(ME.identifier,'B24:InvalidFixedRayleighCoefficients'));
end

fprintf(['smoke_rayleigh_model_form_sensitivity PASS: recalibrated and ' ...
    'fixed-coefficient closures are explicit and exact\n']);
end
