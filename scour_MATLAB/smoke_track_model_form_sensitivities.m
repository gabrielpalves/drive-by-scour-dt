function smoke_track_model_form_sensitivities()
%SMOKE_TRACK_MODEL_FORM_SENSITIVITIES Exact parameter-arm acceptance checks.

p545 = ttbi.zhai_ballast_properties(0.545);
p600 = ttbi.zhai_ballast_properties(0.600);
local_close(p545.effective_ballast_mass_kg,531.421984825421,1e-9);
local_close(p545.effective_ballast_stiffness_n_m,137748174.778731,1e-6);
local_close(p545.effective_subgrade_stiffness_n_m,77508161.7742676,1e-6);
local_close(p600.effective_ballast_mass_kg,561.378867049648,1e-9);
local_close(p600.effective_ballast_stiffness_n_m,142643777.694237,1e-6);
local_close(p600.effective_subgrade_stiffness_n_m,85330086.3569919,1e-6);

baseline = A02_Track();
arms = ttbi.track_model_form_arms();
assert(numel(arms) == 6 && numel(unique({arms.id})) == 6);

one = ttbi.apply_track_model_form_arm(baseline,'consistent-one-seat-v1');
assert(one.Rail.Prop.I == baseline.Rail.Prop.I/2);
assert(one.Rail.Prop.rho == baseline.Rail.Prop.rho/2);
assert(one.Sleeper.Prop.m == baseline.Sleeper.Prop.m/2);
assert(one.Pad.Prop.k == baseline.Pad.Prop.k);
assert(one.Ballast.Prop.k == baseline.Ballast.Prop.k);

two = ttbi.apply_track_model_form_arm(baseline,'consistent-two-rail-v1');
assert(two.Rail.Prop.I == baseline.Rail.Prop.I);
assert(two.Rail.Prop.rho == baseline.Rail.Prop.rho);
assert(two.Sleeper.Prop.m == baseline.Sleeper.Prop.m);
assert(two.Pad.Prop.k == 2*baseline.Pad.Prop.k);
assert(two.Ballast.Prop.m == 2*baseline.Ballast.Prop.m);
assert(two.Ballast.Prop.k == 2*baseline.Ballast.Prop.k);
assert(two.SubBallast.Prop.k == 2*baseline.SubBallast.Prop.k);
assert(isequal(two.BallastOnBeam.Prop,two.Ballast.Prop));

spacing = ttbi.apply_track_model_form_arm( ...
    baseline,'spacing-consistent-0p600-v1');
assert(spacing.Sleeper.spacing == 0.6);
assert(spacing.Ballast.Prop.m == p600.effective_ballast_mass_kg);
assert(spacing.Ballast.Prop.k == p600.effective_ballast_stiffness_n_m);
assert(spacing.SubBallast.Prop.k == p600.effective_subgrade_stiffness_n_m);
assert(spacing.Ballast.Prop.c == baseline.Ballast.Prop.c);
assert(spacing.SubBallast.Prop.c == baseline.SubBallast.Prop.c);

low = ttbi.apply_track_model_form_arm( ...
    baseline,'rail-damping-0p050pct-v1');
high = ttbi.apply_track_model_form_arm( ...
    baseline,'rail-damping-0p200pct-v1');
assert(low.Rail.Damping.per == 0.05);
assert(high.Rail.Damping.per == 0.20);

registered = ttbi.apply_track_model_form_arm( ...
    baseline,'baseline-hybrid-v1');
assert(isequal(rmfield(registered,'ModelForm'),baseline));
assert(strcmp(registered.ModelForm.arm_id,'baseline-hybrid-v1'));

fprintf(['smoke_track_model_form_sensitivities PASS: exact Zhai 0.545/0.600 ' ...
    'properties and six immutable parameter arms\n']);
end

function local_close(actual,expected,absolute_tolerance)
assert(abs(actual-expected) <= absolute_tolerance, ...
    'Unexpected value %.17g; expected %.17g (tol %.3g).', ...
    actual,expected,absolute_tolerance);
end
