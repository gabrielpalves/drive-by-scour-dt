function smoke_f25_solver()
%SMOKE_F25_SOLVER One-passage coupled-solver/window acceptance for F25.
%
% This diagnostic is intentionally nonqualifying and writes no dataset.  It
% exercises the exact F25 geometry, Type-2 profile, healthy scenario, bounded
% EOV mapping, physical8 channels, contact gate, 5,831-point reconstruction,
% and tail trim through the same A01..D01 solver path as F25_Run.

config = ttbi.f25_experiment_config('F25-R');
state = ttbi.f25_state_design(config);
operations = ttbi.f25_sample_operations( ...
    config, state.StateNamedStreamSeedID(1,1));
Damage = ttbi.f25_damage_for_state(state, 1);
Damage.desvio = 0;
Damage.track = [];
passage_index = 1;
Train = A01_Train(operations.speed_mps(passage_index), ...
    operations.a01_vehicle_inputs(:,:,passage_index));
Track = A02_Track();
Beam = A03_Bridge(struct('Prop', struct( ...
    'L', config.L_bridge, 'num_spans', config.num_spans)));
Beam.Prop.E = Beam.Prop.E-Beam.Prop.E*0.003* ...
    (operations.temperature_C(passage_index)-15);
[Calc, Beam, Track] = A04_Options(Beam, Track, config.profile_config);
[Sol, Calc, Train, Beam, ~] = B00_Calculations( ...
    Calc, Train, Track, Beam, Damage);
assert(isfinite(Beam.Modal.f(1)) && Beam.Modal.f(1) >= 3 && ...
    Beam.Modal.f(1) <= 6);
assert(isfinite(Sol.F_tension_max) && ...
    Sol.F_tension_max <= config.contact_max_tension_N);
assert(isfinite(Sol.tension_frac_max) && ...
    Sol.tension_frac_max <= config.contact_max_tension_fraction);
data = D01_DataProcessing(1, 1, Sol, Train, Calc, Damage, struct());
one = config;
one.Npass = 1;
monitoring = ttbi.f25_extract_monitoring_signals(data, one);
assert(isequal(size(monitoring.clean_trimmed), [1 8 5830]));
assert(isequal(size(monitoring.monitoring_tail_sample), [1 8]));
assert(all(isfinite(monitoring.clean_trimmed), 'all'));
assert(all(isfinite(monitoring.monitoring_tail_sample), 'all'));
fprintf(['PASS smoke_f25_solver: f1=%.6g Hz, F_tension=%.6g N, ' ...
    'fraction=%.6g, saved window 5831 -> 5830 + retained tail.\n'], ...
    Beam.Modal.f(1), Sol.F_tension_max, Sol.tension_frac_max);
end
