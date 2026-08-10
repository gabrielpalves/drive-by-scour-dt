function smoke_f25_contract()
%SMOKE_F25_CONTRACT No-solver acceptance for the isolated F25 experiment.

cfg_r = ttbi.f25_experiment_config('F25-R');
cfg_x = ttbi.f25_experiment_config('F25-X');
assert(~strcmp(cfg_r.experiment_id, cfg_x.experiment_id));
assert(strcmp(cfg_r.dataset_id, cfg_x.dataset_id));
assert(strcmp(cfg_r.generation_results_root, ...
    'f25_artifacts/shared_data/generation'));
assert(~strcmp(cfg_r.generation_results_root, 'Results'));
assert(~strcmp(cfg_r.cache_root, cfg_x.cache_root));
assert(~strcmp(cfg_r.training_manifest_root, ...
    cfg_x.training_manifest_root));
assert(~strcmp(cfg_r.training_results_root, cfg_x.training_results_root));
assert(strcmp(cfg_r.channel_schema_id, 'physical8_v1'));
assert(strcmp(cfg_r.python_contract_sha256, ...
    '614ecca52a5dac91c081d826dda1a2ddda028c229824cee60348d841ec9a2b1e'));
assert(cfg_r.partition_seed == 2025080901 && ...
    cfg_r.eov_master_seed == 2025080902 && ...
    cfg_r.noise_master_seed == 2025080903);
assert(cfg_r.n_states == 10 && cfg_r.Npass == 200 && ...
    cfg_r.total_passages == 2000);

catalog = cfg_r.scenarios;
assert(isequal(catalog.labels, ...
    {'Healthy';'DC2';'DC3';'DC4';'DC5';'DC6';'DC7';'DC8';'DC9';'DC10'}));
assert(isequal(catalog.crack_ei_loss', ...
    [0 .22 0 0 0 .22 .22 .22 .22 .14]));
assert(isequal(catalog.central_scour_loss', ...
    [0 0 .05 0 .10 .10 0 .10 .05 .05]));
assert(isequal(catalog.entrance_bearing_stiffness_Nm_per_rad' > 0, ...
    logical([0 0 0 1 1 0 1 1 1 1])));
assert(all(catalog.bearing_vectors_Nm_per_rad(:, 2) == 0));
assert(all(catalog.scour_vectors(:, [1 3]) == 0, 'all'));

design_r = ttbi.f25_state_design(cfg_r);
design_x = ttbi.f25_state_design(cfg_x);
assert(isequal(design_r.StateUID, design_x.StateUID));
assert(isequal(design_r.StateNamedStreamSeedID, ...
    design_x.StateNamedStreamSeedID));
assert(isequal(design_r.PassageNamedStreamSeedID, ...
    design_x.PassageNamedStreamSeedID));
assert(isequal(size(design_r.PassageNamedStreamSeedID), [10 200 2]));

operations = ttbi.f25_sample_operations( ...
    cfg_r, design_r.StateNamedStreamSeedID(1,1));
assert(strcmp(operations.schema, 'f25-operations-v1'));
assert(all(operations.speed_km_h > 70 & operations.speed_km_h < 90));
assert(all(operations.temperature_C > 3 & operations.temperature_C < 33));
assert(all(operations.body_mass_kg > 33000 & ...
    operations.body_mass_kg < 40000, 'all'));
assert(all(operations.primary_suspension_N_per_m > 2*2640e3 & ...
    operations.primary_suspension_N_per_m < 2*2920e3, 'all'));
assert(all(operations.secondary_suspension_N_per_m > 2*942e3 & ...
    operations.secondary_suspension_N_per_m < 2*1042e3, 'all'));
train_probe = A01_Train(operations.speed_mps(1), ...
    operations.a01_vehicle_inputs(:,:,1));
assert(abs(train_probe.Veh(1).Body.m-operations.body_mass_kg(1,1,1)) < 1e-8);
assert(abs(train_probe.Veh(1).Susp.Prim.k(1)- ...
    operations.primary_suspension_N_per_m(1,1,1)) < 1e-8);

dc8 = ttbi.f25_damage_for_state(design_r, 8);
assert(isequal(dc8.scour_rates, [0 .10 0]));
assert(dc8.bearing_left == 1e9 && dc8.bearing_right == 0);
assert(dc8.crack_intensity == .22 && dc8.crack_locs == 29.85 && ...
    dc8.crack_lc == .15);
healthy = ttbi.f25_damage_for_state(design_r, 1);
assert(isempty(healthy.crack_locs) && all(healthy.scour_rates == 0) && ...
    healthy.bearing_left == 0);

% Exercise only configuration/geometry functions, never the coupled solve.
Track = A02_Track();
Beam = A03_Bridge(struct( ...
    'Prop', struct('L', cfg_r.L_bridge, 'num_spans', cfg_r.num_spans)));
[Calc, Beam, Track] = A04_Options(Beam, Track, cfg_r.profile_config);
assert(Calc.Profile.Type == 2);
assert(strcmp(Calc.Profile.profile_contract, ...
    'fernandes-2025-stored-type2-v1'));
assert(Beam.Mesh.Ele.num_per_spacing == 4);

Train = A01_Train(80/3.6, zeros(cfg_r.Nveh, cfg_r.Nprop));
[Calc, Train, Beam] = B43_ModelGeometry(Calc, Train, Track, Beam); %#ok<ASGLU>
assert(Beam.Mesh.Ele.num == 266);
assert(abs(Beam.Mesh.Ele.L - .15) < 1e-14);
assert(abs(Beam.Prop.L - 39.9) < 1e-12);
Beam = B01_ElementsAndCoordinates(Beam);
support_node = find(abs(Beam.Mesh.Nodes.coord - 19.95) < 1e-12);
assert(isequal(support_node, 134)); % source node 133 in zero-based indexing

midpoints = (Beam.Mesh.Nodes.coord(1:end-1) + ...
    Beam.Mesh.Nodes.coord(2:end))/2;
crack_elements = find(abs(midpoints - catalog.crack_block_centre_m) <= ...
    catalog.crack_block_half_length_m);
assert(isequal(crack_elements(:)', [199 200]));
assert(abs(Beam.Mesh.Nodes.coord(199) - 29.70) < 1e-12);
assert(abs(Beam.Mesh.Nodes.coord(201) - 30.00) < 1e-12);

stored = load(cfg_r.profile_asset_path, 'Calc');
assert(isfield(stored, 'Calc') && isfield(stored.Calc, 'Profile'));
assert(isfield(stored.Calc.Profile, 'x') && ...
    isfield(stored.Calc.Profile, 'h'));
stored_x = stored.Calc.Profile.x(:);
assert(numel(stored_x) == numel(stored.Calc.Profile.h));
assert(stored_x(1) <= 0 && stored_x(end) >= Calc.Profile.L);
assert(abs(stored.Calc.Profile.L_bridge - 39.9) < 1e-12);

% Execute the Type-2 loader itself on the live F25 domain. This is profile
% interpolation only; it does not assemble or solve the coupled system.
Calc.Profile.min_dx = .01;
Calc.Position.x = 0:.01:Calc.Profile.L;
Calc.Plot.Profile_original = 0;
live_bridge_before = Calc.Profile.L_bridge;
Calc = B19_GenerateProfile(Calc);
assert(Calc.Profile.Type == 2);
assert(Calc.Profile.L_bridge == live_bridge_before);
assert(numel(Calc.Profile.h) == numel(Calc.Profile.x));
assert(all(isfinite(Calc.Profile.h)));
assert(isempty(Calc.Profile.PSD_X) && isempty(Calc.Profile.PSD_Y));

assert(cfg_r.raw_window_samples == 5831 && ...
    cfg_r.trimmed_window_samples == 5830 && ...
    cfg_r.paa_segments == 583 && ...
    cfg_r.paa_samples_per_segment == 10);
assert(cfg_r.monitoring_window.physical_bridge_samples == 3990);
assert(cfg_r.monitoring_window.source_convention_bridge_samples == 4000);
assert(cfg_r.monitoring_window.extra_beyond_physical_bridge_samples == 10);
assert(cfg_r.monitoring_window.crop_end_untrimmed - ...
    cfg_r.monitoring_window.crop_start + 1 == 5831);

% Manufactured D01-like full RAW verifies interpolation coordinates and the
% saved-output tail convention without assembling the coupled solver.
window_cfg = cfg_r;
window_cfg.Npass = 2;
fixture = struct();
for passage_index = 1:window_cfg.Npass
    base = 0:6999;
    fixture.AceleracaoPrimVag{1,passage_index} = ...
        [base; base+10000; base+20000];
    fixture.AcelWheelsetPrimVag{1,passage_index} = ...
        [base+30000; base+40000; base+50000; base+60000];
    fixture.PitchPrimVag{1,passage_index} = ...
        [base+50000; base+60000; base+70000];
    fixture.DimAcel(1,passage_index) = 7000;
    fixture.DimSpace(1,passage_index) = 7000;
end
monitoring = ttbi.f25_extract_monitoring_signals(fixture, window_cfg);
assert(isequal(size(monitoring.clean_trimmed), [2 8 5830]));
assert(isequal(size(monitoring.monitoring_tail_sample), [2 8]));
assert(monitoring.clean_trimmed(1,1,1) == 1000);
assert(monitoring.clean_trimmed(1,1,end) == 6829);
assert(monitoring.monitoring_tail_sample(1,1) == 6830);
fprintf(['PASS smoke_f25_contract: 10 scenarios x 200 passages, ' ...
    '39.9 m/0.15 m geometry, exact 29.70-30.00 m crack block, ' ...
    'source-locked Type-2 profile, shared R/X state seeds.\n']);
end
