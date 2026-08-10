function policy = contact_gate_policy_definition(source_commit)
%CONTACT_GATE_POLICY_DEFINITION Return the immutable closure-gate policy.

policy = struct();
policy.schema = 'contact-closure-gate-v2';
policy.closure_interpretation = ...
    'bounded-numerical-tension-engineering-v1';
policy.source_commit = source_commit;
policy.stages = {'F40-S', 'F40-M', 'L99-S', 'L99-M'};
policy.expected_states = [31, 31, 39, 39];
policy.expected_passages = 3;
policy.expected_cases = 420;
policy.dt_ms = [1, 0.5, 0.25];
policy.gates_n = [0, 12000, 24000];
policy.fraction_gate = 0.002;
policy.common_dx_m = 0.01;
policy.reconstruction_rtol = 1e-10;
policy.reconstruction_atol = 1e-12;
policy.coarse_nrmse_max = 0.05;
policy.coarse_nmax_max = 0.10;
policy.coarse_corr_min = 0.995;
policy.medium_nrmse_max = 0.02;
policy.medium_nmax_max = 0.05;
policy.medium_corr_min = 0.999;
policy.gci_safety_factor = 1.25;
policy.gci_method = 'actual-step-generalized-richardson-v1';
policy.equivalence_rtol = 1e-10;
policy.equivalence_atol = 1e-12;
policy.gci_p_min = 1e-8;
policy.gci_p_max = 50;
policy.qoi_gci_required = true;
policy.time_grid_ulps = 8;
policy.waveform_monotonic_atol = 1e-12;
policy.finest_identity_atol = 1e-12;
policy.expected_channels = { ...
    'carbody_vertical_acceleration', ...
    'front_bogie_vertical_acceleration', ...
    'rear_bogie_vertical_acceleration', ...
    'wheelset_1_constrained_vertical_acceleration_proxy', ...
    'wheelset_2_constrained_vertical_acceleration_proxy', ...
    'carbody_pitch_rate', ...
    'front_bogie_pitch_rate', 'rear_bogie_pitch_rate'};
policy.channel_schema_id = 'physical8_v1';
end
