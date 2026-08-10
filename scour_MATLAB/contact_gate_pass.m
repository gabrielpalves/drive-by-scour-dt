function pass = contact_gate_pass( ...
        peak_tension_n, tension_fraction, gates_n, fraction_gate)
%CONTACT_GATE_PASS Evaluate post-hoc tension limits for one run.

if ~isfinite(peak_tension_n) || ~isfinite(tension_fraction)
    pass = false(size(gates_n));
else
    pass = (peak_tension_n <= gates_n) & ...
        (tension_fraction <= fraction_gate);
end
end
