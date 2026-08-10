function plain = contact_gate_plain_table(value)
%CONTACT_GATE_PLAIN_TABLE Convert one MATLAB table to a scalar struct.

if ~istable(value)
    error('contact_closure_gate:PlainTable', ...
        'Canonical MAT projection requires a MATLAB table source.');
end
plain = table2struct(value, 'ToScalar', true);
end
