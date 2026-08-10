function root = contact_gate_execution_root()
%CONTACT_GATE_EXECUTION_ROOT Hash the resolved closure-gate executable set.

root = contact_resolved_module_root(contact_gate_module_files(), ...
    'contact_closure:GateModuleShadowed', 'Gate module');
end
