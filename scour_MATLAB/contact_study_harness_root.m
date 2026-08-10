function root = contact_study_harness_root()
%CONTACT_STUDY_HARNESS_ROOT Hash the resolved study executable set.

root = contact_resolved_module_root(contact_study_harness_files(), ...
    'contact_closure:HarnessShadowed', 'Study harness file');
end
