function contact_gate_recover_one_temp(final_path, tmp_path)
%CONTACT_GATE_RECOVER_ONE_TEMP Recover one deterministic atomic temp.

common = contact_closure_common();
if ~isfile(tmp_path)
    return
end
if ~common.regular_nonsymlink(tmp_path) || isfile(final_path)
    error('contact_closure_gate:AmbiguousTemp', ...
        'Atomic temp is foreign or coexists with final artifact: %s', tmp_path);
end
delete(tmp_path);
if isfile(tmp_path)
    error('contact_closure_gate:TempRecovery', ...
        'Could not remove interrupted non-evidence temp: %s', tmp_path);
end
fprintf('[closure] recovered interrupted temp: %s\n', tmp_path);
end
