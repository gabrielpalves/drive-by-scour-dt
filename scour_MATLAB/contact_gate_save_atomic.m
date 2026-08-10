function contact_gate_save_atomic(path, variable_name, value)
%CONTACT_GATE_SAVE_ATOMIC Create one immutable MAT artifact atomically.

tmp = [path, '.tmp'];
if isfile(path) || isfile(tmp)
    error('contact_closure_gate:ImmutableArtifact', ...
        'Refusing existing immutable/stale temporary artifact: %s', path);
end
payload = struct();
payload.(variable_name) = value;
save(tmp, '-struct', 'payload');
[ok, message] = movefile(tmp, path);
if ~ok
    error('contact_closure_gate:AtomicMove', ...
        'Could not install immutable artifact: %s', message);
end
end
