function contact_gate_write_text_atomic(path, text)
%CONTACT_GATE_WRITE_TEXT_ATOMIC Create one immutable UTF-8 text artifact.

tmp = [path, '.tmp'];
if isfile(path) || isfile(tmp)
    error('contact_closure_gate:ImmutableArtifact', ...
        'Refusing existing immutable/stale temporary artifact: %s', path);
end
fid = fopen(tmp, 'wb');
if fid < 0
    error('contact_closure_gate:OutputOpen', ...
        'Could not open temporary artifact: %s', tmp);
end
try
    bytes = unicode2native(char(text), 'UTF-8');
    written = fwrite(fid, bytes, 'uint8');
    if written ~= numel(bytes)
        error('contact_closure_gate:OutputWrite', ...
            'Short write for temporary artifact: %s', tmp);
    end
    close_status = fclose(fid);
    if close_status ~= 0
        error('contact_closure_gate:OutputClose', ...
            'Could not close temporary artifact: %s', tmp);
    end
catch ME
    try %#ok<TRYNC>
        fclose(fid);
    end
    rethrow(ME);
end
[ok, message] = movefile(tmp, path);
if ~ok
    error('contact_closure_gate:AtomicMove', ...
        'Could not install immutable artifact: %s', message);
end
end
