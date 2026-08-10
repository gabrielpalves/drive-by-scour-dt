function contact_gate_write_or_verify_text(path, text)
%CONTACT_GATE_WRITE_OR_VERIFY_TEXT Create or authenticate a text sidecar.

common = contact_closure_common();
expected = reshape(unicode2native(char(text), 'UTF-8'), 1, []);
if isfile(path)
    if ~common.regular_nonsymlink(path) || ...
            ~isequal(common.file_bytes(path), expected)
        error('contact_closure_gate:ImmutableSummary', ...
            'Existing deterministic summary sidecar differs: %s', path);
    end
else
    contact_gate_write_text_atomic(path, text);
end
end
