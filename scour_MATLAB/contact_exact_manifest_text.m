function text = contact_exact_manifest_text(value)
%CONTACT_EXACT_MANIFEST_TEXT Require one nonempty, NUL-free text scalar.

if ~(ischar(value) || (isstring(value) && isscalar(value)))
    error('dataset_digest_manifest:BadManifest', ...
        'Manifest text fields must be scalar text.');
end
text = char(value);
if isempty(text) || contains(text, char(0))
    error('dataset_digest_manifest:BadManifest', ...
        'Manifest text fields must be nonempty and contain no NUL.');
end
end
