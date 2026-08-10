function tf = contact_allclose(a, b, rtol, atol)
%CONTACT_ALLCLOSE Elementwise finite comparison with mixed tolerance.

a = double(a);
b = double(b);
tf = isequal(size(a), size(b)) && all(isfinite(a(:))) && ...
    all(isfinite(b(:))) && ...
    all(abs(a(:) - b(:)) <= ...
        atol + rtol .* max(abs(a(:)), abs(b(:))));
end
