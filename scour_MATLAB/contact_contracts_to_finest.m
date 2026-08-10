function contracts = contact_contracts_to_finest(phi, rtol, atol)
%CONTACT_CONTRACTS_TO_FINEST Check monotone error contraction to finest grid.

phi = double(phi(:));
if numel(phi) ~= 3 || any(~isfinite(phi))
    contracts = false;
    return
end
coarse_error = abs(phi(1) - phi(3));
medium_error = abs(phi(2) - phi(3));
tol = atol + rtol * max([1; abs(phi)]);
contracts = medium_error <= coarse_error + tol;
end
