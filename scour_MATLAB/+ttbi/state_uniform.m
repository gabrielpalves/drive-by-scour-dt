function u = state_uniform(uid, damage_seed, namespace)
%STATE_UNIFORM Deterministic U(0,1) draw bound to a state UID.
    % Deterministic U[0,1) variate keyed by UID and a named latent variable.
    % Thirteen hex digits fit exactly in binary64 (52 bits).
    h = ttbi.sha256(sprintf('%s|damage_seed=%.0f|%s', ...
        namespace, damage_seed, uid));
    u = hex2dec(h(1:13)) / 16^13;
end
