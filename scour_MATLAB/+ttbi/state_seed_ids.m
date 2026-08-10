function ids = state_seed_ids(state_uids, damage_seed)
%STATE_SEED_IDS Root RNG seed per state, derived from its semantic UID.
    % Map semantic UIDs to reproducible uint32 RNG seeds. SHA-256 makes row
    % order irrelevant; the explicit collision gate turns the tiny truncation
    % risk into a fail-closed generation error rather than silent dependence.
    if ~isscalar(damage_seed) || ~isfinite(damage_seed) || ...
            damage_seed < 0 || damage_seed ~= round(damage_seed)
        error('A00: damage_seed must be one nonnegative integer.');
    end
    ids = zeros(numel(state_uids), 1, 'uint32');
    for k = 1:numel(state_uids)
        h = ttbi.sha256(sprintf( ...
            'ttbi-state-seed-v1|damage_seed=%.0f|%s', ...
            damage_seed, state_uids{k}));
        ids(k) = uint32(hex2dec(h(1:8)));
    end
    if any(ids == 0) || numel(unique(ids)) ~= numel(ids)
        error(['A00: StateSeedID collision in this design. Change the explicit ' ...
            'state-seed derivation version (zero is also reserved); never ' ...
            'resolve it by row/DC.']);
    end
end
