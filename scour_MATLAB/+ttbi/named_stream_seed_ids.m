function [state_seeds, passage_seeds] = named_stream_seed_ids( ...
        state_seed_ids, state_uids, Npass, schedule_version, ...
        state_names, passage_names)
%NAMED_STREAM_SEED_IDS Per-state and per-passage named RNG substream seeds.
    % Namespace-separated SHA-derived uint32 seeds. Every state/component pair
    % gets its own deterministic stream; adding a random draw inside one
    % component cannot shift operations or any other EOV. Zero and any collision
    % across the complete design are fail-closed before generation starts.
    n_states_ = numel(state_seed_ids);
    state_seeds = zeros(n_states_, numel(state_names), 'uint32');
    passage_seeds = zeros(n_states_, Npass, numel(passage_names), 'uint32');
    for i_ = 1:n_states_
        for stream_ = 1:numel(state_names)
            key_ = sprintf('%s|root=%u|uid=%s|stream=%s', ...
                schedule_version, state_seed_ids(i_), state_uids{i_}, ...
                state_names{stream_});
            state_seeds(i_, stream_) = ttbi.seed32(key_);
        end
        for pass_ = 1:Npass
            for stream_ = 1:numel(passage_names)
                key_ = sprintf('%s|root=%u|uid=%s|stream=%s|pass=%05d', ...
                    schedule_version, state_seed_ids(i_), state_uids{i_}, ...
                    passage_names{stream_}, pass_);
                passage_seeds(i_, pass_, stream_) = ttbi.seed32(key_);
            end
        end
    end
    all_ids_ = [state_seed_ids(:); state_seeds(:); passage_seeds(:)];
    if any(all_ids_ == 0) || numel(unique(all_ids_)) ~= numel(all_ids_)
        error(['A00: named RNG substream seed collision/zero in the complete ' ...
            'design. Bump random_stream_schedule_version; do not use DC or ' ...
            'silently accept correlated namespaces.']);
    end
end
