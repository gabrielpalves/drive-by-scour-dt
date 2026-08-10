function idx = contact_nearest_dt_index(runs, target_ms)
%CONTACT_NEAREST_DT_INDEX Locate the realized step nearest a target.

[~, idx] = min(abs([runs.actual_dt_ms] - target_ms));
end
