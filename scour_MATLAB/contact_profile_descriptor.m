function cfg = contact_profile_descriptor( ...
        case_info, data, passage_index, descriptor_contract)
%CONTACT_PROFILE_DESCRIPTOR Rebuild the exact saved rail-profile descriptor.

profile_mode = contact_state_text(data, 'profile_mode', ...
    contact_case_text(case_info, 'profile_mode', 'fixed'));
cfg = struct('mode', profile_mode);
switch profile_mode
    case 'fixed'
        % No additional descriptor.
    case 'fixed_scaled'
        cfg.intensity = double(data.profile_log(passage_index));
    case 'psd_fra'
        profile_values = double(data.profile_log(:));
        if any(profile_values ~= profile_values(1))
            error('contact_closure:ProfileNotPersistent', ...
                ['Paper-1 psd_fra qualification requires one persistent profile ' ...
                 'class per state; profile_log varies by passage.']);
        end
        cfg.fra_class = profile_values(1);
        % Derive phase only from the semantic StateUID/named stream.
        cfg.phase_seed = descriptor_contract.profile_phase_seed;
        jitter_mm = 0;
        if isfield(case_info, 'profile_jitter_sd_mm')
            jitter_mm = double(case_info.profile_jitter_sd_mm);
        end
        if jitter_mm ~= 0
            error('contact_closure:UnreproducibleProfileJitter', ...
                ['profile_jitter_sd_mm=%g is nonzero, but the realized jitter ' ...
                 'was not persisted. Exact passage reconstruction is impossible.'], ...
                jitter_mm);
        end
        cfg.jitter_sd_m = 0;
    otherwise
        error('contact_closure:UnknownProfile', ...
            'Unsupported profile_mode "%s".', profile_mode);
end
end
