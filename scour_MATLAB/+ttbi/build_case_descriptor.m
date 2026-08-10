function descriptor = build_case_descriptor(campaign, state)
%BUILD_CASE_DESCRIPTOR Create the short folder name and full human descriptor.
%
% The folder name stays below practical Windows path limits.  The complete
% scientific description is retained in case_info instead of being encoded in
% the directory name.

variability_tag = [ ...
    repmat('N', 1, double(campaign.use_signal_noise)), ...
    repmat('V', 1, double(campaign.use_vehicle_variability)), ...
    repmat('S', 1, double(campaign.use_speed_variability)), ...
    repmat('T', 1, double(campaign.use_temp_variability))];
if isempty(variability_tag)
    variability_tag = 'none';
end

switch campaign.bearing_mode
    case 'off'
        bearing_tag = 'OFF';
    case 'fixed'
        if campaign.Bearing_Intensity > 0
            bearing_tag = 'ON';
        else
            bearing_tag = 'OFF';
        end
    case 'target'
        bearing_tag = 'TGT';
    otherwise
        error('ttbi:UnknownBearingMode', ...
            'Unknown bearing mode "%s".', campaign.bearing_mode);
end

eov_tag = '';
if campaign.use_crack_eov
    if strcmp(campaign.crack_draw, 'per_state')
        eov_tag = [eov_tag, '_crackST'];
    else
        eov_tag = [eov_tag, '_crackON'];
    end
end
if ~strcmp(campaign.profile_mode, 'fixed')
    eov_tag = [eov_tag, '_prof-', campaign.profile_mode];
    if strcmp(campaign.profile_mode, 'psd_fra') && ...
            strcmp(campaign.profile_draw, 'per_state')
        eov_tag = [eov_tag, 'ST'];
    end
end
if campaign.use_track_eov
    if strcmp(campaign.track_draw, 'per_state')
        eov_tag = [eov_tag, '_trackEOVST'];
    else
        eov_tag = [eov_tag, '_trackEOV'];
    end
end
if campaign.use_oor_eov
    eov_tag = [eov_tag, '_oorON'];
end

support_tag = strjoin(string(campaign.scour_supports), '-');
descriptor = struct();
descriptor.case_name = sprintf('%s_L%g_st%d', ...
    campaign.STAGE, campaign.L_bridge, state.n_states);
descriptor.case_desc = sprintf([ ...
    'L%g_%dspan_%s_scourS%s_bear%s%s_dano0-%gpct_' ...
    'states%d_Npass%d_var%s_railC%gm'], ...
    campaign.L_bridge, campaign.num_spans, campaign.damage_mode, ...
    support_tag, bearing_tag, eov_tag, campaign.dano_max * 100, ...
    state.n_states, campaign.Npass, variability_tag, ...
    campaign.rail_end_clearance_m);
descriptor.rail_end_clearance_m = campaign.rail_end_clearance_m;
descriptor.rail_end_clearance_decision_id = ...
    campaign.rail_end_clearance_decision_id;
end
