function [Sol] = B66_ContactForce(Sol,Track,Calc,Train,Damage)

% Calculation of vertical contact forces for each vehicle
% Note: Contact force calculation looks good only if Calc.Options.redux = 0
%   When the redux model option is ised, then the vehicle starts the simulation
%   on a flat rigid surface. At the simulation progresses, each wheel enter
%   the track and produces large vertical deformations on the track, which
%   at the same time produces large oscilations in the contact force.

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Veh = Indexed structure with Veh variables, including at least:
%   ... see script ...
% Model = Structure with Model variables, including at least:
%   ... see script ...
% Calc = Structure with Calc variables, including at least:
%   ... see script ...
% Track = Structure with Track variables, including at least:
%   ... see script ...
% Sol = Structure with Sol variables, including at least:
%   ... see script ...
% % ---- Outputs ----
% Sol = Additional fields to Sol structure variable
% ... see script ...
% -------------------------------------------------------------------------

% Auxiliary variables
ones_1_x_num_t = ones(1,Calc.Solver.num_t);

% **** VBI ****
if Calc.Options.VBI == 1
    
    % Deformation under each wheel
    [Sol] = B17_CalcUat(Sol,Track,Calc,Train.Veh,Damage);

    % Force on Beam (Contact Force)
    for veh_num = 1:Train.Veh(1).Tnum

        ind = (Calc.Veh(veh_num).x_path>=0);
        
        Sol.Veh(veh_num).F_onBeam = ...
            ((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).U) ...
                - Sol.Veh(veh_num).def_under - Calc.Veh(veh_num).h_path.*ind) ...
                .* (Train.Veh(veh_num).Susp.Prim.k' * ones_1_x_num_t) ...
            + ((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).V) ...
                - Sol.Veh(veh_num).vel_under - Sol.Veh(veh_num).def_under_p*Train.vel...
                - Calc.Veh(veh_num).hd_path.*ind) ...
                .* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t) ... 
            - (Sol.Veh(veh_num).acc_under + Sol.Veh(veh_num).def_under_pp*Train.vel^2 ...
                + 2*Sol.Veh(veh_num).vel_under_p*Train.vel) ...
                .* (Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t) ...
            + (Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t) .* Calc.Cte.grav;

        Sol.Veh(veh_num).F_onBeam_max = max(Sol.Veh(veh_num).F_onBeam(:));
        Sol.Veh(veh_num).F_onBeam_min = min(Sol.Veh(veh_num).F_onBeam(:));

        % Checking contact (while on the bridge). AUDIT R3 2026-07-17: the deck
        % start is L_Aw = L_Approach + max_TL*redux_factor (= 123 m for L60),
        % which is where B54 places the on-beam sleepers. The old expression
        % L_Approach + Position.x_0 used x_0 = max_TL + extra_L2, landing 6 m too
        % late (129 m) — so the bridge column of contact_log was mis-windowed.
        L1 = Calc.Profile.L_Aw;
        L2 = Calc.Profile.L_Aw + Calc.Profile.L_bridge;
        ind = and(Calc.Veh(veh_num).x_path>=L1,Calc.Veh(veh_num).x_path<=L2);
        Sol.Veh(veh_num).F_onBridge_max = max(max(Sol.Veh(veh_num).F_onBeam(ind)));
        Sol.Veh(veh_num).F_onBridge_min = min(min(Sol.Veh(veh_num).F_onBeam(ind)));
        Sol.Veh(veh_num).contactLost = Sol.Veh(veh_num).F_onBridge_max>0;

        % Checking contact over the WHOLE on-track path (audit 2026-07-17).
        % Sign convention: grav = -9.81 so compression is NEGATIVE and
        % F_onBeam > 0 is TENSION - a separation the bilateral solver cannot
        % represent. The old bridge-only window missed approach/exit uplift
        % (flat and void impacts mostly occur over plain track). These fields
        % are persisted per passage by A00 (data2save.contact_log) and
        % asserted on by the smoke tests.
        % On-track = [0, Calc.Profile.L], the same mask B50 uses to zero
        % off-profile forces. FIX 2026-07-22 (external audit r3, verified):
        % the old lower-bound-only mask (x_path >= 0) counted post-exit
        % samples (F identically 0 there) in the DENOMINATOR, diluting
        % tension_frac and making the sustained-tension gate read laxer
        % than pre-registered. Applies forward; saved contact_logs keep
        % their generation-time values (all passed with >3x margin).
        ind0 = (Calc.Veh(veh_num).x_path >= 0) & ...
               (Calc.Veh(veh_num).x_path <= Calc.Profile.L);
        Sol.Veh(veh_num).F_onTrack_max     = max(Sol.Veh(veh_num).F_onBeam(ind0));
        Sol.Veh(veh_num).contactLost_track = Sol.Veh(veh_num).F_onTrack_max > 0;
        Sol.Veh(veh_num).tension_frac      = ...
            sum(Sol.Veh(veh_num).F_onBeam(:) > 0 & ind0(:)) / max(sum(ind0(:)),1);

%         %-- Graphical Check --
%         if veh_num == 1
%             figure;
%         end % if veh_num == 1
%         subplot(ceil(Train.Veh(1).Tnum/2),2,veh_num);
%             hold on; box on;
%             plot(Calc.Veh(veh_num).x_path',Sol.Veh(veh_num).F_onBeam');
%             axis tight;
%             plot(L1*[1,1],ylim,'k--');
%             plot(L2*[1,1],ylim,'k--');
%             if Sol.Veh(veh_num).F_onBridge_max > 0 
%                 plot(xlim,[0,0],'r--');
%             end % if Sol.Veh(veh_num).F_onBridge_max > 0                 
%             plot([L1,L2],[1,1]*Sol.Veh(veh_num).F_onBridge_max,'k-');
%             plot([L1,L2],[1,1]*Sol.Veh(veh_num).F_onBridge_min,'k-');
%             xlim([L1,L2]);
%             ylim([Sol.Veh(veh_num).F_onBridge_min,Sol.Veh(veh_num).F_onBridge_max]);
%             plot(xlim,[1,1]*Train.Veh(veh_num).sta_loads(1),'r--'); %axis tight;
%             title(['Vehicle ',num2str(veh_num),' (',num2str(Sol.Veh(veh_num).contactLost),')']);

    end % for veh_num = 1:Train.Veh(1).Tnum

% **** Moving Force ****
elseif Calc.Options.VBI == 0

    % Force on Beam (Contact Force)
    for veh_num = 1:Train.Veh(1).Tnum

        Sol.Veh(veh_num).F_onBeam = ...
            ((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).U)) ...
                .* (Train.Veh(veh_num).Susp.Prim.k' * ones_1_x_num_t) ...
            + ((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).V)) ...
                .* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t) ...
            + (Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t) .* Calc.Cte.grav;
        
        Sol.Veh(veh_num).F_onBeam_max = max(Sol.Veh(veh_num).F_onBeam(:));
        Sol.Veh(veh_num).F_onBeam_min = min(Sol.Veh(veh_num).F_onBeam(:));

        % Checking contact (while on the bridge). AUDIT R4 2026-07-17: use
        % L_Aw (= L_Approach + max_TL*redux_factor) as the deck start, matching
        % the VBI==1 branch (this branch is unused by the campaign, fixed for
        % consistency so the bridge window is never 6 m off).
        L1 = Calc.Profile.L_Aw;
        L2 = Calc.Profile.L_Aw + Calc.Profile.L_bridge;
        ind = and(Calc.Veh(veh_num).x_path>=L1,Calc.Veh(veh_num).x_path<=L2);
        Sol.Veh(veh_num).contactLost = max(max(Sol.Veh(veh_num).F_onBeam.*ind))>0;

        % Whole on-track path (audit 2026-07-17; see the VBI==1 branch)
        % On-track = [0, Calc.Profile.L], the same mask B50 uses to zero
        % off-profile forces. FIX 2026-07-22 (external audit r3, verified):
        % the old lower-bound-only mask (x_path >= 0) counted post-exit
        % samples (F identically 0 there) in the DENOMINATOR, diluting
        % tension_frac and making the sustained-tension gate read laxer
        % than pre-registered. Applies forward; saved contact_logs keep
        % their generation-time values (all passed with >3x margin).
        ind0 = (Calc.Veh(veh_num).x_path >= 0) & ...
               (Calc.Veh(veh_num).x_path <= Calc.Profile.L);
        Sol.Veh(veh_num).F_onTrack_max     = max(Sol.Veh(veh_num).F_onBeam(ind0));
        Sol.Veh(veh_num).contactLost_track = Sol.Veh(veh_num).F_onTrack_max > 0;
        Sol.Veh(veh_num).tension_frac      = ...
            sum(Sol.Veh(veh_num).F_onBeam(:) > 0 & ind0(:)) / max(sum(ind0(:)),1);

    end % for veh_num = 1:Train.Veh(1).Tnum

end % if Calc.Options.VBI == 1

% ---- Checking contact ----
% Aggregates (audit 2026-07-17): bridge-window flag kept for backward
% compatibility; the *_track fields cover the whole on-track path. A00
% persists all of them per passage (data2save.contact_log) so invalid
% passages can be filtered at load time and the smoke tests can assert.
Sol.contactLost_track = double(max([Sol.Veh(:).contactLost_track]) > 0);
Sol.F_tension_max     = max([Sol.Veh(:).F_onTrack_max]);   % >0 = worst tension [N]
Sol.tension_frac_max  = max([Sol.Veh(:).tension_frac]);    % worst per-vehicle fraction
if max([Sol.Veh(:).contactLost]) > 0
    Sol.contactLost = 1;
    disp('There is no permanent contact between wheels and rail');
else
    Sol.contactLost = 0;
end % if max([Sol.Veh(:).contactLost]) > 0
if Sol.contactLost_track > 0 && ~Sol.contactLost
    disp('Contact lost over the approach/exit track (outside the bridge window)');
end

% % -- Graphical check of components --
% veh_num = 1;
% [Sol] = B17_CalcUat(Sol,Track,Calc,Train.Veh);
% figure;        
% subplot(3,1,1); plot(((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).U).* (Train.Veh(veh_num).Susp.Prim.k' * ones_1_x_num_t))'); axis tight;
% subplot(3,1,2); plot((-Sol.Veh(veh_num).def_under.* (Train.Veh(veh_num).Susp.Prim.k' * ones_1_x_num_t))'); axis tight;
% subplot(3,1,3); plot(((- Calc.Veh(veh_num).h_path).* (Train.Veh(veh_num).Susp.Prim.k' * ones_1_x_num_t))'); axis tight;
% figure;
% subplot(4,1,1); plot(((Train.Veh(veh_num).Wheels.N2w * Sol.Veh(veh_num).V).* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,2); plot((- Sol.Veh(veh_num).vel_under .* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,3); plot((- Sol.Veh(veh_num).def_under_p*Train.vel .* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,4); plot(((- Calc.Veh(veh_num).hd_path) .* (Train.Veh(veh_num).Susp.Prim.c' * ones_1_x_num_t))'); axis tight;
% figure;
% subplot(4,1,1); plot((-Sol.Veh(veh_num).acc_under.*(Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,2); plot((-Sol.Veh(veh_num).def_under_pp*Train.vel^2.*(Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,3); plot(((-2*Sol.Veh(veh_num).vel_under_p*Train.vel).*(Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t))'); axis tight;
% subplot(4,1,4); plot(((Train.Veh(veh_num).Wheels.m' * ones_1_x_num_t) .* Calc.Cte.grav)'); axis tight;

% ---- End of script ----