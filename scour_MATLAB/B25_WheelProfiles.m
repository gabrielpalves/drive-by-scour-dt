function [Calc] = B25_WheelProfiles(Calc,Veh,Damage)

% Calculates the profile under each wheel
%
% Optional 3rd argument Damage may carry per-passage WHEEL damage descriptors
% (Stage 3 EOVs; NOT labels) — literature-anchored, see
% docs/stage3_alldamage_spec.md:
%   Damage.oor_flats  : rows [veh, wheel, flat_length_m, depth_m, phase_rad]
%                       -> periodic haversine dip, period 2*pi*R. Depth is
%                       PRE-COMPUTED by the caller (fresh d=L^2/8R, run-in
%                       d=L^2/16R) so B25 stays law-agnostic.
%   Damage.oor_poly   : rows [veh, wheel, order_n, amp_m, phase_rad]
%                       -> continuous polygonization sinusoid
%                       amp*cos(n*x/R + phase).
%   Damage.oor_radius : wheel radius R [m] (default 0.46)
% Both are added BEFORE the derivatives so hd/hdd carry them automatically.
% Mirrors TTBI_2D/b25_wheel_profiles.py.
if nargin < 3, Damage = struct(); end
has_flats = isfield(Damage,'oor_flats') && ~isempty(Damage.oor_flats);
has_poly  = isfield(Damage,'oor_poly')  && ~isempty(Damage.oor_poly);
oor_R = 0.46;
if isfield(Damage,'oor_radius') && ~isempty(Damage.oor_radius)
    oor_R = Damage.oor_radius;
end

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structure with Calculation variables. It should include at least:
%   ... see script ...
% Veh = Indexed Structre with Vehicle variables. It should include at least:
%   ... see script ...
% ---- Output ----
% Calc = Additional fields in the structure:
%   .Veh(i).x_path = Matrix with x coordinates for each wheel in time
%   .Veh(i).h_path = Matrix with irregularity elevation for each wheel in time
%   .Veh(i).hd_path = Matrix with 1st time-derivative of irregularity
%   .Veh(i).hdd_path = Matrix with 2nd time-derivative of irregularity
% -------------------------------------------------------------------------

for veh_num = 1:Veh(1).Tnum
    
    % Initialize variables
    Calc.Veh(veh_num).x_path = zeros(Veh(veh_num).Wheels.num,Calc.Solver.num_t);
    Calc.Veh(veh_num).h_path = Calc.Veh(veh_num).x_path;

    % Profile for each wheel
    %interp_method = 'linear';   % Matlab's default (leads to high frequency artifacts)
    interp_method = 'pchip';    % Piecewise cubic interpolation
    for wheel = 1:Veh(veh_num).Wheels.num
        Calc.Veh(veh_num).x_path(wheel,:) = ...
            Calc.Position.x(Calc.Time.t_0_ind:Calc.Time.t_end_ind) ... 
            - Veh(veh_num).Ax_dist(wheel) - Veh(veh_num).First_wheel_dist;
        Calc.Veh(veh_num).h_path(wheel,:) = ...
            interp1(Calc.Profile.x,Calc.Profile.h,...
            Calc.Veh(veh_num).x_path(wheel,:),interp_method);
    end % for wheel = 1:Veh(veh_num).Wheels.num

    % Removing NaN values
    Calc.Veh(veh_num).h_path(Calc.Veh(veh_num).x_path<Calc.Profile.x(1)) = ...
        Calc.Profile.h(1);
    Calc.Veh(veh_num).h_path(Calc.Veh(veh_num).x_path>Calc.Profile.x(end)) = ...
        Calc.Profile.h(end);

    % ---- Wheel flats + polygonization (per-passage EOVs; NOT labels) ----
    % Added BEFORE the derivatives so hd/hdd carry them automatically.
    if has_flats
        rows = find(Damage.oor_flats(:,1) == veh_num)';
        for r = rows
            wheel = Damage.oor_flats(r,2);
            if wheel < 1 || wheel > Veh(veh_num).Wheels.num, continue; end
            lf    = Damage.oor_flats(r,3);      % flat length [m]
            depth = Damage.oor_flats(r,4);      % pre-computed depth [m]
            phase = Damage.oor_flats(r,5);      % phase [rad]
            circ  = 2*pi*oor_R;                 % wheel circumference [m]
            s = mod(Calc.Veh(veh_num).x_path(wheel,:) + phase/(2*pi)*circ, circ);
            dip = -0.5*depth*(1 - cos(2*pi*s/lf)) .* (s < lf);
            Calc.Veh(veh_num).h_path(wheel,:) = ...
                Calc.Veh(veh_num).h_path(wheel,:) + dip;
        end
    end
    if has_poly
        rows = find(Damage.oor_poly(:,1) == veh_num)';
        for r = rows
            wheel = Damage.oor_poly(r,2);
            if wheel < 1 || wheel > Veh(veh_num).Wheels.num, continue; end
            n_ord = Damage.oor_poly(r,3);       % harmonic order
            amp   = Damage.oor_poly(r,4);       % radius deviation [m]
            phase = Damage.oor_poly(r,5);       % phase [rad]
            Calc.Veh(veh_num).h_path(wheel,:) = ...
                Calc.Veh(veh_num).h_path(wheel,:) + ...
                amp*cos(n_ord*Calc.Veh(veh_num).x_path(wheel,:)/oor_R + phase);
        end
    end

    % 1st derivative in time
    Calc.Veh(veh_num).hd_path = ...
        diff(Calc.Veh(veh_num).h_path,1,2)/Calc.Solver.dt;
    Calc.Veh(veh_num).hd_path = ...
        [Calc.Veh(veh_num).hd_path(:,1),Calc.Veh(veh_num).hd_path];

    % 2nd derivative in time
    Calc.Veh(veh_num).hdd_path = ...
        diff(Calc.Veh(veh_num).hd_path,1,2)/Calc.Solver.dt;
    Calc.Veh(veh_num).hdd_path = ...
        [Calc.Veh(veh_num).hdd_path(:,1),Calc.Veh(veh_num).hdd_path];

    % First point of profile for of each wheel at level zero
    Calc.Veh(veh_num).h_path = Calc.Veh(veh_num).h_path - ...
        Calc.Veh(veh_num).h_path(:,1)*ones(1,size(Calc.Veh(veh_num).h_path,2));

    % % Graphical Check
    % figure; subplot(3,1,1); hold on; box on;
    %     plot(Calc.Veh(veh_num).x_path',Calc.Veh(veh_num).h_path'); ylabel('Profile');
    %     plot(Calc.Veh(veh_num).x_path(:,[1,end])',Calc.Veh(veh_num).h_path(:,[1,end])','.','MarkerSize',20);
    % subplot(3,1,2); hold on; box on;
    %     plot(Calc.Veh(veh_num).x_path',Calc.Veh(veh_num).hd_path'); ylabel('1st Derivative');
    %     plot(Calc.Veh(veh_num).x_path(:,[1,end])',Calc.Veh(veh_num).hd_path(:,[1,end])','.','MarkerSize',20);
    % subplot(3,1,3); hold on; box on;
    %     plot(Calc.Veh(veh_num).x_path',Calc.Veh(veh_num).hdd_path'); ylabel('2nd Derivative');
    %     plot(Calc.Veh(veh_num).x_path(:,[1,end])',Calc.Veh(veh_num).hdd_path(:,[1,end])','.','MarkerSize',20);

end % for veh_num = 1:Veh(1).Tnum

% ---- End of script ----