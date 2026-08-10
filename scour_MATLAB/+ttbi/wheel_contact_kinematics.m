function [total_acc, active] = wheel_contact_kinematics(sol_veh, calc_veh, vel)
%WHEEL_CONTACT_KINEMATICS Total wheelset acceleration + the solver's active mask.
%
% Single source of truth for two quantities that D01_DataProcessing (saved
% channel) and B66_ContactForce (reported contact force) must agree on. They
% previously carried a duplicated chain-rule expression and an inconsistent
% mask; both are defined here once.
%
% ---- Outputs ----
%   total_acc  Total vertical acceleration of the constrained wheelset,
%              following the moving contact coordinate, per wheel x time step:
%
%                  z_w,tt = u_tt + 2*v*u_xt + v^2*u_xx + h_tt
%
%              i.e. the Eulerian rail-field acceleration plus the two
%              convective terms plus profile inertia. This is an IDEALIZED
%              MODEL-PREDICTED quantity used as an axle-box response proxy: it
%              omits mounting dynamics, contact compliance, sensor bandwidth
%              and filtering, and most of the real unsprung assembly. Do not
%              call it a measured or measurable axle-box acceleration.
%
%   active     Logical mask of samples where the wheel sits on an actual rail
%              element. This is the SOLVER'S OWN condition: B65 assembles every
%              coupling and profile contribution inside `if ele_num > 0`, where
%              `ele_num` comes from `Calc.Veh(v).elexj`
%              (B65_DynamicCalcCoupledFaster.m:152/214/280).
%
% ---- Why elexj and not x_path >= 0 (audit 2026-08-09) ----
% B66 previously masked h_path/hd_path (and, as first written, hdd_path) with
% `x_path >= 0`. That is WEAKER than the solver's condition: it excludes
% pre-entry samples but NOT post-exit ones, so profile terms stayed active
% after the wheel left the rail domain while the solver had already dropped
% them. Reported force therefore disagreed with the force actually applied.
% Using `elexj > 0` makes the reconstruction solver-consistent.
%
% The mask change can affect a passage only if its sampled path contains
% post-exit points. The healthy L60/80 km/h diagnostic run on 2026-08-09 had no
% such points and therefore measured exactly zero mask-only delta. Other
% geometries/speeds are not inferred from that result; whole-track diagnostics
% and the contact gate still require qualification for the released source.

active = calc_veh.elexj > 0;

total_acc = sol_veh.acc_under ...
    + 2*vel*sol_veh.vel_under_p ...
    + vel^2*sol_veh.def_under_pp ...
    + calc_veh.hdd_path .* active;

end
