function Track = TrackProp_Zhai_et_al_WithBallastOnBridge(Track)
%TRACKPROP_ZHAI_ET_AL_WITHBALLASTONBRIDGE Return reviewed track properties.
%
% The upstream file was a workspace-mutating script. Making the Track input
% and output explicit preserves every assignment while removing text-based
% dispatch and making MATLAB dependency analysis complete.

% Except for the explicitly labeled rail modal-damping target below, the
% scalar nominal-value lineage is:
% W.M. Zhai, K.Y. Wang, J.H. Lin, Modelling and experiment of railway 
%   ballast vibrations, Journal of Sound and Vibration, Volume 270, 
%   Issues 4–5, 2004, Pages 673-683, ISSN 0022-460X,
%   https://doi.org/10.1016/S0022-460X(03)00186-X.
%
% Zhai's Table 1 states its quantities per rail seat.  This inherited planar
% property set doubles the rail inertia/mass and half-sleeper mass below, but
% retains the tabulated pad, ballast, and sub-ballast values.  That mixed
% one-seat/two-rail scaling convention is preserved pending an upstream
% benchmark or a prospective sensitivity; do not silently double the latter
% parameters.  See docs/numerical_vv_protocol.md.
% Zhai's tabulated set also uses 0.545 m rail-support spacing, whereas this
% generator retains the source-reported values at 0.600 m.  Because the source's
% ballast-mass and ballast/subgrade-stiffness expressions (Mb, Kb, Kf) depend
% on spacing, this is a scope-transferred hybrid parameterization, not a
% spacing-consistent re-derivation.  Preserve
% it only as the declared baseline until the prospective spacing/scaling
% sensitivity or an upstream benchmark is completed.
% This function does not reproduce Zhai's complete five-parameter ballast
% topology: the inherited TTB-2D assembly has no adjacent-ballast Kw/Cw shear
% branch, and its on-bridge variant condenses Mb onto the deck DOFs instead of
% retaining an independent ballast DOF.  Those are inherited model-form
% choices requiring a prospective topology sensitivity; value provenance is
% not physical validation of that condensation.

% ---- Rail ----
Track.Rail.Prop.E = 2.059e11;       % Young's modulus [N/m^2]
Track.Rail.Prop.I = (3.217e-5)*2;   % Second moment of area [m^4]
Track.Rail.Prop.rho = 60.64*2;      % Mass per unit length [kg/m]
Track.Rail.Damping.per = 0.1;       % Target [%], inherited author-chosen;
                                    % not reported by Zhai et al. (2004)

% ---- Pad ----
Track.Pad.Prop.k = 6.5e7;           % Vertical stiffness [N/m]
Track.Pad.Prop.c = 7.5e4;           % Vertical damping [Ns/m]

% ---- Sleeper ----
Track.Sleeper.spacing = 0.6;        % Spacing [m]
Track.Sleeper.Prop.m = 125.5*2;     % Mass [kg]

% ---- Ballast ----
Track.Ballast.Prop.m = 531.4;       % Mass [kg]
Track.Ballast.Prop.k = 137.75e6;    % Stiffness [N/m]
Track.Ballast.Prop.c = 5.88e4;      % Damping [Ns/m]

% ---- SubBallast ----
Track.SubBallast.Prop.k = 77.5e6;   % Stiffness [N/m]
Track.SubBallast.Prop.c = 3.115e4;  % Damping [Ns/m]

% ---- Ballast on Bridge ----
Track.BallastOnBeam.Prop.m = Track.Ballast.Prop.m;
Track.BallastOnBeam.Prop.k = Track.Ballast.Prop.k;
Track.BallastOnBeam.Prop.c = Track.Ballast.Prop.c;

% % ---- Pad under Sleeper on Bridge ----
% Track.PadUnderSleeperOnBeam.Prop.k = 120e6;     % Stiffness [N/m]
% Track.PadUnderSleeperOnBeam.Prop.c = 60e4;      % Damping [Ns/m]

% ---- End of reviewed property function ----
end
