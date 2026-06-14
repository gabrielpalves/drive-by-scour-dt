function Beam = A03_Bridge(Beam)
% Bridge model definition

% Not a function.
% Defines the content of the variable Beam.

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

if nargin < 1 || isempty(Beam) || ~isstruct(Beam)
    Beam = struct();
end

% Geometry — respect values pre-set by the caller (A00 config block), else
% default to the ABLATION / CHAMPION bridge: 40 m, 2 spans (3 supports). The
% drive-by classifier was trained on this geometry, so the held-out DT dataset
% must use it too. For the multi-damage extension, set L = 100, num_spans = 4
% (5 supports) in A00 before calling A03_Bridge.
if ~isfield(Beam, 'Prop') || ~isfield(Beam.Prop, 'L') || isempty(Beam.Prop.L)
    Beam.Prop.L = 40;          % Sum of all spans [m]
end
if ~isfield(Beam.Prop, 'num_spans') || isempty(Beam.Prop.num_spans)
    Beam.Prop.num_spans = 2;   % 2 spans = 3 supports total
end

Beam.Prop.E = 35e9;     % Modulus of elasticity [N/m^2]
Beam.Prop.I = 0.33;     % Second moment of area [m4]
Beam.Damping.per = 3;   % Damping [%]
Beam.Prop.rho = 9.6;  % Mass per unit length [kg/m]

% Boundary conditions
Beam.BC.text = 'UD';

%Beam.BC.text = 'SP';    % Simply supported
%Beam.BC.text = 'FF';    % Fixed-fixed
%Beam.BC.text = 'UD';     % User-defined

% % Boundary conditions - Alternative definitions
%Beam.BC.loc = [0,Beam.Prop.L/2,Beam.Prop.L];  % Location of supports
%Beam.BC.vert_stiff = [-1,-1,-1];              % Vertical stiffness of supports (-1 for perfectly stiff)
%Beam.BC.rot_stiff = [0,0,0];                  % Rotational stiffness of supports (-1 for perfectly stiff)
%Beam.BC.text = 'User-defined';
%Beam.BC.text_long = '2-span bridge';

% ---- End of script ----
end