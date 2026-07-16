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

Beam.Prop.E = 35e9;     % Modulus of elasticity [N/m^2]  (Fernandes: 3.5e10)
Beam.Prop.I = 0.33;     % Second moment of area [m4]     (Fernandes: 0.33)
Beam.Damping.per = 3;   % Damping [%]                    (Fernandes: 3% all modes)
% ===== CRITICAL FIX 2026-07-15: was 9.6 — a 1000x MASS ERROR ============
% B43 sets Prop.A = 1 m^2 and B01/B03 build the mass matrix as rho_n * A_n,
% so this value IS the deck mass per unit length. Fernandes specifies
% rho = 9600 kg/m for this exact bridge (E=3.5e10, I=0.33, 2 x 20 m spans);
% the code carried 9.6, i.e. 1000x too light (a ~0.004 m^2 rod, not a deck).
% Verified against the model's own numbers (scratchpad/beam_freq.py, EB FE with
% the real k_v0 = 344e6 N/m support springs):
%     rho = 9.6   -> f1 = 130.6 Hz  (L40/2-span)  <- unphysical
%     rho = 9600  -> f1 =   4.13 Hz               <- textbook for a 20 m span
% ratio = sqrt(1000) = 31.6x exactly. A 131 Hz deck is effectively rigid to a
% vehicle whose car body sits at 1-3 Hz and bogies at ~10-30 Hz, so the whole
% vehicle-bridge INTERACTION was wrong: the deck contributed almost no dynamic
% signature. EVERY dataset generated before this fix is invalid (all are being
% regenerated anyway). Do not change without re-checking B09/B56 frequencies.
Beam.Prop.rho = 9600;   % Deck mass per unit length [kg/m] (x Prop.A = 1 m^2)

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