% Definition of Track
function Track = A02_Track()
% A FUNCTION (converted from the upstream TTB-2D script): returns the Track
% struct. Takes no arguments because the track properties are loaded from the
% reviewed TrackProp source below rather than sampled per passage.

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -- Mechanical properties --

% Retained for compatibility with the established Track struct schema. Source
% selection is no longer path-driven: the reviewed function is called directly.
Track.Load.path = '';

% Loading predefined list of track properties

% Possibility 1: With Ballast on Bridge
Track = TrackProp_Zhai_et_al_WithBallastOnBridge(Track);

% -- End of script --
end
