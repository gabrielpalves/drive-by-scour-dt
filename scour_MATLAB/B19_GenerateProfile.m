function [Calc] = B19_GenerateProfile(Calc)

% Calculates the irregularity profile for the rail

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structre with calculation variables. It should include at least:
%   .Profile.Type = Type of Profile 
%       0 = Smooth
%       1 = Generated from custom PSD im [m^3]
% ---- Output ----
% Calc = Additional fields in the structure:
%   ... see script ...
% -------------------------------------------------------------------------

% Amplitude intensity multiplier (rail-irregularity damage/EOV toggle;
% default 1 = unchanged). Captured BEFORE the Type-2 branch below replaces
% Calc.Profile with the loaded struct. Mirrors TTBI_2D/b19_generate_profile.py.
if isfield(Calc.Profile,'intensity')
    profile_intensity = Calc.Profile.intensity;
else
    profile_intensity = 1;
end

% Profile sampling rate
if isfield(Calc.Profile,'phase_seed') && ~isempty(Calc.Profile.phase_seed)
    % Per-STATE profile lock (2026-07-12 EOV design): the sampling grid must
    % be SPEED-INDEPENDENT so the seeded phase draw below reproduces the
    % identical physical realization on every passage of the state (the
    % default dx tracks Position.x = v*dt, which varies with train speed).
    % min_dx is fixed per case (B07), so nx and N are too.
    Calc.Profile.dx = Calc.Profile.min_dx;
else
    Calc.Profile.dx = min([min(abs(diff(Calc.Position.x))),Calc.Profile.min_dx]);
end

Calc.Profile.x = (Calc.Position.x(1):Calc.Profile.dx:Calc.Profile.L);
Calc.Profile.nx = length(Calc.Profile.x);

% ---- Type of Road Profile ----

% -- Smooth --
if Calc.Profile.Type == 0
    
    Calc.Profile.h = zeros(1,Calc.Profile.nx);
    Calc.Profile.text = 'Smooth';
    
% -- From PSD --
elseif Calc.Profile.Type == 1
    
    % [Max,Min] spatial frequency
    min_spaf = 1/Calc.Profile.max_WaveLength;
    max_spaf = 1/Calc.Profile.min_WaveLength;
    
    % Auxiliary variables
    N = 2^nextpow2(Calc.Profile.nx);     % Number of frequencies to calculate

    % PSD X values
    PSD_X = (1/Calc.Profile.dx)*(0:N/2)/N;            % Output
    
    % PSD Y values
    PSD_Y = Calc.Profile.PSD_Y_fun(PSD_X,Calc.Profile.inputs);
    
    % Application of spatial frequency limits
    PSD_Y(PSD_X<min_spaf) = 0;
    PSD_Y(PSD_X>max_spaf) = 0;

    % ---- Generating Output ----
    % Profile elevation. If a phase_seed is set (per-STATE profile lock,
    % 2026-07-12 EOV design review), the random phases come from a stream
    % seeded once per damage state -> every passage of that state rides the
    % SAME track realization (EN 13848-2: pass-to-pass geometry repeatability
    % <=0.5 mm @95%). The caller's RNG stream is saved/restored so all other
    % per-passage draws are unaffected.
    if isfield(Calc.Profile,'phase_seed') && ~isempty(Calc.Profile.phase_seed)
        rng_state_ = rng;
        rng(Calc.Profile.phase_seed);
        [Calc.Profile.h] = PSD2profile(PSD_Y,N,Calc.Profile.x);
        rng(rng_state_);
    else
        [Calc.Profile.h] = PSD2profile(PSD_Y,N,Calc.Profile.x);
    end
    % PSD
    Calc.Profile.PSD_X = PSD_X;
    Calc.Profile.PSD_Y = PSD_Y;
    
    %Salvar um novo perfil
    %save('Calc.ProfileData06-06.mat','Calc')   
  
% ---- Profile from a dataset ----
elseif Calc.Profile.Type == 2
    
    A = Calc;
    
    load ('Calc.ProfileData15_05.mat','Calc')
    
    B = Calc.Profile;
    Calc = A;
    Calc.Profile = B;
   
end % if Calc.Profile.Type

% ---- Rail-irregularity intensity toggle ----
% Applied AFTER generation/loading so it works for every profile type.
% Mirrors TTBI_2D/b19_generate_profile.py (h = h * intensity).
Calc.Profile.intensity = profile_intensity;
if profile_intensity ~= 1
    Calc.Profile.h = Calc.Profile.h * profile_intensity;
end

% ---- Per-passage profile jitter (2026-07-12 EOV design review) ----
% Additive white noise on top of the (per-state locked) realization:
% pass-to-pass metrological repeatability / minor wheel wander (EN 13848-2
% D1 longitudinal-level repeatability <=0.5 mm @95%). Drawn from the CURRENT
% per-passage RNG stream, so it DIFFERS passage to passage by construction.
if isfield(Calc.Profile,'jitter_sd_m') && Calc.Profile.jitter_sd_m > 0
    Calc.Profile.h = Calc.Profile.h + ...
        Calc.Profile.jitter_sd_m * randn(size(Calc.Profile.h));
end

%---- Plotting profile ----
if Calc.Plot.Profile_original == 1
    
    % Plotting profile
    figure;
    plot(Calc.Profile.x,Calc.Profile.h*1000);
    aux1 = get(gcf,'Position'); set(gcf,'Position',[aux1(1:2),500,250]);
    xlim([Calc.Profile.x(1),Calc.Profile.x(end)]);
    xlabel('Distance (m)'); ylabel('Profile Elevation (mm)');
    pause(0.25);
    
end % if Calc.Plot.Profile_original = 1

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function [y] = PSD2profile(PSD_Y,N,X)

    % Auxiliary variables
    dx = X(2) - X(1);
    nx = length(X);
    
    % Two-sided PSD
    PSD2 = PSD_Y/2;
    PSD2([1,end]) = 0;
    PSD2 = [PSD2,fliplr(PSD2(2:end-1))];
    
    % Amplitude
    Amplitude = sqrt(PSD2*N/dx);
    % Random Phase values
    phase = 2*pi*rand(1,N/2-1);
    phase = [0,phase,0,-fliplr(phase)];
    % Fourier transform coefficients
    FFT_y = Amplitude .* exp(1i*phase);
    
    % Inverse Fourier Transform
    iFFT_y = ifft(FFT_y,N);
    
    % Reducing results to desired output
    y = iFFT_y(1:nx);

end % function [y] = PSD2profile(PSD_Y,N,X)
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

end % function [Calc] = B19_GenerateProfile(Calc)

% ---- End of function ----