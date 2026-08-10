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

% Optional rail-domain study coordinate contract. Ordinary production runs
% never set this field and retain the historical FFT realization exactly.
% Clearance arms have different finite-domain lengths and global origins; an
% FFT synthesized independently on each live domain would change its frequency
% grid and phase realization. The study instead synthesizes on one fixed
% comparison coordinate and samples h(x-shift), keeping excitation identical
% over the retained bridge window.
common_coordinate_enabled = false;
profile_synthesis_x = Calc.Profile.x;
profile_query_x = Calc.Profile.x;
if isfield(Calc.Profile,'rail_domain_common_coordinate') && ...
        ~isempty(Calc.Profile.rail_domain_common_coordinate)
    common = Calc.Profile.rail_domain_common_coordinate;
    if ~isstruct(common) || ~isscalar(common) || ...
            ~isfield(common,'schema') || ...
            ~strcmp(common.schema,'rail-domain-common-coordinate-v1') || ...
            ~isfield(common,'x_bounds_m') || ...
            ~isnumeric(common.x_bounds_m) || ...
            ~isreal(common.x_bounds_m) || ...
            ~isequal(size(common.x_bounds_m),[1 2]) || ...
            any(~isfinite(common.x_bounds_m)) || ...
            common.x_bounds_m(2) <= common.x_bounds_m(1)
        error('B19_GenerateProfile:InvalidCommonCoordinate', ...
            ['rail_domain_common_coordinate must be a scalar struct with ' ...
             'schema rail-domain-common-coordinate-v1 and finite increasing ' ...
             '1-by-2 x_bounds_m.']);
    end
    translation_m = 0;
    if isfield(Calc.Profile,'rail_domain_translation_m')
        translation_m = Calc.Profile.rail_domain_translation_m;
    end
    profile_synthesis_x = ...
        common.x_bounds_m(1):Calc.Profile.dx:common.x_bounds_m(2);
    profile_query_x = Calc.Profile.x-translation_m;
    coverage_tolerance = 256*eps(max(abs(common.x_bounds_m))+1);
    if numel(profile_synthesis_x) < 2 || ...
            profile_query_x(1) < profile_synthesis_x(1)-coverage_tolerance || ...
            profile_query_x(end) > profile_synthesis_x(end)+coverage_tolerance
        error('B19_GenerateProfile:CommonCoordinateCoverage', ...
            ['The fixed synthesis coordinate [%.17g, %.17g] m does not ' ...
             'cover the translated live profile query [%.17g, %.17g] m.'], ...
            profile_synthesis_x(1),profile_synthesis_x(end), ...
            profile_query_x(1),profile_query_x(end));
    end
    common_coordinate_enabled = true;
end

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
    synthesis_nx = length(profile_synthesis_x);
    N = 2^nextpow2(synthesis_nx);        % Number of frequencies to calculate

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
    % seeded once per damage state -> every passage of that persistent
    % scenario rides the SAME physical track realization. This is a scenario
    % persistence assumption, not an inference from EN 13848-2 measurement
    % repeatability. The caller's RNG stream is saved/restored so all other
    % per-passage draws are unaffected.
    if isfield(Calc.Profile,'phase_seed') && ~isempty(Calc.Profile.phase_seed)
        rng_state_ = rng;
        rng(double(Calc.Profile.phase_seed), 'twister');
        profile_synthesis_h = PSD2profile( ...
            PSD_Y,N,profile_synthesis_x);
        rng(rng_state_);
    else
        profile_synthesis_h = PSD2profile( ...
            PSD_Y,N,profile_synthesis_x);
    end
    if common_coordinate_enabled
        Calc.Profile.h = interp1(profile_synthesis_x, ...
            profile_synthesis_h,profile_query_x,'linear');
        if any(~isfinite(Calc.Profile.h))
            error('B19_GenerateProfile:CommonCoordinateInterpolation', ...
                'Common-coordinate profile interpolation returned nonfinite data.');
        end
        Calc.Profile.rail_domain_common_coordinate.realized_bounds_m = ...
            profile_synthesis_x([1,end]);
        Calc.Profile.rail_domain_common_coordinate.translation_m = ...
            translation_m;
    else
        Calc.Profile.h = profile_synthesis_h;
    end
    % PSD
    Calc.Profile.PSD_X = PSD_X;
    Calc.Profile.PSD_Y = PSD_Y;
    
    %Salvar um novo perfil
    %save('Calc.ProfileData06-06.mat','Calc')   
  
% ---- Legacy stored synthetic realization (Type 2; no registered R11 rung) --
elseif Calc.Profile.Type == 2

    % AUDIT FIX 2026-07-17 (R4): load ONLY the elevation h and map it onto the
    % LIVE x-grid WITHOUT rescaling; PRESERVE all live geometry. History:
    %  - the original `Calc.Profile = B` reverted L_bridge to the stored 39.9 m
    %    (wrong crop / contact span; too short for L99.6);
    %  - R2 fixed geometry but WRAPPED the profile (mod), a step discontinuity
    %    that tripped wheel-rail contact loss on L99.6;
    %  - R3 replaced the wrap with a linear STRETCH — contact-safe, but it
    %    rescaled wavelengths differently per geometry (L60 -4.2%, L99.6 +7.9%,
    %    ~12.7% spectral gap), CONFOUNDING the scale-up comparison.
    % R4 uses a triangle-wave REFLECT fold: the identity on [x0, x0+span] and a
    % mirror beyond it. So the stored realization is used EXACTLY (no rescale)
    % wherever the track is within the stored 325.8 m — which covers the whole
    % deck+crop for BOTH L60 (track 312 m) and L99.6 (deck ends 222.6 m, crop
    % ends ~240 m) — making the fixed baseline the SAME physical irregularity in
    % every geometry. Only L99.6's tail (325.8-351.6 m, ~103 m PAST the deck
    % exit and outside the crop) is mirrored; it just needs to keep wheel-rail
    % contact, which smoke_geometry verifies at 70/80/90 km/h. The stored file
    % is a synthetic realization, so a mirrored out-of-crop tail is immaterial.
    stored_ = load('Calc.ProfileData15_05.mat','Calc');
    x_st_ = stored_.Calc.Profile.x(:)';
    h_st_ = stored_.Calc.Profile.h(:)';
    span_ = x_st_(end) - x_st_(1);
    xq_   = Calc.Profile.x;                          % live grid (set above)
    t_    = mod(xq_ - x_st_(1), 2*span_);            % fold period = 2*span
    xw_   = x_st_(1) + min(t_, 2*span_ - t_);        % identity on [0,span], mirror beyond
    Calc.Profile.h = interp1(x_st_, h_st_, xw_, 'linear');
    % PSD arrays from the stored realization are stale on the live grid; blank
    % them so nothing downstream reads a mismatched spectrum.
    Calc.Profile.PSD_X = [];
    Calc.Profile.PSD_Y = [];

end % if Calc.Profile.Type

% ---- Rail-irregularity intensity toggle ----
% Applied AFTER generation/loading so it works for every profile type.
% Mirrors TTBI_2D/b19_generate_profile.py (h = h * intensity).
Calc.Profile.intensity = profile_intensity;
if profile_intensity ~= 1
    Calc.Profile.h = Calc.Profile.h * profile_intensity;
end

% ---- Dormant compatibility hook: per-passage profile jitter ---------------
% Every campaign preset sets jitter_sd_m=0. EN 13848-2 repeatability is a
% measurement-system property and does not justify physical white roughness.
% If a future physical perturbation is introduced, it must be separately
% justified, band-limited and versioned rather than enabled through this hook.
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
