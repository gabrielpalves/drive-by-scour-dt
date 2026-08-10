function [Beam] = B24_BeamDamping(Beam)

% Calculates the Beam damping matrix 
%   Rayleigh damping is addopted 
%   1st and 2nd beam frequencies are taken as reference (excluding rigid modes)

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .Modal.w = Beam circular frequencies
%   .Modal.num_rigid_modes = Number of rigid modes in the model
%   .Damping.per = Damping percentage
%   .Damping.fixed_rayleigh_coefficients = optional [alpha beta] pair used
%       only by a predeclared model-form sensitivity. Production omits it and
%       recalibrates the coefficients from the current grid/state.
%   .Damping.fixed_rayleigh_source_id = required scalar text when fixed
%       coefficients are supplied
%   .Mesh.Mg = Mass matrix of beam
%   .Mesh.Kg = Stiffness matrix of beam
% % ---- Outputs ----
% Beam = Addition of fields to structure Beam:
%   .Cg = Global Damping matrix
% -------------------------------------------------------------------------

if Beam.Damping.per > 0
    
    % Reference frequencies
    ref_modes = (1:2) + Beam.Modal.num_rigid_modes;
    if numel(Beam.Modal.w) < ref_modes(end)
        error('B24:InsufficientElasticModes', ...
            'Rayleigh damping requires two elastic reference modes.');
    end
    wr = Beam.Modal.w(ref_modes);
    if ~isreal(wr) || any(~isfinite(wr)) || any(wr <= 0)
        error('B24:InvalidReferenceFrequencies', ...
            'Rayleigh reference frequencies must be finite positive reals.');
    end
    Beam.Damping.reference_mode_indices = ref_modes;
    Beam.Damping.reference_frequencies_rad_s = wr;
    
    % Rayleigh's coefficients 'alpha' and 'beta'. The ordinary production
    % path recalibrates from the current first two elastic modes. A fixed
    % pair is accepted only through the explicit sensitivity-only field.
    calibrated = (1/2)*[[1/wr(1) wr(1)];[1/wr(2) wr(2)]]\ ...
        ([1;1]*(Beam.Damping.per/100));
    if ~isreal(calibrated) || any(~isfinite(calibrated))
        error('B24:InvalidRayleighCoefficients', ...
            'Rayleigh damping coefficients are non-finite or complex.');
    end
    if isfield(Beam.Damping,'fixed_rayleigh_coefficients')
        aux1 = Beam.Damping.fixed_rayleigh_coefficients;
        if ~(isnumeric(aux1) && isreal(aux1) && numel(aux1) == 2 && ...
                all(isfinite(aux1(:))) && all(aux1(:) >= 0) && ...
                any(aux1(:) > 0))
            error('B24:InvalidFixedRayleighCoefficients', ...
                ['fixed_rayleigh_coefficients must be two finite, real, ' ...
                 'nonnegative values, not both zero.']);
        end
        if ~isfield(Beam.Damping,'fixed_rayleigh_source_id') || ...
                ~(ischar(Beam.Damping.fixed_rayleigh_source_id) || ...
                (isstring(Beam.Damping.fixed_rayleigh_source_id) && ...
                 isscalar(Beam.Damping.fixed_rayleigh_source_id))) || ...
                isempty(strtrim(char(Beam.Damping.fixed_rayleigh_source_id)))
            error('B24:MissingFixedRayleighSource', ...
                ['A nonempty scalar fixed_rayleigh_source_id is required ' ...
                 'with fixed coefficients.']);
        end
        aux1 = double(aux1(:));
        Beam.Damping.rayleigh_policy = 'fixed-coefficients-sensitivity-v1';
        Beam.Damping.rayleigh_coefficient_source_id = ...
            char(Beam.Damping.fixed_rayleigh_source_id);
    else
        aux1 = calibrated;
        Beam.Damping.rayleigh_policy = 'recalibrated-current-state-grid-v1';
        Beam.Damping.rayleigh_coefficient_source_id = ...
            'current-first-two-elastic-modes';
    end
    Beam.Damping.rayleigh_alpha = aux1(1);
    Beam.Damping.rayleigh_beta = aux1(2);
    Beam.Damping.recalibrated_rayleigh_alpha = calibrated(1);
    Beam.Damping.recalibrated_rayleigh_beta = calibrated(2);
    Beam.Damping.achieved_reference_damping_ratios = ...
        aux1(1)./(2*wr(:)) + aux1(2).*wr(:)/2;

    % Damping matrix
    Beam.Mesh.Cg = aux1(1)*Beam.Mesh.Mg + aux1(2)*Beam.Mesh.Kg;
    
%     % Graphical check
%     w = linspace(wr(1)/2,wr(2)*1.5,100);
%     figure; hold on;
%         plot(w,aux1(1)./(2*w)+aux1(2)*w/2);
%         plot(wr,Beam.Damping.per/100*[1,1],'r.');
%         xlabel('Circular frequency'); ylabel('Damping ratio');

else
    
    % No Damping case
    Beam.Mesh.Cg = Beam.Mesh.Kg*0;
    Beam.Damping.rayleigh_policy = 'zero-damping';
    Beam.Damping.rayleigh_coefficient_source_id = 'damping-target-zero';
    Beam.Damping.rayleigh_alpha = 0;
    Beam.Damping.rayleigh_beta = 0;
    Beam.Damping.recalibrated_rayleigh_alpha = 0;
    Beam.Damping.recalibrated_rayleigh_beta = 0;
    Beam.Damping.achieved_reference_damping_ratios = zeros(0,1);
    
end % if Beam.Damping.per > 0

% ---- End of script ----
