function [Beam] = B09_BeamFrq(Beam,Calc)

% Calculates the beam frequencies given its system matrices

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% ---- Input ----
% Beam = Structure with Beam's variables, including at least:
%   .Kg = Global Stiffness matrix
%   .Mg = Global Mass matrix
%   .BC.num_DOF_fixed = Number of fixed DOF
%   .Prop.L = Beam length
% Calc = Structre with calculation variables. It should include at least:
%   .Options.calc_beam_frq = If value = 1, beam's natural frequencies are calculated
%   .Options.calc_beam_modes = If value = 1, beam's modes of vibration are calculated
% ---- Output ----
% Beam = Addition of fields to structure Beam:
%   .w = Circular frequencies of the beam
%   .f = Natural frequencies of the beam in [Hz]
%   .modes = Modes of vibration of in columns
% -------------------------------------------------------------------------

% Only natural frequencies calculation
if and(Calc.Options.calc_beam_frq == 1,Calc.Options.calc_beam_modes == 0)

    lambda = eig(full(Beam.Mesh.Kg),full(Beam.Mesh.Mg));
    [lambda,~] = local_sorted_eigenvalues( ...
        lambda, Beam.Modal.num_rigid_modes);
    Beam.Modal.w = sqrt(lambda);
    Beam.Modal.f = Beam.Modal.w/(2*pi);

    % Removing values associated to BC
    Beam.Modal.w = Beam.Modal.w(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.f = Beam.Modal.f(Beam.BC.num_DOF_fixed+1:end);

% Natural frequencies and Modes of vibration calculation
elseif and(Calc.Options.calc_beam_frq == 1,Calc.Options.calc_beam_modes == 1)
    
    [V,lambda] = eig(full(Beam.Mesh.Kg),full(Beam.Mesh.Mg));
    [lambda,k] = local_sorted_eigenvalues( ...
        diag(lambda), Beam.Modal.num_rigid_modes);
    V = V(:,k); 
    
    % Normaliztion of eigenvectors
    Factor = diag(V'*Beam.Mesh.Mg*V); 
    Beam.Modal.modes = V/(sqrt(diag(Factor)));
    
    % EigenValues to Natural frequencies
    Beam.Modal.w = sqrt(lambda);
    Beam.Modal.f = Beam.Modal.w/(2*pi);
    
    % Removing values associated to BC
    Beam.Modal.w = Beam.Modal.w(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.f = Beam.Modal.f(Beam.BC.num_DOF_fixed+1:end);
    Beam.Modal.modes(:,1:Beam.BC.num_DOF_fixed) = [];
    
end % if and(Calc.Options.calc_beam_frq == 1,Calc.Options.calc_beam_frq == 0)

% ------------------------------ Plotting ---------------------------------
    
% -- Plotting of calculated Natural frequencies --
if Calc.Plot.P1_Beam_frq == 1
    figure; plot((1:length(Beam.Modal.f)),Beam.Modal.f,'.'); axis tight;
    %figure; semilogy((1:length(Beam.Modal.f)),Beam.Modal.f,'.'); axis tight;
    xlabel('Mode number'); ylabel('Frequency (Hz)');
    title(['Beam Only (1st frq: ',num2str(round(Beam.Modal.f(1),2)),' Hz;',...
        blanks(1),'Last frq: ',num2str(round(Beam.Modal.f(end),2)),' Hz)']);
    pause(0.25);
end % if Calc.Plot.P1_Beam_frq == 1

% -- Plotting Mode shapes --
if Calc.Plot.P2_Beam_modes >= 1
    if Calc.Plot.P2_Beam_modes > 0
        aux1 = ceil(Calc.Plot.P2_Beam_modes/2);
        figure; 
        for k = 1:Calc.Plot.P2_Beam_modes
            aux2 = max(abs(Beam.Modal.modes(1:2:end,k)));
            subplot(aux1,2,k)
            Xdata = Beam.Mesh.Nodes.acum;
            plot(Xdata,Beam.Modal.modes(1:2:end,k)/aux2);
            xlim([0,Beam.Prop.L]);
            title(['Mode ',num2str(k),' (',num2str(round(Beam.Modal.f(k),3)),' Hz)']);
        end % for k = 1:Calc.Plot.P2_Beam_modes
        pause(0.25);
    end % if Calc.Plot.P2_Beam_modes > 0
end % if Calc.Plot.P2_Beam_modes == 1

% ---- End of function ----
end

function [lambda, order] = local_sorted_eigenvalues( ...
        raw_lambda, expected_rigid_modes)
% Generalized EIG does not promise eigenvalue order.  Rayleigh damping and
% fixed-DOF removal both require ascending modal order, so make it explicit
% and reject anything beyond round-off departure from a real PSD spectrum.
if ~isscalar(expected_rigid_modes) || ...
        expected_rigid_modes ~= floor(expected_rigid_modes) || ...
        expected_rigid_modes < 0 || expected_rigid_modes > 2
    error('B09:InvalidRigidModeCount', ...
        'Expected beam rigid-mode count must be an integer in [0,2].');
end
if any(~isfinite(raw_lambda))
    error('B09:NonfiniteEigenvalue', ...
        'Generalized beam eigenproblem returned a non-finite eigenvalue.');
end
scale = max([1; abs(raw_lambda(:))]);
% A global relative tolerance such as 1e-10*max(lambda) is far too loose for
% FE spectra spanning many decades.  Bound round-off by 128 ULPs at the
% largest eigenvalue, then require exactly the independently derived number
% of near-zero rigid modes.
tol = 128 * eps(max(scale));
if any(abs(imag(raw_lambda(:))) > tol)
    error('B09:ComplexEigenvalue', ...
        'Generalized beam eigenproblem returned a materially complex value.');
end
lambda = real(raw_lambda(:));
if any(lambda < -tol)
    error('B09:NegativeEigenvalue', ...
        'Generalized beam eigenproblem returned a materially negative value.');
end
[lambda, order] = sort(lambda, 'ascend');
near_zero = abs(lambda) <= tol;
if sum(near_zero) ~= expected_rigid_modes
    error('B09:RigidModeCountMismatch', ...
        ['Eigenproblem has %d near-zero modes within %.6g, but boundary ' ...
         'conditions predict %d.'], ...
        sum(near_zero), tol, expected_rigid_modes);
end
lambda(near_zero) = 0;
end
