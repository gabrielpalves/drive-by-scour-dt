function [Sol, Calc, Train, Beam, Track, Model] = B00_Calculations(Calc, Train, Track, Beam, Damage)
% Core calculations of TTB-2D simulation.
% Basically this script does the following:
%   - Inputs processing and auxiliary variables generation
%   - Model generation (system matrices and load sequence)
%   - Numerical integration of solution
%   - Generation of additional results based on the solution

% Not a function.
% Uses and modifies the variables: Train, Track, Beam and Calc.
%   Generates the simulation results and saves them in variable Sol.

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -- Model geometry --
[Calc,Train,Beam] = B43_ModelGeometry(Calc,Train,Track,Beam);

% -- Options Processing --
[Calc,Train,Track,Beam] = B07_OptionsProcessing(Calc,Train,Track,Beam);

% ---- Beam Model ----
% Elements and coordinates
[Beam] = B01_ElementsAndCoordinates(Beam,Calc);

% Legacy single-element E modifier (kept for backward compatibility;
% inactive unless the caller sets n_mod/E_mod explicitly).
if isfield(Beam.Prop,'n_mod') && isfield(Beam.Prop,'E_mod')
    Beam.Prop.E_n(Beam.Prop.n_mod) = Beam.Prop.E*Beam.Prop.E_mod;
end

% Crack damage on the BRIDGE beam only: DAMAGED-ELEMENT EI reduction (uniform
% I drop over the affected element(s), cf. Fernandes et al. 2025). NOT the
% Sinha/Friswell/Edwards (2002) model - that one tapers EI linearly over an
% effective length around the crack and is parametrized by crack DEPTH
% (mis-citation corrected in the 2026-07-17 audit).
% Exact mirror of TTBI_2D/damage_config.py::apply_cracks. No-op when the
% Damage struct carries no cracks.
[Beam] = local_apply_cracks(Beam,Damage);

% Boundary conditions
[Beam,Damage] = B02_BoundaryConditions(Beam,Damage);
% Beam system matrices (Mass and Stiffness)
[Beam] = B03_BeamMatrices(Beam);
% Beam frequencies 
[Beam] = B09_BeamFrq(Beam,Calc);
% Beam Damping Matrix
[Beam] = B24_BeamDamping(Beam);

% ---- Track Model ----
% Definition of auxiliary variables
[Track] = B51_RailVariables(Track,Calc);
% Elements and coordinates
[Track.Rail] = B01_ElementsAndCoordinates(Track.Rail);
% Boundary conditions 
[Track.Rail] = B02_BoundaryConditions(Track.Rail,Damage);
% Rail system matrices (Mass and stiffness)
[Track.Rail] = B03_BeamMatrices(Track.Rail);
% Rail frequencies
[Track.Rail] = B09_BeamFrq(Track.Rail,Calc);
% Reference frequencies for damping definition
Track.Rail.Modal.w = [zeros(Track.Rail.Modal.num_rigid_modes,1),Beam.Modal.w(1:2)];
% Beam Damping Matrix
[Track.Rail] = B24_BeamDamping(Track.Rail);

% ---- Complete Model ----
% System matrices (Damage carries the per-passage TRACK-LAYER descriptors —
% Damage.track: ballast patches / hanging sleepers / pad aging+failures)
disp('Building model system matrices ...')
[Model] = B54_ModelMatrices(Beam,Track,Calc,Damage);
% Model boundary conditions
[Model] = B55_ModelBC(Model,Beam,Track);
% Modal analysis of Model
[Model] = B56_ModelFrq(Model,Calc);
% B57_PlotModelModes(Model,Calc);
fprintf('\b'); disp(' DONE');

% ---- Vehicle Model ----
% Vehicles system matrices
[Train.Veh] = B18_TrainVehEq(Train.Veh);
% Vehicle frequencies
[Train.Veh] = B08_VehFreq(Train.Veh,Calc);
% Vehicle position and Solver time array
[Calc] = B10_EndTime(Calc);
[Calc] = B11_TimeSpaceDiscretization(Calc);
% Vehicle static loads
[Train.Veh] = B47_VehStaticLoads(Train.Veh,Calc);

% ---- Irregularity profile ----
% Rail-irregularity amplitude toggle (a damage/EOV feature) -> profile module.
% Mirrors TTBI_2D/b00_calculations.py. Takes precedence over any static value
% set in A04 (per-passage draws from A00 land here via the Damage struct).
if isfield(Damage,'profile_intensity')
    Calc.Profile.intensity = Damage.profile_intensity;
end
% Generation
[Calc] = B19_GenerateProfile(Calc);
% Assigning profile to each wheel (Damage may carry wheel OOR/flat draws)
[Calc] = B25_WheelProfiles(Calc,Train.Veh,Damage);

% -- Model visualization --
% B67_PlotModelVisualization(Calc,Train.Veh,Track,Model);

% ---- Coupled Equations Solver (COUP) ----
% Element number of vertical forces in time
[Calc] = B50_ElementNumOfForce(Track.Rail,Calc);
% Initial coupled system's static deformation
[Sol] = B64_Coupled_InitialStatic(Train.Veh,Model,Calc,Track);
tic; disp('Performing Dynamic Calculations (Coupled System) ...');
[Sol] = B65_DynamicCalcCoupledFaster(Train.Veh,Model,Calc,Track,Sol,Damage);
disp(['Calculation time: ',num2str(round(toc,2)),'s']);

% ---- Additional results from solution ----
% Contact Force
[Sol] = B66_ContactForce(Sol,Track,Calc,Train,Damage);
% Beam deformation
[Sol] = B49_BeamDeformation(Sol,Model,Beam,Calc,Train,1);
% Static beam deformation
[Sol] = B49_BeamDeformation(Sol,Model,Beam,Calc,Train,0);
% Beam Bending Moment
[Sol] = B31_BeamBM(Sol,Model,Beam,Calc,1);
% Static Beam Bending Moment
[Sol] = B31_BeamBM(Sol,Model,Beam,Calc,0);
% Beam Shear Force
[Sol] = B33_BeamShear(Sol,Model,Beam,Calc,1);
% Static Beam Shear Force
[Sol] = B33_BeamShear(Sol,Model,Beam,Calc,0);
% Beam Acceleration
[Sol] = B53_BeamAcceleration(Sol,Model,Beam,Calc);
% Results at selected Beam sections
[Sol] = B58_ResultsBeamSections(Sol,Beam,Calc);

disp('All calculations finished sucessfully');
% C03_TTB_2D_Plots;

% ---- End of script ----

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function [Beam] = local_apply_cracks(Beam,Damage)
% Crack = DAMAGED-ELEMENT EI reduction: every BRIDGE element whose midpoint
% lies within [loc-lc, loc+lc] has its second moment of area I_n scaled by
% the SAME (1 - intensity) - a uniform block, with lc <= 0 affecting only
% the single element containing `loc`. This is the classical damaged-element
% benchmark used by Fernandes et al. (2025) (intensities 0.14/0.22 on a
% ~30 cm element). It is NOT the Sinha/Friswell/Edwards (2002) model, which
% tapers EI piecewise-linearly over ~1.5x beam depth each side of the crack
% and is parametrized by crack depth (audit 2026-07-17: former mis-citation).
% Exact mirror of TTBI_2D/damage_config.py::apply_cracks.

if ~isfield(Damage,'crack_locs') || isempty(Damage.crack_locs)
    return
end
locs = Damage.crack_locs(:)';
intens = [];
if isfield(Damage,'crack_intensity'), intens = Damage.crack_intensity(:)'; end
if numel(intens) == 1
    intens = repmat(intens, 1, numel(locs));
end
lc = 0;
if isfield(Damage,'crack_lc'), lc = Damage.crack_lc; end

acum = Beam.Mesh.Nodes.acum(:)';
mid = (acum(1:end-1) + acum(2:end)) / 2;        % element midpoints [m]
n_ele = numel(mid);

for k = 1:numel(locs)
    if k <= numel(intens), intensity = intens(k); else, intensity = 0; end
    if intensity == 0
        continue
    end
    if lc > 0
        sel = find(abs(mid - locs(k)) <= lc);
    else
        [~,sel] = min(abs(mid - locs(k)));
    end
    sel = sel(sel >= 1 & sel <= n_ele);
    if isempty(sel)
        fprintf('B00/apply_cracks: crack at %g m hit no element - ignored.\n', locs(k));
        continue
    end
    Beam.Prop.I_n(sel) = Beam.Prop.I_n(sel) * (1 - intensity);
end
