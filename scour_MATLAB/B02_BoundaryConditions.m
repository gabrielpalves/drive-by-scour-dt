function [Beam,Damage] = B02_BoundaryConditions(Beam,Damage)

% Definition of DOF with boundary conditions for different configurations

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam's variables, including at least:
%   .BC.loc = location of supports in X direction
%   .BC.vert_stiff = Vertical stiffnes value of each of the supports
%       -1 =    Fixed, no displacement
%       0 =     Free vertical displacement
%       value = Vertical stiffness of the support
%   .BC.rot_stiff = Rotational stiffnes value of each of the supports
%       -1 =    Fixed, no rotation
%       0 =     Free rotation
%       value = Rotational stiffness of the support
% % ---- Outputs ----
% Beam = Addition of fields to structure Beam:
%   .BC.supp_num = Number of supports
%   .BC.loc_ind = Node number closest to the support
%   .BC.DOF_fixed = Array of DOF with fixed boundary condition
%   .BC.DOF_with_values =  %Array of DOF that have some additional stiffness
%   .BC.DOF_stiff_values = Array of stiffness values to be added to Beam.BC.DOF_with_values
%   .BC.num_DOF_fixed = Number of fixed DOF
%   .BC.num_DOF_with_values = Number of values with additional stiffness
%   .num_rigid_modes = Number of rigid modes
% -------------------------------------------------------------------------

% Number of supports
Beam.BC.supp_num = length(Beam.BC.loc); %Defined in B07_OptionsProcessing

% Supports location index
if Beam.BC.supp_num > 0
    [~,Beam.BC.loc_ind] = ...
        min(abs(ones(Beam.BC.supp_num,1)*Beam.Mesh.Nodes.acum - ...
        Beam.BC.loc'*ones(1,Beam.Mesh.Nodes.Tnum)),[],2);
else
    Beam.BC.loc_ind = [];
end % if Beam.BC.supp_num > 0
Beam.BC.loc_ind = Beam.BC.loc_ind';

% Fixed vertical displacement DOF
Beam.BC.DOF_fixed = Beam.BC.loc_ind(Beam.BC.vert_stiff==-1)*2-1;

% Fixed rotational DOF
Beam.BC.DOF_fixed = [Beam.BC.DOF_fixed, ...
    Beam.BC.loc_ind(Beam.BC.rot_stiff==-1)*2];

% Sorting fixed DOF
Beam.BC.DOF_fixed = sort(Beam.BC.DOF_fixed);

%%
% -------------------------------------------------------------------------
% 1. Vertical Stiffness (Scour Damage)
% -------------------------------------------------------------------------
vert_with_values = [];
vert_stiff_values = [];
DOF_Original_value = 344e6;

% Safely initialize Damage fields if they do not exist
if ~isfield(Damage, 'scour_rates')
    Damage.scour_rates = zeros(1, Beam.BC.supp_num);
end

for i = 1:length(Beam.BC.vert_stiff)
    if Beam.BC.vert_stiff(i) > 0
        % MATLAB 1-based indexing: Vertical DOF = node * 2 - 1
        vert_with_values(end+1) = Beam.BC.loc_ind(i) * 2 - 1;
        
        % Apply scour to this specific support
        retained_stiffness = 1.0 - Damage.scour_rates(i);
        vert_stiff_values(end+1) = retained_stiffness * DOF_Original_value;
    end
end

% -------------------------------------------------------------------------
% 2. Rotational Stiffness (Bearing Damage)
% -------------------------------------------------------------------------
rot_with_values = [];
rot_stiff_values = [];

% Safely initialize bearing damage fields
if ~isfield(Damage, 'bearing_left')
    Damage.bearing_left = 0.0;
end
if ~isfield(Damage, 'bearing_right')
    Damage.bearing_right = 0.0;
end

for i = 1:length(Beam.BC.rot_stiff)
    if Beam.BC.rot_stiff(i) > 0
        % MATLAB 1-based indexing: Rotational DOF = node * 2
        rot_with_values(end+1) = Beam.BC.loc_ind(i) * 2;
        
        % Inject bearing damage ONLY at the abutments
        if i == 1
            rot_stiff_values(end+1) = Damage.bearing_left;
        elseif i == Beam.BC.supp_num
            rot_stiff_values(end+1) = Damage.bearing_right;
        else
            rot_stiff_values(end+1) = 0.0; % Intermediate piers free rotation
        end
    end
end

% -------------------------------------------------------------------------
% 3. Combine & Sort Arrays
% -------------------------------------------------------------------------
Beam.BC.DOF_with_values = [vert_with_values, rot_with_values];
combined_stiff_values = [vert_stiff_values, rot_stiff_values];

% Sorting DOFs and aligning the stiffness values to match
[Beam.BC.DOF_with_values, aux2] = sort(Beam.BC.DOF_with_values);
Beam.BC.DOF_stiff_values = combined_stiff_values(aux2);

% Auxiliary variables
Beam.BC.num_DOF_fixed = length(Beam.BC.DOF_fixed);
Beam.BC.num_DOF_with_values = length(Beam.BC.DOF_with_values);
Beam.Modal.num_rigid_modes = max([0, 2 - Beam.BC.num_DOF_fixed]);

% Value to use in the diagonal element when the DOF is fixed
Beam.BC.DOF_fixed_value = DOF_Original_value;

% ---- End of script ----