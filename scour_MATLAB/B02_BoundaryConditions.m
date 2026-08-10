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
if ~isnumeric(Beam.BC.loc) || ~isreal(Beam.BC.loc) || ...
        (~isempty(Beam.BC.loc) && ~isvector(Beam.BC.loc)) || ...
        any(~isfinite(Beam.BC.loc(:)))
    error('B02:InvalidSupportLocations', ...
        'Beam.BC.loc must be a finite real numeric vector.');
end
if ~isnumeric(Beam.BC.vert_stiff) || ~isreal(Beam.BC.vert_stiff) || ...
        (~isempty(Beam.BC.vert_stiff) && ~isvector(Beam.BC.vert_stiff)) || ...
        any(~isfinite(Beam.BC.vert_stiff(:))) || ...
        any(Beam.BC.vert_stiff(:) < 0 & Beam.BC.vert_stiff(:) ~= -1) || ...
        numel(Beam.BC.vert_stiff) ~= Beam.BC.supp_num
    error('B02:InvalidVerticalBC', ...
        'Beam.BC.vert_stiff must have one real numeric entry per support.');
end
if ~isnumeric(Beam.BC.rot_stiff) || ~isreal(Beam.BC.rot_stiff) || ...
        (~isempty(Beam.BC.rot_stiff) && ~isvector(Beam.BC.rot_stiff)) || ...
        any(~isfinite(Beam.BC.rot_stiff(:))) || ...
        any(Beam.BC.rot_stiff(:) < 0 & Beam.BC.rot_stiff(:) ~= -1) || ...
        numel(Beam.BC.rot_stiff) ~= Beam.BC.supp_num
    error('B02:InvalidRotationalBC', ...
        'Beam.BC.rot_stiff must have one real numeric entry per support.');
end
% Downstream DOF lists are row vectors. Normalize accepted vector inputs so
% column-shaped user data cannot trigger implicit expansion or shape drift.
Beam.BC.loc = Beam.BC.loc(:)';
Beam.BC.vert_stiff = Beam.BC.vert_stiff(:)';
Beam.BC.rot_stiff = Beam.BC.rot_stiff(:)';

% Supports location index
if Beam.BC.supp_num > 0
    [~,Beam.BC.loc_ind] = ...
        min(abs(ones(Beam.BC.supp_num,1)*Beam.Mesh.Nodes.acum - ...
        Beam.BC.loc'*ones(1,Beam.Mesh.Nodes.Tnum)),[],2);
else
    Beam.BC.loc_ind = [];
end % if Beam.BC.supp_num > 0
Beam.BC.loc_ind = Beam.BC.loc_ind';
Beam.BC.loc_realized = Beam.Mesh.Nodes.acum(Beam.BC.loc_ind);
Beam.BC.loc_offset = Beam.BC.loc_realized - Beam.BC.loc;
alignment_scale = max([abs(Beam.Mesh.Nodes.acum(:)); ...
    abs(Beam.BC.loc(:)); 1]);
% B01 generates coordinates by cumulative summation.  Scale the floating-
% point tolerance with the number of summed elements so an exactly designed
% fine grid is not rejected solely because its endpoint accumulated roundoff.
% Count the coordinate increments directly so this boundary-condition routine
% also accepts its documented minimal mesh interface (Nodes.acum/Nodes.Tnum),
% without requiring the later B03 element bookkeeping to exist already.
% The tolerance remains orders of magnitude below any physical mesh offset.
mesh_element_count = max(numel(Beam.Mesh.Nodes.acum) - 1, 0);
summation_roundoff_factor = max(256, 2*mesh_element_count);
Beam.BC.loc_tolerance = summation_roundoff_factor * eps(alignment_scale);
% Positive vertical springs are the active campaign's bridge supports.  A
% nearest-node shift changes span lengths and the damage location, so reject
% it rather than silently solving a different geometry.
misaligned_spring_supports = find(Beam.BC.vert_stiff > 0 & ...
    abs(Beam.BC.loc_offset) > Beam.BC.loc_tolerance);
if ~isempty(misaligned_spring_supports)
    i = misaligned_spring_supports(1);
    error('B02:SupportNotOnNode', ...
        ['Positive-spring support %d at %.17g m maps to %.17g m ' ...
         '(offset %.17g m). Choose a support-aligned bridge mesh.'], ...
        i, Beam.BC.loc(i), Beam.BC.loc_realized(i), ...
        Beam.BC.loc_offset(i));
end

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
DOF_Original_value = 344e6;
rigid_vertical_nodes = Beam.BC.loc_ind(Beam.BC.vert_stiff == -1);

% Safely initialize Damage fields if they do not exist
if ~isfield(Damage, 'scour_rates')
    Damage.scour_rates = zeros(1, Beam.BC.supp_num);
end
if ~isnumeric(Damage.scour_rates) || ~isreal(Damage.scour_rates) || ...
        (~isempty(Damage.scour_rates) && ~isvector(Damage.scour_rates)) || ...
        any(~isfinite(Damage.scour_rates(:))) || ...
        any(Damage.scour_rates(:) < 0 | Damage.scour_rates(:) > 1)
    error('B02:InvalidScourRates', ...
        'Damage.scour_rates must be a finite real vector in [0,1].');
end
Damage.scour_rates = Damage.scour_rates(:)';
% The rail model reuses this function with no positive vertical springs and
% legitimately carries the bridge's Damage struct.  Exact support alignment
% is therefore required precisely when scour can enter the assembled matrix.
if any(Beam.BC.vert_stiff > 0) && ...
        numel(Damage.scour_rates) ~= Beam.BC.supp_num
    error('B02:ScourSupportCountMismatch', ...
        ['Damage.scour_rates must have exactly one entry per support when ' ...
         'positive vertical support springs are active.']);
end

positive_vertical_supports = find(Beam.BC.vert_stiff > 0);
positive_vertical_supports = positive_vertical_supports(:)';
% MATLAB 1-based indexing: vertical DOF = node * 2 - 1.
vert_with_values = ...
    Beam.BC.loc_ind(positive_vertical_supports) * 2 - 1;
retained_stiffness = 1.0 - ...
    Damage.scour_rates(positive_vertical_supports);
vert_stiff_values = retained_stiffness * DOF_Original_value;
rigid_vertical_nodes = [rigid_vertical_nodes, ...
    Beam.BC.loc_ind(positive_vertical_supports(vert_stiff_values > 0))];

% -------------------------------------------------------------------------
% 2. Rotational Stiffness (nominal abutment rotational-fixity intervention;
%    NOT physical bearing degradation — see paper claim boundary)
% -------------------------------------------------------------------------
rigid_rotation_constrained = any(Beam.BC.rot_stiff == -1);

% Safely initialize abutment rotational-fixity fields (k_r = 0 is the
% free-rotation baseline)
if ~isfield(Damage, 'bearing_left')
    Damage.bearing_left = 0.0;
end
if ~isfield(Damage, 'bearing_right')
    Damage.bearing_right = 0.0;
end
if ~isnumeric(Damage.bearing_left) || ~isreal(Damage.bearing_left) || ...
        ~isscalar(Damage.bearing_left) || ~isfinite(Damage.bearing_left) || ...
        Damage.bearing_left < 0
    error('B02:InvalidBearingStiffness', ...
        'Damage.bearing_left must be a finite nonnegative scalar [Nm/rad].');
end
if ~isnumeric(Damage.bearing_right) || ~isreal(Damage.bearing_right) || ...
        ~isscalar(Damage.bearing_right) || ~isfinite(Damage.bearing_right) || ...
        Damage.bearing_right < 0
    error('B02:InvalidBearingStiffness', ...
        'Damage.bearing_right must be a finite nonnegative scalar [Nm/rad].');
end

positive_rotational_supports = find(Beam.BC.rot_stiff > 0);
positive_rotational_supports = positive_rotational_supports(:)';
% MATLAB 1-based indexing: rotational DOF = node * 2.
rot_with_values = Beam.BC.loc_ind(positive_rotational_supports) * 2;
rot_stiff_values = zeros(size(positive_rotational_supports));
rot_stiff_values(positive_rotational_supports == 1) = ...
    Damage.bearing_left;
rot_stiff_values(positive_rotational_supports == Beam.BC.supp_num) = ...
    Damage.bearing_right;
rigid_rotation_constrained = rigid_rotation_constrained || ...
    any(rot_stiff_values > 0);

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
% An Euler-Bernoulli beam has two rigid-body coordinates: vertical
% translation and rigid rotation. Count their constraint RANK, not merely
% fixed DOFs: positive elastic support springs also remove rigid modes, while
% two rotational constraints are rank-duplicate for rigid-body motion.
rigid_constraint_rank = 0;
if ~isempty(rigid_vertical_nodes)
    rigid_constraint_rank = 1;
    if numel(unique(rigid_vertical_nodes)) >= 2
        rigid_constraint_rank = 2;
    end
end
if rigid_rotation_constrained && rigid_constraint_rank < 2
    rigid_constraint_rank = rigid_constraint_rank + 1;
end
Beam.Modal.num_rigid_modes = 2 - rigid_constraint_rank;

% Value to use in the diagonal element when the DOF is fixed
Beam.BC.DOF_fixed_value = DOF_Original_value;

% ---- End of script ----
