function [Model] = B54_ModelMatrices(Beam,Track,Calc,Damage)

% Assembles the coupled model (Track+Beam) system matrices
%
% Optional 4th argument Damage may carry per-passage TRACK-LAYER damage
% descriptors in Damage.track (ballast patches, hanging-sleeper groups,
% rail-pad aging/failures) — see local_track_vectors below and
% docs/stage3_alldamage_spec.md. Without it (or with Damage.track absent)
% the assembled matrices are numerically identical to the legacy path.
if nargin < 4, Damage = struct(); end

% *************************************************************************
% *** Script part of TTB-2D tool for Matlab environment.                ***
% *** Licensed under the GNU General Public License v3.0                ***
% *** Author: Daniel Cantero (daniel.cantero@ntnu.no)                   ***
% *** For help, modifications, and collaboration contact the author.    ***
% *************************************************************************

% -------------------------------------------------------------------------
% % ---- Inputs ----
% Beam = Structure with Beam variables, including at least:
%   ... see script ...
% Track = Structure with Track variables, including at least:
%   ... see script ...
% Calc = Structure with Calculation variables, including at least:
%   ... see script ...
% % ---- Outputs ----
% Model = New structure variable with the system matrices for 
%   the Track and Beam models coupled together
%   ... see script ...
% -------------------------------------------------------------------------

% ---- Counting ----
% Total
Track.Sleeper.Tnum = round(Calc.Profile.L/Track.Sleeper.spacing) + 1;
% Approach
Track.Sleeper.num_app = round((Calc.Profile.max_TL*Calc.Options.redux_factor+ ...
    Calc.Profile.L_Approach)/Track.Sleeper.spacing);
if Calc.Profile.extra_L < Calc.Cte.tol
    Track.Sleeper.num_onbeam = round(Beam.Prop.L/Track.Sleeper.spacing) + 1;
    Track.Sleeper.num_aft = round((Calc.Profile.L - ...
        (Calc.Profile.max_TL*Calc.Options.redux_factor+Calc.Profile.L_Approach+ ...
        Beam.Prop.L+Calc.Profile.extra_L))/Track.Sleeper.spacing);
else
    Track.Sleeper.num_onbeam = floor(Beam.Prop.L/Track.Sleeper.spacing) + 1;
    Track.Sleeper.num_aft = round((Calc.Profile.L - ...
        (Calc.Profile.max_TL*Calc.Options.redux_factor+Calc.Profile.L_Approach+...
        Beam.Prop.L+Calc.Profile.extra_L))/Track.Sleeper.spacing) + 1;
end % if Calc.Profile.extra_L < Calc.Cte.tol

% % Check (Should be zero)
% Track.Sleeper.Tnum - (Track.Sleeper.num_app + Track.Sleeper.num_onbeam + Track.Sleeper.num_aft)

% ---- DOF indices ----
Model.Mesh.DOF.rail = 1:Track.Rail.Mesh.DOF.Tnum;
Model.Mesh.DOF.rail_vert = Model.Mesh.DOF.rail(1:2:end);
Model.Mesh.DOF.rail_vert_at_sleepers = ...
    (1:Track.Rail.Mesh.Ele.num_per_spacing:Track.Rail.Mesh.Nodes.Tnum)*2-1;
Model.Mesh.DOF.sleepers = (1:Track.Sleeper.Tnum) + Track.Rail.Mesh.DOF.Tnum;
Model.Mesh.DOF.sleepers_app = Model.Mesh.DOF.sleepers(1:Track.Sleeper.num_app);
Model.Mesh.DOF.sleepers_onbeam = ...
    Model.Mesh.DOF.sleepers(Track.Sleeper.num_app+(1:Track.Sleeper.num_onbeam));
Model.Mesh.DOF.sleepers_aft = ...
    Model.Mesh.DOF.sleepers(Track.Sleeper.num_app+Track.Sleeper.num_onbeam + ...
    (1:Track.Sleeper.num_aft));
Model.Mesh.DOF.ballast_app = Model.Mesh.DOF.sleepers(end) + ...
    (1:Track.Sleeper.num_app);
Model.Mesh.DOF.beam = Model.Mesh.DOF.ballast_app(end) + (1:Beam.Mesh.DOF.Tnum);
Model.Mesh.DOF.beam_vert = Model.Mesh.DOF.beam(1:2:end);
Model.Mesh.DOF.beam_vert_under_sleeper = ...
    Model.Mesh.DOF.beam_vert(1:Beam.Mesh.Ele.num_per_spacing:end);
Model.Mesh.DOF.ballast_aft = Model.Mesh.DOF.beam(end) + (1:Track.Sleeper.num_aft);

if Track.Sleeper.num_aft == 0
    Model.Mesh.DOF.Tnum = Model.Mesh.DOF.beam(end);
else
    Model.Mesh.DOF.Tnum = Model.Mesh.DOF.ballast_aft(end);
end % if Track.Sleeper.num_aft == 0

% ---- X location of some DOF ---- (Useful for plotting)
Model.Mesh.XLoc.rail_vert = Track.Rail.Mesh.Nodes.acum;
Model.Mesh.XLoc.sleepers = ...
    (0:length(Model.Mesh.DOF.sleepers)-1)*Track.Sleeper.spacing;
Model.Mesh.XLoc.ballast_app = ...
    (0:length(Model.Mesh.DOF.ballast_app)-1)*Track.Sleeper.spacing;
Model.Mesh.XLoc.beam_vert = Beam.Mesh.Nodes.acum + Calc.Profile.L_Approach + ...
    Calc.Profile.max_TL*Calc.Options.redux_factor;
Model.Mesh.XLoc.ballast_aft = ...
    ((0:length(Model.Mesh.DOF.ballast_aft)-1)+1)*Track.Sleeper.spacing + ...
    Calc.Profile.L_Approach + Beam.Prop.L + ...
    Calc.Profile.max_TL*Calc.Options.redux_factor + Calc.Profile.extra_L;

% % Graphical check
% figure; hold on; box on;
% plot(Model.Mesh.XLoc.rail_vert,1.5*ones(size(Model.Mesh.XLoc.rail_vert)),'.');
% plot(Model.Mesh.XLoc.sleepers,1*ones(size(Model.Mesh.XLoc.sleepers)),'g.');
% plot(Model.Mesh.XLoc.ballast_app,0.5*ones(size(Model.Mesh.XLoc.ballast_app)),'r.');
% plot(Model.Mesh.XLoc.beam_vert,0.5*ones(size(Model.Mesh.XLoc.beam_vert)),'k.');
% aux1 = 1:Beam.Mesh.Ele.num_per_spacing:length(Model.Mesh.XLoc.beam_vert);
% plot(Model.Mesh.XLoc.beam_vert(aux1),0.5*ones(size(Model.Mesh.XLoc.beam_vert(aux1))),'ko');
% plot(Model.Mesh.XLoc.ballast_aft,0.5*ones(size(Model.Mesh.XLoc.ballast_aft)),'r.');
% ylim([0,2]);
% legend('Rail nodes','Sleepers','Ballast app.','Beam','Beam under Sleeper','Ballast aft.');

% Temporary variable name change
if Track.PadUnderSleeperOnBeam.included == 1
    Track.BallastOnBeam.Prop.m = 0;
    Track.BallastOnBeam.Prop.c = Track.PadUnderSleeperOnBeam.Prop.c;
    Track.BallastOnBeam.Prop.k = Track.PadUnderSleeperOnBeam.Prop.k;
end % if Track.PadUnderSleeperOnBeam.included == 1

% ---- Per-sleeper track-layer property vectors (damage-aware EOVs) ----
% Track-layer damage (Stage 3) enters HERE and only here: descriptors in
% Damage.track are mapped onto per-sleeper stiffness/damping vectors using
% the sleeper x-positions. Healthy (no descriptors) -> uniform vectors that
% assemble numerically identical matrices to the legacy scalar path.
% Mirrors TTBI_2D/b54_model_matrices.py::_track_vectors.
% Second output = placement diagnostics (resolved GLOBAL coordinates of every
% modified sleeper/pad + the frame offset applied) - consumed by the smoke
% tests to assert the damage actually lands on/around the deck.
[TrkV,Model.TrackDmgDbg] = local_track_vectors(Track,Model,Calc,Damage);

% ---------------------- Building Global matrices -------------------------

% Initialize matrices
% Model.Mesh.Mg = zeros(Model.Mesh.DOF.Tnum);
% Model.Mesh.Cg = zeros(Model.Mesh.DOF.Tnum);
% Model.Mesh.Kg = zeros(Model.Mesh.DOF.Tnum);
Model.Mesh.Mg = sparse(Model.Mesh.DOF.Tnum,Model.Mesh.DOF.Tnum);
Model.Mesh.Cg = Model.Mesh.Mg;
Model.Mesh.Kg = Model.Mesh.Mg;

% ---- Diagonal Elementens ----

% Track
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.rail,Track.Rail.Mesh.Mg);
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.rail,Track.Rail.Mesh.Cg);
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.rail,Track.Rail.Mesh.Kg);

% Pads to rail DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.rail_vert_at_sleepers,...
    funDiag(Track.Sleeper.Tnum,TrkV.pad_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.rail_vert_at_sleepers,...
    funDiag(Track.Sleeper.Tnum,TrkV.pad_k));

% Pads to sleepers DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.sleepers,...
    funDiag(Track.Sleeper.Tnum,TrkV.pad_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.sleepers,...
    funDiag(Track.Sleeper.Tnum,TrkV.pad_k));

% Sleepers
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.sleepers,...
    funDiag(Track.Sleeper.Tnum,Track.Sleeper.Prop.m));

% Ballast on approach to sleepers DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_app,...
    funDiag(Track.Sleeper.num_app,TrkV.balA_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_app,...
    funDiag(Track.Sleeper.num_app,TrkV.balA_k));

% Ballast on bridge to sleepers DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_onbeam,...
    funDiag(Track.Sleeper.num_onbeam,TrkV.balB_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_onbeam,...
    funDiag(Track.Sleeper.num_onbeam,TrkV.balB_k));

% Ballast after bridge to sleepers DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_aft,...
    funDiag(Track.Sleeper.num_aft,TrkV.balF_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_aft,...
    funDiag(Track.Sleeper.num_aft,TrkV.balF_k));

% Ballast on approach to Ballast DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.ballast_app,...
    funDiag(Track.Sleeper.num_app,TrkV.balA_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.ballast_app,...
    funDiag(Track.Sleeper.num_app,TrkV.balA_k));

% Ballast on bridge to Bridge DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.beam_vert_under_sleeper,...
    funDiag(Track.Sleeper.num_onbeam,TrkV.balB_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.beam_vert_under_sleeper,...
    funDiag(Track.Sleeper.num_onbeam,TrkV.balB_k));

% Ballast after bridge to Ballast DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.ballast_aft,...
    funDiag(Track.Sleeper.num_aft,TrkV.balF_c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.ballast_aft,...
    funDiag(Track.Sleeper.num_aft,TrkV.balF_k));

% Ballast on approach
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.ballast_app,...
    funDiag(Track.Sleeper.num_app,Track.Ballast.Prop.m));

% Ballast on bridge
% Note: The mass of ballast on bridge is distributed to all the Beam's 
%       vertical DOF, rather than only to the ones under a sleeper.
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.beam_vert,...
    funDiag(Beam.Mesh.Nodes.Tnum,Track.BallastOnBeam.Prop.m/Beam.Mesh.Ele.num_per_spacing));

% Ballast after approach
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.ballast_aft,...
    funDiag(Track.Sleeper.num_aft,Track.Ballast.Prop.m));

% Beam
Model.Mesh.Mg = funAdd1(Model.Mesh.Mg,Model.Mesh.DOF.beam,Beam.Mesh.Mg);
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.beam,Beam.Mesh.Cg);
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.beam,Beam.Mesh.Kg);

% Sub-Ballast on approach to Ballast DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.ballast_app,...
    funDiag(Track.Sleeper.num_app,Track.SubBallast.Prop.c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.ballast_app,...
    funDiag(Track.Sleeper.num_app,Track.SubBallast.Prop.k));

% Ballast after bridge to Ballast DOF
Model.Mesh.Cg = funAdd1(Model.Mesh.Cg,Model.Mesh.DOF.ballast_aft,...
    funDiag(Track.Sleeper.num_aft,Track.SubBallast.Prop.c));
Model.Mesh.Kg = funAdd1(Model.Mesh.Kg,Model.Mesh.DOF.ballast_aft,...
    funDiag(Track.Sleeper.num_aft,Track.SubBallast.Prop.k));

% ---- Off-Diagonal Elementens ----

% Rail and Sleepers
Model.Mesh.Cg = funAdd2(Model.Mesh.Cg,Model.Mesh.DOF.rail_vert_at_sleepers,...
    Model.Mesh.DOF.sleepers,-funDiag(Track.Sleeper.Tnum,TrkV.pad_c));
Model.Mesh.Kg = funAdd2(Model.Mesh.Kg,Model.Mesh.DOF.rail_vert_at_sleepers,...
    Model.Mesh.DOF.sleepers,-funDiag(Track.Sleeper.Tnum,TrkV.pad_k));

% Sleepers and Ballast on approach
Model.Mesh.Cg = funAdd2(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_app,...
    Model.Mesh.DOF.ballast_app,-funDiag(Track.Sleeper.num_app,TrkV.balA_c));
Model.Mesh.Kg = funAdd2(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_app,...
    Model.Mesh.DOF.ballast_app,-funDiag(Track.Sleeper.num_app,TrkV.balA_k));

% Sleepers and Beam
Model.Mesh.Cg = funAdd2(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_onbeam,...
    Model.Mesh.DOF.beam_vert_under_sleeper,-funDiag(Track.Sleeper.num_onbeam,TrkV.balB_c));
Model.Mesh.Kg = funAdd2(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_onbeam,...
    Model.Mesh.DOF.beam_vert_under_sleeper,-funDiag(Track.Sleeper.num_onbeam,TrkV.balB_k));

% Sleepers and Ballast after bridge
Model.Mesh.Cg = funAdd2(Model.Mesh.Cg,Model.Mesh.DOF.sleepers_aft,...
    Model.Mesh.DOF.ballast_aft,-funDiag(Track.Sleeper.num_aft,TrkV.balF_c));
Model.Mesh.Kg = funAdd2(Model.Mesh.Kg,Model.Mesh.DOF.sleepers_aft,...
    Model.Mesh.DOF.ballast_aft,-funDiag(Track.Sleeper.num_aft,TrkV.balF_k));

% Sparse matrix output
Model.Mesh.Mg = sparse(Model.Mesh.Mg);
Model.Mesh.Cg = sparse(Model.Mesh.Cg);
Model.Mesh.Kg = sparse(Model.Mesh.Kg);

% Checking symmetry
checksum = sum([max(max(abs(Model.Mesh.Mg-Model.Mesh.Mg'))),...
    max(max(abs(Model.Mesh.Cg-Model.Mesh.Cg'))),...
    max(max(abs(Model.Mesh.Kg-Model.Mesh.Kg')))]);
if checksum > Calc.Cte.tol
    dips('System matrices are not symmetric');
    error('System matrices are not symmetric');
end % if checksum > Calc.Cte.tol

% Auxiliary variables
Model.Mesh.Ele.DOF = Track.Rail.Mesh.Ele.DOF;
Model.Mesh.Ele.a = Track.Rail.Mesh.Ele.a;

% -------------------------- Subfunctions ---------------------------------

function [InM] = funAdd1(InM,ind1,AddM)

    % Subfunction that adds the values of the matrix "AddM" to the 
    % matrix "InM" at rows and columns defined by "ind1"

    InM(ind1,ind1) = InM(ind1,ind1) + AddM;

function [InM] = funAdd2(InM,ind1,ind2,AddM)

    % Subfunction that adds the values of the matrix "AddM" to the 
    % matrix "InM" at rows "ind1" and columns "ind2"
    
    InM(ind1,ind2) = InM(ind1,ind2) + AddM;
    InM(ind2,ind1) = InM(ind2,ind1) + AddM;

function [OutM] = funDiag(size,value)

    % Subfunction that generates a diagonal matrix of dimensions "size"x"size"
    % with values "value". Accepts a SCALAR (legacy: uniform diagonal) or a
    % VECTOR of length "size" (per-sleeper damage-aware properties).

    if isscalar(value)
        OutM = diag(ones(1,size))*value;
    else
        OutM = diag(value(:).');
    end

function [V,Dbg] = local_track_vectors(Track,Model,Calc,Damage)

    % Per-sleeper track-layer property vectors from Damage.track descriptors.
    % COORDINATE FRAME (audit fix 2026-07-17): descriptors carrying an
    % .x_bridge_local field are in A00's BRIDGE-LOCAL window frame (deck
    % starts at x = x_bridge_local). The sleeper axis here is GLOBAL
    % (0..Calc.Profile.L) and under redux = 0 the deck really starts at
    % num_app*spacing = L_Approach + max_TL (123.0 m for L60) - NOT at
    % L_Approach, as a stale comment here used to claim. The global axis is
    % shifted into the local frame before any selection, so every descriptor
    % lands relative to the REAL deck. Descriptors WITHOUT .x_bridge_local
    % (legacy datasets) are read as global coordinates, exactly as before.
    % Fields (optional):
    %   .ballast_patches : rows [x_start, x_end, eta_k, eta_c] — fouling /
    %                      degradation patch (stiffness+damping multipliers)
    %   .hanging_groups  : rows [x_start, n_consec] — unsupported-sleeper
    %                      group (linearised: ballast support -> ~0)
    %   .pad_stiff_mult  : scalar chi_pad (global pad aging multiplier)
    %   .pad_damp_mult   : scalar beta_pad
    %   .pad_failures    : vector of x-positions of failed pads (k -> ~0)
    %   .x_bridge_local  : deck-start position in the descriptor frame [m]
    % Healthy (no Damage.track) -> uniform vectors; the assembled matrices
    % are numerically identical to the legacy scalar path.
    % Mirror of TTBI_2D/b54_model_matrices.py::_track_vectors.
    % Dbg = resolved GLOBAL positions of modified entries (smoke-test hook).

    n = Track.Sleeper.Tnum;
    x = Model.Mesh.XLoc.sleepers;            % [1 x n] sleeper positions [m]
    % First/last on-beam sleeper in the GLOBAL sleeper-axis frame:
    x_deck0 = Track.Sleeper.num_app * Track.Sleeper.spacing;
    x_deck1 = (Track.Sleeper.num_app + Track.Sleeper.num_onbeam - 1) * ...
        Track.Sleeper.spacing;
    off = 0;                                 % global -> descriptor-frame shift
    if isfield(Damage,'track') && ~isempty(Damage.track) && ...
            isfield(Damage.track,'x_bridge_local') && ...
            ~isempty(Damage.track.x_bridge_local)
        off = x_deck0 - Damage.track.x_bridge_local;
    end
    x = x - off;                             % sleeper axis in descriptor frame
    mult_bal_k = ones(1,n); mult_bal_c = ones(1,n);
    mult_pad_k = ones(1,n); mult_pad_c = ones(1,n);
    KILL = 1e-6;    % "removed" support: not exactly 0 to keep Kg well-posed

    if isfield(Damage,'track') && ~isempty(Damage.track)
        T = Damage.track;
        if isfield(T,'ballast_patches') && ~isempty(T.ballast_patches)
            for r = 1:size(T.ballast_patches,1)
                p = T.ballast_patches(r,:);
                sel = (x >= p(1)) & (x <= p(2));
                % Overlapping patches: the patch with the LARGEST stiffness
                % deviation |log eta_k| governs and supplies BOTH eta_k and
                % eta_c (a sleeper is either predominantly wet-fouled,
                % eta_k<1 & eta_c>1, or dry-fouled, eta_k>1 & eta_c<1 -
                % mixing quantities across patches would be an unphysical
                % hybrid). FIX 2026-07-22 (external audit r3, verified): the
                % old product of stacked draws could leave the documented
                % per-patch bands (k up to ~4 dry-on-dry, ~0.5 wet-on-wet).
                upd = sel & (abs(log(p(3))) > abs(log(mult_bal_k)));
                mult_bal_k(upd) = p(3);
                mult_bal_c(upd) = p(4);
            end
        end
        if isfield(T,'hanging_groups') && ~isempty(T.hanging_groups)
            for r = 1:size(T.hanging_groups,1)
                i0 = find(x >= T.hanging_groups(r,1) - Calc.Cte.tol,1,'first');
                if isempty(i0), continue; end
                idx = i0:min(i0 + T.hanging_groups(r,2) - 1, n);
                mult_bal_k(idx) = KILL;
                mult_bal_c(idx) = KILL;
            end
        end
        if isfield(T,'pad_stiff_mult') && ~isempty(T.pad_stiff_mult)
            mult_pad_k = mult_pad_k*T.pad_stiff_mult;
        end
        if isfield(T,'pad_damp_mult') && ~isempty(T.pad_damp_mult)
            mult_pad_c = mult_pad_c*T.pad_damp_mult;
        end
        if isfield(T,'pad_failures') && ~isempty(T.pad_failures)
            for r = 1:numel(T.pad_failures)
                [~,i0] = min(abs(x - T.pad_failures(r)));
                mult_pad_k(i0) = KILL;
                mult_pad_c(i0) = KILL;
            end
        end
    end

    % Segment slices (approach / on-beam / after) — same indexing as the DOFs
    i_app = 1:Track.Sleeper.num_app;
    i_on  = Track.Sleeper.num_app + (1:Track.Sleeper.num_onbeam);
    i_aft = Track.Sleeper.num_app + Track.Sleeper.num_onbeam + ...
        (1:Track.Sleeper.num_aft);

    % ---- Placement diagnostics (GLOBAL coordinates; smoke-test hook) ----
    x_glob = x + off;
    Dbg = struct();
    Dbg.frame_offset     = off;                    % global - descriptor frame
    Dbg.x_deck_global    = [x_deck0, x_deck1];     % first/last on-beam sleeper
    Dbg.bal_x_global     = x_glob(mult_bal_k ~= 1 & mult_bal_k ~= KILL);
    Dbg.hang_x_global    = x_glob(mult_bal_k == KILL);
    Dbg.padfail_x_global = x_glob(mult_pad_k == KILL | mult_pad_c == KILL);

    V.pad_k  = Track.Pad.Prop.k          * mult_pad_k;         % full track
    V.pad_c  = Track.Pad.Prop.c          * mult_pad_c;
    V.balA_k = Track.Ballast.Prop.k      * mult_bal_k(i_app);  % approach
    V.balA_c = Track.Ballast.Prop.c      * mult_bal_c(i_app);
    V.balB_k = Track.BallastOnBeam.Prop.k* mult_bal_k(i_on);   % on bridge
    V.balB_c = Track.BallastOnBeam.Prop.c* mult_bal_c(i_on);
    V.balF_k = Track.Ballast.Prop.k      * mult_bal_k(i_aft);  % after
    V.balF_c = Track.Ballast.Prop.c      * mult_bal_c(i_aft);

% ---- End of function ----