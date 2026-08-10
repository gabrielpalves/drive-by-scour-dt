function [V,Dbg] = B54_TrackVectors(Track,Model,Calc,Damage)
%B54_TRACKVECTORS Resolve track-EOV descriptors into per-sleeper properties.
%
% This is the testable production helper used by B54_ModelMatrices.  It was
% extracted verbatim from B54's former local_track_vectors in audit r4 so the
% governing-patch overlap rule can be parity-tested against the Python mirror.
%
% COORDINATE FRAME: descriptors carrying x_bridge_local are expressed in
% A00's bridge-local window frame.  Descriptors without it remain global.
% DESCRIPTOR BOUNDARY: malformed/nonfinite rows and coordinates outside the
% modeled sleeper domain are rejected. Hanging-group x is a continuous group
% start (resolved to the first sleeper at or to its right), whereas failed-pad
% x values must lie on the sleeper lattice exactly within numerical tolerance.

n = Track.Sleeper.Tnum;
x = Model.Mesh.XLoc.sleepers;            % [1 x n] sleeper positions [m]
x_deck0 = Track.Sleeper.num_app * Track.Sleeper.spacing;
x_deck1 = (Track.Sleeper.num_app + Track.Sleeper.num_onbeam - 1) * ...
    Track.Sleeper.spacing;
off = 0;                                 % global -> descriptor-frame shift
has_track = isfield(Damage,'track') && ~isempty(Damage.track);
if has_track
    if ~isstruct(Damage.track) || ~isscalar(Damage.track)
        error('B54_TrackVectors:InvalidDamageTrack', ...
            'Damage.track must be empty or a scalar structure.');
    end
    if isfield(Damage.track,'x_bridge_local') && ...
            ~isempty(Damage.track.x_bridge_local)
        local_require_finite_scalar(Damage.track.x_bridge_local, ...
            'B54_TrackVectors:InvalidBridgeLocalCoordinate', ...
            'Damage.track.x_bridge_local');
        off = x_deck0 - Damage.track.x_bridge_local;
    end
end
x = x - off;
tol = max(Calc.Cte.tol,64*eps(max(1,max(abs(x)))));
mult_bal_k = ones(1,n); mult_bal_c = ones(1,n);
mult_pad_k = ones(1,n); mult_pad_c = ones(1,n);
KILL = 1e-6;    % "removed" support: not exactly 0 to keep Kg well-posed

if has_track
    T = Damage.track;
    if isfield(T,'ballast_patches') && ~isempty(T.ballast_patches)
        local_validate_ballast_patches(T.ballast_patches,x,tol);
        for r = 1:size(T.ballast_patches,1)
            p = T.ballast_patches(r,:);
            sel = (x >= p(1)) & (x <= p(2));
            % Overlapping patches: the largest |log eta_k| deviation governs
            % and supplies BOTH eta_k and eta_c.  This prevents stacked
            % products and unphysical k/c hybrids.
            upd = sel & (abs(log(p(3))) > abs(log(mult_bal_k)));
            mult_bal_k(upd) = p(3);
            mult_bal_c(upd) = p(4);
        end
    end
    if isfield(T,'hanging_groups') && ~isempty(T.hanging_groups)
        local_validate_hanging_groups(T.hanging_groups,x,tol);
        for r = 1:size(T.hanging_groups,1)
            i0 = find(x >= T.hanging_groups(r,1) - tol,1,'first');
            idx = i0:(i0 + T.hanging_groups(r,2) - 1);
            mult_bal_k(idx) = KILL;
            mult_bal_c(idx) = KILL;
        end
    end
    if isfield(T,'pad_stiff_mult') && ~isempty(T.pad_stiff_mult)
        local_require_positive_scalar(T.pad_stiff_mult, ...
            'B54_TrackVectors:InvalidPadStiffnessMultiplier', ...
            'Damage.track.pad_stiff_mult');
        mult_pad_k = mult_pad_k*T.pad_stiff_mult;
    end
    if isfield(T,'pad_damp_mult') && ~isempty(T.pad_damp_mult)
        local_require_positive_scalar(T.pad_damp_mult, ...
            'B54_TrackVectors:InvalidPadDampingMultiplier', ...
            'Damage.track.pad_damp_mult');
        mult_pad_c = mult_pad_c*T.pad_damp_mult;
    end
    if isfield(T,'pad_failures') && ~isempty(T.pad_failures)
        pad_indices = local_validate_pad_failures(T.pad_failures,x,tol);
        for r = 1:numel(pad_indices)
            i0 = pad_indices(r);
            mult_pad_k(i0) = KILL;
            mult_pad_c(i0) = KILL;
        end
    end
end

i_app = 1:Track.Sleeper.num_app;
i_on  = Track.Sleeper.num_app + (1:Track.Sleeper.num_onbeam);
i_aft = Track.Sleeper.num_app + Track.Sleeper.num_onbeam + ...
    (1:Track.Sleeper.num_aft);

x_glob = x + off;
Dbg = struct();
Dbg.frame_offset     = off;
Dbg.x_deck_global    = [x_deck0, x_deck1];
Dbg.bal_x_global     = x_glob(mult_bal_k ~= 1 & mult_bal_k ~= KILL);
Dbg.hang_x_global    = x_glob(mult_bal_k == KILL);
Dbg.padfail_x_global = x_glob(mult_pad_k == KILL | mult_pad_c == KILL);
% Resolved multipliers make the overlap rule directly auditable without
% reverse-engineering assembled matrix diagonals.
Dbg.x_descriptor = x;
Dbg.mult_bal_k = mult_bal_k;
Dbg.mult_bal_c = mult_bal_c;
Dbg.mult_pad_k = mult_pad_k;
Dbg.mult_pad_c = mult_pad_c;

V.pad_k  = Track.Pad.Prop.k          * mult_pad_k;
V.pad_c  = Track.Pad.Prop.c          * mult_pad_c;
V.balA_k = Track.Ballast.Prop.k      * mult_bal_k(i_app);
V.balA_c = Track.Ballast.Prop.c      * mult_bal_c(i_app);
V.balB_k = Track.BallastOnBeam.Prop.k* mult_bal_k(i_on);
V.balB_c = Track.BallastOnBeam.Prop.c* mult_bal_c(i_on);
V.balF_k = Track.Ballast.Prop.k      * mult_bal_k(i_aft);
V.balF_c = Track.Ballast.Prop.c      * mult_bal_c(i_aft);
end

function local_validate_ballast_patches(patches,x,tol)
% Validate [x0,x1,eta_k,eta_c] before any row can be ignored or misread.

if ~isnumeric(patches) || ~isreal(patches) || ~ismatrix(patches) || ...
        size(patches,2) ~= 4 || any(~isfinite(patches(:)))
    error('B54_TrackVectors:InvalidBallastPatches', ...
        ['Damage.track.ballast_patches must be a finite real N-by-4 ' ...
         'matrix [x0,x1,eta_k,eta_c].']);
end
if any(patches(:,1) > patches(:,2))
    error('B54_TrackVectors:ReversedBallastPatch', ...
        'Each ballast patch must satisfy x0 <= x1.');
end
if any(patches(:,3) <= 0) || any(patches(:,4) <= 0)
    error('B54_TrackVectors:InvalidBallastMultiplier', ...
        'Ballast eta_k and eta_c multipliers must be strictly positive.');
end
x_min = min(x); x_max = max(x);
if any(patches(:,1) < x_min - tol) || any(patches(:,2) > x_max + tol)
    error('B54_TrackVectors:BallastPatchOutsideDomain', ...
        ['Ballast patch coordinates must lie within the modeled sleeper ' ...
         'domain [%.15g, %.15g] m in the descriptor frame.'],x_min,x_max);
end
for row = 1:size(patches,1)
    if ~any(x >= patches(row,1) & x <= patches(row,2))
        error('B54_TrackVectors:BallastPatchSelectsNoSleeper', ...
            ['Ballast patch row %d selects no sleeper. Supply an interval ' ...
             'that intersects the modeled sleeper lattice.'],row);
    end
end
end

function local_validate_hanging_groups(groups,x,tol)
% A continuous start maps to the first sleeper at or to its right; count is exact.

if ~isnumeric(groups) || ~isreal(groups) || ~ismatrix(groups) || ...
        size(groups,2) ~= 2 || any(~isfinite(groups(:)))
    error('B54_TrackVectors:InvalidHangingGroups', ...
        ['Damage.track.hanging_groups must be a finite real N-by-2 ' ...
         'matrix [x0,count].']);
end
counts = groups(:,2);
if any(counts < 1) || any(counts ~= fix(counts))
    error('B54_TrackVectors:InvalidHangingCount', ...
        'Every hanging-sleeper count must be a positive integer.');
end
x_min = min(x); x_max = max(x);
if any(groups(:,1) < x_min - tol) || any(groups(:,1) > x_max + tol)
    error('B54_TrackVectors:HangingGroupOutsideDomain', ...
        ['Hanging-group starts must lie within the modeled sleeper domain ' ...
         '[%.15g, %.15g] m in the descriptor frame.'],x_min,x_max);
end
for row = 1:size(groups,1)
    first = find(x >= groups(row,1) - tol,1,'first');
    if isempty(first)
        error('B54_TrackVectors:HangingGroupSelectsNoSleeper', ...
            'Hanging-group row %d selects no sleeper.',row);
    end
    if first + counts(row) - 1 > numel(x)
        error('B54_TrackVectors:HangingGroupExceedsDomain', ...
            ['Hanging-group row %d requests %d sleepers but only %d remain ' ...
             'from its resolved start; refusing silent truncation.'], ...
            row,counts(row),numel(x)-first+1);
    end
end
end

function indices = local_validate_pad_failures(positions,x,tol)
% Pad failures are lattice descriptors; arbitrary coordinates must not snap.

if ~isnumeric(positions) || ~isreal(positions) || ~isvector(positions) || ...
        any(~isfinite(positions(:)))
    error('B54_TrackVectors:InvalidPadFailures', ...
        ['Damage.track.pad_failures must be a finite real vector of ' ...
         'descriptor-frame sleeper coordinates [m].']);
end
positions = positions(:)';
x_min = min(x); x_max = max(x);
if any(positions < x_min - tol) || any(positions > x_max + tol)
    error('B54_TrackVectors:PadFailureOutsideDomain', ...
        ['Pad-failure coordinates must lie within the modeled sleeper ' ...
         'domain [%.15g, %.15g] m in the descriptor frame.'],x_min,x_max);
end
indices = zeros(size(positions));
for item = 1:numel(positions)
    [distance,indices(item)] = min(abs(x - positions(item)));
    if distance > tol
        error('B54_TrackVectors:PadFailureOffLattice', ...
            ['Pad-failure coordinate %.15g m is not on the modeled sleeper ' ...
             'lattice (nearest distance %.3g m); refusing silent snapping.'], ...
            positions(item),distance);
    end
end
if numel(unique(indices)) ~= numel(indices)
    error('B54_TrackVectors:DuplicatePadFailure', ...
        'Each failed-pad sleeper may appear at most once.');
end
end

function local_require_positive_scalar(value,identifier,label)
local_require_finite_scalar(value,identifier,label);
if value <= 0
    error(identifier,'%s must be strictly positive.',label);
end
end

function local_require_finite_scalar(value,identifier,label)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
    error(identifier,'%s must be a finite real numeric scalar.',label);
end
end
