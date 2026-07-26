function [V,Dbg] = B54_TrackVectors(Track,Model,Calc,Damage)
%B54_TRACKVECTORS Resolve track-EOV descriptors into per-sleeper properties.
%
% This is the testable production helper used by B54_ModelMatrices.  It was
% extracted verbatim from B54's former local_track_vectors in audit r4 so the
% governing-patch overlap rule can be parity-tested against the Python mirror.
%
% COORDINATE FRAME: descriptors carrying x_bridge_local are expressed in
% A00's bridge-local window frame.  Descriptors without it remain global.

n = Track.Sleeper.Tnum;
x = Model.Mesh.XLoc.sleepers;            % [1 x n] sleeper positions [m]
x_deck0 = Track.Sleeper.num_app * Track.Sleeper.spacing;
x_deck1 = (Track.Sleeper.num_app + Track.Sleeper.num_onbeam - 1) * ...
    Track.Sleeper.spacing;
off = 0;                                 % global -> descriptor-frame shift
if isfield(Damage,'track') && ~isempty(Damage.track) && ...
        isfield(Damage.track,'x_bridge_local') && ...
        ~isempty(Damage.track.x_bridge_local)
    off = x_deck0 - Damage.track.x_bridge_local;
end
x = x - off;
mult_bal_k = ones(1,n); mult_bal_c = ones(1,n);
mult_pad_k = ones(1,n); mult_pad_c = ones(1,n);
KILL = 1e-6;    % "removed" support: not exactly 0 to keep Kg well-posed

if isfield(Damage,'track') && ~isempty(Damage.track)
    T = Damage.track;
    if isfield(T,'ballast_patches') && ~isempty(T.ballast_patches)
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
