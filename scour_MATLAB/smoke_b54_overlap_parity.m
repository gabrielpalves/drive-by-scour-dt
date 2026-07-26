function smoke_b54_overlap_parity
%SMOKE_B54_OVERLAP_PARITY Unit test for the governing ballast-patch rule.
%
% The fixture is mirrored exactly by check_b54_overlap_parity.py.  When the
% B54_PARITY_OUT environment variable is set, resolved multipliers are saved
% for a byte-independent numerical cross-language comparison.

Track.Sleeper.Tnum = 9;
Track.Sleeper.num_app = 2;
Track.Sleeper.num_onbeam = 5;
Track.Sleeper.num_aft = 2;
Track.Sleeper.spacing = 1;
Track.Pad.Prop.k = 11; Track.Pad.Prop.c = 12;
Track.Ballast.Prop.k = 21; Track.Ballast.Prop.c = 22;
Track.BallastOnBeam.Prop.k = 31; Track.BallastOnBeam.Prop.c = 32;

Model.Mesh.XLoc.sleepers = 0:8;
Calc.Cte.tol = 1e-9;

% Global sleeper x=0:8 maps to descriptor x=8:16 because the descriptor's
% deck starts at 10 while the global deck starts at num_app*spacing=2.
T.x_bridge_local = 10;
% Wet patch governs x=9:10. In x=11:13 it overlaps a stronger dry patch;
% abs(log(1.8)) > abs(log(0.8)), so the dry patch supplies BOTH k and c.
T.ballast_patches = [9, 13, 0.8, 2.0; 11, 15, 1.8, 0.6];
Damage.track = T;

[~,Dbg] = B54_TrackVectors(Track,Model,Calc,Damage);
k_mult = Dbg.mult_bal_k;
c_mult = Dbg.mult_bal_c;
expected_k = [1, 0.8, 0.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1];
expected_c = [1, 2.0, 2.0, 0.6, 0.6, 0.6, 0.6, 0.6, 1];
assert(isequal(k_mult,expected_k), ...
    'governing-patch stiffness multipliers differ from the pinned fixture');
assert(isequal(c_mult,expected_c), ...
    'governing patch did not supply its paired damping multiplier');

% Unequal-deviation patches must be order-invariant.
Damage.track.ballast_patches = flipud(T.ballast_patches);
[~,DbgRev] = B54_TrackVectors(Track,Model,Calc,Damage);
assert(isequal(DbgRev.mult_bal_k,k_mult) && ...
       isequal(DbgRev.mult_bal_c,c_mult), ...
    'governing-patch result depends on input row order');

out_path = getenv('B54_PARITY_OUT');
if ~isempty(out_path)
    save(out_path,'k_mult','c_mult','-v7');
end
fprintf('B54 OVERLAP PARITY: MATLAB PASS\n');
end
