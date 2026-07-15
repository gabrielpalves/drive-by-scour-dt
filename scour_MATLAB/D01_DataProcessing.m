function [data] = D01_DataProcessing(i, j, Sol, Train, Calc, Damage, data)
%% RAW (time-domain) save — "Option B", 2026-07-14
% Stores the vehicle responses EXACTLY as the solver produced them (TIME domain,
% un-interpolated, noise-free) plus every parameter Python needs to reproduce the
% space-domain transform and the bridge crop at LOAD time.
%
% WHY: the time->space transform is a LINEAR interpolation, so noise added BEFORE
% it is band-limited/coloured and speed-dependent, while noise added AFTER it is
% white — the two are NOT equivalent (verified: ~0.67x variance, +0.77 lag-1
% autocorrelation, ~1.46x the energy surviving PAA). Saving the raw signal keeps
% BOTH options open forever: the measurement model lives entirely at load time and
% can be changed without ever regenerating the data.
%
% STORAGE is ~neutral: the space grid is 100 samples/m and DimSpace ~= 2.2*DimAcel,
% so the raw time series is about the size of the old cropped space window — but it
% covers the WHOLE passage (approach + bridge + exit) instead of just the crop.
%
% Python mirror: core/dataset._raw_to_space_crop  (np.interp == interp1 'linear').

% ---- Raw time-domain channels (leading vehicle) -------------------------
% Rows: A(1:3) = car-body / front-bogie / rear-bogie VERTICAL acceleration;
%       V(4:6) = car-body / front-bogie / rear-bogie PITCH RATES (V(1:3) are
%                vertical velocities — do NOT use them for pitch);
%       acc_under(1:4) = wheel (unsprung) vertical accelerations.
data.AceleracaoPrimVag{i,j} = Sol.Veh(1).A(1:3,:);
data.PitchPrimVag{i,j}      = Sol.Veh(1).V(4:6,:);
data.AcelRodaPrimVag{i,j}   = Sol.Veh(1).acc_under(1:4,:);

% ---- Legacy baked noise: OFF by policy (use_signal_noise=false => desvio=0) ----
% Kept CONDITIONAL so the legacy pipeline stays reproducible, and so no random
% numbers are drawn when it is off (an unconditional randn would perturb the RNG
% stream). When it IS on, the noise now lands in the TIME domain of the SAVED raw
% signal — the physically correct place (sensors sample in time) — and Python's
% interpolation colours it exactly as the old MATLAB pipeline did.
if Damage.desvio > 0
    data.AcelRodaPrimVag{i,j} = data.AcelRodaPrimVag{i,j} + Damage.desvio * ...
        data.AcelRodaPrimVag{i,j} .* randn(size(data.AcelRodaPrimVag{i,j}));
end

data.Velocidade{i,j} = Train.vel;
data.Posicao{i,j}    = Train.vel*(Calc.Solver.num_t/1000);   % distance travelled [m]

% ---- Everything Python needs to rebuild space domain + bridge crop -------
% Space domain is 100 samples/m. The time->space map is UNIFORM (constant speed
% per passage), exactly as the legacy code did it:
%     xx = linspace(1, DimSpace, DimAcel);  xi = 1:DimSpace;  interp1(xx, y, xi)
% Crop window = 10 m approach skip + bridge span + 18.31 m crossing/after.
% NOTE: with profile_mode='fixed' (Type 2) B19 replaces Calc.Profile wholesale and
% L_bridge reverts to the stored 39.9 m — the documented legacy 40 m window. It is
% recorded per passage below so the crop is always self-describing.
DimAcel  = size(Sol.Veh(1).A, 2);
DimSpace = round(data.Posicao{i,j}*100);

crop_start  = 1001;                                   % ~10 m approach skip
bridge_samp = round(Calc.Profile.L_bridge) * 100;     % bridge span [samples]
crop_end    = min(crop_start - 1 + bridge_samp + 1831, DimSpace);

data.DimAcel(i,j)      = DimAcel;
data.DimSpace(i,j)     = DimSpace;
data.crop_start(i,j)   = crop_start;
data.crop_end(i,j)     = crop_end;
data.bridge_samp(i,j)  = bridge_samp;
data.L_bridge_eff(i,j) = Calc.Profile.L_bridge;

end
