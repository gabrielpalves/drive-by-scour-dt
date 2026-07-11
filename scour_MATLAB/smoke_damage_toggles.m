% _smoke_damage_toggles.m — temporary smoke test for the new damage/EOV toggles
% (crack local-EI, profile intensity, psd_fra profile mode). Run from scour_MATLAB.

%% Test 1: B19 Type-2 (loaded profile) intensity capture & scaling
Calc1 = struct();
Calc1.Profile.Type = 2;
Calc1.Profile.min_dx = 0.01;
Calc1.Profile.L = 10;
Calc1.Position.x = [0 1];
Calc1.Plot.Profile_original = 0;
C_ref = B19_GenerateProfile(Calc1);

Calc2 = Calc1;
Calc2.Profile.intensity = 2.0;
C_scl = B19_GenerateProfile(Calc2);

assert(max(abs(C_scl.Profile.h - 2*C_ref.Profile.h)) < 1e-14, ...
    'T1 FAIL: Type-2 intensity scaling wrong');
fprintf('T1 OK: Type-2 profile x2 intensity (max|h| %.3e -> %.3e m)\n', ...
    max(abs(C_ref.Profile.h)), max(abs(C_scl.Profile.h)));

%% Test 2: full passage — scour + crack + psd_fra profile through B00
Damage = struct();
Damage.desvio        = 0;
Damage.scour_rates   = [0 0.30 0];   % 30% scour at the central pier
Damage.bearing_left  = 0;
Damage.bearing_right = 0;
Damage.crack_locs      = 30.0;       % crack at 30 m
Damage.crack_intensity = 0.22;       % 22% EI loss (Fernandes upper case)
Damage.crack_lc        = 0;          % single element

Train = A01_Train(80/3.6, zeros(5,3));
Track = A02_Track();
Beam_seed = struct();
Beam_seed.Prop.L = 40;
Beam_seed.Prop.num_spans = 2;
Beam = A03_Bridge(Beam_seed);

Profile_cfg = struct('mode','psd_fra','fra_class',5);
[Calc, Beam, Track] = A04_Options(Beam, Track, Profile_cfg);
[Sol, Calc, Train, Beam, Track] = B00_Calculations(Calc, Train, Track, Beam, Damage);

% crack: exactly ONE element with I reduced to 0.78*I
n_red = sum(abs(Beam.Prop.I_n - Beam.Prop.I) > 1e-12);
ratio = min(Beam.Prop.I_n) / Beam.Prop.I;
fprintf('T2: elements with reduced I = %d (expect 1); ratio = %.4f (expect 0.7800)\n', n_red, ratio);
assert(n_red == 1, 'T2 FAIL: crack did not hit exactly one element');
assert(abs(ratio - 0.78) < 1e-9, 'T2 FAIL: crack EI ratio wrong');

% profile: regenerated, non-trivial
assert(std(Calc.Profile.h) > 0, 'T2 FAIL: psd_fra profile is empty');
fprintf('T2: FRA class-5 profile std = %.4g mm over %.0f m\n', ...
    1000*std(Calc.Profile.h), Calc.Profile.L);

% solve produced a solution
assert(isfield(Sol,'Veh') && ~isempty(Sol.Veh), 'T2 FAIL: no vehicle solution');
fprintf('T2 OK: dynamic solve completed with crack + FRA-PSD profile\n');

%% Test 3: legacy path untouched — A04 with NO third argument = Type 2 fixed
[Calc3, ~, ~] = A04_Options(A03_Bridge(struct()), A02_Track());
assert(Calc3.Profile.Type == 2, 'T3 FAIL: legacy default profile type changed');
fprintf('T3 OK: legacy A04 call (2 args) still defaults to the fixed measured profile\n');

disp('ALL SMOKE TESTS PASSED');
