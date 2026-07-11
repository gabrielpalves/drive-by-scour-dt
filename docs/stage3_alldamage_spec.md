# Stage 3 — "all-damage" EOV stage (build spec)

Decided 2026-07-09 (user). Purpose: a CURIOSITY/ROBUSTNESS arm for the ablation paper —
train the champion with the most complete damage/EOV set the TTB-2D formulation supports
and see whether scour estimation survives. The DT paper (Paper 2) keeps the lean EOV set;
Stage 3 is kept in the ablation paper only if the results earn it.

## Design decisions
- **Preset name:** `stage3_alldamage` (A00_Run.m + comprehensive_ablation_multidamage.py).
- **Geometry: L=60 / 3-span / scour supports [2 3]** — SAME bridge as Stages 0/1 so the
  marginal-effect chain is clean: Stage 0 (clean, agg 0.757) → Stage 1 (+bearing heads)
  → Stage 3 (+track/vehicle EOVs). Stage 2 (L=99.6/4-span) remains the scale-up axis;
  do not mix the two axes.
- **Targets/heads:** unchanged from Stage 1 — scour [2,3] + bearing [left,right] (4 heads).
  ALL new damages are per-passage randomized NUISANCES (logged, never labelled).
- **Architecture:** champion-only (PAA_NHiTS) per the fixed-architecture policy.
- **Damage/EOV set:**
  1. crack (existing toggle) + `psd_fra` profile (existing);
  2. **ballast patches** — per-sleeper stiffness/damping multipliers; state-dependent
     (dry: η_k∈[1.2,2.0], η_c∈[0.4,0.8]; wet: η_k∈[0.7,0.9], η_c∈[1.5,4.0]); 1–2
     patches/100 m, length U(5,20) m, GRF/exponential correlation θ_x≈10 m
     (see docs/track_eov_sampling_spec.md — verified numbers);
  3. **hanging sleepers** — linearized (ballast k → ~0 under affected sleepers); groups
     of Discrete-Uniform 1–5 consecutive sleepers, 1–3 groups/100 m, location density
     SPIKED within 15 m of bridge transitions (cited) → interacts with bearing heads;
  4. **rail pads** — aging χ_pad∈[1.0,3.5] Weibull(1.8,2.2), β_pad∈[0.8,1.2], failures
     P=0.005/pad (≈1-yr snapshot assumption), max 3 consecutive. **Designed NEGATIVE
     CONTROL**: pad content is 300–3000 Hz — above the sprung-channel cutoff (~30 Hz)
     AND above the PAA-512 Nyquist (~85–110 Hz @ 70–90 km/h) → predicted effect ≈ 0;
     confirming that validates the band-separation argument.
  5. **wheel flats — LITERATURE-VERIFIED 2026-07-09** (NotebookLM deep research #2;
     prompt in docs/deep_research_prompt_wheel_flats.md). Haversine dip in B25/b25,
     period 2πR (R≈0.46 m), random phase. Sampling (cited unless marked):
     ~12% of in-service wheels carry a flat → generative model: per-bogie independent
     slide events q=0.171, leading axle always flats, trailing axle w.p. 0.40
     [correlation split EXTRAPOLATED from braking mechanics; reproduces the 12%
     wheel marginal and P(bogie2|bogie1)≈0.15]. Flat TYPE: fresh w.p. 0.125 | flat
     (1.5% vs 10.5% split — flats run in within km). Fresh: L~U(10,35) mm, depth
     d=L²/(8R) (rigid chord sagitta); run-in: L~U(30,60) mm, d=L²/(16R) (contact-
     patch-filtered — the recommended kinematic-trajectory approximation).
     Condemning anchors: EN 15313/UIC 510-2 60 mm; AAR 50.8 mm. Repetition frequency
     6.7–8.6 Hz at 70–90 km/h = IN the scour band (massive interference verdict);
     70–90 km/h is also the transcritical onset → momentary contact loss expected
     and physical (matches our smoke observation).
  6. **low-order polygonization (OOR) — MUST be modelled separately (cited verdict).**
     Continuous per-wheel sinusoid added to the wheel irregularity:
     Δr·cos(n·x/R + φ); P(wheel polygonized)=0.30, order n~DU(1,5), amplitude
     ln(Δr[m])~N(−10.0, 0.5) clipped to [10, 120] µm (cited service range).
     Orders 1–5 at 70–90 km/h excite 6.7–43.5 Hz continuously — direct overlap
     with the sprung channels and bridge modes; the continuous (non-transient)
     in-band forcing is the hardest wheel-side confounder for scour.
- **Logging:** per-passage draws → `track_log` (per-damage params) + `oor_log`, saved in
  each NNNN.mat like crack_log/profile_log. Nuisances, NOT labels.

## PAA band analysis + pre-registered contingency
PAA-512 spatial segments: Stage-0/1 window 58.31 m → 0.114 m/seg → Nyquist ≈4.4 cyc/m
≈ 85–110 Hz temporal (70–90 km/h). Scour/bearing (1–15 Hz), confounder band (0.5–30 Hz),
wheel-flat fundamental (~7.7 Hz), sleeper-dip wavelengths (0.6–3 m) ALL survive; only
>~100 Hz impact content is lost (which the sprung channels don't see anyway).
**Contingency if Stage-3 scour MSE inflates vs the Stage-1 baseline:** FIRST re-run the
champion config with `n_segments` = 1024, then 2048 (Nyquist ×2/×4); only if resolution
moves the needle, consider RAW. One-config re-runs, not a new ablation.

## Explicitly OUT (model-scope limitations, for the paper)
- **Pier rotation / tilt from scour** — piers are boundary springs, not members; a deck-
  node rotational spring would model deck-pier fixity, NOT pier tilt (deceptive realism).
  Needs a frame/3D formulation → future work + limitation sentence.
- Lateral/torsional damage, SSI nonlinearity, prescribed support settlement (only
  approximable via profile), prestress loss (confounds with temperature; parked).
- Suspension/vehicle-mass drift: ALREADY covered by use_vehicle_variability (say so in
  the paper — free robustness claim). Sensor faults: DT layer (sensor_health).

## Build status update (2026-07-10): wheel spec LITERATURE-VERIFIED + re-implemented
Deep research #2 (wheel flats/OOR) done; A00/B25/b25/damage_config updated to the
verified spec (fresh-vs-run-in flats with per-type depth laws, per-bogie slide-event
correlation MC-verified to the cited targets [marginal 0.120, P(bogie2|bogie1)=0.17,
fresh/run-in 0.015/0.105], polygonization added as a separate continuous nuisance).
Descriptor format changed: `oor_flats` [veh,wheel,len,depth,phase] + `oor_poly`
[veh,wheel,n,amp,phase] (old 4-col `oor` removed). End-to-end smoke RE-PASSED
2026-07-10: healthy parity bit-exact; fresh-flat wheel 146x, its bogie 3x, sprung
channels CarBody_V 3.6% / RearBog_V 25.7% / CarBody_P 1.6% (polygonization raises the
sprung-channel response vs flats-only, as expected for continuous in-band forcing).

## Build status (2026-07-09): BUILT + SMOKE-TESTED (Python end-to-end)
Implemented in both languages (B54/b54 per-sleeper vectors, B25/b25 wheel flats,
B00/b00 threading, A00 preset + sampling + logs, make_damage track/oor args,
ablation STAGE entry). Unit smokes pass (patch/hanging/pad indexing exact; flat
depth = chord relation). End-to-end `_stage3_smoke.py` (3 live passages) PASSED:
**healthy parity bit-exact (max|A−B| = 0)**; damaged channel pattern as predicted —
flat-carrying wheel 272×, its bogie 5×, sprung champion channels CarBody_V 2.9% /
RearBog_V 17.9% / CarBody_P 1.0% (in-band, attenuated). NOTE: the solver reported
momentary wheel–rail CONTACT LOSS on the flat — physical for flat impacts (B66
handles it), but monitor for large flats; cap flat length if solves get noisy.
MATLAB twin `scour_MATLAB/smoke_stage3.m` pending a run on the user's local MATLAB
(expected: same parity + pattern).

## Build order (each step smoke-tested + MATLAB↔Python parity-checked)
1. Per-sleeper track-property vectors (pad k/c, ballast k/c) threaded through
   B51_RailVariables + B54_ModelMatrices (MATLAB) — THE main cost; currently scalars.
2. Python mirrors (b51/b54) + parity smoke (healthy: byte-identical to scalar path).
3. Sampling layer in A00 (patches/GRF, sleeper groups w/ transition spike, pad aging/
   failures) + numeric knobs in the preset; logging.
4. Wheel flat in B25 (MATLAB) + b25 (Python) + parity smoke.
5. `stage3_alldamage` preset (A00) + ablation STAGE entry (dataset name per case_name
   convention) + rebuild bundle for the Lab PCs.
