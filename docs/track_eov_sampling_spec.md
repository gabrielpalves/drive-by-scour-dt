# Track-layer damage EOV — sampling specification (from NotebookLM deep research)

Transcribed 2026-07-09 from the NotebookLM answer (notebook "Probabilistic Graphical
Models for Predictive Digital Twins at Scale", 106 sources incl. the Track-Layer Deep
Research report; answer also saved as a note in the notebook). Items the model itself
marked *(Extrapolation)* are flagged.

**VERIFICATION PASS DONE 2026-07-09** (second NotebookLM answer: per-number quote +
source mapping against the deep-research report "Track-Layer Degradation in
Train-Track-Bridge Interaction Models..."). CITED (solid): void-depth range + lognormal
params; 50% voiding prevalence (Augustin et al.; Li & Sun); ballast dry/wet multiplier
ranges; GRF theta_x = 3-15 m (typ. 10); patch length U(5,20) m; pad Weibull(1.8, 2.2);
damping [0.8,1.2]; k->0 pad failure; P=0.5% of fastening positions PER YEAR; ARIMA(5,1,0);
max 3 consecutive failed pads; **hanging-sleeper density spikes within 15 m of bridge
transitions**. EXTRAPOLATED (assumptions, state as such): 1-3 sleeper groups /100 m;
group sizes above 5 (cited range = 1-5 consecutive, 5 = wheel-load critical limit);
1-2 ballast patches /100 m; patch upper bound 25 m (cited 20 m); 0.83 failed pads/100 m
(arithmetic); 2 m clustering near joints. Corrections are applied in the table below.

## Sampling specification (per 100 m of track; sleeper spacing 0.6 m ⇒ ~167 sleepers)

### (a) Hanging / unsupported sleepers
- **Representation:** unilateral (non-linear, bilinear) spring contact — the vertical
  ballast reaction F_b,i drops to 0 while the relative displacement ≤ void depth g_v.
  *Linearised fallback for our per-passage linear solver: zeroed/strongly-reduced
  ballast spring under the affected sleepers.*
- **Severity:** void depth g_v = 0.5–3.0 mm (max boundary 5.0 mm);
  distribution **Lognormal**: ln(g_v) ~ N(−0.2, 0.4).
- **Count:** 1–3 clustered groups per 100 m *(Extrapolation: field data shows up to
  ~50% of concrete sleepers have some voiding globally — Augustin et al.; Li & Sun —
  but impactful gaps are highly clustered)*.
- **Clustering (VERIFIED):** group size = **Discrete Uniform 1–5 consecutive sleepers**
  (cited; 5 consecutive = the critical limit for wheel-load magnitude). Values up to 10
  were an extrapolation — use only as a flagged transition-zone extreme.
- **Location (CITED, design input):** density **spikes within 15 m of bridge
  transitions** — sample hanging sleepers with elevated probability in the approach/
  transition zones at the abutments, NOT uniformly. NOTE the interaction: this puts
  track damage adjacent to the abutment BEARING targets — watch bearing-head leakage.

### (b) Ballast degradation & fouling
- **Representation:** state-dependent multipliers on nominal vertical stiffness (η_k)
  and viscous damping (η_c) of the ballast layer.
- **Severity:** dry/compacted-fouled state: η_k ∈ [1.2, 2.0], η_c ∈ [0.4, 0.8];
  wet/saturated state: η_k ∈ [0.7, 0.9], η_c ∈ [1.5, 4.0]; continuous Uniform within
  the state's range. (Note fouling can STIFFEN dry ballast while wet fouling softens
  and adds damping — sample the state, then the multipliers.)
- **Count:** 1–2 patches per 100 m *(Extrapolation; the cited anchor is tamping cycles
  every 20–35 MGT of traffic)*.
- **Clustering (VERIFIED):** patch length **U(5, 20) m** (cited; the earlier 25 m upper
  bound was an extrapolation); spatial mapping = 1D Gaussian (or exponential) random
  field with horizontal scale of fluctuation (correlation length) θ_x = 3–15 m,
  typically 10 m (cited).

### (c) Rail-pad deterioration
- **Representation:** aging = progressive multipliers on pad stiffness (χ_pad) and
  damping (β_pad) (source suggests a Prony series for frequency dependence — beyond
  our constant-parameter pads; use the constant multipliers); failure = zero-stiffness
  spring (k_p,i → 0).
- **Severity:** aging χ_pad ∈ [1.0, 3.5] with **Weibull(λ = 1.8, k = 2.2)** (the
  [1.0, 3.5] range is a synthesis of two cited ranges [1.2, 3.5] and [1.0, 3.0]);
  damping β_pad ∈ [0.8, 1.2]; failed pad: k = 0 with per-pad probability P = 0.005.
  **NOTE (verified): P = 0.5% of fastening positions PER YEAR** — using it as a
  snapshot Bernoulli assumes ~1 year between inspections; state that assumption.
- **Count:** aging applies globally across all ~167 sleepers/100 m; failures
  ≈ 0.83 pads per 100 m *(Extrapolation: 167 × 0.005 Bernoulli)*.
- **Clustering:** sleeper-to-sleeper aging variation via an ARIMA(5,1,0) spatial noise
  model, clustered within ~2 m of rail joints; failures independent, capped at
  max 3 consecutive.

## (i) Channel / frequency sensitivity
- **Unsprung (axle-box/wheel):** 300–3000 Hz, dominated by rail-pad state (pad dynamic
  stiffness controls the rail's pinned-pinned resonance, ~800–1200 Hz).
- **Sprung (bogie/car-body — our champion channels):** suspension low-passes above
  ~30 Hz; **completely insensitive to rail-pad aging**; they capture 0.5–30 Hz —
  bounce/pitch modes (10–30 Hz) excited by low-frequency harmonics of severe
  hanging-sleeper gap closures, plus the global profile dips associated with ballast
  fouling (0.5–30 Hz).

## (ii) Confounding verdict — the key finding
**Mimicry is real and lands in our band.** Localized ballast fouling and hanging-sleeper
groups cause differential settlement → permanent depressions in the track profile →
car-body bounce/pitch excitation at **0.5–15 Hz, the same band as pier-scour / bearing
deck-deflection signatures (1–15 Hz)**. Classifiers trained without track defects can
misclassify track settlement as structural scour (false alarms).

**Separable?** Yes, per the sources, via:
1. **Domain randomization** — inject these damages into the TTBI training set as EOVs
   (exactly our crack/profile mechanism) so the network learns invariance to localized
   track anomalies and keys on the longer-wavelength global deflection signature;
2. Signal decoupling (Augmented Kalman filter / SMC "apparent profile" isolation /
   band-stop around the sleeper-passing frequency f = v/0.6) — the non-ML alternative.

**Implication for us:** the confounding pathway is largely PROFILE-mediated (localized
depressions), which our stationary `psd_fra` PSD randomization does NOT cover — so the
proposed track-EOV stage adds genuinely new coverage, it is not redundant with the
profile EOV. This supports the user's decision to add a dedicated stage.

## Caution — the NotebookLM "how does N-HiTS distinguish them" answer
A follow-up NotebookLM answer explained the separation via the ORIGINAL N-HiTS
architecture (Challu et al.: stacked blocks, backcast-residual subtraction, hierarchical
top-down interpolation). **Our model is NOT that**: it is a CNN + `MultiRatePooling1D`
(parallel max-pools at rates 1/2/4, concatenated — core/models.py); no stacked blocks,
no backcasts. Usable claims for the paper: (1) multi-rate pooling gives the head
simultaneous coarse/fine views, favouring long-wavelength global-deflection content;
(2) domain randomization forces invariance — a HYPOTHESIS this stage tests (success
criterion: flat scour MSE + low false-alarm rate under track EOVs). Do NOT quote the
hierarchical-interpolation mechanism.

## Implementation notes (TTB-2D mapping — to design before the stage)
- Our track layers (`TrackProp_Zhai…`) are currently SCALAR per-layer properties applied
  uniformly to every sleeper; per-sleeper multipliers (needed for all three damages)
  require threading per-sleeper stiffness vectors into the track/model matrices
  (B51/B54 + Python b51/b54). This is the main implementation cost of the stage.
- The bilinear gap contact (true hanging sleeper) is nonlinear; start with the
  linearised zero/reduced-stiffness version (standard in the literature for moderate
  voids) and note the simplification.
- Sample per passage (like crack/profile) unless decided otherwise; log draws in the
  .mat like `crack_log`/`profile_log` (they are nuisances, NOT labels).
- Keep MATLAB and Python (TTBI_2D) implementations mirrored — parity audited 2026-07-09.
