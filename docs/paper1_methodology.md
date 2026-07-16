# Paper 1 â€” Methodology (draft prose)

Drafted 2026-07-09 from the implementation (`scour_MATLAB/`, `core/`, `training/`) and the
talk (`presentation/SHM_Markdown/main.md`). This is the Â§4 material for
`docs/paper1_outline.md`. **Numbers marked âš  need a final check against the code/config at
write-time** (units, sampling rate, exact ranges). Notation can be normalised when we adopt
the journal template.

---

## 4.1 Overview

The proposed drive-by scour-identification pipeline has four stages: (i) a physics-based
trainâ€“trackâ€“bridge interaction (TTBI) simulation generates vehicle acceleration responses
for a range of scour severities under environmental and operational variability; (ii) each
multi-channel response is standardised and compressed by Piecewise Aggregate Approximation
(PAA); (iii) a modular one-dimensional neural network maps the compressed signal to a scour
severity; and (iv) every architecture and sensor configuration is optimised by Bayesian
hyperparameter search and ranked by a repeated-seed robustness criterion. Stages (iii)â€“(iv)
are the object of the ablation: we vary one architectural block or one sensor at a time and
measure the effect on a fixed metric, isolating which design choices genuinely contribute.

*(Figure: pipeline schematic â€” physics â†’ PAA â†’ modular CNN â†’ robustness ranking.)*

## 4.2 Trainâ€“trackâ€“bridge interaction model

Responses are simulated with the two-dimensional TTBI framework of Cantero (TTB-2D /
VEqMon2D) [cite Cantero 2022], which solves the fully coupled vehicleâ€“trackâ€“bridge system in
the vertical plane by direct time integration of the coupled equations of motion.

**Bridge.** A âš  40 m Eulerâ€“Bernoulli beam on three supports (two equal spans), with
modulus of elasticity E = 35 GPa, second moment of area I = 0.33 mâ´, mass per unit length
âš  Ï (verify units), and 3% modal damping. Supports are modelled as vertical springs; the
central foundation carries the scour damage (Â§4.3).

**Vehicle.** A planar multi-body railway vehicle (car body, two bogies, wheelsets, with
primary and secondary suspension), using the calibrated properties of the TTB-2D
"O'Brien-calibrated" vehicle [cite]. The train comprises âš  five successive vehicles;
on-board accelerometers are placed on the **leading vehicle**, from which eight
degrees of freedom (DOFs) are recorded:

| # | DOF | Mass it sits on |
|---|---|---|
| 0 | Car-body vertical | suspended (secondary) |
| 1 | Front-bogie vertical | suspended (primary) |
| 2 | Rear-bogie vertical | suspended (primary) |
| 3 | Wheelset 1 vertical | unsuspended |
| 4 | Wheelset 2 vertical | unsuspended |
| 5 | Car-body pitch rate | suspended |
| 6 | Front-bogie pitch rate | suspended |
| 7 | Rear-bogie pitch rate | suspended |

The suspension acts as a mechanical filter: car-body channels are the most attenuated but
the most practical to instrument (power, protection, access); wheelset channels are the
least filtered but the noisiest. This trade-off is precisely what the sensor-economy
ablation (Â§5.2) quantifies.

**Track.** A layered ballasted-track model after Zhai et al. [cite Zhai 2004] â€” rail
(Eulerâ€“Bernoulli beam) on discrete rail-pad, sleeper, and ballast elements â€” couples the
wheelsets to the bridge and injects realistic high-frequency wheelâ€“rail dynamics.

**Rail irregularity.** A single fixed measured longitudinal-level profile is used for every
passage. **Scope note (report honestly):** rail-profile irregularity is therefore *not* a
source of operational variability in this study; it is held constant so that the recorded
variability is attributable to the controlled factors of Â§4.4. Rail-profile degradation as
an environmental factor is addressed in the follow-up digital-twin work.

## 4.3 Scour damage model

Scour removes soil around the foundation, reducing its vertical and lateral support
stiffness and lowering the global natural frequencies [cite Prendergast & Gavin; Kamariotis
2024]. Consistent with the drive-by scour literature [cite Fernandes 2024, 2025], scour at
the central foundation is idealised as a reduction of the support's vertical stiffness,

  k_v(d) = (1 âˆ’ d) Â· k_v0,  with healthy k_v0 = 3.44 Ã— 10â¸ N/m,

where d âˆˆ [0, 60%] is the scour severity. The severity is discretised in 1% steps, yielding
**61 damage states** (0% healthy â†’ 60%). This spans early, hard-to-detect damage (the
regime of practical interest) through severe loss of support.

## 4.4 Environmental and operational variability, and datasets

To prevent the network from learning a single idealised operating point, each of the 200
passages per damage state is simulated under randomised environmental and operational
variability (EOV):

- **Measurement noise** â€” see the v2 addendum Â§A.4: the legacy model was *multiplicative*
  (std = 5%Â·|signal|), applied to the **wheel channels only**, in the time domain before the
  space interpolation. In the current (v2) design, generation is **noise-free** and noise is a
  configurable **load-time** observation model (this bullet is superseded).
- **Temperature** â€” sampled in âš  [3, 33] Â°C, entering through a temperature-dependent
  modulus of elasticity E(T) (âš  âˆ’0.3%/Â°C about 15 Â°C).
- **Vehicle-property variability** â€” per-vehicle randomisation of âš  three mechanical
  properties (mass/suspension), reflecting load and maintenance state.
- **Train speed** (second dataset only) â€” sampled per passage in âš  [70, 90] km/h.

Two datasets isolate the effect of speed variability, the hardest confounder:

- **D1** â€” noise + temperature + vehicle-property variability (constant nominal speed).
- **D2** â€” D1 **plus** per-passage train-speed variability.

Each dataset contains 61 states Ã— 200 passages = âš  12,200 labelled passages per channel.
Each passage is the leading-vehicle DOF time history over the bridge crossing (plus a short
approach/exit for free vibration), cropped to a fixed spatial window and âš  sampled at
~1 kHz.

## 4.5 Signal preprocessing

Each channel is standardised (zero mean, unit variance) using statistics fitted **only on
the training partition** (fixed 80/20 split, seed 42) to prevent information leakage, then
compressed by **Piecewise Aggregate Approximation (PAA)** to a fixed length of 512 segments:
the sequence is partitioned into equal windows and each is replaced by its mean. PAA acts as
a structural low-pass filter â€” it smooths the high-frequency, high-energy rail-corrugation
and wheelâ€“rail noise while preserving the low-frequency deflection-basin / stiffness-loss
signature â€” and reduces the sequence length by roughly an order of magnitude, cutting
training cost [cite Fernandes 2025 for the PAA-as-EOV-filter precedent]. A continuous
wavelet transform (CWT, Morlet, âš  64 scales) branch is included as a two-dimensional
preprocessing comparator.

## 4.6 The ablated architecture

All architectures share a modular one-dimensional backbone with three independently
toggle-able blocks, so a single implementation spans the ablation grid:

- **Convolutional backbone.** A stack of 1-D convolutionâ€“ReLU(â€“max-pool) layers extracts
  local spatio-temporal features from the (channels Ã— 512) input; depth, width, and kernel
  sizes are hyperparameters (Â§4.7).
- **Space2Vec (optional).** A learnable spatial-position embedding (a linear plus a set of
  periodic components of the normalised along-bridge coordinate), concatenated to the input
  channels â€” grounding the vibration to physical position on the span.
- **LSTM (optional).** A recurrent block adding temporal context across the sequence.
- **N-HiTS multi-rate pooling (optional).** The physics-matched block: the feature sequence
  is sub-sampled at several rates and the views concatenated, so the network simultaneously
  represents fast, localised events and slow, global trends.

**The inductive-bias argument.** The drive-by signal is intrinsically two-timescale:
high-frequency transients from wheelâ€“rail contact and track irregularity, superimposed on
the low-frequency modal sag induced by foundation stiffness loss â€” the component that
actually encodes scour. N-HiTS multi-rate pooling matches this structure by construction:
the fine pooling rate preserves the sharp wheel-impact features while the coarse rates
expose the global deflection trend, letting the classifier read scour from the slow
component without discarding the fast one. This is the paper's central claim, tested in Â§5.1.

**Head and target.** A dense head maps the pooled features to **61 ordinal outputs**. Rather
than treat the 61 states as unordered classes, we regress the (discretised) severity and
score with **mean squared error on the state index**, which preserves the physical damage
ordering: predicting 41% for a true 42% is nearly free, whereas 5% for a true 55% is heavily
penalised. This ordinal formulation is the natural bridge to the continuous multi-foundation
regression of the follow-up work.

## 4.7 Hyperparameter optimisation

For every architecture Ã— sensor configuration, up to âš  26 hyperparameters (learning rate,
weight decay, per-layer filter counts and kernel sizes, pooling flags, dense widths and
dropout, and â€” when active â€” LSTM size and N-HiTS pooling rates) are optimised with Optuna's
**multivariate Tree-structured Parzen Estimator (TPE)**. TPE natively handles the
*conditional* search space (blocks and their parameters appear or disappear with the
architecture flags), which fixed Gaussian-process surrogates do not. Each study runs âš  200
trials with 25% random start-up and successive-halving pruning of unpromising trials; the
objective is the best validation MSE (Â§4.6) over âš  50 epochs with early stopping.

## 4.8 Robustness-based model selection

A single Optuna champion is the *luckiest* trial, which over-states performance and hides
fragility. We therefore re-train each champion over **30 random seeds** and rank
configurations by the 95% upper confidence bound on the mean MSE,

  UCBâ‚‰â‚… = MSEÌ„ + 1.96 Â· Ïƒ_MSE / âˆš30,

rewarding models that are *reliably* good rather than occasionally excellent. We additionally
report a **collapse-rate** â€” the fraction of the 30 seeds whose MSE exceeds a physical
tolerance (a model that has failed to learn the damage ordering) â€” which exposes
configurations, typically single weak sensors, that a point estimate would wrongly endorse
(Â§5.2). Median MSE with inter-quartile range and the collapse-rate are the reported summary
statistics.

---

### Verify-before-submission checklist
- **âš  Deck mass / fundamental frequency (HIGH PRIORITY).** A03 sets `Ï=9.6` and B43 sets
  `A=1`, so deck mass/length = 9.6 kg/m; in isolation this implies a fundamental far above a
  real bridge (~100+ Hz vs a few Hz). Confirm the as-built (deck+track+ballast) fundamental
  from the model's `B09`/`B56` output is physically representative before submission, and
  report it. This underpins the scourâ†’frequency credibility of the whole method.
- Ï/A value + units; sampling rate; signal crop window length; effective spans (mesh snap:
  L40â†’19.8/20.1, L60â†’20.1/19.8/20.1, L99.6â†’4Ã—24.9).
- Exact EOV ranges: temperature range and E(T) law, which/how-many vehicle properties, speed
  range â€” against the FINAL regenerated (noise-free, per-state-EOV) datasets.
- N vehicles (Nveh=5) and which vehicle carries the sensors (`AcelPrimVag` = leading vehicle).
- Optuna: exact HP count, trials (100 multi-damage / more for single-scour), epochs, pruner;
  CWT scale count; PAA n_segments = 512.
- Confirm the "O'Brien-calibrated" vehicle citation and the Zhai track reference; k_v0 value/units.

---

# v3 addendum (2026-07-15) — the staged multi-damage regression campaign

The §4 above documents the single-scour **classification** study, which is now the paper's
*architecture-selection* stage. The paper's spine is the **staged multi-damage regression**
campaign below. Supersedes the v2 addendum. Fold into §4 when porting to the journal
template. Every number here is cited, derived-from-cited (marked INFERENCE), or flagged ⚠
for verification — nothing is a bare assumption.

## A.1 From classification to continuous regression

The 61-ordinal-class formulation (§4.6) becomes **continuous multi-output regression**: one
head per scoured pier (severity %) and one per abutment bearing (seized-%), MSE loss, no
discretisation (`core/task.py` `task='regression'`; heads laid out `[scour…, bearing…]`).
Why: (i) several independent piers make a joint-class scheme combinatorial; (ii) continuous
severity is the state a digital twin tracks; (iii) per-head error and a direct
**localisation** read come for free. Backbone and PAA front end are unchanged — only the
head and loss differ, so the classification→regression comparison is controlled.

## A.2 The ablation ladder — one factor per rung

The organising contribution. The architecture is selected once (all arms, at `s0_scour`)
and then **fixed**, so every later rung changes exactly one scientific factor and any
degradation is *attributable*. The ladder follows the questions a bridge maintainer
actually asks — *can I localise scour? does a bearing fool me? a crack? both? does rail
roughness? track damage? the train itself?* — then repeats the key rungs at scale.

| Rung | Adds | Geometry |
|---|---|---|
| `s0_scour` | scour only — baseline + architecture selection | L60 / 3-span, piers 2,3 |
| `s11_bear` | + bearing (**head**) | " |
| `s12_crack` | + crack (nuisance, no bearing) | " |
| `s13_bearcrack` | + bearing + crack = **all bridge damages** | " |
| `s14_prof` | + rail profile (FRA-4, per-state) = **the roughness rung** | " |
| `s15_track` | + track-layer damage (ballast / hanging sleepers / pads) | " |
| `s16_all` | + wheel OOR & flats = **all damages** | " |
| `s21_scour4` | scour only | **L99.6 / 4-span**, piers 2,3,4 |
| `s22_bearcrack4` | + bearing + crack = all bridge damages | " |
| `s23_all4` | all damages | " |

**Heads = scour + bearing only.** Crack, rail profile, track-layer and wheel damage are
**nuisances**: randomized, logged, never estimated — the network must be *invariant* to
them. Three decisions worth defending explicitly:

- **Crack is a nuisance, not a head.** It does not share scour's support-stiffness pathway
  (bearing does), and — the stronger argument — **cracks are visually inspectable while
  submerged scour is not**. Drive-by monitoring earns its keep precisely on the damage
  inspection cannot see; the crack's job here is to try to *fool* the scour estimate.
- **The rail profile gets its own rung.** It is a *condition*, not a damage: every track has
  irregularity. A fixed baseline profile runs through `s0`–`s13` so those isolate
  bridge-damage interactions, and roughness then enters *alone* at `s14_prof` — the single
  most important open question after the first (deprecated) scale-up collapsed.
- **"All damages" is split** into track (`s15`) then train (`s16`), so a degradation is
  attributable to the rail or the vehicle rather than to an undifferentiated kitchen sink.

Metrics per rung: per-pier MSE, aggregate MSE, **localisation accuracy**, and (from `s11`)
bearing MSE plus a **scour↔bearing leakage** report (false-scour-from-bearing = the
safety-critical direction).

## A.3 EOV as literature-anchored domain randomisation

**Draw frequency.** Persistent *conditions* — the track-profile realisation, any crack, and
the track-layer state — are drawn **once per damage state** and held across that state's 50
passages, with a small per-passage jitter. Track geometry evolves over MGT, not between
trains (EN 13848-2 pass-to-pass repeatability ≤ 0.5 mm; Sato/Shenton degradation), and
published drive-by ML holds the profile fixed per scenario (Locke 2020; NuBe-DBBM /
Sarwar & Cantero; Fernandes). Only per-passage *operational* variability is redrawn every
run: speed [70,90] km/h, temperature [3,33] °C via E(T), vehicle properties, and wheel
damage (a different train of the fleet each passage). This corrects the first scale-up
pilot, which redrew a fresh profile+class and a fresh crack **every passage** — physically
indefensible, and the direct cause of its sprung-channel collapse.

**Rail profile.** FRA **class 4** — the roughest geometry permissible at 70–90 km/h; classes
5–6 are premium track, unrealistically smooth for a scour-prone regional line. One PSD
realisation (phases seeded) per state, plus 0.5 mm additive per-passage jitter.

**Crack.** Prevalence **0.25** — spans carrying a *macroscopic* EI loss in our Sinha 5–30%
band are ~20–30% of an aged concrete/composite inventory (marginal non-structural cracks
reach 70–80% but produce no modelled EI drop). Severity U(0.05, 0.30) EI loss. Location:
**hogging-weighted** (§A.5).

**Track layer (anchored 2026-07-15; full record in `docs/track_eov_sampling_spec.md`).** The
key result is the resolution of a **prevalence paradox**: field data report *~50% of concrete
sleepers show some voiding*, yet a mechanically-limiting model tops out near 9%. Both are
right, because most voids are **sub-threshold**. The void taxonomy: **< 1.0 mm** is
accommodated by rail/fastener elasticity; **1.0–2.5 mm** is the *dynamic-impact* threshold (a
1–2 mm void raises adjacent sleeper–ballast contact force by up to **70%** and wheel loads
~**85%**); **> 2.5 mm** is critical. Only 10–20% of visibly-settled sleepers exceed 1.5–2.0 mm
⇒ **5–10% of sleepers are *impactfully* unsupported**. Resulting spec, per 100 m (rates
**scale with the modelled window** — 120 m at L60, 159.6 m at L99.6):

| Quantity | Value | Basis |
|---|---|---|
| unsupported-sleeper groups | **Poisson λ = 3.0 /100 m** | INFERENCE: 5–10% impactful ÷ ~3 sleepers per group |
| group size | DU(1, 5) consecutive | cited (1–4 m of contiguous settlement at 0.6 m spacing) |
| ballast fouled patches | **Poisson λ = 1.2 /100 m** | INFERENCE: GPR ⇒ FI>30 on **10–20%** of route length; 15% ÷ 12.5 m mean patch |
| patch length | U(5, 20) m | cited |
| failed pads | **p = 0.02 /pad** (≈3–4 per 100 m) | INFERENCE: 0.5%/yr **incidence** × 3–5 yr renewal ⇒ 1.5–3.0% standing prevalence |
| fouling ↔ voiding | **coupled: ×3** hanging density inside a fouled patch | cited mechanism (mud pumping → loss of support; bidirectional) |
| ballast near abutment | **×3** within 20 m | cited: corrective work 4–8× more frequent; settlement rate 3–4× |
| hanging sleepers near abutment | p = 0.6 within ±15 m | cited (transition-zone density spike) |

λ = 3.0 is the **unique** value reconciling the source's recommended λ ∈ [2,3] with its own
5–10% sleeper band (λ·3/167 ⇒ λ ≥ 2.78 for ≥ 5%); Monte-Carlo verified at 5.4%
(`plotting/check_track_stats.py`). ⚠ **Verify before submission:** the pivotal "FI>30 on
10–20% of route length" traces to a non-peer-reviewed deck — re-check against the GPR
literature. Report all table rows as *derived from cited anchors*, not measured.

## A.4 Measurement noise as a load-time observation model

**Generation is noise-free.** D01 saves the **raw, un-interpolated, noise-free time-domain**
signal plus the parameters needed to rebuild the space window (`DimAcel`, `DimSpace`,
`crop_start`, `crop_end`); Python performs the interpolation and bridge crop at **load time**
(`core/dataset._raw_to_space_crop`, an exact `interp1` mirror). The measurement model
therefore lives entirely at load time and can be changed — per channel, in either domain —
**without regenerating data**. Storage is ~neutral: the space grid is 100 samples/m and
`DimSpace ≈ 2.2·DimAcel`, so the raw series is about the size of the old cropped window while
covering the whole passage.

**Why this matters (a finding worth reporting).** Adding noise *before* the time→space
interpolation is **not** equivalent to adding it after. Because `interp1` is linear, noise
injected in the time domain becomes `interp(white)` — **band-limited, coloured, and
speed-dependent** — whereas noise injected in space is white. Measured on our signals:
≈ 0.67× the variance but ≈ **1.46× the energy surviving PAA** (the colouring concentrates
energy at low spatial frequencies, exactly what PAA's averaging preserves). Same nominal 5%,
materially different perturbation. We therefore state the noise model explicitly instead of
leaving it implicit in the pipeline. The ablation injects a uniform 5% multiplicative noise
on every channel at load time; the physically-correct model for future work is an
**additive, signal-independent floor** from sensor datasheets (noise density × √bandwidth).
⚠ **EN 61373** position severities (carbody:bogie:axle) describe the vibration *environment*
for equipment qualification — range and reliability — **not** an acquisition noise floor; do
not scale noise by them.

## A.5 Damage-location priors

Three of the four families carry a location prior already: **scour** only at the modelled
piers, **bearing** only at the abutment rotational springs (both by construction), and
**hanging sleepers** with a *cited* density spike within ±15 m of the bridge transitions.
Ballast patches and pads are uniform over the modelled track — which spans the full approach
+ bridge + exit, since track defects occur off the deck too (the approach is exactly where
the vehicle is excited *before* reaching the bridge).

**Crack location — a correction worth narrating.** We first drew U(0.10, 0.90)·L and
justified it by computing the **moving-load |M| envelope** for our exact geometries
(`plotting/moment_envelope.py`): the envelope is *broad* — only ~4% (L60) / ~2% (L99.6) of
that range ever sees |M| < 35% of max, because nearly every section peaks for *some* train
position — with peaks at mid-span (1.00/0.84) versus over-pier (0.51/0.42), i.e. sagging
dominating ~2:1. That reasoning was **wrong, because the envelope is the wrong lens**: it
answers *where bending is large*, not *where concrete fails*. Real crack prevalence is
**hogging-dominated by > 4:1–5:1** — over an internal support the deck's **top fibre** is in
tension (concrete's weakest mode) *and* takes deck runoff and chlorides, whereas mid-span
soffit cracks tend to close under compression and rarely yield a macroscopic EI loss.
**Eurocode 4 mandates** analysing a cracked section over **15% of the span each side of
internal supports**. Implemented: hogging:sagging = **4:1**, jittered within **±17.5% of a
span** about the chosen section. *(Torsion is not representable in the 2-D vertical model — a
limitation, alongside pier rotation/tilt.)*

## A.6 Inter-pier scour dependence: independent for training, dependent only in the twin

Scour at the piers of one bridge is physically **dependent** — they share a flood, a reach,
and a bed. We nevertheless sample the training states with an **independent LHS** over each
pier's severity, and this is deliberate:

1. **Do not bake the prior into the likelihood.** The network learns
   `p(response | scour state)` — an *observation model*. Inter-pier correlation is
   `p(scour state)` — a *prior*. Training on correlated states teaches the network to infer
   "pier 3 is scoured because pier 2 is" **without evidence**; the twin would then apply the
   same correlated prior again through its DBN and double-count it. Keeping them separate
   yields a clean likelihood and a modular twin.
2. **It is required by the localisation claim.** The central question is *which* pier. A
   correlated training set would rarely present "pier 2 damaged, pier 3 healthy" — the exact
   discrimination under test. Independent LHS plus explicit single-pier anchors guarantee
   those corners are covered.
3. **Coverage.** A broad joint training distribution serves *any* correlation encountered at
   deployment, with no retraining.

The dependence therefore belongs to the **digital twin** (follow-up work), where we model it
**mechanistically** rather than as an imposed copula: all piers respond to the *same*
flood-shock driver with pier-specific sensitivity, so the correlation emerges from shared
hydrology. Since the correlation strength at our ~20–25 m pier spacings is not well
established, it is reported as a **swept sensitivity** (independent → strongly coupled)
against the decision it actually changes, rather than asserted as a number.

## A.7 Champion metric and provenance policy

Rank by **median** aggregate MSE with IQR and a **collapse-rate** (fraction of runs failing to
learn the ordering), with a UCB variant `MSĒ + 1.96·σ/√n`. The multi-damage grid runs **100
Optuna trials** (multivariate TPE, 25% random start-up, successive-halving pruning) × **3
independent seeds** per config; the train/val split is fixed (seed 42) so seed spread isolates
initialisation/HPO variance. A seed-aggregated median leaderboard is the paper-facing table.
Phase 1 sweeps the 8 single channels; Phase 2 runs the auto-selected best pair **plus a
designed mixed unsprung+sprung pair**.

**Provenance:** re-ablations start **from scratch** (tagged studies), never by extending
existing Optuna studies; the noise mode enters the study name, so a noise A/B on identical
data trains separate studies rather than silently resuming.
