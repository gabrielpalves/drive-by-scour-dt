# Historical Paper 1 methodology draft — retired campaign design

> **ARCHIVAL DOCUMENT — NOT AN OPERATIONAL OR PUBLICATION-SOURCE CONTRACT.**
>
> The body below preserves the superseded multi-rung methodology draft and its
> terminology for audit history. Its stages, state counts, HPO policy, channel
> interpretation, execution blocks, qualification instructions, and statistical
> plan must not be used to launch or describe the current Paper 1 campaign.
>
> The controlling current specification is
> [`docs/paper1_campaign_plan.md`](paper1_campaign_plan.md), with the concise
> operator guide in [`README_CAMPAIGN.md`](../README_CAMPAIGN.md) and the sole
> dispatch verdict in [`docs/audit_r5_results.md`](audit_r5_results.md).
> Current manuscript claims must be taken from the authenticated campaign
> manifests, protocol descriptors, final result artifacts, and `paper1/sections/`,
> not from this historical body.
>
> Historical uses of "registered" mean prospectively specified in versioned,
> hash-identified repository source; they do not claim external preregistration.

## 1. Study question and estimands

The study asks whether vertical acceleration and pitch angular velocity
measured on a passing railway vehicle can estimate continuous, pier-specific
loss of vertical foundation-support stiffness and identify the most affected
pier in the presence of operational variability, a competing bearing
mechanism, and modeled bridge/track/vehicle nuisances. It also asks which
registered neural architecture and two-channel response subset perform best
under a fixed, compute-feasible calibration policy.

The primary response target is modeled scour support-stiffness loss in
percentage points. On bearing rungs, nominal abutment rotational fixity is an
additional regression target. Crack, rail profile, track-layer conditions, and
wheel polygonization are logged nuisance variables and are not predicted.

The study has three distinct estimand classes:

1. **Anchor/deployment selection and carried-reference performance.** Inner
   validation selects the block reference at `s0` and `s21`, and separately the
   exploratory deployment winner at `s16` and `s23`. Frozen rungs
   (`s11`–`s15` and `s22`) carry their block reference rather than selecting a
   new winner. The immutable outer test estimates eligible frozen
   winner/comparator performance under that rung's simulated distribution.
2. **L60 paired simulator-intervention contrasts.** Seven registered edges
   estimate the change in achievable scour-estimation error after adding a
   simulator intervention — a mechanism block together with, where
   applicable, its associated output heads and training task — with
   architecture, response-channel pair, hyperparameters, semantic state population,
   split, and training seeds paired as specified below. An edge is an
   intervention/task contrast, not an isolated single-mechanism effect.
3. **L99.6 scale/stress results.** These form a separate physical execution
   block and are interpreted blockwise. They are not one-factor causal
   contrasts with L60.

Because every rung trains a new model, cross-rung changes measure
**rung-specific achievable performance under retraining**. They are not
zero-shot out-of-distribution tests.

## 2. Train–track–bridge interaction model

Vehicle responses are generated with the repository's two-dimensional,
vertically coupled train–track–bridge interaction (TTBI) model, a modified
derivative of Cantero's TTB-2D tool — D. Cantero, *TTB-2D: Train–Track–Bridge
interaction simulation tool for Matlab*, SoftwareX 20 (2022) 101253,
doi:10.1016/j.softx.2022.101253 — whose vehicle equations of motion are
generated with VEqMon2D — D. Cantero, SoftwareX 19 (2022) 101103,
doi:10.1016/j.softx.2022.101103. The exact upstream base is the publication's
v1 code line at commit `28d35528ac6624200a881bcd6130382b81579a01`
(archived repository `ElsevierSoftwareX/SOFTX-D-22-00221`, GPL-3.0); the
repository-local damage mechanisms, campaign sampling rules, serialization,
and qualification gates are **not** part of that upstream release. See
`THIRD_PARTY_NOTICES.md`. The model couples:

- an Euler–Bernoulli multi-span bridge on vertical support springs;
- a layered ballasted track with rail, pads, sleepers, and ballast elements
  whose scalar nominal property values are those of
  `scour_MATLAB/TrackProp_Zhai_et_al_WithBallastOnBridge.m`, taken from
  W. M. Zhai, K. Y. Wang & J. H. Lin, *Modelling and experiment of railway
  ballast vibrations*, Journal of Sound and Vibration 270(4–5) (2004)
  673–683, doi:10.1016/S0022-460X(03)00186-X; the inherited topology omits
  Zhai's adjacent-ballast \(K_w,C_w\) shear branch and condenses on-bridge
  ballast mass onto deck DOFs, so it is not the complete Zhai model;
- a five-vehicle planar train with primary and secondary suspensions; and
- bilateral linear wheel–rail contact solved by direct time integration.

Two geometries are used:

- **L60:** three spans, with scour targets at supports 2 and 3;
- **L99.6:** four spans, with targets at supports 2, 3, and 4.

The adopted deck \(E,I,\rho A\), and 3% damping values originate from the
Fernandes two-by-20 m example. Their reuse for L60 and especially four-by-24.9
m L99.6 is an explicit idealized geometry/scale stress transfer, not a claim
that either longer bridge configuration has been calibrated to a field asset.

The leading vehicle supplies eight candidate response channels. The vehicle
itself is TTB-2D's six-DOF formulation (vertical displacement and pitch of
the car body and of each bogie). Two additional saved rows are not vehicle
responses: `Sol.Veh.acc_under = N(x_w)^T A_rail` samples the Eulerian
(partial-time) rail FE vertical-acceleration field at the instantaneous wheel
coordinates. They are neither wheelset/axle-box acceleration nor total
acceleration following the moving contact point. In `B66_ContactForce`, the
convective terms (v^2u_{,xx}+2v\dot{u}_{,x}) remain separate.

| Index | Response channel | Physical status |
|---:|---|---|
| 0 | car-body vertical acceleration | secondary-suspended |
| 1 | front-bogie vertical acceleration | primary-suspended |
| 2 | rear-bogie vertical acceleration | primary-suspended |
| 3 | Eulerian rail vertical acceleration at moving wheel-1 coordinate | virtual rail-field sample; legacy key `Wheel1_Vert` / `AcelRodaPrimVag[0]` |
| 4 | Eulerian rail vertical acceleration at moving wheel-2 coordinate | virtual rail-field sample; legacy key `Wheel2_Vert` / `AcelRodaPrimVag[1]` |
| 5 | car-body pitch angular velocity | secondary-suspended |
| 6 | front-bogie pitch angular velocity | primary-suspended |
| 7 | rear-bogie pitch angular velocity | primary-suspended |

Thus the candidate set is three vehicle vertical accelerations, two virtual
moving-coordinate rail-acceleration samples, and three vehicle pitch angular
velocities. It is not eight accelerometers, and channels 3/4 do not correspond
to deployable axle-box sensors without changing the observation model.

Implementation-naming note: the campaign code and loader index these eight
channels as "DOF" 0–7 (e.g., the observation-noise RNG is keyed by "global
DOF"). Wherever this document says "channel/DOF" it refers to that
implementation channel index, not to a modeled vehicle degree of freedom —
per the paragraph above, only six of the eight channels correspond to
modeled DOFs.

The manuscript must reproduce the exact implemented bridge, vehicle, track,
damping, mesh, time-step, crop, and solver parameters from the authorized
generation contract. Zhai et al. (2004) state the nominal track quantities
per rail seat and use 0.545 m rail-support spacing. The inherited planar
property file doubles rail inertia/mass and the half-sleeper mass, but retains
the tabulated pad, ballast, and sub-ballast terms at the generator's 0.600 m
spacing. In the source equations, \(M_b\), \(K_b\), and \(K_f\) depend on
spacing; the damping values are not claimed to be spacing-derived. The intended
one-seat/two-rail scaling and the spacing transfer are
therefore separate model-validity questions; the manuscript must not claim
that all per-rail values were summed or that this is a spacing-consistent Zhai
reproduction until an upstream benchmark or prospective sensitivities resolve
  them. Zhai supports the per-seat 531.4 kg value and an independent discrete
  mass at each support; B54 instead condenses that retained value onto the deck
  DOF under each on-bridge sleeper and omits the source's adjacent-mass
  \(K_w,C_w\) shear branch. That inherited topology, and full lumps at both
  bridge endpoints, are not source-supplied bridge rules. The separate rail
  Rayleigh target is 0.1%; it is not reported by Zhai and is retained as an
  inherited author-chosen modeling value. The bridge Rayleigh target remains
  3%.

The production bridge/rail meshes are geometry-specific and support-aligned:
L60 uses 0.2/0.3 m (3/2 elements per 0.6 m sleeper bay), whereas L99.6 uses
0.3/0.3 m (2/2 per bay). Positive-spring supports must lie on nodes to
roundoff tolerance. A universal 0.3 m deck/rail statement is therefore wrong:
on L60 it realizes the internal supports at 20.1 and 39.9 m. In particular,
the corrected deck mass per unit length and resulting healthy modal
frequencies must be reported;
pre-fix datasets with the 1,000× deck-mass error are inadmissible. The
modal gate is two-level: on first passage, every generated state must fall
inside the 0.2–15 Hz admissibility band, and `target_healthy` states must
additionally reproduce the nominal healthy first bending frequency within
the registered acceptance band (L60: ≈4.18 Hz, accepted 3–6 Hz; L99.6:
≈2.75 Hz, accepted 2–4 Hz).

### 2.1 Contact-model scope

The solver is linear and bilateral; it does not simulate separation and
re-contact. Wheel flats are therefore disabled. Wheel polygonization is
required to remain within the registered contact gate, subject to the final
qualification and time-step closure. A two-tier post-solve contact diagnostic
logs brief micro-unloading and aborts generation before invalid data are
admitted if a passage exceeds either the registered peak-tension bound (24 kN)
or path-fraction bound (0.2% of on-track samples), or contains non-finite
contact values. The registered time-step closure study is a required numerical
qualification, not a substitute for a nonlinear contact model.

## 3. Modeled damage and nuisance mechanisms

### 3.1 Scour surrogate

Scour is represented as loss of the target support's vertical stiffness,

$
k_v(d)=(1-d)k_{v0},\qquad 0\le d\le0.60,
$

with the implemented healthy value $(k_{v0}=3.44\times10^8\ \mathrm{N/m})$.
The label \(100d\) is therefore **modeled support-stiffness loss (%)**. It is
not scour-hole depth, embedment loss, eroded soil volume, or a universal
mapping from hydraulic scour to stiffness. The idealization follows the
support-stiffness-loss convention used in relevant drive-by scour studies —
Fernandes, Lopez, Ribeiro & Fadel Miguel (2024, *Struct. Health Monit.*
drive-by autoencoder) and (2025, *Int. J. Struct. Stab. Dyn.* art. 2650316),
with the foundation-frequency sensitivity underlying it from Prendergast,
Hester, Gavin & O'Sullivan (2013, *J. Sound Vib.* 332(25):6685–6702) — subject
to this semantic boundary. These are the same three keys the manuscript cites
at `paper1/sections/numerical_simulation.tex` (`fernandes2024driveby`,
`fernandes2025early`, `prendergast2013investigation`).

### 3.2 Bearing mechanism

The bearing mechanism is a nominal abutment rotational-fixity intervention:
a rotational spring at each abutment whose free-rotation baseline (k_r = 0)
is the reference configuration.
The sampled variable is an analytic nominal fixity ratio φ, mapped **once**
to rotational stiffness through the implemented geometry-dependent
transformation k_r = φ/(1−φ)·4E₁₅I/L_end, evaluated with the fixed 15 °C
reference modulus E₁₅; k_r then remains constant while the deck modulus
varies with temperature between passages. The label is a nominal
free-rotation-to-near-fixed coordinate. It is a
bounded design coordinate for the simulation and regression head, not a direct
percentage of physical bearing material damage or condition rating. Bearing
heads are trained and reported, but model selection on bearing rungs remains
scour-primary.

### 3.3 Crack mechanism

The crack nuisance is a **uniform damaged-element reduction of flexural
rigidity \(EI\)** over the selected finite element. Its location and severity
are drawn from the registered design distribution. This is a defensible
damaged-element benchmark consistent with the repository's Fernandes
comparison; it is **not** the tapered, piecewise-linear Sinha–Friswell–Edwards
crack model and is not parameterized by physical crack depth. The manuscript
must not call it “Sinha damage.”

### 3.4 Rail profile

Every rail profile in the campaign is generated from the same implemented FRA
class-4 power spectral density (corrected cycles-per-metre corner frequency);
no measured profile is used, and only the phase rule changes along the rung
graph. Rungs `s0`–`s13` share one fixed generated realization (registered
phase seed 20260728), common to every state and rung. At `s13→s14`, that
shared realization is replaced by per-state phase realizations of the same
spectrum. On L99.6 the shared realization remains fixed through `s22`;
per-state realizations begin only at `s23`. The profile is held across the
passages of one state. FRA class 4 is a registered benchmark distribution; the paper must
not call it the universally “roughest legal” track condition without a
route/speed-specific regulatory argument. The former per-passage 0.5-mm white
physical jitter is absent because EN 13848-2 measurement repeatability does not
represent physical rail-profile evolution.

### 3.5 Track-layer mechanisms

The track nuisance block includes:

- ballast stiffness/damping patches: Poisson counts (rate 1.2/100 m,
  window-scaled), lengths U(5, 20) m, each patch **independently** wet or dry
  with probability 0.5 each
  (dry: stiffness ×[1.2, 2.0], damping ×[0.4, 0.8]; wet: stiffness
  ×[0.7, 0.9], damping ×[1.5, 4.0]); patch centers placed with an
  author-chosen 3× density
  within 20 m of the abutments; where patches overlap, the patch with the
  largest absolute log-stiffness multiplier governs and carries its
  (η_k, η_c) multipliers **jointly**;
- hanging-sleeper groups: Poisson counts (rate 3.0/100 m, window-scaled),
  author-chosen group size discrete U{1,…,5} consecutive sleepers; 60% of groups directed
  to transition zones (±15 m of the abutments), then fouled-patch placement
  with 3:1 odds; a start proposal is accepted only when the complete sampled
  group fits inside the modeled sleeper window, so the stored count is never
  truncated; and
- rail-pad variability/failure: **one** state-global service-condition stiffness
  scalar (Weibull, scale λ=1.8, shape k=2.2, clipped to [1.0, 3.5]) and
  **one** state-global damping
  scalar in [0.8, 1.2], plus independent per-position Bernoulli(0.02)
  failures on the 0.6 m sleeper lattice. Failed-pad descriptors must match that
  lattice exactly and cannot be silently snapped. The global scalars have no
  time axis and are not a progressive aging law.

Track descriptors are sampled in a bridge-local approach–deck–exit frame and
mapped by the model to the actual global sleeper coordinates, so the sampled
transition/deck mechanisms act on the intended locations. Hanging sleepers
and failed pads are implemented as a **linear support-removal
approximation**: the affected element's stiffness **and** damping are both
multiplied by 1e-6. There is no explicit void depth, sleeper–ballast gap,
closure, impact, or settlement-profile nonlinearity, no ARIMA spatial aging
field, and no consecutive-failure cap. Claims must be limited accordingly.

### 3.6 Wheel out-of-roundness

The train nuisance block includes registered wheel polygonization draws.
Wheel flats are excluded because they violated the bilateral-contact
assumption and were temporally under-resolved. “All” rungs therefore mean all
**modeled vertical-pathway mechanisms**, not every bridge, track, soil, or
vehicle damage mode.

### 3.7 Registered generative priors

The values below are the implemented design distribution, not estimates of a
universal infrastructure population. **None was fitted to data**: no estimator,
fitting sample, or goodness-of-fit assessment exists for any of them. Following
the 2026-08-01 semantic-closure pass they fall into exactly three classes —
values chosen with reference to a primary measurement used as an engineering
proxy across a stated scope boundary; one value **contradicted by the nearest
available measurement and deliberately retained** (the dry-fouling stiffness
band [1.2, 2.0], against Esmaeili's measured mild softening); and
author-chosen design values. The
earlier "derived/inferred" class is **retracted**: the direct-PDF audit found
that the exact distributions and odds previously described that way were not
derived from any source (see `docs/track_eov_sampling_spec.md` and
`paper1/MISSING_PRIMARY_SOURCES.md`). In particular, the 0.5 wet-patch
probability, dry-stiffness band,
pad Weibull/multiplier law, 0.6 transition selection, 3:1 local coupling odds,
and the wheel-OOR occurrence/order/amplitude triplet are not measured
population distributions. The 0.02 pad-failure probability is also wholly
author-chosen: the direct-PDF audit found no primary support for the previously
claimed 0.5% annual-incidence anchor. The manuscript
must preserve those distinctions and cite only what each primary actually
supports; `docs/track_eov_sampling_spec.md` records the evidence boundary.

| Variable | Registered design |
|---|---|
| scour support-stiffness loss | joint LHS on [0, 0.60] per target; five controlled nonzero anchor levels |
| latent bearing fixity | joint LHS on [0, 0.95] per abutment; five controlled nonzero anchor levels |
| crack activation | author-chosen UID-keyed Bernoulli 0.25 in `joint`; forced on in `nuisance_only`; off in the controlled healthy/scour/bearing families; dormant where crack physics is inactive; not a fitted population prevalence |
| crack severity/location | author-chosen throughout: author-chosen \(EI\)-loss severity band U(0.05, 0.30); author-chosen 4:1 hogging:sagging design odds; author-chosen ±0.175-span support-zone window; author-chosen global 0.10–0.90 bridge-length clamp — none fitted to data |
| rail profile | one generated FRA class-4 realization (phase seed 20260728) shared through `s13` on L60 and through `s22` on L99.6; per-state FRA class-4 phase realization at `s14+` (L60) and only at `s23` (L99.6); zero physical passage jitter |
| descriptor sampling window | author-chosen convention: 30 m of approach + the deck + 30 m of exit ⇒ 120 m (L60) and 159.6 m (L99.6); scales both Poisson means, so it carries the same evidentiary status as the rates it multiplies; distinct from the physical approach/exit track lengths |
| hanging sleepers | Poisson rate 3.0 groups/100 m, author-chosen group size discrete U{1,…,5} consecutive; author-chosen 60% transition-zone selection (±15 m of abutments) then author-chosen 3:1 fouled-patch odds; support stiffness and damping ×1e-6 |
| ballast patches | author-chosen Poisson rate 1.2 patches/100 m, author-chosen length U(5, 20) m (the FRA extent context is open-ended reporting among class-5-limit-crossing sites, not a fitted bound), independently wet/dry with author-chosen p=0.5. Dry: k ×[1.2,2.0] **author-chosen and sign-contradicted** by Esmaeili (measured mild softening), c ×[0.4,0.8] **direction/magnitude taken from Esmaeili** (up to −67%) but deliberately milder and not estimated from it. Wet: k ×[0.7,0.9], c ×[1.5,4.0] **flooded-clean-ballast proxy** (measured 0.67 sits below the band; condensed dashpot rises ×2.8 not ×4.0, and its rise is gradual with no stated threshold — ≈unchanged at 5–10 cm, +28% at 15 cm, ×2.8 only at full 35 cm submergence, Table 4.17). Author-chosen 3× center density within its author-chosen 20 m abutment window; max-\|log η_k\| overlap winner carries (η_k, η_c) jointly |
| rail pads | one state-global author-chosen service-condition scalar Weibull(λ=1.8, k=2.2) clipped to [1.0,3.5] + one state-global damping scalar [0.8,1.2]; independent per-position Bernoulli(0.02) failures (author-chosen snapshot stress prior) on the 0.6 m lattice, stiffness and damping ×1e-6 |
| wheel polygonization | per-wheel probability 0.30; order discrete U{1,…,5}; \(\ln(A[\mathrm{m}])\sim N(-10,0.5^2)\), clipped to 10–120 µm — an author-chosen design prior: the cited literature supports the polygonization physics, not these exact occurrence/severity numbers |
| speed and temperature | author-chosen envelopes: correctly oriented 50×2 LHS mapped to [70, 90] km/h and [3, 33] °C, then rounded to the nearest integer km/h and °C; temperature acts on the deck modulus through the author-chosen registered linear law E(T)=E15·[1−0.003(T−15)] (T in °C, −0.3%/°C, ≈9% modulus change across the 30 °C span) |
| vehicle variability | author-chosen: for each passage and each of five vehicles, independent standard-normal multipliers give Gaussian body-mass CV 10%, primary-suspension-stiffness CV 5%, and secondary-suspension-stiffness CV 5%; the other registered vehicle properties remain fixed |

Because the retained dry-ballast stiffness direction is contradicted by the
nearest audited experiment, definitive interpretation requires the opt-in,
CRN-paired retained-stiffening versus reciprocal-softening analysis specified
in [`dry_ballast_stiffness_sign_sensitivity.md`](dry_ballast_stiffness_sign_sensitivity.md).
The reciprocal arm is a matched-log-magnitude sensitivity, not a replacement
field prior.

The 4:1 crack-location odds are a design prior. Eurocode cracked-region
guidance can motivate the support-region window but does not, by itself,
establish those occurrence odds.

## 4. Registered rung graph

### 4.1 L60 controlled edges

The L60 graph contains seven registered edges:

1. `s0_scour → s11_bear`;
2. `s0_scour → s12_crack`;
3. `s11_bear → s13_bearcrack`;
4. `s12_crack → s13_bearcrack`;
5. `s13_bearcrack → s14_prof`;
6. `s14_prof → s15_track`;
7. `s15_track → s16_all`.

Interpretation requires two qualifications.

First, `s0→s11` and `s12→s13` simultaneously activate bearing physics and add
bearing outputs/range-normalized multi-task training. They estimate the cost of
the complete registered bearing-aware task; they do not isolate bearing
physics from representation/head sharing.

Second, `s13→s14` changes the profile regime from one shared fixed baseline to
a distribution of persistent per-state FRA-4 phase realizations. It is not
merely an amplitude perturbation of the same realized profile.

The four-cell bearing-by-crack difference-in-differences,
\((s13-s12)-(s11-s0)\), is a secondary exploratory interaction analysis and is
not an eighth primary edge.

### 4.2 L99.6 block

`s21_scour4` is the independent L99.6 execution, HPO, and reference-selection
anchor. `s22_bearcrack4` jointly activates bearing and crack, and `s23_all4`
jointly adds the profile/track/wheel EOV block. These are scale/stress
comparisons, not a seven-edge one-factor family. The L60 champion, HPO
manifest, or source/runtime-bound execution receipt is never copied into this
block.

## 5. Fixed semantic state design and common random numbers

The state universe is fixed within each geometry so an edge cannot be
confounded by sample size, family composition, or the insertion/removal of
rows:

| Family | L60 | L99.6 | Role |
|---|---:|---:|---|
| `target_healthy` | 50 | 50 | zero-target diagnostic |
| `scour_only` | 50 | 75 | controlled per-pier/level anchors |
| `bearing_only` | 50 | 50 | controlled bearing anchors; dormant where inactive |
| `nuisance_only` | 50 | 50 | controlled crack nuisance anchors; dormant where inactive |
| `joint` | 250 | 250 | primary multivariate population |
| **Total** | **450** | **475** | |

Every state has 50 passages. The 50-passage count is a prospectively fixed,
balanced, compute-feasible operational integration budget — every state
contributes the same passage count under the campaign's compute envelope —
not a power calculation and not a claim of 50 independent samples (see
below). The `joint` population is generated once per
geometry from a master LHS containing all scour coordinates and two latent
bearing coordinates, irrespective of rung activation. Latent crack status is
also defined independently of the rung toggle. The active physics variables
are derived from those latent values.

Every row has a semantic `StateUID` encoding geometry, family, target, level,
and replicate identity. A collision-checked `StateSeedID` and a versioned set
of UID-named random substreams allocate:

- one state-keyed operations stream that generates the complete 50-passage
  speed/temperature/vehicle sequence, plus state-persistent crack,
  profile-state, track, and profile-phase draws;
- an active passage-level wheel-OOR stream and a reserved `profile-passage`
  namespace used only by dormant/deprecated profile modes. In production R11,
  the rail profile is persistent within a state and is not redrawn by passage.

No scientific random draw is keyed to mutable row number or parallel
scheduling. Within a geometry, the complete UID inventory, latent design,
random-stream identities, and split assignment must match exactly across
rungs. These common random numbers are intended to reduce edge variance and
support the registered paired resampling analyses; whether variance is
actually reduced depends on the covariance the shared draws induce, and no
variance-reduction diagnostic is registered. The paired analyses are
fixed-design sensitivity summaries, not inferential guarantees, and the
shared draws do not make the 50 passages independent experimental
units. The semantic state is therefore the analysis and resampling cluster
relative to passages. It must not itself be described as an iid field sample:
the controlled anchors are fixed and the joint states are the points of one
registered LHS realization.

## 6. Operational variability and raw data

For each passage, the generator samples the registered speed range
[70, 90] km/h, temperature range [3, 33] °C, and vehicle-property variability;
speed and temperature are rounded to integer km/h and °C before simulation.
Temperature acts on the deck modulus through the registered linear law
E(T)=E15·[1−0.003(T−15)] with T in °C (−0.3%/°C; ≈9% modulus change across
the registered 30 °C span).
The speed–temperature design uses a correctly oriented
`Npass × number-of-variables` Latin hypercube. Persistent state conditions do
not change between passages.

MATLAB saves raw, un-interpolated, noise-free time histories plus the spatial
crop and transformation metadata. The generator persists its exact numerical
environment descriptor, reviewed source/asset root, semantic state identity,
latent design, named random-stream schedule, contact diagnostics, and
generation fingerprint in the campaign manifest and state files. Resume,
loading, cache reuse, and protocol creation reject missing, qualification-only,
foreign, or inconsistent provenance.

## 7. Preprocessing and observation model

Python reproduces MATLAB's time-to-space interpolation and crop, then applies
true Piecewise Aggregate Approximation: the spatial sequence is partitioned
into equal windows and each window is replaced by its mean, giving 512
segments. A separate affine scaler is fitted to the resulting representation
for each selected channel using training samples only, then applied to all
partitions. This PAA front end follows the compression precedent set in
drive-by monitoring by Fernandes, Lopez, Ribeiro & Fadel Miguel, *Early
Multi-damage Classification in Railway Bridges Using Drive-by Numerical
Measurements with Piecewise Aggregate Approximation and Convolutional Neural
Networks*, Int. J. Struct. Stab. Dyn., art. 2650316,
doi:10.1142/S0219455426503165 — the only paper in that line that applies PAA
(583 segments over 5,830-sample, 1 kHz vertical-acceleration records).
**Boundary:** that study motivates PAA by dimensionality reduction and
training cost (527 → 121 min), not by an explicit low-pass or denoising
argument; the low-pass reading is ours. It also min–max scales to [0, 1]
where we standardize.

The main observation arm, `all_mult`, adds pointwise zero-mean Gaussian
multiplicative noise with standard deviation \(0.05|x|\) to every selected
channel at load time. The RNG is keyed by global DOF, so a DOF receives the
same realization whether evaluated alone, in a pair, or in the full array.
Noise is injected after the raw-to-space transform and before PAA/scaling.

This is deliberately a **symmetric relative-noise stress model**. It does not
encode a rail-qualified IMU's additive noise density, bias stability, full
scale, bandwidth, mounting transfer function, or EN 61373 qualification level.
The result must therefore be worded as a ranking of modeled response
channels/DOFs under the registered stress—not as a comparison of sensor
hardware technologies. The experiment selects channels, not transducers: one
multi-axis IMU can supply more than one modeled channel, while another channel
may require another device/location. Physical sensor count, packaging,
mounting, and hardware placement are not identified. A datasheet-based
observation model is a separate future robustness arm.

## 8. Learning task and architectures

The network performs continuous multi-output regression. Output order is the
target-pier scour heads followed, where active, by left/right bearing heads.
On bearing-active rungs, training uses the registered range-normalized
multi-head MSE (per-head weights ∝ 1/range², normalized to mean one; head
ranges 60 for scour, 95 for bearing) so the larger bearing range cannot
dominate gradient scale; scour-only rungs use the plain MSE. Model selection
on bearing rungs uses scour-head MSE; bearing MSE and scour↔bearing leakage
remain secondary reported metrics.

Four architecture arms share the same PAA and convolutional search space:

1. CNN with multi-rate pooling;
2. CNN + Time2Vec-style spatial encoding with multi-rate pooling (internal
   key `PAA_S2V_NHiTS`);
3. CNN + LSTM with multi-rate pooling;
4. CNN with global average pooling and no multi-rate module.

The fourth arm anchors the pooling comparison as an equal-budget control
family, not a direct pooling ablation: the multi-rate family additionally
searches its pooling-rate configuration (`nhits_pool_rates_key` is searched
only when the multi-rate module is active), so the two hyperparameter search
spaces are not identical and family differences are reported as observed
finite-design error differences, never as isolated-module effects. The
implemented `MultiRatePooling1D` module is N-HiTS-inspired pooling, not the
full N-HiTS forecasting architecture; the manuscript must use the
implementation-level name. It is a fixed-width adaptive temporal pyramid: each
configured level denotes an output-bin count, adaptive max pooling produces
that count for any sequence length, and the level outputs are concatenated.
Consequently RAW and PAA use the identical pooling operation and dense-head
width, and changing sequence length alone does not change parameter count. The
historical configuration key `nhits_pool_rates` is retained, but its values are
adaptive bin counts rather than stride/downsampling factors.

The eight-channel input is a non-selectable response-budget control. Candidate
selection is restricted to registered two-channel/DOF subsets. The
registered comparator — front-bogie vertical acceleration plus the Eulerian
rail acceleration sampled at the moving wheel-1 coordinate, channel indices
`[1, 3]` — is retained under its legacy identifiers. Its former
"sprung+unsprung sensor fusion" rationale is withdrawn; axle-box literature
does not validate this implemented virtual rail-field pairing. The comparator
cannot replace the selected winner unless it wins under the registered rule.

`s16_all` and `s23_all4` additionally reopen the complete
4-architecture × 28-pair × 3-seed matrix as an **exploratory deployment
reselection** under the all-modeled-EOV regime. These candidates use their
block's frozen full-array-calibrated hyperparameters. Their separate winner
cannot overwrite the L60/L99 reference or substitute for the carried L60
reference in the seven-edge primary analysis.

## 9. Compute-feasible hyperparameter calibration

There are two independent calibration blocks:

- L60, anchored at `s0_scour`;
- L99, anchored at `s21_scour4`.

At each anchor, only the **full eight-channel input** receives free HPO:
4 architectures × 3 registered training/HPO seeds × 100 Optuna trials. The
anchor studies use the registered multivariate TPE sampler (25 startup trials
at a 100-trial budget, with constant-liar handling) and a
Successive-Halving pruner (`min_resource=4`, `reduction_factor=3`). Pruning is
active only in these 24 anchor studies. The canonical block manifest stores the
exact best hyperparameters for each architecture–seed cell. At an anchor, this
100-trial calibration phase itself serves as the non-selectable full-array
control and is not repeated as a singleton.

Every response-channel candidate, non-anchor full-array control, and downstream-rung
configuration then runs one real Optuna trial over singleton distributions
copied from its block's authenticated manifest, with the pruner disabled. This
retains the production study/artifact path and removes candidate-specific and
rung-specific HPO lotteries. It does not eliminate finite-search optimization
error: architecture comparisons remain conditional on separate equal-budget
100-trial anchor calibrations and the registered finite seed set.

After the hyperparameter manifest exists, each anchor also runs a diagnostic
8-single-channel × 4-architecture × 3-seed matrix. These are frozen singleton
studies. They characterize individual response channels but do not select the
block reference; reference selection uses the complete
4-architecture × 28-two-channel × 3-seed matrix.

The campaign contains 1,620–1,638 Optuna studies, depending on whether the
designed comparator duplicates the carried pair at the six frozen rungs, but
3,996–4,014 trials:

- 24 anchor studies × 100 trials = 2,400 trials;
- every remaining study contributes exactly one frozen trial.

Any Optuna `FAIL`, OOM, nonterminal surplus, missing complete trial, mutated
search identity, or inconsistent manifest is fatal. Before a study is created,
the physical GPU must pass a durable capacity qualification at the largest
registered structural point for all four architectures, including forward,
five-head loss, backward, and Adam update. The measured **model-envelope
headroom**, total device memory minus this process's peak reserved memory, must
be at least
\(\max(0.20\times\text{total VRAM}, 1\ \mathrm{GiB})\). This is not a
system-wide free-memory measurement and cannot detect other GPU processes.
Qualification and production therefore require an otherwise idle/exclusive
GPU as an operational condition.

This policy supports conditional statements such as “best among these
architectures and two-channel/DOF subsets under the registered
full-array-calibrated policy and realized finite anchor searches.” It is not
evidence of a global architecture, physical-sensor-count, or hardware-placement
optimum.

### 9.1 Registered training and search specification

Every ordinary Optuna training trial uses batch size 32, at most 50 epochs,
early-stopping patience 5, Adam, and `CosineAnnealingLR` with
`T_max = 50` and `eta_min = 0`. The validation objective is scour-head MSE when
bearing heads are present and ordinary MSE otherwise. The registered
hyperparameter domains are:

| Scope | Hyperparameter | Domain |
|---|---|---|
| base | convolutional layers | integer 2–4 |
| base | dense layers | integer 1–3 |
| base | learning rate | log-uniform \(10^{-4}\)–\(10^{-2}\) |
| base | weight decay | log-uniform \(10^{-5}\)–\(10^{-3}\) |
| each convolutional layer | filters | 16–128, step 16 |
| each convolutional layer | kernel size | {2, 3, 5, 7} |
| each convolutional layer | pooling | {true, false} |
| each dense layer | units | 32–256, step 16 |
| each dense layer | dropout | continuous 0.1–0.5 |
| LSTM arm | recurrent layers | integer 1–2 |
| LSTM arm | hidden size | 32–128, step 32 |
| LSTM arm | recurrent dropout | continuous 0.1–0.4, only when layers > 1 |
| multi-rate arms | pooling-rate key | {1_2_4, 1_4_8, 1_3_6, 1_2_4_8} |

The anchor sampler is multivariate TPE with `constant_liar=true` and 25 startup
trials at the 100-trial budget. Its Successive-Halving pruner uses
`min_resource=4`, `reduction_factor=3`, and
`min_early_stopping_rate=0`. A frozen singleton registers every active
hyperparameter as an exact one-point Optuna distribution.

Python, NumPy, PyTorch CPU, and every CUDA device are seeded for each registered
training seed. cuDNN deterministic mode is enabled, cuDNN benchmarking and
TF32 are disabled, float32 matrix multiplication uses `highest` precision,
PyTorch deterministic algorithms hard-fail rather than warn, and cuBLAS uses
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### 9.2 Registered random seeds

| Operation | Seed/policy |
|---|---|
| MATLAB master damage-state design and UID-derived streams | `damage_seed = 1` |
| fixed shared rail-profile realization (phase), rungs `s0`–`s13` (L99.6: through `s22`) | 20260728 |
| canonical 60/20/20 semantic-state split | 42; after UID sorting and the seed-derived within-stratum permutation, the repeating assignment pattern is train/test/validation/train/train |
| Optuna sampler and model training arms | {42, 1337, 2026} |
| load-time relative-noise draws | 42, keyed additionally by global DOF |
| finalist 5-fold × 2-repeat CV partition | 271828 |
| within-rung state bootstrap | 42, 2,000 replicates |
| L60 cross-rung paired state bootstrap | 42, 100,000 replicates |

## 10. Data partition and model-selection firewall

States, not passages, are assigned to a deterministic 60/20/20
train/inner-validation/outer-test split. Assignment is a semantic-UID-stable
hash permutation within registered strata:

- family, target, and anchor level for controlled families;
- registered latent-crack/severity strata for the joint family.

The five replicas in every scour-only or bearing-only target/level cell
allocate exactly 3/1/1 to train/inner-validation/outer-test; healthy and
nuisance-only states are stratified by family. All passages from a state stay
together. The identical UID partition is required across every rung of a
geometry.

Training-channel scalers are fitted on training data only. HPO, early stopping,
architecture/channel-subset selection, and finalist choice use development data only.
The outer test remains inaccessible until the winner and registered comparator
set are frozen and all weights/artifacts are complete.

Only the four channel-search rungs (`s0`, `s16`, `s21`, and `s23`) run
finalist-only 5-fold × 2-repeat state-grouped CV on development states. The
registered comparator set contains the inner-validation winner, the top five
architecture×channel combinations from the complete inner-validation
factorial leaderboard, each architecture's own optimum, and the
same-channel-pair, designed, carried-reference, and full-array controls after
deduplication. Each fold refits the scaler on fold training data and uses
frozen hyperparameters and a checkpoint epoch derived before observing fold
validation. This is a conditional split-stability diagnostic and cannot
re-rank the canonical winner. The immutable outer test is the report set.

## 11. Within-rung reporting

For registered MSE finite-design resampling sensitivity and paired
contrasts, aggregate passage errors within state before resampling. Report:

- scour MSE in squared percentage points, per-pier MSE, and RMSE where useful;
- most-damaged-pier localisation accuracy as a passage-level point estimate,
  restricted to passages whose maximum true scour target is **strictly greater
  than 5 percentage points**;
- bearing MSE on bearing rungs;
- false-scour-from-bearing and false-bearing-from-scour diagnostics;
- median performance over the finite three-seed set and seed IQR;
- state-first finite-design resampling sensitivity intervals for registered
  finalists' scour MSE and all-head MSE;
- complete architecture/channel-subset/seed eligibility and provenance.

Within-rung finalist MSE intervals and paired MSE contrasts use 2,000
state-first bootstrap replicates with seed 42. The seed set
{42, 1337, 2026} is fixed and finite; bootstrap replicates do not resample
seeds, and seed variability is reported separately. Localisation is evaluated
passage by passage in the implementation; because every state contributes
exactly 50 passages, its point estimate is state-balanced, but no state-level
localisation interval is registered. Do not attach the MSE bootstrap interval
to localisation.

## 12. Registered L60 cross-rung paired sensitivity analysis

(The implementing modules are `core/cross_rung_inference.py` and
`check_cross_rung_inference.py`; those file names are historical and live in
the hash-locked runtime root, so they are not renamed. "Inference" in the
file names must not be read as statistical inference — the registered
procedure is descriptive, per the non-claims below.)

The primary cross-rung analysis uses only the immutable outer-test subset of
the exact common 250-state `joint` master population. Controlled anchors remain
diagnostics. The analysis requires:

- the same exact outer-joint `StateUID` set at both endpoints;
- one observation for every UID × registered training-seed cell;
- the L60 `s0` reference architecture and two-channel/DOF pair at every rung;
- the exact architecture-by-seed hyperparameters calibrated on the `s0`
  full-array control;
- one canonical external L60 HPO manifest and execution receipt whose hashes
  match the champion and every result row.

Within a training seed, state-level scour MSE is averaged over the paired outer
states. The registered statistic is the median of those means over the finite
seed set. An edge effect is the right-rung statistic minus the left-rung
statistic, so positive values denote higher error after the registered
intervention (the added mechanism block plus, where applicable, its heads
and training task).

Resampling sensitivity is computed with 100,000 state-first paired bootstrap
replicates. States are resampled first with the same indices at both edge
endpoints; seeds remain paired and are not resampled. Each edge receives a
pointwise central-95% finite-design resampling sensitivity interval and a
wider seven-edge tail-adjusted sensitivity envelope (tail mass α/7, a
Bonferroni-style width rule). Both are descriptive sensitivity summaries of
the fixed finite design: neither is a confidence interval, a
familywise-error-controlled hypothesis test, a significance/superiority
decision, or a joint-sign guarantee. The bootstrap fraction above or below
zero is descriptive, not a p-value or posterior probability.

These sensitivity summaries quantify empirical state-resampling variability
conditional on the exact registered finite anchor/LHS design. Because the
anchors are fixed and the joint population is one LHS realization rather than an
iid field sample, they must not be presented as field-population or
design-superpopulation coverage statements. That interpretation would require
an LHS-aware variance estimator or independently replicated state designs.

## 13. Reproducibility and execution blocks

Production generation is locked to the exact registered MATLAB R2025b Update 5
numerical-stack descriptor. Python is locked to the registered Python
3.13.3/Torch/CUDA environment. Source roots, environment descriptors,
generation configuration, protocol core/full hashes, split manifests, Optuna
study records, weights, scalers, capacity receipts, HPO manifests, and champion
manifests are authenticated and cross-checked.

MATLAB generation uses at most four local process workers. The cap is part of
the generation fingerprint; changing parallel capacity cannot be hidden inside
a resumed campaign folder. Named UID streams make the scientific draws
independent of task scheduling, but the resource cap remains provenance-bound.

### 13.1 Pre-dispatch MATLAB host qualification

Every PC intended to generate production MATLAB data is assigned one stable,
unique `TTBI_QUALIFICATION_HOST_ID`. From clean commit A, that PC freshly
generates and executes transient qualification micros for at least `s0_scour`,
`s16_all`, and `s23_all4`. These three stages exercise the fixed-profile
baseline, the complete L60 nuisance block, and the four-span geometry. The
generated qualification script is derived from the reviewed `A00_Run.m`, is
regenerated for the qualification round from the converged commit, and is never
treated as durable source. The exact source-bound bytes are executed on each
intended host; host receipts are never copied or forged between PCs.

Every qualification run emits a `qualification_host_receipt.json` under schema
`ttbi-matlab-qualification-host-v1`, containing the declared host ID, hostname,
CPU identifier, logical-processor count, MATLAB thread diagnostic, and computer
architecture. The sidecar is bound to the actual MATLAB-environment digest,
canonical qualification-source digest, and exact executed-script digest.
Corresponding stage directories for the required host/environment pairs are
then compared with `compare_generation_releases.py`. Accepted comparison
evidence uses schema `matlab-environment-qualification-receipt-v4` and requires
both host sidecars to be internally consistent and stable across stages. A
numerically equivalent verdict is not accepted implicitly; it requires explicit
review and a new acceptance receipt.

These host fields are *self-attested diagnostics*, bound by a SHA-256 over their
own canonical descriptor. No pre-registered signing key, hardware attestation,
or independent witness is involved. The reproducibility claim this supports is
 therefore **retained-artifact integrity and internal consistency under a
 trusted-operator threat model**: the retained datasets are complete, mutually
 consistent, unmodified since qualification, carry the reviewed source identity
 and contract, and each mandatory stage carries one stable set of host
 diagnostics. It is explicitly *not* a claim that the reviewed source is proven
 to have executed, or that the computation ran on
independent physical machines — a coherently fabricated receipt graph would
satisfy these checks. The mechanism guards against accidental drift, stale or
partial runs, copied receipts and silent substitution, which are the realistic
failure modes of a multi-PC academic campaign; it is not an anti-fraud
mechanism, and no result in this paper depends on it being one.

CPU equality is not required. Two runs with an identical MATLAB-environment
digest are eligible only when their authenticated receipts declare distinct
host IDs. Host identity is intentionally excluded from production `gen_schema`
and `gen_fingerprint`, so independent execution evidence does not alter
scientific dataset identity.

At the present pre-dispatch checkpoint, one real `s0_scour` qualification micro
on this laptop completed 35 states × 3 passages (105 passages) in 27 min 32 s
and validated. It predates source convergence. No second independent host has
run. This is integration and timing evidence for one host, not completed
cross-host qualification, and the campaign remains blocked. Every intended
generation host must still run `s0_scour`, `s16_all`, and `s23_all4` from
commit A; corresponding stage outputs must be authenticated by distinct host
IDs and accepted v4 comparison receipts.

During subsequent production generation, `parfor` may finish states out of
order. The raw-transform parity gate therefore waits specifically for the
complete `0001.mat`, runs the MATLAB reference smoke first, and runs the
dependent Python checker only after MATLAB has written
`matlab_ref_parity.mat`. Later output is not admitted until both sequential
checks pass.

### 13.2 Python execution blocks

Three durable absolute paths coordinate an execution block:

- `TTBI_EXECUTION_RECEIPT_DIR`;
- `TTBI_HYPERPARAMETER_MANIFEST`;
- `CHAMPION_MANIFEST`.

Relative paths and bundle-local defaults are forbidden. L60 and L99 use
separate artifacts. A cross-block copied champion or receipt is rejected.
The champion manifest carries the canonical `frozen_selection_sha256`. After
the anchor publishes it, the anchor prints the manifest's canonical SHA-256.
Every follower must provide that exact value through
`TTBI_BLOCK_REFERENCE_SHA256`; a missing or mismatched operator pin is fatal.
The pin binds followers to the reviewed anchor publication and prevents silent
manifest substitution after the anchor completes.

All seven L60 Python ablations must execute on one exact physical host, GPU,
and registered runtime. All three L99 Python ablations must likewise execute
on one exact host/GPU/runtime, although that block may use a different machine
from L60. MATLAB generation can be distributed independently; the Python
execution/HPO/reference block cannot be divided across computers.

The campaign can be dispatched only from the report-authorized commit and a
complete SHA-256-manifested ten-bundle set. Qualification outputs and benchmark
fixtures are never scientific data. The legacy-named
`benchmark_r5_compute.py` is now the contract-guarded genuine R11 timing
runner. It derives and authenticates a 475-state × 50-passage × 8-channel ×
512-segment workload with five heads, runs one 100-trial anchor HPO study
through the production objective, and runs exactly one shared finalist-CV
refit. `FAIL`, OOM, retry, and replacement are forbidden. The heavy benchmark
has not yet run; its timing and terminal-state evidence must be bound to clean
commit A before dispatch authorization.

## 14. Claim boundary

The implemented experiment can support claims about:

- continuous modeled support-stiffness-loss estimation;
- most-damaged-pier localisation;
- bearing estimation and scour↔bearing leakage;
- conditional architecture/two-channel response-subset ranking under the registered
  full-array-calibrated policy;
- paired changes in achievable L60 performance across the seven registered
  simulator-intervention edges (each edge changes a mechanism block and,
  where applicable, heads, loss/task, and retraining jointly); and
- blockwise behavior on the L99.6 scale/stress design.

It cannot, without additional experiments, support claims about:

- physical scour depth or hydraulic scour evolution;
- every bridge/track/vehicle damage mode;
- field generalization or a calibrated digital-twin likelihood;
- zero-shot OOD robustness;
- globally optimal architecture, physical sensor count or placement, or sensor
  hardware;
- sensitivity, specificity, probability of detection, calibrated damage
  probability, or minimum detectable severity; or
- nonlinear gap/contact behavior for hanging sleepers or wheel flats.

A binary detection claim would require a threshold selected using development
data only and then frozen before the outer test. R11 has no such threshold.

## 15. Before-submission checklist

- ~~Replace all citation placeholders with primary sources and verify the exact
  version of the Cantero/TTBI and Fernandes implementations used.~~ **DONE
  2026-08-01**: TTB-2D = SoftwareX 20:101253 at upstream base commit
  `28d35528…`; VEqMon2D = SoftwareX 19:101103; track properties = Zhai, Wang &
  Lin, JSV 270(4–5):673–683; PAA precedent = Fernandes et al., IJSSD art.
  2650316. Remaining: Garg & Dukkipati (1984) was **dropped** as a citation
  because no local copy exists to verify the FRA class-4 constants against;
  the manuscript now attributes those constants to the TTB-2D generator
  source, which is locally verifiable. Re-add the textbook citation only if a
  copy is obtained.
- Report the complete authorized physical parameter table, effective span
  lengths, sample interval, spatial crop, and healthy modal-frequency checks.
- State the FRA frequency units and profile persistence without converting
  EN 13848-2 metrology into physical process noise.
- Use “support-stiffness loss,” “nominal bearing fixity,” “uniform
  damaged-element \(EI\) loss,” “linear hanging-sleeper support removal,” and
  “all modeled vertical-pathway mechanisms” consistently.
- Pull every sample count, protocol hash, candidate count, timing, and result
  from authenticated R11 artifacts; do not copy historical tables.
- Distinguish inner-validation selection, diagnostic repeated CV, immutable
  outer-test performance, the seven-edge finite-design sensitivity analysis,
  exploratory deployment reselection, and L99 blockwise stress results.
- Report exact analysis-state counts alongside passage counts and do not use
  passages as the inferential sample size or describe the registered LHS states
  as an iid field sample.
- Keep the `all_mult` conclusion at the modeled response-channel/DOF level; do
  not infer physical sensor count, placement, or hardware performance.
