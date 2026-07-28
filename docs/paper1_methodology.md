# Paper 1 — R11 methodology draft

> **Pre-results, implementation-bound draft (2026-07-27).**
>
> This document describes the registered R11 experiment. Dispatch remains
> blocked and no R11 data or ablation result exists. Numerical findings from
> pre-R11 datasets are invalid for this study and must not be imported into the
> manuscript. At submission time, every protocol value must be pulled from the
> authorized commit, its `protocol_descriptor.json` files, authenticated
> manifests, and final result tables.

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
2. **L60 paired mechanism contrasts.** Seven registered edges estimate the
   change in achievable scour-estimation error after adding a mechanism, with
   architecture, response-channel pair, hyperparameters, semantic state population,
   split, and training seeds paired as specified below.
3. **L99.6 scale/stress results.** These form a separate physical execution
   block and are interpreted blockwise. They are not one-factor causal
   contrasts with L60.

Because every rung trains a new model, cross-rung changes measure
**rung-specific achievable performance under retraining**. They are not
zero-shot out-of-distribution tests.

## 2. Train–track–bridge interaction model

Vehicle responses are generated with the repository's two-dimensional,
vertically coupled train–track–bridge interaction (TTBI) model, derived from
the Cantero TTB-2D/VEqMon2D framework [cite the exact release/publication]. The
model couples:

- an Euler–Bernoulli multi-span bridge on vertical support springs;
- a layered ballasted track with rail, pads, sleepers, and ballast elements
  [cite the exact Zhai/TTBI source];
- a five-vehicle planar train with primary and secondary suspensions; and
- bilateral linear wheel–rail contact solved by direct time integration.

Two geometries are used:

- **L60:** three spans, with scour targets at supports 2 and 3;
- **L99.6:** four spans, with targets at supports 2, 3, and 4.

The leading vehicle supplies eight candidate response DOFs:

| Index | Response DOF | Suspension level |
|---:|---|---|
| 0 | car-body vertical acceleration | secondary-suspended |
| 1 | front-bogie vertical acceleration | primary-suspended |
| 2 | rear-bogie vertical acceleration | primary-suspended |
| 3 | wheelset 1 vertical acceleration | unsprung |
| 4 | wheelset 2 vertical acceleration | unsprung |
| 5 | car-body pitch angular velocity | secondary-suspended |
| 6 | front-bogie pitch angular velocity | primary-suspended |
| 7 | rear-bogie pitch angular velocity | primary-suspended |

Thus the candidate set is five vertical-acceleration channels plus three
pitch-angular-velocity channels. It is not eight accelerometers.

The manuscript must reproduce the exact implemented bridge, vehicle, track,
damping, mesh, time-step, crop, and solver parameters from the authorized
generation contract. In particular, the corrected deck mass per unit length
and resulting healthy modal frequencies must be reported; pre-fix datasets
with the 1,000× deck-mass error are inadmissible.

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

\[
k_v(d)=(1-d)k_{v0},\qquad 0\le d\le0.60,
\]

with the implemented healthy value \(k_{v0}=3.44\times10^8\ \mathrm{N/m}\).
The label \(100d\) is therefore **modeled support-stiffness loss (%)**. It is
not scour-hole depth, embedment loss, eroded soil volume, or a universal
mapping from hydraulic scour to stiffness. The idealization follows the
support-stiffness-loss convention used in relevant drive-by scour studies
[cite Fernandes and the foundation-frequency literature], subject to this
semantic boundary.

### 3.2 Bearing mechanism

Bearing degradation is represented by a rotational spring at each abutment.
The sampled variable is an analytic nominal fixity ratio, mapped to rotational
stiffness using the implemented geometry-dependent transformation. It is a
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

Rungs `s0`–`s13` share a fixed baseline longitudinal profile. At
`s13→s14`, that fixed baseline is replaced by a per-state realization from the
implemented FRA class-4 power spectral density, with the corrected
cycles-per-metre corner frequency. The profile is held across the passages of
one state. FRA class 4 is a registered benchmark distribution; the paper must
not call it the universally “roughest legal” track condition without a
route/speed-specific regulatory argument. The former per-passage 0.5-mm white
physical jitter is absent because EN 13848-2 measurement repeatability does not
represent physical rail-profile evolution.

### 3.5 Track-layer mechanisms

The track nuisance block includes:

- ballast stiffness/damping patches, with the governing overlapping patch
  selected by the largest absolute log-stiffness multiplier;
- hanging-sleeper groups; and
- rail-pad variability/failure.

Track descriptors are sampled in a bridge-local approach–deck–exit frame and
mapped by the model to the actual global sleeper coordinates, so the sampled
transition/deck mechanisms act on the intended locations. Hanging sleepers are
implemented as a **linear support-removal approximation**. There is no explicit
void depth, sleeper–ballast gap, closure, impact, or settlement-profile
nonlinearity. Claims must be limited accordingly.

### 3.6 Wheel out-of-roundness

The train nuisance block includes registered wheel polygonization draws.
Wheel flats are excluded because they violated the bilateral-contact
assumption and were temporally under-resolved. “All” rungs therefore mean all
**modeled vertical-pathway mechanisms**, not every bridge, track, soil, or
vehicle damage mode.

### 3.7 Registered generative priors

The values below are the implemented design distribution, not estimates of a
universal infrastructure population. Where the repository derived a prior from
several imperfect field sources, the manuscript must call it a **modeling
prior** and cite the derivation in `docs/track_eov_sampling_spec.md`.

| Variable | Registered design |
|---|---|
| scour support-stiffness loss | joint LHS on [0, 0.60] per target; five controlled nonzero anchor levels |
| latent bearing fixity | joint LHS on [0, 0.95] per abutment; five controlled nonzero anchor levels |
| crack activation | Bernoulli 0.25 in `joint`; forced on in `nuisance_only`; dormant where crack physics is inactive |
| crack severity/location | \(EI\) loss U(0.05, 0.30); 4:1 hogging:sagging design odds; ±0.175 span support zone with global 0.10–0.90 bridge-length clamp |
| rail profile | fixed baseline through `s13`; per-state FRA class-4 phase realization at `s14+`; zero physical passage jitter |
| hanging sleepers | Poisson rate 3.0 groups/100 m, group size discrete U{1,…,5}, registered transition/fouling placement weights |
| ballast patches | Poisson rate 1.2 patches/100 m, length U(5, 20) m, registered wet/dry stiffness/damping multipliers and transition weight |
| rail pads | per-position snapshot-failure modeling prior 0.02 plus registered stiffness/damping multipliers |
| wheel polygonization | per-wheel probability 0.30; order discrete U{1,…,5}; \(\ln(A[\mathrm{m}])\sim N(-10,0.5^2)\), clipped to 10–120 µm |
| speed and temperature | correctly oriented 50×2 LHS mapped to [70, 90] km/h and [3, 33] °C, then rounded to the nearest integer km/h and °C |
| vehicle variability | for each passage and each of five vehicles, independent standard-normal multipliers give Gaussian body-mass CV 10%, primary-suspension-stiffness CV 5%, and secondary-suspension-stiffness CV 5%; the other registered vehicle properties remain fixed |

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
manifest, or physical execution receipt is never copied into this block.

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

Every state has 50 passages. The `joint` population is generated once per
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
rungs. These common random numbers reduce edge variance and make exact paired
inference possible; they do not make the 50 passages independent experimental
units. The semantic state is therefore the analysis and resampling cluster
relative to passages. It must not itself be described as an iid field sample:
the controlled anchors are fixed and the joint states are the points of one
registered LHS realization.

## 6. Operational variability and raw data

For each passage, the generator samples the registered speed range
[70, 90] km/h, temperature range [3, 33] °C, and vehicle-property variability;
speed and temperature are rounded to integer km/h and °C before simulation.
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
partitions. This PAA front end follows the low-pass/compression precedent in
drive-by monitoring [cite the exact Fernandes paper].

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
Training uses the registered range-normalized multi-head MSE so the larger
bearing range cannot dominate gradient scale. Model selection on bearing rungs
uses scour-head MSE; bearing MSE and scour↔bearing leakage remain secondary
reported metrics.

Four architecture arms share the same PAA and convolutional search space:

1. CNN with multi-rate pooling;
2. CNN + Space2Vec with multi-rate pooling;
3. CNN + LSTM with multi-rate pooling;
4. CNN with global average pooling and no multi-rate module.

The fourth arm is the direct pooling-ablation control. The implemented
`MultiRatePooling1D` module is N-HiTS-inspired pooling, not the full N-HiTS
forecasting architecture; the manuscript must use the implementation-level
name.

The eight-channel input is a non-selectable response-budget control. Candidate
selection is restricted to registered two-channel/DOF subsets. The physics-motivated
front-bogie/wheel comparator is retained but cannot replace the selected
winner unless it wins under the registered selection rule.

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

At each anchor, only the **full eight-DOF input** receives free HPO:
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

For registered MSE uncertainty and paired contrasts, aggregate passage errors
within state before resampling. Report:

- scour MSE in squared percentage points, per-pier MSE, and RMSE where useful;
- most-damaged-pier localisation accuracy as a passage-level point estimate,
  restricted to passages whose maximum true scour target is **strictly greater
  than 5 percentage points**;
- bearing MSE on bearing rungs;
- false-scour-from-bearing and false-bearing-from-scour diagnostics;
- median performance over the finite three-seed set and seed IQR;
- state-first uncertainty intervals for registered finalists' scour MSE and
  all-head MSE;
- complete architecture/channel-subset/seed eligibility and provenance.

Within-rung finalist MSE intervals and paired MSE contrasts use 2,000
state-first bootstrap replicates with seed 42. The seed set
{42, 1337, 2026} is fixed and finite; bootstrap replicates do not resample
seeds, and seed variability is reported separately. Localisation is evaluated
passage by passage in the implementation; because every state contributes
exactly 50 passages, its point estimate is state-balanced, but no state-level
localisation interval is registered. Do not attach the MSE bootstrap interval
to localisation.

## 12. Registered L60 cross-rung inference

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
statistic, so positive values denote higher error after adding the mechanism.

Uncertainty is computed with 100,000 state-first paired bootstrap replicates.
States are resampled first with the same indices at both edge endpoints; seeds
remain paired and are not resampled. Each edge receives a pointwise 95%
percentile interval and a Bonferroni familywise interval controlling the family
of exactly seven primary L60 edges. Only the Bonferroni interval may support a
claim about an effect's sign across the ladder. The bootstrap fraction above or
below zero is descriptive, not a p-value or posterior probability.

These percentile intervals quantify empirical state-resampling uncertainty
conditional on the exact registered finite anchor/LHS design. Because the
anchors are fixed and the joint population is one LHS realization rather than an
iid field sample, the intervals must not be presented as field-population or
design-superpopulation coverage intervals. That interpretation would require
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

Every qualification run emits an authenticated
`qualification_host_receipt.json` under schema
`ttbi-matlab-qualification-host-v1`, containing the declared host ID, hostname,
CPU identifier, logical-processor count, MATLAB thread diagnostic, and computer
architecture. The sidecar is bound to the actual MATLAB-environment digest,
canonical qualification-source digest, and exact executed-script digest.
Corresponding stage directories for the required host/environment pairs are
then compared with `compare_generation_releases.py`. Accepted comparison
evidence uses schema `matlab-environment-qualification-receipt-v4` and
authenticates both host sidecars. A numerically equivalent verdict is not
accepted implicitly; it requires explicit review and a new acceptance receipt.

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
  mechanism edges; and
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

- Replace all citation placeholders with primary sources and verify the exact
  version of the Cantero/TTBI and Fernandes implementations used.
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
  outer-test performance, seven-edge confirmatory inference, exploratory
  deployment reselection, and L99 blockwise stress results.
- Report exact analysis-state counts alongside passage counts and do not use
  passages as the inferential sample size or describe the registered LHS states
  as an iid field sample.
- Keep the `all_mult` conclusion at the modeled response-channel/DOF level; do
  not infer physical sensor count, placement, or hardware performance.
