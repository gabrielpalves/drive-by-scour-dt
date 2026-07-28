# R11 multi-damage campaign

> **Pre-authorization snapshot (2026-07-27): DISPATCH BLOCKED.**
>
> This status records the source-convergence state before clean commit A. It is
> intentionally not rewritten by report-only commit B. The sole current
> dispatch authority—both before and after B—is
> `docs/audit_r5_results.md`; an authorized report there supersedes this dated
> snapshot.
>
> No existing `bundle_*.zip`, generated dataset, Optuna database, champion
> manifest, or result table is valid input to R11. The source must first
> converge on clean commit A, pass the full MATLAB/Python qualification and the
> commit-bound production-code-path HPO-plus-CV benchmark, and then be
> independently audited. The legacy-named `benchmark_r5_compute.py` is now a
> contract-guarded genuine R11 runner. It derives and authenticates the
> 475-state x 50-passage x 8-channel x 512-segment, five-head workload, executes
> one 100-trial anchor HPO study through the production objective, and performs
> exactly one shared finalist-CV refit. Optuna `FAIL`, OOM, trial retry, and
> trial replacement are forbidden. A qualifying refit must also be a clean
> first attempt: `attempt_count=1`, `prior_unaccepted_attempt_count=0`,
> `timing_complete=true`, and `memory_complete=true`. At this dated snapshot,
> the heavy run had **not** yet been executed; it must run against clean source
> commit A and publish only non-scientific throughput evidence. Only a
> report-only commit B may change
> `docs/audit_r5_results.md` to the exact
> `**Status: DISPATCH AUTHORIZED.**` verdict. While that report remains
> blocked, `build_stage_bundles.py` must refuse publication and production
> generation/ablation must not start.
>
> At this dated snapshot, MATLAB host qualification was also incomplete. One
> real `s0_scour`
> qualification micro on this laptop completed 35 states × 3 passages
> (105 passages) in 27 min 32 s and validated, but no second independent host
> has run. It also predates source convergence. This is one-host
> integration/timing evidence only; it does not close the required multi-stage,
> cross-host comparison gate.

R11 is a ten-rung, simulation-based experiment for continuous scour
support-stiffness-loss estimation and most-damaged-pier localisation from one
instrumented train passage. It evaluates bearing targets and randomized
crack/profile/track/wheel nuisance mechanisms without representing every
possible railway-bridge damage mode.

The modeled input has **eight response channels/DOFs**: five vertical
accelerations (car body, two bogies, two wheelsets) and three pitch angular
velocities (car body and two bogies). R11 selects channel subsets, not physical
transducers. A two-channel subset may map to one multi-axis IMU or to multiple
devices; the experiment does not identify sensor count, packaging, mounting,
or hardware placement.

## Registered physical design

The L60 branch contains seven controlled edges:

`s0→s11`, `s0→s12`, `s11→s13`, `s12→s13`, `s13→s14`,
`s14→s15`, and `s15→s16`.

The L99.6 branch is a separate scale/stress block. Its results are blockwise;
neither `s0→s21` nor the multi-mechanism L99.6 steps are one-factor causal
contrasts.

| Rung | Active change relative to its parent | Geometry |
|---|---|---|
| `s0_scour` | scour support-stiffness loss only; L60 execution/HPO/reference anchor | L60, 3 spans, piers 2 and 3 |
| `s11_bear` | bearing fixity active and bearing heads added | L60 |
| `s12_crack` | damaged-element crack nuisance active | L60 |
| `s13_bearcrack` | bearing and crack active | L60 |
| `s14_prof` | fixed baseline profile replaced by a per-state FRA-4 realization | L60 |
| `s15_track` | ballast, hanging-sleeper and pad mechanisms active | L60 |
| `s16_all` | wheel polygonization active; flats remain disabled; exploratory 4-architecture × 28-pair deployment reselection | L60 |
| `s21_scour4` | scour-only L99.6 anchor; independent execution/HPO/reference block | L99.6, 4 spans, piers 2–4 |
| `s22_bearcrack4` | bearing and crack active as one blockwise stress step | L99.6 |
| `s23_all4` | profile, track-layer and wheel EOV block active; exploratory 4-architecture × 28-pair deployment reselection | L99.6 |

The generator uses a fixed semantic state universe within each geometry:

- L60: 50 `target_healthy` + 50 `scour_only` + 50 `bearing_only` +
  50 `nuisance_only` + 250 `joint` = **450 states**.
- L99.6: 50 `target_healthy` + 75 `scour_only` + 50 `bearing_only` +
  50 `nuisance_only` + 250 `joint` = **475 states**.
- Every state has 50 passages. Passages are correlated repeated observations;
  the semantic state is the analysis and resampling cluster relative to those
  passages. The fixed anchors and the 250 points from one registered joint LHS
  realization are not an iid sample from a field population.

Inactive rungs retain the same rows and latent draws; their physics toggles are
zeroed rather than adding/removing states. Every state has a semantic
`StateUID`. A collision-checked numeric `StateSeedID` and a versioned schedule
of UID-named state/passage random substreams provide common random numbers
(CRN) without depending on row number, `parfor` order, or which mechanisms are
active. Within a geometry block, the complete UID inventory, latent bearing
design, latent crack design, random-stream identities, and deterministic
60/20/20 UID split must match exactly across rungs.

## What the physical labels mean

- **Scour** is the fractional loss \(d\) in the modeled vertical support
  stiffness, \(k_v(d)=(1-d)k_{v0}\). Report it as **support-stiffness loss
  (%)**, not scour-hole depth or soil-removal depth.
- **Bearing damage** is a nominal rotational-fixity design variable used as a
  regression target. It is not a direct field damage percentage.
- **Crack** is a uniform damaged-element \(EI\)-reduction block. It is not the
  tapered Sinha crack model and must not be cited as one.
- **Hanging sleepers** use a linear support-removal approximation. The model
  has no explicit sleeper–ballast gap/contact-closure nonlinearity or void-depth
  response law.
- **Wheel polygonization** is included; wheel flats are excluded because the
  bilateral linear contact solver cannot represent the separation they caused.

The modeled mechanisms are therefore scientifically useful abstractions with
explicit semantics, not interchangeable measurements of physical damage depth.

## Data, preprocessing, and observation model

MATLAB saves raw, un-interpolated, noise-free time histories and the spatial
crop metadata. Python reproduces the MATLAB space transform, applies true PAA
window means to 512 segments, and fits the resulting per-channel scaler on
training samples only. Persistent conditions are drawn per state; speed, temperature,
vehicle properties, and the registered passage-level mechanisms are drawn per
passage.

The main `all_mult` arm adds pointwise zero-mean Gaussian multiplicative noise
with \(\sigma=0.05|x|\) to every selected channel at load time, with the draw
keyed by global DOF so channel subsets remain noise-paired. Injection occurs
after the raw-to-space transform and before PAA/scaling. This is a
**symmetric relative-noise stress test**. It does not represent a particular
accelerometer datasheet, mounting class, additive noise floor, dynamic range,
or sensor hardware technology. Consequently, the defensible conclusion is
limited to modeled response-channel/DOF ranking under this registered
stress.

## Split, selection, and test firewall

The 60/20/20 train/inner-validation/outer-test split is deterministic,
state-grouped, semantic-UID stable, and stratified by family/target/anchor
level; the joint family also uses its registered latent/severity strata. A
state's passages never cross partitions. The same exact UID partition is
required at every rung of a geometry block.

Hyperparameter and candidate selection use only the development data. The
outer test remains sealed until the inner-validation winner and all registered
comparators are frozen. Finalist-only 5-fold × 2-repeat state-grouped CV uses
development states, fold-local scaling, frozen hyperparameters, and fixed
checkpoint-epoch rules at `s0`, `s16`, `s21`, and `s23` only. Its registered
set is the inner-validation winner, the top five architecture×channel
combinations from the complete inner-validation factorial leaderboard, each
architecture's own optimum, and same-channel-pair/designed/carried/full-array
controls after deduplication. It is a split-stability diagnostic and cannot
re-rank the winner; the immutable outer test is the report set.

Registered within-rung finalist MSE intervals and paired MSE contrasts use
2,000 state-first bootstrap replicates with seed 42. Most-damaged-pier
localisation is a passage-level point estimate restricted to passages whose
maximum true scour target is strictly greater than 5 percentage points; no
state-level localisation interval is registered.

On bearing rungs, training uses the registered range-normalized multi-head
loss, while model selection uses scour-head MSE. Bearing error and
scour↔bearing leakage are secondary reported outcomes, not part of the primary
selection objective.

## Two independent execution and HPO blocks

R11 has two independent physical execution blocks:

- **L60**, anchored at `s0_scour`.
- **L99**, anchored at `s21_scour4`.

Each block has its own authenticated execution receipt, full-array
hyperparameter manifest, and reference champion. The published champion
manifest carries the canonical `frozen_selection_sha256`. Never copy the L60
champion or execution receipt into the L99 block. Cross-block comparisons are
descriptive only.

All seven L60 Python ablations must run on one exact physical host, GPU, and
registered runtime; all three L99 ablations must likewise run on one exact
host/GPU/runtime, which may differ from L60. MATLAB generation may be
distributed separately, but a Python execution block must not be divided
across machines.

Before running a block anchor, set all three coordination paths to durable,
absolute paths outside disposable bundle workspaces:

```powershell
$env:TTBI_EXECUTION_RECEIPT_DIR = 'D:\ttbi-control\l60\receipts'
$env:TTBI_HYPERPARAMETER_MANIFEST = 'D:\ttbi-control\l60\hyperparameters.json'
$env:CHAMPION_MANIFEST = 'D:\ttbi-control\l60\champion.json'
```

The anchor prints the canonical SHA-256 of the published
`CHAMPION_MANIFEST`. Before any follower starts, copy that exact value into:

```powershell
$env:TTBI_BLOCK_REFERENCE_SHA256 = '<exact SHA-256 printed by the anchor>'
```

Followers fail closed if the variable is missing or differs from the canonical
hash of the regular manifest file. This operator-supplied pin prevents an
otherwise valid-looking champion file from being silently substituted between
anchor and follower runs.

Use a different L99 directory/files when starting `s21_scour4`. Relative,
missing, symlinked, malformed, cross-block, cross-runtime, or hash-inconsistent
coordination artifacts are rejected.

At the beginning of each block, the executing GPU is qualified against the
largest registered structural point of all four architectures using an
8-channel × 512 input, five output heads, forward pass, registered loss,
backward pass, and Adam update. Its **model-envelope headroom**, calculated as
total device memory minus this process's peak reserved memory, must be at least
the larger of 20% of total VRAM and 1 GiB. This is not an observation of
system-wide free VRAM and does not detect competing processes, so the GPU must
be operationally idle/exclusive during qualification and execution. The
durable capacity receipt is bound to the execution runtime, source root,
policy, and measured envelope. OOM and all Optuna `FAIL` states are fatal.

## Compute-feasible hyperparameter policy

Hyperparameters are calibrated **only** on the full eight-DOF input at each
block anchor:

- 4 architectures × 3 registered seeds × 100 Optuna trials at `s0_scour`;
- the same 4 × 3 × 100 design independently at `s21_scour4`;
- the registered pruner is enabled only for those 24 anchor studies.

The canonical manifest stores the exact best parameter map separately for each
architecture and seed. At each anchor, the same 100-trial full-array
calibration phase is also the non-selectable eight-DOF control; it is not
duplicated as a singleton. Every two-channel candidate, non-anchor full-array
control, and downstream-rung configuration runs one real Optuna trial over
singleton distributions copied from its block's authenticated manifest, with
no pruner. Thus the normal study/artifact path is exercised while
candidate-specific and rung-specific HPO lotteries are removed. Architecture
comparisons remain conditional on their separate, equal-budget finite
100-trial anchor studies and registered seed set; finite-search optimization
error is not eliminated.

Each anchor also runs a diagnostic 8 single-channel × 4-architecture × 3-seed
matrix using frozen singleton parameters. This diagnostic does not select the
block reference, which is chosen from the complete 4-architecture ×
28-two-channel matrix.

At `s16_all` and `s23_all4`, an exploratory deployment analysis reopens the
complete 4-architecture × 28 two-channel subset matrix × 3 seeds. Those are
still frozen singleton trials calibrated at the corresponding block anchor.
The exploratory winner is written separately, never replaces the L60/L99
reference, and never enters the seven-edge L60 primary analysis.

The complete campaign contains **1,620–1,638 Optuna studies** depending on
whether the designed comparator duplicates the carried pair at the six frozen
rungs, but only **3,996–4,014 actual Optuna
trials**: 2,400 anchor-search trials plus one trial in every other study. These
counts supersede the historical ~1,350-study × 100-trial projection.

Channel-subset/architecture winners support only conditional statements:
“best among the registered architectures and two-channel subsets under the
registered full-array-calibrated policy, realized finite anchor searches,
split, seed set, and simulated distribution.” They do not establish a global
optimum or the superiority of a sensor hardware technology.

## Registered L60 cross-rung inference

After all seven L60 summaries are complete, run
`analyze_cross_rung_contrasts.py` with the exact summary directory for every
L60 rung and explicit, canonical external paths to the L60 champion manifest,
L60 hyperparameter manifest, and L60 execution receipt. The analyzer
recomputes all hashes and refuses embedded or copied metadata as a substitute
for those regular files.

The primary population is only the exact paired outer-test subset of the
250-state `joint` master population. For each registered edge, the statistic is
the change in state-level scour MSE after taking the mean over paired outer
states within each training seed and the median over the finite registered seed
set. Models are trained independently at every rung using paired seeds; only
the hyperparameters, architecture, and response-channel pair are frozen.
Therefore the
effect is a change in **achievable rung-specific performance under retraining**,
not zero-shot out-of-distribution robustness.

Uncertainty uses 100,000 paired state-first bootstrap replicates. The report
contains pointwise 95% intervals and Bonferroni familywise intervals over
exactly the seven registered edges; only the familywise interval may support an
across-ladder sign claim. The bearing-by-crack difference-in-differences is
secondary/exploratory. These percentile intervals quantify empirical
state-resampling uncertainty conditional on the exact registered finite
anchor/LHS design. They are not field-population or design-superpopulation
coverage intervals; that stronger interpretation would require an LHS-aware
variance method or independent replicated state designs.

Interpret the first mechanisms precisely:

- `s0→s11` and `s12→s13` add bearing physics **and** bearing heads/multi-task
  learning, so neither contrast isolates physics alone.
- `s13→s14` changes the shared fixed baseline profile to a per-state FRA-4
  phase distribution.
- L99.6 results remain blockwise and are excluded from the seven-edge
  confirmatory L60 family.

## Claim boundary

The implemented task supports continuous modeled support-stiffness-loss
estimation, most-damaged-pier localisation, bearing estimation/leakage
diagnostics, and registered robustness contrasts under the simulated design
distribution. It does **not** yet support sensitivity, specificity, POD,
calibrated damage probability, or minimum detectable severity: there is no
development-locked binary decision threshold. It also does not establish
field validity, causal effects outside the registered L60 edges, unmodeled
damage coverage, physical sensor count/placement, or hardware-specific sensor
performance.

No R11 result exists until the complete regenerated campaign passes its
provenance checks. Historical values under `results/` must not be quoted as R11
evidence.

## Pre-dispatch MATLAB host qualification

Every PC intended to generate MATLAB campaign data must be qualified against
clean commit A before dispatch authorization:

1. Assign the PC a stable, unique label through
   `TTBI_QUALIFICATION_HOST_ID`. Do not reuse one label for different PCs.
2. Freshly regenerate the transient qualification script from the converged
   source with `python make_micro_smoke.py --qualification --stage <stage>`.
   Qualify at least `s0_scour`, `s16_all`, and `s23_all4` on every required
   MATLAB environment/host. Execute the exact source-bound script bytes on each
   intended host, never copy or forge a host receipt, and do not treat the
   generated script as durable source.
3. Retain each run's authenticated `qualification_host_receipt.json` (schema
   `ttbi-matlab-qualification-host-v1`). It records the declared host ID,
   hostname, CPU identifier, logical-processor count, MATLAB thread diagnostic,
   and computer architecture, and binds those diagnostics to the actual
   MATLAB-environment digest, canonical qualification-source digest, and exact
   executed qualification-script digest.
4. Compare corresponding stage directories for the required host/environment
   pairs with `compare_generation_releases.py` and retain an accepted
   `matlab-environment-qualification-receipt-v4` receipt. A numerically
   equivalent verdict is pending until it is explicitly reviewed and accepted
   with a new receipt.

CPU equality is not a qualification condition. Two runs with the same MATLAB
environment digest may be compared only when their authenticated receipts
declare distinct host IDs. Host identity is deliberately absent from production
`gen_schema` and `gen_fingerprint`: it authenticates independent qualification
execution without changing the scientific dataset identity.

At the dated pre-authorization snapshot above, qualification was **PENDING**:
only the pre-convergence
35-state/105-passage `s0_scour` micro on this laptop has completed
(27 min 32 s). Every intended generation host must still run all three required
stages from commit A, and corresponding stage outputs from authenticated,
distinct host IDs must receive accepted v4 comparison receipts. No second host
has produced matching evidence, so cross-host qualification is not complete
and the dated snapshot's dispatch gate was blocked.

## Dispatch sequence after authorization

The following production steps occur only after the qualification above and
the genuine R11 eight-channel benchmark gate have been completed against
commit A, independently audited, and recorded by report-only commit B. The
benchmark gate requires zero Optuna `FAIL`/OOM and no Optuna-trial
retry/replacement, plus one clean finalist refit with `attempt_count=1`,
`prior_unaccepted_attempt_count=0`, `timing_complete=true`, and
`memory_complete=true`.

1. Build one complete ten-ZIP set from authorized commit B with
   `python build_stage_bundles.py`; verify `bundle_sha256.txt`.
2. Extract every bundle into a fresh workspace. Do not resume or overlay any
   pre-R11 folder.
3. Run the bundle's MATLAB and Python fast preflights. All must pass under the
   exact locked MATLAB/Python/CUDA environment.
4. Generate raw data with MATLAB R2025b Update 5. `A00_Run.m` does not pause,
   and `parfor` may complete states out of order. As soon as the complete
   `0001.mat` appears, run the MATLAB raw-parity smoke; after it writes
   `matlab_ref_parity.mat`, run the dependent Python checker. These two checks
   run sequentially while A00 continues. Stop A00 immediately if either fails,
   and do not admit or use later output until parity passes.
5. Establish the durable L60 coordination paths; complete `s0_scour` before
   its L60 followers. Establish an independent set and complete `s21_scour4`
   before its L99 followers.
6. Run `python comprehensive_ablation_multidamage.py` in each preset bundle.
7. After all L60 rungs finish, run the registered cross-rung analyzer.

The exact preflight list and stage-specific instructions are written into each
future `README_BUNDLE.md`. Historical design decisions and audit chronology are
preserved in `docs/framework_rationale.md`; the current dispatch verdict is
only `docs/audit_r5_results.md`.
