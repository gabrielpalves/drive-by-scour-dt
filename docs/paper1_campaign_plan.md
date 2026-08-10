# Paper 1 campaign plan (bridge damages)

**Opened:** 2026-08-06
**Status:** implementation complete; final source-control and external qualification gates in progress
**Supersedes:** the retired 10-rung campaign throughout the Paper-1 production
code. It does **not** supersede
[`shm_reviewer_readiness_plan.md`](shm_reviewer_readiness_plan.md), which remains
the controlling scientific queue. This document says *what campaign to run*; that
one says *what must be true before any campaign runs*.

The four-stage generator, `physical8_v1`, fixed-width RAW/PAA models, explicit
training grid, all registered training adapters, F25 production path, and
model-form sensitivity harnesses are implemented. The final source-locked
18-case clearance matrix selected 6 m, and the track-parameter and
fixed-Rayleigh response sensitivities are complete; their exact hashes are in
[`paper1_model_form_freeze_20260809.json`](evidence/paper1_model_form_freeze_20260809.json).
Clean commit A, locked-host qualification, genuine CUDA benchmark/capacity
receipts, contact authorization, and dispatch authorization remain gates.

---

## 1. Author decisions locked 2026-08-06

| # | Decision | Choice |
|---|---|---|
| D1 | Geometry set | **F40 + L99.6. L60 dropped** from the Paper 1 campaign; it becomes a later frozen-model length sensitivity, not a coequal block. |
| D2 | Finite rail domain | **Complete.** The final source-locked 6 / 15 / 30 m coupled matrix passed all 18 cases and rule-selected 6 m under decision ID `paper1-rail-domain-clearance-c06-v1`. |
| D3 | Response channels | **Implement `physical8_v1`.** Add total wheelset accelerations; retain `acc_under` separately as virtual rail-field diagnostics. |
| D4 | HPO budget | **5 Optuna restarts × 100 trials on all 16 factorial cells** (8,000 trials), 2,000 selected-pair trials on F40-S, and 2,000 independently in each remaining block: **16,000 trials / 160 studies**. |

Scope: scour, bearing fixity, local flexural-stiffness loss (crack surrogate).
Track, rail, wheel-polygonization and suspension mechanisms are **deferred**, not
deleted — see §7.

---

## 2. Branch strategy: single line, no solver fork

**Decision: do not fork the solver.** One `main`, one commit A, one source root.

> **Settled 2026-08-09 (author).** This was briefly reversed: on 2026-08-06 the
> author chose to strip the deferred mechanisms to a branch, and the sequencing
> for that strip was written into the `ISSUES_FOUND.md` handoff while this
> section still said the opposite — a real contradiction, and my error for not
> reconciling it. Codex's 2026-08-07 review independently recommended **no
> strip**. The author confirmed no strip on 2026-08-09. **The strip sequencing
> in "Claude handoff — 2026-08-06" is superseded and must not be executed.**
> Deferring the mechanisms costs nothing extra later: stripping onto a branch
> stays available at any time, whereas stripping `main` had to happen before
> commit A or not at all.

Reasons specific to this repository:

1. `generator_source_root_sha256` hashes the whole MATLAB tree and
   `python_runtime_source_root_sha256` the whole Python runtime tree. Two branches
   produce two roots by construction. `compare_generation_releases` refuses
   cross-root comparison and qualification receipts are per-root, so data
   generated on one branch could never be pooled with, or cross-checked against,
   the other.
2. The gate suite is single-tree: 89 campaign controls, the generation /
   damage-physics / bridge-mesh mutation anchors, ~30 Python checkers, the MATLAB
   smokes. Two branches means maintaining all of it twice; the realistic outcome
   is that the experimental branch's gates rot and the merge back becomes a
   re-audit rather than a merge.
3. Removing the track/train mechanisms from a "bridge-only" branch would *move*
   the generator root and discard the mesh, ballast-inventory and descriptor work
   completed 2026-08-03, with no scientific gain.
4. Scope is already a configuration decision and it is already tested.
   `_STAGE_INPUTS` toggles bearing / crack / profile / track / polygonization per
   stage, and `smoke_stage3` proves the empty-descriptor healthy case is
   bit-identical (`max|A−B| = 0`). Code present but switched off is provably off.

**What replaces the branch:**

- Paper 1 scope = the stage list in §3 plus a scope paragraph plus
  [`damage_model_reference.md`](damage_model_reference.md). Not a code fork.
- Tag the exact tree the three hosts run (e.g. `campaign-p1-a`) so hosts cannot
  silently diverge.
- Unfinished mechanisms get an experimental branch cut **off commit A**
  (`exp/track-train-damage`), rebased forward. The branch carrying the campaign
  must always be the one with green gates.

---

## 3. Stage set

Four stages, two geometries, replacing the current ten rungs for Paper 1.

| Stage | Geometry | Damage | Purpose |
|---|---|---|---|
| `F40-S` | 2 × 20 m, Fernandes-derived | central support-stiffness loss, 0–60 % | Full RAW/PAA architecture benchmark; the gating dataset |
| `F40-M` | same bridge | scour + bearing fixity + local EI loss | Direct multi-damage extension, Fernandes-comparable |
| `L99-S` | 4 × 24.9 m | supports 2/3/4, continuous scour | Independent long-bridge multi-scour block |
| `L99-M` | same bridge | scour + bearing + local EI loss | Final bridge-damage robustness test |

**F40 mesh:** keep the support-aligned deck mesh, not Fernandes's stated 0.30 m
elements (0.30 m does not divide 20 m or 40 m). Describe it as a
"Fernandes-derived benchmark with a support-alignment mesh correction". The
2026-08-03 support-alignment work already forbids off-node positive-spring
supports, so this is enforced rather than asserted.

**One dataset, two tasks.** Generate the dense 0–60 % severity grid once.
Regression uses the full grid; the classification arm uses the
`{0, 5, 10, 20 %}` subset for direct Fernandes comparability. No extra MATLAB.
With 5 state replicas per severity that gives 5 states per class — thin, so the
classification result is a **comparability check, not a headline**. If you want
it stronger, raise replicas at the four anchor severities only; that complicates
the state matrix and every checker that pins exact state counts, so it is offered
as an option, not assumed.

**EOV set for all four stages — author-decided 2026-08-06:**

- **Profile: `fixed`.** One shared FRA-v2 class-4 phase realization across every
  state and passage. This matches Fernandes et al. (2025), who generate a single
  class-4 PSD profile (their Fig. 3a), and it makes every paired
  healthy/damaged comparison exact — which the V&V ladder in §C needs.
- **Operational variability on**: speed 70–90 km/h, temperature 3–33 °C, vehicle
  property variability.
- **Track-damage EOVs off**: ballast fouling, hanging sleepers, pad failures,
  wheel polygonization. These are the mechanisms whose priors are author-chosen
  or contested; excluding them keeps the reviewer surface small.

Profile *phase* variation is not track damage: track damage changes mechanical
properties via `B54`, whereas phase changes excitation geometry. `psd_fra`
remains implemented but disabled in all four production blocks.

**Consequence to state in limitations:** with a single fixed realization,
rail-profile irregularity is not a source of operational variability in this
data (already recorded at `framework_rationale.md:51`), so no claim of transport
to another track section is available. **Deferred to the track/train-damage work:**
re-running with `psd_fra` alongside the track mechanisms. When that happens,
evaluating the Paper 1 frozen model on varied-phase data measures *distribution
shift*, not robustness to an EOV it was trained under — a legitimate and
interesting experiment, but it must be framed that way.

### Generation budget

| Stage | States | Passages | Solves |
|---|---:|---:|---:|
| F40-S | 305 (61 severities × 5) | 50 | 15,250 |
| F40-M | 425 | 50 | 21,250 |
| L99-S | 475 | 50 | 23,750 |
| L99-M | 475 | 50 | 23,750 |
| **Total** | | | **84,000** |

**Generation is not the long pole.** Author-measured rate 2026-08-06: ~12,000
passages in under one day on a lab PC under `parfor`. At that rate F40-S is
≈1.3 days, F40-M ≈1.8, L99-S ≈2.0, L99-M ≈2.0 (add ~20 % at L99.6 for the longer
path), i.e. **≈7–8 lab-PC-days total, ≈3–5 days wall clock across the fleet**.
The home PC has half the cores, so budget ~2× there. Production is deliberately
capped at four `parfor` workers; the L99 capacity preflight must use that exact
cap rather than reopening the earlier 16-worker memory failure.

Because generation is this cheap, **regenerating F40 is never the expensive
option** — see §10. The long pole is the ablation.

---

## 4. Blocking work queue, in order

Every model-form decision that can move a number closes **before** the fleet
starts. Otherwise a week of three-PC generation is discarded — the situation the
project is already recovering from.

**Step 1 — model-form freeze.** For each open item, decide explicitly:
*(a)* change production now, or *(b)* freeze as inherited/author-chosen plus an
optional sensitivity arm. Only (b) items may be deferred past generation.

**Resolved 2026-08-09 — Codex's classifications, accepted:**

| Item | Class | Resolution |
|---|:--:|---|
| Rail-domain clearance (6/15/30 m) | **(a)** | **Closed:** 18/18 final source-locked cases passed and selected 6 m for production. |
| Per-rail-seat vs two-rail scaling | **(b)** | **Closed:** retain the inherited hybrid baseline; the prospective consistent 1×/2× sensitivity is complete and recorded. **Do not silently double** the remaining parameters. |
| 0.545 → 0.600 m spacing transfer of `Mb`/`Kb`/`Kf` | **(b)** | **Closed:** retain the baseline; the spacing-consistent Zhai-equation sensitivity is complete and recorded. |
| Ballast topology (`Kw`/`Cw` omitted; on-bridge `Mb` condensed to deck) | **(b)** | Freeze as an explicit inherited simplified topology. **If a wheelset channel wins the selection, topology sensitivity becomes mandatory before publication.** |
| Rayleigh handling under refinement | **(a)** | **Closed for the model-form freeze:** recalibrate per grid for production; the fixed-healthy-coefficient sensitivity is complete. Save α, β and reference modes at every level. |
| Rail 0.1 % Rayleigh target | **(b)** | **Closed:** retain as inherited author-chosen, not a Zhai property; the 0.05 % / 0.20 % damping sensitivity is complete. |

Zhai states its properties **per rail seat**, at **0.545 m** spacing, reports
**+12 %** ballast-acceleration overprediction without shear, and concludes that
adjacent-ballast shear is necessary for track-dynamic analysis. The TTB-2D paper
supports calling this a simplified 2-D lumped track model but does not validate
every scaling and bridge-condensation choice.

**Critical ordering consequence:** any sensitivity code needed for these
cross-model comparisons must exist **before commit A**. Adding it afterwards
moves the source root.

**Step 2 — `physical8_v1` channels (D3): complete.** Total constrained-wheelset acceleration is
`z̈_w = u_tt + 2v·u_xt + v²·u_xx + ḧ_w`; `B66_ContactForce.m` includes the
`−m·hdd_path` profile-inertia term; schema identity propagates through MAT,
loader, cache, protocol, result and plotting paths. `acc_under` remains a
separately named virtual rail-field diagnostic. Old MAT files cannot be
back-converted. The manufactured four-term fixture verifies the helper and the
saved D01 field exactly, including masking and legacy-channel preservation.

**Step 3 — contract rewrite: complete; focused mutation suites green.**
`_STAGE_INPUTS`, MATLAB setup/state matrices, semantic pairing, descriptors,
loader contracts, four-stage inference, and training-job enumeration now use
F40-S/F40-M/L99-S/L99-M. Retired ten-rung entrypoints fail closed.

**Step 4 — training side (implemented 2026-08-09; focused checks green; the
final integrated gate suite remains a commit-A closure step).**
- `MultiRatePooling1D` is now fixed-width adaptive temporal-pyramid pooling,
  shared identically by RAW and PAA; sequence length no longer changes dense
  width or parameter count.
- The implementation class is `Time2VecPositionEncoding`, truthfully described
  as "Time2Vec-style spatial-coordinate encoding". `Space2Vec` remains only as
  a compatibility alias for historical imports/checkpoint metadata.
- `training/robustness.py` now has separate development-adjudication and
  post-freeze stability interfaces. Development calls
  `core.statistical_inference.repeated_stratified_group_folds`; the sealed-test
  interface is report-only. Both require prospective seed lists, and the old
  internally generated fixed-split 30-seed routine has been removed.

**Step 5 — manuscript correction pass: complete.** The `.tex` now carries the
physical8 proxy boundary, geometry-specific mesh, inherited hybrid track
scaling/topology, damping closure, and final finite-domain decision without
claiming physical validation.

**Step 6 — closure.** Full MATLAB/Python gate suite → disposition every untracked
path deliberately → clean commit A → recompute both source roots.

**Step 7 — dispatch.** Clearance and response-sensitivity evidence is closed.
Next: clean commit A → genuine RAW benchmark/capacity preflight → host
qualification on all three PCs → contact authorization → six bundles →
generation.

### Production compute-benchmark gate (settled 2026-08-09)

The benchmark capacity prerequisite is executable only through
`capacity_preflight_compute.py`. On the benchmark RTX 5060 host, activate the
exact locked environment at clean commit A, create one canonical absolute
receipt directory outside the repository, and run:

`PYTHONPATH` and `PYTHONHOME` must be absent, not empty;
`CUBLAS_WORKSPACE_CONFIG` must be `:4096:8`. If `CUDA_VISIBLE_DEVICES` is
needed, set it once and keep it unchanged through both the capacity and
benchmark commands. The selected physical GPU must otherwise be idle.

`.\.venv-campaign-py313\Scripts\python.exe -B capacity_preflight_compute.py --receipt-dir "<canonical-absolute-external-directory>"`

The CLI establishes the registered deterministic mode with F40-S seed 104729,
derives the live F40-S execution-runtime binding, and freshly executes all 16
registered worst-case RAW/PAA CUDA probes. It refuses a dirty or changing
worktree, an environment-lock mismatch, an internal/aliased/noncanonical
directory, an existing content-addressed receipt, or runtime/source drift. The
printed create-once receipt path is the exact value supplied to
`benchmark_paper1_compute.py --capacity-receipt`; there is no qualifying
receipt-reuse mode in the publication CLI.

`benchmark_paper1_compute.py` is the dispatch-gating, non-scientific sizing
run. It must execute one **fresh, uninterrupted** 100-trial registered F40-S
anchor-HPO study for `RAW_POS1_LSTM1_MR1` on physical channel 1, with the
registered pruner and 50-epoch cap. It uses a deterministic, non-transcendental
fixture with the full F40-S population shape: **305 state groups × 50 passages,
one RAW channel × 5,831 samples**. Groups are split exactly 183/61/61
(60/20/20) into train, inner validation, and sealed-unused test; the sealed 61
groups are never returned to the selection objective.

The benchmark calls the same registered HPO helper as
`execute_ablation_pipeline`: live execution-plan derivation, exact
execution-block attestation, CUDA-capacity validation before study creation,
v6 protocol/capacity stamping, `training.trainer.Objective`, terminal-state
validation, and registered model construction. Any pre-existing trial,
interruption, `FAIL`, `RUNNING`, `WAITING`, OOM, retry, or replacement makes
that directory permanently nonqualifying; preserve it for diagnosis and choose
a new directory.

The receipt verifier reopens a copied database through real Optuna and checks
the sole study/name/minimize direction, exact contract attributes, all 100
terminal trials, parameters and distributions against the live search space,
finite intermediate/objective values and timestamps, selected trial equal to
`best_trial`, and strict loading of the champion state dictionary into the
registered one-output model. JSON, CSV, progress display, and console output
contain no objective values. The authenticated SQLite database necessarily
retains objective values because Optuna needs them to define the champion.
The receipt additionally binds the exact capacity/environment SHA, execution
receipt, external evidence directory, clean tested commit A, CUDA memory
arithmetic, call ledger, canonical artifact inventory, and root hash. Full
revalidation is permitted from clean A or from clean report-only B only when A
is an ancestor and `A..B` changes exactly `docs/audit_r5_results.md`.

---

## 5. Training and evaluation protocol

**Architecture matrix — 16 cells:**
`{RAW, PAA} × {position encoding off/on} × {LSTM off/on} × {multi-rate pooling off/on}`.

**HPO anchor channel:** front-bogie vertical acceleration, prospectively selected
from the Fernandes-style literature. **Not** car-body pitch rate: its only
support is the pre-correction results, which are invalid. Pitch rate stays a
full candidate in the channel screen. Record the anchor in source before
generation via the existing `hyperparameter_policy_sha256` mechanism.

**Protocol:**

1. Seal the state-grouped outer test set before any HPO.
2. Five Optuna restarts × 100 trials per cell on development data only — **8,000
   trials** (D4).
3. Adjudicate the five winning vectors with grouped development CV
   (`repeated_stratified_group_folds`). Never select on the outer test.
4. Freeze one vector per cell.
5. **Development adjudication** (not stability): 30 fits per cell = 5 HPO winners
   × the 3 explicit folds of one complete grouped-development OOF partition
   (prospectively fixed seed 271828, `n_splits=3`, `n_repeats=1`) × 2
   initialization seeds — **480 fits**. Every fit job carries `fold_index=0..2`;
   for each candidate/initialization the three validation folds are disjoint and
   cover every development state exactly once. This selects the frozen vector
   without opening the outer test. The 30 fits are not 30 independent
   experiments; aggregate to paired out-of-fold scores with state-clustered
   uncertainty.

   > **Corrected 2026-08-09 (Codex, accepted).** The earlier plan called this
   > matrix both the selection mechanism and an unbiased stability estimate. It
   > cannot be both: if it adjudicates the frozen vector, its spread is a
   > selection statistic, not post-selection stability. **Final stability
   > therefore needs a separate predeclared post-freeze refit set, evaluated on
   > the sealed outer test**, with its size and seeds fixed before the outer
   > test is opened. Budget that set explicitly — it is not free, and it is not
   > the 480 fits above.
6. Channel screen: four retained pipelines — best RAW, best PAA, RAW CNN-GAP
   baseline, PAA CNN-GAP baseline (deduplicate if a winner *is* its baseline) —
   over 8 singles + 28 pairs at frozen hyperparameters, 5 paired refits:
   **720 prospectively listed jobs (at most 720 fits)**. An authenticated slot
   alias cites its canonical result and performs no duplicate fit.
7. Re-HPO the four pipelines on the selected pair: 4 × 5 × 100 = **2,000 trials**.
   These are final-pair optimization and cannot be cited as evidence that the
   pair beats the other 27, which received frozen-parameter screening only.
   The count is the pre-outcome four-slot maximum; an authenticated
   winner-equals-baseline alias completes from its canonical slot without a
   duplicate Optuna study.
8. Independently HPO the same four resolved pipelines on the selected pair in
   **each** of F40-M, L99-S, and L99-M: 3 × 4 × 5 × 100 = **6,000 trials**.
   The former transport/rescue trigger is withdrawn; no block inherits another
   block's selected hyperparameters for its primary result. This is likewise a
   pre-outcome four-slot maximum; the authenticated alias rule is unchanged.
9. Freeze each block before opening its outer test. For every **unique resolved
   pipeline** in that block, authenticate all five selected-pair HPO
   completions and choose the single vector with the minimum finite canonical
   inner-validation objective; deterministic ties use ascending HPO restart
   seed and then ascending best-trial number. The immutable block-freeze
   artefact hashes the full five-restart inventory, chosen parameters and
   checkpoint epoch, selected pair, architecture, protocol/source lineage, and
   campaign run. Retained-slot aliases cite the canonical frozen pipeline and
   never duplicate compute. No outer-test index is loaded before this artefact
   and its independently deposited SHA-256 authenticate.
10. Post-freeze sealed-test stability: 30 disjoint initialization seeds for four
   retained slots in each of four blocks = **480 prospectively listed jobs**
   (at most 480 fits; 120 per block before alias deduplication).
   Each job refits the stage-local frozen vector on the complete development
   partition and reports only on the sealed outer test; it cannot select.
11. Secondary frozen transfer from F40-S remains descriptive/non-selection only:
    3 downstream blocks × 4 slots × 5 seeds = **60 prospectively listed jobs**
    (at most 60 fits).
    These jobs use the authenticated F40-S vector/architecture/pair on each
    downstream development partition and report its sealed outer test without
    making any downstream choice.

**Pre-outcome maxima: 16,000 Optuna trial slots (160 listed study jobs) +
1,740 listed refit jobs.** Actual compute can only decrease through the
authenticated retained-slot alias mapping; aliases remain in the complete job
inventory but never duplicate a canonical study or fit.

### Measured compute projection

From the contract-valid R5 benchmark (commit `7a97db1`, artifact
`docs/evidence/r5_compute_benchmark_a0793a1.json`): **one 100-trial Optuna study
= 7,175.6 s ≈ 1.99 h** (0 failed, 29 complete / 71 pruned); **one finalist CV
fold refit = 157.2 s**; peak VRAM 842 MB allocated / 1,760 MB reserved.

| Item | Count | Unit | Total |
|---|---:|---:|---:|
| Listed Optuna jobs, before authenticated alias dedup (80 factorial + 20 F40-S pair + 60 block-local) | ≤160 studies | 1.99 h | ≤318 h |
| Listed refit jobs, before authenticated alias dedup (480 + 720 + 480 + 60) | ≤1,740 fits | 157.2 s | ≤76 h |
| | | | **≈394 h ≈ 16.4 GPU-days** |

**16.4 GPU-days is a single-case extrapolation, NOT a lower bound.** (Wording
corrected 2026-08-09 — Codex is right that "floor" overclaims it.) The source
measurement is one two-channel PAA-512 study on a laptop RTX 4070; the target
GPUs and the RAW cells both differ. Three things push the real figure up:

1. The benchmark study pruned 71/100 trials. Prune rate is dataset- and
   arm-dependent.
2. **Eight of the sixteen cells are RAW.** RAW sequences are roughly an order of
   magnitude longer than PAA-512, so those cells will not cost what the
   benchmarked configuration cost. Treat any multiplier as planning risk, not
   a measured bound; the required desktop benchmark will replace it.
3. The benchmark ran on a laptop with 1.54× thermal variance between two runs of
   identical work. **Benchmark the desktops before committing to a schedule** —
   the commit message says so explicitly.

Before launching the factorial, run one RAW multi-rate cell to measure its
per-trial time and VRAM. That single measurement decides whether the 16,000-trial
design is a three-week job or a three-month one, and it is also the direct test
of whether the adaptive-pooling fix in Step 4 worked.

**Where the rigor comes from** (for the methods section — no paper dictates the
specific budgets, which are prospectively chosen): Cawley & Talbot 2010 on
selection bias; Varma & Simon 2006 on nested assessment; Bouthillier et al. 2021
on sampling multiple variance sources; Reimers & Gurevych 2017 on repeated-run
distributions; Buckley, Ghosh & Pakrashi 2023 for nested CV precedent in SHM.

**Claim boundary:** "PAA outperformed RAW across matched pipeline variants on the
Fernandes-derived two-span benchmark" — not universal superiority. F40-M and
L99 are independently tuned scientific blocks; cross-block comparisons are
descriptive, not a hardware-controlled transport effect.

---

## 6. Fleet allocation

Constraint: never confound architecture with hardware. Within each scientific
block, all compared arms use the same GPU model and numerical stack; physical
host/device UUID may differ across the two matched 5060 Ti machines.

| Host | Role |
|---|---|
| Lab A — 5950X, 32 GB, RTX 5060 Ti | Generate F40-S first (gating dataset), then all neural training |
| Lab B — 5950X, 32 GB, RTX 5060 Ti | Generate L99-S, then all neural training |
| Home — 3700X, 64 GB, RTX 2060 6 GB | Generate F40-M and L99-M; micro studies; ablation work (see below) |

**Correction to earlier advice: the 6 GB card is not disqualified.** The R5
benchmark measured peak VRAM at 842 MB allocated / 1,760 MB reserved, and its
commit message explicitly retracts the "keep the 6 GB card off ablation work"
recommendation. That does not authorize method-correlated Paper-1 work on the
2060. It may carry an entire balanced isolated F25 comparison block only after
its own RAW capacity preflight. The registered F25 science jobs are singles or
pairs (at most two input channels); the preflight additionally executes the
prospectively chosen full-eight, batch-48, five-layer, no-pool RAW k2/k5 cases
as conservative non-job 6-GB dispatch stresses. They are capacity evidence,
not F25 arms or controls. RAW multi-rate VRAM remains an unmeasured host gate.

**But keep the 2060 out of the 16-cell comparison entirely** (Codex 2026-08-09,
accepted — this supersedes my earlier "split by cell" suggestion, which was
wrong). Assigning any subset of cells to a different card correlates hardware
with pipeline, which is the exact confound the identical-GPU rule exists to
prevent, and it applies to the refit workload too since refits are also compared
across pipelines. **The full comparative 16-cell HPO and its refits run on the
two matched 5060 Ti machines.** The 2060 takes generation, smokes, the micro
convergence studies, and non-comparative sensitivity runs.

MATLAB generation is core-bound, so the 16-core lab boxes are the faster
generators — but they are also the two matched GPUs, which is why F40-S
generation must finish first and free Lab A for the factorial. Every host passes
MATLAB-environment and cross-host generation checks before it produces anything
retained.

Bundle set to replace the obsolete ten-stage builder:
`bundle_f40s_generate`, `bundle_f40m_generate`, `bundle_l99s_generate`,
`bundle_l99m_generate`, `bundle_train_labA`, `bundle_train_labB`. The two
training bundles carry disjoint machine-readable job manifests and one common
complete-grid digest. MAT data travels separately with SHA-256 manifests.

---

## 7. Deferred, not deleted

Track damage (ballast fouling, hanging sleepers, rail pads), wheel
polygonization, and suspension damage stay in `main`, toggled off by stage
configuration. They are deferred because their credible parameter ranges,
identifiability and validation are hard — not because the matrix edits are hard.
Several of their priors are already recorded as author-chosen or
contradicted-and-retained. Any future work on them happens on
`exp/track-train-damage` cut off commit A.

Also deferred: full N-HiTS as a Paper 1 arm. Authentic N-HiTS is a forecasting
model with backcast/forecast residual stacks and hierarchical interpolation;
replacing its forecast head with damage classes yields an N-HiTS-derived encoder.
Keep the current mechanism and call it "N-HiTS-inspired multi-rate pooling". A
curiosity pilot may test implementation stability, memory and runtime — not
"keep it if the validation score is good".

---

## 8. Novelty guardrails

- **Do not** claim nobody used vehicle pitch for scour. McGeown et al. 2024
  investigated vehicle pitch as a railway-scour indicator. Their reported
  quantity is pitch *angle* in radians; the stored `PitchPrimVag` channels are
  angular velocities. The defensible claim is narrower and still strong: pitch
  rate selected by a systematic channel comparison under these TTBI/EOV
  conditions, extending the earlier pitch-angle evidence.
- **Do not** claim scour + bearing + crack is itself novel. Fernandes et al.
  already considered that combination with PAA/CNN. Ours is *continuous and
  spatially resolved* where theirs is essentially discrete — say that, not
  "better".
- The crack arm is a **local flexural-rigidity loss surrogate**, not calibrated
  crack depth. Define the damaged region by a fixed physical length, or lock the
  mesh and call it a fixed-length damaged-element benchmark; otherwise severity
  is mesh-dependent.
- The adopted deck E / I / ρA / 3 % set comes from a Fernandes two-by-20 m
  example. Transferring it to L99.6 is an idealized geometry/scale stress, not
  calibration of a 99.6 m bridge.
- Pre-correction performance results guide hypotheses only. They predate the deck
  mass and subsequent corrections and cannot be reused as publication results.

---

## 9. Settled implementation ledger

- Fixed profile, operational EOV on, track/OOR off: encoded in all four stage
  contracts. Alternative profile phases are a later frozen-model distribution-
  shift experiment, not a production training EOV.
- F40-M is exactly 425 states; L99-S/M are 475. F40-S/F40-M share a controlled
  30-state semantic subset; L99-S/M are completely paired.
- The transport/rescue trigger is withdrawn. Each scientific block receives
  independent selected-pair HPO; frozen F40-S transfer is secondary only.
- Removing the old rungs and changing state semantics is recorded as
  `generation-rules-v8`, schema `audit-2026-08-09-r12`.

---

## 11. `F25-R` / `F25-X` — Fernandes (2025) reconstruction and extension

Design per Codex 2026-08-09, accepted. **Not a branch and not a second solver
lineage**: a separate *experiment configuration* on the same tree, same clean
commit A, same source hashes. Operational isolation comes from a dedicated
experiment ID, separate manifests / cache / results roots, and its own `.zip`
bundle.

- **`F25-R`** — publication-faithful reconstruction: his bridge, his ten
  scenarios, his two sensors, Min–Max + PAA, CNN baseline, his splits and EOVs.
- **`F25-X`** — extension on the *same data partitions and seeds*: additional
  physical channels, RAW vs PAA, and our architectures.

Say **"publication-faithful reconstruction", never "exact replication"**, unless
every profile realization, seed, split, preprocessing detail and hyperparameter
is actually recovered. Every choice goes in a deviation table classified as
**exactly reproduced / inferred because underreported / deliberately changed**.

**Resolved scope decision (2026-08-09):** sensor *pairs* are an exploratory
tier only, executed after all singles in a pre-registered order. The primary
comparison remains the complete eight-single-channel table.

### Extracted from the paper (verified against the PDF)

| Quantity | Value |
|---|---|
| Scenarios | 10: `Healthy`, `DC2`–`DC10` (Table 2). First row is labelled `Healthy`, not `DC1` |
| Scour | Proportional reduction of **central** support `kv`; healthy `3.44e5 kN/m` (= `3.44e8 N/m`, **identical to our `k_v0`**); 10% → `3.1e5`, 5% → `3.27e5` |
| Bearing | Healthy `kr = 0` at **both** ends; damage = `kr = 1e9 Nm/rad` at the **entrance**, taken from Feng / O'Brien as the *minimum* damage level |
| Crack | Mid-span of the **second span** (= 30.0 m). Depth ratios 0.1 / 0.05 of the cross-section → **≈22% / ≈14% stiffness reduction**, quoted **for a 30 cm element** |
| Passages | 200 per scenario, 2,000 total |
| Preprocessing | Min–Max, then PAA: `w = 583` segments × 10 values, **5,830 → 583** |
| Split | 100 train / 100 test per scenario; validation = 20% of train (20 passages) |
| HPO | Bayesian optimization, **100 trials, each configuration executed five times** |
| Training | Early stop on val loss, patience 50, max 1000 epochs; LR ×0.5 after 30 stagnant epochs, min LR `1e-6` |
| Metrics | Accuracy via confusion matrices (best run shown) + boxplots over the runs |
| EOVs | Speed 70–90 km/h; temperature 3–33 °C; noise 5%; primary suspension 2640–2920 kN/m; secondary 942–1042 kN/m; car-body mass 33,000–40,000 kg; 3 properties across 5 vehicles |
| Architectures he tried | CNN, CNN–LSTM, CNN–GRU; preprocessing Z-score / VSS / Min–Max; reduction PCA / SAX |

**The EOV set matches our campaign constants exactly** (`vel_min/max` 70/90,
`temp_min/max` 3/33, `Desvio` 0.05, `Nveh` 5, `Nprop` 3). `F25-R` needs no new
EOV code.

**Published CNN search space (verified in §4.4):** convolution filters
`32:16:128`, kernel size `2:5`, one to five convolution layers, optional
max-pooling after each layer (pool size 2), dense width `16:16:64`, batch size
`8:8:48`, and learning rate `1e-5`–`1e-2`. The two reported PAA finalists are
sensor-specific: car body = two 48-filter layers (kernels 2, 3), no pooling,
dense 48, Adam `1e-3`, batch 24; front bogie = 128/96/96 filters (all kernel
3), max-pool 2 after each layer, dense 96, Adam `5e-4`, batch 32. `F25-R`
records these as the publication-faithful baselines even though the executable
PyTorch reconstruction remains a deliberate framework deviation.

### Implementation notes that are not transcription

1. **"10% crack" is not 10% EI loss.** Set the uniform damaged-EI block to
   **22%** and **14%**. Using 10%/5% would reconstruct neither magnitude.
2. **Element index does not transfer.** Use the physical coordinate 30.0 m. Our
   production mesh is 0.20 m support-aligned and 0.30 m is not a multiple of it,
   so the damaged-zone extent cannot match exactly — and his 22%/14% are quoted
   *for a 30 cm element*. Deviation-table entry with a real numerical
   consequence.
3. **His crack mechanics are not reproduced.** Our model uses a uniform-EI
   block rather than his Sinha-style tapered local rigidity law. We preserve
   the reported equivalent 22%/14% losses and exact 0.30 m zone, but classify
   the mechanics as **deliberately changed**. `Icj`, `I0` and the section
   geometry are not implementation gaps for this explicitly surrogate arm.
4. **His HPO protocol is not ours.** 100 trials with each configuration
   evaluated five times is *not* five independent 100-trial studies. `F25-R`
   must use his protocol; our restart design belongs to the main campaign only.
5. **TensorFlow vs PyTorch** is a deliberate deviation — initialisation,
   optimiser defaults and early-stopping semantics differ.
6. **5,830 vs our 5,831-sample window**: trim exactly one sample, pin the
   convention (trim the tail so the window opens at the same place).
7. **39.9 m makes the main-campaign crop 5,821 samples, not 5,831.** Keep the
   production crop corrected. For F25 only, reconstruct the source's 58.30 m
   monitoring convention from the saved full RAW passage using the historical
   round-before-scale bridge term (`round(39.9)*100 = 4,000`), yielding the
   frozen 5,831-point inclusive grid, then trim its tail to 5,830. Record that
   this window extends 0.10 m beyond the exact 39.9 m deck relative to the
   corrected crop; `ttbi.f25_monitoring_window` pins the arithmetic.

### The extension, correctly framed

**Correction to my earlier advice, recorded so it is not repeated:** I twice
proposed quantifying "split leakage" in his passage-level split. **That does not
hold.** His design has **one structural state per class**, varied only by EOV
draws, so state-grouped splitting is impossible by construction and his split is
not leaky.

The real difference is the question being asked. His: *recognise this exact
damage configuration under EOV variation.* Ours: *generalise to unseen
structural states.* Any accuracy gap is a task-difficulty difference, not a flaw
in his work — which is also the correct framing for building on our own group's
paper.

Surviving and sharpened: **fixed crack location.** His crack sits at one
coordinate at one of two depths, so `F25-X` can test whether a model trained
there transfers to other locations, using the location sampling we already have.

Also worth adding, because his metric is ordinal-blind: his scenarios are
severity-ordered, and plain accuracy scores `Healthy`↔`DC10` the same as
`DC9`↔`DC10`. Report his accuracy unmodified for comparability, then add an
ordinal-aware metric (mean absolute class-distance or quadratic-weighted kappa).
Do **not** import MSE or localisation from the regression campaign — wrong task.

**Architecture contribution is narrower than assumed:** he already tried
CN­N–LSTM and CNN–GRU, so a recurrent block is not new. The genuinely new arms are
**multi-rate pooling** and the **Time2Vec-style positional encoding**, plus the
channels he never tested.

### Published results = the reconstruction's acceptance targets

Transcribed by the author from the paper's confusion matrices (car body and
front bogie, each RAW and full-methodology). All four tables have rows summing
to exactly 100, i.e. 100 test passages per scenario. **Source is a
low-resolution published figure — treat as approximate and record that in the
deviation table.**

**Overall 10-class accuracy** (diagonal sum / 1000):

| Sensor | RAW | Full (Min–Max + PAA) | Δ |
|---|---:|---:|---:|
| Car body vertical | 65.1 % | **86.7 %** | +21.6 |
| Front bogie vertical | 82.2 % | **82.1 %** | −0.1 |

**PAA's benefit is sensor-specific, not general.** On the front bogie RAW is
already equal-best and PAA buys only compute — consistent with his own wording
that PAA "maintains" accuracy. The car-body gain is the outlier. Two
consequences: (i) the RAW arm cannot be dropped, and (ii) the car-body gain may
be partly a model-capacity artifact, since his RAW and PAA CNNs were optimised
separately — the matched-capacity comparison is ours to make.

**Per-axis decomposition** (derived here from his matrices; each class maps to a
crack level 0/5/10, a scour level 0/5/10, and bearing present/absent):

| Axis | Car body | Front bogie |
|---|---:|---:|
| Bearing present/absent | **99.9 %** | 96.7 % |
| Scour level | 94.6 % | 92.9 % |
| Crack level | 88.4 % | 88.0 % |
| Overall 10-class | 86.7 % | 82.1 % |

This is the reporting frame, not a global ordinal metric: the ten classes are a
partial order over three independent axes, **not** a single severity ladder, so
quadratic-weighted kappa and similar are inappropriate. Report his accuracy
unmodified for comparability, then decompose. The decomposition is computable
from his published matrices, so the comparison is direct.

**Two named, diagnosed failure modes — the actual `F25-X` targets:**

1. **DC9 ↔ DC10 crack-severity separation.** They differ only in crack level
   (22% vs 14% EI reduction) with scour and bearing identical. Car body:
   DC9→DC10 = 56, DC10→DC9 = 21. Front bogie: DC10 collapses to 12% with
   DC10→DC9 = 69. Every other class is ≥ 88% on both sensors.
2. **Healthy ↔ DC4 on the front bogie.** 28 of 100 DC4 passages called Healthy
   (plus 5 the reverse) — all 33 of its bearing-axis errors. The car body gets
   DC4 = 99%. The front bogie cannot reliably detect bearing damage alone.

### Deck mass: settled, single value

Fernandes **2024** publishes "Mass per unit length | kg/m | 9.6" — physically
impossible for a deck with `I = 0.33 m⁴` (it is a ~4 cm² rod; first mode
≈130 Hz), and it is the value that arrived in this repository with his code and
sat in `A03_Bridge.m` until `1ddc881`. Fernandes **2025 §3 states 9,600 kg/m**,
so it was corrected upstream. Every other quantity in both tables matches our
`A03` and train exactly.

**`F25-R` uses 9,600 kg/m only.** No two-mass design: the 2025 confusion
matrices — our acceptance targets — already come from the corrected deck, so
there is nothing for a mass sensitivity to resolve. Record the 2024 value as
historical context in the deviation table, nothing more.

> **A prediction I made and then refuted — recorded so it is not revived.**
> Before checking the 2025 paper I proposed that the DC9↔DC10 failure was caused
> by a near-rigid 130 Hz deck: bearing damage would still register through the
> quasi-static deflection shape, while crack *severity* acts through local
> flexural dynamics a rigid deck cannot express — predicting that at 9,600 kg/m
> the separation would improve sharply. **Void.** His matrices are already at
> 9,600 kg/m. The deck was never rigid in the data we compare against.

### Source correction: the published CNN does not discard location via GAP

The deck-mass refutation leaves the DC9↔DC10 failure with no known model-mass
confound, so it remains a clean architecture target.

The mechanism maps onto the axis structure. **Scour** (support stiffness) and
**bearing** (end restraint) are *global* changes: they alter the deflection shape
across the whole passage, and both are near-solved (94.6% / 99.9%). A **crack**
is a single element at a fixed coordinate — a *local* curvature anomaly
concentrated in the short window where the wheels cross 30 m. Crack is the hard
axis (88.4% / 88.0%) precisely because its evidence is local.

**Correction from the source figure (2026-08-09):** his reported networks end in
`Flatten` plus a dense layer (27,840→48 for car body; 7,008→96 for front bogie),
not global average pooling. The earlier claim that his architecture discards
location through GAP is false and is withdrawn before any F25 run. Do not use
it in the manuscript or as a post-hoc mechanism for multi-rate gains.

Multi-rate pooling remains a prospectively declared extension because it
constructs features at several temporal scales, but its benefit is now an
**empirical question**, not a source-backed localisation prediction. Report
whether any gain concentrates on the crack axis and DC9/DC10, while bearing
and scour remain stable. A uniform lift instead indicates general capacity or
optimization effects; it does not support a crack-localisation mechanism.

**Corrected sensor hypothesis.** An earlier draft of this section aimed
car-body pitch rate at the bearing axis. That was badly aimed — the car body
already scores 99.9% there, so there is no headroom. The headroom is on the
crack axis. The useful hypothesis is that a **pitch rate is a differential
across the vehicle** and therefore more sensitive to a localised deck feature
than a single vertical acceleration — pointing it at **crack**, not bearing.
Same untested channel, testable hypothesis, right target.

**Arm justification.** The three arms are RAW-CNN → PAA-CNN → PAA + multi-rate:
a one-change-at-a-time ladder where RAW-CNN is the preprocessing control,
PAA-CNN is his method, and multi-rate pooling is the single new ingredient. Do
**not** justify this shortlist by the pre-correction results — those are invalid,
and appealing to them reintroduces exactly the circularity avoided elsewhere.
The ladder is prospectively defensible on its own.

### Frozen `F25` configuration (author-decided 2026-08-09)

| # | Item | Decision |
|---|---|---|
| 1 | Sensor pairs | **Exploratory tier only**, run after all singles. Ordering **pre-registered and documented**; block labelled exploratory. A partial set is then honest rather than a run-order selection artifact. |
| 2 | `F25-X` channels | **All 8 singles** — his 2 plus the 6 he never tested. |
| 3 | HPO regime | **Tiered**: frozen-HP screen first, then unfrozen singles, then frozen-HP pairs. |
| 4 | Runs reported | **20**, matching him. Report the distribution; best-run confusion matrix for visual comparability only. |
| 5 | Crack location | **His element 100 = 29.70–30.00 m**, reproduced exactly (below). |
| 6 | Mesh / geometry | **0.15 m deck mesh, 39.9 m bridge, spans 19.95 m.** |
| 7 | Profile | **His `Calc.ProfileData15_05.mat` realization** — faithfulness over convenience. |
| 8 | Wheelset channels | **Included** (8 channels, first two wheels). ⇒ `F25` now **depends on the `physical8_v1` workstream**. |

#### Why 0.15 m (items 5 + 6 resolve together)

His grid is 133 elements × 0.30 m = **39.9 m**, spans of **19.95 m**; element 100
spans **29.70–30.00 m** and contains mid-second-span (29.925 m), so his text and
his index agree and the bridge is genuinely 39.9 m, not 40.0 m.

**His exact mesh cannot run here:** the central support at 19.95 m falls at
element 66.5 — mid-element — and the 2026-08-03 support-alignment work rejects
off-node positive-spring supports.

A **0.15 m mesh is a uniform refinement of his grid**, so every node of his mesh
remains a node:

- 266 elements over 39.9 m; central support exactly on node 133 at 19.95 m;
- his element 100 is exactly our elements 199–200 — the damaged region is
  **reproduced identically, not approximated**;
- the 0.30 m crack zone is exactly 2 elements.

Manuscript wording: *a 0.15 m mesh, a uniform refinement of the source's 0.30 m
grid that preserves every original node including the damaged-element
boundaries, while placing the central support exactly on a node.* More faithful
than a 0.10 m/40.0 m redesign, and defensible where his own grid would not run.

#### Frozen vs unfrozen — mandatory reporting rule

The two regimes answer different questions: frozen = "how do channels compare at
settings tuned for *his* two channels?"; unfrozen = "how do they compare when
each is tuned?" **They must never appear in the same comparison table.** A table
mixing frozen and tuned channels is not a comparison. Report the complete frozen
table as primary; unfrozen results are a separate, labelled analysis.

Corollary: his protocol tuned each (sensor, method) separately, so the
**faithful** reconstruction of his two channels is the **unfrozen** one. Frozen
is our economising screen for the other six.

#### Budget, at his 500 runs per configuration (100 trials × 5 executions)

| Tier | Runs | Note |
|---|---:|---|
| Frozen-HP singles screen | 3 arms × 8 ch × 20 | ~480 fits, hours |
| Unfrozen singles | 3 × 8 × 500 = **12,000** | ~4–8 days on the 2060 |
| Frozen-HP pairs (exploratory) | 3 × 28 × 20 = 1,680 | ~14–28 h |
| ~~Unfrozen pairs~~ | 42,000 | **Not viable — do not plan for it** |

### Remaining gaps

The decisive move is to **read his code, which the author has, rather than the
paper** — it converts several rows from "inferred" to "exactly reproduced".

- [x] **Profile provenance — CLOSED 2026-08-09.** The author confirms Fernandes
  used `Profile.Type == 2` to load `Calc.ProfileData15_05.mat` in his own
  campaign, and that the file arrived with his code. Reviving that branch for
  `F25-R` is therefore *his own code path*, not a workaround, and it is the
  faithful choice. The main campaign still generates its own FRA class-4
  realization. Consistent with the paper: PSD-generated once, saved, reused.
  - [x] **Executable provenance consequence — CLOSED 2026-08-09.** The asset
    was already present in `bundle_source_files.txt` and therefore already in
    the reviewed generator source root (the earlier "not hashed" statement was
    stale). `ttbi.f25_experiment_config` now also pins its exact SHA-256 and
    `framework_rationale.md` item 15 is superseded. The no-solver F25 smoke
    verifies that the stored 325.8 m realization covers the live F25 domain.
- [x] **Bridge length — CLOSED 2026-08-09.** The source grid and stored
  configuration resolve the bridge at 39.9 m with two 19.95 m spans. The
  executable F25 contract realizes 266 × 0.15 m elements, places the central
  support on node 133 (zero-based), and maps source element 100 exactly to
  refined elements 199–200.
- [ ] **Reported accuracies.** Published as a low-resolution confusion matrix
  plus boxplots. If exact values cannot be recovered, the deviation table must
  say the comparison against published numbers is approximate.
- [x] **Run count — CLOSED 2026-08-09.** Section 5.1 states that the CNN is
  executed 20 times for the reported accuracy distributions. F25 reports all
  20 runs and uses only the best-run confusion matrix for the paper-matched
  visual, never as the distribution summary.

---

## 10. Can the old 40 m results be reused? No.

Asked 2026-08-06. The answer is not a judgement call — the repository already
recorded it, and the physics is decisive.

**The deck mass was 1000× too light.** `A03_Bridge.m` carried
`Beam.Prop.rho = 9.6` while `B43` sets `Prop.A = 1 m²`, so that value *is* the
deck mass per unit length: 9.6 kg/m, about a 0.004 m² rod, not a deck. Verified
with the model's own solver (`plotting/beam_freq.py`):

| Geometry | ρ = 9.6 | ρ = 9600 (correct) |
|---|---:|---:|
| L40 / 2-span | **130.6 Hz** | **4.13 Hz** |
| L60 / 3-span | 132.2 Hz | 4.18 Hz |
| L99.6 / 4-span | 86.9 Hz | 2.75 Hz |

A ~131 Hz deck is effectively **rigid** to a vehicle whose car body sits at
1–3 Hz and bogies at 10–30 Hz. The vehicle–bridge *interaction* — the entire
physical basis of drive-by monitoring — carried almost no deck dynamic
signature. The classifier still read scour because support-stiffness loss still
changes the quasi-static deflection, **which is exactly why it hid for so long:
the ML worked, so nothing looked broken.**

`framework_rationale.md:182` states the consequence directly: all previously
published numbers are *"invalid, not merely superseded — they describe an
unphysical bridge; the paper drafts must stop quoting them even as 'direction'."*

The mass is not the only defect the 61-class campaign carries:

- **Ungrouped passage-level splits** at all five split sites (seed-42
  `train_test_split` over passages) put ~40 train / ~10 validation passages of
  the *same* state, same label, same realization. Every reported MSE measures
  same-state interpolation, not generalization.
- **A 0.5 mm white per-passage profile jitter** that was band-unlimited, adding
  ~0 in-band but ~125 g-equivalent fictitious short-wavelength forcing — loading
  precisely the wheel channels the sensor-placement finding rests on.
- **Speed/temperature LHS transposed**: correlation −0.75, with the (slow, cold)
  and (fast, hot) quadrants never sampled.
- **The 0.3 m deck grid** does not divide a 2 × 20 m bridge, so the central
  support — whose stiffness loss *is* the label — sat off-node until the
  2026-08-03 support-alignment fix.

So "2 sensors ≈ 8" and "PAA + N-HiTS wins" are **hypotheses, not results**.

**Why "reuse them if L99-S agrees" does not work.** Agreement between a correct
L99.6 dataset and an incorrect F40 dataset would show the architecture ranking is
robust across those two datasets. It cannot make the old numbers reportable,
because the old numbers describe a bridge that does not exist. It would also put
two different simulators' outputs in one results table, and the ρ fix is in the
public repository history for any reviewer who reads the methods.

**The legitimate use, which is genuinely valuable.** Cite the old campaign as the
*exploratory basis* for the prospective design — it is what makes the
architecture shortlist and channel candidates pre-declared rather than fished.
Label it pre-correction, report no numbers from it. That strengthens the
"we did not fish" argument at zero cost.

**And regenerating is cheap.** F40-S is 15,250 solves ≈ **1.3 lab-PC-days** at
the measured 12,000 passages/day. Reusing the old results saves about one day of
generation and costs the paper its foundation. Regenerate.
