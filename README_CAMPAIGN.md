# Multi-damage campaign — the 10-rung ladder (2026-07-15)

Everything is **regenerated from scratch**: new ladder, raw/Option-B data format,
anchored EOVs, noise-free generation. The earlier Stage-0/1 datasets and the L100
Stage-2 pilot are **superseded** (kept only as a historical record under `results/`).

> ## 🔴 AUDIT R4 (2026-07-25) — audit-snapshot ZIPs are not dispatchable
>
> The `bundle_*.zip` files present at the time of this audit predate the R4
> methodology/provenance changes. Do not dispatch that snapshot. After
> independent review, commit the reviewed source, then rebuild from that clean
> commit. Dispatch only a complete ten-ZIP set listed in `bundle_sha256.txt`,
> whose `source_commit` equals the reviewed commit and whose SHA-256 entries all
> verify. The builder reads the explicit tracked `bundle_source_files.txt`,
> refuses dirty runtime inputs, invalidates the old manifest before replacing
> any ZIP, and publishes a new manifest only after the complete set is built.
>
> R4 adds: true finalist-only 5-fold × 2-repeat **state-grouped** CV; fold-local
> scaling; an immutable outer-test firewall; state-first cross-seed bootstrap and
> paired contrasts; top-5 and per-architecture finalists; exact pair/seed matrix
> guards; protocol-stamped Optuna studies and exported weights/scalers; full
> source-file verification on cache reuse; a Python/CUDA environment lock; and a
> MATLAB contact time-step closure harness. The complete evidence and remaining
> gates are in `docs/audit_r4_results.md`.
>
> Two empirical gates remain before confirmatory publication claims: run the
> 1/0.5/0.25-ms contact-closure study on the regenerated s23 state 24 and s15
> state 244, and complete the actual campaign. The six `nuisance_only` states
> leave only one outer-test state; that isolated false-positive probe is
> **exploratory**, not confirmatory. The main joint-state estimation experiment
> is unaffected by that wording constraint.
>
> ## 🔴 AUDIT 2026-07-17 — bundles built before this date are INVALID
> An external code audit (verified line-by-line) found seven defects; all are
> fixed in this tree (full record: `docs/framework_rationale.md`, 2026-07-17
> entry). Highlights: speed/temp LHS was transposed (corr −0.75, two EOV
> quadrants never sampled — ALL rungs); track-layer damage landed entirely off
> the deck (bridge really starts at ~123 m, not 30 m — s15/s16/s23); FRA PSD
> corner was in rad/m on a cycles/m axis (~16× short-λ power — s14+); the
> 0.5 mm physical profile jitter is removed; **wheel flats are disabled**
> (bilateral solver can't represent the separation they cause; polygonization
> stays); wheel–rail tension is now logged per passage (`contact_log`); the
> Python split is now **grouped by damage state** (caches retag to `_gs1`);
> Stage 0 again runs **all architectures** (the old stage-name check was wrong) and
> the pre-mass-fix `CHAMPION_ARCH` must be re-selected from the new s0 run.
> **Before generating anything:** rebuild bundles from this tree
> (`python build_stage_bundles.py`), run `smoke_audit.m`, `smoke_stage3.m`,
> `smoke_geometry.m` (MATLAB) and `python check_split_grouping.py` — all must
> print ALL PASS. If a PC already started from a 2026-07-16 bundle, stop it and
> discard.
>
> ### Round-2 audit (2026-07-17) — additional rules
> - **Extract each bundle into a FRESH folder.** A00 now writes a `gen_schema`
>   into the manifest and every `NNNN.mat` and **aborts** if you resume a folder
>   written by different code. Never extract a new bundle over an old `Results/`
>   run — move the old folder aside first.
> - **`fixed`-profile crop fixed:** B19 no longer reverts the bridge to 39.9 m,
>   so every rung crops its true deck (s21/s22 at L99.6 now span all piers). The
>   `s13→s14` step is once again one-factor.
> - **Run `s0_scour` FIRST.** It now selects the architecture from the new
>   leaderboard and writes `results/_champion_arch_<schema>.json`; every later
>   rung reads it and **errors if it is missing** (the old PAA_NHiTS default is
>   gone). Its Python ablation must finish before s11+ ablations start.
> - Optuna studies carry a `SCHEMA_TAG`, so a re-run on a reused results/DB dir
>   cannot resume pre-audit studies. Contact gate (recalibrated twice; final
>   2026-07-22): TWO-TIER on both the generator and the loader — brief
>   micro-unloading is tolerated + logged (peak tension ≤ 24 kN ≈ 20% of the
>   static wheel load AND ≤ 0.2% of path samples); beyond either bound, or
>   non-finite, is FATAL (physics regression).
>
> ### Round-3 audit (2026-07-17) — multi-PC workflow
> - **Study/DB isolation:** every rung's Optuna DB, output and cache dir are now
>   STAGE-prefixed, so running several rungs in one workspace can't cross-
>   contaminate (previously `s11` could resume `s0`'s studies).
> - **Champion across PCs:** after s0 finishes, copy `results/_champion_arch_
>   <schema>.json` to each s11+ PC (or point the `CHAMPION_MANIFEST` env var at
>   it). Bare architecture/pair overrides are rejected: only the complete
>   manifest proves the s0 split, budget, protocol hash, pair and `RUN_TAG`.
>   Every later rung hard-errors if that authenticated lineage is absent.
> - **Provenance:** A00 writes a `gen_fingerprint` (Npass/seeds/severities/…) and
>   aborts resume if it differs; the Python loader requires `gen_schema` +
>   `contact_log` in every `.mat`. Caches are tagged `_gs6` (all older tags
>   orphaned; gs6 = true Keogh PAA + per-DOF-paired noise, audit r3 2026-07-22).
> - **Profile fix (my R2 bug):** the fixed-baseline profile is now stretched
>   (not wrapped) onto the live track, so L99.6/fixed no longer has a seam that
>   caused wheel-rail contact loss. `smoke_geometry` checks contact at 70/80/90
>   km/h.

**One bundle per rung.** A newly rebuilt `bundle_<stage>.zip` is self-contained —
MATLAB generator + Python ablation, both **preset for that rung**, plus the
preflight smokes. Extract into a fresh folder on a Lab PC and run; no editing.
Build only from a clean, independently reviewed commit with
`python build_stage_bundles.py`. Each bundle carries its own `README_BUNDLE.md`.

## The ladder

Within each declared ladder sequence, one data-generating factor is added per
rung; attribution is conditional on the fixed ladder architecture/placement and
the simulated design distribution. The separately labelled deployment selections
are not causal ladder contrasts. Heads = **scour + bearing only**; crack / rail
profile / track-layer / wheel damage are **nuisances** the network must be
invariant to (it never estimates them).

| Bundle | Adds | Geometry |
|---|---|---|
| `bundle_s0_scour` | scour only — baseline + architecture selection | L60 / 3-span, piers 2,3 |
| `bundle_s11_bear` | + bearing (head) | " |
| `bundle_s12_crack` | + crack (nuisance, no bearing) | " |
| `bundle_s13_bearcrack` | + bearing + crack = all **bridge** damages | " |
| `bundle_s14_prof` | + rail profile (FRA-4, per-state) = **the roughness rung** | " |
| `bundle_s15_track` | + track-layer damage (ballast / hanging sleepers / pads) | " |
| `bundle_s16_all` | + wheel OOR (polygonization; flats disabled) = **all modeled EOVs**; **deployment selection** (4 archs × 28 pairs) | " |
| `bundle_s21_scour4` | scour only | **L99.6 / 4-span**, piers 2,3,4 |
| `bundle_s22_bearcrack4` | + bearing + crack | " |
| `bundle_s23_all4` | all modeled EOVs; **deployment selection** (4 archs × 28 pairs) | " |

**Audit r3 (2026-07-22) — applies to every bundle rebuilt after this date:**
true Keogh PAA (window means, not linear resampling); noise realization keyed
by global DOF (sensor-set comparisons are noise-paired); SCOUR-primary
objective (bearing heads trained + reported, never selected on;
range-normalized multi-task loss); 4th architecture arm `PAA_CNN` (no
multi-rate pooling — the pooling-ablation control); the full 8-DOF array runs
at every rung as a NON-selectable sensor-budget control; contact gate 24 kN
two-tier; hanging↔fouling coupling corrected to the documented 3:1 odds and
overlapping ballast patches resolved by worst-patch-governs (**s15/s16/s23
generated before this date must be REGENERATED**; all other rungs' data
stands); tagged re-runs get their own summary dir.

**Deployment selection (Feature B, 2026-07-19):** the two all-EOV rungs
(s16_all, s23_all4) re-open the architecture question — 4 archs × 28 pairs × 3
seeds = **336 studies each**. Their winner is EXPLORATORY: it goes to a separate
`results/_deployment_selection_<stage>_..json` manifest (with a state-level
bootstrap 95% CI on the outer-test MSE) and **never overwrites the s0 ladder
champion**. The claim it supports is exactly *"best among these 4 architectures
× 28 two-sensor pairs at this budget + geometry"* — not a sensor-count or global
optimality claim. The full ladder requires approximately **1,344–1,359 Optuna
studies** (134,400–135,900 useful trials), depending on whether the designed
pair duplicates the carried pair, plus at most about **2,040 fixed-hyperparameter
finalist-CV refits** after comparator deduplication. Benchmark wall time before
dispatching the complete ladder.

Folder names are **short** (`<stage>_L<len>_st<N>`, e.g. `s14_prof_L60_st300`) —
the old ~110-char names blew past Windows MAX_PATH. The full descriptor lives in
`case_info.case_desc` / `case_info.txt`.

State counts since **Feature A** (2026-07-19, explicit state families):
250 joint LHS + 12 `target_healthy` + per-pier `scour_only` anchors
(4 levels × 2 replicas) + per-abutment `bearing_only` anchors (bearing rungs)
+ 6 `nuisance_only` (crack rungs). E.g. s0 = 278, s11 = 294, s12 = 284,
s13–s16 = 300, s21 = 286, s22/s23 = 308. Every family is guaranteed present in
train / inner-val / outer-test by the stratified split (`split_manifest.json`
is written next to the data and verified on every load). Presence is not the
same as adequate inferential sample size: `nuisance_only=6` maps to only one
outer-test state, so its isolated false-positive result is explicitly exploratory.
The R4 MCSE report gives the next-regeneration recommendation (design floor:
50 total states for ten expected evaluation states at a 20% allocation).

## Per-bundle run order

1. **MATLAB — generate.** Open `scour_MATLAB/A00_Run.m` and run it. `STAGE` is
   already set and `use_signal_noise = false` (noise is a load-time model).
   Output → `scour_MATLAB/Results/<case>/`; move it under `data/`.
2. **Once per campaign — verify the transform.** After the FIRST state of the
   first rung exists:
   `smoke_raw_parity('Results/<case>')` in MATLAB, then
   `python check_raw_parity.py "scour_MATLAB/Results/<case>"`.
   **Must print `PARITY PASS`** (max|MATLAB−Python| < 1e-12) before the long runs.
3. **Python — ablate.** `python comprehensive_ablation_multidamage.py`
   (STAGE + SENSOR_NOISE preset), with 100 useful Optuna trials × 3 seeds per
   configuration. `s0` runs the four-architecture × 28-pair factorial (plus a
   diagnostic single-DOF sweep); `s16/s23` repeat the full joint factorial for
   exploratory deployment selection; `s21/s22` search all pairs on the carried
   architecture; frozen rungs run the carried pair, designed `[1,3]` comparator
   and full-array control. Selection uses inner validation only. Finalist CV uses
   development states only. The sealed outer test is opened after both freeze.
   Summary → `results/<stage>_summary_ph-<hash>/`.

Generation and ablation run **concurrently** — with 3 PCs, generate the next rung
while the previous one ablates.

## What changed (and why)

- **Data format (Option B).** D01 saves the **raw, un-interpolated, noise-free
  time-domain** signal + the space/crop parameters; Python rebuilds the space
  window at load time (`core/dataset._raw_to_space_crop`, an exact `interp1`
  mirror). The measurement model now lives entirely at load time and can change
  **without regenerating**. Storage is ~neutral (`DimSpace ≈ 2.2·DimAcel`).
- **Noise.** Generation is noise-free; the ablation injects zero-mean Gaussian
  multiplicative noise with pointwise σ = 5% of `|signal|` on **every** channel
  at load time (`all_mult`). Adding noise
  before vs after the interpolation is *not* equivalent (time-domain noise becomes
  coloured + speed-dependent: ≈0.67× variance but ≈1.46× the energy surviving PAA).
- **EOVs are anchored** (`docs/track_eov_sampling_spec.md`): persistent conditions
  drawn **per state** (EN 13848-2); FRA-4; crack p=0.25 **hogging-weighted 4:1**
  (Eurocode 4); hanging groups **Poisson λ=3.0/100 m**, ballast **λ=1.2/100 m**
  (both window-scaled); pads **p=0.02** (snapshot prevalence, not the annual rate);
  fouling↔voiding **coupled ×3**; ballast **×3** near abutments.
- **Provenance.** Re-ablations start **from scratch** via `RUN_TAG`; the noise mode
  enters the study name, so a noise A/B on identical data trains separate studies
  instead of silently resuming. Protocol descriptors, dataset/source hashes,
  study records, weights, scalers, environment lock and bundle source commit are
  cross-checked before reuse or publication.

## Claim boundary

The implemented confirmatory task is **continuous scour support-stiffness-loss
estimation and most-damaged-pier localisation** under the simulated design
distribution. Do not claim a binary detector, sensitivity/specificity, calibrated
probability of damage or minimum detectable severity: no development-locked
detection threshold has been implemented. The 5% localisation threshold is a
metric eligibility rule, not a binary alarm threshold.

## Requirements

Use **Python 3.13.3** with the direct-dependency versions in
`requirements-campaign-py313-cu128.txt`; the driver verifies
`environment/campaign-py313-cu128.json`, CUDA/cuDNN and deterministic cuBLAS
configuration before creating a study. Use **MATLAB R2025b** (hard-gated in
`A00_Run.m`) with the Statistics and Machine Learning Toolbox (`poissrnd`,
`wblrnd`, `lhsdesign`). The lock is an exact direct-dependency/runtime lock, not
a fully hashed transitive package archive. Valid studies resume only until the
exact pre-registered useful-trial budget is met; manually extended studies are
rejected.
