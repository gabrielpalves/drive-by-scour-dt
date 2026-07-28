# Audit r4 handoff — response to r3, change log, and mandate

> **HISTORICAL HANDOFF — SUPERSEDED BY R11.** This preserves the R4 review
> record, including terminology later corrected. It does not authorize current
> generation, ablation, bundle dispatch or paper claims.

Audience: the external auditor (Codex), round 4. Round 3 was delivered 2026-07-22
against `main@a13ea7f` plus a dirty working tree. Since then two commits landed:

- **`a2d6aaa`** — "Audit r3 (Codex, verified) + contact gate 24 kN" (18 files, +574/−64):
  every r3 blocker either fixed or explicitly queued (this document is the ledger).
- **`e37b3fe`** — `bundle_sha256.txt` manifest; all 10 bundles rebuilt FROM `a2d6aaa`.

Review diff: `git diff a13ea7f..e37b3fe`. The tree at handoff is clean except
untracked artifacts (bundle zips, `results/`, `presentation/`). Full narrative:
`docs/framework_rationale.md`, entry "EXTERNAL AUDIT ROUND 3".

**Process note:** all 11 of your central claims were independently re-verified
against the code before any fix (10 confirmed, with corrections noted below). The
user's directive for this round: **correctness first, no schedule pressure** — runs
were stopped wherever needed. This round you MAY modify code, under the ground
rules in §3.

---

## 1. Ledger — your r3 findings, one by one

### 1.1 PAA — FIXED (your finding was correct and important)
`core/preprocessing.py::_apply_paa` is now true Keogh PAA: exact window means via
integration of the sample-and-hold signal, fractional windows for non-divisible
lengths, global mean preserved exactly, chunked float64 accumulation. Your
`[0,0,0,4] → [0,2]` example is a pinned regression test. New `check_paa.py`
(18 checks, including a brute-force per-sample-overlap reference implementation)
ALL PASS. Caches orphaned via `CACHE_SCHEMA_TAG _gs5→_gs6`; `"paa_impl"` added to
`PREPROC_PROTOCOL` (hash-carried). The name "PAA" is kept because the transform
now *is* PAA — your rename-vs-fix dilemma was resolved by fixing.

### 1.2 Contact — PARTIALLY fixed; closure study ADOPTED and queued
- `B66_ContactForce.m`: on-track mask now `[0, Calc.Profile.L]` in BOTH solver
  branches (matches B50's off-profile force zeroing). You were right that the old
  mask diluted the `tension_frac` denominator.
- The 24 kN two-tier gate stands, with its post-hoc character acknowledged in the
  rationale (recipe: ≈2× worst observed event, rounded to 20% of the ~118 kN
  static load; 7× below the known-regression scale of 144% static). The watch
  item is renewed and binding: **no further raises**; a tens-of-kN or sustained
  event escalates to solver work or censor-with-report.
- Your Δt-refinement proposal is **adopted as the closure** and queued: rerun the
  two flagged states (s23 state 24; s15 state 244) at Δt = 1 / 0.5 / 0.25 ms and
  compare against gates 0/12/24 kN. This is mandate item (c) — you may build it.

### 1.3 Track-EOVs — FIXED (both your concrete findings confirmed)
- Hanging↔fouling coupling: acceptance odds corrected 9:1 → the documented
  **3:1** (`A00_Run.m`; the adjacent ballast-transition block already had the
  correct pattern, which confirmed the intent).
- Overlapping ballast patches: product → **governing-patch rule** — largest
  |log η_k| wins and supplies BOTH η_k and η_c (a sleeper is predominantly wet-
  or dry-fouled; mixing quantities across patches would be an unphysical
  hybrid). `B54_ModelMatrices.m` + the Python mirror
  `TTBI_2D/b54_model_matrices.py` changed in lockstep.
- **Consequence accepted**: s15/s16/s23 data generated before 2026-07-22 is
  invalid (the 9:1 draw is distribution-wide) and is being regenerated. All
  other rungs' data stands.
- Settlement/permanent profile dip as a *geometric* EOV: queued as a separate
  future EOV; the current voids-as-stiffness-removal representation remains and
  is honestly documented (`docs/track_eov_sampling_spec.md`, updated).

### 1.4 Noise — pairing FIXED; model choice defended; additive arm queued
- The RNG is now keyed `[NOISE_RNG_SEED, global_dof]`: a channel receives the
  identical realization regardless of subset or order. Your alone-vs-in-pair
  reproduction was correct and is closed.
- The uniform 5% multiplicative `all_mult` is KEPT as the campaign's deliberate,
  channel-symmetric stress model. Your physical point is accepted: additive
  datasheet floors (noise density × √BW; ADIS16488 / AXO305 / GYPRO4300 in the
  reference shortlist) are the correct sensor model — queued as the
  noise-robustness arm; it needs the datasheet constants transcribed, not code
  alone. (Our EN 61373 caveat — environment severities are NOT noise floors —
  is already documented and stands.)

### 1.5 Objective — FIXED, pre-registered
- `task.objective_value` = **scour-head MSE** on bearing rungs (bearing heads
  trained and reported, never selected on); aggregate elsewhere (identical
  there).
- Training loss = `WeightedHeadMSE`, w_h ∝ 1/range_h², normalized to mean 1
  (SCOUR_RANGE_PCT=60, BEARING_RANGE_PCT=95).
- Both recorded in `TRAIN_PROTOCOL`; selection_metric strings updated in the
  driver and `core/protocol.py` → protocol hash carries them.

### 1.6 Architecture / sensor-count controls — FIXED
- 4th arm **`PAA_CNN`** (`use_nhits=False` → the model's native global-average-
  pool path): the first no-pooling control in the project's history, Part I
  included. Your observation that no arm ever lacked the module was correct.
- **`CONTROL_SETS = [ALL_DOFS]`**: the full 8-DOF array runs at every rung as a
  NON-selectable comparator (selection filters `n_sensors == 2`; the champion
  publish gate refuses non-pairs; the control joins the pre-registered
  comparator set so its test row is always reported). "2 ≈ 8" becomes a
  per-rung measurement.
- Naming discipline adopted: the module stays `MultiRatePooling1D`; paper text
  will say "multi-rate pooling (N-HiTS-inspired)", never claim N-HiTS.

### 1.7 Statistical inference — PARTIALLY adopted; the rest is YOUR MANDATE (§3b)
Adopted now: the scour-primary estimand (1.5) and the full-8/no-pooling controls
(1.6) close part of the claims gap. Acknowledged as real and open: winner's curse
over the candidate grid on one validation split; `nuisance_only = 6` leaves ~1–2
independent test states for the false-positive probe; the median-of-CI-bounds
concern; single grouped split. See mandate.

One pushback, for the record: "the ladder is not one-factor" is overstated. The
DATA-generating process changes exactly one factor per rung; per-rung re-HPO is
part of the estimand ("best achievable at this rung"), not a confound. We accept
the wording discipline (attribution claims are about the data-side factor) but
keep the design.

### 1.8 Provenance — PARTIAL; remainder queued (§3d)
Done: missing best-trial weights now FATAL (`training/pipeline.py`); `RUN_TAG`
joins `SUMMARY_DIR`; cache provenance pins SHA-256 of `file_digests.mat` +
`damage_states.mat` (a digest-chain-consistent tamper invalidates the cache);
`bundle_sha256.txt` written by the builder; bundles rebuilt from a clean commit.
Residual, documented and accepted for now: an *inconsistent* same-size edit of
one `NNNN.mat` is ignored while a cache exists (the cache holds pre-edit bytes)
and is caught at the next rebuild by the per-file SHA validation.
Queued: full per-file SHA verify on cache reuse; `study.user_attr` protocol
dump; cryptographic weights↔best-trial link; environment lockfile; A00-side
digest coverage of `damage_states.mat`/`case_info.mat`.

### 1.9 Documentation
`README_CAMPAIGN.md` de-staled (1000 N → 24 kN two-tier; `_gs2` → `_gs6`; "OOR &
flats" corrected; claim wording per your terminology list). Your 49 CFR §213.9
reading is accepted: FRA-4 will be presented as a roughness *benchmark*
conditioned on service type, not a legal minimum (Class 3 is passenger-legal at
our speeds and rougher). `paper1_methodology.md` is deliberately NOT patched
piecemeal — it is known-stale wholesale and will be rewritten from
`framework_rationale.md` at paper-writing time; treat it as non-authoritative.

---

## 2. Decisions of record — do not relitigate without new evidence

1. Crack is a nuisance, not a head (inspectability + pathway rule); bearing is a
   head (shares scour's support-stiffness pathway).
2. Training uses an INDEPENDENT scour LHS; pier-to-pier correlation lives in the
   digital twin (likelihood-vs-prior factorization).
3. No censoring/resampling of severe states — MNAR; refused three times.
4. Optuna studies are never extended; re-runs start from scratch under a new tag.
5. Per-rung re-HPO stays; ladder attribution claims are data-side.
6. Contact gate: 24 kN / 0.2% / non-finite is final absent a materially larger
   event (then: solver work or censor-with-report, not a raise).
7. s15/s16/s23 regenerate; all other generated data stands; the s0 ablation
   re-runs under the new protocol hash.
8. Scour is described as *support-stiffness loss*; bearing φ is a *nominal
   fixity DOE parameter*; crack is a *damaged-element EI reduction*.

---

## 3. Mandate for r4 (code changes allowed)

In priority order:

**(a) Adversarially re-verify the r3 fixes.** Especially: PAA numerical edge
cases (very short signals, n > L, dtype/copy semantics, the chunk boundary);
`WeightedHeadMSE` under the real trainer (device moves, 1-D head case, AMP if
any); `CONTROL_SETS` interactions (pair-matrix completeness gate, preflight,
median leaderboards, `n_sensors` derivation, deployment rungs where all arms
run); per-DOF noise under `legacy_wheel`/`sprung_mult` modes; `RUN_TAG` summary
dirs vs champion manifest paths; the B54 governing-patch rule vs the Python
mirror (write an overlap parity test — none exists).

**(b) Statistical-inference upgrade — design AND implement.** Repeated
stratified grouped CV for finalists (winner + comparators only; budget-bounded);
hierarchical bootstrap that resamples STATES first and computes the cross-seed
statistic inside each replicate (fixing the median-of-CI-bounds issue); paired
per-state contrasts (winner vs each comparator); an MCSE-based recommendation
for family sizes at the next full regeneration (nuisance_only = 6 is known-thin).
HARD CONSTRAINTS: the outer test-once protocol is untouchable; everything flows
through `core/protocol.py` descriptors (no hand-made tags); deterministic seeds;
ship a `check_*.py` for every new mechanism, in the existing pattern.

**(c) Contact-closure harness (MATLAB).** A tool to re-run a named state/passage
at Δt ∈ {1, 0.5, 0.25} ms and diff signals/contact logs across gates 0/12/24 kN,
for s23#24 and s15#244. Output: a small report the paper's verification appendix
can cite. Do NOT change the gate itself.

**(d) Provenance completions** from §1.8's queue, if budget remains.

### Ground rules for your code changes

1. Any change that alters FEATURE BYTES → bump `CACHE_SCHEMA_TAG`. Any change to
   selection/preprocessing behavior → a named entry in `PREPROC_PROTOCOL` /
   `TRAIN_PROTOCOL` / the protocol descriptor. No silent knobs.
2. Keep every existing suite green and extend it: `check_paa.py`,
   `check_loader_provenance.py` (45), `check_protocol_hash.py`,
   `check_cache_provenance.py`, `check_split_grouping.py`; MATLAB `smoke_audit`,
   `smoke_stage3`, `smoke_geometry`, `checkcode` (A00 baseline = 41 benign).
3. MATLAB edits: `checkcode`-clean, parfor-safe (no loop-variable collisions),
   and covered by a smoke assertion.
4. Do not touch `data/`, `results/`, or the bundle zips; do not rebuild bundles
   (that happens from a clean commit on our side). Do not modify generated
   datasets or their digests.
5. Known-by-design, do NOT "fix": the LHS anti-transposition guard fires below
   Npass=10 (micro-smoke bypass is documented); `legacy_wheel` load-time noise
   only approximates the baked Stage-0/1 noise (documented); hanging sleepers
   are binary support removal (documented); bundle zips are untracked artifacts.
6. One logical change per commit, message states WHAT and WHY. If you believe
   something is wrong but out of mandate — WRITE IT UP, do not change it
   silently. Flag disagreements with §2 as findings, not edits.

### Green-state reference (reproduce before and after your changes)

```
python check_paa.py                    -> PAA: ALL PASS (18)
python check_loader_provenance.py     -> LOADER PROVENANCE: ALL PASS (45)
python check_protocol_hash.py         -> ALL PASS
python check_cache_provenance.py      -> CACHE PROVENANCE: ALL PASS
python check_split_grouping.py        -> SPLIT GROUPING: ALL PASS
matlab -batch "cd scour_MATLAB; smoke_audit"     -> SMOKE AUDIT: ALL PASS
matlab -batch "cd scour_MATLAB; smoke_stage3"    -> SMOKE STAGE-3: ALL PASS
matlab -batch "cd scour_MATLAB; smoke_geometry"  -> SMOKE GEOMETRY: ALL PASS
python _stage3_smoke.py               -> STAGE-3 SMOKE PASSED (mirror parity)
```
