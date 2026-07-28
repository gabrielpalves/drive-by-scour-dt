# External audit R4 — implementation result and publication gate

> **HISTORICAL AUDIT SNAPSHOT — SUPERSEDED BY THE CURRENT R11 GATE.** This
> report applies only to checkpoint `f805fbe`. It does not authorize R11
> generation, ablation, bundle publication or current paper claims. Current
> authority: `README_CAMPAIGN.md` and `docs/audit_r5_results.md`.

Date: 2026-07-25
Audited baseline: `main@6f61fa0`
R4 implementation checkpoint: `f805fbe`

## Verdict

The code now supports a rigorous, reproducible **simulation-based comparative
study of continuous scour support-stiffness-loss estimation and most-damaged-pier
localisation**. The central implementation is suitable for a paper when its
claims are kept inside that scope.

It is not scientifically honest to call the whole research “reviewer-proof” yet.
The implementation audit is largely closed, but the confirmatory evidence does
not exist until the campaign runs. Two pre-publication empirical gates also
remain: the contact time-step closure on the two target passages and an explicit
exploratory label for the one-state `nuisance_only` outer-test probe (or a future
regeneration with more independent nuisance states).

## What R4 closed

### Numerical and physical contracts

- PAA is the actual fractional-window Keogh-style window mean, with short-signal,
  dtype, ownership, non-divisible-window and chunk-boundary tests.
- Multi-head regression loss has exact `(batch, heads)` shape guards,
  range-normalised scour/bearing weights and device/AMP checks.
- Load-time noise is generated from protocol-pinned PCG64 streams keyed by
  **global DOF**, so the same sensor receives the same noise when evaluated
  alone, in a pair, in reverse order or in the full array.
- Overlapping ballast patches use one coherent governing patch (largest
  `|log eta_k|` supplies both stiffness and damping); MATLAB and Python mirrors
  pass an executable overlap parity test.
- The full eight-sensor array is a non-selectable control. The four-architecture
  × 28-pair matrix and exact seed/pair membership are hard-gated before a winner
  can be published.
- The bilateral-contact limitation remains explicit. Wheel flats remain
  disabled; polygonisation stays. A reproducible MATLAB 1/0.5/0.25-ms closure
  harness now reuses saved passage descriptors and reports signal/contact
  convergence without changing the 24-kN/0.2% production gate.

### Selection and inference

- All train/validation/test partitions are grouped by generated damage state;
  passages from one state cannot cross a partition.
- The outer test is immutable and is not opened until the winner and comparator
  set are written to `frozen_selection.json`, comparator artifacts pass
  preflight, and finalist CV finishes.
- Search rungs run 5-fold × 2-repeat state-grouped CV only on a pre-registered
  finalist set: top-five joint candidates, each architecture at its own
  inner-validation optimum, same-placement architecture contrasts, the carried
  and designed pairs, and full-array controls.
- Fold scalers are fitted on fold-training samples only. Hyperparameters and
  training duration are frozen from the canonical inner-validation study;
  repeated-CV validation does not tune a checkpoint or reselect the canonical
  winner.
- Paper-facing uncertainty resamples independent states first and computes the
  finite-seed median inside each bootstrap replicate. Winner–comparator
  contrasts use aligned state/repeat/seed tensors. No bootstrap fraction is
  presented as a p-value or posterior probability.
- Disentanglement/false-positive probes aggregate passages within state first,
  report state and passage counts separately, use positive-part prediction
  amplitude so negative predictions cannot improve a false-positive score, and
  suppress intervals when fewer than two independent states exist.

### Provenance and reproducibility

- Every generated source file is SHA-256 verified before study creation and on
  cache reuse. Dataset sidecars and cache artifacts are hash-linked.
- Optuna studies store the canonical full protocol record. A pre-existing
  unstamped study, protocol mismatch, manually extended useful-trial budget or
  incomplete seed matrix is rejected.
- Champion weights and the scaler are SHA-256 linked to deployment metadata,
  the best Optuna trial and its protocol record. Export is atomic/idempotent
  even after per-trial weights are cleaned up; a standalone deployment verifier
  does not need the Optuna database.
- Python 3.13.3, direct package versions, Torch/CUDA and cuBLAS configuration are
  hash-carried and checked before a study starts. PyTorch nondeterministic
  operations now hard-fail rather than warn. MATLAB generation hard-requires
  R2025b and newly generated manifests record it.
- Bundle contents come from tracked `bundle_source_files.txt`, not an untracked
  historical ZIP. The builder refuses dirty source/manifest inputs and records
  both source commit and transport SHA-256.

### Intentional deployment-package compatibility break

- R4 deliberately makes every pre-R4 standalone digital-twin package
  unloadable. `digital_twin.assets.DigitalAsset` now calls the fail-closed
  standalone-package verifier, which requires the champion-weights and scaler
  SHA-256 digests, scaler filename, protocol hash and canonical protocol
  descriptor. Older packages do not carry that provenance and are rejected
  rather than silently trusted.
- This is an intentional breaking change, not a migration defect: all existing
  packages also predate the corrected bridge-mass campaign and are invalid for
  scientific reuse. They must be regenerated from a verified post-R4 champion;
  missing fields must never be backfilled by assertion alone.
- The prototype-twin figures that use a synthetic observation model do not
  consume these standalone packages. A broader audit of the digital-twin layer
  remains separate from the present MATLAB-generation and ablation-methodology
  audit.

## Executed evidence on this machine

Environment observed: Python 3.13.3; Torch 2.7.0+cu128; CUDA 12.8; MATLAB
R2025b Update 5.

Passed Python checks:

- `check_paa.py`
- `check_weighted_head_mse.py`
- `check_sensor_noise_pairing.py`
- `check_b54_overlap_parity.py`
- `check_protocol_hash.py`
- `check_cache_provenance.py`
- `check_campaign_controls.py`
- `check_statistical_inference.py`
- `check_artifact_provenance.py`
- `check_environment_lock.py`
- `check_loader_provenance.py`
- `check_split_grouping.py`
- `check_familytable_roundtrip.py` (real MATLAB cell/logical encoding)
- `check_raw_parity.py` (worst MATLAB–Python interpolation/crop difference
  `1.980e-13`, below the `1e-12` gate)
- `_stage3_smoke.py` (legacy/empty-descriptor healthy paths bit-identical;
  finite track-damage + wheel-polygonisation response on all eight DOFs)

Passed MATLAB checks:

- `smoke_audit`
- `smoke_contact_closure`
- `smoke_b54_overlap_parity`
- `smoke_stage3` (healthy byte parity, damaged path, global track placement and
  compressive contact)
- `smoke_geometry` (all six fixed/FRA-profile, L60/L99.6 and 70/80/90-km/h
  configurations; live bridge length, full-deck crop, finite full-track profile,
  and compressive contact all passed)
- `smoke_familytable` (MATLAB save/load, resume equality and fingerprint JSON
  contracts)

MATLAB R2025b Code Analyzer reported zero messages for `B54_ModelMatrices.m`,
`B54_TrackVectors.m`, `B66_ContactForce.m`, `contact_closure_study.m` and
`smoke_contact_closure.m`. `A00_Run.m` retains 41 non-fatal analyzer advisories:
unreachable branches created by campaign constants, `parfor` broadcast/performance
advice, obsolete suppressions, one `datestr` deprecation and formatting/
preallocation suggestions.

## Remaining gates

### Before dispatching an ablation

1. Have the R4 commit independently audited.
2. Do **not** use the `bundle_*.zip` files present in the 2026-07-25 audit
   snapshot; they predate R4. After a reviewed rebuild, dispatch only the
   complete ten-ZIP set whose hashes and reviewed `source_commit` all match the
   newly published `bundle_sha256.txt`.
3. On every dispatch machine, run the documented MATLAB/Python preflight suite
   from the reviewed bundle before starting generation or ablation.
4. Benchmark one representative Optuna study and one finalist-CV refit. The
   complete ladder is approximately 1,344–1,359 studies
   (134,400–135,900 useful trials), plus at most about 2,040 fixed-parameter CV
   refits after deduplication.
5. Build bundles only from a clean reviewed commit and verify the persisted
   bundle SHA-256 manifest.

### Before confirmatory contact-validity claims

Run `contact_closure_study` on regenerated:

- `s23_all4`, state 24, worst saved passage;
- `s15_track`, state 244, worst saved passage.

Use `DtMs=[1,0.5,0.25]` and report signal NRMSE/max error/correlation plus
contact peak/fraction against 0/12/24-kN diagnostic gates. The smoke validates
the harness, not the numerical closure of those two production passages.

### Before manuscript submission

- The isolated `nuisance_only` false-positive probe has six total states and
  only one outer-test state. Treat it as exploratory with no CI, as the code
  does. If the paper needs a confirmatory nuisance-only false-positive claim,
  regenerate that family using the MCSE report; the design floor is 50 total
  states for ten expected evaluation states at 20%.
- The 5% all-channel multiplicative noise is a controlled stress model, not a
  rail-qualified accelerometer datasheet model. A separate additive
  noise-density × bandwidth arm remains desirable for external realism.
- Hanging sleepers are linearised support removal, not a unilateral gap model;
  geometric settlement/profile dip is outside the present EOV set. State these
  boundaries and use “all modelled vertical-pathway EOVs,” never “all damage.”
- Legacy completed R8 datasets that lack `case_info.matlab_release` have an
  explicit `UNRECORDED_R8_PRE_AUDIT_R4` attestation. Do not describe their
  MATLAB release as verified. Newly generated datasets record R2025b.
- The environment file locks direct dependencies/runtime versions but is not a
  hash-complete transitive wheel archive. Archive `pip freeze`, GPU/driver
  details and the final source/bundle hashes with the paper artifacts.
- `docs/paper1_methodology.md` and older planning documents contain historical
  30-seed, three-architecture and obsolete campaign language. Rewrite the paper
  methodology from the executed protocol descriptor and result artifacts; do
  not cite those planning files as the final protocol.
- No development-locked binary alarm threshold is implemented. Therefore claim
  continuous severity estimation/localisation, not sensitivity, specificity,
  probability of detection or minimum detectable damage. A future detection
  paper needs a threshold fixed on development data before its sealed test.
- This remains a numerical 2-D TTBI study. Field-transfer, model discrepancy,
  lateral/torsional effects, nonlinear soil–foundation interaction and real
  sensor/vehicle domain shift remain external-validity limitations.

## Publication-safe claim

> Under the specified 2-D TTBI simulation distribution, fixed two-sensor budget,
> four candidate architectures, 28 sensor pairs, three optimiser seeds and
> pre-registered training budget, we compare continuous per-pier scour
> support-stiffness-loss estimation and most-damaged-pier localisation using
> state-grouped development/outer partitions, finalist split-stability analysis
> and state-first uncertainty.

For s16/s23, “best” must additionally be labelled **exploratory deployment
selection** and “best among the evaluated candidates at this budget and
geometry,” not a global optimum or best sensor-count claim.
