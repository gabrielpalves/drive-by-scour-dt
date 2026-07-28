# R11 audit handoff

**Status: DISPATCH BLOCKED (2026-07-27).**

This is a reviewer handoff, not an authorization record. No scientific R11
generation, ablation, qualifying commit-A benchmark, or bundle dispatch has
started. A historical non-scientific timing benchmark at commit `a0793a1`
remains useful only as workload evidence; it cannot qualify the converged R11
source. The authoritative authorization file remains the legacy-named
`docs/audit_r5_results.md`.

## Frozen scientific contract

- Ten rungs in two independent physical blocks: L60 (`s0` anchor, 450 states)
  and L99.6 (`s21` anchor, 475 states), with 50 passages per semantic state.
- Fixed family inventories, semantic `StateUID`, collision-gated SHA-derived
  `StateSeedID`, UID-named random streams, common random numbers, and one exact
  state-grouped 60/20/20 split within each geometry.
- Free HPO only on the full eight-DOF anchor controls:
  4 architectures × 3 seeds × 100 trials independently at `s0` and `s21`.
  Every response-subset or non-anchor-search candidate and every follower uses
  one authenticated singleton trial; `FAIL` and OOM are fatal and no
  Optuna-trial retry/replacement is allowed.
- Each rung is retrained. The primary paired inference is the exact seven-edge
  L60 joint-family analysis with 100,000 state-first bootstrap replicates and
  Bonferroni familywise intervals. L99.6 is blockwise; `s0` to `s21` is
  descriptive only.
- The full-array benchmark runner is implemented and contract-guarded despite
  its legacy filename, `benchmark_r5_compute.py`. Its authenticated workload is
  475 states × 50 passages × 8 channels × 512 segments with five heads, one
  100-trial anchor HPO study, and exactly one shared finalist-CV refit. The
  heavy execution remains pending and must be bound to clean commit A. A
  qualifying run requires zero Optuna `FAIL`/OOM and no Optuna-trial retry or
  replacement. Its accepted shared refit must be a clean first attempt:
  `attempt_count=1`, `prior_unaccepted_attempt_count=0`,
  `timing_complete=true`, and `memory_complete=true`. A refit recovered through
  `--recover-stale` is preserved as interrupted evidence but is nonqualifying
  for authorization and timing.

## Implemented deltas

- Exact MATLAB R2025b Update 5 numerical-stack and source-byte firewall;
  fail-closed working-directory, resume, qualification, and loader boundaries.
- Durable source/runtime/policy-bound execution and CUDA-capacity receipts,
  block-local HPO manifests and champions, strict terminal-state accounting,
  and fatal OOM/`FAIL`.
- The champion manifest carries `frozen_selection_sha256`; the anchor prints
  its canonical SHA-256, and every follower must pin that exact value through
  `TTBI_BLOCK_REFERENCE_SHA256`. The behavioral gate accepts the exact pin and
  rejects a different registered architecture, another valid pair, a rehashed
  foreign capacity receipt, and extra manifest fields.
- The cross-rung analyzer independently requires that external block-reference
  trust root. Every study and downstream artifact carries the exact
  `campaign_run_tag`, `execution_receipt_sha256`, and
  `block_reference_manifest_sha256` lineage (explicit null only where the
  anchor contract requires it). Champion, frozen-selection, deployment,
  finalist-CV, and protocol evidence is published create-once or through the
  one registered monotonic null-to-digest transition.
- Full semantic family-table serialization and independent MATLAB-to-Python
  round-trip verification.
- State-grouped selection/test firewall, frozen finalist CV, artifact and cache
  provenance, weighted five-head loss, and registered cross-rung analyzer.
- Genuine R11 benchmark derivation/stamping and lightweight contract guards.
- Stale root bundle ZIPs are staged for deletion and recoverably quarantined
  under `stale_pre_r11_bundles/`. No replacement bundle has been built.
  Local untracked raw/cache content under `results/Stage1/` and
  `results/Stage2/` is ignored and does not enter delta A; the few historical
  files already tracked there remain unchanged.

## Pre-commit evidence already obtained

These are working-tree convergence checks. They must not be represented as
commit-A evidence if source changes invalidate them.

- MATLAB `smoke_stage3`: **PASS**, 421 s.
- MATLAB `smoke_geometry`: **PASS**, 1393.1 s.
- MATLAB/Python raw parity: **PASS**, reported discrepancy `1.437e-13`.
- Generation contract mutation suite: **PASS**, all 107 injected mutations
  rejected, including a same-value literal substitution and assignment- or
  definition-based runtime rebinds of the canonical protocol schema tag.
- Adversarial release/host comparison suite: **PASS**, 111 checks in 488.125 s.
- MATLAB: `smoke_audit`, `smoke_contact_closure`,
  `smoke_b54_overlap_parity`, `smoke_crn_state_design`,
  `smoke_r11_provenance_serialization`, and `smoke_familytable`: **PASS**.
- Python family-table round trip: **PASS**, 12/12.
- Python HPO, capacity, execution-blocking, source,
  environment-lock, loader, artifact, cache, protocol-hash, split-grouping,
  PAA, sensor-noise pairing, weighted-head loss, statistical-inference, and
  cross-rung-inference checks: **PASS**.
- Current focused counts: execution blocking **65/65**, cross-rung inference
  **54/54**, and benchmark contract **33/33**.
- Campaign-control functional subchecks: **PASS**. The campaign-control checker
  as a whole is **not PASS pre-commit**: its separate “regular tracked Git blob”
  assertion remains intentionally red until the new R11 sources become blobs
  in commit A.
- R4 mutation guards: **PASS**, 23/23 mutations rejected across artifact,
  statistical, and environment groups, with isolated-tree byte restoration and
  every real-tree mutation target byte-unchanged.
- Training-policy mutation guards now register 17 mutations. Their final
  commit-bound rerun is deliberately pending until commit A, because one
  baseline is the tracked-blob campaign gate.
- Benchmark contract checker and Python compilation checks: **PASS**.

The earlier 35-state × 3-passage `s0_scour` laptop micro (27 min 32 s)
predates source convergence and covers one host and one stage. It is useful
integration/timing evidence only and is not a final qualification run.

## Remaining dispatch blockers, in order

1. Finish the integrated source review and quiescent-tree regression, then
   create clean source commit A. Against A, rerun the full mutation harnesses
   and every source-sensitive/commit-bound preflight required by the
   authorization record, including the tracked-blob campaign gate.
2. Run the genuine R11 heavy benchmark on A: exactly 100 registered anchor
   trials, exactly one durably accepted shared CV refit, zero `FAIL`, zero OOM,
   and zero Optuna-trial retry/replacement. The refit must be a clean first
   attempt with `attempt_count=1`, `prior_unaccepted_attempt_count=0`,
   `timing_complete=true`, and `memory_complete=true`; a recovered interrupted
   refit cannot authorize dispatch or supply qualifying timing. Record timing
   and provenance, never objective or prediction values.
3. On every intended MATLAB generation host, derive and execute fresh from A
   the three segregated qualification micros for `s0_scour`, `s16_all`, and
   `s23_all4`. Use stable distinct `TTBI_QUALIFICATION_HOST_ID` values and
   retain every authenticated host receipt. This is qualification, not
   scientific campaign generation.
4. Compare corresponding stage outputs across the required independent hosts
   and explicitly accept each required
   `matlab-environment-qualification-receipt-v4`; comparator success alone is
   insufficient.
5. Give commit A, benchmark evidence, host receipts, comparison receipts, and
   this handoff to Claude for an independent audit.
6. If and only if every gate passes, make report-only commit B by changing only
   `docs/audit_r5_results.md` relative to A.
7. Build and hash one fresh ten-bundle set from B. Do not reuse or overlay any
   pre-R11 bundle or result directory.

## Claim and manuscript boundaries

- Scour is modeled vertical support-stiffness loss, not scour-hole depth.
  Bearing fixity is a nominal design variable. Crack is uniform
  damaged-element `EI` loss, not Sinha. Hanging sleepers are a linear
  support-removal approximation. Wheel flats are excluded.
- `all_mult` is a symmetric multiplicative response-noise stress, not a
  datasheet sensor model. The study ranks modeled response channels/DOFs, not
  physical sensor count, mount, package, technology, or globally optimal
  placement.
- The registered output supports continuous estimation and conditional
  most-damaged-pier localisation. It does not support detection, POD,
  sensitivity/specificity, calibrated damage probability, or minimum
  detectable severity without a development-locked binary threshold.
- Rung-specific retraining estimates achievable rung-specific performance; it
  is not zero-shot transfer or out-of-distribution robustness.
- Bootstrap intervals are finite-design state-resampling uncertainty
  conditional on the fixed anchor/LHS design, not field-population coverage.
- The paper remains blocked until every citation placeholder is replaced by a
  verified primary source, the exact Cantero/TTBI and Fernandes implementation
  versions are identified, and the physical-parameter and terminology tables
  are reconciled with those sources. These are manuscript blockers even after
  software dispatch authorization.
