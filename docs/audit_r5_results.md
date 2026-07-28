# Audit R11 results (legacy filename)

**Status: DISPATCH BLOCKED.**

**Tested source commit:** PENDING

The filename `audit_r5_results.md` is retained because the bundle authorization
gate already binds to it. The contents are the R11 audit record. This file does
**not** yet authorize generation, ablation, bundle publication, or dispatch.

## Locked R11 boundary

The contract under review is `audit-2026-07-27-r11` /
`generation-rules-v6` / `gs8a20260727r11`. All ten production rungs are locked
to the exact MATLAB **25.2.0.3177638 (R2025b) Update 5** numerical-stack
descriptor and the registered Python/Torch/CUDA environment. MATLAB and Python
authenticate their source, numerical assets, environments, semantic state
inventories, protocol descriptors, caches, studies, weights, scalers, capacity
receipts, HPO manifests, and champion manifests. Qualification output remains
segregated and is rejected as campaign data.

The champion manifest carries `frozen_selection_sha256`. The anchor prints its
canonical SHA-256, and every follower must provide that exact value through
`TTBI_BLOCK_REFERENCE_SHA256`; a missing or mismatched operator pin is fatal.
The final audit must confirm this hardening after its implementation checks
complete.

The legacy-named `benchmark_r5_compute.py` is now a contract-guarded genuine R11
runner. It derives and authenticates a
475-state × 50-passage × 8-channel × 512-segment workload with five heads,
runs one 100-trial anchor HPO study through the production objective, and runs
exactly one shared finalist-CV refit. The heavy run has **not** yet executed.
Its final run must be bound to clean commit A and must contain zero Optuna
`FAIL` states, zero OOM events, no Optuna-trial retries/replacements, and
exactly 100 registered anchor trials. The one durably accepted CV refit must
also be a clean first attempt: `attempt_count=1`,
`prior_unaccepted_attempt_count=0`, `timing_complete=true`, and
`memory_complete=true`. The benchmark's explicit `--recover-stale` path can
record an interrupted refit and a later accepted attempt, but that evidence is
non-qualifying for authorization/timing and requires a fresh clean benchmark
run. Only non-scientific
throughput, provenance, hardware, terminal-state, and immutable-artifact
evidence may be reported; objective and prediction values must not be
published.

## Gates still required

1. Freeze the converged source as one clean 40-character lowercase Git commit
   A. Rerun any source-sensitive checks if the tree changes before A.
2. Complete the fast MATLAB/Python, environment/source, artifact/provenance,
   statistical, capacity, mutation, and production-path preflight suites
   against A, including the new block-reference hash pin.
3. Run the genuine R11 heavy benchmark above against A and review its timing,
   provenance, hardware, immutable artifacts, and exact terminal accounting.
4. On **every** intended MATLAB generation host, freshly generate and execute
   all three commit-A qualification stages: `s0_scour`, `s16_all`, and
   `s23_all4`. Each host must use a stable unique
   `TTBI_QUALIFICATION_HOST_ID`. Corresponding stage outputs from independent
   hosts must have distinct authenticated host IDs and accepted
   `matlab-environment-qualification-receipt-v4` pairwise receipts. A
   numerically equivalent comparator exit alone does not authorize anything.
5. Obtain the independent audit of the converged source and evidence.
6. Create report-only commit B. Relative to A, B may change **only**
   `docs/audit_r5_results.md`, retaining the exact R11 legacy heading above,
   replacing the status with the exact
   `**Status: DISPATCH AUTHORIZED.**` verdict and the pending source line with
   the tested A SHA, and recording the reviewed evidence. The report must state
   that the benchmark ran on A, not B.

The completed 35-state × 3-passage `s0_scour` laptop micro is
pre-convergence, one-host integration/timing evidence only. It is not one of
the final commit-A three-stage host qualifications and does not satisfy the
cross-host gate.

Until every gate passes, do not rebuild or dispatch the ten bundles. The
builder must reject this status. The stale pre-R11 ZIP set is quarantined under
`stale_pre_r11_bundles/` and must not be used.
