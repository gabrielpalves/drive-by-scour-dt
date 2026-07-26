# Audit R5 results — IN PROGRESS

**Status: DISPATCH BLOCKED.**

**Tested source commit:** PENDING

This file is reserved for the final, commit-bound R5 audit record. It is
included in the reviewed bundle manifest so every dispatched bundle will carry
the evidence that authorizes that exact source revision. It does **not** yet
authorize generation or ablation.

The final record must be completed only after all of the following have
finished against the converged, clean commit:

- the fast Python and MATLAB campaign preflights;
- the R4 and training-policy mutation campaigns;
- the production-path compute benchmark: 100 useful Optuna trials and one
  shared finalist-fold refit;
- review of the benchmark's timing, provenance, hardware and immutable-fixture
  records; and
- an independent review of the converged source.

The final version must replace the pending line with the same bold label
followed by exactly one 40-character lowercase Git SHA enclosed in backticks.
That SHA identifies clean commit A on which all code checks and the heavy
benchmark ran. Adding the evidence and authorization creates dispatch commit B;
B must differ from A **only** in this report. The builder verifies that A is an
ancestor of B and that `git diff --name-only A..B` is exactly
`docs/audit_r5_results.md`. Thus, the report must not claim that the benchmark
ran on B, and no runtime change may hide in the evidence commit.

The final version must also record the exact commands and outcomes, mutation
totals, benchmark wall-clock evidence and any remaining scientific limitations.
It must not report benchmark objective or prediction values: the legacy cache
is a non-scientific timing fixture only.

Until this status is replaced by a completed audit verdict, do not rebuild or
dispatch the ten campaign bundles. The builder rejects this status before
writing any ZIP. Existing `bundle_*.zip` files are stale.
