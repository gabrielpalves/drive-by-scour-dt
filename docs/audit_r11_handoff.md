# R11 scientific-audit handoff — 2026-08-01

> **STATUS: NOT READY — DO NOT CREATE COMMIT A.**
>
> No unresolved P0 was found in the current technical implementation. The
> working tree nevertheless has open P1 scientific-claim, semantic-source,
> immutable-registration, and tracked-snapshot blockers. The green checks below
> establish convergence of this working tree only; they are not evidence bound
> to a clean commit and do not authorize generation or dispatch.

## 1. Audited snapshot

- Current HEAD: `865728f801c83a642b06a223f2a22b33f2b429b7`.
- `git status --short --untracked-files=all`: 334 entries (73 modified and 261
  untracked) before this handoff/review update.
- Required bundle plus manuscript-source union: 443 files. Of these, 182 are
  regular tracked blobs in HEAD and 261 are absent from HEAD.
- `git ls-files paper1` returns no files; all 14 paper-source files are
  currently untracked.
- The Codex review did not edit any `.tex` file.

The current authenticated-source snapshot, reasserted after capture, is:

- reviewed manifest: 429 entries;
- generator root: 297 files,
  `c34ba6d6ab166b79b2b9e6e6e45fd5ef29d952f5aa43b2a755ba8e2dd9820b3f`;
- Python-runtime root: 124 files,
  `a5d3815db9fdc85dc8ca77626938de07ebcb2a6d362d1f7b79ebb0b625a2c35c`.

These hashes identify the present dirty working tree. They are not a substitute
for a clean commit identifier.

## 2. Technical P1 closures independently accepted

### 2.1 Endpoint bytes, identity, and publication

The release comparator now binds parsing to the authenticated member bytes and
reasserts every endpoint digest and the content root before returning a verdict.
The mutation probes pin the intended guard diagnostics rather than merely
requiring some later defence-in-depth check to fail.

Publication/resume paths reassert source and destination identities around
material mutations, reject stale credentials and unexpected inventory, and use
exclusive publication. Aggregate qualification reruns every retained pair
comparison; dispatch has permanent behavioural mutations for edge
revalidation. The former P2-a and P2-b findings are therefore closed.

### 2.2 MATLAB execution closure and OneDrive/NTFS identity

Static and explicitly reviewed dynamic MATLAB dependencies are included in the
execution closure. Worker processes attest the source they actually execute;
shadowing, missing modules, unexpected modules, aliases, and links fail closed.

On this OneDrive tree, Java NIO returns a null `basic:fileKey`. The new Windows
fallback invokes the absolute System32 `fsutil.exe` via `ProcessBuilder` (no
shell), applies a 10 s timeout, accepts exactly one non-sentinel 128-bit file
identifier, and binds it to a volume identity observed before and after. The
generator and contact implementations are separated into small named helpers.
Direct checks agreed on ordinary-file identity and reported a hard-link count
of one. The final generation-worker smoke exercised and rejected the hard-link,
shadow, stale-credential, resume, and publication probes.

### 2.3 Host-claim boundary

The documentation and code headers now limit host evidence to self-attested
diagnostics plus retained-artifact integrity under a trusted-operator model.
They no longer claim that coherent fabrication by an operator is impossible.
This is the defensible boundary for the present academic campaign.

### 2.4 Readability and modularity at P1

`A00_Run.m` is now a 328-line orchestration script with no local functions.
The `+ttbi` package contains 98 `.m` files; each has one function and a purpose
header. The contact implementation contains 112 `contact_*.m` files with the
same one-function/purpose-header property. The Python contact checker is a
296-line entry module with one top-level function and imports domain-specific
modules.

The historical 35-state × 3-passage refactor oracle compared 3,444 leaves,
9,130,944 numeric values (9,124,430 signal values), with only the expected
source-root field differing. That is appropriate evidence for the two roots it
compared. The final current-tree generation-worker smoke supplies additional
current-root execution evidence; neither test validates the scientific priors.

The verifier split correctly authenticates and reasserts the complete verifier
module root rather than only its entry file. Missing-module, shadow, inventory,
and live-drift mutations pass fail-closed checks.

### 2.5 Handoff

The former handoff was stale and is replaced by this document. This closes the
technical handoff P1, but not the scientific and release blockers below.

## 3. P1 blockers before commit A

### P1-S1 — Exact simulator laws exceed their source support

The manuscript still presents exact laws collectively as modelling priors
assembled from field evidence, although the evidence supports only mechanism
plausibility or contextual bounds. The manuscript must classify each of the
following individually as an author-chosen stress-test prior or engineering
proxy, not as an empirically fitted population law:

- ballast occurrence `1.2/100 m`, extent `U(5,20) m`, wet/dry probability 0.5,
  the exact dry/wet ranges, and the threefold wet-density multiplier;
- hanging-sleeper occurrence `3/100 m`, `DU{1,...,5}`, 60% single-sided and the
  3:1 side ratio;
- pad Weibull shape/scale `(1.8, 2.2)`, damping multiplier `[0.8,1.2]`, and
  prevalence `p=0.02`;
- crack Bernoulli activation `p=0.25`;
- the exact OOR triplet and any other exact distribution without an identified
  fitting dataset.

“Pad aging” is not supported because the model has no age or time axis; this is
service-condition variability. Patch extent may be contextualized by surveys
but the exact `U(5,20)` law was not fitted to them. The present campaign is not
an immutable “released campaign specification” until the deposit exists.

Direct source inspection found:

- Williams et al. (2014) is a lateral-load/insulator investigation and does not
  support a 0.5% annual pad-failure incidence;
- Lundqvist and Dahlberg examine a single 1 mm gap and report large adjacent
  force/displacement effects, not a universal count distribution;
- the RAILCON paper does not fit `DU{1,...,5}` or establish five universal
  critical sleepers;
- Shi et al. sweep 0, 1, 3, and 5 unsupported sleepers; three is often the worst
  case, and five is not a universal critical limit.

### P1-S2 — Statistical “conservative” language is unproved

The seven-edge tail-adjusted result is a wider descriptive sensitivity envelope.
It is not a confidence interval, a familywise-error-controlled procedure, or a
formally conservative inferential guarantee. Remaining generic “conservative”
claims in the introduction, data-processing, results, conclusion, and
limitations must be narrowed accordingly.

### P1-S3 — Semantic citation/source graph is open

The syntactic BibTeX graph is closed: 61 used keys, 61 unique definitions, no
missing keys, unused entries, or duplicates. It is not semantically closed.

Essential local sources not yet connected to the corresponding manuscript
claims include Esmaeili (2017), Wangtawesap (2023), Kitahara (2024),
Lazarevic/RAILCON (2016), Siahkouhi (2025), Oregui (2016), Sainz-Aja (2020),
and Woo and Park (2017). FRA ORD-22/01, FRA RR22-32, RIVAS (2013), and Shi
(2024) are additionally required wherever their associated claims are retained.

Conversely, the cited keys `garg1984dynamics`, `sadeghi2018gpr`,
`selig1994track`, `chrismer2018fouling`, `husoy2024defects`, and
`musgrave2024ballast` lack an identifiable local source artifact. Exact-source
placeholders remain for the Cantero/Zhai implementation provenance and the
Fernandes PAA method; the Zhai BibTeX note still says `VERIFY`. TTB-2D
provenance exists in repository notices but is not propagated to the manuscript.

### P1-S4 — No immutable protocol registration

There is no real, dated OSF/Zenodo protocol locator. Methodology files being
source-locked inside a mutable working tree and a promised future data DOI are
not protocol registration. Finalize the protocol only after the text and source
graph converge, deposit it immutably, insert the real locator/date, and re-audit
that exact pre-commit tree.

### P1-R1 — Required files are not a clean tracked snapshot

Commit A cannot be created from the present 73-modified/261-untracked tree.
All 443 required bundle and manuscript files must be regular tracked blobs, and
the clean-tree/commit-bound gates must be rerun after the scientific corrections
and immutable locator are present.

## 4. Downstream qualification blocker (after A, not a pre-A excuse)

The frozen lock requires MATLAB `25.2.0.3177638` (R2025b Update 5). The installed
R2025b is `25.2.0.3312555` (Update 6, 2026-06-30). Do not retarget the lock to
make this host pass. On the final tree, `smoke_contact_closure` traversed the
closure and stopped at the intended exact-stack gate:

`Closure qualification requires the exact locked MATLAB R2025b Update 5 numerical stack.`

This mismatch does not prevent finishing the pre-A manuscript and snapshot
work. It does prevent this host from supplying the exact post-A MATLAB
qualification receipt.

The following also remain downstream of a clean commit A:

- a fresh, genuine 100-trial benchmark started from zero;
- real complete-graph host/endpoint qualification receipts and retained pair
  comparisons;
- contact closure over the authorized real datasets;
- final dispatch authorization and the scientific campaign itself.

Synthetic/micro gates prove enforcement logic, not completion of these real
campaign steps.

## 5. Final current-tree evidence

The following source-sensitive tests were rerun on the converged tree and count
only when they ran without overlap from another live-source mutation harness:

| Check | Result |
|---|---|
| `smoke_generation_worker` | PASS, 1,092.3 s; real one-state/one-passage ProcessPool execution, 8,806 integration steps, worker attestation, resume and adversarial boundary probes |
| `check_generation_release_comparison.py` | PASS, 1,824.073 s, 158 explicit PASS cases |
| `check_qualification_receipt_inventory.py` | PASS, 1,184.822 s, 75 explicit PASS cases; H=2 complete graph with 3 retained pair receipts |
| `check_dispatch_authorization.py` | PASS, 4.780 s, 63 PASS and one symlink-privilege N/A |
| `check_artifact_provenance.py` | PASS, 59.230 s, 52 PASS |
| `check_contact_closure_gate.py` | PASS, 273.5 s; complete 327-case synthetic gate |
| `check_generation_refactor_equivalence.py` | PASS, 41.7 s |
| `check_source_provenance.py` | PASS, 145.623 s |
| `check_import_path_guard.py` | PASS, 3.846 s |
| `check_environment_lock.py` | PASS, 234.862 s; validates the lock policy, not that Update 6 equals Update 5 |
| `check_loader_provenance.py` | PASS, 189.3 s |
| `check_protocol_hash.py` | PASS, 229.9 s |
| `check_generation_contract.py` | PASS, 175 generation plus 13 damage mutations |
| `check_profile_pad_contract.py` | PASS, 21 mutations |

PAA, sensor-noise, split-grouping, statistical-inference, track-prior, weighted
head-MSE, benchmark-contract, cache-provenance, capacity, cross-rung,
execution-blocking, family-table, and hyperparameter checks also passed during
this convergence. Python compilation passed for all 122 manifested `.py` files,
and `git diff --check` passed apart from line-ending warnings.

Expected non-green states remain:

- `check_campaign_controls.py`: every check passes except the required regular
  tracked-blob gate (443 required, 182 tracked, 261 missing);
- `check_training_policy_mutation_guards.py`: correctly rejects the dirty,
  non-commit-bound tree;
- `check_raw_parity.py`: N/A until real data and MATLAB output exist.

The full R4 29/29 mutation suite passed before the final diagnostic/doc-only
patches, but was not rerun afterward. It must be rerun from clean commit A.

### Execution rule for mutation harnesses

Several checkers deliberately mutate live source and restore it. Run those
harnesses one at a time with a quiescent tree. During this audit, parallel
execution caused one checker to observe another checker's intended temporary
mutation and fail closed. Such overlapped results were discarded and rerun
serially. Heavy read-only suites may run in parallel only when no live-source
mutation harness is active.

## 6. P2 readability queue

The principal modularity objection is no longer P1. Literal one-function-per-
file cleanup remains for production files with local helpers:

- `save_progress.m` (7 functions);
- `B54_ModelMatrices.m` (4);
- `B65_DynamicCalcCoupledFaster.m`, `B65_DynamicCalcCoupled.m`,
  `A04_Options.m`, `B19_GenerateProfile.m`, `B00_Calculations.m`, and
  `B09_BeamFrq.m` (2 each).

Several smokes also contain local helpers. Larger single-purpose functions such
as `campaign_setup.m` should be split only where responsibilities are genuinely
independent. Preserve simple call sites, one clear purpose per file, and comments
that explain scientific decisions rather than restating syntax. Any additional
source refactor should happen before commit A, followed by the same equivalence
and source-sensitive checks.

## 7. Required sequence

Before A:

1. Correct the `.tex` claim boundaries and statistical wording.
2. Close the semantic source graph and remove all source placeholders.
3. Finalize and immutably deposit the protocol; insert its real locator/date.
4. Re-audit the exact text, citations, source inventory, and protocol hash.
5. Make all 443 required files regular tracked blobs and obtain a clean tree.

Commit A:

6. Create one commit containing the converged code, manifest, documentation,
   immutable locator, bibliography, and manuscript sources.
7. Run every commit-bound and source-sensitive gate serially; reject any drift.

After A:

8. Qualify on the exact locked Update 5 stack.
9. Run the fresh 100-trial benchmark and retain its complete evidence.
10. Build and revalidate the real complete host graph.
11. Run real-data contact closure, then final dispatch authorization.
12. Only then start the scientific campaign.

Until steps 1–5 are complete: **NOT READY; do not create commit A.**
