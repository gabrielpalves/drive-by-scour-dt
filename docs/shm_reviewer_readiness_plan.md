# SHM reviewer-readiness plan

**Status:** Paper-1 source decisions frozen; external qualification and release gates remain

**Scope:** scientific validity, interpretability, and reproducibility for an
SHM/railway-bridge audience; no external preregistration deposit is required.

The earlier audit could correctly say that P1-R1 was the only remaining item
under its narrow pre-A source-control scope. The expanded goal is stronger:
the scientific changes below should be completed before establishing a new
clean commit A and recomputing source hashes.

## Paper-1 disposition — 2026-08-09 (controls the historical checklist below)

The checklist below predates the settled four-block campaign and is retained as
an audit trail. The following disposition resolves its apparent open-item
conflicts; an unchecked historical item does not override this table.

| Topic | Paper-1 disposition |
|---|---|
| Weighted track sampler and campaign-specific track priors | **Deferred mechanism hardening, not a Paper-1 generation blocker.** Track EOV/damage is disabled by authenticated configuration in all four production blocks. The existing fail-closed descriptor and finite-attempt guards remain active; rare-failure and prior-boundary tests stay in the later track-mechanism queue. |
| Ballast `Kw`/`Cw` topology and on-bridge condensation | **Accepted inherited simplified topology.** No topology arm is added before generation. If a constrained-wheelset channel wins selection, the preregistered contingency is to implement the arm under a new source root or not headline that channel. |
| One-seat/two-rail scaling, 0.545→0.600 m transfer, and rail damping | **Complete.** The inherited hybrid baseline remains production; prospective paired one-seat, two-rail, spacing-consistent, and 0.05%/0.20% damping arms have run. |
| Fixed-healthy Rayleigh closure | **Complete.** Production remains state/grid-recalibrated; the fixed-healthy-coefficient sensitivity has run for scour, bearing fixity, and crack. |
| Finite rail domain | **Complete.** The final 18-case F40/L99 × operating-point × 6/15/30 m matrix passed and selected 6 m under `paper1-rail-domain-clearance-c06-v1`. |
| Manuscript mesh/channel/track/damping corrections | **Complete.** The manuscript now states the geometry-specific meshes, `physical8_v1` idealized constrained-wheelset proxy, inherited hybrid scaling/topology, and damping boundaries. |
| Numerical mesh/time-step packages and upstream/contact comparisons | **Implemented harnesses, external execution/authorization pending.** Exact-host qualification and contact closure block dispatch, not source implementation. Broader upstream reproduction remains a pre-publication validation task and cannot be inferred from the finite-domain result. |
| Training robustness | **Implemented, execution pending.** The campaign registers 160 HPO studies and 1,740 refit jobs before authenticated alias deduplication, including grouped development OOF, channel screening, post-freeze stability, and secondary frozen transfer. |
| Source/release closure | **In progress.** Deliberate path disposition, the final serial suite, clean commit A, CUDA capacity/benchmark receipts, exact-host qualification, contact authorization, and bundle verification remain mandatory. |

The exact pre-commit model-form receipts and their claim boundaries are recorded
in [`evidence/paper1_model_form_freeze_20260809.json`](evidence/paper1_model_form_freeze_20260809.json).

## A. Correctness blockers

- [x] Fix the digital-twin multi-support percent-to-TTBI conversion and test
  0/30/60% (repository-only audit: `check_digital_twin_scour_units.py`; not
  shipped in dispatch ZIPs).
- [x] Make nuisance-only false-scour probes run on every regression rung;
  bearing probes remain conditional on bearing targets. Add an `s12` mutation
  test.
- [x] Make finite-attempt weighted track-location sampling fail closed rather
  than silently retaining the last rejected proposal.
- [ ] Test the production weighted sampler itself, including its rare
  rejection-limit failure path.
- [x] Fail closed on malformed/nonfinite active track and polygon descriptors,
  invalid indices, coordinate-frame/lattice errors, group overflow, duplicates,
  and nonpositive multipliers; never silently truncate, snap, or ignore them.
- [ ] Enforce each campaign-specific prior range at the state-to-solver boundary
  (not only mathematical positivity) without hard-coding a generic TTBI helper
  to one study's priors.
- [x] Remove or quarantine pre-R11 numerical performance comments from current
  production modules.
- [x] Replace the support-snapping production mesh with geometry-specific
  support-aligned bridge densities (L60 bridge/rail 3/2 elements per sleeper
  bay; L99.6 2/2), and reject off-node positive-spring supports.
- [x] Make B54's declared on-bridge ballast inventory mesh-invariant: one
  531.4 kg lump per support-aligned sleeper assigned to the bridge, with an
  assembly-isolation check. Zhai supports that per-seat value and a discrete
  independent ballast mass at each support point. The deck attachment and
  full endpoint lumps are inherited model-form/domain-partition choices, not
  source-supplied bridge rules.
- [ ] Resolve the inherited ballast-topology transfer. Zhai retains independent
  ballast motion and adjacent-mass shear elements \(K_w,C_w\), and its no-shear
  comparison overpredicts measured ballast acceleration by 12%. The present
  model omits \(K_w,C_w\) throughout and, on the bridge, condenses \(M_b\)
  directly onto deck DOFs. Freeze and run a prospective topology sensitivity
  or supply a bridge-condensation derivation; the corrected mass inventory
  alone is not physical validation.
- [ ] Resolve the inherited equivalent-plane parameter scaling. Zhai labels
  Table 1 values per rail seat, while the track-property function doubles the
  rail and sleeper terms but not pad/ballast/sub-ballast terms. Require an
  upstream benchmark or prospective one-seat/two-rail sensitivity; do not
  silently double the remaining parameters.
- [ ] Resolve the 0.545-to-0.600 m support-spacing transfer. The generator
  retains Zhai's \(M_b\), \(K_b\), and \(K_f\) values unchanged at a different
  sleeper spacing, although their source expressions depend on spacing.
  Classify it as a proxy-informed hybrid baseline
  and require an upstream benchmark or prospective consistency sensitivity;
  do not silently recompute values after response results are seen.
- [ ] Retain and report the rail Rayleigh target `0.1%` as inherited
  author-chosen, not as a Zhai property; add it to the damping-sensitivity
  ledger alongside the state-specific/fixed-healthy-Rayleigh comparison.
- [ ] Correct the manuscript's universal 0.3 m deck/rail mesh statement.
  Production is L60 bridge/rail 0.2/0.3 m and L99.6 0.3/0.3 m; the L60
  historical 0.3 m bridge grid moves the internal supports by 0.1 m.
- [ ] Withdraw the manuscript's statement that every per-rail track quantity
  is summed into a two-rail planar model until the per-rail-seat scaling issue
  above is resolved.
- [ ] Correct the manuscript's two `AcelRodaPrimVag` channel descriptions. They
  are rail FE acceleration-field values interpolated under the wheel positions,
  not total wheelset accelerations. Non-`.tex` code and documentation now use
  the correct semantics; `.tex` remains untouched under the current audit
  constraint.

## B. Mechanism-level scientific verification

- [x] Establish one canonical implementation map:
  [`damage_model_reference.md`](damage_model_reference.md).
- [x] Add one-at-a-time matrix-delta tests for ballast, unsupported sleepers,
  pad service condition, and pad failure.
- [x] Add an isolated wheel-polygonization amplitude/order/phase/derivative
  test.
- [ ] For every mechanism, run a deterministic paired healthy/damaged passage
  and record modal, RMS, PSD-band, peak-response, and contact signatures.
  The catalog/extractor exists and one canonical healthy-versus-30%-scour pair
  has passed; the other eight interventions and the 50-passage studies remain.
- [ ] For structural stiffness mechanisms, add a fixed-healthy-Rayleigh
  sensitivity. Production recalibrates bridge and rail damping from each
  state's damaged modes, so the current dynamic counterfactual changes both
  (K) and assembled (C).
- [ ] Add simple source-supported trend checks where the literature supplies a
  defensible direction or magnitude. Do not invent acceptance bands.

## C. Numerical V&V ladder

The exact prospective cases, QoIs, refinement levels, artifacts, and claim
boundaries are specified in
[`numerical_vv_protocol.md`](numerical_vv_protocol.md).

- [x] Analytic element, assembly, boundary-condition, static, modal, damping,
  and isolated B54 structural-block checks using an independently integrated
  Euler--Bernoulli oracle; seven plausible mutations are rejected.
- [ ] Bridge/track mesh refinement for modal frequencies and response QoIs.
  The geometry-specific sequence, support-alignment audit, metric helpers,
  coupled preflight, and nonqualifying micro package exist; per-rail-seat
  scaling, 0.545-to-0.600 m parameter transfer, ballast topology, a frozen
  finite stress-case table, and qualifying verification remain unresolved.
- [ ] Time-step refinement for the same QoIs plus contact metrics.
- [ ] Freeze mesh-level Rayleigh handling before refinement. Production
  recalibrates \(\alpha,\beta\) from mesh-dependent modes; retain all
  coefficients/reference modes and compare recalibrated-per-grid with one
  fixed-coefficient sensitivity.
- [ ] Run finite rail-domain convergence. The current ten extra sleeper bays
  give only 6 m of added rail and no absorbing boundary, whereas Zhai reports
  convergence when a moving wheel remains at least 15 m from a rail end.
  Compare realized 6/15/30 m clearances on the saved bridge window and contact
  histories before treating the current boundary as negligible.
- [ ] Reproduce one upstream TTB-2D or published TTBI benchmark.
- [ ] Separate numerical verification, literature/model validation, and future
  field validation in every report.

## D. Experimental robustness

- [x] Retain and smoke-test the paired dry-ballast reciprocal-softening versus
  retained-stiffening transform while keeping track EOV disabled in every
  Paper-1 block.
- [ ] On `exp/track-train-damage`, register a track-active experiment and an
  authenticated arm-pair evaluator before executing the frozen-model estimand
  in [`dry_ballast_stiffness_sign_sensitivity.md`](dry_ballast_stiffness_sign_sensitivity.md).
- [ ] Add reduced replicated-LHS and passage-count convergence studies.
- [ ] Increase training seeds from three to at least five where feasible, or
  limit claims explicitly if compute prevents it.
- [ ] Add mean/dummy and regularized linear/PAA baselines.
- [ ] Report MAE and RMSE/MSE, per-pier bias, severity-stratified residuals,
  out-of-range predictions, and localization uncertainty.
- [ ] Add sensitivity to the five-percentage-point localization threshold.
- [ ] Add a realistic sensor sensitivity (additive floor, bias/drift,
  bandwidth/quantization) distinct from the current multiplicative stress.
- [ ] Add a clearly secondary frozen-model cross-rung/zero-shot analysis to
  expose dependence on retraining.

## E. Interpretation and release boundary

- [ ] Treat any future track-damage experiment as a bundled track-family
  intervention unless auxiliary one-factor runs support mechanism-specific
  claims.
- [ ] Keep nominal bearing fixity, profile phase, and polygonization described
  as nuisance/design variables rather than physical-damage labels.
- [ ] Label the Fernandes two-by-20 m bridge-property set when it is transferred
  to L60 and L99.6 as an idealized geometry/scale stress, not calibration of
  those longer bridge configurations.
- [ ] Keep `AcelRodaPrimVag` and `Wheel*_Vert` explicitly marked as legacy
  identifiers wherever compatibility requires them; figures and scientific
  prose must use the canonical rail-under-wheel Eulerian-acceleration meaning.
- [ ] Keep the live Python TTBI path nonqualifying until damaged-response parity
  is closed.
- [ ] Inspect suspicious untracked root files before any staging; never stage
  the entire dirty tree blindly.
- [ ] After scientific convergence: recompute source roots, run the complete
  MATLAB/Python qualification suite, create one clean commit A, and record exact
  commands/environment/data hashes. A Git revision and availability statement
  are sufficient for SHM reproducibility; OSF/Zenodo preregistration is not a
  requirement for this study.
