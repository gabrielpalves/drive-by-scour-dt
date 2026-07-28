# Paper 1 — R11 manuscript outline

> **Planning document; no R11 results yet (2026-07-27).**
>
> All pre-R11 numerical results, champions, figures, and datasets are excluded.
> Populate the results section only from authenticated outputs of the complete
> regenerated campaign. The authoritative methods draft is
> `docs/paper1_methodology.md`; the dispatch verdict is
> `docs/audit_r5_results.md`.

## Working title

**Common-random-number ablation of architectures and vehicle response
channels for multi-pier railway-bridge scour monitoring**

Alternative, after results justify it:

**Drive-by estimation of multi-pier support-stiffness loss under competing
bridge, track, and vehicle mechanisms**

Avoid “damage depth,” “all damages,” “N-HiTS architecture,”
“hardware-optimal sensors,” and “zero-shot robustness” in the title.

## One-sentence paper claim

Under a registered 2-D TTBI simulation, fixed semantic state populations,
full-array-calibrated hyperparameters, and a state-grouped test firewall, the
study quantifies how registered architecture/response-channel-subset choices and
seven modeled L60 mechanism additions change achievable multi-pier
support-stiffness-loss estimation after rung-specific retraining.

This is the intended scope, not a result. The abstract must replace “quantifies”
with the observed effect sizes only after the complete campaign.

## Contributions to test

1. **A fixed-population CRN experiment.** L60 uses 450 semantic states at every
   rung and L99.6 uses 475, with identical latent designs, UID-stable splits,
   and named random substreams within each geometry.
2. **A compute-feasible, anchor-only HPO policy.** Free 100-trial HPO is
   restricted to the full eight-DOF input at independent L60/s0 and L99/s21
   anchors; every other candidate and follower uses one exact frozen singleton
   trial. Architecture comparisons remain conditional on separate finite
   equal-budget searches and the registered seed set.
3. **A four-arm architecture experiment with an actual pooling control.** The
   plain CNN/GAP arm tests whether the implemented multi-rate pooling adds value;
   the module is described as N-HiTS-inspired, not as full N-HiTS.
4. **A two-channel response-subset study with a full-array control.** Conclusions
   are conditional on the registered full-array-calibrated policy and the
   symmetric relative-noise stress. The code selects channels/DOFs, not physical
   transducer count, packages, mounts, placement, or sensor technologies.
5. **Multi-target estimation and confounding diagnostics.** The task estimates
   pier-specific support-stiffness loss, identifies the most affected pier, and
   measures bearing error and scour↔bearing leakage where bearing heads exist.
6. **Seven paired L60 mechanism contrasts.** The primary analysis uses the exact
   common outer-joint StateUIDs, paired training seeds, 100,000 state-first
   bootstrap replicates, and Bonferroni familywise intervals.
7. **Transparent physics and claim boundaries.** Scour is stiffness loss, crack
   is a uniform damaged-element \(EI\) block, hanging sleepers are linear
   support removal, and L99.6 is a blockwise scale/stress experiment.

Do not state these as successful findings before the registered analyses exist.

## 1. Introduction

### Motivation

- Scour threatens railway-bridge availability and safety, while conventional
  inspection can be intermittent and difficult during hydraulic events.
- Drive-by monitoring can reuse an in-service vehicle, but feasibility alone
  does not answer the implemented design questions: which modeled response
  channels and architecture, what is gained by the full response array, which
  pier, and what happens when other mechanisms share the signal pathway?
- A naive ablation can be misleading if rungs use different state inventories,
  passages from one state leak across splits, every candidate receives an
  independent HPO lottery, or nuisance randomness is unpaired.

### Gap

Position the paper against primary drive-by scour work by Cantero, Fernandes
and collaborators, plus relevant bearing/track-roughness studies. Existing work
already establishes important feasibility, PAA, EOV, and sensor-placement
precedents; do not claim those concepts as new. The methodological delta is the
registered architecture/response-channel-subset experiment with fixed semantic
populations, common random numbers, separate execution blocks, and paired
state-level inference. Physical sensor-placement optimization remains outside
the implemented experiment.

### Research questions

- RQ1: Under the registered full-array calibration, which architecture and
  two-channel response subset minimize inner-validation scour MSE at each
  geometry anchor?
- RQ2: What performance is lost or gained by using the selected two-channel subset
  rather than the full eight-DOF response array?
- RQ3: How accurately can one passage estimate multiple pier-specific
  stiffness-loss targets and localize the most affected pier?
- RQ4: How do bearing, crack, persistent FRA-4 profile variability, track-layer
  mechanisms, and wheel polygonization change achievable L60 performance?
- RQ5: How does the method behave in the separate four-span L99.6 scale/stress
  block?

### Claimed scope in the introduction

Say “modeled support-stiffness loss” rather than physical scour depth. Say
“response channel/DOF” rather than physical sensor count or placement. Say
“performance under rung-specific retraining” rather than zero-shot OOD
robustness.

## 2. Related work

Organize by four strands:

1. drive-by bridge/railway monitoring and TTBI simulation;
2. scour-as-support-stiffness-loss and bridge frequency/deflection response;
3. multi-damage/confounder studies, including bearing and damaged-element
   crack representations;
4. preprocessing, architecture, uncertainty, and sensor placement in drive-by
   ML.

### Literature attribution guardrails

- Cite Cantero for the applicable TTBI formulation/version.
- Cite Fernandes only for mechanisms and methods actually shared with the
  implementation.
- Do not cite Sinha for the uniform damaged-element crack block; explain that
  Sinha's model is tapered and crack-depth parameterized.
- Do not use EN 13848-2 measurement repeatability as a physical rail-profile
  change model.
- Describe FRA class 4 as the registered benchmark, not the universally
  roughest permissible track.
- Distinguish hanging-sleeper support removal from a nonlinear
  void-depth/gap-impact model.

The verified paper-by-paper comparison belongs in
`docs/paper1_related_work.md`; re-check all final citations against primary
sources.

## 3. Methods

### 3.1 Registered TTBI model

- 2-D vertical train–track–bridge coupling;
- L60 three-span and L99.6 four-span bridges;
- five-vehicle train, leading-vehicle measurements;
- eight candidate response DOFs;
- exact authorized physical parameters, mesh, solver, time step, crop, and
  healthy modal-frequency sanity checks;
- bilateral-contact limitation, flats disabled, contact diagnostics and
  time-step closure.

### 3.2 Damage and nuisance semantics

- scour: \(k_v(d)=(1-d)k_{v0}\), target = support-stiffness loss%;
- bearing: nominal rotational fixity, not physical damage%;
- crack: uniform damaged-element \(EI\) loss, not Sinha/crack depth;
- profile: fixed baseline through `s13`, then per-state FRA-4 phase distribution;
- ballast/pad/hanging sleepers: registered linear track-layer mechanisms;
- hanging-sleeper gap/void depth absent;
- wheel polygonization included with probability/order/lognormal-amplitude law
  reported exactly; flats excluded.

### 3.3 Rung graph

Show the L60 directed graph, not a single ladder:

```text
             ┌─ s11_bear ─────┐
s0_scour ────┤                ├─ s13_bearcrack ─ s14_prof ─ s15_track ─ s16_all
             └─ s12_crack ────┘
```

Registered L60 edges are
`s0→s11`, `s0→s12`, `s11→s13`, `s12→s13`, `s13→s14`,
`s14→s15`, and `s15→s16`.

State explicitly that the bearing edges add bearing physics, output heads, and
multi-task learning together. State that `s13→s14` replaces the fixed profile
with the per-state FRA-4 distribution.

Present `s21→s22→s23` in a separate panel labelled **L99.6 blockwise
scale/stress design**. Do not draw a causal `s0→s21` arrow.

### 3.4 State populations and CRN

Report the fixed families:

- L60: 50 healthy + 50 scour-only + 50 bearing-only + 50 nuisance-only +
  250 joint = 450;
- L99.6: 50 + 75 + 50 + 50 + 250 = 475;
- 50 passages/state.

Explain semantic StateUIDs, latent bearing/crack designs, master joint LHS,
collision-gated StateSeedIDs, and UID-named state/passage streams. Clarify that
the state is the analysis/resampling cluster relative to passages, which are
correlated repeated observations. The fixed anchors and one joint LHS realization
are not an iid field sample.

### 3.5 Data and preprocessing

- noise-free raw MATLAB signals;
- exact time-to-space transform in Python;
- training-only per-channel scaling;
- PAA window means to 512 segments;
- persistent vs passage-level draws; production rail profiles persist per state,
  while wheel OOR redraws by passage and `profile-passage` is only a reserved
  dormant/deprecated namespace;
- 50×2 speed/temperature LHS rounded to integer km/h/°C and exact five-vehicle
  Gaussian variability laws (body mass 10% CV, primary and secondary stiffness
  5% CV each);
- global-DOF-keyed `all_mult` observation noise.

Call `all_mult` a symmetric 5% relative-noise stress, not a physical
accelerometer noise model.

### 3.6 Architecture and response-channel design

Compare:

- CNN + multi-rate pooling;
- Space2Vec CNN + multi-rate pooling;
- LSTM CNN + multi-rate pooling;
- plain CNN + GAP.

Treat the full array as a non-selectable response-budget control and candidate
two-channel subsets as the reduced-input budget. `s0` and `s21` independently select
their block reference architecture/pair; each anchor's 100-trial full-array
calibration also supplies that anchor's non-selectable eight-DOF control.
Each anchor also evaluates 8 single channels × 4 architectures × 3 seeds as a
frozen diagnostic that cannot select the reference.
`s16` and `s23` reopen the exact 4-architecture × 28-pair × 3-seed matrix with
frozen singleton parameters as exploratory deployment analyses. Their
separate winners cannot rewrite the block reference used for registered
contrasts.

State explicitly that the five vertical-acceleration and three pitch-angular-
velocity channels are modeled response quantities, not eight accelerometers.
A subset can map to one or several devices; physical device count, package,
mount, and placement are not identified.

### 3.7 HPO and execution policy

- two independent physical blocks: L60/s0 and L99/s21;
- at each anchor: full eight DOFs only, 4 architectures × 3 seeds × 100 trials,
  registered multivariate TPE and Successive-Halving pruner enabled;
- canonical best parameters stored per architecture × seed;
- all candidates/followers: one exact Optuna singleton trial, no pruner;
- total 1,620–1,638 studies and 3,996–4,014 trials, with the range determined
  by designed-pair/carried-pair deduplication at six frozen rungs;
- zero `FAIL` tolerance and OOM fatal;
- capacity preflight checks total VRAM minus this process's peak reserved
  memory, not system-wide free VRAM; require an otherwise idle/exclusive GPU;
- all seven L60 Python rungs share one exact host/GPU/runtime and all three L99
  rungs share one exact host/GPU/runtime, which may differ from L60;
- durable absolute execution/HPO/champion paths; the champion carries
  `frozen_selection_sha256`, and every follower pins the exact anchor-printed
  canonical champion-manifest hash through `TTBI_BLOCK_REFERENCE_SHA256`; no
  cross-block copying or post-anchor substitution;
- the legacy-named benchmark runner is now a contract-guarded genuine R11
  runner: authenticated 475-state × 50-passage × 8-channel × 512-segment,
  five-head workload; one 100-trial anchor HPO study; and exactly one shared
  finalist-CV refit. Its heavy commit-A execution remains pending, with zero
  `FAIL`, OOM, retry, or replacement allowed.

The methods table must report batch size 32, maximum 50 epochs, patience 5,
Adam, cosine scheduling (`T_max=50`, `eta_min=0`), the complete conditional
search domains, multivariate TPE with 25 startup trials and constant-liar
handling, and the Successive-Halving settings (`min_resource=4`,
`reduction_factor=3`, `min_early_stopping_rate=0`). Report the exact
determinism policy and seed registry from the authorized protocol rather than a
generic “fixed seeds” statement.

### 3.8 Split and selection firewall

- deterministic semantic-UID-stable 60/20/20 split;
- state grouped and stratified by registered family/target/level/joint strata;
- exact same partition across rungs of one geometry;
- training-only scaler;
- inner validation for HPO and selection;
- finalist 5-fold × 2-repeat grouped CV on development data only, diagnostic
  and unable to re-rank; after deduplication its fixed set is the winner, top
  five architecture×channel combinations from the complete inner-validation
  factorial leaderboard, each architecture's optimum, and same-pair, designed,
  carried-reference, and full-array controls;
- immutable outer test opened after freeze.

### 3.9 Outcomes and inference

Within rungs:

- scour MSE/per-pier MSE and RMSE;
- most-damaged-pier localization as a passage-level point estimate only when
  maximum true scour is strictly greater than 5 percentage points; no
  registered state-level localization interval;
- bearing MSE and leakage diagnostics;
- finite-seed median and seed IQR;
- 2,000-draw, seed-42 state-first uncertainty for registered MSE report
  comparators.

Across L60:

- analyzer supplied with canonical external champion, HPO-manifest, execution-
  receipt, and all seven exact rung-summary paths; embedded copies are
  insufficient;
- exact paired outer-test joint StateUID × seed cells;
- mean over states within seed, median over fixed seeds;
- right-minus-left edge effect in squared percentage points;
- 100,000 paired state-first bootstrap replicates;
- pointwise 95% and seven-edge Bonferroni familywise intervals;
- exploratory bearing×crack difference-in-differences.

Describe both bootstrap analyses as empirical finite-design state-resampling
uncertainty conditional on the registered anchor/LHS design. Do not claim
field-population or design-superpopulation coverage without LHS-aware variance
estimation or independent replicated state designs.

### 3.10 MATLAB host qualification and production parity

- assign every intended MATLAB generation PC one stable, unique
  `TTBI_QUALIFICATION_HOST_ID`;
- freshly regenerate transient qualification scripts from clean commit A; run
  at least `s0_scour`, `s16_all`, and `s23_all4` on every required host;
- retain each authenticated `qualification_host_receipt.json`, whose hardware
  diagnostics are bound to the actual MATLAB environment, qualification source,
  and exact executed script;
- compare corresponding stage outputs for the required host/environment pairs
  and retain accepted `matlab-environment-qualification-receipt-v4` evidence;
- require explicit acceptance for numerical equivalence; do not infer
  qualification from a comparator exit log alone;
- CPU equality is not required. Equal MATLAB-environment digests are allowed
  only for distinct authenticated host IDs, and host identity is not part of
  production `gen_schema` or `gen_fingerprint`;
- report the present 35-state/105-passage `s0_scour` laptop micro (27 min 32 s)
  only as pre-convergence, one-host integration/timing evidence. Every intended
  host must still run all three stages from commit A, and corresponding outputs
  from distinct authenticated host IDs require accepted v4 receipts; and
- in production, wait specifically for a complete `0001.mat`, then run the
  MATLAB raw-parity smoke and dependent Python checker sequentially. Do not
  admit later output until both pass.

## 4. Results — populate only after R11 completion

### 4.1 Qualification and sample accounting

Report:

- authorized source commit and protocol hashes;
- every per-host MATLAB qualification sidecar and accepted v4 pairwise
  comparison receipt, plus environment/execution/capacity receipt identities;
- exact state/passages counts and excluded passage reasons;
- contact-closure result;
- complete HPO terminal-state accounting;
- no split or artifact-provenance failures.

### 4.2 Anchor architecture/response-subset selection

Use separate L60 and L99 tables. For each architecture/pair report the
three-seed inner-validation median/IQR and eligibility. Keep outer-test values
out of selection tables.

### 4.3 Two-channel versus full-array performance

For each registered rung, compare the frozen two-channel reference against the full
array with paired state-level effect estimates. Avoid “2 ≈ 8” unless the
predefined equivalence/noninferiority criterion exists; otherwise report the
effect and interval.

### 4.4 Multi-pier estimation and localization

Report per-pier error and aggregate scour error with their registered
state-resampling intervals. Report localisation separately as the registered
passage-level point estimate; do not attach the MSE interval to it. Do not call
the task “detection.”

### 4.5 Bearing estimation and leakage

Separate:

- bearing-head predictive error;
- false scour response in bearing-only states;
- false bearing response in scour-only states;
- the combined cost of bearing physics plus extra heads/multi-task training.

### 4.6 Seven L60 edge effects

Primary table:

| Edge | Mechanism/task change | Estimate (right−left) | Pointwise 95% CI | Bonferroni familywise CI | Outer analysis states |
|---|---|---:|---:|---:|---:|
| `s0→s11` | bearing physics + heads/multi-task | pending | pending | pending | pending |
| `s0→s12` | crack nuisance | pending | pending | pending | pending |
| `s11→s13` | crack nuisance with bearing task | pending | pending | pending | pending |
| `s12→s13` | bearing physics + heads/multi-task | pending | pending | pending | pending |
| `s13→s14` | fixed profile → per-state FRA-4 | pending | pending | pending | pending |
| `s14→s15` | track-layer mechanisms | pending | pending | pending | pending |
| `s15→s16` | wheel polygonization | pending | pending | pending | pending |

Only the Bonferroni interval supports a familywise sign statement. Report the
exploratory difference-in-differences separately.

### 4.7 L99.6 blockwise scale/stress results

Report L99 independently, with its own anchor/reference/protocol identities.
Do not interpret `s0→s21` as an L60-to-L99 treatment effect and do not insert
L99 rows into the seven-edge family.

### 4.8 Diagnostic repeated CV

Report split-stability ranks/intervals separately and state that CV did not
re-rank the canonical selection or replace the outer-test estimate.

## 5. Discussion

Discuss only mechanisms supported by paired estimates and their intervals.
Potential topics:

- whether multi-rate pooling adds value beyond the plain CNN/GAP control;
- whether selected response-channel subsets change between geometry/EOV deployment
  regimes;
- whether the full array materially improves over the selected two-channel budget;
- which registered nuisance additions dominate error;
- whether bearing leakage is operationally important; and
- how the L99.6 block differs without converting the comparison into a causal
  scale effect.

### Required limitations

- simulation-only, no field validation;
- 2-D vertical model; no lateral/torsional/pier-tilt response;
- support spring is not full soil–structure interaction or hydraulic scour;
- label is stiffness loss, not depth;
- bearing fixity is a design variable;
- crack is uniform damaged-element \(EI\) loss, not crack depth/Sinha;
- hanging sleepers lack gap/contact/void-depth nonlinearity;
- flats excluded and wheel–rail contact is bilateral;
- `all_mult` is not a datasheet-based sensor model;
- candidate space, physical sensor mapping, and finite full-array-calibrated
  HPO searches bound all optimality claims;
- finite-design bootstrap intervals are not field-population coverage
  intervals;
- rung-specific retraining is not zero-shot OOD evaluation;
- no detection/POD/sensitivity/minimum-severity claim without a locked binary
  threshold;
- L99.6 is blockwise, not confirmatory one-factor inference.

## 6. Conclusion

Summarize only authenticated R11 outcomes. Keep the final sentence at the level
of modeled support-stiffness-loss estimation under the registered simulation
and state the field-validation requirement.

## Figure plan

| Figure | Content | Source/status |
|---|---|---|
| 1 | TTBI model, bridge geometries, eight response DOFs | redraw from authorized model |
| 2 | L60 directed edge graph + separate L99 block | create from registered protocol |
| 3 | fixed state families, latent design and named CRN streams | create |
| 4 | PAA + four architecture arms, including plain CNN/GAP control | redraw |
| 5 | split/HPO/test firewall and two independent execution blocks | create |
| 6 | L60 anchor architecture/response-subset results | regenerate |
| 7 | selected two-channel vs full-array paired effects | regenerate |
| 8 | seven-edge estimates with pointwise and Bonferroni intervals | regenerate |
| 9 | per-pier parity/localization and bearing leakage | regenerate |
| 10 | L99 blockwise results | regenerate |

## Table plan

1. Literature delta table, verified against primary sources.
2. Authorized TTBI and damage/EOV parameter table.
3. State families, counts, persistence, and inferential roles.
4. Architecture/HPO/search-budget specification.
5. L60 and L99 anchor selection tables.
6. Per-rung outer-test performance and analysis-state counts.
7. Seven L60 paired edge effects.
8. L99 blockwise stress results.
9. Limitations-to-claim mapping.

## Final manuscript audit

- Every result path resolves to the authorized source/protocol/execution/HPO
  lineage.
- No historical figure or result number remains.
- “State” and “passage” counts are never conflated.
- Selection, diagnostic CV, outer-test reporting, seven-edge inference,
  exploratory deployment selection, and L99 blockwise analysis are visibly
  separated.
- All multiplicity claims use the registered seven-edge family.
- All physical labels and model limitations use the terminology above.
- No statement elevates the symmetric relative-noise arm to a hardware sensor
  comparison.
- No text calls one-host micro evidence a completed cross-host qualification or
  treats CPU equality as a scientific requirement.
- The reported full-array benchmark comes from the reviewed genuine R11
  eight-channel runner, never from the legacy two-channel fixture.
