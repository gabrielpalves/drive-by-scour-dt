# SHM-group talk — staged ablation for drive-by damage ID (slide plan)

Audience: Prof. Todd + the UCSD SHM group. Goal: convey the **staged ablation design**, the
**classification → regression** move, the **champion-selection metric**, the **tests we run**,
and the **whys** — at a level that invites methodological critique. ~15–20 min + questions.
Built from `docs/paper1_outline.md`; every slide has a "why it's there" so you can defend it.

Convention below: **[N] Title** — bullets = on-slide; *say:* = the spoken why; *fig:* = visual.

---

**[1] Title + one-line thesis.**
- "Drive-by monitoring turns a train into a mobile sensor. We ask the design questions after
  feasibility — which architecture, how few sensors, can one pass localise + disentangle
  damage, and how does it behave under realistic variability — with a **staged ablation**."
- *say:* set expectation that this is a methodology talk, not a single result.

**[2] Problem + motivation.**
- Scour = leading, invisible, flood-accelerating cause of bridge failure; inspection misses
  early subsurface erosion. Direct SHM is costly per-bridge; drive-by scales across a line.
- *say:* the operational hook — bridges are a small, outsourced fraction of a route (Tereza
  Cristina), so "few sensors on an in-service train" is the realistic deployment.
- *fig:* TTBI schematic (train–track–bridge) + scour lowering support stiffness.

**[3] Prior art in one slide (Fernandes line) + our gap.**
- They establish feasibility + multi-damage *classification* (PAA+CNN, Bayesian opt, EOVs,
  car-body vs bogie). We build ON that (same group lineage — say so).
- Gap = the *design* questions, answered *systematically*.
- *say:* honesty guardrail — we are not claiming they lack robustness/EOV; our delta is the
  **staged, one-factor-at-a-time** character.

**[4] The organising idea: a STAGED ablation (the spine).**
- Diagram: Stage 0 (localise: 2 piers) → Stage 1 (+bearing: disentangle) → Stage 1c/1f
  (+crack, +profile EOV) → Stage 2 (scale: 4 spans / ~100 m) → Stage 3 (all-damage).
- Rule: **select the architecture once, then FIX it** → each stage changes exactly one factor.
- *say:* this is the core methodological contribution — it's what lets us attribute cause. A
  monolithic "train on everything" experiment can't tell you whether a drop came from the
  extra pier, the second damage, or the roughness.
- *fig:* the staged pipeline (make this the anchor slide you return to).

**[5] Why classification → regression.**
- Single-scour study: 61 **ordinal classes** (0–60% @1%), MSE on the class index (ordinal so
  41-vs-42% is cheap, 5-vs-55% is not).
- Multi-damage: **continuous per-target regression heads** (one per pier + per bearing).
- Three whys: (i) multiple independent piers ⇒ joint classes blow up combinatorially;
  (ii) continuous severity is the state a **digital twin** tracks; (iii) it gives per-pier /
  per-head error + a direct **localisation** read.
- *say:* same backbone, same PAA front end — only the head + loss change. Clean control.

**[6] The architecture we ablate (and the physics claim).**
- Modular 1-D CNN + 3 toggle blocks: Space2Vec (spatial embedding), LSTM (recurrence),
  **N-HiTS multi-rate pooling**.
- **Physics claim:** the drive-by signal is two-timescale — fast wheel/rail transients over the
  slow deflection basin that actually encodes scour. Multi-rate pooling represents both by
  construction.
- *fig:* signal with the two timescales annotated; the modular block diagram.
- *say:* this is why we expect pooling to win *and* why spatial-embedding/recurrence shouldn't
  help the already-strong channels — capacity without matching structure adds variance.

**[7] The champion-selection metric (a methodological point Todd's group will probe).**
- We do NOT rank by the single luckiest Optuna trial. We rank by **median error + IQR + a
  collapse-rate** (fraction of runs that fail to learn the ordering), with a **UCB** variant
  `MSĒ + 1.96·σ/√n`.
- Multi-damage grid: **3 independent seeds/config**; train/val split fixed (seed 42) so seed
  spread = init/HPO variance only. Single-scour study used 30-seed UCB.
- *say:* the collapse-rate is itself a *result* — some single-sensor configs look fine on the
  lucky trial and are unusable in deployment. This is the "reliably good, not occasionally
  excellent" argument. Invite critique here (it's a selection-statistics choice).

**[8] The tests we run (the ablation grid).**
- Phase 1: single-DOF sweep (8 sensors). Phase 2: auto-selected best pair (+ designed mixed
  pair). Per stage: per-pier MSE, aggregate MSE, **localisation accuracy**, and (Stage 1+)
  bearing MSE + a **scour↔bearing leakage** report.
- HPO: Optuna multivariate TPE, 25% random start, 100 trials, pruning.
- *say:* deliberately reduced from the full single-scour grid (that established the
  architecture); later stages are champion-only so compute buys *seeds/stages*, not arms.

**[9] Result — Stage 0: localise + quantify from one pass.**
- Champion pair: **aggregate MSE 0.757 (~0.87 pp RMSE), per-pier 0.72/0.79, localisation
  0.990**; best single 0.86 (pair only ~12% better).
- *say:* one drive-by pass resolves *which* of two independent piers is scoured and *how much*
  — no aggregate/max fallback.
- *fig:* Stage-0 parity (pred vs true, per pier).

**[10] Result — Stage 1: disentangling scour from a seized bearing.**
- Add left/right bearing heads. Scour cost: **0.757 → 1.80** MSE (0.87→1.34 pp). Bearing skill
  real (RMSE ~2–3 seized-%). **False-scour-from-bearing leakage = 2.37 pp** (the safety-critical
  direction). Pair flips to FrontBogie_Vert+CarBody_Pitch (bearing-sensitivity + entry abutment).
- *say:* this is the "does a second, co-located damage fool the scour estimate?" question,
  answered with a number, not a hope. Tie to the k_r sensitivity sweep (methods fig).
- *fig:* Stage-1 4-head parity + a leakage bar.

**[11] Result — the roughness finding (the interesting one).**
- Under realistic **per-passage** roughness (first pilot), **all sprung channels collapsed** to
  predict-the-mean; **unsprung wheel channels survived** → the clean-track ranking **inverted**.
- Physics: suspension filtering + car-body resonance masks the quasi-static scour signal;
  the unsprung mass traces profile+deflection directly (axle-box-acceleration literature).
- **Mixed pair FrontBogie_Vert+Wheel1 beat wheel+wheel by 17%** despite the bogie alone being
  collapse-level — a TSD/two-axle **residual (profile-reference) fusion** mechanism.
- **Caveat (say it):** pilot used deprecated geometry + an over-aggressive per-passage EOV;
  we corrected the EOV design (per-STATE profile, FRA-4, EN 13848-2 rationale) and are
  regenerating — direction is robust, magnitudes pending.
- *say:* this is where the group's roughness/UQ instincts will engage — lean in.

**[12] Whys, gathered (the intellectual core).**
- Pooling wins because it matches two-timescale physics (not tuning luck) — consistent across
  classification AND regression.
- "Which sensors" is **regime-dependent**: sprung win on clean track, unsprung under roughness,
  and a **mixed pair** hedges — so sensor economy is really a **fusion** question.
- Selection by robustness (collapse-rate) not point estimate.

**[13] Limitations (say them before they ask).**
- Simulation-only; 2-D vertical (no lateral/torsion/pier-tilt; support-spring SSI only);
  bearing range ~0–31% fixity; profile as FRA-4 domain randomisation, not measured evolution;
  the deck-mass/frequency sanity-check is on the verify list.

**[14] Forward link → the digital twin (teaser, optional).**
- One line: continuous severity + localisation + leakage feed a **value-of-information DT**
  (Paper 2). If asked, switch to the DT explainer (`docs/dt_torzoni_explainer.md`).

**[15] Backup slides.**
- 3-arm architecture table; leave-one-out; k_r sweep; EOV design table (per-state vs
  per-passage, FRA-4, EN 13848-2); noise-domain caveat (before/after interpolation);
  Torzoni DT staging.

---

## Delivery notes
- Anchor on slide [4] (the staged diagram) — return to it between results so the audience
  always knows which factor is being varied.
- Two "invite critique" moments for Todd's group: the **selection metric** [7] and the
  **roughness/EOV** finding [11]. These are where methodological feedback is most valuable.
- Numbers to have memorised: localisation 0.99; 2≈8; scour 0.76→1.80 with bearing; leakage
  2.4 pp; mixed-pair +17%. Everything else can live on slides.
- If short on time, cut [5]-detail and [8]-detail, never [4], [7], [11].
