# Paper 1 — Outline & Draft Skeleton (v2, staged multi-damage)

**Working title:** *A staged, physics-matched ablation for drive-by identification of
scour and bearing damage in multi-span railway bridges*

(Alt, if the reviewer wants the architecture foregrounded: *Multi-rate temporal pooling as a
physics-matched inductive bias for drive-by railway-bridge damage identification: a staged
architecture-and-sensor ablation*.)

**Status:** updated 2026-07-15 to the final **10-rung ladder**, the anchored EOV design, the
load-time noise model, and the corrected crack prior. Supersedes the v1 skeleton
(single-scour classification only). Results below come from the *superseded* Stage-0/Stage-1
datasets and the L100 pilot — the whole campaign is being **regenerated from scratch** on the
new ladder, so treat every §5 number as a *direction*, not a final value, until it lands.
Numbers marked ⚠ need a check at write-time (see the verify list).

**Target venue:** TBD (confirm with advisor). Ranked fit — MSSP (top; signal-processing +
drive-by community) → Structural Health Monitoring (Sage) → CACIE (ML angle) → Engineering
Structures (application) → Sensors (fast fallback). Framing knob: MSSP wants the
signal/physics novelty foregrounded; CACIE the ML; Eng. Struct. the application.

---

## 0. The one-paragraph story (say this first, in the talk and the abstract)

Drive-by monitoring turns an in-service train into a mobile sensor for the bridges it
crosses. Prior work (esp. the Fernandes line) establishes that vehicle accelerations carry a
detectable scour signature. We ask the *engineering-design* questions that come next and
answer them with a **staged ablation** that changes **one factor at a time**: (1) *which*
architecture suits this signal, and why — we isolate each block and argue **multi-rate
temporal pooling (N-HiTS) as a physics-matched inductive bias**; (2) *how few* on-board
sensors suffice — a full sensor-economy sweep gives **two sensors ≈ eight**; (3) can a single
pass **localise and quantify** scour at **several independent piers** — yes, at ~99%
localisation; (4) does a **co-located bearing** damage confound the scour estimate — we add a
bearing head and measure the **cross-leakage**; and (5) how does the method behave when
**environmental/operational variability (EOV)** — track roughness, cracks — is injected as
*domain randomisation*. Along the way we (a) move from coarse **classification** to
**continuous ordinal regression**, the natural state for a downstream digital twin, and (b)
select the champion by a **robustness criterion** (median error + collapse-rate + UCB) rather
than the single luckiest tuning trial. A recurring, literature-consistent finding: under
realistic **per-passage roughness**, suspension-filtered car-body/bogie channels lose the
quasi-static scour signature while **unsprung (axle-level) channels retain it** — and a
**mixed unsprung+sprung sensor pair** recovers performance via a residual (profile-reference)
mechanism.

---

## 1. Contributions (state these explicitly in the intro)

1. **A staged ablation methodology for drive-by damage ID.** We select the architecture ONCE
   on the simplest task and then FIX it, so every later rung varies exactly one scientific
   factor. The ladder follows the questions a bridge maintainer actually asks — *can I
   localise scour? does a bearing fool me? a crack? both? does rail roughness? track damage?
   the train itself?* — and answers each with a number rather than a hope. This isolates
   *what causes what* in a way a monolithic "train on everything" experiment cannot.
   **(The organising contribution.)** Ten rungs:
   `s0_scour → s11_bear → s12_crack → s13_bearcrack → s14_prof → s15_track → s16_all`
   (L60/3-span), then `s21_scour4 → s22_bearcrack4 → s23_all4` (L99.6/4-span).
2. **A physics-matched architectural inductive bias.** N-HiTS multi-rate pooling mirrors the
   two-timescale physics of the drive-by signal — high-frequency wheel/rail-contact transients
   vs. the low-frequency modal/quasi-static deflection induced by support-stiffness loss — and
   is the consistent top performer across sensor sets and across the classification→regression
   change. Spatial embedding (Space2Vec) and recurrence (LSTM) do **not** help the strong
   channels and can hurt them. Reframes the architecture choice as *physics*, not tuning luck.
3. **A sensor-economy result: 2 ≈ 8, and a physics of *which* two.** A single-DOF → leave-one-
   out → pair → forward sweep shows a two-sensor pair matches the full array, and identifies
   which channels carry the signature and which are dead weight — with an interpretable,
   suspension-chain explanation.
4. **Multi-pier localisation + quantification from one pass**, and a **scour–bearing
   disentanglement** measurement (per-head error + a **cross-leakage** metric: how much false
   scour a seized bearing induces, the safety-critical direction).
5. **A robustness-based model-selection criterion** (median error + IQR + **collapse-rate**,
   with a UCB variant), rather than the single luckiest Optuna trial — rewards models that are
   *reliably* good and exposes fragile single-sensor configs a point estimate would hide.
6. **A literature-anchored EOV domain-randomisation design** — every nuisance parameter cited
   or derived-from-cited, with the draw *frequency* itself argued from physics (persistent
   conditions per state, not per passage; EN 13848-2, Sato/Shenton) — and an empirical,
   physically-explained result that **per-passage roughness collapses sprung-channel scour
   regression while unsprung channels survive**, with a **mixed-pair fusion** remedy.
   *(Report the direction now; final magnitudes after the regeneration.)*
7. **A separation of likelihood from prior.** Inter-pier scour is physically dependent, yet we
   train on an **independent** LHS: the network must learn `p(response | state)`, and baking
   the correlation `p(state)` into it would (i) let the model infer a pier's condition without
   evidence and (ii) destroy the localisation claim it is meant to test. The dependence is
   modelled *mechanistically* in the digital twin (shared flood driver), not in the training
   set — a modularity argument we believe generalises to other multi-damage SHM learners.

> Positioning vs. Fernandes (prior art): they establish drive-by scour *feasibility* (and a
> multi-damage *classification*). We answer *which architecture and why*, *how few sensors*,
> *can we localise across piers and disentangle a second damage*, and *how to select and
> stress-test the model* — as a staged, one-factor-at-a-time study. See the delta table in §3.

---

## 2. Introduction (draft beats)

- Scour = a leading cause of bridge failure worldwide, submerged and invisible, accelerating
  with climate-driven flood frequency; visual inspection misses early subsurface erosion
  between cycles. [Kamariotis 2024; HEC-18; Lamb 2019; Prendergast & Gavin]
- Direct SHM (structure-mounted sensors) is costly to deploy per-bridge across a network;
  **drive-by / indirect** SHM turns an in-service train into a mobile sensor, scalable over a
  line, and is attractive precisely where bridges are a small, outsourced fraction of the
  route (motivating anecdote: Ferrovia Tereza Cristina, ~160 km). [reviews: Malekjafarian/
  OBrien; Corbally & Malekjafarian; Fernandes 2024–2026]
- Gap: detection *feasibility* is established, but the **design questions** a network operator
  faces before deployment are open — which architecture, how few sensors, can one pass handle
  several piers, does a second damage confound the estimate, and how does the method behave
  under realistic operational variability. Existing work typically fixes a black-box CNN and
  compares a couple of sensor positions.
- Our answer: a **staged ablation** (the six contributions). Roadmap sentence.

---

## 3. Related work + the Fernandes delta table

**VERIFIED DRAFT: see `docs/paper1_related_work.md`** — per-paper summaries (2024 JCSHM DAE;
2025 IJSSD multi-damage PAA+CNN classification; 2026a ASCE speed→~100% car body; 2026b LATAM
CNN-LSTM on the real Canelas bridge), the verified Table 1, the drop-in delta paragraph, and
honesty guardrails. **v2 additions to fold in:** the roughness/axle-box strand that grounds
our EOV + fusion findings — see `papers/Drive-By Scour ML Literature Design` and
`paper1_related_work.md` §"Roughness, suspension filtering, and sensor fusion". Key anchors:
profile is held FIXED-per-scenario in published drive-by ML (Locke 2020; NuBe-DBBM / Sarwar &
Cantero; Fernandes fixed FRA-4); axle-box acceleration retains the geometry the suspension
filters out; two-axle/TSD residual methods cancel roughness by fusing axles (OBrien/Keenahan).

**Delta in one line:** Fernandes establishes *feasibility* and multi-damage *classification*
(PAA, CNN/CNN-LSTM, Bayesian opt, multi-run robustness, EOVs incl. track profile, car-body-vs-
bogie). Our advances: the **staged one-factor-at-a-time design**, the **component-level
architecture ablation** (Space2Vec / LSTM / **N-HiTS pooling as two-timescale physics**), the
**full sensor-economy ablation → 2 ≈ 8**, **multi-pier localisation + scour–bearing leakage**,
the **classification→ordinal-regression** move, the **collapse-rate/UCB** selection, and the
**roughness→sprung-collapse / unsprung-survival + mixed-pair fusion** finding. Do NOT claim
they lack robustness/EOV/PAA/sensor comparison — they have all four; our delta is the
*systematic, staged* character. (Full table + guardrails in `paper1_related_work.md`.)

---

## 4. Methodology

**FULL DRAFT PROSE: see `docs/paper1_methodology.md`** (v2 in progress — being expanded to the
regression formulation, the staged design, the champion metric, and the EOV design). The
beats below are the summary.

**4.1 Physical model — TTBI.** Cantero 2-D train–track–bridge interaction (TTB-2D / VEqMon2D);
multi-span Euler–Bernoulli deck on spring supports; Zhai layered ballasted track; 5-vehicle
planar train, sensors on the **leading** vehicle, 8 DOFs (vertical accel: car-body, 2 bogies,
2 wheelsets; pitch rate: car-body, 2 bogies). ⚠ **Bridge dynamics sanity-check REQUIRED**:
the deck `ρ·A = 9.6 kg/m` in isolation implies an unrealistically high fundamental — confirm
the as-built (deck+track+ballast) fundamental frequency is physically representative (few Hz)
from the model's own `B09`/`B56` output before submission; report it and the effective spans.

**4.2 Damage models.**
- **Scour** = vertical support-stiffness loss at a target pier, `k_v(d)=(1−d)·k_v0`
  (⚠ `k_v0≈3.44e8 N/m`), `d∈[0,60%]`; lowers global stiffness / quasi-static deflection under
  the moving load. [Prendergast & Gavin; Kamariotis 2024; Fernandes]
- **Bearing** (Stage 1+) = abutment **rotational** spring, healthy `k_r=0` (free), seizure →
  large `k_r`; mechanism identical to Fernandes 2025 and Khan 2022, sampled continuously over
  `[0,1e9]` Nm/rad and regressed as a **seized-%** head. The `1e9` anchor = Fernandes's
  minimum drive-by-detectable seized bearing; our own sensitivity sweep (below) shows it maps
  to ~0–31% analytic fixity — a documented **range** limitation, not a noise floor.
- **Crack** (Stage 1c+) = damaged-element EI reduction (uniform I drop on the affected ~0.3 m
  element(s), cf. Fernandes et al. 2025), a **nuisance/EOV** (not a label). ⚠ 2026-07-17: do NOT
  cite this as Sinha — Sinha/Friswell/Edwards (2002) is a *tapered* (piecewise-linear) EI model
  parametrized by crack depth; ours is the classical damaged-element benchmark.
- **Track-layer + wheel OOR** (Stage 3) = ballast/hanging-sleeper/pad + wheel-flat nuisances.

**4.3 From classification to regression (a deliberate reformulation).** The single-scour
architecture study framed severity as **61 ordinal classes** (0–60% in 1% steps) scored by MSE
on the class index — ordinal because a 41-vs-42% confusion is cheap and 5-vs-55% is not. The
multi-damage study drops the discretisation and regresses **continuous per-target heads**
(one per scoured pier + one per bearing), MSE loss. Why: (i) multiple independent piers make a
joint-class scheme combinatorial; (ii) continuous severity is the state a **digital twin**
tracks; (iii) it exposes **per-pier / per-head** error and a **localisation** read directly.
Same backbone, same PAA front end — only the head and loss change (`core/task.py`).

**4.4 EOV as literature-anchored domain randomisation.** *(Full spec: methodology §A.3.)*
Persistent **conditions** — the track-profile realisation, any crack, the track-layer state —
are drawn **once per damage state** and held across its passages (track geometry evolves over
MGT, not between trains: EN 13848-2 ≤0.5 mm repeatability; Sato/Shenton), plus a small
per-passage jitter; only *operational* variability is redrawn per passage (speed, temperature,
vehicle properties, wheel damage). Profile fixed at **FRA 4**. **This corrects the first
scale-up pilot**, which redrew a fresh profile+class and crack *every passage* — the direct
cause of its sprung-channel collapse (§5.5). Track-layer numbers are now **anchored** via a
resolution of the *prevalence paradox*: the reported "~50% of sleepers show some voiding" and
a ~9% mechanical model are both right because most voids are **sub-threshold** (impact
threshold 1.0–2.5 mm; only 10–20% of settled sleepers exceed 1.5–2.0 mm ⇒ **5–10%
impactfully unsupported**) → Poisson λ=3.0 groups and λ=1.2 fouled patches per 100 m, pads at
**snapshot prevalence** 2% (the cited 0.5% is an annual *incidence* rate), fouling↔voiding
**coupled ×3**, ballast **×3** near abutments. ⚠ one pivotal GPR figure needs a source
re-check.

**4.5 Measurement noise as a load-time observation model (not baked into the physics).**
Generation is **noise-free**: the generator saves the **raw time-domain** signal plus the
space-transform/crop parameters, and the interpolation, crop and noise injection all happen at
**load time** — so the sensor model is configurable per channel and per experiment without
ever regenerating data. **A reportable finding:** adding noise before vs after the time→space
interpolation is **not** equivalent — because the interpolation is linear, time-domain noise
becomes band-limited/coloured and speed-dependent (≈0.67× variance but ≈**1.46× the energy
surviving PAA** vs white noise added after). Same nominal 5%, different perturbation; we
state the model explicitly rather than leave it implicit. Future work: an **additive
datasheet-anchored** floor. *(EN 61373 severities describe the vibration ENVIRONMENT for
qualification — range/reliability — **not** an acquisition noise floor.)*

**4.5b Damage-location priors, and a correction we report.** Scour (piers), bearing
(abutments) and hanging sleepers (cited ±15 m transition spike) carry location priors by
construction or citation. For the **crack** we first drew uniformly and justified it from the
computed **moving-load |M| envelope** (broad: only ~2–4% of the range ever sees |M|<35% of
max; peaks favour mid-span ~2:1). That was **wrong — the envelope is the wrong lens**: it
answers *where bending is large*, not *where concrete fails*. Real cracking is
**hogging-dominated >4:1** (top-fibre tension over supports + runoff/chlorides; mid-span
soffit cracks close under compression), and **Eurocode 4 mandates** a cracked section over 15%
of span each side of internal supports. Now: hogging:sagging **4:1**, ±17.5% of a span.

**4.6 Signal preprocessing — PAA.** Per-channel standardisation fitted on the training split
only (fixed 80/20, seed 42, **grouped by damage state** — 2026-07-17: validation holds out
whole states/files, never passages of a trained state, so val = unseen-state generalisation),
then **PAA to 512 segments** as a structural low-pass filter:
smooths high-frequency rail/wheel content, preserves the deflection-basin signature, ~10×
length reduction. (CWT scalogram branch as a 2-D comparator.) [Fernandes 2025 PAA precedent]

**4.7 The ablated architecture.** Shared 1-D CNN backbone + three toggle-able blocks —
**Space2Vec** (spatial-position embedding), **LSTM** (recurrence), **N-HiTS multi-rate
pooling** (the physics-matched block). *The inductive-bias argument:* the drive-by signal is
two-timescale (fast wheel–rail transients over slow deflection); multi-rate pooling represents
both by construction, letting the model read the slow scour/bearing component without
discarding the fast one. Tested in §5.1.

**4.8 Architecture policy + the ladder (select once, then fix).** All arms run at `s0_scour`
(the architecture-selection rung, and the paper's architecture table); the winner is **fixed**
for every later rung, which then varies only its one scientific factor. Ladder and the three
design decisions (crack = nuisance; profile = its own rung; all-damages split track/train) in
methodology §A.2. **Heads = scour + bearing only**; crack, profile, track and wheel damage are
nuisances the network must be *invariant* to. Re-runs are, by policy, **from scratch** (tagged
studies) — never silent extension.

**4.9 Inter-pier scour: independent in training, dependent only in the twin.** See
methodology §A.6 — the likelihood-vs-prior argument, and why a correlated training set would
destroy the localisation claim.

**4.9 Hyperparameter optimisation.** Optuna multivariate **TPE** (handles the conditional
block search space), 25% random start-up, **100 trials**/study (⚠ single-scour study used more),
successive-halving pruning, objective = best validation MSE over ≤50 epochs with early stopping.

**4.10 Robustness-based selection.** Rank by **median** error with IQR and a **collapse-rate**
(fraction of seeds/configs whose error exceeds a physical tolerance — a model that failed to
learn the ordering), with a **UCB** variant `MSĒ + 1.96 σ/√n`. The multi-damage grid runs
**3 independent seeds** per config (init/HPO variance; the train/val split is fixed) and reports
the median leaderboard; the single-scour study used 30-seed UCB. Rationale: the single luckiest
trial over-states performance and hides fragility (e.g. a single pitch channel that collapses).

---

## 5. Results

**5.1 Architecture ablation (single-scour, classification — the selection stage).**
- PAA ≫ RAW. Among blocks, **N-HiTS is the consistent contributor**; S2V and LSTM help only
  weak channels and hurt strong ones. Champion = **PAA + N-HiTS**. [full 3-arm/27-row table]
- Consistency across classification (single-scour) and regression (multi-scour) is itself
  evidence the inductive bias — not a dataset quirk — drives the win.

**5.2 Sensor economy (2 ≈ 8).** Single-DOF importance, leave-one-out, best pairs, forward
sweep. Champion pair (single-scour: RearBogie_Vert + CarBody_Pitch) ≈ full array at zero
collapse; ~3 sensors saturate. The *which two* has a suspension-chain reading (§6).

**5.3 `s0_scour` — multi-pier localisation + quantification (regression).** L60 / 3-span, scour
at piers 2 & 3. *(Superseded-dataset values:)* **champion pair RearBogie_Vert + CarBody_Pitch:
aggregate MSE 0.757 (RMSE ~0.87 pp), per-pier 0.72 / 0.79 (balanced), localisation 0.990.**
Best single RearBogie_Vert 0.86 → the pair is only ~12% better = near-single-sensor
sufficiency. One pass localises AND quantifies two independent piers → no aggregate/max
fallback needed.

**5.4 `s11_bear` — bearing disentanglement.** Add left/right bearing seized-% heads.
*(Superseded-dataset values:)* **champion-pair scour MSE 0.757 → 1.798** (RMSE 0.87 → 1.34 pp;
localisation 0.988) = the measured cost of sharing the network with a second damage.
**Bearing heads carry real skill** (pair bearing MSE 7.0; left 4.8 / right 9.2 → RMSE ~2.2 /
3.0 seized-%, tight parity), consistent with the k_r sensitivity sweep (10–16% rel-RMS at 1e9;
CarBody_Pitch most bearing-sensitive). **Cross-leakage:** false-scour-from-bearing **2.37 pp**,
false-bearing-from-scour 3.43 seized-%. The auto-pair **flips** to FrontBogie_Vert +
CarBody_Pitch — explained by bearing sensitivity (CarBody_Pitch) and entry-abutment proximity
(FrontBogie_Vert); right-bearing is harder than left because the crop ends before the exit
abutment (deck-continuity information only). *Architecture-consistency note:* S2V edges the
champion on the pair (~10%) but loses 7/8 singles — reported as robustness; policy unchanged.

**5.5 `s12`–`s13` — does a crack fool the scour estimate?** The bridge-damage rungs: crack
alone, then bearing + crack (the Fernandes-comparable set). Read `scour_mse` and the leakage
columns against `s0`/`s11`. *(Awaiting the regenerated chain.)*

**5.6 `s14_prof` — the roughness rung (the interesting one).** *(Direction from the deprecated
L100 pilot; magnitudes pending.)* Injecting **per-passage** roughness collapsed **all six
sprung channels** to predict-the-mean (scour MSE ≈ label variance; localisation ≈ chance)
while **wheel (unsprung) channels retained skill** (pair scour RMSE ~11 pp, localisation
0.744) — the clean-track ranking **inverted**. Physics: suspension filtering + car-body
resonance mask the quasi-static scour signature under strong roughness, while the unsprung
mass traces profile + deflection directly (§6). **Mixed-pair pilot (budget-matched):**
**FrontBogie_Vert + Wheel1 beat Wheel1 + Wheel2 by 17% scour MSE** (105 vs 126) and +5 pp
localisation, *despite FrontBogie_Vert alone being collapse-level* — the TSD/two-axle residual
(profile-reference) mechanism; CarBody + Wheel1 was **worse** than wheel+wheel (the sprung
partner must sit low in the suspension chain). **The key open question:** the pilot used a
physically indefensible *per-passage* redraw; with the corrected *per-state* profile the
sprung channels may well survive. `s14_prof` is the clean test, and either outcome is a
result — collapse confirms a real deployment limit; survival shows the collapse was an artefact
of over-randomisation and localises the blame precisely.

**5.7 `s15`–`s16` — do the rail and the train interfere?** The maintainer-facing question,
answered separately for track-layer damage and for wheel damage so any degradation is
attributable to one or the other.

**5.8 Scale-up (`s21`–`s23`).** Does the pair still suffice with 3 piers and ~100 m? Is the
middle pier harder? How much do the EOVs inflate per-pier MSE at scale?

**5.9 Summary tables.** (i) Architecture (3-arm). (ii) Sensor economy. (iii) **The ladder
table** — one row per rung with scour MSE, localisation, bearing MSE and leakage, so the
marginal cost of each factor is read directly down a column. Pull exact numbers from
`results/<stage>_summary/leaderboard_median.csv` at write-time.

---

## 6. Discussion

- **Why multi-rate pooling helps** — tie N-HiTS pooling to the two-timescale physics; contrast
  a plain CNN / CWT; explain why S2V and LSTM degrade strong channels (added capacity/variance
  with no matching structure). The paper's intellectual core.
- **The suspension-chain reading of sensor economy** — under clean track, sprung channels win
  (they integrate the deflection basin); under strong roughness, the same filtering that helps
  now *removes* the quasi-static signal, and unsprung channels win. The best sensor set is
  therefore **regime-dependent**, and a **mixed unsprung+sprung pair** is a robust hedge
  (profile reference + inertial response). This reframes "which sensors" as a fusion question.
- **Localise + disentangle from one pass** — operational value: a lightly instrumented
  in-service train (2–3 sensors) resolves *which* pier and separates scour from a seized
  bearing, with a bounded, measured false-scour leakage.
- **Robustness vs point estimates** — the collapse-rate story; fragile single-sensor configs.
- **Limitations** — simulation-only (no field data); 2-D vertical model (no lateral/torsion/
  pier-tilt; SSI/soil not modelled beyond a support spring); bearing **range** covers ~0–31%
  fixity; profile modelled as FRA-4 domain randomisation (not measured-track evolution); the
  deck-mass/frequency sanity-check (§4.1). State each plainly.
- **Forward link** — multi-foundation dependent scour, flood/shock evolution, and the
  value-of-information **digital twin** (Paper 2; DT explainer in `docs/dt_torzoni_explainer.md`).

---

## 7. Conclusion
Restate the six contributions with headline numbers (localisation 0.99; 2 ≈ 8; scour cost of
bearing 0.76→1.80; leakage 2.4 pp; roughness inverts the ranking and mixed-pair fusion
recovers 17%). One sentence on Paper 2.

---

## Figure inventory (existing → paper)

| Paper fig | Source | Status |
|---|---|---|
| Concept: TTBI + scour + bearing | scour.png, TTBI_Cantero.png (+ new bearing schematic) | keep/extend |
| Example signals (sprung vs unsprung) | signals.png, pitch_signals.png | keep |
| Architecture schematic (modular blocks) | CNN.png | keep |
| F1 Architecture ablation (3-arm) | `results/Stage0/.../stage0_multiscour_summary/` + single-scour figs | regenerate |
| F2 Sensor economy (single→LOO→pair→sweep) | single-scour `ablation_analysis/figures/` | keep (font later) |
| F3 Stage-0 parity + per-pier / localisation | `stage0_multiscour_summary/parity_best.png` | keep |
| F4 Stage-1 parity (4 heads) + leakage | `stage1_bearing_summary/parity_best.png`, `disentanglement.csv` | keep |
| F5 Bearing k_r sensitivity sweep | `results/bearing_sensitivity/` | keep (methods fig) |
| F6 Roughness collapse + mixed-pair bar | Stage-2 pilot `stage2_4span_L100pilot_summary/` | REDO after regen |
| (opt) Champion confusion / ordinal error | `plotting/confusion.py` | generate if wanted |

---

## Open items before submission (verify list)

- **⚠ Deck mass / fundamental frequency** (§4.1) — confirm the as-built fundamental is physical;
  report it + effective spans (L40→19.8/20.1; L60→20.1/19.8/20.1; L99.6→4×24.9).
- **⚠ k_v0** healthy support stiffness value + units; the scour k_v(d) law location in code.
- **⚠ EOV ranges** — noise model (now load-time; state the domain caveat), temperature law,
  vehicle-property count, speed range — against the FINAL regenerated datasets.
- **⚠ Optuna** — trials/epochs/pruner for each study; CWT scale count; PAA n_segments = 512.
- Verify each Fernandes Table-1 cell (classes, sensor sets) against `papers/`.
- Re-pull every §5 number from the `*_summary/` CSVs at write-time; mark pilot numbers as pilot.
- Confirm the "O'Brien-calibrated" vehicle and Zhai track citations.
- Decide venue → port to the journal LaTeX template; redraw figures to publication fonts.

## Reference shortlist
Cantero (2022, VEqMon2D; 2-D TTBI); Kamariotis et al. (2024); Fernandes et al. (2024, 2025,
2026a, 2026b); Khan et al. (2022, continuous bearing k_r); Zhai et al. (track); Prendergast &
Gavin (scour–frequency); Locke et al. (2020, fixed profile); Sarwar & Cantero / NuBe-DBBM;
Corbally & Malekjafarian; OBrien/Keenahan (two-axle/TSD residual); axle-box-acceleration
strand; EN 13848-2 (track geometry repeatability); FRA track classes; HEC-18; Lamb et al.
(2019). Full PDFs / the roughness deep-research report in `papers/`.
