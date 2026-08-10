# Track-layer damage EOV — sampling specification (from NotebookLM deep research)

> **ADDENDUM 2026-08-01 — SECOND-PASS (ADVERSARIAL) SOURCE AUDIT.** Every
> source behind a surviving anchor was re-read page-anchored and then re-read
> again by an independent pass instructed to refute the first verdict. **No
> campaign value changes.** Five anchors weaken; record them as follows.
>
> 1. **Wet-ballast multipliers (Wangtawesap 2023) — weaker than recorded.**
>    The measured full-submergence stiffness factor is **0.67**, which is
>    *below* the implemented band [0.7, 0.9], and the ×4 damping figure is
>    the **per-zone** coefficient: the **condensed** sleeper–ballast dashpot
>    (the quantity a lumped model actually scales) rises only ×2.8. The
>    condensed-damping rise is gradual with **no stated threshold**
>    (Table 4.17: ≈unchanged at 5–10 cm, +28% at 15 cm, ×2.8 only at full
>    35 cm submergence) — the earlier "no rise below ~15 cm" note is
>    **withdrawn** (2026-08-02). The scope is flooded
>    **clean** ballast; wet-and-fouled use remains our extrapolation.
> 2. **Dry-fouling damping (Esmaeili 2017) — direction only.** Damping falls
>    up to 67% (≈×0.33); the implemented [0.4, 0.8] is milder and was not
>    estimated from it. The dry **stiffness** band [1.2, 2.0] is contradicted
>    (measured mild softening: the quoted 54.7 → 46.6 kN/mm pair in Sec. 3.4's
>    text is the 5 wt% tire-derived-aggregate case; the plain-ballast change
>    ≈82 → 64 kN/mm, ×0.78, is back-calculated from the text's stated
>    33.5%/27.6% TDA-induced drops, and Fig. 14 prints those bars as
>    0.08227 → 0.06441 under an MN/m caption — a ×1000 unit inconsistency
>    with the prose's kN/mm, disclosed not resolved — correction 2026-08-02:
>    an earlier
>    note claiming no plain-ballast pair is reported is **withdrawn**) and is
>    retained knowingly.
> 3. **>50% poor support — now cited to the primary.** The Augustin et al.
>    (2003) chapter was located inside the local Popp & Schiehlen volume,
>    **pp. 317–336, statement on p. 330**, and reads as engineering
>    experience rather than a survey. The "as cited in" chain via
>    RAILCON/Li & Sun is no longer needed. Kitahara et al. found **no**
>    hanging sleepers in their own field campaign.
> 4. **Five is not the critical count (Shi et al. 2024).** The study models
>    0–5 consecutively voided sleepers at 80 km/h; its wheel–rail
>    contact-force comparison reports four of those cases (0, 1, 3, 5) and is
>    non-monotonic, peaking at **three**: 104.5 kN maximum wheel–rail force
>    at three, 90.8 kN at five, 66.1 kN fully supported. Do not present
>    DU{1,…,5} as bracketing a worst case.
> 5. **P(OOR)=0.30 is weaker than "unsupported" — it is unsupportable.** The
>    RIVAS 5%/7% figures come from two UK classes whose tread problems the
>    report describes, with **no stated sampling frame or selection
>    rationale**, so they are not even a fleet rate, let alone an any-severity
>    rate. Reported radial amplitudes reach ~0.9 mm for **developed
>    polygonization** (Iwnicki) and up to ~2.5 mm for **general OOR removed at
>    the wheel lathe** (RIVAS) — two distinct scopes, not one range — against
>    our 10–120 µm clip, and **no source reports an amplitude distribution**.
>
> Transition-zone geometry: Siahkouhi et al. (2025) supply 4–8× maintenance
> frequency (itself cited from Wang & Markine 2018) but **no cone/influence
> length**; the ±15 m window and the ×3 densities have no source basis and
> are author-chosen. FRA RR 22-32's "up to 15×" saturated settlement is
> repeated from Wilk et al., not measured by FRA.
>
> Full verdict table, including the citation traps (two different Shi papers;
> Kitahara's defective Augustin reference; the Sainz-Aja spelling), is in
> `paper1/MISSING_PRIMARY_SOURCES.md` under "SEMANTIC CLOSURE PASS".
> `paper1/sections/numerical_simulation.tex` now classifies each value
> individually against these verdicts.

> **UPDATE 2026-07-15 — BASELINE COUNTS IMPLEMENTED; evidence limits retained
> below.** Second deep-research pass (Gemini, too large for NotebookLM) →
> `papers/Track Defect Prevalence Data Search.{md,docx}`; prompt =
> `docs/deep_research_prompt_track_counts.md`. **Direct-PDF correction
> 2026-07-31:** the report's gap taxonomy, 10–20% exceedance fraction,
> 5–10% "impactfully unsupported" prevalence and ~85% wheel-load statement
> were not supported by the cited primaries and must not be used to calibrate
> the campaign. The defensible anchors are narrower: Kitahara reports that
> poor support is often widespread while citing Augustin; Lundqvist and
> Dahlberg find, for one modeled 1 mm gap, +70% adjacent sleeper–ballast force
> and +40% adjacent-sleeper displacement; RAILCON separately summarizes
> wheel–rail-force increases up to 80% across prior studies. None identifies a
> population count distribution. The λ=3.0 law remains an author-chosen stress
> prior, and its 5.4% arithmetic share is conditional on an assumed mean group
> size of three, not a field-prevalence estimate. Active values live in
> `scour_MATLAB/+ttbi/campaign_setup.m`.
> **Reproducible MC verification restored 2026-07-28 (upgraded same day per
> the R11 re-audit):** the reviewed `check_track_prior_stats.py` (seed
> 20260728, 200k draws; live campaign-setup drift guard over every prior constant it
> uses; fully-contained sampling; union/unique estimates inside pinned
> assertions) verifies: RAW unsupported-sleeper incidence 5.4%
> (**overlap-ignoring arithmetic**) alongside an EFFECTIVE unique
> unsupported share ≈5.25% (**homogeneous-placement MC** — production
> transition/fouling weights change overlap, not totals); RAW fouled-length
> fraction 15% with homogeneous union ≈13.8–13.9%; ≈3.33 failed
> pads/100 m; and the prior-level fouling-rate sensitivity
> λ∈{0.6, 1.2, 2.4}/100 m ⇒ raw {7.5%, 15%, 30%} (homogeneous union
> ≈{7.2%, 13.9%, 25.6%}). All derived shares quoted here are prior-level
> arithmetic or homogeneous MC values, NOT implemented effective
> prevalence. The checker is not the production sampler, and a
> response-level λ sweep remains out of scope for the frozen R11 campaign:
>
> | Quantity | OLD (un-cited) | CURRENT IMPLEMENTED PRIOR |
> |---|---|---|
> | hanging-sleeper groups | DU{0..3} per state (fixed) | **Poisson λ = 3.0 per 100 m**, window-scaled author-chosen stress prior |
> | ballast fouled patches | DU{0..2} per state (fixed) | **Poisson λ = 1.2 per 100 m**, window-scaled author-chosen design rate; source context below |
> | failed pads | an alleged 0.5%/yr rate was misattributed to Williams et al. (2014) | **`p=0.02`/pad** = implemented author-chosen snapshot stress prior; no audited primary established the rate or conversion ⇒ arithmetic expectation ~3–4 positions/100 m |
> | pad service-condition field | proposed sleeper-wise ARIMA aging process | **one state-wise global Weibull stiffness multiplier and one global uniform damping multiplier**; no spatial or temporal aging law is implemented |
> | fouling ↔ voiding | independent (a documented vulnerability) | **coupled** by author-chosen **×3 inside a fouled patch**, implemented as 3:1 inside vs outside placement odds (the literature motivates the mechanism, not the exact odds) |
> | ballast near bridge | uniform | **author-chosen ×3 within 20 m of an abutment**, motivated by qualitative transition vulnerability and maintenance context rather than a measured defect-density ratio |
> | counts vs window length | fixed count regardless of length | **rates per 100 m, scaled by the author-chosen descriptor-window convention** (30 m approach + deck + 30 m exit ⇒ 120 m at L60, 159.6 m at L99.6; the 30 m margins have no source basis and carry the same evidentiary status as the rates they scale) — the old fixed draw was itself an error |
>
> **Historical λ rationale — rejected by the direct-PDF audit.** The research
> report proposed λ=2.0–3.0 and separately asserted an unsupported 5–10%
> "impactful" prevalence. Their arithmetic intersection was previously used to
> select λ=3.0. Because that prevalence target has no verified primary basis,
> the derivation does not calibrate λ. The implemented value remains a frozen
> author-chosen stress prior; 3×3/167=5.4% is only arithmetic under an assumed
> mean group size, not an empirical target or validation.
>
> **Historical 2026-07-19 NotebookLM inference — superseded by the
> 2026-07-31 direct-source audit.** The block below records why the frozen
> value was retained, but its heterogeneous percentages are context, not a
> calibration of the implemented Poisson law. The "FI>30 on 10–20% of route
> length" **does not exist as a citable network constant**; the Slideshare
> anchor is retired. Context assembled at that time (λ = 1.2 kept, no protocol
> change):
> * **Citable extent envelope**: Norway national defect-rate proxy **~9.4%** (3,400 defects
>   / 36,100 inspections 2014–2024; Husøy, Lau, Løhren & Hoff 2024, Civil-Comp, DOI
>   10.4203/ccc.7.24.3 — a defect *rate*, order-of-magnitude proxy only, not % length) →
>   the only continuous regional-line GPR survey: **66% fouled-or-worse at Selig FI≥20,
>   12% highly fouled FI≥40** over 17 km (Sadeghi et al. 2018, *J. Applied Geophysics* 151,
>   DOI 10.1016/j.jappgeo.2018.02.020) → US heavy-haul "majority fouled" (FRA 2017 RIVIT
>   deck — non-peer-reviewed, cite only as corroboration). Our **15% of length** sits inside
>   this envelope, close to the regional survey's 12% highly-fouled fraction.
> * **Threshold semantics corrected**: Selig's own classes are fouled 20≤FI<40, highly
>   fouled ≥40; "FI>30 = highly fouled" is the **Zetica/FRA GPR (BFI) schema** (FRA
>   DOT/FRA/ORD-22/01, Table 5) and FI≈30 is the **functional drainage-loss / end-of-life
>   limit** (Chrismer & Hyslip 2018, AREMA — industry paper). Paper wording: our patches
>   represent ballast *at or beyond the FI≈30 drainage-loss limit* — between Selig "fouled"
>   and "highly fouled".
> * **Patch-length bounds now have source context**: FRA field records report
>   degraded-location extents from 5 ft or less up to 110 ft (about ≤1.5 m to
>   33.5 m) — an open-ended range, and specifically among sites crossing the
>   track class-5 profile safety limit (DOT/FRA/ORD-22/01 §4.3, Fig. 22,
>   printed p. 34) — and Guo
>   et al. average their GPR fouling indicator over 2.4 m (four-sleeper)
>   windows to suppress fluctuation of the 5 cm channel values, a smoothing
>   choice rather than a stated minimum resolvable feature (2023, *Int. J.
>   Rail Transportation*, DOI
>   10.1080/23248378.2022.2064346). Neither source fits the campaign's
>   U(5,20) m distribution; that distribution remains author-chosen.
> * **Secondary-line vulnerability now citable**: ballast design life 25 yr (main) vs up to
>   50 yr (secondary rural) — Musgrave 2024 (Network Rail perspective, DOI
>   10.17265/2328-2142/2024.05.004).
> * **Planned sensitivity (not implemented or run):** a future
>   λ ∈ {0.6, 1.2, 2.4} sweep would span about 7.5–30% of length. The current
>   campaign implements λ=1.2 only; report it as an author-chosen design point,
>   not a measured constant and not generically "conservative" without an
>   outcome-specific sensitivity result.
> The historical report called its own table entries inferences from adjacent
> anchors. The direct-PDF audit below supersedes that blanket characterization:
> several exact distributions and odds are author-chosen rather than derived.
>
> **Crack (secondary question) — modeling priors retained, location
> support-weighted.** `P(crack)=0.25` is an author-chosen modeling prior. A
> historical research report proposed a **20–30%** context range, but that is
> not a directly audited prevalence for this campaign population and does not
> calibrate the Bernoulli law. The location prior weights hogging regions because the
> deck top fibre is in tension over internal supports and is exposed to
> runoff/chlorides. Eurocode 4's cracked-section treatment motivates using a
> support-region window; it does **not** supply occurrence odds. The implemented
> **4:1 hogging:sagging weight is therefore an explicit design prior**, with
> locations placed within **±17.5% of a span** about the chosen section. The
> **±17.5% window itself, the U(0.05, 0.30) EI-loss severity band, and the
> global 10–90% bridge-length clamp are equally author-chosen design values**
> — none is fitted to data or supplied by any source.

Transcribed 2026-07-09 from the NotebookLM answer (notebook "Probabilistic Graphical
Models for Predictive Digital Twins at Scale", 106 sources incl. the Track-Layer Deep
Research report; answer also saved as a note in the notebook). Items the model itself
marked *(Extrapolation)* are flagged.

> **PRIMARY-SOURCE AUDIT 2026-07-28 (supersedes the "CITED (solid)" labels
> below where they conflict).** The primary PDFs behind the deep-research
> citations were fetched and read directly (page-anchored; verdict table in
> `paper1/MISSING_PRIMARY_SOURCES.md`). Outcome: **wet-ballast multipliers**
> are genuinely stated (Wangtawesap 2023 Chulalongkorn thesis: stiffness
> ×~0.7 flooded, damping ×1.5–4.0 — flooded clean ballast, not fouled+wet);
> **A broad “usually over 50% poorly/completely unsupported” statement** appears
> in Kitahara et al. 2024 citing Augustin et al.; it is mechanism/occurrence
> context, not a measured network prevalence usable to calibrate the count law.
> **70% contact-force increase** is Lundqvist & Dahlberg 2005 at a
> 1 mm gap specifically; **dry-fouling damping reduction** is supported by
> Esmaeili 2017 (factors ~0.33–0.56). DEMOTED to author-chosen/synthesized
> design priors (numbers absent from the primaries): pad
> **Weibull(1.8, 2.2)** and the aging ranges (no Weibull in any pad paper;
> Woo & Park fit an ARRHENIUS lifetime model; Oregui's field-worn pads are
> ≈40% SOFTER in complex modulus at the 12–18 kN preloads, and more than
> half softer at 6 kN — aging direction contested; large Sainz-Aja factors are
> temperature/toe-load/frequency effects), **dry-fouling stiffness increase
> [1.2, 2.0]** (Esmaeili's dry-sand box tests show a mild DECREASE),
> **void-depth 0.5–3.0 mm lognormal** (no range/distribution in any paper;
> Sysyn: distribution "not known"; moot in R11 — binary removal), and the
> **OOR triplet** (P=0.30 unsupported — RIVAS reports 5–7% of wheels above
> 1.0 mm; 10–120 µm amplitudes are far below reported developed-OOR levels
> ≥0.5 mm, defensible only as maintained-wheel modeling; orders 1–5
> dominance is fleet-specific: ICE 2–3, Stockholm order 3 in ~60% of
> wheels, Chinese heavy-haul 1–3, other fleets high-order). The campaign
> values remain the frozen registered design; only their evidentiary
> labels change.

**HISTORICAL VERIFICATION NOTE (2026-07-09; superseded by the direct-PDF
audit above).** This NotebookLM mapping originally labelled the void-depth
lognormal, dry-stiffness band, pad Weibull/ranges, and other values as "CITED
(solid)". That label was not supported when the primary PDFs were read and
must not be used. The surviving primary anchors are the wet-ballast proxy,
dry-fouling damping direction/range overlap, >50% poor support, the 1 mm-gap
contact-force example, patch-length context, qualitative transition clustering,
and qualitative pad-condition evidence. The alleged annual 0.5% pad-incidence
anchor was not present in Williams et al. (2014) or the other audited pad and
fastening sources. ARIMA(5,1,0) and a maximum of three
consecutive failed pads were recommendations in the source report, **not
implemented campaign mechanisms**. The following were extrapolations even in
the historical note: 1-3 sleeper groups /100 m;
the exact DU{1,…,5} group-size law (RAILCON reports 1–4 m occurrence,
summarizes 1–4-sleeper simulations, and cites correlations up to six; it does
not fit the campaign distribution or identify five as a universal limit);
1-2 ballast patches /100 m; patch upper bound 25 m (cited 20 m); about 3.3 failed
pads/100 m under the implemented `p=0.02` snapshot prior (arithmetic); 2 m
clustering near joints. Corrections are applied in the table below.

## Sampling specification (per 100 m of track; sleeper spacing 0.6 m ⇒ ~167 sleepers)

### (a) Hanging / unsupported sleepers
- **Representation:** unilateral (non-linear, bilinear) spring contact — the vertical
  ballast reaction F_b,i drops to 0 while the relative displacement ≤ void depth g_v.
  *Linearised fallback for our per-passage linear solver: zeroed/strongly-reduced
  ballast spring under the affected sleepers.*
- **Severity:** void depth g_v = 0.5–3.0 mm (max boundary 5.0 mm);
  distribution **Lognormal**: ln(g_v) ~ N(−0.2, 0.4).
  ⚠ **NOT IMPLEMENTED (audit 2026-07-17):** the code stores only
  `[x_start, n_consec]` and applies the same near-zero support (×1e-6) to every
  group — i.e. every hanging group behaves FULLY voided regardless of depth.
  The g_v distribution above is a spec-only refinement. The paper must describe
  the implemented model as a BINARY removed-support linearisation (severity
  gradation and gap-closure impact both absent); do not claim g_v sampling.
  If g_v is ever implemented it must actually enter the mechanics (a gap state
  or depth-dependent stiffness), not just the log.
- **Count actually implemented:** Poisson with rate **λ=3.0 per 100 m**,
  scaled by the modeled window. This is an untruncated non-negative count:
  zero and values above three are possible. The rate is an author-chosen
  stress prior; it is not a field-observed count distribution.
- **Clustering design:** group size = **Discrete Uniform 1–5 consecutive
  sleepers**. RAILCON contextualizes consecutive occurrence over 1–4 m,
  summarizes simulations with one to four sleepers, and cites correlation work
  up to six; it does not fit a discrete-uniform population law or establish
  five as a universal critical limit. The campaign distribution is
  author-chosen. Values up to 10 were a historical extrapolation only.
- **Location design:** sources motivate elevated transition-zone incidence,
  but the exact ±15 m window and sampling probabilities are author-chosen.
  NOTE the interaction: this puts
  track damage adjacent to the abutment BEARING targets — watch bearing-head leakage.
- **Transition-zone selection probability actually implemented:** each
  hanging-group placement proposal is drawn from a ±15 m abutment transition
  zone with probability p_transition = 0.6 (the two abutments equiprobable),
  and uniformly over the modelled window otherwise, before the fouling↔voiding
  acceptance step of section (b) is applied. The 0.6 value
  (`hang_p_transition`, flagged as an assumption in
  `scour_MATLAB/+ttbi/campaign_setup.m`) is an
  author-chosen modeling assumption quantifying the cited qualitative
  density-spike rule above; it is not a cited constant. A proposed start is
  accepted only if all 1--5 sampled sleepers fit on the realized sleeper
  lattice inside the modeled window; the stored count is exact and is never
  truncated at the exit boundary.

### (b) Ballast degradation & fouling
- **Representation:** state-dependent multipliers on nominal vertical stiffness (η_k)
  and viscous damping (η_c) of the ballast layer.
- **Severity:** dry/high-stiffness scenario: η_k ∈ [1.2, 2.0], η_c ∈ [0.4, 0.8];
  wet/saturated proxy: η_k ∈ [0.7, 0.9], η_c ∈ [1.5, 4.0]; continuous Uniform within
  the state's range. The dry-stiffness band is an author-chosen stress scenario,
  not an empirically calibrated dry-fouling law: the audited dry-sand box test
  softened mildly, and the direction depends on fouling material/compaction.
  The wet band is borrowed from flooded clean ballast and is therefore a proxy,
  not a measured wet-and-fouled joint distribution. Sample the registered
  scenario first, then its two multipliers.
- **Wet-vs-dry state selection actually implemented:** each sampled patch
  independently draws its state first — wet/saturated with probability
  p_wet = 0.5, dry/compacted-fouled otherwise — then draws η_k and η_c
  uniformly from that state's ranges above. The 0.5 value (`ballast_p_wet`
  in `scour_MATLAB/+ttbi/campaign_setup.m`) is an author-chosen modeling
  assumption: no citable field
  prevalence for the wet fraction of fouled patches was identified.
- **Count actually implemented:** Poisson with rate **λ=1.2 per 100 m**,
  scaled by the modeled window. It is untruncated, so zero and values above
  two are possible. The rate is an author-chosen design point compared against
  source-reported extent context; it is not a fitted field count law.
- **Geometry actually implemented:** each patch is a rectangular interval with
  author-chosen length **U(5, 20) m** within source-contextualized bounds. Its
  start coordinate is drawn by rejection sampling with an evidence-motivated,
  author-chosen ×3 density near bridge transitions. No Gaussian/exponential random
  field or correlation-length parameter is implemented. The cited
  θ_x=3–15 m range is literature context for a possible future spatial-field
  model and must not be attributed to this campaign.
- **Overlap rule (audit r3, 2026-07-22):** where Poisson-placed patches overlap, the
  patch with the **largest stiffness deviation |log η_k| governs and supplies BOTH
  η_k and η_c** (a sleeper is either predominantly wet- or dry-fouled). The earlier
  implementation MULTIPLIED stacked draws, which could leave the documented per-patch
  bands (up to η_k ≈ 4 dry-on-dry); fixed in B54 + the Python mirror.
- **Coupling correction (audit r3, 2026-07-22):** the fouling↔voiding coupling is
  enforced by rejection sampling with acceptance ODDS **3:1 inside vs outside a
  fouled patch** — an author-chosen quantification of a qualitative physical
  association. The pre-fix weights compounded to 9:1
  (mult²); datasets generated with the 9:1 draw (s15/s16/s23 before this date)
  must be regenerated.

### (c) Rail-pad service-condition variability and failure
- **Representation:** one persistent per-state service-condition multiplier on
  pad stiffness (χ_pad) and one on damping (β_pad). There is no time axis,
  monotonic trajectory, fatigue accumulation, or calibrated aging law, so these
  scalars must not be presented as progressive deterioration. A failed pad is
  represented by setting **both stiffness and viscous damping multipliers to
  \(10^{-6}\)**, i.e. a numerically regularized removed-pad approximation, not
  an exactly zero-stiffness-only spring.
- **Severity:** service-condition scenario χ_pad ∈ [1.0, 3.5] with
  **Weibull(λ = 1.8, k = 2.2)**; damping β_pad ∈ [0.8, 1.2]; and an
  independent failed-pad event with per-position snapshot probability
  **p = 0.02**. In the solver, "failed" means both stiffness and damping
  multipliers are `1e-6`, not exact zero.
  The Weibull family/parameters and multiplier bands are author-chosen design
  priors, not a fitted physical aging law. The primary literature contains
  competing directions (field-worn softening versus fatigue/condition-related
  stiffening), and none of the audited pad papers specifies this Weibull.
  **Evidence boundary:** the previously claimed 0.5% annual incidence was a
  source-mapping error; Williams et al. (2014) does not report it, and no
  audited primary in the local library supplied an equivalent rate. The
  `p=0.02` campaign value is therefore an author-chosen snapshot stress prior,
  not an inferred standing prevalence or measured snapshot estimate.
- **Spatial rule actually implemented:** one stiffness service-condition multiplier and one
  damping multiplier are drawn per persistent state and applied globally to all
  pads. Independently, every unique 0.6-m sleeper/pad lattice position in the
  bridge-local track window receives one Bernoulli(`p=0.02`) failure draw. Failed
  positions are stored without replacement and mapped one-to-one to the same
  sleeper lattice in `B54_TrackVectors.m`. Off-lattice, duplicate, nonfinite,
  or out-of-modeled-sleeper-domain descriptors fail before assembly; no
  nearest-sleeper snapping is permitted at this boundary.
- **Count:** the expectation is approximately **3.33** failed positions per
  100 m *((100/0.6) × 0.02 Bernoulli = 3.333)* — the value the reviewed
  `check_track_prior_stats.py` pins. Do not quote 3.34, which comes from
  rounding the lattice to 167 positions before multiplying. There is no imposed cap on consecutive failures.
  Runs of adjacent failures can arise only from the independent Bernoulli draws.
- **Explicit limitation:** the source report's proposed sleeper-wise ARIMA(5,1,0)
  field, joint-adjacent clustering, and maximum run length of three were not
  implemented. They must not be attributed to the campaign or inferred from
  the global service-condition scalar.

## (i) Channel / frequency sensitivity
- **Moving-coordinate rail-field channels:** legacy channels 3/4 are
  (N(x_w)^T A_{rail}), not unsprung/axle-box acceleration. Pad state can affect
  rail response and the rail pinned-pinned resonance is often reported around
  800–1200 Hz, but the campaign's 1 kHz solver sample rate (500 Hz Nyquist)
  cannot resolve it or support a claim over 500–3000 Hz. Any pad sensitivity
  must be confined to the resolved band and cannot be presented as measured
  axle-box sensitivity.
- **Sprung (bogie/car-body — historically selected channels):** suspension
  low-passes above ~30 Hz, so reduced sensitivity to rail-pad condition is the
  historical hypothesis to test, not an assumed zero response. In current
  terminology this is sensitivity to the pad service-condition scalar. These channels capture
  resolved bounce/pitch modes that may be excited by the implemented localized
  support-property changes. The campaign does not model nonlinear gap closure
  or a ballast-settlement profile.

## (ii) Confounding hypothesis — what this design can test
Localized support changes can excite car-body bounce/pitch content in the same
low-frequency band as modeled bridge changes, so a learner trained without
track defects may produce false scour response. This is a motivating
hypothesis, not a result. The implemented EOV changes support stiffness and
damping (and approximates hanging sleepers by support removal); it does
**not** generate permanent settlement depressions in the rail profile.

**Separable?** Not established by the sources — separability is a modeling
expectation to be evaluated, not a guaranteed property. The sources motivate
two mitigation routes:
1. **Domain randomization** — inject these damages into the TTBI training set as
   EOVs (exactly our crack/profile mechanism), then evaluate whether scour
   performance remains robust to localized track anomalies; this does not
   guarantee mathematical invariance;
2. Signal decoupling (Augmented Kalman filter / SMC "apparent profile" isolation /
   band-stop around the sleeper-passing frequency f = v/0.6) — the non-ML alternative.

**Implication for us:** the support-property pathway is absent from stationary
`psd_fra` roughness randomization, so the track-EOV stage adds a distinct
modeled perturbation. It does not, by itself, cover the profile-mediated
settlement pathway described in field literature.

## Caution — the NotebookLM "how does N-HiTS distinguish them" answer
A follow-up NotebookLM answer explained the separation via the ORIGINAL N-HiTS
architecture (Challu et al.: stacked blocks, backcast-residual subtraction, hierarchical
top-down interpolation). **Our model is NOT that**: it is a CNN + `MultiRatePooling1D`
(parallel max-pools at rates 1/2/4, concatenated — core/models.py); no stacked blocks,
no backcasts. Usable claims for the paper: (1) multi-rate pooling gives the head
simultaneous coarse/fine views, favouring long-wavelength global-deflection content;
(2) domain randomization may improve nuisance robustness — a hypothesis this
stage tests (success criterion: flat scour MSE + low false-alarm rate under
track EOVs), not an invariance guarantee. Do NOT quote the
hierarchical-interpolation mechanism.

## Nominal track-property scope (2026-08-03 source check)

Zhai et al. (2004), Eq. (5) and Table 1, define one 531.4 kg independent
vibrating ballast mass at each rail support point. The inherited B54 bridge
topology instead has no on-bridge ballast DOF and condenses one retained value
onto each support-aligned deck DOF; the corrected total no longer changes with
bridge mesh density. Deck attachment and full endpoint lumps are inherited
model-form/domain-partition choices, not source-supplied bridge rules. Zhai
also couples adjacent ballast masses through \(K_w,C_w\); the repository omits
that shear branch everywhere. Table 1 states the nominal track quantities per
rail seat. The inherited planar
property function doubles rail and sleeper terms but not pad, ballast, or
sub-ballast terms, so the intended one-seat versus two-rail scaling is still an
explicit model-validity question. Zhai's set also uses 0.545 m support spacing,
while the generator retains the source \(M_b\), \(K_b\), and \(K_f\) values at
0.600 m without re-evaluating their spacing-dependent expressions; this is a
separate proxy-informed hybrid transfer requiring a benchmark or prospective
sensitivity. The damping values are not treated as spacing-derived. The EOV
multipliers in this specification operate on the retained nominal convention;
they resolve none of the topology, scaling, or spacing baseline questions.
The separate 0.1% rail Rayleigh target is likewise inherited author-chosen,
not a Zhai Table 1 value; it is a damping-model sensitivity rather than a track
damage prior.

## Historical implementation-planning notes

This section records the pre-stage design notes. Current code and the audited
campaign contract supersede any future-tense statement below.
- Our track layers (`TrackProp_Zhai…`) are currently SCALAR per-layer properties applied
  uniformly to every sleeper; per-sleeper multipliers (needed for all three damages)
  require threading per-sleeper stiffness vectors into the track/model matrices
  (B51/B54 + Python b51/b54). This is the main implementation cost of the stage.
- The bilinear gap contact (true hanging sleeper) is nonlinear; start with the
  linearised zero/reduced-stiffness version (standard in the literature for moderate
  voids) and note the simplification.
- Current campaign policy samples persistent crack/profile/track conditions per
  state and logs the draws; they are nuisances, not labels. Vehicle-side wheel
  conditions remain per-passage.
- Keep MATLAB and Python (TTBI_2D) implementations mirrored — parity audited 2026-07-09.
