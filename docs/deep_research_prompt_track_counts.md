# Deep-research prompt — track-defect COUNTS and PREVALENCE per 100 m

> **HISTORICAL RESEARCH PROMPT.** Current R11 evaluates empirical robustness
> under the simulated nuisance distribution; it does not assume or claim
> mathematical invariance.

Paste the block below into NotebookLM Deep Research (notebook: "Probabilistic Graphical
Models for Predictive Digital Twins at Scale" — it already holds the Track-Layer Deep
Research report and ~106 sources). Written 2026-07-14.

**Why this prompt exists.** `docs/track_eov_sampling_spec.md` anchored the track-EOV
*severities* and *cluster sizes* to literature, but explicitly flagged the **counts per
100 m** as *(Extrapolation)*: "1–3 sleeper groups /100 m", "1–2 ballast patches /100 m",
"0.83 failed pads/100 m (arithmetic)". A 2026-07-14 code fix also exposed a second gap:
the counts previously could never be **zero**, so every simulated state carried track
damage — no sound-track case existed. The *prevalence* (how much of a real line is sound)
is therefore also un-anchored. These are the last un-cited numbers in the damage model
(the wheel side is fully verified), and everything else in the campaign is literature-
anchored — so this closes the loop. Verify per-number with quotes, as we did for the
wheel-flat and track-EOV specs.

---

**Topic: How much of an operating ballasted railway line is actually defective? Counts,
spatial extent, and prevalence (including the sound-track case) of ballast fouling/
degradation, unsupported ("hanging") sleepers, and failed rail pads, per 100 m of track**

**Context.** We simulate a 2-D vertical train–track–bridge interaction model (TTB-2D,
Zhai-type track: rail beam on discrete pad springs, sleeper masses, ballast spring–damper
layer at **0.6 m sleeper spacing ⇒ ~167 sleepers per 100 m**). Track-layer defects are
injected as **randomized training nuisances** (domain randomization) for a machine-learning
drive-by bridge-**scour** detector: the network must be *invariant* to them, and never
estimates them. The modelled track is a **regional/secondary ballasted line at 70–90 km/h**,
comprising a ~30 m approach, the bridge (60 m or 99.6 m), and a ~30 m exit.

We already have CITED (do not re-derive): void depth 0.5–3.0 mm, ln(g_v) ~ N(−0.2, 0.4);
unsupported-sleeper group size = Discrete Uniform **1–5 consecutive** (5 = wheel-load
critical limit); ballast fouled-state multipliers (dry η_k ∈ [1.2,2.0], η_c ∈ [0.4,0.8];
wet η_k ∈ [0.7,0.9], η_c ∈ [1.5,4.0]); fouled-patch length **U(5, 20) m**; 1-D random-field
correlation length θ_x ≈ 3–15 m (typ. 10); pad aging χ ∈ [1.0,3.5] Weibull(1.8, 2.2);
pad failure k→0 at **P = 0.5 % of fastening positions PER YEAR**; hanging-sleeper density
**spikes within 15 m of bridge transitions**; max 3 consecutive failed pads.

**What we are missing — and our current (un-cited) placeholders:**
- ballast fouled patches per 100 m = **Discrete Uniform {0, 1, 2}** ⇒ P(no patch) = 1/3
- unsupported-sleeper groups per 100 m = **Discrete Uniform {0, 1, 2, 3}** ⇒ P(none) = 1/4
- failed pads per 100 m ≈ **0.83** (= 167 × 0.005, treating a per-YEAR rate as a snapshot)

**Research questions.**

1. **Reconcile the prevalence paradox (the key question).** Our own sourcing says field data
   show *"up to ~50 % of concrete sleepers have some voiding"* (Augustin et al.; Li & Sun),
   yet our model's maximum is 3 groups × 5 sleepers = 15 of 167 ≈ **9 %**. Resolve this:
   (a) what **void-depth threshold** makes a sleeper mechanically "unsupported" (i.e. loses
   ballast reaction) in a dynamic sense, versus merely showing a measurable gap?
   (b) Of sleepers with "some voiding", what fraction exceed ~0.5 mm / ~1 mm / ~2 mm (our
   lognormal support)? (c) Given that, what fraction of sleepers on a real line are
   *impactfully* unsupported? A 50 % figure and a 9 % model are only compatible if most
   voids are sub-threshold — confirm or refute with numbers.

2. **Unsupported-sleeper CLUSTER COUNT per 100 m.** How many **distinct contiguous groups**
   of unsupported sleepers occur per 100 m of plain line? If cluster counts are not reported
   directly, supply the two ingredients so we can derive it: (a) the **fraction of sleepers**
   unsupported above the threshold from Q1, and (b) the **distribution of contiguous run
   length** (we already have DU 1–5 cited). Then count ≈ fraction × 167 / mean_run.
   Sources to check: void/gap measurement surveys, sleeper-support condition assessment,
   dynamic track-stiffness or track-geometry-car measurements, unsupported-sleeper detection
   studies.

3. **Ballast fouling EXTENT per 100 m.** What percentage of **route length** on an operating
   ballasted line is fouled/degraded enough to alter vertical support stiffness (i.e. to
   justify our η_k/η_c multipliers)? **Ground-penetrating-radar (GPR) ballast-fouling
   surveys** are the ideal source — they map fouling continuously over kilometres and
   typically report % of length by fouling class. Give: (a) the % of length in a
   stiffness-altering fouled state (by fouling index / class if possible), and (b) the
   distribution of contiguous fouled-zone lengths (we have U(5,20) m cited). Then
   patches per 100 m ≈ fraction × 100 / mean_zone_length. Relate to the tamping-cycle
   anchor (every 20–35 MGT) if that is the only quantitative route.

4. **The SOUND-track case (the prevalence question).** For a randomly chosen 100 m of
   operating regional/secondary ballasted line **between maintenance interventions**, what
   is the probability of **no** stiffness-altering ballast fouling patch, and of **no** group
   of impactfully unsupported sleepers? Are our placeholders P(no patch) = 1/3 and
   P(no group) = 1/4 plausible, too optimistic, or too pessimistic? If the literature
   implies most plain line is sound, give the supported figure.

5. **Rail-pad failures: RATE vs PREVALENCE.** Is the cited P = 0.5 % of fastening positions
   *per year* an **incidence rate** (new failures per year) or a **standing prevalence**
   (fraction failed at any inspection)? What is the observed **snapshot** fraction of failed
   / missing / severely degraded pads on an operating line, and what inspection-and-renewal
   interval does that imply? Our 0.83 pads/100 m assumes ~1 year of accumulated failures —
   state whether that is defensible.

6. **Line-class / tonnage dependence.** Do these prevalences differ materially by line
   class, annual tonnage (MGT), sleeper type (concrete vs timber), or maintenance regime?
   Give the figures appropriate to a **secondary/regional line at 70–90 km/h**, and say
   explicitly if the available data come from high-speed or heavy-haul lines instead.

7. **Co-occurrence / correlation.** Are ballast fouling and unsupported sleepers
   **statistically correlated** (fouling → differential settlement → voids), or effectively
   independent? We currently draw them **independently** per 100 m. If they are correlated,
   give the strength/mechanism and a practical way to couple them (e.g. condition voiding
   probability on the local fouling state).

8. **Near-bridge specificity.** We already model a **cited** hanging-sleeper density spike
   within 15 m of bridge transitions. Is **ballast fouling** likewise elevated near bridges
   and transitions (impact loading, drainage/runoff, stiffness discontinuity)? If so, by
   what factor relative to plain line, and over what distance?

**Deliverable.** For each quantity below, give the value, the citation (author/year/venue),
and a direct quote or number from the source; mark anything you must infer as INFERENCE and
anything unavailable as UNVERIFIED. End with a drop-in sampling specification, per 100 m of
ballasted regional line, between maintenance:

| Quantity | Distribution | Source / status |
|---|---|---|
| P(no stiffness-altering ballast patch) | | |
| ballast patches per 100 m, given ≥1 | | |
| P(no impactfully-unsupported sleeper group) | | |
| unsupported-sleeper groups per 100 m, given ≥1 | | |
| fraction of sleepers impactfully unsupported (threshold from Q1) | | |
| failed pads per 100 m (snapshot prevalence) | | |
| ballast↔voiding correlation (if any) | | |
| near-bridge multiplier for ballast fouling | | |

---

## Secondary (only if the session has room — same "prevalence" question type)

**Deck-crack prevalence.** Our crack nuisance uses P(a given bridge state carries a crack)
= **0.25**, from an earlier report's "20–30 %" recommendation, which was its most weakly
sourced number. For **concrete / composite railway bridge decks**, what fraction of spans in
an operating population exhibit a crack severe enough to cause a **local flexural-stiffness
(EI) reduction of ~5–30 %** (our Sinha-type severity range)? Inspection-database or
bridge-condition-survey statistics preferred. Also: is crack occurrence concentrated at
**mid-span (sagging)** versus **over-pier (hogging)** regions, and by what ratio? (We
computed that our moving-load |M| envelope peaks ~2:1 mid-span vs over-pier, but real
hogging regions may crack more than their moment share due to top-fibre tension plus
water/chloride ingress — confirm or refute.)
