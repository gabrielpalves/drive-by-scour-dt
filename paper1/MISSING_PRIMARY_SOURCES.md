# Primary sources — status after the UCSD fetch runs (2026-07-28, two rounds)

## ✅ SEMANTIC CLOSURE PASS — 2026-08-01 (P1-S3)

Every source the manuscript now cites for a numerical or mechanism claim was
re-read directly from the local PDF, page-anchored, and then **adversarially
re-read by an independent pass instructed to refute the first verdict**. Five
first-pass `SUPPORTS` verdicts were downgraded to `PARTIAL` by that second
pass; the manuscript wording follows the *downgraded* verdicts.

### Bibliography changes

**Added (13 entries, each with a local PDF):** `augustin2003settlement`,
`lazarevic2016sleeper`, `shi2024unsupported`, `kitahara2024hanging`,
`esmaeili2017fouled`, `wangtawesap2023drainage`, `oregui2016railpad`,
`woo2017lifetime`, `sainzaja2020railpad`, `fra2022ballastwaiver`,
`fra2022rainysettlement`, `rivas2013wheel`, `siahkouhi2025transition`.

**Removed (6 entries, no local artifact):** `sadeghi2018gpr`,
`selig1994track`, `chrismer2018fouling`, `husoy2024defects`,
`musgrave2024ballast` — all five had propped up a single false sentence
claiming the patch-extent prior was "anchored to" GPR surveys; that sentence
is deleted and the extent context now comes from `fra2022ballastwaiver` and
`guo2023gpr`, both held locally. `garg1984dynamics` — the FRA class-4 PSD
constants are now attributed to the TTB-2D generator source, which is
locally verifiable; re-add the textbook only if a copy is obtained.

Result (updated 2026-08-02; the counts reflect the P1-S4 withdrawal, which
removed `nosek2018preregistration`, and the R12 scope-caveat wording pass):
67 cited keys, 67 defined entries, zero missing, zero unused, zero
duplicates; `main.pdf` compiles to 42 pages with no undefined citation or
reference.

### Verdict table (second-pass verdicts)

| Claim the manuscript needs | Source | Verdict | Scope restriction now stated in the text |
|---|---|---|---|
| Wet ballast stiffness ×0.7–0.9 | Wangtawesap 2023 | PARTIAL | 7,100→4,750 kN/m at full submergence = ×0.67, i.e. just *below* our band; measured on flooded **clean** ballast, so wet+fouled use is our extrapolation |
| Wet ballast damping ×1.5–4.0 | Wangtawesap 2023 | PARTIAL | ×4 per-zone at full submergence but only ×2.8 for the **condensed** dashpot; the condensed rise is gradual with no stated threshold (Table 4.17: ≈unchanged at 5–10 cm, +28% at 15 cm, ×2.8 only at full 35 cm submergence) — the earlier "no rise below ~15 cm" reading is withdrawn (2026-08-02) |
| Dry fouling damping reduction ×0.4–0.8 | Esmaeili 2017 | PARTIAL | damping falls up to 67% (≈×0.33); our band is deliberately milder and was not estimated from it |
| Dry fouling stiffness ×1.2–2.0 | Esmaeili 2017 | **NOT_SUPPORTED** | measured stiffness *decreases*: the quoted 54.7→46.6 kN/mm pair (Sec. 3.4 running text; no stiffness table exists) is the 5 wt% tire-derived-aggregate case; Fig. 14 labels the plain-ballast (T0) bars 0.08227→0.06441 under an MN/m caption — a ×1000 unit inconsistency with the prose's kN/mm, disclosed not resolved; back-calculating the text's stated 33.5%/27.6% TDA drops gives ≈82→64 kN/mm (×0.78) — correction 2026-08-02: an earlier note claiming no plain-ballast pair is reported is withdrawn; sign disagreement retained and declared |
| >50% sleepers poorly supported | Augustin 2003 (p. 330), via Kitahara 2024 | SUPPORTS | stated as engineering experience, not a survey; Kitahara found no hanging sleepers in their own campaign |
| 1 mm void → +70% adjacent force | Lundqvist & Dahlberg 2005 | SUPPORTS | also +40% displacement; only 0.5 and 1 mm gaps simulated; no count or void-depth distribution |
| Group size DU{1,…,5} | Lazarević et al. 2016 | **NOT_SUPPORTED** | reports 1–4 m consecutive extent, 1–4-sleeper simulations, correlation to six; fits no count law |
| Five as critical limit | Shi et al. 2024 | **NOT_SUPPORTED** | non-monotonic: 104.5 kN at **three** vs 90.8 kN at five vs 66.1 kN supported |
| Patch length U(5,20) m | FRA ORD-22/01 + Guo 2023 | **NOT_SUPPORTED** | FRA reports degraded locations from 5 ft or less up to 110 ft (about ≤1.5 m to 33.5 m — open-ended), specifically among sites crossing the class-5 profile safety limit (Sec. 4.3, Fig. 22); Guo averages the indicator over 2.4 m windows to suppress 5 cm channel fluctuation — a smoothing choice, not a resolution limit; neither fits a law |
| Fouling index ≈30 as severity limit | FRA ORD-22/01 Table 5 | PARTIAL | FI > 30 = "Highly Fouled" in the report's own **FRA/Zetica BFI interpretation schema**, flagged for likely water presence. NOT Selig's own classes, which are fouled 20 ≤ FI < 40 and highly fouled ≥ 40 — do not attribute the 30 threshold to Selig |
| Wet fouling severity | FRA RR 22-32 | PARTIAL | "up to 15×" settlement at BFI 40 is **repeated from Wilk et al.**, not FRA's own measurement |
| Transition-zone clustering | Siahkouhi et al. 2025 | PARTIAL | 4–8× maintenance frequency is itself cited from Wang & Markine 2018; **no cone length** of 15–24 m is established — our ±15 m window and 3× density are ours |
| Pad Weibull(1.8, 2.2) | Oregui / Woo & Park / Sainz-Aja | **NOT_SUPPORTED** | "Weibull" appears in none of the three; Woo & Park fit **Arrhenius** (50% spring-constant criterion) |
| Pad stiffening direction | Oregui 2016 | PARTIAL (contradicts) | 10-year field-worn pads ≈40% **softer** in complex modulus at 12–18 kN preload (less than half the new modulus at 6 kN), damping nearly absent → aging direction contested. Use 40% at the registered preload; the looser "40–50%" in older notes conflates the two preload cases |
| Large pad stiffness factors as aging | Sainz-Aja 2020 | **NOT_SUPPORTED** | those are temperature / toe-load / frequency effects |
| Pad failure p=0.02 | all pad papers + Williams 2014 | **NOT_SUPPORTED** | no failure rate in any of them. Williams 2014 is a lateral-load/insulator study; exhaustive token search found **no 0.5%** and no per-year statistic anywhere |
| Wheel OOR P=0.30 | RIVAS D2.7 + Iwnicki 2023 | **NOT_SUPPORTED** | closest is 5%/7% above 1.0 mm — on two UK classes whose tread problems the report describes, with **no stated sampling frame or selection rationale**, so not even a fleet rate |
| Orders 1–5 dominant | Iwnicki 2023 | PARTIAL | fleet-specific: ICE 2–3, Stockholm order 3 in ~60%, Chinese heavy-haul 1–3, but 9–28 elsewhere |
| Amplitude 10–120 µm lognormal | RIVAS + Iwnicki | **NOT_SUPPORTED** | developed polygonization reaches ~0.9 mm (Iwnicki); general OOR removed at the wheel lathe reaches ~2.5 mm (RIVAS) — two distinct scopes, not one range; no amplitude distribution reported anywhere |
| PAA precedent | Fernandes et al. 2025 (IJSSD 2650316) | SUPPORTS | 583 segments over 5,830 samples; motivated by dimensionality/training cost, **not** denoising; min–max where we standardize |
| TTB-2D / VEqMon2D identity | Cantero 2022 ×2 | SUPPORTS | SoftwareX 20:101253 and 19:101103, both code version "v1", GPL-3.0 |

### Citation traps recorded during the audit

- **Two different Shi papers.** `shi2024unsupported` is Can Shi et al.,
  Transp. Geotech. 45:101221. RAILCON's own reference [5] is J. Shi, Chan &
  Burrow, Proc. IMechE Part F 227(6):657–667, 2013 — a *different* work,
  also present locally. Do not merge them.
- **Augustin chapter.** Kitahara's own reference entry for it is defective
  (misspells the fourth author, omits the volume editors, series, and page
  range). The correct entry is Augustin, Gudehus, Huber & Schünemann (2003),
  in Popp & Schiehlen (eds.), *Lecture Notes in Applied Mechanics* 6,
  Springer, pp. 317–336. Do not copy Kitahara's version.
- **Sainz-Aja spelling.** The local filename reads `Sainza-Aja`; the printed
  surname is **Sainz-Aja**. The BibTeX entry uses the printed form.
- **Li & Sun 1992** remains unobtainable (Chinese). It is no longer needed:
  the >50% statement is now cited to Augustin directly, whose chapter is
  local inside the Popp & Schiehlen volume.


> **Current status (2026-07-31): the planned local acquisition pass is closed;
> the evidence graph is not.** Remaining work is BibTeX/citation wiring and
> claim cleanup, not another blind fetch. Several frozen numbers are explicitly
> author-chosen because no audited primary estimates them; local PDF presence
> must never be confused with support for an adjacent numerical claim.
> Earlier "still missing" and "only remaining gap" passages below are retained
> as workflow history and are superseded by the definitive direct-PDF audit
> later in this file. Unsupported numerical priors must remain explicitly
> author-chosen rather than being dressed with adjacent literature.

## HISTORICAL — evidence-graph closure checklist (2026-07-31), SUPERSEDED

> **Superseded by the SEMANTIC CLOSURE PASS (2026-08-01) above.** Every item
> below has since been closed: the graph is 68 used / 68 defined (not 61/61;
> now 67 / 67 after the P1-S4 withdrawal removed `nosek2018preregistration`),
> all eight "still need BibTeX entries" sources are entered and cited, and all
> six "no identifiable local artifact" keys were removed. Retained only as a
> record of what the closure pass was answering.

- The TeX/BibTeX graph is syntactically closed: 61 citation keys are used and
  the same 61 are defined, with zero undefined and zero unused keys.
- It is not semantically closed. Essential audited local sources still need
  BibTeX entries and manuscript wiring: Esmaeili (2017), Wangtawesap (2023),
  Kitahara et al. (2024), Lazarević/RAILCON (2016), Siahkouhi et al. (2025),
  Oregui et al. (2016), Sainz-Aja et al. (2020), and Woo & Park (2017).
  Depending on retained wording, FRA ORD-22/01, FRA RR22-32, RIVAS (2013),
  and Shi et al. (2024) also require explicit entries.
- Conversely, no identifiable local artifact was found for six sources already
  cited in the manuscript: `garg1984dynamics`, `sadeghi2018gpr`,
  `selig1994track`, `chrismer2018fouling`, `husoy2024defects`, and
  `musgrave2024ballast`. A bibliography entry is not local traceability.
- The manuscript still needs the claim-level corrections listed in the
  definitive audit below. A broad “field-evidence” sentence cannot substitute
  for source-by-source boundaries.

> **Historical Round-2 acquisition update (same day; superseded):** the second
> targeted fetch closed its then-planned download list. It did not close the
> evidence graph. Newly obtained + renamed:
> `Siahkouhi_2025_track_bridge_transition_zone_monitoring_review.pdf` (the
> ASCE JPEODX.PVENG-1608 review — 4–8× corrective ops, 15–24 m cone),
> `Popp_Schiehlen_2003_system_dynamics_railway_track_subgrade_book.pdf`
> (the FULL Springer volume containing the **Augustin et al. 2003**
> chapter, via UFSC),
> `Kitahara_2024_track_geometry_hanging_sleeper_detection_ASCE.pdf`
> (published version, replacing the preprint-only status), plus useful
> context papers: `FRA_2022_RR22-32_rainy_section_track_settlement_model.pdf`
> (BFI × moisture settlement — supports the wet/fouled interaction
> narrative), `Tutumluer_2011_field_validated_DEM_railroad_ballast.pdf`,
> `Olsson_2024_numerical_framework_ballast_settlement.pdf`,
> `Grossoni_2021_ballasted_track_settlement_VTI.pdf` (settlement-vs-MGT
> context for tamping cycles), and
> `Esen_2023_IN2ZONE_next_generation_railway_transition.pdf`.
> Li & Sun 1992 is confirmed unobtainable (Chinese) → cite **as cited in**
> the RAILCON/Lazarević line.
> **At that point**, the only planned workflow item was the NotebookLM
> number→source mapping. The direct-PDF audit later found the additional
> citation and claim-level gaps listed above.

Fetch run done; ~47 new files landed in `papers/` and the cryptic ones were
renamed to the `Author_Year_description` convention (map at the bottom).

## 🎉 Mystery resolved: "Augustin et al." and "Li & Sun"

You couldn't find them because they are obscure — but they are REAL, and the
attribution chain is now recovered:
`RAILCON2016_determination_sleeper_support_conditions.pdf` (the paper you
fetched as `1401.pdf`) cites them as its refs [1] and [2]:

- **Augustin, S., Gudehus, G., Huber, G., Schünemann, A.** (2003),
  "Numerical model and laboratory tests on settlement of ballast track", in
  K. Popp & W. Schiehlen (eds.), *System Dynamics and Long-Term Behaviour of
  Railway Vehicles, Track and Subgrade*, Springer, Berlin.
  → **obtainable**: Springer book chapter (SpringerLink; Lecture Notes in
  Applied Mechanics series). Add to a future fetch.
- **Li, Z.F., Sun, J.G.** (1992), "Maintenance and cause of unsupported
  sleeper", *China Railway Build*, Vol. 2, pp. 15–17.
  → Chinese-language, effectively unobtainable; if the 50% voiding figure is
  kept, cite it **as cited in** the RAILCON/Lazarević line (or drop it in
  favor of the ASCE/Kitahara ">50% poorly supported" source).

Update `docs/track_eov_sampling_spec.md:88` accordingly when convenient —
the attribution is legitimate, just chained.

## ✅ Obtained (fetch-list items)

| Item | File (new name) |
|---|---|
| Jing hanging-sleeper DEM (Extrica) | `Jing_2015_hanging_sleeper_DEM_ballast_interaction.pdf` |
| RAILCON 2016 (grafar 1401) | `RAILCON2016_determination_sleeper_support_conditions.pdf` |
| Sysyn sleeper–ballast impact | `Sysyn_2021_sleeper_ballast_impact_unsupported_sleepers.pdf` |
| GPR hanging sleepers (Sensors 26) | `Yang_2026_GPR_hanging_sleepers_detection.pdf` |
| Fouled ballast bridge vibrations (IJSEA) | `Farsi_2024_fouled_ballast_railway_bridge_vibrations.pdf` |
| Williams 2014 RailTEC (historically alleged 0.5%/yr) | `Williams-et-al.-2014-Experimental-Field-Investigation-of-the-Effects-of.pdf` — direct review: lateral-load/insulator study; it does **not** report that rate |
| LTU fastening inspection | `Chandran_2022_LTU_thesis_train_based_fastening_inspection.pdf` (full PhD thesis) |
| Huddersfield pad ML (AES 151:102927) | `Ferreno_2021_rail_pad_properties_machine_learning.pdf` |
| Rail-pad bibliometric review (Appl. Sci.) | `Guillen_2026_rail_pad_applications_bibliometric_review.pdf` |
| FRA DOT/FRA/ORD-22/01 | `FRA_2022_ORD-22-01_fouled_ballast_waiver_operations.pdf` |
| CJCE tandem/side-by-side scour equations | `Nandi_2024_scour_tandem_side_by_side_eccentric_piers.pdf` |
| Extreme scour frequency / collapse risk | `Rifo_2022_extreme_scour_frequency_collapse_risk.pdf` |

## ⚠ Historical preprint-only status (superseded; published PDF now present)

- **Kitahara, Masuda, Kai, Nagayama, Su, Tanaka** →
  `Kitahara_2024_track_geometry_hanging_sleeper_detection_preprint.docx`
  (R1 manuscript of the ASCE-ASME J. Risk Uncertainty Part A paper,
  DOI 10.1061/AJRUA6.RUENG-1259 — the ">50% of sleepers poorly supported"
  source). Fine for reading; cite the published version.

## Historical NotebookLM number→source mapping (received 2026-07-28)

The per-number mapping was retrieved from the notebook. Cross-referenced
against `papers/`, it was useful as a candidate-source index. It is **not an
evidence verdict**: the later direct-PDF audit in this file supersedes every
row-level attribution below.

| Registered number | Candidate source(s) proposed by NotebookLM | Local file and direct-audit caveat |
|---|---|---|
| Ballast DRY ×[1.2,2.0] / ×[0.4,0.8] | Esmaeili et al. 2017, Soil Dyn. Earthq. Eng. 98:1–11, 10.1016/j.soildyn.2017.03.033 | ✅ Esmaeili_2017; damping direction/range overlap only. Its dry-sand test mildly softens, so the campaign's high-stiffness band is author-chosen. |
| Ballast WET ×[0.7,0.9] / ×[1.5,4.0] | Wangtawesap, Chulalongkorn Univ. thesis (drainage/ballasted track-bridge) | ✅ Wangtawesap_2023; the numbers describe flooded clean ballast and serve only as a proxy for the campaign's wet-fouled scenario. |
| GRF θx = 3–15 m (typ. 10) | Fenton, ALERT Geomaterials 2014 lecture notes + Firouzianbandpey et al., CPT/Kriging spatial-correlation paper | ✅ Fenton_random_fields_chapter + 2015_Firouzianbandpey (context only — GRF NOT implemented) |
| Pad Weibull(1.8, 2.2); ranges [1.2,3.5]+[1.0,3.0]; damping [0.8,1.2]; k→0 failure | Sainz-Aja et al. 2020, Mech. Mater. 148:103505; Oregui et al. 2016, JSV 363:460–472; Woo & Park 2017 (RailTEC) | ✅ all three; none fits a Weibull, and the reported aging/condition directions conflict. The active law is author-chosen, not a fitted prior. |
| Void depth ln(g_v)~N(−0.2,0.4), 0.5–3.0 mm | Lundqvist & Dahlberg 2005; Sysyn et al. 2021 | ✅ both; neither states this distribution/range. It is not implemented in R11. |
| Tamping 20–35 MGT | **Zhao et al. 2006, Transp. Res. Record 1943(1):50–56, 10.1177/0361198106194300107** + Charoenwong et al. 2024, Transp. Geotech., 10.1016/j.trgeo.2024.101193 | ✅ both now local; context only, not an active R11 prior. |
| OOR P=0.30; orders 1–5; 10–120 µm lognormal | **NOT individually mapped** — Report B grouped these in one "Cited" cell with no per-number citation | ✅ Iwnicki review and RIVAS D2.7 checked; neither supports the triplet. Treat all three values as author-chosen. |

## ✅ Round-3 fetch complete (2026-07-28, evening)

All three fetch targets landed and were renamed:
`Zhao_2006_optimizing_ballast_tamping_renewal_policies.pdf` (TRR
1943(1):50–56 — the last number-bearing primary),
`RIVAS_2013_D2.7_wheel_maintenance_measures_ground_vibration.pdf`
(submitted 14/11/2013), and
`Pieringer_2014_contact_modelling_wheel_flats_Wear.pdf` (Wear
314:273–281 — Pieringer, Kropp & Nielsen; strengthens the wheel-flat
exclusion rationale). Also saved:
`NotebookLM_2026-07-28_number_source_mapping_track_wheels.docx` — the
export of the NotebookLM Q&A that produced the mapping table above.

## ✅ Deep-research reports on disk (regenerations, 2026-07-28)

The originals (2026-07-09) proved unrecoverable from NotebookLM, so both
reports were REGENERATED via Gemini (with the notebook attached) and are
now in `papers/`:
`Track Damage in Bridge Scour Detection - Gemini regen 2026-07-28.docx` and
`Railway Wheel Flats Occurrence Severity Modelling - Gemini regen
2026-07-28.docx`. Both were read in full (2-agent assessment). **Caveats
that make these documentation, not evidence:**

1. **They are regenerations, not the originals** — several numbers drift
   from the frozen spec (dry-fouling stiffness "+75–100%" vs ×[1.2,2.0];
   hanging groups 2–5 vs 1–5; flats U(10,35)/U(30,60) mm vs the historical
   U(20,60); no Weibull parameters, no void-lognormal parameters, no λ
   counts per 100 m, no tamping 20–35 MGT, no 15 m transition numbers, no
   P(OOR)=0.30, no 10–120 µm amplitude law). The campaign values remain
   the frozen registered priors; the direct-PDF audit above is the
   standing ground truth.
2. **Export corruption (track report):** every numeric value is an
   embedded PNG image and the media pool is cross-contaminated with the
   wheel report's formulas — many values are unrecoverable from the file
   itself. Several wheel-report equation images are cropped at the source
   (blank right-hand sides).
3. **Circular citations:** the workhorse reference in both regens —
   "Probabilistic Graphical Models for Predictive Digital Twins at Scale",
   no URL — is the NotebookLM notebook itself (or its Kapteyn–Willcox seed
   paper). Gemini cited the attached notebook as the source for most
   priors: circular, and consistent with the audit's finding that those
   numbers have no external primary.

Regen claims that DO align with the audit: Stockholm ~60% order-3
(Chalmers/Iwnicki), maintained-wheel amplitudes "<0.5 mm", wet ballast
−30% stiffness / +40–300% damping.

## ✅ Optional final fetches — ALL OBTAINED (2026-07-28, night)

- `Nielsen_2024_Chalmers_damaged_wheels_WILD_regulations.pdf` — Nielsen,
  Fehrlund, Maglio, Söderström, Ekberg, Kabo & Vernersson, Chalmers
  Research Report 2024:04 (60 mm condemning limit, WILD assessment).
- `Steenbergen_2007_wheel_flat_contact_geometry_partI.pdf` — VSD
  45(12):1097–1116, DOI 10.1080/00423110701199982 (we now hold Parts I+II).
- `Milne_2021_track_level_support_stiffness_variation_VTI.pdf` — Milne,
  Harkness, Le Pen & Powrie, VSD 59(2):245–268,
  DOI 10.1080/00423114.2019.1677920 (along-track support-stiffness
  variability).
- `Pieringer_2014_contact_modelling_wheel_flats_authors_version.pdf` —
  confirmed: the CPL author's version of the SAME Wear 314:273–281 paper
  already on disk (kept as the open-access duplicate).

**The targeted acquisition pass is closed; the evidence graph is not.** The
next step is the §3 citation-upgrade pass: wire the audited local primaries into
BibTeX and the manuscript, remove or qualify unsupported mappings, and keep
demoted numbers explicitly labeled as author-chosen priors. Full local
traceability for every cited source must be verified separately; PDF presence
alone does not establish support for a numerical claim.

## 🔎 First-pass primary-source audit (direct page-anchored read, 2026-07-28) — superseded 2026-08-01

> This was the definitive record until the 2026-08-01 adversarial re-read at
> the top of this file, which **downgraded five of its `SUPPORTS` verdicts to
> `PARTIAL`**. Where the two disagree, the 2026-08-01 table binds.

All relevant local PDFs were read directly (pdftotext, page-anchored
quotes; full verdicts with quotes in the workflow record). Per prior:

**Supported or narrowly contextualized by an audited primary (citable in the
§3 upgrade pass with the stated caveat):**
- **WET ballast multipliers** ✅ Wangtawesap 2023 thesis states them almost
  exactly: ~30% stiffness reduction fully flooded (Table 4.17 factors
  0.97→0.67) and damping +50% (30 cm) to +300% (flooded) = factors
  1.5–4.0. Caveat: flooded *clean* ballast (drainage problem), not
  fouled+wet. FRA RR 22-32 corroborates severity: settlement "up to 15
  times greater" saturated vs dry at BFI = 40 (via Wilk et al. refs).
- **Broad poor-support occurrence context, not a prevalence estimate**:
  Kitahara 2024, p.1, reports that usually over 50% of sleepers are poorly
  supported or completely unsupported, citing Augustin et al. 2003. Use only as a
  Kitahara-citing-Augustin mechanism/occurrence statement; it cannot calibrate
  the campaign Poisson law or a network prevalence. The Springer volume is
  local, but the relevant page still needs manual/OCR confirmation.
- **70% contact-force increase** ✅ Lundqvist & Dahlberg 2005 — but tied
  specifically to a 1 mm gap (single hanging sleeper, adjacent-sleeper
  force; also +40% displacement, railseat stress 4→8 MPa).
- **Dry-fouling damping reduction** ~✅ Esmaeili 2017: damping falls up to
  67% with dry sand fouling (factors ~0.33–0.56; our [0.4,0.8] band
  overlaps but is milder); regression ξ = 0.002T²−0.012T−0.0015F+0.19.

**NOT supported — keep/label as author-chosen design priors:**
- **Pad-failure `p=0.02` and the alleged 0.5%/year anchor**: Williams
  et al. (2014) studies lateral-load distribution and insulator failure; it
  gives no annual pad-failure incidence. Searches of the other local pad and
  fastening primaries found no equivalent rate. The active Bernoulli
  probability is wholly author-chosen, not an incidence-to-prevalence
  conversion.
- **Discrete-Uniform 1–5 hanging-sleeper group size**: RAILCON 2016 reports
  consecutive occurrence over 1–4 m, summarizes simulations with one to four
  sleepers, and cites correlation studies up to six; it does not fit DU{1,…,5}
  or establish five as a universal critical limit. The campaign distribution
  is a modeling synthesis.
- **P(OOR) = 0.30**: no occurrence statistic near 30% exists in Iwnicki
  2023 or RIVAS D2.7. RIVAS: 5%/7% of wheels with OOR > 1.0 mm (two UK
  fleets). P=0.30 can stand only as an assumption about *any-severity*
  polygonization, which no source quantifies.
- **OOR amplitude 10–120 µm lognormal**: no such range or distribution in
  any source. Reported in-service amplitudes for *developed* OOR are much
  larger (0.5–2.5 mm; re-profiling limits ~0.5 mm; 0.15 mm clip-crack
  threshold). Defensible framing: our band models MAINTAINED in-service
  wheels below intervention limits (roughness ~25 dB re 1 µm ≈ 18 µm falls
  inside it) — a design choice, not a cited statistic.
- **Orders 1–5 dominance**: fleet-specific, not universal. Supported for
  ICE (orders 2–3), Stockholm metro (order 3 in ~60% of wheels), Chinese
  heavy-haul (1–3); other fleets are high-order (9–28). Cite as
  "low-order polygonization documented in several fleet studies", not
  as general dominance.
- **Pad Weibull(1.8, 2.2) + aging ranges [1.2,3.5]/[1.0,3.0] + damping
  [0.8,1.2]**: "Weibull" appears in NONE of the four pad papers. Woo &
  Park fit an ARRHENIUS lifetime model (time to 50% spring-constant
  change ≈ factor 1.5 criterion), not a Weibull, not a multiplier
  distribution. Worse, the aging DIRECTION is contested: Oregui's 10-year
  field-worn pads are ≈40% SOFTER in complex modulus at the 12–18 kN
  preloads (more than half softer at 6 kN) with near-zero damping, while fatigue
  tests stiffen. Sainz-Aja's large factors (up to 437% EPDM) are
  temperature/toe-load/frequency effects, not aging. The wide stiffness
  range is defensible as SERVICE-CONDITION variability; the Weibull shape
  is purely a modeling choice.
- **Dry-fouling stiffness increase [1.2, 2.0]**: Esmaeili's box tests show
  a mild stiffness DECREASE with dry sand fouling (54.7→46.6 kN/mm);
  only qualitative "increases the track rigidity" language exists.
  Direction depends on fouling type (fines/compaction vs sand).
- **Void-depth 0.5–3.0 mm lognormal**: no paper states the range or any
  distribution (Sysyn: depth distribution along track "not known...
  future research"; L&D simulate 0.5/1 mm; Sysyn sweeps to ~4.2 mm;
  Kitahara uses 7 mm; Yang (China) works in centimetres). Moot for R11
  (binary support removal implemented), but the spec's law is a synthesis.

**Historical impact assessment — superseded 2026-07-31:** no numeric campaign
value changes, but manuscript wording still needs cleanup. In particular,
`sections/numerical_simulation.tex` must not imply that every exact prior was
assembled from field evidence, and `sections/limitations.tex` must not call the
whole prior family generically conservative. The unsupported values remain
valid only as explicitly author-chosen conditional design scenarios.
The spec's "CITED (solid)" labels for pad Weibull, dry-fouling stiffness,
void lognormal, and the OOR triplet are DEMOTED by this audit — see the
dated addendum in `docs/track_eov_sampling_spec.md`. This is exactly the
"separate literature-backed priors from author-chosen assumptions"
tightening Codex requested (P2).

Skip the rest of Bibliography B: 12 items are already in `papers/`; the
others are grey/web sources (TfW, Rail Engineer, TSB reports, L.B. Foster,
Scribd), flat-detection method papers (mechanism disabled in R11), or
redundant free items (IntechOpen chapter, Concordia thesis, PHM/FaultSeg,
Chalmers WILD/fast-model, Oxford axle-box, Emerald denoising, EN 15313
paywalled standard).

(Corrected status: Augustin 2003 ✅ in the Popp & Schiehlen volume;
Li & Sun 1992 → cite-as-cited-in; transition review ✅ Siahkouhi 2025;
the alleged 0.5%/yr pad rate is **not** in Williams 2014; DU{1,…,5} is
**not** established by RAILCON 2016; >50% poor support ✅ Kitahara 2024
+ Augustin attribution chain.)

## 🎁 Bonus fetches beyond the list (now available for citation upgrades)

- **TTBI**: `Zhai_2019_train_track_bridge_interaction_review.pdf` (VSD
  57(7):984–1027 — a strong extra citation for the paper's §3.1).
- **Wheel OOR/polygonization**: `Iwnicki_2023_out_of_round_wheels_polygonisation_review.pdf`
  (VSD 61(7):1787–1830 — the modern successor to Nielsen & Johansson 2000;
  check it for the 30% occurrence and 10–120 µm amplitude numbers),
  `Tao_2021_high_order_polygonal_wear_metro_wheels.pdf`,
  `Zhang_2009_high_speed_wheel_ovalization_ICCTP.pdf`,
  `Song_2019_wheel_polygons_dynamic_track_performance.pdf`,
  `Song_2020_polygonized_wheel_detection_axlebox_TFA.pdf`,
  `Xu_2025_wheel_polygonalization_gear_transmission.pdf`.
- **Wheel flats (exclusion rationale)**: `Steenbergen_2008_wheel_flat_contact_geometry_partII.pdf`,
  `Vale_2021_wheel_flats_ballasted_slab_tracks.pdf`,
  `alemi-et-al-2016-condition-monitoring-approaches...pdf`, plus the
  detection batch (Liu 2022, Peng 2025, Zhou 2020, Mosleh 2021, Gao
  2019/2020, Mohammadi 2023, Komorski 2021, Pecile 2016, Naseri 2024).
- **Pads**: `Woo-and-Park-2017-Lifetime-Evaluation-of-Rail-Pads...pdf`
  (lifetime/aging evidence, but it does not ground the campaign Weibull prior),
  `Sainza-Aja_2020_...static-and-dynamic-stiffness-of-rail-pads.pdf`,
  `Oregui_2016_Obtaining_railpad_properties...pdf`.
- **Ballast/water**: `Wangtawesap_2023_Impact-of-drainage-problem...pdf`
  (wet-ballast dynamics — candidate for the wet-multiplier rationale),
  `Esmaeili_2017_...sand-fouled-ballast...pdf`, `Guo_2023_ballast_fouling_GPR.pdf` (PDF of the already-cited bib entry).
- **Random fields**: `Fenton_random_fields_chapter.pdf`,
  `2015_Firouzianbandpey_...spatial-correlation-length...pdf` (grounds the
  GRF θx literature context).
- **Tamping**: `Charoenwong_2025_Prediction_of_future_ballast_tamping.pdf`
  (candidate for the 20–35 MGT anchor).
- **Unsupported sleepers extras**: `Shi_2013_unsupported_sleepers_heavy_haul_embankment.pdf`,
  `Shi_2024_Dynamic_impact_of_unsupported_sleepers.pdf`,
  `Lazarevic_2015_sleeper_support_micro_tremor_analysis.pdf`,
  `Esen_2026_self-levelling_sleeper_concept.pdf`,
  `RAILCON2016_performance_requirements_rail_fastening_systems.pdf`.

**Next step (after Codex's review settles):** a citation-upgrade pass on the
paper's §3 — read the candidates above, confirm which registered numbers
they actually support, and replace "modeling prior documented in the
campaign specification" wording with primary citations where justified.
Do NOT wire numbers to papers before reading them.

## Rename map (old → new)

`dot_59915_DS1` → FRA_2022_ORD-22-01_fouled_ballast_waiver_operations ·
`FULLTEXT01` → Chandran_2022_LTU_thesis_train_based_fastening_inspection ·
`1-s2.0-S096599782030973X-main` → Ferreno_2021_rail_pad_properties_machine_learning ·
`1401` → RAILCON2016_determination_sleeper_support_conditions ·
`1435` → RAILCON2016_performance_requirements_rail_fastening_systems ·
`zhang2009` → Zhang_2009_high_speed_wheel_ovalization_ICCTP ·
`applsci-16-05323` → Guillen_2026_rail_pad_applications_bibliometric_review ·
`applsci-11-07127` → Vale_2021_wheel_flats_ballasted_slab_tracks ·
`applsci-12-06837` → Liu_2022_wheel_flat_VMD_axlebox_acceleration ·
`applsci-09-04165` → Song_2019_wheel_polygons_dynamic_track_performance ·
`applsci-10-01613` → Song_2020_polygonized_wheel_detection_axlebox_TFA ·
`applsci-15-07962` → Peng_2025_wheel_flat_recognition_wayside_force ·
`applsci-10-01297` → Zhou_2020_wheel_flat_multisensor_arrays ·
`applsci-11-04002` → Mosleh_2021_wheel_flat_spectral_kurtosis ·
`sensors-26-01905` → Yang_2026_GPR_hanging_sleepers_detection ·
`sensors-19-03614` → Gao_2019_wheel_flat_parallelogram_mechanism ·
`sensors-20-04969` → Gao_2020_wheel_flat_optical_position_sensor ·
`sensors-23-01910` → Mohammadi_2023_unsupervised_wayside_wheel_flat_detection ·
`machines-13-00323` → Xu_2025_wheel_polygonalization_gear_transmission ·
`sustainability-13-07740` → Sysyn_2021_sleeper_ballast_impact_unsupported_sleepers ·
`IJSEA13021009` → Farsi_2024_fouled_ballast_railway_bridge_vibrations ·
"Out-of-round railway wheels and polygonisation" → Iwnicki_2023_out_of_round_wheels_polygonisation_review ·
"An investigation into...high-order polygonal wear..." → Tao_2021_high_order_polygonal_wear_metro_wheels ·
"The role of the contact geometry...Part II" → Steenbergen_2008_wheel_flat_contact_geometry_partII ·
"Micro-analysis of hanging sleeper..." → Jing_2015_hanging_sleeper_DEM_ballast_interaction ·
"Train track bridge dynamic interaction..." → Zhai_2019_train_track_bridge_interaction_review ·
"Frequency analysis of extreme scour depths..." → Rifo_2022_extreme_scour_frequency_collapse_risk ·
"developing-new-equations..." → Nandi_2024_scour_tandem_side_by_side_eccentric_piers ·
"Ballast fouling inspection...GPR" → Guo_2023_ballast_fouling_GPR ·
`ASCE_ASME_PartA_Kitahara...docx` → Kitahara_2024_track_geometry_hanging_sleeper_detection_preprint ·
`05_Fenton_RandomFields` → Fenton_random_fields_chapter ·
`Influence_of_unsupported_sleepers..._AHCC` → Shi_2013_unsupported_sleepers_heavy_haul_embankment ·
`lazarević-et-al-2015...` → Lazarevic_2015_sleeper_support_micro_tremor_analysis

Round 2: `siahkouhi-et-al-2025-...` → Siahkouhi_2025_track_bridge_transition_zone_monitoring_review ·
`dot_64657_DS1` → FRA_2022_RR22-32_rainy_section_track_settlement_model ·
"Tutumluer et al 2011" → Tutumluer_2011_field_validated_DEM_railroad_ballast ·
`1-s2.0-S2352146523011699-main` → Esen_2023_IN2ZONE_next_generation_railway_transition ·
`1-s2.0-S2214391223002131-main` → Olsson_2024_numerical_framework_ballast_settlement ·
`1-s2.0-S2214391220303214-main` → Grossoni_2021_ballasted_track_settlement_VTI ·
`978-3-540-45476-2` → Popp_Schiehlen_2003_system_dynamics_railway_track_subgrade_book ·
`kitahara-et-al-2024-...` → Kitahara_2024_track_geometry_hanging_sleeper_detection_ASCE

## Round-5 fetch wishlist (2026-08-02; boundary clarified same day after R13)

Mapped by cross-referencing all 67 cited keys against `papers/` holdings.

**Closure boundary (corrected 2026-08-02 after R14; supersedes both earlier
drafts of this paragraph):** the campaign prior/EOV values fall into the
manuscript's three declared classes. The scope-caveated engineering proxies
and the one contradicted-and-retained band trace to held primary PDFs (the
audited set above); the **author-chosen values trace to no source by
design** — that is their declared evidentiary status, not a gap. The
model-parameter sets (track properties, vehicle properties, FRA class-4 PSD
constants) are attributed in the manuscript to the generator's own
hash-locked property files, which name their sources
(`TrackProp_Zhai_et_al_WithBallastOnBridge.m` → Zhai et al.;
`TrainProp_ObrienCalibrate.m` → O'Brien et al., DOI printed in its header).
**That is implementation provenance, not verification of what Zhai 2004 or
O'Brien 2018 contain** — the manuscript's provenance paragraph now states
this explicitly — and upgrading it to direct content verification requires
the two PDFs. **Tier 1 below is therefore a scheduled pre-A step
(acquisition planned 2026-08-03)**; tiers 2–3 remain pre-submission
diligence. The other unheld citations back qualitative attribution,
precedent, or methods statements only.

**Tier 1 — highest-value upgrades (model-parameter and intro-statistic
sources):**

1. `garg1984dynamics` — Garg & Dukkipati 1984 textbook (user fetching).
   On arrival: verify the FRA class-4 PSD constants (A_v = 0.5376, corner
   0.8245/(2π) cycles/m) against its parameterization, then restore the
   citation alongside the generator-source attribution.
2. `zhai2004track` — Zhai, Wang & Lin 2004, J. Sound Vib. 270(4–5):673–683.
   The track-parameter table's values are manuscript-attributed to the
   generator's track-property file (held, hashed), which cites this paper;
   fetching it upgrades that chain to direct verification.
3. `obrien2018vehicle` — OBrien, Quirke, Bowe & Cantero 2018, Struct. Health
   Monit., DOI 10.1177/1475921717744479. The vehicle values are
   manuscript-attributed to `TrainProp_ObrienCalibrate.m` (held, hashed,
   DOI in header) — a **repository-local** configuration file, not one of
   the seven upstream TTB-2D pre-implemented train types. *(Retraction
   2026-08-02, per R14: an earlier draft of this item claimed the held
   TTB-2D SoftwareX paper's Table 1 corroborates this train; it does not —
   Table 1 lists seven other configurations.)* Fetching the PDF verifies the
   implemented values against the publication and definitively settles the
   source model's own DOF count (the R12 confusion).
4. `wardhana2003analysis` — Wardhana & Hadipriono 2003, J. Perform. Constr.
   Facil. 17(3). Backs "leading cause of bridge collapse worldwide"; the
   paper contains specific failure-share statistics worth quoting exactly.
5. `arneson2012hec18` — FHWA HEC-18 (free download from FHWA). Same intro
   claim + inspection-timing claim.
6. `lamb2019scour` — Lamb et al. 2019 (British railway scour risk). Backs
   the "substantial share of weather-related risk" claim.

**Tier 2 — characterization claims (we describe what the source does):**

7. `malekjafarian2015review` — cited 3×, incl. the suspension-level
   filtering claim (Shock & Vibration, open access — easy).
8. `molodova2011axle` — shares the sprung/unsprung filtering claim.
9. `sinha2002simplified` — cited twice as a NEGATIVE contrast ("not the
   tapered crack-depth model of Sinha et al."); holding it protects that
   characterization.
10. `nielsen2000out` — polygonization survey; numeric OOR claims already
    covered by held Iwnicki 2023, so context-only.
11. `locke2020using` — backs "profile held fixed within a scenario" claim
    about prior ML studies.
12. `kamariotis2023framework` — CHECK: held `Kamariotis_2024_Quantifying_
    VoSHM.pdf` may be the companion paper, not the cited 2023 MSSP
    framework paper. Verify and fetch the 2023 one if different.

**Tier 3 — precedent-narrative only (fetch if convenient):**
`yang2004extracting`, `prendergast2013/2014/2016`, `fitzgerald2019drive`,
`keenahan2014use`, `obrien2015apparent` (drive-by history), plus the ML/stats
methods classics (`keogh2001dimensionality`, `challu2023nhits`,
`kazemi2019time2vec`, `hochreiter1997long`, `srivastava2014dropout`,
`akiba2019optuna`, `bergstra2011algorithms`, `jamieson2016non`,
`kingma2015adam`, `loshchilov2017sgdr`, `mckay1979comparison`,
`law2015simulation`, `efron1979bootstrap`, `davison1997bootstrap`,
`dunn1961multiple`) — all standard, most freely available. Standards
(`en1994`, `en13848`, `en61373`) are paywalled; cited only for what they
are, so optional.

## Acquisition update (2026-08-03; supersedes the wishlist status)

The Tier 1--3 article acquisition is complete except for the explicitly
unavailable Garg chapter and two methodological books:

- Zhai 2004, O'Brien 2018, Wardhana 2003, HEC-18, Lamb 2019, all Tier 2
  characterization papers, all drive-by precedent papers, and the cited
  Kamariotis 2023 paper are now held under descriptive names in `papers/`.
- **Zhai direct check (2026-08-03):** Eq. (5) and Table 1 define one discrete
  independent vibrating ballast mass at each rail support point, and the code's
  base rail, pad, half-sleeper, ballast, and sub-ballast numbers match the table
  at its stated *per rail seat* scope. This resolves B54's mesh-dependent
  numerical inventory: one retained 531.4 kg value now attaches at each deck
  DOF under a sleeper assigned to the bridge. It does **not** validate that
  attachment. Zhai retains an independent ballast DOF and adjacent-mass shear
  \(K_w,C_w\); the repository omits the shear branch and condenses on-bridge
  ballast mass onto the deck. Zhai's no-shear comparison is 12% above the
  measured ballast acceleration and its conclusion calls shear necessary for
  track dynamics. Deck condensation, shear omission, and full endpoint lumps
  are inherited model-form/domain-partition choices, not Zhai bridge rules.
  The check also exposes two additional open conventions:
  the planar property file doubles rail and sleeper terms but not the other
  per-seat terms; and Zhai's set uses 0.545 m support spacing while the
  generator retains its \(M_b\), \(K_b\), and \(K_f\) values at 0.600 m
  without re-evaluating their spacing-dependent expressions. No claim may say
  all per-rail quantities were summed or that this is a spacing-consistent
  Zhai reproduction until upstream benchmarking or prospective sensitivities
  resolve those transfers. `Track.Rail.Damping.per = 0.1%` is not in Zhai and
  is classified separately as an inherited author-chosen Rayleigh target.
- The 13 article/conference methods sources are now held and renamed:
  Keogh 2001, Challu 2023, Kazemi 2019, Hochreiter & Schmidhuber 1997,
  Srivastava et al. 2014, Akiba et al. 2019, Bergstra et al. 2011, Jamieson &
  Talwalkar 2016, Kingma & Ba 2015, Loshchilov & Hutter 2017, McKay et al.
  1979, Efron 1979, and Dunn 1961.
- Garg & Dukkipati Chapters 1 and 2 are held. Chapter 3 is unavailable through
  UCSD, UFSC, and the author's other attempted access route. No manuscript
  claim may rely on unverified Chapter 3 content; documented unavailability is
  sufficient and is not a pre-A blocker.
- The only unheld methods references are Law (2015), *Simulation Modeling and
  Analysis*, 5th ed., and Davison & Hinkley (1997), *Bootstrap Methods and
  Their Application*. Both are standard book-level methodological support,
  not sources for a campaign physical prior or numeric TTBI claim. Obtain them
  if convenient for direct pre-submission checking; their absence does not
  block implementation or generation.
- The three paywalled standards remain optional because the manuscript cites
  them only for their identity/scope, not for an unverified numeric design law.

Direct source check of the held Kamariotis et al. (2023) MSSP paper confirms
that its Table 1 supplies the generic gradual-deterioration prior now used in
the digital-twin scenario (`A` mean 1.94e-4/COV 0.40, `B` mean 2.0/COV 0.10,
process-noise mean -0.005/COV 0.10). It does **not** calibrate those variables
to hydraulic scour. Its compound-Poisson example uses rate 0.04/year and
lognormal jump mean 3.75/COV 0.25; the repository defaults 0.10/year,
5.0/0.60 are author-chosen placeholders and must not be attributed to that
table.

All newly acquired PDFs were identified from their title pages, checked for
duplicate hashes and filename collisions, renamed without changing SHA-256
content, and rendered again after the move.
