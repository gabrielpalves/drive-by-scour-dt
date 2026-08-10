# Deep-research brief: GPR ballast-fouling extent ("FI>30 on 10–20% of route length")

> **DIRECT-SOURCE CORRECTION (2026-07-31; supersedes the “resolved” inference
> below).** The reports summarized here use different denominators, thresholds,
> line contexts, and evidence classes; they do not estimate the implemented
> Poisson count law. Consequently `ballast_rate_100m = 1.2` is an author-chosen
> design rate, not a population estimate or a generically conservative inferred
> prior. Published zone lengths can contextualize the 5–20 m bounds, but do not
> fit the exact `U(5,20)` distribution. The proposed sensitivity sweep remains
> unimplemented, and several cited items still lack complete local traceability.

**Why (2026-07-19):** the ballast-patch rate in our generator (`A00_Run.m`:
`ballast_rate_100m = 1.2` fouled patches per 100 m = 15% of route length ÷ 12.5 m
mean patch) is derived from ONE pivotal anchor — "GPR surveys put highly fouled
track (Fouling Index FI > 30) at 10–20% of route length" — whose citation in our
2026-07-15 deep-research report is a **Slideshare deck**, not a peer-reviewed
source. Before Paper 1 we must either re-anchor it to citable literature or
demote λ=1.2 to a declared assumption with a sensitivity sweep.

---

## PROMPT (paste into Gemini Deep Research)

I am modelling ballast fouling for a train–track–bridge simulation of a
European-style secondary/regional rail line, and I need to verify one specific
quantitative claim with peer-reviewed or official (infrastructure manager /
government report) sources. Please research the following and give exact
quotes, page/section numbers, and full citations (DOI where possible) for every
number you report.

1. **The claim to verify:** "Ground-penetrating-radar (GPR) condition surveys
   find that 10–20% of surveyed route length is 'highly fouled' or 'fouled'
   ballast (Fouling Index FI > 30)." Is there ANY peer-reviewed GPR survey,
   national rail-network condition report, or thesis with network-scale data
   that states what PERCENTAGE of route length falls in a fouled/highly-fouled
   ballast class? I need the percentage, the fouling threshold used, the
   network/line type (main line vs secondary/regional), the country, and the
   surveyed length. Candidate leads I already suspect: a 2022 Taylor & Francis
   paper on GPR ballast fouling inspection and quantification; MDPI *Sensors*
   papers on GPR ballast condition indices; work by P. Anbazhagan (IISc
   Bangalore) on GPR fouling quantification; Network Rail / Deutsche Bahn /
   ProRail / SNCF ballast condition statistics; US FRA or AREMA reports; any
   PhD theses with GPR line-survey statistics.

2. **The threshold question:** Selig & Waters' Fouling Index is FI = P4 + P200
   (percent passing the No. 4 and No. 200 sieves). In Selig's own
   classification, which FI ranges are "moderately fouled", "fouled", and
   "highly fouled"? Specifically: is "highly fouled" FI ≥ 40 (as I believe) and
   does ANY standard classification use FI ≥ 30 as its "fouled" boundary? If
   the 10–20% figure exists only for a different threshold (e.g. FI ≥ 20 or
   "moderately fouled and worse"), report that explicitly — the threshold
   changes what my simulated patches represent.

3. **Patch geometry:** any published data on the LENGTH of individual fouled
   zones/patches along track (mean/distribution, e.g. from GPR B-scans). My
   model uses patch length ~ U(5, 20) m (mean 12.5 m) — is that supported?

4. **Secondary lines vs main lines:** any evidence on whether fouling extent is
   higher on secondary/regional lines (lower renewal budgets, older ballast)
   vs main lines — even qualitative statements from infrastructure managers.

5. **If the network-level percentage genuinely does not exist in citable form**,
   say so plainly and list the CLOSEST citable quantitative statements (e.g.
   "X km of line Y surveyed, Z% flagged for ballast cleaning"), so I can either
   re-derive my rate from those or declare it an explicit assumption.

Output format: for each numbered item, a short answer first, then the
supporting quotes with full citations. Flag every number that comes from a
non-peer-reviewed source (slide decks, blogs, marketing material) — those are
exactly what I am trying to replace.

---

## What to do with the answer (for the assistant, next session)

- If a citable % + threshold exists → update `ballast_rate_100m` derivation in
  `docs/track_eov_sampling_spec.md` + the A00 comment; if the value moves
  materially (>±50% on λ), that is a PROTOCOL change → new protocol hash, note
  in framework_rationale; regenerate only if the campaign has not started.
- If not → keep λ=1.2, declare it an explicit assumption in Paper 1 §methods,
  and add the planned sensitivity sweep (λ ∈ {0.6, 1.2, 2.4}) to the
  future-work / robustness section.

---

## ✅ RESOLVED 2026-07-19 — user ran the research (NotebookLM over the
## deep-research corpus: https://notebooklm.google.com/notebook/8c4bab81-d0af-44cc-a125-ba9fa87e3b61)

**Verdict: the middle path — the exact claim is NOT citable, but λ=1.2 IS
inside a citable envelope, so the VALUE stays (no protocol change, nothing
regenerates) and only the DERIVATION text changed** (applied same day to
`A00_Run.m` comments, `track_eov_sampling_spec.md`, `paper1_methodology.md`).

Findings (per question):
1. **Extent**: no network-wide "10–20% at FI>30" exists. Citable envelope:
   Norway ~9.4% *defect rate* (3,400/36,100 inspections 2014–2024; Husøy et
   al. 2024, Civil-Comp, DOI 10.4203/ccc.7.24.3 — NOTE: a rate per
   inspection, NOT % of length; use as order-of-magnitude proxy only);
   Iran 17-km regional GPR: 66% at Selig FI≥20, **12% highly fouled FI≥40**
   (Sadeghi et al. 2018, J. Appl. Geophys. 151, DOI
   10.1016/j.jappgeo.2018.02.020 — the strongest anchor, and its 12%
   highly-fouled fraction is close to our 15%); US heavy-haul "majority
   fouled" (FRA 2017 RIVIT deck — NON-peer-reviewed, corroboration only).
2. **Thresholds**: Selig canonical = fouled 20≤FI<40, highly fouled ≥40
   (quoted in Sadeghi 2018 Table 1). "FI>30 = highly fouled" is the
   Zetica/FRA **BFI** schema (FRA DOT/FRA/ORD-22/01 Table 5, §5.3.1 —
   official but non-peer-reviewed). FI≈30 = functional drainage-loss /
   end-of-ballast-life limit (Chrismer & Hyslip 2018, AREMA — industry).
   **Paper wording**: patches = ballast at/beyond the FI≈30 drainage-loss
   limit (between Selig "fouled" and "highly fouled").
3. **Patch length U(5,20) m: ~~SUPPORTED~~ — superseded by the 2026-07/08
   direct-PDF audits (verdict now NOT_SUPPORTED; see
   `paper1/MISSING_PRIMARY_SOURCES.md`).** FRA reports degraded locations
   from 5 ft or less up to 110 ft (≈≤1.5–33.5 m, open-ended), among sites
   crossing the class-5 profile limit (DOT/FRA/ORD-22/01 §4.3, Fig. 22,
   printed p. 34; the p. 31 analogue is the class-4 limit, 5 ft or less up
   to 80 ft);
   Guo et al. average their GPR indicator over 2.4 m windows to suppress
   5 cm channel fluctuation — a smoothing choice, not a minimum resolvable
   feature (Guo et al. 2023, Int. J. Rail Transp. 11(2), DOI
   10.1080/23248378.2022.2064346 — this is the "Tandfonline GPR fouling"
   paper the old brief hoped for). Neither source fits U(5,20) m; the law is
   author-chosen with source context only.
4. **Secondary lines: SUPPORTED** — ballast design life 25 yr main vs up to
   50 yr secondary rural (Musgrave 2024, Network Rail perspective, DOI
   10.17265/2328-2142/2024.05.004 — practitioner journal, cite as practice
   reference).

**Remaining pre-submission diligence** (cheap, do during paper writing):
spot-verify the four load-bearing quotes against the actual PDFs (NotebookLM
quotes are excerpts; page numbers must be confirmed), and keep the
peer-review flags in the citations exactly as recorded above — two of the
five anchors (FRA report, AREMA paper) are official/industry, not
peer-reviewed, and the paper should say which is which.
