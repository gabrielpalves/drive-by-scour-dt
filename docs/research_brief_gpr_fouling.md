# Deep-research brief: GPR ballast-fouling extent ("FI>30 on 10–20% of route length")

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
