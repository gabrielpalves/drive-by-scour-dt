# Paper 1 — draft notes (Claude, 2026-07-28)

> **Sixth pass — Codex scientific continuation, 2026-08-03 (controlling).**
> The “P1-R1 only” queue below is superseded. P1-S4 remains closed: no
> external preregistration/Zenodo/OSF deposit is required. The numerical
> foundation now has geometry-specific support-aligned meshes, an independent
> Euler--Bernoulli structural oracle, fail-closed nonqualifying package
> validators, a mesh-invariant declared bridge-ballast inventory, and one
> corrected deterministic scour-response pair. The held Zhai primary exposed
> four separate baseline limits: the inherited no-shear/deck-condensed ballast
> topology, per-rail-seat/two-rail scaling, transfer of spacing-dependent
> \(M_b/K_b/K_f\) from 0.545 to 0.600 m, and a 0.1% rail Rayleigh target that
> is inherited author-chosen rather than sourced to Zhai.
> The two virtual moving-rail channels are Eulerian rail FE acceleration-field
> samples, not wheelset/axle-box signals.
>
> **Current pre-A queue:** (1) after the present no-`.tex` audit constraint is
> lifted, correct the manuscript's universal 0.3 m mesh, blanket per-rail
> summation, 0.1% rail-damping provenance, and wheelset-channel descriptions;
> (2) resolve or prospectively sensitivity-test the Zhai topology/scaling/
> spacing transfers and Rayleigh closure; (3) freeze the finite V&V stress-case
> table and complete coupled mesh/time/rail-domain response refinement, one
> upstream reproduction, the remaining eight mechanism
> signatures/registered passage studies, the authenticated dry-ballast arm
> evaluation, and the robustness items in
> `docs/shm_reviewer_readiness_plan.md`; then (4) execute P1-R1 as the final
> clean-commit/source-root closure. Do not dispatch from the present worktree.

> **Fifth pass — 2026-08-02 (post-R13, same day).** Codex R13 confirmed most
> R12 repairs and withdrew two of its own R12 claims (the Wangtawesap 15 cm
> damping threshold — thesis Table 4.17 shows a gradual rise with no
> threshold; and the Shi "no 0–5 sweep" statement), but returned residual
> defects, all now fixed: crack-law values individually labeled author-chosen
> in both docs; "cross-rung inference" headings renamed to "paired
> sensitivity analysis" (module file names stay, flagged as historical);
> track spec "Separable? Yes per the sources" corrected; Esmaeili
> plain-ballast pair acknowledged (Fig. 14: ≈82→64 kN/mm — my earlier "no
> plain-ballast pair" was an over-correction); Wangtawesap sentence rewritten
> from Table 4.17 exact values; O'Brien vehicle values generator-attributed
> (`TrainProp_ObrienCalibrate.m` prints the DOI); round-5 fetch list reframed
> as attribution upgrades, not closure holes; commit-A recipe corrected (see
> the action block below — adding the 261 files alone is not sufficient).
> Full account: ISSUES_FOUND.md "Claude response — 2026-08-02 (R13)".

> **Fourth pass — 2026-08-02 (post-R12).** Codex R12 accepted P1-S4's closure
> by withdrawal (**the OSF/Zenodo deposit is dropped** — author decision
> 2026-08-01; `docs/protocol_deposit.md` is retained as historical only) but
> returned NOT PASS on P1-S1/S2/S3 with specific semantic defects, all now
> fixed: implemented vehicle corrected to TTB-2D's six-DOF formulation with
> wheelset channels as derived responses; descriptor window classified
> author-chosen; three-class prior taxonomy restored in the methodology doc;
> Guo "cannot resolve" and CRN variance/inference overclaims narrowed;
> Lundqvist/Oregui/Esmaeili/Shi/FRA/RIVAS scope caveats attached everywhere;
> cross-document values and the stale 68/40pp counts synchronized (current
> state: 67 = 67 citations, `main.pdf` 42 pages after the R12 wording pass). **The only remaining pre-A
> item is P1-R1**: `git add` the 261 untracked required files (443 required /
> 182 tracked / 0 missing from disk), then commit A. The deposit steps in
> older passes below are superseded — do not perform them.

> **Third pass — 2026-08-01 (P1-S1/S2/S3 closed).** The Codex R11 re-review
> returned NOT PASS with five open P1s. Three are now executed and two are
> staged; the full account is the "Claude response — 2026-08-01" section at
> the end of `ISSUES_FOUND.md`, and the source verdicts are the "SEMANTIC
> CLOSURE PASS" table in `MISSING_PRIMARY_SOURCES.md`. Headlines:
>
> - Every exact simulator law is now individually classified in
>   `sections/numerical_simulation.tex` as anchored-with-scope-caveat,
>   contradicted-but-retained, or author-chosen. Only two values survive as
>   anchored, and both are transferred across a scope boundary that the text
>   now states.
> - All five "conservative" statistical claims are narrowed to a descriptive
>   width rule.
> - Bibliography: **+13 verified entries, −6 untraceable ones**; 68 used = 68
>   defined, zero missing/unused/duplicate; `main.pdf` = 40 pages, biber
>   clean.
>
> **Historical pre-A queue (superseded 2026-08-03)** (formerly mirrored in `docs/framework_rationale.md`
> item 21 and the round-5 wishlist; the deposit item that used to lead this
> list is dropped — see the fourth-pass note above):
>
> 1. **Codex's final pass** over the R12/R13/R14 corrections.
> 2. **Tier-1 primary verification (scheduled 2026-08-03, UCSD):**
>    (a) `garg1984dynamics` (Garg & Dukkipati 1984) — verify the FRA class-4
>    PSD constants (A_v = 0.5376, corner 0.8245/(2π) cycles/m) and restore
>    the citation; (b) Zhai 2004 — verify the track-property values;
>    (c) O'Brien 2018 — verify the vehicle values and the source model's own
>    DOF count. Completing (b)+(c) removes the manuscript's "not
>    independently re-verified" implementation-provenance caveat; any paper
>    that proves unobtainable leaves the caveat standing and closes the
>    sub-item as documented.
> 3. **P1-R1 → commit A** — note (per Codex R13) that adding the 261
>    required untracked files alone is NOT sufficient: stage the ~74 tracked
>    modifications as well, `git add` the 261 required untracked files (443
>    required, 182 tracked, 0 missing from disk), and explicitly disposition
>    every remaining untracked path (add if wanted, otherwise ignore) so the
>    tree is clean; then commit A and confirm the campaign-controls
>    tracked-blob gate goes green.

> **Second pass (same day, post-Codex-handoff):** all 8 P1 manuscript
> blockers from `docs/audit_r11_handoff.md` fixed and recompiled (36 pp,
> clean). Key wording now in force: "prospectively specified and
> source-locked" (with "registered" defined as shorthand for exactly that
> in §1.4); finite-design resampling *sensitivity intervals* + seven-edge
> *tail-adjusted sensitivity envelope* with explicit non-claims (no CI/
> FWER/significance/superiority/joint-sign); equal-budget architecture
> *family* comparisons; simulator intervention/task contrasts; generated
> FRA-v2 profile (phase seed 20260728, never "measured"); dual modal gates
> (0.2–15 Hz all states + healthy-target 3–6/2–4 Hz); bearing fixity on the
> fixed E15 reference (free-rotation-to-near-fixed); exact track-EOV
> implementation semantics. `docs/paper1_methodology.md` aligned to the
> same semantics + the implementation (loss, profile, track EOVs, bearing).
> ~~**Consider an OSF/Zenodo protocol deposit before generation**~~
> *(superseded 2026-08-01: deposit dropped; P1-S4 closed by claim
> withdrawal)*. See ISSUES_FOUND.md for the changed-file list
> (Codex re-audit) and MISSING_PRIMARY_SOURCES.md for the UCSD fetch list.

## What exists

```
paper1/
  main.tex                     — preamble, title, authors, abstract include, section includes
  references.bib               — full bibliography (entries marked VERIFY need primary-source checks)
  sections/abstract.tex        — 1-paragraph abstract; 2 result sentences are \pending
  sections/introduction.tex    — motivation, related work (verified Fernandes summaries), gap,
                                 RQ1–RQ5, contributions, delta table (Table 1)
  sections/framework.tex       — 3 design principles, estimand classes, rung graph (TikZ Fig. 2),
                                 rung table, execution blocks
  sections/numerical_simulation.tex — TTBI model + parameter table (values read from working tree),
                                 channels table, contact scope, all 6 mechanism families with
                                 registered priors, state families + CRN, operational variability
  sections/data_processing.tex — preprocessing chain, noise arm, 4 architecture arms, channel
                                 experiment, HPO policy + search-domain table, split/firewall,
                                 metrics + registered paired sensitivity analyses
  sections/results.tex         — SCAFFOLD ONLY: registered table shells, every cell \pending
  sections/limitations.tex     — model/label semantics, observation model, statistical scope,
                                 practical feasibility & implementation challenges
  sections/conclusion.tex      — framework summary + \pending outcomes paragraph
  ISSUES_FOUND.md              — repo issues found while writing (three-way channel with Codex)
  NOTES_FOR_AUTHOR.md          — this file
```

## Hard rules baked into the draft

- **No pre-R11 numbers anywhere.** The campaign has not run
  (`docs/audit_r5_results.md`: DISPATCH BLOCKED). Every result slot is a red
  `\pending{...}` marker. Do not fill any of them from old datasets — your own
  registered protocol (docs/paper1_outline.md) forbids it.
- Terminology follows the registered guardrails: "modeled support-stiffness
  loss" (never scour depth), "response channels/DOFs" (never sensor count or
  placement), "N-HiTS-inspired multi-rate pooling" (never the N-HiTS
  architecture), retraining (never zero-shot), no detection/POD language for
  our task, L99.6 blockwise (never an 8th edge).

## Structure vs. your proposal

Your Fernandes-2025-style structure was adopted with two adjustments:
1. **Related work lives inside the Introduction** (as in Fernandes 2025),
   with the verified delta table as Table 1. A standalone section would
   duplicate the intro's build-on framing.
2. **"Limitations" is "Limitations and practical feasibility"** — the
   registered claim-boundary list first (that's what protects the paper in
   review), then the Fernandes-style practical/implementation-challenges
   discussion.

Mapping: Proposed framework = §2 (brief, registered-design overview);
Numerical simulation = §3 (TTBI + damage implementation/selection/
randomization); Data processing = §4 (normalization, architectures, ablation,
HPO, metrics — with the Fernandes deltas explicit); Results and discussion =
§5 (scaffold); Limitations = §6; Conclusion = §7.

## Decisions made while writing (flag if you disagree)

- **Title** = the outline's working title (CRN ablation). The alternative
  title is a comment in main.tex.
- **Working tree values in Table 3 (parameters)**: values were extracted from
  the current MATLAB source (read-only) and are correct as of today, but
  Codex is actively editing — the table caption says every value must be
  re-verified against the commit-A protocol descriptors at submission.
- **Standardization vs. min-max**: the paper states we use per-channel
  standardization fitted on training states (differs from Fernandes' min-max)
  with a one-line rationale (sign/zero-crossing preservation, firewall).
- **Bearing lit**: continuous fixity motivated by Khan 2022; the 1e9 N·m/rad
  landmark attributed to the Feng/Fernandes line, mapped to phi ≈ 0.30 (L60).
- **Time2Vec vs Space2Vec**: see ISSUES_FOUND.md item 5.
- **Biblatex numeric** (biber backend) replaces the template's `style=nejm`
  (not generally installed); `\addbibresource{references.bib}` (sample.bib
  was empty — left in place, unused; delete when convenient).
- UCSD affiliation added as a second affiliation — adjust as appropriate.

## What still needs you

1. **Figures 1, 3, 4** are placeholders (model schematic; processing chain/
   architectures; firewall diagram could be added). Figure 2 (rung graph) is
   real TikZ. Results figures wait for the campaign.
2. **Venue choice** — the draft is venue-neutral `article` class; margins/
   abstract length currently follow the old template.
3. **Acknowledgments / funding / author list** — stubs in main.tex.
4. **Compilation**: ✅ builds locally — `main.pdf` (35 pages) compiles clean
   (zero errors, zero undefined citations/refs) with your MiKTeX 25.4 at
   `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64` (pdflatex → biber →
   pdflatex ×2). Note: **siunitx was removed** from the preamble — your
   MiKTeX has siunitx 3.5.5 against an older expl3 (2025-06-09), which
   breaks at compile time; units are written in plain LaTeX instead. If you
   want siunitx back (e.g., on Overleaf it works fine), reintroduce it and
   update MiKTeX's l3kernel locally.
5. **Bibliography**: all risky entries were verified via Crossref/publisher
   records (2026-07-28): Fernandes ×4 with full author names and exact
   titles; Cantero TTB-2D (SoftwareX 20:101253) and VEqMon2D; O'Brien 2018
   (SHM 17(6):1425–1440 — the train-property source); Zhai 2004 (JSV
   270:673–683); Bragança/Souza two-part Applied Sciences reviews; McGeown
   2024 (Sensors 24(5):1684); **Tola 2025 (J. Bridge Eng. 30(7):04025043)
   added and cited** — field drive-by scour detection with RILA, Cantero
   co-author, directly in our niche; Feng 2023 (CACIE 38:1935–1954); Khan
   2022 (SIE 18(8):1177–1191); Duran 2026 (MSSP 252:114255); corrected
   titles for the GPR/ballast sources (Sadeghi, Guo, Husøy, Musgrave,
   Chrismer & Hyslip). Hand-check remaining: AREMA pagination, Time2Vec
   venue, EOV-prior primary sources (ISSUES_FOUND item 6).
6. **Adversarial verification ran** (4 audits): numeric consistency vs the
   registered protocol (all headline values consistent; seed-registry table
   added as Table tab:seeds), terminology guardrails (all violations fixed),
   LaTeX integrity (all cite/ref/environment checks pass), and pre-R11
   contamination (zero historical numbers; two hedging rewords applied).
7. **Abstract/conclusion/results \pending markers** — fill only from
   authenticated R11 outputs, then delete the \pending and \verifyref macro
   definitions in main.tex.

## Suggested Codex review request

Paste to Codex: "Review paper1/sections/numerical_simulation.tex and
paper1/sections/data_processing.tex against the current code
(scour_MATLAB/, core/, training/). Check every stated parameter, formula,
and pipeline order. Record discrepancies in paper1/ISSUES_FOUND.md under
'Codex review'. Do not edit the .tex files; the author arbitrates."
