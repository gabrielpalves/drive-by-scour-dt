# Paper 1 — Related work: the Fernandes line (verified against the PDFs)

> **NON-AUTHORITATIVE HISTORICAL DRAFT.** The primary-source summaries below
> remain useful, but every “this work”, architecture, novelty, response-budget
> and result claim predates R11 and is superseded. Rebuild those sections from
> the executed R11 protocol and results. The current block is
> `MultiRatePooling1D` (N-HiTS-inspired), not N-HiTS, and no current `2 ≈ 8`
> result exists before the campaign runs.

Drafted 2026-07-09 from the primary sources in `papers/` (extracted with `pdftotext`).
This is the §3 "related work + delta" material for `docs/paper1_outline.md`, with the
Table-1 cells now verified. **Note on framing:** Fernandes, Lopez, Ribeiro, Miguel et al.
are at **UFSC (the same department the first author comes from) and CONSTRUCT/Porto** —
these are closely related, partly same-group works. Cite them as the foundation this paper
builds on; the deltas below are specific methodological advances, stated so a reviewer
(quite possibly one of these authors) sees them as fair.

---

## What each paper actually does (verified)

**Fernandes, Lopez & Ribeiro (2024).** *Drive-by scour damage detection in railway bridges
using deep autoencoder and different sensor placement strategies.* J. Civil Structural
Health Monitoring 14:1895–1916. DOI 10.1007/s13349-024-00821-w.
- **Unsupervised** deep autoencoder on **raw** vertical acceleration (no preprocessing);
  anomaly via a **Kullback–Leibler-divergence** damage index; ROC-curve evaluation.
- Studies **sensor placement** across vehicle positions; best detection from the **front
  and rear bogies of the first and last vehicles**.
- Detects **5% and 10%** scour (local pier-foundation stiffness reduction). Proof of
  concept; quantifies how many crossing events are needed for confident inference.

**Fernandes, Lopez, Ribeiro & Miguel (2025).** *Early Multi-damage Classification in Railway
Bridges Using Drive-by Numerical Measurements with PAA and CNN.* Int. J. Structural
Stability and Dynamics, 2650316 (28 pp). DOI 10.1142/S0219455426503165.
- **Supervised multi-damage classification** of three simultaneous damage types — **cracks,
  bearing-device damage, and scour**.
- Pipeline: Min–Max normalization → **PAA** dimensionality reduction → **CNN**; architecture
  and hyperparameters chosen by **Bayesian optimization**.
- Robustness via **confusion matrices and boxplots** over data/network randomness; a
  **sensitivity analysis** of each EOV (**track profile**, vehicle velocity, signal noise,
  temperature, mechanical-property variation).
- Sensors: **car body + front bogie of the leading vehicle**. Finding: mechanical-property
  variability most degrades the car-body sensor.

**Fernandes, Minski, de Souza, Ribeiro, Miguel & Lopez (2026).** *Early Scour Damage
Detection Using Drive-By Monitoring Data through Supervised Learning.* ASCE J. Structural
Design and Construction Practice. DOI 10.1061/JSDCCC.SCENG-1785.
- **Supervised CNN** classifying **levels of scour**; two variants — acceleration only, and
  acceleration **+ vehicle speed**.
- Sensors: car body + front bogie of the first vehicle; robustness via confusion matrices
  and boxplots.
- Headline: including **vehicle speed** lifts the **car-body** sensor to **~100%** accuracy
  across scenarios — the "speed rescues the car body" result.

**Fernandes, Lopez, Ribeiro & Dutta (2026).** *Drive-by Monitoring for Scour Damage
Classification in Deep Foundations of a Railway Bridge Using Optimized CNN.* LATAM-SHM 2026
(e-J. NDT, ndt.net 32457).
- Scour-depth classification on a **high-fidelity FE model of the real Canelas Railway
  Bridge**, calibrated to **field modal parameters**.
- Model: a **hybrid CNN–LSTM**, Bayesian-optimized **per sensor configuration**; statistical
  evaluation over multiple runs.
- EOVs: speed, mass, temperature-dependent stiffness, **track irregularities**, noise.
  Result: consistent accuracy across scenarios and sensor locations.

---

## Table 1 — This work vs. the Fernandes line (verified)

| Work | Task / damage | Preprocess + model | Model / sensor selection | Robustness | Key result / scope |
|---|---|---|---|---|---|
| Fernandes 2024 (JCSHM) | detect scour (unsupervised) | raw signal + deep autoencoder | fixed AE; sensor **placement** compared | KL index, ROC | 5–10% scour via front/rear bogies |
| Fernandes 2025 (IJSSD) | classify crack+bearing+scour (multi-damage) | Min–Max + PAA + CNN | Bayesian-opt CNN; 2 fixed sensors | confusion matrices, boxplots, EOV sensitivity | high acc.; car body sensitive to mech.-property EOV |
| Fernandes 2026a (ASCE) | classify scour levels | CNN (+ speed channel) | Bayesian-opt CNN; car body + front bogie | confusion matrices, boxplots | speed → ~100% on car body |
| Fernandes 2026b (LATAM) | classify scour depths (real Canelas bridge) | CNN–LSTM | Bayesian-opt **per sensor config** | multi-run stats | consistent acc. across sensors |
| **This work** | **continuous multi-output scour regression** | **RAW/PAA × modular {Time2Vec-style position encoding, LSTM, fixed-width multi-rate pooling}** | **architecture-family comparison + response-channel-budget screen** | **grouped development adjudication, then separate post-freeze sealed-test stability** | **two-channel input compared with full eight-channel input; simulated-response scope** |

---

## The delta, in prose (drop-in paragraph)

> Collectively, the drive-by scour literature — and the Fernandes line in particular
> [2024; 2025; 2026a; 2026b] — has established that (i) vehicle-borne accelerations carry a
> detectable foundation-scour signature, (ii) PAA is an effective dimensionality-reduction
> and EOV-suppression step, and (iii) supervised CNN (and CNN–LSTM) models, tuned by
> Bayesian optimization and assessed over repeated runs, classify scour reliably under
> environmental and operational variability, including on a field-calibrated bridge. These
> works settle *feasibility*. They do not, however, resolve which architecture
> is right for this signal or how performance changes with the **modeled
> response-channel budget**. This simulation does not determine how few
> on-board sensors are needed: two candidate rows are virtual
> moving-coordinate rail samples. Prior models
> fix an architecture (chosen as a black box by Bayesian optimization) and compare a small
> set of sensor positions (typically the car body and front bogie of the leading vehicle).
> This paper instead (a) runs a **component-level architecture ablation** that isolates the
> contribution of each block — Time2Vec-style spatial-coordinate encoding,
> recurrence (LSTM), and
> **N-HiTS multi-rate pooling** — and argues the last as a *physics-matched inductive bias*
> mirroring the two-timescale drive-by signal (high-frequency
> rail/contact-path response vs. the low-frequency modal sag of stiffness
> loss); (b) conducts a systematic **response-channel-budget ablation** over
> all eight channels—three vehicle vertical accelerations, two virtual
> Eulerian rail-field samples, and three vehicle pitch rates—using
> single-channel importance, leave-one-out, best pairs, and a forward sweep,
> and compares a **two-channel input with the full eight-channel array**; and
> (c) adjudicates candidates using prospectively seeded repeated grouped folds
> on development states, freezes the selected vector, and reserves a distinct
> predeclared multi-seed refit set on the sealed outer test for report-only
> stability. Finally, we frame the problem
> as **fine ordinal quantification** of scour severity (1% steps, MSE preserving the damage
> ordering) rather than coarse discrete classification, giving the downstream digital twin a
> continuous state to track.

---

## Honesty guardrails (so the delta survives review)

- **Do NOT claim** Fernandes lacks robustness testing, EOV handling, PAA, or sensor
  comparison — they have all of these (boxplots, confusion matrices, EOV sensitivity,
  track-profile variation, car-body-vs-bogie). Our delta is the **systematic** character:
  component-level ablation + full response-channel-budget study + physics framing + ordinal metric +
  collapse-rate — not "we do X and they don't."
- Candidate methodological elements are the N-HiTS-inspired pooling arm, the
  Space2Vec-style block, the full-versus-two-channel response-budget comparison,
  and collapse-rate/UCB95 selection. Treat none as a successful finding or
  hardware-economy result until authenticated R11 outputs exist.
- 2025's EOV set already includes **track profile** — so do not present "profile as an EOV"
  as our novelty (that belongs to Paper 2's *degradation-as-EOV*, which is different).
- Verify the exact scour-class definitions and passage counts if you cite specific numbers;
  the abstracts above are the confirmed claims.

---

## Roughness, suspension filtering, and response-channel fusion (v2 strand — grounds §4.4/§5.5/§6)

From the deep-research report `papers/Drive-By Scour ML Literature Design.{md,docx}` (NotebookLM/
Gemini synthesis; treat per-number claims with the usual verify-before-cite discipline — the
report's own citations are listed there). This strand supports the EOV design and the
roughness/fusion result; fold the confirmed items into §3 and cite the primary sources.

- **Published drive-by ML holds the profile FIXED per scenario, not redrawn per passage.**
  Locke et al. (2020, *J. Sound & Vibration*) use one fixed high-resolution roughness realisation
  and report robustness "provided the roughness profile remains constant"; the NuBe-DBBM
  benchmark (Sarwar & Cantero) uses fixed predefined profiles (P00/PA1/PA2); Fernandes fixes an
  FRA class per scenario with per-passage speed/mass/suspension variation. ⇒ our **per-STATE**
  profile draw is the literature-consistent choice; the first pilot's per-passage class+phase
  redraw is the outlier (and the collapse cause).
- **Physical pass-to-pass profile persistence.** The fixed-scenario literature
  above and MGT-scale settlement models (Sato; Shenton) support holding the
  physical profile over repeated passages. EN 13848-2 repeatability is a
  measurement-system property and is not evidence for physical persistence.
- **FRA class benchmark.** FRA class 4 is the campaign's deliberately rough
  controlled benchmark, not “the roughest legally permissible” at every speed;
  that legal claim depends on track class and service category. Classes 5–6
  represent progressively tighter geometry. State the benchmark choice and its
  limitation rather than inferring a route's legal/empirical class.
- **Suspension filtering versus the implemented moving-rail channel.** Sprung
  masses (car body ~1–3 Hz, bogie) are low-pass isolated from the rail. However,
  channels 3/4 are not unsprung or axle-box accelerations: they are
  (N(x_w)^T A_{rail}), the Eulerian rail FE acceleration sampled at moving
  wheel coordinates. Axle-box monitoring literature therefore cannot directly
  explain their behavior. Any robustness contrast must be reported empirically
  as vehicle-channel versus virtual moving-rail-channel behavior.
- **Two-axle / TSD residual fusion.** OBrien, Keenahan et al. provide relevant
  background for actual axle responses, but they do not validate the implemented
  `[1,3]` comparator. A profile-reference interpretation for front-bogie
  acceleration plus the virtual moving-rail sample must be demonstrated from
  these equations/results, not inferred from axle-box or TSD precedent.
- **Scour drive-by with roughness.** Prendergast/Fitzgerald/Malekjafarian wavelet-ODS scour work
  averages over repeated passes on the *same* profile to pull the signature out; most drive-by
  scour work is **classification**, not continuous regression ⇒ reinforces the regression-under-
  roughness difficulty we report.

*Guardrail:* do not over-claim novelty on "roughness matters" (well known),
and do not call the observed ranking a sprung-versus-unsprung sensor inversion.
The defensible contrast is between modeled vehicle responses and virtual
moving-coordinate rail-field samples within the staged design.
