# Paper 1 — Related work: the Fernandes line (verified against the PDFs)

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
| **This work** | **fine ordinal scour quantification (61 cls @1%, MSE)** | **PAA/CWT × modular {Space2Vec, LSTM, N-HiTS}** | **component-level architecture ablation + full sensor-economy ablation (single→LOO→pair→sweep, 8 DOFs)** | **30-seed UCB95 + collapse-rate** | **2 sensors ≈ 8; physics-matched N-HiTS pooling** |

---

## The delta, in prose (drop-in paragraph)

> Collectively, the drive-by scour literature — and the Fernandes line in particular
> [2024; 2025; 2026a; 2026b] — has established that (i) vehicle-borne accelerations carry a
> detectable foundation-scour signature, (ii) PAA is an effective dimensionality-reduction
> and EOV-suppression step, and (iii) supervised CNN (and CNN–LSTM) models, tuned by
> Bayesian optimization and assessed over repeated runs, classify scour reliably under
> environmental and operational variability, including on a field-calibrated bridge. These
> works settle *feasibility*. They do not, however, resolve two design questions a network
> operator faces before deployment: **which architecture** is right for this signal and
> **why**, and **how few on-board sensors** — and which — are actually needed. Prior models
> fix an architecture (chosen as a black box by Bayesian optimization) and compare a small
> set of sensor positions (typically the car body and front bogie of the leading vehicle).
> This paper instead (a) runs a **component-level architecture ablation** that isolates the
> contribution of each block — spatial embedding (Space2Vec), recurrence (LSTM), and
> **N-HiTS multi-rate pooling** — and argues the last as a *physics-matched inductive bias*
> mirroring the two-timescale drive-by signal (high-frequency wheel–rail transients vs. the
> low-frequency modal sag of stiffness loss); (b) conducts a **systematic sensor-economy
> ablation** over all eight vehicle DOFs — single-DOF importance, leave-one-out, best pairs,
> and a forward sweep — showing a **two-sensor configuration matches the full eight-DOF
> array**; and (c) ranks every candidate by a **30-seed upper-confidence bound plus a
> collapse-rate**, rewarding models that are *reliably* good and exposing single-sensor
> configurations that a point estimate would wrongly endorse. Finally, we frame the problem
> as **fine ordinal quantification** of scour severity (1% steps, MSE preserving the damage
> ordering) rather than coarse discrete classification, giving the downstream digital twin a
> continuous state to track.

---

## Honesty guardrails (so the delta survives review)

- **Do NOT claim** Fernandes lacks robustness testing, EOV handling, PAA, or sensor
  comparison — they have all of these (boxplots, confusion matrices, EOV sensitivity,
  track-profile variation, car-body-vs-bogie). Our delta is the **systematic** character:
  component-level ablation + full sensor-economy + physics framing + ordinal metric +
  collapse-rate — not "we do X and they don't."
- The genuinely novel elements are: **N-HiTS pooling (and its physics rationale)**, the
  **Space2Vec block**, the **2 ≈ 8 sensor-economy result**, and the **collapse-rate/UCB95
  selection**. Lead with those.
- 2025's EOV set already includes **track profile** — so do not present "profile as an EOV"
  as our novelty (that belongs to Paper 2's *degradation-as-EOV*, which is different).
- Verify the exact scour-class definitions and passage counts if you cite specific numbers;
  the abstracts above are the confirmed claims.
