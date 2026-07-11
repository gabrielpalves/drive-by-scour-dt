# Paper 1 — Outline & Draft Skeleton

**Working title:** *A physics-matched deep-learning architecture and sensor-economy
ablation for drive-by scour identification in railway bridges*

**Status:** venue-agnostic skeleton (2026-07-09). Port to the journal LaTeX template
once the target is confirmed. Grounded in the completed single-scour ablation
(`ablation_analysis/summary_per_config.csv`, `presentation/SHM_Markdown/main.md`,
figures in `ablation_analysis/figures/`).

**Target venue:** TBD (confirm with advisor). Ranked fit — MSSP (top; signal-processing +
drive-by community) → Structural Health Monitoring (Sage) → CACIE (ML angle) →
Engineering Structures (application) → Sensors (fast fallback). Framing knob: MSSP wants
the signal/physics novelty foregrounded; CACIE the ML; Eng. Struct. the application.

**Scope fence (Todd "don't go too crazy"):** single foundation, scour only, 40 m bridge.
Multi-foundation / multi-damage / the digital twin are **Paper 2** — referenced here only
as future work.

---

## 1. Contributions (state these explicitly in the intro)

1. **A physics-matched architectural inductive bias.** N-HiTS multi-rate pooling mirrors
   the two-timescale physics of the drive-by signal — high-frequency wheel/rail-contact
   transients vs. the low-frequency modal sag induced by foundation stiffness loss — and
   is shown to be the consistent top performer across sensor sets. This reframes the
   architecture choice as *physics*, not hyperparameter luck.
2. **A sensor-economy result: 2 ≈ 8.** A systematic single-DOF → leave-one-out → pair →
   forward-sweep ablation shows a 2-sensor pair matches the full 8-DOF array
   (median MSE 0.414 vs 0.386; both zero collapse), and identifies *which* channels carry
   the scour signature and which are dead weight.
3. **A robustness-based model-selection criterion.** Ranking by a 30-seed
   UCB95 (`MSĒ + 1.96 σ/√30`) plus a **collapse-rate**, rather than the single luckiest
   Optuna trial — rewards models that are *reliably* good, and exposes fragile
   single-sensor configurations a point estimate would hide.

> Positioning vs. Fernandes (the prior art): they establish drive-by scour *feasibility*;
> we answer *which architecture* and *how few sensors*, with a physics argument and a
> robustness ranking. See the delta table in §3.

---

## 2. Introduction (draft beats)

- Scour = leading cause of bridge failure worldwide (>50–60% of collapses), invisible
  (submerged), accelerating with climate-driven flood frequency; visual inspection misses
  early subsurface erosion between cycles. [cite Kamariotis 2024; HEC-18; Lamb 2019]
- Direct SHM (structure-mounted sensors) is costly to deploy per-bridge on a network;
  **drive-by / indirect** SHM turns an in-service train into a mobile sensor, scalable
  across a line. [cite Braganca 2023 review; Souza 2023 review; Fernandes 2024–2026]
- Gap: drive-by scour *detection* is established, but two practical design questions are
  under-explored — (i) *which* deep-learning architecture is right for this signal, and
  why; (ii) *how few* on-board sensors suffice, and which. Existing work fixes a CNN and
  studies sensor *placement* qualitatively; none runs an architecture×sensor ablation with
  a robustness-based selection.
- Contributions (the three above). Roadmap sentence.

---

## 3. Related work + the Fernandes delta table

**VERIFIED DRAFT: see `docs/paper1_related_work.md`** — per-paper summaries (2024 JCSHM DAE;
2025 IJSSD multi-damage PAA+CNN; 2026a ASCE speed→~100% car body; 2026b LATAM CNN-LSTM on
the real Canelas bridge), the verified Table 1, the drop-in delta paragraph, and honesty
guardrails. All four checked against the PDFs with `pdftotext` on 2026-07-09.

Narrative: direct vs indirect SHM; vibration-based scour indicators (natural-frequency drop
with scour — Prendergast & Gavin ~40% at 10 m); the Fernandes line as the closest prior art
(same UFSC/Porto group — frame as the foundation we build on).

**Delta in one line:** Fernandes establishes *feasibility* (PAA, CNN/CNN-LSTM, Bayesian opt,
multi-run robustness, EOVs incl. track profile, car-body-vs-bogie). Our advances are the
**component-level architecture ablation** (isolating Space2Vec / LSTM / **N-HiTS pooling as
two-timescale physics**), the **full sensor-economy ablation → 2 ≈ 8**, the **30-seed UCB95 +
collapse-rate** selection, and **fine ordinal quantification** (vs coarse classes). Do NOT
claim they lack robustness/EOV/PAA/sensor comparison — they have all four; our delta is the
*systematic* character. (Full table + guardrails in `paper1_related_work.md`.)

---

## 4. Methodology

**FULL DRAFT PROSE: see `docs/paper1_methodology.md`** — expanded §4.1–4.8 (pipeline
overview, TTBI model with the verified bridge/vehicle/track params, scour model,
EOV+datasets, PAA preprocessing, the modular architecture with the N-HiTS-as-physics
argument, Optuna TPE, and the UCB95 + collapse-rate selection), with a
verify-before-submission checklist. The beats below are the summary.

**4.1 Physical model — TTBI (train–track–bridge interaction).**
- Cantero 2-D TTBI (VEqMon2D); 40 m bridge; vehicle model; rail-track model (Zhai
  layered pad/sleeper/ballast). [cite Cantero 2022, Cantero 2-D TTBI]
- Scour = vertical foundation stiffness loss; healthy kv = 3.44e5 kN/m; damage =
  (1−rate)·kv, rate ∈ [0, 60%]. Single central foundation.
- **Scope note (report honestly):** one fixed measured rail profile (`Profile.Type=2`) —
  rail-profile irregularity is *not* an operational-variability source in this dataset;
  documented limitation, resolved in Paper 2 via PSD-regenerated profiles.

**4.2 Datasets & operational variability.**
- Two datasets: (D1) noise + temperature + vehicle-property variability; (D2) D1 + train
  speed per passage (the harder set).
- 61 damage cases (0–60% in 1% steps); 200 passages per case. Eight vehicle DOFs:
  vertical accel on car body / front bogie / rear bogie / 2 wheels; pitch rate on car body
  / front bogie / rear bogie.

**4.3 Signal preprocessing.**
- PAA (piecewise aggregate approximation, 512 segments) as a structural low-pass filter:
  smooths high-frequency rail-corrugation noise, preserves the macro deflection-basin /
  stiffness-loss signature. Per-channel standardisation. (CWT scalogram arm as the 2-D
  comparison.) [cite Fernandes 2025 for PAA-as-EOV-filter precedent]

**4.4 The modular architecture (the ablated object).**
- Shared 1-D CNN backbone with three optional blocks:
  - **Space2Vec** — learnable spatial embedding grounding the signal to bridge position.
  - **LSTM** — bidirectional temporal context.
  - **N-HiTS multi-rate pooling** — the physics-matched block (multi-timescale sub-sampling
    → high-freq wheel impacts *and* low-freq modal sag). **This is the inductive-bias claim.**
- Ordinal head: 61 outputs, MSE on the class index preserves the physical damage ordering
  (a 41% vs 42% confusion is cheap; 5% vs 55% is not). [Figure: CNN.png]

**4.5 Hyperparameter optimisation.**
- Optuna, multivariate TPE (handles the conditional search space as blocks appear/vanish),
  25% random startup, up to ~26 hyperparameters, 200 trials/config.

**4.6 Robustness-based selection (a methodological contribution).**
- Each Optuna champion re-trained over **30 seeds**; rank by
  **UCB95 = MSĒ + 1.96·σ/√30** and report the **collapse-rate** (fraction of seeds whose
  MSE blows past a physical tolerance). Rewards *reliably* good models; flags fragile
  single-sensor configs (e.g. front-bogie-pitch-alone). [uses `optuna_best.csv`,
  `summary_per_config.csv`]

---

## 5. Results

**5.1 Architecture ablation (8-DOF).** [Figs: fig1_architecture_full8dof.png;
architecture_ablation_*_MSE_{ntv,all}.png]
- PAA ≫ RAW (RAW discards nothing useful but keeps chaotic noise → worse).
- Among blocks, **N-HiTS is the consistent contributor**; the PAA+N-HiTS family is the
  reliable top performer. LSTM helps most in the single-sensor regime (see 5.2).
- Report the median-MSE ranking with IQR and collapse-rate.

**5.2 Sensor sensitivity.**
- **Single-DOF importance** [fig2_single_dof_importance.png]: best single channel =
  **CarBody_Pitch** (median MSE 0.543), then RearBogie_Vert (0.703); wheels and
  **FrontBogie_Pitch collapse** (unusable alone — collapse-rate up to ~0.83). Pitch on the
  car body carries the cleanest scour signature; the front-bogie pitch is actively harmful.
- **Leave-one-out** [fig3_leave_one_out.png]: dropping **CarBody_Vert** *improves* the
  7-DOF set (0.345 < full-8 0.386) — it adds noise, not signal. Physical read: redundant
  with the pitch channels.
- **Best pairs** [fig4_best_pairs.png]: **RearBogie_Vert + CarBody_Pitch → 0.414**, matching
  the full 8-DOF (0.386) at zero collapse → **the 2 ≈ 8 headline**. Chosen over the
  marginally-similar CarBody_Pitch + FrontBogie_Pitch because *both* of its single-sensor
  fall-backs remain usable (FrontBogie_Pitch alone collapses) — resilience, not just mean.
- **Forward sweep / focused combos** [cross_sweep_lines.png; focused_dof_arch_ablation.png]:
  ~3 sensors saturate; beyond that, diminishing returns.

**5.3 Summary table.**

| Configuration | n DOF | median MSE | collapse | note |
|---|---|---|---|---|
| Full 8-DOF (PAA+N-HiTS) | 8 | 0.386 | 0.00 | reference |
| Best LOO (drop CarBody_Vert) | 7 | 0.345 | 0.00 | beats full-8 |
| **Champion pair (RearBogie_Vert + CarBody_Pitch)** | **2** | **0.414** | **0.00** | **2 ≈ 8** |
| Best single (CarBody_Pitch) | 1 | 0.543 | 0.01 | usable solo |
| FrontBogie_Pitch alone | 1 | ~423 | ~0.83 | collapses |

(MSE in ordinal class-index units ≡ % scour; champion RMSE ≈ 0.64% scour. Verify numbers
against the CSV at write-time.)

---

## 6. Discussion

- **Why the pooling helps** — tie N-HiTS multi-rate pooling back to the two-timescale
  physics; contrast with why a plain CNN or CWT under-performs. This is the paper's
  intellectual core.
- **Operational implication** — a lightly instrumented in-service train (2–3 sensors) is
  enough; connects to the real constraint (bridges are a small, outsourced fraction of a
  line — the Ferrovia Tereza Cristina anecdote as motivation).
- **Robustness vs. point estimates** — the collapse-rate story: some single-sensor configs
  look fine on the lucky trial but are unusable in deployment.
- **Limitations** — single foundation; single fixed rail profile (no profile-EOV);
  simulation-only (no field data); ordinal-classification vs continuous regression.
- **Forward link** — multi-foundation regression, dependent scour, and the value-of-SHM
  digital twin are Paper 2.

---

## 7. Conclusion
- Restate the three contributions with the headline numbers. One sentence on Paper 2.

---

## Figure inventory (existing → paper)

| Paper fig | Source (in `ablation_analysis/figures/` or talk) | Redraw? |
|---|---|---|
| Concept: TTBI + scour | scour.png, TTBI_Cantero.png | keep now, redraw later |
| Example signals | signals.png, pitch_signals.png | keep |
| Architecture schematic | CNN.png | keep |
| F1 Architecture ablation (8-DOF) | fig1_architecture_full8dof.png | keep (font later) |
| F2 Single-DOF importance | fig2_single_dof_importance.png | keep |
| F3 Leave-one-out | fig3_leave_one_out.png | keep |
| F4 Best pairs | fig4_best_pairs.png | keep |
| (opt) Forward sweep | cross_sweep_lines.png | keep |
| (opt) Champion confusion matrix | from `plotting/confusion.py` | generate if wanted |

*User note (2026-07-09): several figures need font-size/label redraws — deferred; data is
locked, so redraws are trivial later. Keep current figures for the draft.*

---

## Open items before submission
- Confirm target venue with advisor → I scaffold the matching LaTeX template.
- Verify every Table-1 (Fernandes) cell against `papers/` (exact classes, sensor sets).
- Re-pull the exact numbers in §5.3 from `summary_per_config.csv` at write-time.
- Decide whether to add a champion confusion matrix (5.1/5.2) and an S2V/LSTM per-sensor
  panel.
- Redraw figures to publication font sizes.

## Reference shortlist
Cantero (2022, VEqMon2D; 2-D TTBI); Kamariotis et al. (2024); Fernandes et al.
(2024, 2025, 2026); Braganca et al. (2023 review); Souza et al. (2023 review);
Prendergast & Gavin (scour–frequency); HEC-18; Lamb et al. (2019). Full PDFs in `papers/`.
