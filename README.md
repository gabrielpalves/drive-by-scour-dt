# Drive-by Digital Twin for Railway-Bridge Scour

A physics-informed **Digital Twin (DT)** that monitors **scour** (and, later,
bearing) damage on railway bridges from **drive-by** vibration — sensors on the
*passing train*, not on the structure. The project has two parts:

1. **Ablation study** *(Paper 1, complete)* — which neural architecture and which
   vehicle sensors best classify scour from drive-by signals.
2. **DT framework** *(Paper 2, in progress)* — evolve the bridge over its life
   (gradual + flood-shock scour), estimate its state each passage, and choose
   maintenance actions to minimise the expected **monetary** life-cycle cost
   (Value-of-SHM).

Everything is designed to be **toggled from a single place** so scenarios and
methodologies can be switched on/off and compared.

---

## Pipeline at a glance

```
 TTBI-2D physics  ──►  champion CNN classifier  ──►  Bayes belief filter  ──►  planner  ──►  € cost
 (damaged bridge)      (PAA + N-HiTS)               (DBN / discrete filter)   (decision)     (Value-of-SHM)
        ▲                                                                         │
        └────────────── scour evolution (Kamariotis gradual + flood shock) ◄──────┘
```

---

## Repository structure

| Path | What it is |
|------|------------|
| `TTBI_2D/` | Train–Track–Bridge Interaction physics engine (Cantero 2022, Python port). |
| `TTBI_2D/damage_config.py` | **Single toggle point** for bridge geometry + all damage types (scour / bearing / crack / rail-profile). |
| `core/` | CNN model builder, preprocessing (PAA/CWT/…), dataset/scaler utilities. |
| `training/` | Offline ablation pipeline (Optuna HPO, champion export). |
| `comprehensive_ablation_PC_{7,11}.py` | Drivers that ran the 540-run ablation (split across two PCs). |
| `plotting/aggregate_ablation.py` | Builds the master tables + the winner-only ablation figures. |
| `plotting/presentation_plots.py` | Boxplots, confusion matrices, all-architecture sweeps. |
| `digital_twin/assets.py` | `ScourModel` (gradual + flood shock), `PhysicalAsset`, `DigitalAsset`. |
| `digital_twin/scour_multi.py` | `MultiScourModel` — correlated multi-foundation scour (Gaussian copula). |
| `digital_twin/costs.py` | `CostModel` — verified Kamariotis € figures + fragility. |
| `digital_twin/planner.py` | Planners: threshold, `CostBenefitPlanner`, `HeuristicPlanner`, `POMDPPlanner`, `HybridPlanner`. |
| `digital_twin/simulation.py` | `DTSimulator` — the toggleable belief-filter + decision + € accounting loop. |
| `digital_twin/harness.py` | Value-of-SHM comparison across planners (mock or live). |
| `digital_twin/calibrate.py` | Calibrate `p_advance`; fragility sensitivity sweep. |
| `digital_twin/dbn.py` | pgmpy Dynamic Bayesian Network (Torzoni-style belief filtering). |
| `run_dt.py` | **DT entry point** — edit the CONFIG block and run. |
| `models/champion_PAA_NHiTS_full8dof/` | The committed champion classifier (weights, scaler, metadata, confusion). |

Heavy / local-only artefacts (`PC7/`, `PC11/`, `Torzoni2024/`, `Torzoni2026/`,
`data/`, `papers/`, `ablation_analysis/`, `presentation/`) are **git-ignored**.

---

## Environment

A `.venv` (Python 3.14) with `torch` (CUDA), `scipy`, `pgmpy`, `scikit-learn`,
`pandas`, `matplotlib`, `joblib`, `PyWavelets`. TensorFlow is **not** required.

```bash
.venv/Scripts/python run_dt.py            # Windows
```

---

## Usage

### 1. Ablation (Paper 1) — already run
The study is `4 architectures × 45 DOF-sets × 3 seeds = 540 champion models`.
Re-generate the analysis tables and figures from the result trees:

```bash
.venv/Scripts/python plotting/aggregate_ablation.py     # tables + winner figures
.venv/Scripts/python plotting/presentation_plots.py     # boxplots, confusion, sweeps
```

**Headline results:** `PAA + N-HiTS` is the best and most consistent architecture;
two sensors (`RearBogie_Vert + CarBody_Pitch`) ≈ the full 8-DOF array;
`CarBody_Pitch` is the best single channel. Use **median MSE + collapse-rate**
(not mean — occasional training collapses inflate the mean 10–100×).

### 2. Digital Twin (Paper 2)
Edit the CONFIG block in `run_dt.py`, then run:

```bash
.venv/Scripts/python run_dt.py
```

It evolves the bridge (monthly, with floods), runs each selected planner, and
prints the expected discounted € life-cycle cost — the Value-of-SHM comparison
— saving `ablation_analysis/vosh_comparison.csv`.

### 3. Calibration
```bash
.venv/Scripts/python -m digital_twin.calibrate
```
Estimates the belief-filter `p_advance` from the true scour model and runs a
fragility **sensitivity sweep** (we keep failure modelling reliability-light, so
results are reported as robust across fragility assumptions rather than for one
curve).

---

## What you can toggle

**Bridge & damage** — `TTBI_2D/damage_config.py`:
```python
Beam   = configure_bridge(Beam, length=100.0, num_spans=4)        # or support_locs=[fractions]
Damage = make_damage(num_supports=5,
                     scour_rates=[0, .30, 0, .20, 0],   # per-pier vertical-stiffness loss
                     bearing_rot_stiff=[1e9, 0,0,0, 0], # abutment bearing (0 = healthy)
                     crack_locs=[55.0], crack_intensity=0.22,   # Sinha-style local EI loss
                     profile_intensity=1.0)             # rail-irregularity amplitude
```
All default to healthy, so a feature is "off" by leaving it at its default; the
legacy single-foundation, scour-only 40 m bridge is reproduced exactly.

**DT run** — `run_dt.py`:
- `MODE` — `mock` (fast synthetic truth) or `live` (TTBI + champion classifier).
- `PLANNERS` — any of `do_nothing`, `cost_vi`, `heuristic`, `pomdp`, `hybrid`.
- `DT_YEARS`, `N_STEPS` — simulation clock (default monthly, 30 years).
- `ENABLE_SHOCK` — Kamariotis flood term on/off.

**Decision layer** (orthogonal switches sharing one `CostModel`):
- *planner*: cost-optimal MDP / Kamariotis heuristic / myopic-VoI POMDP / hybrid
  (POMDP-inspect + cost-VI-repair). *Active-inference* (Torzoni 2026) is planned.
- *monitoring*: drive-by / periodic inspection / inspect-after-flood / none.
- actions: `do_nothing` / `inspect` / `repair` (scour repair is all-or-nothing).

---

## Status & known limitations

- **Ablation:** complete and analysed.
- **DT framework:** the full loop runs and is validated **in mock mode**; planners,
  scour evolution (incl. floods), copula multi-foundation scour, and € accounting
  all work and are step-size-invariant.
- **Live mode:** runs end-to-end (~2 min/passage) but the champion currently
  **mispredicts on DT-generated signals** — a train/serve skew between the Python
  `run_single_passage` and the MATLAB training-data pipeline. The intended fix is
  to **sample a held-out signal library generated by the training pipeline**
  rather than re-simulating live; this also makes fine time-stepping cheap.
- **Multi-damage:** the physics + scour generator support it; the decision loop is
  currently single-scour (61 classes) and will be extended once a multi-output
  classifier is trained.

---

## Key references
- **Cantero (2022)** — TTBI / VEqMon2D vehicle–bridge interaction.
- **Kamariotis et al. (2024)** — quantifying the value of SHM; gradual + shock scour; € costs.
- **Torzoni et al. (2024, 2026)** — DT decision frameworks (DBN; active inference).
- **Fernandes et al. (2024–2026)** — drive-by railway-bridge damage / multi-damage / scour.
- **Adnan (2026)** — scour interaction between nearby foundations (copula proximity term).
