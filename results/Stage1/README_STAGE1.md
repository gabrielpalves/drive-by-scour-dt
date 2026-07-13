# Multi-damage ablation bundle — STAGE 1 (bearing disentanglement)

Data generator (MATLAB) + ablation (Python) for **Stage 1**, kept together for Google Drive.
Extract **into the existing repo root** (the folder with `data/`). No data is included, so
your generated folders and `scour_MATLAB/Results/` are untouched.

Stage 1 adds the **abutment bearing state as a labelled regression target** alongside scour,
to answer: *when scour and bearing share one network, does bearing corrupt the scour
estimate?* (the head-vs-nuisance question).

## 1. Data generation (MATLAB) — already set to Stage 1
`scour_MATLAB/A00_Run.m` has `STAGE = 'stage1_bearing'` (bearing = joint-sampled target →
`data.bearing_vector`; no crack/profile EOV). Run it to (re)generate the dataset; output goes
to `scour_MATLAB/Results/<case>/`. Move/copy it under `data/`.

## 2. Ablation (Python)
```
python comprehensive_ablation_multidamage.py
```
`STAGE = "stage1_bearing"` at the top sets:
- `DATASET = "L60_3span_multi_scour_scourS2-3_bearTGT_dano0-60pct_states267_Npass50_varNVST"`
  — **CONFIRM this matches your actual folder** (the `states<N>` count depends on
  `n_states_multi`; with the default 250 it is 267 = 17 anchors + 250).
- `TARGET_SUPPORTS = [2, 3]`, `BEARING_TARGETS = ["left", "right"]`.

It runs the same turnkey flow (Phase 1 single-DOF → auto-pick best pair → Phase 2 pair) with
**4 heads** (scour ×2 + bearing ×2). Everything resumable.

**Results:** `results/stage1_bearing_summary/`
- `leaderboard.csv` — per model: agg MSE, localisation acc (scour heads only),
  **`scour_mse` vs `bearing_mse`**, and per-head MSE (`mse_support_2/3`, `mse_bearing_left/right`).
- `disentanglement.csv` — the Stage-1 answer: **`scour_leak_on_bearing_%`** (false scour a
  seized bearing induces on scour-only-absent states — ideally ~0, the dangerous case) and
  `bearing_leak_on_scour_%`.
- `parity_best.png` — predicted-vs-true per head (scour in %, bearing in seized %).

**How to read it:** compare `scour_mse` here to the Stage-0 scour MSE (0.414 champion pair).
Flat + low `scour_leak_on_bearing` = the network disentangled cleanly (bearing didn't corrupt
scour). Rising = bearing confounds scour → the head earns its place.

## 3. Switching stages
Both `A00_Run.m` (MATLAB) and `comprehensive_ablation_multidamage.py` (Python) have a `STAGE`
selector at the top: `stage0_multiscour | stage1_bearing | stage2_4span` (+ `stage1_eov`,
`stage1_full` in MATLAB). Keep the two in sync.

## Requirements
`torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn, PyWavelets, tqdm`
(the env that ran Stage 0). Fixes from the Stage-0 bundle are included (auto-create
`database/`, UTF-8/ASCII-safe logs).
