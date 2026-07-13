# Multi-damage bundle — the remaining campaign (Stage 2 + Stage 3)

> UPDATE: the Stage-0 LSTM completion (steps 2–3 below) is DONE — champion
> CONFIRMED = PAA_NHiTS (LSTM hurt the pair +158%). Keep `CHAMPION_ARCH =
> "PAA_NHiTS"`. Remaining work: Stage-2 data gen (this PC), Stage-3 data gen
> (the second Lab PC), then the champion-only ablations as data lands.

Data generator (MATLAB) + ablation (Python) for everything left, kept together for
Google Drive. Extract **into the existing repo root** (the folder with `data/`). No data
included — your generated folders, `scour_MATLAB/Results/`, and the finished Stage-0
`database/` + `results/` are untouched (they are what makes the resume work).

## Architecture policy (encoded in the script)
**Select ONCE at Stage 0, then FIX.** Stage 0 runs **all 3 arms** — PAA+N-HiTS,
PAA+S2V+N-HiTS, and the newly enabled **PAA+LSTM+N-HiTS** (CNN → LSTM → multi-rate pool,
LSTM size Optuna-tuned). Stage 1/2 then run **`CHAMPION_ARCH` only**, so each later stage
varies one factor at a time (bearing head at Stage 1; scale + EOV at Stage 2), never the
backbone too. The gate lives at the top of `comprehensive_ablation_multidamage.py`.

## Run order

### 1. NOW, MATLAB — generate Stage-2 data (long; runs in parallel with step 2)
`scour_MATLAB/A00_Run.m` ships with `STAGE = 'stage2_4span'`: **L=99.6 m / 4 spans of
24.9 m**, scour piers **[2 3 4]**, `bearing_mode='target'`, crack + FRA-PSD profile EOV
ON. Output → `scour_MATLAB/Results/<case>/`; move/copy under `data/` when done. Expected
folder:
```
L99.6_4span_multi_scour_scourS2-3-4_bearTGT_crackON_prof-psd_fra_dano0-60pct_states271_Npass50_varNVST
```
(`states271` = 21 anchors + 250 LHS — CONFIRM against what A00 writes.)

**Mesh-node check DONE (2026-07-09): L=99.6 was chosen instead of 100 precisely so all 5
supports land EXACTLY on mesh nodes** (24.9 m span = 83 × 0.3 m elements; verified in
Python replicating B43/B01/B02). A nominal L=100 would be re-meshed to 99.9 m with
snapped spans 24.9/24.9/25.2/24.9 and a floating-point tie at the middle support. Do not
change `L_bridge` without re-checking grid-exactness (L must be a multiple of
4 × 1.2 m for four equal on-node spans).

### 2. NOW, Python — complete the Stage-0 architecture table (resumable)
```
python comprehensive_ablation_multidamage.py
```
Ships with `STAGE = "stage0_multiscour"`. The pipeline skips every study that already has
its 75 trials, so the 18 finished N-HiTS/S2V studies are NOT retrained — **only the 9 new
PAA_LSTM_NHiTS studies run** (8 single-DOF + the champion pair; the pair choice is ranked
by PAA_NHiTS and stays RearBogie_Vert+CarBody_Pitch). Needs the Stage-0 `database/`,
`results/MD0_*`, and `data/…states259…` folders already on this PC.

New summary → `results/stage0_multiscour_summary/` (`leaderboard.csv` now 27 rows = 3
arms × 9 configs, `parity_best.png`). The old 2-arm summary at `results/MD0_summary/` is
superseded.

### 3. Set the champion (one line)
Read the new leaderboard. If PAA_NHiTS still wins (expected — Stage 0 is already at
~0.87 pp RMSE, near the noise floor), leave `CHAMPION_ARCH = "PAA_NHiTS"`. Only if the
LSTM arm wins, set `CHAMPION_ARCH = "PAA_LSTM_NHiTS"` — and treat a *marginal* win with
caution (heaviest arm, single seed, no MC).

### 4. When Stage-1 data lands — Stage-1 ablation (champion only)
Flip `STAGE = "stage1_bearing"` (one line; DATASET/TARGETS/BEARING_TARGETS set
automatically — CONFIRM the `states267` folder name). Runs the champion only (4 heads:
scour ×2 + bearing ×2). Results → `results/stage1_bearing_summary/`:
- `leaderboard.csv` — `scour_mse` vs `bearing_mse` + per-head MSE.
- `disentanglement.csv` — **`scour_leak_on_bearing_%`** (ideal ~0; the dangerous case)
  and `bearing_leak_on_scour_%`.
- Read against the Stage-0 baseline (champion pair agg MSE 0.757): flat `scour_mse` +
  low leak = bearing disentangles cleanly.

### 5. When Stage-2 data lands — Stage-2 ablation (champion only)
Flip `STAGE = "stage2_4span"`. Runs the champion only (5 heads: scour ×3 + bearing ×2)
through the same turnkey flow (single-DOF sweep → auto-pair → pair). Results →
`results/stage2_4span_summary/` (same files as Stage 1, plus `mse_support_2/3/4`).
Watch: does the pair still suffice with 3 piers + ~100 m span + EOV; is the middle pier
(3) harder; does EOV inflate per-pier MSE vs the Stage-0 ~0.75. Note: with the
`psd_fra` profile the crop window is the TRUE full-bridge window (~11831 samples), not
the 5831-sample legacy window the 'fixed'-profile stages produce.

### 6. NEW — Stage 3 "all-damage" (second Lab PC; kitchen-sink robustness arm)
On the SECOND Lab PC, set `A00_Run.m` `STAGE = 'stage3_alldamage'` and run:
Stage-1 bridge/targets (L60/3-span, scour [2,3] + bearing target) + crack +
`psd_fra` profile + **track-layer damage** (ballast fouling patches, hanging-
sleeper groups spiked near the abutment transitions, rail-pad aging/failures —
per-sleeper, data-anchored sampling per `docs/track_eov_sampling_spec.md`) +
**wheel OOR/flats** (haversine dips, depth = L²/16R). All per-passage randomized
NUISANCES, logged in each file (`track_log`, `oor_log`) — never labels.

**Before the long run:** execute `smoke_stage3.m` once in MATLAB (a few minutes).
It must print `healthy parity max|A-B| = 0 (true)` — the per-sleeper code is
bit-identical when healthy (already verified end-to-end on the Python mirror;
sprung channels move only 1–18% under damage while the flat-carrying wheel
explodes ~270×, as the band analysis predicts). A momentary "no permanent
contact" solver message during flat impacts is physical and handled.

Expected folder:
```
L60_3span_multi_scour_scourS2-3_bearTGT_crackON_prof-psd_fra_trackEOV_oorON_dano0-60pct_states267_Npass50_varNVST
```
Then ablate with `STAGE = "stage3_alldamage"` (champion-only, 4 heads; results →
`results/stage3_alldamage_summary/`). Read `scour_mse` against the Stage-1
baseline: flat = invariance demonstrated; inflated = re-run the champion config
with `n_segments` 1024/2048 BEFORE blaming the EOVs (PAA-resolution contingency).

## Requirements
`torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn, PyWavelets,
tqdm` (the env that ran Stage 0). Earlier bundle fixes included (auto-create
`database/`, UTF-8/ASCII-safe logs). Everything is resumable — re-running any stage
continues where it stopped.
