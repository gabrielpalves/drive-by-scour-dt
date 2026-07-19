# Multi-damage bundle v7 — the remaining campaign (2026-07-12 EOV redesign)

> UPDATE 2026-07-12: Stage 1 is DONE + ANALYSED (bearing-head CONFIRMED).
> The first Stage-2 run is DEPRECATED — it was generated with the pre-fix
> L=100 geometry AND the old per-passage EOV draws, which collapsed every
> sprung channel (see `docs/framework_rationale.md`, Stage-2 forensics entry).
> The EOV design was revised per the deep-research report
> (`papers/Drive-By Scour ML Literature Design`): **persistent conditions
> (crack, profile realization, track layer) are now drawn once per damage
> STATE and held for its 50 passages**; profile class is FIXED at FRA 4 with
> 0.5 mm per-passage jitter; crack prevalence is 0.25 (was 1.0/passage).
> New folder tags: `crackST`, `prof-psd_fraST`, `trackEOVST`.

Data generator (MATLAB) + ablation (Python), kept together for Google Drive.
Extract **into the existing repo root on the Lab PC** (the folder with `data/`).
No data included — generated folders, `scour_MATLAB/Results/`, and finished
`database/` + `results/` are untouched (they make the resume work).

## Architecture policy (unchanged)
Champion selected at Stage 0 and FIXED: `CHAMPION_ARCH = "PAA_NHiTS"`. All
remaining stages run champion-only, one factor at a time.

## Ablation grid (NEW 2026-07-12)
`N_TRIALS = 100` (TPE startup auto-scales to 25 random trials) and
`SEEDS = [42, 1337, 2026]` — 3 independent Optuna runs per config; the
leaderboard gets one row per seed plus a seed-aggregated
`leaderboard_median.csv` (median + IQR; the paper-facing table). The train/val
split is fixed (random_state=42), so seed spread = init/HPO variance only.
`EXTRA_PAIRS` lets a stage run designed pairs besides the auto top-2.
**Warning:** re-running an already-finished stage with this grid EXTENDS its
studies (+25 trials each) and trains 2 extra seeds — do that only deliberately
(see "optional Stage-1 extension" below).

## Run order

### 0. NOW, Python (runs in parallel with any MATLAB job) — mixed-pair pilot
```
python pilot_stage2_L100_mixed_pairs.py
```
Trains ONLY two new pairs on the existing L100 Stage-2 data (75 trials, seed
42 — matched to the old budget): Wheel1_Vert+FrontBogie_Vert and
Wheel1_Vert+CarBody_Vert — the TSD-residual fusion hypothesis (wheel = profile
reference, sprung = inertial response). Every old study is skipped. Summary →
`results/stage2_4span_L100pilot_summary/` (original summary untouched).
Question answered: does wheel+sprung beat wheel+wheel (126 scour MSE) under
heavy roughness EOV?

### 1. NOW, MATLAB — `smoke_stage3.m` once (a few minutes)
Must print healthy parity `max|A-B| = 0`. Required before the Stage-3
generation; run it now so it never blocks the queue.

### 2. MATLAB — generate `stage1_crack` (BRIDGE-damage EOV stage)
`A00_Run.m` ships with `STAGE = 'stage1_crack'`: L60/3-span, scour [2,3] +
bearing target + **per-STATE crack** (p=0.25, EI-loss U(0.05,0.30), location
U(0.1,0.9)·L), profile FIXED. This is the Fernandes-comparable stage (scour +
bearing + crack) and isolates the crack effect from the profile effect.
Expected folder:
```
L60_3span_multi_scour_scourS2-3_bearTGT_crackST_dano0-60pct_states267_Npass50_varNVST
```
(`states267` = 17 anchors + 250 LHS — CONFIRM against what A00 writes.)

### 3. MATLAB — generate `stage1_full` (adds the RAIL-side EOV)
Flip `STAGE = 'stage1_full'`: stage1_crack + `psd_fra` profile with **one
FRA-class-4 realization per state** (phases locked via a per-state seed in
B19) + 0.5 mm additive per-passage jitter. This is the Stage-2 collapse
ATTRIBUTION run on the known-good L60 geometry. Expected folder:
```
L60_3span_multi_scour_scourS2-3_bearTGT_crackST_prof-psd_fraST_dano0-60pct_states267_Npass50_varNVST
```

### 4. MATLAB — REGENERATE Stage 2 at L=99.6 (revised EOVs)
Flip `STAGE = 'stage2_4span'`: L=99.6 m / 4 spans of 24.9 m (all 5 supports
EXACTLY on mesh nodes — do NOT change L_bridge; L=100 re-meshes to 99.9 m with
a floating-point tie at the middle support, which is precisely what the
deprecated first run used), scour piers [2 3 4] + bearing + revised EOVs.
Expected folder:
```
L99.6_4span_multi_scour_scourS2-3-4_bearTGT_crackST_prof-psd_fraST_dano0-60pct_states271_Npass50_varNVST
```
(`states271` = 21 anchors + 250 LHS.)

### 5. MATLAB — Stage 3 all-damage (last in the queue)
Flip `STAGE = 'stage3_alldamage'`: stage1_full + track-layer damage (**now
per-STATE**: ballast patches, hanging-sleeper groups, pad aging/failures) +
wheel OOR/flats (still per-PASSAGE = a different train of the fleet each
passage). Expected folder:
```
L60_3span_multi_scour_scourS2-3_bearTGT_crackST_prof-psd_fraST_trackEOVST_oorON_dano0-60pct_states267_Npass50_varNVST
```

### 6. Python — ablate each stage as its data lands
Flip `STAGE` in `comprehensive_ablation_multidamage.py` to
`stage1_crack` → `stage1_full` → `stage2_4span` → `stage3_alldamage` (champion
only, 100 trials × 3 seeds; results → `results/<STAGE>_summary/`). Read each
`scour_mse` against the chain baselines (Stage-0 pair 0.757; Stage-1 pair
1.798) and the `disentanglement.csv` leak columns. If a stage inflates badly,
re-run its champion config with `n_segments` 1024/2048 BEFORE blaming the EOVs
(PAA-resolution contingency).

### Optional — Stage-1 consistency extension
Re-running `STAGE = "stage1_bearing"` with the new grid extends every Stage-1
champion study to 100 trials and adds seeds 1337/2026 (S2V studies stay
untouched — champion-only gate). Improves the paper's architecture-consistency
table; run it whenever the PC has idle time. The published 75-trial numbers
stay reproducible from the committed database.

## Requirements
`torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn,
PyWavelets, tqdm` (the env that ran Stage 0). Everything is resumable —
re-running any stage continues where it stopped.
