# Multi-damage campaign — the 10-rung ladder (2026-07-15)

Everything is **regenerated from scratch**: new ladder, raw/Option-B data format,
anchored EOVs, noise-free generation. The earlier Stage-0/1 datasets and the L100
Stage-2 pilot are **superseded** (kept only as a historical record under `results/`).

**One bundle per rung.** Each `bundle_<stage>.zip` is self-contained — MATLAB
generator + Python ablation, both **preset for that rung**, plus the parity smoke.
Extract into the repo root on a Lab PC and run; no editing. Rebuild any time with
`python build_stage_bundles.py`. Each bundle carries its own `README_BUNDLE.md`.

## The ladder

One factor per rung, so any degradation is **attributable**. Heads = **scour +
bearing only**; crack / rail profile / track-layer / wheel damage are **nuisances**
the network must be invariant to (it never estimates them).

| Bundle | Adds | Geometry |
|---|---|---|
| `bundle_s0_scour` | scour only — baseline + architecture selection | L60 / 3-span, piers 2,3 |
| `bundle_s11_bear` | + bearing (head) | " |
| `bundle_s12_crack` | + crack (nuisance, no bearing) | " |
| `bundle_s13_bearcrack` | + bearing + crack = all **bridge** damages | " |
| `bundle_s14_prof` | + rail profile (FRA-4, per-state) = **the roughness rung** | " |
| `bundle_s15_track` | + track-layer damage (ballast / hanging sleepers / pads) | " |
| `bundle_s16_all` | + wheel OOR & flats = **all damages** | " |
| `bundle_s21_scour4` | scour only | **L99.6 / 4-span**, piers 2,3,4 |
| `bundle_s22_bearcrack4` | + bearing + crack | " |
| `bundle_s23_all4` | all damages | " |

Folder names are **short** (`<stage>_L<len>_st<N>`, e.g. `s14_prof_L60_st267`) —
the old ~110-char names blew past Windows MAX_PATH. The full descriptor lives in
`case_info.case_desc` / `case_info.txt`.

## Per-bundle run order

1. **MATLAB — generate.** Open `scour_MATLAB/A00_Run.m` and run it. `STAGE` is
   already set and `use_signal_noise = false` (noise is a load-time model).
   Output → `scour_MATLAB/Results/<case>/`; move it under `data/`.
2. **Once per campaign — verify the transform.** After the FIRST state of the
   first rung exists:
   `smoke_raw_parity('Results/<case>')` in MATLAB, then
   `python check_raw_parity.py "scour_MATLAB/Results/<case>"`.
   **Must print `PARITY PASS`** (max|MATLAB−Python| < 1e-12) before the long runs.
3. **Python — ablate.** `python comprehensive_ablation_multidamage.py`
   (STAGE + SENSOR_NOISE preset). 100 Optuna trials × 3 seeds, Phase-1 single-DOF
   sweep → auto pair + the designed mixed pair `[1,3]`.
   Summary → `results/<stage>_summary/` (+ `leaderboard_median.csv` = the
   paper-facing table).

Generation and ablation run **concurrently** — with 3 PCs, generate the next rung
while the previous one ablates.

## What changed (and why)

- **Data format (Option B).** D01 saves the **raw, un-interpolated, noise-free
  time-domain** signal + the space/crop parameters; Python rebuilds the space
  window at load time (`core/dataset._raw_to_space_crop`, an exact `interp1`
  mirror). The measurement model now lives entirely at load time and can change
  **without regenerating**. Storage is ~neutral (`DimSpace ≈ 2.2·DimAcel`).
- **Noise.** Generation is noise-free; the ablation injects a uniform 5%
  multiplicative noise on **every** channel at load time (`all_mult`). Adding noise
  before vs after the interpolation is *not* equivalent (time-domain noise becomes
  coloured + speed-dependent: ≈0.67× variance but ≈1.46× the energy surviving PAA).
- **EOVs are anchored** (`docs/track_eov_sampling_spec.md`): persistent conditions
  drawn **per state** (EN 13848-2); FRA-4; crack p=0.25 **hogging-weighted 4:1**
  (Eurocode 4); hanging groups **Poisson λ=3.0/100 m**, ballast **λ=1.2/100 m**
  (both window-scaled); pads **p=0.02** (snapshot prevalence, not the annual rate);
  fouling↔voiding **coupled ×3**; ballast **×3** near abutments.
- **Provenance.** Re-ablations start **from scratch** via `RUN_TAG`; the noise mode
  enters the study name, so a noise A/B on identical data trains separate studies
  instead of silently resuming.

## Requirements
`torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn,
PyWavelets, tqdm` (the env that ran Stage 0), plus MATLAB with the Statistics
toolbox (`poissrnd`, `wblrnd`, `lhsdesign`). Everything is resumable — re-running
a rung continues where it stopped.
