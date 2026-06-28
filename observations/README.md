# `observations/` — held-out drive-by signal batches for the Digital Twin

One **subfolder per case**. A "case" is a unique combination of bridge geometry,
damage configuration, and operational variability. Keeping each case in its own
folder guarantees a batch generated for one setup is never sampled as another's
observations.

## How a batch gets here

1. Run `scour_MATLAB/A00_Run.m` (configure the case at the top of that file).
   It writes to `scour_MATLAB/Results/<case_name>/`, where `<case_name>` already
   encodes the case, e.g.:

   ```
   L40_2span_scourS2_bearOFF_dano0-60pct_x91_Npass15_varNVST
   ```

   and contains:
   - `0001.mat … 00NN.mat` — one file per damage state (`data.Dano` = scour fraction)
   - `case_info.txt` / `case_info.mat` — the manifest (geometry, damage, variability)
   - `tempo_*.mat` — run timing (ignored by the loader)

2. Move that whole folder here:

   ```
   observations/L40_2span_scourS2_bearOFF_dano0-60pct_x91_Npass15_varNVST/
   ```

3. Point the twin at it in `run_dt.py`:

   ```python
   MODE        = "library"
   LIBRARY_DIR = "observations/L40_2span_scourS2_bearOFF_dano0-60pct_x91_Npass15_varNVST"
   ```

## Must match the champion classifier

The drive-by champion (`models/champion_PAA_NHiTS_2sensor_RBvert_CBpitch`) was
trained on the **40 m / 2-span / 3-support, central-pier scour, bearing-off**
bridge. A batch the twin samples must use the **same geometry** (see `case_info.txt`)
or the classifier will mispredict. A different bridge (e.g. the multi-damage
100 m / 4-span set) needs its **own retrained model** and its own subfolder here.

`SignalLibrary` (`digital_twin/signal_library.py`) reads only the indexed
`NNNN.mat` files and stores all 8 DOF channels; the champion selects its active
DOFs (here 2 = RearBogie_Vert, 5 = CarBody_Pitch) from each signal.
