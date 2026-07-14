"""
build_stage_bundles.py  (2026-07-14)
Produce ONE self-contained bundle per stage: MATLAB generator (A00 STAGE preset)
+ Python ablation (driver STAGE preset + noise mode) + all core code + a per-stage
README. Each bundle extracts into the repo root on a Lab PC and runs with NO editing.

Two PCs -> run two bundles at once. File list is inherited from the last working
campaign bundle (multidamage_stage2_bundle.zip) so nothing is dropped.
"""
import os, re, zipfile, io

REPO = os.path.dirname(os.path.abspath(__file__))
SRC_BUNDLE = os.path.join(REPO, "multidamage_stage2_bundle.zip")

# stage -> (noise mode for the ablation, needs MATLAB generation?)
# all_mult   = uniform 5% mult noise on every channel (data generated noise-free)
# sprung_mult= 5% on sprung channels only (data already carries baked WHEEL noise)
STAGES = {
    "stage0_multiscour": ("sprung_mult", False),
    "stage1_bearing":    ("sprung_mult", False),
    "stage1_crack":      ("all_mult",    True),
    "stage1_full":       ("all_mult",    True),
    "stage2_4span":      ("all_mult",    True),
    "stage3_alldamage":  ("all_mult",    True),
}

def set_a00_stage(t, stage):
    t2, n = re.subn(r"^STAGE = '[^']*';", f"STAGE = '{stage}';", t, count=1, flags=re.M)
    assert n == 1, "A00 STAGE line not found/replaced"
    return t2

def set_driver_stage(t, stage):
    t2, n = re.subn(r'^(STAGE = )"[^"]*"', rf'\1"{stage}"', t, count=1, flags=re.M)
    assert n == 1, "driver STAGE line not found/replaced"
    return t2

def set_driver_noise(t, mode):
    t2 = t.replace("SENSOR_NOISE = None",
                   f'SENSOR_NOISE = {{"mode": "{mode}", "desvio": 0.05}}', 1)
    assert t2 != t, "driver SENSOR_NOISE line not found/replaced"
    return t2

def readme(stage, mode, gen):
    lines = [f"# Bundle: {stage}", ""]
    if gen:
        lines += [
            "## 1. MATLAB - generate the data (noise-free)",
            "Open `scour_MATLAB/A00_Run.m` and just RUN it: `STAGE` is already set to",
            f"`{stage}` and `use_signal_noise = false` (noise is added later, at load time).",
            "Output -> `scour_MATLAB/Results/<case>/`; move/copy it under `data/`.",
            "",
            "## 2. Python - ablate",
            "`python comprehensive_ablation_multidamage.py` (STAGE + SENSOR_NOISE preset).",
            f"Noise: `{mode}` = uniform 5% multiplicative on EVERY channel (the data is",
            "noise-free, so this makes all channels equally noisy). 100 trials x 3 seeds;",
            f"summary -> `results/{stage}_summary/` (+ `leaderboard_median.csv`).",
        ]
    else:
        lines += [
            "## Re-ablation only - the data already exists (do NOT regenerate)",
            f"The {stage} dataset was generated earlier and carries baked WHEEL noise.",
            "Just run the ablation:",
            "`python comprehensive_ablation_multidamage.py`",
            f"Noise: `{mode}` = 5% multiplicative on the SPRUNG channels only, so all",
            "channels end up ~5% noisy WITHOUT double-noising the already-noisy wheels.",
            "Study names are noise-tagged (`_nz-sprung_mult`) -> the published noiseless",
            "studies are NOT touched; new fresh studies train from scratch.",
            "",
            "CAVEAT (state it if reported): wheels then carry the legacy COLORED/",
            "speed-dependent baked noise while sprung channels carry load-time WHITE",
            "noise - a second-order difference after PAA (see the noise-domain finding).",
            "For perfectly uniform white noise, regenerate this stage noise-free instead:",
            "the included `A00_Run.m` is preset (STAGE + use_signal_noise=false) to do so,",
            "then switch the driver's SENSOR_NOISE mode to `all_mult`.",
        ]
    lines += ["", "## Requirements",
              "torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn,",
              "PyWavelets, tqdm. Everything is resumable.", ""]
    return "\n".join(lines)

# inherit the canonical file list
with zipfile.ZipFile(SRC_BUNDLE) as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
    blobs = {n: z.read(n) for n in names}

DRIVER = "comprehensive_ablation_multidamage.py"
A00 = "scour_MATLAB/A00_Run.m"
# pilot is L100-specific; drop it from the clean per-stage bundles
names = [n for n in names if n != "pilot_stage2_L100_mixed_pairs.py"]

built = []
for stage, (mode, gen) in STAGES.items():
    out = os.path.join(REPO, f"bundle_{stage}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            data = blobs[n]
            if n == DRIVER:
                t = data.decode("utf-8")
                t = set_driver_stage(t, stage)
                t = set_driver_noise(t, mode)
                data = t.encode("utf-8")
            elif n == A00:
                data = set_a00_stage(data.decode("utf-8"), stage).encode("utf-8")
            z.writestr(n, data)
        z.writestr("README_BUNDLE.md", readme(stage, mode, gen))
    built.append((f"bundle_{stage}.zip", len(names) + 1, mode, "GEN" if gen else "reablate"))

print(f"{'bundle':32} {'files':>6}  {'noise':12} {'action'}")
for b, nf, mode, act in built:
    print(f"{b:32} {nf:6}  {mode:12} {act}")
