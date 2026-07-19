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

# THE LADDER (2026-07-14). Everything is regenerated from scratch (raw/Option-B
# format + the new EOV design), so every stage is a GEN stage and every stage
# ablates with `all_mult` = uniform 5% multiplicative noise on EVERY channel,
# injected at LOAD time onto noise-free data. One factor per rung.
STAGES = {
    # id                (noise mode, one-line what-it-adds)
    "s0_scour":       ("all_mult", "scour only - baseline + architecture selection"),
    "s11_bear":       ("all_mult", "+ bearing (HEAD)"),
    "s12_crack":      ("all_mult", "+ crack (nuisance, no bearing)"),
    "s13_bearcrack":  ("all_mult", "+ bearing + crack = all BRIDGE damages"),
    "s14_prof":       ("all_mult", "+ rail profile FRA-4 per-state = the ROUGHNESS rung"),
    "s15_track":      ("all_mult", "+ track-layer damage (ballast/hanging sleepers/pads)"),
    "s16_all":        ("all_mult", "+ wheel OOR (polygonization; flats disabled) = ALL damages"),
    "s21_scour4":     ("all_mult", "4-span L99.6, scour only"),
    "s22_bearcrack4": ("all_mult", "4-span, + bearing + crack = all BRIDGE damages"),
    "s23_all4":       ("all_mult", "4-span, all damages"),
}
# Extra files that post-date the source bundle's manifest.
EXTRA_FILES = ["scour_MATLAB/smoke_raw_parity.m", "check_raw_parity.py",
               "README_CAMPAIGN.md",
               # 2026-07-17 audit test layer - ship with every bundle:
               "scour_MATLAB/smoke_audit.m", "scour_MATLAB/smoke_stage3.m",
               "scour_MATLAB/smoke_geometry.m", "check_split_grouping.py",
               "check_loader_provenance.py", "scour_MATLAB/save_progress.m",
               # 2026-07-18/19 audit layer: cache-provenance test + the unified
               # protocol-hash module (the driver IMPORTS core/protocol.py — a
               # bundle without it cannot even start) and its adversarial test.
               "check_cache_provenance.py",
               "core/protocol.py", "check_protocol_hash.py",
               # Feature A MATLAB<->Python family-table contract smoke (run the
               # .m in MATLAB first, then the Python side).
               "scour_MATLAB/smoke_familytable.m",
               "check_familytable_roundtrip.py"]

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

def readme(stage, mode, adds):
    return "\n".join([
        f"# Bundle: {stage}  —  {adds}", "",
        "Self-contained: MATLAB generator + Python ablation, both PRESET for this",
        "stage. Extract into the repo root on the Lab PC. No editing needed.", "",
        "## 0. AUDIT 2026-07-17 — read first",
        "- Extract into a FRESH working copy; do NOT extract over an old run. A00",
        "  writes a gen_schema and ABORTS if you resume a folder from other code.",
        "- One-time sanity (MATLAB): `smoke_audit`, `smoke_stage3`, `smoke_geometry`;",
        "  (Python): `python check_split_grouping.py`, `python check_loader_provenance.py`,",
        "  `python check_cache_provenance.py`, `python check_protocol_hash.py`.",
        "  All must print ALL PASS.",
        "- Run STAGE s0_scour FIRST: it selects the architecture and writes",
        "  results/_champion_arch_<schema>_ph-<hash>.json; every later rung reads it",
        "  and errors if it is missing (no hardcoded champion anymore). On a DIFFERENT",
        "  PC, copy that JSON here (or set CHAMPION_MANIFEST / CHAMPION_ARCH_OVERRIDE).",
        "- PROTOCOL HASH (2026-07-19): every study/summary/manifest name carries a",
        "  SHA-256 of the full protocol (dataset fingerprint, split, seeds, trials,",
        "  pruner, search space, noise, targets). If ANY of those change, names",
        "  change and old studies are orphaned — never resumed. The exact hashed",
        "  descriptor is written to the summary dir as protocol_descriptor.json.",
        "- Study/DB/cache dirs are STAGE-prefixed, so rungs never cross-contaminate.", "",
        "## 1. MATLAB — generate the data (noise-free, RAW format)",
        f"Open `scour_MATLAB/A00_Run.m` and RUN it. `STAGE` is already `{stage}` and",
        "`use_signal_noise = false` — measurement noise is added later, at LOAD time.",
        "D01 saves the RAW, un-interpolated TIME-domain signal plus the space/crop",
        "parameters; Python rebuilds the space window at load time (Option B), so the",
        "noise model can change forever without regenerating.",
        "Output -> `scour_MATLAB/Results/<stage>_L<len>_st<N>/`; move it under `data/`.",
        "Folder names are SHORT now (Windows MAX_PATH); the full descriptor lives in",
        "`case_info.case_desc` / `case_info.txt`.", "",
        "## 2. (once) verify the MATLAB<->Python transform",
        "After the FIRST state exists, in MATLAB:  `smoke_raw_parity('Results/<case>')`",
        "then:  `python check_raw_parity.py \"scour_MATLAB/Results/<case>\"`",
        "Must print PARITY PASS (max|MATLAB-Python| < 1e-12) before the long run.", "",
        "## 3. Python — ablate",
        "`python comprehensive_ablation_multidamage.py` (STAGE + SENSOR_NOISE preset).",
        f"Noise: `{mode}` = uniform 5% multiplicative on EVERY channel, injected at load",
        "time onto the noise-free data. 100 trials x 3 seeds + the mixed pair [1,3];",
        f"summary -> `results/{stage}_summary_ph-<hash>/` (+ leaderboards +",
        "`protocol_descriptor.json`).", "",
        "## Heads vs nuisances",
        "HEADS = scour (per pier) + bearing (per abutment) ONLY. Crack, rail profile,",
        "track-layer and wheel damage are NUISANCES: randomized, logged, never",
        "estimated — the network must be INVARIANT to them.", "",
        "## Requirements",
        "torch, optuna, scikit-learn, numpy, scipy, joblib, matplotlib, seaborn,",
        "PyWavelets, tqdm. Everything is resumable.", ""])

# The source bundle supplies only the FILE LIST; contents are read from the REPO
# working tree so a bundle can never ship stale code.
with zipfile.ZipFile(SRC_BUNDLE) as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
names = [n for n in names if n != "pilot_stage2_L100_mixed_pairs.py"]   # L100-specific
names = [n for n in names if n != "README_STAGE2.md"]                   # renamed -> README_CAMPAIGN.md
names = sorted(set(names) | set(EXTRA_FILES))

blobs, missing = {}, []
for n in names:
    p = os.path.join(REPO, n.replace("/", os.sep))
    if os.path.isfile(p):
        with open(p, "rb") as fh:
            blobs[n] = fh.read()
    else:
        missing.append(n)
if missing:
    raise SystemExit(f"MISSING from the repo (fix before shipping): {missing}")

DRIVER = "comprehensive_ablation_multidamage.py"
A00 = "scour_MATLAB/A00_Run.m"

# drop stale bundles from earlier stage naming
for f in os.listdir(REPO):
    if f.startswith("bundle_") and f.endswith(".zip"):
        os.remove(os.path.join(REPO, f))

built = []
for stage, (mode, adds) in STAGES.items():
    out = os.path.join(REPO, f"bundle_{stage}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            data = blobs[n]
            if n == DRIVER:
                t = set_driver_stage(data.decode("utf-8"), stage)
                data = set_driver_noise(t, mode).encode("utf-8")
            elif n == A00:
                data = set_a00_stage(data.decode("utf-8"), stage).encode("utf-8")
            z.writestr(n, data)
        z.writestr("README_BUNDLE.md", readme(stage, mode, adds))
    built.append((f"bundle_{stage}.zip", os.path.getsize(out) // 1024, adds))

print(f"{'bundle':30} {'KB':>5}  adds")
for b, kb, adds in built:
    print(f"{b:30} {kb:5}  {adds}")
print(f"\n{len(built)} bundles x {len(names)+1} files, contents read from the repo working tree.")
