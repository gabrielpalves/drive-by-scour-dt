"""make_micro_smoke.py — build the MICRO end-to-end smoke environment.

Validated 2026-07-19 on the author's PC (MATLAB R2025b + Python 3.13/optuna):
this reproduces the full campaign machinery at toy scale in ~30 min total,
catching integration problems BEFORE a multi-day generation run.

What it emits (both are TEXT PATCHES of the real files, so the smoke always
exercises the CURRENT code verbatim — never a re-implementation):

  1. scour_MATLAB/micro_A00_smoke.m  — the real A00_Run.m with only size knobs
     patched: 3 target_healthy + 2 piers x (2 levels x 2 reps) + 6 joint
     = 17 states x 3 passages (~15 min on a 32-worker pool). The LHS
     degeneracy guard is bypassed below Npass=10 (it is calibrated for
     Npass~50: with 2-3 samples |corr|=1 always and 4-quadrant occupancy is
     impossible — found the hard way, 2026-07-19).
  2. (--dryrun) <scratch>/dryrun/ with the generated micro dataset copied to
     data/ and dryrun_driver.py — the real ablation driver patched to
     N_TRIALS=2, EPOCHS=2, SEEDS=[42] against the micro dataset. Run it with a
     Python that has optuna; it executes the WHOLE s0 pipeline (phase 1,
     champion pick, 28-pair sweep, strict summarize, test-once, manifests).

Suggested smoke sequence on a fresh machine (see docs/framework_rationale.md
R7.2 entry for the full record):
    python make_micro_smoke.py
    (MATLAB)  micro_A00_smoke
    (MATLAB)  smoke_raw_parity('Results/s0_scour_L60_st17')
    python check_raw_parity.py "scour_MATLAB/Results/s0_scour_L60_st17"
    del scour_MATLAB/Results/s0_scour_L60_st17/0005.mat   -> rerun micro (resume)
    python make_micro_smoke.py --dryrun <scratch_dir>
    cd <scratch_dir>/dryrun && python dryrun_driver.py
"""
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
MICRO_DS = "s0_scour_L60_st17"          # 3 + 8 + 6 states (see patches below)


def write_micro_a00() -> str:
    """Patch A00_Run.m size knobs -> scour_MATLAB/micro_A00_smoke.m."""
    src = open(os.path.join(REPO, 'scour_MATLAB', 'A00_Run.m'),
               encoding='utf-8').read()
    patches = [
        (r"^n_states_multi   = 250;", "n_states_multi   = 6;      % MICRO-SMOKE"),
        (r"^Npass = 50;",             "Npass = 3;                  % MICRO-SMOKE"),
        (r"^n_healthy_states  = 12;", "n_healthy_states  = 3;     % MICRO-SMOKE"),
        (r"^n_anchor_levels  = 4;",   "n_anchor_levels  = 2;      % MICRO-SMOKE"),
        # LHS guard calibrated for Npass ~ 50 (see module docstring):
        (r"^    if abs\(corr_st_\) > 0\.6 \|\| ~all\(occ_\)",
         "    if Npass >= 10 && (abs(corr_st_) > 0.6 || ~all(occ_))"
         "   % MICRO-SMOKE bypass"),
    ]
    for pat, rep in patches:
        src, n = re.subn(pat, rep, src, count=1, flags=re.M)
        assert n == 1, f"A00 knob not found (A00 changed?): {pat}"
    out = os.path.join(REPO, 'scour_MATLAB', 'micro_A00_smoke.m')
    open(out, 'w', encoding='utf-8').write(src)
    return out


def write_dryrun(scratch: str) -> str:
    """Copy the generated micro dataset + emit the patched dry-run driver."""
    micro_src = os.path.join(REPO, 'scour_MATLAB', 'Results', MICRO_DS)
    assert os.path.exists(os.path.join(micro_src, '_GENERATION_COMPLETE')), \
        f"micro dataset not complete at {micro_src} — run micro_A00_smoke first"
    dry = os.path.join(scratch, 'dryrun')
    os.makedirs(os.path.join(dry, 'data'), exist_ok=True)
    dst = os.path.join(dry, 'data', MICRO_DS)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(micro_src, dst)
    # a fresh copy must not inherit the source's split manifest
    sm = os.path.join(dst, 'split_manifest.json')
    if os.path.exists(sm):
        os.remove(sm)
    src = open(os.path.join(REPO, 'comprehensive_ablation_multidamage.py'),
               encoding='utf-8').read()
    patches = [
        (r'"s0_scour":       \("s0_scour_L60_st278",',
         f'"s0_scour":       ("{MICRO_DS}",'),
        (r"^N_TRIALS       = 100.*$", "N_TRIALS       = 2            # DRY-RUN"),
        (r"^EPOCHS         = 50$",    "EPOCHS         = 2            # DRY-RUN"),
        (r"^SEEDS          = \[42, 1337, 2026\]$",
         "SEEDS          = [42]         # DRY-RUN"),
    ]
    for pat, rep in patches:
        src, n = re.subn(pat, rep, src, count=1, flags=re.M)
        assert n == 1, f"driver knob not found (driver changed?): {pat}"
    src = src.replace("import csv\nimport json\nimport os",
                      f"import sys; sys.path.insert(0, {REPO!r})\n"
                      "import csv\nimport json\nimport os")
    out = os.path.join(dry, 'dryrun_driver.py')
    open(out, 'w', encoding='utf-8').write(src)
    return out


if __name__ == "__main__":
    if "--dryrun" in sys.argv:
        i = sys.argv.index("--dryrun")
        scratch = sys.argv[i + 1] if len(sys.argv) > i + 1 else os.path.join(
            REPO, "_micro_dryrun_scratch")
        print("dry-run driver ->", write_dryrun(scratch))
    else:
        print("micro A00 ->", write_micro_a00())
