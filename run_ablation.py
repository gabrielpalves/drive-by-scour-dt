"""Run this machine's ablation jobs from VS Code.

Open this file, edit the five values under CONFIGURE, and press Run. No shell,
no wrapper: it sets the environment the campaign code reads, runs the capacity
probe once, then executes every job assigned to this machine, in order.

Interrupted (power cut, Windows update, closed window)? Just press Run again.
Each job resumes where it stopped, and finished jobs return immediately.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# --- CONFIGURE ------------------------------------------------------------
# The extracted training bundle (the folder holding training_job_manifest.json)
BUNDLE = Path(r"D:\ttbi\bundle_train_labA")

# Where the four generated datasets live. It must already contain
# F40-S_*, F40-M_*, L99-S_* and L99-M_*, and its name must be "data".
DATA = Path(r"D:\ttbi\ttbi-campaign\data")

# Durable place for results, caches, Optuna studies and receipts. Keep it
# OUTSIDE the extracted bundle so re-extracting never deletes your results.
WORK = Path(r"D:\ttbi\ttbi-campaign")

# One shared label for this campaign run, the same on every machine.
RUN_TAG = "paper1-2026-09"
# --------------------------------------------------------------------------

os.environ["TTBI_TRAINING_JOB_MANIFEST"] = str(BUNDLE / "training_job_manifest.json")
os.environ["TTBI_DATA_ROOT"] = str(DATA)
os.environ["TTBI_RESULTS_ROOT"] = str(WORK / "results")
os.environ["TTBI_CACHE_ROOT"] = str(WORK / "cache")
os.environ["TTBI_STUDY_ROOT"] = str(WORK / "studies")
os.environ["TTBI_EXECUTION_RECEIPT_DIR"] = str(WORK / "receipts")
os.environ["TTBI_CAMPAIGN_RUN_TAG"] = RUN_TAG

for key in (
    "TTBI_RESULTS_ROOT", "TTBI_CACHE_ROOT",
    "TTBI_STUDY_ROOT", "TTBI_EXECUTION_RECEIPT_DIR",
):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

if not (BUNDLE / "training_job_manifest.json").is_file():
    raise SystemExit(f"No training_job_manifest.json in {BUNDLE} -- fix BUNDLE above.")
if not DATA.is_dir():
    raise SystemExit(f"{DATA} does not exist -- fix DATA above.")

# The campaign code resolves its own sources relative to the bundle root.
os.chdir(BUNDLE)
sys.path.insert(0, str(BUNDLE))

import capacity_preflight_compute  # noqa: E402
import comprehensive_ablation_multidamage as ablation  # noqa: E402

manifest = ablation.load_training_job_manifest()
jobs = list(manifest["assigned_job_ids"])

stages = sorted({job["stage"] for job in manifest["jobs"]})
missing = [s for s in stages if not any(DATA.glob(f"{s}_*"))]
if missing:
    raise SystemExit(
        f"No dataset in {DATA} for stage(s): {', '.join(missing)}. "
        f"This machine's jobs need all of: {', '.join(stages)}."
    )

print(f"{manifest['machine_role']}: {len(jobs)} jobs, stages {', '.join(stages)}")
print(f"data    {DATA}")
print(f"results {WORK / 'results'}")
print()

print("capacity probe ...")
if capacity_preflight_compute.main(
    ["--all-stages", "--receipt-dir", os.environ["TTBI_EXECUTION_RECEIPT_DIR"]]
) != 0:
    raise SystemExit("capacity probe failed -- this GPU cannot hold the largest job.")
print("capacity probe ok\n")

started = time.time()
for index, job_id in enumerate(jobs, start=1):
    elapsed = (time.time() - started) / 3600.0
    print(f"[{index}/{len(jobs)}] {job_id}  ({elapsed:.1f} h elapsed)", flush=True)
    if ablation.main(["--execute-job", job_id]) != 0:
        raise SystemExit(f"job {job_id} failed -- nothing after it was started.")

print(f"\nAll {len(jobs)} jobs finished in {(time.time() - started) / 3600.0:.1f} h.")
print(f"Results are under {WORK / 'results'}.")
