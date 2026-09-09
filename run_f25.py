"""Run the Fernandes-2025 reconstruction (F25-R then F25-X) from VS Code.

Open this file, set WORKSPACE below, and press Run. No shell. It runs the
capacity probe once, then every job of both plans in their generated order,
which is the dependency order for a single machine.

Interrupted (power cut, Windows update, closed window)? Just press Run again.
Finished jobs return immediately and an interrupted HPO study continues from
the trials already stored.

The MATLAB generation must have happened first, once:
    F25_Run('F25-R', 4)
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

# --- CONFIGURE ------------------------------------------------------------
# The workspace where bundle_F25-R.zip and bundle_F25-X.zip were co-extracted.
WORKSPACE = Path(r"D:\f25\workspace")

# A machine that dies mid-trial leaves that trial stored as RUNNING forever,
# and the executor refuses to resume a study containing one. Removing it drops
# no results (the trial never produced any) and keeps every completed trial.
# Set to False to be asked to clean it up by hand instead.
CLEAN_INTERRUPTED_TRIALS = True
# --------------------------------------------------------------------------

DATASET_MARKER = (
    WORKSPACE / "f25_artifacts" / "shared_data" / "generation"
    / "fernandes-2025-f25-data-v1" / "_F25_GENERATION_COMPLETE"
)


def clean_interrupted_trials(database: Path) -> int:
    """Drop trials that never completed. Returns how many were removed."""
    if not database.is_file():
        return 0
    with sqlite3.connect(database) as probe:
        stale = [
            row[0] for row in probe.execute("select trial_id, state from trials")
            if row[1] != "COMPLETE"
        ]
    if not stale:
        return 0
    shutil.copy2(database, database.parent / (database.name + ".bak"))
    with sqlite3.connect(database) as connection:
        for table in (
            "trial_values", "trial_params", "trial_user_attributes",
            "trial_system_attributes", "trial_intermediate_values",
            "trial_heartbeats",
        ):
            for trial_id in stale:
                try:
                    connection.execute(
                        f"delete from {table} where trial_id=?", (trial_id,)
                    )
                except sqlite3.OperationalError:
                    pass
        for trial_id in stale:
            connection.execute("delete from trials where trial_id=?", (trial_id,))
    return len(stale)


def run_capacity_probe(probe, failure_message: str) -> None:
    """Run the capacity gate, treating an existing receipt as already passed.

    Both capacity gates refuse to overwrite a receipt for the same machine and
    sources, on purpose: the receipt is run evidence. On a rerun -- after a
    reboot, or simply the second time this script is started -- that refusal
    means the gate was already passed here, not that anything is wrong.
    """
    print("capacity probe ...")
    saved_argv = sys.argv
    sys.argv = [saved_argv[0] if saved_argv else "probe"]
    try:
        code = probe()
    except BaseException as error:  # noqa: BLE001 - inspected and re-raised
        if "already exists" not in str(error):
            raise
        print(f"capacity receipt already on file, kept as evidence\n"
              f"  ({error})\n")
        return
    finally:
        sys.argv = saved_argv
    if code != 0:
        raise SystemExit(f"capacity probe failed -- {failure_message}.")
    print("capacity probe ok\n")


def completion_artifact(job: dict) -> Path:
    if job["phase"] == "hpo":
        return WORKSPACE / job["manifest_path"] / "winner.json"
    return WORKSPACE / job["results_path"] / "metrics.json"


def main() -> None:
    for experiment in ("F25-R", "F25-X"):
        if not (WORKSPACE / f"f25_training_plan.{experiment}.json").is_file():
            raise SystemExit(
                f"No f25_training_plan.{experiment}.json in {WORKSPACE}. "
                "Both bundles must be co-extracted into this one workspace."
            )
    if not DATASET_MARKER.is_file():
        raise SystemExit(
            f"The shared F25 dataset is not complete ({DATASET_MARKER} is "
            "missing). Run F25_Run('F25-R', 4) in MATLAB first."
        )

    os.chdir(WORKSPACE)
    sys.path.insert(0, str(WORKSPACE))
    import check_f25_capacity
    from training import f25_executor

    plans = {
        experiment: json.loads(
            (WORKSPACE / f"f25_training_plan.{experiment}.json").read_text("utf-8")
        )
        for experiment in ("F25-R", "F25-X")
    }
    total = sum(len(plan["jobs"]) for plan in plans.values())
    done = sum(
        1 for plan in plans.values() for job in plan["jobs"]
        if completion_artifact(job).is_file()
    )
    print(f"workspace {WORKSPACE}")
    print(f"{total} jobs total ({len(plans['F25-R']['jobs'])} F25-R + "
          f"{len(plans['F25-X']['jobs'])} F25-X); {done} already finished\n")

    run_capacity_probe(
        # main() reads sys.argv directly, so hand it a clean one.
        lambda: check_f25_capacity.main(),
        "this GPU cannot hold the largest F25 job",
    )

    started = time.time()
    index = 0
    for experiment, plan in plans.items():
        for job in plan["jobs"]:
            index += 1
            elapsed = (time.time() - started) / 3600.0
            state = "done" if completion_artifact(job).is_file() else "run"
            print(f"[{index}/{total}] {state}  {job['job_id']}  ({elapsed:.1f} h)",
                  flush=True)

            if job["phase"] == "hpo" and CLEAN_INTERRUPTED_TRIALS:
                removed = clean_interrupted_trials(
                    WORKSPACE / job["manifest_path"] / "study.sqlite3"
                )
                if removed:
                    print(f"      removed {removed} interrupted trial(s); the "
                          "completed ones were kept (backup: study.sqlite3.bak)")

            try:
                f25_executor.main([
                    "run-job", "--experiment", experiment,
                    "--job-id", job["job_id"],
                ])
            except Exception as error:  # noqa: BLE001 - report and stop
                raise SystemExit(
                    f"job {job['job_id']} failed: {error}\n"
                    "Nothing after it was started."
                ) from error

    print(f"\nAll {total} jobs finished in "
          f"{(time.time() - started) / 3600.0:.1f} h.")
    print(f"Evidence is under {WORKSPACE / 'f25_artifacts'}.")


if __name__ == "__main__":
    main()
