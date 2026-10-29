"""Audit the returned F25 archive and regenerate manuscript evidence.

Usage: python analyze_f25_results.py [--archive f25.zip]
Reads JSON, SQLite and numeric MATLAB metadata; never executes archived code
or deserializes model checkpoints. Requires numpy, matplotlib and scipy.
All reported dispersions are sample SDs over the same 20 initialization seeds,
conditional on one fixed catalog, profile and passage partition.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
from decimal import ROUND_HALF_UP, Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
PREFIX = "f25/workspace/"
TIERS = ["F25-R-unfrozen-source-singles", "F25-X-01-frozen-hp-singles",
         "F25-X-02-unfrozen-singles", "F25-X-03-frozen-hp-pairs"]
SENSORS = {0: "Carbody vertical", 1: "Front-bogie vertical", 2: "Rear-bogie vertical",
           5: "Carbody pitch rate", 6: "Front-bogie pitch rate", 7: "Rear-bogie pitch rate"}
SENSOR_CODES = {i: f"S{i:02}" for i in SENSORS}
# Manuscript labels; CSVs keep the archive's physical8_v1 codes.
DISPLAY = {"S00": "CV", "S01": "FV", "S02": "RV", "S05": "CP", "S06": "FP", "S07": "RP"}
ARMS = ["RAW-CNN", "PAA-CNN", "PAA-multirate"]
ARM_LABELS = ["RAW", "PAA", "PAA-MR"]
METRICS = ["overall_ten_class_accuracy", "bearing_present_absent_accuracy",
           "scour_level_accuracy", "crack_level_accuracy"]
# (crack depth / section depth, support stiffness loss, bearing presence).
AXES = np.array([[0, 0, 0], [.1, 0, 0], [0, .05, 0], [0, 0, 1],
                 [0, .1, 1], [.1, .1, 0], [.1, 0, 1], [.1, .1, 1],
                 [.1, .05, 1], [.05, .05, 1]])
SEEDS = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
         1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123]
HPO_SEEDS = [104729, 130363, 155921, 196613, 228017]
# Overall accuracies of the best of 20 runs in Fernandes et al. (2025): the sums of
# the diagonals printed in their Figs. 9 and 12 (equal to core.f25_experiment_contract).
PUBLISHED_BEST_RUN_PCT = {("RAW-CNN", "S00"): 65.1, ("PAA-CNN", "S00"): 86.7,
                          ("RAW-CNN", "S01"): 82.2, ("PAA-CNN", "S01"): 82.1}


def half_up(numerator, denominator, places):
    """Round an exact ratio half-up; float formatting can round 72.285 down."""
    quantum = Decimal(1).scaleb(-places)
    return f"{(Decimal(int(numerator)) / Decimal(int(denominator))).quantize(quantum, ROUND_HALF_UP)}"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capacity_hash(receipt):
    return digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "f25.zip")
    args = parser.parse_args()
    out = ROOT / "results" / "f25"
    figures = ROOT / "paper1" / "figures"
    tables = ROOT / "paper1" / "tables"
    for directory in (out, figures, tables):
        directory.mkdir(parents=True, exist_ok=True)
    audit = {"archive": args.archive.name, "archive_sha256": file_hash(args.archive),
             "scope": "internal archive consistency; not external authentication or a numerical V&V verdict",
             "checks": {}, "limitations": [
                 "No checkpoints were deserialized and no model was retrained.",
                 "No per-passage predictions are retained: confusion-based metrics can be recomputed, paired passage tests cannot.",
                 "Historical repeated/aborted fits and wall-clock/GPU time are not reconstructed.",
                 "SD measures initialization variability conditional on this data, profile and split.",
                 "Channel rankings read the shared test set and are descriptive/exploratory."]}
    runs, summaries, hpo_rows, records, continuations = [], [], [], [], []
    confusion_by_job, artifact_by_job, job_by_id = {}, {}, {}
    confusion_sum_by_job, correct_by_job = {}, {}
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive members")

        def raw(path):
            return archive.read(PREFIX + path)

        def read(path):
            def pairs(items):
                result = {}
                for key, value in items:
                    require(key not in result, f"Duplicate JSON key: {path}:{key}")
                    result[key] = value
                return result
            return json.loads(raw(path), object_pairs_hook=pairs,
                              parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

        plans = [read(f"f25_training_plan.{experiment}.json") for experiment in ("F25-R", "F25-X")]
        manifests = [read(f"f25_bundle_manifest.{experiment}.json") for experiment in ("F25-R", "F25-X")]
        contract = plans[0]["f25_contract_sha256"]
        for plan, manifest in zip(plans, manifests):
            unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
            require(digest(unsigned) == plan["plan_sha256"], "Plan hash mismatch")
            unsigned = {k: v for k, v in manifest.items() if k != "bundle_manifest_sha256"}
            require(digest(unsigned) == manifest["bundle_manifest_sha256"], "Bundle manifest hash mismatch")
            require(manifest["training_plan_sha256"] == plan["plan_sha256"], "Bundle/plan mismatch")
            require(manifest["f25_contract_sha256"] == contract, "Contract mismatch")
            require(digest(manifest["source_files"]) == manifest["source_root_sha256"], "Source inventory hash mismatch")
            require(hashlib.sha256(raw("bundle_source_files.txt")).hexdigest() == manifest["source_manifest_sha256"], "Source manifest bytes mismatch")
            for row in manifest["source_files"]:
                require(hashlib.sha256(raw(row["path"])).hexdigest() == row["sha256"], f"Source bytes mismatch: {row['path']}")
            for row in manifest["generated_artifacts"].values():
                require(hashlib.sha256(raw(row["path"])).hexdigest() == row["sha256"], "Generated dispatch artifact mismatch")
            for job in plan["jobs"]:
                job_by_id[job["job_id"]] = job
        require(len(job_by_id) == 107, "Expected 107 jobs")
        require(manifests[0]["source_files"] == manifests[1]["source_files"], "R/X source mismatch")
        audit["checks"]["bundle_sources"] = {"files": manifests[0]["source_file_count"],
                                                "source_commit": manifests[0]["source_commit"],
                                                "source_root_sha256": manifests[0]["source_root_sha256"]}
        block = read("f25_artifacts/execution_block_receipt.json")
        for job_id, job in job_by_id.items():
            record = read(job["manifest_path"] + "/run_record.json")
            records.append(record)
            require(record["job"] == job, f"Job record mismatch {job_id}")
            require(record["job_sha256"] == digest(job), f"Job digest mismatch {job_id}")
            require(record["execution_block_receipt_sha256"] == digest(block), "Execution block mismatch")
            binding = record["capacity_receipt_binding"]
            capacity = read(binding["relative_path"])
            require(capacity_hash(capacity) == binding["receipt_sha256"] == capacity["receipt_sha256"]
                    and capacity["accepted"], "Capacity binding mismatch")
            path = (job["manifest_path"] + "/winner.json" if job["phase"] == "hpo"
                    else job["results_path"] + "/metrics.json")
            result = read(path)
            artifact_by_job[job_id] = result
            require(result["job_id"] == job_id and result["job_sha256"] == digest(job), "Artifact job mismatch")
            require(result["run_record_sha256"] == digest(record), "Artifact run record mismatch")
            require(result["f25_contract_sha256"] == contract, "Artifact contract mismatch")
            if "driver_continuation" in result:
                binding = result["driver_continuation"]
                receipt = read(binding["receipt_path"])
                require(hashlib.sha256(raw(binding["receipt_path"])).hexdigest() == binding["receipt_file_sha256"], "Continuation file hash mismatch")
                require(digest(receipt["continuation"]) == binding["continuation_sha256"], "Continuation content hash mismatch")
                continuation = receipt["continuation"]
                require(continuation["original_run_record_sha256"] == digest(record), "Continuation original record mismatch")
                current = continuation["current_execution_record"]
                require(current["job"] == job and current["data_binding"] == record["data_binding"], "Continuation scientific binding changed")
                require(current["python_runtime_source_sha256"] == record["python_runtime_source_sha256"], "Continuation source changed")
                capacity_binding = current["capacity_receipt_binding"]
                require(capacity_hash(read(capacity_binding["relative_path"])) == capacity_binding["receipt_sha256"], "Continuation capacity mismatch")
                continuations.append({"job_id": job_id, "previous_driver": continuation["previous_driver"],
                                      "current_driver": continuation["current_driver"]})
            if job["phase"] == "hpo":
                database_bytes = raw(job["manifest_path"] + "/study.sqlite3")
                with tempfile.TemporaryDirectory(prefix="f25-audit-") as scratch:
                    database = Path(scratch) / "study.sqlite3"
                    database.write_bytes(database_bytes)
                    with closing(sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)) as connection:
                        trials = connection.execute("select trial_id,number,state from trials order by number").fetchall()
                        require(len(trials) == 100 and all(t[2] == "COMPLETE" for t in trials), f"Incomplete HPO {job_id}")
                        values = dict(connection.execute("select trial_id,value from trial_values").fetchall())
                        attrs = {}
                        for tid, key, value in connection.execute("select trial_id,key,value_json from trial_user_attributes"):
                            attrs.setdefault(tid, {})[key] = json.loads(value)
                        for tid, number, state in trials:
                            require(attrs[tid]["execution_seeds"] == HPO_SEEDS, "HPO seed mismatch")
                            scores = attrs[tid]["validation_accuracies"]
                            require(len(scores) == 5 and all(math.isfinite(x) for x in scores), "HPO missing scores")
                            require(abs(np.mean(scores) - values[tid]) < 1e-12, "HPO objective mismatch")
                        winner_trial = next(t for t in trials if t[1] == result["trial_number"])
                        require(abs(values[winner_trial[0]] - max(values.values())) < 1e-12, "HPO winner not best")
                        require(attrs[winner_trial[0]]["validation_accuracies"] == result["validation_accuracies"], "HPO winner scores mismatch")
                        require(abs(values[winner_trial[0]] - result["mean_validation_accuracy"]) < 1e-12, "HPO winner objective mismatch")
                        winner_params = {}
                        for name, value, distribution in connection.execute(
                                "select param_name,param_value,distribution_json from trial_params where trial_id=?", (winner_trial[0],)):
                            distribution = json.loads(distribution)
                            if distribution["name"] == "CategoricalDistribution":
                                value = distribution["attributes"]["choices"][int(value)]
                            winner_params[name] = value
                        require(winner_params == result["params"], "HPO winner params mismatch")
                hpo_rows.append({"job_id": job_id, "tier": job["tier_id"], "configuration": job["configuration_id"],
                                 "completed_proposals": 100, "executions_per_proposal": 5,
                                 "winner_trial": result["trial_number"],
                                 "validation_accuracy_mean_pct": 100 * result["mean_validation_accuracy"],
                                 "params_json": json.dumps(result["params"], sort_keys=True)})
                continue
            require(result["report_seeds"] == SEEDS and [r["seed"] for r in result["results"]] == SEEDS, "Report seed mismatch")
            require(hashlib.sha256(raw(result["anchor_winner_path"])).hexdigest() == result["anchor_winner_sha256"], "Anchor file mismatch")
            require(read(result["anchor_winner_path"])["params"] == result["params"], "Anchor params mismatch")
            cms = []
            for row in result["results"]:
                cm = np.asarray(row["metrics"]["confusion_matrix"])
                require(cm.shape == (10, 10) and np.issubdtype(cm.dtype, np.integer) and np.all(cm >= 0), "Invalid confusion counts")
                require(np.all(cm.sum(axis=1) == 100), "Test class counts differ from 100")
                derived = [np.trace(cm) / 1000]
                for axis in (2, 1, 0):
                    derived.append(float(cm[AXES[:, axis, None] == AXES[None, :, axis]].sum() / 1000))
                for key, value in zip(METRICS, derived):
                    require(abs(row["metrics"][key] - value) < 1e-12, f"Metric mismatch {job_id}:{key}")
                cms.append(cm)
                runs.append({"job_id": job_id, "tier": job["tier_id"], "configuration": job["configuration_id"],
                             "arm": job["arm_id"], "sensor_codes": "+".join(SENSOR_CODES[i] for i in job["sensor_indices"]),
                             "seed": row["seed"], "best_epoch": row["best_epoch"],
                             "parameter_count": row["parameter_count"], "flattened_units": row["flattened_units"],
                             "validation_accuracy_pct": 100 * row["validation_accuracy"], "test_loss": row["test_loss"],
                             "constant_class_prediction": int(np.count_nonzero(cm.sum(axis=0)) == 1),
                             **{key + "_pct": 100 * value for key, value in zip(METRICS, derived)}})
            confusion_by_job[job_id] = np.mean(cms, axis=0)
            confusion_sum_by_job[job_id] = np.sum(cms, axis=0)
            correct_by_job[job_id] = [int(np.trace(cm)) for cm in cms]
            summary = {"job_id": job_id, "tier": job["tier_id"], "configuration": job["configuration_id"],
                       "arm": job["arm_id"], "sensor_codes": "+".join(SENSOR_CODES[i] for i in job["sensor_indices"]),
                       "sensor_names": " + ".join(SENSORS[i] for i in job["sensor_indices"]), "n_seeds": 20,
                       "parameter_count": result["results"][0]["parameter_count"],
                       "flattened_units": result["results"][0]["flattened_units"]}
            for key in METRICS:
                values = np.array([100 * r["metrics"][key] for r in result["results"]])
                summary.update({key + "_mean_pct": float(values.mean()), key + "_sd_pp": float(values.std(ddof=1)),
                                key + "_min_pct": float(values.min()), key + "_max_pct": float(values.max())})
            summaries.append(summary)
        # Rehash the returned state bytes, including case_info binding.
        generation = plans[0]["shared_generation_root"]
        digests = read(generation + "/file_digests.json")
        for item in digests["files"]:
            with archive.open(PREFIX + generation + "/" + item["name"]) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            require(actual == item["sha256"], f"MAT state bytes mismatch {item['name']}")
        require(digest(digests["files"]) == digests["digest_root_sha256"], "MAT root mismatch")
        require(hashlib.sha256(raw(generation + "/case_info.mat")).hexdigest() == records[0]["data_binding"]["case_info_sha256"], "Case metadata hash mismatch")
        for record in records:
            require(record["data_binding"] == records[0]["data_binding"], "Data binding differs between jobs")
            require(record["partition_sha256"] == records[0]["partition_sha256"], "Partition differs between jobs")
            require(record["data_binding"]["verified_state_files"] == digests["files"], "Record states mismatch")
        metadata = loadmat(io.BytesIO(raw(generation + "/case_info.mat")), simplify_cells=True)
        audit["checks"]["case_info_fields"] = sorted(k for k in metadata if not k.startswith("__"))
        audit["checks"].update({"retained_hpo_studies": len(hpo_rows), "completed_hpo_proposals": len(hpo_rows)*100,
                                "retained_report_configurations": len(summaries), "reported_seed_fits": len(runs),
                                "state_file_hashes_verified": len(digests["files"]),
                                "partition_sha256": records[0]["partition_sha256"], "f25_contract_sha256": contract,
                                "python_runtime_source_sha256": records[0]["python_runtime_source_sha256"],
                                "driver_continuations": continuations})
    require(len(hpo_rows) == 22 and len(summaries) == 85 and len(runs) == 1700, "Campaign inventory mismatch")
    # F25-R front-bogie and F25-X frozen front-bogie are matched deterministic controls.
    duplicate_controls = []
    for arm in ("RAW", "PAA"):
        r = artifact_by_job[f"F25-R/{TIERS[0]}/report/{arm}__S01"]["results"]
        x = artifact_by_job[f"F25-X/{TIERS[1]}/report/{arm}__S01"]["results"]
        require(r == x, "Expected frozen-anchor duplicate differs")
        duplicate_controls.append(arm + "__S01")
    audit["checks"]["identical_anchor_control_rows"] = duplicate_controls
    audit["checks"]["constant_class_fits"] = sum(r["constant_class_prediction"] for r in runs)
    audit["checks"]["fits_at_or_below_10pct"] = int(sum(r[METRICS[0] + "_pct"] <= 10 + 1e-9 for r in runs))
    audit["checks"]["best_epoch_1000"] = sum(r["best_epoch"] == 1000 for r in runs)
    audit["checks"]["best_epoch_at_least_950"] = sum(r["best_epoch"] >= 950 for r in runs)
    audit["checks"]["runtime_stacks"] = list({canonical(r["execution_runtime"]["execution_compatibility_descriptor"]).decode()
                                               for r in records})
    audit["checks"]["runtime_stacks"].sort()
    write_csv(out / "configuration_summary.csv", summaries)
    write_csv(out / "seed_metrics.csv", runs)
    write_csv(out / "hpo_summary.csv", hpo_rows)
    write_csv(out / "mean_confusion_matrices.csv", [
        {"job_id": job_id, "true_class": i, "predicted_class": j, "mean_count_per_seed": float(cm[i, j])}
        for job_id, cm in confusion_by_job.items() for i in range(10) for j in range(10)])
    key = METRICS[0] + "_mean_pct"
    best = {tier: max((s for s in summaries if s["tier"] == tier), key=lambda s: s[key]) for tier in TIERS}
    audit["descriptive_best_by_tier"] = best
    # Seed-paired differences, expressly descriptive (same test passages reused).
    differences = []
    for arm in ARMS:
        for sensor in SENSOR_CODES.values():
            fr = next(s for s in summaries if s["tier"] == TIERS[1] and s["arm"] == arm and s["sensor_codes"] == sensor)
            un = next(s for s in summaries if s["tier"] == TIERS[2] and s["arm"] == arm and s["sensor_codes"] == sensor)
            a = np.array([r[METRICS[0] + "_pct"] for r in runs if r["job_id"] == fr["job_id"]])
            b = np.array([r[METRICS[0] + "_pct"] for r in runs if r["job_id"] == un["job_id"]])
            delta = b - a
            differences.append({"arm": arm, "sensor": sensor, "mean_unfrozen_minus_frozen_pp": float(delta.mean()),
                                "sample_sd_paired_difference_pp": float(delta.std(ddof=1)), "positive_seeds": int((delta>0).sum())})
    write_csv(out / "tuning_differences.csv", differences)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    colors = ["#3375a5", "#d57b2a", "#31846b"]
    # Boxes (median, IQR, 1.5-IQR whiskers) plus every run: the 20-run
    # distributions are skewed by collapsed runs, which mean +/- SD would hide.
    jitter = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6), sharey=True)
    for ax, tier, title in zip(axes, TIERS[1:3], ["(a) Shared hyperparameters", "(b) Channel-specific tuning"]):
        for k, arm in enumerate(ARMS):
            data = [next(s for s in summaries if s["tier"] == tier and s["arm"] == arm and s["sensor_codes"] == SENSOR_CODES[i]) for i in SENSORS]
            values = [[100 * r["metrics"][METRICS[0]] for r in artifact_by_job[s["job_id"]]["results"]] for s in data]
            positions = np.arange(6) + (k-1)*.27
            ax.boxplot(values, positions=positions, widths=.22, whis=1.5, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": colors[k], "alpha": .3, "edgecolor": colors[k]},
                       medianprops={"color": colors[k], "lw": 1.4}, whiskerprops={"color": colors[k]},
                       capprops={"color": colors[k]})
            for position, accuracy in zip(positions, values):
                ax.plot(position + jitter.uniform(-.07, .07, len(accuracy)), accuracy, "o", ms=2.2,
                        color=colors[k], mec="none", alpha=.85)
            ax.plot([], [], "s", ms=6, color=colors[k], alpha=.6, label=ARM_LABELS[k])
        ax.set_xticks(range(6), [DISPLAY[c] for c in SENSOR_CODES.values()])
        ax.set_xlim(-.6, 5.6)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 100)
        ax.axhline(10, color=".65", ls="--", lw=.7)
        ax.grid(axis="y", alpha=.2)
        ax.set_xlabel("Response channel")
        ax.set_ylabel("Ten-class test accuracy (%)")
    axes[1].legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"f25_singles.{ext}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.3), sharex=True, sharey=True)
    for ax, arm, title in zip(axes, ARMS, ["(a) RAW", "(b) PAA", "(c) PAA-MR"]):
        values = np.full((6, 6), np.nan)
        texts = {}
        for s in summaries:
            if s["tier"] != TIERS[3] or s["arm"] != arm:
                continue
            i, j = sorted(list(SENSOR_CODES).index(int(code[1:])) for code in s["sensor_codes"].split("+"))
            values[i, j] = s[key]
            texts[i, j] = half_up(sum(correct_by_job[s["job_id"]]), 200, 1)
        im = ax.imshow(np.ma.masked_invalid(values), vmin=10, vmax=85, cmap="viridis")
        for i, j in zip(*np.where(np.isfinite(values))):
            ax.text(j, i, texts[i, j], ha="center", va="center", fontsize=7,
                    color="white" if values[i,j] < 50 else "black")
        ax.set_xticks(range(6), [DISPLAY[c] for c in SENSOR_CODES.values()])
        ax.set_yticks(range(6), [DISPLAY[c] for c in SENSOR_CODES.values()])
        ax.set_title(title, loc="left")
    fig.subplots_adjust(right=.90, bottom=.20, wspace=.16)
    fig.colorbar(im, cax=fig.add_axes([.92, .24, .015, .55]), label="Mean accuracy (%)")
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"f25_pairs.{ext}", dpi=200)
    plt.close(fig)
    selected = best[TIERS[2]]
    cm = confusion_by_job[selected["job_id"]]
    cm_sum = confusion_sum_by_job[selected["job_id"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    im = ax.imshow(cm, vmin=0, vmax=100, cmap="Blues")
    labels = ["Healthy"] + [f"DC{i}" for i in range(2, 11)]
    for i in range(10):
        for j in range(10):
            if cm[i,j] >= .5:
                ax.text(j, i, half_up(cm_sum[i, j], 20, 1), ha="center", va="center", fontsize=7,
                        color="white" if cm[i,j]>60 else "black")
    ax.set_xticks(range(10), labels, rotation=45, ha="right")
    ax.set_yticks(range(10), labels)
    ax.set_xlabel("Predicted scenario")
    ax.set_ylabel("True scenario")
    fig.colorbar(im, ax=ax, label="Mean row percentage over 20 runs (%)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"f25_confusion.{ext}", dpi=200)
    plt.close(fig)
    def cell(row):
        mean = half_up(sum(correct_by_job[row["job_id"]]), 200, 2)
        return f"${mean} \\pm {row[METRICS[0] + '_sd_pp']:.2f}$"

    def median(row):
        middle = sorted(correct_by_job[row["job_id"]])[9:11]
        return half_up(sum(middle), 20, 2)
    for tier, filename in [(TIERS[1], "f25_frozen_singles.tex"), (TIERS[2], "f25_tuned_singles.tex")]:
        lines = ["% Generated by analyze_f25_results.py; percent mean +/- sample SD.",
                 r"\begin{tabular}{llrrr}", r"\toprule", r"Code & Response & RAW & PAA & PAA-MR \\", r"\midrule"]
        for sensor in SENSORS:
            data = [next(s for s in summaries if s["tier"] == tier and s["arm"] == arm and s["sensor_codes"] == SENSOR_CODES[sensor]) for arm in ARMS]
            lines.append(f"{DISPLAY[SENSOR_CODES[sensor]]} & {SENSORS[sensor]} & " + " & ".join(cell(s) for s in data) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (tables / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["% Generated by analyze_f25_results.py; source values: best of 20 runs, from the printed matrices.",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Response & Input & Mean $\pm$ SD (\%) & Min (\%) & Max (\%) & Source, best run (\%) \\", r"\midrule"]
    for s in sorted((s for s in summaries if s["tier"] == TIERS[0]), key=lambda s:(s["sensor_codes"], ARMS.index(s["arm"]))):
        lines.append(f"{DISPLAY[s['sensor_codes']]} & {s['arm'].replace('-CNN','')} & {cell(s)} & {s[METRICS[0] + '_min_pct']:.1f} & {s[METRICS[0] + '_max_pct']:.1f}"
                     f" & {PUBLISHED_BEST_RUN_PCT[s['arm'], s['sensor_codes']]:.1f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "f25_reconstruction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["% Generated by analyze_f25_results.py; exploratory test-read rankings.",
             r"\begin{tabular}{llrr}", r"\toprule", r"Arm & Response pair & Mean $\pm$ SD (\%) & Median (\%) \\", r"\midrule"]
    for arm in ARMS:
        ranked = sorted((s for s in summaries if s["tier"] == TIERS[3] and s["arm"] == arm), key=lambda s:s[key], reverse=True)
        for s in ranked[:3]:
            pair = "+".join(DISPLAY[code] for code in sorted(s["sensor_codes"].split("+")))
            lines.append(f"{ARM_LABELS[ARMS.index(arm)]} & {pair} & {cell(s)} & {median(s)}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "f25_exploratory_pairs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Selected hyperparameters of the source-channel and best tuned models (the
    # F25-R S01 vectors are the frozen anchors of F25-X).
    lines = ["% Generated by analyze_f25_results.py; winners of the 100-proposal searches.",
             r"\begin{tabular}{lllllrrrr}", r"\toprule",
             r"Study & Model & Channel & Filters & Kernels & Dense & Batch & Learning rate & Parameters \\", r"\midrule"]
    for tier, configuration in [(TIERS[0], "RAW__S00"), (TIERS[0], "PAA__S00"), (TIERS[0], "RAW__S01"),
                                (TIERS[0], "PAA__S01"), (TIERS[2], "RAW__S05"), (TIERS[2], "PAA__S05"),
                                (TIERS[2], "PAA-MR__S05")]:
        params = json.loads(next(h for h in hpo_rows if h["tier"] == tier and h["configuration"] == configuration)["params_json"])
        layers = range(params["n_conv_layers"])
        filters = "--".join(f"{params[f'filters_l{i}']}{'$^{p}$' if params[f'pool_l{i}'] else ''}" for i in layers)
        kernels = "--".join(str(params[f"kernel_l{i}"]) for i in layers)
        mantissa, exponent = f"{params['learning_rate']:.1e}".split("e")
        count = next(s for s in summaries if s["tier"] == tier and s["configuration"] == configuration)["parameter_count"]
        arm, code = configuration.split("__")
        lines.append(f"{'Reconstruction' if tier == TIERS[0] else 'Extension, tuned'} & {arm}{'-CNN' if arm != 'PAA-MR' else ''} & {DISPLAY[code]} & {filters} & {kernels}"
                     f" & {params['dense_units']} & {params['batch_size']} & ${mantissa}\\times10^{{{int(exponent)}}}$ & {count:,}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "f25_hyperparameters.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": True, "hpo_studies": len(hpo_rows), "report_configurations": len(summaries),
                      "seed_fits": len(runs), "best_by_tier": {t:{"configuration":s["configuration"], "sensor":s["sensor_names"],
                       "mean_pct":s[key], "sd_pp":s[METRICS[0] + '_sd_pp']} for t,s in best.items()}}, indent=2))


if __name__ == "__main__":
    main()
