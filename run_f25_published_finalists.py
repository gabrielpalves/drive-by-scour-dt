"""Train the published Fernandes et al. (2025) PAA finalists on the F25 data.

The F25 reconstruction (``paper1/``) tuned new CNNs and did not recover the
published best-run accuracies.  This runner trains the two architectures the
source published (Sec. 4.4, Fig. 6) on exactly the F25 states, partition,
noise realisation, training-partition MinMax, PAA (583 x 10), report seeds and
training loop, so the gap can be apportioned:

    published best - F25 tuned best
      = (published best - finalist best)    data/simulator/framework differences
      + (finalist best - F25 tuned best)    optimisation (HPO) outcome

The Keras-defaults arm (``PAA-PUB-KERAS__*``) trains the same two finalists
with Keras initializers and the Keras Adam update, which splits the first term
into framework defaults and data/simulator.

Everything that defines the experiment is imported from the frozen F25 sources
(``core.f25_experiment_contract``, ``core.f25_models``,
``training.f25_executor``); this file only feeds them the published parameter
vectors.  It reads a finished F25 run -- an F25 workspace folder or the
returned ``f25.zip`` -- and never executes code from it: JSON and MAT files are
parsed as data and the two F25 checkpoints used for the data check are loaded
with ``torch.load(weights_only=True)``.

VS Code: open this file and press Run (settings below).  Command line::

    python run_f25_published_finalists.py           # 5 finalists x 20 seeds
    python run_f25_published_finalists.py --smoke   # CPU, 2 seeds, 3 epochs
    python run_f25_published_finalists.py --paper-table   # manuscript table only

Interrupted?  Run it again: finished seeds are kept and skipped.
Details: results/f25_published_finalists/README.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- CONFIGURE (VS Code: edit, then press Run) --------------------------------
# The finished F25 run: an F25 workspace folder (it contains f25_artifacts/)
# or the returned archive f25.zip.  None = this folder when it is an F25
# workspace, otherwise f25.zip next to this file.
F25_SOURCE: Path | None = None
OUTPUT_DIR = ROOT / "results" / "f25_published_finalists"
# Also train the front-bogie finalist with F25's valid/floor layers (71 x 96).
INCLUDE_VALID_PADDING_VARIANT = True
# Also train both finalists with Keras initializers and the Keras Adam update.
INCLUDE_KERAS_DEFAULTS_ARM = True
# ------------------------------------------------------------------------------

import argparse
import contextlib
import csv
import hashlib
import io
import json
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Iterator
from unittest import mock
import zipfile

import numpy as np
import scipy
import torch

# Importing the executor also pins CUBLAS_WORKSPACE_CONFIG before CUDA starts.
from training import f25_executor as executor
from core.f25_experiment_contract import (
    CLASS_COUNT,
    EARLY_STOPPING_PATIENCE_EPOCHS,
    LEARNING_RATE_PLATEAU_FACTOR,
    LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
    MAXIMUM_EPOCHS,
    MINIMUM_LEARNING_RATE,
    PAA_BLOCK_COUNT,
    PASSAGES_PER_CLASS,
    PUBLISHED_OVERALL_ACCURACY,
    REPORT_SEEDS,
    add_source_noise,
    apply_training_minmax,
    build_contract,
    fit_trimmed_training_minmax,
    paa_blocks_of_ten,
    partition_indices,
    partition_sha256,
)
from core.f25_models import build_f25_model
from core.f25_training_contract import SHARED_GENERATION_ROOT, canonical_json_sha256
from f25_published_finalist_models import (
    ALL_FINALISTS,
    F25_ARM_ID,
    FRONT_BOGIE_VALID,
    KERAS_FINALISTS,
    PUBLISHED_FINALISTS,
    FinalistSpec,
    KerasAdam,
    build_finalist_model,
    selftest as model_selftest,
)


REPORT_SCHEMA = "f25-published-finalist-report-v1"
SEED_SCHEMA = "f25-published-finalist-seed-v1"
FINGERPRINT_SCHEMA = "f25-published-finalist-fingerprint-v1"
DATA_CHECK_SCHEMA = "f25-published-finalist-data-check-v1"
REPORT_TIER = "F25-R-unfrozen-source-singles"
BUNDLE_MANIFEST = "f25_bundle_manifest.F25-R.json"
# The sources that define the data, model and training path; each must be
# byte-identical to the bundle that produced the F25 results.
FROZEN_SOURCES = (
    "core/f25_experiment_contract.py",
    "core/f25_models.py",
    "core/f25_training_contract.py",
    "core/temporal_pooling.py",
    "training/f25_executor.py",
)
SMOKE_SEEDS = 2
SMOKE_EPOCHS = 3
# The F25 checkpoints were scored on a CUDA host; the CPU re-score agrees to
# float32 rounding, so the test loss is compared with a relative tolerance.
DATA_CHECK_LOSS_RTOL = 1.0e-4
# Manuscript table (--paper-table), in print order:
# (finalist_id, response, padding, framework defaults).
PAPER_TABLE = ROOT / "paper1" / "tables" / "f25_published_finalists.tex"
PAPER_TABLE_ROWS = (
    ("PAA-PUB__S00", "CV", "None", "PyTorch"),
    ("PAA-PUB-KERAS__S00", "CV", "None", "Keras"),
    ("PAA-PUB__S01", "FV", "Same", "PyTorch"),
    ("PAA-PUB-KERAS__S01", "FV", "Same", "Keras"),
    ("PAA-PUB-VALID__S01", "FV", "None", "PyTorch"),
)


class F25FinalistError(RuntimeError):
    """Raised when the finalist run cannot prove it reuses the F25 inputs."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _report_manifest(configuration_id: str) -> str:
    return f"f25_artifacts/F25-R/manifests/{REPORT_TIER}/report/{configuration_id}"


def _report_results(configuration_id: str) -> str:
    return f"f25_artifacts/F25-R/results/{REPORT_TIER}/report/{configuration_id}"


class F25Source:
    """Read-only view of a finished F25 run: a workspace folder or f25.zip."""

    _RECEIPT = f"{SHARED_GENERATION_ROOT}/file_digests.json"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.archive: zipfile.ZipFile | None = None
        self.prefix = ""
        if self.path.is_dir():
            if not (self.path / self._RECEIPT).is_file():
                raise F25FinalistError(
                    f"{self.path} is not an F25 workspace (no {self._RECEIPT})"
                )
        elif zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            matches = [
                name
                for name in self.archive.namelist()
                if name.endswith("/" + self._RECEIPT) or name == self._RECEIPT
            ]
            if len(matches) != 1:
                raise F25FinalistError(
                    f"{self.path.name} must hold exactly one F25 workspace; "
                    f"found {len(matches)}"
                )
            self.prefix = matches[0][: -len(self._RECEIPT)]
        else:
            raise F25FinalistError(f"F25 source is neither a folder nor a zip: {path}")

    def describe(self) -> str:
        if self.archive is None:
            return f"workspace {self.path}"
        return f"archive {self.path} ({self.prefix or '/'})"

    def exists(self, relative: str) -> bool:
        if self.archive is None:
            return (self.path / relative).is_file()
        try:
            self.archive.getinfo(self.prefix + relative)
        except KeyError:
            return False
        return True

    def read(self, relative: str) -> bytes:
        if self.archive is None:
            return (self.path / relative).read_bytes()
        try:
            return self.archive.read(self.prefix + relative)
        except KeyError as exc:
            raise F25FinalistError(f"{relative} is missing from {self.path.name}") from exc

    def read_json(self, relative: str) -> tuple[dict[str, Any], bytes]:
        raw = self.read(relative)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise F25FinalistError(f"{relative} is not one JSON object")
        return value, raw

    @contextlib.contextmanager
    def generation_dir(self, state_names: list[str]) -> Iterator[Path]:
        """Yield a folder holding the state files (extracted to local temp)."""

        if self.archive is None:
            yield self.path / SHARED_GENERATION_ROOT
            return
        with tempfile.TemporaryDirectory(prefix="f25_finalists_") as temporary:
            target = Path(temporary)
            for name in state_names:
                member = f"{self.prefix}{SHARED_GENERATION_ROOT}/{name}"
                with self.archive.open(member) as source, (target / name).open(
                    "xb"
                ) as sink:
                    while chunk := source.read(1 << 22):
                        sink.write(chunk)
            yield target


def default_source() -> Path:
    if (ROOT / SHARED_GENERATION_ROOT / "file_digests.json").is_file():
        return ROOT
    if (ROOT / "f25.zip").is_file():
        return ROOT / "f25.zip"
    raise F25FinalistError(
        "No F25 run found: put this file in the F25 workspace, place f25.zip "
        "next to it, or set F25_SOURCE / --source."
    )


def load_reference(source: F25Source, *, allow_source_drift: bool) -> dict[str, Any]:
    """Authenticate the F25 dataset binding, report results and sources."""

    contract_sha = build_contract()["contract_sha256"]
    receipt, _ = source.read_json(f"{SHARED_GENERATION_ROOT}/file_digests.json")
    marker = source.read(f"{SHARED_GENERATION_ROOT}/_F25_GENERATION_COMPLETE")
    if (
        receipt.get("python_contract_sha256") != contract_sha
        or marker.decode("utf-8")
        != (
            "schema=f25-generation-complete-v1\n"
            f"gen_fingerprint={receipt.get('gen_fingerprint')}\n"
            f"python_contract_sha256={contract_sha}\n"
            f"digest_root_sha256={receipt.get('digest_root_sha256')}\n"
        )
    ):
        raise F25FinalistError("F25 generation receipt/marker do not bind this contract")
    receipt_files = [
        {"name": row["name"], "sha256": row["sha256"]} for row in receipt["files"]
    ]

    configurations: dict[str, Any] = {}
    for spec in PUBLISHED_FINALISTS:
        config_id = spec.configuration_id
        run_record, run_record_raw = source.read_json(
            f"{_report_manifest(config_id)}/run_record.json"
        )
        metrics, metrics_raw = source.read_json(
            f"{_report_results(config_id)}/metrics.json"
        )
        job = run_record.get("job", {})
        if (
            run_record.get("f25_contract_sha256") != contract_sha
            or metrics.get("f25_contract_sha256") != contract_sha
            or run_record.get("partition_sha256") != partition_sha256()
            or job.get("arm_id") != F25_ARM_ID
            or job.get("sensor_indices") != [spec.sensor_index]
            or metrics.get("job_id") != job.get("job_id")
            or metrics.get("report_seeds") != list(REPORT_SEEDS)
            or [row["seed"] for row in metrics.get("results", [])]
            != list(REPORT_SEEDS)
        ):
            raise F25FinalistError(f"F25 report {config_id} is foreign or incomplete")
        configurations[config_id] = {
            "run_record": run_record,
            "run_record_sha256": _sha256_bytes(run_record_raw),
            "metrics": metrics,
            "metrics_sha256": _sha256_bytes(metrics_raw),
        }

    bindings = [
        {key: value for key, value in entry["run_record"]["data_binding"].items() if key != "root"}
        for entry in configurations.values()
    ]
    binding = configurations[PUBLISHED_FINALISTS[0].configuration_id]["run_record"][
        "data_binding"
    ]
    if (
        any(other != bindings[0] for other in bindings[1:])
        or binding.get("verified_state_files") != receipt_files
        or binding.get("gen_fingerprint") != receipt.get("gen_fingerprint")
        or binding.get("verified_state_root_sha256")
        != canonical_json_sha256(receipt_files)
    ):
        raise F25FinalistError("F25 report jobs do not share the receipted dataset")

    sources: dict[str, Any] = {}
    manifest_commit = None
    if source.exists(BUNDLE_MANIFEST):
        manifest, _ = source.read_json(BUNDLE_MANIFEST)
        manifest_commit = manifest.get("source_commit")
        recorded = {row["path"]: row["sha256"] for row in manifest["source_files"]}
    else:
        recorded = {}
    for relative in FROZEN_SOURCES:
        local = _sha256_bytes((ROOT / relative).read_bytes())
        sources[relative] = {
            "local_sha256": local,
            "f25_bundle_sha256": recorded.get(relative),
            "identical": local == recorded.get(relative),
        }
    drift = [name for name, row in sources.items() if not row["identical"]]
    if drift or manifest_commit != binding.get("bundle_source_commit"):
        message = (
            "local F25 sources differ from the bundle that produced the F25 "
            f"results: {drift or 'bundle manifest/commit unavailable'}"
        )
        if not allow_source_drift:
            raise F25FinalistError(message + " (override: --allow-source-drift)")
        print(f"WARNING: {message}")
    return {
        "binding": binding,
        "receipt_files": receipt_files,
        "configurations": configurations,
        "frozen_sources": sources,
        "bundle_source_commit": manifest_commit,
    }


def prepare_channel(
    clean: np.ndarray, physical_channel: int, binding: dict[str, Any]
) -> dict[str, Any]:
    """Noise -> training MinMax -> PAA, statement for statement as
    ``executor._prepare_dataset`` does after its provenance gate.

    ``clean`` holds one channel, (2000, 1, 5830), class-major rows.
    """

    noisy = np.empty(clean.shape, dtype=np.float32)
    for class_index in range(CLASS_COUNT):
        for passage_index in range(PASSAGES_PER_CLASS):
            row = class_index * PASSAGES_PER_CLASS + passage_index
            noisy[row, 0] = add_source_noise(
                clean[row, 0],
                class_index=class_index,
                passage_index=passage_index,
                channel_index=physical_channel,
            ).astype(np.float32)
    split = partition_indices()
    local_passage = np.tile(np.arange(PASSAGES_PER_CLASS), CLASS_COUNT)
    masks = {
        name: np.isin(local_passage, indices) for name, indices in split.items()
    }
    calibration = fit_trimmed_training_minmax(
        noisy[masks["train"]], channel_axis=1, time_axis=2
    )
    scaled = apply_training_minmax(noisy, calibration, channel_axis=1).astype(
        np.float32
    )
    del noisy
    scaled = paa_blocks_of_ten(scaled, time_axis=2).astype(np.float32)
    if scaled.shape[-1] != PAA_BLOCK_COUNT:
        raise F25FinalistError("F25 PAA did not yield 583 blocks")
    labels = np.repeat(np.arange(CLASS_COUNT, dtype=np.int64), PASSAGES_PER_CLASS)
    return {
        "X": scaled,
        "y": labels,
        "masks": masks,
        "minmax": {
            "minimum": list(calibration.minimum),
            "maximum": list(calibration.maximum),
        },
        "partition_sha256": partition_sha256(),
        "data_binding": binding,
        "selected_channels": 1,
    }


def load_datasets(
    source: F25Source, reference: dict[str, Any], channels: list[int]
) -> dict[int, dict[str, Any]]:
    """Load clean states with the executor's own loader, then prepare each channel."""

    names = [row["name"] for row in reference["receipt_files"]]
    started = time.perf_counter()
    with source.generation_dir(names) as generation_dir:
        # Verifies every state's SHA-256 (before and after reading) and its
        # payload semantics against the binding the F25 report jobs used.
        clean = executor._load_clean_selected(
            generation_dir, channels, reference["binding"]
        )
    datasets = {
        channel: prepare_channel(
            clean[:, [local], :], channel, reference["binding"]
        )
        for local, channel in enumerate(channels)
    }
    print(
        f"dataset ready: {clean.shape[0]} passages, channels {channels}, "
        f"{time.perf_counter() - started:.0f} s"
    )
    return datasets


def check_data(
    source: F25Source, reference: dict[str, Any], datasets: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Re-score the F25 best-run checkpoints on the prepared test partitions.

    Identical confusion matrices (and test loss to float32 rounding) prove the
    finalists see the very inputs the F25 report runs saw.
    """

    rows = []
    for spec in PUBLISHED_FINALISTS:
        config_id = spec.configuration_id
        relative = f"{_report_results(config_id)}/best_visual_checkpoint.pt"
        raw = source.read(relative)
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        metrics = reference["configurations"][config_id]["metrics"]
        recorded = next(
            row for row in metrics["results"] if row["seed"] == checkpoint["seed"]
        )
        if checkpoint["params"] != metrics["params"]:
            raise F25FinalistError(f"{config_id} checkpoint and metrics disagree")
        executor._set_determinism(0)
        model = build_f25_model(
            arm_id=F25_ARM_ID, in_channels=1, params=checkpoint["params"], device="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = datasets[spec.sensor_index]
        test = dataset["masks"]["test"]
        X_test = torch.from_numpy(np.ascontiguousarray(dataset["X"][test]))
        y_test = torch.from_numpy(np.ascontiguousarray(dataset["y"][test]))
        loss, _accuracy, predicted = executor._evaluate(
            model, X_test, y_test, int(checkpoint["params"]["batch_size"]),
            torch.device("cpu"),
        )
        reproduced = executor._classification_metrics(y_test.numpy(), predicted)
        expected_matrix = recorded["metrics"]["confusion_matrix"]
        relative_loss_error = abs(loss - recorded["test_loss"]) / recorded["test_loss"]
        row = {
            "configuration_id": config_id,
            "checkpoint_sha256": _sha256_bytes(raw),
            "seed": int(checkpoint["seed"]),
            "recorded_overall_accuracy": recorded["metrics"]["overall_ten_class_accuracy"],
            "reproduced_overall_accuracy": reproduced["overall_ten_class_accuracy"],
            "recorded_test_loss": recorded["test_loss"],
            "reproduced_test_loss": loss,
            "relative_test_loss_error": relative_loss_error,
            "confusion_matrix_identical": (
                reproduced["confusion_matrix"] == expected_matrix
                and checkpoint["metrics"]["confusion_matrix"] == expected_matrix
            ),
        }
        row["passed"] = bool(
            row["confusion_matrix_identical"]
            and relative_loss_error <= DATA_CHECK_LOSS_RTOL
        )
        rows.append(row)
        print(
            f"data check {config_id}: F25 seed {row['seed']} re-scored "
            f"{row['reproduced_overall_accuracy']:.4f} (recorded "
            f"{row['recorded_overall_accuracy']:.4f}), confusion matrix "
            f"{'identical' if row['confusion_matrix_identical'] else 'DIFFERENT'}, "
            f"test-loss rel. error {relative_loss_error:.1e}"
        )
    return {
        "schema": DATA_CHECK_SCHEMA,
        "method": (
            "F25 best-run checkpoints re-scored on this run's prepared test "
            "partition (CPU)"
        ),
        "loss_relative_tolerance": DATA_CHECK_LOSS_RTOL,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def runtime_descriptor(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if cuda else platform.processor()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": "torch.use_deterministic_algorithms(True), "
        "cudnn.deterministic, no TF32 (executor._set_determinism)",
    }


def fingerprint(
    spec: FinalistSpec,
    dataset: dict[str, Any],
    reference: dict[str, Any],
    runtime: dict[str, Any],
    max_epochs: int,
) -> dict[str, Any]:
    """Everything a stored seed result depends on; resuming requires equality."""

    return {
        "schema": FINGERPRINT_SCHEMA,
        "finalist": spec.describe(),
        "params": spec.params,
        "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE_EPOCHS,
        "plateau": [
            LEARNING_RATE_PLATEAU_FACTOR,
            LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
            MINIMUM_LEARNING_RATE,
        ],
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "gen_fingerprint": reference["binding"]["gen_fingerprint"],
        "verified_state_root_sha256": reference["binding"]["verified_state_root_sha256"],
        "partition_sha256": dataset["partition_sha256"],
        "minmax": dataset["minmax"],
        "frozen_sources": {
            name: row["local_sha256"] for name, row in reference["frozen_sources"].items()
        },
        "runtime": {
            key: runtime[key]
            for key in ("torch", "torch_cuda", "cudnn", "device", "device_name")
        },
    }


@contextlib.contextmanager
def finalist_factory(spec: FinalistSpec) -> Iterator[None]:
    """Let ``executor._train_once`` build ``spec`` instead of an F25 winner.

    ``_train_once`` resolves ``build_f25_model`` from its module globals right
    after seeding; swapping that one name reuses the frozen loop unchanged.
    For the Keras arm it also resolves ``torch.optim.Adam`` at call time, which
    is swapped for :class:`KerasAdam` the same way.
    """

    def build(*, arm_id: str, in_channels: int, params: dict[str, Any], device: Any):
        if arm_id != F25_ARM_ID or in_channels != 1 or params != spec.params:
            raise F25FinalistError("executor asked for a model other than the finalist")
        return build_finalist_model(spec, device=device)

    optimizers: list[KerasAdam] = []

    class TrackedKerasAdam(KerasAdam):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            optimizers.append(self)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(executor, "build_f25_model", build))
        if spec.uses_keras_defaults:
            stack.enter_context(mock.patch.object(torch.optim, "Adam", TrackedKerasAdam))
        yield
    if spec.uses_keras_defaults and (
        len(optimizers) != 1 or optimizers[0].keras_step == 0
    ):
        raise F25FinalistError(f"{spec.finalist_id} did not train with the Keras Adam")


def train_seed(
    spec: FinalistSpec,
    dataset: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    max_epochs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with finalist_factory(spec):
        result = executor._train_once(
            job={"arm_id": F25_ARM_ID, "representation": "PAA",
                 "sensor_indices": [spec.sensor_index]},
            dataset=dataset,
            params=spec.params,
            seed=seed,
            device=device,
            max_epochs=max_epochs,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS,
            evaluate_test=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if (
        result["flattened_units"] != spec.flatten_units
        or result["parameter_count"] != spec.parameter_count
    ):
        raise F25FinalistError(f"{spec.finalist_id} trained a different network")
    return {
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "validation_loss": result["validation_loss"],
        "validation_accuracy": result["validation_accuracy"],
        "test_loss": result["test_loss"],
        "parameter_count": result["parameter_count"],
        "flattened_units": result["flattened_units"],
        "metrics": executor._classification_metrics(
            result["test_truth"], result["test_prediction"]
        ),
        # _train_once stops `patience` epochs after the best one.
        "epochs_run": min(result["best_epoch"] + EARLY_STOPPING_PATIENCE_EPOCHS, max_epochs),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def single_class(row: dict[str, Any]) -> bool:
    """All test passages predicted as one class."""

    columns = np.asarray(row["metrics"]["confusion_matrix"]).sum(axis=0)
    return int(np.count_nonzero(columns)) == 1


def at_chance(row: dict[str, Any]) -> bool:
    """Test accuracy no better than guessing among ten balanced classes."""

    return row["metrics"]["overall_ten_class_accuracy"] <= 1.0 / CLASS_COUNT + 1.0e-12


def failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "chance_level_runs": sum(at_chance(row) for row in rows),
        "single_class_runs": sum(single_class(row) for row in rows),
    }


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def percent(fraction: float, places: int = 1) -> str:
    """Fraction -> percent rounded half-up, as in analyze_f25_results.half_up.

    Accuracies are multiples of 0.001, so a median or mean can land exactly on
    a rounding tie (0.5795 -> 57.95 %); plain float formatting prints 57.9.
    """

    exact = Decimal(f"{100 * fraction:.6f}")
    return str(exact.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [row["metrics"]["overall_ten_class_accuracy"] for row in rows]
    best = max(range(len(rows)), key=lambda index: (overall[index], -index))
    summary: dict[str, Any] = {
        "runs": len(rows),
        "overall_ten_class_accuracy": distribution(overall),
        "best_seed": rows[best]["seed"],
        **failure_counts(rows),
        "median_best_epoch": statistics.median(row["best_epoch"] for row in rows),
        "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in rows),
    }
    for metric in (
        "bearing_present_absent_accuracy",
        "scour_level_accuracy",
        "crack_level_accuracy",
    ):
        summary[metric] = distribution([row["metrics"][metric] for row in rows])
    return summary


def comparison(
    spec: FinalistSpec, summary: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    entry = reference["configurations"][spec.configuration_id]
    tuned = distribution(
        [row["metrics"]["overall_ten_class_accuracy"] for row in entry["metrics"]["results"]]
    )
    published = next(
        row.paa_cnn for row in PUBLISHED_OVERALL_ACCURACY if row.sensor_id == spec.sensor_id
    )
    finalist = summary["overall_ten_class_accuracy"]
    return {
        "published_best_run_overall_accuracy": published,
        "published_source": (
            "Fernandes et al. (2025) Figs. 9/12 diagonal sums "
            "(core.f25_experiment_contract.PUBLISHED_OVERALL_ACCURACY)"
        ),
        "f25_tuned": {
            "configuration_id": spec.configuration_id,
            "job_id": entry["metrics"]["job_id"],
            "metrics_sha256": entry["metrics_sha256"],
            "params": entry["metrics"]["params"],
            "overall_ten_class_accuracy": tuned,
            **failure_counts(entry["metrics"]["results"]),
        },
        "gap_published_minus_f25_tuned_best": published - tuned["max"],
        "gap_published_minus_finalist_best": published - finalist["max"],
        "gap_finalist_best_minus_f25_tuned_best": finalist["max"] - tuned["max"],
        "gap_finalist_mean_minus_f25_tuned_mean": finalist["mean"] - tuned["mean"],
    }


def run_finalist(
    spec: FinalistSpec,
    dataset: dict[str, Any],
    reference: dict[str, Any],
    data_check: dict[str, Any] | None,
    *,
    seeds: list[int],
    device: torch.device,
    max_epochs: int,
    runtime: dict[str, Any],
    out_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    directory = out_dir / spec.finalist_id
    stamp = fingerprint(spec, dataset, reference, runtime, max_epochs)
    stamp_sha = canonical_json_sha256(stamp)
    stamp_path = directory / "fingerprint.json"
    if stamp_path.is_file():
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(stored) != stamp_sha:
            raise F25FinalistError(
                f"{directory} holds seeds from another machine, data or setting; "
                "move it away or choose another output folder"
            )
    else:
        executor._atomic_json(stamp_path, stamp)

    rows = []
    trained_seconds: list[float] = []
    for index, seed in enumerate(seeds, start=1):
        path = directory / "runs" / f"seed_{seed}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("fingerprint_sha256") != stamp_sha or record["row"]["seed"] != seed:
                raise F25FinalistError(f"{path} belongs to another run")
            row, state = record["row"], "kept"
        else:
            row = train_seed(spec, dataset, seed=seed, device=device, max_epochs=max_epochs)
            executor._atomic_json(
                path,
                {"schema": SEED_SCHEMA, "fingerprint_sha256": stamp_sha, "row": row},
            )
            trained_seconds.append(row["wall_seconds"])
            state = "trained"
        rows.append(row)
        remaining = len(seeds) - index
        eta = (
            f", ~{remaining * statistics.fmean(trained_seconds) / 60:.0f} min "
            "left on this finalist"
            if trained_seconds and remaining
            else ""
        )
        print(
            f"[{spec.finalist_id} {index:>2}/{len(seeds)}] seed {seed} {state}: "
            f"test acc {row['metrics']['overall_ten_class_accuracy']:.4f}, "
            f"best epoch {row['best_epoch']} of {row['epochs_run']}, "
            f"{row['wall_seconds']:.0f} s{eta}",
            flush=True,
        )

    summary = summarize(rows)
    entry = reference["configurations"][spec.configuration_id]
    artifact = {
        "schema": REPORT_SCHEMA,
        "job_id": f"F25-PF/published-finalists/report/{spec.finalist_id}",
        "smoke": smoke,
        "finalist": spec.describe(),
        "f25_contract_sha256": build_contract()["contract_sha256"],
        "params": spec.params,
        "params_source": (
            "core.f25_experiment_contract.PUBLISHED_PAA_CNN_TEMPLATES"
            f"[{spec.template_index}]"
        ),
        "report_seeds": seeds,
        "results": rows,
        "best_run_role": "visual comparability only; distribution uses all runs",
        "summary": summary,
        "comparison": comparison(spec, summary, reference),
        "training_protocol": {
            "loop": "training.f25_executor._train_once (frozen F25 loop, unmodified)",
            "optimizer": (
                "Adam with the Keras update "
                "(f25_published_finalist_models.KerasAdam, epsilon 1e-7)"
                if spec.uses_keras_defaults
                else "Adam"
            ),
            "initialization": (
                "Keras: glorot_uniform kernels, zero biases"
                if spec.uses_keras_defaults
                else "PyTorch module defaults"
            ),
            "loss": "cross-entropy on logits (softmax output)",
            "max_epochs": max_epochs,
            "early_stopping_patience_epochs": EARLY_STOPPING_PATIENCE_EPOCHS,
            "plateau_factor": LEARNING_RATE_PLATEAU_FACTOR,
            "plateau_patience_epochs": LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "restored_weights": "lowest validation loss",
        },
        "data": {
            "data_binding_source": f"{_report_manifest(spec.configuration_id)}/run_record.json",
            "data_binding_source_sha256": entry["run_record_sha256"],
            "gen_fingerprint": reference["binding"]["gen_fingerprint"],
            "verified_state_root_sha256": reference["binding"]["verified_state_root_sha256"],
            "bundle_source_commit": reference["binding"]["bundle_source_commit"],
            "partition_sha256": dataset["partition_sha256"],
            "minmax": dataset["minmax"],
            "split_sizes": {
                name: int(mask.sum()) for name, mask in dataset["masks"].items()
            },
            "data_check": data_check,
        },
        "frozen_sources": reference["frozen_sources"],
        "runtime": runtime,
        "f25_report_accelerator": entry["run_record"]["execution_runtime"][
            "execution_environment_descriptor"
        ]["accelerator"]["name"],
        "fingerprint_sha256": stamp_sha,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    executor._atomic_json(directory / "metrics.json", artifact)
    return artifact


SUMMARY_FIELDS = (
    "finalist_id", "configuration_id", "role", "framework_defaults", "conv_padding",
    "pool_ceil_mode", "flatten_units", "published_flatten_units", "parameter_count", "runs",
    "mean", "sd", "min", "max", "median", "chance_level_runs", "single_class_runs",
    "median_best_epoch", "mean_wall_seconds", "published_best_run",
    "f25_tuned_mean", "f25_tuned_best", "f25_tuned_chance_level_runs",
    "gap_published_minus_finalist_best", "gap_finalist_best_minus_f25_tuned_best",
    "smoke",
)


def write_summary(artifacts: list[dict[str, Any]], out_dir: Path) -> Path:
    rows = []
    for artifact in artifacts:
        finalist = artifact["finalist"]
        overall = artifact["summary"]["overall_ten_class_accuracy"]
        compared = artifact["comparison"]
        rows.append({
            "finalist_id": finalist["finalist_id"],
            "configuration_id": finalist["configuration_id"],
            "role": finalist["role"],
            "framework_defaults": finalist.get("framework_defaults", "pytorch"),
            "conv_padding": finalist["conv_padding"],
            "pool_ceil_mode": finalist["pool_ceil_mode"],
            "flatten_units": finalist["flatten_units"],
            "published_flatten_units": finalist["published_flatten_units"],
            "parameter_count": finalist["parameter_count"],
            "runs": artifact["summary"]["runs"],
            **{key: f"{overall[key]:.6f}" for key in ("mean", "sd", "min", "max", "median")},
            "chance_level_runs": artifact["summary"]["chance_level_runs"],
            "single_class_runs": artifact["summary"]["single_class_runs"],
            "median_best_epoch": artifact["summary"]["median_best_epoch"],
            "mean_wall_seconds": f"{artifact['summary']['mean_wall_seconds']:.1f}",
            "published_best_run": f"{compared['published_best_run_overall_accuracy']:.3f}",
            "f25_tuned_mean": f"{compared['f25_tuned']['overall_ten_class_accuracy']['mean']:.6f}",
            "f25_tuned_best": f"{compared['f25_tuned']['overall_ten_class_accuracy']['max']:.6f}",
            "f25_tuned_chance_level_runs": compared["f25_tuned"]["chance_level_runs"],
            "gap_published_minus_finalist_best": f"{compared['gap_published_minus_finalist_best']:.6f}",
            "gap_finalist_best_minus_f25_tuned_best": f"{compared['gap_finalist_best_minus_f25_tuned_best']:.6f}",
            "smoke": artifact["smoke"],
        })
    path = out_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def write_paper_table(out_dir: Path, path: Path = PAPER_TABLE) -> Path:
    """Write the manuscript table from the stored confusion matrices.

    Accuracies are recounted from the diagonals and rounded half-up from exact
    decimals, like the tables of analyze_f25_results.py.
    """

    def rounded(value: Decimal, places: int) -> str:
        return str(value.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))

    lines = [
        "% Generated by run_f25_published_finalists.py --paper-table; "
        "percent over the 20 report seeds.",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Response & Padding & Defaults & Mean $\pm$ SD (\%) & Median (\%) "
        r"& Max (\%) & Runs at 10\% \\",
        r"\midrule",
    ]
    for finalist_id, response, padding, defaults in PAPER_TABLE_ROWS:
        report = json.loads((out_dir / finalist_id / "metrics.json").read_text(encoding="utf-8"))
        rows = report["results"]
        if report["smoke"] or [row["seed"] for row in rows] != list(REPORT_SEEDS):
            raise F25FinalistError(f"{finalist_id}: not a full report over the 20 seeds")
        values = []
        for row in rows:
            matrix = np.asarray(row["metrics"]["confusion_matrix"], dtype=np.int64)
            value = Decimal(100 * int(np.trace(matrix))) / Decimal(int(matrix.sum()))
            if abs(float(value) / 100 - row["metrics"]["overall_ten_class_accuracy"]) > 1.0e-9:
                raise F25FinalistError(f"{finalist_id} seed {row['seed']}: accuracy != matrix")
            values.append(value)
        with localcontext() as context:
            context.prec = 50
            mean = sum(values) / len(values)
            sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)).sqrt()
            median = statistics.median(values)
        at_ten = sum(value <= Decimal(100) / CLASS_COUNT for value in values)
        lines.append(
            f"{response} & {padding} & {defaults} & ${rounded(mean, 2)} \\pm {rounded(sd, 2)}$ & "
            f"{rounded(median, 2)} & {rounded(max(values), 1)} & {at_ten}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise F25FinalistError("CUDA was requested but this PyTorch sees no GPU")
    return torch.device(choice)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=F25_SOURCE,
                        help="F25 workspace folder or f25.zip (default: auto)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None,
                        help="default: cuda if available (cpu for --smoke)")
    parser.add_argument("--finalists", nargs="+",
                        choices=[spec.finalist_id for spec in ALL_FINALISTS])
    parser.add_argument("--no-valid-variant", action="store_true",
                        help="skip the valid/floor front-bogie sensitivity arm")
    parser.add_argument("--no-keras-arm", action="store_true",
                        help="skip the Keras-defaults (initializers + Adam) arm")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_SEEDS} seeds x {SMOKE_EPOCHS} epochs into <out-dir>/smoke")
    parser.add_argument("--smoke-seeds", type=int, default=SMOKE_SEEDS)
    parser.add_argument("--smoke-epochs", type=int, default=SMOKE_EPOCHS)
    parser.add_argument("--skip-data-check", action="store_true",
                        help="do not re-score the F25 checkpoints (not recommended)")
    parser.add_argument("--allow-source-drift", action="store_true",
                        help="run even if local F25 sources differ from the F25 bundle")
    parser.add_argument("--paper-table", action="store_true",
                        help=f"only write {PAPER_TABLE.relative_to(ROOT).as_posix()} "
                             "from the stored results (no training)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.paper_table:
        print(f"written: {write_paper_table(args.out_dir)}")
        return 0
    if args.finalists:
        specs = [spec for spec in ALL_FINALISTS if spec.finalist_id in args.finalists]
    else:
        specs = list(PUBLISHED_FINALISTS)
        if INCLUDE_VALID_PADDING_VARIANT and not args.no_valid_variant:
            specs.append(FRONT_BOGIE_VALID)
        if INCLUDE_KERAS_DEFAULTS_ARM and not args.no_keras_arm:
            specs.extend(KERAS_FINALISTS)
    if args.smoke:
        if not 1 <= args.smoke_seeds <= len(REPORT_SEEDS) or args.smoke_epochs < 1:
            raise F25FinalistError("smoke needs 1-20 seeds and at least one epoch")
        seeds = list(REPORT_SEEDS[: args.smoke_seeds])
        max_epochs = args.smoke_epochs
        out_dir = args.out_dir / "smoke"
        device = resolve_device(args.device or "cpu")
    else:
        seeds = list(REPORT_SEEDS)
        max_epochs = MAXIMUM_EPOCHS
        out_dir = args.out_dir
        device = resolve_device(args.device or "auto")
        if device.type == "cpu":
            print("WARNING: no GPU; the full job on CPU takes many hours.")

    for row in model_selftest():
        print(
            f"{row['finalist_id']:<19} {row['feature_map'][0]} x "
            f"{row['feature_map'][1]} = {row['flatten_units']:,} flattened "
            f"(published {row['published_flatten_units']:,}), "
            f"{row['parameter_count']:,} parameters"
        )
    source = F25Source(args.source or default_source())
    print(f"F25 source: {source.describe()}")
    reference = load_reference(source, allow_source_drift=args.allow_source_drift)
    channels = sorted({spec.sensor_index for spec in PUBLISHED_FINALISTS})
    datasets = load_datasets(source, reference, channels)
    data_check = None
    if not args.skip_data_check:
        data_check = check_data(source, reference, datasets)
        executor._atomic_json(out_dir / "data_check.json", data_check)
        if not data_check["passed"]:
            raise F25FinalistError(
                "prepared data do not reproduce the F25 report checkpoints; "
                f"see {out_dir / 'data_check.json'}"
            )

    runtime = runtime_descriptor(device)
    print(
        f"training on {runtime['device_name']} ({device.type}); seeds "
        f"{seeds[0]}..{seeds[-1]} ({len(seeds)}), max {max_epochs} epochs"
        f"{'  [SMOKE]' if args.smoke else ''}\n"
    )
    artifacts = [
        run_finalist(
            spec,
            datasets[spec.sensor_index],
            reference,
            data_check,
            seeds=seeds,
            device=device,
            max_epochs=max_epochs,
            runtime=runtime,
            out_dir=out_dir,
            smoke=args.smoke,
        )
        for spec in specs
    ]
    summary_path = write_summary(artifacts, out_dir)

    print(
        f"\n{'test accuracy, %':<20} {'flatten':>7} {'params':>9}  "
        f"{'mean+-sd':>11}  {'min':>5}  {'med':>5}  {'max':>5} | "
        f"{'F25 tuned max':>13} | {'published':>9}"
    )
    for artifact in artifacts:
        overall = artifact["summary"]["overall_ten_class_accuracy"]
        compared = artifact["comparison"]
        print(
            f"{artifact['finalist']['finalist_id']:<20} "
            f"{artifact['finalist']['flatten_units']:>7,} "
            f"{artifact['finalist']['parameter_count']:>9,}  "
            f"{percent(overall['mean']):>5}+-{percent(overall['sd']):>4}  "
            f"{percent(overall['min']):>5}  {percent(overall['median']):>5}  "
            f"{percent(overall['max']):>5} | "
            f"{percent(compared['f25_tuned']['overall_ten_class_accuracy']['max']):>13} | "
            f"{percent(compared['published_best_run_overall_accuracy']):>9}"
        )
    print(f"\nwritten: {summary_path} and <finalist>/metrics.json under {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; finished seeds are saved -- run again to continue")
        raise SystemExit(130)
    except (F25FinalistError, executor.F25ExecutionError) as error:
        print(f"F25 FINALISTS ABORTED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
