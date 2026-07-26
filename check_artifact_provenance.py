"""Adversarial checks for Optuna protocol and champion-weight provenance.

Run with the campaign environment:
    python check_artifact_provenance.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import optuna
import torch

from core.dataset import _cache_stem
from core.artifact_provenance import verify_standalone_dt_package
from core.protocol import protocol_hash
from training.pipeline import (
    _stamp_study_protocol,
    export_digital_twin_package,
    verify_digital_twin_package,
)


fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    ok = bool(condition)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails += int(not ok)


def rejects(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        check(name, True)
    else:
        check(name, False)


def rejects_with_message(name: str, fn, message: str) -> None:
    """Require the intended fail-closed error, not an incidental exception."""

    try:
        fn()
    except RuntimeError as exc:
        check(name, message in str(exc))
    except Exception:
        check(name, False)
    else:
        check(name, False)


optuna.logging.set_verbosity(optuna.logging.WARNING)
config = {
    "name": "artifact-fixture",
    "name_short": "PAA_CNN",
    "method": "PAA",
    "dofs": [1, 3],
    "task": "regression",
    "target_supports": [2, 3],
    "bearing_targets": None,
    "discretization": 1,
    "seed": 42,
    "protocol_hash": None,
    "protocol_descriptor": {
        "core": {"protocol_version": 2},
        "rung": {"dataset": "fixture"},
        # Tuple is intentional: Optuna's SQLite JSON round-trip returns a list.
        "search_space_fixture": {"learning_rate": (1e-4, 1e-3)},
    },
}
config["protocol_hash"] = protocol_hash(config["protocol_descriptor"])

with tempfile.TemporaryDirectory(prefix="artifact-prov-") as tmp:
    output = Path(tmp, "out")
    cache = Path(tmp, "cache")
    output.mkdir()
    cache.mkdir()

    storage_url = f"sqlite:///{Path(tmp, 'study.db').resolve().as_posix()}"
    storage = optuna.storages.RDBStorage(storage_url)
    study = optuna.create_study(
        study_name="artifact-fixture", storage=storage
    )
    _stamp_study_protocol(
        study,
        config=config,
        dataset_name="fixture",
        n_trials=1,
        epochs=2,
        sampler_seed=42,
        use_pruner=False,
    )
    study = optuna.load_study(
        study_name="artifact-fixture", storage=storage
    )
    _stamp_study_protocol(
        study,
        config=config,
        dataset_name="fixture",
        n_trials=1,
        epochs=2,
        sampler_seed=42,
        use_pruner=False,
    )
    check("SQLite restart accepts canonically identical tuple/list protocol",
          study.user_attrs["ttbi_protocol_record"]["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"])))
    study.optimize(lambda trial: trial.suggest_float("lr", 1e-4, 1e-3), n_trials=1)
    record = study.user_attrs.get("ttbi_protocol_record")
    check("study stores full protocol descriptor",
          record["protocol_hash"] == config["protocol_hash"]
          and record["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"])))

    trial_path = output / (
        f"weights_{config['name']}_trial_{study.best_trial.number}.pth"
    )
    torch.save({"weight": torch.arange(8, dtype=torch.float32)}, trial_path)
    scaler_path = cache / f"scaler_{_cache_stem('fixture', config)}.pkl"
    joblib.dump({"mean": [0.0, 0.0], "scale": [1.0, 1.0]}, scaler_path)
    scaler_sha = __import__("hashlib").sha256(scaler_path.read_bytes()).hexdigest()
    cache_prov = cache / f"cache_{_cache_stem('fixture', config)}_prov.json"
    cache_prov.write_text(json.dumps({
        "source": {"fixture": True},
        "artifacts": {"scaler": scaler_sha},
    }), encoding="utf-8")
    scaler_source_bytes = scaler_path.read_bytes()
    tampered_source = bytearray(scaler_source_bytes)
    tampered_source[-1] ^= 1
    scaler_path.write_bytes(tampered_source)
    rejects("pre-export cache-scaler tamper rejected",
            lambda: export_digital_twin_package(
                study, config, "fixture", str(cache), str(output)
            ))
    scaler_path.write_bytes(scaler_source_bytes)
    export_digital_twin_package(
        study, config, "fixture", str(cache), str(output)
    )
    metadata = verify_digital_twin_package(study, config, str(output))
    check("standalone deployment package verifies without Optuna DB",
          verify_standalone_dt_package(
              str(output / "DT_champion_weights.pth"),
              str(output / "DT_metadata.json"),
              str(output / "DT_scaler.pkl"),
          )["protocol_hash"] == config["protocol_hash"])
    check("metadata links study and best trial",
          metadata["study_name"] == study.study_name
          and metadata["best_trial_number"] == study.best_trial.number)
    check("metadata carries full protocol descriptor",
          metadata["protocol_descriptor"]
          == json.loads(json.dumps(config["protocol_descriptor"])))
    check("trial weights removed only after linked champion publication",
          not trial_path.exists()
          and (output / "DT_champion_weights.pth").is_file()
          and (output / "DT_scaler.pkl").is_file())
    export_digital_twin_package(
        study, config, "fixture", str(cache), str(output)
    )
    check("completed package export is idempotent after trial cleanup",
          not trial_path.exists()
          and verify_digital_twin_package(
              study, config, str(output)
          )["best_trial_number"] == study.best_trial.number)

    metadata_path = output / "DT_metadata.json"
    metadata_original = metadata_path.read_text(encoding="utf-8")
    missing_field = json.loads(metadata_original)
    del missing_field["scaler_sha256"]
    metadata_path.write_text(json.dumps(missing_field), encoding="utf-8")
    rejects_with_message(
        "standalone verifier rejects a missing required provenance field",
        lambda: verify_standalone_dt_package(
            str(output / "DT_champion_weights.pth"),
            str(metadata_path),
            str(output / "DT_scaler.pkl"),
        ),
        "metadata lacks provenance fields",
    )
    metadata_path.write_text(metadata_original, encoding="utf-8")

    champion = output / "DT_champion_weights.pth"
    original = champion.read_bytes()
    corrupt = bytearray(original)
    corrupt[-1] ^= 1
    champion.write_bytes(corrupt)
    rejects("standalone verifier rejects one-byte champion tamper",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path),
                str(output / "DT_scaler.pkl"),
            ))
    rejects("one-byte champion tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    champion.write_bytes(original)

    exported_scaler = output / "DT_scaler.pkl"
    scaler_bytes = exported_scaler.read_bytes()
    scaler_corrupt = bytearray(scaler_bytes)
    scaler_corrupt[-1] ^= 1
    exported_scaler.write_bytes(scaler_corrupt)
    rejects("standalone verifier rejects one-byte scaler tamper",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    rejects("one-byte scaler tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    exported_scaler.write_bytes(scaler_bytes)

    altered = json.loads(metadata_original)
    altered["protocol_descriptor"]["core"]["protocol_version"] = 999
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects descriptor/hash disagreement",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    rejects("metadata protocol-descriptor tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["scaler_filename"] = "different_scaler.pkl"
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("standalone verifier rejects scaler filename substitution",
            lambda: verify_standalone_dt_package(
                str(champion), str(metadata_path), str(exported_scaler),
            ))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    altered = json.loads(metadata_original)
    altered["active_dofs"] = [0]
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("deployment-semantics metadata tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    metadata_path.write_text(metadata_original, encoding="utf-8")

    original_record = study.user_attrs["ttbi_protocol_record"]
    altered_record = json.loads(json.dumps(original_record))
    altered_record["protocol_descriptor"]["core"]["protocol_version"] = 999
    study.set_user_attr("ttbi_protocol_record", altered_record)
    rejects("Optuna protocol-record tamper rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))
    study.set_user_attr("ttbi_protocol_record", original_record)

    altered = json.loads(metadata_path.read_text(encoding="utf-8"))
    altered["best_trial_number"] += 1
    metadata_path.write_text(json.dumps(altered), encoding="utf-8")
    rejects("metadata-to-best-trial mismatch rejected",
            lambda: verify_digital_twin_package(study, config, str(output)))

    mixed = optuna.create_study(study_name="unstamped-existing")
    mixed.optimize(lambda trial: 0.0, n_trials=1)
    rejects(
        "existing protocol-hashed trials cannot be stamped retroactively",
        lambda: _stamp_study_protocol(
            mixed,
            config={**config, "name": "unstamped-existing"},
            dataset_name="fixture",
            n_trials=1,
            epochs=2,
            sampler_seed=42,
            use_pruner=False,
        ),
    )
    # Release SQLite before TemporaryDirectory removes the database on Windows.
    storage.engine.dispose()

print()
if fails:
    raise SystemExit(f"ARTIFACT PROVENANCE: {fails} FAILURE(S)")
print("ARTIFACT PROVENANCE: ALL PASS")
