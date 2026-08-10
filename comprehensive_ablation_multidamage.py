"""Manifest-gated migration entrypoint for the four-block Paper-1 campaign.

The former ten-rung driver is retired.  This module now authenticates exactly
one of the deterministic Lab-A/Lab-B job manifests generated from
``core.paper1_training_contract``.  Validation, listing, and materialization do
not import PyTorch, Optuna, data, studies, or model weights.  Execution is
delegated to the phase adapters in :mod:`training.paper1_executor`. Factorial
HPO, grouped-development adjudication, the authenticated channel screen,
selected-pair HPO, block-local freezing, post-freeze stability, and secondary
transfer are executable in dependency order. Every downstream phase remains
fail-closed until its registered upstream artefacts exist. No legacy rung,
frozen-transport, rescue, or outcome-dependent allocation path remains
reachable.

Environment
-----------
``TTBI_TRAINING_JOB_MANIFEST`` must be an absolute path to canonical JSON for
one complete, source-derived Lab-A or Lab-B manifest.

Examples
--------
``py -3.13 comprehensive_ablation_multidamage.py --validate-manifest``

``py -3.13 comprehensive_ablation_multidamage.py --list-jobs``

``py -3.13 comprehensive_ablation_multidamage.py --materialize-job <job_id>``

``py -3.13 comprehensive_ablation_multidamage.py --execute-job <job_id>``

``py -3.13 comprehensive_ablation_multidamage.py --publish-adjudication <file>``

``py -3.13 comprehensive_ablation_multidamage.py --publish-channel-selection <file>``
The latter writes the full channel tensor to ``<file>`` and the compact
downstream selection to ``TTBI_PAPER1_SELECTION_ARTIFACT``.

``py -3.13 comprehensive_ablation_multidamage.py --publish-block-freeze F40-S <file>``
authenticates all five selected-pair HPO restarts for every unique resolved
pipeline in that block before depositing the report-only freeze artefact.
"""

from __future__ import annotations

import os as _bootstrap_os
import sys as _bootstrap_sys
for _unsafe_python_path_variable in ("PYTHONPATH", "PYTHONHOME"):
    if _unsafe_python_path_variable in _bootstrap_os.environ:
        raise RuntimeError(
            f"{_unsafe_python_path_variable} must be absent before scientific "
            "imports"
        )

_bootstrap_source_root = _bootstrap_os.path.abspath(
    _bootstrap_os.path.dirname(__file__)
)
_bootstrap_first_path = _bootstrap_sys.path[0] or _bootstrap_os.getcwd()
if (
    _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    or _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_first_path
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_source_root
    ))
):
    raise RuntimeError(
        "reviewed repository root must be the canonical first import path"
    )
_bootstrap_guard_dir = _bootstrap_os.path.join(
    _bootstrap_source_root, "campaign_import_guard"
)
_bootstrap_guard_init = _bootstrap_os.path.join(
    _bootstrap_guard_dir, "__init__.py"
)
if (
    not _bootstrap_os.path.isfile(_bootstrap_guard_init)
    or _bootstrap_os.path.islink(_bootstrap_guard_init)
    or _bootstrap_os.path.normcase(_bootstrap_os.path.abspath(
        _bootstrap_guard_dir
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_dir
    ))
    or any(
        entry.casefold().startswith("__init__.")
        and entry != "__init__.py"
        for entry in _bootstrap_os.listdir(_bootstrap_guard_dir)
    )
):
    raise RuntimeError(
        "reviewed campaign import guard package is absent or ambiguous"
    )
_bootstrap_loaded_guard = _bootstrap_sys.modules.get("campaign_import_guard")
if _bootstrap_loaded_guard is not None and (
    _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        getattr(_bootstrap_loaded_guard, "__file__", "")
    ))
    != _bootstrap_os.path.normcase(_bootstrap_os.path.realpath(
        _bootstrap_guard_init
    ))
    or getattr(_bootstrap_loaded_guard, "_BOUNDARY_ENFORCED", False) is not True
):
    raise RuntimeError(
        "preloaded campaign import guard is not the reviewed enforced module"
    )
from campaign_import_guard import (  # noqa: E402
    enforce_import_boundary as _enforce_import_boundary,
)
_enforce_import_boundary()

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from core.campaign_contract import EXPECTED_PROTOCOL_SCHEMA_TAG
from core.paper1_dispatch import (
    TRAINING_HOSTS,
    TRAINING_MANIFEST_SCHEMA,
    training_manifests,
)
from core.paper1_training_contract import canonical_json_bytes


SCHEMA_TAG = EXPECTED_PROTOCOL_SCHEMA_TAG
TRAINING_JOB_MANIFEST_ENV = "TTBI_TRAINING_JOB_MANIFEST"
MIGRATION_STATUS = "manifest-qualified-four-stage-execution"


class TrainingManifestError(RuntimeError):
    """The supplied machine manifest is absent, mutable, or foreign."""


def _manifest_path(explicit_path: str | os.PathLike[str] | None) -> Path:
    raw = (
        os.fspath(explicit_path)
        if explicit_path is not None
        else os.environ.get(TRAINING_JOB_MANIFEST_ENV, "")
    )
    if not raw:
        raise TrainingManifestError(
            f"{TRAINING_JOB_MANIFEST_ENV} is required and must name one "
            "absolute Lab-A/Lab-B manifest"
        )
    path = Path(raw)
    if not path.is_absolute():
        raise TrainingManifestError(
            f"{TRAINING_JOB_MANIFEST_ENV} must be an absolute path"
        )
    if path.is_symlink() or not path.is_file():
        raise TrainingManifestError(
            "training job manifest must be one regular, non-symlink file"
        )
    try:
        if path.absolute() != path.resolve(strict=True):
            raise TrainingManifestError(
                "training job manifest path must already be canonical"
            )
    except OSError as exc:
        raise TrainingManifestError(
            f"cannot resolve training job manifest: {exc}"
        ) from exc
    return path


def load_training_job_manifest(
    explicit_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Authenticate and return one exact deterministic host manifest."""

    path = _manifest_path(explicit_path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingManifestError(
            f"training job manifest is unreadable/non-JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TrainingManifestError("training job manifest must be one object")
    if raw != canonical_json_bytes(value):
        raise TrainingManifestError(
            "training job manifest bytes are not canonical JSON"
        )
    role = value.get("machine_role")
    if value.get("schema") != TRAINING_MANIFEST_SCHEMA or role not in TRAINING_HOSTS:
        raise TrainingManifestError(
            "training job manifest schema/machine role is not registered"
        )
    expected = training_manifests()[str(role)]
    if value != expected:
        raise TrainingManifestError(
            "training job manifest differs from the complete source-derived "
            f"{role} allocation"
        )
    return value


def select_registered_job(
    manifest: dict[str, Any], job_id: str
) -> dict[str, Any]:
    """Select one exact job from the authenticated machine allocation."""

    if not isinstance(job_id, str) or not job_id:
        raise TrainingManifestError("job_id must be a nonempty string")
    matches = [job for job in manifest["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise TrainingManifestError(
            f"job_id {job_id!r} is not assigned exactly once to "
            f"{manifest['machine_role']}"
        )
    return matches[0]


def execute_registered_job(
    job: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Execute one exact assigned job through its four-stage phase adapter."""

    from training.paper1_executor import execute_manifest_job

    return execute_manifest_job(job, manifest)


def publish_adjudication(path: str) -> dict[str, Any]:
    from training.paper1_executor import publish_development_adjudication_artifact

    return publish_development_adjudication_artifact(path)


def publish_channel_selection(path: str) -> dict[str, dict[str, Any]]:
    from training.paper1_executor import publish_channel_selection_artifacts

    return publish_channel_selection_artifacts(channel_output_path=path)


def publish_block_freeze(stage: str, path: str) -> dict[str, Any]:
    from training.paper1_executor import publish_block_freeze_artifact

    return publish_block_freeze_artifact(stage, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--validate-manifest",
        action="store_true",
        help="authenticate this host's complete job manifest and exit",
    )
    actions.add_argument(
        "--list-jobs",
        action="store_true",
        help="print assigned job IDs in registered order",
    )
    actions.add_argument(
        "--materialize-job",
        metavar="JOB_ID",
        help="print one assigned job as canonical JSON",
    )
    actions.add_argument(
        "--execute-job",
        metavar="JOB_ID",
        help="execute/resume one exact assigned job through its phase adapter",
    )
    actions.add_argument(
        "--publish-adjudication",
        metavar="FILE",
        help="authenticate all 480 OOF results and publish their aggregate",
    )
    actions.add_argument(
        "--publish-channel-selection",
        metavar="FILE",
        help=(
            "authenticate all channel jobs and publish the full tensor plus "
            "TTBI_PAPER1_SELECTION_ARTIFACT"
        ),
    )
    actions.add_argument(
        "--publish-block-freeze",
        nargs=2,
        metavar=("STAGE", "FILE"),
        help=(
            "authenticate all five selected-pair HPO restarts per unique "
            "pipeline and publish one stage-local freeze artefact"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_training_job_manifest()
    if args.validate_manifest:
        print(
            f"PASS {manifest['machine_role']} training manifest: "
            f"{manifest['assigned_job_count']} disjoint jobs, complete-grid "
            f"SHA-256 {manifest['complete_grid_sha256']}"
        )
        return 0
    if args.list_jobs:
        for job_id in manifest["assigned_job_ids"]:
            print(job_id)
        return 0
    if args.publish_adjudication:
        artifact = publish_adjudication(args.publish_adjudication)
        print(
            "PASS development adjudication artefact: "
            f"{artifact['artifact_sha256']}"
        )
        return 0
    if args.publish_channel_selection:
        artifacts = publish_channel_selection(args.publish_channel_selection)
        print(
            "PASS channel/downstream selection artefacts: "
            f"{artifacts['channel_selection']['artifact_sha256']} / "
            f"{artifacts['selection']['artifact_sha256']}"
        )
        return 0
    if args.publish_block_freeze:
        stage, path = args.publish_block_freeze
        artifact = publish_block_freeze(stage, path)
        print(
            f"PASS {stage} block freeze artefact: "
            f"{artifact['artifact_sha256']}"
        )
        return 0
    requested = args.materialize_job or args.execute_job
    job = select_registered_job(manifest, requested)
    if args.materialize_job:
        sys.stdout.buffer.write(canonical_json_bytes(job) + b"\n")
        return 0
    completion = execute_registered_job(job, manifest)
    print(
        f"PASS {job['job_id']} ({job['phase']}): "
        f"{completion.get('schema', 'completed')}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingManifestError as exc:
        raise SystemExit(f"TRAINING MANIFEST REJECTED: {exc}") from exc
