"""Compute the registered two-pair Paper-1 matched-block analysis.

Example::

    py -3.13 analyze_cross_rung_contrasts.py \
      --payload F40-S=D:/runs/F40-S/matched_metrics.json \
      --payload F40-M=D:/runs/F40-M/matched_metrics.json \
      --payload L99-S=D:/runs/L99-S/matched_metrics.json \
      --payload L99-M=D:/runs/L99-M/matched_metrics.json \
      --output D:/runs/paper1_matched_block_inference.json

Every input must be an absolute, canonical, non-symlink file containing exact
canonical JSON.  The output path must not exist.  The registered 100,000-draw
StateUID bootstrap and seed cannot be changed from this command line.
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
import os
from pathlib import Path
import sys

from core.cross_rung_inference import (  # noqa: E402
    MATCHED_BLOCK_BOOTSTRAP_N,
    MATCHED_BLOCK_BOOTSTRAP_SEED,
    REGISTERED_BLOCK_PAIRS,
    analyze_registered_matched_block_files,
)


REGISTERED_STAGES = tuple(
    dict.fromkeys(
        stage
        for _pair, left, right, _count in REGISTERED_BLOCK_PAIRS
        for stage in (left, right)
    )
)


def _stage_path(raw: str) -> tuple[str, Path]:
    stage, separator, path_text = raw.partition("=")
    if not separator or stage not in REGISTERED_STAGES or not path_text:
        raise argparse.ArgumentTypeError(
            f"--payload must be STAGE=ABS_PATH for one of {REGISTERED_STAGES}"
        )
    path = Path(path_text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("payload path must be absolute")
    return stage, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        action="append",
        required=True,
        type=_stage_path,
        metavar="STAGE=ABS_PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payloads = dict(args.payload)
    if len(payloads) != len(args.payload) or tuple(payloads) != REGISTERED_STAGES:
        parser.error(
            "--payload must provide F40-S, F40-M, L99-S, L99-M exactly once "
            "in registered order"
        )
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    result = analyze_registered_matched_block_files(
        payloads,
        args.output,
        n_boot=MATCHED_BLOCK_BOOTSTRAP_N,
        bootstrap_seed=MATCHED_BLOCK_BOOTSTRAP_SEED,
    )
    print(
        f"PASS matched-block inference -> {args.output} "
        f"(result_sha256={result['result_sha256']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"MATCHED-BLOCK INFERENCE REJECTED: {exc}") from exc
