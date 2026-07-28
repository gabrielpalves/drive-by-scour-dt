"""Compute the pre-registered seven-edge L60 CRN analysis.

Example
-------
py -3.13 analyze_cross_rung_contrasts.py \
  --champion-manifest D:/ttbi-control/champion.json \
  --block-reference-sha256 <64-lowercase-hex> \
  --hyperparameter-manifest D:/ttbi-control/l60_hyperparameters.json \
  --execution-receipt D:/ttbi-control/execution_l60.json \
  --summary s0_scour=D:/runs/s0_summary \
  --summary s11_bear=D:/runs/s11_summary \
  --summary s12_crack=D:/runs/s12_summary \
  --summary s13_bearcrack=D:/runs/s13_summary \
  --summary s14_prof=D:/runs/s14_summary \
  --summary s15_track=D:/runs/s15_summary \
  --summary s16_all=D:/runs/s16_summary \
  --output-dir D:/runs/l60_cross_rung
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.cross_rung_inference import (
    REGISTERED_L60_STAGES,
    analyze_registered_l60_contrasts,
)


def _summary_mapping(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"--summary must be STAGE=PATH, got {value!r}"
            )
        stage, path = value.split("=", 1)
        if stage not in REGISTERED_L60_STAGES:
            raise argparse.ArgumentTypeError(
                f"unknown L60 stage {stage!r}; expected {REGISTERED_L60_STAGES}"
            )
        if not path:
            raise argparse.ArgumentTypeError(
                f"--summary {stage}= needs a directory path"
            )
        if stage in result:
            raise argparse.ArgumentTypeError(
                f"duplicate --summary entry for {stage}"
            )
        result[stage] = path
    if set(result) != set(REGISTERED_L60_STAGES):
        missing = sorted(set(REGISTERED_L60_STAGES) - set(result))
        raise argparse.ArgumentTypeError(
            f"--summary is missing registered stage(s) {missing}"
        )
    return result


def _absolute_durable_path(value: str) -> str:
    if not Path(value).is_absolute():
        raise argparse.ArgumentTypeError(
            f"expected an absolute durable path, got {value!r}"
        )
    return value


def _lower_sha256(value: str) -> str:
    if (
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise argparse.ArgumentTypeError(
            "expected one 64-character lowercase SHA-256 digest"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compute the pre-registered semantic-UID-paired L60 "
            "cross-rung contrasts. The seven primary edges are fixed in code; "
            "custom edge selection is intentionally unsupported."
        )
    )
    parser.add_argument(
        "--champion-manifest",
        required=True,
        type=_absolute_durable_path,
        help=(
            "Absolute durable path to the authenticated s0 block-reference "
            "JSON used throughout the L60 block."
        ),
    )
    parser.add_argument(
        "--block-reference-sha256",
        required=True,
        type=_lower_sha256,
        help=(
            "Independently retained canonical SHA-256 printed by the completed "
            "s0 anchor; internal stage pins cannot replace this trust root."
        ),
    )
    parser.add_argument(
        "--hyperparameter-manifest",
        required=True,
        type=_absolute_durable_path,
        help=(
            "Absolute durable path to the authenticated full-factorial L60/s0 "
            "hyperparameter manifest; its recomputed canonical SHA must match "
            "champion and every row."
        ),
    )
    parser.add_argument(
        "--execution-receipt",
        required=True,
        type=_absolute_durable_path,
        help=(
            "Absolute durable path to the canonical L60 execution-block "
            "receipt. Its regular-file bytes, runtime binding, protocol core, "
            "and run tag must match the champion, HPO manifest, and every rung "
            "artifact."
        ),
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="One exact summary directory for each registered L60 stage.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the contrast CSVs and authenticated analysis manifest.",
    )
    args = parser.parse_args()
    try:
        summaries = _summary_mapping(args.summary)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    result = analyze_registered_l60_contrasts(
        summaries,
        args.champion_manifest,
        args.hyperparameter_manifest,
        args.execution_receipt,
        args.output_dir,
        expected_block_reference_sha256=
            args.block_reference_sha256,
    )
    print(
        "REGISTERED L60 CROSS-RUNG ANALYSIS: COMPLETE\n"
        f"  summary:  {result['summary_path']}\n"
        f"  cells:    {result['cells_path']}\n"
        f"  manifest: {result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
