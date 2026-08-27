"""Adversarial acceptance for four-stage matched-block inference.

Run: ``python check_cross_rung_inference.py``
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

from core.campaign_contract import (
    EXPECTED_CHANNEL_SCHEMA_ID,
    campaign_contract_sha256,
    campaign_stage_contract,
)
import core.cross_rung_inference as inference


FAILURES = 0
SEEDS = [1009, 1013, 1019]


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def raises(name: str, function, fragment: str = "") -> None:
    try:
        function()
    except RuntimeError as exc:
        check(name, not fragment or fragment in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001 - mutation diagnostic
        check(name, False, f"unexpected {type(exc).__name__}: {exc}")
    else:
        check(name, False, "mutation was accepted")


def _payloads() -> dict[str, dict]:
    values: dict[str, dict] = {}
    pair_offsets = {
        "F40-S": 0.0,
        "F40-M": 1.25,
        "L99-S": 2.0,
        "L99-M": 2.5,
    }
    for pair_id, left, right, _count in inference.REGISTERED_BLOCK_PAIRS:
        matched = inference.registered_matched_uids(pair_id)
        # Exact paired test subset, with at least one held-out semantic replica
        # from every five-state controlled stratum in the F40 case.
        test_uids = set(matched[4::5])
        for stage in (left, right):
            inventory = inference.registered_stage_uid_inventory(stage)
            partitions = {uid: "train" for uid in inventory}
            for uid in test_uids:
                partitions[uid] = "test"
            rows = []
            for uid_position, uid in enumerate(matched):
                if uid not in test_uids:
                    continue
                for seed_position, seed in enumerate(SEEDS):
                    rows.append({
                        "stage": stage,
                        "state_uid": uid,
                        "seed": seed,
                        "scour_mse": (
                            pair_offsets[stage]
                            + uid_position / 1000.0
                            + seed_position / 10000.0
                        ),
                    })
            contract = campaign_stage_contract(stage)
            values[stage] = {
                "schema": inference.MATCHED_BLOCK_INPUT_SCHEMA,
                "stage": stage,
                "dataset": contract["dataset"],
                "channel_schema_id": EXPECTED_CHANNEL_SCHEMA_ID,
                "campaign_contract_sha256": campaign_contract_sha256(stage),
                "evaluation_role": "post_freeze_sealed_test_stability",
                "pipeline_slot": "raw_cnn_gap_baseline",
                "input_selector": [1, 5],
                "metric_name": "scour_mse",
                "registered_seeds": list(SEEDS),
                "generated_state_uids": list(inventory),
                "partition_by_uid": partitions,
                "metric_rows": rows,
            }
    return values


def main() -> None:
    print("PAPER1 MATCHED-BLOCK INFERENCE CHECKS")
    check(
        "only the two registered four-stage pairs exist",
        inference.REGISTERED_BLOCK_PAIRS == (
            ("F40-S__F40-M", "F40-S", "F40-M", 30),
            ("L99-S__L99-M", "L99-S", "L99-M", 475),
        ),
    )
    f40 = inference.registered_matched_uids("F40-S__F40-M")
    l99 = inference.registered_matched_uids("L99-S__L99-M")
    check(
        "F40 intersection is exactly five healthy plus 25 controlled scour states",
        len(f40) == 30
        and sum("family=target_healthy" in uid for uid in f40) == 5
        and sum("family=scour_only" in uid for uid in f40) == 25
        and all(
            any(f"level={level:04d}" in uid for level in (12, 24, 36, 48, 60))
            for uid in f40 if "family=scour_only" in uid
        ),
    )
    check(
        "L99 pair is the complete 475-state inventory",
        len(l99) == 475
        and tuple(l99) == inference.registered_stage_uid_inventory("L99-S")
        == inference.registered_stage_uid_inventory("L99-M"),
    )
    check(
        "two-pair multiplicity adjustment is prospective",
        inference.POINTWISE_CENTRAL_MASS == 0.95
        and inference.FAMILYWISE_CENTRAL_MASS == 0.975
        and inference.MATCHED_BLOCK_BOOTSTRAP_N == 100_000,
    )

    payloads = _payloads()
    result = inference.analyze_registered_matched_blocks(
        payloads, n_boot=2000, bootstrap_seed=42
    )
    by_pair = {row["pair_id"]: row for row in result["pairs"]}
    check(
        "constant-offset F40 fixture gives exact paired effect",
        np.isclose(
            by_pair["F40-S__F40-M"]["estimate_right_minus_left"], 1.25
        )
        and np.allclose(
            by_pair["F40-S__F40-M"]
            ["two_pair_adjusted_resampling_sensitivity_interval"],
            [1.25, 1.25],
        ),
    )
    check(
        "constant-offset L99 fixture gives exact paired effect",
        np.isclose(
            by_pair["L99-S__L99-M"]["estimate_right_minus_left"], 0.5
        )
        and np.allclose(
            by_pair["L99-S__L99-M"]
            ["pointwise_resampling_sensitivity_interval"],
            [0.5, 0.5],
        ),
    )
    check(
        "result explicitly forbids population/superiority interpretation",
        all(
            row["population_confidence_interval"] is False
            and row["automatic_superiority_claim"] is False
            for row in result["pairs"]
        )
        and len(result["result_sha256"]) == 64,
    )

    mutant = deepcopy(payloads)
    mutant["F40-S"]["generated_state_uids"].pop()
    raises(
        "missing generated F40 UID is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "generated StateUID inventory",
    )
    mutant = deepcopy(payloads)
    shared = f40[0]
    mutant["F40-M"]["partition_by_uid"][shared] = "val"
    raises(
        "matched F40 partition drift is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "matched UID partition differs",
    )
    mutant = deepcopy(payloads)
    mutant["L99-M"]["metric_rows"].pop()
    raises(
        "missing L99 StateUID x seed cell is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "exact matched test UID x seed grid",
    )
    mutant = deepcopy(payloads)
    mutant["F40-M"]["metric_rows"][0]["scour_mse"] = -0.01
    raises(
        "negative MSE is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "invalid sealed-test metric cell",
    )
    mutant = deepcopy(payloads)
    mutant["F40-M"]["pipeline_slot"] = "outcome-picked-pipeline"
    raises(
        "different endpoint pipeline is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "pipeline/input/registered seed axes",
    )
    mutant = deepcopy(payloads)
    mutant["L99-S"]["metric_rows"].append(
        deepcopy(mutant["L99-S"]["metric_rows"][0])
    )
    raises(
        "duplicate metric cell is rejected",
        lambda: inference.analyze_registered_matched_blocks(mutant, n_boot=100),
        "duplicate StateUID x seed cell",
    )

    with tempfile.TemporaryDirectory(prefix="paper1-matched-inference-") as td:
        root = Path(td).resolve()
        paths = {}
        for stage, payload in payloads.items():
            path = root / f"{stage}.json"
            path.write_bytes(inference._canonical_bytes(payload))
            paths[stage] = path
        output = root / "result.json"
        file_result = inference.analyze_registered_matched_block_files(
            paths, output, n_boot=2000, bootstrap_seed=42
        )
        check(
            "canonical files produce one atomic authenticated result",
            output.is_file()
            and output.read_bytes() == inference._canonical_bytes(file_result)
            and file_result == result,
        )
        noncanonical = root / "noncanonical.json"
        noncanonical.write_text(
            json.dumps(payloads["F40-S"], indent=2), encoding="ascii"
        )
        bad_paths = dict(paths)
        bad_paths["F40-S"] = noncanonical
        raises(
            "noncanonical endpoint payload bytes are rejected",
            lambda: inference.analyze_registered_matched_block_files(
                bad_paths, root / "bad-result.json", n_boot=100
            ),
            "not canonical JSON",
        )

    source = Path(inference.__file__).read_text(encoding="utf-8")
    check(
        "retired L60 ladder identifiers are absent",
        not any(token in source for token in (
            "REGISTERED_L60", "s0_scour", "s11_bear"
        )),
    )

    print()
    if FAILURES:
        raise SystemExit(
            f"MATCHED-BLOCK INFERENCE: {FAILURES} CHECK(S) FAILED"
        )
    print("MATCHED-BLOCK INFERENCE: ALL PASS")


if __name__ == "__main__":
    main()
