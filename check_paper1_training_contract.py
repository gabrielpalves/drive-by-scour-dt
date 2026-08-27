"""No-GPU acceptance test for the Paper-1 training job grid."""

from __future__ import annotations

from collections import Counter

from core.paper1_training_contract import (
    ANCHOR_CHANNEL_INDEX,
    ANCHOR_CHANNEL_NAME,
    CHANNEL_NAMES,
    DEVELOPMENT_N_REPEATS,
    DEVELOPMENT_N_SPLITS,
    DEVELOPMENT_PARTITION_SEED,
    ELIGIBLE_SENSOR_INDICES,
    ELIGIBLE_SENSOR_NAMES,
    EXCLUDED_PROXY_INDICES,
    FACTORIAL_CELLS,
    HPO_RESTART_SEEDS,
    OUTER_SPLIT_SEED,
    POST_FREEZE_STABILITY_SEEDS,
    STAGE_ORDER,
    channel_screen_inputs,
    complete_job_grid,
    development_adjudication_jobs,
    frozen_transfer_jobs,
    hpo_jobs,
    post_freeze_stability_jobs,
    resolve_retained_pipelines,
)


def main() -> None:
    assert len(FACTORIAL_CELLS) == 16
    assert Counter(cell.representation for cell in FACTORIAL_CELLS) == {
        "RAW": 8,
        "PAA": 8,
    }
    assert ANCHOR_CHANNEL_INDEX == 1
    assert OUTER_SPLIT_SEED == 42
    assert ANCHOR_CHANNEL_NAME == "front_bogie_vertical_acceleration"
    assert CHANNEL_NAMES[3:5] == (
        "wheelset_1_constrained_vertical_acceleration_proxy",
        "wheelset_2_constrained_vertical_acceleration_proxy",
    )
    assert EXCLUDED_PROXY_INDICES == (3, 4)
    assert ELIGIBLE_SENSOR_INDICES == (0, 1, 2, 5, 6, 7)
    assert ELIGIBLE_SENSOR_NAMES == tuple(
        CHANNEL_NAMES[index] for index in ELIGIBLE_SENSOR_INDICES
    )
    inputs = channel_screen_inputs()
    assert len(inputs) == 21
    assert sum(len(value) == 1 for value in inputs) == 6
    assert sum(len(value) == 2 for value in inputs) == 15
    assert all(
        set(value) <= set(ELIGIBLE_SENSOR_INDICES) for value in inputs
    )

    hpo = hpo_jobs()
    counts = Counter((job["stage"], job["phase"]) for job in hpo)
    assert counts == {
        ("F40-S", "f40s_factorial_hpo"): 80,
        ("F40-S", "f40s_selected_pair_hpo"): 20,
        ("F40-M", "block_selected_pair_hpo"): 20,
        ("L99-S", "block_selected_pair_hpo"): 20,
        ("L99-M", "block_selected_pair_hpo"): 20,
    }
    assert {job["trials"] for job in hpo} == {100}
    assert {job["hpo_restart_seed"] for job in hpo} == set(
        HPO_RESTART_SEEDS
    )
    assert len(development_adjudication_jobs()) == 480
    adjudication = development_adjudication_jobs()
    assert {job["development_partition_seed"] for job in adjudication} == {
        DEVELOPMENT_PARTITION_SEED
    }
    assert {job["fold_index"] for job in adjudication} == set(
        range(DEVELOPMENT_N_SPLITS)
    )
    assert DEVELOPMENT_N_REPEATS == 1
    assert len(post_freeze_stability_jobs()) == 4 * 4 * 30
    assert len(POST_FREEZE_STABILITY_SEEDS) == 30
    assert len(frozen_transfer_jobs()) == 3 * 4 * 5

    # Baseline winners must deduplicate, while non-baseline winners retain the
    # complete four-arm comparison in stable role order.
    assert len(resolve_retained_pipelines(
        best_raw="RAW_POS0_LSTM0_MR0",
        best_paa="PAA_POS0_LSTM0_MR0",
    )) == 2
    assert len(resolve_retained_pipelines(
        best_raw="RAW_POS1_LSTM1_MR1",
        best_paa="PAA_POS1_LSTM1_MR1",
    )) == 4

    grid = complete_job_grid()
    assert grid["stage_order"] == list(STAGE_ORDER)
    assert grid["channel_names"] == list(CHANNEL_NAMES)
    assert grid["eligible_sensor_indices"] == list(ELIGIBLE_SENSOR_INDICES)
    assert grid["excluded_proxy_indices"] == list(EXCLUDED_PROXY_INDICES)
    assert len(grid["phases"]["channel_screen"]) == 420
    assert len(grid["complete_grid_sha256"]) == 64
    assert grid["transport_rescue_policy"] == (
        "withdrawn; every block has independent HPO"
    )
    print(
        "PASS paper1 training contract: 16 factorial cells, 160 HPO studies "
        "(16,000 trials), 6+15 eligible channel screen, 30-seed sealed-test stability"
    )


if __name__ == "__main__":
    main()
