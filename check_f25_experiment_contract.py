"""CPU-only acceptance and mutation test for the isolated F25 contract."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from core.f25_experiment_contract import (
    BRIDGE_LENGTH_M,
    CHANNELS,
    CHANNEL_SCHEMA_ID,
    CRACK_ELEMENT_NUMBERS_ONE_BASED,
    CRACK_ZONE_M,
    DECK_ELEMENT_COUNT,
    DECK_MESH_M,
    DEVIATION_CLASSIFICATIONS,
    DEVIATION_ROWS,
    DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD,
    F25ContractError,
    F25_R_CHANNELS,
    F25_R_EXPERIMENT,
    F25_R_UNFROZEN,
    F25_X_EXPERIMENT,
    F25_X_FROZEN_PAIRS,
    F25_X_FROZEN_SINGLES,
    F25_X_TIERS,
    F25_X_UNFROZEN_SINGLES,
    HEALTHY_CENTRAL_SUPPORT_KV_N_PER_M,
    HPO_EXECUTIONS_PER_CONFIGURATION,
    HPO_EXECUTIONS_PER_PROPOSAL,
    HPO_EXECUTION_SEEDS,
    HPO_PROPOSALS_PER_CONFIGURATION,
    FROZEN_HP_ANCHOR_CHANNEL,
    FROZEN_HP_ANCHOR_CONTRACT_ID,
    FROZEN_HP_ANCHORS,
    LEARNING_RATE_PLATEAU_FACTOR,
    LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS,
    MAXIMUM_EPOCHS,
    MINIMUM_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE_EPOCHS,
    MONITORING_BRIDGE_TERM_SAMPLES,
    MONITORING_CROP_END_ONE_BASED,
    MONITORING_CROP_START_ONE_BASED,
    MONITORING_POST_TERM_SAMPLES,
    NOISE_E_LEVEL,
    NOISE_SNR_LABEL,
    NOISE_STANDARD_DEVIATION_DDOF,
    PAA_BLOCK_COUNT,
    PAA_BLOCK_SIZE,
    PARTITION_COUNTS,
    PARTITION_ORDER,
    PROFILE_RELATIVE_PATH,
    PROFILE_SHA256,
    PROFILE_TYPE,
    PUBLISHED_DIAGNOSTIC_COUNTS_PER_100,
    PUBLISHED_OVERALL_ACCURACY,
    PUBLISHED_PAA_CNN_TEMPLATES,
    PUBLISHED_PAA_AXIS_ACCURACY,
    REPORT_SEEDS,
    SCENARIOS,
    SOURCE_WINDOW_SAMPLES,
    SOURCE_CNN_SEARCH_SPACE,
    SPAN_LENGTHS_M,
    TAIL_SAMPLES_TRIMMED,
    TOTAL_PASSAGES,
    TRAIN_PER_CLASS,
    TRIMMED_WINDOW_SAMPLES,
    VALIDATION_PER_CLASS,
    TEST_PER_CLASS,
    apply_training_minmax,
    add_source_noise,
    build_contract,
    canonical_json_sha256,
    fit_training_minmax,
    extract_monitoring_window,
    lexicographic_sensor_pairs,
    paa_blocks_of_ten,
    partition_indices,
    partition_sha256,
    prepare_paa,
    prepare_raw,
    trim_native_window,
    validate_contract_payload,
)


# These digests deliberately live outside the authority module: changing a
# seed/allocation rule or any canonical contract datum requires an explicit
# acceptance-fixture update rather than silently blessing itself.
EXPECTED_PARTITION_SHA256 = "96e61a6ef27c997a65d1755f3b2ed28505fac775ed12a8fa60f7cf8e8c1360cb"
EXPECTED_CONTRACT_SHA256 = "614ecca52a5dac91c081d826dda1a2ddda028c229824cee60348d841ec9a2b1e"


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(message)

    def rejects(self, action, message: str) -> None:
        self.count += 1
        try:
            action()
        except F25ContractError:
            return
        raise AssertionError(message)


def _scenario_signature():
    return tuple(
        (
            scenario.label,
            scenario.crack_depth_ratio,
            scenario.crack_ei_loss_fraction,
            scenario.central_scour_kv_loss_fraction,
            scenario.entrance_bearing_kr_nm_per_rad,
        )
        for scenario in SCENARIOS
    )


def main() -> None:
    checks = Checks()

    checks.require(CHANNEL_SCHEMA_ID == "physical8_v1", "channel schema drifted")
    checks.require(
        CHANNELS == (
            "carbody_vertical_acceleration",
            "front_bogie_vertical_acceleration",
            "rear_bogie_vertical_acceleration",
            "wheelset_1_constrained_vertical_acceleration_proxy",
            "wheelset_2_constrained_vertical_acceleration_proxy",
            "carbody_pitch_rate",
            "front_bogie_pitch_rate",
            "rear_bogie_pitch_rate",
        ),
        "physical8_v1 channel order drifted",
    )
    checks.require(F25_R_CHANNELS == CHANNELS[:2], "F25-R source sensors drifted")

    expected_scenarios = (
        ("Healthy", None, 0.00, 0.00, 0.0),
        ("DC2", 0.10, 0.22, 0.00, 0.0),
        ("DC3", None, 0.00, 0.05, 0.0),
        ("DC4", None, 0.00, 0.00, 1.0e9),
        ("DC5", None, 0.00, 0.10, 1.0e9),
        ("DC6", 0.10, 0.22, 0.10, 0.0),
        ("DC7", 0.10, 0.22, 0.00, 1.0e9),
        ("DC8", 0.10, 0.22, 0.10, 1.0e9),
        ("DC9", 0.10, 0.22, 0.05, 1.0e9),
        ("DC10", 0.05, 0.14, 0.05, 1.0e9),
    )
    checks.require(_scenario_signature() == expected_scenarios, "Table 2 drifted")
    checks.require(
        SCENARIOS[2].central_support_kv_n_per_m == 326_800_000.0
        and SCENARIOS[4].central_support_kv_n_per_m == 309_600_000.0,
        "scour fractions are not applied to the 3.44e8 N/m healthy stiffness",
    )
    checks.require(
        DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD == 1.0e9
        and all(scenario.exit_bearing_kr_nm_per_rad == 0.0 for scenario in SCENARIOS),
        "bearing magnitude/location drifted",
    )
    checks.require(
        SCENARIOS[8].axis_signature[1:] == SCENARIOS[9].axis_signature[1:]
        and SCENARIOS[8].axis_signature[0] == 0.10
        and SCENARIOS[9].axis_signature[0] == 0.05,
        "DC9/DC10 no longer isolate crack depth",
    )

    checks.require(
        BRIDGE_LENGTH_M == 39.9
        and SPAN_LENGTHS_M == (19.95, 19.95)
        and DECK_MESH_M == 0.15
        and DECK_ELEMENT_COUNT == 266,
        "F25 refined bridge geometry drifted",
    )
    checks.require(
        CRACK_ZONE_M == (29.70, 30.00)
        and CRACK_ELEMENT_NUMBERS_ONE_BASED == (199, 200),
        "source crack zone is not exactly preserved",
    )
    checks.require(
        PROFILE_TYPE == 2
        and PROFILE_RELATIVE_PATH == "scour_MATLAB/Calc.ProfileData15_05.mat"
        and PROFILE_SHA256
        == "71c69d9923bdc184a2c8448e0e0e6debb1670302908e093b758f57c36147465d",
        "fixed Type-2 profile binding drifted",
    )

    checks.require(TOTAL_PASSAGES == 2000, "passage total drifted")
    checks.require(
        (TEST_PER_CLASS, VALIDATION_PER_CLASS, TRAIN_PER_CLASS) == (100, 20, 80)
        and PARTITION_ORDER == ("test", "validation", "train")
        and PARTITION_COUNTS == (100, 20, 80),
        "100/20/80 per-class allocation drifted",
    )
    split = partition_indices()
    checks.require(
        tuple(split) == PARTITION_ORDER
        and tuple(len(split[name]) for name in PARTITION_ORDER) == PARTITION_COUNTS,
        "partition order/counts drifted",
    )
    checks.require(
        np.array_equal(
            np.sort(np.concatenate([split[name] for name in PARTITION_ORDER])),
            np.arange(200),
        ),
        "partition is not a disjoint cover",
    )
    checks.require(
        partition_sha256() == EXPECTED_PARTITION_SHA256,
        "deterministic partition allocation/seed drifted",
    )

    checks.require(
        SOURCE_WINDOW_SAMPLES == 5831
        and TAIL_SAMPLES_TRIMMED == 1
        and TRIMMED_WINDOW_SAMPLES == 5830
        and PAA_BLOCK_SIZE == 10
        and PAA_BLOCK_COUNT == 583,
        "window/PAA dimensions drifted",
    )
    full_raw = np.arange(7000, dtype=np.float64).reshape(1, 1, 7000)
    reconstructed = extract_monitoring_window(full_raw)
    checks.require(
        MONITORING_CROP_START_ONE_BASED == 1001
        and MONITORING_CROP_END_ONE_BASED == 6831
        and MONITORING_BRIDGE_TERM_SAMPLES == 4000
        and MONITORING_POST_TERM_SAMPLES == 1831
        and reconstructed.shape == (1, 1, 5831)
        and reconstructed[0, 0, 0] == 1000.0
        and reconstructed[0, 0, -1] == 6830.0,
        "inclusive full-RAW monitoring-window reconciliation drifted",
    )
    checks.rejects(
        lambda: extract_monitoring_window(np.zeros((1, 1, 6830))),
        "short full RAW response did not fail closed",
    )
    source = np.arange(5831, dtype=np.float64).reshape(1, 1, 5831)
    source[..., -1] = 1.0e12  # must be removed before fitting MinMax
    calibration = fit_training_minmax(source)
    checks.require(
        calibration.minimum == (0.0,) and calibration.maximum == (5829.0,),
        "tail was not trimmed before training MinMax",
    )
    trimmed = trim_native_window(source)
    raw = prepare_raw(source, calibration)
    paa = prepare_paa(source, calibration)
    checks.require(trimmed.shape == (1, 1, 5830), "tail trim shape drifted")
    checks.require(raw.shape == (1, 1, 5830), "RAW shape drifted")
    checks.require(paa.shape == (1, 1, 583), "PAA shape drifted")
    checks.require(
        np.isclose(paa[0, 0, 0], 4.5 / 5829.0, rtol=0.0, atol=1e-15)
        and np.isclose(paa[0, 0, -1], 5824.5 / 5829.0, rtol=0.0, atol=1e-15),
        "MinMax-before-PAA transform values drifted",
    )
    validation = trimmed + 5829.0
    validation_scaled = apply_training_minmax(validation, calibration)
    checks.require(
        validation_scaled.max() > 1.0,
        "validation/test values are being clipped to training MinMax bounds",
    )
    checks.require(
        np.array_equal(paa, paa_blocks_of_ten(raw)),
        "prepare_paa is not RAW MinMax followed by exact PAA",
    )
    clean_signal = trimmed[0, 0]
    noisy_a = add_source_noise(
        clean_signal, class_index=0, passage_index=0, channel_index=0
    )
    noisy_b = add_source_noise(
        clean_signal, class_index=0, passage_index=0, channel_index=0
    )
    noisy_other_channel = add_source_noise(
        clean_signal, class_index=0, passage_index=0, channel_index=1
    )
    checks.require(
        NOISE_E_LEVEL == 0.05
        and NOISE_SNR_LABEL == 20
        and NOISE_STANDARD_DEVIATION_DDOF == 0
        and np.array_equal(noisy_a, noisy_b)
        and not np.array_equal(noisy_a, noisy_other_channel),
        "Eq.(2) shared physical-identity noise contract drifted",
    )
    checks.require(
        np.isclose(
            np.std(noisy_a - clean_signal, ddof=0)
            / np.std(clean_signal, ddof=0),
            0.05,
            rtol=0.03,
            atol=0.0,
        ),
        "Eq.(2) no longer uses 0.05 x per-signal population sigma",
    )
    checks.rejects(
        lambda: add_source_noise(
            clean_signal, class_index=10, passage_index=0, channel_index=0
        ),
        "invalid source-noise class key did not fail closed",
    )
    checks.rejects(
        lambda: trim_native_window(np.zeros((1, 1, 5830))),
        "wrong source length did not fail closed",
    )
    checks.rejects(
        lambda: paa_blocks_of_ten(np.zeros((1, 1, 5831))),
        "wrong PAA input length did not fail closed",
    )
    checks.rejects(
        lambda: fit_training_minmax(np.ones((2, 1, 5831))),
        "zero MinMax span did not fail closed",
    )
    checks.rejects(
        lambda: trim_native_window(np.full((1, 1, 5831), np.nan)),
        "non-finite source did not fail closed",
    )

    pairs = lexicographic_sensor_pairs()
    checks.require(len(pairs) == 28 and len(set(pairs)) == 28, "8C2 pair count drifted")
    checks.require(
        pairs == tuple(sorted(pairs))
        and pairs[0] == tuple(sorted(CHANNELS))[:2]
        and pairs[-1] == tuple(sorted(CHANNELS))[-2:],
        "pair inventory is not lexicographic and pre-outcome",
    )
    checks.require(
        F25_R_UNFROZEN.configuration_count == 4
        and F25_R_UNFROZEN.hpo_fit_budget == 2000
        and F25_R_UNFROZEN.report_fit_budget == 80,
        "F25-R two-sensor/two-source-arm unfrozen budget drifted",
    )
    checks.require(
        tuple(tier.tier_id for tier in F25_X_TIERS)
        == (
            "F25-X-01-frozen-hp-singles",
            "F25-X-02-unfrozen-singles",
            "F25-X-03-frozen-hp-pairs",
        ),
        "F25-X execution tier order drifted",
    )
    checks.require(
        (
            F25_X_FROZEN_SINGLES.configuration_count,
            F25_X_FROZEN_SINGLES.core_table_fit_budget,
            F25_X_UNFROZEN_SINGLES.configuration_count,
            F25_X_UNFROZEN_SINGLES.core_table_fit_budget,
            F25_X_FROZEN_PAIRS.configuration_count,
            F25_X_FROZEN_PAIRS.core_table_fit_budget,
        )
        == (24, 480, 24, 12000, 84, 1680),
        "F25-X 480/12000/1680 core budgets drifted",
    )
    checks.require(
        F25_X_UNFROZEN_SINGLES.report_fit_budget == 480,
        "post-HPO 20-seed report budget became hidden",
    )
    checks.require(
        HPO_PROPOSALS_PER_CONFIGURATION == 100
        and HPO_EXECUTIONS_PER_PROPOSAL == 5
        and HPO_EXECUTIONS_PER_CONFIGURATION == 500
        and len(HPO_EXECUTION_SEEDS) == 5,
        "source HPO semantics drifted",
    )
    checks.require(
        SOURCE_CNN_SEARCH_SPACE.convolution_filters
        == (32, 48, 64, 80, 96, 112, 128)
        and SOURCE_CNN_SEARCH_SPACE.convolution_kernel_sizes == (2, 3, 4, 5)
        and SOURCE_CNN_SEARCH_SPACE.convolution_layer_counts == (1, 2, 3, 4, 5)
        and SOURCE_CNN_SEARCH_SPACE.max_pool_size == 2
        and SOURCE_CNN_SEARCH_SPACE.dense_units == (16, 32, 48, 64)
        and SOURCE_CNN_SEARCH_SPACE.batch_sizes == (8, 16, 24, 32, 40, 48)
        and SOURCE_CNN_SEARCH_SPACE.learning_rate_range == (1.0e-5, 1.0e-2),
        "published CNN HPO bounds/steps drifted",
    )
    car_template, bogie_template = PUBLISHED_PAA_CNN_TEMPLATES
    checks.require(
        (
            car_template.convolution_filters,
            car_template.convolution_kernel_sizes,
            car_template.max_pool_after_layer,
            car_template.flatten_units,
            car_template.dense_units,
            car_template.learning_rate,
            car_template.batch_size,
        )
        == ((48, 48), (2, 3), (False, False), 27840, 48, 1.0e-3, 24)
        and (
            bogie_template.convolution_filters,
            bogie_template.convolution_kernel_sizes,
            bogie_template.max_pool_after_layer,
            bogie_template.flatten_units,
            bogie_template.dense_units,
            bogie_template.learning_rate,
            bogie_template.batch_size,
        )
        == ((128, 96, 96), (3, 3, 3), (True, True, True), 7008, 96, 5.0e-4, 32),
        "published sensor-specific PAA CNN finalists drifted",
    )
    checks.require(
        LEARNING_RATE_PLATEAU_FACTOR == 0.5
        and LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS == 30
        and MINIMUM_LEARNING_RATE == 1.0e-6
        and EARLY_STOPPING_PATIENCE_EPOCHS == 50
        and MAXIMUM_EPOCHS == 1000,
        "source training schedule drifted",
    )
    checks.require(
        FROZEN_HP_ANCHOR_CHANNEL == CHANNELS[1]
        and tuple(anchor.arm_id for anchor in FROZEN_HP_ANCHORS)
        == ("RAW-CNN", "PAA-CNN", "PAA-multirate")
        and all(anchor.anchor_channel == CHANNELS[1] for anchor in FROZEN_HP_ANCHORS)
        and F25_X_FROZEN_SINGLES.hyperparameter_provenance
        == FROZEN_HP_ANCHOR_CONTRACT_ID
        and F25_X_FROZEN_PAIRS.hyperparameter_provenance
        == FROZEN_HP_ANCHOR_CONTRACT_ID,
        "front-bogie frozen-HP provenance drifted",
    )
    checks.require(
        len(REPORT_SEEDS) == 20 and len(set(REPORT_SEEDS)) == 20,
        "20-run reporting inventory drifted",
    )

    checks.require(
        F25_R_EXPERIMENT.experiment_id == "F25-R"
        and F25_X_EXPERIMENT.experiment_id == "F25-X"
        and F25_R_EXPERIMENT.manifest_root != F25_X_EXPERIMENT.manifest_root
        and F25_R_EXPERIMENT.cache_root != F25_X_EXPERIMENT.cache_root
        and F25_R_EXPERIMENT.results_root != F25_X_EXPERIMENT.results_root,
        "experiment identity/artifact roots are not isolated",
    )
    checks.require(
        F25_R_EXPERIMENT.shared_data_contract_id
        == F25_X_EXPERIMENT.shared_data_contract_id
        and F25_R_EXPERIMENT.partition_seed == F25_X_EXPERIMENT.partition_seed
        and F25_R_EXPERIMENT.eov_master_seed == F25_X_EXPERIMENT.eov_master_seed
        and F25_R_EXPERIMENT.source_lineage_policy
        == F25_X_EXPERIMENT.source_lineage_policy,
        "F25-R/F25-X do not share data partitions/seeds/source lineage",
    )

    overall = {
        target.sensor_id: (target.raw_cnn, target.paa_cnn)
        for target in PUBLISHED_OVERALL_ACCURACY
    }
    checks.require(
        overall == {CHANNELS[0]: (0.651, 0.867), CHANNELS[1]: (0.822, 0.821)},
        "published RAW/full overall targets drifted",
    )
    axes = {
        target.sensor_id: (
            target.bearing_present_absent,
            target.scour_level,
            target.crack_level,
            target.overall_ten_class,
        )
        for target in PUBLISHED_PAA_AXIS_ACCURACY
    }
    checks.require(
        axes
        == {
            CHANNELS[0]: (0.999, 0.946, 0.884, 0.867),
            CHANNELS[1]: (0.967, 0.929, 0.880, 0.821),
        },
        "published per-axis targets drifted",
    )
    checks.require(
        PUBLISHED_DIAGNOSTIC_COUNTS_PER_100
        == {
            "carbody_DC9_as_DC10": 56,
            "carbody_DC10_as_DC9": 21,
            "front_bogie_DC10_correct": 12,
            "front_bogie_DC10_as_DC9": 69,
            "front_bogie_DC4_as_Healthy": 28,
            "front_bogie_Healthy_as_DC4": 5,
        },
        "published diagnostic failure counts drifted",
    )

    classifications = Counter(row.classification for row in DEVIATION_ROWS)
    checks.require(
        set(classifications) == set(DEVIATION_CLASSIFICATIONS)
        == {
            "exactly reproduced",
            "inferred because underreported",
            "deliberately changed",
        },
        "deviation table vocabulary drifted",
    )
    partition_row = next(
        row for row in DEVIATION_ROWS if row.item_id == "partition_allocation_and_seed"
    )
    checks.require(
        partition_row.classification == "inferred because underreported",
        "partition allocation/seed is falsely claimed exact",
    )

    payload = build_contract()
    checks.require(
        payload["contract_sha256"] == EXPECTED_CONTRACT_SHA256,
        "canonical F25 contract digest drifted",
    )
    validate_contract_payload(payload)
    checks.count += 1

    # Recomputing the digest cannot bless a scientifically drifted payload: the
    # exact authority comparison must still reject it.
    mutant = copy.deepcopy(payload)
    mutant["scenarios"][1]["crack_ei_loss_fraction"] = 0.10
    mutant_body = dict(mutant)
    mutant_body.pop("contract_sha256")
    mutant["contract_sha256"] = canonical_json_sha256(mutant_body)
    checks.rejects(
        lambda: validate_contract_payload(mutant),
        "self-consistent but scientifically drifted scenario was accepted",
    )
    digest_mutant = copy.deepcopy(payload)
    digest_mutant["profile"]["sha256"] = "0" * 64
    checks.rejects(
        lambda: validate_contract_payload(digest_mutant),
        "payload with stale digest was accepted",
    )

    print(
        "F25 EXPERIMENT CONTRACT: PASS "
        f"({checks.count} checks; 10 classes/2,000 passages; "
        "F25-X budgets 480 + 12,000 + 1,680)"
    )


if __name__ == "__main__":
    main()
