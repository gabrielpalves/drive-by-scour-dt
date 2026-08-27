"""Authoritative, isolated contract for the F25-R/F25-X experiment.

This module deliberately has no dependency on PyTorch, Optuna, MATLAB, or the
main Paper-1 campaign driver.  It freezes the Fernandes (2025) reconstruction
and extension as data: geometry, damage classes, partitions, preprocessing,
sensor tiers, compute budgets, published comparison targets, and the deviation
table.  A future generator/trainer must bind its manifest to
``build_contract()["contract_sha256"]`` rather than reinterpreting prose.

``F25-R`` means *publication-faithful reconstruction*, not exact replication.
``F25-X`` is an extension on the same generated passages, partitions, and
predeclared seeds.  The two experiment IDs have separate artifact roots but use
one qualified source lineage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_SCHEMA = "f25-experiment-contract-v1"
RECONSTRUCTION_LABEL = "publication-faithful reconstruction"
SOURCE_LINEAGE_POLICY = "same clean commit A and identical source hashes"

DEVIATION_CLASSIFICATIONS = (
    "exactly reproduced",
    "inferred because underreported",
    "deliberately changed",
)

CHANNEL_SCHEMA_ID = "physical8_v1"
CHANNELS = (
    "carbody_vertical_acceleration",
    "front_bogie_vertical_acceleration",
    "rear_bogie_vertical_acceleration",
    "wheelset_1_constrained_vertical_acceleration_proxy",
    "wheelset_2_constrained_vertical_acceleration_proxy",
    "carbody_pitch_rate",
    "front_bogie_pitch_rate",
    "rear_bogie_pitch_rate",
)
# Preserve all eight generated response rows in ``physical8_v1`` while
# separating storage/V&V diagnostics from scientifically comparable sensors.
# The constrained-wheelset rows are kinematic proxies, not independent wheel
# DOFs or instrument models, so F25-X must not rank or combine them as sensors.
EXCLUDED_PROXY_INDICES = (3, 4)
ELIGIBLE_SENSOR_INDICES = (0, 1, 2, 5, 6, 7)
ELIGIBLE_SENSOR_CHANNELS = tuple(CHANNELS[index] for index in ELIGIBLE_SENSOR_INDICES)
F25_R_CHANNELS = CHANNELS[:2]
F25_X_CHANNELS = ELIGIBLE_SENSOR_CHANNELS

PROFILE_TYPE = 2
PROFILE_RELATIVE_PATH = "scour_MATLAB/Calc.ProfileData15_05.mat"
PROFILE_SHA256 = (
    "71c69d9923bdc184a2c8448e0e0e6de"
    "bb1670302908e093b758f57c36147465d"
)

BRIDGE_LENGTH_M = 39.9
SPAN_LENGTHS_M = (19.95, 19.95)
DECK_MESH_M = 0.15
DECK_ELEMENT_COUNT = 266
CENTRAL_SUPPORT_X_M = 19.95
CRACK_ZONE_M = (29.70, 30.00)
CRACK_ELEMENT_NUMBERS_ONE_BASED = (199, 200)
DECK_MASS_KG_PER_M = 9600.0
HEALTHY_CENTRAL_SUPPORT_KV_N_PER_M = 3.44e8
HEALTHY_END_BEARING_KR_NM_PER_RAD = 0.0
DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD = 1.0e9

PASSAGES_PER_CLASS = 200
CLASS_COUNT = 10
TOTAL_PASSAGES = PASSAGES_PER_CLASS * CLASS_COUNT
TEST_PER_CLASS = 100
VALIDATION_PER_CLASS = 20
TRAIN_PER_CLASS = 80
PARTITION_ORDER = ("test", "validation", "train")
PARTITION_COUNTS = (TEST_PER_CLASS, VALIDATION_PER_CLASS, TRAIN_PER_CLASS)
PARTITION_ALGORITHM = "numpy-PCG64-permutation-first-test-then-validation-then-train"
PARTITION_SEED = 2_025_080_901
EOV_MASTER_SEED = 2_025_080_902
SHARED_DATA_CONTRACT_ID = "f25-shared-data-and-partition-v1"

SOURCE_WINDOW_SAMPLES = 5831
TRIMMED_WINDOW_SAMPLES = 5830
TAIL_SAMPLES_TRIMMED = 1
# F25 reconstructs the paper's 58.30 m / 5,830-point signal from this
# repository's inclusive full-RAW grid.  The generic corrected 39.9 m crop is
# 3990 + 1831 = 5821 points; the reconstruction uses the historical
# round-before-scale bridge term (4000) and an inclusive grid (5831), then trims
# its final point.  This reconstruction-only crop extends 0.10 m farther.
MONITORING_SAMPLES_PER_M = 100
MONITORING_CROP_START_ONE_BASED = 1001
MONITORING_BRIDGE_TERM_SAMPLES = 4000
MONITORING_POST_TERM_SAMPLES = 1831
MONITORING_CROP_END_ONE_BASED = 6831
CORRECTED_GENERIC_BRIDGE_TERM_SAMPLES = 3990
CORRECTED_GENERIC_WINDOW_SAMPLES = 5821
WINDOW_RECONCILIATION_EXTRA_SAMPLES = 10
WINDOW_RECONCILIATION_EXTRA_M = 0.10
PAA_BLOCK_SIZE = 10
PAA_BLOCK_COUNT = 583
MINMAX_SCOPE = "training-partition per physical channel"
MINMAX_CLIP = False
RAW_PREPROCESSING_STEPS = (
    "extract source-era monitoring crop from full RAW: samples 1001..6831",
    "trim one tail sample: 5831 -> 5830",
    "add Eq.(2) Gaussian noise at 0.05 x per-signal population standard deviation",
    "apply training-partition per-channel MinMax",
)
PAA_PREPROCESSING_STEPS = RAW_PREPROCESSING_STEPS + (
    "non-overlapping arithmetic means: 583 blocks x 10 samples",
)

NOISE_E_LEVEL = 0.05
NOISE_SNR_LABEL = 20
NOISE_STANDARD_DEVIATION_DDOF = 0
NOISE_MASTER_SEED = 2_025_080_903
NOISE_CONTRACT_ID = "f25-shared-preprocessing-noise-v1"
NOISE_ORDER_POLICY = (
    "crop and tail-trim clean physical signals; add one shared seeded noise "
    "realization per class/passage/channel; then fit/apply MinMax; then PAA"
)

HPO_PROPOSALS_PER_CONFIGURATION = 100
HPO_EXECUTIONS_PER_PROPOSAL = 5
HPO_EXECUTIONS_PER_CONFIGURATION = (
    HPO_PROPOSALS_PER_CONFIGURATION * HPO_EXECUTIONS_PER_PROPOSAL
)
# These are repeated executions of each proposal, not five optimizer restarts.
HPO_EXECUTION_SEEDS = (104729, 130363, 155921, 196613, 228017)
REPORT_SEEDS = (
    1009,
    1013,
    1019,
    1021,
    1031,
    1033,
    1039,
    1049,
    1051,
    1061,
    1063,
    1069,
    1087,
    1091,
    1093,
    1097,
    1103,
    1109,
    1117,
    1123,
)


class F25ContractError(ValueError):
    """Raised when an F25 contract, binding, or transform fails closed."""


@dataclass(frozen=True)
class GeometryContract:
    bridge_length_m: float = BRIDGE_LENGTH_M
    span_lengths_m: tuple[float, float] = SPAN_LENGTHS_M
    deck_mesh_m: float = DECK_MESH_M
    deck_element_count: int = DECK_ELEMENT_COUNT
    central_support_x_m: float = CENTRAL_SUPPORT_X_M
    crack_zone_m: tuple[float, float] = CRACK_ZONE_M
    crack_element_numbers_one_based: tuple[int, int] = (
        CRACK_ELEMENT_NUMBERS_ONE_BASED
    )
    deck_mass_kg_per_m: float = DECK_MASS_KG_PER_M


@dataclass(frozen=True)
class DamageScenario:
    class_index: int
    label: str
    crack_depth_ratio: float | None
    crack_ei_loss_fraction: float
    central_scour_kv_loss_fraction: float
    entrance_bearing_kr_nm_per_rad: float
    exit_bearing_kr_nm_per_rad: float = HEALTHY_END_BEARING_KR_NM_PER_RAD

    @property
    def central_support_kv_n_per_m(self) -> float:
        return HEALTHY_CENTRAL_SUPPORT_KV_N_PER_M * (
            1.0 - self.central_scour_kv_loss_fraction
        )

    @property
    def axis_signature(self) -> tuple[float, float, int]:
        """Return crack-depth, scour-loss, bearing-presence axis levels."""

        return (
            0.0 if self.crack_depth_ratio is None else self.crack_depth_ratio,
            self.central_scour_kv_loss_fraction,
            int(self.entrance_bearing_kr_nm_per_rad > 0.0),
        )


SCENARIOS = (
    DamageScenario(0, "Healthy", None, 0.00, 0.00, 0.0),
    DamageScenario(1, "DC2", 0.10, 0.22, 0.00, 0.0),
    DamageScenario(2, "DC3", None, 0.00, 0.05, 0.0),
    DamageScenario(
        3, "DC4", None, 0.00, 0.00, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
    DamageScenario(
        4, "DC5", None, 0.00, 0.10, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
    DamageScenario(5, "DC6", 0.10, 0.22, 0.10, 0.0),
    DamageScenario(
        6, "DC7", 0.10, 0.22, 0.00, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
    DamageScenario(
        7, "DC8", 0.10, 0.22, 0.10, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
    DamageScenario(
        8, "DC9", 0.10, 0.22, 0.05, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
    DamageScenario(
        9, "DC10", 0.05, 0.14, 0.05, DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD
    ),
)


@dataclass(frozen=True)
class EOVContract:
    speed_km_h: tuple[float, float] = (70.0, 90.0)
    temperature_c: tuple[float, float] = (3.0, 33.0)
    measurement_noise_sigma_fraction: float = 0.05
    primary_suspension_n_per_m: tuple[float, float] = (2.640e6, 2.920e6)
    secondary_suspension_n_per_m: tuple[float, float] = (0.942e6, 1.042e6)
    carbody_mass_kg: tuple[float, float] = (33_000.0, 40_000.0)
    vehicle_count: int = 5
    varied_properties_per_vehicle: int = 3


@dataclass(frozen=True)
class ModelArm:
    arm_id: str
    representation: str
    architecture: str
    pooling: str
    source_role: str


RAW_CNN = ModelArm(
    "RAW-CNN", "RAW-5830", "CNN", "flatten+dense", "source control"
)
PAA_CNN = ModelArm(
    "PAA-CNN", "PAA-583", "CNN", "flatten+dense", "source methodology"
)
PAA_MULTIRATE = ModelArm(
    "PAA-multirate",
    "PAA-583",
    "CNN",
    "multi-rate",
    "F25-X architecture extension",
)
SOURCE_RECONSTRUCTION_ARMS = (RAW_CNN, PAA_CNN)
F25_X_ARMS = (RAW_CNN, PAA_CNN, PAA_MULTIRATE)


@dataclass(frozen=True)
class CNNHyperparameterSearchSpace:
    convolution_filters: tuple[int, ...] = (32, 48, 64, 80, 96, 112, 128)
    convolution_kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    convolution_layer_counts: tuple[int, ...] = (1, 2, 3, 4, 5)
    optional_max_pool_after_each_layer: tuple[bool, bool] = (False, True)
    max_pool_size: int = 2
    dense_units: tuple[int, ...] = (16, 32, 48, 64)
    batch_sizes: tuple[int, ...] = (8, 16, 24, 32, 40, 48)
    learning_rate_range: tuple[float, float] = (1.0e-5, 1.0e-2)
    learning_rate_sampling: str = "log-uniform (inferred because underreported)"


SOURCE_CNN_SEARCH_SPACE = CNNHyperparameterSearchSpace()


@dataclass(frozen=True)
class PublishedPAACNNTemplate:
    sensor_id: str
    input_samples: int
    convolution_filters: tuple[int, ...]
    convolution_kernel_sizes: tuple[int, ...]
    max_pool_after_layer: tuple[bool, ...]
    max_pool_size: int
    flatten_units: int
    dense_units: int
    dense_activation: str
    output_units: int
    output_activation: str
    optimizer: str
    learning_rate: float
    batch_size: int


PUBLISHED_PAA_CNN_TEMPLATES = (
    PublishedPAACNNTemplate(
        sensor_id=CHANNELS[0],
        input_samples=583,
        convolution_filters=(48, 48),
        convolution_kernel_sizes=(2, 3),
        max_pool_after_layer=(False, False),
        max_pool_size=2,
        flatten_units=27_840,
        dense_units=48,
        dense_activation="ReLU",
        output_units=10,
        output_activation="softmax",
        optimizer="Adam",
        learning_rate=1.0e-3,
        batch_size=24,
    ),
    PublishedPAACNNTemplate(
        sensor_id=CHANNELS[1],
        input_samples=583,
        convolution_filters=(128, 96, 96),
        convolution_kernel_sizes=(3, 3, 3),
        max_pool_after_layer=(True, True, True),
        max_pool_size=2,
        flatten_units=7_008,
        dense_units=96,
        dense_activation="ReLU",
        output_units=10,
        output_activation="softmax",
        optimizer="Adam",
        learning_rate=5.0e-4,
        batch_size=32,
    ),
)

LEARNING_RATE_PLATEAU_FACTOR = 0.5
LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS = 30
MINIMUM_LEARNING_RATE = 1.0e-6
EARLY_STOPPING_PATIENCE_EPOCHS = 50
MAXIMUM_EPOCHS = 1000


@dataclass(frozen=True)
class FrozenHyperparameterAnchor:
    arm_id: str
    anchor_channel: str
    authenticated_source: str
    permitted_derivation: str


FROZEN_HP_ANCHOR_CHANNEL = CHANNELS[1]
FROZEN_HP_ANCHOR_CONTRACT_ID = "f25-x-front-bogie-arm-anchors-v1"
FROZEN_HP_ANCHORS = (
    FrozenHyperparameterAnchor(
        "RAW-CNN",
        FROZEN_HP_ANCHOR_CHANNEL,
        "authenticated F25-R front-bogie RAW-CNN winner artifact",
        "none; use the complete authenticated vector",
    ),
    FrozenHyperparameterAnchor(
        "PAA-CNN",
        FROZEN_HP_ANCHOR_CHANNEL,
        "authenticated F25-R front-bogie PAA-CNN winner artifact",
        "none; use the complete authenticated vector",
    ),
    FrozenHyperparameterAnchor(
        "PAA-multirate",
        FROZEN_HP_ANCHOR_CHANNEL,
        "authenticated F25-R front-bogie PAA-CNN winner artifact",
        "replace only flatten+dense pooling with registered multi-rate pooling",
    ),
)


def single_sensor_sets(
    channels: Sequence[str] = F25_X_CHANNELS,
) -> tuple[tuple[str, ...], ...]:
    """Return the eligible channel-order-preserving singleton inventory."""

    return tuple((str(channel),) for channel in channels)


def lexicographic_sensor_pairs(
    channels: Sequence[str] = F25_X_CHANNELS,
) -> tuple[tuple[str, str], ...]:
    """Return all eligible 6-choose-2 pairs in lexicographic order."""

    # Sort by the stable channel identifiers themselves, not by an observed
    # score and not by the physical8 storage order.
    return tuple(itertools.combinations(sorted(str(channel) for channel in channels), 2))


@dataclass(frozen=True)
class TierPlan:
    tier_id: str
    regime: str
    reporting_role: str
    arms: tuple[ModelArm, ...]
    sensor_sets: tuple[tuple[str, ...], ...]
    hpo_proposals_per_configuration: int
    hpo_executions_per_proposal: int
    report_refits_per_configuration: int
    hyperparameter_provenance: str

    @property
    def configuration_count(self) -> int:
        return len(self.arms) * len(self.sensor_sets)

    @property
    def hpo_fit_budget(self) -> int:
        return (
            self.configuration_count
            * self.hpo_proposals_per_configuration
            * self.hpo_executions_per_proposal
        )

    @property
    def report_fit_budget(self) -> int:
        return self.configuration_count * self.report_refits_per_configuration

    @property
    def core_table_fit_budget(self) -> int:
        """Budget shown in campaign-plan §11 before final HPO refits.

        Frozen tiers consist of their 20 report refits.  Unfrozen tiers consist
        of their 100 x 5 HPO executions; the separately exposed report budget
        prevents the final 20-seed distribution from becoming hidden compute.
        """

        return self.hpo_fit_budget or self.report_fit_budget


F25_R_UNFROZEN = TierPlan(
    tier_id="F25-R-unfrozen-source-singles",
    regime="unfrozen",
    reporting_role="publication-faithful reconstruction",
    arms=SOURCE_RECONSTRUCTION_ARMS,
    sensor_sets=single_sensor_sets(F25_R_CHANNELS),
    hpo_proposals_per_configuration=HPO_PROPOSALS_PER_CONFIGURATION,
    hpo_executions_per_proposal=HPO_EXECUTIONS_PER_PROPOSAL,
    report_refits_per_configuration=len(REPORT_SEEDS),
    hyperparameter_provenance=(
        "separately optimized for each source sensor and source arm"
    ),
)

F25_X_FROZEN_SINGLES = TierPlan(
    tier_id="F25-X-01-frozen-hp-singles",
    regime="frozen",
    reporting_role="primary complete frozen table",
    arms=F25_X_ARMS,
    sensor_sets=single_sensor_sets(),
    hpo_proposals_per_configuration=0,
    hpo_executions_per_proposal=0,
    report_refits_per_configuration=len(REPORT_SEEDS),
    hyperparameter_provenance=FROZEN_HP_ANCHOR_CONTRACT_ID,
)
F25_X_UNFROZEN_SINGLES = TierPlan(
    tier_id="F25-X-02-unfrozen-singles",
    regime="unfrozen",
    reporting_role="separate tuned-singles analysis",
    arms=F25_X_ARMS,
    sensor_sets=single_sensor_sets(),
    hpo_proposals_per_configuration=HPO_PROPOSALS_PER_CONFIGURATION,
    hpo_executions_per_proposal=HPO_EXECUTIONS_PER_PROPOSAL,
    report_refits_per_configuration=len(REPORT_SEEDS),
    hyperparameter_provenance="separately optimized for each sensor and arm",
)
F25_X_FROZEN_PAIRS = TierPlan(
    tier_id="F25-X-03-frozen-hp-pairs",
    regime="frozen",
    reporting_role="exploratory; run only after all singles",
    arms=F25_X_ARMS,
    sensor_sets=lexicographic_sensor_pairs(),
    hpo_proposals_per_configuration=0,
    hpo_executions_per_proposal=0,
    report_refits_per_configuration=len(REPORT_SEEDS),
    hyperparameter_provenance=FROZEN_HP_ANCHOR_CONTRACT_ID,
)
F25_X_TIERS = (
    F25_X_FROZEN_SINGLES,
    F25_X_UNFROZEN_SINGLES,
    F25_X_FROZEN_PAIRS,
)


@dataclass(frozen=True)
class ExperimentIsolation:
    experiment_id: str
    experiment_kind: str
    manifest_root: str
    cache_root: str
    results_root: str
    bundle_name: str
    shared_data_contract_id: str = SHARED_DATA_CONTRACT_ID
    partition_seed: int = PARTITION_SEED
    eov_master_seed: int = EOV_MASTER_SEED
    source_lineage_policy: str = SOURCE_LINEAGE_POLICY


F25_R_EXPERIMENT = ExperimentIsolation(
    experiment_id="F25-R",
    experiment_kind=RECONSTRUCTION_LABEL,
    manifest_root="f25_artifacts/F25-R/manifests",
    cache_root="f25_artifacts/F25-R/cache",
    results_root="f25_artifacts/F25-R/results",
    bundle_name="bundle_F25-R.zip",
)
F25_X_EXPERIMENT = ExperimentIsolation(
    experiment_id="F25-X",
    experiment_kind="extension",
    manifest_root="f25_artifacts/F25-X/manifests",
    cache_root="f25_artifacts/F25-X/cache",
    results_root="f25_artifacts/F25-X/results",
    bundle_name="bundle_F25-X.zip",
)
EXPERIMENTS = (F25_R_EXPERIMENT, F25_X_EXPERIMENT)


@dataclass(frozen=True)
class PublishedOverallAccuracy:
    sensor_id: str
    raw_cnn: float
    paa_cnn: float


PUBLISHED_OVERALL_ACCURACY = (
    PublishedOverallAccuracy(CHANNELS[0], 0.651, 0.867),
    PublishedOverallAccuracy(CHANNELS[1], 0.822, 0.821),
)


@dataclass(frozen=True)
class PublishedAxisAccuracy:
    sensor_id: str
    bearing_present_absent: float
    scour_level: float
    crack_level: float
    overall_ten_class: float


PUBLISHED_PAA_AXIS_ACCURACY = (
    PublishedAxisAccuracy(CHANNELS[0], 0.999, 0.946, 0.884, 0.867),
    PublishedAxisAccuracy(CHANNELS[1], 0.967, 0.929, 0.880, 0.821),
)
PUBLISHED_TARGET_PROVENANCE = (
    "approximate transcription from low-resolution published confusion matrices"
)
PUBLISHED_DIAGNOSTIC_COUNTS_PER_100 = {
    "carbody_DC9_as_DC10": 56,
    "carbody_DC10_as_DC9": 21,
    "front_bogie_DC10_correct": 12,
    "front_bogie_DC10_as_DC9": 69,
    "front_bogie_DC4_as_Healthy": 28,
    "front_bogie_Healthy_as_DC4": 5,
}
REPORTING_METRICS = (
    "overall 10-class accuracy",
    "bearing present/absent accuracy",
    "three-level scour accuracy",
    "three-level crack-depth accuracy",
    "per-run metric distribution over 20 predeclared seeds",
    "best-run confusion matrix for visual comparability only",
)


@dataclass(frozen=True)
class DeviationRow:
    item_id: str
    classification: str
    applies_to: str
    source_choice: str
    implemented_choice: str
    rationale: str


DEVIATION_ROWS = (
    DeviationRow(
        "table2_damage_scenarios",
        "exactly reproduced",
        "F25-R/F25-X",
        "Healthy plus DC2-DC10",
        "the ten source-backed damage combinations in SCENARIOS",
        "Table 2 mapping is available",
    ),
    DeviationRow(
        "bridge_geometry",
        "exactly reproduced",
        "F25-R/F25-X",
        "39.9 m bridge; two 19.95 m spans",
        "39.9 m bridge; two 19.95 m spans",
        "source geometry and stored profile agree",
    ),
    DeviationRow(
        "crack_zone",
        "exactly reproduced",
        "F25-R/F25-X",
        "source element 100: [29.70, 30.00] m",
        "two refined elements spanning [29.70, 30.00] m",
        "uniform refinement preserves both source boundaries",
    ),
    DeviationRow(
        "profile_realization",
        "exactly reproduced",
        "F25-R/F25-X",
        "Profile.Type == 2; Calc.ProfileData15_05.mat",
        f"Profile.Type == {PROFILE_TYPE}; SHA-256 {PROFILE_SHA256}",
        "the original fixed realization is available and hashed",
    ),
    DeviationRow(
        "split_counts",
        "exactly reproduced",
        "F25-R/F25-X",
        "100 train candidates and 100 test; 20% of train for validation",
        "100 test, 20 validation, 80 train per class",
        "the published allocation counts are explicit",
    ),
    DeviationRow(
        "source_monitoring_window",
        "exactly reproduced",
        "F25-R/F25-X",
        "58.30 m / 5,830 data points",
        "full-RAW samples 1001..6831 inclusive, then trim sample 6831",
        "the final learning signal has exactly the published 5,830 points",
    ),
    DeviationRow(
        "noise_equation",
        "exactly reproduced",
        "F25-R/F25-X",
        "a_calc + 0.05 N(0,1) sigma(a_calc), SNR label 20",
        "per-passage/per-channel population sigma before MinMax/PAA",
        "Eq. (2) fixes the magnitude and transform order",
    ),
    DeviationRow(
        "hpo_execution_semantics",
        "exactly reproduced",
        "unfrozen F25-R/F25-X tiers",
        "100 proposals; every proposal executed five times",
        "one 100-proposal study with five seeded executions per proposal",
        "five executions are not reinterpreted as five optimizer studies",
    ),
    DeviationRow(
        "cnn_search_and_paa_finalists",
        "exactly reproduced",
        "F25-R",
        "published numeric CNN search domain and sensor-specific PAA finalists",
        "registered numeric bounds/steps plus PUBLISHED_PAA_CNN_TEMPLATES",
        "source PDF figures and hyperparameter table are available",
    ),
    DeviationRow(
        "reported_run_count",
        "exactly reproduced",
        "F25-R/F25-X",
        "CNN executed 20 times for the accuracy distribution",
        "20 predeclared report seeds; best confusion matrix is visual only",
        "source section 5.1 explicitly states the count",
    ),
    DeviationRow(
        "partition_allocation_and_seed",
        "inferred because underreported",
        "F25-R/F25-X",
        "allocation algorithm and random seed not reported",
        f"{PARTITION_ALGORITHM}; seed {PARTITION_SEED}",
        "predeclared once and shared by both experiment IDs",
    ),
    DeviationRow(
        "eov_and_training_seeds",
        "inferred because underreported",
        "F25-R/F25-X",
        "individual random seeds not reported",
        "explicit shared EOV, HPO-execution, and 20 report seed inventories",
        "prevents reconstruction/extension partition or seed drift",
    ),
    DeviationRow(
        "minmax_calibration_scope",
        "inferred because underreported",
        "F25-R/F25-X",
        "MinMax reported without calibration-axis detail",
        MINMAX_SCOPE,
        "fit on training data only, separately for every physical channel",
    ),
    DeviationRow(
        "published_accuracy_precision",
        "inferred because underreported",
        "F25-R acceptance comparison",
        "low-resolution confusion-matrix figure",
        PUBLISHED_TARGET_PROVENANCE,
        "targets are comparison references, not falsely exact thresholds",
    ),
    DeviationRow(
        "learning_rate_sampling_distribution",
        "inferred because underreported",
        "F25-R/F25-X unfrozen tiers",
        "learning-rate bounds 1e-5..1e-2; proposal distribution not stated",
        "log-uniform sampling inside the exact published bounds",
        "predeclares a scale-appropriate distribution without claiming it exact",
    ),
    DeviationRow(
        "frozen_front_bogie_anchor",
        "inferred because underreported",
        "F25-X frozen tiers only",
        "no source frozen-channel comparison protocol",
        "one authenticated front-bogie vector per registered arm",
        "predeclares one vector per arm and avoids outcome-selected anchors",
    ),
    DeviationRow(
        "deck_mesh",
        "deliberately changed",
        "F25-R/F25-X",
        "0.30 m source grid with central support mid-element",
        "0.15 m uniform refinement; 266 elements",
        "preserves every source node and crack boundary while aligning support",
    ),
    DeviationRow(
        "native_window_tail",
        "deliberately changed",
        "F25-R/F25-X",
        "published model consumes 5,830 samples",
        "trim only the final sample from the native 5,831-sample window",
        "keeps the opening coordinate and enables 583 exact blocks of ten",
    ),
    DeviationRow(
        "round_before_scale_window_reconciliation",
        "deliberately changed",
        "F25-R/F25-X only",
        "paper reports a 58.30 m / 5,830-point input",
        "repository inclusive recrop uses round(39.9)x100, then tail-trims",
        "adds 10 samples (0.10 m) versus the corrected 5,821-sample crop",
    ),
    DeviationRow(
        "crack_mechanics",
        "deliberately changed",
        "F25-R/F25-X",
        "Sinha tapered local-EI crack model parameterized by crack depth",
        "uniform 0.30 m EI block with equivalent 22% or 14% EI loss",
        "preserves source zone and equivalent magnitude, not local taper mechanics",
    ),
    DeviationRow(
        "learning_framework",
        "deliberately changed",
        "F25-R/F25-X",
        "TensorFlow implementation",
        "PyTorch implementation",
        "initialization, optimizer defaults, and stopping semantics can differ",
    ),
    DeviationRow(
        "extension_channels_and_pairs",
        "deliberately changed",
        "F25-X only",
        "two vertical single sensors",
        "six learning-eligible singles plus 15 pre-registered exploratory pairs; "
        "wheelset proxies excluded",
        "tests channel hypotheses without treating diagnostic proxies as sensors",
    ),
    DeviationRow(
        "multirate_arm",
        "deliberately changed",
        "F25-X only",
        "source CNN with flatten and dense layers",
        "PAA CNN with multi-rate pooling as a separate third arm",
        "architecture extension; no global-average-pooling source claim is made",
    ),
    DeviationRow(
        "multirate_frozen_derivation",
        "deliberately changed",
        "F25-X frozen tiers only",
        "no source multi-rate arm",
        "front-bogie PAA-CNN vector with only the pooling mechanism replaced",
        "keeps the frozen extension one-change-at-a-time and fail-closed",
    ),
)


@dataclass(frozen=True)
class MinMaxCalibration:
    """Per-channel MinMax parameters fitted on the training partition only."""

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.minimum or len(self.minimum) != len(self.maximum):
            raise F25ContractError("MinMax calibration channel inventory is invalid")
        lo = np.asarray(self.minimum, dtype=np.float64)
        hi = np.asarray(self.maximum, dtype=np.float64)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise F25ContractError("MinMax calibration must be finite")
        if np.any(hi <= lo):
            raise F25ContractError("every MinMax channel must have positive span")


def _normal_axis(axis: int, ndim: int, name: str) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise F25ContractError(f"{name}={axis} is invalid for ndim={ndim}")
    return normalized


def extract_monitoring_window(
    full_raw: np.ndarray | Sequence[float], *, time_axis: int = -1
) -> np.ndarray:
    """Extract repository full-RAW samples 1001..6831 (MATLAB indexing)."""

    array = np.asarray(full_raw)
    if array.ndim == 0:
        raise F25ContractError("full RAW response must have a time axis")
    axis = _normal_axis(time_axis, array.ndim, "time_axis")
    if array.shape[axis] < MONITORING_CROP_END_ONE_BASED:
        raise F25ContractError(
            "full RAW response does not cover the F25 source-era monitoring crop"
        )
    selection = [slice(None)] * array.ndim
    # Convert inclusive MATLAB coordinates 1001..6831 to Python's half-open
    # zero-based slice 1000:6831.
    selection[axis] = slice(
        MONITORING_CROP_START_ONE_BASED - 1,
        MONITORING_CROP_END_ONE_BASED,
    )
    result = array[tuple(selection)]
    if result.shape[axis] != SOURCE_WINDOW_SAMPLES or not np.all(np.isfinite(result)):
        raise F25ContractError("extracted F25 monitoring window is invalid")
    return result


def trim_native_window(
    samples: np.ndarray | Sequence[float], *, time_axis: int = -1
) -> np.ndarray:
    """Trim exactly the final sample of a native 5,831-sample window."""

    array = np.asarray(samples)
    if array.ndim == 0:
        raise F25ContractError("F25 window must have a time axis")
    axis = _normal_axis(time_axis, array.ndim, "time_axis")
    if array.shape[axis] != SOURCE_WINDOW_SAMPLES:
        raise F25ContractError(
            f"F25 source window must be exactly {SOURCE_WINDOW_SAMPLES} samples; "
            f"got {array.shape[axis]}"
        )
    if not np.all(np.isfinite(array)):
        raise F25ContractError("F25 source window contains NaN or infinity")
    selection = [slice(None)] * array.ndim
    selection[axis] = slice(0, TRIMMED_WINDOW_SAMPLES)
    return array[tuple(selection)]


def add_source_noise(
    clean_trimmed_signal: np.ndarray | Sequence[float],
    *,
    class_index: int,
    passage_index: int,
    channel_index: int,
) -> np.ndarray:
    """Apply source Eq.(2) to one clean, trimmed passage/channel signal.

    The random key contains only immutable physical-data identifiers.  It does
    not contain split, experiment, representation, model, or training IDs, so
    R/X and RAW/PAA consume the same noisy realization.
    """

    signal = np.asarray(clean_trimmed_signal, dtype=np.float64)
    if signal.ndim != 1 or signal.shape[0] != TRIMMED_WINDOW_SAMPLES:
        raise F25ContractError(
            "source noise requires one 5,830-sample passage/channel signal"
        )
    if not np.all(np.isfinite(signal)):
        raise F25ContractError("source-noise input contains NaN or infinity")
    identifiers = (class_index, passage_index, channel_index)
    limits = (CLASS_COUNT, PASSAGES_PER_CLASS, len(CHANNELS))
    if any(
        not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) >= limit
        for value, limit in zip(identifiers, limits)
    ):
        raise F25ContractError("source-noise physical identifiers are invalid")
    seed_sequence = np.random.SeedSequence(
        [NOISE_MASTER_SEED, *(int(value) for value in identifiers)]
    )
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    sigma = float(np.std(signal, ddof=NOISE_STANDARD_DEVIATION_DDOF))
    return signal + NOISE_E_LEVEL * sigma * generator.standard_normal(signal.shape)


def fit_trimmed_training_minmax(
    noisy_training_samples: np.ndarray | Sequence[float],
    *,
    channel_axis: int = -2,
    time_axis: int = -1,
) -> MinMaxCalibration:
    """Fit per-channel MinMax after crop, trim, and source-noise injection."""

    trimmed = np.asarray(noisy_training_samples)
    if trimmed.ndim < 2 or not np.all(np.isfinite(trimmed)):
        raise F25ContractError("MinMax fitting requires finite channel/time data")
    channel = _normal_axis(channel_axis, trimmed.ndim, "channel_axis")
    time = _normal_axis(time_axis, trimmed.ndim, "time_axis")
    if channel == time:
        raise F25ContractError("channel_axis and time_axis must differ")
    if trimmed.shape[time] != TRIMMED_WINDOW_SAMPLES:
        raise F25ContractError(
            "post-noise MinMax fitting requires exactly 5,830 time samples"
        )
    reduction_axes = tuple(index for index in range(trimmed.ndim) if index != channel)
    lo = np.min(trimmed, axis=reduction_axes)
    hi = np.max(trimmed, axis=reduction_axes)
    return MinMaxCalibration(
        tuple(float(value) for value in np.ravel(lo)),
        tuple(float(value) for value in np.ravel(hi)),
    )


def fit_training_minmax(
    training_windows: np.ndarray | Sequence[float],
    *,
    channel_axis: int = -2,
    time_axis: int = -1,
) -> MinMaxCalibration:
    """Fit MinMax on native windows when noise was already applied upstream.

    The campaign implementation should normally call
    :func:`fit_trimmed_training_minmax` after :func:`add_source_noise`.  This
    native-window convenience path exists for deterministic clean fixtures.
    """

    trimmed = trim_native_window(training_windows, time_axis=time_axis)
    return fit_trimmed_training_minmax(
        trimmed,
        channel_axis=channel_axis,
        time_axis=time_axis,
    )


def apply_training_minmax(
    trimmed_samples: np.ndarray | Sequence[float],
    calibration: MinMaxCalibration,
    *,
    channel_axis: int = -2,
) -> np.ndarray:
    """Apply frozen training MinMax values without clipping validation/test."""

    array = np.asarray(trimmed_samples, dtype=np.float64)
    if array.ndim < 2 or not np.all(np.isfinite(array)):
        raise F25ContractError("MinMax input must be a finite channel/time array")
    channel = _normal_axis(channel_axis, array.ndim, "channel_axis")
    if array.shape[channel] != len(calibration.minimum):
        raise F25ContractError(
            "MinMax calibration channel count does not match the input"
        )
    shape = [1] * array.ndim
    shape[channel] = len(calibration.minimum)
    lo = np.asarray(calibration.minimum, dtype=np.float64).reshape(shape)
    hi = np.asarray(calibration.maximum, dtype=np.float64).reshape(shape)
    # Do not clip: values beyond the training extrema must remain observable.
    return (array - lo) / (hi - lo)


def prepare_raw(
    samples: np.ndarray | Sequence[float],
    calibration: MinMaxCalibration,
    *,
    channel_axis: int = -2,
    time_axis: int = -1,
) -> np.ndarray:
    """Execute the frozen RAW pipeline: tail trim, then training MinMax."""

    trimmed = trim_native_window(samples, time_axis=time_axis)
    return apply_training_minmax(
        trimmed, calibration, channel_axis=channel_axis
    )


def paa_blocks_of_ten(
    minmax_samples: np.ndarray | Sequence[float], *, time_axis: int = -1
) -> np.ndarray:
    """Average 5,830 MinMax-scaled samples into 583 non-overlapping blocks."""

    array = np.asarray(minmax_samples, dtype=np.float64)
    if array.ndim == 0 or not np.all(np.isfinite(array)):
        raise F25ContractError("PAA input must be a finite array")
    axis = _normal_axis(time_axis, array.ndim, "time_axis")
    if array.shape[axis] != TRIMMED_WINDOW_SAMPLES:
        raise F25ContractError(
            f"PAA input must be exactly {TRIMMED_WINDOW_SAMPLES} samples"
        )
    moved = np.moveaxis(array, axis, -1)
    blocked = moved.reshape(*moved.shape[:-1], PAA_BLOCK_COUNT, PAA_BLOCK_SIZE)
    reduced = blocked.mean(axis=-1)
    return np.moveaxis(reduced, -1, axis)


def prepare_paa(
    samples: np.ndarray | Sequence[float],
    calibration: MinMaxCalibration,
    *,
    channel_axis: int = -2,
    time_axis: int = -1,
) -> np.ndarray:
    """Execute trim -> training MinMax -> PAA in the mandated order."""

    minmax = prepare_raw(
        samples,
        calibration,
        channel_axis=channel_axis,
        time_axis=time_axis,
    )
    return paa_blocks_of_ten(minmax, time_axis=time_axis)


def partition_indices(seed: int = PARTITION_SEED) -> dict[str, np.ndarray]:
    """Return the shared deterministic local-passage split for every class.

    Local passage indices 0..199 are permuted once, then the same allocation is
    applied within each damage class.  ``F25-R`` and ``F25-X`` must call this
    function (or verify its digest), so comparisons remain paired.
    """

    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise F25ContractError("partition seed must be a non-negative integer")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    permutation = generator.permutation(PASSAGES_PER_CLASS)
    test_stop = TEST_PER_CLASS
    validation_stop = test_stop + VALIDATION_PER_CLASS
    return {
        "test": permutation[:test_stop].copy(),
        "validation": permutation[test_stop:validation_stop].copy(),
        "train": permutation[validation_stop:].copy(),
    }


def partition_sha256(seed: int = PARTITION_SEED) -> str:
    """Hash split names, shapes, dtypes, and little-endian int64 bytes."""

    digest = hashlib.sha256()
    split = partition_indices(seed)
    for name in PARTITION_ORDER:
        values = np.asarray(split[name], dtype="<i8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(values.shape).encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tier_payload(tier: TierPlan) -> dict[str, Any]:
    payload = _jsonable(tier)
    payload.update(
        {
            "configuration_count": tier.configuration_count,
            "hpo_fit_budget": tier.hpo_fit_budget,
            "report_fit_budget": tier.report_fit_budget,
            "core_table_fit_budget": tier.core_table_fit_budget,
        }
    )
    return payload


def _contract_body() -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "naming_rule": RECONSTRUCTION_LABEL,
        "source_lineage_policy": SOURCE_LINEAGE_POLICY,
        "channel_schema_id": CHANNEL_SCHEMA_ID,
        "channels": list(CHANNELS),
        "eligible_sensor_indices": list(ELIGIBLE_SENSOR_INDICES),
        "eligible_sensor_channels": list(ELIGIBLE_SENSOR_CHANNELS),
        "excluded_proxy_indices": list(EXCLUDED_PROXY_INDICES),
        "geometry": _jsonable(GeometryContract()),
        "profile": {
            "type": PROFILE_TYPE,
            "relative_path": PROFILE_RELATIVE_PATH,
            "sha256": PROFILE_SHA256,
        },
        "scenarios": _jsonable(SCENARIOS),
        "passages": {
            "classes": CLASS_COUNT,
            "per_class": PASSAGES_PER_CLASS,
            "total": TOTAL_PASSAGES,
        },
        "partition": {
            "shared_data_contract_id": SHARED_DATA_CONTRACT_ID,
            "order": list(PARTITION_ORDER),
            "counts_per_class": dict(zip(PARTITION_ORDER, PARTITION_COUNTS)),
            "algorithm": PARTITION_ALGORITHM,
            "seed": PARTITION_SEED,
            "sha256": partition_sha256(),
        },
        "eov": _jsonable(EOVContract()),
        "eov_master_seed": EOV_MASTER_SEED,
        "preprocessing": {
            "source_samples": SOURCE_WINDOW_SAMPLES,
            "tail_samples_trimmed": TAIL_SAMPLES_TRIMMED,
            "trimmed_samples": TRIMMED_WINDOW_SAMPLES,
            "monitoring_window": {
                "samples_per_m": MONITORING_SAMPLES_PER_M,
                "crop_start_one_based": MONITORING_CROP_START_ONE_BASED,
                "bridge_term_samples": MONITORING_BRIDGE_TERM_SAMPLES,
                "post_term_samples": MONITORING_POST_TERM_SAMPLES,
                "crop_end_one_based": MONITORING_CROP_END_ONE_BASED,
                "corrected_generic_window_samples": CORRECTED_GENERIC_WINDOW_SAMPLES,
                "reconciliation_extra_samples": WINDOW_RECONCILIATION_EXTRA_SAMPLES,
                "reconciliation_extra_m": WINDOW_RECONCILIATION_EXTRA_M,
            },
            "noise": {
                "contract_id": NOISE_CONTRACT_ID,
                "equation": "a_calc + E_level * N(0,1) * population_std(a_calc)",
                "e_level": NOISE_E_LEVEL,
                "snr_label": NOISE_SNR_LABEL,
                "standard_deviation_ddof": NOISE_STANDARD_DEVIATION_DDOF,
                "master_seed": NOISE_MASTER_SEED,
                "order_policy": NOISE_ORDER_POLICY,
            },
            "minmax_scope": MINMAX_SCOPE,
            "minmax_clip": MINMAX_CLIP,
            "paa_block_size": PAA_BLOCK_SIZE,
            "paa_block_count": PAA_BLOCK_COUNT,
            "raw_steps": list(RAW_PREPROCESSING_STEPS),
            "paa_steps": list(PAA_PREPROCESSING_STEPS),
        },
        "hpo": {
            "optimizer_studies_per_configuration": 1,
            "proposals_per_configuration": HPO_PROPOSALS_PER_CONFIGURATION,
            "executions_per_proposal": HPO_EXECUTIONS_PER_PROPOSAL,
            "executions_per_configuration": HPO_EXECUTIONS_PER_CONFIGURATION,
            "execution_seeds": list(HPO_EXECUTION_SEEDS),
            "source_cnn_search_space": _jsonable(SOURCE_CNN_SEARCH_SPACE),
        },
        "source_training_schedule": {
            "optimizer": "Adam",
            "learning_rate_plateau_factor": LEARNING_RATE_PLATEAU_FACTOR,
            "learning_rate_plateau_patience_epochs": (
                LEARNING_RATE_PLATEAU_PATIENCE_EPOCHS
            ),
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "early_stopping_patience_epochs": EARLY_STOPPING_PATIENCE_EPOCHS,
            "maximum_epochs": MAXIMUM_EPOCHS,
        },
        "report_seeds": list(REPORT_SEEDS),
        "experiments": _jsonable(EXPERIMENTS),
        "F25-R": {
            "selected_channels": list(F25_R_CHANNELS),
            "published_paa_cnn_templates": _jsonable(PUBLISHED_PAA_CNN_TEMPLATES),
            "tiers": [_tier_payload(F25_R_UNFROZEN)],
        },
        "F25-X": {
            "selected_channels": list(F25_X_CHANNELS),
            "tier_order": [tier.tier_id for tier in F25_X_TIERS],
            "tiers": [_tier_payload(tier) for tier in F25_X_TIERS],
            "frozen_hp_anchor_contract_id": FROZEN_HP_ANCHOR_CONTRACT_ID,
            "frozen_hp_anchors": _jsonable(FROZEN_HP_ANCHORS),
            "mandatory_reporting_separation": (
                "frozen and unfrozen results must never share a comparison table"
            ),
        },
        "published_targets": {
            "overall_accuracy": _jsonable(PUBLISHED_OVERALL_ACCURACY),
            "paa_axis_accuracy": _jsonable(PUBLISHED_PAA_AXIS_ACCURACY),
            "provenance": PUBLISHED_TARGET_PROVENANCE,
            "diagnostic_counts_per_100": dict(PUBLISHED_DIAGNOSTIC_COUNTS_PER_100),
            "reporting_metrics": list(REPORTING_METRICS),
        },
        "deviation_classifications": list(DEVIATION_CLASSIFICATIONS),
        "deviation_rows": _jsonable(DEVIATION_ROWS),
    }


def build_contract() -> dict[str, Any]:
    """Return a fresh canonical contract plus a digest over its body."""

    body = _contract_body()
    payload = dict(body)
    payload["contract_sha256"] = canonical_json_sha256(body)
    return payload


def validate_contract_payload(payload: Mapping[str, Any]) -> None:
    """Reject any payload that is incomplete, self-inconsistent, or drifted."""

    if not isinstance(payload, Mapping):
        raise F25ContractError("F25 contract payload must be a mapping")
    candidate = _jsonable(payload)
    supplied_digest = candidate.pop("contract_sha256", None)
    if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
        raise F25ContractError("F25 contract_sha256 is absent or malformed")
    if canonical_json_sha256(candidate) != supplied_digest:
        raise F25ContractError("F25 contract payload digest does not verify")
    expected = _contract_body()
    if candidate != expected:
        raise F25ContractError("F25 contract payload differs from authority")


def _validate_relative_root(path: str) -> None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != path:
        raise F25ContractError(f"artifact root is not normalized and relative: {path}")


def validate_contract() -> None:
    """Run semantic invariants in addition to exact payload authentication."""

    if tuple(scenario.class_index for scenario in SCENARIOS) != tuple(range(10)):
        raise F25ContractError("scenario indices are not contiguous 0..9")
    if tuple(scenario.label for scenario in SCENARIOS) != (
        "Healthy",
        "DC2",
        "DC3",
        "DC4",
        "DC5",
        "DC6",
        "DC7",
        "DC8",
        "DC9",
        "DC10",
    ):
        raise F25ContractError("scenario label order drifted")
    for scenario in SCENARIOS:
        if scenario.crack_depth_ratio is None:
            if scenario.crack_ei_loss_fraction != 0.0:
                raise F25ContractError("EI loss exists without a crack depth")
        elif (scenario.crack_depth_ratio, scenario.crack_ei_loss_fraction) not in {
            (0.10, 0.22),
            (0.05, 0.14),
        }:
            raise F25ContractError("crack depth/EI correction mapping drifted")
        if scenario.central_scour_kv_loss_fraction not in {0.0, 0.05, 0.10}:
            raise F25ContractError("unregistered central-scour level")
        if scenario.entrance_bearing_kr_nm_per_rad not in {
            0.0,
            DAMAGED_ENTRANCE_BEARING_KR_NM_PER_RAD,
        } or scenario.exit_bearing_kr_nm_per_rad != 0.0:
            raise F25ContractError("bearing location/value drifted")

    if not np.isclose(sum(SPAN_LENGTHS_M), BRIDGE_LENGTH_M, rtol=0.0, atol=1e-12):
        raise F25ContractError("span lengths do not sum to bridge length")
    if not np.isclose(
        DECK_ELEMENT_COUNT * DECK_MESH_M, BRIDGE_LENGTH_M, rtol=0.0, atol=1e-12
    ):
        raise F25ContractError("mesh count does not cover the bridge")
    if not np.isclose(CENTRAL_SUPPORT_X_M / DECK_MESH_M, 133.0):
        raise F25ContractError("central support is not on refined grid index 133")
    crack_indices = np.asarray(CRACK_ZONE_M) / DECK_MESH_M
    if not np.allclose(crack_indices, (198.0, 200.0), rtol=0.0, atol=1e-12):
        raise F25ContractError("crack zone is not exactly two refined elements")
    if DECK_ELEMENT_COUNT != 2 * 133 or CRACK_ELEMENT_NUMBERS_ONE_BASED != (199, 200):
        raise F25ContractError("refined element identity drifted")
    if len(PROFILE_SHA256) != 64 or set(PROFILE_SHA256) - set("0123456789abcdef"):
        raise F25ContractError("profile SHA-256 is malformed")
    if (
        MONITORING_BRIDGE_TERM_SAMPLES + MONITORING_POST_TERM_SAMPLES
        != SOURCE_WINDOW_SAMPLES
        or MONITORING_CROP_END_ONE_BASED - MONITORING_CROP_START_ONE_BASED + 1
        != SOURCE_WINDOW_SAMPLES
        or SOURCE_WINDOW_SAMPLES - TAIL_SAMPLES_TRIMMED
        != TRIMMED_WINDOW_SAMPLES
        or TRIMMED_WINDOW_SAMPLES != PAA_BLOCK_COUNT * PAA_BLOCK_SIZE
    ):
        raise F25ContractError("monitoring trim/PAA arithmetic drifted")
    if (
        CORRECTED_GENERIC_BRIDGE_TERM_SAMPLES + MONITORING_POST_TERM_SAMPLES
        != CORRECTED_GENERIC_WINDOW_SAMPLES
        or SOURCE_WINDOW_SAMPLES - CORRECTED_GENERIC_WINDOW_SAMPLES
        != WINDOW_RECONCILIATION_EXTRA_SAMPLES
        or WINDOW_RECONCILIATION_EXTRA_M
        != WINDOW_RECONCILIATION_EXTRA_SAMPLES / MONITORING_SAMPLES_PER_M
    ):
        raise F25ContractError("round-before-scale window reconciliation drifted")
    if (
        NOISE_E_LEVEL != 0.05
        or NOISE_SNR_LABEL != 20
        or NOISE_STANDARD_DEVIATION_DDOF != 0
        or "then fit/apply MinMax; then PAA" not in NOISE_ORDER_POLICY
    ):
        raise F25ContractError("source Eq.(2) noise contract drifted")

    split = partition_indices()
    if tuple(len(split[name]) for name in PARTITION_ORDER) != PARTITION_COUNTS:
        raise F25ContractError("partition counts drifted")
    combined = np.concatenate([split[name] for name in PARTITION_ORDER])
    if not np.array_equal(np.sort(combined), np.arange(PASSAGES_PER_CLASS)):
        raise F25ContractError("partition is not a disjoint complete allocation")

    if len(CHANNELS) != 8 or len(set(CHANNELS)) != 8:
        raise F25ContractError("physical8_v1 channel inventory drifted")
    if (
        EXCLUDED_PROXY_INDICES != (3, 4)
        or ELIGIBLE_SENSOR_INDICES != (0, 1, 2, 5, 6, 7)
        or set(EXCLUDED_PROXY_INDICES) & set(ELIGIBLE_SENSOR_INDICES)
        or set(EXCLUDED_PROXY_INDICES) | set(ELIGIBLE_SENSOR_INDICES)
        != set(range(len(CHANNELS)))
        or ELIGIBLE_SENSOR_CHANNELS
        != tuple(CHANNELS[index] for index in ELIGIBLE_SENSOR_INDICES)
        or F25_X_CHANNELS != ELIGIBLE_SENSOR_CHANNELS
    ):
        raise F25ContractError("F25-X scientific sensor eligibility drifted")
    if F25_R_CHANNELS != CHANNELS[:2]:
        raise F25ContractError("F25-R is not limited to the two source sensors")
    if tuple(arm.pooling for arm in SOURCE_RECONSTRUCTION_ARMS) != (
        "flatten+dense",
        "flatten+dense",
    ):
        raise F25ContractError("published CNNs are not flatten+dense baselines")
    if SOURCE_CNN_SEARCH_SPACE != CNNHyperparameterSearchSpace():
        raise F25ContractError("source CNN search domain drifted")
    if tuple(template.sensor_id for template in PUBLISHED_PAA_CNN_TEMPLATES) != F25_R_CHANNELS:
        raise F25ContractError("published PAA finalist sensor inventory drifted")
    if (
        PUBLISHED_PAA_CNN_TEMPLATES[0].flatten_units != 27_840
        or PUBLISHED_PAA_CNN_TEMPLATES[1].flatten_units != 7_008
        or tuple(anchor.arm_id for anchor in FROZEN_HP_ANCHORS)
        != tuple(arm.arm_id for arm in F25_X_ARMS)
        or any(anchor.anchor_channel != FROZEN_HP_ANCHOR_CHANNEL for anchor in FROZEN_HP_ANCHORS)
    ):
        raise F25ContractError("published finalists or frozen arm anchors drifted")
    pairs = lexicographic_sensor_pairs()
    if len(pairs) != 15 or len(set(pairs)) != 15 or tuple(sorted(pairs)) != pairs:
        raise F25ContractError("F25-X pair order is not complete eligible 6C2")

    expected_tier_numbers = (
        (F25_R_UNFROZEN, 4, 2000, 80, 2000),
        (F25_X_FROZEN_SINGLES, 18, 0, 360, 360),
        (F25_X_UNFROZEN_SINGLES, 18, 9000, 360, 9000),
        (F25_X_FROZEN_PAIRS, 45, 0, 900, 900),
    )
    for tier, configurations, hpo, report, table in expected_tier_numbers:
        actual = (
            tier.configuration_count,
            tier.hpo_fit_budget,
            tier.report_fit_budget,
            tier.core_table_fit_budget,
        )
        if actual != (configurations, hpo, report, table):
            raise F25ContractError(f"tier budget drifted for {tier.tier_id}: {actual}")
    if (
        F25_X_FROZEN_SINGLES.hyperparameter_provenance
        != FROZEN_HP_ANCHOR_CONTRACT_ID
        or F25_X_FROZEN_PAIRS.hyperparameter_provenance
        != FROZEN_HP_ANCHOR_CONTRACT_ID
    ):
        raise F25ContractError("frozen singles/pairs do not share registered anchors")
    if len(REPORT_SEEDS) != 20 or len(set(REPORT_SEEDS)) != 20:
        raise F25ContractError("report seed inventory is not 20 unique seeds")
    if len(HPO_EXECUTION_SEEDS) != 5 or len(set(HPO_EXECUTION_SEEDS)) != 5:
        raise F25ContractError("HPO execution seed inventory is not five seeds")

    roots: list[str] = []
    for experiment in EXPERIMENTS:
        for root in (
            experiment.manifest_root,
            experiment.cache_root,
            experiment.results_root,
        ):
            _validate_relative_root(root)
            roots.append(root)
        if (
            experiment.shared_data_contract_id != SHARED_DATA_CONTRACT_ID
            or experiment.partition_seed != PARTITION_SEED
            or experiment.eov_master_seed != EOV_MASTER_SEED
            or experiment.source_lineage_policy != SOURCE_LINEAGE_POLICY
        ):
            raise F25ContractError("F25-R/F25-X shared-data/lineage binding drifted")
    if len(set(roots)) != len(roots) or F25_R_EXPERIMENT.experiment_id == F25_X_EXPERIMENT.experiment_id:
        raise F25ContractError("F25 experiment IDs or artifact roots overlap")

    classes = tuple(row.classification for row in DEVIATION_ROWS)
    if set(classes) != set(DEVIATION_CLASSIFICATIONS):
        raise F25ContractError("deviation table does not use exactly three classes")
    if any(value not in DEVIATION_CLASSIFICATIONS for value in classes):
        raise F25ContractError("unregistered deviation classification")
    if len({row.item_id for row in DEVIATION_ROWS}) != len(DEVIATION_ROWS):
        raise F25ContractError("deviation item identifiers collide")

    validate_contract_payload(build_contract())


validate_contract()
