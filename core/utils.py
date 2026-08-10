"""
core/utils.py
=============
Three unrelated but universally needed utilities:

    1. set_global_seed      - reproducibility across Python / NumPy / PyTorch.
    2. define_save_locations - canonical naming for Optuna DBs and output dirs.
    3. DOF definitions      - single source of truth for channel names and indices.

Imported by:
    ablation.ipynb          - seed, save locations, DOF maps
    training/trainer.py     - set_global_seed at the top of every trial
    training/pipeline.py    - set_global_seed, define_save_locations
    digital_twin/assets.py  - DOF maps (channel selection at inference time)

No ML framework is imported here beyond torch (for seed control).
No other core module is imported here - this file has no internal deps.
"""

import os
import random

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

DETERMINISM_POLICY = {
    # Executable JSON-compatible inputs. TRAIN_PROTOCOL embeds this exact
    # mapping, so the unified protocol hash describes runtime behaviour rather
    # than a prose promise that can drift from the implementation.
    "python_random": "seed",
    "python_hash_seed_environment": "seed",
    "numpy_random": "seed",
    "torch_cpu_random": "seed",
    "torch_cuda_random": "all_devices",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "torch_deterministic_algorithms": True,
    "torch_deterministic_warn_only": False,
    "cublas_workspace_config": ":4096:8",
}


def _validated_seed(seed: int) -> int:
    """Return an integer accepted by every registered campaign RNG."""

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    # numpy.random.seed is the narrowest API in the registered stack.
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must lie in [0, 2**32 - 1]")
    return seed


def _validate_determinism_policy(policy: dict) -> None:
    required = set(DETERMINISM_POLICY)
    if not isinstance(policy, dict) or set(policy) != required:
        got = set(policy) if isinstance(policy, dict) else type(policy).__name__
        raise ValueError(
            "determinism policy must define exactly "
            f"{sorted(required)!r}; got {got!r}.")
    for key in (
        "python_random",
        "python_hash_seed_environment",
        "numpy_random",
        "torch_cpu_random",
    ):
        if policy[key] != "seed":
            raise ValueError(
                f"unsupported determinism policy {key}={policy[key]!r}")
    if policy["torch_cuda_random"] != "all_devices":
        raise ValueError(
            "torch_cuda_random must be 'all_devices' for the campaign")
    for key in (
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "torch_deterministic_algorithms",
        "torch_deterministic_warn_only",
    ):
        if not isinstance(policy[key], bool):
            raise ValueError(f"determinism policy {key} must be boolean")
    matmul_precision = policy["float32_matmul_precision"]
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "float32_matmul_precision must be one of "
            "{'highest', 'high', 'medium'}")
    # In PyTorch 2.7 these are two interfaces to the same CUDA matmul mode.
    # Reject an internally contradictory protocol rather than silently letting
    # the last assignment win.
    tf32_implied_by_precision = matmul_precision != "highest"
    if policy["cuda_matmul_allow_tf32"] != tf32_implied_by_precision:
        raise ValueError(
            "cuda_matmul_allow_tf32 conflicts with "
            "float32_matmul_precision")
    workspace = policy["cublas_workspace_config"]
    if not isinstance(workspace, str) or not workspace:
        raise ValueError(
            "cublas_workspace_config must be a non-empty string")


def set_global_seed(
    seed: int,
    policy: dict = DETERMINISM_POLICY,
) -> None:
    """Seed the registered RNGs and apply the executable determinism policy.

    The active campaign passes ``TRAIN_PROTOCOL["determinism"]`` explicitly.
    The default preserves older utility callers while pointing at the same
    canonical mapping; there is deliberately no default *seed*.
    """

    seed = _validated_seed(seed)
    _validate_determinism_policy(policy)

    random.seed(seed)
    # This controls child interpreters; Python's current hash salt is fixed at
    # interpreter startup, so campaign logic never relies on salted hash().
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuBLAS reads this before its first CUDA operation. Never silently preserve
    # a conflicting value supplied by the shell or another entry point.
    required_workspace = policy["cublas_workspace_config"]
    actual_workspace = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", required_workspace)
    if actual_workspace != required_workspace:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the registered "
            f"determinism policy: {actual_workspace!r} != "
            f"{required_workspace!r}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = \
        policy["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
    torch.use_deterministic_algorithms(
        policy["torch_deterministic_algorithms"],
        warn_only=policy["torch_deterministic_warn_only"],
    )

    actual_numeric_mode = {
        "cuda_matmul_allow_tf32":
            bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision":
            torch.get_float32_matmul_precision(),
    }
    expected_numeric_mode = {
        key: policy[key]
        for key in (
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "float32_matmul_precision",
        )
    }
    if actual_numeric_mode != expected_numeric_mode:
        raise RuntimeError(
            "numeric execution mode differs from determinism policy: "
            f"{actual_numeric_mode!r} != {expected_numeric_mode!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Canonical save-location naming
# ──────────────────────────────────────────────────────────────────────────────

def define_save_locations(
    name:          str,
    dataset:       str,
    DOFs:          list[int],
    discretization: int = 1,
) -> tuple[str, str, str]:
    """
    Build the three canonical path strings for one ablation phase.

    The suffix encodes every axis of variation so that phases with different
    datasets, DOF subsets, or discretisation steps never share files:

        n  - dataset contains 'noise'
        t  - dataset contains 'temperature'
        v  - dataset contains 'vehicle'
        s  - dataset contains 'speed'
        _DOF_<d0>_<d1>_...  - active DOF indices
        _disc<k>           - discretisation step

    Example:
        define_save_locations('architectures', 'data_noise_vehicle_temperature',
                              [0, 1, 6], discretization=5)
        ->  database_name = "sqlite:///database/ttbi_architectures_ablation_ntv_DOF_0_1_6_disc5.db"
           output_dir    = "results/results_architectures_ntv_DOF_0_1_6_disc5"
           cache_dir     = "cache/data_cache_ntv_DOF_0_1_6_disc5"

    Args:
        name           (str):       Phase label, e.g. 'architectures',
                                    'DOFs_sensitivity', 'ForwardSweep_3Sensors'.
        dataset        (str):       Dataset folder name (used to build the suffix).
        DOFs           (list[int]): Active DOF indices for this phase.
        discretization (int):       Damage discretisation step (default 1 -> 61 classes).

    Returns:
        database_name (str): SQLite URL for optuna.create_study(storage=...).
        output_dir    (str): Root directory for model weights, plots, and JSON files.
        cache_dir     (str): Directory for preprocessed .npy and scaler files.
    """
    tag = ''
    if 'noise'       in dataset: tag += 'n'
    if 'temperature' in dataset: tag += 't'
    if 'vehicle'     in dataset: tag += 'v'
    if 'speed'       in dataset: tag += 's'
    if not tag:
        tag = 'all'

    if DOFs:
        tag += '_DOF' + ''.join(f'_{d}' for d in DOFs)

    tag += f'_disc{discretization}'

    database_name = f"sqlite:///database/{name}_{tag}.db"
    output_dir    = f"results/{name}_{tag}"
    cache_dir     = f"cache/{name}_{tag}"

    return database_name, output_dir, cache_dir


# ──────────────────────────────────────────────────────────────────────────────
# 3. DOF definitions - single source of truth
# ──────────────────────────────────────────────────────────────────────────────

# Ordered legacy schema identifiers: index == channel integer used throughout
# the codebase. Do not rename Wheel1_Vert/Wheel2_Vert in stored results.
DOF_NAMES: list[str] = [
    "CarBody_Vert",      # 0
    "FrontBogie_Vert",   # 1
    "RearBogie_Vert",    # 2
    "Wheel1_Vert",       # 3
    "Wheel2_Vert",       # 4
    "CarBody_Pitch",     # 5
    "FrontBogie_Pitch",  # 6
    "RearBogie_Pitch",   # 7
]

DOF_DESCRIPTIONS: list[str] = [
    "car-body vertical acceleration",
    "front-bogie vertical acceleration",
    "rear-bogie vertical acceleration",
    ("idealized constrained-wheelset vertical acceleration at wheel 1 "
     "(axle-box response proxy; frozen key Wheel1_Vert)"),
    ("idealized constrained-wheelset vertical acceleration at wheel 2 "
     "(axle-box response proxy; frozen key Wheel2_Vert)"),
    "car-body pitch angular velocity",
    "front-bogie pitch angular velocity",
    "rear-bogie pitch angular velocity",
]

# Convenience look-ups derived from DOF_NAMES - do not edit these directly.
DOF_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(DOF_NAMES)}
IDX_TO_DOF_NAME: dict[int, str] = {i: name for i, name in enumerate(DOF_NAMES)}


def dof_label(dof: int) -> str:
    """
    Human-readable label for a DOF index, used in plot titles and filenames.

    Converts underscores to spaces and formats direction as a parenthetical:
        6  ->  "Front Bogie (Pitch)"
        0  ->  "Car Body (Vert)"

    Args:
        dof (int): DOF index in [0, 7].

    Returns:
        str: Formatted label.

    Raises:
        KeyError: If dof is not in [0, 7].
    """
    if dof in (3, 4):
        return DOF_DESCRIPTIONS[dof]

    raw   = IDX_TO_DOF_NAME[dof]           # e.g. "FrontBogie_Pitch"
    parts = raw.split('_')                  # ["FrontBogie", "Pitch"]

    # Split camel-case component name into words
    import re
    component = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', parts[0])  # "Front Bogie"
    direction = parts[1]                                         # "Pitch"

    return f"{component} ({direction})"
