"""
core/utils.py
=============
Three unrelated but universally needed utilities:

    1. set_global_seed      — reproducibility across Python / NumPy / PyTorch.
    2. define_save_locations — canonical naming for Optuna DBs and output dirs.
    3. DOF definitions      — single source of truth for channel names and indices.

Imported by:
    ablation.ipynb          — seed, save locations, DOF maps
    training/trainer.py     — set_global_seed at the top of every trial
    training/pipeline.py    — set_global_seed, define_save_locations
    digital_twin/assets.py  — DOF maps (channel selection at inference time)

No ML framework is imported here beyond torch (for seed control).
No other core module is imported here — this file has no internal deps.
"""

import os
import random

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """
    Lock all random-number generators to `seed` for full reproducibility.

    Covers Python stdlib, NumPy, PyTorch CPU, and every CUDA device.
    Also forces deterministic cuDNN kernels and disables its auto-tuner —
    this trades a small throughput cost for bitwise-identical runs.

    Call once at the top of the ablation notebook and once at the start of
    every Optuna trial (inside train_and_evaluate) to guard against state
    accumulated between trials.

    Args:
        seed (int): Any non-negative integer. Default 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # multi-GPU setups

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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

        n  — dataset contains 'noise'
        t  — dataset contains 'temperature'
        v  — dataset contains 'vehicle'
        s  — dataset contains 'speed'
        _DOF_<d0>_<d1>_…  — active DOF indices
        _disc<k>           — discretisation step

    Example:
        define_save_locations('architectures', 'data_noise_vehicle_temperature',
                              [0, 1, 6], discretization=5)
        →  database_name = "sqlite:///database/ttbi_architectures_ablation_ntv_DOF_0_1_6_disc5.db"
           output_dir    = "results/results_architectures_ntv_DOF_0_1_6_disc5"
           cache_dir     = "cache/data_cache_ntv_DOF_0_1_6_disc5"

    Args:
        name           (str):       Phase label, e.g. 'architectures',
                                    'DOFs_sensitivity', 'ForwardSweep_3Sensors'.
        dataset        (str):       Dataset folder name (used to build the suffix).
        DOFs           (list[int]): Active DOF indices for this phase.
        discretization (int):       Damage discretisation step (default 1 → 61 classes).

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
# 3. DOF definitions — single source of truth
# ──────────────────────────────────────────────────────────────────────────────

# Ordered list: index == DOF integer used throughout the codebase.
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

# Convenience look-ups derived from DOF_NAMES — do not edit these directly.
DOF_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(DOF_NAMES)}
IDX_TO_DOF_NAME: dict[int, str] = {i: name for i, name in enumerate(DOF_NAMES)}


def dof_label(dof: int) -> str:
    """
    Human-readable label for a DOF index, used in plot titles and filenames.

    Converts underscores to spaces and formats direction as a parenthetical:
        6  →  "Front Bogie (Pitch)"
        0  →  "Car Body (Vert)"

    Args:
        dof (int): DOF index in [0, 7].

    Returns:
        str: Formatted label.

    Raises:
        KeyError: If dof is not in [0, 7].
    """
    raw   = IDX_TO_DOF_NAME[dof]           # e.g. "FrontBogie_Pitch"
    parts = raw.split('_')                  # ["FrontBogie", "Pitch"]

    # Split camel-case component name into words
    import re
    component = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', parts[0])  # "Front Bogie"
    direction = parts[1]                                         # "Pitch"

    return f"{component} ({direction})"
