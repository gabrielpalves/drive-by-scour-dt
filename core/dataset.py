"""
core/dataset.py
===============
Raw data loading, processed-data caching, and the PyTorch Dataset wrapper
for memory-mapped arrays.

Imported by:
    training/trainer.py    — DataLoader construction inside train_and_evaluate
    training/robustness.py — same, inside run_single_training
    training/pipeline.py   — plot_cached_confusion_matrix, export_digital_twin_package

The digital twin does NOT import this module; it drives the physics engine
directly via digital_twin/physics.py rather than reading pre-recorded .mat files.
"""

import os
import pickle

import joblib
import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split

from core.preprocessing import TTBIPreprocessor


# ──────────────────────────────────────────────────────────────────────────────
# 1. Raw .mat loader
# ──────────────────────────────────────────────────────────────────────────────

def load_ttbi_dataset(
    filepath:        str,
    requested_dofs:  list[int],
    n_passages:      int = 200,
    target_supports: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw TTBI vibration passages from a folder of numbered .mat files
    and return them as NumPy arrays ready for preprocessing.

    Two labelling modes
    -------------------
    * Single-output (default, target_supports=None) — LEGACY. Each file
      (0001.mat … 0061.mat) is one damage level and the label is the FILE INDEX
      (0–60 %). y has shape (N,), int. Used by the single-scour ablation.
    * Multi-output (target_supports given) — STAGE 0+. Each file holds an
      independent per-support scour state; the label is the scour VECTOR at the
      requested support indices, read from data.scour_vector (regression target,
      % scour). y has shape (N, len(target_supports)), float. All NNNN.mat in the
      folder are scanned (not capped at 61). `target_supports` are 1-based MATLAB
      support indices (matching A00's scour_supports), e.g. [2, 3] for the two
      internal piers of a 3-span bridge.

    Up to n_passages passages are loaded per file.

    DOF mapping
    -----------
    The requested_dofs list selects which physical channels to extract.
    Valid indices and their physical meaning:

        0  CarBody_Vert      ← AcelPrimVag[0]
        1  FrontBogie_Vert   ← AcelPrimVag[1]
        2  RearBogie_Vert    ← AcelPrimVag[2]
        3  Wheel1_Vert       ← AcelRodaPrimVag[0]
        4  Wheel2_Vert       ← AcelRodaPrimVag[1]
        5  CarBody_Pitch      ← PitchPrimVag[0]
        6  FrontBogie_Pitch   ← PitchPrimVag[1]
        7  RearBogie_Pitch    ← PitchPrimVag[2]

    Args:
        filepath       (str):       Sub-folder name inside 'data/'.
        requested_dofs (list[int]): Ordered list of DOF indices to extract.
        n_passages     (int):       Maximum passages to load per damage file.
                                    Capped to however many the file actually has.

    Returns:
        X (np.ndarray): float32, shape (N, len(requested_dofs), sequence_length)
        y (np.ndarray): int64,   shape (N,) — damage label in [0, 60].

    Raises:
        FileNotFoundError: If the dataset folder does not exist.
    """
    dataset_path = os.path.join('data', filepath)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    # DOF index → (field_name, row_index) inside the MATLAB struct
    _DOF_SOURCE = {
        0: ('AcelPrimVag',     0),
        1: ('AcelPrimVag',     1),
        2: ('AcelPrimVag',     2),
        3: ('AcelRodaPrimVag', 0),
        4: ('AcelRodaPrimVag', 1),
        5: ('PitchPrimVag',    0),
        6: ('PitchPrimVag',    1),
        7: ('PitchPrimVag',    2),
    }

    # ── Multi-output mode: per-pier scour vector labels ───────────────────────
    if target_supports is not None:
        return _load_multi_output(dataset_path, requested_dofs, n_passages,
                                  target_supports, _DOF_SOURCE)

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    for damage_label in range(61):
        filename = f"{damage_label + 1:04d}.mat"
        filepath_ = os.path.join(dataset_path, filename)

        try:
            mat         = sio.loadmat(filepath_)
            data_struct = mat['data'][0, 0]

            available   = data_struct['AcelPrimVag'].shape[1]
            to_load     = min(n_passages, available)

            for p in range(to_load):
                channels = []
                for dof in requested_dofs:
                    field, row = _DOF_SOURCE[dof]
                    channels.append(data_struct[field][0, p][row, :])

                X_list.append(np.vstack(channels))   # (C, L)
                y_list.append(damage_label)

        except FileNotFoundError:
            print(f"  [!] Missing file: {filename}")
        except KeyError as e:
            print(f"  [!] Field not found in {filename}: {e}")
        except Exception as e:
            print(f"  [!] Error processing {filename}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)
    return X, y


def _load_multi_output(
    dataset_path:    str,
    requested_dofs:  list[int],
    n_passages:      int,
    target_supports: list[int],
    dof_source:      dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-output loader: scan all NNNN.mat, read the per-support scour VECTOR.

    Each file's label is data.scour_vector at the (1-based) `target_supports`
    indices, in % scour (the MATLAB value is a 0–1 fraction). Returns
    X (N, C, L) float32 and y (N, n_targets) float32 — a regression target.
    """
    tgt0 = [int(s) - 1 for s in target_supports]      # 1-based MATLAB -> 0-based
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    idx = 0
    while True:
        fname = f"{idx + 1:04d}.mat"
        fp = os.path.join(dataset_path, fname)
        if not os.path.exists(fp):
            break
        try:
            data_struct = sio.loadmat(fp)['data'][0, 0]
            names = data_struct.dtype.names or ()
            if 'scour_vector' not in names:
                raise KeyError(
                    f"{fname}: multi-output load needs data.scour_vector — "
                    f"regenerate the dataset with A00 damage_mode='multi_scour'.")
            vec = np.ravel(data_struct['scour_vector']).astype(float)   # per-support fractions
            label = vec[tgt0] * 100.0                                    # % at the targets

            available = data_struct['AcelPrimVag'].shape[1]
            for p in range(min(n_passages, available)):
                channels = [data_struct[dof_source[dof][0]][0, p][dof_source[dof][1], :]
                            for dof in requested_dofs]
                X_list.append(np.vstack(channels))                       # (C, L)
                y_list.append(label.astype(np.float32))
        except KeyError:
            raise
        except Exception as e:
            print(f"  [!] Error processing {fname}: {e}")
        idx += 1

    if not X_list:
        raise RuntimeError(f"No multi-output passages loaded from {dataset_path}")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)                              # (N, n_targets)
    print(f"  [multi-output] {X.shape[0]} passages, {idx} states, "
          f"{y.shape[1]} targets (supports {target_supports}).")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 2. Memory-mapped PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MemmapDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that reads samples on-demand from a memory-mapped NumPy
    array, keeping RAM usage constant regardless of dataset size.

    The indirection through `indices` lets the same on-disk array serve
    both the training and validation DataLoaders without duplication.

    Args:
        X_memmap (np.ndarray): Memory-mapped feature array, shape (N, C, L) or
                               (N, C, Sc, L) for CWT data.
        y_memmap (np.ndarray): Memory-mapped label array, shape (N,).
        indices  (np.ndarray): Integer indices selecting the partition
                               (e.g. train_idx or val_idx from train_test_split).
    """

    def __init__(
        self,
        X_memmap: np.ndarray,
        y_memmap: np.ndarray,
        indices:  np.ndarray,
        label_dtype: torch.dtype = torch.long,
    ):
        self.X           = X_memmap
        self.y           = y_memmap
        self.indices     = indices
        # long for classification (class index), float for regression (continuous
        # per-pier scour vector). See core.task.label_dtype.
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        # .copy() releases the memmap lock so the GC can reclaim the page
        x = torch.tensor(self.X[real_idx].copy()).float()
        y = torch.tensor(self.y[real_idx].copy()).to(self.label_dtype)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cache_stem(dataset_name: str, config: dict) -> str:
    """
    Build the base filename fragment that uniquely identifies a cache file.

    Format:  <dataset>_<method>_dofs_<d0>_<d1>_…_disc<k>

    Kept private; the three public filenames (features, labels, scaler) are
    constructed in get_or_create_cache by appending the appropriate suffix.
    """
    clean   = os.path.splitext(os.path.basename(dataset_name))[0]
    dof_str = "_".join(map(str, config['dofs']))
    disc    = config.get('discretization', 1)
    stem    = f"{clean}_{config['method']}_dofs_{dof_str}_disc{disc}"
    # Regression labels differ from classification (continuous per-pier vector),
    # so give them a distinct cache; classification stems stay byte-identical.
    if config.get('task') == 'regression':
        tgt = "_".join(map(str, config.get('target_supports', [])))
        stem += f"_reg_t{tgt}"
    return stem


def get_or_create_cache(
    config:       dict,
    dataset_name: str,
    cache_dir:    str,
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Return processed data as memory-mapped arrays, creating the cache on the
    first call and reading it on every subsequent call.

    Cache layout (inside cache_dir)
    --------------------------------
        cache_<stem>.npy        — processed feature array
        cache_<stem>_labels.npy — discretised label array
        scaler_<stem>.pkl       — fitted sklearn scaler  (or .pt for PyTorch)

    Leak-free contract
    ------------------
    The scaler is fitted on the canonical training partition (seed 42,
    test_size=0.20) derived from the *full* dataset, so the same indices
    are used whether the cache is being created or already exists.  This
    means you can safely call get_or_create_cache from any downstream
    function without re-fitting risk.

    Args:
        config       (dict): Ablation step config — must contain 'method',
                             'dofs', and optionally 'discretization'.
        dataset_name (str):  Name of the sub-folder inside 'data/'.
        cache_dir    (str):  Directory where cache files are stored.

    Returns:
        X_processed (np.ndarray): Memory-mapped feature array.
        y           (np.ndarray): Memory-mapped discretised label array.
        scaler      (object):     Fitted sklearn scaler, or None if the
                                  preprocessor has no scaler (future methods).
    """
    os.makedirs(cache_dir, exist_ok=True)
    stem = _cache_stem(dataset_name, config)

    feat_path   = os.path.join(cache_dir, f"cache_{stem}.npy")
    label_path  = os.path.join(cache_dir, f"cache_{stem}_labels.npy")
    scaler_path = os.path.join(cache_dir, f"scaler_{stem}.pkl")
    scaler_path_pt = scaler_path.replace('.pkl', '.pt')

    # ── Fast path: cache exists ───────────────────────────────────────────────
    cache_ready = os.path.exists(feat_path) and os.path.exists(label_path)
    scaler_ready = os.path.exists(scaler_path) or os.path.exists(scaler_path_pt)

    if cache_ready and scaler_ready:
        X_processed = np.load(feat_path,  mmap_mode='r')
        y           = np.load(label_path, mmap_mode='r')
        scaler      = _load_scaler(scaler_path, scaler_path_pt)
        return X_processed, y, scaler

    # ── Slow path: cache miss — process and save ──────────────────────────────
    dof_str = "_".join(map(str, config['dofs']))
    regression = config.get('task') == 'regression'
    print(f"  [CACHE MISS] Processing '{config['method']}' data (DOFs: {dof_str}"
          f"{', regression' if regression else ''})...")

    X_raw, y_raw = load_ttbi_dataset(
        filepath=dataset_name,
        requested_dofs=config['dofs'],
        n_passages=200,
        # Regression reads the per-pier scour VECTOR at the target supports;
        # classification (target_supports=None) keeps the file-index class label.
        target_supports=config.get('target_supports') if regression else None,
    )

    # Canonical train partition for leak-free scaler fitting (seed fixed at 42)
    all_indices          = np.arange(len(y_raw))
    canonical_train_idx, _ = train_test_split(
        all_indices, test_size=0.20, random_state=42
    )

    preprocessor = TTBIPreprocessor(method=config['method'], n_segments=512)
    X_processed  = preprocessor.transform(
        X_raw,
        fit_scaler=True,
        fit_indices=canonical_train_idx,
    )

    if regression:
        # Continuous per-pier scour targets (%), shape (N, n_targets) — no
        # discretisation. The model regresses these directly (MSE loss).
        y_out = y_raw.astype(np.float32)
    else:
        # Discretise labels: damage 0–60 → class 0–(60/disc)
        disc  = config.get('discretization', 1)
        y_out = np.round(y_raw / disc).astype(int)

    # Persist feature and label arrays
    np.save(feat_path,  X_processed)
    np.save(label_path, y_out)

    # Persist scaler
    _save_scaler(preprocessor.scaler, scaler_path)

    print(f"  [CACHE SAVED] → {cache_dir}")

    # Reload in memory-mapped mode to match the fast-path return type
    X_processed = np.load(feat_path,  mmap_mode='r')
    y           = np.load(label_path, mmap_mode='r')
    scaler      = preprocessor.scaler

    return X_processed, y, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Internal scaler I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_scaler(scaler: object, pkl_path: str) -> None:
    """Save a scaler, routing to torch.save for nn.Module scalers."""
    if isinstance(scaler, torch.nn.Module) or torch.is_tensor(scaler):
        pt_path = pkl_path.replace('.pkl', '.pt')
        torch.save(scaler, pt_path)
        print(f"  [Save] PyTorch scaler → {pt_path}")
    elif scaler is None:
        # Write a sentinel so the fast-path existence check still passes
        with open(pkl_path, 'w') as f:
            f.write("NO_SCALER_USED")
        print("  [Warning] No scaler produced by this preprocessor.")
    else:
        joblib.dump(scaler, pkl_path)
        print(f"  [Save] sklearn scaler → {pkl_path}")


def _load_scaler(pkl_path: str, pt_path: str) -> object:
    """Load a scaler from whichever file format exists."""
    if os.path.exists(pkl_path):
        content = open(pkl_path).read(20) if os.path.getsize(pkl_path) < 100 else None
        if content and content.strip() == "NO_SCALER_USED":
            return None
        return joblib.load(pkl_path)
    if os.path.exists(pt_path):
        return torch.load(pt_path)
    return None