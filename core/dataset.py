"""
core/dataset.py
===============
Raw data loading, processed-data caching, and the PyTorch Dataset wrapper
for memory-mapped arrays.

Imported by:
    training/trainer.py    - DataLoader construction inside train_and_evaluate
    training/robustness.py - same, inside run_single_training
    training/pipeline.py   - plot_cached_confusion_matrix, export_digital_twin_package

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
    bearing_targets: list | None = None,
    bearing_max:     float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw TTBI vibration passages from a folder of numbered .mat files
    and return them as NumPy arrays ready for preprocessing.

    Two labelling modes
    -------------------
    * Single-output (default, target_supports=None) - LEGACY. Each file
      (0001.mat ... 0061.mat) is one damage level and the label is the FILE INDEX
      (0-60 %). y has shape (N,), int. Used by the single-scour ablation.
    * Multi-output (target_supports given) - STAGE 0+. Each file holds an
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

        0  CarBody_Vert      <- AcelPrimVag[0]
        1  FrontBogie_Vert   <- AcelPrimVag[1]
        2  RearBogie_Vert    <- AcelPrimVag[2]
        3  Wheel1_Vert       <- AcelRodaPrimVag[0]
        4  Wheel2_Vert       <- AcelRodaPrimVag[1]
        5  CarBody_Pitch      <- PitchPrimVag[0]
        6  FrontBogie_Pitch   <- PitchPrimVag[1]
        7  RearBogie_Pitch    <- PitchPrimVag[2]

    Args:
        filepath       (str):       Sub-folder name inside 'data/'.
        requested_dofs (list[int]): Ordered list of DOF indices to extract.
        n_passages     (int):       Maximum passages to load per damage file.
                                    Capped to however many the file actually has.

    Returns:
        X (np.ndarray): float32, shape (N, len(requested_dofs), sequence_length)
        y (np.ndarray): int64,   shape (N,) - damage label in [0, 60].

    Raises:
        FileNotFoundError: If the dataset folder does not exist.
    """
    dataset_path = os.path.join('data', filepath)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    # DOF index -> (field_name, row_index) inside the MATLAB struct
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
                                  target_supports, _DOF_SOURCE,
                                  bearing_targets=bearing_targets,
                                  bearing_max=bearing_max)

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
    bearing_targets: list | None = None,
    bearing_max:     float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-output loader: scan all NNNN.mat, read the scour (+ bearing) VECTORS.

    Label layout: [scour% at target_supports]  (Stage 0), or, when
    `bearing_targets` is given (Stage 1), followed by the bearing heads:
        [scour_1..scour_S, bearing_1..bearing_B]
    * Scour   = data.scour_vector at the (1-based) `target_supports`, x100 (%).
    * Bearing = data.bearing_vector at the requested targets ('left'->0,
      'right'->1), normalised by `bearing_max` (the seized stiffness, Nm/rad)
      x100 -> a "seized %" on the SAME 0-100 scale as scour, so the MSE loss
      balances the heads instead of being swamped by the 1e9-scale stiffness.
      `bearing_max` defaults to the dataset manifest (case_info.mat /
      damage_states.mat); if absent, the observed max is used (with a warning).

    Returns X (N, C, L) float32 and y (N, n_scour[+n_bearing]) float32.
    """
    tgt0 = [int(s) - 1 for s in target_supports]      # 1-based MATLAB -> 0-based

    bidx = None
    if bearing_targets:
        _name = {'left': 0, 'l': 0, '0': 0, 'right': 1, 'r': 1, '1': 1}
        bidx = []
        for b in bearing_targets:
            k = _name.get(str(b).lower())
            if k is None:
                raise ValueError("bearing_targets entries must be 'left'/'right' "
                                 f"(or 0/1), got {b!r}")
            bidx.append(k)
        if bearing_max is None:
            bearing_max = _read_bearing_max(dataset_path)

    X_list:  list[np.ndarray] = []
    y_scour: list[np.ndarray] = []
    y_bear:  list[np.ndarray] = []                      # raw Nm/rad, normalised later

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
                    f"{fname}: multi-output load needs data.scour_vector - "
                    f"regenerate the dataset with A00 damage_mode='multi_scour'.")
            slabel = np.ravel(data_struct['scour_vector']).astype(float)[tgt0] * 100.0
            if bidx is not None:
                if 'bearing_vector' not in names:
                    raise KeyError(
                        f"{fname}: bearing heads requested but no "
                        f"data.bearing_vector - regenerate with A00 "
                        f"STAGE='stage1_bearing' (bearing_mode='target').")
                bvec = np.ravel(data_struct['bearing_vector']).astype(float)[bidx]

            available = data_struct['AcelPrimVag'].shape[1]
            for p in range(min(n_passages, available)):
                channels = [data_struct[dof_source[dof][0]][0, p][dof_source[dof][1], :]
                            for dof in requested_dofs]
                X_list.append(np.vstack(channels))                       # (C, L)
                y_scour.append(slabel.astype(np.float32))
                if bidx is not None:
                    y_bear.append(bvec.astype(np.float32))
        except KeyError:
            raise
        except Exception as e:
            print(f"  [!] Error processing {fname}: {e}")
        idx += 1

    if not X_list:
        raise RuntimeError(f"No multi-output passages loaded from {dataset_path}")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_scour, dtype=np.float32)                              # (N, n_scour)

    if bidx is not None:
        B = np.array(y_bear, dtype=np.float32)                          # (N, n_bearing) raw
        if bearing_max is None or bearing_max <= 0:
            bearing_max = float(B.max()) or 1.0
            print(f"  [multi-output] bearing_max not in manifest - normalising "
                  f"by observed max {bearing_max:.3g} Nm/rad.")
        y = np.hstack([y, (B / bearing_max) * 100.0]).astype(np.float32)

    extra = f" + {len(bidx)} bearing" if bidx else ""
    print(f"  [multi-output] {X.shape[0]} passages, {idx} states, "
          f"{y.shape[1]} heads ({len(tgt0)} scour{extra}).")
    return X, y


def _read_bearing_max(dataset_path: str) -> float | None:
    """Bearing normalisation constant [Nm/rad] from the dataset manifest.

    Prefers case_info.mat (bearing_max_Nm_rad written by A00), then the max of
    damage_states.mat::BearingStates. Returns None when neither is present, so
    the caller falls back to the observed max.
    """
    ci = os.path.join(dataset_path, 'case_info.mat')
    if os.path.exists(ci):
        try:
            info = sio.loadmat(ci)['case_info'][0, 0]
            if 'bearing_max_Nm_rad' in (info.dtype.names or ()):
                v = float(np.ravel(info['bearing_max_Nm_rad'])[0])
                if v > 0:
                    return v
        except Exception:
            pass
    ds = os.path.join(dataset_path, 'damage_states.mat')
    if os.path.exists(ds):
        try:
            bs = sio.loadmat(ds).get('BearingStates')
            if bs is not None and np.size(bs) and float(np.max(bs)) > 0:
                return float(np.max(bs))
        except Exception:
            pass
    return None


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

    Format:  <dataset>_<method>_dofs_<d0>_<d1>_..._disc<k>

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
        bt = config.get('bearing_targets')
        if bt:   # Stage 1 bearing heads -> distinct cache from the scour-only one
            stem += "_b" + "_".join(str(b) for b in bt)
    sn = config.get('sensor_noise')
    if sn:   # load-time noise injection -> its own cache, never collides with clean
        stem += f"_noise-{sn['mode']}" + (f"-{sn['desvio']}" if 'desvio' in sn else "")
    return stem


def _inject_sensor_noise(X: np.ndarray, dofs: list[int], sn: dict) -> np.ndarray:
    """
    Load-time measurement-noise injection (noise policy 2026-07-12).

    Generation is noise-free from stage1_crack onward (A00 use_signal_noise =
    false -> D01 adds nothing), so any noise a study needs is injected HERE,
    where the model stays configurable per experiment and per channel — sensor
    grade depends on the mounting position (EN 61373 vibration severity:
    carbody < bogie < axle; see papers/'Confiabilidade Sensores MEMS
    Ferroviários'). Deterministic (fixed RNG seed 42), so a cache rebuild
    reproduces identical features; the cache stem carries a noise tag so noisy
    and clean caches never collide.

    Modes:
      {'mode': 'legacy_wheel', 'desvio': 0.05}
          The legacy MATLAB D01 model: multiplicative gaussian
          (std = desvio·|signal|) on the WHEEL channels only (global DOFs 3,4).
          Reproduces the Stage-0/1 training distribution on noise-free data.
    Per-channel additive noise-floor modes: add here when the noise-robustness
    arm lands (anchor levels to the rail-qualified IMU datasheets in papers/).
    """
    rng = np.random.default_rng(42)
    X = np.array(X, dtype=np.float32, copy=True)
    if sn['mode'] == 'legacy_wheel':
        desvio = float(sn.get('desvio', 0.05))
        for i, d in enumerate(dofs):
            if d in (3, 4):   # wheel channels only
                X[:, i, :] += (desvio * X[:, i, :] *
                               rng.standard_normal(X[:, i, :].shape)
                               .astype(np.float32))
    else:
        raise ValueError(f"unknown sensor_noise mode {sn['mode']!r}")
    return X


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
        cache_<stem>.npy        - processed feature array
        cache_<stem>_labels.npy - discretised label array
        scaler_<stem>.pkl       - fitted sklearn scaler  (or .pt for PyTorch)

    Leak-free contract
    ------------------
    The scaler is fitted on the canonical training partition (seed 42,
    test_size=0.20) derived from the *full* dataset, so the same indices
    are used whether the cache is being created or already exists.  This
    means you can safely call get_or_create_cache from any downstream
    function without re-fitting risk.

    Args:
        config       (dict): Ablation step config - must contain 'method',
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

    # ── Slow path: cache miss - process and save ──────────────────────────────
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
        # Stage 1: also read bearing_vector as extra heads (None -> scour only).
        bearing_targets=config.get('bearing_targets') if regression else None,
        bearing_max=config.get('bearing_max') if regression else None,
    )

    # Optional load-time sensor noise (config['sensor_noise']; None = the
    # noise-free chain default). Applied to the RAW signals, before
    # preprocessing/PAA, exactly where MATLAB D01 used to apply it.
    if config.get('sensor_noise'):
        X_raw = _inject_sensor_noise(X_raw, config['dofs'], config['sensor_noise'])

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
        # Continuous per-pier scour targets (%), shape (N, n_targets) - no
        # discretisation. The model regresses these directly (MSE loss).
        y_out = y_raw.astype(np.float32)
    else:
        # Discretise labels: damage 0-60 -> class 0-(60/disc)
        disc  = config.get('discretization', 1)
        y_out = np.round(y_raw / disc).astype(int)

    # Persist feature and label arrays
    np.save(feat_path,  X_processed)
    np.save(label_path, y_out)

    # Persist scaler
    _save_scaler(preprocessor.scaler, scaler_path)

    print(f"  [CACHE SAVED] -> {cache_dir}")

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
        print(f"  [Save] PyTorch scaler -> {pt_path}")
    elif scaler is None:
        # Write a sentinel so the fast-path existence check still passes
        with open(pkl_path, 'w') as f:
            f.write("NO_SCALER_USED")
        print("  [Warning] No scaler produced by this preprocessor.")
    else:
        joblib.dump(scaler, pkl_path)
        print(f"  [Save] sklearn scaler -> {pkl_path}")


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