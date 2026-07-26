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

import hashlib
import hmac
import json
import os
import pickle
import re
import threading
import time

import joblib
import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's bytes (audit R7 P3: cache-artifact digest)."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


# Full source-byte verification is expensive for multi-GB datasets, but it must
# happen before a cached feature array can certify those source bytes.  Cache a
# successful pass per process and invalidate it whenever any recorded file's
# size/mtime/ctime changes.  A lock prevents the many sensor configurations
# launched by one process from hashing the same dataset concurrently.
_SOURCE_VERIFY_LOCK = threading.Lock()
_SOURCE_VERIFY_CACHE: dict[str, tuple[str, tuple]] = {}


def _unique_tmp(path: str) -> str:
    """A PROCESS-UNIQUE temp name for `path` (audit R7.1 P3): two processes
    building the same cache stem concurrently must not collide on one '.tmp' name
    (that caused PermissionError on Windows). Each writes its own temp, then the
    atomic os.replace makes last-writer-wins — both produce identical bytes."""
    return f"{path}.{os.getpid()}.tmp"


def _atomic_np_save(path: str, arr: np.ndarray) -> None:
    """np.save to a process-unique temp file in the same dir, then atomically
    replace (audit R7 P3). A crash mid-write leaves the .tmp, never a partial
    final cache file."""
    tmp = _unique_tmp(path)
    with open(tmp, 'wb') as fh:      # file handle -> np.save writes EXACTLY to tmp
        np.save(fh, arr)             # (no auto ".npy" suffix when given a handle)
    os.replace(tmp, path)            # atomic on the same volume (POSIX + Windows)


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON atomically (process-unique temp + os.replace) — audit R7 P3."""
    tmp = _unique_tmp(path)
    with open(tmp, 'w') as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


class _CacheStemLock:
    """Cross-PROCESS and cross-THREAD advisory lock for ONE cache stem (audit R7.1
    P5). Serialises concurrent BUILDS of the same cache so they cannot race on
    temp files or publish a half-written set. The lock is a file created with
    O_CREAT|O_EXCL (atomic on POSIX + Windows), so a second builder — thread OR
    process — blocks until the first releases. A lock left by a crashed builder is
    stolen after `stale_after` seconds so it can never deadlock forever."""

    def __init__(self, lock_path: str, timeout: float = 1800.0,
                 poll: float = 0.25, stale_after: float = 3600.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after
        self.fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:                                  # steal a stale (crashed) lock
                    if time.time() - os.path.getmtime(self.lock_path) > self.stale_after:
                        os.remove(self.lock_path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(
                        f"cache lock {self.lock_path} held > {self.timeout}s — a "
                        f"builder is stuck or a stale lock remains; remove it.")
                time.sleep(self.poll)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            os.remove(self.lock_path)
        except OSError:
            pass


def _assert_groups_canonical(groups: np.ndarray, n_states, npass) -> None:
    """Groups must be the EXACT canonical BLOCK vector [0]*npass + [1]*npass + ...
    (audit R7.1 P7). Checking only the ID set + counts is too weak: a count-
    preserving SWAP between two states (state A's passages labelled B and vice
    versa) passes that but breaks the state<->passage correspondence the leak-free
    split relies on. Requiring the exact contiguous block vector catches swaps,
    permutations, and interleaving. No-op when n_states/npass are unknown."""
    if n_states is None or not npass:
        return
    expected = np.repeat(np.arange(int(n_states), dtype=groups.dtype), int(npass))
    if groups.shape != expected.shape or not np.array_equal(groups, expected):
        raise RuntimeError(
            f"groups are not the canonical contiguous block vector "
            f"[0..{int(n_states) - 1}] x {npass} (len {len(groups)} vs "
            f"{len(expected)}) — corrupt/tampered/permuted grouping; delete the "
            f"cache dir and rebuild.")

from core.preprocessing import TTBIPreprocessor

# Contact-validity gate (audit R5 2026-07-17; recalibrated 2026-07-19 at first
# campaign dispatch; F-tier recalibrated AGAIN 2026-07-22 on the second observed
# event). The solver couples wheel and rail BILATERALLY, so brief wheel
# unloading past zero shows up as small spurious TENSION instead of a few-ms
# separation + re-contact. Two regimes, two-tier gate:
#   * TOLERATED (logged, reported): brief micro-unloading — peak tension a
#     small fraction of the ~118 kN static wheel load, on a tiny fraction of
#     the path. Observed events (both physically plausible superposition
#     tails; censoring/resampling them would MNAR-bias exactly the severe
#     cases the paper is about):
#       - s23_all4 state 24 (60% scour + FRA-4 + track damage + poly OOR):
#         6.4 kN (5.4% static) on 0.042% of samples.
#       - s15_track state 244 (50%/13% scour + track damage): 13.4 kN (11.4%
#         static) on 0.063% of samples — ONE sample at dt=1 ms, on the track
#         portion (off-bridge; the void rung is DESIGNED to hammer the contact
#         patch, so its unloading tail is naturally heavier).
#   * FATAL (physics regression): tension beyond 20% of static (24 kN, ~1.8x
#     the worst observed event) OR sustained tension (> 0.2% of path samples)
#     OR non-finite. The known true regressions sit far above: the R3
#     profile-seam bug gave 170 kN (144% of static); wheel flats exceed the
#     uplift threshold 12-38x. Watch item unchanged: an event of tens of kN
#     or sustained means the tail is heavier than believed -> revisit
#     (unilateral contact vs censor-with-report), do NOT raise again.
# The generator (A00 F_CONTACT_TOL_N / F_CONTACT_FRACTOL) enforces the SAME
# two-tier rule per passage at generation time; values must stay identical.
_CONTACT_F_TOL_N  = 24000.0  # FATAL above this peak tension [N] (20% static)
_CONTACT_FRAC_TOL = 0.002    # FATAL above this fraction of path samples in tension
_re_state = re.compile(r'\d{4}\.mat$')   # NNNN.mat state-file matcher

# Payload-validation tolerances (audit R7 P4).
_SCOUR_EPS = 1e-6           # float slack on the scour [0, dano_max%] range check
_CROP_RAGGED_TOL = 4        # max samples the per-passage crop may differ by
                            # (round() on DimSpace can vary the window by ~1-2);
                            # anything larger means a non-canonical spatial grid.

# Provenance: every multi-output .mat must carry this generator schema (written
# by A00) AND a gen_fingerprint matching its manifest. Loading a file without
# them, or with a different value, aborts — so pre-audit or R2/R3 data can never
# silently enter the pipeline. Must equal A00_Run.m's gen_schema exactly.
_EXPECTED_GEN_SCHEMA = 'audit-2026-07-25-r9'

# ── PROTOCOL CONSTANTS (unified protocol_hash, 2026-07-19) ────────────────────
# Single source of truth: the split/preprocessing code BELOW reads these same
# constants, and core/protocol.py folds them into the protocol hash. Changing
# any of them therefore (a) changes behaviour and (b) changes every study/
# manifest/summary name in lockstep — nothing stale can be resumed.
SPLIT_SEED      = 42     # canonical grouped-split RNG seed (all arms/seeds/PCs)
SPLIT_TEST_FRAC = 0.20   # outer-TEST fraction of STATES (reported metrics only)
SPLIT_VAL_FRAC  = 0.20   # inner-VAL fraction of STATES (HPO + selection)
N_SEGMENTS      = 512    # PAA segment count (TTBIPreprocessor)
NOISE_RNG_SEED  = 42     # load-time sensor-noise RNG (deterministic rebuilds)
LOAD_N_PASSAGES = 200    # passage cap requested from the loader (manifest npass
                         # is authoritative; the loader enforces exact ==)
CACHE_SCHEMA_TAG = "_gs6"  # cache-contract tag appended to every cache stem
                           # (unchanged at R9: feature transform is unchanged;
                           # source-schema/fingerprint sidecars orphan R8 data)


# ── Feature A (2026-07-19): state families + stratified grouped split ────────
# The generator (A00) tags every state with an explicit FAMILY identity and
# ships it in damage_states.mat; the split below stratifies on it so that every
# family — and every (family, target) anchor stratum — is guaranteed to land in
# train, inner-val AND outer-test. This replaces the random GroupShuffleSplit,
# whose anchor coverage was only probabilistic (audit R6 C8C had a loud WARNING
# for the empty-probe case; now it is a structural guarantee).
STATE_FAMILIES = ('target_healthy', 'scour_only', 'bearing_only',
                  'nuisance_only', 'joint')
# Deterministic per-stratum assignment pattern = EXACTLY 60/20/20. Order is
# deliberate: a 1-state stratum goes to TRAIN (a family the model never trained
# on must not appear only at test), a 2-state stratum spans train+test, 3+
# spans all three partitions.
STRATIFY_PATTERN = ('train', 'test', 'val', 'train', 'train')
N_JOINT_SEV_BINS = 3     # joint-block severity bins (max normalized head)


def split_protocol() -> dict:
    """The split policy as DATA, for the protocol hash (core/protocol.py).

    Every entry is a constant `canonical_grouped_splits` itself uses — this
    function must never describe anything the code does not read."""
    return {
        "splitter":   "family-STRATIFIED deterministic split, grouped by damage "
                      "STATE (Feature A 2026-07-19; replaces GroupShuffleSplit)",
        "layout":     "3-way train/inner-val/outer-test; inner-val = HPO+selection, "
                      "outer-test = reported metrics only",
        "seed":       SPLIT_SEED,
        "test_frac":  SPLIT_TEST_FRAC,
        "val_frac":   SPLIT_VAL_FRAC,
        "pattern":    list(STRATIFY_PATTERN),
        "strata":     "family | family+anchor_target | joint+crack_on+sev_bin",
        "n_joint_sev_bins": N_JOINT_SEV_BINS,
        "families":   list(STATE_FAMILIES),
    }


# Preprocessing policy as DATA, for the protocol hash. Same rule: only values
# the loading/preprocessing code actually reads (or gates on) appear here.
PREPROC_PROTOCOL = {
    "n_segments":         N_SEGMENTS,
    "scaler_fit":         "canonical grouped TRAIN partition only",
    "raw_to_space":       "interp1-mirror time->space + bridge crop (Option B)",
    "noise_rng_seed":     NOISE_RNG_SEED,
    "load_n_passages":    LOAD_N_PASSAGES,
    "contact_f_tol_N":    _CONTACT_F_TOL_N,
    "contact_frac_tol":   _CONTACT_FRAC_TOL,
    "crop_ragged_tol":    _CROP_RAGGED_TOL,
    # Audit r3 (2026-07-22): both fixes change feature bytes -> named here so
    # the protocol hash moves with them and can never silently regress.
    "paa_impl":           "keogh-window-mean-v2 (exact fractional windows)",
    "noise_pairing":      "per-global-dof rng [seed, dof] (subset-invariant)",
    # Spell out the generator instead of relying on default_rng's current
    # default. This pins cross-PC/cache-rebuild bytes across NumPy upgrades.
    "noise_bit_generator": "numpy.random.PCG64",
    "source_byte_verification":
        "SHA-256 every file_digests entry before protocol/cache reuse; "
        "memoized only while size+mtime+ctime are unchanged; always hash "
        "case_info.mat and damage_states.mat into the protocol identity "
        "(including legacy state-only digest tables); re-read current identity "
        "at every cache entry and require equality with the study descriptor",
}


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        X      (np.ndarray): float32, shape (N, len(requested_dofs), sequence_length)
        y      (np.ndarray): int64,   shape (N,) - damage label in [0, 60].
        groups (np.ndarray): int64,   shape (N,) - source MAT-file index per
                             sample (= damage STATE id). All passages of one
                             file share one group; used for leak-free grouped
                             splitting (audit fix 2026-07-17).

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
    # Legacy single-scour mode: label == file index, so group == label. A
    # grouped split would hold out whole CLASSES here - callers must fall back
    # to the per-passage split for this mode (see canonical_train_val_split).
    return X, y, y.copy()


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

    Returns X (N, C, L) float32, y (N, n_scour[+n_bearing]) float32, and
    groups (N,) int64 = source MAT-file index per sample (one file = one
    damage STATE; all its passages share the group id). The groups vector is
    what makes a leak-free grouped train/val split possible (audit fix
    2026-07-17): passages of one state carry the same label AND the same
    persistent nuisance realization (crack, locked profile, track defects on
    the EOV rungs), so splitting them per-passage lets the network recognise
    the state instead of generalising.
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

    # ── Manifest-driven strict validation (audit R4/R5 2026-07-17) ───────────
    # The campaign is multi-day and distributed, so the loader must REJECT
    # partial, mixed-provenance, or corrupt datasets rather than train on them.
    # Everything below is keyed off the dataset's own case_info.mat manifest,
    # which is now MANDATORY, with an exact file inventory and a completion
    # marker written by A00 only when generation finished.
    exp_states, exp_npass, exp_schema, exp_fp = _read_manifest(dataset_path)
    if exp_states is None or exp_schema is None or exp_fp is None or exp_npass is None:
        raise RuntimeError(
            f"{dataset_path}: missing/incomplete case_info.mat manifest "
            f"(n_states/passages_per_state/gen_schema/gen_fingerprint required). "
            f"Regenerate with the current A00.")
    # passages_per_state must be a positive integer: a 0/negative count would let
    # the old `exp_npass if exp_npass else available` fallback silently substitute
    # the file's own passage count and disable the manifest-vs-file cross-check
    # (audit R6 C4). Npass is fixed for the whole campaign, so this is exact.
    if exp_npass <= 0 or exp_states <= 0:
        raise RuntimeError(
            f"{dataset_path}: manifest passages_per_state={exp_npass}, "
            f"n_states={exp_states} — both must be positive. Corrupt manifest.")
    if exp_schema != _EXPECTED_GEN_SCHEMA:
        raise RuntimeError(
            f"{dataset_path}: manifest gen_schema={exp_schema!r} != expected "
            f"{_EXPECTED_GEN_SCHEMA!r} — regenerate with the current A00.")
    # Completion marker must EXIST and its CONTENT must match the manifest (audit
    # R7 P3): A00 writes the schema + fingerprint into it, so a stale marker from a
    # different run (or a hand-written empty file) is rejected, not trusted.
    marker = os.path.join(dataset_path, '_GENERATION_COMPLETE')
    if not os.path.exists(marker):
        raise RuntimeError(
            f"{dataset_path}: no _GENERATION_COMPLETE marker — generation did "
            f"not finish (or was interrupted). Do not train on a partial run.")
    with open(marker) as _mfh:
        mk = [ln.strip() for ln in _mfh.read().splitlines() if ln.strip()]
    # marker = schema, fingerprint, ROOT source-digest (audit R7.1 P4).
    if mk[:2] != [exp_schema, exp_fp] or len(mk) < 3:
        raise RuntimeError(
            f"{dataset_path}: _GENERATION_COMPLETE content {mk[:3]} != manifest "
            f"[{exp_schema!r}, {exp_fp!r}, <root-digest>] — stale/foreign/old "
            f"marker (R7 needs the source root digest as line 3). Regenerate.")
    marker_root = mk[2]
    # Source content integrity (audit R7.1 P4): file_digests.mat is MANDATORY; its
    # root must equal the marker's, and each state file's bytes must hash to its
    # recorded SHA. This catches bit-flips / corruption / same-size overwrites
    # that the (name,size) inventory cannot.
    src_digests, src_root = _read_file_digests(dataset_path)
    if src_digests is None or src_root is None:
        raise RuntimeError(
            f"{dataset_path}: missing/invalid file_digests.mat — R7 requires "
            f"per-state SHA-256 digests. Regenerate with the current A00.")
    if src_root != marker_root or _root_digest(src_digests) != src_root:
        raise RuntimeError(
            f"{dataset_path}: source root digest mismatch (marker {marker_root[:12]}…, "
            f"file_digests {src_root[:12]}…, recomputed "
            f"{_root_digest(src_digests)[:12]}…) — tampered/corrupt digest set.")
    # Exact inventory: no missing AND no EXTRA numbered files beyond n_states.
    nnnn = [f for f in os.listdir(dataset_path) if _re_state.fullmatch(f)]
    if len(nnnn) != exp_states:
        raise RuntimeError(
            f"{dataset_path}: {len(nnnn)} NNNN.mat files but manifest says "
            f"n_states={exp_states} (extra or missing state files).")
    n_files = exp_states
    # Physical scour ceiling (audit R7.1 P5): dano_max is MANDATORY (no [0,100]
    # fallback) and must be a fraction in (0, 1]. The label then must lie in
    # [0, dano_max%].
    dano_max = _read_dano_max(dataset_path)
    if dano_max is None or not (0.0 < dano_max <= 1.0):
        raise RuntimeError(
            f"{dataset_path}: manifest scour_dano_max_frac={dano_max} missing or "
            f"outside (0, 1] — R7 requires a valid scour ceiling. Regenerate.")
    scour_max_pct = dano_max * 100.0
    # State-family table (Feature A, 2026-07-19): MANDATORY. Each file's own
    # state_family is cross-checked against the table row below, so a renamed
    # NNNN.mat carrying another state's payload is caught even when its
    # provenance strings are internally valid.
    state_table = read_state_table(dataset_path)
    if len(state_table['family']) != n_files:
        raise RuntimeError(
            f"{dataset_path}: state table has {len(state_table['family'])} rows "
            f"but the manifest says n_states={n_files} — mismatched artifacts.")

    X_list:  list[np.ndarray] = []
    y_scour: list[np.ndarray] = []
    y_bear:  list[np.ndarray] = []            # fixity ratio, or legacy raw Nm/rad
    g_list:  list[int]        = []            # source file index per sample
    n_seen = 0                               # passages loaded (contact gate is hard-fail)
    bearing_is_fixity = False
    # Tolerated micro-tension incidence across the dataset (2026-07-19):
    # reported in the load summary; the paper quotes it as the scope of the
    # bilateral-contact approximation on the severe-EOV rungs.
    n_tension_passages = 0
    worst_tension_N = 0.0
    worst_tension_frac = 0.0

    for idx in range(n_files):
        fname = f"{idx + 1:04d}.mat"
        fp = os.path.join(dataset_path, fname)
        # Contiguity: a gap means an incomplete/interrupted campaign — abort
        # instead of silently training on a truncated dataset.
        if not os.path.exists(fp):
            raise RuntimeError(
                f"{dataset_path}: expected {n_files} contiguous state files but "
                f"{fname} is MISSING. Dataset is incomplete — finish generation "
                f"(or fix the manifest n_states) before training.")
        # Content integrity (audit R7.1 P4): the file's BYTES must hash to the SHA
        # recorded at generation. Catches corruption / same-size overwrite before
        # we even parse it. (Slow-path only — the cache fast path never re-reads
        # source .mats.)
        if fname not in src_digests:
            raise RuntimeError(f"{fname}: not in file_digests.mat — foreign/extra file.")
        actual_sha = _sha256_file(fp)
        if actual_sha != src_digests[fname]:
            raise RuntimeError(
                f"{fname}: SHA-256 {actual_sha[:12]}… != recorded "
                f"{src_digests[fname][:12]}… — corrupt or overwritten state file.")
        # No blanket exception swallow (audit R4): any failure on a state file is
        # fatal — a corrupt/foreign file must stop the run, not be skipped.
        data_struct = sio.loadmat(fp)['data'][0, 0]
        names = data_struct.dtype.names or ()
        if 'scour_vector' not in names:
            raise KeyError(
                f"{fname}: multi-output load needs data.scour_vector - "
                f"regenerate the dataset with A00 damage_mode='multi_scour'.")
        # Provenance gate (audit R3/R4): every file must carry the current
        # generator schema AND a gen_fingerprint matching the manifest, so a
        # foreign or mixed-provenance file (e.g. R2/R3 wrap data copied in)
        # fails fast instead of training silently.
        fsch = (str(np.ravel(data_struct['gen_schema'])[0])
                if 'gen_schema' in names else None)
        if fsch != _EXPECTED_GEN_SCHEMA:
            raise KeyError(
                f"{fname}: gen_schema={fsch!r} != expected "
                f"{_EXPECTED_GEN_SCHEMA!r}. This file predates the current audit "
                f"schema — regenerate it with the current A00.")
        ffp = (str(np.ravel(data_struct['gen_fingerprint'])[0])
               if 'gen_fingerprint' in names else None)
        if ffp is None:
            raise KeyError(f"{fname}: no gen_fingerprint — regenerate with the "
                           f"current A00 (provenance is mandatory).")
        if exp_fp is not None and ffp != exp_fp:
            raise RuntimeError(
                f"{fname}: gen_fingerprint differs from the manifest — this file "
                f"was generated with a DIFFERENT configuration (mixed dataset). "
                f"Do not train on it.")
        # Family identity (Feature A): the file must declare its state_family
        # AND it must equal the table row for this index.
        if 'state_family' not in names:
            raise RuntimeError(
                f"{fname}: no state_family — pre-Feature-A file. Regenerate "
                f"with the current A00.")
        ffam = str(np.ravel(data_struct['state_family'])[0])
        if ffam != state_table['family'][idx]:
            raise RuntimeError(
                f"{fname}: state_family {ffam!r} != table row "
                f"{state_table['family'][idx]!r} — renamed/mislabelled state "
                f"file (payload does not belong at index {idx + 1}).")
        # Scour label sanity (audit R6 C4 + R7 P4). Validate the FULL vector's
        # finiteness (a NaN in ANY support — even one we don't select — signals a
        # corrupt solve), then bound the SELECTED labels to the physical scour
        # ceiling [0, dano_max%] (NOT merely [0, 100]).
        # Range-check the ENTIRE scour_vector, not just the selected supports
        # (audit R7.1 P5): a -10% or 80% value in an UN-selected support still
        # signals a corrupt solve. Finiteness + [0, dano_max%] on the whole vector.
        full_scour = np.ravel(data_struct['scour_vector']).astype(float)
        full_scour_pct = full_scour * 100.0
        if (not np.all(np.isfinite(full_scour))
                or np.any(full_scour_pct < 0.0)
                or np.any(full_scour_pct > scour_max_pct + _SCOUR_EPS)):
            raise RuntimeError(
                f"{fname}: scour_vector {full_scour.tolist()} non-finite or outside "
                f"[0, {scour_max_pct:.3g}] % (dano_max) in some support — corrupt "
                f"or foreign state. Regenerate this state.")
        slabel = full_scour[tgt0] * 100.0
        if bidx is not None:
            # R7 requires the fixity label (audit R7 P4): the legacy raw-k_r
            # `bearing_vector` fallback is dropped — no r6/r7 data lacks fixity, so
            # a missing bearing_fixity means a foreign/old file, not a valid one.
            if 'bearing_fixity' not in names:
                raise KeyError(
                    f"{fname}: bearing heads requested but no data.bearing_fixity "
                    f"— the R7 schema requires the fixity label. Regenerate with "
                    f"the current A00 (bearing_mode='target').")
            full_fix = np.ravel(data_struct['bearing_fixity']).astype(float)
            # Range-check the WHOLE fixity vector (audit R7.1 P5), not just bidx.
            if not np.all(np.isfinite(full_fix)) or np.any(full_fix < 0.0) or np.any(full_fix > 1.0):
                raise RuntimeError(
                    f"{fname}: bearing_fixity {full_fix.tolist()} non-finite or "
                    f"outside [0, 1] in some support — corrupt state. Regenerate.")
            bvec = full_fix[bidx]
            bearing_is_fixity = True

        # RAW format is MANDATORY for R7 (audit R7.1 P5/P6): A00 saves SIX raw
        # metadata fields alongside the un-interpolated TIME channels. Require ALL
        # SIX (A00 writes DimAcel/DimSpace/crop_start/crop_end/bridge_samp/
        # L_bridge_eff), and require each to be a (1, npass_here) row — a wrong
        # length (e.g. Npass+1) is a corrupt/foreign file.
        _RAW_FIELDS = ('DimSpace', 'DimAcel', 'crop_start', 'crop_end',
                       'bridge_samp', 'L_bridge_eff')
        missing_raw = [f for f in _RAW_FIELDS if f not in names]
        if missing_raw:
            raise KeyError(
                f"{fname}: missing RAW-format field(s) {missing_raw} — R7 requires "
                f"the raw un-interpolated format. Regenerate with the current A00.")
        for f in _RAW_FIELDS:
            v = np.ravel(np.asarray(data_struct[f]))
            if v.size != exp_npass or not np.all(np.isfinite(v)):
                raise RuntimeError(
                    f"{fname}: RAW field {f} has {v.size} entries (or non-finite) "
                    f"!= Npass {exp_npass} — corrupt/foreign state.")
        raw_fmt = True

        # Contact-validity gate (audit R5; TWO-TIER since 2026-07-19 — see the
        # _CONTACT_F_TOL_N comment at the top of this module). Brief
        # micro-unloading (small tension, tiny path fraction) is TOLERATED and
        # counted for the dataset summary; peak tension beyond 10% of the
        # static wheel load OR sustained tension is a physics regression and
        # aborts — never silently censored (MNAR).
        # contact_log must also be present, exactly (Npass, 4), and finite.
        if 'contact_log' not in names:
            raise KeyError(f"{fname}: no contact_log (audit-schema file must "
                           f"carry it) - regenerate with the current A00.")
        clog = np.atleast_2d(np.asarray(data_struct['contact_log'], dtype=float))
        available = data_struct['AcelPrimVag'].shape[1]
        npass_here = exp_npass                       # manifest count, now guaranteed > 0
        if clog.shape != (npass_here, 4):
            raise RuntimeError(
                f"{fname}: contact_log shape {clog.shape} != expected "
                f"({npass_here}, 4) — corrupt or partially-written file.")
        # Finiteness on EVERY column (audit R6 C4): a NaN in any contact metric
        # signals a corrupt solve, not just the tension column.
        if not np.all(np.isfinite(clog)):
            raise RuntimeError(
                f"{fname}: contact_log has non-finite entries — "
                f"corrupt solve output; regenerate this state.")
        # Physical ranges (audit R7 P4): the two lost-contact flags (cols 0,1) are
        # booleans and tension_frac_max (col 2) is non-negative. Values like 9 or
        # -1 mean a corrupt/foreign log, not a valid solve.
        if not np.all(np.isin(clog[:, 0:2], (0.0, 1.0))):
            raise RuntimeError(
                f"{fname}: contact_log lost-contact flags (cols 1-2) must be 0/1, "
                f"got values {np.unique(clog[:, 0:2]).tolist()} — corrupt state.")
        if np.any(clog[:, 2] < 0.0) or np.any(clog[:, 2] > 1.0):
            raise RuntimeError(
                f"{fname}: contact_log tension_frac_max (col 3) outside [0, 1] "
                f"(min {clog[:, 2].min():.3g}, max {clog[:, 2].max():.3g}) — "
                f"corrupt state; regenerate.")
        # EXACT count (audit R6 C4): the old one-sided `available < npass_here`
        # let a file with MORE passages than the manifest declares load its extra
        # passages, which the contact gate below (spanning only npass_here rows)
        # never audited. Require an exact match and iterate exactly npass_here.
        if available != npass_here:
            raise RuntimeError(
                f"{fname}: {available} signal passages but manifest says "
                f"passages_per_state={npass_here} — count mismatch (partial write "
                f"or foreign file). Regenerate this state.")
        # ALL THREE channel fields are MANDATORY and must have exactly npass_here
        # passages (audit R7.1 P5) — regardless of which DOFs this call requests. A
        # complete r7 state carries every channel; a missing PitchPrimVag (etc.) is
        # a partial/foreign file even if the current study doesn't read it.
        for _field in ('AcelPrimVag', 'AcelRodaPrimVag', 'PitchPrimVag'):
            if _field not in names:
                raise KeyError(
                    f"{fname}: missing channel field {_field} — incomplete r7 state.")
            if data_struct[_field].shape[1] != npass_here:
                raise RuntimeError(
                    f"{fname}: {_field} has {data_struct[_field].shape[1]} passages "
                    f"!= Npass {npass_here} — inconsistent per-channel counts.")
        hot = np.where((clog[:, 3] > _CONTACT_F_TOL_N)
                       | (clog[:, 2] > _CONTACT_FRAC_TOL))[0]
        if hot.size:
            raise RuntimeError(
                f"{fname}: {hot.size} passage(s) beyond the contact gate "
                f"(peak tension > {_CONTACT_F_TOL_N:.0f} N or tension on > "
                f"{_CONTACT_FRAC_TOL:.1%} of the path; worst F "
                f"{clog[hot, 3].max():.3g} N, worst frac "
                f"{clog[hot, 2].max():.3g}, passages {hot[:8].tolist()}). "
                f"This exceeds the tolerated brief micro-unloading tier — "
                f"physics regression; inspect the generator before training.")
        # Tolerated-tier incidence (2026-07-19): count passages with ANY
        # tension for the dataset summary — the paper reports this number as
        # the scope of the bilateral-contact approximation.
        _tension_rows = clog[:, 3] > 0.0
        n_tension_passages += int(_tension_rows.sum())
        if _tension_rows.any():
            worst_tension_N = max(worst_tension_N, float(clog[:, 3].max()))
            worst_tension_frac = max(worst_tension_frac, float(clog[:, 2].max()))

        # Iterate EXACTLY the audited passage count (audit R6 C4): bound on
        # npass_here (== available, enforced above), never on n_passages/available,
        # so no passage outside the contact-gated window can be loaded.
        for p in range(min(n_passages, npass_here)):
            n_seen += 1
            # Crop-window validity (audit R7.1 P5): 1 <= crop_start <= crop_end <=
            # DimSpace, and DimAcel finite/positive. crop_end < crop_start would
            # otherwise yield a zero-length signal that np truncates silently.
            dim_space = float(np.ravel(data_struct['DimSpace'][0, p])[0])
            dim_acel  = float(np.ravel(data_struct['DimAcel'][0, p])[0])
            crop_s    = float(np.ravel(data_struct['crop_start'][0, p])[0])
            crop_e    = float(np.ravel(data_struct['crop_end'][0, p])[0])
            # Integrality too (audit R7.1 P6): these are sample indices/counts,
            # later fed through int(); a fractional value would be silently
            # truncated. Require finite AND integer-valued AND ordered.
            if not (np.isfinite([dim_space, dim_acel, crop_s, crop_e]).all()
                    and float(dim_space).is_integer() and float(dim_acel).is_integer()
                    and float(crop_s).is_integer() and float(crop_e).is_integer()
                    and dim_acel >= 1 and 1 <= crop_s <= crop_e <= dim_space):
                raise RuntimeError(
                    f"{fname}: passage {p} invalid crop/space params "
                    f"(DimSpace={dim_space}, DimAcel={dim_acel}, crop "
                    f"[{crop_s}, {crop_e}]) — need finite integer "
                    f"1<=crop_start<=crop_end<=DimSpace. Corrupt state; regenerate.")
            # ALL raw channel arrays for this passage must be finite (audit R7.1
            # P6) — not just the requested DOFs — 2-D and DimAcel-long. The row
            # count is vehicle-model-specific (not hard-coded); the requested rows'
            # existence is enforced when they are indexed below.
            for _src in ('AcelPrimVag', 'AcelRodaPrimVag', 'PitchPrimVag'):
                arr = np.asarray(data_struct[_src][0, p])
                if arr.ndim != 2 or arr.shape[1] != int(dim_acel) \
                        or not np.all(np.isfinite(arr)):
                    raise RuntimeError(
                        f"{fname}: passage {p} channel {_src} shape {arr.shape} not "
                        f"(rows, DimAcel={int(dim_acel)}) or non-finite — corrupt state.")
            channels = []
            for dof in requested_dofs:
                src, row = dof_source[dof]
                ch = data_struct[src][0, p][row, :]
                if raw_fmt:
                    ch = _raw_to_space_crop(
                        ch, dim_acel, dim_space, crop_s, crop_e)
                channels.append(ch)
            stacked = np.vstack(channels)                            # (C, L)
            # Signal sanity (audit R6 C4): a NaN/Inf acceleration sample would flow
            # straight into PAA/training. Reject the whole state on the first one.
            if not np.all(np.isfinite(stacked)):
                raise RuntimeError(
                    f"{fname}: passage {p} has non-finite acceleration samples — "
                    f"corrupt solve output; regenerate this state.")
            X_list.append(stacked)
            y_scour.append(slabel.astype(np.float32))
            g_list.append(idx)
            if bidx is not None:
                y_bear.append(bvec.astype(np.float32))

    if not X_list:
        raise RuntimeError(f"No multi-output passages loaded from {dataset_path}")
    # Crop-length consistency (audit R7 P4): round() on DimSpace can vary the crop
    # by ~1-2 samples, which we absorb by truncating to the common minimum. A
    # LARGER spread means the spatial grid is not canonical (a generator bug), so
    # reject it instead of silently truncating a big chunk of signal.
    lmin = min(x.shape[1] for x in X_list)
    lmax = max(x.shape[1] for x in X_list)
    if lmax - lmin > _CROP_RAGGED_TOL:
        raise RuntimeError(
            f"{dataset_path}: per-passage crop lengths span {lmin}..{lmax} "
            f"(> {_CROP_RAGGED_TOL} samples) — the spatial grid is not canonical. "
            f"Inspect DimSpace / crop_start / crop_end in the generator.")
    if lmax != lmin:
        print(f"  [multi-output] crop lengths differ by <= {_CROP_RAGGED_TOL} "
              f"samples - truncating all to {lmin}.")
        X_list = [x[:, :lmin] for x in X_list]
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_scour, dtype=np.float32)                              # (N, n_scour)
    groups = np.array(g_list, dtype=np.int64)                            # (N,)

    if bidx is not None:
        B = np.array(y_bear, dtype=np.float32)                          # (N, n_bearing)
        if bearing_is_fixity:
            # already a ratio in [0,1] -> report as fixity %
            y = np.hstack([y, B * 100.0]).astype(np.float32)
            print("  [multi-output] bearing label = FIXITY ratio (%).")
        else:
            if bearing_max is None or bearing_max <= 0:
                bearing_max = float(B.max()) or 1.0
                print(f"  [multi-output] bearing_max not in manifest - normalising "
                      f"by observed max {bearing_max:.3g} Nm/rad.")
            y = np.hstack([y, (B / bearing_max) * 100.0]).astype(np.float32)

    # Groups structure (audit R7 P3): the split's leak-freeness relies on exactly
    # n_files groups labelled 0..n_files-1, each repeated EXACTLY npass_here times.
    _assert_groups_canonical(groups, n_files, exp_npass)
    # Completeness: every expected state must have contributed at least one
    # surviving passage, and X/y/groups must be the same length (audit R4).
    n_states_loaded = len(set(g_list))
    if n_states_loaded != n_files:
        raise RuntimeError(
            f"{dataset_path}: loaded {n_states_loaded} states but expected "
            f"{n_files} — a state contributed no valid passages.")
    if not (len(X) == len(y) == len(groups)):
        raise RuntimeError(
            f"{dataset_path}: length mismatch X={len(X)} y={len(y)} "
            f"groups={len(groups)} — loader bug or corrupt data.")

    extra = f" + {len(bidx)} bearing" if bidx else ""
    print(f"  [multi-output] {X.shape[0]} passages, {n_files} states, "
          f"{y.shape[1]} heads ({len(tgt0)} scour{extra}).")
    if n_tension_passages:
        print(f"  [contact] tolerated micro-tension tier: {n_tension_passages}/"
              f"{X.shape[0]} passages ({n_tension_passages / X.shape[0]:.2%}) "
              f"show brief wheel unloading past zero (worst "
              f"{worst_tension_N:.3g} N = "
              f"{worst_tension_N / 1.18e5:.1%} of static, worst path fraction "
              f"{worst_tension_frac:.3g}). All within the gate "
              f"(F <= {_CONTACT_F_TOL_N:.0f} N, frac <= {_CONTACT_FRAC_TOL}); "
              f"report this incidence with the results.")
    return X, y, groups


def _read_manifest(dataset_path: str):
    """(n_states, Npass, gen_schema, gen_fingerprint) from case_info.mat, or
    (None, None, None, None) if there is no manifest (audit R4 2026-07-17).
    These drive the loader's strict completeness/provenance checks."""
    ci = os.path.join(dataset_path, 'case_info.mat')
    if not os.path.exists(ci):
        return None, None, None, None
    info = sio.loadmat(ci)['case_info'][0, 0]
    names = info.dtype.names or ()
    def _int(f):
        return int(np.ravel(info[f])[0]) if f in names else None
    def _str(f):
        return str(np.ravel(info[f])[0]) if f in names else None
    return (_int('n_states'), _int('passages_per_state'),
            _str('gen_schema'), _str('gen_fingerprint'))


def _read_manifest_optional_string(dataset_path: str, field: str) -> str | None:
    """Read an optional scalar string from case_info.mat without weakening R8."""
    ci = os.path.join(dataset_path, "case_info.mat")
    if not os.path.exists(ci):
        return None
    info = sio.loadmat(ci)["case_info"][0, 0]
    names = info.dtype.names or ()
    return str(np.ravel(info[field])[0]) if field in names else None


def _read_file_digests(dataset_path: str):
    """(digests dict {fname: sha256}, root_sha) from file_digests.mat, or
    (None, None) if absent (audit R7.1 P4: source content integrity). A00 writes
    a SHA-256 of every NNNN.mat plus a root = SHA-256 of the sorted per-file
    digests. The loader verifies each file's bytes against this — so a bit-flip /
    disk-or-transport corruption / same-size overwrite is caught, since none of
    those preserve a SHA even when they preserve file size."""
    fd = os.path.join(dataset_path, 'file_digests.mat')
    if not os.path.exists(fd):
        return None, None
    info = sio.loadmat(fd)
    if 'file_digests' not in info:
        return None, None
    s = info['file_digests'][0, 0]
    names = s.dtype.names or ()
    # A single 'digest_lines' char blob ("0001.mat:<sha>\n0002.mat:<sha>...") plus
    # a 'root' — a plain-string round-trip avoids scipy cell-array quirks.
    lines = str(np.ravel(s['digest_lines'])[0]) if 'digest_lines' in names else ""
    root = str(np.ravel(s['root'])[0]) if 'root' in names else None
    per: dict = {}
    for ln in lines.splitlines():
        ln = ln.strip()
        if ':' in ln:
            fn, sh = ln.split(':', 1)
            per[fn.strip()] = sh.strip()
    return (per if per else None), root


def _root_digest(per_file: dict) -> str:
    """Root = SHA-256 over the sorted 'fname:sha' lines (audit R7.1 P4)."""
    joined = "\n".join(f"{k}:{per_file[k]}" for k in sorted(per_file))
    return hashlib.sha256(joined.encode()).hexdigest()


def verify_source_file_bytes(dataset_path: str) -> dict:
    """Verify every source file recorded in ``file_digests.mat``.

    This closes the fast-cache gap where a same-size edit of ``NNNN.mat`` could
    previously go unnoticed until a cache rebuild.  The first call in a process
    hashes every recorded file.  Later cache lookups reuse that result only
    while every file retains the same size, mtime and ctime; any change forces a
    new full pass.  The digest table may also list sidecars (new generators list
    ``case_info.mat`` and ``damage_states.mat``), while old state-only tables
    remain valid. Those two required sidecars are always hashed separately into
    the protocol identity, with a flag stating whether the generation-time root
    covered them.

    Returns a compact, JSON-safe verification summary for protocol provenance.
    Raises ``RuntimeError`` on an incomplete, unsafe or mismatching digest set.
    """
    dataset_path = os.path.realpath(os.path.abspath(dataset_path))
    per_file, root = _read_file_digests(dataset_path)
    if not per_file or not root:
        raise RuntimeError(
            f"{dataset_path}: missing/invalid file_digests.mat; source bytes "
            "cannot be certified for cache reuse.")
    if not hmac.compare_digest(_root_digest(per_file), str(root).lower()):
        raise RuntimeError(
            f"{dataset_path}: file_digests.mat is internally inconsistent "
            "(recomputed root differs from stored root).")

    actual_states = sorted(
        name for name in os.listdir(dataset_path) if _re_state.fullmatch(name)
    )
    recorded_states = sorted(name for name in per_file if _re_state.fullmatch(name))
    if recorded_states != actual_states:
        raise RuntimeError(
            f"{dataset_path}: state-file digest inventory mismatch; recorded "
            f"{len(recorded_states)} states but found {len(actual_states)}.")

    def _signature() -> tuple:
        rows = []
        for name, expected in sorted(per_file.items()):
            # Digest entries are filenames, never paths.  Rejecting traversal
            # also makes the verification scope auditable and dataset-local.
            if os.path.basename(name) != name or name in (".", ".."):
                raise RuntimeError(
                    f"{dataset_path}: unsafe digest entry {name!r}.")
            expected = str(expected).lower()
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise RuntimeError(
                    f"{dataset_path}: invalid SHA-256 for digest entry {name!r}.")
            path = os.path.join(dataset_path, name)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"{dataset_path}: digest-listed source file is missing: {name}.")
            stat = os.stat(path)
            rows.append((
                name, int(stat.st_size), int(stat.st_mtime_ns),
                int(stat.st_ctime_ns), expected,
            ))
        return tuple(rows)

    with _SOURCE_VERIFY_LOCK:
        before = _signature()
        cached = _SOURCE_VERIFY_CACHE.get(dataset_path)
        if cached != (str(root).lower(), before):
            for name, _size, _mtime, _ctime, expected in before:
                actual = _sha256_file(os.path.join(dataset_path, name))
                if not hmac.compare_digest(actual, expected):
                    raise RuntimeError(
                        f"{dataset_path}: source SHA-256 mismatch for {name}; "
                        "the dataset is corrupt, incomplete or was edited after "
                        "generation. Refusing cached and uncached training.")
            after = _signature()
            if after != before:
                raise RuntimeError(
                    f"{dataset_path}: source files changed during SHA-256 "
                    "verification; retry only after generation/copying stops.")
            _SOURCE_VERIFY_CACHE[dataset_path] = (str(root).lower(), after)

    sidecars = sorted(name for name in per_file if not _re_state.fullmatch(name))
    # R8 datasets generated before audit r4 contain a state-only digest table.
    # They remain scientifically usable, but their two interpretation-critical
    # sidecars must still participate in the full protocol identity: otherwise
    # editing damage_states.mat could change the split/labels while resuming the
    # same Optuna study. New A00 tables list these files and therefore also
    # authenticate them against the generation root; old tables get an explicit
    # live SHA binding (identity, not a retroactive generation-time signature).
    required_sidecar_sha256: dict[str, str] = {}
    for name in ("case_info.mat", "damage_states.mat"):
        path = os.path.join(dataset_path, name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{dataset_path}: required provenance sidecar is missing: {name}.")
        stat_before = os.stat(path)
        required_sidecar_sha256[name] = _sha256_file(path)
        stat_after = os.stat(path)
        signature_before = (
            int(stat_before.st_size), int(stat_before.st_mtime_ns),
            int(stat_before.st_ctime_ns),
        )
        signature_after = (
            int(stat_after.st_size), int(stat_after.st_mtime_ns),
            int(stat_after.st_ctime_ns),
        )
        if signature_after != signature_before:
            raise RuntimeError(
                f"{dataset_path}: {name} changed during SHA-256 verification; "
                "retry only after generation/copying stops.")
    return {
        "root": str(root).lower(),
        "entry_count": len(per_file),
        "state_count": len(recorded_states),
        "sidecars": sidecars,
        "required_sidecar_sha256": required_sidecar_sha256,
        "required_sidecars_generation_digest_covered":
            all(name in per_file for name in required_sidecar_sha256),
    }


def read_state_table(dataset_path: str) -> dict:
    """The per-state FAMILY table from damage_states.mat (Feature A, 2026-07-19).

    A00 writes, row-aligned with DamageStates:
        StateFamily  — identity tag per state (one of STATE_FAMILIES); assigned
                       by the GENERATOR at construction, never derived from
                       labels downstream (with per-state persistent nuisances,
                       `y` alone cannot distinguish target_healthy from
                       nuisance_only — the Codex R7.1 correction).
        AnchorTarget — scour_only: pier index; bearing_only: 1=left, 2=right.
        AnchorLevel  — severity level index (1..n_anchor_levels), 0 otherwise.
        CrackOn      — pre-drawn per-state crack activation (forced OFF for the
                       controlled-probe families, ON for nuisance_only).
    MANDATORY for every regression dataset: the stratified split and the
    family-identity disentanglement probe both consume it. A dataset without it
    predates Feature A — regenerate, do not guess families from labels."""
    ds = os.path.join(dataset_path, 'damage_states.mat')
    if not os.path.exists(ds):
        raise RuntimeError(
            f"{dataset_path}: no damage_states.mat — the stratified split needs "
            f"the state-family table. Regenerate with the current A00.")
    m = sio.loadmat(ds)
    required = ('StateFamily', 'AnchorTarget', 'AnchorLevel', 'CrackOn',
                'DamageStates', 'BearingFixity')
    missing = [f for f in required if f not in m]
    if missing:
        raise RuntimeError(
            f"{dataset_path}: damage_states.mat lacks {missing} — pre-Feature-A "
            f"dataset. Regenerate with the current A00 (families are generated, "
            f"never inferred).")
    family = [str(np.ravel(c)[0]) for c in np.ravel(m['StateFamily'])]
    bad = sorted({f for f in family if f not in STATE_FAMILIES})
    if bad:
        raise RuntimeError(
            f"{dataset_path}: unknown state_family value(s) {bad} — expected "
            f"one of {STATE_FAMILIES}. Corrupt/foreign family table.")
    table = {
        'family':         family,
        'anchor_target':  np.ravel(m['AnchorTarget']).astype(int),
        'anchor_level':   np.ravel(m['AnchorLevel']).astype(int),
        'crack_on':       np.ravel(m['CrackOn']).astype(bool),
        'damage_states':  np.atleast_2d(np.asarray(m['DamageStates'], dtype=float)),
        'bearing_fixity': np.atleast_2d(np.asarray(m['BearingFixity'], dtype=float)),
    }
    n = len(family)
    if not (len(table['anchor_target']) == len(table['anchor_level'])
            == len(table['crack_on']) == table['damage_states'].shape[0]
            == table['bearing_fixity'].shape[0] == n):
        raise RuntimeError(
            f"{dataset_path}: state-table arrays are not row-aligned "
            f"(family={n}, target={len(table['anchor_target'])}, "
            f"level={len(table['anchor_level'])}, crack={len(table['crack_on'])}, "
            f"scour={table['damage_states'].shape[0]}, "
            f"fixity={table['bearing_fixity'].shape[0]}) — corrupt table.")
    return table


def _read_dano_max(dataset_path: str) -> float | None:
    """The scour ceiling `dano_max` (a fraction, e.g. 0.60) from case_info.mat,
    or None if the manifest predates it (audit R7 P4). Used to bound the scour
    label to the physically-generated range rather than merely [0, 100] %."""
    ci = os.path.join(dataset_path, 'case_info.mat')
    if not os.path.exists(ci):
        return None
    info = sio.loadmat(ci)['case_info'][0, 0]
    # A00 writes it as `scour_dano_max_frac`; accept the shorter aliases too.
    for f in ('scour_dano_max_frac', 'dano_max_frac', 'dano_max'):
        if f in (info.dtype.names or ()):
            v = float(np.ravel(info[f])[0])
            if v > 0:
                return v
    return None


def _count_state_files(dataset_path: str) -> int:
    """Number of contiguous NNNN.mat starting at 0001 (fallback when a manifest
    lacks n_states). Stops at the first gap."""
    n = 0
    while os.path.exists(os.path.join(dataset_path, f"{n + 1:04d}.mat")):
        n += 1
    return n


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

def _stable_stratum_seed(seed: int, stratum_key: str) -> int:
    """Deterministic, platform-independent RNG seed for one stratum's
    within-stratum permutation (SHA-256 of 'seed|key', first 8 bytes).
    Python's hash() is salted per process, so it must never be used here."""
    digest = hashlib.sha256(f"{seed}|{stratum_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _stratum_keys(table: dict, dano_max: float | None) -> list[str]:
    """One stratum key per state (Feature A). Strata are the units the split
    balances across partitions:
      * scour_only / bearing_only -> (family, anchor target): each pier's/side's
        anchor set (n_anchor_levels x n_anchor_reps = 8 states) spans all three
        partitions; the severity LEVELS inside are shuffled, and which level
        landed where is pre-registered in split_manifest.json (levels are NOT
        their own strata — with 2 replicas a per-level stratum would never
        reach inner-val under the assignment pattern).
      * target_healthy / nuisance_only -> the family itself.
      * joint -> crack flag x severity bin (max normalized head, terciles):
        balances the crack confound and the severity range across partitions.
    """
    ds, bf = table['damage_states'], table['bearing_fixity']
    # Normalisers for the joint severity bins. dano_max comes from the manifest;
    # bearing uses the observed table max (bin edges only — not a label scale).
    dmax = float(dano_max) if dano_max else max(float(ds.max()), 1e-12)
    bmax = float(bf.max())
    keys = []
    for i, fam in enumerate(table['family']):
        if fam in ('scour_only', 'bearing_only'):
            keys.append(f"{fam}|target{int(table['anchor_target'][i])}")
        elif fam in ('target_healthy', 'nuisance_only'):
            keys.append(fam)
        else:                                                    # joint
            sev_s = float(ds[i].max()) / dmax
            sev_b = (float(bf[i].max()) / bmax) if bmax > 0 else 0.0
            sev = min(max(sev_s, sev_b), 1.0)
            sev_bin = min(int(sev * N_JOINT_SEV_BINS), N_JOINT_SEV_BINS - 1)
            keys.append(f"joint|crack{int(bool(table['crack_on'][i]))}|sev{sev_bin}")
    return keys


def _write_or_verify_split_manifest(dataset_dir: str, table: dict,
                                    strata: list[str], assignment: list[str],
                                    seed: int) -> None:
    """Persist the split as an auditable, FINGERPRINTED record (Feature A).

    Written once per dataset to data/<dataset>/split_manifest.json. On every
    later call the freshly-recomputed manifest must equal the stored one
    byte-for-byte (canonical JSON) — catching nondeterminism, a foreign file,
    or a code change that silently altered the assignment. The manifest binds
    to the dataset via its gen_fingerprint."""
    _, _, _, gen_fp = _read_manifest(dataset_dir)
    fam_counts: dict = {}
    for fam in sorted(set(table['family'])):
        fam_counts[fam] = {p: sum(1 for s, a in enumerate(assignment)
                                  if table['family'][s] == fam and a == p)
                           for p in ('train', 'val', 'test')}
    payload = {
        "split_manifest_version": 1,
        "policy": {"seed": seed, "test_frac": SPLIT_TEST_FRAC,
                   "val_frac": SPLIT_VAL_FRAC,
                   "pattern": list(STRATIFY_PATTERN),
                   "n_joint_sev_bins": N_JOINT_SEV_BINS},
        "dataset_gen_fingerprint": gen_fp,
        "n_states": len(assignment),
        "state_family": list(table['family']),
        "stratum": list(strata),
        "assignment": list(assignment),
        "partition_counts_by_family": fam_counts,
    }
    path = os.path.join(dataset_dir, 'split_manifest.json')
    canon = json.dumps(payload, sort_keys=True)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                stored = json.dumps(json.load(fh), sort_keys=True)
        except Exception as e:                                   # noqa: BLE001
            raise RuntimeError(f"{path}: unreadable split manifest ({e}) — "
                               f"delete it to let the split re-persist.") from e
        if stored != canon:
            raise RuntimeError(
                f"{path}: stored split manifest DIFFERS from the recomputed "
                f"split — foreign file, tampering, or a split-policy change "
                f"without a schema bump. Refusing to train on an ambiguous "
                f"split; investigate (delete the manifest ONLY if the dataset "
                f"was legitimately regenerated).")
    else:
        _atomic_write_json(path, payload)
        print(f"  [split] wrote split manifest -> {path}")


def canonical_grouped_splits(
    n_samples: int,
    groups:    np.ndarray | None = None,
    seed:      int = SPLIT_SEED,        # protocol constants (hashed) — change
    test_frac: float = SPLIT_TEST_FRAC, # them THERE, never per-call; every
    val_frac:  float = SPLIT_VAL_FRAC,  # caller must use the canonical values
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """THE canonical 3-way GROUPED split (train, inner-val, outer-test).
    AUDIT R5 2026-07-17; STRATIFIED-BY-FAMILY since Feature A (2026-07-19).

    Grouped by damage STATE (a state's passages never straddle partitions), so:
      - TRAIN (60%): scaler fit + model fit.
      - inner-VAL (20%): Optuna HPO objective + champion/pair SELECTION.
      - outer-TEST (20%): UNTOUCHED until the final reported metrics, so the
        numbers in the paper are not the same set that selected the model.

    Stratification (Feature A): states are assigned to partitions PER STRATUM
    (family / anchor target / joint crack x severity bin — see _stratum_keys)
    with a deterministic seeded permutation + the fixed 60/20/20 pattern, so
    every family is GUARANTEED present in all three partitions (the old random
    GroupShuffleSplit left ~15% cross-seed odds of an empty disentanglement
    probe — audit R6 C8C). Fully deterministic: same dataset + seed -> same
    split on every machine; recorded + verified via split_manifest.json.

    `dataset_name` is REQUIRED in grouped mode (it locates the state-family
    table). groups=None (legacy single-scour classification): per-passage
    80/20, empty test — bit-compatible with the old classification studies.
    """
    idx = np.arange(n_samples)
    if groups is None:
        print("  [split] WARNING: ungrouped per-passage split (legacy "
              "classification mode) - no outer test set.")
        tr, va = train_test_split(idx, test_size=val_frac, random_state=seed)
        return tr, va, np.array([], dtype=int)
    groups = np.asarray(groups)
    if groups.shape[0] != n_samples:
        raise ValueError(f"groups has {groups.shape[0]} entries for "
                         f"{n_samples} samples - stale cache? Delete the "
                         f"cache dir and rebuild.")
    if dataset_name is None:
        raise RuntimeError(
            "canonical_grouped_splits: grouped mode now REQUIRES dataset_name "
            "(Feature A stratified split reads the state-family table from "
            "data/<dataset>/damage_states.mat). Update the caller.")
    # The 60/20/20 pattern IS the fraction policy; other fractions would need a
    # different pattern — refuse silently-inconsistent arguments.
    if (test_frac, val_frac) != (SPLIT_TEST_FRAC, SPLIT_VAL_FRAC):
        raise ValueError(
            f"stratified split implements exactly test={SPLIT_TEST_FRAC}/"
            f"val={SPLIT_VAL_FRAC} via STRATIFY_PATTERN; got ({test_frac}, "
            f"{val_frac}). Change the protocol constants, not the call.")
    dataset_dir = os.path.join('data', dataset_name)
    table = read_state_table(dataset_dir)
    n_states = len(table['family'])
    # groups must be exactly the states 0..n_states-1 (the canonical block
    # vector is asserted elsewhere; here we bind it to the TABLE's row count).
    uniq = np.unique(groups)
    if uniq.min() != 0 or uniq.max() != n_states - 1 or len(uniq) != n_states:
        raise RuntimeError(
            f"groups reference states {uniq.min()}..{uniq.max()} "
            f"({len(uniq)} unique) but the state table has {n_states} rows — "
            f"stale cache or mismatched dataset. Rebuild the cache.")
    strata = _stratum_keys(table, _read_dano_max(dataset_dir))
    # ── deterministic per-stratum assignment ────────────────────────────────
    assignment: list[str | None] = [None] * n_states
    for key in sorted(set(strata)):
        members = [s for s in range(n_states) if strata[s] == key]
        # Seeded permutation removes any construction-order artifact (anchor
        # levels ascend by build order) while staying fully deterministic.
        rng = np.random.default_rng(_stable_stratum_seed(seed, key))
        perm = rng.permutation(len(members))
        for pos, mi in enumerate(perm):
            assignment[members[mi]] = STRATIFY_PATTERN[pos % len(STRATIFY_PATTERN)]
    assert all(a is not None for a in assignment)
    # ── coverage guarantee (the whole point of Feature A) ───────────────────
    # Every family with >= len(pattern) states must appear in ALL partitions.
    for fam in sorted(set(table['family'])):
        members = [s for s in range(n_states) if table['family'][s] == fam]
        got = {assignment[s] for s in members}
        if len(members) >= len(STRATIFY_PATTERN) and got != {'train', 'val', 'test'}:
            raise RuntimeError(
                f"stratified split: family {fam!r} ({len(members)} states) "
                f"landed only in {sorted(got)} — coverage guarantee violated "
                f"(assignment bug; do not train).")
    _write_or_verify_split_manifest(dataset_dir, table, strata, assignment, seed)
    # ── states -> sample indices via groups ─────────────────────────────────
    part_states = {p: np.array([s for s in range(n_states) if assignment[s] == p])
                   for p in ('train', 'val', 'test')}
    tr = idx[np.isin(groups, part_states['train'])]
    va = idx[np.isin(groups, part_states['val'])]
    te = idx[np.isin(groups, part_states['test'])]
    return tr, va, te


def canonical_train_val_split(
    n_samples: int,
    groups:    np.ndarray | None = None,
    seed:      int = SPLIT_SEED,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(train, inner-val) from the canonical 3-way split — for scaler fit, HPO,
    model fit and SELECTION. The outer test set is deliberately excluded here so
    nothing that trains or selects ever sees it. Use canonical_test_split for
    the final reported metrics. dataset_name is REQUIRED in grouped mode
    (Feature A stratified split)."""
    tr, va, _ = canonical_grouped_splits(n_samples, groups, seed=seed,
                                         dataset_name=dataset_name)
    return tr, va


def canonical_test_split(
    n_samples: int,
    groups:    np.ndarray | None = None,
    seed:      int = SPLIT_SEED,
    dataset_name: str | None = None,
) -> np.ndarray:
    """The untouched outer TEST indices — ONLY for final reported metrics."""
    _, _, te = canonical_grouped_splits(n_samples, groups, seed=seed,
                                        dataset_name=dataset_name)
    return te


def _cache_stem(dataset_name: str, config: dict) -> str:
    """
    Build the base filename fragment that uniquely identifies a cache file.

    Format:  <dataset>_<method>_dofs_<d0>_<d1>_..._disc<k>_gs5

    The trailing schema tag versions everything baked into the cache: the split
    policy (scaler fit on the canonical grouped TRAIN partition) AND the loaded
    data itself. gs1 = grouped split (2026-07-17); gs2 = R3 (fixed-profile
    geometry/crop + contact filtering); gs3 = R5 (3-way train/val/test split
    + reflect-fold profile); gs4 = R7 (atomic cache write + artifact digests +
    inventory + stricter loader payload); **gs5** = R8 (family-STRATIFIED split
    + mandatory state table + per-file state_family); **gs6** = audit r3
    2026-07-22 (TRUE Keogh PAA replaces the linear resampler + per-global-DOF
    paired noise RNG + segment count named in the stem). Older-tag caches are
    orphaned, never reused.

    Kept private; the public filenames (features, labels, groups, scaler) are
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
    # Segment count in the stem (audit r3 2026-07-22): N_SEGMENTS is a module
    # constant, so editing it used to leave the stem unchanged and silently
    # reuse a stale-length cache. Now the stem names the length it contains.
    stem += f"_seg{N_SEGMENTS}"
    stem += CACHE_SCHEMA_TAG   # data+split+cache-contract schema tag (R7)
    return stem


def _raw_to_space_crop(y_time, dim_acel, dim_space, crop_start, crop_end) -> np.ndarray:
    """MATLAB D01 mirror: uniform time->space resample + bridge crop ("Option B").

    From 2026-07-14 the generator saves the RAW, un-interpolated, noise-free TIME
    -domain signal plus these parameters, instead of a pre-interpolated cropped
    window. This rebuilds exactly what the legacy MATLAB pipeline baked in:

        xx = linspace(1, DimSpace, DimAcel);  xi = 1:DimSpace
        y_space = interp1(xx, y_time, xi)      % linear  == np.interp
        y_crop  = y_space(crop_start:crop_end) % 1-based, inclusive

    The time->space map is uniform because the speed is constant within a passage.
    Doing it HERE keeps the whole measurement model at load time: noise can be
    injected in the TIME domain (physically correct - and then coloured by this
    interpolation, exactly as the legacy baked noise was) or in SPACE afterwards
    (white), without ever regenerating the data. See the noise-domain finding.
    """
    xx = np.linspace(1.0, float(dim_space), int(dim_acel))
    xi = np.arange(1, int(dim_space) + 1, dtype=float)
    y_space = np.interp(xi, xx, np.asarray(y_time, dtype=float))
    return y_space[int(crop_start) - 1:int(crop_end)]      # 1-based inclusive


def _inject_sensor_noise(X: np.ndarray, dofs: list[int], sn: dict) -> np.ndarray:
    """
    Load-time measurement-noise injection (noise policy 2026-07-12).

    Generation is noise-free from stage1_crack onward (A00 use_signal_noise =
    false -> D01 adds nothing), so any noise a study needs is injected HERE,
    where the model stays configurable per experiment and per channel.
    Per-channel LEVELS must come from sensor DATASHEETS (noise density
    [ug/rtHz] x sqrt(bandwidth) -> additive floor; e.g. the rail-qualified
    IMUs in the reference shortlist). NOTE: EN 61373 position severities
    (carbody < bogie < axle) describe the VIBRATION ENVIRONMENT for equipment
    qualification - relevant to sensor RANGE selection and reliability/aging
    (papers/'Confiabilidade Sensores MEMS Ferroviários'), NOT the acquisition
    noise floor; do not scale noise by them. Deterministic (fixed RNG seed
    42), so a cache rebuild reproduces identical features; the cache stem
    carries a noise tag so noisy and clean caches never collide.

    Modes:
      {'mode': 'legacy_wheel', 'desvio': 0.05}
          The legacy MATLAB D01 noise: multiplicative gaussian
          (std = desvio·|signal|) on the WHEEL channels only (global DOFs 3,4).
          APPROXIMATES (does NOT bit-reproduce) the Stage-0/1 baked-in noise:
          D01 added the noise in the TIME domain BEFORE the space interpolation,
          whereas here it is added in the SPACE domain after. Because interp1 is
          linear, baked noise = interp(white) = band-limited/COLORED and
          speed-dependent, with ~0.67x the variance but ~1.46x the energy
          surviving PAA (verified: scratchpad/noise_domain_check.py). Same
          nominal 5%, materially different perturbation - fine as a robustness
          knob, NOT a reproduction. See framework_rationale (noise-domain entry).
    Per-channel additive noise-floor modes (the physically-correct model: a
    signal-INDEPENDENT floor from sensor datasheets, noise-density x sqrt(BW)):
    add here when the noise-robustness arm lands.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    desvio = float(sn.get('desvio', 0.05))
    WHEELS = (3, 4)

    # NOISE PAIRING (fix 2026-07-22, external audit r3, verified): the RNG is
    # keyed by the GLOBAL DOF id, so a given channel receives the IDENTICAL
    # realization no matter which subset it is loaded in or in what order —
    # sensor-set comparisons are noise-paired. The pre-fix single sequential
    # generator made the draw depend on the channel's position within the
    # loaded subset (same DOF: different noise alone vs in a pair), leaking
    # realization variance into config comparisons.
    def add_mult(mask_dofs):
        for i, d in enumerate(dofs):
            if d in mask_dofs:
                rng_d = np.random.Generator(
                    np.random.PCG64([NOISE_RNG_SEED, int(d)]))
                X[:, i, :] += (desvio * X[:, i, :] *
                               rng_d.standard_normal(X[:, i, :].shape)
                               .astype(np.float32))

    if sn['mode'] == 'legacy_wheel':
        # Reproduce (approximately) the legacy baked model: mult noise, WHEELS only.
        add_mult(WHEELS)
    elif sn['mode'] == 'all_mult':
        # Channel-symmetric Gaussian multiplicative noise on EVERY channel
        # (pointwise sigma = desvio*|signal|). Use on NOISE-FREE data
        # (varVST / new stages) to make all channels equally noisy. Do NOT use on
        # legacy baked-wheel data (varNVST Stage-0/1) - the wheels would be
        # DOUBLE-noised; use 'sprung_mult' there instead.
        add_mult(set(dofs))
    elif sn['mode'] == 'sprung_mult':
        # Multiplicative 5% on the SPRUNG channels only (all DOFs except wheels
        # 3,4). For the legacy baked-wheel data: brings the sprung channels up to
        # ~5% without re-noising the already-noisy wheels. NOTE the channels then
        # differ in noise CHARACTER (wheels = baked colored/speed-dep; sprung =
        # load-time white) - see the domain caveat above; state it if reported.
        add_mult({d for d in dofs if d not in WHEELS})
    else:
        raise ValueError(f"unknown sensor_noise mode {sn['mode']!r}")
    return X


def get_or_create_cache(
    config:       dict,
    dataset_name: str,
    cache_dir:    str,
) -> tuple[np.ndarray, np.ndarray, object, np.ndarray | None]:
    """
    Return processed data as memory-mapped arrays, creating the cache on the
    first call and reading it on every subsequent call.

    Cache layout (inside cache_dir)
    --------------------------------
        cache_<stem>.npy        - processed feature array
        cache_<stem>_labels.npy - discretised label array
        cache_<stem>_groups.npy - damage-state (file) index per sample
        scaler_<stem>.pkl       - fitted sklearn scaler  (or .pt for PyTorch)

    Leak-free contract (AUDIT FIX 2026-07-17)
    -----------------------------------------
    The scaler is fitted on the canonical training partition produced by
    canonical_train_val_split (seed 42, test_size=0.20, GROUPED by damage
    state for regression), so the same indices are used whether the cache is
    being created or already exists, and no validation state ever touches the
    scaler. The <stem> carries a split-policy tag, so caches built under the
    old leaky per-passage split are orphaned, not reused.

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
        groups      (np.ndarray | None): damage-state id per sample - pass to
                                  canonical_train_val_split. None only in
                                  legacy classification mode (label == file).
    """
    os.makedirs(cache_dir, exist_ok=True)
    stem = _cache_stem(dataset_name, config)
    regression = config.get('task') == 'regression'

    feat_path   = os.path.join(cache_dir, f"cache_{stem}.npy")
    label_path  = os.path.join(cache_dir, f"cache_{stem}_labels.npy")
    group_path  = os.path.join(cache_dir, f"cache_{stem}_groups.npy")
    prov_path   = os.path.join(cache_dir, f"cache_{stem}_prov.json")
    scaler_path = os.path.join(cache_dir, f"scaler_{stem}.pkl")
    scaler_path_pt = scaler_path.replace('.pkl', '.pt')

    # Provenance of the SOURCE dataset (audit R4/R5/R6): schema + fingerprint +
    # STATE COUNT + completion marker + a lightweight FILE INVENTORY from its
    # manifest, so a cache built from an older regeneration (or from a dataset
    # whose files were later deleted / truncated / swapped) is invalidated
    # instead of silently reused.
    #
    # Inventory (audit R6 C3): the manifest fingerprint alone does not notice a
    # same-count file SWAP (0001.mat -> 9999.mat) or an in-place edit that changes
    # a file's size. We store sorted (filename, size) pairs — a pure os.stat pass,
    # NO MAT byte reads, so the cache's whole purpose (avoid re-deserialising GBs)
    # is preserved. mtime is deliberately excluded: unzipping a bundle resets it
    # and would trigger spurious rebuilds.
    cur_schema = cur_fp = cur_nstates = cur_npass = cur_dano = None
    cur_complete = False
    cur_inventory: list = []
    cur_manifest_sha = None
    cur_src_root = None
    cur_digests_sha = None
    cur_states_sha = None
    src_dir = os.path.join('data', dataset_name)
    if regression and os.path.isdir(src_dir):
        cur_nstates, cur_npass, cur_schema, cur_fp = _read_manifest(src_dir)
        cur_dano = _read_dano_max(src_dir)
        state_files = sorted(f for f in os.listdir(src_dir) if _re_state.fullmatch(f))
        n_present = len(state_files)
        cur_inventory = [[f, os.path.getsize(os.path.join(src_dir, f))]
                         for f in state_files]
        ci_path = os.path.join(src_dir, 'case_info.mat')
        if os.path.exists(ci_path):
            cur_manifest_sha = _sha256_file(ci_path)   # manifest edit -> cache stale
        # Audit r3 (2026-07-22): also pin the per-state DIGEST TABLE itself
        # (file_digests.mat) and the family table (damage_states.mat). Any
        # tamper that keeps the digest chain self-consistent must rewrite
        # these files -> cache invalidated. Audit r4 additionally verifies every
        # recorded source byte before either cache reuse or rebuild, so even an
        # inconsistent same-size edit of one NNNN.mat is rejected immediately.
        fd_path = os.path.join(src_dir, 'file_digests.mat')
        cur_digests_sha = _sha256_file(fd_path) if os.path.exists(fd_path) else None
        dsm_path = os.path.join(src_dir, 'damage_states.mat')
        cur_states_sha = _sha256_file(dsm_path) if os.path.exists(dsm_path) else None
        # `complete` requires the marker to EXIST *and* its content (schema+fp) to
        # match the manifest (audit R7.1 P3): a stale/wrong-content marker on the
        # fast path must not certify a source as complete.
        marker = os.path.join(src_dir, '_GENERATION_COMPLETE')
        marker_ok = False
        if os.path.exists(marker):
            try:
                mk = [ln.strip() for ln in open(marker).read().splitlines() if ln.strip()]
                # schema + fp + a ROOT source digest (line 3) all required.
                marker_ok = (mk[:2] == [cur_schema, cur_fp] and len(mk) >= 3)
                if marker_ok:
                    cur_src_root = mk[2]      # regeneration -> new root -> cache stale
            except Exception:
                marker_ok = False
        cur_complete = (marker_ok and cur_nstates is not None
                        and n_present == cur_nstates)
    # SOURCE-provenance fields (compared as a block on the fast path). Includes the
    # source ROOT digest (audit R7.1 P4), so regenerating the dataset — which
    # rewrites file_digests + the marker root — invalidates a stale cache.
    cur_src = {"gen_schema": cur_schema, "gen_fingerprint": cur_fp,
               "n_states": cur_nstates, "passages_per_state": cur_npass,
               "dano_max": cur_dano, "manifest_sha256": cur_manifest_sha,
               "source_root": cur_src_root, "complete": cur_complete,
               "inventory": cur_inventory,
               "file_digests_sha256": cur_digests_sha,
               "damage_states_sha256": cur_states_sha}

    # Audit r4: a cache hit is allowed only after the recorded SOURCE bytes
    # themselves have passed SHA-256 verification.  Memoisation makes the first
    # sensor configuration pay the sequential disk read; later configurations
    # perform only a stat pass unless a source file changed.
    if regression:
        verify_source_file_bytes(src_dir)
        # Bind the bytes used by this cache call to the descriptor that named
        # the Optuna study. Without this second identity check, a dataset could
        # be legitimately regenerated/replaced after driver import: the cache
        # would then train on new bytes under the old protocol hash.
        expected_provenance = (
            (config.get("protocol_descriptor") or {})
            .get("rung", {})
            .get("dataset_provenance")
        )
        if config.get("protocol_hash") and expected_provenance is None:
            raise RuntimeError(
                f"{dataset_name}: protocol-hashed regression config has no "
                "dataset_provenance descriptor; refusing an unbound cache.")
        if expected_provenance is not None:
            # Lazy import avoids a module-level dataset <-> protocol cycle.
            from core.protocol import read_dataset_provenance
            current_provenance = read_dataset_provenance(src_dir)
            if current_provenance != expected_provenance:
                raise RuntimeError(
                    f"{dataset_name}: source identity changed after the protocol "
                    "hash was computed. Refusing to reuse/build a cache under "
                    "the stale study identity; restart the driver so it computes "
                    "a fresh protocol hash.")

    # ── Fast path (audit R7.1 P5): validate a reusable cache and return it, or
    # return None. Factored out so it can be re-checked INSIDE the build lock. ──
    def _try_reuse():
        cache_ready = (os.path.exists(feat_path) and os.path.exists(label_path)
                       and os.path.exists(group_path))
        scaler_ready = os.path.exists(scaler_path) or os.path.exists(scaler_path_pt)
        if not (cache_ready and scaler_ready):
            return None
        prov_ok = True
        if regression:
            try:
                with open(prov_path) as fh:
                    stored = json.load(fh)
                # (a) source provenance must match, and (b) the cache ARTIFACTS
                # (incl. the scaler) must hash to what was recorded at build time.
                prov_ok = (stored.get("source") == cur_src)
                if prov_ok:
                    arts = stored.get("artifacts", {})
                    sc = scaler_path if os.path.exists(scaler_path) else scaler_path_pt
                    prov_ok = (arts.get("feat")   == _sha256_file(feat_path)
                               and arts.get("labels") == _sha256_file(label_path)
                               and arts.get("groups") == _sha256_file(group_path)
                               and arts.get("scaler") == _sha256_file(sc))
                    if not prov_ok:
                        print("  [CACHE STALE] artifact digest mismatch "
                              "(corrupt/tampered cache) - rebuilding.")
            except Exception:                                    # missing/unreadable
                prov_ok = False
            if not cur_complete:                                 # explicit gate
                prov_ok = False
            if not prov_ok:
                print("  [CACHE STALE] source provenance changed / incomplete "
                      "- rebuilding.")
        if not prov_ok:
            return None
        X_processed = np.load(feat_path,  mmap_mode='r')
        y           = np.load(label_path, mmap_mode='r')
        scaler      = _load_scaler(scaler_path, scaler_path_pt)
        groups      = np.load(group_path) if regression else None
        if groups is not None:
            if not (len(X_processed) == len(y) == len(groups)):
                raise RuntimeError(
                    f"cache length mismatch X={len(X_processed)} y={len(y)} "
                    f"groups={len(groups)} for stem {stem} - delete the cache dir.")
            _assert_groups_canonical(groups, cur_nstates, cur_npass)
        return X_processed, y, scaler, groups

    hit = _try_reuse()
    if hit is not None:
        return hit

    # ── Slow path UNDER a per-stem LOCK (audit R7.1 P5): serialise concurrent
    # builds so two processes/threads never race on temp files or on Windows'
    # mmap-vs-os.replace. Re-check inside the lock (another builder may have just
    # finished), then build and publish arrays + scaler + digest sidecar. ──
    with _CacheStemLock(os.path.join(cache_dir, f"cache_{stem}.lock")):
        hit = _try_reuse()
        if hit is not None:
            return hit
        dof_str = "_".join(map(str, config['dofs']))
        print(f"  [CACHE MISS] Processing '{config['method']}' data (DOFs: {dof_str}"
              f"{', regression' if regression else ''})...")

        X_raw, y_raw, groups_raw = load_ttbi_dataset(
            filepath=dataset_name,
            requested_dofs=config['dofs'],
            n_passages=LOAD_N_PASSAGES,   # protocol constant (hashed)
            target_supports=config.get('target_supports') if regression else None,
            bearing_targets=config.get('bearing_targets') if regression else None,
            bearing_max=config.get('bearing_max') if regression else None,
        )

        # Optional load-time sensor noise (applied to the RAW signals, before PAA).
        if config.get('sensor_noise'):
            X_raw = _inject_sensor_noise(X_raw, config['dofs'], config['sensor_noise'])

        # Canonical train partition for leak-free scaler fitting (seed 42,
        # grouped + family-stratified; dataset_name locates the state table).
        canonical_train_idx, _ = canonical_train_val_split(
            len(y_raw), groups_raw if regression else None,
            dataset_name=dataset_name if regression else None)

        preprocessor = TTBIPreprocessor(method=config['method'],
                                        n_segments=N_SEGMENTS)  # protocol constant
        X_processed  = preprocessor.transform(
            X_raw, fit_scaler=True, fit_indices=canonical_train_idx)

        if regression:
            y_out = y_raw.astype(np.float32)
        else:
            disc  = config.get('discretization', 1)
            y_out = np.round(y_raw / disc).astype(int)

        # Persist arrays ATOMICALLY (audit R7 P3). Under the lock, so no concurrent
        # builder is mmap-reading these while we os.replace them.
        _atomic_np_save(feat_path,  X_processed)
        _atomic_np_save(label_path, y_out)
        _atomic_np_save(group_path, np.asarray(groups_raw, dtype=np.int64))
        _save_scaler(preprocessor.scaler, scaler_path)

        # Provenance sidecar written LAST (audit R4/R7): source provenance + the
        # SHA-256 of every cache artifact (incl. the scaler). Written last so the
        # SET is effectively atomic — a crash before this leaves NO prov and the
        # partial cache is rebuilt.
        if regression:
            sc_file = scaler_path if os.path.exists(scaler_path) else scaler_path_pt
            prov = {"source": cur_src,
                    "artifacts": {"feat":   _sha256_file(feat_path),
                                  "labels": _sha256_file(label_path),
                                  "groups": _sha256_file(group_path),
                                  "scaler": _sha256_file(sc_file)}}
            _atomic_write_json(prov_path, prov)

        print(f"  [CACHE SAVED] -> {cache_dir}")

        X_processed = np.load(feat_path,  mmap_mode='r')
        y           = np.load(label_path, mmap_mode='r')
        scaler      = preprocessor.scaler
        groups      = np.asarray(groups_raw, dtype=np.int64) if regression else None
        return X_processed, y, scaler, groups


# ──────────────────────────────────────────────────────────────────────────────
# Internal scaler I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_scaler(scaler: object, pkl_path: str) -> None:
    """Save a scaler ATOMICALLY (temp + os.replace, audit R7 P3), routing to
    torch.save for nn.Module scalers."""
    if isinstance(scaler, torch.nn.Module) or torch.is_tensor(scaler):
        pt_path = pkl_path.replace('.pkl', '.pt')
        tmp = _unique_tmp(pt_path)
        torch.save(scaler, tmp); os.replace(tmp, pt_path)
        print(f"  [Save] PyTorch scaler -> {pt_path}")
    elif scaler is None:
        # Write a sentinel so the fast-path existence check still passes
        tmp = _unique_tmp(pkl_path)
        with open(tmp, 'w') as f:
            f.write("NO_SCALER_USED")
        os.replace(tmp, pkl_path)
        print("  [Warning] No scaler produced by this preprocessor.")
    else:
        tmp = _unique_tmp(pkl_path)
        joblib.dump(scaler, tmp); os.replace(tmp, pkl_path)
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
