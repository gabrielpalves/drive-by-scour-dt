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
import math
import os
import pickle
from pathlib import Path
import re
import threading
import time

import joblib
import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split

from core.campaign_contract import (
    EXPECTED_CHANNEL_SCHEMA_ID,
    EXPECTED_GENERATION_BEHAVIOR_VERSION,
    EXPECTED_GEN_SCHEMA,
    EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    EXPECTED_RAIL_END_CLEARANCE_M,
    campaign_stage_contract,
    generation_config_expectations,
)
from core.environment import (
    load_environment_lock,
    matlab_environment_descriptor,
)
from core.generation_state_contract import (
    STATE_DATA_FIELDS,
    STATE_TOP_LEVEL_FIELDS,
    require_canonical_state_names,
    require_exact_fields,
    validate_bearing_fixity,
    validate_contact_log,
    validate_raw_metadata,
)
from core.source_provenance import (
    generator_source_root,
    python_runtime_source_root,
)


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
_UNSET = object()


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
# event). The solver couples wheel and rail BILATERALLY and has no
# separation/re-contact state. Every positive reaction is therefore an
# out-of-domain tensile artifact, not a simulated physical contact event.
# The fixed two-tier gate is a prospectively source-locked engineering admissibility
# envelope, not a literature-validated separation criterion:
#   * TOLERATED (logged, reported): bounded tensile artifact — peak tension a
#     limited fraction of the ~118 kN static wheel load, on a tiny fraction of
#     the path. The historical observations used to calibrate the envelope
#     were:
#       - s23_all4 state 24 (60% scour + FRA-4 + track damage + poly OOR):
#         6.4 kN (5.4% static) on 0.042% of samples.
#       - s15_track state 244 (50%/13% scour + track damage): 13.4 kN (11.4%
#         static) on 0.063% of samples — ONE sample at dt=1 ms, on the track
#         portion (off-bridge).
#   * INADMISSIBLE: tension beyond 20% of static (24 kN, ~1.8x
#     the worst observed event) OR sustained tension (> 0.2% of path samples)
#     OR non-finite. The known true regressions sit far above: the R3
#     profile-seam bug gave 170 kN (144% of static); wheel flats exceed the
#     uplift threshold 12-38x. Watch item unchanged: an event of tens of kN
#     or sustained requires revisiting the model/domain decision
#     (unilateral contact vs explicit exclusion), not another threshold raise.
# A separate exhaustive qualification gate must demonstrate time-step and
# waveform closure for this finite numerical design before dispatch.
# The generator (A00 F_CONTACT_TOL_N / F_CONTACT_FRACTOL) enforces the SAME
# two-tier rule per passage at generation time; values must stay identical.
_CONTACT_F_TOL_N  = 24000.0  # INADMISSIBLE above this peak tension [N] (20% static)
_CONTACT_FRAC_TOL = 0.002    # INADMISSIBLE above this path fraction
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
_EXPECTED_GEN_SCHEMA = EXPECTED_GEN_SCHEMA
_EXPECTED_CHANNEL_SCHEMA_ID = EXPECTED_CHANNEL_SCHEMA_ID
_EXPECTED_GENERATION_BEHAVIOR_VERSION = (
    EXPECTED_GENERATION_BEHAVIOR_VERSION
)
_ENVIRONMENT_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / 'environment'
    / 'campaign-py313-cu128.json'
)
_CAMPAIGN_ENVIRONMENT_LOCK = load_environment_lock(_ENVIRONMENT_LOCK_PATH)
_EXPECTED_MATLAB_ENVIRONMENT = (
    _CAMPAIGN_ENVIRONMENT_LOCK['spec']['matlab_environment']
)
_EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR = matlab_environment_descriptor(
    _EXPECTED_MATLAB_ENVIRONMENT
)
_EXPECTED_MATLAB_ENVIRONMENT_SHA256 = (
    _CAMPAIGN_ENVIRONMENT_LOCK['spec']['matlab_environment_sha256']
)
# Kept as a reviewer-readable compatibility symbol, but deliberately derived
# from the authenticated full environment descriptor rather than duplicated.
_EXPECTED_MATLAB_RELEASE = _EXPECTED_MATLAB_ENVIRONMENT['release']

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
CACHE_SCHEMA_TAG = "_gs9"  # R12/v8: physical8_v1 wheelset channels plus the
                           # R11 UID-stable split/cache provenance. Pre-gs9
                           # caches contain the legacy virtual-rail DOFs 3-4.


_PHYSICAL8_DOF_SOURCE = {
    0: ('AcelPrimVag',         0),
    1: ('AcelPrimVag',         1),
    2: ('AcelPrimVag',         2),
    3: ('AcelWheelsetPrimVag', 0),
    4: ('AcelWheelsetPrimVag', 1),
    5: ('PitchPrimVag',        0),
    6: ('PitchPrimVag',        1),
    7: ('PitchPrimVag',        2),
}


def _resolve_dof_source(channel_schema_id: str) -> dict[int, tuple[str, int]]:
    """Resolve deployed DOFs only after authenticating the manifest schema."""

    if channel_schema_id != _EXPECTED_CHANNEL_SCHEMA_ID:
        raise RuntimeError(
            f"unsupported channel_schema_id={channel_schema_id!r}; expected "
            f"{_EXPECTED_CHANNEL_SCHEMA_ID!r}. Regenerate from the reviewed "
            "physical8_v1 source instead of guessing a channel mapping."
        )
    return _PHYSICAL8_DOF_SOURCE


# ── Feature A (2026-07-19): state families + stratified grouped split ────────
# The generator (A00) tags every state with an explicit FAMILY identity and
# ships it in damage_states.mat; the split below stratifies on it so that every
# family — and every (family, target) anchor stratum — is guaranteed to land in
# train, inner-val AND outer-test. This replaces the random GroupShuffleSplit,
# whose anchor coverage was only probabilistic (audit R6 C8C had a loud WARNING
# for the empty-probe case; now it is a structural guarantee).
STATE_FAMILIES = ('target_healthy', 'scour_only', 'bearing_only',
                  'nuisance_only', 'joint')
# Deterministic per-stratum assignment pattern = EXACTLY 60/20/20.  R11 applies
# it to a seeded permutation of lexicographically sorted *semantic UIDs*, never
# row/DC indices.  Therefore corresponding L60 states retain their partition
# after a rung inserts bearing-only or nuisance-only rows.
STRATIFY_PATTERN = ('train', 'test', 'val', 'train', 'train')
N_JOINT_SEV_BINS = 3     # joint-block severity bins (max normalized head)


def split_protocol() -> dict:
    """The split policy as DATA, for the protocol hash (core/protocol.py).

    Every entry is a constant `canonical_grouped_splits` itself uses — this
    function must never describe anything the code does not read."""
    return {
        "splitter":   "semantic-UID-stable family-STRATIFIED deterministic "
                      "split, grouped by generated damage STATE",
        "layout":     "3-way train/inner-val/outer-test; inner-val = HPO+selection, "
                      "outer-test = reported metrics only",
        "seed":       SPLIT_SEED,
        "test_frac":  SPLIT_TEST_FRAC,
        "val_frac":   SPLIT_VAL_FRAC,
        "pattern":    list(STRATIFY_PATTERN),
        "identity":   "StateUID (row/DC-independent); sorted inside each stratum "
                      "before the seeded permutation",
        "strata":     "family | anchor family+target+level | "
                      "joint+latent_crack_on+scour_severity_bin",
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
        "under the exact R11 source-digests-v2 scope; re-read current identity "
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
        3  Wheel1_Vert*      <- AcelWheelsetPrimVag[0]
        4  Wheel2_Vert*      <- AcelWheelsetPrimVag[1]
        5  CarBody_Pitch      <- PitchPrimVag[0] (angular velocity)
        6  FrontBogie_Pitch   <- PitchPrimVag[1] (angular velocity)
        7  RearBogie_Pitch    <- PitchPrimVag[2] (angular velocity)

    ``Wheel1_Vert`` and ``Wheel2_Vert`` are frozen public identifiers. Under
    ``physical8_v1`` they carry the idealized model-predicted constrained-
    wheelset acceleration along the moving contact coordinate, used as an
    axle-box response proxy. It is not an instrument model or a measured
    axle-box signal. The legacy Eulerian rail-field diagnostic remains stored
    in ``AcelRodaPrimVag`` but is not a deployed DOF.

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

    # Legacy single-output files do not have case_info provenance. They may use
    # only the current deployed mapping; multi-output production below resolves
    # this mapping from its authenticated case_info.channel_schema_id.
    dof_source = _resolve_dof_source(_EXPECTED_CHANNEL_SCHEMA_ID)

    # ── Multi-output mode: per-pier scour vector labels ───────────────────────
    if target_supports is not None:
        return _load_multi_output(dataset_path, requested_dofs, n_passages,
                                  target_supports,
                                  bearing_targets=bearing_targets,
                                  bearing_max=bearing_max)

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    for damage_label in range(61):
        filename = f"{damage_label + 1:04d}.mat"
        filepath_ = os.path.join(dataset_path, filename)

        try:
            mat         = sio.loadmat(filepath_, mat_dtype=True)
            data_struct = mat['data'][0, 0]

            available   = data_struct['AcelPrimVag'].shape[1]
            to_load     = min(n_passages, available)

            for p in range(to_load):
                channels = []
                for dof in requested_dofs:
                    field, row = dof_source[dof]
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
    bearing_targets: list | None = None,
    bearing_max:     float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-output loader: scan all NNNN.mat, read the scour (+ bearing) VECTORS.

    Label layout: [scour% at target_supports]  (Stage 0), or, when
    `bearing_targets` is given (Stage 1), followed by the bearing heads:
        [scour_1..scour_S, bearing_1..bearing_B]
    * Scour   = data.scour_vector at the (1-based) `target_supports`, x100 (%).
    * Bearing = data.bearing_fixity at the requested targets ('left'->0,
      'right'->1), multiplied by 100. This is the dimensionless nominal
      end-restraint coordinate, not a seized-bearing or material-damage
      percentage. `bearing_vector` stores the corresponding absolute spring
      stiffness for traceability but is not the current learning label.

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
    manifest_generation = _validate_campaign_generation_metadata(dataset_path)
    dof_source = _resolve_dof_source(
        manifest_generation["channel_schema_id"]
    )
    manifest_release = manifest_generation["matlab_release"]
    manifest_qualification = manifest_generation[
        "release_qualification_run"
    ]
    # Completion marker must EXIST and its CONTENT must match the manifest (audit
    # R7 P3): A00 writes the schema + fingerprint into it, so a stale marker from a
    # different run (or a hand-written empty file) is rejected, not trusted.
    mk = list(_read_completion_marker(dataset_path))
    # marker = schema, fingerprint, ROOT source-digest (audit R7.1 P4).
    if mk[:2] != [exp_schema, exp_fp]:
        raise RuntimeError(
            f"{dataset_path}: _GENERATION_COMPLETE content {mk[:3]} != manifest "
            f"[{exp_schema!r}, {exp_fp!r}, <root-digest>] — stale/foreign/old "
            f"marker (exactly three nonempty lines are required). Regenerate.")
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
    require_canonical_state_names(
        os.listdir(dataset_path),
        exp_states,
        dataset_path,
        error_type=RuntimeError,
    )
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
    # Bounded tensile-artifact incidence across the dataset (2026-07-19):
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
        loaded_state = sio.loadmat(fp, mat_dtype=True)
        require_exact_fields(
            (
                name
                for name in loaded_state
                if not str(name).startswith("__")
            ),
            STATE_TOP_LEVEL_FIELDS,
            fname,
            error_type=RuntimeError,
        )
        if 'data' not in loaded_state:
            raise KeyError(f"{fname}: no data payload.")
        data_struct = loaded_state['data'][0, 0]
        names = data_struct.dtype.names or ()
        require_exact_fields(
            names,
            STATE_DATA_FIELDS,
            f"{fname}: data",
            error_type=RuntimeError,
        )
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
        _validate_state_generation_metadata(
            loaded_state,
            data_struct,
            names,
            manifest_generation,
            expected_schema=exp_schema,
            expected_fingerprint=exp_fp,
            expected_state_uid=state_table["state_uid"][idx],
            expected_state_seed_id=int(state_table["state_seed_id"][idx]),
            expected_random_stream_schedule_version=
                state_table["random_stream_schedule_version"],
            owner=fname,
        )
        frelease = (str(np.ravel(data_struct['matlab_release'])[0])
                    if 'matlab_release' in names else None)
        fcampaign_release = (
            str(np.ravel(data_struct['campaign_matlab_release'])[0])
            if 'campaign_matlab_release' in names else None
        )
        fqualification = (
            _coerce_matlab_logical(
                data_struct['release_qualification_run'],
                f"{fname}: data.release_qualification_run",
            )
            if 'release_qualification_run' in names else None
        )
        if (frelease != manifest_release
                or fcampaign_release
                != manifest_generation["campaign_matlab_release"]
                or fqualification is not manifest_qualification):
            raise RuntimeError(
                f"{fname}: per-state MATLAB release/qualification provenance "
                f"({frelease!r}, {fcampaign_release!r}, {fqualification!r}) "
                f"does not match the campaign manifest/policy "
                f"({manifest_release!r}, "
                f"{manifest_generation['campaign_matlab_release']!r}, "
                f"{manifest_qualification!r}). Regenerate; never mix or "
                f"restamp state files.")
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
        # R11 semantic identity / CRN alignment.  Numbered filenames are merely
        # storage rows; scientific pairing is keyed by the generator-issued UID
        # and stream.  Require every per-file value and compare it with the exact
        # row of damage_states.mat before reading a single signal.
        identity_fields = (
            'state_uid', 'state_seed_id', 'latent_bearing_fixity',
            'random_stream_schedule_version', 'state_named_stream_seed_id',
            'passage_named_stream_seed_id', 'latent_crack_on', 'crack_on',
            'bearing_fixity',
            'scour_supports',
        )
        missing_identity = [
            field for field in identity_fields if field not in names
        ]
        if missing_identity:
            raise RuntimeError(
                f"{fname}: missing semantic-state field(s) {missing_identity} — "
                "pre-R11/partial payload; regenerate."
            )
        uid_values = np.ravel(np.asarray(data_struct['state_uid']))
        if (
            uid_values.size != 1
            or not isinstance(uid_values[0], (str, np.str_))
        ):
            raise RuntimeError(f"{fname}: state_uid must be exactly one text scalar.")
        file_uid = str(uid_values[0])
        if file_uid != state_table['state_uid'][idx]:
            raise RuntimeError(
                f"{fname}: state_uid {file_uid!r} != table row UID "
                f"{state_table['state_uid'][idx]!r} — renamed/swapped state file."
            )
        seed_values = np.ravel(
            np.asarray(data_struct['state_seed_id'], dtype=np.float64)
        )
        if (
            seed_values.size != 1
            or not np.isfinite(seed_values[0])
            or seed_values[0] != np.floor(seed_values[0])
            or not (1 <= seed_values[0] <= np.iinfo(np.uint32).max)
        ):
            raise RuntimeError(
                f"{fname}: state_seed_id must be one positive uint32-compatible "
                "integer."
            )
        file_seed_id = int(seed_values[0])
        if file_seed_id != int(state_table['state_seed_id'][idx]):
            raise RuntimeError(
                f"{fname}: state_seed_id {file_seed_id} != table row stream "
                f"{int(state_table['state_seed_id'][idx])} — misaligned state."
            )
        file_schedule = _matlab_text_scalar(
            data_struct["random_stream_schedule_version"],
            f"{fname}: data.random_stream_schedule_version",
        )
        file_state_named = np.asarray(
            data_struct["state_named_stream_seed_id"], dtype=np.float64
        )
        file_passage_named = np.asarray(
            data_struct["passage_named_stream_seed_id"], dtype=np.float64
        )
        if (
            file_schedule
            != state_table["random_stream_schedule_version"]
            or file_state_named.shape
            != (1, len(state_table["state_stream_names"]))
            or file_passage_named.shape
            != (
                exp_npass,
                len(state_table["passage_stream_names"]),
            )
            or not np.all(np.isfinite(file_state_named))
            or not np.all(np.isfinite(file_passage_named))
            or np.any(file_state_named < 1)
            or np.any(file_passage_named < 1)
            or np.any(file_state_named > np.iinfo(np.uint32).max)
            or np.any(file_passage_named > np.iinfo(np.uint32).max)
            or not np.all(file_state_named == np.floor(file_state_named))
            or not np.all(file_passage_named == np.floor(file_passage_named))
            or not np.array_equal(
                file_state_named.astype(np.uint32).ravel(),
                state_table["state_named_stream_seed_id"][idx],
            )
            or not np.array_equal(
                file_passage_named.astype(np.uint32),
                state_table["passage_named_stream_seed_id"][idx],
            )
        ):
            raise RuntimeError(
                f"{fname}: named RNG schedule/stream IDs are malformed or "
                "differ from damage_states.mat."
            )
        file_latent_bearing = np.ravel(
            np.asarray(data_struct['latent_bearing_fixity'], dtype=float)
        )
        if (
            file_latent_bearing.shape != (2,)
            or not np.all(np.isfinite(file_latent_bearing))
            or np.any(file_latent_bearing < 0.0)
            or np.any(file_latent_bearing >= 1.0)
            or not np.array_equal(
                file_latent_bearing,
                state_table['latent_bearing_fixity'][idx],
            )
        ):
            raise RuntimeError(
                f"{fname}: latent_bearing_fixity is malformed or differs from "
                "the damage_states.mat row."
            )
        file_latent_crack = _strict_binary_vector(
            data_struct['latent_crack_on'],
            f"{fname}: data.latent_crack_on",
        )
        file_active_crack = _strict_binary_vector(
            data_struct['crack_on'], f"{fname}: data.crack_on"
        )
        if file_latent_crack.size != 1 or file_active_crack.size != 1:
            raise RuntimeError(
                f"{fname}: latent_crack_on/crack_on must be scalar logicals."
            )
        if (
            bool(file_latent_crack[0])
            != bool(state_table['latent_crack_on'][idx])
            or bool(file_active_crack[0])
            != bool(state_table['crack_on'][idx])
        ):
            raise RuntimeError(
                f"{fname}: latent/active crack flags differ from the "
                "damage_states.mat row."
            )
        file_supports = np.ravel(
            np.asarray(data_struct['scour_supports'], dtype=float)
        )
        if (
            not np.all(np.isfinite(file_supports))
            or not np.all(file_supports == np.floor(file_supports))
            or file_supports.astype(int).tolist()
            != [int(value) for value in target_supports]
        ):
            raise RuntimeError(
                f"{fname}: scour_supports {file_supports.tolist()} != registered "
                f"targets {[int(value) for value in target_supports]}."
            )
        # Scour label sanity (audit R6 C4 + R7 P4). Validate the FULL vector's
        # finiteness (a NaN in ANY support — even one we don't select — signals a
        # corrupt solve), then bound the SELECTED labels to the physical scour
        # ceiling [0, dano_max%] (NOT merely [0, 100]).
        # Range-check the ENTIRE scour_vector, not just the selected supports
        # (audit R7.1 P5): a -10% or 80% value in an UN-selected support still
        # signals a corrupt solve. Finiteness + [0, dano_max%] on the whole vector.
        full_scour = np.ravel(data_struct['scour_vector']).astype(float)
        full_scour_pct = full_scour * 100.0
        if (
            full_scour.shape != state_table['damage_states'][idx].shape
            or not np.array_equal(
                full_scour, state_table['damage_states'][idx]
            )
        ):
            raise RuntimeError(
                f"{fname}: scour_vector does not exactly match row {idx + 1} "
                "of DamageStates — renamed/misaligned state payload."
            )
        if (not np.all(np.isfinite(full_scour))
                or np.any(full_scour_pct < 0.0)
                or np.any(full_scour_pct > scour_max_pct + _SCOUR_EPS)):
            raise RuntimeError(
                f"{fname}: scour_vector {full_scour.tolist()} non-finite or outside "
                f"[0, {scour_max_pct:.3g}] % (dano_max) in some support — corrupt "
                f"or foreign state. Regenerate this state.")
        slabel = full_scour[tgt0] * 100.0
        full_fix = np.ravel(data_struct['bearing_fixity']).astype(float)
        if (
            full_fix.shape != (2,)
            or not np.all(np.isfinite(full_fix))
            or np.any(full_fix < 0.0)
            or np.any(full_fix >= 1.0)
            or not np.array_equal(full_fix, state_table['bearing_fixity'][idx])
        ):
            raise RuntimeError(
                f"{fname}: bearing_fixity is malformed or does not exactly "
                f"match row {idx + 1} of BearingFixity."
            )
        if bidx is not None:
            # R7 requires the fixity label (audit R7 P4): the legacy raw-k_r
            # `bearing_vector` fallback is dropped — no r6/r7 data lacks fixity, so
            # a missing bearing_fixity means a foreign/old file, not a valid one.
            if 'bearing_fixity' not in names:
                raise KeyError(
                    f"{fname}: bearing heads requested but no data.bearing_fixity "
                    f"— the R7 schema requires the fixity label. Regenerate with "
                    f"the current A00 (bearing_mode='target').")
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
        validate_raw_metadata(
            {field: data_struct[field] for field in STATE_DATA_FIELDS},
            n_passages=exp_npass,
            bridge_length_m=manifest_generation["L_bridge_m"],
            owner=f"{fname}: data",
            error_type=RuntimeError,
        )
        raw_fmt = True

        # Contact-validity gate (audit R5; TWO-TIER since 2026-07-19 — see the
        # _CONTACT_F_TOL_N comment at the top of this module). A bounded
        # tensile artifact is TOLERATED and counted for the dataset summary;
        # peak tension beyond 20% of the static wheel load OR sustained
        # tension is inadmissible and aborts — never silently censored (MNAR).
        # contact_log must also be present, exactly (Npass, 4), and finite.
        if 'contact_log' not in names:
            raise KeyError(f"{fname}: no contact_log (audit-schema file must "
                           f"carry it) - regenerate with the current A00.")
        clog = validate_contact_log(
            data_struct['contact_log'],
            n_passages=exp_npass,
            max_tension_N=_CONTACT_F_TOL_N,
            max_tension_fraction=_CONTACT_FRAC_TOL,
            owner=f"{fname}: data.contact_log",
            error_type=RuntimeError,
        )
        available = data_struct['AcelPrimVag'].shape[1]
        npass_here = exp_npass                       # manifest count, now guaranteed > 0
        # EXACT count (audit R6 C4): the old one-sided `available < npass_here`
        # let a file with MORE passages than the manifest declares load its extra
        # passages, which the contact gate below (spanning only npass_here rows)
        # never audited. Require an exact match and iterate exactly npass_here.
        if available != npass_here:
            raise RuntimeError(
                f"{fname}: {available} signal passages but manifest says "
                f"passages_per_state={npass_here} — count mismatch (partial write "
                f"or foreign file). Regenerate this state.")
        # All four stored channel fields are mandatory and must have exactly npass_here
        # passages (audit R7.1 P5) — regardless of which DOFs this call requests. A
        # complete r7 state carries every channel; a missing PitchPrimVag (etc.) is
        # a partial/foreign file even if the current study doesn't read it.
        # AcelRodaPrimVag remains a solver diagnostic; AcelWheelsetPrimVag is
        # the physical8_v1 source for deployed DOFs 3-4. Both stay mandatory.
        for _field in (
            'AcelPrimVag', 'AcelRodaPrimVag',
            'AcelWheelsetPrimVag', 'PitchPrimVag',
        ):
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
                f"This exceeds the source-locked bilateral-model admissibility "
                f"envelope; inspect the generator/solver before training and "
                f"do not raise the threshold post hoc.")
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
            # P6) — not just the requested DOFs — 2-D, exact-row and
            # DimAcel-long.
            expected_rows = {
                "AcelPrimVag": 3,
                "AcelRodaPrimVag": 4,
                "AcelWheelsetPrimVag": 4,
                "PitchPrimVag": 3,
            }
            for _src in (
                'AcelPrimVag', 'AcelRodaPrimVag',
                'AcelWheelsetPrimVag', 'PitchPrimVag',
            ):
                arr = np.asarray(data_struct[_src][0, p])
                if arr.ndim != 2 \
                        or arr.shape[0] != expected_rows[_src] \
                        or arr.shape[1] != int(dim_acel) \
                        or not np.all(np.isfinite(arr)):
                    raise RuntimeError(
                        f"{fname}: passage {p} channel {_src} shape {arr.shape} not "
                        f"({expected_rows[_src]}, DimAcel={int(dim_acel)}) or "
                        "non-finite — corrupt state.")
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
        print(f"  [contact] bounded tensile-artifact tier: {n_tension_passages}/"
              f"{X.shape[0]} passages ({n_tension_passages / X.shape[0]:.2%}) "
              f"show brief wheel unloading past zero (worst "
              f"{worst_tension_N:.3g} N = "
              f"{worst_tension_N / 1.18e5:.1%} of static, worst path fraction "
              f"{worst_tension_frac:.3g}). All within the gate "
              f"(F <= {_CONTACT_F_TOL_N:.0f} N, frac <= {_CONTACT_FRAC_TOL}); "
              f"report this incidence with the results.")
    return X, y, groups


def _load_scalar_mat_struct(
    path: str,
    variable: str,
    owner: str,
    *,
    missing_ok: bool = False,
):
    """Load exactly one scalar MATLAB struct from one regular local MAT file."""
    if not os.path.lexists(path):
        if missing_ok:
            return None
        raise RuntimeError(f"{owner}: {os.path.basename(path)} is missing.")
    if os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError(
            f"{owner}: {os.path.basename(path)} must be a regular local file."
        )
    try:
        loaded = sio.loadmat(path, mat_dtype=True)
    except Exception as exc:
        raise RuntimeError(
            f"{owner}: {os.path.basename(path)} cannot be parsed."
        ) from exc
    if variable not in loaded:
        raise RuntimeError(
            f"{owner}: {os.path.basename(path)} lacks {variable} struct."
        )
    raw = np.asarray(loaded[variable])
    if raw.size != 1 or raw.dtype.names is None:
        raise RuntimeError(
            f"{owner}: {variable} must be exactly one scalar MATLAB struct "
            f"(got shape {raw.shape})."
        )
    return np.ravel(raw)[0]


def _read_manifest(dataset_path: str):
    """(n_states, Npass, gen_schema, gen_fingerprint) from case_info.mat, or
    (None, None, None, None) if there is no manifest (audit R4 2026-07-17).
    These drive the loader's strict completeness/provenance checks."""
    ci = os.path.join(dataset_path, 'case_info.mat')
    owner = f"{dataset_path}: case_info"
    info = _load_scalar_mat_struct(
        ci, "case_info", owner, missing_ok=True
    )
    if info is None:
        return None, None, None, None
    names = info.dtype.names or ()
    n_states = _required_mat_int(info, names, "n_states", owner)
    n_passages = _required_mat_int(
        info, names, "passages_per_state", owner
    )
    if n_states <= 0 or n_passages <= 0:
        raise RuntimeError(
            f"{owner}: n_states and passages_per_state must be positive "
            "integer scalars."
        )
    schema = _required_mat_text(info, names, "gen_schema", owner)
    fingerprint = _required_mat_text(
        info, names, "gen_fingerprint", owner
    )
    if schema != _EXPECTED_GEN_SCHEMA:
        raise RuntimeError(
            f"{owner}: gen_schema={schema!r} is not the canonical current "
            f"schema {_EXPECTED_GEN_SCHEMA!r}."
        )
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise RuntimeError(
            f"{owner}: gen_fingerprint must be one lowercase SHA-256."
        )
    return n_states, n_passages, schema, fingerprint


def _coerce_matlab_logical(value, label: str) -> bool:
    """Parse one MATLAB logical/numeric scalar without truthy-string coercion."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise RuntimeError(f"{label} must be one scalar logical (got shape {arr.shape}).")
    item = np.ravel(arr)[0]
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, (np.integer, int, np.floating, float)):
        number = float(item)
        if np.isfinite(number) and number in (0.0, 1.0):
            return bool(int(number))
    raise RuntimeError(
        f"{label} must be MATLAB logical false/true (or numeric 0/1), "
        f"got {item!r}.")


def _required_mat_text(struct, names: tuple, field: str, owner: str) -> str:
    """Read one required scalar MATLAB text field without truthy coercion."""
    if field not in names:
        raise RuntimeError(f"{owner}.{field} is missing.")
    arr = np.asarray(struct[field])
    if arr.size != 1:
        raise RuntimeError(
            f"{owner}.{field} must be one text scalar (got shape {arr.shape})."
        )
    value = str(np.ravel(arr)[0])
    if not value or "\x00" in value or "\r" in value:
        raise RuntimeError(
            f"{owner}.{field} must be nonempty canonical text."
        )
    return value


def _required_mat_int(struct, names: tuple, field: str, owner: str) -> int:
    """Read one finite integer-valued MATLAB scalar."""
    if field not in names:
        raise RuntimeError(f"{owner}.{field} is missing.")
    arr = np.asarray(struct[field])
    if arr.size != 1:
        raise RuntimeError(
            f"{owner}.{field} must be one scalar (got shape {arr.shape})."
        )
    item = np.ravel(arr)[0]
    if isinstance(item, (np.bool_, bool)) or not isinstance(
        item, (np.integer, int, np.floating, float)
    ):
        raise RuntimeError(f"{owner}.{field} must be a numeric integer scalar.")
    value = float(item)
    if not np.isfinite(value) or not value.is_integer():
        raise RuntimeError(f"{owner}.{field} must be a finite integer.")
    return int(value)


def _required_mat_number(
    struct,
    names: tuple,
    field: str,
    owner: str,
) -> float:
    """Read one finite MATLAB numeric scalar without accepting logicals."""

    if field not in names:
        raise RuntimeError(f"{owner}.{field} is missing.")
    arr = np.asarray(struct[field])
    if arr.size != 1:
        raise RuntimeError(
            f"{owner}.{field} must be one scalar (got shape {arr.shape})."
        )
    item = np.ravel(arr)[0]
    if isinstance(item, (np.bool_, bool)) or not isinstance(
        item, (np.integer, int, np.floating, float)
    ):
        raise RuntimeError(f"{owner}.{field} must be one numeric scalar.")
    value = float(item)
    if not np.isfinite(value):
        raise RuntimeError(f"{owner}.{field} must be finite.")
    return value


def _required_mat_logical(
    struct,
    names: tuple,
    field: str,
    owner: str,
) -> bool:
    if field not in names:
        raise RuntimeError(f"{owner}.{field} is missing.")
    return _coerce_matlab_logical(struct[field], f"{owner}.{field}")


def _strict_json_object(text: str, owner: str) -> dict:
    """Parse one finite duplicate-free JSON object."""

    def _pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def _constant(value):
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{owner} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} must contain one JSON object.")
    return value


def _json_values_equal(actual, expected) -> bool:
    """JSON equality that never treats true/false as the numbers 1/0."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return (
            np.isfinite(float(actual))
            and np.isfinite(float(expected))
            and float(actual) == float(expected)
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_values_equal(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_values_equal(actual[key], expected[key])
            for key in expected
        )
    return type(actual) is type(expected) and actual == expected


def _parse_support_text(value: str, owner: str) -> list[int]:
    match = re.fullmatch(r"\[(\d+(?: \d+)*)?\]", value)
    if match is None:
        raise RuntimeError(
            f"{owner} must use canonical MATLAB support text such as '[2 3]'."
        )
    return [int(token) for token in match.group(1).split()] if match.group(1) else []


def _read_manifest_generation_metadata(dataset_path: str) -> dict | None:
    """Read the complete R11 generator/environment/source attestation."""
    ci = os.path.join(dataset_path, "case_info.mat")
    owner = f"{dataset_path}: case_info"
    info = _load_scalar_mat_struct(
        ci, "case_info", owner, missing_ok=True
    )
    if info is None:
        return None
    names = info.dtype.names or ()
    return {
        "gen_fingerprint": _required_mat_text(
            info, names, "gen_fingerprint", owner
        ),
        "channel_schema_id": _required_mat_text(
            info, names, "channel_schema_id", owner
        ),
        "state_design_kind": _required_mat_text(
            info, names, "state_design_kind", owner
        ),
        "generation_config_json": _required_mat_text(
            info, names, "generation_config_json", owner
        ),
        "case_name": _required_mat_text(
            info, names, "case_name", owner
        ),
        "stage": _required_mat_text(
            info, names, "stage", owner
        ),
        "damage_mode": _required_mat_text(
            info, names, "damage_mode", owner
        ),
        "rail_end_clearance_m": _required_mat_number(
            info, names, "rail_end_clearance_m", owner
        ),
        "rail_end_clearance_decision_id": _required_mat_text(
            info, names, "rail_end_clearance_decision_id", owner
        ),
        "L_bridge_m": _required_mat_number(
            info, names, "L_bridge_m", owner
        ),
        "num_spans": _required_mat_int(
            info, names, "num_spans", owner
        ),
        "num_supports": _required_mat_int(
            info, names, "num_supports", owner
        ),
        "scour_supports": _parse_support_text(
            _required_mat_text(info, names, "scour_supports", owner),
            f"{owner}.scour_supports",
        ),
        "n_states": _required_mat_int(
            info, names, "n_states", owner
        ),
        "passages_per_state": _required_mat_int(
            info, names, "passages_per_state", owner
        ),
        "scour_dano_max_frac": _required_mat_number(
            info, names, "scour_dano_max_frac", owner
        ),
        "family_counts": {
            "target_healthy": _required_mat_int(
                info, names, "n_target_healthy", owner
            ),
            "scour_only": _required_mat_int(
                info, names, "n_scour_only", owner
            ),
            "bearing_only": _required_mat_int(
                info, names, "n_bearing_only", owner
            ),
            "nuisance_only": _required_mat_int(
                info, names, "n_nuisance_only", owner
            ),
            "joint": _required_mat_int(
                info, names, "n_joint", owner
            ),
        },
        "bearing_mode": _required_mat_text(
            info, names, "bearing_mode", owner
        ),
        "bearing_label": _required_mat_text(
            info, names, "bearing_label", owner
        ),
        "use_crack_eov": _required_mat_logical(
            info, names, "use_crack_eov", owner
        ),
        "crack_draw": _required_mat_text(
            info, names, "crack_draw", owner
        ),
        "profile_mode": _required_mat_text(
            info, names, "profile_mode", owner
        ),
        "profile_draw": _required_mat_text(
            info, names, "profile_draw", owner
        ),
        "profile_jitter_sd_mm": _required_mat_number(
            info, names, "profile_jitter_sd_mm", owner
        ),
        "use_track_eov": _required_mat_logical(
            info, names, "use_track_eov", owner
        ),
        "track_draw": _required_mat_text(
            info, names, "track_draw", owner
        ),
        "track_L_app": _required_mat_number(
            info, names, "track_L_app", owner
        ),
        "track_L_after": _required_mat_number(
            info, names, "track_L_after", owner
        ),
        "use_oor_eov": _required_mat_logical(
            info, names, "use_oor_eov", owner
        ),
        "oor_flats_enabled": _required_mat_logical(
            info, names, "oor_flats_enabled", owner
        ),
        "use_signal_noise": _required_mat_logical(
            info, names, "use_signal_noise", owner
        ),
        "use_vehicle_variability": _required_mat_logical(
            info, names, "use_vehicle_variability", owner
        ),
        "use_speed_variability": _required_mat_logical(
            info, names, "use_speed_variability", owner
        ),
        "use_temp_variability": _required_mat_logical(
            info, names, "use_temp_variability", owner
        ),
        "generation_behavior_version": _required_mat_text(
            info, names, "generation_behavior_version", owner
        ),
        "matlab_release": _required_mat_text(
            info, names, "matlab_release", owner
        ),
        "campaign_matlab_release": _required_mat_text(
            info, names, "campaign_matlab_release", owner
        ),
        "release_qualification_run": _required_mat_logical(
            info, names, "release_qualification_run", owner
        ),
        "actual_matlab_environment_descriptor": _required_mat_text(
            info, names, "actual_matlab_environment_descriptor", owner
        ),
        "actual_matlab_environment_sha256": _required_mat_text(
            info, names, "actual_matlab_environment_sha256", owner
        ),
        "campaign_matlab_environment_descriptor": _required_mat_text(
            info, names, "campaign_matlab_environment_descriptor", owner
        ),
        "campaign_matlab_environment_sha256": _required_mat_text(
            info, names, "campaign_matlab_environment_sha256", owner
        ),
        "generator_source_root_sha256": _required_mat_text(
            info, names, "generator_source_root_sha256", owner
        ),
        "generator_source_digest_lines": _required_mat_text(
            info, names, "generator_source_digest_lines", owner
        ),
        "generator_source_file_count": _required_mat_int(
            info, names, "generator_source_file_count", owner
        ),
        "qualification_source_sha256": _required_mat_text(
            info, names, "qualification_source_sha256", owner
        ),
    }


def _read_manifest_release_metadata(
    dataset_path: str,
) -> tuple[str | None, bool | None]:
    """Compatibility view over the complete R11 generation attestation."""
    metadata = _read_manifest_generation_metadata(dataset_path)
    if metadata is None:
        return None, None
    return (
        metadata["matlab_release"],
        metadata["release_qualification_run"],
    )


def _validate_campaign_generation_metadata(
    dataset_path: str,
    *,
    expected_stage: str | None = None,
    expected_dataset: str | None = None,
    expected_target_supports: list[int] | None = None,
    expected_bearing_targets: list[str] | None | object = _UNSET,
) -> dict:
    """Validate coherent provenance and, when supplied, the rung contract."""
    metadata = _read_manifest_generation_metadata(dataset_path)
    if metadata is None:
        raise RuntimeError(
            f"{dataset_path}: case_info.mat is missing; no R11 provenance."
        )

    config_json = metadata["generation_config_json"]
    config_sha = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(config_sha, metadata["gen_fingerprint"]):
        raise RuntimeError(
            f"{dataset_path}: generation_config_json hashes to {config_sha}, "
            f"not case_info.gen_fingerprint={metadata['gen_fingerprint']}. "
            "The configuration attestation is corrupt or was restamped."
        )
    generation_config = _strict_json_object(
        config_json,
        f"{dataset_path}: case_info.generation_config_json",
    )
    dynamic_expected = {
        "schema": _EXPECTED_GEN_SCHEMA,
        "channel_schema_id": metadata["channel_schema_id"],
        "state_design_kind": metadata["state_design_kind"],
        "generation_behavior_version":
            metadata["generation_behavior_version"],
        "campaign_matlab_release":
            metadata["campaign_matlab_release"],
        "campaign_matlab_environment_sha256":
            metadata["campaign_matlab_environment_sha256"],
        "generator_source_root_sha256":
            metadata["generator_source_root_sha256"],
        "qualification_source_sha256":
            metadata["qualification_source_sha256"],
        "STAGE": metadata["stage"],
        "n_states": metadata["n_states"],
        "Npass": metadata["passages_per_state"],
        "rail_end_clearance_m": metadata["rail_end_clearance_m"],
        "rail_end_clearance_decision_id":
            metadata["rail_end_clearance_decision_id"],
    }
    dynamic_mismatches = {
        field: (generation_config.get(field), wanted)
        for field, wanted in dynamic_expected.items()
        if field not in generation_config
        or not _json_values_equal(generation_config[field], wanted)
    }
    if dynamic_mismatches:
        raise RuntimeError(
            f"{dataset_path}: hashed generation config disagrees with its "
            f"manifest/source contract: {dynamic_mismatches}."
        )

    sha_fields = (
        "actual_matlab_environment_sha256",
        "campaign_matlab_environment_sha256",
        "generator_source_root_sha256",
    )
    for field in sha_fields:
        if re.fullmatch(r"[0-9a-f]{64}", metadata[field]) is None:
            raise RuntimeError(
                f"{dataset_path}: case_info.{field} is not lowercase SHA-256."
            )

    # Actual MATLAB versions may differ between qualified PCs.  What remains
    # fail-closed is the provenance itself: each descriptor must authenticate
    # its SHA and the actual release must agree with its descriptor.  The
    # campaign descriptor is the immutable known-good reference carried by the
    # generation contract, not a live-host allow-list.
    descriptor_fields = tuple(sorted(_EXPECTED_MATLAB_ENVIRONMENT))
    parsed_descriptors: dict[str, dict[str, str]] = {}
    for prefix in ("actual", "campaign"):
        descriptor = metadata[f"{prefix}_matlab_environment_descriptor"]
        descriptor_sha = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(
            descriptor_sha,
            metadata[f"{prefix}_matlab_environment_sha256"],
        ):
            raise RuntimeError(
                f"{dataset_path}: {prefix} MATLAB environment descriptor/SHA "
                "is internally inconsistent."
            )
        rows = descriptor.split("\n")
        if any(row.count("=") != 1 for row in rows):
            raise RuntimeError(
                f"{dataset_path}: malformed {prefix} MATLAB descriptor."
            )
        parsed = dict(row.split("=", 1) for row in rows)
        if tuple(sorted(parsed)) != descriptor_fields or len(rows) != len(parsed):
            raise RuntimeError(
                f"{dataset_path}: {prefix} MATLAB descriptor fields differ "
                "from the provenance contract."
            )
        parsed_descriptors[prefix] = parsed
    if (
        re.fullmatch(r"R\d{4}[ab]", metadata["matlab_release"]) is None
        or parsed_descriptors["actual"]["release"]
        != metadata["matlab_release"]
    ):
        raise RuntimeError(
            f"{dataset_path}: actual MATLAB release disagrees with its "
            "authenticated descriptor."
        )

    digest_blob = metadata["generator_source_digest_lines"]
    digest_rows = digest_blob.split("\n")
    digest_names: list[str] = []
    for row in digest_rows:
        if row.count(":") != 1:
            raise RuntimeError(
                f"{dataset_path}: malformed generator source digest row "
                f"{row!r}."
            )
        name, digest = row.split(":", 1)
        parts = name.split("/")
        if (
            not name.startswith("scour_MATLAB/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in parts)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(
                f"{dataset_path}: unsafe/invalid generator digest row "
                f"{row!r}."
            )
        digest_names.append(name)
    if (
        digest_names != sorted(digest_names)
        or len(digest_names) != len(set(digest_names))
        or len(digest_names) != len(
            {name.casefold() for name in digest_names}
        )
        or len(digest_names) != metadata["generator_source_file_count"]
    ):
        raise RuntimeError(
            f"{dataset_path}: generator digest inventory is unsorted, "
            "duplicate/case-colliding, or disagrees with its file count."
        )
    recomputed_generator_root = hashlib.sha256(
        digest_blob.encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(
        recomputed_generator_root,
        metadata["generator_source_root_sha256"],
    ):
        raise RuntimeError(
            f"{dataset_path}: generator source digest lines/root disagree."
        )

    live_generator = generator_source_root(
        Path(__file__).resolve().parents[1]
    )
    live_mismatches = {}
    if not hmac.compare_digest(
        metadata["generator_source_root_sha256"],
        live_generator.sha256,
    ):
        live_mismatches["generator_source_root_sha256"] = (
            metadata["generator_source_root_sha256"],
            live_generator.sha256,
        )
    if metadata["generator_source_digest_lines"] != live_generator.digest_lines:
        live_mismatches["generator_source_digest_lines"] = (
            "<dataset digest lines>",
            "<live reviewed digest lines>",
        )
    if metadata["generator_source_file_count"] != live_generator.file_count:
        live_mismatches["generator_source_file_count"] = (
            metadata["generator_source_file_count"],
            live_generator.file_count,
        )
    if live_mismatches:
        raise RuntimeError(
            f"{dataset_path}: generator source attestation is internally "
            f"coherent but does not match the live reviewed MATLAB source: "
            f"{live_mismatches}. Regenerate from the converged source tree."
        )

    expected = {
        "channel_schema_id": _EXPECTED_CHANNEL_SCHEMA_ID,
        "generation_behavior_version":
            _EXPECTED_GENERATION_BEHAVIOR_VERSION,
        "campaign_matlab_release": _EXPECTED_MATLAB_RELEASE,
        "campaign_matlab_environment_descriptor":
            _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
        "campaign_matlab_environment_sha256":
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        "qualification_source_sha256": "PRODUCTION",
        "release_qualification_run": False,
        "rail_end_clearance_m": EXPECTED_RAIL_END_CLEARANCE_M,
        "rail_end_clearance_decision_id":
            EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    }
    mismatches = {
        field: (metadata.get(field), wanted)
        for field, wanted in expected.items()
        if metadata.get(field) != wanted
    }
    if mismatches:
        raise RuntimeError(
            f"{dataset_path}: R11 generator/environment/source attestation "
            f"does not match the reviewed production contract: {mismatches}. "
            "Regenerate from the converged bundle; qualification output and "
            "restamped/foreign data are never campaign input."
        )

    expected_state_design_kind = (
        "dense-scour-61x5-v1"
        if metadata["stage"] == "F40-S"
        else "five-family-multidamage-v2"
    )
    if metadata["state_design_kind"] != expected_state_design_kind:
        raise RuntimeError(
            f"{dataset_path}: state_design_kind="
            f"{metadata['state_design_kind']!r} is incompatible with stage "
            f"{metadata['stage']!r}; expected {expected_state_design_kind!r}."
        )

    if expected_stage is not None:
        contract = campaign_stage_contract(expected_stage)
        expected_dataset = (
            contract["dataset"]
            if expected_dataset is None
            else expected_dataset
        )
        if expected_dataset != contract["dataset"]:
            raise RuntimeError(
                f"{dataset_path}: requested dataset {expected_dataset!r} is "
                f"not the registered {expected_stage!r} dataset "
                f"{contract['dataset']!r}."
            )
        expected_targets = contract["learning"]["target_supports"]
        expected_bearings = contract["learning"]["bearing_targets"]
        if (
            expected_target_supports is not None
            and list(expected_target_supports) != expected_targets
        ):
            raise RuntimeError(
                f"{dataset_path}: requested target supports "
                f"{expected_target_supports!r} differ from the registered "
                f"{expected_targets!r} for {expected_stage}."
            )
        normalized_bearings = (
            list(expected_bearing_targets)
            if expected_bearing_targets is not _UNSET
            and expected_bearing_targets is not None
            else None
        )
        if (
            expected_bearing_targets is not _UNSET
            and normalized_bearings != expected_bearings
        ):
            raise RuntimeError(
                f"{dataset_path}: requested bearing targets "
                f"{normalized_bearings!r} differ from the registered "
                f"{expected_bearings!r} for {expected_stage}."
            )
        actual_basename = os.path.basename(
            os.path.normpath(os.path.abspath(dataset_path))
        )
        expected_manifest = {
            "case_name": expected_dataset,
            "stage": expected_stage,
            "state_design_kind": expected_state_design_kind,
            "damage_mode": contract["scenario"]["damage_mode"],
            "L_bridge_m": contract["geometry"]["L_bridge_m"],
            "num_spans": contract["geometry"]["num_spans"],
            "num_supports": contract["geometry"]["num_supports"],
            "scour_supports": contract["geometry"]["scour_supports"],
            "n_states": contract["sampling"]["n_states"],
            "passages_per_state":
                contract["sampling"]["passages_per_state"],
            "scour_dano_max_frac":
                contract["sampling"]["scour_dano_max_frac"],
            "family_counts": contract["sampling"]["family_counts"],
            "bearing_mode": contract["scenario"]["bearing_mode"],
            "bearing_label": "fixity_ratio",
            "use_crack_eov": contract["scenario"]["use_crack_eov"],
            "crack_draw": "per_state",
            "profile_mode": contract["scenario"]["profile_mode"],
            "profile_draw": "fixed_shared",
            "profile_jitter_sd_mm": 0.0,
            "rail_end_clearance_m":
                contract["scenario"]["rail_end_clearance_m"],
            "rail_end_clearance_decision_id":
                contract["scenario"]["rail_end_clearance_decision_id"],
            "use_track_eov": contract["scenario"]["use_track_eov"],
            "track_draw": "per_state",
            "track_L_app": 30.0,
            "track_L_after": 30.0,
            "use_oor_eov": contract["scenario"]["use_oor_eov"],
            "oor_flats_enabled":
                contract["scenario"]["oor_flats_enabled"],
            "use_signal_noise":
                contract["scenario"]["use_signal_noise"],
            "use_vehicle_variability": True,
            "use_speed_variability": True,
            "use_temp_variability": True,
        }
        manifest_mismatches = {
            field: (metadata.get(field), wanted)
            for field, wanted in expected_manifest.items()
            if field not in metadata
            or not _json_values_equal(metadata[field], wanted)
        }
        if actual_basename != expected_dataset:
            manifest_mismatches["dataset_directory_basename"] = (
                actual_basename,
                expected_dataset,
            )
        config_expected = generation_config_expectations(expected_stage)
        config_mismatches = {
            field: (generation_config.get(field), wanted)
            for field, wanted in config_expected.items()
            if field not in generation_config
            or not _json_values_equal(generation_config[field], wanted)
        }
        if manifest_mismatches or config_mismatches:
            raise RuntimeError(
                f"{dataset_path}: dataset is not the registered "
                f"{expected_stage} scientific scenario. Manifest/path "
                f"mismatches={manifest_mismatches}; hashed-config "
                f"mismatches={config_mismatches}. Refusing a renamed or "
                "wrong-rung dataset before any study/cache is created."
            )
        metadata["campaign_contract"] = contract
    return metadata


def _validate_state_top_level_generation_metadata(
    loaded: dict,
    manifest: dict,
    *,
    expected_schema: str,
    expected_fingerprint: str,
    expected_state_uid: str,
    expected_state_seed_id: int,
    expected_random_stream_schedule_version: str,
    owner: str,
) -> None:
    """Cross-check one state's cheap top-level R11 stamps."""
    top_names = tuple(loaded)
    top_expected = {
        "file_gen_schema": expected_schema,
        "file_gen_fingerprint": expected_fingerprint,
        "file_matlab_release": manifest["matlab_release"],
        "file_campaign_matlab_release":
            manifest["campaign_matlab_release"],
        "file_actual_matlab_environment_sha256":
            manifest["actual_matlab_environment_sha256"],
        "file_campaign_matlab_environment_sha256":
            manifest["campaign_matlab_environment_sha256"],
        "file_generator_source_root_sha256":
            manifest["generator_source_root_sha256"],
        "file_qualification_source_sha256":
            manifest["qualification_source_sha256"],
    }
    top_actual = {
        field: _required_mat_text(loaded, top_names, field, owner)
        for field in top_expected
    }
    top_state_uid = _required_mat_text(
        loaded, top_names, "file_state_uid", owner
    )
    top_state_seed_id = _required_mat_int(
        loaded, top_names, "file_state_seed_id", owner
    )
    top_schedule = _required_mat_text(
        loaded,
        top_names,
        "file_random_stream_schedule_version",
        owner,
    )
    top_qualification = _required_mat_logical(
        loaded,
        top_names,
        "file_release_qualification_run",
        owner,
    )
    top_mismatches = {
        field: (top_actual[field], wanted)
        for field, wanted in top_expected.items()
        if top_actual[field] != wanted
    }
    if top_qualification is not manifest["release_qualification_run"]:
        top_mismatches["file_release_qualification_run"] = (
            top_qualification,
            manifest["release_qualification_run"],
        )
    if top_state_uid != expected_state_uid:
        top_mismatches["file_state_uid"] = (
            top_state_uid,
            expected_state_uid,
        )
    if (
        top_state_seed_id != int(expected_state_seed_id)
        or not 1 <= top_state_seed_id <= np.iinfo(np.uint32).max
    ):
        top_mismatches["file_state_seed_id"] = (
            top_state_seed_id,
            int(expected_state_seed_id),
        )
    if top_schedule != expected_random_stream_schedule_version:
        top_mismatches["file_random_stream_schedule_version"] = (
            top_schedule,
            expected_random_stream_schedule_version,
        )
    if top_mismatches:
        raise RuntimeError(
            f"{owner}: top-level R11 provenance differs from case_info: "
            f"{top_mismatches}."
        )


def _validate_state_generation_metadata(
    loaded: dict,
    data_struct,
    names: tuple,
    manifest: dict,
    *,
    expected_schema: str,
    expected_fingerprint: str,
    expected_state_uid: str,
    expected_state_seed_id: int,
    expected_random_stream_schedule_version: str,
    owner: str,
) -> None:
    """Cross-check a state's top-level stamps and nested R11 payload."""
    _validate_state_top_level_generation_metadata(
        loaded,
        manifest,
        expected_schema=expected_schema,
        expected_fingerprint=expected_fingerprint,
        expected_state_uid=expected_state_uid,
        expected_state_seed_id=expected_state_seed_id,
        expected_random_stream_schedule_version=
            expected_random_stream_schedule_version,
        owner=owner,
    )

    nested_expected = {
        "channel_schema_id": manifest["channel_schema_id"],
        "matlab_release": manifest["matlab_release"],
        "campaign_matlab_release": manifest["campaign_matlab_release"],
        "actual_matlab_environment_descriptor":
            manifest["actual_matlab_environment_descriptor"],
        "actual_matlab_environment_sha256":
            manifest["actual_matlab_environment_sha256"],
        "campaign_matlab_environment_descriptor":
            manifest["campaign_matlab_environment_descriptor"],
        "campaign_matlab_environment_sha256":
            manifest["campaign_matlab_environment_sha256"],
        "generator_source_root_sha256":
            manifest["generator_source_root_sha256"],
        "generator_source_digest_lines":
            manifest["generator_source_digest_lines"],
        "qualification_source_sha256":
            manifest["qualification_source_sha256"],
    }
    nested_actual = {
        field: _required_mat_text(
            data_struct,
            names,
            field,
            f"{owner}: data",
        )
        for field in nested_expected
    }
    nested_count = _required_mat_int(
        data_struct,
        names,
        "generator_source_file_count",
        f"{owner}: data",
    )
    nested_qualification = _required_mat_logical(
        data_struct,
        names,
        "release_qualification_run",
        f"{owner}: data",
    )
    nested_mismatches = {
        field: (nested_actual[field], wanted)
        for field, wanted in nested_expected.items()
        if nested_actual[field] != wanted
    }
    if nested_count != manifest["generator_source_file_count"]:
        nested_mismatches["generator_source_file_count"] = (
            nested_count,
            manifest["generator_source_file_count"],
        )
    if nested_qualification is not manifest["release_qualification_run"]:
        nested_mismatches["release_qualification_run"] = (
            nested_qualification,
            manifest["release_qualification_run"],
        )
    if nested_mismatches:
        raise RuntimeError(
            f"{owner}: nested R11 provenance differs from case_info: "
            f"{nested_mismatches}."
        )


def validate_dataset_state_provenance_stamps(
    dataset_path: str,
    n_states: int,
    manifest: dict,
    *,
    expected_schema: str,
    expected_fingerprint: str,
) -> None:
    """Validate every lightweight state stamp without loading signal payloads."""
    state_table = read_state_table(dataset_path)
    if len(state_table["state_uid"]) != int(n_states):
        raise RuntimeError(
            f"{dataset_path}: state table contains "
            f"{len(state_table['state_uid'])} semantic identities but the "
            f"manifest declares {int(n_states)} states."
        )
    variables = (
        "file_gen_schema",
        "file_gen_fingerprint",
        "file_state_uid",
        "file_state_seed_id",
        "file_random_stream_schedule_version",
        "file_matlab_release",
        "file_campaign_matlab_release",
        "file_release_qualification_run",
        "file_actual_matlab_environment_sha256",
        "file_campaign_matlab_environment_sha256",
        "file_generator_source_root_sha256",
        "file_qualification_source_sha256",
    )
    for index in range(1, int(n_states) + 1):
        name = f"{index:04d}.mat"
        path = os.path.join(dataset_path, name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{dataset_path}: expected state stamp file is missing: {name}."
            )
        loaded = sio.loadmat(
            path,
            variable_names=variables,
            mat_dtype=True,
        )
        _validate_state_top_level_generation_metadata(
            loaded,
            manifest,
            expected_schema=expected_schema,
            expected_fingerprint=expected_fingerprint,
            expected_state_uid=state_table["state_uid"][index - 1],
            expected_state_seed_id=int(
                state_table["state_seed_id"][index - 1]
            ),
            expected_random_stream_schedule_version=
                state_table["random_stream_schedule_version"],
            owner=name,
        )


def _validate_campaign_release_metadata(dataset_path: str) -> tuple[str, bool]:
    """Compatibility gate returning (release, qualification) after R11 checks."""
    metadata = _validate_campaign_generation_metadata(dataset_path)
    return (
        metadata["matlab_release"],
        metadata["release_qualification_run"],
    )


def _read_completion_marker(dataset_path: str) -> tuple[str, str, str]:
    """Read one canonical three-line R11 completion marker.

    Platform newline translation is allowed (MATLAB writes the file), but the
    logical text must contain exactly three nonempty, unpadded lines and one
    final newline. Blank annotations, leading/trailing spaces, BOMs and
    symlinks fail closed.
    """
    marker = os.path.join(dataset_path, '_GENERATION_COMPLETE')
    if (
        not os.path.isfile(marker)
        or os.path.islink(marker)
    ):
        raise RuntimeError(
            f"{dataset_path}: missing regular _GENERATION_COMPLETE marker."
        )
    try:
        with open(marker, encoding='utf-8', newline=None) as handle:
            content = handle.read()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"{dataset_path}: completion marker is unreadable/non-UTF-8."
        ) from exc
    lines = content.split('\n')
    if (
        len(lines) != 4
        or lines[-1] != ''
        or any(not line or line != line.strip() for line in lines[:3])
    ):
        raise RuntimeError(
            f"{dataset_path}: completion marker must be exactly three "
            "nonempty, unpadded lines plus a final newline."
        )
    return lines[0], lines[1], lines[2]


def _read_file_digests(dataset_path: str):
    """Read the exact R11 ``source-digests-v2`` dataset-byte contract.

    A valid table contains canonical, sorted lowercase SHA-256 rows for every
    numbered state plus ``case_info.mat`` and ``damage_states.mat``. R11
    deliberately orphaned all earlier schemas, so a legacy state-only table,
    an unknown scope/schema, duplicate/non-canonical rows, or an extra/missing
    digest is an error rather than a compatibility path.

    Returns ``(None, None)`` only when the table itself is absent. Malformed
    tables raise ``RuntimeError``.
    """
    fd = os.path.join(dataset_path, 'file_digests.mat')
    if not os.path.exists(fd):
        return None, None
    if not os.path.isfile(fd) or os.path.islink(fd):
        raise RuntimeError(
            f"{dataset_path}: file_digests.mat must be a regular local file."
        )
    s = _load_scalar_mat_struct(
        fd,
        "file_digests",
        f"{dataset_path}: file_digests",
    )
    names = s.dtype.names or ()
    required_fields = {'schema', 'scope', 'digest_lines', 'root'}
    if set(names) != required_fields:
        raise RuntimeError(
            f"{dataset_path}: file_digests fields {sorted(names)!r} must be "
            f"exactly {sorted(required_fields)!r}."
        )
    owner = f"{dataset_path}: file_digests"
    schema = _required_mat_text(s, names, 'schema', owner)
    scope = _required_mat_text(s, names, 'scope', owner)
    if schema != 'source-digests-v2':
        raise RuntimeError(
            f"{dataset_path}: unsupported digest schema {schema!r}; "
            "R11 requires 'source-digests-v2'."
        )
    expected_scope = 'NNNN.mat+case_info.mat+damage_states.mat'
    if scope != expected_scope:
        raise RuntimeError(
            f"{dataset_path}: digest scope {scope!r} is incomplete/foreign; "
            f"R11 requires {expected_scope!r}."
        )

    lines = _required_mat_text(s, names, 'digest_lines', owner)
    root = _required_mat_text(s, names, 'root', owner)
    if re.fullmatch(r'[0-9a-f]{64}', root) is None:
        raise RuntimeError(
            f"{dataset_path}: file_digests.root is not lowercase SHA-256."
        )

    per: dict[str, str] = {}
    rows = lines.split('\n')
    for row in rows:
        if row.count(':') != 1:
            raise RuntimeError(
                f"{dataset_path}: malformed digest row {row!r}."
            )
        name, digest = row.split(':', 1)
        if (
            not name
            or os.path.basename(name) != name
            or name in {'.', '..'}
            or name in per
            or re.fullmatch(r'[0-9a-f]{64}', digest) is None
        ):
            raise RuntimeError(
                f"{dataset_path}: unsafe, duplicate or invalid digest row "
                f"{row!r}."
            )
        per[name] = digest

    canonical = '\n'.join(f'{name}:{per[name]}' for name in sorted(per))
    if lines != canonical:
        raise RuntimeError(
            f"{dataset_path}: digest rows are not canonical/sorted."
        )
    if len(per) != len({name.casefold() for name in per}):
        raise RuntimeError(
            f"{dataset_path}: digest inventory contains case-colliding names."
        )
    actual_states = sorted(
        name for name in os.listdir(dataset_path) if _re_state.fullmatch(name)
    )
    expected_names = set(actual_states) | {'case_info.mat', 'damage_states.mat'}
    if set(per) != expected_names:
        raise RuntimeError(
            f"{dataset_path}: R11 digest inventory mismatch; "
            f"missing={sorted(expected_names - set(per))}, "
            f"extra={sorted(set(per) - expected_names)}."
        )
    calculated_root = hashlib.sha256(lines.encode('utf-8')).hexdigest()
    if not hmac.compare_digest(calculated_root, root):
        raise RuntimeError(
            f"{dataset_path}: file_digests root disagrees with canonical rows."
        )
    return per, root


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
    new full pass. R11 requires ``case_info.mat`` and ``damage_states.mat`` in
    the same generation-time root as every state file; legacy state-only tables
    are rejected.

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
            if not os.path.isfile(path) or os.path.islink(path):
                raise RuntimeError(
                    f"{dataset_path}: digest-listed source file is missing, "
                    f"non-regular or symlinked: {name}.")
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
    # Defence in depth: _read_file_digests already requires both sidecars in
    # the R11 root; hash them again here for the compact protocol record.
    required_sidecar_sha256: dict[str, str] = {}
    for name in ("case_info.mat", "damage_states.mat"):
        path = os.path.join(dataset_path, name)
        if not os.path.isfile(path) or os.path.islink(path):
            raise RuntimeError(
                f"{dataset_path}: required provenance sidecar is missing, "
                f"non-regular or symlinked: {name}.")
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
        "required_sidecars_generation_digest_covered": True,
    }


_STATE_UID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:|=,+-]{0,255}\Z")


def _matlab_cellstr_vector(raw, owner: str) -> list[str]:
    """Read one MATLAB cellstr/string vector without stringifying junk."""

    values: list[str] = []
    for index, cell in enumerate(np.ravel(np.asarray(raw, dtype=object))):
        flat = np.ravel(np.asarray(cell))
        if flat.size != 1:
            raise RuntimeError(
                f"{owner}[{index}] must contain exactly one text scalar."
            )
        value = flat[0]
        if not isinstance(value, (str, np.str_)):
            raise RuntimeError(
                f"{owner}[{index}] is {type(value).__name__}, not text."
            )
        values.append(str(value))
    return values


def _matlab_text_scalar(raw, owner: str) -> str:
    flat = np.ravel(np.asarray(raw))
    if flat.size != 1 or not isinstance(flat[0], (str, np.str_)):
        raise RuntimeError(f"{owner} must be exactly one text scalar.")
    value = str(flat[0])
    if not value or value != value.strip():
        raise RuntimeError(f"{owner} must be nonempty and unpadded.")
    return value


def _strict_binary_vector(raw, owner: str) -> np.ndarray:
    """Return bool only when every stored value is exactly finite 0 or 1."""

    try:
        values = np.ravel(np.asarray(raw, dtype=float))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{owner} must be a numeric/logical vector.") from exc
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0.0, 1.0))):
        raise RuntimeError(f"{owner} must contain only finite logical 0/1 values.")
    return values.astype(bool)


def _strict_positive_uint32_vector(raw, owner: str) -> np.ndarray:
    """Return unique, positive, uint32-compatible RNG stream identifiers."""

    try:
        values = np.ravel(np.asarray(raw, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{owner} must be an integer vector.") from exc
    if (
        not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
        or np.any(values < 1)
        or np.any(values > np.iinfo(np.uint32).max)
    ):
        raise RuntimeError(f"{owner} must contain integers in [1, 2^32-1].")
    result = values.astype(np.uint32)
    if len(np.unique(result)) != len(result):
        raise RuntimeError(f"{owner} contains duplicate state-stream IDs.")
    return result


def _strict_nonnegative_int_vector(raw, owner: str) -> np.ndarray:
    try:
        values = np.ravel(np.asarray(raw, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{owner} must be an integer vector.") from exc
    if (
        not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
        or np.any(values < 0)
        or np.any(values > np.iinfo(np.int32).max)
    ):
        raise RuntimeError(f"{owner} must contain nonnegative integers.")
    return values.astype(np.int64)


def _identity_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _state_identity_damage_seed(dataset_path: str) -> int:
    """Read the generator seed used by the UID→StateSeedID derivation."""

    path = os.path.join(dataset_path, "case_info.mat")
    try:
        info = sio.loadmat(path, mat_dtype=True)["case_info"][0, 0]
    except Exception as exc:
        raise RuntimeError(
            f"{dataset_path}: cannot read case_info.mat for StateSeedID "
            "authentication."
        ) from exc
    names = info.dtype.names or ()
    if "generation_config_json" in names:
        raw = np.ravel(info["generation_config_json"])
        if raw.size != 1:
            raise RuntimeError(
                f"{dataset_path}: generation_config_json must be scalar text."
            )
        try:
            config = json.loads(str(raw[0]))
            value = config["damage_seed"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{dataset_path}: generation_config_json lacks valid damage_seed."
            ) from exc
    elif "damage_seed" in names:  # isolated mutation-test fixture
        raw = np.ravel(info["damage_seed"])
        value = raw[0] if raw.size == 1 else None
    else:
        raise RuntimeError(
            f"{dataset_path}: case_info lacks generation_config_json/damage_seed."
        )
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) != math.floor(float(value))
        or not 0 <= int(value) <= np.iinfo(np.uint32).max
    ):
        raise RuntimeError(f"{dataset_path}: damage_seed must be a uint32 integer.")
    return int(value)


def _expected_state_seed_id(uid: str, damage_seed: int) -> int:
    token = f"ttbi-state-seed-v1|damage_seed={damage_seed}|{uid}"
    return int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)


def _expected_named_seed(
    schedule: str,
    root_seed: int,
    uid: str,
    stream: str,
    passage: int | None = None,
) -> int:
    token = (
        f"{schedule}|root={root_seed}|uid={uid}|stream={stream}"
        + ("" if passage is None else f"|pass={passage:05d}")
    )
    return int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)


def state_identity_descriptor(table: dict) -> dict:
    """Canonical semantic-state inventory for split/cache/protocol binding."""

    uids = [str(value) for value in table["state_uid"]]
    sorted_uids = sorted(uids)
    joint_uids = sorted(
        uid for uid, family in zip(uids, table["family"], strict=True)
        if family == "joint"
    )
    records = []
    latent_records = []
    for row, uid in enumerate(uids):
        latent_record = {
            "uid": uid,
            "state_seed_id": int(table["state_seed_id"][row]),
            "family": str(table["family"][row]),
            "anchor_target": int(table["anchor_target"][row]),
            "anchor_level": int(table["anchor_level"][row]),
            "latent_bearing_fixity": [
                float(value) for value in table["latent_bearing_fixity"][row]
            ],
            "latent_crack_on": bool(table["latent_crack_on"][row]),
            "damage_state": [
                float(value) for value in table["damage_states"][row]
            ],
        }
        latent_records.append(latent_record)
        records.append({
            **latent_record,
            "crack_on": bool(table["crack_on"][row]),
            "bearing_fixity": [
                float(value) for value in table["bearing_fixity"][row]
            ],
        })
    records.sort(key=lambda record: record["uid"])
    latent_records.sort(key=lambda record: record["uid"])
    family_counts = {
        family: int(sum(value == family for value in table["family"]))
        for family in STATE_FAMILIES
    }
    return {
        "schema": "ttbi-semantic-state-identity-v2",
        "damage_seed": int(table["damage_seed"]),
        "random_stream_schedule_version":
            table["random_stream_schedule_version"],
        "state_stream_names": list(table["state_stream_names"]),
        "passage_stream_names": list(table["passage_stream_names"]),
        "passages_per_state": int(
            table["passage_named_stream_seed_id"].shape[1]
        ),
        "state_uid_count": len(uids),
        "state_uid_inventory": sorted_uids,
        "state_uid_inventory_sha256": _identity_sha256(sorted_uids),
        "state_uid_row_order": uids,
        "state_uid_row_order_sha256": _identity_sha256(uids),
        "state_seed_id_by_uid_sha256": _identity_sha256([
            [uid, int(seed)]
            for uid, seed in sorted(
                zip(uids, table["state_seed_id"], strict=True),
                key=lambda pair: pair[0],
            )
        ]),
        "state_named_stream_by_uid_sha256": _identity_sha256([
            [
                uid,
                [
                    int(value)
                    for value in table["state_named_stream_seed_id"][row]
                ],
            ]
            for row, uid in sorted(
                enumerate(uids), key=lambda pair: pair[1]
            )
        ]),
        "passage_named_stream_by_uid_sha256": _identity_sha256([
            [
                uid,
                [
                    [int(value) for value in passage]
                    for passage in
                    table["passage_named_stream_seed_id"][row]
                ],
            ]
            for row, uid in sorted(
                enumerate(uids), key=lambda pair: pair[1]
            )
        ]),
        "joint_state_uid_count": len(joint_uids),
        "joint_state_uid_inventory": joint_uids,
        "joint_state_uid_inventory_sha256": _identity_sha256(joint_uids),
        "family_counts": family_counts,
        # This is the causal-pairing identity: it deliberately excludes active
        # CrackOn/BearingFixity, which are the rung treatments, while binding
        # the complete latent design and scour realization by semantic UID.
        "latent_design_root_sha256": _identity_sha256(latent_records),
        # Full active identity remains useful for cache/protocol provenance and
        # is expected to differ when a rung activates bearing or crack physics.
        "state_identity_root_sha256": _identity_sha256(records),
    }


def read_state_table(dataset_path: str) -> dict:
    """Read and authenticate the row-aligned semantic state/CRN table.

    ``StateUID`` is independent of row/DC, while ``StateSeedID`` is its unique
    RNG-stream identity.  Both latent and active bearing/crack variables are
    mandatory.  Missing values are never reconstructed from row order or labels.
    """

    ds = os.path.join(dataset_path, 'damage_states.mat')
    if not os.path.exists(ds):
        raise RuntimeError(
            f"{dataset_path}: no damage_states.mat — the semantic-state split "
            f"needs the generator's CRN table. Regenerate with current A00.")
    m = sio.loadmat(ds, mat_dtype=True)
    required = (
        'StateFamily', 'AnchorTarget', 'AnchorLevel', 'StateUID',
        'StateSeedID', 'LatentBearingFixity', 'LatentCrackOn', 'CrackOn',
        'StateNamedStreamSeedID', 'PassageNamedStreamSeedID',
        'PassageNamedStreamSeedIDFlat', 'random_stream_schedule_version',
        'state_stream_names', 'passage_stream_names',
        'DamageStates', 'BearingFixity',
    )
    missing = [field for field in required if field not in m]
    if missing:
        raise RuntimeError(
            f"{dataset_path}: damage_states.mat lacks {missing} — pre-R11 "
            f"semantic-state dataset. Regenerate; never infer UID/stream/latent "
            f"values from row order or labels.")
    family = _matlab_cellstr_vector(
        m['StateFamily'], f"{dataset_path}: StateFamily"
    )
    bad = sorted({value for value in family if value not in STATE_FAMILIES})
    if bad:
        raise RuntimeError(
            f"{dataset_path}: unknown state_family value(s) {bad} — expected "
            f"one of {STATE_FAMILIES}. Corrupt/foreign family table.")
    state_uid = _matlab_cellstr_vector(
        m['StateUID'], f"{dataset_path}: StateUID"
    )
    bad_uid = [
        uid for uid in state_uid if _STATE_UID_RE.fullmatch(uid) is None
    ]
    if bad_uid:
        raise RuntimeError(
            f"{dataset_path}: malformed StateUID value(s) {bad_uid[:3]!r}; "
            "UIDs must be nonempty canonical printable-ASCII tokens."
        )
    if len(set(state_uid)) != len(state_uid):
        raise RuntimeError(
            f"{dataset_path}: StateUID contains duplicates — semantic identity "
            "must be one-to-one with generated states."
        )
    try:
        damage_states = np.atleast_2d(
            np.asarray(m['DamageStates'], dtype=float)
        )
        bearing_fixity = np.atleast_2d(
            np.asarray(m['BearingFixity'], dtype=float)
        )
        latent_bearing = np.atleast_2d(
            np.asarray(m['LatentBearingFixity'], dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{dataset_path}: physical/latent state matrices must be numeric."
        ) from exc
    if (
        not np.all(np.isfinite(damage_states))
        or not np.all(np.isfinite(bearing_fixity))
        or not np.all(np.isfinite(latent_bearing))
    ):
        raise RuntimeError(f"{dataset_path}: state matrices contain non-finite values.")
    if (
        np.any(damage_states < 0.0)
        or np.any(damage_states > 1.0)
        or np.any(bearing_fixity < 0.0)
        or np.any(bearing_fixity >= 1.0)
        or np.any(latent_bearing < 0.0)
        or np.any(latent_bearing >= 1.0)
    ):
        raise RuntimeError(
            f"{dataset_path}: scour is outside [0,1] or fixity is outside [0,1)."
        )
    if bearing_fixity.shape[1] != 2 or latent_bearing.shape[1] != 2:
        raise RuntimeError(
            f"{dataset_path}: BearingFixity and LatentBearingFixity must each "
            f"have two columns; got {bearing_fixity.shape} and "
            f"{latent_bearing.shape}."
        )
    state_seed_id = _strict_positive_uint32_vector(
        m['StateSeedID'], f"{dataset_path}: StateSeedID"
    )
    damage_seed = _state_identity_damage_seed(dataset_path)
    expected_seed_ids = np.asarray([
        _expected_state_seed_id(uid, damage_seed) for uid in state_uid
    ], dtype=np.uint32)
    if np.any(expected_seed_ids == 0):
        raise RuntimeError(
            f"{dataset_path}: UID seed derivation produced reserved stream 0; "
            "the generator must fail closed and version the derivation."
        )
    if not np.array_equal(state_seed_id, expected_seed_ids):
        mismatch = int(np.flatnonzero(state_seed_id != expected_seed_ids)[0])
        raise RuntimeError(
            f"{dataset_path}: StateSeedID row {mismatch + 1} is not the "
            "registered SHA-256 derivation of its StateUID/damage_seed — "
            "misaligned or restamped stream table."
        )
    schedule = _matlab_text_scalar(
        m["random_stream_schedule_version"],
        f"{dataset_path}: random_stream_schedule_version",
    )
    state_stream_names = _matlab_cellstr_vector(
        m["state_stream_names"], f"{dataset_path}: state_stream_names"
    )
    passage_stream_names = _matlab_cellstr_vector(
        m["passage_stream_names"], f"{dataset_path}: passage_stream_names"
    )
    if (
        schedule != "uid-named-substreams-v2"
        or state_stream_names != [
            "operations", "crack", "profile-state", "track", "profile-phase"
        ]
        or passage_stream_names != ["profile-passage", "oor-passage"]
    ):
        raise RuntimeError(
            f"{dataset_path}: foreign named-substream schedule/names "
            f"({schedule!r}, {state_stream_names!r}, "
            f"{passage_stream_names!r})."
        )
    state_named = np.asarray(m["StateNamedStreamSeedID"], dtype=np.float64)
    passage_named = np.asarray(
        m["PassageNamedStreamSeedID"], dtype=np.float64
    )
    passage_named_flat = np.asarray(
        m["PassageNamedStreamSeedIDFlat"], dtype=np.float64
    )
    if (
        state_named.shape != (len(state_uid), len(state_stream_names))
        or passage_named.ndim != 3
        or passage_named.shape[0] != len(state_uid)
        or passage_named.shape[2] != len(passage_stream_names)
        or passage_named_flat.shape
        != (len(state_uid), passage_named.shape[1] * len(passage_stream_names))
    ):
        raise RuntimeError(
            f"{dataset_path}: named stream matrices have wrong shapes "
            f"{state_named.shape}, {passage_named.shape}, "
            f"{passage_named_flat.shape}."
        )
    for owner, values in (
        ("StateNamedStreamSeedID", state_named),
        ("PassageNamedStreamSeedID", passage_named),
        ("PassageNamedStreamSeedIDFlat", passage_named_flat),
    ):
        if (
            not np.all(np.isfinite(values))
            or not np.all(values == np.floor(values))
            or np.any(values < 1)
            or np.any(values > np.iinfo(np.uint32).max)
        ):
            raise RuntimeError(
                f"{dataset_path}: {owner} must contain positive uint32 integers."
            )
    state_named = state_named.astype(np.uint32)
    passage_named = passage_named.astype(np.uint32)
    passage_named_flat = passage_named_flat.astype(np.uint32)
    expected_state_named = np.empty_like(state_named)
    expected_passage_named = np.empty_like(passage_named)
    for row, uid in enumerate(state_uid):
        root_seed = int(state_seed_id[row])
        for stream_index, stream_name in enumerate(state_stream_names):
            expected_state_named[row, stream_index] = _expected_named_seed(
                schedule, root_seed, uid, stream_name
            )
        for passage_index in range(passage_named.shape[1]):
            for stream_index, stream_name in enumerate(passage_stream_names):
                expected_passage_named[
                    row, passage_index, stream_index
                ] = _expected_named_seed(
                    schedule,
                    root_seed,
                    uid,
                    stream_name,
                    passage=passage_index + 1,
                )
    if (
        np.any(expected_state_named == 0)
        or np.any(expected_passage_named == 0)
        or not np.array_equal(state_named, expected_state_named)
        or not np.array_equal(passage_named, expected_passage_named)
        or not np.array_equal(
            passage_named_flat,
            expected_passage_named.reshape(
                len(state_uid), -1, order="F"
            ),
        )
    ):
        raise RuntimeError(
            f"{dataset_path}: named substream IDs are zero, misaligned, or do "
            "not match the registered SHA-256 namespace derivation."
        )
    all_stream_ids = np.concatenate([
        state_seed_id.ravel(),
        state_named.ravel(),
        passage_named.ravel(),
    ])
    if len(np.unique(all_stream_ids)) != len(all_stream_ids):
        raise RuntimeError(
            f"{dataset_path}: root/named RNG stream IDs collide."
        )
    table = {
        'family': family,
        'anchor_target': _strict_nonnegative_int_vector(
            m['AnchorTarget'], f"{dataset_path}: AnchorTarget"
        ),
        'anchor_level': _strict_nonnegative_int_vector(
            m['AnchorLevel'], f"{dataset_path}: AnchorLevel"
        ),
        'state_uid': state_uid,
        'state_seed_id': state_seed_id,
        # Explicit alias: this is the stable RNG stream, not a numbered-file row.
        'state_stream_id': state_seed_id.copy(),
        'damage_seed': damage_seed,
        'random_stream_schedule_version': schedule,
        'state_stream_names': state_stream_names,
        'passage_stream_names': passage_stream_names,
        'state_named_stream_seed_id': state_named,
        'passage_named_stream_seed_id': passage_named,
        'latent_bearing_fixity': latent_bearing,
        'latent_crack_on': _strict_binary_vector(
            m['LatentCrackOn'], f"{dataset_path}: LatentCrackOn"
        ),
        'crack_on': _strict_binary_vector(
            m['CrackOn'], f"{dataset_path}: CrackOn"
        ),
        'damage_states': damage_states,
        'bearing_fixity': bearing_fixity,
    }
    n = len(family)
    if not (
        len(table['anchor_target']) == len(table['anchor_level'])
        == len(table['state_uid']) == len(table['state_seed_id'])
        == len(table['latent_crack_on']) == len(table['crack_on'])
        == table['latent_bearing_fixity'].shape[0]
        == table['damage_states'].shape[0]
        == table['bearing_fixity'].shape[0] == n
    ):
        raise RuntimeError(
            f"{dataset_path}: state-table arrays are not row-aligned "
            f"(family={n}, target={len(table['anchor_target'])}, "
            f"level={len(table['anchor_level'])}, uid={len(table['state_uid'])}, "
            f"seed={len(table['state_seed_id'])}, "
            f"latent_crack={len(table['latent_crack_on'])}, "
            f"crack={len(table['crack_on'])}, "
            f"latent_fixity={table['latent_bearing_fixity'].shape[0]}, "
            f"scour={table['damage_states'].shape[0]}, "
            f"fixity={table['bearing_fixity'].shape[0]}) — corrupt table.")
    for row, family_name in enumerate(table["family"]):
        target = int(table["anchor_target"][row])
        level = int(table["anchor_level"][row])
        if family_name in ("scour_only", "bearing_only"):
            if target <= 0 or level <= 0:
                raise RuntimeError(
                    f"{dataset_path}: {family_name} row {row + 1} needs positive "
                    "AnchorTarget and AnchorLevel."
                )
        elif target != 0 or level != 0:
            raise RuntimeError(
                f"{dataset_path}: non-anchor family {family_name!r} row "
                f"{row + 1} must have AnchorTarget=AnchorLevel=0."
            )
        latent_crack = bool(table["latent_crack_on"][row])
        active_crack = bool(table["crack_on"][row])
        if family_name in ("target_healthy", "scour_only", "bearing_only"):
            if latent_crack or active_crack:
                raise RuntimeError(
                    f"{dataset_path}: controlled family {family_name!r} row "
                    f"{row + 1} cannot carry latent/active crack."
                )
        if family_name == "nuisance_only" and not latent_crack:
            raise RuntimeError(
                f"{dataset_path}: latent nuisance-only row {row + 1} must have "
                "LatentCrackOn=true (active CrackOn may be dormant by rung)."
            )
        if active_crack and not latent_crack:
            raise RuntimeError(
                f"{dataset_path}: row {row + 1} activates crack without its "
                "registered latent crack draw."
            )
    if not (
        np.array_equal(
            table["bearing_fixity"],
            np.zeros_like(table["bearing_fixity"]),
        )
        or np.array_equal(
            table["bearing_fixity"], table["latent_bearing_fixity"]
        )
    ):
        raise RuntimeError(
            f"{dataset_path}: active BearingFixity must be either wholly dormant "
            "(all zero) or exactly the latent CRN matrix."
        )
    if not (
        np.array_equal(
            table["crack_on"], np.zeros_like(table["crack_on"])
        )
        or np.array_equal(table["crack_on"], table["latent_crack_on"])
    ):
        raise RuntimeError(
            f"{dataset_path}: active CrackOn must be either wholly dormant or "
            "exactly the latent CRN vector."
        )
    return table


def _read_dano_max(dataset_path: str) -> float | None:
    """The scour ceiling `dano_max` (a fraction, e.g. 0.60) from case_info.mat,
    or None if the manifest predates it (audit R7 P4). Used to bound the scour
    label to the physically-generated range rather than merely [0, 100] %."""
    ci = os.path.join(dataset_path, 'case_info.mat')
    if not os.path.exists(ci):
        return None
    info = sio.loadmat(ci, mat_dtype=True)['case_info'][0, 0]
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
            info = sio.loadmat(ci, mat_dtype=True)['case_info'][0, 0]
            if 'bearing_max_Nm_rad' in (info.dtype.names or ()):
                v = float(np.ravel(info['bearing_max_Nm_rad'])[0])
                if v > 0:
                    return v
        except Exception:
            pass
    ds = os.path.join(dataset_path, 'damage_states.mat')
    if os.path.exists(ds):
        try:
            bs = sio.loadmat(ds, mat_dtype=True).get('BearingStates')
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
    """One registered semantic stratum per state.

    * anchor families use ``family + target + level``.  R11 generates five
      semantic replicas of every level, so the 3/1/1 pattern puts that exact
      severity in train/validation/test;
    * target-healthy and latent nuisance-only use their family;
    * joint uses *latent* crack status and scour-only severity.  Active crack
      and bearing mechanisms differ by rung, so using either would destroy the
      common-random-number partition parity the paired contrasts require.
    """
    ds = table['damage_states']
    dmax = float(dano_max) if dano_max else max(float(ds.max()), 1e-12)
    keys = []
    for i, fam in enumerate(table['family']):
        if fam in ('scour_only', 'bearing_only'):
            keys.append(
                f"{fam}|target{int(table['anchor_target'][i])}"
                f"|level{int(table['anchor_level'][i])}"
            )
        elif fam in ('target_healthy', 'nuisance_only'):
            keys.append(fam)
        else:                                                    # joint
            sev_s = float(ds[i].max()) / dmax
            sev = min(max(sev_s, 0.0), 1.0)
            sev_bin = min(int(sev * N_JOINT_SEV_BINS), N_JOINT_SEV_BINS - 1)
            keys.append(
                f"joint|latentcrack"
                f"{int(bool(table['latent_crack_on'][i]))}"
                f"|scoursev{sev_bin}"
            )
    return keys


def _canonical_state_assignment(
    table: dict,
    dano_max: float | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return row-aligned strata/partitions from semantic UIDs, never DC rows."""

    strata = _stratum_keys(table, dano_max)
    n_states = len(table["state_uid"])
    assignment: list[str | None] = [None] * n_states
    for key in sorted(set(strata)):
        # Sorting first removes all construction-order/DC dependence.  The RNG
        # then sees the same ordered UID set in every geometry-compatible rung.
        members = sorted(
            (row for row in range(n_states) if strata[row] == key),
            key=lambda row: table["state_uid"][row],
        )
        rng = np.random.default_rng(_stable_stratum_seed(seed, key))
        permutation = rng.permutation(len(members))
        for position, member_position in enumerate(permutation):
            assignment[members[int(member_position)]] = STRATIFY_PATTERN[
                position % len(STRATIFY_PATTERN)
            ]
    if any(value is None for value in assignment):
        raise RuntimeError("semantic UID split left an unassigned state.")
    return strata, [str(value) for value in assignment]


def semantic_split_descriptor(
    table: dict,
    dano_max: float | None,
    seed: int = SPLIT_SEED,
) -> dict:
    """Canonical UID→partition record embedded in provenance artifacts."""

    strata, assignment = _canonical_state_assignment(table, dano_max, seed)
    records = [
        {
            "state_uid": table["state_uid"][row],
            "state_seed_id": int(table["state_seed_id"][row]),
            "stratum": strata[row],
            "partition": assignment[row],
        }
        for row in range(len(assignment))
    ]
    records.sort(key=lambda record: record["state_uid"])
    return {
        "schema": "ttbi-semantic-split-v1",
        "seed": int(seed),
        "assignment_by_uid": records,
        "assignment_by_uid_sha256": _identity_sha256(records),
        "partition_counts": {
            partition: int(sum(
                record["partition"] == partition for record in records
            ))
            for partition in ("train", "val", "test")
        },
    }


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
    identity = state_identity_descriptor(table)
    split_identity = semantic_split_descriptor(
        table, _read_dano_max(dataset_dir), seed
    )
    assignment_by_uid = split_identity["assignment_by_uid"]
    payload = {
        "split_manifest_version": 2,
        "policy": {"seed": seed, "test_frac": SPLIT_TEST_FRAC,
                   "val_frac": SPLIT_VAL_FRAC,
                   "pattern": list(STRATIFY_PATTERN),
                   "n_joint_sev_bins": N_JOINT_SEV_BINS,
                   "ordering": "lexicographically sorted StateUID within stratum",
                   "joint_basis": "latent crack + scour-only severity"},
        "dataset_gen_fingerprint": gen_fp,
        "n_states": len(assignment),
        "state_identity": identity,
        "state_family": list(table['family']),
        "state_uid": list(table["state_uid"]),
        "state_seed_id": [
            int(value) for value in table["state_seed_id"]
        ],
        "stratum": list(strata),
        "assignment": list(assignment),
        "assignment_by_uid": assignment_by_uid,
        "assignment_by_uid_sha256":
            split_identity["assignment_by_uid_sha256"],
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

    R11 stratification assigns semantic UIDs per stratum (family; anchor
    family×target×level; joint latent-crack×scour-severity) using a seeded
    permutation of sorted UIDs plus the fixed 60/20/20 pattern, so
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
    strata, assignment = _canonical_state_assignment(
        table, _read_dano_max(dataset_dir), seed
    )
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

    Format:  <dataset>_<method>_dofs_<d0>_<d1>_..._disc<k>_gs9

    The trailing schema tag versions everything baked into the cache: the split
    policy (scaler fit on the canonical grouped TRAIN partition) AND the loaded
    data itself. gs1 = grouped split (2026-07-17); gs2 = R3 (fixed-profile
    geometry/crop + contact filtering); gs3 = R5 (3-way train/val/test split
    + reflect-fold profile); gs4 = R7 (atomic cache write + artifact digests +
    inventory + stricter loader payload); **gs5** = R8 (family-STRATIFIED split
    + mandatory state table + per-file state_family); **gs6** = audit r3
    2026-07-22 (TRUE Keogh PAA replaces the linear resampler + per-global-DOF
    paired noise RNG + segment count named in the stem); **gs7** = R11 semantic
    UID/stream authentication and row-invariant UID split (which changes the
    TRAIN states used to fit the scaler); **gs9** = physical8_v1 DOFs 3-4 and
    channel-schema-bound provenance. Older-tag caches are orphaned.

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
    For physically measurable vehicle channels, per-channel levels must come
    from sensor DATASHEETS (noise density
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
          Frozen compatibility key: multiplicative Gaussian robustness noise
          (std = desvio·|signal|) on deployed DOFs 3-4. Under physical8_v1
          these are idealized constrained-wheelset/axle-box response proxies.
          This mode does NOT reproduce the old baked AcelRodaPrimVag noise.
    Per-channel additive noise-floor modes require a signal-independent floor
    from sensor datasheets plus an explicit observation model; the idealized
    wheelset proxy must not be described as a measured axle-box signal.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    desvio = float(sn.get('desvio', 0.05))
    WHEELSET_PROXY_CHANNELS = (3, 4)

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
        # Frozen mode key; physical8_v1 applies it to wheelset proxy DOFs 3-4.
        add_mult(WHEELSET_PROXY_CHANNELS)
    elif sn['mode'] == 'all_mult':
        # Channel-symmetric Gaussian multiplicative noise on EVERY channel
        # (pointwise sigma = desvio*|signal|). Use on NOISE-FREE data
        # (varVST / new stages) to make all channels equally noisy. Do NOT use on
        # legacy baked AcelRoda data (varNVST Stage-0/1) - those channels would be
        # DOUBLE-noised; use 'sprung_mult' there instead.
        add_mult(set(dofs))
    elif sn['mode'] == 'sprung_mult':
        # Frozen mode key: multiplicative noise on sprung-vehicle channels,
        # excluding the wheelset proxy indices 3-4.
        add_mult({d for d in dofs if d not in WHEELSET_PROXY_CHANNELS})
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
    cur_state_identity = None
    cur_semantic_split = None
    cur_matlab_release = None
    cur_release_qualification = None
    cur_generation_metadata = None
    cur_runtime_source = None
    src_dir = os.path.join('data', dataset_name)
    if regression and os.path.isdir(src_dir):
        cur_nstates, cur_npass, cur_schema, cur_fp = _read_manifest(src_dir)
        # R11: qualification/foreign-stack/source data is rejected even on a
        # cache hit; a valid cache can never launder an ineligible dataset.
        cur_generation_metadata = _validate_campaign_generation_metadata(
            src_dir
        )
        cur_runtime_source = python_runtime_source_root()
        cur_matlab_release = cur_generation_metadata["matlab_release"]
        cur_release_qualification = cur_generation_metadata[
            "release_qualification_run"
        ]
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
        state_table = read_state_table(src_dir)
        cur_state_identity = state_identity_descriptor(state_table)
        cur_semantic_split = semantic_split_descriptor(
            state_table, cur_dano, SPLIT_SEED
        )
        # `complete` requires the marker to EXIST *and* its content (schema+fp) to
        # match the manifest (audit R7.1 P3): a stale/wrong-content marker on the
        # fast path must not certify a source as complete.
        marker_ok = False
        try:
            mk = list(_read_completion_marker(src_dir))
            # schema + fp + a ROOT source digest (line 3) all required.
            marker_ok = mk[:2] == [cur_schema, cur_fp]
            if marker_ok:
                cur_src_root = mk[2]      # regeneration -> new root -> cache stale
        except RuntimeError:
            marker_ok = False
        cur_complete = (marker_ok and cur_nstates is not None
                        and n_present == cur_nstates)
    # SOURCE-provenance fields (compared as a block on the fast path). Includes the
    # source ROOT digest (audit R7.1 P4), so regenerating the dataset — which
    # rewrites file_digests + the marker root — invalidates a stale cache.
    cur_src = {"gen_schema": cur_schema, "gen_fingerprint": cur_fp,
               "channel_schema_id": (
                   cur_generation_metadata.get("channel_schema_id")
                   if cur_generation_metadata else None
               ),
               "state_design_kind": (
                   cur_generation_metadata.get("state_design_kind")
                   if cur_generation_metadata else None
               ),
               "python_runtime_source_root_sha256": (
                   cur_runtime_source.sha256
                   if cur_runtime_source else None
               ),
               "python_runtime_source_file_count": (
                   cur_runtime_source.file_count
                   if cur_runtime_source else None
               ),
               "generation_behavior_version": (
                   cur_generation_metadata.get("generation_behavior_version")
                   if cur_generation_metadata else None
               ),
               "matlab_release": cur_matlab_release,
               "campaign_matlab_release": (
                   cur_generation_metadata.get("campaign_matlab_release")
                   if cur_generation_metadata else None
               ),
               "actual_matlab_environment_descriptor": (
                   cur_generation_metadata.get(
                       "actual_matlab_environment_descriptor"
                   ) if cur_generation_metadata else None
               ),
               "actual_matlab_environment_sha256": (
                   cur_generation_metadata.get(
                       "actual_matlab_environment_sha256"
                   ) if cur_generation_metadata else None
               ),
               "campaign_matlab_environment_descriptor": (
                   cur_generation_metadata.get(
                       "campaign_matlab_environment_descriptor"
                   ) if cur_generation_metadata else None
               ),
               "campaign_matlab_environment_sha256": (
                   cur_generation_metadata.get(
                       "campaign_matlab_environment_sha256"
                   ) if cur_generation_metadata else None
               ),
               "generator_source_root_sha256": (
                   cur_generation_metadata.get(
                       "generator_source_root_sha256"
                   ) if cur_generation_metadata else None
               ),
               "generator_source_file_count": (
                   cur_generation_metadata.get(
                       "generator_source_file_count"
                   ) if cur_generation_metadata else None
               ),
               "qualification_source_sha256": (
                   cur_generation_metadata.get(
                       "qualification_source_sha256"
                   ) if cur_generation_metadata else None
               ),
               "release_qualification_run": cur_release_qualification,
               "n_states": cur_nstates, "passages_per_state": cur_npass,
               "dano_max": cur_dano, "manifest_sha256": cur_manifest_sha,
               "dataset_content_root_sha256": cur_src_root,
               "complete": cur_complete,
               "inventory": cur_inventory,
               "file_digests_sha256": cur_digests_sha,
               "damage_states_sha256": cur_states_sha,
               "state_identity": cur_state_identity,
               "semantic_split": cur_semantic_split}

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
            rung = (
                (config.get("protocol_descriptor") or {})
                .get("rung", {})
            )
            required_rung = {
                "stage", "dataset", "target_supports", "bearing_targets",
                "dataset_provenance",
            }
            if (
                not isinstance(rung, dict)
                or not required_rung.issubset(rung)
                or rung.get("dataset") != dataset_name
            ):
                raise RuntimeError(
                    f"{dataset_name}: protocol-hashed cache config lacks the "
                    "exact stage/dataset/target rung contract."
                )
            current_provenance = read_dataset_provenance(
                src_dir,
                expected_stage=rung.get("stage"),
                expected_dataset=rung.get("dataset"),
                expected_target_supports=rung.get("target_supports"),
                expected_bearing_targets=rung.get("bearing_targets"),
            )
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
