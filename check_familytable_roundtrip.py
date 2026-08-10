"""Authenticate the real MATLAB serialization of the current state table.

Run after ``smoke_familytable``.  This check uses the production reader, then
independently derives every UID-root and named-substream SHA-256 seed.  It also
checks MATLAB's flattened passage-seed array against NumPy's Fortran layout.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import _stratum_keys, read_state_table  # noqa: E402


SMOKE_DIR = os.path.join(
    "scour_MATLAB", "Results", "_smoke_familytable"
)
DAMAGE_SEED = 1
SCHEDULE = "uid-named-substreams-v2"
STATE_STREAM_NAMES = [
    "operations", "crack", "profile-state", "track", "profile-phase"
]
PASSAGE_STREAM_NAMES = ["profile-passage", "oor-passage"]
EXPECTED_FAMILY = (
    ["target_healthy"] * 2
    + ["scour_only"] * 2
    + ["bearing_only"]
    + ["nuisance_only"]
    + ["joint"] * 2
)
EXPECTED_UID = [
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=target_healthy|target=00|level=0000|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=target_healthy|target=00|level=0000|rep=002",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=scour_only|target=02|level=0001|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=scour_only|target=03|level=0002|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=bearing_only|target=01|level=0001|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=nuisance_only|target=00|level=0000|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=joint|target=00|level=0001|rep=001",
    "ttbi-state-v2|Lmm=060000|spans=3|scour=0203|"
    "family=joint|target=00|level=0002|rep=001",
]

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails += 1


def sha_seed(token: str) -> np.uint32:
    return np.uint32(int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16))


def expected_root_seed(uid: str) -> np.uint32:
    return sha_seed(
        f"ttbi-state-seed-v1|damage_seed={DAMAGE_SEED}|{uid}"
    )


def expected_named_seed(
    root: int,
    uid: str,
    stream: str,
    passage: int | None = None,
) -> np.uint32:
    suffix = "" if passage is None else f"|pass={passage:05d}"
    return sha_seed(
        f"{SCHEDULE}|root={root}|uid={uid}|stream={stream}{suffix}"
    )


damage_path = os.path.join(SMOKE_DIR, "damage_states.mat")
if not os.path.exists(damage_path):
    print(
        f"  [FAIL] {SMOKE_DIR} not found - run smoke_familytable "
        "in MATLAB first."
    )
    sys.exit(2)

table = read_state_table(SMOKE_DIR)
raw_table = sio.loadmat(damage_path)
check(
    "real MATLAB cellstr family parses",
    table["family"] == EXPECTED_FAMILY,
    repr(table["family"]),
)
check(
    "canonical semantic StateUIDs parse exactly",
    table["state_uid"] == EXPECTED_UID,
    repr(table["state_uid"]),
)
check(
    "AnchorTarget/AnchorLevel preserve the eight-row design",
    table["anchor_target"].tolist() == [0, 0, 2, 3, 1, 0, 0, 0]
    and table["anchor_level"].tolist() == [0, 0, 1, 2, 1, 0, 0, 0],
)
check(
    "latent/active crack vectors preserve real MATLAB logicals",
    table["latent_crack_on"].tolist()
    == [False, False, False, False, False, True, True, False]
    and np.array_equal(table["crack_on"], table["latent_crack_on"]),
)
check(
    "damage and latent/active bearing matrices preserve values",
    abs(table["damage_states"][2, 1] - 0.15) < 1e-12
    and abs(table["damage_states"][6, 2] - 0.40) < 1e-12
    and abs(table["latent_bearing_fixity"][4, 0] - 0.2375) < 1e-12
    and np.array_equal(
        table["bearing_fixity"], table["latent_bearing_fixity"]
    ),
)
raw_bearing_states = np.asarray(raw_table["BearingStates"], dtype=float)
raw_k_ref_bear = float(np.ravel(raw_table["k_ref_bear"])[0])
expected_bearing_states = (
    raw_k_ref_bear
    * table["bearing_fixity"]
    / (1.0 - table["bearing_fixity"])
)
check(
    "serialized bearing stiffness is exactly the nominal-fixity transform",
    raw_k_ref_bear == 2.31e9
    and np.array_equal(raw_bearing_states, expected_bearing_states),
)

expected_roots = np.asarray(
    [expected_root_seed(uid) for uid in EXPECTED_UID], dtype=np.uint32
)
check(
    "StateSeedID is the positive unique SHA-256 UID/damage-seed derivation",
    np.array_equal(table["state_seed_id"], expected_roots)
    and np.all(expected_roots > 0)
    and len(np.unique(expected_roots)) == len(expected_roots),
    repr(expected_roots.tolist()),
)
check(
    "stable stream alias and damage_seed authenticate",
    np.array_equal(table["state_stream_id"], expected_roots)
    and table["damage_seed"] == DAMAGE_SEED,
)
check(
    "registered schedule and component namespaces parse exactly",
    table["random_stream_schedule_version"] == SCHEDULE
    and table["state_stream_names"] == STATE_STREAM_NAMES
    and table["passage_stream_names"] == PASSAGE_STREAM_NAMES,
)

expected_state = np.empty((len(EXPECTED_UID), len(STATE_STREAM_NAMES)), dtype=np.uint32)
passage = table["passage_named_stream_seed_id"]
expected_passage = np.empty_like(passage)
for row, (uid, root) in enumerate(
    zip(EXPECTED_UID, expected_roots.tolist(), strict=True)
):
    for stream_index, stream in enumerate(STATE_STREAM_NAMES):
        expected_state[row, stream_index] = expected_named_seed(
            root, uid, stream
        )
    for passage_index in range(passage.shape[1]):
        for stream_index, stream in enumerate(PASSAGE_STREAM_NAMES):
            expected_passage[row, passage_index, stream_index] = (
                expected_named_seed(
                    root, uid, stream, passage=passage_index + 1
                )
            )
check(
    "state and passage named streams match an independent SHA-256 oracle",
    np.array_equal(table["state_named_stream_seed_id"], expected_state)
    and np.array_equal(passage, expected_passage),
)

raw = sio.loadmat(damage_path)
flat = np.asarray(raw["PassageNamedStreamSeedIDFlat"], dtype=np.uint32)
all_ids = np.concatenate(
    [expected_roots.ravel(), expected_state.ravel(), expected_passage.ravel()]
)
check(
    "MATLAB flat passage layout is the registered Fortran reshape",
    np.array_equal(
        flat, expected_passage.reshape(len(EXPECTED_UID), -1, order="F")
    ),
)
check(
    "complete root/named stream universe is nonzero and collision-free",
    np.all(all_ids > 0) and len(np.unique(all_ids)) == len(all_ids),
)

keys = _stratum_keys(table, 0.60)
expected_keys = [
    "target_healthy",
    "target_healthy",
    "scour_only|target2|level1",
    "scour_only|target3|level2",
    "bearing_only|target1|level1",
    "nuisance_only",
    "joint|latentcrack1|scoursev2",
    "joint|latentcrack0|scoursev0",
]
check(
    "current strata use latentcrack{0,1} nomenclature and latent status",
    keys == expected_keys,
    repr(keys),
)

print()
if fails:
    print(f"FAMILY-TABLE ROUNDTRIP: {fails} CHECK(S) FAILED")
    sys.exit(1)
print("FAMILY-TABLE ROUNDTRIP: ALL PASS")
