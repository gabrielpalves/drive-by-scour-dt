"""Mutation-tested semantic-UID grouped split contract.

Run with the exact campaign interpreter:
    py -3.13 check_split_grouping.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from core.dataset import (  # noqa: E402
    _EXPECTED_GEN_SCHEMA,
    _EXPECTED_MATLAB_RELEASE,
    STATE_FAMILIES,
    canonical_grouped_splits,
    canonical_train_val_split,
    read_state_table,
)


FIXTURE_FP = "a" * 64
FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(
        f"  [{'PASS' if condition else 'FAIL'}] {name}"
        + (f" - {detail}" if detail else "")
    )
    FAILURES += int(not condition)


def _uid(family: str, target: int, level: int, replica: int) -> str:
    return (
        "ttbi-state-v1|Lmm=060000|spans=3|scour=0203|"
        f"family={family}|target={target:02d}|level={level:04d}|"
        f"rep={replica:03d}"
    )


def _stream(uid: str, damage_seed: int = 1) -> int:
    token = f"ttbi-state-seed-v1|damage_seed={damage_seed}|{uid}"
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    if value == 0:
        raise AssertionError("fixture hit the reserved zero stream")
    return value


def _named_stream(
    schedule: str,
    root: int,
    uid: str,
    name: str,
    passage: int | None = None,
) -> int:
    token = (
        f"{schedule}|root={root}|uid={uid}|stream={name}"
        + ("" if passage is None else f"|pass={passage:05d}")
    )
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    if value == 0:
        raise AssertionError("fixture hit reserved named stream zero")
    return value


def _fixture_rows(seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def append(
        family: str,
        target: int,
        level: int,
        replica: int,
        scour: np.ndarray,
        latent_bearing: np.ndarray,
        latent_crack: bool,
        uid_level: int | None = None,
    ) -> None:
        uid = _uid(
            family, target, level if uid_level is None else uid_level, replica
        )
        rows.append({
            "family": family,
            "target": target,
            "level": level,
            "uid": uid,
            "stream": _stream(uid),
            "scour": np.asarray(scour, dtype=float),
            "latent_bearing": np.asarray(latent_bearing, dtype=float),
            "latent_crack": bool(latent_crack),
        })

    zero_s = np.zeros(4)
    zero_b = np.zeros(2)
    for replica in range(1, 51):
        append("target_healthy", 0, 0, replica, zero_s, zero_b, False)
    scour_levels = np.linspace(0.12, 0.60, 5)
    for target in (2, 3):
        for replica in range(1, 6):
            for level, severity in enumerate(scour_levels, start=1):
                scour = np.zeros(4)
                scour[target - 1] = severity
                append(
                    "scour_only", target, level, replica,
                    scour, zero_b, False,
                )
    bearing_levels = np.linspace(0.19, 0.95, 5)
    for target in (1, 2):
        for replica in range(1, 6):
            for level, severity in enumerate(bearing_levels, start=1):
                bearing = np.zeros(2)
                bearing[target - 1] = severity
                append(
                    "bearing_only", target, level, replica,
                    zero_s, bearing, False,
                )
    for replica in range(1, 51):
        append("nuisance_only", 0, 0, replica, zero_s, zero_b, True)
    for joint in range(1, 251):
        scour = np.zeros(4)
        scour[[1, 2]] = rng.random(2) * 0.60
        latent_bearing = rng.random(2) * 0.95
        latent_crack = (
            int(hashlib.sha256(
                f"latent|{_uid('joint', 0, joint, 1)}".encode("ascii")
            ).hexdigest()[:8], 16) / 2**32 < 0.25
        )
        append(
            "joint", 0, 0, 1, scour, latent_bearing, latent_crack,
            uid_level=joint,
        )
    assert len(rows) == 450
    return rows


def build_fixture(
    ds_dir: str,
    *,
    order: np.ndarray | None = None,
    mechanisms_active: bool = True,
) -> int:
    rows = _fixture_rows()
    if order is not None:
        rows = [rows[int(index)] for index in order]
    active_bearing = np.vstack([
        row["latent_bearing"] for row in rows
    ])
    active_crack = np.asarray([
        row["latent_crack"] for row in rows
    ], dtype=np.uint8)
    schedule = "uid-named-substreams-v2"
    state_names = [
        "operations", "crack", "profile-state", "track", "profile-phase"
    ]
    passage_names = ["profile-passage", "oor-passage"]
    state_named = np.asarray([
        [
            _named_stream(schedule, row["stream"], row["uid"], name)
            for name in state_names
        ]
        for row in rows
    ], dtype=np.uint32)
    passage_named = np.asarray([
        [[
            _named_stream(
                schedule, row["stream"], row["uid"], name, passage=1
            )
            for name in passage_names
        ]]
        for row in rows
    ], dtype=np.uint32)
    if not mechanisms_active:
        active_bearing = np.zeros_like(active_bearing)
        active_crack = np.zeros_like(active_crack)
    os.makedirs(ds_dir, exist_ok=True)
    sio.savemat(os.path.join(ds_dir, "damage_states.mat"), {
        "StateFamily": np.asarray(
            [row["family"] for row in rows], dtype=object
        ).reshape(-1, 1),
        "AnchorTarget": np.asarray(
            [row["target"] for row in rows], dtype=float
        ).reshape(-1, 1),
        "AnchorLevel": np.asarray(
            [row["level"] for row in rows], dtype=float
        ).reshape(-1, 1),
        "StateUID": np.asarray(
            [row["uid"] for row in rows], dtype=object
        ).reshape(-1, 1),
        "StateSeedID": np.asarray(
            [row["stream"] for row in rows], dtype=np.uint32
        ).reshape(-1, 1),
        "StateNamedStreamSeedID": state_named,
        "PassageNamedStreamSeedID": passage_named,
        "PassageNamedStreamSeedIDFlat": passage_named.reshape(
            len(rows), -1, order="F"
        ),
        "random_stream_schedule_version": schedule,
        "state_stream_names": np.asarray(
            state_names, dtype=object
        ).reshape(1, -1),
        "passage_stream_names": np.asarray(
            passage_names, dtype=object
        ).reshape(1, -1),
        "LatentBearingFixity": np.vstack([
            row["latent_bearing"] for row in rows
        ]),
        "LatentCrackOn": np.asarray([
            row["latent_crack"] for row in rows
        ], dtype=np.uint8).reshape(-1, 1),
        "CrackOn": active_crack.reshape(-1, 1),
        "DamageStates": np.vstack([row["scour"] for row in rows]),
        "BearingFixity": active_bearing,
        "BearingStates": active_bearing * 1e9,
        "k_ref_bear": 2.31e9,
        "scour_supports": np.asarray([[2.0, 3.0]]),
    })
    sio.savemat(os.path.join(ds_dir, "case_info.mat"), {
        "case_info": {
            "n_states": len(rows),
            "passages_per_state": 1,
            "gen_schema": _EXPECTED_GEN_SCHEMA,
            "gen_fingerprint": FIXTURE_FP,
            "matlab_release": _EXPECTED_MATLAB_RELEASE,
            "campaign_matlab_release": _EXPECTED_MATLAB_RELEASE,
            "release_qualification_run": False,
            "scour_dano_max_frac": 0.60,
            "damage_seed": 1,
        }
    })
    return len(rows)


def _partition_by_uid(dataset: str, n_states: int) -> dict[str, str]:
    groups = np.arange(n_states)
    canonical_grouped_splits(n_states, groups, dataset_name=dataset)
    manifest = json.load(
        open(os.path.join("data", dataset, "split_manifest.json"), encoding="utf-8")
    )
    return {
        record["state_uid"]: record["partition"]
        for record in manifest["assignment_by_uid"]
    }


tmp = tempfile.mkdtemp(prefix="splitcheck_")
cwd0 = os.getcwd()
try:
    os.chdir(tmp)
    dataset = "fixture_on"
    n_states = build_fixture(os.path.join("data", dataset))
    npass = 5
    groups = np.repeat(np.arange(n_states), npass)
    tr, va, te = canonical_grouped_splits(
        len(groups), groups, dataset_name=dataset
    )
    gt, gv, ge = set(groups[tr]), set(groups[va]), set(groups[te])
    check(
        "train/val/test sample sets disjoint and exhaustive",
        not set(tr) & set(va)
        and not set(tr) & set(te)
        and not set(va) & set(te)
        and len(tr) + len(va) + len(te) == len(groups),
    )
    check(
        "no generated state straddles partitions",
        not gt & gv and not gt & ge and not gv & ge,
    )
    check(
        "canonical train/val excludes outer test",
        not set(canonical_train_val_split(
            len(groups), groups, dataset_name=dataset
        )[0]) & set(te)
        and not set(canonical_train_val_split(
            len(groups), groups, dataset_name=dataset
        )[1]) & set(te),
    )
    table = read_state_table(os.path.join("data", dataset))
    part_of = np.full(n_states, "", dtype=object)
    for state in gt:
        part_of[state] = "train"
    for state in gv:
        part_of[state] = "val"
    for state in ge:
        part_of[state] = "test"
    family = np.asarray(table["family"])
    check(
        "every latent design family spans all partitions",
        all(
            set(part_of[family == name]) == {"train", "val", "test"}
            for name in STATE_FAMILIES
        ),
    )
    anchor_level_coverage = True
    for family_name, targets in (
        ("scour_only", (2, 3)),
        ("bearing_only", (1, 2)),
    ):
        for target in targets:
            for level in range(1, 6):
                mask = (
                    (family == family_name)
                    & (table["anchor_target"] == target)
                    & (table["anchor_level"] == level)
                )
                anchor_level_coverage &= (
                    list(part_of[mask]).count("train") == 3
                    and list(part_of[mask]).count("val") == 1
                    and list(part_of[mask]).count("test") == 1
                )
    check(
        "each anchor family×target×level has exact 3/1/1 replicas",
        anchor_level_coverage,
    )
    joint = family == "joint"
    check(
        "joint latent-crack strata span all partitions",
        all(
            set(part_of[joint & (table["latent_crack_on"] == value)])
            == {"train", "val", "test"}
            for value in (False, True)
        ),
    )

    # Exact CRN invariance: row order and active mechanisms change, semantic UID
    # inventory does not. A row-index splitter would fail this mutation.
    permutation = np.random.default_rng(991).permutation(n_states)
    dataset_off = "fixture_off_reordered"
    build_fixture(
        os.path.join("data", dataset_off),
        order=permutation,
        mechanisms_active=False,
    )
    on_map = _partition_by_uid(dataset, n_states)
    off_map = _partition_by_uid(dataset_off, n_states)
    check(
        "common UID partition parity survives row/DC reorder + dormant mechanisms",
        on_map == off_map,
    )
    on_manifest = json.load(open(
        os.path.join("data", dataset, "split_manifest.json"), encoding="utf-8"
    ))
    off_manifest = json.load(open(
        os.path.join("data", dataset_off, "split_manifest.json"), encoding="utf-8"
    ))
    check(
        "row-index assignments changed while semantic mapping stayed fixed",
        on_manifest["assignment"] != off_manifest["assignment"]
        and on_manifest["assignment_by_uid_sha256"]
        == off_manifest["assignment_by_uid_sha256"],
    )
    crn_invariants = (
        "damage_seed",
        "random_stream_schedule_version",
        "state_stream_names",
        "passage_stream_names",
        "passages_per_state",
        "state_uid_inventory",
        "state_uid_inventory_sha256",
        "state_seed_id_by_uid_sha256",
        "state_named_stream_by_uid_sha256",
        "passage_named_stream_by_uid_sha256",
        "joint_state_uid_inventory",
        "joint_state_uid_inventory_sha256",
        "family_counts",
        "latent_design_root_sha256",
    )
    on_identity = on_manifest["state_identity"]
    off_identity = off_manifest["state_identity"]
    check(
        "all causal CRN identities survive row reorder + mechanism activation",
        all(on_identity[field] == off_identity[field]
            for field in crn_invariants)
        and on_identity["state_uid_row_order_sha256"]
        != off_identity["state_uid_row_order_sha256"]
        and on_identity["state_identity_root_sha256"]
        != off_identity["state_identity_root_sha256"],
    )

    manifest_path = os.path.join("data", dataset, "split_manifest.json")
    tampered = json.load(open(manifest_path, encoding="utf-8"))
    old_partition = tampered["assignment_by_uid"][0]["partition"]
    tampered["assignment_by_uid"][0]["partition"] = (
        "test" if old_partition != "test" else "train"
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(tampered, handle)
    try:
        canonical_grouped_splits(len(groups), groups, dataset_name=dataset)
        check("tampered split manifest rejected", False)
    except RuntimeError:
        check("tampered split manifest rejected", True)
    os.remove(manifest_path)
    canonical_grouped_splits(len(groups), groups, dataset_name=dataset)

    def mutate_state_table(name: str, mutation, expected: str) -> None:
        target = os.path.join("data", name)
        shutil.copytree(os.path.join("data", dataset), target)
        path = os.path.join(target, "damage_states.mat")
        loaded = sio.loadmat(path)
        payload = {
            key: value for key, value in loaded.items()
            if not key.startswith("__")
        }
        mutation(payload)
        sio.savemat(path, payload)
        try:
            read_state_table(target)
            check(expected, False, "mutation was accepted")
        except RuntimeError:
            check(expected, True)

    mutate_state_table(
        "mut_duplicate_uid",
        lambda payload: payload["StateUID"].__setitem__(
            (1, 0), payload["StateUID"][0, 0]
        ),
        "duplicate StateUID rejected",
    )
    mutate_state_table(
        "mut_missing_uid",
        lambda payload: payload.pop("StateUID"),
        "missing StateUID rejected",
    )
    mutate_state_table(
        "mut_misaligned_stream",
        lambda payload: payload["StateSeedID"].__setitem__(
            ([0, 1], 0), payload["StateSeedID"][[1, 0], 0]
        ),
        "UID/StateSeedID misalignment rejected",
    )
    mutate_state_table(
        "mut_duplicate_stream",
        lambda payload: payload["StateSeedID"].__setitem__(
            (1, 0), payload["StateSeedID"][0, 0]
        ),
        "duplicate StateSeedID rejected",
    )
    mutate_state_table(
        "mut_named_stream",
        lambda payload: payload["StateNamedStreamSeedID"].__setitem__(
            (0, 0), int(payload["StateNamedStreamSeedID"][0, 0]) + 1
        ),
        "mutated named RNG substream rejected",
    )

    # A unique but renamed UID cannot be authenticated against the persisted
    # UID split: the next split call must reject it rather than silently assign
    # by the unchanged DC row.
    renamed_dir = os.path.join("data", "mut_renamed_uid")
    shutil.copytree(os.path.join("data", dataset), renamed_dir)
    renamed_path = os.path.join(renamed_dir, "damage_states.mat")
    loaded = sio.loadmat(renamed_path)
    payload = {
        key: value for key, value in loaded.items()
        if not key.startswith("__")
    }
    renamed_uid = str(np.ravel(payload["StateUID"][0, 0])[0]) + "-renamed"
    payload["StateUID"][0, 0] = np.asarray([renamed_uid])
    payload["StateSeedID"][0, 0] = _stream(renamed_uid)
    sio.savemat(renamed_path, payload)
    try:
        canonical_grouped_splits(
            n_states, np.arange(n_states), dataset_name="mut_renamed_uid"
        )
        check("renamed UID rejected by persisted split identity", False)
    except RuntimeError:
        check("renamed UID rejected by persisted split identity", True)

    try:
        canonical_grouped_splits(len(groups), groups)
        check("grouped mode without dataset_name rejected", False)
    except RuntimeError:
        check("grouped mode without dataset_name rejected", True)

    from sklearn.model_selection import train_test_split  # noqa: E402

    tr_legacy, va_legacy = canonical_train_val_split(1000, None)
    tr_expected, va_expected = train_test_split(
        np.arange(1000), test_size=0.20, random_state=42
    )
    check(
        "legacy classification fallback bit-compatible",
        np.array_equal(tr_legacy, tr_expected)
        and np.array_equal(va_legacy, va_expected),
    )
finally:
    os.chdir(cwd0)
    shutil.rmtree(tmp, ignore_errors=True)


real = sorted(glob.glob(
    os.path.join("cache", "**", "cache_*_gs*_groups.npy"),
    recursive=True,
))
if not real:
    print("  [SKIP] no real group caches in cache/ yet")
else:
    for group_path in real:
        groups = np.load(group_path)
        check(
            os.path.basename(group_path),
            groups.min() == 0
            and len(np.unique(groups)) == groups.max() + 1,
            f"states={groups.max() + 1}",
        )

print()
print(
    "SPLIT GROUPING: ALL PASS"
    if FAILURES == 0
    else f"SPLIT GROUPING: {FAILURES} CHECK(S) FAILED"
)
sys.exit(1 if FAILURES else 0)
