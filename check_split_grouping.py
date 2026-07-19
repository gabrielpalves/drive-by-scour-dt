"""check_split_grouping.py - grouped + STRATIFIED split invariants.

Run:  python check_split_grouping.py
Fast, dependency-light, no real dataset needed (synthesizes a Feature-A family
fixture; additionally audits any real _gs4+ group caches found in cache/).
MUST print ALL PASS before rebuilding bundles / launching runs.

History: 2026-07-17 = grouped split (leak fix); 2026-07-19 Feature A = the
split is now STRATIFIED BY STATE FAMILY (read from damage_states.mat) with a
persisted, verified split_manifest.json. This check covers both eras'
invariants plus the new guarantees.

Checks
------
1. No damage state (group) straddles train / inner-val / outer-test — the leak
   the 2026-07-17 audit found.
2. Deterministic; ~60/20/20 by STATES; canonical_train_val_split excludes test.
3. EVERY family present in all three partitions (the Feature-A guarantee that
   replaced the R6 C8C warning), and every (scour_only, pier) / (bearing_only,
   side) anchor stratum spans all three partitions.
4. Joint-block crack flag balanced: crack-on AND crack-off joint states appear
   in every partition.
5. split_manifest.json is written, re-verified on the next call, and a
   TAMPERED manifest is rejected.
6. Grouped mode without dataset_name raises (no silent un-stratified path).
7. The legacy fallback (groups=None) still returns the historical per-passage
   partition (bit-compat with old classification studies).
"""
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
from core.dataset import (canonical_train_val_split,          # noqa: E402
                          canonical_grouped_splits, read_state_table,
                          STATE_FAMILIES, _EXPECTED_GEN_SCHEMA)

fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails += 1


# ── Feature-A family fixture: an s13-like layout (300 states) ────────────────
# 12 target_healthy + 2 piers x (4 levels x 2 reps) scour_only + 2 sides x 8
# bearing_only + 6 nuisance_only + 250 joint. Mirrors A00's construction.
def build_fixture(ds_dir, n_joint=250, seed=7):
    rng = np.random.default_rng(seed)
    n_supp, dano_max, fix_max = 4, 0.60, 0.95
    fam, atgt, alvl, crack = [], [], [], []
    scour_rows, fix_rows = [], []
    for _ in range(12):                                   # target_healthy
        fam.append('target_healthy'); atgt.append(0); alvl.append(0); crack.append(False)
        scour_rows.append(np.zeros(n_supp)); fix_rows.append(np.zeros(2))
    levels = np.linspace(dano_max / 4, dano_max, 4)
    for pier in (2, 3):                                   # scour_only x 2 reps
        for _rep in range(2):
            for lv, sv in enumerate(levels, start=1):
                fam.append('scour_only'); atgt.append(pier); alvl.append(lv)
                crack.append(False)
                row = np.zeros(n_supp); row[pier - 1] = sv
                scour_rows.append(row); fix_rows.append(np.zeros(2))
    blevels = np.linspace(fix_max / 4, fix_max, 4)
    for side in (1, 2):                                   # bearing_only x 2 reps
        for _rep in range(2):
            for lv, bv in enumerate(blevels, start=1):
                fam.append('bearing_only'); atgt.append(side); alvl.append(lv)
                crack.append(False)
                row = np.zeros(2); row[side - 1] = bv
                scour_rows.append(np.zeros(n_supp)); fix_rows.append(row)
    for _ in range(6):                                    # nuisance_only
        fam.append('nuisance_only'); atgt.append(0); alvl.append(0); crack.append(True)
        scour_rows.append(np.zeros(n_supp)); fix_rows.append(np.zeros(2))
    for _ in range(n_joint):                              # joint LHS block
        fam.append('joint'); atgt.append(0); alvl.append(0)
        crack.append(bool(rng.random() < 0.25))
        row = np.zeros(n_supp); row[[1, 2]] = rng.random(2) * dano_max
        scour_rows.append(row); fix_rows.append(rng.random(2) * fix_max)
    os.makedirs(ds_dir, exist_ok=True)
    sio.savemat(os.path.join(ds_dir, 'damage_states.mat'), {
        'StateFamily':  np.array(fam, dtype=object).reshape(-1, 1),
        'AnchorTarget': np.array(atgt, dtype=float).reshape(-1, 1),
        'AnchorLevel':  np.array(alvl, dtype=float).reshape(-1, 1),
        'CrackOn':      np.array(crack, dtype=np.uint8).reshape(-1, 1),
        'DamageStates': np.vstack(scour_rows),
        'BearingFixity': np.vstack(fix_rows),
        'BearingStates': np.vstack(fix_rows) * 1e9,      # k_r stand-in
        'k_ref_bear': 2.31e9, 'scour_supports': np.array([[2.0, 3.0]])})
    n_states = len(fam)
    sio.savemat(os.path.join(ds_dir, 'case_info.mat'),
                {'case_info': {'n_states': n_states, 'passages_per_state': 50,
                               'gen_schema': _EXPECTED_GEN_SCHEMA,
                               'gen_fingerprint': 'fp-splitfix',
                               'scour_dano_max_frac': dano_max}})
    return n_states


tmp = tempfile.mkdtemp(prefix="splitcheck_")
cwd0 = os.getcwd()
try:
    os.chdir(tmp)                       # the split resolves data/<dataset_name>
    DS = "fixture_split"
    n_states = build_fixture(os.path.join("data", DS))
    npass = 50
    groups = np.repeat(np.arange(n_states), npass)

    # ---- 1+2: grouped-split invariants (as before, on the stratified path) --
    tr, va, te = canonical_grouped_splits(len(groups), groups, dataset_name=DS)
    check("train/val/test pairwise disjoint",
          len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(tr, te)) == 0
          and len(np.intersect1d(va, te)) == 0)
    check("all samples used (train+val+test)",
          len(tr) + len(va) + len(te) == len(groups))
    gt, gv, ge = set(groups[tr]), set(groups[va]), set(groups[te])
    check("no state straddles train/val/test",
          not (gt & gv) and not (gt & ge) and not (gv & ge))
    fv, ft = len(gv) / n_states, len(ge) / n_states
    check("~20% states in inner-val", 0.15 <= fv <= 0.25, f"val = {fv:.1%}")
    check("~20% states in outer test", 0.15 <= ft <= 0.25, f"test = {ft:.1%}")
    tr2, va2, te2 = canonical_grouped_splits(len(groups), groups, dataset_name=DS)
    check("deterministic (same call -> identical split)",
          np.array_equal(tr, tr2) and np.array_equal(va, va2)
          and np.array_equal(te, te2))
    check("canonical_train_val_split excludes test",
          len(np.intersect1d(canonical_train_val_split(
              len(groups), groups, dataset_name=DS)[0], te)) == 0
          and len(np.intersect1d(canonical_train_val_split(
              len(groups), groups, dataset_name=DS)[1], te)) == 0)
    blocks = {g: np.sort(np.where(groups == g)[0]) for g in list(ge)[:5]}
    check("passage blocks kept whole (test)",
          all(np.all(np.isin(b, te)) for b in blocks.values()))

    # ---- 3: FAMILY + anchor-stratum coverage guarantees ---------------------
    table = read_state_table(os.path.join("data", DS))
    fam_arr = np.array(table['family'])
    part_of = np.full(n_states, '', dtype=object)
    for g in gt: part_of[g] = 'train'
    for g in gv: part_of[g] = 'val'
    for g in ge: part_of[g] = 'test'
    cov_ok = all({'train', 'val', 'test'} <=
                 set(part_of[fam_arr == f]) for f in sorted(set(fam_arr)))
    check("EVERY family present in all 3 partitions", cov_ok,
          str({f: sorted(set(part_of[fam_arr == f])) for f in sorted(set(fam_arr))}))
    strat_ok = True
    for f, tgts in (("scour_only", (2, 3)), ("bearing_only", (1, 2))):
        for t in tgts:
            m = (fam_arr == f) & (table['anchor_target'] == t)
            if {'train', 'val', 'test'} - set(part_of[m]):
                strat_ok = False
    check("every (family, target) anchor stratum spans all 3 partitions", strat_ok)

    # ---- 4: joint crack balance --------------------------------------------
    m_j = fam_arr == 'joint'
    ok_crack = all({'train', 'val', 'test'} <=
                   set(part_of[m_j & (table['crack_on'] == flag)])
                   for flag in (True, False))
    check("joint crack-on AND crack-off in all partitions", ok_crack)

    # ---- 5: split manifest persisted + verified + tamper-detected -----------
    man_path = os.path.join("data", DS, "split_manifest.json")
    check("split_manifest.json written", os.path.exists(man_path))
    man = json.load(open(man_path))
    check("manifest binds dataset fingerprint",
          man.get("dataset_gen_fingerprint") == "fp-splitfix")
    check("manifest assignment matches the returned split",
          all(man["assignment"][g] == part_of[g] for g in range(n_states)))
    # tamper: flip one state's recorded partition -> next call must refuse
    man["assignment"][0] = "test" if man["assignment"][0] != "test" else "train"
    json.dump(man, open(man_path, "w"))
    try:
        canonical_grouped_splits(len(groups), groups, dataset_name=DS)
        check("TAMPERED split manifest rejected", False, "split ran anyway")
    except RuntimeError as e:
        check("TAMPERED split manifest rejected", True, type(e).__name__)
    os.remove(man_path)                 # restore for any later use

    # ---- 6: no silent un-stratified grouped path ----------------------------
    try:
        canonical_grouped_splits(len(groups), groups)
        check("grouped mode without dataset_name rejected", False)
    except RuntimeError:
        check("grouped mode without dataset_name rejected", True)

    # ---- 7: legacy per-passage fallback (classification) --------------------
    from sklearn.model_selection import train_test_split  # noqa: E402
    trL, vaL = canonical_train_val_split(1000, None)
    trR, vaR = train_test_split(np.arange(1000), test_size=0.20, random_state=42)
    check("legacy fallback bit-compat",
          np.array_equal(trL, trR) and np.array_equal(vaL, vaR))
finally:
    os.chdir(cwd0)
    shutil.rmtree(tmp, ignore_errors=True)

# ---- real caches, if present (structure audit only: the dataset dir name ----
# cannot be recovered from a cache stem reliably, so no split is recomputed) --
real = sorted(glob.glob(os.path.join("cache", "**", "cache_*_gs*_groups.npy"),
                        recursive=True))
if not real:
    print("  [SKIP] no real group caches in cache/ yet")
else:
    for gp in real:
        g = np.load(gp)
        sizes = np.bincount(g)
        check(os.path.basename(gp),
              g.min() == 0 and len(np.unique(g)) == g.max() + 1,
              f"states={g.max() + 1}, passage counts={np.unique(sizes)[:5]}")

print()
print("SPLIT GROUPING: ALL PASS" if fails == 0 else
      f"SPLIT GROUPING: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
