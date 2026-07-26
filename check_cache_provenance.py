"""check_cache_provenance.py - adversarial CACHE-provenance test (audit R7.1 P4/P5).

Run:  python check_cache_provenance.py   (needs numpy + scipy + torch + sklearn)

Builds a tiny VALID r7 dataset (with source SHA digests + 3-line marker) under a
temp data/ folder, exercises get_or_create_cache, then verifies that every
tamper is caught: swapped/corrupted feature/label/GROUPS/scaler artifacts, a
count-preserving group SWAP, a same-size source .mat overwrite, a wrong-content
marker, an interrupted publication (missing sidecar), and CONCURRENT builds of
the same stem (the per-stem lock must serialise them without error). MUST print
ALL PASS before trusting the cache on a multi-day campaign.
"""
import concurrent.futures as cf
import hashlib
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import get_or_create_cache, _EXPECTED_GEN_SCHEMA  # noqa: E402
from core.protocol import read_dataset_provenance  # noqa: E402

# Keep the integration fixture at the campaign's 512 PAA segments.  PAA is a
# reduction and now correctly rejects n_segments > raw length.
NST, NP, L, FP = 4, 4, 512, "fp-cache-prov"
CFG = {'method': 'PAA', 'dofs': [0], 'task': 'regression', 'target_supports': [2, 3]}
fails = 0


def _cellrow(nrows):
    a = np.empty((1, NP), dtype=object)
    for p in range(NP):
        a[0, p] = np.random.RandomState(p).randn(nrows, L).astype(float)
    return a


def _raw_meta():
    return {'DimSpace': np.full((1, NP), float(L)), 'DimAcel': np.full((1, NP), float(L)),
            'crop_start': np.ones((1, NP)), 'crop_end': np.full((1, NP), float(L)),
            'bridge_samp': np.full((1, NP), float(L)), 'L_bridge_eff': np.full((1, NP), 60.0)}


def _finalize(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if re.fullmatch(r'\d{4}\.mat', f))
    per = {f: hashlib.sha256(open(os.path.join(data_dir, f), 'rb').read()).hexdigest()
           for f in files}
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    sio.savemat(os.path.join(data_dir, 'file_digests.mat'),
                {'file_digests': {'digest_lines': lines, 'root': root}})
    with open(os.path.join(data_dir, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{FP}\n{root}\n")


def _build_dataset(data_dir):
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)
    sio.savemat(os.path.join(data_dir, 'case_info.mat'), {'case_info': {
        'n_states': NST, 'passages_per_state': NP, 'gen_schema': _EXPECTED_GEN_SCHEMA,
        'gen_fingerprint': FP, 'scour_dano_max_frac': 0.60}})
    # Feature A (2026-07-19): the state-family table is MANDATORY for the
    # loader + stratified split; all fixture states are plain 'joint'.
    sio.savemat(os.path.join(data_dir, 'damage_states.mat'), {
        'StateFamily':  np.array(['joint'] * NST, dtype=object).reshape(-1, 1),
        'AnchorTarget': np.zeros((NST, 1)), 'AnchorLevel': np.zeros((NST, 1)),
        'CrackOn':      np.zeros((NST, 1), dtype=np.uint8),
        'DamageStates': np.tile([0.0, 0.1, 0.2, 0.0], (NST, 1)),
        'BearingFixity': np.zeros((NST, 2))})
    clog = np.column_stack([np.zeros(NP)] * 3 + [-1e5 * np.ones(NP)])
    for i in range(1, NST + 1):
        d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
             'gen_fingerprint': FP, 'state_family': 'joint',
             'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
             'PitchPrimVag': _cellrow(3), 'contact_log': clog, **_raw_meta()}
        sio.savemat(os.path.join(data_dir, f"{i:04d}.mat"), {'data': d})
    _finalize(data_dir)


def check(name, cond):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def _art(cache_dir, suffix):
    return [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(suffix)][0]


def _rebuilds_ok(ds, cache_dir, expect_shape):
    """get_or_create_cache should rebuild-and-load (identical shape) after a tamper."""
    import gc
    X, y, sc, g = get_or_create_cache(CFG, ds, cache_dir)
    ok = tuple(X.shape) == expect_shape and sorted(np.unique(g).tolist()) == list(range(NST))
    del X, y, g
    gc.collect()
    return ok


def main():
    global fails
    root = tempfile.mkdtemp(prefix="cacheprov_")
    ds = "_cacheprov_ds"
    data_dir = os.path.join("data", ds)
    cache_dir = os.path.join(root, "cache")
    try:
        _build_dataset(data_dir)

        # 1. First build + prov structure
        import gc
        X, y, sc, g = get_or_create_cache(CFG, ds, cache_dir)
        shape = tuple(X.shape)
        import json
        prov = json.load(open(_art(cache_dir, '_prov.json')))
        check("prov has source + 4 artifact digests",
              set(prov['artifacts']) == {'feat', 'labels', 'groups', 'scaler'})
        check("source carries root + manifest digest + dano_max",
              bool(prov['source'].get('source_root')) and bool(prov['source'].get('manifest_sha256'))
              and prov['source'].get('dano_max') == 0.60)
        check("no leftover .tmp / .lock", not any(f.endswith(('.tmp', '.lock'))
                                                  for f in os.listdir(cache_dir)))
        del X, y, g
        gc.collect()

        # 2. Fast-path reuse (no rebuild)
        X2, y2, sc2, g2 = get_or_create_cache(CFG, ds, cache_dir)
        check("fast-path reuse identical", tuple(X2.shape) == shape)
        del X2, y2, g2
        gc.collect()

        # 3. Tamper each artifact (feat/labels/groups/scaler) -> rebuild
        for suf in ('.npy', '_labels.npy', '_groups.npy'):
            f = _art(cache_dir, suf)
            b = bytearray(open(f, 'rb').read()); b[-1] ^= 0xFF; open(f, 'wb').write(b)
            check(f"tampered {suf} -> rebuilt", _rebuilds_ok(ds, cache_dir, shape))
        sc_f = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
                if f.startswith('scaler_')][0]
        b = bytearray(open(sc_f, 'rb').read()); b[-1] ^= 0xFF; open(sc_f, 'wb').write(b)
        check("tampered scaler -> rebuilt", _rebuilds_ok(ds, cache_dir, shape))

        # 4. Semantic group SWAP (count-preserving) -> rebuilt to canonical block
        gpath = _art(cache_dir, '_groups.npy')
        swapped = np.load(gpath).copy()
        swapped[:NP], swapped[NP:2 * NP] = 1, 0        # swap states 0 and 1, counts intact
        np.save(gpath, swapped)
        check("count-preserving group swap -> rebuilt", _rebuilds_ok(ds, cache_dir, shape))

        # 5. Interrupted publication: sidecar missing -> rebuild
        os.remove(_art(cache_dir, '_prov.json'))
        check("missing prov sidecar -> rebuilt", _rebuilds_ok(ds, cache_dir, shape))

        # 6. Same-size SOURCE .mat overwrite -> source SHA rejects even while a
        #    valid cache + provenance sidecar exist (audit r4 fast-path closure).
        sfile = os.path.join(data_dir, "0002.mat")
        bb = bytearray(open(sfile, 'rb').read()); bb[len(bb) // 2] ^= 0x01
        open(sfile, 'wb').write(bb)                     # same size, NO _finalize
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("same-size source corruption rejected", False)
        except Exception:
            check("same-size source corruption rejected", True)
        _build_dataset(data_dir)                        # restore a clean dataset
        shutil.rmtree(cache_dir, ignore_errors=True)

        # 7. Valid replacement after driver import must not train under the old
        #    protocol identity (TOCTOU closure).
        expected_source = read_dataset_provenance(data_dir)
        bound_cfg = {
            **CFG,
            "protocol_hash": "fixture-old-protocol",
            "protocol_descriptor": {
                "rung": {"dataset_provenance": expected_source}
            },
        }
        get_or_create_cache(bound_cfg, ds, cache_dir)
        table_path = os.path.join(data_dir, "damage_states.mat")
        table = sio.loadmat(table_path)
        table["AnchorLevel"] = np.ones((NST, 1))
        sio.savemat(table_path, table)
        try:
            get_or_create_cache(bound_cfg, ds, cache_dir)
            check("post-import valid source replacement rejects stale protocol", False)
        except RuntimeError:
            check("post-import valid source replacement rejects stale protocol", True)
        _build_dataset(data_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

        # 8. Wrong marker content -> source 'incomplete' -> rebuild -> loader rejects
        get_or_create_cache(CFG, ds, cache_dir)         # fresh valid cache
        with open(os.path.join(data_dir, '_GENERATION_COMPLETE'), 'w') as fh:
            fh.write("WRONG\nWRONG\nWRONG\n")
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("wrong marker content rejected", False)
        except Exception:
            check("wrong marker content rejected", True)
        _build_dataset(data_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

        # 9. CONCURRENT builds of the SAME stem (per-stem lock serialises them)
        def _one(_):
            X, y, sc, g = get_or_create_cache(CFG, ds, cache_dir)
            n = len(g)
            del X, y, g
            return n
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(_one, range(4)))
        check("4 concurrent threads build without error",
              all(n == NST * NP for n in res)
              and not any(f.endswith(('.tmp', '.lock')) for f in os.listdir(cache_dir)))

        print("\nCACHE PROVENANCE: ALL PASS" if fails == 0 else
              f"\nCACHE PROVENANCE: {fails} CHECK(S) FAILED")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
