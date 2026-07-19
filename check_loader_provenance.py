"""check_loader_provenance.py - adversarial loader-validation test (audit R4).

Run:  python check_loader_provenance.py   (needs numpy + scipy)

Builds a tiny VALID multi-output dataset in a temp dir, asserts it loads, then
mutates it into each bad variant and asserts the loader REJECTS it. This proves
that stale (R2/R3 schema), incomplete (gap / short state), corrupt (bad
contact_log / NaN tension), and mixed-provenance (fingerprint mismatch) datasets
cannot silently enter the pipeline. MUST print ALL PASS before launching runs.
"""
import hashlib
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import _load_multi_output, _EXPECTED_GEN_SCHEMA  # noqa: E402

_DOF_SOURCE = {0: ('AcelPrimVag', 0), 1: ('AcelPrimVag', 1), 2: ('AcelPrimVag', 2),
               3: ('AcelRodaPrimVag', 0), 4: ('AcelRodaPrimVag', 1),
               5: ('PitchPrimVag', 0), 6: ('PitchPrimVag', 1), 7: ('PitchPrimVag', 2)}
NST, NP, L = 3, 4, 64          # 3 states, 4 passages, length 64
FP = "fp-canonical-v1"
fails = 0


def _cellrow(nrows, npass=NP, nan_passage=None):
    a = np.empty((1, npass), dtype=object)
    for p in range(npass):
        arr = np.random.RandomState(p).randn(nrows, L).astype(float)
        if nan_passage is not None and p == nan_passage:
            arr[0, 0] = np.nan
        a[0, p] = arr
    return a


def _raw_meta(npass=NP):
    # RAW-format metadata (mandatory in r7): identity space map, full-window crop.
    # All SIX fields A00 writes; integer-valued crop/dim, finite bridge params.
    return {'DimSpace': np.full((1, npass), float(L)),
            'DimAcel':  np.full((1, npass), float(L)),
            'crop_start': np.ones((1, npass)),
            'crop_end':   np.full((1, npass), float(L)),
            'bridge_samp': np.full((1, npass), float(L)),
            'L_bridge_eff': np.full((1, npass), 60.0)}


def _finalize(path):
    """Recompute source SHA-256 digests + root and (re)write file_digests.mat and
    the 3-line completion marker (schema, fp, root). Called after any STATE write
    so the loader's per-file SHA + root checks reach the case's intended guard."""
    files = sorted(f for f in os.listdir(path) if re.fullmatch(r'\d{4}\.mat', f))
    per = {f: hashlib.sha256(open(os.path.join(path, f), 'rb').read()).hexdigest()
           for f in files}
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    sio.savemat(os.path.join(path, 'file_digests.mat'),
                {'file_digests': {'digest_lines': lines, 'root': root}})
    with open(os.path.join(path, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{FP}\n{root}\n")


def _write_state(path, idx, schema=_EXPECTED_GEN_SCHEMA, fp=FP,
                 clog=None, npass=NP, drop_fp=False, drop_clog=False,
                 family='joint', drop_family=False):
    if clog is None:
        clog = np.column_stack([np.zeros(npass), np.zeros(npass),
                                np.zeros(npass), -1.0e5 * np.ones(npass)])
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
         'gen_schema': schema, 'state_family': family,   # Feature A identity
         'AcelPrimVag': _cellrow(3),
         'AcelRodaPrimVag': _cellrow(4), 'PitchPrimVag': _cellrow(3),
         'contact_log': np.asarray(clog, dtype=float), **_raw_meta(npass)}
    if not drop_fp:
        d['gen_fingerprint'] = fp
    if drop_clog:
        del d['contact_log']
    if drop_family:
        del d['state_family']
    sio.savemat(os.path.join(path, f"{idx:04d}.mat"), {'data': d})
    _finalize(path)      # refresh digests+marker so SHA checks pass -> intended guard


def _write_state_table(path, n_states=NST, families=None):
    """damage_states.mat family table (Feature A) — MANDATORY for the loader."""
    fams = families or ['joint'] * n_states
    sio.savemat(os.path.join(path, 'damage_states.mat'), {
        'StateFamily':  np.array(fams, dtype=object).reshape(-1, 1),
        'AnchorTarget': np.zeros((n_states, 1)), 'AnchorLevel': np.zeros((n_states, 1)),
        'CrackOn':      np.zeros((n_states, 1), dtype=np.uint8),
        'DamageStates': np.tile([0.0, 0.1, 0.2, 0.0], (n_states, 1)),
        'BearingFixity': np.zeros((n_states, 2))})


def _write_manifest(path, n_states=NST, npass=NP, schema=_EXPECTED_GEN_SCHEMA,
                    fp=FP, dano_max=0.60):
    ci = {'n_states': n_states, 'passages_per_state': npass,
          'gen_schema': schema, 'gen_fingerprint': fp,
          'scour_dano_max_frac': dano_max}   # A00's real field name
    sio.savemat(os.path.join(path, 'case_info.mat'), {'case_info': ci})


def _mark_complete(path, schema=_EXPECTED_GEN_SCHEMA, fp=FP):
    # Marker content is validated (audit R7 P3): schema then fingerprint, one each.
    with open(os.path.join(path, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{schema}\n{fp}\n")


def _build(path, manifest=True, complete=True, state_table=True, **state_kw):
    os.makedirs(path, exist_ok=True)
    if manifest:
        _write_manifest(path)
    if state_table:                         # Feature A: family table mandatory
        _write_state_table(path)
    for i in range(1, NST + 1):
        _write_state(path, i, **state_kw)   # each write finalizes digests+marker
    if not complete:                        # simulate an interrupted run
        os.remove(os.path.join(path, '_GENERATION_COMPLETE'))


def _load(path):
    return _load_multi_output(path, [0], 200, [2, 3], _DOF_SOURCE)


def check(name, fn, should_raise):
    global fails
    try:
        fn()
        ok = not should_raise
        detail = "loaded" if not should_raise else "LOADED but should have been REJECTED"
    except Exception as e:                                       # noqa: BLE001
        ok = should_raise
        detail = f"rejected ({type(e).__name__})" if should_raise else f"unexpected: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} - {detail}")
    if not ok:
        fails += 1


root = tempfile.mkdtemp(prefix="advloader_")
try:
    # 0. VALID dataset loads
    p = os.path.join(root, "valid"); _build(p)
    check("valid dataset loads", lambda: _load(p), should_raise=False)

    # 1. Stale schema (simulates an R2/R3 file)
    p = os.path.join(root, "stale"); _build(p)
    _write_state(p, 2, schema="audit-2026-07-17")   # old schema on state 2
    check("stale gen_schema rejected", lambda: _load(p), should_raise=True)

    # 2. Missing gen_fingerprint
    p = os.path.join(root, "nofp"); _build(p)
    _write_state(p, 2, drop_fp=True)
    check("missing gen_fingerprint rejected", lambda: _load(p), should_raise=True)

    # 3. Fingerprint mismatch (mixed provenance)
    p = os.path.join(root, "mixfp"); _build(p)
    _write_state(p, 3, fp="fp-DIFFERENT")
    check("fingerprint mismatch rejected", lambda: _load(p), should_raise=True)

    # 4. Missing middle file (gap) -> incomplete
    p = os.path.join(root, "gap"); _build(p)
    os.remove(os.path.join(p, "0002.mat"))
    check("missing state file (gap) rejected", lambda: _load(p), should_raise=True)

    # 5. contact_log wrong shape (short: 2 rows for 4 passages)
    p = os.path.join(root, "shortclog"); _build(p)
    _write_state(p, 2, clog=np.zeros((2, 4)))
    check("short contact_log rejected", lambda: _load(p), should_raise=True)

    # 6. NaN in F_tension_max
    p = os.path.join(root, "nanF"); _build(p)
    bad = np.column_stack([np.zeros(NP), np.zeros(NP), np.zeros(NP),
                           np.array([np.nan, -1e5, -1e5, -1e5])])
    _write_state(p, 2, clog=bad)
    check("NaN F_tension rejected", lambda: _load(p), should_raise=True)

    # 7. Missing contact_log
    p = os.path.join(root, "noclog"); _build(p)
    _write_state(p, 2, drop_clog=True)
    check("missing contact_log rejected", lambda: _load(p), should_raise=True)

    # 8. Real tension over tolerance -> hard-fail on the FIRST offending passage
    p = os.path.join(root, "tension"); _build(p)
    one_hot = np.column_stack([np.zeros(NP), np.zeros(NP), np.zeros(NP),
                               np.array([5.0e4] + [-1e5] * (NP - 1))])  # 1/NP passages
    _write_state(p, 2, clog=one_hot)
    check("single tension passage hard-fails", lambda: _load(p), should_raise=True)

    # 9. No manifest at all -> reject (manifest is mandatory, audit R5)
    p = os.path.join(root, "nomanifest"); _build(p, manifest=False)
    check("missing manifest rejected", lambda: _load(p), should_raise=True)

    # 10. No completion marker -> reject
    p = os.path.join(root, "notdone"); _build(p, complete=False)
    check("missing completion marker rejected", lambda: _load(p), should_raise=True)

    # 11. Extra numbered file beyond n_states -> reject (exact inventory)
    p = os.path.join(root, "extra"); _build(p)
    _write_state(p, NST + 1)     # 0004.mat when manifest says n_states=3
    check("extra state file rejected", lambda: _load(p), should_raise=True)

    # ── audit R6 C4: exact passage count + label/signal/clog finiteness ──────
    def _write_raw(path, idx, d):
        for k, v in _raw_meta().items():   # ensure RAW metadata unless overridden
            d.setdefault(k, v)
        sio.savemat(os.path.join(path, f"{idx:04d}.mat"), {'data': d})
        _finalize(path)

    def _good_clog(npass=NP):
        return np.column_stack([np.zeros(npass), np.zeros(npass),
                                np.zeros(npass), -1.0e5 * np.ones(npass)])

    # 12. MORE signal passages than the manifest declares -> exact-count reject
    #     (the extra passages would otherwise escape the contact-tension gate).
    p = os.path.join(root, "toomany"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3, NP + 1),
                      'AcelRodaPrimVag': _cellrow(4, NP + 1),
                      'PitchPrimVag': _cellrow(3, NP + 1),
                      'contact_log': _good_clog(NP)})   # clog still NP rows
    check("too many passages rejected", lambda: _load(p), should_raise=True)

    # 13. NaN scour label -> reject
    p = os.path.join(root, "nanlabel"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, np.nan, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("NaN scour label rejected", lambda: _load(p), should_raise=True)

    # 14. Out-of-range scour label (1.5 -> 150%) -> reject
    p = os.path.join(root, "biglabel"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 1.5, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("150% scour label rejected", lambda: _load(p), should_raise=True)

    # 15. NaN acceleration sample -> reject
    p = os.path.join(root, "nansig"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3, nan_passage=1),
                      'AcelRodaPrimVag': _cellrow(4), 'PitchPrimVag': _cellrow(3),
                      'contact_log': _good_clog()})
    check("NaN acceleration rejected", lambda: _load(p), should_raise=True)

    # 16. NaN in a NON-tension contact_log column (col 1) -> reject (all cols now)
    p = os.path.join(root, "nanclog"); _build(p)
    bad_c = _good_clog(); bad_c[0, 1] = np.nan
    _write_state(p, 2, clog=bad_c)
    check("NaN in contact_log col1 rejected", lambda: _load(p), should_raise=True)

    # 17. Npass=0 in the manifest -> reject (was silently substituted before)
    p = os.path.join(root, "zeronpass"); _build(p)
    _write_manifest(p, npass=0)   # overwrite manifest with passages_per_state=0
    check("Npass=0 manifest rejected", lambda: _load(p), should_raise=True)

    # ── audit R7 P3/P4: marker content, dano_max ceiling, flags, channel counts ──
    # 18. Completion marker with WRONG content (schema/fp) -> reject
    p = os.path.join(root, "badmarker"); _build(p)
    _mark_complete(p, schema="audit-WRONG", fp="fp-WRONG")   # overwrite marker
    check("wrong marker content rejected", lambda: _load(p), should_raise=True)

    # 19. Scour label above dano_max (80% when dano_max=60%) -> reject
    p = os.path.join(root, "overdano"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 0.80, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("scour > dano_max rejected", lambda: _load(p), should_raise=True)

    # 20. NaN in a NON-selected scour support (index 0) -> reject (full-vector check)
    p = os.path.join(root, "nanunsel"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[np.nan, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("NaN in unselected support rejected", lambda: _load(p), should_raise=True)

    # 21. Impossible contact flag (col 1 = 9) -> reject (flags must be 0/1)
    p = os.path.join(root, "badflag"); _build(p)
    bad_f = _good_clog(); bad_f[0, 0] = 9.0
    _write_state(p, 2, clog=bad_f)
    check("contact flag=9 rejected", lambda: _load(p), should_raise=True)

    # 22. Negative tension_frac (col 2 = -1) -> reject
    p = os.path.join(root, "negfrac"); _build(p)
    bad_fr = _good_clog(); bad_fr[0, 2] = -1.0
    _write_state(p, 2, clog=bad_fr)
    check("negative tension_frac rejected", lambda: _load(p), should_raise=True)

    # 23. Wheel channel has MORE passages than AcelPrimVag -> reject (per-channel count)
    p = os.path.join(root, "wheeltoomany"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4, NP + 1),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("wheel channel count mismatch rejected", lambda: _load(p), should_raise=True)

    # ── audit R7.1 P5: RAW mandatory, crop validity, dano_max, full-vector ranges ──
    def _write_bare(path, idx, d):     # write WITHOUT auto RAW metadata
        sio.savemat(os.path.join(path, f"{idx:04d}.mat"), {'data': d})
        _finalize(path)

    # 24. Missing RAW field (no DimSpace) -> reject (r7 requires the raw format)
    p = os.path.join(root, "noraw"); _build(p)
    _write_bare(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                       'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                       'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                       'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("missing RAW field (DimSpace) rejected", lambda: _load(p), should_raise=True)

    # 25. crop_end < crop_start -> reject (zero-length signal)
    p = os.path.join(root, "badcrop"); _build(p)
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
         'gen_fingerprint': FP, 'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
         'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog(), **_raw_meta()}
    d['crop_start'] = np.full((1, NP), 40.0); d['crop_end'] = np.full((1, NP), 10.0)
    _write_bare(p, 2, d)
    check("crop_end < crop_start rejected", lambda: _load(p), should_raise=True)

    # 26. dano_max missing from manifest -> reject (mandatory in r7)
    p = os.path.join(root, "nodano"); _build(p)
    sio.savemat(os.path.join(p, 'case_info.mat'), {'case_info': {
        'n_states': NST, 'passages_per_state': NP,
        'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP}})   # no dano_max
    check("missing dano_max rejected", lambda: _load(p), should_raise=True)

    # 27. dano_max out of range (1.5) -> reject
    p = os.path.join(root, "danobig"); _build(p)
    _write_manifest(p, dano_max=1.5)
    check("dano_max=1.5 rejected", lambda: _load(p), should_raise=True)

    # 28. tension_frac_max = 1.5 -> reject (must be in [0,1])
    p = os.path.join(root, "fracbig"); _build(p)
    bad_tf = _good_clog(); bad_tf[0, 2] = 1.5
    _write_state(p, 2, clog=bad_tf)
    check("tension_frac=1.5 rejected", lambda: _load(p), should_raise=True)

    # 29. Missing PitchPrimVag (even though not requested) -> reject (r7 needs all channels)
    p = os.path.join(root, "nopitch"); _build(p)
    _write_bare(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                       'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                       'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                       'contact_log': _good_clog(), **_raw_meta()})
    check("missing PitchPrimVag rejected", lambda: _load(p), should_raise=True)

    # 30. 80% scour in an UN-selected support (index 0) -> reject (full-vector range)
    p = os.path.join(root, "unselbig"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.80, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()})
    check("unselected support 80% rejected", lambda: _load(p), should_raise=True)

    # 31. Missing a RAW field (bridge_samp) -> reject (all 6 required)
    p = os.path.join(root, "nobridge"); _build(p)
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
         'gen_fingerprint': FP, 'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
         'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog(), **_raw_meta()}
    del d['bridge_samp']
    _write_bare(p, 2, d)
    check("missing bridge_samp rejected", lambda: _load(p), should_raise=True)

    # 32. Fractional crop_start (1.5) -> reject (integrality; would be int()-truncated)
    p = os.path.join(root, "fraccrop"); _build(p)
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
         'gen_fingerprint': FP, 'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
         'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog(), **_raw_meta()}
    d['crop_start'] = np.full((1, NP), 1.5)
    _write_bare(p, 2, d)
    check("fractional crop_start rejected", lambda: _load(p), should_raise=True)

    # 33. RAW metadata with Npass+1 entries -> reject
    p = os.path.join(root, "rawlong"); _build(p)
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
         'gen_fingerprint': FP, 'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
         'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog(), **_raw_meta()}
    d['DimSpace'] = np.full((1, NP + 1), float(L))
    _write_bare(p, 2, d)
    check("RAW metadata Npass+1 rejected", lambda: _load(p), should_raise=True)

    # 34. Non-finite in an UN-requested channel (PitchPrimVag) -> reject (all channels)
    p = os.path.join(root, "nanpitch"); _build(p)
    _write_raw(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                      'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                      'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                      'PitchPrimVag': _cellrow(3, nan_passage=1), 'contact_log': _good_clog()})
    check("non-finite unrequested channel rejected", lambda: _load(p), should_raise=True)

    # ── audit R7.1 P4: source content integrity (per-state SHA + root) ──────────
    # 35. SAME-SIZE byte corruption of a state file (NO digest refresh) -> reject
    p = os.path.join(root, "bitflip"); _build(p)
    fpath = os.path.join(p, "0002.mat")
    b = bytearray(open(fpath, 'rb').read()); b[len(b) // 2] ^= 0x01   # flip 1 bit, same size
    open(fpath, 'wb').write(b)                                        # do NOT _finalize
    check("same-size .mat corruption rejected (SHA)", lambda: _load(p), should_raise=True)

    # 36. Missing file_digests.mat -> reject (mandatory in r7)
    p = os.path.join(root, "nodigests"); _build(p)
    os.remove(os.path.join(p, "file_digests.mat"))
    check("missing file_digests rejected", lambda: _load(p), should_raise=True)

    # 37. Marker root != file_digests root -> reject
    p = os.path.join(root, "badroot"); _build(p)
    with open(os.path.join(p, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{FP}\ndeadbeef\n")   # wrong root line 3
    check("marker root mismatch rejected", lambda: _load(p), should_raise=True)

    # 38. VALID dataset with source digests STILL loads (positive control)
    p = os.path.join(root, "valid2"); _build(p)
    check("valid dataset (with source digests) loads", lambda: _load(p), should_raise=False)

    # ── Feature A (2026-07-19): state-family identity ────────────────────────
    # 39. Missing damage_states.mat family table -> reject (mandatory)
    p = os.path.join(root, "notable"); _build(p, state_table=False)
    check("missing state-family table rejected", lambda: _load(p), should_raise=True)

    # 40. File without state_family -> reject (pre-Feature-A file)
    p = os.path.join(root, "nofam"); _build(p)
    _write_state(p, 2, drop_family=True)
    check("missing per-file state_family rejected", lambda: _load(p), should_raise=True)

    # 41. File family != table row -> reject (renamed/mislabelled state file)
    p = os.path.join(root, "mixfam"); _build(p)
    _write_state(p, 3, family='target_healthy')     # table says 'joint'
    check("state_family/table mismatch rejected", lambda: _load(p), should_raise=True)

    print()
    print("LOADER PROVENANCE: ALL PASS" if fails == 0 else
          f"LOADER PROVENANCE: {fails} CHECK(S) FAILED")
finally:
    shutil.rmtree(root, ignore_errors=True)

sys.exit(1 if fails else 0)
