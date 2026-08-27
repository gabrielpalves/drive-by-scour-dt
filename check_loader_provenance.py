"""Adversarial R11 loader/provenance validation.

Run:  python check_loader_provenance.py   (needs numpy + scipy)

Builds a tiny VALID multi-output dataset in a temp dir, asserts it loads, then
mutates it into each bad variant and asserts the loader REJECTS it. This proves
that stale, incomplete, corrupt, mixed-provenance, qualification, incoherent
numerical-environment, and wrong reviewed-source datasets cannot silently enter
the pipeline.  A different, internally authenticated MATLAB release is also
proved portable. MUST print ALL PASS before launching runs.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset import (                                            # noqa: E402
    _EXPECTED_CHANNEL_SCHEMA_ID,
    _EXPECTED_GENERATION_BEHAVIOR_VERSION,
    _EXPECTED_GEN_SCHEMA,
    _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
    _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    _EXPECTED_MATLAB_RELEASE,
    _load_multi_output,
    _read_manifest_generation_metadata,
)
from core.campaign_contract import (                                  # noqa: E402
    EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    EXPECTED_RAIL_END_CLEARANCE_M,
)
from core.source_provenance import generator_source_root              # noqa: E402

NST, NP, L = 3, 4, 64          # 3 states, 4 passages, length 64
BRIDGE_LENGTH_M = 0.01
DIM_SPACE = 1001
fails = 0
_GENERATOR = generator_source_root(Path(__file__).resolve().parent)
_QUALIFICATION_SHA = "a" * 64
DAMAGE_SEED = 1
RNG_SCHEDULE = "uid-named-substreams-v2"
STATE_STREAM_NAMES = (
    "operations", "crack", "profile-state", "track", "profile-phase"
)
PASSAGE_STREAM_NAMES = ("profile-passage", "oor-passage")
STATE_UIDS = tuple(f"fixture-state-{index:03d}" for index in range(1, NST + 1))


def _portable_actual_matlab_environment():
    """Return a coherent live-host identity unlike the campaign reference."""
    replacements = {
        "matlab_product_version": "24.2",
        "parallel_toolbox_version": "24.2",
        "release": "R2024b",
        "statistics_toolbox_version": "24.2",
        "version": "24.2.0.9999999 (R2024b) Portable Fixture",
    }
    rows = []
    for row in _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR.split("\n"):
        key, value = row.split("=", 1)
        rows.append(f"{key}={replacements.get(key, value)}")
    descriptor = "\n".join(rows)
    return {
        "matlab_release": "R2024b",
        "actual_matlab_environment_descriptor": descriptor,
        "actual_matlab_environment_sha256": hashlib.sha256(
            descriptor.encode("utf-8")
        ).hexdigest(),
    }


def _state_seed_id(uid):
    token = f"ttbi-state-seed-v1|damage_seed={DAMAGE_SEED}|{uid}"
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    assert value != 0
    return value


def _named_seed(root, uid, stream, passage=None):
    token = (
        f"{RNG_SCHEDULE}|root={root}|uid={uid}|stream={stream}"
        + ("" if passage is None else f"|pass={passage:05d}")
    )
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    assert value != 0
    return value


def _generation_config_json():
    return json.dumps(
        {
            "schema": _EXPECTED_GEN_SCHEMA,
            "channel_schema_id": _EXPECTED_CHANNEL_SCHEMA_ID,
            "state_design_kind": "five-family-multidamage-v2",
            "generation_behavior_version":
                _EXPECTED_GENERATION_BEHAVIOR_VERSION,
            "campaign_matlab_release": _EXPECTED_MATLAB_RELEASE,
            "campaign_matlab_environment_sha256":
                _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
            "generator_source_root_sha256": _GENERATOR.sha256,
            "qualification_source_sha256": "PRODUCTION",
            "STAGE": "fixture_stage",
            "n_states": NST,
            "Npass": NP,
            "damage_seed": DAMAGE_SEED,
            "rail_end_clearance_m": EXPECTED_RAIL_END_CLEARANCE_M,
            "rail_end_clearance_decision_id":
                EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


GENERATION_CONFIG_JSON = _generation_config_json()
FP = hashlib.sha256(GENERATION_CONFIG_JSON.encode("utf-8")).hexdigest()


def _state_identity(idx, npass=NP):
    uid = f"fixture-state-{idx:03d}"
    root = _state_seed_id(uid)
    state_named = np.asarray([[
        _named_seed(root, uid, name) for name in STATE_STREAM_NAMES
    ]], dtype=np.uint32)
    passage_named = np.asarray([
        [
            _named_seed(root, uid, name, passage=passage)
            for name in PASSAGE_STREAM_NAMES
        ]
        for passage in range(1, npass + 1)
    ], dtype=np.uint32)
    return {
        "state_uid": uid,
        "state_seed_id": np.uint32(root),
        "random_stream_schedule_version": RNG_SCHEDULE,
        "state_named_stream_seed_id": state_named,
        "passage_named_stream_seed_id": passage_named,
        "latent_bearing_fixity": np.zeros((1, 2)),
        "latent_crack_on": np.array([[False]], dtype=np.bool_),
        "crack_on": np.array([[False]], dtype=np.bool_),
        "bearing_fixity": np.zeros((1, 2)),
        "scour_supports": np.array([[2, 3]], dtype=np.uint32),
    }


def _nested_provenance(*, matlab_release=_EXPECTED_MATLAB_RELEASE,
                       qualification=False,
                       qualification_source="PRODUCTION"):
    return {
        'gen_schema': _EXPECTED_GEN_SCHEMA,
        'gen_fingerprint': FP,
        'channel_schema_id': _EXPECTED_CHANNEL_SCHEMA_ID,
        'matlab_release': matlab_release,
        'campaign_matlab_release': _EXPECTED_MATLAB_RELEASE,
        'actual_matlab_environment_descriptor':
            _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
        'actual_matlab_environment_sha256':
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        'campaign_matlab_environment_descriptor':
            _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
        'campaign_matlab_environment_sha256':
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        'generator_source_root_sha256': _GENERATOR.sha256,
        'generator_source_digest_lines': _GENERATOR.digest_lines,
        'generator_source_file_count': _GENERATOR.file_count,
        'qualification_source_sha256': qualification_source,
        'release_qualification_run': qualification,
    }


def _top_level_provenance(nested):
    return {
        'file_gen_schema': nested['gen_schema'],
        'file_gen_fingerprint': nested['gen_fingerprint'],
        'file_state_uid': nested['state_uid'],
        'file_state_seed_id': nested['state_seed_id'],
        'file_random_stream_schedule_version':
            nested['random_stream_schedule_version'],
        'file_matlab_release': nested['matlab_release'],
        'file_campaign_matlab_release':
            nested['campaign_matlab_release'],
        'file_release_qualification_run':
            nested['release_qualification_run'],
        'file_actual_matlab_environment_sha256':
            nested['actual_matlab_environment_sha256'],
        'file_campaign_matlab_environment_sha256':
            nested['campaign_matlab_environment_sha256'],
        'file_generator_source_root_sha256':
            nested['generator_source_root_sha256'],
        'file_qualification_source_sha256':
            nested['qualification_source_sha256'],
    }


def _save_state(path, idx, data, *, nested_overrides=None, nested_drop=(),
                top_overrides=None, top_drop=(), defaults_npass=NP):
    nested = _complete_payload_defaults(defaults_npass)
    nested.update(_nested_provenance())
    nested.update(_state_identity(idx))
    nested.update(data)
    nested.update(nested_overrides or {})
    stamp_source = dict(nested)
    for field in nested_drop:
        nested.pop(field, None)
    # save_progress.m derives these stamps from the final nested payload.
    # Individual tests can then mutate one representation independently.
    top = _top_level_provenance(stamp_source)
    top.update(top_overrides or {})
    for field in top_drop:
        top.pop(field, None)
    sio.savemat(
        os.path.join(path, f"{idx:04d}.mat"),
        {'data': nested, **top},
        long_field_names=True,
    )
    _finalize(path)


def _cellrow(nrows, npass=NP, nan_passage=None):
    a = np.empty((1, npass), dtype=object)
    for p in range(npass):
        arr = np.random.RandomState(p).randn(nrows, L).astype(float)
        if nan_passage is not None and p == nan_passage:
            arr[0, 0] = np.nan
        a[0, p] = arr
    return a


def _offset_cellrow(nrows, offset, npass=NP):
    values = _cellrow(nrows, npass)
    for passage in range(npass):
        values[0, passage] = values[0, passage] + float(offset)
    return values


def _raw_meta(npass=NP):
    # Exact generator equations, with a deliberately tiny bridge for a compact
    # behavioral fixture.
    bridge_samp = round(100 * BRIDGE_LENGTH_M)
    velocity = DIM_SPACE * 10.0 / L
    crop_start = 1001
    crop_end = min(crop_start - 1 + bridge_samp + 1831, DIM_SPACE)
    return {'DimSpace': np.full((1, npass), float(DIM_SPACE)),
            'DimAcel':  np.full((1, npass), float(L)),
            'crop_start': np.full((1, npass), float(crop_start)),
            'crop_end':   np.full((1, npass), float(crop_end)),
            'bridge_samp': np.full((1, npass), float(bridge_samp)),
            'L_bridge_eff': np.full((1, npass), BRIDGE_LENGTH_M),
            'Velocidade': np.full((1, npass), velocity)}


def _complete_payload_defaults(npass=NP):
    """One valid instance of the exact closed MATLAB state contract."""
    return {
        'AcelPrimVag': _cellrow(3, npass),
        'AcelRodaPrimVag': _offset_cellrow(4, 1000, npass),
        'AcelWheelsetPrimVag': _offset_cellrow(4, 2000, npass),
        'PitchPrimVag': _cellrow(3, npass),
        'Dano': 0.2,
        'Temperatura': np.full((1, npass), 20.0),
        'VehiclesProps': np.ones((5, 3)),
        'beam_f1_Hz': 4.0,
        'bearing_vector': np.zeros((1, 2)),
        'crack_log': np.zeros((npass, 3)),
        'profile_mode': 'fixed',
        'profile_log': np.ones((1, npass)),
        'track_log': np.zeros((1, npass)),
        'oor_log': np.zeros((1, npass)),
        'contact_log': np.column_stack([
            np.zeros(npass),
            np.zeros(npass),
            np.zeros(npass),
            -1.0e5 * np.ones(npass),
        ]),
        'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
        'state_family': 'joint',
        **_raw_meta(npass),
    }


def _finalize(path):
    """Recompute source SHA-256 digests + root and (re)write file_digests.mat and
    the 3-line completion marker (schema, fp, root). Called after any STATE write
    so the loader's per-file SHA + root checks reach the case's intended guard."""
    files = [
        f for f in os.listdir(path)
        if re.fullmatch(r'\d{4}\.mat', f)
        or f in {'case_info.mat', 'damage_states.mat'}
    ]
    per = {
        f: hashlib.sha256(Path(path, f).read_bytes()).hexdigest()
        for f in sorted(files)
    }
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    sio.savemat(
        os.path.join(path, 'file_digests.mat'),
        {'file_digests': {
            'schema': 'source-digests-v2',
            'scope': 'NNNN.mat+case_info.mat+damage_states.mat',
            'digest_lines': lines,
            'root': root,
        }},
    )
    with open(os.path.join(path, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{FP}\n{root}\n")


def _foreign_generator_attestation():
    """A canonical and internally coherent source identity unlike this tree."""
    rows = _GENERATOR.digest_lines.split("\n")
    name, digest = rows[0].split(":", 1)
    replacement = "0" * 64 if digest != "0" * 64 else "1" * 64
    rows[0] = f"{name}:{replacement}"
    digest_lines = "\n".join(rows)
    return {
        "generator_source_root_sha256":
            hashlib.sha256(digest_lines.encode("utf-8")).hexdigest(),
        "generator_source_digest_lines": digest_lines,
        "generator_source_file_count": len(rows),
    }


def _write_state(path, idx, schema=_EXPECTED_GEN_SCHEMA, fp=FP,
                 clog=None, npass=NP, drop_fp=False, drop_clog=False,
                 family='joint', drop_family=False,
                 matlab_release=_EXPECTED_MATLAB_RELEASE,
                 qualification=False, drop_release=False,
                 drop_qualification=False,
                 qualification_source="PRODUCTION",
                 nested_overrides=None, nested_drop=(),
                 top_overrides=None, top_drop=()):
    if clog is None:
        clog = np.column_stack([np.zeros(npass), np.zeros(npass),
                                np.zeros(npass), -1.0e5 * np.ones(npass)])
    d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
         'state_family': family,   # Feature A identity
         'AcelPrimVag': _cellrow(3),
         'AcelRodaPrimVag': _cellrow(4), 'PitchPrimVag': _cellrow(3),
         'contact_log': np.asarray(clog, dtype=float), **_raw_meta(npass)}
    if drop_clog:
        del d['contact_log']
    if drop_family:
        del d['state_family']
    overrides = {
        'gen_schema': schema,
        'gen_fingerprint': fp,
        'matlab_release': matlab_release,
        'release_qualification_run': qualification,
        'qualification_source_sha256': qualification_source,
        **(nested_overrides or {}),
    }
    drops = set(nested_drop)
    if drop_clog:
        drops.add('contact_log')
    if drop_family:
        drops.add('state_family')
    if drop_fp:
        drops.add('gen_fingerprint')
    if drop_release:
        drops.add('matlab_release')
    if drop_qualification:
        drops.add('release_qualification_run')
    _save_state(
        path, idx, d,
        nested_overrides=overrides,
        nested_drop=drops,
        top_overrides=top_overrides,
        top_drop=top_drop,
        defaults_npass=npass,
    )


def _write_state_table(path, n_states=NST, families=None):
    """damage_states.mat family table (Feature A) — MANDATORY for the loader."""
    fams = families or ['joint'] * n_states
    identities = [_state_identity(index) for index in range(1, n_states + 1)]
    state_named = np.vstack([
        identity["state_named_stream_seed_id"] for identity in identities
    ])
    passage_named = np.stack([
        identity["passage_named_stream_seed_id"] for identity in identities
    ])
    sio.savemat(os.path.join(path, 'damage_states.mat'), {
        'StateFamily':  np.array(fams, dtype=object).reshape(-1, 1),
        'AnchorTarget': np.zeros((n_states, 1)),
        'AnchorLevel': np.zeros((n_states, 1)),
        'StateUID': np.asarray(
            [identity["state_uid"] for identity in identities],
            dtype=object,
        ).reshape(-1, 1),
        'StateSeedID': np.asarray(
            [identity["state_seed_id"] for identity in identities],
            dtype=np.uint32,
        ).reshape(-1, 1),
        'LatentBearingFixity': np.zeros((n_states, 2)),
        'LatentCrackOn': np.zeros((n_states, 1), dtype=np.uint8),
        'CrackOn': np.zeros((n_states, 1), dtype=np.uint8),
        'StateNamedStreamSeedID': state_named,
        'PassageNamedStreamSeedID': passage_named,
        'PassageNamedStreamSeedIDFlat': passage_named.reshape(
            n_states, -1, order="F"
        ),
        'random_stream_schedule_version': RNG_SCHEDULE,
        'state_stream_names': np.asarray(
            STATE_STREAM_NAMES, dtype=object
        ).reshape(1, -1),
        'passage_stream_names': np.asarray(
            PASSAGE_STREAM_NAMES, dtype=object
        ).reshape(1, -1),
        'DamageStates': np.tile(
            [0.0, 0.1, 0.2, 0.0], (n_states, 1)
        ),
        'BearingFixity': np.zeros((n_states, 2)),
        'scour_supports': np.array([[2, 3]], dtype=np.uint32),
    })


def _write_manifest(path, n_states=NST, npass=NP, schema=_EXPECTED_GEN_SCHEMA,
                    fp=FP, dano_max=0.60,
                    matlab_release=_EXPECTED_MATLAB_RELEASE,
                    qualification=False, drop_release=False,
                    drop_campaign_release=False, drop_qualification=False,
                    qualification_source="PRODUCTION",
                    metadata_overrides=None, metadata_drop=(),
                    drop_dano=False):
    ci = {'n_states': n_states, 'passages_per_state': npass,
          'gen_schema': schema, 'gen_fingerprint': fp,
          'channel_schema_id': _EXPECTED_CHANNEL_SCHEMA_ID,
          'state_design_kind': 'five-family-multidamage-v2',
          'generation_config_json': GENERATION_CONFIG_JSON,
          'case_name': 'fixture_dataset',
          'stage': 'fixture_stage',
          'damage_mode': 'multi_scour',
          'rail_end_clearance_m': EXPECTED_RAIL_END_CLEARANCE_M,
          'rail_end_clearance_decision_id':
              EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
          'L_bridge_m': BRIDGE_LENGTH_M,
          'num_spans': 3,
          'num_supports': 4,
          'scour_supports': '[2 3]',
          'scour_dano_max_frac': dano_max,
          'n_target_healthy': 0,
          'n_scour_only': 0,
          'n_bearing_only': 0,
          'n_nuisance_only': 0,
          'n_joint': n_states,
          'bearing_mode': 'off',
          'bearing_label': 'fixity_ratio',
          'use_crack_eov': False,
          'crack_draw': 'per_state',
          'profile_mode': 'fixed',
          'profile_draw': 'fixed_shared',
          'profile_jitter_sd_mm': 0.0,
          'use_track_eov': False,
          'track_draw': 'per_state',
          'track_L_app': 30.0,
          'track_L_after': 30.0,
          'use_oor_eov': False,
          'oor_flats_enabled': False,
          'use_signal_noise': False,
          'use_vehicle_variability': True,
          'use_speed_variability': True,
          'use_temp_variability': True,
          'generation_behavior_version':
              _EXPECTED_GENERATION_BEHAVIOR_VERSION,
          'matlab_release': matlab_release,
          'campaign_matlab_release': _EXPECTED_MATLAB_RELEASE,
          'release_qualification_run': qualification,
          'actual_matlab_environment_descriptor':
              _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
          'actual_matlab_environment_sha256':
              _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
          'campaign_matlab_environment_descriptor':
              _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
          'campaign_matlab_environment_sha256':
              _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
          'generator_source_root_sha256': _GENERATOR.sha256,
          'generator_source_digest_lines': _GENERATOR.digest_lines,
          'generator_source_file_count': _GENERATOR.file_count,
          'qualification_source_sha256': qualification_source}
    ci.update(metadata_overrides or {})
    if drop_release:
        del ci['matlab_release']
    if drop_campaign_release:
        del ci['campaign_matlab_release']
    if drop_qualification:
        del ci['release_qualification_run']
    if drop_dano:
        del ci['scour_dano_max_frac']
    for field in metadata_drop:
        ci.pop(field, None)
    sio.savemat(
        os.path.join(path, 'case_info.mat'),
        {'case_info': ci},
        long_field_names=True,
    )


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


def _rewrite_coherent_actual_environment(path, environment):
    """Apply one authenticated actual environment to manifest and all states."""
    actual_fields = {
        key: value
        for key, value in environment.items()
        if key != "matlab_release"
    }
    _write_manifest(
        path,
        matlab_release=environment["matlab_release"],
        metadata_overrides=actual_fields,
    )
    for state_idx in range(1, NST + 1):
        _write_state(
            path,
            state_idx,
            matlab_release=environment["matlab_release"],
            nested_overrides=actual_fields,
        )


def _load(path):
    return _load_multi_output(path, [0], 200, [2, 3])


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

    def _physical_dofs_are_deployed():
        X, _, _ = _load_multi_output(p, [3, 4], 200, [2, 3])
        if X.shape[1] != 2 or not np.all(X > 1000.0):
            raise AssertionError(
                "DOFs 3-4 did not come from the offset AcelWheelset fixture"
            )

    check("physical8_v1 maps DOFs 3-4 to AcelWheelsetPrimVag",
          _physical_dofs_are_deployed, should_raise=False)

    p = os.path.join(root, "missingchannelschema"); _build(p)
    _write_manifest(p, metadata_drop=("channel_schema_id",))
    check("missing manifest channel schema rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "wrongchannelschema"); _build(p)
    _write_manifest(
        p, metadata_overrides={"channel_schema_id": "legacy_virtual8"}
    )
    check("foreign manifest channel schema rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "missingclearance"); _build(p)
    _write_manifest(p, metadata_drop=("rail_end_clearance_m",))
    check("missing explicit rail-end clearance rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "wrongclearance"); _build(p)
    _write_manifest(p, metadata_overrides={"rail_end_clearance_m": 15.0})
    check("wrong rail-end clearance rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "missingclearanceid"); _build(p)
    _write_manifest(p, metadata_drop=("rail_end_clearance_decision_id",))
    check("missing clearance decision identity rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "wrongclearanceid"); _build(p)
    _write_manifest(
        p,
        metadata_overrides={
            "rail_end_clearance_decision_id":
                "paper1-rail-domain-clearance-c15-v1"
        },
    )
    check("wrong clearance decision identity rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "missingstatechannelschema"); _build(p)
    _write_state(p, 2, nested_drop=("channel_schema_id",))
    check("missing per-state channel schema rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "mixedstatechannelschema"); _build(p)
    _write_state(
        p, 2,
        nested_overrides={"channel_schema_id": "legacy_virtual8"},
    )
    check("mixed per-state channel schema rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "missingwheelsetpayload"); _build(p)
    _write_state(p, 2, nested_drop=("AcelWheelsetPrimVag",))
    check("missing physical wheelset payload rejected", lambda: _load(p),
          should_raise=True)

    # Strict scalar/canonical manifest grammar: no int() truncation, no [0]
    # selection from arrays, and no non-SHA provenance aliases.
    p = os.path.join(root, "fractionalmanifest"); _build(p)
    _write_manifest(p, n_states=NST + 0.5)
    check("fractional n_states manifest rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "arraycountmanifest"); _build(p)
    _write_manifest(p, npass=np.array([NP, NP]))
    check("multi-element passage count rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "arrayschemamanifest"); _build(p)
    _write_manifest(
        p,
        schema=np.array(
            [_EXPECTED_GEN_SCHEMA, _EXPECTED_GEN_SCHEMA],
            dtype=object,
        ),
    )
    check("multi-element gen_schema rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "arrayfpmanifest"); _build(p)
    _write_manifest(p, fp=np.array([FP, "d" * 64], dtype=object))
    check("multi-element gen_fingerprint rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "uppercasefpmanifest"); _build(p)
    _write_manifest(p, fp=FP.upper())
    check("noncanonical uppercase fingerprint rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "arraystructmanifest"); _build(p)
    ci_path = os.path.join(p, "case_info.mat")
    ci_raw = sio.loadmat(ci_path)["case_info"]
    sio.savemat(
        ci_path,
        {"case_info": np.concatenate((ci_raw, ci_raw), axis=1)},
        long_field_names=True,
    )
    _finalize(p)
    check("multi-element case_info struct rejected", lambda: _load(p),
          should_raise=True)
    check(
        "generation-metadata reader rejects multi-element case_info struct",
        lambda: _read_manifest_generation_metadata(p),
        should_raise=True,
    )

    p = os.path.join(root, "directorymanifest"); _build(p)
    ci_path = os.path.join(p, "case_info.mat")
    os.remove(ci_path)
    os.mkdir(ci_path)
    check("non-regular case_info.mat rejected", lambda: _load(p),
          should_raise=True)

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
        d.setdefault('state_family', 'joint')
        _save_state(path, idx, d)

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
    def _write_bare(path, idx, d, *, missing=()):
        d.setdefault('state_family', 'joint')
        _save_state(path, idx, d, nested_drop=missing)

    # 24. Missing RAW field (no DimSpace) -> reject (r7 requires the raw format)
    p = os.path.join(root, "noraw"); _build(p)
    _write_bare(p, 2, {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]),
                       'gen_schema': _EXPECTED_GEN_SCHEMA, 'gen_fingerprint': FP,
                       'AcelPrimVag': _cellrow(3), 'AcelRodaPrimVag': _cellrow(4),
                       'PitchPrimVag': _cellrow(3), 'contact_log': _good_clog()},
                missing=('DimSpace',))
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
    _write_manifest(p, drop_dano=True)
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
                       'contact_log': _good_clog(), **_raw_meta()},
                missing=('PitchPrimVag',))
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
    _write_bare(p, 2, d, missing=('bridge_samp',))
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

    p = os.path.join(root, "arraydigeststruct"); _build(p)
    fd_path = os.path.join(p, "file_digests.mat")
    fd_raw = sio.loadmat(fd_path)["file_digests"]
    sio.savemat(
        fd_path,
        {"file_digests": np.concatenate((fd_raw, fd_raw), axis=1)},
        long_field_names=True,
    )
    check("multi-element file_digests struct rejected", lambda: _load(p),
          should_raise=True)

    # 37. Marker root != file_digests root -> reject
    p = os.path.join(root, "badroot"); _build(p)
    with open(os.path.join(p, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{FP}\ndeadbeef\n")   # wrong root line 3
    check("marker root mismatch rejected", lambda: _load(p), should_raise=True)

    # 38. Exact marker grammar: a fourth nonempty line is a restamp/tamper,
    # even when the three canonical lines remain correct.
    p = os.path.join(root, "extramarker"); _build(p)
    with open(os.path.join(p, '_GENERATION_COMPLETE'), 'a') as fh:
        fh.write("RESTAMPED\n")
    check("extra completion-marker line rejected", lambda: _load(p),
          should_raise=True)

    # 39. VALID dataset with source digests STILL loads (positive control)
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

    # ── two-tier contact gate (recalibrated 2026-07-19) ──────────────────────
    # 42. SUSTAINED tension (small F but frac over 0.2% of the path) -> reject
    p = os.path.join(root, "sustained"); _build(p)
    sus = np.column_stack([np.zeros(NP), np.array([1.0] + [0.0] * (NP - 1)),
                           np.array([0.01] + [0.0] * (NP - 1)),      # frac 1% >> tol
                           np.array([5.0e3] + [-1e5] * (NP - 1))])   # F below F-tol
    _write_state(p, 2, clog=sus)
    check("sustained tension (frac over tol) rejected", lambda: _load(p),
          should_raise=True)

    # 43. The REAL s23_all4 state-24 event (6.4 kN on 0.042% of the path) is in
    # the TOLERATED micro-unloading tier -> must LOAD (and be counted/summarised)
    p = os.path.join(root, "microtension"); _build(p)
    tol = np.column_stack([np.zeros(NP), np.array([1.0] + [0.0] * (NP - 1)),
                           np.array([0.00042] + [0.0] * (NP - 1)),
                           np.array([6419.0] + [-1e5] * (NP - 1))])
    _write_state(p, 2, clog=tol)
    check("tolerated micro-tension tier ACCEPTED", lambda: _load(p),
          should_raise=False)

    # 44. The REAL s15_track state-244 event (13.4 kN for ONE sample = 0.063%
    # of the path, off-bridge) — the event that drove the 2026-07-22 F-tier
    # recalibration (12 -> 24 kN) -> now in the TOLERATED tier, must LOAD.
    p = os.path.join(root, "microtension2"); _build(p)
    tol2 = np.column_stack([np.zeros(NP), np.array([1.0] + [0.0] * (NP - 1)),
                            np.array([0.000633] + [0.0] * (NP - 1)),
                            np.array([13419.2] + [-1e5] * (NP - 1))])
    _write_state(p, 2, clog=tol2)
    check("s15 state-244 event (13.4 kN, 1 sample) ACCEPTED post-recal",
          lambda: _load(p), should_raise=False)

    # 45. Brief but ABOVE the recalibrated 24 kN cap -> still hard-fails
    # (the F-tier exists; only its level moved).
    p = os.path.join(root, "overcap"); _build(p)
    ovr = np.column_stack([np.zeros(NP), np.array([1.0] + [0.0] * (NP - 1)),
                           np.array([0.0005] + [0.0] * (NP - 1)),
                           np.array([25000.0] + [-1e5] * (NP - 1))])
    _write_state(p, 2, clog=ovr)
    check("brief tension above 24 kN cap rejected", lambda: _load(p),
          should_raise=True)

    # -- R11 portable numerical-environment / qualification firewall --------
    # Exact MATLAB versions are provenance, not an allow-list.  Internal
    # descriptor authentication and dataset-wide consistency remain fail-closed
    # before signals are admitted, as does qualification anti-laundering.
    p = os.path.join(root, "qualification"); _build(p)
    _write_manifest(p, qualification=True)
    check("qualification dataset rejected", lambda: _load(p), should_raise=True)

    p = os.path.join(root, "noqualification"); _build(p)
    _write_manifest(p, drop_qualification=True)
    check("missing qualification marker rejected", lambda: _load(p),
          should_raise=True)

    portable_environment = _portable_actual_matlab_environment()
    p = os.path.join(root, "portablematlab"); _build(p)
    _rewrite_coherent_actual_environment(p, portable_environment)

    def _portable_matlab_loads_with_reference_intact():
        metadata = _read_manifest_generation_metadata(p)
        if (
            metadata["matlab_release"]
            != portable_environment["matlab_release"]
            or metadata["actual_matlab_environment_descriptor"]
            != portable_environment["actual_matlab_environment_descriptor"]
            or metadata["actual_matlab_environment_sha256"]
            != portable_environment["actual_matlab_environment_sha256"]
            or metadata["campaign_matlab_release"]
            != _EXPECTED_MATLAB_RELEASE
            or metadata["campaign_matlab_environment_descriptor"]
            != _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR
            or metadata["campaign_matlab_environment_sha256"]
            != _EXPECTED_MATLAB_ENVIRONMENT_SHA256
        ):
            raise AssertionError("portable/reference provenance was not retained")
        _load(p)

    check(
        "coherent different MATLAB release loads with exact campaign reference",
        _portable_matlab_loads_with_reference_intact,
        should_raise=False,
    )

    p = os.path.join(root, "releasevsdescriptor"); _build(p)
    _write_manifest(p, matlab_release="R2024b")
    check(
        "actual MATLAB release/descriptor disagreement rejected",
        lambda: _load(p),
        should_raise=True,
    )

    p = os.path.join(root, "mixedactualenvironment"); _build(p)
    _rewrite_coherent_actual_environment(p, portable_environment)
    _write_state(p, 2)
    check(
        "mixing actual MATLAB environments within one dataset rejected",
        lambda: _load(p),
        should_raise=True,
    )

    p = os.path.join(root, "nocampaignrelease"); _build(p)
    _write_manifest(p, drop_campaign_release=True)
    check("missing campaign release rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "badlogical"); _build(p)
    _write_manifest(p, qualification=np.array([0, 1]))
    check("non-scalar qualification marker rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "staterelease"); _build(p)
    _write_state(p, 2, matlab_release="R2023b")
    check("per-state release mismatch rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "statequal"); _build(p)
    _write_state(p, 2, qualification=True)
    check("per-state qualification mismatch rejected", lambda: _load(p),
          should_raise=True)

    p = os.path.join(root, "statequalmissing"); _build(p)
    _write_state(p, 2, drop_qualification=True)
    check("missing per-state qualification marker rejected", lambda: _load(p),
          should_raise=True)

    # ── R11 full numerical-environment/source attestation ────────────────
    # Every case starts from a fully valid fixture, so each rejection reaches
    # the named guard instead of being masked by an unrelated missing field.
    manifest_mutations = (
        ("generation behaviour mismatch rejected",
         {'generation_behavior_version': 'generation-rules-v3'}, ()),
        ("missing actual environment descriptor rejected", {},
         ('actual_matlab_environment_descriptor',)),
        ("actual environment SHA mismatch rejected",
         {'actual_matlab_environment_sha256': '0' * 64}, ()),
        ("missing campaign environment descriptor rejected", {},
         ('campaign_matlab_environment_descriptor',)),
        ("campaign environment SHA mismatch rejected",
         {'campaign_matlab_environment_sha256': '1' * 64}, ()),
        ("missing generator source root rejected", {},
         ('generator_source_root_sha256',)),
        ("generator source digest-lines mismatch rejected",
         {'generator_source_digest_lines':
              _GENERATOR.digest_lines + '\nTAMPERED:0'}, ()),
        ("generator source file-count mismatch rejected",
         {'generator_source_file_count': _GENERATOR.file_count + 1}, ()),
        ("production qualification-source mismatch rejected",
         {'qualification_source_sha256': '2' * 64}, ()),
    )
    for case_no, (label, overrides, drops) in enumerate(
            manifest_mutations, start=1):
        p = os.path.join(root, f"r11manifest{case_no:02d}"); _build(p)
        _write_manifest(
            p,
            metadata_overrides=overrides,
            metadata_drop=drops,
        )
        check(label, lambda p=p: _load(p), should_raise=True)

    p = os.path.join(root, "coherentforeigncampaignenvironment"); _build(p)
    _write_manifest(
        p,
        metadata_overrides={
            "campaign_matlab_environment_descriptor":
                portable_environment["actual_matlab_environment_descriptor"],
            "campaign_matlab_environment_sha256":
                portable_environment["actual_matlab_environment_sha256"],
        },
    )
    check(
        "coherent foreign campaign/reference environment rejected",
        lambda: _load(p),
        should_raise=True,
    )

    # Self-consistency is not authenticity: rewrite manifest + every state stamp
    # to one canonical foreign source identity, then recompute dataset digests
    # and the completion marker. It must still fail against the live source tree.
    p = os.path.join(root, "r11coherentforeign"); _build(p)
    foreign = _foreign_generator_attestation()
    _write_manifest(p, metadata_overrides=foreign)
    for state_idx in range(1, NST + 1):
        _write_state(p, state_idx, nested_overrides=foreign)
    check("coherent foreign generator source rejected",
          lambda: _load(p), should_raise=True)

    top_level_fields = (
        'file_gen_schema',
        'file_gen_fingerprint',
        'file_state_uid',
        'file_state_seed_id',
        'file_random_stream_schedule_version',
        'file_matlab_release',
        'file_campaign_matlab_release',
        'file_release_qualification_run',
        'file_actual_matlab_environment_sha256',
        'file_campaign_matlab_environment_sha256',
        'file_generator_source_root_sha256',
        'file_qualification_source_sha256',
    )
    for case_no, field in enumerate(top_level_fields, start=1):
        p = os.path.join(root, f"r11top{case_no:02d}"); _build(p)
        bad_value = True if field == 'file_release_qualification_run' else (
            '3' * 64 if field.endswith('sha256') else 'TAMPERED'
        )
        _write_state(p, 2, top_overrides={field: bad_value})
        check(f"top-level state stamp {field} mismatch rejected",
              lambda p=p: _load(p), should_raise=True)

    p = os.path.join(root, "r11topmissing"); _build(p)
    _write_state(
        p, 2,
        top_drop=('file_generator_source_root_sha256',),
    )
    check("missing top-level state stamp rejected", lambda: _load(p),
          should_raise=True)

    # R11 semantic identity is duplicated only for the cheap UID/root/schedule
    # stamps. Keep those top-level values valid while independently mutating
    # each nested identity field, proving the table-alignment guards themselves
    # are reached rather than merely a generic stamp mismatch.
    baseline_identity = _state_identity(2)
    bad_state_named = baseline_identity[
        "state_named_stream_seed_id"
    ].copy()
    bad_state_named[0, 0] = np.uint32(int(bad_state_named[0, 0]) + 1)
    bad_passage_named = baseline_identity[
        "passage_named_stream_seed_id"
    ].copy()
    bad_passage_named[0, 0] = np.uint32(
        int(bad_passage_named[0, 0]) + 1
    )
    semantic_mutations = (
        (
            "nested StateUID/table mismatch rejected",
            {"state_uid": "fixture-state-999"},
            {"file_state_uid": baseline_identity["state_uid"]},
        ),
        (
            "nested StateSeedID/table mismatch rejected",
            {"state_seed_id": np.uint32(
                int(baseline_identity["state_seed_id"]) + 1
            )},
            {"file_state_seed_id": baseline_identity["state_seed_id"]},
        ),
        (
            "nested RNG schedule/table mismatch rejected",
            {"random_stream_schedule_version": "uid-named-substreams-v999"},
            {
                "file_random_stream_schedule_version":
                    baseline_identity["random_stream_schedule_version"]
            },
        ),
        (
            "nested state named-substream mutation rejected",
            {"state_named_stream_seed_id": bad_state_named},
            {},
        ),
        (
            "nested passage named-substream mutation rejected",
            {"passage_named_stream_seed_id": bad_passage_named},
            {},
        ),
        (
            "nested latent bearing/table mismatch rejected",
            {"latent_bearing_fixity": np.array([[0.1, 0.0]])},
            {},
        ),
        (
            "nested latent crack/table mismatch rejected",
            {"latent_crack_on": np.array([[True]], dtype=np.bool_)},
            {},
        ),
        (
            "nested active crack/table mismatch rejected",
            {"crack_on": np.array([[True]], dtype=np.bool_)},
            {},
        ),
        (
            "nested active bearing/table mismatch rejected",
            {"bearing_fixity": np.array([[0.1, 0.0]])},
            {},
        ),
        (
            "nested scour-support/table mismatch rejected",
            {"scour_supports": np.array([[2, 4]], dtype=np.uint32)},
            {},
        ),
    )
    for case_no, (label, overrides, top_overrides) in enumerate(
        semantic_mutations, start=1
    ):
        p = os.path.join(root, f"r11semantic{case_no:02d}")
        _build(p)
        _write_state(
            p,
            2,
            nested_overrides=overrides,
            top_overrides=top_overrides,
        )
        check(label, lambda p=p: _load(p), should_raise=True)

    p = os.path.join(root, "r11semanticmissing")
    _build(p)
    _write_state(
        p,
        2,
        nested_drop=("state_named_stream_seed_id",),
    )
    check(
        "missing nested semantic-state field rejected",
        lambda: _load(p),
        should_raise=True,
    )

    nested_mutations = (
        ('actual_matlab_environment_descriptor', 'TAMPERED'),
        ('actual_matlab_environment_sha256', '4' * 64),
        ('campaign_matlab_environment_descriptor', 'TAMPERED'),
        ('campaign_matlab_environment_sha256', '5' * 64),
        ('generator_source_root_sha256', '6' * 64),
        ('generator_source_digest_lines',
         _GENERATOR.digest_lines + '\nTAMPERED:0'),
        ('generator_source_file_count', _GENERATOR.file_count + 1),
        ('qualification_source_sha256', '7' * 64),
    )
    for case_no, (field, value) in enumerate(nested_mutations, start=1):
        p = os.path.join(root, f"r11nested{case_no:02d}"); _build(p)
        _write_state(p, 2, nested_overrides={field: value})
        check(f"nested state attestation {field} mismatch rejected",
              lambda p=p: _load(p), should_raise=True)

    p = os.path.join(root, "r11nestedmissing"); _build(p)
    _write_state(
        p, 2,
        nested_drop=('campaign_matlab_environment_descriptor',),
    )
    check("missing nested state attestation rejected", lambda: _load(p),
          should_raise=True)

    # A coherent qualification dataset (manifest + nested payload + top-level
    # stamps all agree) must still never be accepted as production campaign
    # input. This is the anti-laundering policy, not merely a mismatch check.
    p = os.path.join(root, "r11qualificationlaunder"); _build(p)
    _write_manifest(
        p,
        qualification=True,
        qualification_source=_QUALIFICATION_SHA,
    )
    for state_idx in range(1, NST + 1):
        _write_state(
            p, state_idx,
            qualification=True,
            qualification_source=_QUALIFICATION_SHA,
        )
    check("coherent qualification dataset cannot be laundered into production",
          lambda: _load(p), should_raise=True)

    print()
    print("LOADER PROVENANCE: ALL PASS" if fails == 0 else
          f"LOADER PROVENANCE: {fails} CHECK(S) FAILED")
finally:
    shutil.rmtree(root, ignore_errors=True)

sys.exit(1 if fails else 0)
