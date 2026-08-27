"""Adversarial R11 cache-provenance test.

Run:  python check_cache_provenance.py   (needs numpy + scipy + torch + sklearn)

Builds a tiny valid R11 dataset (with source SHA digests + exact 3-line marker) under a
temp data/ folder, exercises get_or_create_cache, then verifies that every
tamper is caught: swapped/corrupted feature/label/GROUPS/scaler artifacts, a
count-preserving group SWAP, a same-size source .mat overwrite, a wrong-content
marker, incoherent/mixed environment provenance, qualification laundering,
an interrupted publication (missing sidecar), and concurrent builds of the
same stem.  A coherent non-reference MATLAB release is accepted and recorded.
MUST print ALL PASS before trusting a multi-day campaign cache.
"""
import concurrent.futures as cf
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
from core.dataset import (                                           # noqa: E402
    _EXPECTED_CHANNEL_SCHEMA_ID,
    _EXPECTED_GENERATION_BEHAVIOR_VERSION,
    _EXPECTED_GEN_SCHEMA,
    _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
    _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    _EXPECTED_MATLAB_RELEASE,
    get_or_create_cache,
)
from core.campaign_contract import (                                 # noqa: E402
    EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
    EXPECTED_RAIL_END_CLEARANCE_M,
)
from core.protocol import read_dataset_provenance  # noqa: E402
from core.source_provenance import generator_source_root              # noqa: E402

# Keep the integration fixture at the campaign's 512 PAA segments and use one
# physically coherent 60 m / 70 km/h RAW passage.  The 3600 time samples map to
# 7000 spatial samples; the registered [1001, 7000] crop therefore contains the
# complete 6000-sample bridge and remains large enough for 512-segment PAA.
NST, NP, L = 4, 4, 3600
BRIDGE_LENGTH_M = 60.0
DIM_SPACE = 7000
SPEED_MPS = 70.0 / 3.6
CFG = {'method': 'PAA', 'dofs': [0], 'task': 'regression', 'target_supports': [2, 3]}
fails = 0
_GENERATOR = generator_source_root(Path(__file__).resolve().parent)
_QUALIFICATION_SHA = "a" * 64
DAMAGE_SEED = 1
RNG_SCHEDULE = "uid-named-substreams-v2"
STATE_STREAM_NAMES = (
    "operations", "crack", "profile-state", "track", "profile-phase"
)
PASSAGE_STREAM_NAMES = ("profile-passage", "oor-passage")


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


def _generation_config_json(
    *,
    generator_source_root_sha256=_GENERATOR.sha256,
    qualification_source_sha256="PRODUCTION",
):
    """Canonical, hashed subset of the generator configuration for this fixture."""
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
            "generator_source_root_sha256": generator_source_root_sha256,
            "qualification_source_sha256": qualification_source_sha256,
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


def _state_identity(idx):
    uid = f"fixture-state-{idx:03d}"
    root = _state_seed_id(uid)
    return {
        "state_uid": uid,
        "state_seed_id": np.uint32(root),
        "random_stream_schedule_version": RNG_SCHEDULE,
        "state_named_stream_seed_id": np.asarray([[
            _named_seed(root, uid, name) for name in STATE_STREAM_NAMES
        ]], dtype=np.uint32),
        "passage_named_stream_seed_id": np.asarray([
            [
                _named_seed(root, uid, name, passage=passage)
                for name in PASSAGE_STREAM_NAMES
            ]
            for passage in range(1, NP + 1)
        ], dtype=np.uint32),
        "latent_bearing_fixity": np.zeros((1, 2)),
        "latent_crack_on": np.array([[False]], dtype=np.bool_),
        "crack_on": np.array([[False]], dtype=np.bool_),
        "bearing_fixity": np.zeros((1, 2)),
        "scour_supports": np.array([[2, 3]], dtype=np.uint32),
    }


def _generation_metadata(*, qualification=False,
                         qualification_source="PRODUCTION",
                         generator_attestation=None,
                         actual_environment=None):
    actual_environment = actual_environment or {
        'matlab_release': _EXPECTED_MATLAB_RELEASE,
        'actual_matlab_environment_descriptor':
            _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
        'actual_matlab_environment_sha256':
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    }
    metadata = {
        'channel_schema_id': _EXPECTED_CHANNEL_SCHEMA_ID,
        'generation_behavior_version':
            _EXPECTED_GENERATION_BEHAVIOR_VERSION,
        'matlab_release': actual_environment['matlab_release'],
        'campaign_matlab_release': _EXPECTED_MATLAB_RELEASE,
        'release_qualification_run': qualification,
        'actual_matlab_environment_descriptor':
            actual_environment['actual_matlab_environment_descriptor'],
        'actual_matlab_environment_sha256':
            actual_environment['actual_matlab_environment_sha256'],
        'campaign_matlab_environment_descriptor':
            _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
        'campaign_matlab_environment_sha256':
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        'generator_source_root_sha256': _GENERATOR.sha256,
        'generator_source_digest_lines': _GENERATOR.digest_lines,
        'generator_source_file_count': _GENERATOR.file_count,
        'qualification_source_sha256': qualification_source,
    }
    metadata.update(generator_attestation or {})
    return metadata


def _top_level_stamps(data):
    return {
        'file_gen_schema': data['gen_schema'],
        'file_gen_fingerprint': data['gen_fingerprint'],
        'file_state_uid': data['state_uid'],
        'file_state_seed_id': data['state_seed_id'],
        'file_random_stream_schedule_version':
            data['random_stream_schedule_version'],
        'file_matlab_release': data['matlab_release'],
        'file_campaign_matlab_release': data['campaign_matlab_release'],
        'file_release_qualification_run':
            data['release_qualification_run'],
        'file_actual_matlab_environment_sha256':
            data['actual_matlab_environment_sha256'],
        'file_campaign_matlab_environment_sha256':
            data['campaign_matlab_environment_sha256'],
        'file_generator_source_root_sha256':
            data['generator_source_root_sha256'],
        'file_qualification_source_sha256':
            data['qualification_source_sha256'],
    }


def _cellrow(nrows):
    a = np.empty((1, NP), dtype=object)
    for p in range(NP):
        a[0, p] = np.random.RandomState(p).randn(nrows, L).astype(float)
    return a


def _raw_meta():
    return {
        'DimSpace': np.full((1, NP), float(DIM_SPACE)),
        'DimAcel': np.full((1, NP), float(L)),
        'crop_start': np.full((1, NP), 1001.0),
        'crop_end': np.full((1, NP), float(DIM_SPACE)),
        'bridge_samp': np.full((1, NP), 100.0 * BRIDGE_LENGTH_M),
        'L_bridge_eff': np.full((1, NP), BRIDGE_LENGTH_M),
        'Velocidade': np.full((1, NP), SPEED_MPS),
    }


def _empty_log_column():
    """Return one MATLAB-cell-like Npass x 1 column of empty log entries."""
    values = np.empty((NP, 1), dtype=object)
    for passage in range(NP):
        values[passage, 0] = np.empty((0, 0), dtype=np.float64)
    return values


def _complete_payload_defaults(contact_log):
    """Production-shaped physical fields in the closed state contract."""
    return {
        'AcelPrimVag': _cellrow(3),
        'AcelRodaPrimVag': _cellrow(4),
        'AcelWheelsetPrimVag': _cellrow(4),
        'PitchPrimVag': _cellrow(3),
        'Dano': np.float64(0.2),
        'Temperatura': np.full((1, NP), 20.0, dtype=np.float64),
        'VehiclesProps': np.zeros((5, 3, NP), dtype=np.float64),
        'beam_f1_Hz': np.float64(4.0),
        'bearing_vector': np.zeros((1, 2), dtype=np.float64),
        'crack_log': np.zeros((NP, 3), dtype=np.float64),
        'profile_mode': 'fixed',
        'profile_log': np.ones((NP, 1), dtype=np.float64),
        'track_log': _empty_log_column(),
        'oor_log': _empty_log_column(),
        'contact_log': np.asarray(contact_log, dtype=np.float64),
        **_raw_meta(),
    }


def _finalize(data_dir, *, fingerprint=None):
    files = sorted(
        f for f in os.listdir(data_dir)
        if re.fullmatch(r'\d{4}\.mat', f)
        or f in {'case_info.mat', 'damage_states.mat'}
    )
    per = {
        f: hashlib.sha256(Path(data_dir, f).read_bytes()).hexdigest()
        for f in files
    }
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    sio.savemat(
        os.path.join(data_dir, 'file_digests.mat'),
        {'file_digests': {
            'schema': 'source-digests-v2',
            'scope': 'NNNN.mat+case_info.mat+damage_states.mat',
            'digest_lines': lines,
            'root': root,
        }},
    )
    if fingerprint is None:
        # Read the first raw struct element so this helper can also finalize the
        # deliberate multi-element case_info corruption used below.  The loader,
        # unlike this digest/marker helper, must and does reject that grammar.
        info = sio.loadmat(
            os.path.join(data_dir, "case_info.mat")
        )["case_info"]
        fingerprint = str(
            np.ravel(info[0, 0]["gen_fingerprint"])[0]
        )
    with open(os.path.join(data_dir, '_GENERATION_COMPLETE'), 'w') as fh:
        fh.write(f"{_EXPECTED_GEN_SCHEMA}\n{fingerprint}\n{root}\n")


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


def _build_dataset(data_dir, *, qualification=False,
                   qualification_source="PRODUCTION",
                   generator_attestation=None,
                   metadata_overrides=None, metadata_drop=(),
                   actual_environment=None,
                   state_generation_overrides=None):
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)
    generation = _generation_metadata(
        qualification=qualification,
        qualification_source=qualification_source,
        generator_attestation=generator_attestation,
        actual_environment=actual_environment,
    )
    generation_config_json = _generation_config_json(
        generator_source_root_sha256=
            generation["generator_source_root_sha256"],
        qualification_source_sha256=qualification_source,
    )
    fingerprint = hashlib.sha256(
        generation_config_json.encode("utf-8")
    ).hexdigest()
    case_info = {
        'n_states': NST,
        'passages_per_state': NP,
        'gen_schema': _EXPECTED_GEN_SCHEMA,
        'gen_fingerprint': fingerprint,
        'generation_config_json': generation_config_json,
        'state_design_kind': 'five-family-multidamage-v2',
        'case_name': '_cacheprov_ds',
        'stage': 'fixture_stage',
        'damage_mode': 'multi_scour',
        'rail_end_clearance_m': EXPECTED_RAIL_END_CLEARANCE_M,
        'rail_end_clearance_decision_id':
            EXPECTED_RAIL_END_CLEARANCE_DECISION_ID,
        'L_bridge_m': BRIDGE_LENGTH_M,
        'num_spans': 3,
        'num_supports': 4,
        'scour_supports': '[2 3]',
        'scour_dano_max_frac': 0.60,
        'n_target_healthy': 0,
        'n_scour_only': 0,
        'n_bearing_only': 0,
        'n_nuisance_only': 0,
        'n_joint': NST,
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
        **generation}
    case_info.update(metadata_overrides or {})
    for field in metadata_drop:
        case_info.pop(field, None)
    sio.savemat(
        os.path.join(data_dir, 'case_info.mat'),
        {'case_info': case_info},
        long_field_names=True,
    )
    identities = [_state_identity(index) for index in range(1, NST + 1)]
    state_named = np.vstack([
        identity["state_named_stream_seed_id"] for identity in identities
    ])
    passage_named = np.stack([
        identity["passage_named_stream_seed_id"] for identity in identities
    ])
    sio.savemat(os.path.join(data_dir, 'damage_states.mat'), {
        'StateFamily': np.array(['joint'] * NST, dtype=object).reshape(-1, 1),
        'AnchorTarget': np.zeros((NST, 1)),
        'AnchorLevel': np.zeros((NST, 1)),
        'StateUID': np.asarray(
            [identity["state_uid"] for identity in identities],
            dtype=object,
        ).reshape(-1, 1),
        'StateSeedID': np.asarray(
            [identity["state_seed_id"] for identity in identities],
            dtype=np.uint32,
        ).reshape(-1, 1),
        'LatentBearingFixity': np.zeros((NST, 2)),
        'LatentCrackOn': np.zeros((NST, 1), dtype=np.uint8),
        'CrackOn': np.zeros((NST, 1), dtype=np.uint8),
        'StateNamedStreamSeedID': state_named,
        'PassageNamedStreamSeedID': passage_named,
        'PassageNamedStreamSeedIDFlat': passage_named.reshape(
            NST, -1, order="F"
        ),
        'random_stream_schedule_version': RNG_SCHEDULE,
        'state_stream_names': np.asarray(
            STATE_STREAM_NAMES, dtype=object
        ).reshape(1, -1),
        'passage_stream_names': np.asarray(
            PASSAGE_STREAM_NAMES, dtype=object
        ).reshape(1, -1),
        'DamageStates': np.tile([0.0, 0.1, 0.2, 0.0], (NST, 1)),
        'BearingStates': np.zeros((NST, 2)),
        'BearingFixity': np.zeros((NST, 2)),
        'k_ref_bear': 1.0,
        'scour_supports': np.array([[2, 3]], dtype=np.uint32),
    })
    clog = np.column_stack([np.zeros(NP)] * 3 + [-1e5 * np.ones(NP)])
    for i in range(1, NST + 1):
        state_generation = dict(generation)
        state_generation.update(
            (state_generation_overrides or {}).get(i, {})
        )
        d = {'scour_vector': np.array([[0.0, 0.1, 0.2, 0.0]]), 'gen_schema': _EXPECTED_GEN_SCHEMA,
             'gen_fingerprint': fingerprint, 'state_family': 'joint',
             **_state_identity(i),
             **state_generation,
             **_complete_payload_defaults(clog)}
        # generation_behavior_version belongs to case_info/fingerprint, not
        # the per-state payload written by A00.
        d.pop('generation_behavior_version')
        sio.savemat(
            os.path.join(data_dir, f"{i:04d}.mat"),
            {'data': d, **_top_level_stamps(d)},
            long_field_names=True,
        )
    _finalize(data_dir, fingerprint=fingerprint)


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
              bool(prov['source'].get('dataset_content_root_sha256'))
              and bool(prov['source'].get('manifest_sha256'))
              and prov['source'].get('dano_max') == 0.60
              and prov['source'].get('matlab_release')
                  == _EXPECTED_MATLAB_RELEASE
              and prov['source'].get('release_qualification_run') is False
              and prov['source'].get('actual_matlab_environment_sha256')
                  == _EXPECTED_MATLAB_ENVIRONMENT_SHA256
              and prov['source'].get('campaign_matlab_environment_sha256')
                  == _EXPECTED_MATLAB_ENVIRONMENT_SHA256
              and prov['source'].get('generator_source_root_sha256')
                  == _GENERATOR.sha256
              and prov['source'].get('generator_source_file_count')
                  == _GENERATOR.file_count
              and prov['source'].get('qualification_source_sha256')
                  == "PRODUCTION"
              and prov['source'].get('channel_schema_id')
                  == _EXPECTED_CHANNEL_SCHEMA_ID
              and prov['source'].get('state_design_kind')
                  == 'five-family-multidamage-v2')
        check("no leftover .tmp / .lock", not any(f.endswith(('.tmp', '.lock'))
                                                  for f in os.listdir(cache_dir)))
        del X, y, g
        gc.collect()

        # 2. Fast-path reuse (no rebuild)
        X2, y2, sc2, g2 = get_or_create_cache(CFG, ds, cache_dir)
        check("fast-path reuse identical", tuple(X2.shape) == shape)
        del X2, y2, g2
        gc.collect()

        # 2b. A different actual MATLAB release is admissible when its
        # descriptor/SHA and every state stamp are coherent.  The existing
        # reference-host cache must be replaced, while the immutable campaign
        # reference remains exact in the new sidecar.
        portable_environment = _portable_actual_matlab_environment()
        _build_dataset(
            data_dir,
            actual_environment=portable_environment,
        )
        Xp, yp, scp, gp = get_or_create_cache(CFG, ds, cache_dir)
        portable_prov = json.load(open(_art(cache_dir, '_prov.json')))
        check(
            "coherent different MATLAB release rebuilds portable cache",
            tuple(Xp.shape) == shape
            and portable_prov['source']['matlab_release']
                == portable_environment['matlab_release']
            and portable_prov['source'][
                'actual_matlab_environment_descriptor'
            ] == portable_environment[
                'actual_matlab_environment_descriptor'
            ]
            and portable_prov['source'][
                'actual_matlab_environment_sha256'
            ] == portable_environment['actual_matlab_environment_sha256']
            and portable_prov['source']['actual_matlab_environment_sha256']
                != _EXPECTED_MATLAB_ENVIRONMENT_SHA256
            and portable_prov['source']['campaign_matlab_release']
                == _EXPECTED_MATLAB_RELEASE
            and portable_prov['source'][
                'campaign_matlab_environment_descriptor'
            ] == _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR
            and portable_prov['source'][
                'campaign_matlab_environment_sha256'
            ] == _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        )
        del Xp, yp, gp
        gc.collect()

        default_actual = {
            'matlab_release': _EXPECTED_MATLAB_RELEASE,
            'actual_matlab_environment_descriptor':
                _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
            'actual_matlab_environment_sha256':
                _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        }
        portable_rejections = (
            (
                "actual descriptor/SHA incoherence rejected on cache path",
                {
                    "actual_environment": portable_environment,
                    "metadata_overrides": {
                        "actual_matlab_environment_sha256": "0" * 64,
                    },
                },
            ),
            (
                "actual release/descriptor disagreement rejected on cache path",
                {
                    "actual_environment": portable_environment,
                    "metadata_overrides": {
                        "matlab_release": _EXPECTED_MATLAB_RELEASE,
                    },
                },
            ),
            (
                "coherent foreign campaign/reference rejected on cache path",
                {
                    "metadata_overrides": {
                        "campaign_matlab_environment_descriptor":
                            portable_environment[
                                "actual_matlab_environment_descriptor"
                            ],
                        "campaign_matlab_environment_sha256":
                            portable_environment[
                                "actual_matlab_environment_sha256"
                            ],
                    },
                },
            ),
            (
                "mixed actual MATLAB environments rejected on cache path",
                {
                    "actual_environment": portable_environment,
                    "state_generation_overrides": {2: default_actual},
                },
            ),
        )
        for label, build_kwargs in portable_rejections:
            _build_dataset(data_dir, **build_kwargs)
            try:
                get_or_create_cache(CFG, ds, cache_dir)
                check(label, False)
            except RuntimeError:
                check(label, True)

        _build_dataset(data_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
        X, y, sc, g = get_or_create_cache(CFG, ds, cache_dir)
        check("reference fixture restored after portability checks",
              tuple(X.shape) == shape)
        del X, y, g
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
                # This tiny cache-only fixture is deliberately not one of the
                # registered campaign rungs, but it must still carry the full
                # descriptor grammar before exercising the source-identity
                # TOCTOU guard.
                "rung": {
                    "stage": None,
                    "dataset": ds,
                    "target_supports": [2, 3],
                    "bearing_targets": None,
                    "dataset_provenance": expected_source,
                }
            },
        }
        get_or_create_cache(bound_cfg, ds, cache_dir)
        table_path = os.path.join(data_dir, "damage_states.mat")
        table = sio.loadmat(table_path)
        table["AnchorLevel"] = np.ones((NST, 1))
        sio.savemat(table_path, table)
        _finalize(data_dir)
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

        # 9. Exact marker grammar also applies on the cache fast path: an extra
        # nonempty line cannot be ignored as harmless annotation/restamping.
        get_or_create_cache(CFG, ds, cache_dir)
        with open(os.path.join(data_dir, '_GENERATION_COMPLETE'), 'a') as fh:
            fh.write("RESTAMPED\n")
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("extra marker line rejected on cache fast path", False)
        except Exception:
            check("extra marker line rejected on cache fast path", True)
        _build_dataset(data_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

        # 10. CONCURRENT builds of the SAME stem (per-stem lock serialises them)
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

        # 11. A previously valid cache must not launder even a COHERENT
        # qualification dataset whose manifest, nested payloads and top-level
        # stamps all agree.
        _build_dataset(
            data_dir,
            qualification=True,
            qualification_source=_QUALIFICATION_SHA,
        )
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("coherent qualification source rejected on cache fast path",
                  False)
        except RuntimeError:
            check("coherent qualification source rejected on cache fast path",
                  True)

        # 12. The fast path revalidates the full R11 manifest rather than
        # trusting the already-published cache sidecar.
        _build_dataset(data_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)
        get_or_create_cache(CFG, ds, cache_dir)
        manifest_mutations = (
            ("actual environment drift rejected on cache fast path",
             "actual_matlab_environment_sha256", "0" * 64),
            ("generator source drift rejected on cache fast path",
             "generator_source_root_sha256", "1" * 64),
            ("generation behaviour drift rejected on cache fast path",
             "generation_behavior_version", "generation-rules-v3"),
            ("channel schema drift rejected on cache fast path",
             "channel_schema_id", "legacy_virtual8"),
        )
        for label, field, bad_value in manifest_mutations:
            ci_path = os.path.join(data_dir, "case_info.mat")
            ci = sio.loadmat(ci_path, simplify_cells=True)["case_info"]
            ci[field] = bad_value
            sio.savemat(
                ci_path,
                {"case_info": ci},
                long_field_names=True,
            )
            try:
                get_or_create_cache(CFG, ds, cache_dir)
                check(label, False)
            except RuntimeError:
                check(label, True)
            _build_dataset(data_dir)

        clearance_mutations = (
            (
                "missing clearance rejected on cache fast path",
                {"metadata_drop": ("rail_end_clearance_m",)},
            ),
            (
                "wrong clearance rejected on cache fast path",
                {"metadata_overrides": {"rail_end_clearance_m": 15.0}},
            ),
            (
                "missing clearance decision ID rejected on cache fast path",
                {"metadata_drop": ("rail_end_clearance_decision_id",)},
            ),
            (
                "wrong clearance decision ID rejected on cache fast path",
                {"metadata_overrides": {
                    "rail_end_clearance_decision_id":
                        "paper1-rail-domain-clearance-c15-v1"
                }},
            ),
        )
        for label, mutation_kwargs in clearance_mutations:
            _build_dataset(data_dir, **mutation_kwargs)
            try:
                get_or_create_cache(CFG, ds, cache_dir)
                check(label, False)
            except RuntimeError:
                check(label, True)

        # 13. Scalar-struct grammar also protects an already valid cache. A
        # duplicated case_info struct cannot be silently reduced with [0, 0].
        _build_dataset(data_dir)
        ci_path = os.path.join(data_dir, "case_info.mat")
        ci_raw = sio.loadmat(ci_path)["case_info"]
        sio.savemat(
            ci_path,
            {"case_info": np.concatenate((ci_raw, ci_raw), axis=1)},
            long_field_names=True,
        )
        _finalize(data_dir)
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("multi-element case_info rejected on cache fast path", False)
        except RuntimeError:
            check("multi-element case_info rejected on cache fast path", True)

        # 14. file_digests must itself be exactly one scalar struct before any
        # field is selected.
        _build_dataset(data_dir)
        fd_path = os.path.join(data_dir, "file_digests.mat")
        fd_raw = sio.loadmat(fd_path)["file_digests"]
        sio.savemat(
            fd_path,
            {"file_digests": np.concatenate((fd_raw, fd_raw), axis=1)},
            long_field_names=True,
        )
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("multi-element digest struct rejected on cache fast path",
                  False)
        except RuntimeError:
            check("multi-element digest struct rejected on cache fast path",
                  True)

        # 15. A foreign source identity may be perfectly self-consistent across
        # manifest, state payloads/stamps, dataset digests and marker. It remains
        # ineligible because it does not identify the live reviewed MATLAB bytes.
        _build_dataset(
            data_dir,
            generator_attestation=_foreign_generator_attestation(),
        )
        try:
            get_or_create_cache(CFG, ds, cache_dir)
            check("coherent foreign source rejected on cache fast path", False)
        except RuntimeError:
            check("coherent foreign source rejected on cache fast path", True)

        print("\nCACHE PROVENANCE: ALL PASS" if fails == 0 else
              f"\nCACHE PROVENANCE: {fails} CHECK(S) FAILED")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
