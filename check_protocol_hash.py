"""Adversarial R11 test of the unified protocol/source hash.

Run:  python check_protocol_hash.py   (needs numpy + scipy + torch/sklearn;
                                       does NOT need optuna or MATLAB)

What is proven here (MUST print ALL PASS before any campaign run):
  A. Canonical hashing is deterministic (key order / set order can't change it)
     and REJECTS non-canonical types instead of silently repr()-ing them.
  B. SEARCH_SPACE (the hashed data) and trainer._suggest_params (the executing
     code) agree EXACTLY - every suggest_* call, for every gate combination
     (n_conv x n_dense x lstm on/off x lstm layers x nhits on/off), matches the
     declared spec in name, kind, range and ORDER. This is the anti-drift proof:
     if someone edits a range in one place only, this fails.
  C. Every protocol knob changes the CORE hash (seeds, order of seeds, trials,
     epochs, noise, architecture flags, extra pairs, stage sets, schema tag,
     train protocol, search space, split constants, pruner config).
  D. Dataset/stage identity changes the FULL hash but NOT the core hash
     (the cross-rung champion-carry condition).
  E. read_dataset_provenance HARD-FAILS on incomplete/tampered/qualification
     datasets and on numerical-environment or reviewed-generator drift.
  F. descriptor_diff names the exact knob that differs (the error-message path).
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
from core.protocol import (canonical_json, protocol_hash, short_hash,      # noqa: E402
                           descriptor_diff, build_protocol_descriptors,
                           read_dataset_provenance, OPTUNA_PROTOCOL)
import core.protocol as cprotocol                                          # noqa: E402
from core.dataset import (                                                # noqa: E402
    _EXPECTED_GENERATION_BEHAVIOR_VERSION,
    _EXPECTED_GEN_SCHEMA,
    _EXPECTED_MATLAB_ENVIRONMENT_DESCRIPTOR,
    _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    _EXPECTED_MATLAB_RELEASE,
)
import core.dataset as cds                                                 # noqa: E402
from core.execution_environment import EXECUTION_BLOCK_POLICY             # noqa: E402
from core.hyperparameter_policy import HYPERPARAMETER_POLICY              # noqa: E402
from core.capacity_preflight import CAPACITY_PREFLIGHT_POLICY             # noqa: E402
from core.campaign_contract import EXPECTED_PROTOCOL_SCHEMA_TAG           # noqa: E402
from core.source_provenance import (                                      # noqa: E402
    ENVIRONMENT_LOCK,
    REQUIREMENTS_LOCK,
    generator_source_root,
    python_runtime_source_root,
)
from training.trainer import (TRAIN_PROTOCOL, SEARCH_SPACE,                # noqa: E402
                              _suggest_params)

fails = 0
REPO = Path(__file__).resolve().parent
_GENERATOR = generator_source_root(REPO)
_PYTHON_RUNTIME = python_runtime_source_root(REPO)
_FIXTURE_N_STATES = 3
_FIXTURE_NPASS = 4
_FIXTURE_DAMAGE_SEED = 1
_FIXTURE_STAGE = "fixture_stage"
_FIXTURE_DATASET = "fixture_dataset"
_RNG_SCHEDULE = "uid-named-substreams-v2"
_STATE_STREAM_NAMES = (
    "operations", "crack", "profile-state", "track", "profile-phase",
)
_PASSAGE_STREAM_NAMES = ("profile-passage", "oor-passage")
_STATE_UIDS = tuple(
    f"fixture-state-{index:03d}"
    for index in range(1, _FIXTURE_N_STATES + 1)
)


def _generation_config_json(
    variant,
    *,
    schema=_EXPECTED_GEN_SCHEMA,
    behavior=_EXPECTED_GENERATION_BEHAVIOR_VERSION,
    campaign_release=_EXPECTED_MATLAB_RELEASE,
    campaign_environment_sha256=_EXPECTED_MATLAB_ENVIRONMENT_SHA256,
    generator_source_root_sha256=_GENERATOR.sha256,
    qualification_source_sha256="PRODUCTION",
):
    """Canonical, hash-authenticated config for one standalone unit fixture."""
    return json.dumps(
        {
            "schema": schema,
            "generation_behavior_version": behavior,
            "campaign_matlab_release": campaign_release,
            "campaign_matlab_environment_sha256":
                campaign_environment_sha256,
            "generator_source_root_sha256":
                generator_source_root_sha256,
            "qualification_source_sha256": qualification_source_sha256,
            "STAGE": _FIXTURE_STAGE,
            "n_states": _FIXTURE_N_STATES,
            "Npass": _FIXTURE_NPASS,
            "damage_seed": _FIXTURE_DAMAGE_SEED,
            # The nonce models a genuine regeneration while leaving the
            # semantic state universe unchanged.
            "unit_fixture_variant": str(variant),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _fixture_fingerprint(variant):
    return hashlib.sha256(
        _generation_config_json(variant).encode("utf-8")
    ).hexdigest()


FP_A = _fixture_fingerprint("A")
FP_B = _fixture_fingerprint("B")
FP_OTHER = "c" * 64
FP_TAMPERED = "d" * 64


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails += 1


def check_raises(name, fn, exc=Exception):
    """PASS iff fn() raises exc (the guard fired)."""
    try:
        fn()
        check(name, False, "did NOT raise")
    except exc as e:
        check(name, True, f"rejected ({type(e).__name__})")
    except Exception as e:                                       # noqa: BLE001
        check(name, False, f"raised the WRONG type: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# A. Canonical hashing determinism
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- A. canonical hashing ---")
h1 = protocol_hash({"b": 1, "a": {"y": 2, "x": [3, 4]}})
h2 = protocol_hash({"a": {"x": [3, 4], "y": 2}, "b": 1})
check("key order does not change the hash", h1 == h2)
h3 = protocol_hash({"s": {"q", "p", "r"}})
h4 = protocol_hash({"s": {"r", "p", "q"}})
check("set iteration order does not change the hash", h3 == h4)
check("LIST order DOES change the hash (SEEDS order is significant)",
      protocol_hash({"l": [1, 2]}) != protocol_hash({"l": [2, 1]}))
check_raises("non-canonical type (object) rejected",
             lambda: canonical_json({"bad": object()}), TypeError)
check("short_hash is a 12-char prefix", short_hash(h1) == h1[:12])

# The protocol's executable-code identity is not a hand-maintained version
# string: it is the content root of every reviewed Python/environment input.
with tempfile.TemporaryDirectory(prefix="runtime-root-") as runtime_tmp:
    runtime_root = Path(runtime_tmp)
    manifest_source = REPO / "bundle_source_files.txt"
    shutil.copy2(manifest_source, runtime_root / manifest_source.name)
    manifest_names = [
        line for line in manifest_source.read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    runtime_names = [
        name for name in manifest_names
        if name.endswith(".py")
        or name in {ENVIRONMENT_LOCK, REQUIREMENTS_LOCK}
    ]
    for name in runtime_names:
        source = REPO.joinpath(*name.split("/"))
        target = runtime_root.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copied_before = python_runtime_source_root(runtime_root)
    sensitivity_target = runtime_root / "core" / "protocol.py"
    sensitivity_target.write_bytes(
        sensitivity_target.read_bytes() + b"\n# provenance sensitivity\n"
    )
    copied_after = python_runtime_source_root(runtime_root)
    check("one reviewed Python byte change moves runtime source root",
          copied_before.sha256 != copied_after.sha256
          and copied_before.file_count == copied_after.file_count)


# ══════════════════════════════════════════════════════════════════════════════
# B. SEARCH_SPACE <-> _suggest_params anti-drift proof
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- B. search space vs _suggest_params ---")


class RecordingTrial:
    """Stand-in for optuna.Trial: records every suggest_* call verbatim and
    returns prescribed values so we can steer the conditional gates."""

    def __init__(self, prescribed):
        self.prescribed = prescribed
        self.calls = []          # (name, kind, args...) in call order

    def suggest_int(self, name, low, high, step=None):
        self.calls.append((name, "int" if step is None else "int_step",
                           low, high) + ((step,) if step is not None else ()))
        return self.prescribed.get(name, low)

    def suggest_float(self, name, low, high, log=False):
        self.calls.append((name, "logfloat" if log else "float", low, high))
        return self.prescribed.get(name, low)

    def suggest_categorical(self, name, choices):
        self.calls.append((name, "cat", tuple(choices)))
        return self.prescribed.get(name, choices[0])


def _spec_to_call(name, spec):
    """INDEPENDENT interpretation of a SEARCH_SPACE spec tuple into the exact
    recorded-call shape. Deliberately re-implements (does not import) the
    trainer's _suggest_one mapping, so a bug there cannot self-certify."""
    kind = spec[0]
    if kind == "int":
        return (name, "int", spec[1], spec[2])
    if kind == "int_step":
        return (name, "int_step", spec[1], spec[2], spec[3])
    if kind == "float":
        return (name, "float", spec[1], spec[2])
    if kind == "logfloat":
        return (name, "logfloat", spec[1], spec[2])
    if kind == "cat":
        return (name, "cat", tuple(spec[1]))
    raise ValueError(f"unknown spec kind {kind!r}")


def _expected_calls(n_conv, n_dense, use_lstm, n_lstm, use_nhits):
    """The exact ordered call list _suggest_params must make, derived from
    SEARCH_SPACE alone (structure per its documented conditional shape)."""
    SS = SEARCH_SPACE
    exp = [_spec_to_call("n_conv_layers",  SS["base"]["n_conv_layers"]),
           _spec_to_call("n_dense_layers", SS["base"]["n_dense_layers"]),
           _spec_to_call("lr",             SS["base"]["lr"]),
           _spec_to_call("weight_decay",   SS["base"]["weight_decay"])]
    for i in range(n_conv):
        for key, spec in SS["per_conv_layer"].items():
            exp.append(_spec_to_call(f"{key}_l{i}", spec))
    for i in range(n_dense):
        for key, spec in SS["per_dense_layer"].items():
            exp.append(_spec_to_call(f"{key}_l{i}", spec))
    if use_lstm:
        exp.append(_spec_to_call("lstm_num_layers", SS["lstm"]["lstm_num_layers"]))
        exp.append(_spec_to_call("lstm_hidden_size", SS["lstm"]["lstm_hidden_size"]))
        if n_lstm > 1:
            exp.append(_spec_to_call("lstm_dropout", SS["lstm"]["lstm_dropout"]))
    if use_nhits:
        exp.append(_spec_to_call("nhits_pool_rates_key",
                                 SS["nhits"]["nhits_pool_rates_key"]))
    return exp


mismatches = []
n_combos = 0
for n_conv in (2, 3, 4):
    for n_dense in (1, 2, 3):
        for use_lstm, n_lstm in ((False, 0), (True, 1), (True, 2)):
            for use_nhits in (False, True):
                n_combos += 1
                trial = RecordingTrial({"n_conv_layers": n_conv,
                                        "n_dense_layers": n_dense,
                                        "lstm_num_layers": n_lstm})
                cfg = {"use_lstm": use_lstm, "use_nhits": use_nhits}
                params = _suggest_params(trial, cfg)
                exp = _expected_calls(n_conv, n_dense, use_lstm, n_lstm, use_nhits)
                if trial.calls != exp:
                    mismatches.append((cfg, n_conv, n_dense, n_lstm,
                                       trial.calls, exp))
                # every suggested name must land in params (and nothing else)
                if set(params) != {c[0] for c in trial.calls}:
                    mismatches.append((cfg, "params/calls key mismatch",
                                       set(params), {c[0] for c in trial.calls}))
check(f"all {n_combos} gate combinations match SEARCH_SPACE exactly",
      not mismatches, f"{len(mismatches)} mismatches: {mismatches[:2]}")


# ══════════════════════════════════════════════════════════════════════════════
# Fixture dataset dirs for C/D/E
# ══════════════════════════════════════════════════════════════════════════════
def _state_seed_id(uid):
    token = (
        "ttbi-state-seed-v1|"
        f"damage_seed={_FIXTURE_DAMAGE_SEED}|{uid}"
    )
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    if value == 0:
        raise RuntimeError("unit fixture derived the reserved RNG seed zero")
    return value


def _named_seed(root_seed, uid, stream, passage=None):
    token = (
        f"{_RNG_SCHEDULE}|root={root_seed}|uid={uid}|stream={stream}"
        + ("" if passage is None else f"|pass={passage:05d}")
    )
    value = int(hashlib.sha256(token.encode("ascii")).hexdigest()[:8], 16)
    if value == 0:
        raise RuntimeError("unit fixture derived the reserved RNG seed zero")
    return value


def _write_semantic_state_table(path, *, damage_level=0.0):
    """Write the complete R11 semantic UID/CRN sidecar used by production."""
    state_seed_ids = np.asarray(
        [_state_seed_id(uid) for uid in _STATE_UIDS],
        dtype=np.uint32,
    )
    state_named = np.asarray([
        [
            _named_seed(int(root), uid, stream)
            for stream in _STATE_STREAM_NAMES
        ]
        for uid, root in zip(_STATE_UIDS, state_seed_ids, strict=True)
    ], dtype=np.uint32)
    passage_named = np.asarray([
        [
            [
                _named_seed(int(root), uid, stream, passage=passage)
                for stream in _PASSAGE_STREAM_NAMES
            ]
            for passage in range(1, _FIXTURE_NPASS + 1)
        ]
        for uid, root in zip(_STATE_UIDS, state_seed_ids, strict=True)
    ], dtype=np.uint32)
    sio.savemat(
        os.path.join(path, "damage_states.mat"),
        {
            "StateFamily": np.asarray(
                ["joint"] * _FIXTURE_N_STATES, dtype=object
            ).reshape(-1, 1),
            "AnchorTarget": np.zeros((_FIXTURE_N_STATES, 1)),
            "AnchorLevel": np.zeros((_FIXTURE_N_STATES, 1)),
            "StateUID": np.asarray(
                _STATE_UIDS, dtype=object
            ).reshape(-1, 1),
            "StateSeedID": state_seed_ids.reshape(-1, 1),
            "LatentBearingFixity":
                np.zeros((_FIXTURE_N_STATES, 2)),
            "LatentCrackOn":
                np.zeros((_FIXTURE_N_STATES, 1), dtype=np.uint8),
            "CrackOn":
                np.zeros((_FIXTURE_N_STATES, 1), dtype=np.uint8),
            "StateNamedStreamSeedID": state_named,
            "PassageNamedStreamSeedID": passage_named,
            "PassageNamedStreamSeedIDFlat": passage_named.reshape(
                _FIXTURE_N_STATES, -1, order="F"
            ),
            "random_stream_schedule_version": _RNG_SCHEDULE,
            "state_stream_names": np.asarray(
                _STATE_STREAM_NAMES, dtype=object
            ).reshape(1, -1),
            "passage_stream_names": np.asarray(
                _PASSAGE_STREAM_NAMES, dtype=object
            ).reshape(1, -1),
            "DamageStates": np.full(
                (_FIXTURE_N_STATES, 4),
                float(damage_level),
                dtype=float,
            ),
            "BearingFixity": np.zeros((_FIXTURE_N_STATES, 2)),
            "scour_supports": np.asarray([[2, 3]], dtype=np.uint32),
        },
        long_field_names=True,
    )


def _fixture(path, variant="A", schema=_EXPECTED_GEN_SCHEMA, manifest=True,
             digests=True, marker=True, tamper_root=False, marker_fp=None,
             matlab_release=_EXPECTED_MATLAB_RELEASE,
             qualification=False, drop_release=False,
             drop_campaign_release=False, drop_qualification=False,
             marker_extra_line=None, metadata_overrides=None,
             metadata_drop=(), state_stamp_overrides=None,
             state_stamp_drop=(), state_stamp_index=2):
    """Minimal R11 identity with manifest, state stamps, digests and marker."""
    os.makedirs(path, exist_ok=True)
    generation = {
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
        'qualification_source_sha256': (
            "a" * 64 if qualification else "PRODUCTION"
        ),
    }
    generation.update(metadata_overrides or {})
    for field in metadata_drop:
        generation.pop(field, None)
    generation_config_json = _generation_config_json(
        variant,
        schema=schema,
        behavior=generation.get(
            "generation_behavior_version",
            _EXPECTED_GENERATION_BEHAVIOR_VERSION,
        ),
        campaign_release=generation.get(
            "campaign_matlab_release", _EXPECTED_MATLAB_RELEASE
        ),
        campaign_environment_sha256=generation.get(
            "campaign_matlab_environment_sha256",
            _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
        ),
        generator_source_root_sha256=generation.get(
            "generator_source_root_sha256", _GENERATOR.sha256
        ),
        qualification_source_sha256=generation.get(
            "qualification_source_sha256",
            "a" * 64 if qualification else "PRODUCTION",
        ),
    )
    fp = hashlib.sha256(
        generation_config_json.encode("utf-8")
    ).hexdigest()
    if manifest:
        ci = {'n_states': _FIXTURE_N_STATES,
              'passages_per_state': _FIXTURE_NPASS,
              'gen_schema': schema, 'gen_fingerprint': fp,
              'generation_config_json': generation_config_json,
              'case_name': _FIXTURE_DATASET,
              'stage': _FIXTURE_STAGE,
              'damage_mode': 'multi_scour',
              'L_bridge_m': 60.0,
              'num_spans': 3,
              'num_supports': 4,
              'scour_supports': '[2 3]',
              'scour_dano_max_frac': 0.60,
              'n_target_healthy': 0,
              'n_scour_only': 0,
              'n_bearing_only': 0,
              'n_nuisance_only': 0,
              'n_joint': _FIXTURE_N_STATES,
              'bearing_mode': 'off',
              'bearing_label': 'fixity_ratio',
              'use_crack_eov': False,
              'crack_draw': 'per_state',
              'profile_mode': 'fixed',
              'profile_draw': 'per_state',
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
        if drop_release:
            del ci['matlab_release']
        if drop_campaign_release:
            del ci['campaign_matlab_release']
        if drop_qualification:
            del ci['release_qualification_run']
        sio.savemat(os.path.join(path, 'case_info.mat'),
                    {'case_info': ci}, long_field_names=True)
    _write_semantic_state_table(path)
    canonical_generation = {
        'matlab_release': matlab_release,
        'campaign_matlab_release': _EXPECTED_MATLAB_RELEASE,
        'release_qualification_run': qualification,
        'actual_matlab_environment_sha256':
            generation.get(
                'actual_matlab_environment_sha256',
                _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
            ),
        'campaign_matlab_environment_sha256':
            generation.get(
                'campaign_matlab_environment_sha256',
                _EXPECTED_MATLAB_ENVIRONMENT_SHA256,
            ),
        'generator_source_root_sha256':
            generation.get('generator_source_root_sha256', _GENERATOR.sha256),
        'qualification_source_sha256':
            generation.get(
                'qualification_source_sha256',
                "a" * 64 if qualification else "PRODUCTION",
            ),
    }
    top_stamps = {
        'file_gen_schema': schema,
        'file_gen_fingerprint': fp,
        'file_matlab_release': canonical_generation['matlab_release'],
        'file_campaign_matlab_release':
            canonical_generation['campaign_matlab_release'],
        'file_release_qualification_run':
            canonical_generation['release_qualification_run'],
        'file_actual_matlab_environment_sha256':
            canonical_generation['actual_matlab_environment_sha256'],
        'file_campaign_matlab_environment_sha256':
            canonical_generation['campaign_matlab_environment_sha256'],
        'file_generator_source_root_sha256':
            canonical_generation['generator_source_root_sha256'],
        'file_qualification_source_sha256':
            canonical_generation['qualification_source_sha256'],
    }
    # Audit r4 verifies bytes; R11 also parses every top-level provenance stamp
    # before a study can be created.
    for state_index, (state_uid, state_seed_id) in enumerate(
        zip(
            _STATE_UIDS,
            (_state_seed_id(uid) for uid in _STATE_UIDS),
            strict=True,
        ),
        start=1,
    ):
        stamps = dict(top_stamps)
        stamps.update({
            'file_state_uid': state_uid,
            'file_state_seed_id': np.uint32(state_seed_id),
            'file_random_stream_schedule_version': _RNG_SCHEDULE,
        })
        if state_index == state_stamp_index:
            stamps.update(state_stamp_overrides or {})
            for field in state_stamp_drop:
                stamps.pop(field, None)
        sio.savemat(
            os.path.join(path, f"{state_index:04d}.mat"),
            {'fixture_state_index': state_index, **stamps},
            long_field_names=True,
        )
    per = {}
    for filename in tuple(
        f"{index:04d}.mat"
        for index in range(1, _FIXTURE_N_STATES + 1)
    ) + ('case_info.mat', 'damage_states.mat'):
        if not Path(path, filename).is_file():
            continue
        payload = Path(path, filename).read_bytes()
        per[filename] = hashlib.sha256(payload).hexdigest()
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    if digests:
        stored_root = ("0" * 64) if tamper_root else root
        sio.savemat(
            os.path.join(path, 'file_digests.mat'),
            {'file_digests': {
                'schema': 'source-digests-v2',
                'scope': 'NNNN.mat+case_info.mat+damage_states.mat',
                'digest_lines': lines,
                'root': stored_root,
            }},
        )
    if marker:
        with open(os.path.join(path, '_GENERATION_COMPLETE'), 'w') as fh:
            fh.write(f"{schema}\n{marker_fp or fp}\n{root}\n")
            if marker_extra_line is not None:
                fh.write(f"{marker_extra_line}\n")


def _refresh_digest_contract(path):
    """Publish a new valid v2 byte root after a deliberate sidecar edit."""
    case_info = sio.loadmat(
        os.path.join(path, 'case_info.mat'),
        simplify_cells=True,
    )['case_info']
    names = sorted(
        f for f in os.listdir(path)
        if re.fullmatch(r'\d{4}\.mat', f)
        or f in {'case_info.mat', 'damage_states.mat'}
    )
    per = {
        name: hashlib.sha256(Path(path, name).read_bytes()).hexdigest()
        for name in names
    }
    lines = "\n".join(f"{name}:{per[name]}" for name in names)
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
    with open(
        os.path.join(path, '_GENERATION_COMPLETE'),
        'w',
        encoding='utf-8',
        newline='\n',
    ) as handle:
        handle.write(
            f"{case_info['gen_schema']}\n"
            f"{case_info['gen_fingerprint']}\n{root}\n"
        )


def _mutate_digest_contract(path, mutation):
    """Mutate a table and keep its root/marker coherent to reach strict guards."""
    digest_path = os.path.join(path, 'file_digests.mat')
    table = sio.loadmat(
        digest_path,
        simplify_cells=True,
    )['file_digests']
    mutation(table)
    lines = str(table['digest_lines'])
    table['root'] = hashlib.sha256(lines.encode()).hexdigest()
    sio.savemat(
        digest_path,
        {'file_digests': table},
        long_field_names=True,
    )
    marker_path = os.path.join(path, '_GENERATION_COMPLETE')
    marker = Path(marker_path).read_text(encoding='utf-8').splitlines()
    Path(marker_path).write_text(
        f"{marker[0]}\n{marker[1]}\n{table['root']}\n",
        encoding='utf-8',
        newline='\n',
    )


def _args(dataset_dir, **over):
    """Baseline build_protocol_descriptors kwargs; **over mutates one knob."""
    base = dict(
        stage="s0_scour", dataset="fixture_ds", dataset_dir=dataset_dir,
        target_supports=[2, 3], bearing_targets=None,
        task="regression", discretization=1,
        seeds=[42, 1337, 2026], n_trials=100, epochs=50,
        use_pruner=True,
        sensor_noise=None,
        architectures=[{"name_short": "PAA_NHiTS", "method": "PAA",
                        "use_space2vec": False, "use_lstm": False,
                        "use_nhits": True, "model_type": "1D_MODULAR"}],
        extra_pairs=[[1, 3]],
        pair_search_stages={"s0_scour", "s16_all"},
        arch_selection_stages={"s0_scour"},
        multi_arch_pair_selection_stages={"s0_scour", "s16_all"},
        schema_tag=EXPECTED_PROTOCOL_SCHEMA_TAG,
        train_protocol=TRAIN_PROTOCOL,
        search_space=SEARCH_SPACE,
        execution_block_policy=EXECUTION_BLOCK_POLICY,
        hyperparameter_policy=HYPERPARAMETER_POLICY,
        capacity_preflight_policy=CAPACITY_PREFLIGHT_POLICY,
    )
    base.update(over)
    return base


root_dir = tempfile.mkdtemp(prefix="protohash_")
_real_descriptor_provenance_reader = cprotocol.read_dataset_provenance


def _standalone_fixture_provenance(dataset_dir, **_registered_expectations):
    """Use the real reader while isolating generic descriptor unit fixtures.

    The three-state dataset intentionally is not a registered 450/475-state
    campaign rung.  Production descriptor assembly still passes and enforces
    all registered-stage expectations; this local adapter suppresses only
    those expectations for sections C/D/F.  Section E calls the unpatched real
    reader directly.
    """
    return _real_descriptor_provenance_reader(dataset_dir)


try:
    ds_a = os.path.join(root_dir, "dsA"); _fixture(ds_a, variant="A")
    ds_b = os.path.join(root_dir, "dsB"); _fixture(ds_b, variant="B")
    cprotocol.read_dataset_provenance = _standalone_fixture_provenance

    # ══════════════════════════════════════════════════════════════════════════
    # C. Core-hash sensitivity: every knob must move the hash
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- C. core-hash knob sensitivity ---")
    core0, full0 = build_protocol_descriptors(**_args(ds_a))
    ch0, fh0 = protocol_hash(core0), protocol_hash(full0)

    check("structured protocol schema is version 6",
          core0["protocol_version"] == 6)
    check("protocol binds live Python runtime source root",
          core0["code"]["python_runtime_source_root_sha256"]
              == _PYTHON_RUNTIME.sha256
          and core0["code"]["python_runtime_source_file_count"]
              == _PYTHON_RUNTIME.file_count
          and _PYTHON_RUNTIME.file_count > 0)
    check("selection metric is the executable objective policy",
          core0["selection"]["selection_metric"] == TRAIN_PROTOCOL["objective"])
    check("execution policy is hash-carried but runtime hardware is not",
          core0["execution_blocking"] == EXECUTION_BLOCK_POLICY
          and "execution_runtime" not in core0
          and full0["rung"]["execution_block"] == "l60"
          and full0["rung"]["execution_anchor"] == "s0_scour")

    core0b, full0b = build_protocol_descriptors(**_args(ds_a))
    check("rebuild with identical inputs -> identical hashes",
          protocol_hash(core0b) == ch0 and protocol_hash(full0b) == fh0)

    alternate_objective = {
        **TRAIN_PROTOCOL,
        "objective": {
            **TRAIN_PROTOCOL["objective"],
            "regression_with_bearing_heads": "mse",
        },
    }
    core_objective, _ = build_protocol_descriptors(
        **_args(ds_a, train_protocol=alternate_objective))
    check("selection metric follows a re-registered objective policy",
          core_objective["selection"]["selection_metric"]
          == alternate_objective["objective"])

    KNOBS = {
        "seeds value":      {"seeds": [42, 1337, 9999]},
        "seeds ORDER":      {"seeds": [1337, 42, 2026]},
        "n_trials":         {"n_trials": 101},
        "epochs":           {"epochs": 51},
        "pruner enabled":   {"use_pruner": False},
        "sensor_noise":     {"sensor_noise": {"mode": "all_mult", "desvio": 0.05}},
        "arch flag flip":   {"architectures": [{"name_short": "PAA_NHiTS",
                                                "method": "PAA",
                                                "use_space2vec": False,
                                                "use_lstm": True,   # flipped
                                                "use_nhits": True,
                                                "model_type": "1D_MODULAR"}]},
        "extra_pairs":      {"extra_pairs": [[0, 3]]},
        "pair_search set":  {"pair_search_stages": {"s0_scour"}},
        "schema_tag":       {"schema_tag": "gs5a-r8"},
        "train_protocol":   {"train_protocol": {**TRAIN_PROTOCOL, "batch_size": 64}},
        "objective policy": {
            "train_protocol": {
                **TRAIN_PROTOCOL,
                "objective": {
                    **TRAIN_PROTOCOL["objective"],
                    "regression_with_bearing_heads": "mse",
                },
            },
        },
        "trial-seed policy": {
            "train_protocol": {
                **TRAIN_PROTOCOL,
                "trial_seed": {
                    **TRAIN_PROTOCOL["trial_seed"],
                    "key": "alternate_seed",
                },
            },
        },
        "determinism policy": {
            "train_protocol": {
                **TRAIN_PROTOCOL,
                "determinism": {
                    **TRAIN_PROTOCOL["determinism"],
                    "cudnn_benchmark": True,
                },
            },
        },
        "search_space":     {"search_space": {**SEARCH_SPACE,
                                              "base": {**SEARCH_SPACE["base"],
                                                       "lr": ("logfloat", 1e-5, 1e-2)}}},
        "task":             {"task": "classification"},
        "discretization":   {"discretization": 5},
        # Feature B knobs (2026-07-19): deployment stages + bootstrap policy.
        "deployment set":   {"deployment_selection_stages": {"s16_all"}},
        "multi-arch pair set": {
            "multi_arch_pair_selection_stages": {"s0_scour"}
        },
        "environment lock": {
            "environment_lock": {
                "path": "environment/campaign.json",
                "sha256": "a" * 64,
                "spec": {"python": "3.13.3"},
            }
        },
        "execution-block policy": {
            "execution_block_policy": {
                **EXECUTION_BLOCK_POLICY,
                "cross_block_inference": {
                    "s0_scour_to_s21_scour4": {
                        **EXECUTION_BLOCK_POLICY[
                            "cross_block_inference"
                        ]["s0_scour_to_s21_scour4"],
                        "rationale": (
                            EXECUTION_BLOCK_POLICY[
                                "cross_block_inference"
                            ]["s0_scour_to_s21_scour4"]["rationale"]
                            + " (alternate registered wording)"
                        ),
                    },
                },
            },
        },
        "hyperparameter policy": {
            "hyperparameter_policy": {
                **HYPERPARAMETER_POLICY,
                "selection_scope":
                    HYPERPARAMETER_POLICY["selection_scope"] + " (mutation)",
            },
        },
        "capacity-preflight policy": {
            "capacity_preflight_policy": {
                **CAPACITY_PREFLIGHT_POLICY,
                "minimum_remaining_headroom": {
                    **CAPACITY_PREFLIGHT_POLICY[
                        "minimum_remaining_headroom"
                    ],
                    "fraction_of_total_memory": 0.19,
                },
            },
        },
        "bootstrap policy": {"bootstrap": {"unit": "state", "n_boot": 1000,
                                           "seed": 42, "ci": 0.95}},
        "statistical inference": {
            "statistical_inference": {
                "finalist_cv": {"n_splits": 5, "n_repeats": 2, "seed": 271828},
                "outer_test": {"hierarchical_bootstrap": "state-first"},
            }
        },
    }
    for label, over in KNOBS.items():
        core_i, _ = build_protocol_descriptors(**_args(ds_a, **over))
        check(f"core hash changes on: {label}", protocol_hash(core_i) != ch0)

    # split constants live in core.dataset; patch one and confirm it propagates.
    _orig = cds.SPLIT_TEST_FRAC
    try:
        cds.SPLIT_TEST_FRAC = 0.25
        core_i, _ = build_protocol_descriptors(**_args(ds_a))
        check("core hash changes on: split test_frac constant",
              protocol_hash(core_i) != ch0)
    finally:
        cds.SPLIT_TEST_FRAC = _orig

    # pruner config lives in core.protocol.OPTUNA_PROTOCOL (pipeline builds from
    # it); patch one value and confirm it propagates.
    _orig = OPTUNA_PROTOCOL["pruner"]["reduction_factor"]
    try:
        OPTUNA_PROTOCOL["pruner"]["reduction_factor"] = 2
        core_i, _ = build_protocol_descriptors(**_args(ds_a))
        check("core hash changes on: pruner reduction_factor",
              protocol_hash(core_i) != ch0)
    finally:
        OPTUNA_PROTOCOL["pruner"]["reduction_factor"] = _orig

    # ══════════════════════════════════════════════════════════════════════════
    # D. Full vs core separation (the cross-rung champion-carry condition)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- D. full-hash vs core-hash separation ---")
    core_b, full_b = build_protocol_descriptors(**_args(ds_b, dataset="fixture_dsB"))
    check("dataset regeneration changes the FULL hash",
          protocol_hash(full_b) != fh0)
    check("dataset regeneration does NOT change the CORE hash",
          protocol_hash(core_b) == ch0)
    ds_sidecar = os.path.join(root_dir, "fixture_sidecar")
    _fixture(ds_sidecar, variant="A")
    _, full_sidecar_0 = build_protocol_descriptors(**_args(ds_sidecar))
    _write_semantic_state_table(ds_sidecar, damage_level=0.1)
    _refresh_digest_contract(ds_sidecar)
    core_sidecar_1, full_sidecar_1 = build_protocol_descriptors(
        **_args(ds_sidecar))
    check("valid sidecar replacement moves the dataset content root",
          protocol_hash(full_sidecar_1) != protocol_hash(full_sidecar_0))
    check("sidecar identity does not change the CORE hash",
          protocol_hash(core_sidecar_1) == ch0)
    core_s, full_s = build_protocol_descriptors(
        **_args(ds_a, stage="s11_bear", bearing_targets=["left", "right"]))
    check("stage/targets change the FULL hash", protocol_hash(full_s) != fh0)
    check("stage/targets do NOT change the CORE hash", protocol_hash(core_s) == ch0)
    core_l99, full_l99 = build_protocol_descriptors(
        **_args(ds_a, stage="s21_scour4"))
    check("L99 rung is assigned to its independent execution block",
          full_l99["rung"]["execution_block"] == "l99"
          and full_l99["rung"]["execution_anchor"] == "s21_scour4"
          and protocol_hash(core_l99) == ch0)
    check("s0->s21 is explicitly descriptive/non-confirmatory",
          core0["execution_blocking"]["cross_block_inference"]
              ["s0_scour_to_s21_scour4"]["confirmatory"] is False
          and core0["execution_blocking"]["cross_block_inference"]
              ["s0_scour_to_s21_scour4"]["status"]
              == "descriptive_nonconfirmatory")

    # ══════════════════════════════════════════════════════════════════════════
    # E. Dataset-provenance hard-fails
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- E. provenance hard-fails ---")
    check_raises("missing dataset dir rejected",
                 lambda: read_dataset_provenance(os.path.join(root_dir, "nope")),
                 RuntimeError)
    p = os.path.join(root_dir, "nomanifest"); _fixture(p, manifest=False)
    check_raises("missing manifest rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "oldschema"); _fixture(p, schema="audit-2026-07-17-r5")
    check_raises("wrong gen_schema rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "nodigests"); _fixture(p, digests=False)
    check_raises("missing file_digests rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "badroot"); _fixture(p, tamper_root=True)
    check_raises("tampered digest root rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "legacydigest"); _fixture(p)
    _mutate_digest_contract(
        p, lambda table: table.update(schema='source-digests-v1'))
    check_raises("legacy digest-table schema rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "wrongscope"); _fixture(p)
    _mutate_digest_contract(
        p, lambda table: table.update(scope='NNNN.mat'))
    check_raises("state-only digest scope rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "missingsidecardigest"); _fixture(p)
    _mutate_digest_contract(
        p,
        lambda table: table.update(
            digest_lines='\n'.join(
                row for row in str(table['digest_lines']).splitlines()
                if not row.startswith('damage_states.mat:')
            )
        ),
    )
    check_raises("missing sidecar digest rejected despite coherent root",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "duplicatedigest"); _fixture(p)
    _mutate_digest_contract(
        p,
        lambda table: table.update(
            digest_lines=(
                str(table['digest_lines']) + '\n'
                + str(table['digest_lines']).splitlines()[0]
            )
        ),
    )
    check_raises("duplicate digest row rejected despite coherent root",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "unsorteddigest"); _fixture(p)
    _mutate_digest_contract(
        p,
        lambda table: table.update(
            digest_lines='\n'.join(
                reversed(str(table['digest_lines']).splitlines())
            )
        ),
    )
    check_raises("unsorted digest rows rejected despite coherent root",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "extrastate"); _fixture(p)
    shutil.copy2(
        os.path.join(p, "0003.mat"),
        os.path.join(p, "0004.mat"),
    )
    _refresh_digest_contract(p)
    check_raises(
        "coherently digested extra state beyond manifest rejected",
        lambda: read_dataset_provenance(p),
        RuntimeError,
    )
    p = os.path.join(root_dir, "nomarker"); _fixture(p, marker=False)
    check_raises("missing completion marker rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "mixmarker"); _fixture(p, marker_fp=FP_OTHER)
    check_raises("marker/manifest fingerprint mismatch rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "extramarker"); _fixture(
        p, marker_extra_line="RESTAMPED")
    check_raises("extra completion-marker line rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "blankmarker"); _fixture(
        p, marker_extra_line="")
    check_raises("extra blank completion-marker line rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "paddedmarker"); _fixture(p)
    marker_path = Path(p, "_GENERATION_COMPLETE")
    marker_path.write_text(
        " " + marker_path.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    check_raises("padded completion-marker line rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "qualification"); _fixture(
        p, qualification=True)
    check_raises("qualification dataset rejected before study creation",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "noqualification"); _fixture(
        p, drop_qualification=True)
    check_raises("missing qualification marker rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "wrongrelease"); _fixture(
        p, matlab_release="R2023b")
    check_raises("wrong MATLAB release rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "nocampaignrelease"); _fixture(
        p, drop_campaign_release=True)
    check_raises("missing campaign release policy rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)

    r11_manifest_mutations = (
        ("wrong generation behaviour rejected",
         {'generation_behavior_version': 'generation-rules-v3'}, ()),
        ("missing actual MATLAB environment descriptor rejected", {},
         ('actual_matlab_environment_descriptor',)),
        ("wrong actual MATLAB environment SHA rejected",
         {'actual_matlab_environment_sha256': '0' * 64}, ()),
        ("missing campaign MATLAB environment descriptor rejected", {},
         ('campaign_matlab_environment_descriptor',)),
        ("wrong campaign MATLAB environment SHA rejected",
         {'campaign_matlab_environment_sha256': '1' * 64}, ()),
        ("wrong generator source root rejected",
         {'generator_source_root_sha256': '2' * 64}, ()),
        ("wrong generator digest lines rejected",
         {'generator_source_digest_lines':
              _GENERATOR.digest_lines + '\nTAMPERED:0'}, ()),
        ("wrong generator source count rejected",
         {'generator_source_file_count': _GENERATOR.file_count + 1}, ()),
        ("wrong production qualification source rejected",
         {'qualification_source_sha256': '3' * 64}, ()),
    )
    for mutation_no, (label, overrides, drops) in enumerate(
            r11_manifest_mutations, start=1):
        p = os.path.join(root_dir, f"r11manifest{mutation_no:02d}")
        _fixture(
            p,
            metadata_overrides=overrides,
            metadata_drop=drops,
        )
        check_raises(
            label,
            lambda p=p: read_dataset_provenance(p),
            RuntimeError,
        )

    # The protocol reader now performs an inexpensive pass over every state's
    # top-level stamps. Recompute the digest chain after each deliberate stamp
    # mutation (inside _fixture), proving semantic validation rather than a
    # generic byte-integrity failure.
    r11_state_stamp_mutations = (
        ("state schema stamp mismatch rejected",
         {'file_gen_schema': 'audit-TAMPERED'}, ()),
        ("state fingerprint stamp mismatch rejected",
         {'file_gen_fingerprint': FP_TAMPERED}, ()),
        ("state MATLAB release stamp mismatch rejected",
         {'file_matlab_release': 'R2023b'}, ()),
        ("state campaign release stamp mismatch rejected",
         {'file_campaign_matlab_release': 'R2023b'}, ()),
        ("state qualification-mode stamp mismatch rejected",
         {'file_release_qualification_run': True}, ()),
        ("state actual-environment stamp mismatch rejected",
         {'file_actual_matlab_environment_sha256': '4' * 64}, ()),
        ("state campaign-environment stamp mismatch rejected",
         {'file_campaign_matlab_environment_sha256': '5' * 64}, ()),
        ("state generator-root stamp mismatch rejected",
         {'file_generator_source_root_sha256': '6' * 64}, ()),
        ("state qualification-source stamp mismatch rejected",
         {'file_qualification_source_sha256': '7' * 64}, ()),
        ("missing state provenance stamp rejected", {},
         ('file_generator_source_root_sha256',)),
    )
    for mutation_no, (label, overrides, drops) in enumerate(
            r11_state_stamp_mutations, start=1):
        p = os.path.join(root_dir, f"r11state{mutation_no:02d}")
        _fixture(
            p,
            state_stamp_overrides=overrides,
            state_stamp_drop=drops,
        )
        check_raises(
            label,
            lambda p=p: read_dataset_provenance(p),
            RuntimeError,
        )

    prov = read_dataset_provenance(ds_a)
    check("valid fixture provenance reads back",
          prov["gen_fingerprint"] == FP_A and prov["n_states"] == 3
          and prov["passages_per_state"] == 4
          and prov["dataset_content_root_sha256"]
          and prov["actual_matlab_environment_sha256"]
              == _EXPECTED_MATLAB_ENVIRONMENT_SHA256
          and prov["campaign_matlab_environment_sha256"]
              == _EXPECTED_MATLAB_ENVIRONMENT_SHA256
          and prov["generator_source_root_sha256"] == _GENERATOR.sha256
          and prov["generator_source_file_count"] == _GENERATOR.file_count
          and prov["qualification_source_sha256"] == "PRODUCTION")

    # ══════════════════════════════════════════════════════════════════════════
    # F. descriptor_diff names the changed knob
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- F. descriptor diff ---")
    core_t, _ = build_protocol_descriptors(**_args(ds_a, n_trials=101))
    diffs = descriptor_diff(core0, core_t)
    check("diff names exactly the changed leaf",
          diffs == ["optuna.n_trials: 100 != 101"], str(diffs))

finally:
    cprotocol.read_dataset_provenance = _real_descriptor_provenance_reader
    shutil.rmtree(root_dir, ignore_errors=True)

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
sys.exit(1 if fails else 0)
