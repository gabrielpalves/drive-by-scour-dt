"""check_protocol_hash.py - adversarial test of the unified protocol hash
(audit R7.1 remaining item 1, 2026-07-19).

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
  E. read_dataset_provenance HARD-FAILS on every incomplete/tampered dataset
     variant (missing manifest / wrong schema / missing digests / tampered
     root / missing marker / marker-manifest mismatch).
  F. descriptor_diff names the exact knob that differs (the error-message path).
"""
import hashlib
import os
import shutil
import sys
import tempfile

import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.protocol import (canonical_json, protocol_hash, short_hash,      # noqa: E402
                           descriptor_diff, build_protocol_descriptors,
                           read_dataset_provenance, OPTUNA_PROTOCOL)
from core.dataset import _EXPECTED_GEN_SCHEMA                              # noqa: E402
import core.dataset as cds                                                 # noqa: E402
from training.trainer import (TRAIN_PROTOCOL, SEARCH_SPACE,                # noqa: E402
                              _suggest_params)

fails = 0


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
def _fixture(path, fp="fp-A", schema=_EXPECTED_GEN_SCHEMA, manifest=True,
             digests=True, marker=True, tamper_root=False, marker_fp=None):
    """Minimal on-disk dataset identity: manifest + digests + marker. (The state
    .mat files themselves are the LOADER's concern - read_dataset_provenance
    binds identity only, per its docstring.)"""
    os.makedirs(path, exist_ok=True)
    if manifest:
        sio.savemat(os.path.join(path, 'case_info.mat'),
                    {'case_info': {'n_states': 3, 'passages_per_state': 4,
                                   'gen_schema': schema, 'gen_fingerprint': fp,
                                   'scour_dano_max_frac': 0.60}})
    per = {'0001.mat': hashlib.sha256(b'one').hexdigest(),
           '0002.mat': hashlib.sha256(b'two').hexdigest(),
           '0003.mat': hashlib.sha256(b'three').hexdigest()}
    lines = "\n".join(f"{k}:{per[k]}" for k in sorted(per))
    root = hashlib.sha256(lines.encode()).hexdigest()
    if digests:
        stored_root = ("0" * 64) if tamper_root else root
        sio.savemat(os.path.join(path, 'file_digests.mat'),
                    {'file_digests': {'digest_lines': lines, 'root': stored_root}})
    if marker:
        with open(os.path.join(path, '_GENERATION_COMPLETE'), 'w') as fh:
            fh.write(f"{schema}\n{marker_fp or fp}\n{root}\n")


def _args(dataset_dir, **over):
    """Baseline build_protocol_descriptors kwargs; **over mutates one knob."""
    base = dict(
        stage="s0_scour", dataset="fixture_ds", dataset_dir=dataset_dir,
        target_supports=[2, 3], bearing_targets=None,
        task="regression", discretization=1,
        seeds=[42, 1337, 2026], n_trials=100, epochs=50,
        sensor_noise=None,
        architectures=[{"name_short": "PAA_NHiTS", "method": "PAA",
                        "use_space2vec": False, "use_lstm": False,
                        "use_nhits": True, "model_type": "1D_MODULAR"}],
        extra_pairs=[[1, 3]],
        pair_search_stages={"s0_scour", "s16_all"},
        arch_selection_stages={"s0_scour"},
        schema_tag="gs4a20260718r7",
        train_protocol=TRAIN_PROTOCOL,
        search_space=SEARCH_SPACE,
    )
    base.update(over)
    return base


root_dir = tempfile.mkdtemp(prefix="protohash_")
try:
    ds_a = os.path.join(root_dir, "dsA"); _fixture(ds_a, fp="fp-A")
    ds_b = os.path.join(root_dir, "dsB"); _fixture(ds_b, fp="fp-B")  # regenerated

    # ══════════════════════════════════════════════════════════════════════════
    # C. Core-hash sensitivity: every knob must move the hash
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- C. core-hash knob sensitivity ---")
    core0, full0 = build_protocol_descriptors(**_args(ds_a))
    ch0, fh0 = protocol_hash(core0), protocol_hash(full0)

    core0b, full0b = build_protocol_descriptors(**_args(ds_a))
    check("rebuild with identical inputs -> identical hashes",
          protocol_hash(core0b) == ch0 and protocol_hash(full0b) == fh0)

    KNOBS = {
        "seeds value":      {"seeds": [42, 1337, 9999]},
        "seeds ORDER":      {"seeds": [1337, 42, 2026]},
        "n_trials":         {"n_trials": 101},
        "epochs":           {"epochs": 51},
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
        "search_space":     {"search_space": {**SEARCH_SPACE,
                                              "base": {**SEARCH_SPACE["base"],
                                                       "lr": ("logfloat", 1e-5, 1e-2)}}},
        "task":             {"task": "classification"},
        "discretization":   {"discretization": 5},
        # Feature B knobs (2026-07-19): deployment stages + bootstrap policy.
        "deployment set":   {"deployment_selection_stages": {"s16_all"}},
        "bootstrap policy": {"bootstrap": {"unit": "state", "n_boot": 1000,
                                           "seed": 42, "ci": 0.95}},
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
    core_s, full_s = build_protocol_descriptors(
        **_args(ds_a, stage="s11_bear", bearing_targets=["left", "right"]))
    check("stage/targets change the FULL hash", protocol_hash(full_s) != fh0)
    check("stage/targets do NOT change the CORE hash", protocol_hash(core_s) == ch0)

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
    p = os.path.join(root_dir, "nomarker"); _fixture(p, marker=False)
    check_raises("missing completion marker rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    p = os.path.join(root_dir, "mixmarker"); _fixture(p, marker_fp="fp-OTHER")
    check_raises("marker/manifest fingerprint mismatch rejected",
                 lambda: read_dataset_provenance(p), RuntimeError)
    prov = read_dataset_provenance(ds_a)
    check("valid fixture provenance reads back",
          prov["gen_fingerprint"] == "fp-A" and prov["n_states"] == 3
          and prov["passages_per_state"] == 4)

    # ══════════════════════════════════════════════════════════════════════════
    # F. descriptor_diff names the changed knob
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- F. descriptor diff ---")
    core_t, _ = build_protocol_descriptors(**_args(ds_a, n_trials=101))
    diffs = descriptor_diff(core0, core_t)
    check("diff names exactly the changed leaf",
          diffs == ["optuna.n_trials: 100 != 101"], str(diffs))

finally:
    shutil.rmtree(root_dir, ignore_errors=True)

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
sys.exit(1 if fails else 0)
