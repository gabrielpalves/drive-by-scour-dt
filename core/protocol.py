"""
core/protocol.py
================
The UNIFIED PROTOCOL HASH (audit R7.1 remaining item 1, 2026-07-19).

Why this module exists
----------------------
Every provenance mechanism built in R4-R7.1 guards ONE artifact class:
gen_fingerprint guards the .mat data, the cache sidecar guards the .npy caches,
SCHEMA_TAG guards study names against *code-version* mixing. What was still
missing is a single value that pins down the ENTIRE experimental protocol -
"if any knob that could change a reported number changes, every downstream
name/manifest changes with it, and nothing stale can be resumed or reused."

This module provides that value as a SHA-256 over a CANONICAL-JSON descriptor
of the whole protocol:

    dataset  : gen_schema + gen_fingerprint + dataset-content root + counts
    split    : policy / seed / fractions      (from core.dataset constants)
    training : SEEDS, N_TRIALS, EPOCHS, batch size, patience, optimizer, ...
    optuna   : sampler + pruner configuration (OPTUNA_PROTOCOL below - the
               pipeline BUILDS its sampler/pruner from this dict, so the hash
               cannot drift from the running code)
    search   : the full hyperparameter search space (training.trainer.SEARCH_SPACE,
               which _suggest_params reads - again: data IS the code)
    noise    : the full SENSOR_NOISE dict (or None)
    targets  : TARGET_SUPPORTS / BEARING_TARGETS
    preproc  : PAA n_segments etc.            (from core.dataset constants)
    code     : SCHEMA_TAG + expected gen_schema + cache schema tag + SHA-256
               root over the executing Python/environment source boundary

TWO hashes, not one (this distinction is the heart of the design):

  * protocol_core_hash - over everything EXCEPT the stage/dataset identity.
        Each geometry block has an independent anchor-selected reference:
        s0/L60 for the registered L60 graph and s21/L99 for the L99 stress
        block. Followers require the same selection protocol and their exact
        block-local, content-addressed manifest; no selection crosses geometry
        blocks. The s0→s21 performance comparison remains descriptive only.
  * protocol_hash (full) - core + stage + dataset name/targets/provenance.
        This is the per-rung identity: it goes into every study name and the
        summary-dir name, so re-running after ANY change (even just a dataset
        regeneration, which changes the dataset-content root) can never resume
        an old Optuna study or overwrite an old summary in place.

Single-source-of-truth rule
---------------------------
Nothing in the descriptor is a hand-maintained *description* of behaviour.
Every entry is either (a) a constant the executing code itself reads
(split seed, n_segments, pruner config, search space), or (b) a value read
from the generated artifacts (gen_fingerprint, root digest). This is the
lesson from the R5 MATLAB fingerprint (sprintf description -> drift-prone;
canonical struct hash -> safe).

Imported by:
    comprehensive_ablation_multidamage.py - descriptor assembly + hashing
    training/pipeline.py                  - OPTUNA_PROTOCOL (sampler/pruner cfg)
    check_protocol_hash.py                - the adversarial self-test
"""

import hashlib
import json
import os

from core.source_provenance import python_runtime_source_root


# ──────────────────────────────────────────────────────────────────────────────
# 1. Optuna protocol - THE configuration the pipeline builds its sampler and
#    pruner from. Lives here (not in training/pipeline.py) because this module
#    must stay importable on machines WITHOUT optuna (the audit-check PC), and
#    pipeline.py imports optuna at module level.
# ──────────────────────────────────────────────────────────────────────────────
OPTUNA_PROTOCOL = {
    "direction": "minimize",
    "sampler": {
        "class": "TPESampler",
        # n_startup_trials is a RULE, not a number: it scales with n_trials.
        # Recorded as the rule string + the operands so the hash changes if the
        # rule changes. pipeline._create_or_resume_study evaluates exactly this.
        "n_startup_rule": "max(10, n_trials // 4)",
        "multivariate": True,
        "constant_liar": True,
    },
    "pruner": {
        "class": "SuccessiveHalvingPruner",
        "min_resource": 4,          # >= 4 epochs before any pruning
        "reduction_factor": 3,      # keep top 1/3 at each rung
        "min_early_stopping_rate": 0,
    },
    # R11 fail-closed policy: an OOM or any other FAIL invalidates the study.
    # Capacity is qualified before study creation; retrying a failed model
    # would make candidate compute conditional on hidden recovery behavior.
    "max_fail_slack": 0,
    "failure_policy": "any FAIL, including CPU/CUDA OOM, is fatal and is never retried",
    # What counts as a finished study is executable through
    # core.hyperparameter_policy.validate_terminal_study.
    "finished_rule": (
        "FAIL=RUNNING=WAITING=0 and total=COMPLETE+PRUNED=n_trials "
        "and COMPLETE>=1; frozen singleton additionally requires "
        "COMPLETE=1 and PRUNED=0"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Canonical hashing
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_default(obj):
    """json.dumps fallback for the few non-JSON types we allow in descriptors.

    Sets are ORDER-CANONICALISED (sorted) because two runs must never hash
    differently just because a set iterated in a different order. Anything else
    is REJECTED loudly - silently str()-ing an unexpected object (a numpy
    scalar, a Path, a function) would make the hash depend on repr details we
    never audited."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(
        f"protocol descriptor contains a non-canonical type {type(obj).__name__}"
        f" ({obj!r}) - convert it to plain int/float/str/bool/list/dict first.")


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace, canonicalised sets.

    Key ordering inside dicts does NOT matter (sort_keys=True); ordering inside
    LISTS does (deliberately - e.g. SEEDS order matters because SEEDS[0] is the
    parity-plot representative)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_canonical_default)


def protocol_hash(descriptor: dict) -> str:
    """Full SHA-256 hex digest of a protocol descriptor."""
    return hashlib.sha256(canonical_json(descriptor).encode("utf-8")).hexdigest()


# 12 hex chars = 48 bits. Embedded in study/dir names for readability; the FULL
# hash is what manifests store and validation compares.
SHORT_LEN = 12


def short_hash(full_hex: str) -> str:
    """The name-embeddable short form of a full protocol hash."""
    return full_hex[:SHORT_LEN]


def descriptor_diff(a: dict, b: dict, _prefix: str = "") -> list[str]:
    """Human-readable leaf-level differences between two descriptors.

    Used in the champion-manifest mismatch error so the user sees WHICH knob
    changed (e.g. 'optuna.pruner.reduction_factor: 3 != 2') instead of just
    two opaque hashes. Recurses into dicts; lists/scalars compare atomically."""
    diffs: list[str] = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        path = f"{_prefix}{k}"
        if k not in a:
            diffs.append(f"{path}: <absent> != {b[k]!r}")
        elif k not in b:
            diffs.append(f"{path}: {a[k]!r} != <absent>")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            diffs.extend(descriptor_diff(a[k], b[k], _prefix=f"{path}."))
        elif a[k] != b[k]:
            diffs.append(f"{path}: {a[k]!r} != {b[k]!r}")
    return diffs


def _runtime_source_identity():
    """Re-hash reviewed Python/runtime bytes for every descriptor assembly."""
    return python_runtime_source_root()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset provenance reader (for the FULL hash)
# ──────────────────────────────────────────────────────────────────────────────

def read_dataset_provenance(
    dataset_dir: str,
    *,
    expected_stage: str | None = None,
    expected_dataset: str | None = None,
    expected_target_supports: list[int] | None = None,
    expected_bearing_targets: list[str] | None | object = None,
) -> dict:
    """Read the generated dataset's identity for the full protocol hash.

    HARD-FAILS (RuntimeError) unless the dataset is COMPLETE and self-consistent:
      * case_info.mat manifest present with schema + fingerprint;
      * gen_schema equals the loader's expected schema (fail HERE, before hours
        of study time, not later inside the loader);
      * file_digests.mat present, root digest matches its own per-file lines;
      * case_info.mat and damage_states.mat bytes enter the full protocol hash,
        even for legacy R8 digest tables that listed state files only;
      * _GENERATION_COMPLETE marker present with schema/fp/root agreeing with
        the manifest + digests.
    This deliberately re-checks what the loader also checks: the protocol hash
    is computed ONCE at driver start, and a wrong dataset must stop the run
    before any study is created under a name derived from it.

    Audit r4 additionally verifies every recorded source file's bytes here,
    before any study is created.  The cache path repeats this call, which is a
    cheap stat-only check after the process-local successful hash pass.
    """
    # Imported here (not module level) to keep this module import-light for the
    # check scripts; core.dataset pulls in torch/sklearn.
    from core.dataset import (_read_manifest, _read_file_digests, _root_digest,
                              _read_dano_max, _EXPECTED_GEN_SCHEMA,
                              verify_source_file_bytes,
                              _validate_campaign_generation_metadata,
                              validate_dataset_state_provenance_stamps,
                              _read_completion_marker, read_state_table,
                              state_identity_descriptor,
                              semantic_split_descriptor, SPLIT_SEED)

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"protocol hash: dataset dir {dataset_dir!r} does not exist - the "
            f"protocol hash includes the dataset fingerprint, so the GENERATED "
            f"dataset must be present (and complete) before the ablation starts.")
    n_states, npass, gen_schema, gen_fp = _read_manifest(dataset_dir)
    if not (n_states and npass and gen_schema and gen_fp):
        raise RuntimeError(
            f"protocol hash: {dataset_dir} has no complete case_info.mat manifest "
            f"(n_states={n_states}, npass={npass}, schema={gen_schema!r}) - "
            f"regenerate with the current A00.")
    if gen_schema != _EXPECTED_GEN_SCHEMA:
        raise RuntimeError(
            f"protocol hash: {dataset_dir} was generated under schema "
            f"{gen_schema!r} but this code expects {_EXPECTED_GEN_SCHEMA!r} - "
            f"regenerate the dataset (do not run studies against foreign data).")
    generation_kwargs = {
        "expected_stage": expected_stage,
        "expected_dataset": expected_dataset,
        "expected_target_supports": expected_target_supports,
    }
    # ``None`` is a meaningful registered value (scour-only stages), so pass
    # bearing targets whenever a stage contract is requested.
    if expected_stage is not None:
        generation_kwargs["expected_bearing_targets"] = (
            list(expected_bearing_targets)
            if expected_bearing_targets is not None else None
        )
    generation = _validate_campaign_generation_metadata(
        dataset_dir,
        **generation_kwargs,
    )
    per_file, root = _read_file_digests(dataset_dir)
    if not per_file or not root:
        raise RuntimeError(
            f"protocol hash: {dataset_dir} lacks file_digests.mat - the source "
            f"root digest is a required hash input. Regenerate with current A00.")
    if _root_digest(per_file) != root:
        raise RuntimeError(
            f"protocol hash: {dataset_dir} file_digests.mat is INTERNALLY "
            f"inconsistent (recomputed root != stored root) - corrupt/tampered.")
    recorded_states = sorted(
        name for name in per_file
        if (
            len(name) == 8
            and name[:4].isdigit()
            and name.endswith(".mat")
        )
    )
    expected_states = [f"{index:04d}.mat" for index in range(1, n_states + 1)]
    if recorded_states != expected_states:
        raise RuntimeError(
            f"protocol hash: {dataset_dir} state inventory is not exactly the "
            f"manifest's contiguous 0001..{n_states:04d} range "
            f"(recorded={recorded_states}, expected={expected_states}).")
    mk = list(_read_completion_marker(dataset_dir))
    if mk != [gen_schema, gen_fp, root]:
        raise RuntimeError(
            f"protocol hash: {dataset_dir} marker content disagrees with "
            f"manifest/digests (marker={mk}, expected="
            f"{[gen_schema, gen_fp, root]}) - mixed or tampered dataset.")
    byte_verification = verify_source_file_bytes(dataset_dir)
    if byte_verification.get("state_count") != int(n_states):
        raise RuntimeError(
            f"protocol hash: {dataset_dir} verified "
            f"{byte_verification.get('state_count')!r} state files but the "
            f"manifest declares n_states={n_states}.")
    validate_dataset_state_provenance_stamps(
        dataset_dir,
        n_states,
        generation,
        expected_schema=gen_schema,
        expected_fingerprint=gen_fp,
    )
    state_table = read_state_table(dataset_dir)
    state_identity = state_identity_descriptor(state_table)
    semantic_split = semantic_split_descriptor(
        state_table, _read_dano_max(dataset_dir), SPLIT_SEED
    )
    if state_identity["state_uid_count"] != int(n_states):
        raise RuntimeError(
            f"protocol hash: {dataset_dir} semantic UID inventory has "
            f"{state_identity['state_uid_count']} states but manifest declares "
            f"{n_states}."
        )
    return {
        "gen_schema":          gen_schema,
        "gen_fingerprint":     gen_fp,
        "dataset_content_root_sha256": root,
        "n_states":            int(n_states),
        "passages_per_state":  int(npass),
        "scour_dano_max_frac": _read_dano_max(dataset_dir),
        "generation_behavior_version":
            generation["generation_behavior_version"],
        "channel_schema_id": generation["channel_schema_id"],
        "state_design_kind": generation["state_design_kind"],
        "rail_end_clearance_m": generation["rail_end_clearance_m"],
        "rail_end_clearance_decision_id":
            generation["rail_end_clearance_decision_id"],
        "matlab_release": generation["matlab_release"],
        "campaign_matlab_release":
            generation["campaign_matlab_release"],
        "actual_matlab_environment_descriptor":
            generation["actual_matlab_environment_descriptor"],
        "actual_matlab_environment_sha256":
            generation["actual_matlab_environment_sha256"],
        "campaign_matlab_environment_descriptor":
            generation["campaign_matlab_environment_descriptor"],
        "campaign_matlab_environment_sha256":
            generation["campaign_matlab_environment_sha256"],
        "generator_source_root_sha256":
            generation["generator_source_root_sha256"],
        "generator_source_file_count":
            generation["generator_source_file_count"],
        "qualification_source_sha256":
            generation["qualification_source_sha256"],
        "release_qualification_run":
            generation["release_qualification_run"],
        "campaign_contract": generation.get("campaign_contract"),
        "state_identity": state_identity,
        "semantic_split": semantic_split,
        "matlab_environment_attestation":
            "exact R11 numerical-stack descriptor and reviewed generator "
            "source root enforced independently by A00 and Python",
        "source_byte_verification": byte_verification,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Descriptor assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_protocol_descriptors(
    *,
    # ---- identity of THIS rung (full hash only) ----
    stage:            str,
    dataset:          str,
    dataset_dir:      str,
    target_supports:  list,
    bearing_targets:  list | None,
    # ---- the dataset-independent protocol (core hash) ----
    task:                 str,
    discretization:       int,
    seeds:                list,
    n_trials:             int,
    epochs:               int,
    use_pruner:           bool,
    sensor_noise:         dict | None,
    architectures:        list,
    extra_pairs:          list,
    pair_search_stages:   set,
    arch_selection_stages: set,
    schema_tag:           str,
    train_protocol:       dict,
    search_space:         dict,
    execution_block_policy: dict,
    hyperparameter_policy: dict,
    capacity_preflight_policy: dict,
    environment_lock:     dict | None = None,
    # Feature B (2026-07-19): deployment-selection stages + the bootstrap-CI
    # policy are protocol too. Defaults keep older callers/tests valid.
    deployment_selection_stages: set = frozenset(),
    multi_arch_pair_selection_stages: set = frozenset(),
    bootstrap:            dict | None = None,
    # Audit r4: repeated grouped-CV and state-first cross-seed inference are
    # executable protocol, not post-processing decoration.  The whole policy is
    # supplied as data by the driver and therefore moves the core hash.
    statistical_inference: dict | None = None,
    # Audit r3 (2026-07-22): non-selectable sensor-budget controls (the full
    # 8-channel array) reported as comparators at every rung. Default keeps older
    # callers/tests valid.
    control_sets:         list = (),
) -> tuple[dict, dict]:
    """Assemble (core_descriptor, full_descriptor).

    core  = everything that defines HOW models are trained/selected - shared
            by both independently anchored geometry blocks; every follower
            validates its own block-local reference against it.
    full  = {"core": core, "rung": <stage/dataset identity + provenance>} -
            unique per rung AND per dataset regeneration; keys study names.

    All inputs are passed EXPLICITLY (keyword-only) rather than read from the
    driver's globals, so the check script can exercise this function with
    fixture values and so the reviewer can see, in one signature, the complete
    list of what the hash covers."""
    from core.dataset import (
        split_protocol,
        PREPROC_PROTOCOL,
        _EXPECTED_GEN_SCHEMA,
        _EXPECTED_CHANNEL_SCHEMA_ID,
    )
    from core.dataset import CACHE_SCHEMA_TAG
    from core.execution_environment import (
        canonical_execution_block_policy,
        execution_block_for_stage,
    )

    runtime_source = _runtime_source_identity()
    execution_policy = canonical_execution_block_policy(
        execution_block_policy
    )
    execution_block, execution_anchor = execution_block_for_stage(
        stage, execution_policy
    )
    core = {
        # v7: v6 plus an explicit physical channel-schema code contract.
        #
        # v6: v5 plus the executable anchor-HPO/frozen-singleton policy and
        # fail-closed CUDA capacity preflight.  Candidate-specific tuning and
        # recoverable OOMs are therefore impossible without moving the hash.
        #
        # v5: v4's content-addressed runtime source boundary plus a hash-carried
        # physical execution-block design. Actual machine/GPU identity remains
        # a runtime attestation, never an input to this shared protocol hash.
        #
        # v4: v3's structured executable training/objective policy plus a
        # content-addressed boundary over the executing Python/environment
        # files. It cannot be confused with descriptors that relied only on
        # manually maintained schema tags.
        "protocol_version": 7,
        "code": {
            # Code-version markers: the driver/loader schema tag, the generator
            # schema the loader requires, and the cache contract tag. Bumping any
            # of them (r7 -> r8) changes every downstream name, as intended.
            "schema_tag":          schema_tag,
            "expected_gen_schema": _EXPECTED_GEN_SCHEMA,
            "expected_channel_schema_id": _EXPECTED_CHANNEL_SCHEMA_ID,
            "cache_schema_tag":    CACHE_SCHEMA_TAG,
            "python_runtime_source_root_sha256": runtime_source.sha256,
            "python_runtime_source_file_count": runtime_source.file_count,
        },
        "task": {"task": task, "discretization": int(discretization)},
        # Split + preprocessing come from core.dataset CONSTANTS that the split
        # and cache code itself uses - see the single-source-of-truth rule above.
        "split":         split_protocol(),
        "preprocessing": PREPROC_PROTOCOL,
        "training":      train_protocol,
        "optuna": {
            **OPTUNA_PROTOCOL,
            "n_trials": int(n_trials),
            "epochs":   int(epochs),
            "use_pruner": bool(use_pruner),
            # SEEDS order is significant (SEEDS[0] = canonical representative),
            # so it is hashed as an ORDERED list.
            "seeds":    [int(s) for s in seeds],
        },
        "search_space": search_space,
        "environment_lock": environment_lock,
        "execution_blocking": execution_policy,
        "hyperparameter_execution": hyperparameter_policy,
        "capacity_preflight": capacity_preflight_policy,
        # None (clean chain) hashes differently from every configured dict, and
        # any dict key/value change (mode, desvio, ...) changes the hash.
        "sensor_noise": sensor_noise,
        # The full arm definitions, not just names: an arch FLAG change (e.g.
        # use_lstm) with an unchanged name must still change the protocol.
        "architectures": architectures,
        "selection": {
            # Embed the exact executable objective policy instead of a second
            # prose description that could drift from TRAIN_PROTOCOL.
            "selection_metric":       train_protocol["objective"],
            "extra_pairs":            [sorted(int(d) for d in p) for p in extra_pairs],
            "control_sets":           [sorted(int(d) for d in c) for c in control_sets],
            "control_arch_policy":    (
                "all trained architectures are compared on the selected pair "
                "and full-array control at full-factorial stages; frozen rungs "
                "report winner/carried architectures (deduplicated)"
            ),
            "pair_search_stages":     sorted(pair_search_stages),
            "arch_selection_stages":  sorted(arch_selection_stages),
            # Feature B: deployment rungs re-open arch x pair; comparators are
            # keyed (architecture, pair); reported CIs are state-level bootstrap.
            "deployment_selection_stages": sorted(deployment_selection_stages),
            "multi_arch_pair_selection_stages":
                sorted(multi_arch_pair_selection_stages),
            "comparator_key":         "(architecture, pair)",
            "bootstrap":              bootstrap,
            "statistical_inference":  statistical_inference,
        },
    }
    full = {
        "core": core,
        "rung": {
            "stage":            stage,
            "dataset":          dataset,
            "execution_block":  execution_block,
            "execution_anchor": execution_anchor,
            "target_supports":  [int(t) for t in target_supports],
            "bearing_targets":  list(bearing_targets) if bearing_targets else None,
            "dataset_provenance": read_dataset_provenance(
                dataset_dir,
                expected_stage=stage,
                expected_dataset=dataset,
                expected_target_supports=target_supports,
                expected_bearing_targets=bearing_targets,
            ),
        },
    }
    return core, full
