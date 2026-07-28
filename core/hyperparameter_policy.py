"""Authenticated, compute-feasible hyperparameter execution policy.

The campaign has two deliberately different Optuna modes:

``anchor_hpo``
    A 100-trial search, with the registered pruner, performed only for the
    full eight-channel input at the physical-execution-block anchor.

``frozen_singleton``
    One real Optuna trial whose distributions are singleton domains copied
    from the authenticated anchor manifest.  This preserves the normal study
    and artifact path without silently reopening hyperparameter selection for
    every sensor subset and downstream damage rung.

The validators in this module are intentionally strict.  A protocol-hashed
campaign config must declare its mode and, in frozen mode, carry the complete
manifest whose SHA-256 it cites.  This makes the short ``hyperparameter_source``
object a verifiable inclusion claim rather than unauthenticated metadata.
Legacy callers retain historical behaviour only when they carry no campaign
marker at all. A partially stripped campaign config fails closed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from core.execution_environment import (
    EXECUTION_BLOCK_POLICY,
    canonical_execution_block_policy,
    validate_execution_runtime,
)
from core.source_provenance import python_runtime_source_root


POLICY_SCHEMA = "ttbi-hyperparameter-execution-policy-v1"
RUN_PLAN_SCHEMA = "ttbi-hyperparameter-run-plan-v2"
MANIFEST_SCHEMA = "ttbi-hyperparameter-manifest-v3"
STUDY_IDENTITY_SCHEMA = "ttbi-anchor-study-identity-v2"
SOURCE_SCHEMA = "ttbi-frozen-hyperparameter-source-v1"

ANCHOR_HPO_MODE = "anchor_hpo"
FROZEN_SINGLETON_MODE = "frozen_singleton"
LEGACY_MODE = "legacy"

ARCHITECTURES = (
    "PAA_NHiTS",
    "PAA_S2V_NHiTS",
    "PAA_LSTM_NHiTS",
    "PAA_CNN",
)
SEEDS = (42, 1337, 2026)
FULL_DOF_INPUT = tuple(range(8))

HYPERPARAMETER_POLICY = {
    "schema": POLICY_SCHEMA,
    "execution_blocks": {
        "l60": {"anchor_stage": "s0_scour"},
        "l99": {"anchor_stage": "s21_scour4"},
    },
    "architectures": list(ARCHITECTURES),
    "seeds": list(SEEDS),
    "anchor_hpo": {
        "active_dofs": list(FULL_DOF_INPUT),
        "n_trials": 100,
        "use_registered_pruner": True,
    },
    "frozen_singleton": {
        "n_trials": 1,
        "use_registered_pruner": False,
        "optuna_distribution": "one-point domain for every active parameter",
    },
    "selection_scope": (
        "hyperparameters are selected only on the full eight-DOF control at "
        "each execution-block anchor and are frozen by architecture and seed "
        "for every sensor subset and downstream rung in that block"
    ),
    "failure_contract": {
        "failed_trials_allowed": 0,
        "running_or_waiting_trials_allowed": 0,
        "oom_is_fatal": True,
        "terminal_budget": "COMPLETE+PRUNED equals the derived mode budget",
        "at_least_one_complete": True,
        "frozen_exact_state": "one COMPLETE and zero PRUNED",
    },
    "run_plan_schema": RUN_PLAN_SCHEMA,
    "manifest_schema": MANIFEST_SCHEMA,
    "study_identity_schema": STUDY_IDENTITY_SCHEMA,
    "source_schema": SOURCE_SCHEMA,
}

_HEX = frozenset("0123456789abcdef")
_CAMPAIGN_CONFIG_MARKERS = {
    "protocol_hash",
    "protocol_core_hash",
    "protocol_descriptor",
    "hyperparameter_mode",
    "frozen_hyperparameters",
    "hyperparameter_manifest",
    "hyperparameter_manifest_sha256",
    "hyperparameter_source",
    "execution_runtime",
    "campaign_run_tag",
    "execution_receipt_sha256",
    "block_reference_manifest_sha256",
}
_MANIFEST_KEYS = {
    "schema",
    "policy",
    "policy_sha256",
    "execution_block",
    "anchor_stage",
    "protocol_core_hash",
    "anchor_protocol_hash",
    "anchor_dataset",
    "run_tag",
    "execution_receipt_sha256",
    "execution_runtime",
    "python_runtime_source_root_sha256",
    "python_runtime_source_file_count",
    "entries",
}
_ENTRY_KEYS = {
    "architecture",
    "seed",
    "study_identity",
    "study_identity_sha256",
    "params",
    "params_sha256",
}
_IDENTITY_KEYS = {
    "schema",
    "execution_block",
    "anchor_stage",
    "architecture",
    "seed",
    "active_dofs",
    "study_name",
    "protocol_hash",
    "dataset",
    "model_name",
    "execution_environment_sha256",
    "campaign_run_tag",
    "execution_receipt_sha256",
    "study_protocol_record_sha256",
    "effective_n_trials",
    "effective_use_pruner",
    "terminal_counts",
    "best_trial_number",
    "best_trial_value",
    "best_params_sha256",
}
_TERMINAL_COUNT_KEYS = {
    "COMPLETE",
    "PRUNED",
    "FAIL",
    "RUNNING",
    "WAITING",
    "total",
}
_SOURCE_KEYS = {
    "execution_block",
    "anchor_stage",
    "architecture",
    "seed",
    "study_identity_sha256",
    "params_sha256",
}
_RUN_PLAN_KEYS = {
    "schema",
    "mode",
    "execution_block",
    "anchor_stage",
    "stage",
    "dataset",
    "protocol_hash",
    "protocol_core_hash",
    "architecture",
    "seed",
    "active_dofs",
    "effective_n_trials",
    "effective_use_pruner",
    "requested_n_trials",
    "requested_use_pruner",
    "policy_sha256",
    "campaign_run_tag",
    "execution_receipt_sha256",
    "block_reference_manifest_sha256",
    "hyperparameter_manifest_sha256",
    "hyperparameter_source",
}


class HyperparameterPolicyError(RuntimeError):
    """A campaign config or manifest violates the registered HPO policy."""


def canonical_json_value(value: Any) -> Any:
    """Return the unique JSON value accepted by the provenance contract."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HyperparameterPolicyError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return json.loads(payload)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def policy_sha256() -> str:
    """Hash of the exact executable policy object."""

    return canonical_json_sha256(HYPERPARAMETER_POLICY)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HyperparameterPolicyError(f"{label} must be an integer")
    return int(value)


def _strict_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HyperparameterPolicyError(f"{label} must be non-empty text")
    return value


def _validated_architecture(value: object) -> str:
    architecture = _strict_nonempty_text(value, "architecture")
    if architecture not in ARCHITECTURES:
        raise HyperparameterPolicyError(
            f"unregistered architecture {architecture!r}; expected "
            f"{list(ARCHITECTURES)!r}"
        )
    return architecture


def _validated_seed(value: object) -> int:
    seed = _strict_int(value, "seed")
    if seed not in SEEDS:
        raise HyperparameterPolicyError(
            f"unregistered seed {seed}; expected {list(SEEDS)!r}"
        )
    return seed


def _validated_dofs(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise HyperparameterPolicyError("active DOFs must be a list or tuple")
    dofs = tuple(_strict_int(item, "DOF index") for item in value)
    if len(dofs) != len(set(dofs)):
        raise HyperparameterPolicyError("active DOFs contain duplicates")
    if any(index not in FULL_DOF_INPUT for index in dofs):
        raise HyperparameterPolicyError("active DOFs must lie in [0, 7]")
    if not dofs:
        raise HyperparameterPolicyError("at least one active DOF is required")
    return dofs


def _protocol_rung(config: dict) -> dict:
    descriptor = config.get("protocol_descriptor")
    if not isinstance(descriptor, dict):
        raise HyperparameterPolicyError(
            "protocol-hashed config lacks a protocol descriptor"
        )
    rung = descriptor.get("rung")
    if not isinstance(rung, dict):
        raise HyperparameterPolicyError(
            "protocol-hashed config lacks its rung descriptor"
        )
    required = {"stage", "execution_block", "execution_anchor"}
    missing = sorted(required - set(rung))
    if missing:
        raise HyperparameterPolicyError(
            f"protocol rung is missing fields {missing!r}"
        )
    return rung


def _runtime_source_identity() -> tuple[str, int]:
    root = python_runtime_source_root()
    return root.sha256, root.file_count


def _expected_anchor(block: str) -> str:
    blocks = HYPERPARAMETER_POLICY["execution_blocks"]
    if block not in blocks:
        raise HyperparameterPolicyError(
            f"execution block {block!r} is not registered"
        )
    return blocks[block]["anchor_stage"]


def _validate_policy_alignment() -> None:
    """Reject drift between HPO anchors and physical execution blocks."""

    execution = canonical_execution_block_policy(EXECUTION_BLOCK_POLICY)
    expected = {
        key: {"anchor_stage": value["anchor_stage"]}
        for key, value in execution["blocks"].items()
    }
    if HYPERPARAMETER_POLICY["execution_blocks"] != expected:
        raise HyperparameterPolicyError(
            "hyperparameter anchors drifted from EXECUTION_BLOCK_POLICY"
        )


def validate_run_plan(plan: dict) -> dict:
    """Validate the complete derived plan, including mode-specific budgets.

    Terminal-study validation must not merely trust an integer called
    ``effective_n_trials``.  Otherwise a mutated anchor plan with budget one
    could authenticate a one-trial search.  This validator makes the policy's
    100-trial anchor and one-COMPLETE frozen modes structural invariants.
    """

    if not isinstance(plan, dict):
        raise HyperparameterPolicyError(
            "hyperparameter run plan must be a mapping"
        )
    value = canonical_json_value(plan)
    if set(value) != _RUN_PLAN_KEYS:
        raise HyperparameterPolicyError(
            "hyperparameter run-plan fields differ from the contract"
        )
    if value["schema"] != RUN_PLAN_SCHEMA:
        raise HyperparameterPolicyError("unsupported hyperparameter run plan")

    requested_n_trials = _strict_int(
        value["requested_n_trials"], "caller-requested n_trials"
    )
    if requested_n_trials < 1:
        raise HyperparameterPolicyError(
            "caller-requested n_trials must be positive"
        )
    if not isinstance(value["requested_use_pruner"], bool):
        raise HyperparameterPolicyError(
            "caller-requested pruner mode must be boolean"
        )
    effective_n_trials = _strict_int(
        value["effective_n_trials"], "effective trial budget"
    )
    if effective_n_trials < 1:
        raise HyperparameterPolicyError(
            "effective trial budget must be positive"
        )
    if not isinstance(value["effective_use_pruner"], bool):
        raise HyperparameterPolicyError(
            "effective pruner mode must be boolean"
        )

    mode = value["mode"]
    if mode == LEGACY_MODE:
        if any(
            value[key] is not None
            for key in (
                "execution_block",
                "anchor_stage",
                "stage",
                "protocol_hash",
                "protocol_core_hash",
                "policy_sha256",
                "campaign_run_tag",
                "execution_receipt_sha256",
                "block_reference_manifest_sha256",
                "hyperparameter_manifest_sha256",
                "hyperparameter_source",
            )
        ):
            raise HyperparameterPolicyError(
                "legacy run plan carries campaign-only provenance"
            )
        if effective_n_trials != requested_n_trials:
            raise HyperparameterPolicyError(
                "legacy effective budget differs from the caller request"
            )
        if (
            value["effective_use_pruner"]
            is not value["requested_use_pruner"]
        ):
            raise HyperparameterPolicyError(
                "legacy effective pruner differs from the caller request"
            )
        return value

    if mode not in {ANCHOR_HPO_MODE, FROZEN_SINGLETON_MODE}:
        raise HyperparameterPolicyError(
            f"unregistered hyperparameter run mode {mode!r}"
        )
    _validate_policy_alignment()
    block = _strict_nonempty_text(
        value["execution_block"], "run-plan execution block"
    )
    anchor = _strict_nonempty_text(
        value["anchor_stage"], "run-plan anchor"
    )
    if anchor != _expected_anchor(block):
        raise HyperparameterPolicyError(
            "run plan carries an unregistered execution block/anchor"
        )
    stage = _strict_nonempty_text(value["stage"], "run-plan stage")
    _strict_nonempty_text(value["dataset"], "run-plan dataset")
    campaign_run_tag = value["campaign_run_tag"]
    if not isinstance(campaign_run_tag, str):
        raise HyperparameterPolicyError(
            "campaign run plan lacks its exact run_tag"
        )
    if not _is_sha256(value["execution_receipt_sha256"]):
        raise HyperparameterPolicyError(
            "campaign run plan lacks a valid execution receipt SHA-256"
        )
    block_reference_sha = value["block_reference_manifest_sha256"]
    if stage == anchor:
        if block_reference_sha is not None:
            raise HyperparameterPolicyError(
                "block-anchor run plan cannot cite its not-yet-published "
                "reference manifest"
            )
    elif not _is_sha256(block_reference_sha):
        raise HyperparameterPolicyError(
            "follower run plan lacks a valid block-reference manifest SHA-256"
        )
    architecture = _validated_architecture(value["architecture"])
    seed = _validated_seed(value["seed"])
    dofs = _validated_dofs(value["active_dofs"])
    if (
        not _is_sha256(value["protocol_hash"])
        or not _is_sha256(value["protocol_core_hash"])
        or value["policy_sha256"] != policy_sha256()
    ):
        raise HyperparameterPolicyError(
            "run plan carries invalid protocol/policy hashes"
        )

    if mode == ANCHOR_HPO_MODE:
        if stage != anchor or dofs != FULL_DOF_INPUT:
            raise HyperparameterPolicyError(
                "anchor HPO plan is not the full eight-DOF block anchor"
            )
        if (
            effective_n_trials
            != HYPERPARAMETER_POLICY["anchor_hpo"]["n_trials"]
            or value["effective_use_pruner"]
            is not HYPERPARAMETER_POLICY[
                "anchor_hpo"
            ]["use_registered_pruner"]
        ):
            raise HyperparameterPolicyError(
                "anchor HPO plan has a non-registered budget/pruner mode"
            )
        if (
            value["hyperparameter_manifest_sha256"] is not None
            or value["hyperparameter_source"] is not None
        ):
            raise HyperparameterPolicyError(
                "anchor HPO plan cites frozen hyperparameters"
            )
    else:
        if (
            stage == anchor and dofs == FULL_DOF_INPUT
        ):
            raise HyperparameterPolicyError(
                "full-array block anchor cannot use frozen-singleton mode"
            )
        if (
            effective_n_trials
            != HYPERPARAMETER_POLICY["frozen_singleton"]["n_trials"]
            or value["effective_use_pruner"]
            is not HYPERPARAMETER_POLICY[
                "frozen_singleton"
            ]["use_registered_pruner"]
        ):
            raise HyperparameterPolicyError(
                "frozen plan has a non-registered budget/pruner mode"
            )
        manifest_sha = value["hyperparameter_manifest_sha256"]
        source = value["hyperparameter_source"]
        if not _is_sha256(manifest_sha):
            raise HyperparameterPolicyError(
                "frozen plan lacks its manifest SHA-256"
            )
        if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
            raise HyperparameterPolicyError(
                "frozen run-plan source fields differ from the contract"
            )
        if (
            source["execution_block"] != block
            or source["anchor_stage"] != anchor
            or source["architecture"] != architecture
            or source["seed"] != seed
            or not _is_sha256(source["study_identity_sha256"])
            or not _is_sha256(source["params_sha256"])
        ):
            raise HyperparameterPolicyError(
                "frozen run-plan source does not match the selected arm"
            )
    return value


def validate_study_identity(identity: dict) -> dict:
    """Validate one completed anchor-study identity."""

    if not isinstance(identity, dict):
        raise HyperparameterPolicyError("study identity must be a mapping")
    value = canonical_json_value(identity)
    if set(value) != _IDENTITY_KEYS:
        raise HyperparameterPolicyError(
            "study identity fields differ from the registered contract"
        )
    if value["schema"] != STUDY_IDENTITY_SCHEMA:
        raise HyperparameterPolicyError("unsupported study identity schema")
    block = _strict_nonempty_text(
        value["execution_block"], "study execution block"
    )
    anchor = _strict_nonempty_text(value["anchor_stage"], "study anchor")
    if anchor != _expected_anchor(block):
        raise HyperparameterPolicyError("study identity carries the wrong anchor")
    architecture = _validated_architecture(value["architecture"])
    seed = _validated_seed(value["seed"])
    if _validated_dofs(value["active_dofs"]) != FULL_DOF_INPUT:
        raise HyperparameterPolicyError(
            "anchor study identity was not calibrated on all eight DOFs"
        )
    _strict_nonempty_text(value["study_name"], "study name")
    _strict_nonempty_text(value["dataset"], "study dataset")
    _strict_nonempty_text(value["model_name"], "study model name")
    if not isinstance(value["campaign_run_tag"], str):
        raise HyperparameterPolicyError(
            "anchor study identity run_tag must be text"
        )
    for key in (
        "protocol_hash",
        "execution_environment_sha256",
        "execution_receipt_sha256",
        "study_protocol_record_sha256",
        "best_params_sha256",
    ):
        if not _is_sha256(value[key]):
            raise HyperparameterPolicyError(
                f"study identity has invalid {key}"
            )
    if _strict_int(
        value["effective_n_trials"], "effective trial budget"
    ) != HYPERPARAMETER_POLICY["anchor_hpo"]["n_trials"]:
        raise HyperparameterPolicyError(
            "anchor study identity has the wrong trial budget"
        )
    if value["effective_use_pruner"] is not True:
        raise HyperparameterPolicyError(
            "anchor study identity did not use the registered pruner"
        )
    counts = value["terminal_counts"]
    if not isinstance(counts, dict) or set(counts) != _TERMINAL_COUNT_KEYS:
        raise HyperparameterPolicyError("malformed study terminal counts")
    counts = {
        key: _strict_int(counts[key], f"terminal count {key}")
        for key in _TERMINAL_COUNT_KEYS
    }
    if any(count < 0 for count in counts.values()):
        raise HyperparameterPolicyError("terminal counts cannot be negative")
    if counts["total"] != sum(
        counts[key] for key in _TERMINAL_COUNT_KEYS - {"total"}
    ):
        raise HyperparameterPolicyError("terminal counts do not sum to total")
    if (
        counts["FAIL"] != 0
        or counts["RUNNING"] != 0
        or counts["WAITING"] != 0
        or counts["COMPLETE"] < 1
        or counts["COMPLETE"] + counts["PRUNED"]
        != HYPERPARAMETER_POLICY["anchor_hpo"]["n_trials"]
    ):
        raise HyperparameterPolicyError(
            "anchor study is not a clean, exact-budget terminal study"
        )
    best_number = _strict_int(
        value["best_trial_number"], "best trial number"
    )
    if not 0 <= best_number < counts["total"]:
        raise HyperparameterPolicyError(
            "best trial number lies outside the exact anchor trial set"
        )
    best_value = value["best_trial_value"]
    if (
        isinstance(best_value, bool)
        or not isinstance(best_value, (int, float))
        or not math.isfinite(float(best_value))
        or float(best_value) < 0.0
    ):
        raise HyperparameterPolicyError(
            "best trial MSE must be finite and non-negative"
        )
    # Keep these locals intentionally evaluated: their validators carry the
    # exact architecture/seed contract.
    del architecture, seed
    return value


def _validate_params(params: object) -> dict:
    if not isinstance(params, dict) or not params:
        raise HyperparameterPolicyError(
            "manifest hyperparameters must be a non-empty mapping"
        )
    value = canonical_json_value(params)
    if not all(isinstance(key, str) and key for key in value):
        raise HyperparameterPolicyError(
            "hyperparameter names must be non-empty strings"
        )
    return value


class _ParameterValidationTrial:
    """Trial stub used to execute the live frozen-domain validator."""

    def __init__(self, values: dict):
        self.values = values

    def suggest_int(self, name, low, high, step=1):
        value = self.values[name]
        if low != high or value != low:
            raise HyperparameterPolicyError(
                f"parameter {name!r} did not register as a singleton integer"
            )
        return value

    def suggest_float(self, name, low, high, **_kwargs):
        value = float(self.values[name])
        if float(low) != float(high) or value != float(low):
            raise HyperparameterPolicyError(
                f"parameter {name!r} did not register as a singleton float"
            )
        return value

    def suggest_categorical(self, name, choices):
        if list(choices) != [self.values[name]]:
            raise HyperparameterPolicyError(
                f"parameter {name!r} did not register as a singleton category"
            )
        return self.values[name]


def validate_registered_params(architecture: str, params: dict) -> dict:
    """Validate exact conditional keysets/ranges via the live trainer code."""

    architecture = _validated_architecture(architecture)
    values = _validate_params(params)
    flags = {
        "PAA_NHiTS": (False, False, True),
        "PAA_S2V_NHiTS": (True, False, True),
        "PAA_LSTM_NHiTS": (False, True, True),
        "PAA_CNN": (False, False, False),
    }
    use_space2vec, use_lstm, use_nhits = flags[architecture]
    config = {
        "name_short": architecture,
        "use_space2vec": use_space2vec,
        "use_lstm": use_lstm,
        "use_nhits": use_nhits,
        "frozen_hyperparameters": values,
    }
    try:
        from training.trainer import _suggest_params
        reproduced = _suggest_params(
            _ParameterValidationTrial(values), config
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HyperparameterPolicyError(
            f"invalid registered parameters for {architecture}: {exc}"
        ) from exc
    if canonical_json_value(reproduced) != values:
        raise HyperparameterPolicyError(
            f"parameters for {architecture} did not reproduce exactly"
        )
    return values


def validate_manifest(
    manifest: dict,
    *,
    expected_runtime: dict | None = None,
    expected_run_tag: str | None = None,
    expected_execution_receipt_sha256: str | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
) -> dict:
    """Validate and authenticate a complete 4-architecture x 3-seed manifest."""

    _validate_policy_alignment()
    if not isinstance(manifest, dict):
        raise HyperparameterPolicyError("hyperparameter manifest must be a mapping")
    value = canonical_json_value(manifest)
    if set(value) != _MANIFEST_KEYS:
        raise HyperparameterPolicyError(
            "manifest fields differ from the registered contract"
        )
    if value["schema"] != MANIFEST_SCHEMA:
        raise HyperparameterPolicyError("unsupported hyperparameter manifest schema")
    if value["policy"] != canonical_json_value(HYPERPARAMETER_POLICY):
        raise HyperparameterPolicyError(
            "manifest embeds a different hyperparameter policy"
        )
    if (
        not _is_sha256(value["policy_sha256"])
        or value["policy_sha256"] != policy_sha256()
    ):
        raise HyperparameterPolicyError("manifest policy SHA-256 is invalid")

    block = _strict_nonempty_text(
        value["execution_block"], "manifest execution block"
    )
    anchor = _strict_nonempty_text(value["anchor_stage"], "manifest anchor")
    if anchor != _expected_anchor(block):
        raise HyperparameterPolicyError("manifest carries the wrong anchor")
    protocol_core_hash = value["protocol_core_hash"]
    anchor_protocol_hash = value["anchor_protocol_hash"]
    anchor_dataset = _strict_nonempty_text(
        value["anchor_dataset"], "manifest anchor dataset"
    )
    run_tag = value["run_tag"]
    if not isinstance(run_tag, str):
        raise HyperparameterPolicyError("manifest run_tag must be text")
    execution_receipt_sha = value["execution_receipt_sha256"]
    if not _is_sha256(execution_receipt_sha):
        raise HyperparameterPolicyError(
            "manifest execution receipt SHA-256 is invalid"
        )
    if expected_run_tag is not None and run_tag != expected_run_tag:
        raise HyperparameterPolicyError(
            "manifest belongs to another campaign run_tag"
        )
    if (
        expected_execution_receipt_sha256 is not None
        and execution_receipt_sha
        != expected_execution_receipt_sha256
    ):
        raise HyperparameterPolicyError(
            "manifest belongs to another execution receipt"
        )
    if not _is_sha256(protocol_core_hash):
        raise HyperparameterPolicyError(
            "manifest protocol-core SHA-256 is invalid"
        )
    if not _is_sha256(anchor_protocol_hash):
        raise HyperparameterPolicyError(
            "manifest anchor protocol SHA-256 is invalid"
        )
    runtime = validate_execution_runtime(value["execution_runtime"])
    if (
        runtime["execution_block"] != block
        or runtime["anchor_stage"] != anchor
    ):
        raise HyperparameterPolicyError(
            "manifest runtime does not match its execution block and anchor"
        )
    if expected_runtime is not None:
        expected = validate_execution_runtime(expected_runtime)
        if runtime != expected:
            raise HyperparameterPolicyError(
                "manifest runtime differs from the current execution runtime"
            )

    source_sha = value["python_runtime_source_root_sha256"]
    source_count = _strict_int(
        value["python_runtime_source_file_count"], "runtime source file count"
    )
    if not _is_sha256(source_sha) or source_count < 1:
        raise HyperparameterPolicyError(
            "manifest runtime source identity is invalid"
        )
    if expected_source_root_sha256 is None:
        expected_source_root_sha256, live_count = _runtime_source_identity()
        if expected_source_file_count is None:
            expected_source_file_count = live_count
    if source_sha != expected_source_root_sha256:
        raise HyperparameterPolicyError(
            "manifest was produced by a different Python runtime source root"
        )
    if (
        expected_source_file_count is not None
        and source_count != expected_source_file_count
    ):
        raise HyperparameterPolicyError(
            "manifest runtime source file count differs from the live source"
        )

    entries = value["entries"]
    if not isinstance(entries, list):
        raise HyperparameterPolicyError("manifest entries must be a list")
    expected_pairs = {
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in SEEDS
    }
    seen: set[tuple[str, int]] = set()
    study_hashes: set[str] = set()
    study_names: set[str] = set()
    study_record_hashes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise HyperparameterPolicyError(
                "manifest entry fields differ from the registered contract"
            )
        architecture = _validated_architecture(entry["architecture"])
        seed = _validated_seed(entry["seed"])
        pair = (architecture, seed)
        if pair in seen:
            raise HyperparameterPolicyError(
                f"duplicate manifest entry for {pair!r}"
            )
        seen.add(pair)
        identity = validate_study_identity(entry["study_identity"])
        if (
            identity["execution_block"] != block
            or identity["anchor_stage"] != anchor
            or identity["architecture"] != architecture
            or identity["seed"] != seed
            or identity["execution_environment_sha256"]
            != runtime["execution_environment_sha256"]
            or identity["campaign_run_tag"] != run_tag
            or identity["execution_receipt_sha256"]
            != execution_receipt_sha
            or identity["protocol_hash"] != anchor_protocol_hash
            or identity["dataset"] != anchor_dataset
        ):
            raise HyperparameterPolicyError(
                f"manifest entry {pair!r} has inconsistent study identity"
            )
        identity_sha = entry["study_identity_sha256"]
        if (
            not _is_sha256(identity_sha)
            or identity_sha != canonical_json_sha256(identity)
        ):
            raise HyperparameterPolicyError(
                f"manifest entry {pair!r} has invalid study identity SHA-256"
            )
        if identity_sha in study_hashes:
            raise HyperparameterPolicyError(
                "two architecture/seed entries cite the same study identity"
            )
        study_hashes.add(identity_sha)
        study_name = identity["study_name"]
        if study_name in study_names:
            raise HyperparameterPolicyError(
                "two architecture/seed entries cite the same Optuna study"
            )
        study_names.add(study_name)
        study_record_sha = identity["study_protocol_record_sha256"]
        if study_record_sha in study_record_hashes:
            raise HyperparameterPolicyError(
                "two architecture/seed entries cite the same study protocol "
                "record"
            )
        study_record_hashes.add(study_record_sha)
        params = validate_registered_params(
            architecture, entry["params"]
        )
        if (
            not _is_sha256(entry["params_sha256"])
            or entry["params_sha256"] != canonical_json_sha256(params)
            or identity["best_params_sha256"] != entry["params_sha256"]
        ):
            raise HyperparameterPolicyError(
                f"manifest entry {pair!r} has invalid parameter SHA-256"
            )
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        extra = sorted(seen - expected_pairs)
        raise HyperparameterPolicyError(
            "manifest is not the complete architecture x seed factorial "
            f"(missing={missing!r}, extra={extra!r})"
        )
    # Canonical ordering is part of the representation, not just membership.
    expected_order = [
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in SEEDS
    ]
    actual_order = [
        (entry["architecture"], entry["seed"]) for entry in entries
    ]
    if actual_order != expected_order:
        raise HyperparameterPolicyError(
            "manifest entries are not in canonical architecture x seed order"
        )
    return value


def build_manifest(
    entries: Iterable[dict],
    *,
    execution_runtime: dict,
    protocol_core_hash: str,
    anchor_protocol_hash: str,
    anchor_dataset: str,
    run_tag: str,
    execution_receipt_sha256: str,
    source_root_sha256: str | None = None,
    source_file_count: int | None = None,
) -> tuple[dict, str]:
    """Build and immediately validate a canonical anchor manifest."""

    runtime = validate_execution_runtime(execution_runtime)
    if source_root_sha256 is None or source_file_count is None:
        live_sha, live_count = _runtime_source_identity()
        source_root_sha256 = source_root_sha256 or live_sha
        source_file_count = (
            live_count if source_file_count is None else source_file_count
        )
    value = {
        "schema": MANIFEST_SCHEMA,
        "policy": HYPERPARAMETER_POLICY,
        "policy_sha256": policy_sha256(),
        "execution_block": runtime["execution_block"],
        "anchor_stage": runtime["anchor_stage"],
        "protocol_core_hash": protocol_core_hash,
        "anchor_protocol_hash": anchor_protocol_hash,
        "anchor_dataset": anchor_dataset,
        "run_tag": run_tag,
        "execution_receipt_sha256": execution_receipt_sha256,
        "execution_runtime": runtime,
        "python_runtime_source_root_sha256": source_root_sha256,
        "python_runtime_source_file_count": source_file_count,
        "entries": list(entries),
    }
    canonical = validate_manifest(
        value,
        expected_runtime=runtime,
        expected_run_tag=run_tag,
        expected_execution_receipt_sha256=execution_receipt_sha256,
        expected_source_root_sha256=source_root_sha256,
        expected_source_file_count=source_file_count,
    )
    return canonical, canonical_json_sha256(canonical)


def write_manifest(
    path: str | os.PathLike[str],
    manifest: dict,
    *,
    expected_runtime: dict | None = None,
    expected_run_tag: str | None = None,
    expected_execution_receipt_sha256: str | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
) -> str:
    """Atomically publish canonical bytes without replacing prior evidence."""

    value = validate_manifest(
        manifest,
        expected_runtime=expected_runtime,
        expected_run_tag=expected_run_tag,
        expected_execution_receipt_sha256=(
            expected_execution_receipt_sha256
        ),
        expected_source_root_sha256=expected_source_root_sha256,
        expected_source_file_count=expected_source_file_count,
    )
    payload = canonical_json_bytes(value)
    destination = Path(path)
    if not destination.is_absolute():
        raise HyperparameterPolicyError(
            f"hyperparameter manifest path must be absolute: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise HyperparameterPolicyError(
            "hyperparameter manifest parent is not a regular directory"
        )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != payload
            ):
                raise HyperparameterPolicyError(
                    "refusing to overwrite a differing existing "
                    f"hyperparameter manifest: {destination}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_runtime: dict | None = None,
    expected_run_tag: str | None = None,
    expected_execution_receipt_sha256: str | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
) -> dict:
    """Load only exact canonical bytes authenticated by ``expected_sha256``."""

    if not _is_sha256(expected_sha256):
        raise HyperparameterPolicyError(
            "expected manifest SHA-256 is missing or invalid"
        )
    source = Path(path)
    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
    ):
        raise HyperparameterPolicyError(
            f"manifest is not an absolute regular non-symlink file: {source}"
        )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise HyperparameterPolicyError(
            "manifest file bytes do not match the expected SHA-256"
        )
    try:
        parsed = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HyperparameterPolicyError(
            "manifest is not canonical ASCII JSON"
        ) from exc
    if payload != canonical_json_bytes(parsed):
        raise HyperparameterPolicyError(
            "manifest file is JSON-equivalent but not in canonical byte form"
        )
    return validate_manifest(
        parsed,
        expected_runtime=expected_runtime,
        expected_run_tag=expected_run_tag,
        expected_execution_receipt_sha256=(
            expected_execution_receipt_sha256
        ),
        expected_source_root_sha256=expected_source_root_sha256,
        expected_source_file_count=expected_source_file_count,
    )


def build_manifest_entry(
    *,
    study_identity: dict,
    params: dict,
) -> dict:
    """Build the exact manifest entry for one clean anchor study."""

    identity = validate_study_identity(study_identity)
    parameters = validate_registered_params(
        identity["architecture"], params
    )
    return canonical_json_value({
        "architecture": identity["architecture"],
        "seed": identity["seed"],
        "study_identity": identity,
        "study_identity_sha256": canonical_json_sha256(identity),
        "params": parameters,
        "params_sha256": canonical_json_sha256(parameters),
    })


def select_frozen_config(
    manifest: dict,
    *,
    architecture: str,
    seed: int,
    expected_runtime: dict | None = None,
    expected_run_tag: str | None = None,
    expected_execution_receipt_sha256: str | None = None,
    expected_source_root_sha256: str | None = None,
    expected_source_file_count: int | None = None,
) -> dict:
    """Return the authenticated config fields for one frozen singleton trial."""

    value = validate_manifest(
        manifest,
        expected_runtime=expected_runtime,
        expected_run_tag=expected_run_tag,
        expected_execution_receipt_sha256=(
            expected_execution_receipt_sha256
        ),
        expected_source_root_sha256=expected_source_root_sha256,
        expected_source_file_count=expected_source_file_count,
    )
    architecture = _validated_architecture(architecture)
    seed = _validated_seed(seed)
    selected = next(
        entry for entry in value["entries"]
        if entry["architecture"] == architecture and entry["seed"] == seed
    )
    source = {
        "execution_block": value["execution_block"],
        "anchor_stage": value["anchor_stage"],
        "architecture": architecture,
        "seed": seed,
        "study_identity_sha256": selected["study_identity_sha256"],
        "params_sha256": selected["params_sha256"],
    }
    return {
        "hyperparameter_mode": FROZEN_SINGLETON_MODE,
        "frozen_hyperparameters": deepcopy(selected["params"]),
        "hyperparameter_manifest": deepcopy(value),
        "hyperparameter_manifest_sha256": canonical_json_sha256(value),
        "hyperparameter_source": source,
    }


def _validate_frozen_claim(
    config: dict,
    *,
    architecture: str,
    seed: int,
    block: str,
    anchor: str,
    runtime: dict | None,
    protocol_core_hash: str,
    expected_run_tag: str,
    expected_execution_receipt_sha256: str,
) -> tuple[str, dict]:
    required = {
        "frozen_hyperparameters",
        "hyperparameter_manifest",
        "hyperparameter_manifest_sha256",
        "hyperparameter_source",
    }
    missing = sorted(required - set(config))
    if missing:
        raise HyperparameterPolicyError(
            f"frozen config is missing authenticated fields {missing!r}"
        )
    manifest = validate_manifest(
        config["hyperparameter_manifest"],
        expected_runtime=runtime,
        expected_run_tag=expected_run_tag,
        expected_execution_receipt_sha256=(
            expected_execution_receipt_sha256
        ),
    )
    manifest_sha = config["hyperparameter_manifest_sha256"]
    if (
        not _is_sha256(manifest_sha)
        or manifest_sha != canonical_json_sha256(manifest)
    ):
        raise HyperparameterPolicyError(
            "hyperparameter manifest SHA-256 does not authenticate the manifest"
        )
    if (
        manifest["execution_block"] != block
        or manifest["anchor_stage"] != anchor
    ):
        raise HyperparameterPolicyError(
            "frozen config cites a manifest from another execution block"
        )
    if manifest["protocol_core_hash"] != protocol_core_hash:
        raise HyperparameterPolicyError(
            "frozen config cites a manifest from another core protocol"
        )
    source = config["hyperparameter_source"]
    if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
        raise HyperparameterPolicyError(
            "hyperparameter_source must contain exactly the registered fields"
        )
    expected_entry = next(
        entry for entry in manifest["entries"]
        if entry["architecture"] == architecture and entry["seed"] == seed
    )
    expected_source = {
        "execution_block": block,
        "anchor_stage": anchor,
        "architecture": architecture,
        "seed": seed,
        "study_identity_sha256": expected_entry["study_identity_sha256"],
        "params_sha256": expected_entry["params_sha256"],
    }
    if canonical_json_value(source) != expected_source:
        raise HyperparameterPolicyError(
            "hyperparameter_source is not the matching manifest entry"
        )
    params = _validate_params(config["frozen_hyperparameters"])
    if (
        canonical_json_sha256(params) != expected_entry["params_sha256"]
        or params != expected_entry["params"]
    ):
        raise HyperparameterPolicyError(
            "frozen hyperparameters differ from the authenticated manifest entry"
        )
    return manifest_sha, expected_source


def derive_execution_plan(
    config: dict,
    *,
    dataset_name: str | None = None,
    requested_n_trials: int,
    requested_use_pruner: bool,
    execution_runtime: dict | None = None,
) -> dict:
    """Derive the only permitted budget/pruner combination for ``config``."""

    requested_n_trials = _strict_int(
        requested_n_trials, "caller-requested n_trials"
    )
    if requested_n_trials < 1:
        raise HyperparameterPolicyError("requested n_trials must be positive")
    if not isinstance(requested_use_pruner, bool):
        raise HyperparameterPolicyError("requested use_pruner must be boolean")

    if not config.get("protocol_hash"):
        campaign_markers = sorted(
            set(config).intersection(_CAMPAIGN_CONFIG_MARKERS)
        )
        if campaign_markers:
            raise HyperparameterPolicyError(
                "config carries campaign markers but no valid protocol_hash: "
                f"{campaign_markers!r}"
            )
        return validate_run_plan({
            "schema": RUN_PLAN_SCHEMA,
            "mode": LEGACY_MODE,
            "execution_block": None,
            "anchor_stage": None,
            "stage": None,
            "dataset": dataset_name,
            "protocol_hash": None,
            "protocol_core_hash": None,
            "architecture": config.get("name_short"),
            "seed": config.get("seed"),
            "active_dofs": list(config.get("dofs", [])),
            "effective_n_trials": requested_n_trials,
            "effective_use_pruner": requested_use_pruner,
            "requested_n_trials": requested_n_trials,
            "requested_use_pruner": requested_use_pruner,
            "policy_sha256": None,
            "campaign_run_tag": None,
            "execution_receipt_sha256": None,
            "block_reference_manifest_sha256": None,
            "hyperparameter_manifest_sha256": None,
            "hyperparameter_source": None,
        })

    _validate_policy_alignment()
    rung = _protocol_rung(config)
    descriptor = config["protocol_descriptor"]
    from core.protocol import protocol_hash as _protocol_hash
    if _protocol_hash(descriptor) != config.get("protocol_hash"):
        raise HyperparameterPolicyError(
            "config protocol descriptor does not reproduce protocol_hash"
        )
    core_descriptor = descriptor.get("core")
    if not isinstance(core_descriptor, dict):
        raise HyperparameterPolicyError(
            "protocol descriptor lacks its core descriptor"
        )
    if _protocol_hash(core_descriptor) != config.get("protocol_core_hash"):
        raise HyperparameterPolicyError(
            "protocol core descriptor does not reproduce protocol_core_hash"
        )
    stage = _strict_nonempty_text(rung["stage"], "protocol stage")
    rung_dataset = _strict_nonempty_text(
        rung.get("dataset"), "protocol dataset"
    )
    if dataset_name is None or rung_dataset != dataset_name:
        raise HyperparameterPolicyError(
            "pipeline dataset differs from the protocol rung dataset"
        )
    if not _is_sha256(config.get("protocol_hash")):
        raise HyperparameterPolicyError("config protocol hash is invalid")
    protocol_core_hash = config.get("protocol_core_hash")
    if not _is_sha256(protocol_core_hash):
        raise HyperparameterPolicyError(
            "protocol-hashed config lacks a valid protocol_core_hash"
        )
    block = _strict_nonempty_text(
        rung["execution_block"], "protocol execution block"
    )
    anchor = _strict_nonempty_text(
        rung["execution_anchor"], "protocol execution anchor"
    )
    if anchor != _expected_anchor(block):
        raise HyperparameterPolicyError(
            "protocol rung carries an unregistered block/anchor pair"
        )
    lineage_fields = {
        "campaign_run_tag",
        "execution_receipt_sha256",
        "block_reference_manifest_sha256",
    }
    missing_lineage = sorted(lineage_fields - set(config))
    if missing_lineage:
        raise HyperparameterPolicyError(
            "protocol-hashed config lacks execution/reference lineage fields "
            f"{missing_lineage!r}"
        )
    campaign_run_tag = config["campaign_run_tag"]
    if not isinstance(campaign_run_tag, str):
        raise HyperparameterPolicyError(
            "protocol-hashed config run_tag must be text"
        )
    execution_receipt_sha = config["execution_receipt_sha256"]
    if not _is_sha256(execution_receipt_sha):
        raise HyperparameterPolicyError(
            "protocol-hashed config lacks a valid execution receipt SHA-256"
        )
    block_reference_sha = config["block_reference_manifest_sha256"]
    if stage == anchor:
        if block_reference_sha is not None:
            raise HyperparameterPolicyError(
                "block-anchor config cannot cite its not-yet-published "
                "reference manifest"
            )
    elif not _is_sha256(block_reference_sha):
        raise HyperparameterPolicyError(
            "follower config lacks a valid block-reference manifest SHA-256"
        )
    architecture = _validated_architecture(config.get("name_short"))
    seed = _validated_seed(config.get("seed"))
    dofs = _validated_dofs(config.get("dofs"))
    runtime = (
        validate_execution_runtime(execution_runtime)
        if execution_runtime is not None else None
    )
    if runtime is not None and (
        runtime["execution_block"] != block
        or runtime["anchor_stage"] != anchor
    ):
        raise HyperparameterPolicyError(
            "execution runtime differs from the protocol block/anchor"
        )

    expected_mode = (
        ANCHOR_HPO_MODE
        if stage == anchor and dofs == FULL_DOF_INPUT
        else FROZEN_SINGLETON_MODE
    )
    mode = config.get("hyperparameter_mode")
    if mode not in {ANCHOR_HPO_MODE, FROZEN_SINGLETON_MODE}:
        raise HyperparameterPolicyError(
            "protocol-hashed config must explicitly declare "
            f"hyperparameter_mode as {expected_mode!r}"
        )
    if mode != expected_mode:
        raise HyperparameterPolicyError(
            f"{stage}/{list(dofs)} requires {expected_mode!r}, not {mode!r}"
        )

    manifest_sha: str | None = None
    source: dict | None = None
    if mode == ANCHOR_HPO_MODE:
        forbidden = {
            "frozen_hyperparameters",
            "hyperparameter_manifest",
            "hyperparameter_manifest_sha256",
            "hyperparameter_source",
        }.intersection(config)
        if forbidden:
            raise HyperparameterPolicyError(
                "anchor HPO config carries frozen-manifest fields "
                f"{sorted(forbidden)!r}"
            )
        effective_n_trials = HYPERPARAMETER_POLICY["anchor_hpo"]["n_trials"]
        effective_pruner = HYPERPARAMETER_POLICY[
            "anchor_hpo"
        ]["use_registered_pruner"]
    else:
        manifest_sha, source = _validate_frozen_claim(
            config,
            architecture=architecture,
            seed=seed,
            block=block,
            anchor=anchor,
            runtime=runtime,
            protocol_core_hash=protocol_core_hash,
            expected_run_tag=campaign_run_tag,
            expected_execution_receipt_sha256=execution_receipt_sha,
        )
        effective_n_trials = HYPERPARAMETER_POLICY[
            "frozen_singleton"
        ]["n_trials"]
        effective_pruner = HYPERPARAMETER_POLICY[
            "frozen_singleton"
        ]["use_registered_pruner"]

    return validate_run_plan({
        "schema": RUN_PLAN_SCHEMA,
        "mode": mode,
        "execution_block": block,
        "anchor_stage": anchor,
        "stage": stage,
        "dataset": rung_dataset,
        "protocol_hash": config["protocol_hash"],
        "protocol_core_hash": protocol_core_hash,
        "architecture": architecture,
        "seed": seed,
        "active_dofs": list(dofs),
        "effective_n_trials": effective_n_trials,
        "effective_use_pruner": effective_pruner,
        "requested_n_trials": requested_n_trials,
        "requested_use_pruner": requested_use_pruner,
        "policy_sha256": policy_sha256(),
        "campaign_run_tag": campaign_run_tag,
        "execution_receipt_sha256": execution_receipt_sha,
        "block_reference_manifest_sha256": block_reference_sha,
        "hyperparameter_manifest_sha256": manifest_sha,
        "hyperparameter_source": source,
    })


def terminal_counts(study: Any) -> dict:
    """Return exact Optuna terminal-state counts without importing Optuna."""

    names = ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
    counts = {name: 0 for name in names}
    for trial in study.trials:
        state = getattr(trial.state, "name", str(trial.state).split(".")[-1])
        if state not in counts:
            raise HyperparameterPolicyError(
                f"study contains unsupported trial state {state!r}"
            )
        counts[state] += 1
    counts["total"] = len(study.trials)
    return counts


def validate_terminal_study(study: Any, plan: dict) -> dict:
    """Enforce the zero-FAIL, exact-budget publication gate."""

    plan = validate_run_plan(plan)
    counts = terminal_counts(study)
    budget = _strict_int(plan.get("effective_n_trials"), "effective budget")
    useful = counts["COMPLETE"] + counts["PRUNED"]
    if (
        counts["FAIL"] != 0
        or counts["RUNNING"] != 0
        or counts["WAITING"] != 0
        or counts["COMPLETE"] < 1
        or useful != budget
        or counts["total"] != budget
    ):
        raise HyperparameterPolicyError(
            "study is not a clean exact-budget terminal study: "
            f"counts={counts!r}, budget={budget}"
        )
    if plan["mode"] == FROZEN_SINGLETON_MODE and (
        counts["COMPLETE"] != 1 or counts["PRUNED"] != 0
    ):
        raise HyperparameterPolicyError(
            "frozen singleton study must contain exactly one COMPLETE trial "
            "and no PRUNED trial"
        )
    return counts


def anchor_study_identity(
    *,
    study: Any,
    config: dict,
    plan: dict,
    dataset_name: str,
    study_protocol_record: dict,
) -> dict:
    """Create the manifest identity after an anchor study passes its gate."""

    plan = validate_run_plan(plan)
    if plan["mode"] != ANCHOR_HPO_MODE:
        raise HyperparameterPolicyError(
            "only anchor HPO studies can produce manifest identities"
        )
    rederived = derive_execution_plan(
        config,
        dataset_name=dataset_name,
        requested_n_trials=plan["requested_n_trials"],
        requested_use_pruner=plan["requested_use_pruner"],
        execution_runtime=config.get("execution_runtime"),
    )
    if plan != rederived:
        raise HyperparameterPolicyError(
            "anchor identity plan does not derive from its config/runtime"
        )
    counts = validate_terminal_study(study, plan)
    record = canonical_json_value(study_protocol_record)
    stored_record = getattr(study, "user_attrs", {}).get(
        "ttbi_protocol_record"
    )
    if canonical_json_value(stored_record) != record:
        raise HyperparameterPolicyError(
            "anchor identity record is not the record stored with the study"
        )
    runtime = validate_execution_runtime(config.get("execution_runtime"))
    expected_record_fields = {
        "protocol_hash": config["protocol_hash"],
        "dataset": dataset_name,
        "model_name": config["name"],
        "seed": plan["seed"],
        "n_trials": plan["effective_n_trials"],
        "use_pruner": plan["effective_use_pruner"],
        "execution_environment_sha256":
            runtime["execution_environment_sha256"],
        "campaign_run_tag": plan["campaign_run_tag"],
        "execution_receipt_sha256": plan["execution_receipt_sha256"],
        "block_reference_manifest_sha256":
            plan["block_reference_manifest_sha256"],
        "hyperparameter_execution_plan": plan,
    }
    if (
        not isinstance(record, dict)
        or any(
            record.get(key) != canonical_json_value(expected)
            for key, expected in expected_record_fields.items()
        )
    ):
        raise HyperparameterPolicyError(
            "anchor study protocol record is inconsistent with its plan/config"
        )
    identity = {
        "schema": STUDY_IDENTITY_SCHEMA,
        "execution_block": plan["execution_block"],
        "anchor_stage": plan["anchor_stage"],
        "architecture": plan["architecture"],
        "seed": plan["seed"],
        "active_dofs": plan["active_dofs"],
        "study_name": study.study_name,
        "protocol_hash": config["protocol_hash"],
        "dataset": dataset_name,
        "model_name": config["name"],
        "execution_environment_sha256":
            runtime["execution_environment_sha256"],
        "campaign_run_tag": plan["campaign_run_tag"],
        "execution_receipt_sha256": plan["execution_receipt_sha256"],
        "study_protocol_record_sha256": canonical_json_sha256(record),
        "effective_n_trials": plan["effective_n_trials"],
        "effective_use_pruner": plan["effective_use_pruner"],
        "terminal_counts": counts,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "best_params_sha256": canonical_json_sha256(study.best_params),
    }
    return validate_study_identity(identity)
