"""Data-only policy for the optional contemporary architecture challengers.

The primary Paper-1 factorial deliberately continues to hash
``training.trainer.SEARCH_SPACE`` exactly as before.  ModernTCN and TSLANet
use this separate registry so enabling, disabling, or extending a challenger
cannot masquerade as a change to the historical 16-cell search space.

This module has no torch/training imports and is safe to use from contracts,
the model factory, checks, and dispatch code.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any


CHALLENGER_SEARCH_POLICY_SCHEMA = "paper1-challenger-search-policy-v3"

# Exact workload qualified by ``check_modern_architecture_cuda.py``.  Keeping
# the scope inside the hashed policy prevents a later sensor, target, sequence,
# or batch-size expansion from silently reusing an M=1 capacity result.
CHALLENGER_CAPACITY_SCOPE = {
    "stage": "F40-S",
    "input_selector": "f40s_selected_pair",
    "input_channels": 2,
    "representation_input_lengths": {
        "RAW": 5_831,
        "PAA": 512,
    },
    "output_heads": 1,
    "batch_size": 32,
    "minimum_remaining_headroom": {
        "fraction_of_total_memory": 0.20,
        "absolute_bytes": 1_073_741_824,
    },
    "peak_metric": "torch.cuda.max_memory_reserved",
    "measured_step": "second_complete_adam_step_after_optimizer_warmup",
}

# Optuna persists categorical values as JSON-friendly scalars.  Keep the
# scalar-to-architecture decoders beside the search domain so changing the
# meaning of a frozen label necessarily changes the policy hash.
CHALLENGER_PATCH_LAYOUTS: dict[str, tuple[int, int]] = {
    "8_4": (8, 4),
    "16_8": (16, 8),
    "32_16": (32, 16),
    "64_32": (64, 32),
}
MODERN_TCN_LAYOUTS: dict[
    str, dict[str, tuple[int, ...]]
] = {
    "32x1_64x1": {"dims": (32, 64), "depths": (1, 1)},
    "32x1_64x1_128x2": {
        "dims": (32, 64, 128),
        "depths": (1, 1, 2),
    },
    "64x1_96x2_128x2": {
        "dims": (64, 96, 128),
        "depths": (1, 2, 2),
    },
}

# Constructor choices that are intentionally fixed rather than optimized.
# These values are passed explicitly by ``core.models.build_model`` and are
# part of the policy identity, avoiding silent dependence on class defaults.
CHALLENGER_FACTORY_POLICY = {
    "MODERN_TCN": {
        "stage_stride": 2,
        "small_kernel_size": 5,
        "small_kernel_merged": False,
        # Both memory knobs are OFF only for the registered F40-S selected-pair
        # scope (M=2).  The CUDA preflight binds its shapes to the hashed scope
        # above, warms Adam, and measures a second complete optimizer step.
        # Any scope or search-space expansion requires a fresh qualification;
        # neither knob is inferred to be necessary merely because M exceeds 1.
        "activation_checkpointing": False,
        "depthwise_channel_chunk_size": None,
    },
    "TSLANET": {
        "drop_path_rate_source": "tsla_dropout",
        "use_asb": True,
        "use_icb": True,
        "adaptive_filter": True,
        "spectral_threshold_init": 1.0,
        "spectral_mask_temperature": 0.1,
        "spectral_eps": 1e-6,
        "position_bins": 64,
        "head_hidden_dim": None,
    },
}

# An empty tuple denotes native global-average pooling.  Every non-empty tuple
# is the exact adaptive temporal-pyramid max-pooling bin layout already used by
# the incumbent modular family.  The shared registry prevents head labels from
# drifting between HPO and model construction.
CHALLENGER_POOLING_LAYOUTS: dict[str, tuple[int, ...]] = {
    "gap": (),
    "1_2_4": (1, 2, 4),
    "1_4_8": (1, 4, 8),
    "1_3_6": (1, 3, 6),
    "1_2_4_8": (1, 2, 4, 8),
}


CHALLENGER_OPTIMIZER_SPACE = {
    "lr": ("logfloat", 1e-4, 1e-2),
    "weight_decay": ("logfloat", 1e-5, 1e-3),
}


CHALLENGER_SEARCH_SPACES = {
    "MODERN_TCN": {
        "modern_layout": (
            "cat",
            [
                "32x1_64x1",
                "32x1_64x1_128x2",
                "64x1_96x2_128x2",
            ],
        ),
        "modern_patch_layout": (
            "cat", list(CHALLENGER_PATCH_LAYOUTS)
        ),
        "modern_kernel_size": ("cat", [15, 31, 63]),
        "modern_expansion_ratio": ("cat", [2.0, 3.0, 4.0]),
        "modern_dropout": ("float", 0.0, 0.3),
        "challenger_pooling": ("cat", list(CHALLENGER_POOLING_LAYOUTS)),
    },
    "TSLANET": {
        "tsla_embed_dim": ("int_step", 32, 96, 32),
        "tsla_depth": ("int", 1, 3),
        "tsla_patch_layout": (
            "cat", list(CHALLENGER_PATCH_LAYOUTS)
        ),
        "tsla_mlp_ratio": ("cat", [1.5, 2.0, 3.0]),
        "tsla_dropout": ("float", 0.0, 0.3),
        "challenger_pooling": ("cat", list(CHALLENGER_POOLING_LAYOUTS)),
    },
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def challenger_search_policy() -> dict[str, Any]:
    """Return a detached, JSON-compatible policy descriptor."""

    # JSON round-tripping deliberately normalizes spec tuples to arrays, which
    # is exactly how the policy is persisted in manifests and hashes.
    return json.loads(
        _canonical_json_bytes(
            {
                "schema": CHALLENGER_SEARCH_POLICY_SCHEMA,
                "optimizer": CHALLENGER_OPTIMIZER_SPACE,
                "families": CHALLENGER_SEARCH_SPACES,
                "pooling_layouts": CHALLENGER_POOLING_LAYOUTS,
                "patch_layouts": CHALLENGER_PATCH_LAYOUTS,
                "modern_tcn_layouts": MODERN_TCN_LAYOUTS,
                "factory": CHALLENGER_FACTORY_POLICY,
                "capacity_scope": CHALLENGER_CAPACITY_SCOPE,
            }
        )
    )


def challenger_search_policy_sha256() -> str:
    """Return the canonical identity of the complete challenger HPO domain."""

    return hashlib.sha256(
        _canonical_json_bytes(challenger_search_policy())
    ).hexdigest()


def registered_challenger_search_spaces() -> dict[str, dict[str, tuple]]:
    """Return a mutable detached copy for callers that need local inspection."""

    return deepcopy(CHALLENGER_SEARCH_SPACES)


def _matches_categorical_choice(value: object, choice: object) -> bool:
    """Compare one choice without Python's bool/int or int/float aliases."""

    if isinstance(choice, bool):
        return isinstance(value, bool) and value is choice
    if isinstance(choice, Integral):
        return (
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and int(value) == int(choice)
        )
    if isinstance(choice, Real):
        return (
            isinstance(value, Real)
            and not isinstance(value, Integral)
            and math.isfinite(float(value))
            and float(value) == float(choice)
        )
    return type(value) is type(choice) and value == choice


def _validate_parameter(name: str, value: object, spec: tuple) -> object:
    kind = spec[0]
    if kind == "cat":
        for choice in spec[1]:
            if _matches_categorical_choice(value, choice):
                return deepcopy(choice)
        raise ValueError(
            f"unregistered challenger parameter {name}={value!r}; "
            f"choose one of {list(spec[1])!r}"
        )

    if kind in {"int", "int_step"}:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"challenger parameter {name!r} must be an integer")
        integer = int(value)
        low, high = int(spec[1]), int(spec[2])
        step = int(spec[3]) if kind == "int_step" else 1
        if not low <= integer <= high or (integer - low) % step:
            raise ValueError(
                f"challenger parameter {name}={value!r} is outside its "
                f"registered integer domain"
            )
        return integer

    if kind in {"float", "logfloat"}:
        if (
            isinstance(value, bool)
            or isinstance(value, Integral)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"challenger parameter {name!r} must be one finite float"
            )
        number = float(value)
        if not float(spec[1]) <= number <= float(spec[2]):
            raise ValueError(
                f"challenger parameter {name}={value!r} is outside its "
                f"registered continuous domain"
            )
        return number

    raise ValueError(f"unsupported challenger parameter specification {spec!r}")


def validate_challenger_parameters(
    model_type: str,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed unless ``params`` is exactly one registered HPO point.

    Validation lives with the hashed domain and is used by the model factory,
    so direct artifact reconstruction cannot bypass the same categorical,
    numeric, type, or extra-key rules enforced during HPO/freeze.
    """

    if model_type not in CHALLENGER_SEARCH_SPACES:
        raise ValueError(f"unknown challenger model family {model_type!r}")
    if not isinstance(params, Mapping):
        raise ValueError("challenger parameters must be a mapping")
    if any(not isinstance(name, str) for name in params):
        raise ValueError("challenger parameter keys must be strings")

    complete_space = {
        **CHALLENGER_OPTIMIZER_SPACE,
        **CHALLENGER_SEARCH_SPACES[model_type],
    }
    expected = set(complete_space)
    supplied = set(params)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            "challenger parameter keys do not match the registered domain: "
            f"missing={missing!r}, extra={extra!r}"
        )
    normalized = deepcopy(dict(params))
    for name, spec in complete_space.items():
        normalized[name] = _validate_parameter(name, params[name], spec)
    return normalized


__all__ = [
    "CHALLENGER_CAPACITY_SCOPE",
    "CHALLENGER_FACTORY_POLICY",
    "CHALLENGER_OPTIMIZER_SPACE",
    "CHALLENGER_PATCH_LAYOUTS",
    "CHALLENGER_POOLING_LAYOUTS",
    "CHALLENGER_SEARCH_POLICY_SCHEMA",
    "CHALLENGER_SEARCH_SPACES",
    "MODERN_TCN_LAYOUTS",
    "challenger_search_policy",
    "challenger_search_policy_sha256",
    "registered_challenger_search_spaces",
    "validate_challenger_parameters",
]
