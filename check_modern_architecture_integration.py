"""Integrated contract checks for the ModernTCN and TSLANet challengers.

Run:  python check_modern_architecture_integration.py

The challengers deliberately remain outside the registered 16-cell factorial.
This check closes the path that is already useful before a pilot campaign:
registered HPO domain -> frozen singleton -> universal model factory ->
variable-length multi-head forward/backward.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from core.challenger_policy import (
    CHALLENGER_CAPACITY_SCOPE,
    CHALLENGER_FACTORY_POLICY,
    CHALLENGER_PATCH_LAYOUTS,
    CHALLENGER_POOLING_LAYOUTS,
    CHALLENGER_SEARCH_SPACES,
    MODERN_TCN_LAYOUTS,
    challenger_search_policy,
    challenger_search_policy_sha256,
    validate_challenger_parameters,
)
from core.models import MultiRatePooling1D, build_model
from core.modern_tcn import ModernTCNRegressor
from core.paper1_training_contract import FACTORIAL_CELLS
from core.temporal_pooling import (
    AdaptiveTemporalPooling1D,
    deterministic_adaptive_max_pool1d,
)
from core.tslanet import TSLANetRegressor
from training.trainer import _suggest_params


EXPECTED_FAMILY_SPACE = {
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
            "cat",
            ["8_4", "16_8", "32_16", "64_32"],
        ),
        "modern_kernel_size": ("cat", [15, 31, 63]),
        "modern_expansion_ratio": ("cat", [2.0, 3.0, 4.0]),
        "modern_dropout": ("float", 0.0, 0.3),
        "challenger_pooling": (
            "cat",
            ["gap", "1_2_4", "1_4_8", "1_3_6", "1_2_4_8"],
        ),
    },
    "TSLANET": {
        "tsla_embed_dim": ("int_step", 32, 96, 32),
        "tsla_depth": ("int", 1, 3),
        "tsla_patch_layout": (
            "cat",
            ["8_4", "16_8", "32_16", "64_32"],
        ),
        "tsla_mlp_ratio": ("cat", [1.5, 2.0, 3.0]),
        "tsla_dropout": ("float", 0.0, 0.3),
        "challenger_pooling": (
            "cat",
            ["gap", "1_2_4", "1_4_8", "1_3_6", "1_2_4_8"],
        ),
    },
}


class RecordingTrial:
    """Small Optuna-compatible recorder with optional prescribed values."""

    def __init__(self, prescribed: dict | None = None) -> None:
        self.prescribed = prescribed or {}
        self.calls: list[tuple] = []

    def suggest_int(self, name, low, high, step=None):
        self.calls.append((name, "int", low, high, step))
        return self.prescribed.get(name, low)

    def suggest_float(self, name, low, high, log=False):
        self.calls.append((name, "float", low, high, log))
        return self.prescribed.get(name, low)

    def suggest_categorical(self, name, choices):
        choices = tuple(choices)
        self.calls.append((name, "cat", choices))
        return self.prescribed.get(name, choices[0])


def rejects(label: str, function) -> None:
    try:
        function()
    except (KeyError, TypeError, ValueError):
        print(f"  [PASS] {label}")
        return
    raise AssertionError(f"guard did not fire: {label}")


def expected_names(model_type: str) -> list[str]:
    return [
        "lr",
        "weight_decay",
        *CHALLENGER_SEARCH_SPACES[model_type],
    ]


def structural_combinations(space: dict) -> int:
    count = 1
    for spec in space.values():
        if spec[0] == "cat":
            count *= len(spec[1])
        elif spec[0] == "int":
            count *= spec[2] - spec[1] + 1
        elif spec[0] == "int_step":
            count *= (spec[2] - spec[1]) // spec[3] + 1
    return count


def exercise_family(model_type: str, expected_class: type[torch.nn.Module]) -> None:
    sampled_trial = RecordingTrial()
    config = {
        "model_type": model_type,
        "method": "PAA",
        "task": "regression",
        "target_supports": [2],
        "bearing_targets": [],
        "use_space2vec": False,
        "use_lstm": False,
        "use_nhits": False,
    }
    params = _suggest_params(sampled_trial, config)
    names = [call[0] for call in sampled_trial.calls]
    assert names == expected_names(model_type)
    assert list(params) == names
    assert not {
        "n_conv_layers",
        "n_dense_layers",
        "nhits_pool_rates_key",
    }.intersection(params)

    # A frozen candidate must register the exact same keyset as singleton
    # Optuna domains. This is the path later used for repeated refits.
    frozen_trial = RecordingTrial()
    frozen = _suggest_params(
        frozen_trial,
        {**config, "frozen_hyperparameters": params},
    )
    assert frozen == params
    assert [call[0] for call in frozen_trial.calls] == expected_names(model_type)

    model, n_outputs = build_model(
        config,
        params,
        (2, 8, 512),
        torch.device("cpu"),
    )
    assert isinstance(model, expected_class)
    assert n_outputs == 1
    x_paa = torch.randn(2, 8, 512, requires_grad=True)
    y_paa = model(x_paa)
    y_paa.square().mean().backward()
    assert y_paa.shape == (2, 1)
    assert bool(torch.isfinite(y_paa).all())
    assert x_paa.grad is not None and bool(torch.isfinite(x_paa.grad).all())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    optimizer.step()

    # Run the same encoder family with the incumbent-compatible head.  The
    # pooling label is part of the frozen HPO identity, not a silent factory
    # default, and must remain length agnostic.
    pyramid_params = {**params, "challenger_pooling": "1_2_4"}
    pyramid_model, _ = build_model(
        config,
        pyramid_params,
        (2, 8, 257),
        torch.device("cpu"),
    )
    pyramid_output = pyramid_model(torch.randn(2, 8, 257))
    assert pyramid_output.shape == (2, 1)
    assert pyramid_model.pool.output_multiplier == 7
    print(f"  [PASS] {model_type}: shared temporal-pyramid head")

    # Simulate the artifact's JSON metadata + state_dict reconstruction path.
    metadata = json.loads(json.dumps({
        "architecture_flags": {"model_type": model_type},
        "optimal_hyperparameters": params,
    }))
    rebuild_config = {
        **config,
        "model_type": metadata["architecture_flags"]["model_type"],
    }
    rebuilt, _ = build_model(
        rebuild_config,
        metadata["optimal_hyperparameters"],
        (2, 8, 512),
        torch.device("cpu"),
    )
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    rebuilt.eval()
    reload_probe = torch.randn(2, 8, 512)
    with torch.no_grad():
        assert torch.equal(model(reload_probe), rebuilt(reload_probe))

    if model_type == "MODERN_TCN":
        model.structural_reparam()
        deploy_rebuilt, _ = build_model(
            rebuild_config,
            metadata["optimal_hyperparameters"],
            (2, 8, 512),
            torch.device("cpu"),
        )
        deploy_rebuilt.eval()
        deploy_rebuilt.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert torch.equal(model(reload_probe), deploy_rebuilt(reload_probe))
        print("  [PASS] MODERN_TCN: factory reloads fused deploy checkpoint")

    # The same parameterization must accept the exact L99 RAW length and the
    # five heads used by the critical multi-damage/bearing scenario.
    critical_config = {
        **config,
        "method": "RAW",
        "target_supports": [2, 3, 4],
        "bearing_targets": ["left", "right"],
    }
    critical, n_outputs = build_model(
        critical_config,
        params,
        (1, 8, 11791),
        torch.device("cpu"),
    )
    critical.eval()
    with torch.no_grad():
        y_raw = critical(torch.randn(1, 8, 11791))
    assert n_outputs == 5
    assert y_raw.shape == (1, 5)
    assert bool(torch.isfinite(y_raw).all())
    print(
        f"  [PASS] {model_type}: HPO/freeze/train/rebuild/PAA/L99-RAW"
    )


def main() -> None:
    torch.manual_seed(2026)
    torch.set_num_threads(1)

    family_space = CHALLENGER_SEARCH_SPACES
    assert family_space == EXPECTED_FAMILY_SPACE
    assert set(family_space["MODERN_TCN"]["modern_layout"][1]) == set(
        MODERN_TCN_LAYOUTS
    )
    assert set(family_space["MODERN_TCN"]["modern_patch_layout"][1]) == set(
        CHALLENGER_PATCH_LAYOUTS
    )
    assert set(family_space["TSLANET"]["tsla_patch_layout"][1]) == set(
        CHALLENGER_PATCH_LAYOUTS
    )
    print("  [PASS] search-space categorical registries match the factory")
    policy = challenger_search_policy()
    assert policy["patch_layouts"] == json.loads(
        json.dumps(CHALLENGER_PATCH_LAYOUTS)
    )
    assert policy["modern_tcn_layouts"] == json.loads(
        json.dumps(MODERN_TCN_LAYOUTS)
    )
    assert policy["factory"] == CHALLENGER_FACTORY_POLICY
    assert policy["capacity_scope"] == CHALLENGER_CAPACITY_SCOPE
    original_patch = CHALLENGER_PATCH_LAYOUTS["8_4"]
    original_sha = challenger_search_policy_sha256()
    try:
        CHALLENGER_PATCH_LAYOUTS["8_4"] = (7, 3)
        assert challenger_search_policy_sha256() != original_sha
    finally:
        CHALLENGER_PATCH_LAYOUTS["8_4"] = original_patch
    assert challenger_search_policy_sha256() == original_sha
    original_temperature = CHALLENGER_FACTORY_POLICY["TSLANET"][
        "spectral_mask_temperature"
    ]
    try:
        CHALLENGER_FACTORY_POLICY["TSLANET"][
            "spectral_mask_temperature"
        ] = 0.2
        assert challenger_search_policy_sha256() != original_sha
    finally:
        CHALLENGER_FACTORY_POLICY["TSLANET"][
            "spectral_mask_temperature"
        ] = original_temperature
    assert challenger_search_policy_sha256() == original_sha
    original_input_channels = CHALLENGER_CAPACITY_SCOPE["input_channels"]
    try:
        CHALLENGER_CAPACITY_SCOPE["input_channels"] = 8
        assert challenger_search_policy_sha256() != original_sha
    finally:
        CHALLENGER_CAPACITY_SCOPE["input_channels"] = original_input_channels
    assert challenger_search_policy_sha256() == original_sha
    print(
        "  [PASS] policy hash covers decoders, factory knobs, and capacity scope"
    )
    assert structural_combinations(family_space["MODERN_TCN"]) == 540
    assert structural_combinations(family_space["TSLANET"]) == 540
    print("  [PASS] both challengers register 108 encoder x 5 head choices")

    probe = torch.tensor(
        [[[1.0, 3.0, 2.0, 4.0], [-1.0, 2.0, 0.0, 5.0]]]
    )
    incumbent_pool = MultiRatePooling1D((1, 2, 4))
    challenger_pool = AdaptiveTemporalPooling1D((1, 2, 4))
    assert torch.equal(incumbent_pool(probe), challenger_pool(probe))
    assert tuple(CHALLENGER_POOLING_LAYOUTS["1_2_4"]) == (1, 2, 4)
    print("  [PASS] challenger temporal pyramid equals incumbent pooling")
    for length in (1, 2, 7, 17):
        source = torch.randn(2, 3, length)
        for output_bins in (1, 2, 4, 8):
            assert torch.equal(
                deterministic_adaptive_max_pool1d(source, output_bins),
                F.adaptive_max_pool1d(source, output_bins),
            )
    print("  [PASS] deterministic pooling matches PyTorch adaptive windows")

    factorial_ids = {cell.cell_id for cell in FACTORIAL_CELLS}
    assert len(factorial_ids) == 16
    assert not {"MODERN_TCN", "TSLANET"}.intersection(factorial_ids)
    print("  [PASS] challengers remain outside the 16-cell factorial")

    exercise_family("MODERN_TCN", ModernTCNRegressor)
    exercise_family("TSLANET", TSLANetRegressor)

    modern = _suggest_params(RecordingTrial(), {"model_type": "MODERN_TCN"})
    rejects(
        "frozen challenger rejects extra keys",
        lambda: _suggest_params(
            RecordingTrial(),
            {
                "model_type": "MODERN_TCN",
                "frozen_hyperparameters": {**modern, "foreign": 1},
            },
        ),
    )
    rejects(
        "challenger rejects legacy architecture flags",
        lambda: _suggest_params(
            RecordingTrial(),
            {"model_type": "TSLANET", "use_nhits": True},
        ),
    )
    rejects(
        "challenger factory rejects active legacy architecture flags",
        lambda: build_model(
            {
                "model_type": "MODERN_TCN",
                "method": "RAW",
                "task": "regression",
                "target_supports": [2],
                "use_nhits": True,
            },
            modern,
            (1, 8, 512),
            torch.device("cpu"),
        ),
    )
    rejects(
        "challenger HPO rejects the 2-D CWT representation",
        lambda: _suggest_params(
            RecordingTrial(),
            {"model_type": "TSLANET", "method": "PAA_CWT"},
        ),
    )
    rejects(
        "challenger factory rejects the 2-D CWT representation",
        lambda: build_model(
            {
                "model_type": "MODERN_TCN",
                "method": "PAA_CWT",
                "task": "regression",
                "target_supports": [2],
            },
            modern,
            (1, 8, 32, 512),
            torch.device("cpu"),
        ),
    )
    invalid_tsla = _suggest_params(
        RecordingTrial(), {"model_type": "TSLANET"}
    )
    invalid_tsla["tsla_embed_dim"] = "96"
    rejects(
        "factory rejects permissive numeric-string coercion",
        lambda: build_model(
            {
                "model_type": "TSLANET",
                "method": "RAW",
                "task": "regression",
                "target_supports": [2],
            },
            invalid_tsla,
            (1, 8, 512),
            torch.device("cpu"),
        ),
    )
    for key, invalid_value in (
        ("modern_kernel_size", 999),
        ("modern_expansion_ratio", 99.0),
        ("modern_dropout", 0.9),
    ):
        rejects(
            f"factory rejects out-of-domain {key}",
            lambda key=key, invalid_value=invalid_value: build_model(
                {
                    "model_type": "MODERN_TCN",
                    "method": "RAW",
                    "task": "regression",
                    "target_supports": [2],
                },
                {**modern, key: invalid_value},
                (1, 8, 512),
                torch.device("cpu"),
            ),
        )
    valid_tsla = _suggest_params(RecordingTrial(), {"model_type": "TSLANET"})
    numpy_typed_tsla = {
        **valid_tsla,
        "tsla_embed_dim": np.int64(valid_tsla["tsla_embed_dim"]),
        "tsla_depth": np.int64(valid_tsla["tsla_depth"]),
        "tsla_mlp_ratio": np.float64(valid_tsla["tsla_mlp_ratio"]),
    }
    normalized_tsla = validate_challenger_parameters(
        "TSLANET", numpy_typed_tsla
    )
    assert type(normalized_tsla["tsla_embed_dim"]) is int
    assert type(normalized_tsla["tsla_depth"]) is int
    assert type(normalized_tsla["tsla_mlp_ratio"]) is float
    numpy_model, _ = build_model(
        {
            "model_type": "TSLANET",
            "method": "PAA",
            "task": "regression",
            "target_supports": [2],
        },
        numpy_typed_tsla,
        (1, 8, 64),
        torch.device("cpu"),
    )
    assert numpy_model.blocks[0].asb.mask_temperature == (
        CHALLENGER_FACTORY_POLICY["TSLANET"]["spectral_mask_temperature"]
    )
    assert numpy_model.blocks[0].asb.eps == (
        CHALLENGER_FACTORY_POLICY["TSLANET"]["spectral_eps"]
    )
    print("  [PASS] factory canonicalizes NumPy scalars and pins ASB constants")
    for key, invalid_value in (
        ("tsla_embed_dim", 33),
        ("tsla_depth", 4),
        ("tsla_mlp_ratio", 9.0),
        ("tsla_dropout", 0.9),
    ):
        rejects(
            f"factory rejects out-of-domain {key}",
            lambda key=key, invalid_value=invalid_value: build_model(
                {
                    "model_type": "TSLANET",
                    "method": "RAW",
                    "task": "regression",
                    "target_supports": [2],
                },
                {**valid_tsla, key: invalid_value},
                (1, 8, 512),
                torch.device("cpu"),
            ),
        )
    rejects(
        "challenger factory rejects extra parameter keys",
        lambda: build_model(
            {
                "model_type": "TSLANET",
                "method": "RAW",
                "task": "regression",
                "target_supports": [2],
            },
            {**valid_tsla, "foreign": 1},
            (1, 8, 512),
            torch.device("cpu"),
        ),
    )
    rejects(
        "HPO rejects an unknown model family",
        lambda: _suggest_params(RecordingTrial(), {"model_type": "TYPO"}),
    )
    rejects(
        "factory rejects an unknown model family",
        lambda: build_model(
            {
                "model_type": "TYPO",
                "task": "regression",
                "target_supports": [2],
            },
            {},
            (1, 8, 512),
            torch.device("cpu"),
        ),
    )
    print("MODERN ARCHITECTURE INTEGRATION: ALL PASS")


if __name__ == "__main__":
    main()
