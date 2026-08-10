"""PyTorch CNNs for the isolated Fernandes-2025 reconstruction/extension.

The reconstruction backbone deliberately ends in a literal ``Flatten`` and
dense layer.  It does not reuse the main campaign's global-average classifier.
The extension changes only that aggregation step to a registered adaptive
multi-rate pyramid while retaining the same convolution/dense parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SCHEMA = "f25-pytorch-cnn-v1"
MULTIRATE_LEVELS = (1, 2, 4)


class F25ModelError(ValueError):
    """Raised when an F25 model vector cannot define the frozen architecture."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise F25ModelError(f"{name} must be a positive integer")
    return value


def normalize_model_params(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise F25ModelError("F25 model params must be a dictionary")
    n_layers = _positive_int(params.get("n_conv_layers"), "n_conv_layers")
    if n_layers > 5:
        raise F25ModelError("F25 source search permits at most five CNN layers")
    filters = []
    kernels = []
    pools = []
    for index in range(n_layers):
        filters.append(_positive_int(params.get(f"filters_l{index}"), f"filters_l{index}"))
        kernels.append(_positive_int(params.get(f"kernel_l{index}"), f"kernel_l{index}"))
        pool = params.get(f"pool_l{index}")
        if not isinstance(pool, bool):
            raise F25ModelError(f"pool_l{index} must be bool")
        pools.append(pool)
    dense_units = _positive_int(params.get("dense_units"), "dense_units")
    batch_size = _positive_int(params.get("batch_size"), "batch_size")
    learning_rate = params.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 1.0e-5 <= float(learning_rate) <= 1.0e-2
    ):
        raise F25ModelError("learning_rate must lie in [1e-5, 1e-2]")
    return {
        "n_conv_layers": n_layers,
        "filters": tuple(filters),
        "kernels": tuple(kernels),
        "pools": tuple(pools),
        "dense_units": dense_units,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
    }


class _AdaptiveMultiRateFlatten(nn.Module):
    def __init__(self, levels: Sequence[int] = MULTIRATE_LEVELS):
        super().__init__()
        levels = tuple(_positive_int(value, "multi-rate level") for value in levels)
        if not levels or len(set(levels)) != len(levels):
            raise F25ModelError("multi-rate levels must be nonempty and distinct")
        self.levels = levels
        self.output_bins = sum(levels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[-1] < 1:
            raise F25ModelError("multi-rate input must be (batch, channels, time)")
        return torch.cat(
            [F.adaptive_max_pool1d(tensor, level).flatten(1) for level in self.levels],
            dim=1,
        )


class F25CNN(nn.Module):
    """Source CNN with flatten+dense or the one-change multi-rate extension."""

    def __init__(
        self,
        *,
        in_channels: int,
        input_samples: int,
        params: dict[str, Any],
        pooling: str = "flatten+dense",
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        input_samples = _positive_int(input_samples, "input_samples")
        n_classes = _positive_int(n_classes, "n_classes")
        normalized = normalize_model_params(params)
        if pooling not in {"flatten+dense", "multi-rate"}:
            raise F25ModelError("pooling must be flatten+dense or multi-rate")
        self.schema = MODEL_SCHEMA
        self.pooling_kind = pooling
        self.input_samples = input_samples
        self.normalized_params = normalized

        modules: list[nn.Module] = []
        current_channels = in_channels
        current_samples = input_samples
        for filters, kernel, use_pool in zip(
            normalized["filters"], normalized["kernels"], normalized["pools"]
        ):
            if kernel > current_samples:
                raise F25ModelError(
                    "convolution kernel exceeds the live temporal dimension"
                )
            # Keras Conv1D defaults to valid padding; retaining it is the
            # closest PyTorch reconstruction of the source implementation.
            modules.extend(
                [nn.Conv1d(current_channels, filters, kernel_size=kernel), nn.ReLU()]
            )
            current_samples = current_samples - kernel + 1
            if use_pool:
                if current_samples < 2:
                    raise F25ModelError("max pooling would erase the time dimension")
                modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_samples //= 2
            current_channels = filters
        self.convolutions = nn.Sequential(*modules)

        if pooling == "flatten+dense":
            self.aggregation: nn.Module = nn.Flatten(start_dim=1)
            flattened_units = current_channels * current_samples
        else:
            multirate = _AdaptiveMultiRateFlatten()
            self.aggregation = multirate
            flattened_units = current_channels * multirate.output_bins
        self.flattened_units = flattened_units
        self.dense = nn.Linear(flattened_units, normalized["dense_units"])
        self.activation = nn.ReLU()
        self.output = nn.Linear(normalized["dense_units"], n_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[-1] != self.input_samples:
            raise F25ModelError(
                f"F25 CNN expects (batch, channels, {self.input_samples})"
            )
        tensor = self.convolutions(tensor)
        tensor = self.aggregation(tensor)
        tensor = self.activation(self.dense(tensor))
        return self.output(tensor)


def build_f25_model(
    *,
    arm_id: str,
    in_channels: int,
    params: dict[str, Any],
    device: torch.device | str,
) -> F25CNN:
    if arm_id == "RAW-CNN":
        input_samples = 5830
        pooling = "flatten+dense"
    elif arm_id == "PAA-CNN":
        input_samples = 583
        pooling = "flatten+dense"
    elif arm_id == "PAA-multirate":
        input_samples = 583
        pooling = "multi-rate"
    else:
        raise F25ModelError(f"unknown F25 arm: {arm_id}")
    return F25CNN(
        in_channels=in_channels,
        input_samples=input_samples,
        params=params,
        pooling=pooling,
        n_classes=10,
    ).to(device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

