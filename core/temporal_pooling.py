"""Length-agnostic temporal heads shared by contemporary challengers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def deterministic_adaptive_max_pool1d(
    x: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """PyTorch-equivalent adaptive max pooling with deterministic CUDA grad.

    ``torch.adaptive_max_pool1d`` dispatches its CUDA backward through an
    operation that PyTorch 2.7 marks non-deterministic.  Adaptive bins are
    nevertheless simple integer windows.  Taking ``Tensor.max`` over those
    exact windows reproduces the forward definition and has a deterministic
    CUDA backward under the campaign policy.
    """

    if x.ndim != 3 or x.size(-1) < 1:
        raise ValueError(
            "x must have shape (batch, channels, non-empty sequence_length)"
        )
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size < 1
    ):
        raise ValueError("output_size must be a positive integer")
    length = x.size(-1)
    pooled = []
    for index in range(output_size):
        start = (index * length) // output_size
        end = ((index + 1) * length + output_size - 1) // output_size
        pooled.append(x[..., start:end].max(dim=-1).values)
    return torch.stack(pooled, dim=-1)


class AdaptiveTemporalPooling1D(nn.Module):
    """Pool ``(B,C,N)`` features to fixed width without losing channel order.

    ``pyramid_bins=None`` (or an empty sequence) is global-average pooling.
    Otherwise, each positive bin count produces an adaptive max-pooled temporal
    level and all levels are concatenated.  This is the same mathematical head
    used by the incumbent's ``MultiRatePooling1D``, exposed here without an
    import cycle so ModernTCN and TSLANet can be compared under the same head.
    """

    def __init__(self, pyramid_bins: Sequence[int] | None = None) -> None:
        super().__init__()
        if pyramid_bins is None:
            bins: tuple[int, ...] = ()
        else:
            if isinstance(pyramid_bins, (str, bytes)):
                raise ValueError("pyramid_bins must be a sequence of integers")
            try:
                bins = tuple(pyramid_bins)
            except TypeError as exc:
                raise ValueError(
                    "pyramid_bins must be a sequence of integers"
                ) from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in bins
        ):
            raise ValueError("every pyramid bin count must be a positive integer")
        if len(set(bins)) != len(bins):
            raise ValueError("pyramid bin counts must be distinct")
        self.pyramid_bins = bins
        self.output_multiplier = sum(bins) if bins else 1
        self.mode = "temporal_pyramid_max" if bins else "global_average"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(0) < 1 or x.size(1) < 1 or x.size(2) < 1:
            raise ValueError(
                "x must have shape (non-empty batch, channels, sequence_length)"
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if not self.pyramid_bins:
            return x.mean(dim=-1)
        return torch.cat(
            [
                deterministic_adaptive_max_pool1d(x, bins).flatten(1)
                for bins in self.pyramid_bins
            ],
            dim=1,
        )


__all__ = [
    "AdaptiveTemporalPooling1D",
    "deterministic_adaptive_max_pool1d",
]
