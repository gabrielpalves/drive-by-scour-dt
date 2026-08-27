"""Length-agnostic TSLANet-inspired encoder for multi-target regression.

This module adapts the two characteristic TSLANet components -- adaptive
spectral filtering and interactive convolution -- to the repository's
``(batch, channels, sequence_length)`` convention. The ASB -> ICB ordering,
pre-normalization, ICB algebra, residual stochastic depth, mean pooling, and
linear/LayerNorm initialization follow the pinned reference implementation.

The following intentional differences make the model suitable for this
repository:

* the last partial patch is zero-padded instead of discarded;
* a fixed-size learnable positional template is linearly resampled to the
  live patch count instead of allocating length-specific parameters; and
* the adaptive threshold is positive by construction and deterministically
  initialized, rather than being a directly optimized random scalar;
* the adaptive hard mask uses a sigmoid straight-through estimator (STE).
  Its forward values are binary, while its surrogate backward pass trains
  both the threshold and the spectral energy. The upstream STE only supplies
  a surrogate threshold gradient; and
* the native mean head can be replaced by the incumbent-compatible adaptive
  temporal-pyramid max head through an explicitly registered HPO choice.

Consequently, no parameter depends on the input length and one trained model
can consume both PAA and RAW sequences. These differences are part of the
model definition; this module must be described as ``TSLANet-inspired`` and
not as an exact reproduction of upstream TSLANet.

Reference implementation and paper:
    https://github.com/emadeldeen24/TSLANet
    commit ca0e88416d3ae49fd50e399c44ae94868378a94d
    https://proceedings.mlr.press/v235/eldele24a.html

The implementation is intentionally pure PyTorch and has no training or
plotting side effects. It is a regression adaptation trained from scratch;
the upstream self-supervised pretraining stage is not reproduced. Challenger
experiments use the repository-wide Adam plus cosine-annealing protocol rather
than upstream's AdamW recipe. This deliberately holds the optimization budget
constant across model families; it evaluates the architecture adaptation, not
the complete upstream training pipeline. See ``THIRD_PARTY_NOTICES.md``. No
upstream source file is vendored here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.temporal_pooling import AdaptiveTemporalPooling1D


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1)")
    return value


class DropPath(nn.Module):
    """Per-sample stochastic depth used on a complete residual branch.

    This is the dependency-free equivalent of the ``timm.DropPath`` operation
    used upstream. One Bernoulli decision is shared by every token and feature
    of a sample; ordinary elementwise dropout would define a different model.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = _probability("drop_prob", drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * keep_mask.div_(keep_prob)


class PatchEmbedding1D(nn.Module):
    """Overlapping Conv1d patch embedding with deterministic right padding.

    Unlike upstream, which drops an incomplete tail for a registered fixed
    length, this adaptation zero-pads the tail so every physical sample is
    represented. It also permits sequences shorter than ``patch_size``. No
    complete patch is made solely from padding.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 16,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        if stride is None:
            stride = max(1, self.patch_size // 2)
        self.stride = _positive_int("stride", stride)
        if self.stride > self.patch_size:
            raise ValueError("stride cannot exceed patch_size (patches must cover the signal)")

        self.projection = nn.Conv1d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
        )

    def right_padding(self, sequence_length: int) -> int:
        """Return padding required for a complete final patch."""
        sequence_length = _positive_int("sequence_length", sequence_length)
        if sequence_length < self.patch_size:
            return self.patch_size - sequence_length
        remainder = (sequence_length - self.patch_size) % self.stride
        return (self.stride - remainder) % self.stride

    def n_patches(self, sequence_length: int) -> int:
        padded = sequence_length + self.right_padding(sequence_length)
        return 1 + (padded - self.patch_size) // self.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, sequence_length)")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.size(1)}"
            )
        if x.size(-1) < 1:
            raise ValueError("sequence_length must be non-empty")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

        pad_right = self.right_padding(x.size(-1))
        if pad_right:
            x = F.pad(x, (0, pad_right))
        return self.projection(x).transpose(1, 2)  # (B, patches, embed_dim)


class AdaptiveSpectralBlock(nn.Module):
    """Learnable global spectral filter with an optional adaptive branch.

    FFT is applied along the patch axis.  A channel-wise complex filter models
    the full spectrum.  When ``adaptive_filter`` is enabled, a second filter is
    applied only to frequencies whose energy exceeds a learned, sample-wise
    median-normalized threshold. The hard mask uses a smooth straight-through
    estimator: its forward value is exactly binary, while the sigmoid
    surrogate in the backward pass trains the positive threshold and allows
    gradients through spectral energy. This is an intentional, explicitly
    documented departure from the upstream threshold-only surrogate.
    """

    def __init__(
        self,
        dim: int,
        adaptive_filter: bool = True,
        threshold_init: float = 1.0,
        mask_temperature: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        if not isinstance(adaptive_filter, bool):
            raise ValueError("adaptive_filter must be bool")
        if not math.isfinite(threshold_init) or threshold_init <= 0:
            raise ValueError("threshold_init must be finite and positive")
        if not math.isfinite(mask_temperature) or mask_temperature <= 0:
            raise ValueError("mask_temperature must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.adaptive_filter = adaptive_filter
        self.mask_temperature = float(mask_temperature)
        self.eps = float(eps)

        # Final dimension stores real and imaginary parts for view_as_complex.
        self.complex_weight = nn.Parameter(torch.empty(self.dim, 2))
        self.complex_weight_high = nn.Parameter(torch.empty(self.dim, 2))
        nn.init.trunc_normal_(self.complex_weight, std=0.02)
        nn.init.trunc_normal_(self.complex_weight_high, std=0.02)

        # Stable inverse softplus keeps the public initialization meaningful
        # even for unusually large, but finite, thresholds.
        inverse_softplus = (
            float(threshold_init)
            if threshold_init > 20.0
            else math.log(math.expm1(float(threshold_init)))
        )
        self.log_threshold = nn.Parameter(torch.tensor([inverse_softplus]))

    @property
    def threshold(self) -> torch.Tensor:
        return F.softplus(self.log_threshold) + self.eps

    def adaptive_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Return a ``(batch, frequencies, 1)`` hard-forward mask."""
        if not spectrum.is_complex() or spectrum.ndim != 3:
            raise ValueError("spectrum must be complex with shape (batch, freq, dim)")
        if spectrum.size(-1) != self.dim:
            raise ValueError(f"expected spectral feature dimension {self.dim}")

        energy = spectrum.abs().square().sum(dim=-1)
        # ``Tensor.median(dim=...)`` has no deterministic CUDA implementation
        # in the campaign's pinned PyTorch release. Sorting and selecting the
        # lower middle value is mathematically identical to torch.median for a
        # one-dimensional even/odd sample and remains deterministic on CUDA.
        ordered_energy = energy.sort(dim=1).values
        median_index = (energy.size(1) - 1) // 2
        median_energy = ordered_energy[
            :, median_index:median_index + 1
        ]
        normalized = energy / (median_energy + self.eps)
        threshold = self.threshold.to(device=normalized.device, dtype=normalized.dtype)

        hard = (normalized > threshold).to(normalized.dtype)
        soft = torch.sigmoid(
            (normalized - threshold) / self.mask_temperature
        )
        # Hard forward, sigmoid backward. Do not detach ``normalized`` here:
        # allowing the mask to shape spectral features is a deliberate part of
        # this inspired adaptation, not a claim of exact upstream fidelity.
        # This ordering also preserves the exact hard value numerically in the
        # forward pass (the algebraically equivalent hard + soft - detach(soft)
        # can retain cancellation round-off around zero).
        mask = soft + (hard - soft).detach()
        return mask.unsqueeze(-1)

    @staticmethod
    def _complex(parameter: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        parts = parameter.to(dtype=dtype).contiguous()
        return torch.view_as_complex(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim or x.size(1) < 1:
            raise ValueError(
                f"x must have shape (batch, non-empty patches, {self.dim})"
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

        original_dtype = x.dtype
        # CPU FFT does not support float16/bfloat16; float32 is also the stable
        # accumulation path used by the reference implementation.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.rfft(work, dim=1, norm="ortho")
        weight = self._complex(self.complex_weight, work.dtype)
        filtered = spectrum * weight.view(1, 1, -1)

        if self.adaptive_filter:
            mask = self.adaptive_mask(spectrum)
            high_weight = self._complex(self.complex_weight_high, work.dtype)
            filtered = filtered + spectrum * mask * high_weight.view(1, 1, -1)

        output = torch.fft.irfft(filtered, n=x.size(1), dim=1, norm="ortho")
        return output.to(dtype=original_dtype)


class InteractiveConvolutionBlock(nn.Module):
    """Gated interaction between pointwise and local temporal convolutions."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        dropout = _probability("dropout", dropout)

        self.dim = dim
        self.pointwise = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.temporal = nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1)
        self.projection = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"x must have shape (batch, patches, {self.dim})")
        channels_first = x.transpose(1, 2)
        pointwise = self.pointwise(channels_first)
        temporal = self.temporal(channels_first)
        pointwise_gate = self.dropout(self.activation(pointwise))
        temporal_gate = self.dropout(self.activation(temporal))
        mixed = pointwise * temporal_gate + temporal * pointwise_gate
        return self.projection(mixed).transpose(1, 2)


class TSLABlock(nn.Module):
    """Length-agnostic TSLANet layer with a residual update."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_asb: bool = True,
        use_icb: bool = True,
        adaptive_filter: bool = True,
        spectral_threshold_init: float = 1.0,
        spectral_mask_temperature: float = 0.1,
        spectral_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not isinstance(use_asb, bool) or not isinstance(use_icb, bool):
            raise ValueError("use_asb and use_icb must be bool")
        dropout = _probability("dropout", dropout)
        drop_path = _probability("drop_path", drop_path)

        self.use_asb = use_asb
        self.use_icb = use_icb
        self.norm_asb = nn.LayerNorm(dim) if use_asb else nn.Identity()
        self.asb = (
            AdaptiveSpectralBlock(
                dim,
                adaptive_filter=adaptive_filter,
                threshold_init=spectral_threshold_init,
                mask_temperature=spectral_mask_temperature,
                eps=spectral_eps,
            )
            if use_asb
            else nn.Identity()
        )
        self.norm_icb = nn.LayerNorm(dim) if use_icb else nn.Identity()
        self.icb = (
            InteractiveConvolutionBlock(
                dim,
                hidden_dim=max(1, int(dim * mlp_ratio)),
                dropout=dropout,
            )
            if use_icb
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_asb and not self.use_icb:
            return x
        update = x
        if self.use_asb:
            update = self.asb(self.norm_asb(update))
        if self.use_icb:
            update = self.icb(self.norm_icb(update))
        return x + self.drop_path(update)


class TSLANetRegressor(nn.Module):
    """Compact TSLANet-inspired multi-output regressor.

    Args:
        in_channels: Number of selected response channels.
        output_dim: Number of continuous regression targets.
        embed_dim: Patch/token feature width.
        depth: Number of TSLANet blocks.
        patch_size: Temporal samples per patch.
        patch_stride: Patch step; defaults to 50% overlap.
        mlp_ratio: ICB hidden-width multiplier (upstream default: 3).
        dropout: Input and ICB dropout probability (and hidden-head dropout
            when ``head_hidden_dim`` is set). Unless
            ``drop_path_rate`` is provided, it is also the maximum residual
            stochastic-depth rate, matching upstream's coupled schedule.
        drop_path_rate: Maximum residual stochastic-depth rate. Rates increase
            linearly from zero at the first block to this value at the last.
        use_asb: Enable the Adaptive Spectral Block.
        use_icb: Enable the Interactive Convolution Block.
        adaptive_filter: Enable ASB's thresholded second spectral branch.
        spectral_threshold_init: Positive initial ASB energy threshold.
        spectral_mask_temperature: Positive sigmoid-STE temperature.
        spectral_eps: Positive ASB normalization stabilizer.
        position_bins: Size of the learnable normalized-position template.
            It is linearly interpolated (``align_corners=False`` semantics) to
            the live patch count; ``None`` disables it. Upstream instead owns
            one positional tensor for one fixed input length.
        head_hidden_dim: Optional hidden width before the regression output.
        pool_bins: ``None`` selects the upstream-style global temporal mean.
            A tuple selects the incumbent-compatible adaptive temporal-pyramid
            max-pooling head. The encoder is unchanged in either case.

    Input shape is ``(B, in_channels, L)`` and output shape is
    ``(B, output_dim)`` for any positive ``L``.
    """

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        *,
        embed_dim: int = 64,
        depth: int = 2,
        patch_size: int = 16,
        patch_stride: int | None = None,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        drop_path_rate: float | None = None,
        use_asb: bool = True,
        use_icb: bool = True,
        adaptive_filter: bool = True,
        spectral_threshold_init: float = 1.0,
        spectral_mask_temperature: float = 0.1,
        spectral_eps: float = 1e-6,
        position_bins: int | None = 64,
        head_hidden_dim: int | None = None,
        pool_bins: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.output_dim = _positive_int("output_dim", output_dim)
        embed_dim = _positive_int("embed_dim", embed_dim)
        depth = _positive_int("depth", depth)
        dropout = _probability("dropout", dropout)
        if drop_path_rate is None:
            drop_path_rate = dropout
        else:
            drop_path_rate = _probability("drop_path_rate", drop_path_rate)
        if position_bins is not None:
            position_bins = _positive_int("position_bins", position_bins)
        if head_hidden_dim is not None:
            head_hidden_dim = _positive_int("head_hidden_dim", head_hidden_dim)

        self.patch_embedding = PatchEmbedding1D(
            self.in_channels,
            embed_dim,
            patch_size=patch_size,
            stride=patch_stride,
        )
        if position_bins is None:
            self.register_parameter("position_embedding", None)
        else:
            self.position_embedding = nn.Parameter(
                torch.empty(1, position_bins, embed_dim)
            )
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.input_dropout = nn.Dropout(dropout)
        if depth == 1:
            drop_path_rates = [0.0]
        else:
            drop_path_rates = [
                drop_path_rate * index / (depth - 1)
                for index in range(depth)
            ]
        self.blocks = nn.ModuleList(
            TSLABlock(
                embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_rates[index],
                use_asb=use_asb,
                use_icb=use_icb,
                adaptive_filter=adaptive_filter,
                spectral_threshold_init=spectral_threshold_init,
                spectral_mask_temperature=spectral_mask_temperature,
                spectral_eps=spectral_eps,
            )
            for index in range(depth)
        )
        self.pool = AdaptiveTemporalPooling1D(pool_bins)
        head_input_dim = embed_dim * self.pool.output_multiplier

        if head_hidden_dim is None:
            # The registered challenger uses the same direct linear head as
            # upstream, with output width adapted from classes to targets.
            self.head = nn.Linear(head_input_dim, self.output_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, self.output_dim),
            )

        # Match upstream for the module types it initializes explicitly.
        # Conv1d layers retain PyTorch defaults, as they do in the reference.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _position_tokens(self, n_patches: int) -> torch.Tensor | None:
        if self.position_embedding is None:
            return None
        if self.position_embedding.size(1) == n_patches:
            return self.position_embedding
        if self.position_embedding.size(1) == 1:
            return self.position_embedding.expand(-1, n_patches, -1)

        # ``upsample_linear1d_backward`` is not deterministic on CUDA in the
        # pinned PyTorch release. Build the same align_corners=False linear
        # interpolation as a fixed matrix multiplication instead. Its only
        # trainable operand is the position template, so cuBLAS follows the
        # campaign's deterministic workspace contract for both passes.
        source_bins = self.position_embedding.size(1)
        dtype = self.position_embedding.dtype
        device = self.position_embedding.device
        output_index = torch.arange(n_patches, device=device, dtype=dtype)
        source_position = (
            (output_index + 0.5) * (source_bins / n_patches) - 0.5
        ).clamp(0.0, float(source_bins - 1))
        left_index = source_position.floor().to(torch.long)
        right_index = (left_index + 1).clamp(max=source_bins - 1)
        right_weight = source_position - left_index.to(dtype)
        left_weight = 1.0 - right_weight
        interpolation = (
            F.one_hot(left_index, num_classes=source_bins).to(dtype)
            * left_weight.unsqueeze(1)
            + F.one_hot(right_index, num_classes=source_bins).to(dtype)
            * right_weight.unsqueeze(1)
        )
        position = interpolation @ self.position_embedding.squeeze(0)
        return position.unsqueeze(0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(x)
        position = self._position_tokens(tokens.size(1))
        if position is not None:
            tokens = tokens + position.to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        # Upstream pools block outputs directly; there is deliberately no
        # additional terminal LayerNorm in this adaptation.
        return self.pool(tokens.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


__all__ = [
    "DropPath",
    "PatchEmbedding1D",
    "AdaptiveSpectralBlock",
    "InteractiveConvolutionBlock",
    "TSLABlock",
    "TSLANetRegressor",
]
