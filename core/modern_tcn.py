"""ModernTCN adaptation for length-agnostic multi-target regression.

The backbone follows the defining tensor organization of ModernTCN: input
variables are preserved explicitly and every stage carries ``(B, M, D, N)``
(``M`` variables, ``D`` embedding channels, ``N`` temporal patches).  The
patch stem is shared but applied independently to every variable.  Each block
then combines:

* a reparameterizable large- plus small-kernel depthwise temporal convolution
  with ``groups=M*D``;
* ConvFFN1, grouped by ``M``, which mixes embedding channels independently
  inside each variable; and
* ConvFFN2, applied after ``M`` and ``D`` are permuted and grouped by ``D``,
  which mixes variables independently inside each embedding channel.

The upstream classification head is sequence-length dependent.  The local
regression adaptation applies GELU and either temporal averaging or the
incumbent-compatible temporal-pyramid max head, then concatenates the
still-distinct ``M`` variable embeddings before a linear multi-output head.
Consequently, parameters depend on the known sensor count but not on the
RAW/PAA sequence length.

Upstream design reference:
    https://github.com/luodhhh/ModernTCN
    commit 56a9a2c018385cd5acef015378cae7f084d1b11c

This is a repository-local adaptation, not a verbatim upstream source file.
See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.temporal_pooling import AdaptiveTemporalPooling1D


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dropout_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dropout must be a real number in [0, 1)")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be a real number in [0, 1)")
    return probability


def _integer_expansion_ratio(value: object) -> int:
    """Validate ModernTCN's integer ConvFFN expansion ratio."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("expansion_ratio must be an integer-valued real >= 1")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 1.0 or not ratio.is_integer():
        raise ValueError("expansion_ratio must be an integer-valued real >= 1")
    return int(ratio)


def _stage_positive_ints(
    name: str,
    value: int | Sequence[int],
    n_stages: int,
) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or one per stage")
    if isinstance(value, int):
        return (_positive_int(name, value),) * n_stages
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a positive integer or one per stage")
    values = tuple(
        _positive_int(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(values) != n_stages:
        raise ValueError(f"{name} must provide one value per stage")
    return values


def _stage_optional_positive_ints(
    name: str,
    value: int | Sequence[int | None] | None,
    n_stages: int,
) -> tuple[int | None, ...]:
    if value is None:
        return (None,) * n_stages
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be None, a positive integer, or one value per stage"
        )
    if isinstance(value, int):
        return (_positive_int(name, value),) * n_stages
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{name} must be None, a positive integer, or one value per stage"
        )
    values: list[int | None] = []
    for index, item in enumerate(value):
        values.append(
            None if item is None else _positive_int(f"{name}[{index}]", item)
        )
    if len(values) != n_stages:
        raise ValueError(f"{name} must provide one value per stage")
    return tuple(values)


def _same_padding_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply stride-one SAME padding for both odd and even kernels."""

    total = kernel_size - 1
    left = total // 2
    return F.pad(x, (left, total - left))


def _right_pad_to_ceil(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> torch.Tensor:
    """Replicate-pad so a strided valid convolution returns ``ceil(N/S)``.

    ModernTCN extends the right edge before overlapping patch embedding and
    before an incomplete downsampling step.  Computing the required extent
    makes that policy valid for arbitrary RAW/PAA lengths, including signals
    shorter than one patch.
    """

    length = x.size(-1)
    output_length = (length + stride - 1) // stride
    required_length = (output_length - 1) * stride + kernel_size
    right = max(0, required_length - length)
    return F.pad(x, (0, right), mode="replicate") if right else x


def _fuse_conv_bn(
    conv: nn.Conv1d,
    batch_norm: nn.BatchNorm1d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the inference-equivalent convolution kernel and bias."""

    if batch_norm.running_mean is None or batch_norm.running_var is None:
        raise RuntimeError("reparameterization requires tracked BatchNorm state")
    weight = conv.weight
    conv_bias = (
        conv.bias
        if conv.bias is not None
        else torch.zeros(
            conv.out_channels,
            dtype=weight.dtype,
            device=weight.device,
        )
    )
    if batch_norm.affine:
        assert batch_norm.weight is not None and batch_norm.bias is not None
        scale = batch_norm.weight / torch.sqrt(
            batch_norm.running_var + batch_norm.eps
        )
        bias = batch_norm.bias + (conv_bias - batch_norm.running_mean) * scale
    else:
        scale = torch.rsqrt(batch_norm.running_var + batch_norm.eps)
        bias = (conv_bias - batch_norm.running_mean) * scale
    return weight * scale.reshape(-1, 1, 1), bias


def _depthwise_conv1d(
    x: torch.Tensor,
    conv: nn.Conv1d,
    channel_chunk_size: int | None,
) -> torch.Tensor:
    """Execute one depthwise Conv1d in independent channel chunks.

    Grouped depthwise channels have no cross-channel reduction. Splitting only
    the execution batch therefore leaves the operator and its parameters
    unchanged while preventing cuDNN from reserving a very large workspace for
    long RAW tensors.
    """

    if channel_chunk_size is None or x.size(1) <= channel_chunk_size:
        return conv(x)
    if conv.groups != conv.in_channels or conv.out_channels != conv.in_channels:
        raise RuntimeError("channel chunking is defined only for depthwise Conv1d")

    outputs: list[torch.Tensor] = []
    for start in range(0, conv.in_channels, channel_chunk_size):
        stop = min(start + channel_chunk_size, conv.in_channels)
        bias = None if conv.bias is None else conv.bias[start:stop]
        outputs.append(
            F.conv1d(
                x[:, start:stop],
                conv.weight[start:stop],
                bias,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=stop - start,
            )
        )
    return torch.cat(outputs, dim=1)


class _ConvBNBranch(nn.Module):
    """One externally SAME-padded Conv1d + BatchNorm training branch."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        channel_chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.channel_chunk_size = channel_chunk_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)

    def convolution(self, x: torch.Tensor) -> torch.Tensor:
        x = _same_padding_1d(x, self.kernel_size)
        return _depthwise_conv1d(x, self.conv, self.channel_chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.convolution(x))


class _ReparamLargeKernelConv(nn.Module):
    """ModernTCN large/small depthwise branches with deploy-time fusion."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int,
        small_kernel: int | None,
        *,
        small_kernel_merged: bool = False,
        channel_chunk_size: int | None = 64,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.small_kernel = small_kernel
        self.channel_chunk_size = channel_chunk_size
        self.activation_checkpointing = activation_checkpointing
        if small_kernel is not None and small_kernel > kernel_size:
            raise ValueError("small kernel cannot exceed the large kernel")

        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = _ConvBNBranch(
                channels,
                kernel_size,
                groups,
                channel_chunk_size,
            )
            if small_kernel is not None:
                self.small_conv = _ConvBNBranch(
                    channels,
                    small_kernel,
                    groups,
                    channel_chunk_size,
                )

    @property
    def is_reparameterized(self) -> bool:
        return hasattr(self, "lkb_reparam")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_reparameterized:
            x = _same_padding_1d(x, self.kernel_size)
            return _depthwise_conv1d(
                x,
                self.lkb_reparam,
                self.channel_chunk_size,
            )

        def run_branch(branch: _ConvBNBranch) -> torch.Tensor:
            # Only the convolution is recomputed. Its following BatchNorm is
            # deliberately outside the checkpoint and updates running state
            # exactly once per optimizer forward.
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and x.requires_grad
            ):
                convolved = checkpoint(
                    branch.convolution,
                    x,
                    use_reentrant=True,
                    preserve_rng_state=True,
                )
            else:
                convolved = branch.convolution(x)
            return branch.bn(convolved)

        output = run_branch(self.lkb_origin)
        if hasattr(self, "small_conv"):
            output = output + run_branch(self.small_conv)
        return output

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse both inference branches without modifying this module."""

        if self.is_reparameterized:
            return self.lkb_reparam.weight, self.lkb_reparam.bias
        if self.training:
            raise RuntimeError(
                "call eval() before fusing BatchNorm-based large kernels"
            )
        kernel, bias = _fuse_conv_bn(
            self.lkb_origin.conv,
            self.lkb_origin.bn,
        )
        if hasattr(self, "small_conv"):
            small_kernel, small_bias = _fuse_conv_bn(
                self.small_conv.conv,
                self.small_conv.bn,
            )
            # Align the two cross-correlation kernels according to their
            # explicit SAME-padding origins. This also handles mixed parity.
            large_left = (self.kernel_size - 1) // 2
            small_left = (self.small_kernel - 1) // 2
            kernel_left = large_left - small_left
            kernel_right = self.kernel_size - self.small_kernel - kernel_left
            kernel = kernel + F.pad(
                small_kernel,
                (kernel_left, kernel_right),
            )
            bias = bias + small_bias
        return kernel, bias

    def merge_kernel(self) -> None:
        """Replace training branches with one inference-equivalent Conv1d."""

        if self.is_reparameterized:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.prepare_deploy_structure()
        with torch.no_grad():
            self.lkb_reparam.weight.copy_(kernel)
            self.lkb_reparam.bias.copy_(bias)

    def prepare_deploy_structure(self) -> None:
        """Create the fused-branch schema without requiring source weights.

        This is used when loading a deploy-form checkpoint into a fresh model.
        The temporary initialization is isolated from caller RNG state and is
        immediately overwritten by ``load_state_dict``.
        """

        if self.is_reparameterized:
            return
        reference = self.lkb_origin.conv.weight
        cuda_devices = (
            [reference.device.index]
            if reference.is_cuda and reference.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            merged = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=True,
                device=reference.device,
                dtype=reference.dtype,
            )
        self.lkb_reparam = merged
        del self.lkb_origin
        if hasattr(self, "small_conv"):
            del self.small_conv


class _ModernTCNBlock(nn.Module):
    """One faithful ``(B,M,D,N)`` ModernTCN backbone block."""

    def __init__(
        self,
        nvars: int,
        dmodel: int,
        large_size: int,
        small_size: int | None,
        ffn_ratio: int,
        dropout: float,
        *,
        small_kernel_merged: bool = False,
        activation_checkpointing: bool = True,
        depthwise_channel_chunk_size: int | None = 64,
    ) -> None:
        super().__init__()
        self.nvars = nvars
        self.dmodel = dmodel
        self.dff = dmodel * ffn_ratio
        self.activation_checkpointing = activation_checkpointing

        self.dw = _ReparamLargeKernelConv(
            channels=nvars * dmodel,
            kernel_size=large_size,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            channel_chunk_size=depthwise_channel_chunk_size,
            activation_checkpointing=activation_checkpointing,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN1: each of M groups mixes D -> D_ff independently.
        self.ffn1pw1 = nn.Conv1d(
            nvars * dmodel,
            nvars * self.dff,
            kernel_size=1,
            groups=nvars,
        )
        self.ffn1act = nn.GELU()
        self.ffn1drop1 = nn.Dropout(dropout)
        self.ffn1pw2 = nn.Conv1d(
            nvars * self.dff,
            nvars * dmodel,
            kernel_size=1,
            groups=nvars,
        )
        self.ffn1drop2 = nn.Dropout(dropout)

        # ConvFFN2: after permuting M and D, each of D groups mixes variables.
        self.ffn2pw1 = nn.Conv1d(
            dmodel * nvars,
            dmodel * nvars * ffn_ratio,
            kernel_size=1,
            groups=dmodel,
        )
        self.ffn2act = nn.GELU()
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2pw2 = nn.Conv1d(
            dmodel * nvars * ffn_ratio,
            dmodel * nvars,
            kernel_size=1,
            groups=dmodel,
        )
        self.ffn2drop2 = nn.Dropout(dropout)

    def _conv_ffns(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two grouped FFNs to one explicit ``(B,M,D,N)`` tensor."""

        batch, nvars, dmodel, length = x.shape
        x = x.reshape(batch, nvars * dmodel, length)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))

        x = x.reshape(batch, nvars, dmodel, length)
        x = x.permute(0, 2, 1, 3).reshape(
            batch,
            dmodel * nvars,
            length,
        )
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        return x.reshape(batch, dmodel, nvars, length).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, nvars, dmodel, length = x.shape
        if nvars != self.nvars or dmodel != self.dmodel:
            raise ValueError(
                "block received incompatible (M,D): "
                f"expected ({self.nvars},{self.dmodel}), "
                f"got ({nvars},{dmodel})"
            )

        x = self.dw(x.reshape(batch, nvars * dmodel, length))
        x = self.norm(x.reshape(batch * nvars, dmodel, length))
        x = x.reshape(batch, nvars, dmodel, length)

        # The expanded FFN tensors dominate memory for long RAW batches.
        # Checkpoint only this BN-free region: recomputation therefore cannot
        # update running statistics twice, while the mathematical block and
        # dropout RNG stream are preserved exactly.
        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            x = checkpoint(
                self._conv_ffns,
                x,
                use_reentrant=True,
                preserve_rng_state=True,
            )
        else:
            x = self._conv_ffns(x)
        return residual + x


class _ModernTCNStage(nn.Module):
    def __init__(
        self,
        nvars: int,
        dmodel: int,
        depth: int,
        large_size: int,
        small_size: int | None,
        ffn_ratio: int,
        dropout: float,
        *,
        small_kernel_merged: bool = False,
        activation_checkpointing: bool = True,
        depthwise_channel_chunk_size: int | None = 64,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _ModernTCNBlock(
                    nvars=nvars,
                    dmodel=dmodel,
                    large_size=large_size,
                    small_size=small_size,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    small_kernel_merged=small_kernel_merged,
                    activation_checkpointing=activation_checkpointing,
                    depthwise_channel_chunk_size=depthwise_channel_chunk_size,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _StageTransition(nn.Module):
    """Shared per-variable stage downsampling that preserves the M axis."""

    def __init__(
        self,
        nvars: int,
        in_features: int,
        out_features: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.nvars = nvars
        self.in_features = in_features
        self.out_features = out_features
        self.stride = stride
        self.norm = nn.BatchNorm1d(in_features)
        self.projection = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=stride,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nvars, features, length = x.shape
        if nvars != self.nvars or features != self.in_features:
            raise ValueError("stage transition received incompatible (M,D)")
        x = x.reshape(batch * nvars, features, length)
        x = self.norm(x)
        x = _right_pad_to_ceil(x, self.stride, self.stride)
        x = self.projection(x)
        return x.reshape(batch, nvars, self.out_features, x.size(-1))


class ModernTCNRegressor(nn.Module):
    """ModernTCN backbone with a length-agnostic regression head.

    Args:
        in_channels: Number of input variables/sensors ``M``.
        output_dim: Number of continuous regression targets.
        dims: Embedding width ``D`` of each temporal hierarchy stage.
        depths: Number of ModernTCN blocks in each stage.
        kernel_size: Large temporal kernel, shared or specified per stage.
        small_kernel_size: Reparameterization branch kernel, shared or per
            stage. ``None`` disables the small branch; the faithful default is
            5. Small kernels may have either parity but cannot exceed the
            corresponding large kernel.
        patch_size: Width of the per-variable temporal patch embedding.
        patch_stride: Patch embedding stride.
        stage_stride: Shared temporal downsampling ratio between stages.
        expansion_ratio: Integer ConvFFN width ratio, as in ModernTCN.
        dropout: Dropout probability in both ConvFFNs and the head.
        small_kernel_merged: Construct deploy-form large kernels directly.
            Normal training should retain the default ``False`` and call
            :meth:`structural_reparam` only after training and ``eval()``.
        activation_checkpointing: Recompute the BN-free ConvFFN region during
            backward to bound RAW-batch memory. This changes compute/memory
            tradeoff only, not the architecture or BatchNorm updates.
        depthwise_channel_chunk_size: Maximum independent channel group passed
            to one depthwise Conv1d execution. The default bounds deterministic
            cuDNN workspace on long RAW batches; ``None`` executes all ``M*D``
            channels at once. This is an execution policy, not a model change.
        pool_bins: ``None`` or an empty tuple selects native global temporal
            averaging. A non-empty tuple selects the same adaptive temporal-
            pyramid max-pooling head available to the incumbent; the
            ModernTCN backbone is unchanged.

    Input shape:
        ``(batch, in_channels, sequence_length)`` for any non-empty RAW/PAA
        length.

    Output shape:
        ``(batch, output_dim)``; a singleton output dimension is retained.

    Regression-head adaptation:
        The final backbone tensor remains ``(B,M,D,N)``. GELU and the selected
        fixed-width temporal pool remove only ``N``; the ordered features
        preserve sensor identity until the linear multi-output projection.
    """

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        *,
        dims: Sequence[int] = (32, 64, 128),
        depths: Sequence[int] = (1, 1, 2),
        kernel_size: int | Sequence[int] = 31,
        small_kernel_size: int | Sequence[int | None] | None = 5,
        patch_size: int = 16,
        patch_stride: int = 8,
        stage_stride: int = 2,
        expansion_ratio: float = 2.0,
        dropout: float = 0.1,
        small_kernel_merged: bool = False,
        activation_checkpointing: bool = True,
        depthwise_channel_chunk_size: int | None = 64,
        pool_bins: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = _positive_int("in_channels", in_channels)
        self.n_vars = self.in_channels
        self.output_dim = _positive_int("output_dim", output_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.patch_stride = _positive_int("patch_stride", patch_stride)
        if self.patch_stride > self.patch_size:
            raise ValueError(
                "patch_stride cannot exceed patch_size (patches must cover "
                "the signal)"
            )
        self.stage_stride = _positive_int("stage_stride", stage_stride)
        self.dropout_probability = _dropout_probability(dropout)
        self.expansion_ratio = _integer_expansion_ratio(expansion_ratio)
        if not isinstance(small_kernel_merged, bool):
            raise ValueError("small_kernel_merged must be boolean")
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing
        if depthwise_channel_chunk_size is not None:
            depthwise_channel_chunk_size = _positive_int(
                "depthwise_channel_chunk_size",
                depthwise_channel_chunk_size,
            )
        self.depthwise_channel_chunk_size = depthwise_channel_chunk_size

        if isinstance(dims, (str, bytes)) or not isinstance(dims, Sequence):
            raise ValueError("dims must be a non-empty sequence of positive integers")
        if isinstance(depths, (str, bytes)) or not isinstance(depths, Sequence):
            raise ValueError(
                "depths must be a non-empty sequence of positive integers"
            )
        stage_dims = tuple(
            _positive_int(f"dims[{index}]", value)
            for index, value in enumerate(dims)
        )
        stage_depths = tuple(
            _positive_int(f"depths[{index}]", value)
            for index, value in enumerate(depths)
        )
        if not stage_dims:
            raise ValueError("dims must be a non-empty sequence of positive integers")
        if len(stage_dims) != len(stage_depths):
            raise ValueError("dims and depths must contain the same number of stages")

        stage_kernels = _stage_positive_ints(
            "kernel_size",
            kernel_size,
            len(stage_dims),
        )
        stage_small_kernels = _stage_optional_positive_ints(
            "small_kernel_size",
            small_kernel_size,
            len(stage_dims),
        )
        for index, (small, large) in enumerate(
            zip(stage_small_kernels, stage_kernels, strict=True)
        ):
            if small is not None and small > large:
                raise ValueError(
                    f"small_kernel_size[{index}] cannot exceed "
                    f"kernel_size[{index}]"
                )

        self.dims = stage_dims
        self.depths = stage_depths
        self.kernel_sizes = stage_kernels
        self.small_kernel_sizes = stage_small_kernels

        # Applied after reshaping B,M,L -> B*M,1,L: sensors never become input
        # channels of this convolution, and the same stem is shared over M.
        self.patch_embedding = nn.Conv1d(
            1,
            stage_dims[0],
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=0,
        )
        self.patch_norm = nn.BatchNorm1d(stage_dims[0])

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for index, (features, depth, large, small) in enumerate(
            zip(
                stage_dims,
                stage_depths,
                stage_kernels,
                stage_small_kernels,
                strict=True,
            )
        ):
            self.stages.append(
                _ModernTCNStage(
                    nvars=self.n_vars,
                    dmodel=features,
                    depth=depth,
                    large_size=large,
                    small_size=small,
                    ffn_ratio=self.expansion_ratio,
                    dropout=self.dropout_probability,
                    small_kernel_merged=small_kernel_merged,
                    activation_checkpointing=self.activation_checkpointing,
                    depthwise_channel_chunk_size=(
                        self.depthwise_channel_chunk_size
                    ),
                )
            )
            if index < len(stage_dims) - 1:
                self.transitions.append(
                    _StageTransition(
                        nvars=self.n_vars,
                        in_features=features,
                        out_features=stage_dims[index + 1],
                        stride=self.stage_stride,
                    )
                )

        self.head_activation = nn.GELU()
        self.pool = AdaptiveTemporalPooling1D(pool_bins)
        self.head_dropout = nn.Dropout(self.dropout_probability)
        self.regression_head = nn.Linear(
            self.n_vars * stage_dims[-1] * self.pool.output_multiplier,
            self.output_dim,
        )

    def temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final explicit ``(B,M,D,N)`` representation."""

        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, in_channels, sequence_length)"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.size(1)}"
            )
        if x.size(0) < 1 or x.size(2) < 1:
            raise ValueError("batch and sequence dimensions must be non-empty")

        batch, nvars, length = x.shape
        x = x.reshape(batch * nvars, 1, length)
        x = _right_pad_to_ceil(x, self.patch_size, self.patch_stride)
        x = self.patch_norm(self.patch_embedding(x))
        x = x.reshape(batch, nvars, self.dims[0], x.size(-1))

        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.transitions):
                x = self.transitions[index](x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ordered, fixed-width ``(B,M*D_last)`` head features."""

        x = self.head_activation(self.temporal_features(x))
        batch, nvars, features, length = x.shape
        x = self.pool(x.reshape(batch * nvars, features, length))
        return x.reshape(batch, nvars * features * self.pool.output_multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.head_dropout(self.forward_features(x)))

    def structural_reparam(self) -> "ModernTCNRegressor":
        """Fuse every large/small Conv-BN pair for inference deployment.

        Conversion is intentionally restricted to evaluation mode because
        training-mode BatchNorm uses batch statistics and therefore has no
        fixed convolutional equivalent. The conversion is idempotent.
        """

        if self.training:
            raise RuntimeError("call eval() before structural_reparam()")
        for module in list(self.modules()):
            if isinstance(module, _ReparamLargeKernelConv):
                module.merge_kernel()
        return self

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Load either training-branch or fused deployment checkpoints.

        Fresh factory models use training-form branches.  A checkpoint saved
        after :meth:`structural_reparam` instead contains ``lkb_reparam``
        tensors; that complete schema is detected and recreated before the
        ordinary strict load. Mixed or incomplete schemas are rejected before
        mutating the receiver.
        """

        keys = set(state_dict)
        modules = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, _ReparamLargeKernelConv)
        ]
        deploy_keys = {
            f"{name}.lkb_reparam.{field}"
            for name, _module in modules
            for field in ("weight", "bias")
        }
        has_deploy = any(".lkb_reparam." in key for key in keys)
        has_training = any(
            ".lkb_origin." in key or ".small_conv." in key for key in keys
        )
        if has_deploy and has_training:
            raise RuntimeError(
                "ModernTCN checkpoint mixes training and deploy branch schemas"
            )
        if has_deploy:
            missing_deploy = sorted(deploy_keys - keys)
            unexpected_deploy = sorted(
                key
                for key in keys
                if ".lkb_reparam." in key and key not in deploy_keys
            )
            if missing_deploy or unexpected_deploy:
                raise RuntimeError(
                    "incomplete ModernTCN deploy checkpoint: "
                    f"missing={missing_deploy!r}, "
                    f"unexpected={unexpected_deploy!r}"
                )
            for _name, module in modules:
                module.prepare_deploy_structure()
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )


__all__ = ["ModernTCNRegressor"]
