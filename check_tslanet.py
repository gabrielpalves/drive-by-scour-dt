"""Focused CPU/CUDA checks for the TSLANet-inspired regressor.

Run with the pinned campaign interpreter:
    .venv-campaign-py313/Scripts/python -B check_tslanet.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.tslanet import (
    AdaptiveSpectralBlock,
    DropPath,
    InteractiveConvolutionBlock,
    PatchEmbedding1D,
    TSLABlock,
    TSLANetRegressor,
)
from core.utils import DETERMINISM_POLICY, set_global_seed


fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        fails += 1


def rejects(name: str, function, error=ValueError) -> None:
    try:
        function()
    except error:
        check(name, True)
    except Exception as exc:
        print(f"    unexpected {type(exc).__name__}: {exc}")
        check(name, False)
    else:
        check(name, False)


def cuda_training_snapshot(seed: int) -> dict[str, torch.Tensor]:
    """Run one seeded stochastic CUDA step and return exact audit tensors."""
    set_global_seed(seed, DETERMINISM_POLICY)
    device = torch.device("cuda")
    cuda_model = TSLANetRegressor(
        8,
        3,
        embed_dim=16,
        depth=2,
        patch_size=16,
        patch_stride=8,
        dropout=0.2,
        adaptive_filter=True,
        position_bins=32,
    ).to(device)
    cuda_model.train()
    optimizer = torch.optim.SGD(cuda_model.parameters(), lr=1e-3)
    cuda_x = torch.randn(4, 8, 513, device=device, requires_grad=True)
    target = torch.randn(4, 3, device=device)

    optimizer.zero_grad(set_to_none=True)
    cuda_y = cuda_model(cuda_x)
    loss = F.mse_loss(cuda_y, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    snapshot = {
        "output": cuda_y.detach().cpu().clone(),
        "loss": loss.detach().cpu().clone(),
        "input_grad": cuda_x.grad.detach().cpu().clone(),
    }
    for name, parameter in cuda_model.named_parameters():
        if parameter.grad is not None:
            snapshot[f"gradient/{name}"] = parameter.grad.detach().cpu().clone()
    for name, value in cuda_model.state_dict().items():
        snapshot[f"state/{name}"] = value.detach().cpu().clone()
    return snapshot


print("TSLANET CHECKS")
# Apply the same executable policy as campaign trials before the first CUDA
# operation. This also rejects a conflicting shell workspace configuration.
set_global_seed(2026, DETERMINISM_POLICY)
torch.set_num_threads(1)
check(
    "canonical determinism policy active",
    os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    == DETERMINISM_POLICY["cublas_workspace_config"]
    and torch.are_deterministic_algorithms_enabled()
    == DETERMINISM_POLICY["torch_deterministic_algorithms"]
    and torch.backends.cudnn.deterministic
    == DETERMINISM_POLICY["cudnn_deterministic"]
    and torch.backends.cudnn.benchmark
    == DETERMINISM_POLICY["cudnn_benchmark"],
)

model = TSLANetRegressor(
    in_channels=8,
    output_dim=4,
    embed_dim=24,
    depth=2,
    patch_size=31,
    patch_stride=16,
    mlp_ratio=1.5,
    dropout=0.0,
    adaptive_filter=True,
    position_bins=32,
)

# The same parameterization accepts PAA plus the exact F40/L99 RAW lengths.
for length in (512, 5831, 11791):
    batch_size = 2 if length < 11791 else 1
    x = torch.randn(batch_size, 8, length, requires_grad=True)
    prediction = model(x)
    loss = prediction.square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    check(
        f"output shape at L={length}",
        prediction.shape == (batch_size, 4),
    )
    check(f"finite forward/backward at L={length}", bool(torch.isfinite(prediction).all())
          and x.grad is not None and bool(torch.isfinite(x.grad).all()))

spectral = model.blocks[0].asb
check("adaptive spectral weight receives gradient",
      spectral.complex_weight_high.grad is not None
      and bool(torch.isfinite(spectral.complex_weight_high.grad).all()))
check("adaptive threshold receives gradient",
      spectral.log_threshold.grad is not None
      and bool(torch.isfinite(spectral.log_threshold.grad).all()))

# Lock the characteristic upstream ASB and ICB equations independently of the
# end-to-end shape checks.
base_asb = AdaptiveSpectralBlock(4, adaptive_filter=False)
asb_input = torch.randn(2, 9, 4)
asb_spectrum = torch.fft.rfft(asb_input, dim=1, norm="ortho")
asb_weight = torch.view_as_complex(base_asb.complex_weight.contiguous())
expected_asb = torch.fft.irfft(
    asb_spectrum * asb_weight.view(1, 1, -1),
    n=asb_input.size(1),
    dim=1,
    norm="ortho",
)
check(
    "ASB base branch matches the upstream spectral equation",
    torch.equal(base_asb(asb_input), expected_asb),
)

icb = InteractiveConvolutionBlock(4, hidden_dim=7, dropout=0.0).eval()
icb_input = torch.randn(2, 11, 4)
icb_channels = icb_input.transpose(1, 2)
icb_pointwise = icb.pointwise(icb_channels)
icb_temporal = icb.temporal(icb_channels)
expected_icb = icb.projection(
    icb_pointwise * icb.activation(icb_temporal)
    + icb_temporal * icb.activation(icb_pointwise)
).transpose(1, 2)
check(
    "ICB matches the upstream interactive-convolution equation",
    torch.equal(icb(icb_input), expected_icb),
)
truncated_width_block = TSLABlock(
    5,
    mlp_ratio=1.5,
    dropout=0.0,
    drop_path=0.0,
).eval()
check(
    "ICB hidden width follows upstream integer truncation",
    truncated_width_block.icb.pointwise.out_channels == 7,
)

ordered_block = TSLABlock(
    4,
    mlp_ratio=2.0,
    dropout=0.0,
    drop_path=0.0,
).eval()
block_input = torch.randn(2, 11, 4)
expected_update = ordered_block.asb(ordered_block.norm_asb(block_input))
expected_update = ordered_block.icb(ordered_block.norm_icb(expected_update))
check(
    "TSLA block preserves pre-norm ASB-to-ICB residual ordering",
    torch.equal(ordered_block(block_input), block_input + expected_update),
)

# The inspired STE is explicitly hard in the forward pass and sigmoidal in
# the backward pass. It intentionally trains both threshold and spectrum.
ste = AdaptiveSpectralBlock(
    2,
    threshold_init=1.0,
    mask_temperature=0.5,
)
spectral_parts = torch.tensor(
    [[
        [[0.5, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.4142135, 0.0], [0.0, 0.0]],
        [[2.0, 0.0], [0.0, 0.0]],
    ]],
    requires_grad=True,
)
ste_mask = ste.adaptive_mask(torch.view_as_complex(spectral_parts))
ste_mask.sum().backward()
check(
    "adaptive mask is binary in the forward pass",
    bool(torch.logical_or(ste_mask == 0, ste_mask == 1).all()),
)
check(
    "sigmoid STE trains threshold with the correct sign",
    ste.log_threshold.grad is not None
    and bool(torch.isfinite(ste.log_threshold.grad).all())
    and float(ste.log_threshold.grad) < 0.0,
)
check(
    "sigmoid STE propagates through spectral energy",
    spectral_parts.grad is not None
    and bool(torch.isfinite(spectral_parts.grad).all())
    and bool((spectral_parts.grad != 0).any()),
)

# Patch padding covers all samples and handles sub-patch sequences.
patcher = PatchEmbedding1D(3, 8, patch_size=16, stride=8)
check("non-divisible patch count includes final partial patch",
      patcher.n_patches(31) == 3 and patcher(torch.randn(1, 3, 31)).shape == (1, 3, 8))
check("sequence shorter than a patch is accepted",
      patcher(torch.randn(2, 3, 7)).shape == (2, 1, 8))
check(
    "right padding is minimal and adds no all-padding patch",
    patcher.right_padding(31) == 1
    and patcher.right_padding(32) == 0
    and patcher.right_padding(7) == 9
    and patcher.n_patches(32) == 3,
)

# The normalized positional template must implement the documented
# align_corners=False interpolation without becoming length-specific.
template_model = TSLANetRegressor(
    1,
    1,
    embed_dim=3,
    depth=1,
    patch_size=4,
    position_bins=4,
    use_asb=False,
    use_icb=False,
)
with torch.no_grad():
    template_model.position_embedding.copy_(
        torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    )
template_tokens = template_model._position_tokens(7)
expected_tokens = F.interpolate(
    template_model.position_embedding.transpose(1, 2),
    size=7,
    mode="linear",
    align_corners=False,
).transpose(1, 2)
check(
    "positional template matches linear interpolation semantics",
    template_tokens is not None
    and torch.allclose(template_tokens, expected_tokens, atol=1e-6, rtol=1e-6),
)

# The ablation toggle removes only the adaptive branch, not base FFT filtering.
no_adaptive = TSLANetRegressor(
    8, 3, embed_dim=16, depth=1, patch_size=16,
    adaptive_filter=False, position_bins=None,
)
x = torch.randn(2, 8, 517, requires_grad=True)
y = no_adaptive(x)
y.sum().backward()
check("adaptive-filter-off forward/backward", y.shape == (2, 3)
      and bool(torch.isfinite(y).all()) and x.grad is not None)
check("adaptive-filter toggle is registered",
      isinstance(no_adaptive.blocks[0].asb, AdaptiveSpectralBlock)
      and no_adaptive.blocks[0].asb.adaptive_filter is False)

# Upstream architectural details retained by the adaptation.
regularized = TSLANetRegressor(
    8,
    3,
    embed_dim=16,
    depth=3,
    patch_size=16,
    dropout=0.3,
)
check(
    "residual regularization uses scheduled DropPath",
    isinstance(regularized.blocks[0].drop_path, nn.Identity)
    and isinstance(regularized.blocks[1].drop_path, DropPath)
    and isinstance(regularized.blocks[2].drop_path, DropPath)
    and regularized.blocks[1].drop_path.drop_prob == 0.15
    and regularized.blocks[2].drop_path.drop_prob == 0.3,
)
drop_path = DropPath(0.25).train()
drop_path_output = drop_path(torch.ones(32, 5, 7))
flat_drop_path = drop_path_output.flatten(1)
check(
    "DropPath shares one decision across each residual sample",
    all(torch.unique(row).numel() == 1 for row in flat_drop_path)
    and bool(
        torch.logical_or(
            drop_path_output == 0.0,
            drop_path_output == (1.0 / 0.75),
        ).all()
    ),
)
check(
    "mean pooling has no extra terminal normalization",
    not hasattr(model, "final_norm"),
)
linear_biases = [
    module.bias
    for module in model.modules()
    if isinstance(module, nn.Linear) and module.bias is not None
]
layer_norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
check(
    "upstream Linear and LayerNorm initialization contract",
    bool(linear_biases)
    and all(torch.count_nonzero(bias) == 0 for bias in linear_biases)
    and bool(layer_norms)
    and all(
        torch.equal(norm.weight, torch.ones_like(norm.weight))
        and torch.equal(norm.bias, torch.zeros_like(norm.bias))
        for norm in layer_norms
    ),
)

# With stochastic layers disabled, repeated CPU inference is exactly stable.
model.eval()
probe = torch.randn(1, 8, 513)
with torch.no_grad():
    first = model(probe)
    second = model(probe)
check("deterministic repeated CPU inference", torch.equal(first, second))

# Repeat a genuinely stochastic training step from initialization through the
# optimizer update. Exact equality covers outputs, loss, input/parameter
# gradients and the updated state_dict, including DropPath and dropout RNGs.
if torch.cuda.is_available():
    first_cuda = cuda_training_snapshot(8128)
    second_cuda = cuda_training_snapshot(8128)
    check(
        "finite CUDA training-step audit tensors",
        first_cuda["output"].shape == (4, 3)
        and first_cuda.keys() == second_cuda.keys()
        and all(bool(torch.isfinite(value).all()) for value in first_cuda.values()),
    )
    check(
        "exact repeated CUDA stochastic training step",
        first_cuda.keys() == second_cuda.keys()
        and all(
            torch.equal(first_cuda[name], second_cuda[name])
            for name in first_cuda
        ),
    )
else:
    print("  [SKIP] exact repeated CUDA stochastic training step (CUDA unavailable)")

rejects("wrong input channel count rejected",
        lambda: model(torch.randn(2, 7, 512)))
rejects("uncovered patch stride rejected",
        lambda: PatchEmbedding1D(8, 16, patch_size=8, stride=9))
rejects("non-floating input rejected",
        lambda: model(torch.ones(2, 8, 512, dtype=torch.int64)), TypeError)

print()
print("TSLANET: ALL PASS" if fails == 0 else f"TSLANET: {fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
