"""Focused structural and numerical checks for the ModernTCN regressor.

Run:  python check_modern_tcn.py
"""

from __future__ import annotations

import torch

from core.modern_tcn import ModernTCNRegressor


failures = 0


def check(name: str, condition: bool) -> None:
    global failures
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures += 1


def rejects(name: str, function, exception_type=ValueError) -> None:
    try:
        function()
    except exception_type:
        check(name, True)
    except Exception as error:
        print(f"    unexpected {type(error).__name__}: {error}")
        check(name, False)
    else:
        check(name, False)


print("MODERNTCN REGRESSOR CHECKS")
torch.manual_seed(2026)

model = ModernTCNRegressor(
    in_channels=8,
    output_dim=3,
    dims=(12, 20),
    depths=(1, 2),
    kernel_size=(16, 31),  # mixed parity also exercises exact SAME padding
    small_kernel_size=(4, 5),
    patch_size=16,
    patch_stride=8,
    stage_stride=2,
    expansion_ratio=2.0,
    dropout=0.0,
)

# These checks encode ModernTCN's defining variable-aware structure and would
# fail against the former ConvNeXt-like implementation that folded M into the
# patch embedding's input channels.
first_block = model.stages[0].blocks[0]
check(
    "patch embedding is independent per variable",
    model.patch_embedding.in_channels == 1
    and model.patch_embedding.out_channels == 12,
)
check(
    "temporal DWConv groups equal M*D",
    first_block.dw.lkb_origin.conv.groups == 8 * 12
    and first_block.dw.small_conv.conv.groups == 8 * 12,
)
check(
    "ConvFFN1 is grouped by M",
    first_block.ffn1pw1.groups == 8
    and first_block.ffn1pw2.groups == 8,
)
check(
    "ConvFFN2 exists and is grouped by D after permutation",
    first_block.ffn2pw1.groups == 12
    and first_block.ffn2pw2.groups == 12
    and first_block.ffn2pw1.in_channels == 12 * 8,
)
check(
    "stage transition does not fold M into feature channels",
    model.transitions[0].projection.in_channels == 12
    and model.transitions[0].projection.out_channels == 20,
)

# Exercise representative PAA and RAW lengths with the same parameterization.
model.train()
paa = torch.randn(3, 8, 512, requires_grad=True)
paa_prediction = model(paa)
check("PAA output shape", paa_prediction.shape == (3, 3))
paa_prediction.square().mean().backward()
check(
    "PAA forward/backward finite",
    bool(torch.isfinite(paa_prediction).all())
    and paa.grad is not None
    and bool(torch.isfinite(paa.grad).all()),
)
check(
    "all trainable parameter gradients exist and are finite",
    all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
)

model.zero_grad(set_to_none=True)
raw = torch.randn(1, 8, 11_791, requires_grad=True)
raw_prediction = model(raw)
check("RAW output shape", raw_prediction.shape == (1, 3))
raw_prediction.sum().backward()
check(
    "RAW forward/backward finite",
    raw.grad is not None and bool(torch.isfinite(raw.grad).all()),
)

# Right-edge replication must make even a sub-patch signal executable.
short_prediction = model(torch.randn(2, 8, 5))
check("sub-patch sequence supported", short_prediction.shape == (2, 3))

model.eval()
probe = torch.randn(2, 8, 193)
with torch.no_grad():
    first = model(probe)
    second = model(probe)
    temporal = model.temporal_features(probe)
    pooled = model.forward_features(probe)
check("eval forward repeatable", torch.equal(first, second))
check(
    "explicit variable axis is preserved as (B,M,D,N)",
    temporal.ndim == 4
    and temporal.shape[:3] == (2, 8, 20)
    and temporal.size(-1) > 0,
)
check(
    "adaptive head is length independent and keeps ordered M*D features",
    pooled.shape == (2, 8 * 20)
    and model.regression_head.in_features == 8 * 20,
)

# Activation checkpointing is an execution policy, not an architectural
# variant. With dropout disabled it must preserve forward values, gradients,
# and the single BatchNorm update performed by the ordinary forward.
checkpointed = ModernTCNRegressor(
    in_channels=3,
    output_dim=2,
    dims=(8,),
    depths=(1,),
    kernel_size=15,
    small_kernel_size=5,
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
    activation_checkpointing=True,
    depthwise_channel_chunk_size=4,
)
eager = ModernTCNRegressor(
    in_channels=3,
    output_dim=2,
    dims=(8,),
    depths=(1,),
    kernel_size=15,
    small_kernel_size=5,
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
    activation_checkpointing=False,
    depthwise_channel_chunk_size=4,
)
eager.load_state_dict(checkpointed.state_dict())
unchunked = ModernTCNRegressor(
    in_channels=3,
    output_dim=2,
    dims=(8,),
    depths=(1,),
    kernel_size=15,
    small_kernel_size=5,
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
    activation_checkpointing=False,
    depthwise_channel_chunk_size=None,
)
unchunked.load_state_dict(checkpointed.state_dict())
checkpointed.train()
eager.train()
unchunked.train()
checkpoint_input = torch.randn(2, 3, 97)
checkpoint_input_a = checkpoint_input.clone().requires_grad_(True)
checkpoint_input_b = checkpoint_input.clone().requires_grad_(True)
checkpoint_input_c = checkpoint_input.clone().requires_grad_(True)
checkpoint_output = checkpointed(checkpoint_input_a)
eager_output = eager(checkpoint_input_b)
unchunked_output = unchunked(checkpoint_input_c)
checkpoint_output.square().sum().backward()
eager_output.square().sum().backward()
unchunked_output.square().sum().backward()
checkpoint_grad_error = max(
    float((left.grad - right.grad).abs().max())
    for left, right in zip(
        checkpointed.parameters(),
        eager.parameters(),
        strict=True,
    )
    if left.requires_grad
)
checkpoint_input_grad_error = float(
    (checkpoint_input_a.grad - checkpoint_input_b.grad).abs().max()
)
chunk_grad_error = max(
    float((left.grad - right.grad).abs().max())
    for left, right in zip(eager.parameters(), unchunked.parameters(), strict=True)
    if left.requires_grad
)
chunk_input_grad_error = float(
    (checkpoint_input_b.grad - checkpoint_input_c.grad).abs().max()
)
check(
    "checkpoint on/off forward parity",
    torch.equal(checkpoint_output, eager_output),
)
check(
    "checkpoint on/off gradient parity",
    checkpoint_grad_error <= 1.0e-7 and checkpoint_input_grad_error <= 1.0e-7,
)
check(
    "depthwise channel chunk on/off forward parity",
    bool(torch.allclose(eager_output, unchunked_output, atol=1.0e-7, rtol=1.0e-6)),
)
check(
    "depthwise channel chunk on/off gradient parity",
    chunk_grad_error <= 1.0e-6 and chunk_input_grad_error <= 1.0e-6,
)
checkpoint_bn_steps = [
    int(module.num_batches_tracked)
    for module in checkpointed.modules()
    if isinstance(module, torch.nn.BatchNorm1d)
]
eager_bn_steps = [
    int(module.num_batches_tracked)
    for module in eager.modules()
    if isinstance(module, torch.nn.BatchNorm1d)
]
check(
    "checkpointing does not double-update BatchNorm",
    checkpoint_bn_steps == eager_bn_steps
    and bool(checkpoint_bn_steps)
    and all(steps == 1 for steps in checkpoint_bn_steps),
)

# A partially frozen stem/backbone can feed a tensor without requires_grad
# into the still-trainable FFNs. Checkpointing must fall back to eager mode so
# those trainable weights retain their graph.
partially_frozen = ModernTCNRegressor(
    in_channels=2,
    output_dim=1,
    dims=(8,),
    depths=(1,),
    kernel_size=15,
    small_kernel_size=5,
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
    activation_checkpointing=True,
)
for frozen_module in (
    partially_frozen.patch_embedding,
    partially_frozen.patch_norm,
    partially_frozen.stages[0].blocks[0].dw,
    partially_frozen.stages[0].blocks[0].norm,
):
    for parameter in frozen_module.parameters():
        parameter.requires_grad_(False)
partially_frozen.train()
partially_frozen(torch.randn(2, 2, 65)).square().mean().backward()
ffn_parameters = [
    parameter
    for name, parameter in partially_frozen.named_parameters()
    if ".ffn" in name
]
check(
    "checkpointing preserves FFN gradients after partial freezing",
    bool(ffn_parameters)
    and all(parameter.grad is not None for parameter in ffn_parameters)
    and all(bool(torch.isfinite(parameter.grad).all()) for parameter in ffn_parameters),
)

# Train-form large+small Conv-BN branches must fuse into a single convolution
# without changing predictions. Mixed even/odd kernels guard alignment logic.
torch.manual_seed(41)
deploy_model = ModernTCNRegressor(
    in_channels=3,
    output_dim=2,
    dims=(8, 12),
    depths=(1, 1),
    kernel_size=(16, 15),
    small_kernel_size=(4, 5),
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
)
deploy_model.train()
with torch.no_grad():
    for _ in range(3):
        deploy_model(torch.randn(4, 3, 129))
deploy_model.eval()
deploy_probe = torch.randn(2, 3, 131)
with torch.no_grad():
    before_merge = deploy_model(deploy_probe)
deploy_model.structural_reparam()
with torch.no_grad():
    after_merge = deploy_model(deploy_probe)
max_reparam_error = float((before_merge - after_merge).abs().max())
merged_depthwise = [
    block.dw
    for stage in deploy_model.stages
    for block in stage.blocks
]
check(
    "large+small kernels structurally reparameterize",
    bool(merged_depthwise)
    and all(module.is_reparameterized for module in merged_depthwise)
    and all(not hasattr(module, "lkb_origin") for module in merged_depthwise),
)
check(
    f"reparameterization prediction parity (max error={max_reparam_error:.3e})",
    max_reparam_error <= 2.0e-5,
)
with torch.no_grad():
    repeat_merge = deploy_model.structural_reparam()(deploy_probe)
check(
    "structural reparameterization is idempotent",
    torch.equal(after_merge, repeat_merge),
)

# A fused checkpoint is a first-class portable artifact: a fresh training-form
# instance must detect its schema, rebuild deploy branches, and load strictly.
deploy_state = deploy_model.state_dict()
deploy_reload = ModernTCNRegressor(
    in_channels=3,
    output_dim=2,
    dims=(8, 12),
    depths=(1, 1),
    kernel_size=(16, 15),
    small_kernel_size=(4, 5),
    patch_size=8,
    patch_stride=4,
    expansion_ratio=2,
    dropout=0.0,
)
deploy_reload.eval()
deploy_reload.load_state_dict(deploy_state, strict=True)
with torch.no_grad():
    reloaded_deploy_output = deploy_reload(deploy_probe)
reloaded_depthwise = [
    block.dw
    for stage in deploy_reload.stages
    for block in stage.blocks
]
check(
    "deploy checkpoint rebuild/load round-trip",
    all(module.is_reparameterized for module in reloaded_depthwise)
    and torch.equal(after_merge, reloaded_deploy_output),
)

incomplete_deploy_state = dict(deploy_state)
incomplete_deploy_state.pop(
    next(key for key in incomplete_deploy_state if key.endswith("lkb_reparam.bias"))
)
rejects(
    "incomplete deploy checkpoint rejected before load",
    lambda: ModernTCNRegressor(
        in_channels=3,
        output_dim=2,
        dims=(8, 12),
        depths=(1, 1),
        kernel_size=(16, 15),
        small_kernel_size=(4, 5),
        patch_size=8,
        patch_stride=4,
        expansion_ratio=2,
        dropout=0.0,
    ).load_state_dict(incomplete_deploy_state, strict=True),
    RuntimeError,
)

training_form = ModernTCNRegressor(
    in_channels=2,
    output_dim=1,
    dims=(8,),
    depths=(1,),
    kernel_size=7,
    patch_size=4,
    patch_stride=4,
    dropout=0.0,
)
check(
    "single target dimension retained",
    training_form(torch.randn(4, 2, 17)).shape == (4, 1),
)
rejects(
    "training-mode reparameterization rejected",
    training_form.structural_reparam,
    RuntimeError,
)

rejects("wrong input rank rejected", lambda: model(torch.randn(2, 512)))
rejects("wrong channel count rejected", lambda: model(torch.randn(2, 7, 512)))
rejects("empty sequence rejected", lambda: model(torch.randn(2, 8, 0)))
rejects(
    "mismatched stage configuration rejected",
    lambda: ModernTCNRegressor(
        in_channels=8, output_dim=3, dims=(8, 16), depths=(1,)
    ),
)
rejects(
    "small kernel larger than large kernel rejected",
    lambda: ModernTCNRegressor(
        in_channels=8,
        output_dim=3,
        dims=(8,),
        depths=(1,),
        kernel_size=3,
        small_kernel_size=5,
    ),
)
rejects(
    "non-integer ModernTCN expansion ratio rejected",
    lambda: ModernTCNRegressor(
        in_channels=8,
        output_dim=3,
        expansion_ratio=1.5,
    ),
)
rejects(
    "invalid dropout rejected",
    lambda: ModernTCNRegressor(in_channels=8, output_dim=3, dropout=1.0),
)
rejects(
    "uncovered patch stride rejected",
    lambda: ModernTCNRegressor(
        in_channels=8,
        output_dim=3,
        patch_size=8,
        patch_stride=9,
    ),
)

if failures:
    raise SystemExit(f"FAIL: {failures} ModernTCN check(s) failed")
print("PASS: variable-aware ModernTCN supports RAW/PAA regression")
