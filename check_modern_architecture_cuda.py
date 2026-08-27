"""CUDA capacity smoke for the two contemporary challenger families.

This is a development preflight, not a campaign authorization receipt. It
executes a warm-up followed by a measured complete optimizer step at the
maximum registered structural point for the optional F40-S challenger scope:
RAW/PAA, batch 32, the authenticated two-sensor pair and one scour target. Run
it on every candidate training GPU before implementing/authorizing a
challenger dispatcher.  Shapes and head count are cross-checked against the
hashed challenger and campaign contracts rather than duplicated silently here.

PowerShell:
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    python -B check_modern_architecture_cuda.py
"""

from __future__ import annotations

import gc
import hashlib

import torch

from core import task
from core.campaign_contract import campaign_stage_contract
from core.capacity_preflight import CAPACITY_PREFLIGHT_POLICY
from core.challenger_policy import (
    CHALLENGER_CAPACITY_SCOPE,
    CHALLENGER_OPTIMIZER_SPACE,
    CHALLENGER_PATCH_LAYOUTS,
    CHALLENGER_POOLING_LAYOUTS,
    CHALLENGER_SEARCH_SPACES,
    MODERN_TCN_LAYOUTS,
    challenger_search_policy,
    validate_challenger_parameters,
)
from core.models import build_model
from core.dataset import PREPROC_PROTOCOL
from core.paper1_challenger_contract import (
    CHALLENGER_INPUT_CHANNEL_COUNT,
    CHALLENGER_INPUT_SELECTOR,
    CHALLENGER_STAGE,
    complete_challenger_job_grid,
)
from core.utils import set_global_seed
from training.trainer import (
    TRAIN_PROTOCOL,
    _suggest_params,
    make_optimizer,
)


BATCH_SIZE = int(CHALLENGER_CAPACITY_SCOPE["batch_size"])
INPUT_CHANNELS = int(CHALLENGER_CAPACITY_SCOPE["input_channels"])
OUTPUT_HEADS = int(CHALLENGER_CAPACITY_SCOPE["output_heads"])
INPUT_LENGTHS = dict(
    CHALLENGER_CAPACITY_SCOPE["representation_input_lengths"]
)
MINIMUM_HEADROOM_FRACTION = float(
    CHALLENGER_CAPACITY_SCOPE["minimum_remaining_headroom"][
        "fraction_of_total_memory"
    ]
)
MINIMUM_HEADROOM_BYTES = int(
    CHALLENGER_CAPACITY_SCOPE["minimum_remaining_headroom"]["absolute_bytes"]
)


class ExactTrial:
    """Accept only singleton domains matching one frozen parameter mapping."""

    def __init__(self, values: dict) -> None:
        self.values = values

    def suggest_int(self, name, low, high, step=1):
        value = self.values[name]
        assert low == high == value
        return value

    def suggest_float(self, name, low, high, **_kwargs):
        value = float(self.values[name])
        assert float(low) == float(high) == value
        return value

    def suggest_categorical(self, name, choices):
        assert list(choices) == [self.values[name]]
        return self.values[name]


def config_for(model_type: str, representation: str) -> dict:
    learning = campaign_stage_contract(CHALLENGER_STAGE)["learning"]
    return {
        "name": f"capacity_{model_type}_{representation}",
        "model_type": model_type,
        "method": representation,
        "task": "regression",
        "target_supports": learning["target_supports"],
        "bearing_targets": learning["bearing_targets"],
        "use_space2vec": False,
        "use_lstm": False,
        "use_nhits": False,
    }


def _categorical_capacity_choice(name: str, choices: list) -> object:
    """Derive the capacity endpoint from the live categorical registry."""

    if name == "modern_layout":
        def layout_cost(label: str) -> tuple[int, int, int]:
            layout = MODERN_TCN_LAYOUTS[label]
            dims = layout["dims"]
            depths = layout["depths"]
            return (
                sum(depth * dim * dim for dim, depth in zip(dims, depths)),
                max(dims),
                sum(depths),
            )

        return max(choices, key=layout_cost)
    if name in {"modern_patch_layout", "tsla_patch_layout"}:
        return min(
            choices,
            key=lambda label: (
                CHALLENGER_PATCH_LAYOUTS[label][1],
                CHALLENGER_PATCH_LAYOUTS[label][0],
            ),
        )
    if name == "challenger_pooling":
        return max(
            choices,
            key=lambda label: (
                sum(CHALLENGER_POOLING_LAYOUTS[label]),
                len(CHALLENGER_POOLING_LAYOUTS[label]),
            ),
        )
    if all(isinstance(choice, (int, float)) for choice in choices):
        return max(choices)
    raise AssertionError(
        f"no capacity ordering registered for categorical parameter {name!r}"
    )


def registered_capacity_parameters(model_type: str) -> dict:
    """Compute the structural endpoint directly from every live HPO domain."""

    complete_space = {
        **CHALLENGER_OPTIMIZER_SPACE,
        **CHALLENGER_SEARCH_SPACES[model_type],
    }
    params = {}
    for name, spec in complete_space.items():
        kind = spec[0]
        if kind == "cat":
            params[name] = _categorical_capacity_choice(name, list(spec[1]))
        elif kind in {"int", "int_step"}:
            params[name] = int(spec[2])
        elif kind in {"float", "logfloat"}:
            params[name] = float(spec[2])
        else:
            raise AssertionError(
                f"unsupported capacity-domain kind {kind!r} for {name!r}"
            )
    return validate_challenger_parameters(model_type, params)


def validate_registered_maxima() -> None:
    assert TRAIN_PROTOCOL["batch_size"] == BATCH_SIZE
    assert challenger_search_policy()["capacity_scope"] == (
        CHALLENGER_CAPACITY_SCOPE
    )
    assert CHALLENGER_CAPACITY_SCOPE["stage"] == CHALLENGER_STAGE == "F40-S"
    assert CHALLENGER_CAPACITY_SCOPE["input_selector"] == (
        CHALLENGER_INPUT_SELECTOR
    )
    assert INPUT_CHANNELS == CHALLENGER_INPUT_CHANNEL_COUNT == 2
    challenger = complete_challenger_job_grid()
    jobs = [
        job
        for phase_jobs in challenger["phases"].values()
        for job in phase_jobs
    ]
    assert jobs and all(
        job["input_selector"] == CHALLENGER_INPUT_SELECTOR
        and job["input_channel_count"] == INPUT_CHANNELS
        and job["requires_primary_selection_artifact"] is True
        for job in jobs
    )
    stage = campaign_stage_contract(CHALLENGER_STAGE)
    expected_outputs = len(stage["learning"]["target_supports"]) + len(
        stage["learning"]["bearing_targets"] or []
    )
    assert expected_outputs == OUTPUT_HEADS
    post_window_samples = CAPACITY_PREFLIGHT_POLICY["largest_raw_case"][
        "post_window_samples"
    ]
    expected_raw_length = (
        round(100 * stage["geometry"]["L_bridge_m"]) + post_window_samples
    )
    assert INPUT_LENGTHS == {
        "RAW": expected_raw_length,
        "PAA": PREPROC_PROTOCOL["n_segments"],
    }
    assert set(INPUT_LENGTHS) == {
        arm["representation"] for arm in challenger["challenger_arms"]
    }
    assert set(CHALLENGER_SEARCH_SPACES) == {
        arm["model_type"] for arm in challenger["challenger_arms"]
    }
    for model_type in CHALLENGER_SEARCH_SPACES:
        params = registered_capacity_parameters(model_type)
        assert set(params) == {
            *CHALLENGER_OPTIMIZER_SPACE,
            *CHALLENGER_SEARCH_SPACES[model_type],
        }
        config = {
            **config_for(model_type, "RAW"),
            "frozen_hyperparameters": params,
        }
        assert _suggest_params(ExactTrial(params), config) == params


def probe(model_type: str, representation: str, device: torch.device) -> dict:
    params = registered_capacity_parameters(model_type)
    config = config_for(model_type, representation)
    input_shape = (
        BATCH_SIZE,
        INPUT_CHANNELS,
        INPUT_LENGTHS[representation],
    )
    model = optimizer = features = target = output = loss = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        model, n_outputs = build_model(config, params, input_shape, device)
        assert n_outputs == OUTPUT_HEADS
        optimizer = make_optimizer(
            model.parameters(), params, TRAIN_PROTOCOL["optimizer"]
        )
        criterion = task.make_criterion(config, TRAIN_PROTOCOL["loss"]).to(device)
        features = torch.randn(input_shape, dtype=torch.float32, device=device)
        target = torch.randn(
            BATCH_SIZE,
            OUTPUT_HEADS,
            dtype=torch.float32,
            device=device,
        )
        model.train()

        # First step materializes Adam's state tensors and warms allocator
        # paths.  Capacity is deliberately measured on the next full step.
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        assert output.shape == target.shape
        loss = criterion(output, target)
        assert bool(torch.isfinite(loss))
        loss.backward()
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        missing_gradients = [
            name for name, parameter in trainable if parameter.grad is None
        ]
        assert not missing_gradients, (
            f"trainable parameters without gradients: {missing_gradients!r}"
        )
        assert all(
            bool(torch.isfinite(parameter.grad).all())
            for _name, parameter in trainable
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        del loss, output
        loss = output = None
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)

        output = model(features)
        assert output.shape == target.shape
        loss = criterion(output, target)
        assert bool(torch.isfinite(loss))
        loss.backward()
        missing_gradients = [
            name for name, parameter in trainable if parameter.grad is None
        ]
        assert not missing_gradients, (
            f"trainable parameters without gradients: {missing_gradients!r}"
        )
        assert all(
            bool(torch.isfinite(parameter.grad).all())
            for _name, parameter in trainable
        )
        optimizer.step()
        assert all(
            bool(torch.isfinite(parameter).all())
            for _name, parameter in trainable
        )
        optimizer_tensors = [
            (f"optimizer.{name}.{state_name}", state_value)
            for name, parameter in trainable
            for state_name, state_value in optimizer.state[parameter].items()
            if torch.is_tensor(state_value)
        ]
        assert optimizer_tensors
        assert all(
            bool(torch.isfinite(state_value).all())
            for _name, state_value in optimizer_tensors
        )
        torch.cuda.synchronize(device)
        digest = hashlib.sha256()
        for name, tensor in (
            [("output", output), ("loss", loss)]
            + list(model.state_dict().items())
            + optimizer_tensors
        ):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(value.numpy().tobytes())
            digest.update(b"\n")
        return {
            "allocated": int(torch.cuda.max_memory_allocated(device)),
            "reserved": int(torch.cuda.max_memory_reserved(device)),
            "step_sha256": digest.hexdigest(),
        }
    finally:
        del loss, output, target, features, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def gib(value: int) -> float:
    return value / (1024 ** 3)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA challenger capacity check requires one CUDA GPU")
    validate_registered_maxima()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    properties = torch.cuda.get_device_properties(device)
    total = int(properties.total_memory)
    required = max(
        int(MINIMUM_HEADROOM_FRACTION * total),
        MINIMUM_HEADROOM_BYTES,
    )
    print(f"GPU: {properties.name} ({gib(total):.2f} GiB)")
    print(f"Required free headroom after peak: {gib(required):.2f} GiB")

    failures = []
    case_index = 0
    for model_type in CHALLENGER_SEARCH_SPACES:
        for representation in INPUT_LENGTHS:
            case_seed = 2026 + case_index
            case_index += 1
            set_global_seed(case_seed, TRAIN_PROTOCOL["determinism"])
            first = probe(model_type, representation, device)
            set_global_seed(case_seed, TRAIN_PROTOCOL["determinism"])
            second = probe(model_type, representation, device)
            peak_allocated = max(first["allocated"], second["allocated"])
            peak_reserved = max(first["reserved"], second["reserved"])
            headroom = total - peak_reserved
            repeatable = first["step_sha256"] == second["step_sha256"]
            passed = headroom >= required and repeatable
            print(
                f"  [{'PASS' if passed else 'FAIL'}] "
                f"{model_type:10s} {representation:3s} "
                f"allocated_peak={gib(peak_allocated):.2f} GiB "
                f"reserved_peak={gib(peak_reserved):.2f} GiB "
                f"headroom={gib(headroom):.2f} GiB "
                f"repeatable={repeatable}"
            )
            if not passed:
                failures.append((model_type, representation))
    if failures:
        raise SystemExit(f"challenger CUDA capacity failed: {failures!r}")
    print("MODERN ARCHITECTURE CUDA CAPACITY: ALL PASS")


if __name__ == "__main__":
    main()
