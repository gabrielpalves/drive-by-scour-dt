"""The two PAA finalist CNNs published by Fernandes et al. (2025, IJSSD).

Section 4.4 and Fig. 6 of the source report one tuned architecture per sensor;
``core.f25_experiment_contract.PUBLISHED_PAA_CNN_TEMPLATES`` freezes both.
The carbody finalist (two valid convolutions, no pooling, 580 x 48 = 27,840
flattened units) is exactly an F25 search vector, so it is built by the frozen
``core.f25_models.F25CNN``.  The front-bogie finalist pools 583 -> 292 -> 146
-> 73 (73 x 96 = 7,008), which only 'same' convolutions with ceil-mode pooling
reproduce; ``F25CNN`` hard-codes valid convolutions and floor pooling, so that
variant is built by :class:`PublishedFinalistCNN`, a mirror of ``F25CNN`` with
those two switches.  The valid/floor front-bogie variant (71 x 96 = 6,816) is
kept as a padding-sensitivity arm.

The source trained with TensorFlow/Keras; F25 uses PyTorch defaults.  The
Keras-defaults arm trains both published finalists again with the two Keras
defaults that change the optimisation: glorot_uniform kernels with zero biases
(:func:`keras_initialize_`) and the Keras Adam update (:class:`KerasAdam`).
Everything else, including the training loop, stays F25's.

This module lives outside the reviewed ``core``/``training`` packages on
purpose: it adds no source to their authenticated inventories.

Self-test::

    python f25_published_finalist_models.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from core.f25_experiment_contract import (
    CHANNELS,
    CLASS_COUNT,
    PAA_BLOCK_COUNT,
    PUBLISHED_PAA_CNN_TEMPLATES,
    PublishedPAACNNTemplate,
)
from core.f25_models import (
    F25CNN,
    F25ModelError,
    build_f25_model,
    normalize_model_params,
    parameter_count,
)


MODEL_SCHEMA = "f25-published-finalist-cnn-v1"
F25_ARM_ID = "PAA-CNN"
FRAMEWORK_DEFAULTS = ("pytorch", "keras")
KERAS_ADAM_EPSILON = 1.0e-7  # tf.keras.optimizers.Adam default


def template_params(template: PublishedPAACNNTemplate) -> dict[str, Any]:
    """Express a published template as an F25 executor parameter vector."""

    if (
        template.input_samples != PAA_BLOCK_COUNT
        or template.max_pool_size != 2
        or template.output_units != CLASS_COUNT
        or template.optimizer != "Adam"
        or template.dense_activation != "ReLU"
        or template.output_activation != "softmax"
        or not (
            len(template.convolution_filters)
            == len(template.convolution_kernel_sizes)
            == len(template.max_pool_after_layer)
        )
    ):
        raise F25ModelError("published template departs from the F25 CNN family")
    params: dict[str, Any] = {"n_conv_layers": len(template.convolution_filters)}
    for index, (filters, kernel, pool) in enumerate(
        zip(
            template.convolution_filters,
            template.convolution_kernel_sizes,
            template.max_pool_after_layer,
        )
    ):
        params[f"filters_l{index}"] = filters
        params[f"kernel_l{index}"] = kernel
        params[f"pool_l{index}"] = pool
    params["dense_units"] = template.dense_units
    params["batch_size"] = template.batch_size
    params["learning_rate"] = template.learning_rate
    return params


def feature_width(
    params: dict[str, Any], *, conv_padding: str, pool_ceil_mode: bool
) -> int:
    """Temporal width after the convolution stack, from the layer arithmetic."""

    normalized = normalize_model_params(params)
    width = PAA_BLOCK_COUNT
    for kernel, use_pool in zip(normalized["kernels"], normalized["pools"]):
        if conv_padding == "valid":
            width = width - kernel + 1
        elif conv_padding != "same":
            raise F25ModelError("conv_padding must be 'valid' or 'same'")
        if use_pool:
            width = math.ceil(width / 2) if pool_ceil_mode else width // 2
    return width


def analytic_parameter_count(params: dict[str, Any], flatten_units: int) -> int:
    """Keras/PyTorch parameter count of the published layer list."""

    normalized = normalize_model_params(params)
    count = 0
    in_channels = 1
    for filters, kernel in zip(normalized["filters"], normalized["kernels"]):
        count += in_channels * filters * kernel + filters
        in_channels = filters
    dense = normalized["dense_units"]
    return count + flatten_units * dense + dense + dense * CLASS_COUNT + CLASS_COUNT


@dataclass(frozen=True)
class FinalistSpec:
    finalist_id: str
    template_index: int
    conv_padding: str
    pool_ceil_mode: bool
    role: str
    framework_defaults: str = "pytorch"

    def __post_init__(self) -> None:
        if self.framework_defaults not in FRAMEWORK_DEFAULTS:
            raise F25ModelError(f"framework_defaults must be one of {FRAMEWORK_DEFAULTS}")

    @property
    def uses_keras_defaults(self) -> bool:
        return self.framework_defaults == "keras"

    @property
    def template(self) -> PublishedPAACNNTemplate:
        return PUBLISHED_PAA_CNN_TEMPLATES[self.template_index]

    @property
    def params(self) -> dict[str, Any]:
        return template_params(self.template)

    @property
    def sensor_id(self) -> str:
        return self.template.sensor_id

    @property
    def sensor_index(self) -> int:
        return CHANNELS.index(self.sensor_id)

    @property
    def configuration_id(self) -> str:
        """The F25-R report configuration trained on the same sensor."""

        return f"PAA__S{self.sensor_index:02d}"

    @property
    def flatten_units(self) -> int:
        width = feature_width(
            self.params,
            conv_padding=self.conv_padding,
            pool_ceil_mode=self.pool_ceil_mode,
        )
        return width * self.template.convolution_filters[-1]

    @property
    def parameter_count(self) -> int:
        return analytic_parameter_count(self.params, self.flatten_units)

    @property
    def uses_frozen_f25_model(self) -> bool:
        return self.conv_padding == "valid" and not self.pool_ceil_mode

    def describe(self) -> dict[str, Any]:
        # PyTorch-default specs omit the framework keys, so the fingerprints of
        # the runs made before the Keras arm existed still match on resume.
        keras = (
            {
                "framework_defaults": "keras",
                "initialization": "glorot_uniform kernels, zero biases (Keras Conv1D/Dense)",
                "optimizer_update": (
                    "Keras Adam: epsilon 1e-7 added to sqrt(v), bias corrections "
                    "folded into the step size"
                ),
            }
            if self.uses_keras_defaults
            else {}
        )
        return {
            **keras,
            "finalist_id": self.finalist_id,
            "role": self.role,
            "sensor_id": self.sensor_id,
            "sensor_index": self.sensor_index,
            "configuration_id": self.configuration_id,
            "model_class": (
                "core.f25_models.F25CNN"
                if self.uses_frozen_f25_model
                else f"{__name__}.PublishedFinalistCNN"
            ),
            "input_samples": self.template.input_samples,
            "convolution_filters": list(self.template.convolution_filters),
            "convolution_kernel_sizes": list(
                self.template.convolution_kernel_sizes
            ),
            "max_pool_after_layer": list(self.template.max_pool_after_layer),
            "conv_padding": self.conv_padding,
            "pool_ceil_mode": self.pool_ceil_mode,
            "flatten_units": self.flatten_units,
            "published_flatten_units": self.template.flatten_units,
            "dense_units": self.template.dense_units,
            "parameter_count": self.parameter_count,
            "batch_size": self.template.batch_size,
            "learning_rate": self.template.learning_rate,
        }


CARBODY = FinalistSpec(
    finalist_id="PAA-PUB__S00",
    template_index=0,
    conv_padding="valid",
    pool_ceil_mode=False,
    role="published finalist (Fernandes et al. 2025, Fig. 6(a))",
)
FRONT_BOGIE = FinalistSpec(
    finalist_id="PAA-PUB__S01",
    template_index=1,
    conv_padding="same",
    pool_ceil_mode=True,
    role="published finalist (Fernandes et al. 2025, Fig. 6(b))",
)
FRONT_BOGIE_VALID = FinalistSpec(
    finalist_id="PAA-PUB-VALID__S01",
    template_index=1,
    conv_padding="valid",
    pool_ceil_mode=False,
    role="padding sensitivity: front-bogie finalist with F25 valid/floor layers",
)
CARBODY_KERAS = FinalistSpec(
    finalist_id="PAA-PUB-KERAS__S00",
    template_index=0,
    conv_padding="valid",
    pool_ceil_mode=False,
    role="framework sensitivity: carbody finalist with Keras initializers and Adam",
    framework_defaults="keras",
)
FRONT_BOGIE_KERAS = FinalistSpec(
    finalist_id="PAA-PUB-KERAS__S01",
    template_index=1,
    conv_padding="same",
    pool_ceil_mode=True,
    role="framework sensitivity: front-bogie finalist with Keras initializers and Adam",
    framework_defaults="keras",
)
PUBLISHED_FINALISTS = (CARBODY, FRONT_BOGIE)
KERAS_FINALISTS = (CARBODY_KERAS, FRONT_BOGIE_KERAS)
ALL_FINALISTS = (CARBODY, FRONT_BOGIE, FRONT_BOGIE_VALID, *KERAS_FINALISTS)


class PublishedFinalistCNN(nn.Module):
    """``F25CNN`` with selectable convolution padding and pooling rounding.

    Layer order, attribute names and forward pass mirror ``F25CNN`` so that the
    valid/floor configuration has identical state-dict keys, initial weights
    (under one seed) and outputs; :func:`selftest` checks this.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        input_samples: int,
        params: dict[str, Any],
        conv_padding: str,
        pool_ceil_mode: bool,
        n_classes: int = CLASS_COUNT,
    ) -> None:
        super().__init__()
        if conv_padding not in {"valid", "same"}:
            raise F25ModelError("conv_padding must be 'valid' or 'same'")
        normalized = normalize_model_params(params)
        self.schema = MODEL_SCHEMA
        self.pooling_kind = "flatten+dense"
        self.conv_padding = conv_padding
        self.pool_ceil_mode = bool(pool_ceil_mode)
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
            # Integer zero padding for 'valid' keeps the kernels F25CNN uses;
            # 'same' matches Keras (extra pad on the right for even kernels).
            padding: int | str = 0 if conv_padding == "valid" else "same"
            modules.extend(
                [
                    nn.Conv1d(
                        current_channels, filters, kernel_size=kernel, padding=padding
                    ),
                    nn.ReLU(),
                ]
            )
            if conv_padding == "valid":
                current_samples = current_samples - kernel + 1
            if use_pool:
                if current_samples < 2:
                    raise F25ModelError("max pooling would erase the time dimension")
                modules.append(
                    nn.MaxPool1d(
                        kernel_size=2, stride=2, ceil_mode=self.pool_ceil_mode
                    )
                )
                current_samples = (
                    math.ceil(current_samples / 2)
                    if self.pool_ceil_mode
                    else current_samples // 2
                )
            current_channels = filters
        self.convolutions = nn.Sequential(*modules)
        self.aggregation = nn.Flatten(start_dim=1)
        self.flattened_units = current_channels * current_samples
        self.dense = nn.Linear(self.flattened_units, normalized["dense_units"])
        self.activation = nn.ReLU()
        self.output = nn.Linear(normalized["dense_units"], n_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[-1] != self.input_samples:
            raise F25ModelError(
                f"published finalist expects (batch, channels, {self.input_samples})"
            )
        tensor = self.convolutions(tensor)
        tensor = self.aggregation(tensor)
        tensor = self.activation(self.dense(tensor))
        return self.output(tensor)


def keras_glorot_limit(module: nn.Conv1d | nn.Linear) -> float:
    """Bound of Keras glorot_uniform, with Keras' fans (kernel size x channels)."""

    if isinstance(module, nn.Conv1d):
        kernel = module.kernel_size[0]
        fan_in, fan_out = module.in_channels * kernel, module.out_channels * kernel
    else:
        fan_in, fan_out = module.in_features, module.out_features
    return math.sqrt(6.0 / (fan_in + fan_out))


def keras_initialize_(model: nn.Module) -> nn.Module:
    """Re-initialize like Keras Conv1D/Dense: glorot_uniform kernels, zero biases.

    The draws come from the CPU generator, which the executor seeds, whatever
    the model's device, so CPU and GPU runs start from the same weights.  (The
    PyTorch default bound of the carbody dense layer, 1/sqrt(27,840), is 2.4x
    smaller than the glorot bound.)
    """

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                limit = keras_glorot_limit(module)
                kernel = torch.empty(module.weight.shape, dtype=module.weight.dtype)
                module.weight.copy_(kernel.uniform_(-limit, limit))
                module.bias.zero_()
    return model


class KerasAdam(torch.optim.Adam):
    """``torch.optim.Adam`` stepping exactly like ``tf.keras.optimizers.Adam``.

    Keras adds epsilon (default 1e-7) to sqrt(v) and folds both bias
    corrections into the step size,

        p -= lr * sqrt(1 - b2**t) / (1 - b1**t) * m / (sqrt(v) + eps),

    whereas PyTorch adds its epsilon (default 1e-8) after correcting v.  The two
    coincide when PyTorch uses eps / sqrt(1 - b2**t), which :meth:`step` sets.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1.0e-3,
        *,
        eps: float = KERAS_ADAM_EPSILON,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, lr=lr, eps=eps, **kwargs)
        self.keras_epsilon = eps
        self.keras_step = 0

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        self.keras_step += 1
        for group in self.param_groups:
            beta2 = group["betas"][1]
            group["eps"] = self.keras_epsilon / math.sqrt(1.0 - beta2**self.keras_step)
        return super().step(closure)


def build_finalist_model(
    spec: FinalistSpec, *, device: torch.device | str
) -> nn.Module:
    """Build one finalist on ``device`` from the current torch RNG state."""

    if spec.uses_frozen_f25_model:
        model: nn.Module = build_f25_model(
            arm_id=F25_ARM_ID, in_channels=1, params=spec.params, device=device
        )
    else:
        model = PublishedFinalistCNN(
            in_channels=1,
            input_samples=spec.template.input_samples,
            params=spec.params,
            conv_padding=spec.conv_padding,
            pool_ceil_mode=spec.pool_ceil_mode,
        ).to(device)
    if (
        model.flattened_units != spec.flatten_units
        or parameter_count(model) != spec.parameter_count
    ):
        raise F25ModelError(f"{spec.finalist_id} does not match its layer arithmetic")
    if spec.uses_keras_defaults:
        keras_initialize_(model)
    return model


def selftest() -> list[dict[str, Any]]:
    """Check flatten sizes, parameter counts, feature maps and F25 parity."""

    rows = []
    for spec in ALL_FINALISTS:
        torch.manual_seed(0)
        model = build_finalist_model(spec, device="cpu")
        with torch.no_grad():
            features = model.convolutions(torch.zeros(2, 1, PAA_BLOCK_COUNT))
            logits = model(torch.zeros(2, 1, PAA_BLOCK_COUNT))
        width = spec.flatten_units // spec.template.convolution_filters[-1]
        if (
            tuple(features.shape)
            != (2, spec.template.convolution_filters[-1], width)
            or tuple(logits.shape) != (2, CLASS_COUNT)
        ):
            raise F25ModelError(f"{spec.finalist_id} feature map is not {width} wide")
        if (
            spec in (*PUBLISHED_FINALISTS, *KERAS_FINALISTS)
            and spec.flatten_units != spec.template.flatten_units
        ):
            raise F25ModelError(
                f"{spec.finalist_id} flattens {spec.flatten_units} units, "
                f"published {spec.template.flatten_units}"
            )
        rows.append(
            {
                "finalist_id": spec.finalist_id,
                "feature_map": [width, spec.template.convolution_filters[-1]],
                "flatten_units": spec.flatten_units,
                "published_flatten_units": spec.template.flatten_units,
                "parameter_count": parameter_count(model),
                "model_class": type(model).__name__,
            }
        )
        if spec.uses_keras_defaults:
            _check_keras_initialization(spec, model)
        elif spec.uses_frozen_f25_model:
            # The mirror must be F25CNN when its two switches are off.
            torch.manual_seed(1)
            reference = F25CNN(
                in_channels=1, input_samples=PAA_BLOCK_COUNT, params=spec.params
            )
            torch.manual_seed(1)
            mirror = PublishedFinalistCNN(
                in_channels=1,
                input_samples=PAA_BLOCK_COUNT,
                params=spec.params,
                conv_padding="valid",
                pool_ceil_mode=False,
            )
            ref_state = reference.state_dict()
            mirror_state = mirror.state_dict()
            probe = torch.randn(3, 1, PAA_BLOCK_COUNT)
            if (
                list(ref_state) != list(mirror_state)
                or any(
                    not torch.equal(ref_state[key], mirror_state[key])
                    for key in ref_state
                )
                or not torch.equal(reference(probe), mirror(probe))
            ):
                raise F25ModelError(
                    f"PublishedFinalistCNN(valid) differs from F25CNN for {spec.finalist_id}"
                )
    _check_keras_adam()
    return rows


def _check_keras_initialization(spec: FinalistSpec, model: nn.Module) -> None:
    """Zero biases, kernels filling the glorot bound, same weights on any device."""

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            limit = keras_glorot_limit(module)
            peak = float(module.weight.abs().max())
            if torch.count_nonzero(module.bias) or not 0.9 * limit < peak <= limit:
                raise F25ModelError(
                    f"{spec.finalist_id} {name} is not glorot_uniform with zero bias"
                )
    if torch.cuda.is_available():
        states = []
        for device in ("cpu", "cuda"):
            torch.manual_seed(7)
            built = build_finalist_model(spec, device=device)
            states.append({key: value.cpu() for key, value in built.state_dict().items()})
        if any(not torch.equal(states[0][key], states[1][key]) for key in states[0]):
            raise F25ModelError(f"{spec.finalist_id} starts differently on CPU and GPU")


def _check_keras_adam(steps: int = 40) -> None:
    """KerasAdam against the tf.keras Adam update written out, in float64.

    Gradients of order 1e-6 make epsilon matter, so PyTorch's own update
    (epsilon added after the bias correction) fails the same comparison.
    """

    start = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    target = torch.linspace(0.5, -0.3, 7, dtype=torch.float64)
    lr, beta1, beta2 = 1.0e-3, 0.9, 0.999

    def gradient(point: torch.Tensor) -> torch.Tensor:
        return 1.0e-6 * (point - target)

    expected = start.clone()
    m = torch.zeros_like(start)
    v = torch.zeros_like(start)
    for t in range(1, steps + 1):
        g = gradient(expected)
        m += (g - m) * (1.0 - beta1)
        v += (g * g - v) * (1.0 - beta2)
        alpha = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
        expected -= m * alpha / (v.sqrt() + KERAS_ADAM_EPSILON)

    finals = []
    for optimizer_class in (KerasAdam, torch.optim.Adam):
        point = start.clone().requires_grad_(True)
        optimizer = optimizer_class([point], lr=lr, eps=KERAS_ADAM_EPSILON)
        for _ in range(steps):
            point.grad = gradient(point.detach())
            optimizer.step()
        finals.append(point.detach())
    if not torch.allclose(finals[0], expected, rtol=0.0, atol=1.0e-12):
        raise F25ModelError("KerasAdam does not reproduce the Keras Adam update")
    if torch.allclose(finals[1], expected, rtol=0.0, atol=1.0e-6):
        raise F25ModelError("the Keras Adam check cannot tell Keras from PyTorch Adam")


if __name__ == "__main__":
    for row in selftest():
        print(
            f"{row['finalist_id']:<20} {row['model_class']:<22} feature map "
            f"{row['feature_map'][0]} x {row['feature_map'][1]} = "
            f"{row['flatten_units']:>6,} flattened (published "
            f"{row['published_flatten_units']:,}), "
            f"{row['parameter_count']:,} parameters"
        )
    print(
        "PASS published-finalist model self-test (F25CNN parity, Keras "
        "initialization, Keras Adam update)"
    )
