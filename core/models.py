"""
core/models.py
==============
All PyTorch model classes used in the ablation study and the digital twin.

Imported by:
    ablation.ipynb      - training, robustness evaluation, export
    drive_by_DT.py      - loading champion weights for online inference

Do NOT import training or plotting modules here; this file has no
side-effects and no dependency outside torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.task import n_outputs as _task_n_outputs
from core.temporal_pooling import deterministic_adaptive_max_pool1d


# ──────────────────────────────────────────────────────────────────────────────
# 1. Time2Vec-style Spatial-Coordinate Encoding
# ──────────────────────────────────────────────────────────────────────────────

class Time2VecPositionEncoding(nn.Module):
    """
    Time2Vec-style encoding of a normalized spatial coordinate.

    The input coordinate is position along the sampled response, not clock
    time. The module borrows Time2Vec's one-linear-plus-periodic functional
    form and concatenates the resulting coordinate features to the response
    channels before the CNN. It has no sequence-length-dependent parameters.

    Args:
        seq_len (int | None): Deprecated compatibility metadata. It does not
            affect parameters or the accepted input length.
        out_features (int): Total embedding dimension (1 linear + out_features-1 periodic).
    """

    def __init__(self, seq_len: int | None = None, out_features: int = 8):
        super().__init__()
        if seq_len is not None and (
            isinstance(seq_len, bool) or int(seq_len) != seq_len or seq_len < 1
        ):
            raise ValueError("seq_len must be a positive integer when supplied")
        if (
            isinstance(out_features, bool)
            or int(out_features) != out_features
            or out_features < 1
        ):
            raise ValueError("out_features must be a positive integer")
        seq_len = None if seq_len is None else int(seq_len)
        out_features = int(out_features)
        self.seq_len = seq_len
        self.out_features = out_features

        self.w_linear = nn.Parameter(torch.randn(1, 1))
        self.p_linear = nn.Parameter(torch.randn(1, 1))

        self.w_periodic = nn.Parameter(torch.randn(out_features - 1, 1))
        self.p_periodic = nn.Parameter(torch.randn(out_features - 1, 1))

    def forward(self, x_space: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_space: (Batch, 1, Sequence_Length)  - normalised [0, 1] position vector.
        Returns:
            (Batch, out_features, Sequence_Length)
        """
        if x_space.ndim != 3 or x_space.size(1) != 1:
            raise ValueError(
                "x_space must have shape (batch, 1, sequence_length)"
            )
        linear   = self.w_linear   * x_space + self.p_linear
        periodic = torch.sin(self.w_periodic * x_space + self.p_periodic)
        return torch.cat([linear, periodic], dim=1)


# Historical configs and checkpoint metadata use ``Space2Vec`` and the
# ``space2vec.*`` state-dict prefix. Retain the import alias and module
# attribute name, but keep the implementation and documentation truthful.
Space2Vec = Time2VecPositionEncoding


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-Rate Pooling Module (N-HiTS style)
# ──────────────────────────────────────────────────────────────────────────────

class MultiRatePooling1D(nn.Module):
    """
    Fixed-width adaptive temporal-pyramid max pooling.

    Each configured level produces exactly that many temporal bins, regardless
    of the input sequence length. Concatenating the levels therefore yields
    ``channels * sum(pool_rates)`` features for both RAW and PAA inputs. The
    historical ``pool_rates`` name is retained for configuration compatibility;
    its values now denote adaptive pyramid bin counts, not stride factors.

    Args:
        pool_rates (tuple[int]): Positive adaptive output-bin counts, e.g.
                                 (1, 2, 4).
    """

    def __init__(self, pool_rates: tuple = (1, 2, 4)):
        super().__init__()
        try:
            levels = tuple(pool_rates)
        except TypeError as exc:
            raise ValueError("pool_rates must be a non-empty sequence") from exc
        if not levels:
            raise ValueError("pool_rates must be a non-empty sequence")
        if any(
            isinstance(level, bool)
            or not isinstance(level, int)
            or level < 1
            for level in levels
        ):
            raise ValueError(
                "every adaptive pyramid bin count must be a positive integer"
            )
        if len(set(levels)) != len(levels):
            raise ValueError("adaptive pyramid bin counts must be distinct")
        self.pool_rates = levels
        self.pyramid_bins = levels
        self.output_bins = sum(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Features, Sequence_Length)
        Returns:
            (Batch, Features * sum(pool_rates)) - fixed-width 2-D tensor.
        """
        if x.ndim != 3 or x.size(-1) < 1:
            raise ValueError(
                "x must have shape (batch, features, non-empty sequence_length)"
            )
        pooled_outputs = []
        for output_bins in self.pyramid_bins:
            pooled = deterministic_adaptive_max_pool1d(x, output_bins)
            pooled_outputs.append(pooled.reshape(x.size(0), -1))
        return torch.cat(pooled_outputs, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Modular 1-D Network (ablation backbone)
# ──────────────────────────────────────────────────────────────────────────────

class SpaceAwareModularNetwork(nn.Module):
    """
    Fully modular architecture supporting dynamic insertion of a Time2Vec-style
    spatial-coordinate encoding, LSTM, and multi-rate pooling blocks.

    Architecture flags (all bool, all default False):
        use_space2vec  - prepend spatial-coordinate encoding before CNN.
        use_lstm       - insert a unidirectional LSTM after CNN.
        use_nhits      - replace global-average-pool with multi-rate pool.

    Args:
        n_segments (int):   Sequence length after preprocessing.
        n_classes  (int):   Number of damage classes.
        in_channels (int):  Number of active DOFs (input channels).
        params (dict):      Hyperparameter dict from Optuna or a saved study.
        use_space2vec (bool): Include Time2Vec-style position encoding.
        use_lstm      (bool): Include LSTM layer.
        use_nhits     (bool): Use multi-rate pooling instead of GAP.
        s2v_features  (int):  Output dimension of the position encoding.
    """

    def __init__(
        self,
        n_segments:    int,
        n_classes:     int,
        in_channels:   int,
        params:        dict,
        use_space2vec: bool = False,
        use_lstm:      bool = False,
        use_nhits:     bool = False,
        s2v_features:  int  = 8,
    ):
        super().__init__()
        self.params        = params
        self.use_space2vec = use_space2vec
        self.use_lstm      = use_lstm
        self.use_nhits     = use_nhits
        self.n_segments    = n_segments

        # ── 1. Position encoding (legacy state key: space2vec) ────────────────
        if self.use_space2vec:
            self.space2vec   = Time2VecPositionEncoding(
                seq_len=n_segments, out_features=s2v_features
            )
            cnn_in_channels  = in_channels + s2v_features
        else:
            cnn_in_channels  = in_channels

        # ── 2. Dynamic CNN layers ─────────────────────────────────────────────
        self.cnn_layers    = nn.ModuleList()
        n_conv_layers      = params.get('n_conv_layers', 2)

        for i in range(n_conv_layers):
            out_ch  = params[f'n_filters_l{i}']
            k_size  = params[f'kernel_size_l{i}']
            self.cnn_layers.append(nn.Conv1d(cnn_in_channels, out_ch, kernel_size=k_size, padding='same'))
            self.cnn_layers.append(nn.ReLU())
            if params.get(f'pooling_l{i}', False):
                self.cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            cnn_in_channels = out_ch

        # ── 3. Optional LSTM ──────────────────────────────────────────────────
        if self.use_lstm:
            lstm_hidden      = params.get('lstm_hidden_size', 64)
            lstm_layers      = params.get('lstm_num_layers', 1)
            lstm_dropout     = params.get('lstm_dropout', 0.2) if lstm_layers > 1 else 0.0
            self.lstm        = nn.LSTM(
                input_size=cnn_in_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            current_features = lstm_hidden
        else:
            current_features = cnn_in_channels

        # ── 4. Pooling / sub-sampling ─────────────────────────────────────────
        if self.use_nhits:
            pool_rates           = params.get('nhits_pool_rates', (1, 2, 4))
            self.multi_rate_pool = MultiRatePooling1D(pool_rates=pool_rates)
            flattened_size = current_features * self.multi_rate_pool.output_bins
        else:
            flattened_size = current_features  # global-average-pool -> scalar per channel

        # ── 5. Dense classification head ──────────────────────────────────────
        self.dense_layers = nn.ModuleList()
        n_dense_layers    = params.get('n_dense_layers', 1)
        in_features       = flattened_size

        for i in range(n_dense_layers):
            out_features = params[f'n_dense_units_l{i}']
            self.dense_layers.append(nn.Linear(in_features, out_features))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(params.get(f'dropout_l{i}', 0.2)))
            in_features = out_features

        self.final_layer = nn.Linear(in_features, n_classes)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, in_channels, n_segments)
        Returns:
            logits: (Batch, n_classes)
        """
        batch_size = x.size(0)

        # 1. Spatial-coordinate encoding. Use the live sequence length so one
        # fixed parameterization accepts both RAW and PAA representations.
        if self.use_space2vec:
            space_vec = torch.linspace(
                0, 1, steps=x.size(-1), device=x.device, dtype=x.dtype
            )
            space_vec = space_vec.view(1, 1, -1).expand(batch_size, 1, -1)
            x = torch.cat([x, self.space2vec(space_vec)], dim=1)

        # 2. CNN
        for layer in self.cnn_layers:
            x = layer(x)

        # 3. LSTM  - expects (Batch, SeqLen, Features)
        if self.use_lstm:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1)

        # 4. Pooling
        if self.use_nhits:
            x = self.multi_rate_pool(x)   # -> (Batch, flattened)
        else:
            x = torch.mean(x, dim=2)      # global average pool -> (Batch, Features)

        # 5. Dense head
        for layer in self.dense_layers:
            x = layer(x)

        return self.final_layer(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 2-D CNN for CWT scalograms
# ──────────────────────────────────────────────────────────────────────────────

class Simple2DCNN(nn.Module):
    """
    Dynamic 2-D CNN designed for Continuous Wavelet Transform scalograms.

    Input shape: (Batch, Channels, Scales_Height, Sequence_Width)

    Args:
        in_channels  (int): Number of active DOFs.
        n_classes    (int): Number of damage classes.
        params       (dict): Optuna hyperparameter dict.
        image_height (int): Number of CWT scale bins (default 64).
        image_width  (int): Sequence length after PAA downsampling (default 512).
    """

    def __init__(
        self,
        in_channels:  int,
        n_classes:    int,
        params:       dict,
        image_height: int = 64,
        image_width:  int = 512,
    ):
        super().__init__()
        self.params = params
        self.layers = nn.ModuleList()

        current_h        = image_height
        current_w        = image_width
        current_channels = in_channels
        n_conv_layers    = params['n_conv_layers']

        # ── 2-D convolutional layers ──────────────────────────────────────────
        for i in range(n_conv_layers):
            out_ch  = params[f'n_filters_l{i}']
            k_size  = params[f'kernel_size_l{i}']   # single int -> (k, k) square kernel
            self.layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=k_size, padding='same'))
            self.layers.append(nn.ReLU())
            if params.get(f'pooling_l{i}', False):
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_h //= 2
                current_w //= 2
            current_channels = out_ch

        # ── Flatten ───────────────────────────────────────────────────────────
        self.layers.append(nn.Flatten())
        flattened_size = current_channels * current_h * current_w

        # ── Dense head ────────────────────────────────────────────────────────
        n_dense_layers = params['n_dense_layers']
        in_features    = flattened_size

        for i in range(n_dense_layers):
            out_features = params[f'n_dense_units_l{i}']
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(params.get(f'dropout_l{i}', 0.2)))
            in_features = out_features

        self.layers.append(nn.Linear(in_features, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Channels, Scales_Height, Sequence_Width)
        Returns:
            logits: (Batch, n_classes)
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 5. Factory function
# ──────────────────────────────────────────────────────────────────────────────

# Legacy N-HiTS key -> adaptive pyramid-bin tuple for Optuna categorical storage.
_POOL_RATE_MAP: dict[str, tuple] = {
    "1_2_4":   (1, 2, 4),
    "1_4_8":   (1, 4, 8),
    "1_3_6":   (1, 3, 6),
    "1_2_4_8": (1, 2, 4, 8),
}

def build_model(
    config:      dict,
    params:      dict,
    input_shape: tuple,
    device:      torch.device,
) -> tuple[nn.Module, int]:
    """
    Universal factory that routes to the correct architecture based on `config`.

    Args:
        config (dict): Ablation step config - keys: 'method', 'model_type',
                       'use_space2vec', 'use_lstm', 'use_nhits', 'discretization'.
        params (dict): Optuna best_params (or any compatible dict).
        input_shape (tuple): Shape of the processed data array, e.g.
                             (N, channels, seq_len) for 1-D models or
                             (N, channels, scales, seq_len) for 2-D.
        device (torch.device): Target device.

    Returns:
        (model, n_outputs) - n_outputs is the final-layer size: the number of
        damage CLASSES for a classification config, or the number of target
        supports (continuous heads) for a regression config (config['task']).
        The architecture is identical either way; only the head size changes.
    """
    # Resolve the legacy string key to adaptive pyramid bin counts.
    if 'nhits_pool_rates_key' in params:
        params = dict(params)   # don't mutate the caller's dict
        params['nhits_pool_rates'] = _POOL_RATE_MAP[params['nhits_pool_rates_key']]

    # Output size from the task: classes (classification) or target count
    # (regression). Defaults to classification when config has no 'task' key,
    # so existing single-scour champions and the DT are unaffected.
    n_out = _task_n_outputs(config)
    in_channels = input_shape[1]

    model_type = str(config.get("model_type", "1D_MODULAR")).upper()

    method = config.get("method")
    if method == "PAA_CWT" and model_type not in {"1D_MODULAR", "2D_CNN"}:
        raise ValueError(
            f"model_type {model_type!r} consumes 1-D RAW/PAA tensors and is "
            "incompatible with the 2-D PAA_CWT representation"
        )

    if model_type in {"MODERN_TCN", "TSLANET"}:
        active_legacy_flags = [
            name
            for name in ("use_space2vec", "use_lstm", "use_nhits")
            if config.get(name, False)
        ]
        if active_legacy_flags:
            raise ValueError(
                f"model_type {model_type!r} is incompatible with active "
                f"legacy modules {active_legacy_flags!r}"
            )
        from core.challenger_policy import validate_challenger_parameters

        params = validate_challenger_parameters(model_type, params)

    if model_type == "2D_CNN" or method == "PAA_CWT":
        model = Simple2DCNN(
            in_channels=in_channels,
            n_classes=n_out,
            params=params,
            image_height=input_shape[2],
            image_width=input_shape[3],
        ).to(device)
    elif model_type == "1D_MODULAR":
        model = SpaceAwareModularNetwork(
            n_segments=input_shape[2],
            n_classes=n_out,
            in_channels=in_channels,
            params=params,
            use_space2vec=config.get('use_space2vec', False),
            use_lstm=config.get('use_lstm',      False),
            use_nhits=config.get('use_nhits',     False),
        ).to(device)
    elif model_type == "MODERN_TCN":
        from core.challenger_policy import (
            CHALLENGER_FACTORY_POLICY,
            CHALLENGER_PATCH_LAYOUTS,
            CHALLENGER_POOLING_LAYOUTS,
            MODERN_TCN_LAYOUTS,
        )
        from core.modern_tcn import ModernTCNRegressor

        fixed = CHALLENGER_FACTORY_POLICY[model_type]
        patch_size, patch_stride = CHALLENGER_PATCH_LAYOUTS[
            params["modern_patch_layout"]
        ]
        layout = MODERN_TCN_LAYOUTS[params["modern_layout"]]
        pool_bins = CHALLENGER_POOLING_LAYOUTS[params["challenger_pooling"]]
        model = ModernTCNRegressor(
            in_channels=in_channels,
            output_dim=n_out,
            dims=layout["dims"],
            depths=layout["depths"],
            kernel_size=params["modern_kernel_size"],
            patch_size=patch_size,
            patch_stride=patch_stride,
            stage_stride=fixed["stage_stride"],
            small_kernel_size=fixed["small_kernel_size"],
            expansion_ratio=params["modern_expansion_ratio"],
            dropout=params["modern_dropout"],
            small_kernel_merged=fixed["small_kernel_merged"],
            activation_checkpointing=fixed["activation_checkpointing"],
            depthwise_channel_chunk_size=(
                fixed["depthwise_channel_chunk_size"]
            ),
            pool_bins=pool_bins,
        ).to(device)
    elif model_type == "TSLANET":
        from core.challenger_policy import (
            CHALLENGER_FACTORY_POLICY,
            CHALLENGER_PATCH_LAYOUTS,
            CHALLENGER_POOLING_LAYOUTS,
        )
        from core.tslanet import TSLANetRegressor

        fixed = CHALLENGER_FACTORY_POLICY[model_type]
        patch_size, patch_stride = CHALLENGER_PATCH_LAYOUTS[
            params["tsla_patch_layout"]
        ]
        pool_bins = CHALLENGER_POOLING_LAYOUTS[params["challenger_pooling"]]
        model = TSLANetRegressor(
            in_channels=in_channels,
            output_dim=n_out,
            embed_dim=params["tsla_embed_dim"],
            depth=params["tsla_depth"],
            patch_size=patch_size,
            patch_stride=patch_stride,
            mlp_ratio=params["tsla_mlp_ratio"],
            dropout=params["tsla_dropout"],
            drop_path_rate=params[fixed["drop_path_rate_source"]],
            use_asb=fixed["use_asb"],
            use_icb=fixed["use_icb"],
            adaptive_filter=fixed["adaptive_filter"],
            spectral_threshold_init=fixed["spectral_threshold_init"],
            spectral_mask_temperature=fixed["spectral_mask_temperature"],
            spectral_eps=fixed["spectral_eps"],
            position_bins=fixed["position_bins"],
            head_hidden_dim=fixed["head_hidden_dim"],
            pool_bins=pool_bins,
        ).to(device)
    else:
        raise ValueError(
            f"unsupported model_type {model_type!r}; expected 1D_MODULAR, "
            "2D_CNN, MODERN_TCN, or TSLANET"
        )

    return model, n_out
