"""
core/models.py
==============
All PyTorch model classes used in the ablation study and the digital twin.

Imported by:
    ablation.ipynb      — training, robustness evaluation, export
    drive_by_DT.py      — loading champion weights for online inference

Do NOT import training or plotting modules here; this file has no
side-effects and no dependency outside torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.task import n_outputs as _task_n_outputs


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spatial Embedding Module
# ──────────────────────────────────────────────────────────────────────────────

class Space2Vec(nn.Module):
    """
    Learnable spatial embedding layer (adapted from Time2Vec).

    Grounds the vibration signal to physical coordinates on the bridge
    by concatenating a linear and a set of periodic position features
    to the raw channel tensor before the CNN layers.

    Args:
        seq_len (int): Length of the input sequence (number of spatial samples).
        out_features (int): Total embedding dimension (1 linear + out_features-1 periodic).
    """

    def __init__(self, seq_len: int, out_features: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.out_features = out_features

        self.w_linear = nn.Parameter(torch.randn(1, 1))
        self.p_linear = nn.Parameter(torch.randn(1, 1))

        self.w_periodic = nn.Parameter(torch.randn(out_features - 1, 1))
        self.p_periodic = nn.Parameter(torch.randn(out_features - 1, 1))

    def forward(self, x_space: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_space: (Batch, 1, Sequence_Length)  — normalised [0, 1] position vector.
        Returns:
            (Batch, out_features, Sequence_Length)
        """
        linear   = self.w_linear   * x_space + self.p_linear
        periodic = torch.sin(self.w_periodic * x_space + self.p_periodic)
        return torch.cat([linear, periodic], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-Rate Pooling Module (N-HiTS style)
# ──────────────────────────────────────────────────────────────────────────────

class MultiRatePooling1D(nn.Module):
    """
    Extracts features at multiple temporal resolutions by sub-sampling
    the sequence at different pooling rates, then concatenating all views.

    Args:
        pool_rates (tuple[int]): Downsampling factors, e.g. (1, 2, 4).
                                 Rate 1 = no pooling (full resolution).
    """

    def __init__(self, pool_rates: tuple = (1, 2, 4)):
        super().__init__()
        self.pool_rates = pool_rates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Features, Sequence_Length)
        Returns:
            (Batch, sum_of_pooled_features)  — flat 2-D tensor.
        """
        pooled_outputs = []
        for rate in self.pool_rates:
            pooled = F.max_pool1d(x, kernel_size=rate, stride=rate) if rate > 1 else x
            pooled_outputs.append(pooled.flatten(start_dim=1))
        return torch.cat(pooled_outputs, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Modular 1-D Network (ablation backbone)
# ──────────────────────────────────────────────────────────────────────────────

class SpaceAwareModularNetwork(nn.Module):
    """
    Fully modular architecture supporting dynamic insertion of Space2Vec,
    LSTM, and N-HiTS blocks for physical ablation studies.

    Architecture flags (all bool, all default False):
        use_space2vec  — prepend spatial embedding before CNN.
        use_lstm       — insert a bidirectional LSTM after CNN.
        use_nhits      — replace global-average-pool with multi-rate pool.

    Args:
        n_segments (int):   Sequence length after preprocessing.
        n_classes  (int):   Number of damage classes.
        in_channels (int):  Number of active DOFs (input channels).
        params (dict):      Hyperparameter dict from Optuna or a saved study.
        use_space2vec (bool): Include Space2Vec layer.
        use_lstm      (bool): Include LSTM layer.
        use_nhits     (bool): Use multi-rate pooling instead of GAP.
        s2v_features  (int):  Output dimension of Space2Vec embedding.
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

        # ── 1. Space2Vec ──────────────────────────────────────────────────────
        if self.use_space2vec:
            self.space2vec   = Space2Vec(seq_len=n_segments, out_features=s2v_features)
            cnn_in_channels  = in_channels + s2v_features
        else:
            cnn_in_channels  = in_channels

        # ── 2. Dynamic CNN layers ─────────────────────────────────────────────
        self.cnn_layers    = nn.ModuleList()
        current_seq_len    = n_segments
        n_conv_layers      = params.get('n_conv_layers', 2)

        for i in range(n_conv_layers):
            out_ch  = params[f'n_filters_l{i}']
            k_size  = params[f'kernel_size_l{i}']
            self.cnn_layers.append(nn.Conv1d(cnn_in_channels, out_ch, kernel_size=k_size, padding='same'))
            self.cnn_layers.append(nn.ReLU())
            if params.get(f'pooling_l{i}', False):
                self.cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_seq_len //= 2
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
            flattened_size = sum(
                current_features * (current_seq_len // rate) for rate in pool_rates
            )
        else:
            flattened_size = current_features  # global-average-pool → scalar per channel

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

        # 1. Spatial embedding
        if self.use_space2vec:
            space_vec = torch.linspace(0, 1, steps=self.n_segments, device=x.device)
            space_vec = space_vec.view(1, 1, -1).expand(batch_size, 1, -1)
            x = torch.cat([x, self.space2vec(space_vec)], dim=1)

        # 2. CNN
        for layer in self.cnn_layers:
            x = layer(x)

        # 3. LSTM  — expects (Batch, SeqLen, Features)
        if self.use_lstm:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1)

        # 4. Pooling
        if self.use_nhits:
            x = self.multi_rate_pool(x)   # → (Batch, flattened)
        else:
            x = torch.mean(x, dim=2)      # global average pool → (Batch, Features)

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
            k_size  = params[f'kernel_size_l{i}']   # single int → (k, k) square kernel
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

# N-HiTS pool-rate string → tuple, for Optuna's categorical storage format.
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
        config (dict): Ablation step config — keys: 'method', 'model_type',
                       'use_space2vec', 'use_lstm', 'use_nhits', 'discretization'.
        params (dict): Optuna best_params (or any compatible dict).
        input_shape (tuple): Shape of the processed data array, e.g.
                             (N, channels, seq_len) for 1-D models or
                             (N, channels, scales, seq_len) for 2-D.
        device (torch.device): Target device.

    Returns:
        (model, n_outputs) — n_outputs is the final-layer size: the number of
        damage CLASSES for a classification config, or the number of target
        supports (continuous heads) for a regression config (config['task']).
        The architecture is identical either way; only the head size changes.
    """
    # Resolve N-HiTS string key back to a tuple (Optuna stores it as a string).
    if 'nhits_pool_rates_key' in params:
        params = dict(params)   # don't mutate the caller's dict
        params['nhits_pool_rates'] = _POOL_RATE_MAP[params['nhits_pool_rates_key']]

    # Output size from the task: classes (classification) or target count
    # (regression). Defaults to classification when config has no 'task' key,
    # so existing single-scour champions and the DT are unaffected.
    n_out = _task_n_outputs(config)
    in_channels = input_shape[1]

    if config.get('model_type') == '2D_CNN' or config.get('method') == 'PAA_CWT':
        model = Simple2DCNN(
            in_channels=in_channels,
            n_classes=n_out,
            params=params,
            image_height=input_shape[2],
            image_width=input_shape[3],
        ).to(device)
    else:
        model = SpaceAwareModularNetwork(
            n_segments=input_shape[2],
            n_classes=n_out,
            in_channels=in_channels,
            params=params,
            use_space2vec=config.get('use_space2vec', False),
            use_lstm=config.get('use_lstm',      False),
            use_nhits=config.get('use_nhits',     False),
        ).to(device)

    return model, n_out