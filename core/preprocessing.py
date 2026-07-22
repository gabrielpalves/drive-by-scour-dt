"""
core/preprocessing.py
=====================
Signal preprocessing for the TTBI ablation and digital twin pipelines.

The single public class, TTBIPreprocessor, is the canonical transform used
in both scripts.  The DT's old fft_preprocess() function is retired - 
DigitalAsset.estimate_state() should call preprocessor.transform() instead.

Imported by:
    core/dataset.py         - cache-miss path (fit + transform on raw data)
    digital_twin/assets.py  - online inference (transform only, scaler pre-loaded)
"""

import time
import pickle

import numpy as np
import pywt
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TTBIPreprocessor:
    """
    Unified signal preprocessor for multi-channel TTBI vibration data.

    Supports four methods, selectable at construction time:

        'raw'     - no signal transform; only scale.
        'paa'     - Piecewise Aggregate Approximation via linear interpolation.
        'fft'     - one-sided magnitude spectrum, resampled to n_segments bins.
        'cwt'     - Continuous Wavelet Transform (Morlet); produces 2-D scalograms.
                    PAA is applied first to prevent OOM on long sequences.

    Scaling is always physics-preserving and applied per channel:
        1-D methods  -> StandardScaler  (preserves zero-crossings of vibration signals)
        2-D / CWT    -> MinMaxScaler    (preserves image-like intensity contrast)

    Scaler fitting is leak-free: call transform(..., fit_scaler=True,
    fit_indices=train_idx) to fit only on training samples, then transform
    the full dataset with the fitted scaler in the same call.

    Args:
        method     (str): One of 'raw', 'paa', 'fft', 'cwt' (case-insensitive).
                          'paa_cwt' is accepted as an alias for 'cwt'.
        n_segments (int): Target sequence length for PAA / FFT bin count
                          and the time-axis width of CWT scalograms.
        cwt_scales (int): Number of frequency scales for CWT (scalogram height).
    """

    def __init__(self, method: str = 'raw', n_segments: int = 512, cwt_scales: int = 64):
        self.method     = method.lower()
        self.n_segments = n_segments
        self.cwt_scales = cwt_scales
        self.is_fit     = False

        if self.method in ('cwt', 'paa_cwt'):
            # 2-D scalograms are strictly positive -> MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            # 1-D vibration signals oscillate around zero -> StandardScaler
            self.scaler = StandardScaler()

    # ──────────────────────────────────────────────────────────────────────────
    # Private signal transforms
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_paa(self, X: np.ndarray) -> np.ndarray:
        """
        True Piecewise Aggregate Approximation (Keogh et al. 2001): split the
        sequence axis into n_segments equal-width windows and AVERAGE each
        window. Non-divisible lengths use exact fractional windows (every raw
        sample contributes to each window in proportion to its overlap),
        computed from the cumulative integral of the sample-and-hold signal —
        so the global mean is preserved exactly and every sample carries equal
        total weight.

        FIX 2026-07-22 (external audit r3, verified): the pre-fix version was
        a LINEAR-INTERPOLATION RESAMPLER (point subsampling, no averaging /
        anti-aliasing) — e.g. [0,0,0,4] -> 2 segments gave [0,4]; true PAA
        gives [0,2]. That contradicted the PAA definition cited in the
        methodology (Keogh; Fernandes 2025) and under-filtered exactly the
        high-frequency channels (wheels). Any change here is covered by the
        cache schema tag (gs6) + protocol hash ("paa_impl").

        Args:
            X: (Samples, Channels, Length)
        Returns:
            (Samples, Channels, n_segments)  float32
        """
        n_samp, n_chan, length = X.shape
        n = self.n_segments
        if length == n:
            return np.asarray(X, dtype=np.float32)
        # Window edges in sample coordinates; I(t) = integral of the
        # sample-and-hold signal is piecewise-linear between integer t, so
        # each window mean is (I(hi) - I(lo)) / (hi - lo), exact for
        # fractional edges. float64 accumulation; chunked over samples to
        # bound the transient cumsum memory.
        edges = np.linspace(0.0, float(length), n + 1)
        idx   = np.minimum(edges.astype(np.int64), length - 1)
        frac  = (edges - idx)[None, None, :]
        seg_w = float(length) / n
        out = np.empty((n_samp, n_chan, n), dtype=np.float32)
        for s0 in range(0, n_samp, 512):
            blk  = X[s0:s0 + 512]
            csum = np.concatenate(
                [np.zeros(blk.shape[:2] + (1,), dtype=np.float64),
                 np.cumsum(blk, axis=2, dtype=np.float64)], axis=2)
            integ = csum[..., idx] + frac * (csum[..., idx + 1] - csum[..., idx])
            out[s0:s0 + 512] = ((integ[..., 1:] - integ[..., :-1]) / seg_w)
        return out

    def _apply_fft(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the one-sided magnitude spectrum of each channel, then
        resample to n_segments frequency bins via PAA.

        Args:
            X: (Samples, Channels, Length)
        Returns:
            (Samples, Channels, n_segments)  float32
        """
        fft_mag = np.abs(np.fft.rfft(X, axis=2))
        return self._apply_paa(fft_mag)

    def _apply_cwt(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Morlet CWT scalograms for every channel of every sample,
        parallelised across samples with joblib.

        PAA is applied first to cap sequence length and prevent OOM.

        Args:
            X: (Samples, Channels, Length)
        Returns:
            (Samples, Channels, cwt_scales, n_segments)  float32
        """
        X_down              = self._apply_paa(X)
        samples, channels, length = X_down.shape
        scales              = np.arange(1, self.cwt_scales + 1)

        print(f"  -> Computing CWT (shape: {samples}x{channels}x{self.cwt_scales}x{length})...")
        t0 = time.perf_counter()

        def _process_sample(i: int) -> np.ndarray:
            out = np.zeros((channels, self.cwt_scales, length), dtype=np.float32)
            for c in range(channels):
                coeffs, _ = pywt.cwt(X_down[i, c, :], scales, 'morl')
                out[c]    = np.abs(coeffs)
            return out

        results = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(_process_sample)(i) for i in range(samples)
        )
        print(f"  -> CWT done in {time.perf_counter() - t0:.2f}s")
        return np.stack(results)   # (Samples, Channels, cwt_scales, n_segments)

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def transform(
        self,
        X:           np.ndarray,
        fit_scaler:  bool            = False,
        fit_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the selected signal transform, then scale per channel.

        Leak-free scaling
        -----------------
        Pass fit_scaler=True and fit_indices=train_idx on the first call
        (typically the cache-miss path in dataset.py).  The scaler is fitted
        exclusively on the training partition and then applied to the full
        array - no test-set statistics bleed into the scaler.

        On subsequent calls (e.g. online inference in the digital twin) pass
        fit_scaler=False; the previously fitted scaler is used as-is.

        Args:
            X           (np.ndarray): Raw data, shape (Samples, Channels, Length)
                                      or (Samples, Channels, Scales, Length) if
                                      already processed externally.
            fit_scaler  (bool):       Fit the internal scaler before transforming.
            fit_indices (np.ndarray): Integer indices of training samples.
                                      Required when fit_scaler=True to prevent leakage.
                                      If None, the entire X is used to fit (only safe
                                      when X is already the training partition).

        Returns:
            Scaled array of the same dtype (float32), shape:
                1-D methods -> (Samples, Channels, n_segments)
                CWT         -> (Samples, Channels, cwt_scales, n_segments)

        Raises:
            ValueError:  Unknown method string.
            RuntimeError: transform called before scaler was fitted.
        """
        # ── 1. Signal transform ───────────────────────────────────────────────
        method_map = {
            'raw':     lambda x: x,
            'paa':     self._apply_paa,
            'fft':     self._apply_fft,
            'cwt':     self._apply_cwt,
            'paa_cwt': self._apply_cwt,
        }
        if self.method not in method_map:
            raise ValueError(
                f"Unknown preprocessing method '{self.method}'. "
                f"Choose from: {list(method_map)}."
            )
        X_processed = method_map[self.method](X)

        # ── 2. Per-channel scaling ────────────────────────────────────────────
        # The scaler always operates on a 2-D matrix of shape
        # (total_observations, n_channels), regardless of whether the data
        # is 1-D or 2-D.  We transpose so channels become columns, flatten
        # the spatial dimensions into rows, scale, then invert.

        original_shape = X_processed.shape
        channels       = original_shape[1]

        # Move channels to last axis before flattening
        if X_processed.ndim == 3:           # (S, C, L)
            X_t = np.transpose(X_processed, (0, 2, 1))        # -> (S, L, C)
        elif X_processed.ndim == 4:         # (S, C, Sc, L)
            X_t = np.transpose(X_processed, (0, 2, 3, 1))     # -> (S, Sc, L, C)
        else:
            raise ValueError(f"Unexpected array ndim={X_processed.ndim} after transform.")

        X_flat = X_t.reshape(-1, channels)  # (S*spatial, C)

        if fit_scaler:
            if fit_indices is not None:
                # Reshape back to isolate training samples, then flatten
                X_train_t    = X_t[fit_indices].reshape(-1, channels)
                self.scaler.fit(X_train_t)
            else:
                self.scaler.fit(X_flat)
            self.is_fit = True
        else:
            if not self.is_fit:
                raise RuntimeError(
                    "Scaler has not been fitted yet. "
                    "Call transform(..., fit_scaler=True) on training data first, "
                    "or load a saved scaler via load_scaler()."
                )

        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled_t    = X_scaled_flat.reshape(X_t.shape)

        # Invert the transpose
        if X_processed.ndim == 3:
            X_scaled = np.transpose(X_scaled_t, (0, 2, 1))    # -> (S, C, L)
        else:
            X_scaled = np.transpose(X_scaled_t, (0, 3, 1, 2)) # -> (S, C, Sc, L)

        return X_scaled.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Scaler persistence (used by export_digital_twin_package and DigitalAsset)
    # ──────────────────────────────────────────────────────────────────────────

    def save_scaler(self, filepath: str) -> None:
        """Persist the fitted scaler. Uses joblib (or a text sentinel when no
        scaler was used), matching core.dataset._save_scaler so the online
        DigitalAsset can load packages produced by the training pipeline."""
        import joblib
        if self.scaler is None:
            with open(filepath, 'w') as f:
                f.write("NO_SCALER_USED")
        else:
            joblib.dump(self.scaler, filepath)
        print(f"Scaler saved -> {filepath}")

    def load_scaler(self, filepath: str) -> None:
        """
        Load a previously saved scaler so transform() can be called without
        re-fitting.  Call this in DigitalAsset.__init__() when loading the
        offline package. Handles the formats written by the training pipeline:
        a 'NO_SCALER_USED' text sentinel, or a joblib-dumped sklearn scaler.
        """
        import joblib
        try:
            with open(filepath, 'r') as f:
                if f.read(20).strip() == "NO_SCALER_USED":
                    self.scaler = None
                    self.is_fit = True
                    print(f"Scaler: none used <- {filepath}")
                    return
        except (UnicodeDecodeError, ValueError):
            pass   # binary file -> a real (joblib) scaler
        self.scaler = joblib.load(filepath)
        self.is_fit = True
        print(f"Scaler loaded <- {filepath}")