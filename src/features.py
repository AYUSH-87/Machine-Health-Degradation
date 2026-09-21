"""Vibration feature extraction module.

Extracts time-domain statistical metrics and frequency-domain spectral features
from multi-sensor industrial vibration streams over windowed segments.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_time_domain_features(signal: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    """Compute standard statistical vibration time-domain indicators for a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of vibration sensor amplitudes.
    eps : float
        Epsilon to prevent division by zero.

    Returns
    -------
    dict
        Computed time-domain indicators.
    """
    n = len(signal)
    if n == 0:
        return {
            "rms": 0.0, "std": 0.0, "variance": 0.0, "peak": 0.0, "peak_to_peak": 0.0,
            "crest_factor": 0.0, "kurtosis": 3.0, "skewness": 0.0,
            "shape_factor": 1.0, "impulse_factor": 1.0,
        }

    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal, ddof=0))
    variance_val = float(np.var(signal, ddof=0))
    rms_val = float(np.sqrt(np.mean(np.square(signal))))

    max_val = float(np.max(signal))
    min_val = float(np.min(signal))
    peak_val = float(np.max(np.abs(signal)))
    p2p_val = float(max_val - min_val)

    mean_abs = float(np.mean(np.abs(signal)))

    # Crest factor: Peak / RMS (high values indicate early impulsive bearing faults)
    crest_factor = peak_val / (rms_val + eps)

    # Kurtosis: 4th central moment (Gaussian = 3.0; defect impacts cause spikes > 4.0)
    if std_val > eps:
        kurtosis_val = float(stats.kurtosis(signal, fisher=False))  # Pearson definition (normal=3)
        skewness_val = float(stats.skew(signal))
    else:
        kurtosis_val = 3.0
        skewness_val = 0.0

    # Shape factor and impulse factor
    shape_factor = rms_val / (mean_abs + eps)
    impulse_factor = peak_val / (mean_abs + eps)

    return {
        "rms": round(rms_val, 6),
        "std": round(std_val, 6),
        "variance": round(variance_val, 6),
        "peak": round(peak_val, 6),
        "peak_to_peak": round(p2p_val, 6),
        "crest_factor": round(crest_factor, 6),
        "kurtosis": round(kurtosis_val, 6),
        "skewness": round(skewness_val, 6),
        "shape_factor": round(shape_factor, 6),
        "impulse_factor": round(impulse_factor, 6),
    }


def compute_frequency_domain_features(
    signal: np.ndarray,
    sampling_rate_hz: float = 1000.0,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Compute spectral frequency-domain indicators via Fast Fourier Transform (FFT).

    Parameters
    ----------
    signal : np.ndarray
        1D array of vibration sensor amplitudes.
    sampling_rate_hz : float
        Sensor sampling frequency in Hz.
    eps : float
        Epsilon for division stability.

    Returns
    -------
    dict
        Dominant frequency, spectral centroid, and spectral energy.
    """
    n = len(signal)
    if n < 4:
        return {
            "dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_energy": 0.0,
        }

    # Detrend DC component
    signal_detrended = signal - np.mean(signal)

    # Compute one-sided real FFT
    fft_vals = np.fft.rfft(signal_detrended)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    magnitudes = np.abs(fft_vals) / n

    # Skip 0 Hz DC bin for dominant frequency estimation
    if len(magnitudes) > 1:
        dom_idx = int(np.argmax(magnitudes[1:])) + 1
        dominant_freq = float(freqs[dom_idx])
    else:
        dominant_freq = 0.0

    # Spectral centroid: center of gravity of spectrum
    mag_sum = float(np.sum(magnitudes))
    if mag_sum > eps:
        spectral_centroid = float(np.sum(freqs * magnitudes) / mag_sum)
    else:
        spectral_centroid = 0.0

    # Spectral energy: total power in FFT spectrum
    spectral_energy = float(np.sum(np.square(magnitudes)))

    return {
        "dominant_frequency": round(dominant_freq, 4),
        "spectral_centroid": round(spectral_centroid, 4),
        "spectral_energy": round(spectral_energy, 6),
    }


class VibrationFeatureExtractor:
    """Orchestrates windowed feature extraction across all sensor channels."""

    def __init__(
        self,
        sensor_columns: List[str],
        sampling_rate_hz: float = 1000.0,
        window_size: int = 200,
        window_step: int = 100,
        time_domain_features: Optional[List[str]] = None,
        frequency_domain_features: Optional[List[str]] = None,
    ):
        self.sensor_columns = sensor_columns
        self.sampling_rate_hz = sampling_rate_hz
        self.window_size = window_size
        self.window_step = window_step
        self.time_domain_features = time_domain_features or [
            "rms", "std", "variance", "peak", "peak_to_peak",
            "crest_factor", "kurtosis", "skewness", "shape_factor", "impulse_factor"
        ]
        self.frequency_domain_features = frequency_domain_features or [
            "dominant_frequency", "spectral_centroid", "spectral_energy"
        ]

    def extract_windowed_features(
        self,
        df: pd.DataFrame,
        timestamps: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract multi-sensor features across sliding windows, preserving chronological order.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned multi-sensor DataFrame.
        timestamps : pd.Series, optional
            Timestamp or sequence order corresponding to each row.

        Returns
        -------
        features_df : pd.DataFrame
            Feature matrix (rows = windows, cols = sensor_feature).
        window_timestamps : pd.Series
            Timestamp at the center/end of each window.
        """
        n_samples = len(df)
        effective_window = min(self.window_size, n_samples)
        effective_step = max(1, min(self.window_step, n_samples))

        if n_samples < effective_window:
            effective_window = n_samples

        records: List[Dict[str, float]] = []
        out_timestamps: List[Union[str, int]] = []

        start = 0
        while start + effective_window <= n_samples:
            end = start + effective_window
            window_df = df.iloc[start:end]

            row_features: Dict[str, float] = {}

            for sensor in self.sensor_columns:
                sig = window_df[sensor].to_numpy(dtype=np.float64)

                td_feats = compute_time_domain_features(sig)
                fd_feats = compute_frequency_domain_features(sig, sampling_rate_hz=self.sampling_rate_hz)

                for f_name in self.time_domain_features:
                    if f_name in td_feats:
                        row_features[f"{sensor}_{f_name}"] = td_feats[f_name]

                for f_name in self.frequency_domain_features:
                    if f_name in fd_feats:
                        row_features[f"{sensor}_{f_name}"] = fd_feats[f_name]

            records.append(row_features)

            # Preserve timestamp of window center
            mid_idx = start + effective_window // 2
            if timestamps is not None and len(timestamps) > mid_idx:
                out_timestamps.append(timestamps.iloc[mid_idx])
            else:
                out_timestamps.append(mid_idx)

            start += effective_step

        # If no full window fit, extract single feature row over the entire slice
        if not records and n_samples > 0:
            row_features = {}
            for sensor in self.sensor_columns:
                sig = df[sensor].to_numpy(dtype=np.float64)
                td = compute_time_domain_features(sig)
                fd = compute_frequency_domain_features(sig, sampling_rate_hz=self.sampling_rate_hz)
                for f_name in self.time_domain_features:
                    row_features[f"{sensor}_{f_name}"] = td.get(f_name, 0.0)
                for f_name in self.frequency_domain_features:
                    row_features[f"{sensor}_{f_name}"] = fd.get(f_name, 0.0)
            records.append(row_features)
            out_timestamps.append(timestamps.iloc[-1] if timestamps is not None and len(timestamps) > 0 else 0)

        features_df = pd.DataFrame(records)
        window_timestamps_series = pd.Series(out_timestamps, name="timestamp")

        return features_df, window_timestamps_series
