"""
neurosync/src/eeg/features.py
Band-power feature extraction from EEG epochs.

WHY THIS EXISTS:
Raw EEG is a 2D array of voltages — not directly useful for a classifier.
We extract "band power" features: the energy of the signal in each frequency
band, per channel. These are the standard features in BCI systems (used in
real P300 spellers, motor-imagery BCIs, etc.). The resulting feature vector
is compact, interpretable, and physiologically meaningful.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}


def bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    fs: int = 256,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to EEG data."""
    nyq = fs / 2
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=-1)


def compute_band_power(data: np.ndarray, fs: int = 256) -> np.ndarray:
    """
    Compute average band power across all channels for each frequency band.

    Args:
        data: EEG epoch, shape (n_channels, n_samples)
        fs: sampling rate in Hz

    Returns:
        features: 1D array of shape (n_bands * n_channels,)
                  — can be reduced to (n_bands,) if averaging across channels
    """
    n_channels = data.shape[0]
    features = []

    for band_name, (low, high) in BANDS.items():
        filtered = bandpass_filter(data, low, high, fs)
        # Log band power per channel (log stabilizes variance)
        band_power = np.log(np.mean(filtered ** 2, axis=-1) + 1e-8)
        features.extend(band_power.tolist())

    return np.array(features, dtype=np.float32)


def extract_features_batch(
    X: np.ndarray,
    fs: int = 256,
) -> np.ndarray:
    """
    Extract features from a batch of EEG epochs.

    Args:
        X: shape (n_epochs, n_channels, n_samples)
        fs: sampling rate

    Returns:
        features: shape (n_epochs, n_features)
    """
    return np.array([compute_band_power(epoch, fs) for epoch in X])


def get_band_power_summary(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """
    Return a human-readable dict of average band powers (averaged across channels).
    Used for the UI dashboard display.

    Returns:
        e.g. {"alpha": 0.72, "theta": 0.38, ...}  (normalized 0–1)
    """
    result = {}
    powers = []
    for band_name, (low, high) in BANDS.items():
        filtered = bandpass_filter(data, low, high, fs)
        power = float(np.mean(filtered ** 2))
        powers.append(power)
        result[band_name] = power

    # Normalize to [0, 1] for visualization
    max_p = max(powers) + 1e-8
    return {k: round(v / max_p, 3) for k, v in result.items()}
