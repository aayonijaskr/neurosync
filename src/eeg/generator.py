"""
neurosync/src/eeg/generator.py
Synthetic EEG signal generator with controllable cognitive states.

WHY THIS EXISTS:
Real BCI systems read signals from hardware electrodes. Since we're simulating,
we generate physiologically realistic EEG by summing sinusoidal components
at known brain-wave frequencies, then adding colored noise. The key insight
is that different cognitive states have different frequency-band signatures —
this is empirically established neuroscience, not fiction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class EEGConfig:
    sampling_rate: int = 256     # Hz — standard EEG sampling rate
    duration: float = 2.0        # seconds per epoch
    n_channels: int = 8          # EEG channels (e.g., F3, F4, C3, C4, P3, P4, O1, O2)
    noise_level: float = 0.1     # additive noise amplitude


# Frequency bands (Hz) — empirically established in BCI literature
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# Cognitive state profiles: relative amplitude of each band
# These approximate real EEG signatures from the literature:
#   - FOCUS: high alpha (relaxed attention), moderate beta, low theta
#   - HIGH_EFFORT: high theta (working memory load), high gamma, suppressed alpha
STATE_PROFILES = {
    "focus": {
        "delta": 0.3,
        "theta": 0.35,
        "alpha": 0.75,   # dominant — marker of calm, focused attention
        "beta":  0.45,
        "gamma": 0.25,
    },
    "high_effort": {
        "delta": 0.25,
        "theta": 0.72,   # elevated — marker of cognitive load / working memory
        "alpha": 0.22,   # suppressed — alpha desynchronization under load
        "beta":  0.60,
        "gamma": 0.65,   # elevated — associated with active processing
    },
}


def generate_eeg_epoch(
    state: str = "focus",
    config: EEGConfig = None,
    noise_multiplier: float = 1.0,
    random_seed: int = None,
) -> np.ndarray:
    """
    Generate a synthetic multi-channel EEG epoch for a given cognitive state.

    Returns:
        np.ndarray of shape (n_channels, n_samples)
    """
    if config is None:
        config = EEGConfig()
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = int(config.sampling_rate * config.duration)
    t = np.linspace(0, config.duration, n_samples)
    profile = STATE_PROFILES.get(state, STATE_PROFILES["focus"])
    signal = np.zeros((config.n_channels, n_samples))

    for ch in range(config.n_channels):
        # Sum sinusoids across all frequency bands
        channel_signal = np.zeros(n_samples)
        for band_name, (f_low, f_high) in BANDS.items():
            amplitude = profile[band_name]
            # Pick a representative frequency in the band (with slight per-channel variation)
            center_freq = (f_low + f_high) / 2 + np.random.uniform(-1.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            channel_signal += amplitude * np.sin(2 * np.pi * center_freq * t + phase)

        # Add colored (1/f) noise — more realistic than white noise
        white = np.random.randn(n_samples)
        freqs = np.fft.rfftfreq(n_samples, 1 / config.sampling_rate)
        freqs[0] = 1  # avoid div-by-zero
        colored_noise = np.fft.irfft(np.fft.rfft(white) / np.sqrt(freqs))
        colored_noise = colored_noise[:n_samples]
        colored_noise /= colored_noise.std() + 1e-8

        signal[ch] = channel_signal + config.noise_level * noise_multiplier * colored_noise

    return signal


def generate_dataset(
    n_epochs_per_class: int = 100,
    config: EEGConfig = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset of EEG epochs for both cognitive states.

    Returns:
        X: shape (n_epochs_total, n_channels, n_samples)
        y: shape (n_epochs_total,)  — 0=focus, 1=high_effort
    """
    if config is None:
        config = EEGConfig()
    np.random.seed(seed)

    X_focus = np.array([
        generate_eeg_epoch("focus", config) for _ in range(n_epochs_per_class)
    ])
    X_effort = np.array([
        generate_eeg_epoch("high_effort", config) for _ in range(n_epochs_per_class)
    ])

    X = np.concatenate([X_focus, X_effort], axis=0)
    y = np.array([0] * n_epochs_per_class + [1] * n_epochs_per_class)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]
