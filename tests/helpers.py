#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Signals, measurements and the reference filter design the tests share."""

import math

import numpy as np

PAIRS = [
    (24000, 16000),
    (24000, 8000),
    (16000, 8000),
    (48000, 16000),
    (48000, 8000),
    (48000, 24000),
    (8000, 16000),
    (8000, 24000),
    (16000, 24000),
    (16000, 48000),
    (24000, 48000),
    (44100, 16000),
    (16000, 44100),
    (22050, 24000),
]


def noise(rate: int, secs: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(rate * secs)) * 8000).astype(np.int16)


def tone(rate: int, freq: float, secs: float = 2.0) -> np.ndarray:
    t = np.arange(int(rate * secs)) / rate
    return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)


def chunks(signal: np.ndarray, size: int) -> list[np.ndarray]:
    return [signal[k : k + size] for k in range(0, len(signal) - size + 1, size)]


def interleave(*signals: np.ndarray) -> np.ndarray:
    return np.stack(signals, axis=1).reshape(-1)


def level_db(out: np.ndarray, rate: int, probe: float) -> float:
    """Level at probe Hz of a resampled tone, relative to the tone, from a quarter second in."""
    y = out[rate // 4 : rate // 4 + rate].astype(float)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y)))) / (len(y) / 4)
    freqs = np.fft.rfftfreq(len(y), 1 / rate)
    return 20 * np.log10(max(spectrum[np.argmin(np.abs(freqs - probe))], 1e-9) / 16000)


def design(in_rate: int, out_rate: int):
    """The filter the extension designs, split into one reversed filter per phase.

    Returns up, down, the phases and the middle tap on the input upsampled by up.
    """
    divisor = math.gcd(in_rate, out_rate)
    up, down = out_rate // divisor, in_rate // divisor
    up_rate = in_rate * up
    nyquist = min(in_rate, out_rate) / 2
    transition = nyquist * 0.1
    cutoff = nyquist - transition / 2
    estimate = (90.0 - 7.95) / (2.285 * 2 * math.pi * transition / up_rate) + 1
    taps = math.ceil(estimate) | 1
    length = -(-taps // up) * up
    t = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / up_rate * t) * np.kaiser(taps, 0.1102 * (90.0 - 8.7))
    h = np.concatenate([h * up / h.sum(), np.zeros(length - taps)])
    phases = h.reshape(-1, up).T[:, ::-1].astype(np.float32)
    return up, down, phases, (taps - 1) // 2


def filtered(phases: np.ndarray, up: int, buffer: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The outputs at positions on the buffer upsampled by up, as int16."""
    starts, phase = np.divmod(positions, up)
    windows = np.lib.stride_tricks.sliding_window_view(buffer, phases.shape[1])[starts]
    out = np.einsum("ij,ij->i", windows, phases[phase])
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
