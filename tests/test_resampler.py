#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from pipecat_resampler import Resampler

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


def run(r: Resampler, parts) -> list[np.ndarray]:
    return [np.frombuffer(r.process(p.tobytes()), dtype=np.int16) for p in parts]


def chunks(signal: np.ndarray, size: int) -> list[np.ndarray]:
    return [signal[k : k + size] for k in range(0, len(signal) - size + 1, size)]


def reference(in_rate: int, out_rate: int, signal: np.ndarray) -> np.ndarray:
    """The same filter design and output positions, computed directly in numpy."""
    divisor = math.gcd(in_rate, out_rate)
    up, down = out_rate // divisor, in_rate // divisor
    up_rate = in_rate * up
    nyquist = min(in_rate, out_rate) / 2
    transition = nyquist * 0.1
    cutoff = nyquist - transition / 2
    length = (90.0 - 7.95) / (2.285 * 2 * math.pi * transition / up_rate) + 1
    length = math.ceil(math.ceil(length) / up) * up
    t = np.arange(length) - (length - 1) / 2
    taps = np.sinc(2 * cutoff / up_rate * t) * np.kaiser(length, 0.1102 * (90.0 - 8.7))
    taps *= up / taps.sum()
    phases = taps.reshape(-1, up).T[:, ::-1].astype(np.float32)
    width = phases.shape[1]
    buffer = np.concatenate([np.zeros(width - 1, dtype=np.float32), signal.astype(np.float32)])
    count = -(-(len(signal) * up) // down)
    positions = np.arange(count) * down
    starts, phase = np.divmod(positions, up)
    windows = np.lib.stride_tricks.sliding_window_view(buffer, width)[starts]
    out = np.einsum("ij,ij->i", windows, phases[phase])
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
@pytest.mark.parametrize("ms", [10, 20, 40])
def test_equal_chunks_give_equal_chunks(in_rate, out_rate, ms):
    if (in_rate * ms) % 1000 or (out_rate * ms) % 1000:
        pytest.skip("chunk isn't a whole number of samples at both rates")
    size = in_rate * ms // 1000
    out = run(Resampler(in_rate, out_rate), chunks(noise(in_rate), size))
    assert {len(o) for o in out} == {out_rate * ms // 1000}


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_chunking_does_not_change_the_output(in_rate, out_rate):
    signal = noise(in_rate)
    whole = np.frombuffer(Resampler(in_rate, out_rate).process(signal.tobytes()), dtype=np.int16)
    rng = np.random.default_rng(1)
    edges = np.cumsum(rng.integers(1, in_rate // 20, 400))
    parts = np.split(signal, edges[edges < len(signal)])
    pieces = np.concatenate(run(Resampler(in_rate, out_rate), parts))
    assert np.array_equal(whole, pieces)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_matches_the_reference_design(in_rate, out_rate):
    signal = noise(in_rate, 0.5)
    out = np.frombuffer(Resampler(in_rate, out_rate).process(signal.tobytes()), dtype=np.int16)
    expected = reference(in_rate, out_rate, signal)
    assert len(out) == len(expected)
    assert np.max(np.abs(out.astype(int) - expected.astype(int))) <= 1


def level_db(in_rate, out_rate, freq, probe):
    t = np.arange(in_rate * 2) / in_rate
    signal = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    out = np.concatenate(run(Resampler(in_rate, out_rate), chunks(signal, in_rate // 50)))
    y = out[out_rate // 4 : out_rate // 4 + out_rate].astype(float)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y)))) / (len(y) / 4)
    freqs = np.fft.rfftfreq(len(y), 1 / out_rate)
    return 20 * np.log10(max(spectrum[np.argmin(np.abs(freqs - probe))], 1e-9) / 16000)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_passband_is_flat(in_rate, out_rate):
    tone = 0.8 * min(in_rate, out_rate) / 2
    assert abs(level_db(in_rate, out_rate, tone, tone)) < 0.05


@pytest.mark.parametrize("in_rate,out_rate", [p for p in PAIRS if p[0] > p[1]])
def test_aliasing_is_attenuated(in_rate, out_rate):
    nyquist = out_rate / 2
    tone = 1.25 * nyquist
    assert level_db(in_rate, out_rate, tone, 2 * nyquist - tone) < -85


@pytest.mark.parametrize("in_rate,out_rate", [p for p in PAIRS if p[0] < p[1]])
def test_images_are_attenuated(in_rate, out_rate):
    tone = 0.75 * in_rate / 2
    assert level_db(in_rate, out_rate, tone, in_rate - tone) < -85


def test_flush_returns_the_delayed_audio():
    r = Resampler(24000, 16000)
    impulse = np.zeros(2400, dtype=np.int16)
    impulse[0] = 16000
    out = np.frombuffer(r.process(impulse.tobytes()) + r.flush(), dtype=np.int16)
    assert len(out) == 1600 + math.ceil(r.delay * 16000 / 24000)
    peak = int(np.argmax(np.abs(out)))
    assert abs(peak - r.delay * 16000 / 24000) <= 1


def test_flush_and_reset_start_a_new_stream():
    signal = noise(24000, 0.2)
    fresh = Resampler(24000, 16000).process(signal.tobytes())
    r = Resampler(24000, 16000)
    r.process(noise(24000, 0.2, seed=5).tobytes())
    r.flush()
    assert r.process(signal.tobytes()) == fresh
    r.process(noise(24000, 0.2, seed=6).tobytes())
    r.reset()
    assert r.process(signal.tobytes()) == fresh


def test_same_rate_passes_audio_through():
    r = Resampler(16000, 16000)
    data = noise(16000, 0.1).tobytes()
    assert r.process(data) == data
    assert r.flush() == b""
    assert r.delay == 0


def test_accepts_any_buffer():
    data = noise(24000, 0.1)
    expected = Resampler(24000, 16000).process(data.tobytes())
    assert Resampler(24000, 16000).process(memoryview(data.tobytes())) == expected
    assert Resampler(24000, 16000).process(bytearray(data.tobytes())) == expected
    assert Resampler(24000, 16000).process(data) == expected


def test_empty_input_gives_empty_output():
    assert Resampler(24000, 16000).process(b"") == b""


def test_odd_byte_count_is_refused():
    with pytest.raises(ValueError, match="16-bit"):
        Resampler(24000, 16000).process(b"\x00\x01\x02")


@pytest.mark.parametrize("in_rate,out_rate", [(0, 16000), (16000, -1)])
def test_bad_rates_are_refused(in_rate, out_rate):
    with pytest.raises(ValueError, match="positive"):
        Resampler(in_rate, out_rate)


def test_clips_instead_of_wrapping():
    loud = np.tile(np.array([32767, -32768], dtype=np.int16), 2400)
    out = np.frombuffer(Resampler(24000, 16000).process(loud.tobytes()), dtype=np.int16)
    assert out.max() <= 32767 and out.min() >= -32768


def test_properties():
    r = Resampler(24000, 16000)
    assert (r.in_rate, r.out_rate) == (24000, 16000)
    assert r.delay > 0
    assert r.kernel in {"avx2", "sse2", "neon", "generic"}


KERNEL_CHECK = """
import numpy as np, sys
from pipecat_resampler import Resampler
rng = np.random.default_rng(0)
signal = (rng.standard_normal(48000) * 8000).astype(np.int16).tobytes()
for i, o in [(24000, 16000), (24000, 8000), (8000, 24000), (44100, 16000)]:
    r = Resampler(i, o)
    sys.stdout.buffer.write(r.process(signal[: 2 * i]))
print(r.kernel, file=sys.stderr)
"""


def kernel_output(kernel: str) -> tuple[str, np.ndarray]:
    env = dict(os.environ, PIPECAT_RESAMPLER_KERNEL=kernel)
    done = subprocess.run(
        [sys.executable, "-c", KERNEL_CHECK], env=env, capture_output=True, check=True
    )
    return done.stderr.decode().strip(), np.frombuffer(done.stdout, dtype=np.int16)


def test_every_kernel_gives_the_same_output():
    used, generic = kernel_output("generic")
    assert used == "generic"
    for kernel in ("sse2", "avx2", "neon"):
        used, out = kernel_output(kernel)
        assert len(out) == len(generic)
        assert np.max(np.abs(out.astype(int) - generic.astype(int))) <= 1
