#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import math

import numpy as np
import pytest
from helpers import PAIRS, chunks, design, filtered, interleave, level_db, noise, tone

from pipecat_resampler import StreamResampler


def run(r: StreamResampler, parts) -> list[np.ndarray]:
    return [np.frombuffer(r.process(p.tobytes()), dtype=np.int16) for p in parts]


def reference(in_rate: int, out_rate: int, signal: np.ndarray) -> np.ndarray:
    """The same filter design and output positions, computed directly in numpy."""
    up, down, phases, _ = design(in_rate, out_rate)
    width = phases.shape[1]
    buffer = np.concatenate([np.zeros(width - 1, dtype=np.float32), signal.astype(np.float32)])
    count = -(-(len(signal) * up) // down)
    return filtered(phases, up, buffer, np.arange(count) * down)


def tone_level_db(in_rate, out_rate, freq, probe):
    signal = tone(in_rate, freq)
    out = np.concatenate(run(StreamResampler(in_rate, out_rate), chunks(signal, in_rate // 50)))
    return level_db(out, out_rate, probe)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
@pytest.mark.parametrize("ms", [10, 20, 40])
def test_equal_chunks_give_equal_chunks(in_rate, out_rate, ms):
    if (in_rate * ms) % 1000 or (out_rate * ms) % 1000:
        pytest.skip("chunk isn't a whole number of samples at both rates")
    size = in_rate * ms // 1000
    out = run(StreamResampler(in_rate, out_rate), chunks(noise(in_rate), size))
    assert {len(o) for o in out} == {out_rate * ms // 1000}


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_chunking_does_not_change_the_output(in_rate, out_rate):
    signal = noise(in_rate)
    whole = np.frombuffer(
        StreamResampler(in_rate, out_rate).process(signal.tobytes()), dtype=np.int16
    )
    rng = np.random.default_rng(1)
    edges = np.cumsum(rng.integers(1, in_rate // 20, 400))
    parts = np.split(signal, edges[edges < len(signal)])
    pieces = np.concatenate(run(StreamResampler(in_rate, out_rate), parts))
    assert np.array_equal(whole, pieces)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_matches_the_reference_design(in_rate, out_rate):
    signal = noise(in_rate, 0.5)
    out = np.frombuffer(
        StreamResampler(in_rate, out_rate).process(signal.tobytes()), dtype=np.int16
    )
    expected = reference(in_rate, out_rate, signal)
    assert len(out) == len(expected)
    assert np.max(np.abs(out.astype(int) - expected.astype(int))) <= 1


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_passband_is_flat(in_rate, out_rate):
    freq = 0.8 * min(in_rate, out_rate) / 2
    assert abs(tone_level_db(in_rate, out_rate, freq, freq)) < 0.05


@pytest.mark.parametrize("in_rate,out_rate", [p for p in PAIRS if p[0] > p[1]])
def test_aliasing_is_attenuated(in_rate, out_rate):
    nyquist = out_rate / 2
    freq = 1.25 * nyquist
    assert tone_level_db(in_rate, out_rate, freq, 2 * nyquist - freq) < -85


@pytest.mark.parametrize("in_rate,out_rate", [p for p in PAIRS if p[0] < p[1]])
def test_images_are_attenuated(in_rate, out_rate):
    freq = 0.75 * in_rate / 2
    assert tone_level_db(in_rate, out_rate, freq, in_rate - freq) < -85


def test_flush_returns_the_delayed_audio():
    r = StreamResampler(24000, 16000)
    impulse = np.zeros(2400, dtype=np.int16)
    impulse[0] = 16000
    out = np.frombuffer(r.process(impulse.tobytes()) + r.flush(), dtype=np.int16)
    assert len(out) == 1600 + math.ceil(r.delay * 16000 / 24000)
    peak = int(np.argmax(np.abs(out)))
    assert abs(peak - r.delay * 16000 / 24000) <= 1


def test_flush_and_reset_start_a_new_stream():
    signal = noise(24000, 0.2)
    fresh = StreamResampler(24000, 16000).process(signal.tobytes())
    r = StreamResampler(24000, 16000)
    r.process(noise(24000, 0.2, seed=5).tobytes())
    r.flush()
    assert r.process(signal.tobytes()) == fresh
    r.process(noise(24000, 0.2, seed=6).tobytes())
    r.reset()
    assert r.process(signal.tobytes()) == fresh


@pytest.mark.parametrize("channels", [1, 2])
def test_flush_is_empty_when_no_audio_went_in(channels):
    chunk = noise(24000, 0.04 * channels).tobytes()
    r = StreamResampler(24000, 8000, channels=channels)
    assert r.flush() == b""
    r.process(b"")
    assert r.flush() == b""
    r.process(chunk)
    assert r.flush() != b""
    assert r.flush() == b""
    r.process(chunk)
    r.reset()
    assert r.flush() == b""


def test_same_rate_passes_audio_through():
    r = StreamResampler(16000, 16000)
    data = noise(16000, 0.1).tobytes()
    assert r.process(data) == data
    assert r.flush() == b""
    assert r.delay == 0


def test_accepts_any_buffer():
    data = noise(24000, 0.1)
    expected = StreamResampler(24000, 16000).process(data.tobytes())
    assert StreamResampler(24000, 16000).process(memoryview(data.tobytes())) == expected
    assert StreamResampler(24000, 16000).process(bytearray(data.tobytes())) == expected
    assert StreamResampler(24000, 16000).process(data) == expected


def test_empty_input_gives_empty_output():
    assert StreamResampler(24000, 16000).process(b"") == b""


def test_odd_byte_count_is_refused():
    with pytest.raises(ValueError, match="16-bit"):
        StreamResampler(24000, 16000).process(b"\x00\x01\x02")


@pytest.mark.parametrize("in_rate,out_rate", [(0, 16000), (16000, -1)])
def test_bad_rates_are_refused(in_rate, out_rate):
    with pytest.raises(ValueError, match="positive"):
        StreamResampler(in_rate, out_rate)


def test_clips_instead_of_wrapping():
    loud = np.tile(np.array([32767, -32768], dtype=np.int16), 2400)
    out = np.frombuffer(StreamResampler(24000, 16000).process(loud.tobytes()), dtype=np.int16)
    assert out.max() <= 32767 and out.min() >= -32768


def test_properties():
    r = StreamResampler(24000, 16000)
    assert (r.in_rate, r.out_rate, r.channels) == (24000, 16000, 1)
    assert r.delay > 0
    assert r.kernel in {"avx2", "sse2", "neon", "generic"}
    assert StreamResampler(24000, 16000, channels=2).channels == 2


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("in_rate,out_rate", [(48000, 16000), (24000, 16000), (16000, 44100)])
def test_each_channel_matches_resampling_it_alone(channels, in_rate, out_rate):
    signals = [noise(in_rate, seed=c) for c in range(channels)]
    frames = interleave(*signals).reshape(-1, channels)
    rng = np.random.default_rng(1)
    edges = np.cumsum(rng.integers(1, in_rate // 20, 400))
    parts = np.split(frames, edges[edges < len(frames)])
    r = StreamResampler(in_rate, out_rate, channels=channels)
    out = np.concatenate(run(r, parts)).reshape(-1, channels)
    for c, signal in enumerate(signals):
        alone = StreamResampler(in_rate, out_rate).process(signal.tobytes())
        assert np.array_equal(out[:, c], np.frombuffer(alone, dtype=np.int16))


@pytest.mark.parametrize("ms", [10, 20, 40])
def test_stereo_chunks_stay_steady(ms):
    frames = interleave(noise(48000), noise(48000, seed=1))
    size = 2 * 48000 * ms // 1000
    r = StreamResampler(48000, 16000, channels=2)
    out = run(r, chunks(frames, size))
    assert {len(o) for o in out} == {2 * 16000 * ms // 1000}


def test_flush_returns_every_channel():
    left, right = noise(24000, 0.1), noise(24000, 0.1, seed=1)
    r = StreamResampler(24000, 16000, channels=2)
    r.process(interleave(left, right).tobytes())
    tail = np.frombuffer(r.flush(), dtype=np.int16).reshape(-1, 2)
    for c, signal in enumerate((left, right)):
        mono = StreamResampler(24000, 16000)
        mono.process(signal.tobytes())
        assert np.array_equal(tail[:, c], np.frombuffer(mono.flush(), dtype=np.int16))


def test_same_rate_passes_stereo_through():
    data = interleave(noise(16000, 0.1), noise(16000, 0.1, seed=1)).tobytes()
    assert StreamResampler(16000, 16000, channels=2).process(data) == data


def test_partial_frames_are_refused():
    with pytest.raises(ValueError, match="whole frames"):
        StreamResampler(48000, 16000, channels=2).process(b"\x00" * 6)


@pytest.mark.parametrize("channels", [0, -2])
def test_bad_channel_counts_are_refused(channels):
    with pytest.raises(ValueError, match="channels must be positive"):
        StreamResampler(16000, 8000, channels=channels)
