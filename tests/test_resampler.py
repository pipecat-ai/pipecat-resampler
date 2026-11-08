#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from helpers import PAIRS, design, filtered, interleave, level_db, noise, tone

from pipecat_resampler import Resampler


def resample(in_rate: int, out_rate: int, signal: np.ndarray, channels: int = 1) -> np.ndarray:
    out = Resampler(in_rate, out_rate, channels=channels).resample(signal.tobytes())
    return np.frombuffer(out, dtype=np.int16)


def reference(in_rate: int, out_rate: int, signal: np.ndarray) -> np.ndarray:
    """The same filter design and output positions, computed directly in numpy."""
    up, down, phases, center = design(in_rate, out_rate)
    width = phases.shape[1]
    count = (2 * len(signal) * up + down) // (2 * down)
    positions = center + np.arange(count) * down
    after = max(0, positions[-1] // up + 1 - len(signal))
    buffer = np.concatenate([np.zeros(width - 1), signal, np.zeros(after)]).astype(np.float32)
    return filtered(phases, up, buffer, positions)


def tone_level_db(in_rate, out_rate, freq, probe):
    return level_db(resample(in_rate, out_rate, tone(in_rate, freq)), out_rate, probe)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
@pytest.mark.parametrize("frames", [0, 1, 2, 3, 4801, 4802, 44101])
def test_output_is_the_clip_at_the_output_rate(in_rate, out_rate, frames):
    out = resample(in_rate, out_rate, noise(in_rate, 6)[:frames])
    assert len(out) == math.floor(frames * out_rate / in_rate + 0.5)


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_matches_the_reference_design(in_rate, out_rate):
    signal = noise(in_rate, 0.5)
    out = resample(in_rate, out_rate, signal)
    expected = reference(in_rate, out_rate, signal)
    assert len(out) == len(expected)
    assert np.max(np.abs(out.astype(int) - expected.astype(int))) <= 1


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
def test_output_lines_up_with_the_input(in_rate, out_rate):
    freq = 0.3 * min(in_rate, out_rate) / 2
    signal = np.rint(np.sin(2 * np.pi * freq * np.arange(in_rate) / in_rate) * 16000)
    out = resample(in_rate, out_rate, signal.astype(np.int16)).astype(float)
    ideal = np.sin(2 * np.pi * freq * np.arange(len(out)) / out_rate) * 16000
    middle = slice(len(out) // 4, 3 * len(out) // 4)
    assert np.max(np.abs(out[middle] - ideal[middle])) <= 3


@pytest.mark.parametrize("in_rate,out_rate", PAIRS)
@pytest.mark.parametrize("where", [0, 0.5, 0.9])
def test_an_impulse_comes_out_where_it_went_in(in_rate, out_rate, where):
    clip = np.zeros(in_rate // 10, dtype=np.int16)
    k = int(where * (len(clip) - 1))
    clip[k] = 16000
    out = resample(in_rate, out_rate, clip)
    assert abs(int(np.argmax(np.abs(out))) - k * out_rate / in_rate) <= 0.5


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


def test_calls_do_not_affect_each_other():
    signal = noise(24000, 0.2)
    fresh = Resampler(24000, 16000).resample(signal.tobytes())
    r = Resampler(24000, 16000)
    r.resample(noise(24000, 0.2, seed=5).tobytes())
    assert r.resample(signal.tobytes()) == fresh
    assert r.resample(signal.tobytes()) == fresh


def test_threads_can_share_a_resampler():
    clips = [noise(48000, 0.5, seed=s).tobytes() for s in range(16)]
    expected = [Resampler(48000, 16000).resample(c) for c in clips]
    r = Resampler(48000, 16000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(r.resample, clips)) == expected


def test_releases_the_gil_while_it_works():
    r = Resampler(48000, 16000)
    clip = noise(48000, 30).tobytes()
    started = threading.Event()
    finished = False

    def work():
        nonlocal finished
        started.set()
        r.resample(clip)
        finished = True

    # With a long switch interval the worker only gives up the GIL by itself,
    # so this thread runs before the work is done only if resample() lets go.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(100)
    try:
        worker = threading.Thread(target=work)
        worker.start()
        started.wait()
        ran_during_the_work = not finished
        worker.join()
    finally:
        sys.setswitchinterval(interval)
    assert ran_during_the_work


def test_same_rate_passes_audio_through():
    data = noise(16000, 0.1).tobytes()
    assert Resampler(16000, 16000).resample(data) == data
    stereo = interleave(noise(16000, 0.1), noise(16000, 0.1, seed=1)).tobytes()
    assert Resampler(16000, 16000, channels=2).resample(stereo) == stereo


def test_accepts_any_buffer():
    data = noise(24000, 0.1)
    expected = Resampler(24000, 16000).resample(data.tobytes())
    assert Resampler(24000, 16000).resample(memoryview(data.tobytes())) == expected
    assert Resampler(24000, 16000).resample(bytearray(data.tobytes())) == expected
    assert Resampler(24000, 16000).resample(data) == expected


def test_empty_input_gives_empty_output():
    assert Resampler(24000, 16000).resample(b"") == b""


def test_partial_frames_are_refused():
    with pytest.raises(ValueError, match="16-bit"):
        Resampler(24000, 16000).resample(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="whole frames"):
        Resampler(48000, 16000, channels=2).resample(b"\x00" * 6)


@pytest.mark.parametrize("in_rate,out_rate", [(0, 16000), (16000, -1)])
def test_bad_rates_are_refused(in_rate, out_rate):
    with pytest.raises(ValueError, match="positive"):
        Resampler(in_rate, out_rate)


@pytest.mark.parametrize("channels", [0, -2])
def test_bad_channel_counts_are_refused(channels):
    with pytest.raises(ValueError, match="channels must be positive"):
        Resampler(16000, 8000, channels=channels)


def test_clips_instead_of_wrapping():
    loud = np.tile(np.array([32767, -32768], dtype=np.int16), 2400)
    out = resample(24000, 16000, loud)
    assert out.max() <= 32767 and out.min() >= -32768


def test_properties():
    r = Resampler(24000, 16000)
    assert (r.in_rate, r.out_rate, r.channels) == (24000, 16000, 1)
    assert r.kernel in {"avx2", "sse2", "neon", "generic"}
    assert Resampler(24000, 16000, channels=2).channels == 2


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("in_rate,out_rate", [(48000, 16000), (24000, 16000), (16000, 44100)])
def test_each_channel_matches_resampling_it_alone(channels, in_rate, out_rate):
    signals = [noise(in_rate, seed=c) for c in range(channels)]
    out = resample(in_rate, out_rate, interleave(*signals), channels).reshape(-1, channels)
    for c, signal in enumerate(signals):
        assert np.array_equal(out[:, c], resample(in_rate, out_rate, signal))
