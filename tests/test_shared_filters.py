#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import pytest
from helpers import chunks, noise

from pipecat_resampler import Resampler, StreamResampler, _resampler


def filters() -> int:
    return _resampler._filter_count()


def test_resamplers_between_the_same_rates_share_a_filter():
    before = filters()
    stream = StreamResampler(32000, 12000)
    stereo = StreamResampler(32000, 12000, channels=2)
    clip = Resampler(32000, 12000)
    assert filters() == before + 1

    other = StreamResampler(12000, 32000)
    assert filters() == before + 2

    del stream, stereo, clip, other
    assert filters() == before


def test_a_filter_lasts_until_its_last_resampler_goes():
    signal = noise(32000, 0.2)
    expected = StreamResampler(32000, 12000).process(signal.tobytes())
    before = filters()
    first = StreamResampler(32000, 12000)
    second = StreamResampler(32000, 12000)

    del first
    assert filters() == before + 1
    assert second.process(signal.tobytes()) == expected

    del second
    assert filters() == before


def test_streams_sharing_a_filter_keep_their_own_state():
    left, right = noise(32000, 0.5), noise(32000, 0.5, seed=1)
    a, b = StreamResampler(32000, 12000), StreamResampler(32000, 12000)
    out_a, out_b = [], []
    for x, y in zip(chunks(left, 320), chunks(right, 320)):
        out_a.append(a.process(x.tobytes()))
        out_b.append(b.process(y.tobytes()))

    for signal, out in ((left, out_a), (right, out_b)):
        alone = StreamResampler(32000, 12000)
        expected = [alone.process(x.tobytes()) for x in chunks(signal, 320)]
        assert out == expected


def test_a_failed_resampler_takes_no_filter():
    before = filters()
    with pytest.raises(ValueError):
        StreamResampler(32000, 12000, channels=0)
    with pytest.raises(ValueError):
        Resampler(0, 12000)
    assert filters() == before


def test_a_shared_filter_gives_the_same_output():
    signal = noise(32000, 0.5)
    alone = np.frombuffer(Resampler(32000, 12000).resample(signal.tobytes()), dtype=np.int16)
    keep = StreamResampler(32000, 12000)
    shared = np.frombuffer(Resampler(32000, 12000).resample(signal.tobytes()), dtype=np.int16)
    assert np.array_equal(alone, shared)
    del keep
