#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import subprocess
import sys

import numpy as np

KERNEL_CHECK = """
import numpy as np, sys
from pipecat_resampler import Resampler, StreamResampler
rng = np.random.default_rng(0)
signal = (rng.standard_normal(48000) * 8000).astype(np.int16).tobytes()
for i, o in [(24000, 16000), (24000, 8000), (8000, 24000), (44100, 16000), (48000, 16000)]:
    sys.stdout.buffer.write(StreamResampler(i, o).process(signal[: 2 * i]))
    sys.stdout.buffer.write(Resampler(i, o).resample(signal[: 2 * i]))
r = StreamResampler(24000, 16000, channels=2)
sys.stdout.buffer.write(r.process(signal[: 4 * 24000]))
sys.stdout.buffer.write(Resampler(24000, 16000, channels=2).resample(signal[: 4 * 24000]))
print(r.kernel, Resampler(24000, 16000).kernel, file=sys.stderr)
"""


def kernel_output(kernel: str) -> tuple[str, np.ndarray]:
    env = dict(os.environ, PIPECAT_RESAMPLER_KERNEL=kernel)
    done = subprocess.run(
        [sys.executable, "-c", KERNEL_CHECK], env=env, capture_output=True, check=True
    )
    return done.stderr.decode().strip(), np.frombuffer(done.stdout, dtype=np.int16)


def test_every_kernel_gives_the_same_output():
    used, generic = kernel_output("generic")
    assert used == "generic generic"
    for kernel in ("sse2", "avx2", "neon"):
        used, out = kernel_output(kernel)
        assert len(out) == len(generic)
        assert np.max(np.abs(out.astype(int) - generic.astype(int))) <= 1
