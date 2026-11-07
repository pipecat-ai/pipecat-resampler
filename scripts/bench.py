#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Time pipecat-resampler against soxr's stream resampler, called from Python.

Prints microseconds per chunk, median of five runs, at 10, 20 and 40 ms chunks.
"""

import statistics
import time

import numpy as np
import soxr

from pipecat_resampler import Resampler

PAIRS = [
    (24000, 16000),
    (24000, 8000),
    (16000, 8000),
    (8000, 16000),
    (48000, 16000),
    (44100, 16000),
]


def per_chunk_us(make, parts):
    runs = []
    for _ in range(5):
        process = make()
        start = time.perf_counter()
        for p in parts:
            process(p)
        runs.append((time.perf_counter() - start) / len(parts) * 1e6)
    return statistics.median(runs)


def main():
    print(f"kernel: {Resampler(24000, 16000).kernel}")
    for ms in (10, 20, 40):
        print(f"\n{ms} ms chunks (µs per chunk)")
        print(f"  {'pair':<14}{'pipecat':>10}{'soxr VHQ':>10}")
        for i, o in PAIRS:
            rng = np.random.default_rng(0)
            parts = [
                (rng.standard_normal(i * ms // 1000) * 8000).astype(np.int16) for _ in range(500)
            ]
            ours = per_chunk_us(
                lambda i=i, o=o: Resampler(i, o).process, [p.tobytes() for p in parts]
            )
            theirs = per_chunk_us(
                lambda i=i, o=o: (
                    soxr.ResampleStream(i, o, 1, dtype="int16", quality="VHQ").resample_chunk
                ),
                parts,
            )
            print(f"  {f'{i}->{o}':<14}{ours:>10.1f}{theirs:>10.1f}")


if __name__ == "__main__":
    main()
