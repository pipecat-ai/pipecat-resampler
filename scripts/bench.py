#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Time pipecat-resampler against other streaming resamplers, called from Python.

Prints microseconds per chunk, median of five runs: first at 24 -> 16 kHz by chunk
size, then by rate pair at 40 ms chunks, the size Pipecat's output transport sends.
soxr is a dev dependency; samplerate, PyAV, the Speex library and Pipecat's numpy
resampler are measured when they're installed.
"""

import ctypes
import ctypes.util
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


def pipecat_resampler(i, o):
    process = Resampler(i, o).process
    return lambda x: process(x.tobytes())


def soxr_vhq(i, o):
    return soxr.ResampleStream(i, o, 1, dtype="int16", quality="VHQ").resample_chunk


def soxr_qq(i, o):
    return soxr.ResampleStream(i, o, 1, dtype="int16", quality="QQ").resample_chunk


def samplerate_medium(i, o):
    import samplerate

    r = samplerate.Resampler("sinc_medium", channels=1)
    ratio = o / i

    def run(x):
        y = r.process(x.astype(np.float32) * (1 / 32768), ratio)
        return np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)

    return run


def pyav(i, o):
    import av

    r = av.AudioResampler(format="s16", layout="mono", rate=o)

    def run(x):
        frame = av.AudioFrame.from_ndarray(x.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = i
        return [f.to_ndarray() for f in r.resample(frame)]

    return run


def speex_q7(i, o):
    path = ctypes.util.find_library("speexdsp")
    if not path:
        raise OSError("libspeexdsp isn't installed")
    lib = ctypes.CDLL(path)
    lib.speex_resampler_init.restype = ctypes.c_void_p
    lib.speex_resampler_init.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.speex_resampler_process_int.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    err = ctypes.c_int(0)
    state = lib.speex_resampler_init(1, i, o, 7, ctypes.byref(err))
    out = np.zeros(o, dtype=np.int16)
    in_len, out_len = ctypes.c_uint32(), ctypes.c_uint32()

    def run(x):
        in_len.value, out_len.value = len(x), len(out)
        lib.speex_resampler_process_int(
            state, 0, x.ctypes.data, ctypes.byref(in_len), out.ctypes.data, ctypes.byref(out_len)
        )
        return out[: out_len.value].copy()

    return run


def pipecat_numpy(i, o):
    from pipecat.audio.resamplers.sinc_stream_resampler import SincStreamAudioResampler

    r = SincStreamAudioResampler()

    def run(x):
        # resample() never awaits, so it runs to completion on the first send.
        coro = r.resample(x.tobytes(), i, o)
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value

    return run


def available():
    candidates = {
        "pipecat-resampler": pipecat_resampler,
        "soxr VHQ": soxr_vhq,
        "soxr QQ": soxr_qq,
        "samplerate sinc_medium": samplerate_medium,
        "PyAV": pyav,
        "Speex quality 7": speex_q7,
        "Pipecat numpy": pipecat_numpy,
    }
    found = {}
    for name, make in candidates.items():
        try:
            make(24000, 16000)(np.zeros(960, dtype=np.int16))
        except Exception:
            continue
        found[name] = make
    return found


def per_chunk_us(make, i, o, ms):
    rng = np.random.default_rng(0)
    parts = [(rng.standard_normal(i * ms // 1000) * 8000).astype(np.int16) for _ in range(500)]
    # An untimed pass first, so the CPU and caches are warm for every resampler.
    warm_up = make(i, o)
    for p in parts:
        warm_up(p)
    runs = []
    for _ in range(5):
        process = make(i, o)
        start = time.perf_counter()
        for p in parts:
            process(p)
        runs.append((time.perf_counter() - start) / len(parts) * 1e6)
    return statistics.median(runs)


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    resamplers = available()
    print(f"pipecat-resampler kernel: {Resampler(24000, 16000).kernel}\n")

    print("24 -> 16 kHz, µs per chunk\n")
    sizes = (10, 20, 40)
    table(
        ["Resampler"] + [f"{ms} ms" for ms in sizes],
        [
            [name] + [f"{per_chunk_us(make, 24000, 16000, ms):.1f}" for ms in sizes]
            for name, make in resamplers.items()
        ],
    )

    print("\n40 ms chunks, µs per chunk\n")
    table(
        ["Resampler"] + [f"{i / 1000:g} -> {o / 1000:g} kHz" for i, o in PAIRS],
        [
            [name] + [f"{per_chunk_us(make, i, o, 40):.1f}" for i, o in PAIRS]
            for name, make in resamplers.items()
        ],
    )


if __name__ == "__main__":
    main()
