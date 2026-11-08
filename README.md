<h1><div align="center">
 <img alt="pipecat-resampler" width="300px" height="auto" src="https://github.com/pipecat-ai/pipecat-resampler/raw/refs/heads/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-resampler)](https://pypi.org/project/pipecat-resampler) ![Tests](https://github.com/pipecat-ai/pipecat-resampler/actions/workflows/wheels.yaml/badge.svg) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

# pipecat-resampler

A small C extension for Python that converts 16-bit PCM audio, mono or with interleaved channels, from one sample rate to another, as a stream of chunks or a whole clip at once. It's made for real-time voice pipelines, such as the ones built with [Pipecat](https://github.com/pipecat-ai/pipecat).

```python
from pipecat_resampler import Resampler, StreamResampler

# Streaming: equal chunks in give equal chunks out, the first one included.
stream = StreamResampler(24000, 16000)
chunk_16k = stream.process(chunk_24k)  # bytes in, bytes out
tail = stream.flush()  # the audio the filter is still delaying

stereo = StreamResampler(48000, 16000, channels=2)
frames_16k = stereo.process(frames_48k)  # interleaved frames in and out

# A whole clip in one call: the output lines up with the input.
resampler = Resampler(48000, 16000)
clip_16k = resampler.resample(clip_48k)
```

## Platforms

The extension uses Python's stable ABI, so one wheel per platform covers Python 3.11 and later.

| Platform | Wheels |
|---|---|
| Linux x86-64 and ARM64 | manylinux (glibc 2.17 or later) and musllinux (Alpine 3.13 or later) |
| macOS | x86-64 and ARM64 |
| Windows | x86-64 |

## Why it exists

A voice pipeline resamples whenever its parts run at different rates: WebRTC audio arrives at 48 kHz while a speech recognizer wants 16 kHz, or a 24 kHz TTS voice goes out on an 8 kHz phone call. The conversion then runs every 10 to 40 ms for every stream in the process, and now and then on a whole clip, such as a tone to play or the audio a turn detector reads.

We wanted one resampler built for that:

- **Sinc quality.** A band-limited polyphase sinc filter, flat to 90% of the band, with about 90 dB of aliasing rejection.
- **Fast.** C with SIMD, so one process can carry many streams at once.
- **Small.** About 0.1 MB installed, with no runtime dependencies.
- **Steady output.** The same input size gives the same output size on every call, so nothing downstream has to absorb bursts or wait for audio that's held back.
- **Low latency.** A fixed delay of a few milliseconds.
- **Easy to ship.** BSD 2-Clause licensed, like Pipecat, with wheels for Linux (glibc 2.17 or later, and Alpine), macOS and Windows.

Each library we tried covers part of this list, and none covered all of it.

## Comparison

All resamplers were called from Python on an Intel i9-10920X. CPU times come from `scripts/bench.py`; lower is better. Run `uv run python scripts/bench.py` to measure on your machine. It times soxr, and also samplerate, PyAV and the Speex library when they're installed.

### Streaming: output and quality

Each resampler converted 24 kHz noise to 16 kHz in 40 ms chunks. Aliasing is how loud a 10 kHz tone comes out after it folds down to 6 kHz.

| Resampler | Samples per 40 ms chunk (640 expected) | Aliasing | Install size |
|---|---|---|---|
| **pipecat-resampler** | 640 | -98 dB | 0.1 MB |
| soxr VHQ | 0 to 1100 | -93 dB | 0.5 MB |
| soxr QQ | 639 to 640 | -3 dB | 0.5 MB |
| samplerate (libsamplerate) `sinc_medium` | 640, first chunk 594 | -100 dB | 1.8 MB |
| PyAV (FFmpeg) | 640, first chunk 624 | -88 dB | 127 MB |
| Speex, quality 7 | 640 | -100 dB | no PyPI binding |

- **soxr** filters with FFTs over large blocks, so with small chunks VHQ's output comes in bursts. QQ is steady and fast, but it uses cubic interpolation, so content above the new rate folds back into the speech. soxr is LGPL 2.1 licensed, and its Linux wheels need glibc 2.27 or later (2.26 on ARM), with none for Alpine.
- **samplerate** is steady after its first chunk and very clean. It computes the filter for each output sample, which handles any ratio but costs about 15 times soxr's CPU.
- **PyAV** wraps FFmpeg's resampler, which is steady after its first chunk. It brings FFmpeg along, and each call costs about 30 µs whatever the chunk size.
- **Speex** is steady and clean. It has no Python binding on PyPI, so we measured the system library through ctypes.

### Streaming: by chunk size

One stream at 24 → 16 kHz, in microseconds per chunk:

| Resampler | 10 ms | 20 ms | 40 ms |
|---|---|---|---|
| **pipecat-resampler** | 2.0 | 3.9 | 7.5 |
| soxr VHQ | 5.3 | 8.0 | 13.4 |
| soxr QQ | 3.6 | 4.8 | 7.7 |
| samplerate `sinc_medium` | 58.5 | 110.4 | 215.0 |
| PyAV | 29.5 | 32.1 | 33.6 |
| Speex quality 7 | 10.2 | 16.6 | 29.7 |

pipecat-resampler is the fastest at every chunk size, even against soxr QQ, which interpolates instead of filtering, and its lead grows as chunks get smaller.

### Streaming: by rate pair

One stream with 40 ms chunks, the size Pipecat's output transport sends, in microseconds per chunk:

| Resampler | 24 → 16 kHz | 24 → 8 kHz | 16 → 8 kHz | 8 → 16 kHz | 48 → 16 kHz | 44.1 → 16 kHz |
|---|---|---|---|---|---|---|
| **pipecat-resampler** | 7.6 | 6.2 | 4.7 | 6.4 | 11.9 | 19.4 |
| soxr VHQ | 13.5 | 8.9 | 6.1 | 6.8 | 15.3 | 19.0 |
| soxr QQ | 7.4 | 5.1 | 4.9 | 7.5 | 7.8 | 7.7 |
| samplerate `sinc_medium` | 214.9 | 197.2 | 138.7 | 155.6 | 381.6 | 410.9 |
| PyAV | 34.2 | 34.1 | 33.0 | 33.5 | 34.0 | 36.4 |
| Speex quality 7 | 29.1 | 33.0 | 21.6 | 21.5 | 62.0 | 241.0 |

pipecat-resampler uses less CPU than soxr VHQ at every pair but 44.1 → 16 kHz, where the two are about the same: its filter has too many phases to compute four outputs at a time. pipecat-resampler's slowest case here, 19 µs per 40 ms chunk, is under 0.05% of one core.

### Streaming: multiple streams

A process carrying many calls runs many streams at once. Here, 64 streams at the same rates were each given a 40 ms chunk in turn, in microseconds per chunk:

| Resampler | 24 → 16 kHz | 48 → 16 kHz | 44.1 → 16 kHz |
|---|---|---|---|
| **pipecat-resampler** | 7.7 | 11.9 | 19.2 |
| soxr VHQ | 16.9 | 20.2 | 25.6 |
| soxr QQ | 7.6 | 7.9 | 7.8 |
| samplerate `sinc_medium` | 214.3 | 382.0 | 408.8 |
| PyAV | 33.7 | 36.8 | 37.7 |
| Speex quality 7 | 29.4 | 62.1 | 241.5 |

pipecat-resampler costs the same per chunk as with one stream, because the streams share one filter and it stays in the CPU's cache. soxr VHQ costs a quarter to a third more than alone, and more than pipecat-resampler at all three pairs.

### Whole clips: quality

`Resampler` against `soxr.resample()` at its two highest qualities, converting a 2 s clip from 24 kHz to 16 kHz. Error is how far a mix of tones up to 6.8 kHz comes out from the exact signal. Offset is how far the output sits from the input in time, in samples. Gain at 7.6 kHz shows how much of the top of the band each one keeps.

| Resampler | Error | Offset | Gain at 7.6 kHz |
|---|---|---|---|
| **pipecat-resampler** | -82 dB | 0.000 | -6.0 dB |
| soxr VHQ | -77 dB | 0.000 | -3.0 dB |
| soxr HQ | -78 dB | 0.000 | -2.9 dB |

- All three line the output up with the input and keep the clip's length: n samples in give `round(n * out_rate / in_rate)` out.
- soxr's longer filters keep more of the top of the band and reject aliasing past -120 dB. pipecat-resampler's 90 dB puts what it lets through at about the last bit of 16-bit audio.
- soxr adds dither to its 16-bit output, which is part of its error here and turns digital silence into noise in the last bit. pipecat-resampler rounds, so silence stays silent.
- Both release the GIL while they work.

### Whole clips: by clip length

At 24 → 16 kHz, in milliseconds per clip:

| Resampler | 0.2 s | 2 s | 8 s |
|---|---|---|---|
| **pipecat-resampler** | 0.04 | 0.37 | 1.48 |
| soxr VHQ | 0.16 | 0.74 | 2.32 |
| soxr HQ | 0.09 | 0.49 | 1.74 |

pipecat-resampler is the quickest at every length, most of all on short clips, because a `Resampler` designs its filter once while `soxr.resample()` designs one on every call.

### Whole clips: by rate pair

For 2 s clips, in milliseconds per clip:

| Resampler | 24 → 16 kHz | 24 → 8 kHz | 16 → 8 kHz | 8 → 16 kHz | 48 → 16 kHz | 44.1 → 16 kHz |
|---|---|---|---|---|---|---|
| **pipecat-resampler** | 0.38 | 0.30 | 0.23 | 0.30 | 0.59 | 0.93 |
| soxr VHQ | 0.74 | 0.45 | 0.24 | 0.35 | 0.77 | 1.16 |
| soxr HQ | 0.48 | 0.27 | 0.17 | 0.24 | 0.53 | 0.72 |

pipecat-resampler is faster than soxr VHQ at 24 → 16, 24 → 8, 48 → 16 and 44.1 → 16 kHz, and about the same at 16 → 8 and 8 → 16 kHz. soxr HQ is faster at every pair but 24 → 16 kHz.

## How it works

**Filter.** It's a polyphase Kaiser-windowed sinc filter, the design WebRTC and Speex use. It keeps the lower rate's band flat up to 90% of its Nyquist frequency and attenuates aliasing and images by about 90 dB. It has an odd number of taps, so each output sample sits exactly on its middle tap.

**Streaming.** Each `process()` call works out how many output samples its input is worth and carries any fraction of a sample into the next call, which keeps the output steady for any rates and chunk size. Instead of holding output back, the filter delays the audio by a few milliseconds; `flush()` returns that delayed end, and `reset()` discards it.

**Whole clips.** `resample()` treats the clip as silent before and after it. Output sample n is the clip at input sample n × in_rate / out_rate, and the output keeps the clip's length, rounded to the nearest sample. A call doesn't change the resampler, so threads can share one.

**Channels.** With several channels, the audio is interleaved frames, and each channel is filtered on its own, so stereo costs twice what mono does.

**Shared filters.** Resamplers between the same two rates share one filter, built with the first and freed with the last. That saves the most at 44.1 kHz, where a filter takes about 200 KB and a few milliseconds to build, and it keeps many streams fast.

**Cost.** The filter does about 115 multiply-adds for every sample at the higher of the two rates, so 48 kHz input needs twice the work of 24 kHz input. Four outputs that share a phase can be computed in one pass, which loads each tap once for all four. Rates like 48 → 16, 24 → 16 and 8 → 16 kHz have a filter with at most four phases, so nearly every output is computed that way. The 44.1 kHz family has 160 phases or more, so a chunk only has four outputs on a phase when it's long, as with 40 ms chunks at 44.1 → 16 kHz and whole clips. NEON computes those four at a time and the rest one at a time, and the x86 kernels compute the whole family one at a time. The phases are stored in the order outputs use them, so a run of outputs reads its taps front to back.

**Kernels.** The inner loop uses the fastest kernel the CPU supports: AVX2 with FMA or SSE2 on x86-64, NEON on ARM64, and plain C elsewhere. AVX2, SSE2 and NEON compute four outputs per pass where the filter allows it. NEON's four-wide pass gives each output exactly what its one-wide pass does, so it mixes the two on any filter without the output depending on how the input is chunked. Plain C, and SSE2 where it computes one output at a time, add mirrored samples before multiplying where a phase's taps are symmetric, which halves the multiplications. Set `PIPECAT_RESAMPLER_KERNEL` to `generic`, `sse2`, `avx2` or `neon` to force a kernel.

## Development

```bash
uv sync
uv run pytest
uv run python scripts/bench.py
```

Wheels are built and tested with [cibuildwheel](https://cibuildwheel.pypa.io) on every pull request and every push to `main`.

Add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for every change users will notice. A release renames that section to the new version and date, and its GitHub release notes are taken from it.

## License

pipecat-resampler is released under the [BSD 2-Clause License](LICENSE).
