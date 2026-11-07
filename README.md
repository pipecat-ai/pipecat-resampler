# pipecat-resampler

A small C extension for Python that converts 16-bit PCM audio, mono or with interleaved channels, from one sample rate to another, chunk by chunk. It's made for real-time audio, such as voice pipelines built with [Pipecat](https://github.com/pipecat-ai/pipecat). Equal chunks in give equal chunks out: 40 ms of audio in always gives 40 ms out, the first chunk included.

```python
from pipecat_resampler import Resampler

resampler = Resampler(24000, 16000)
chunk_16k = resampler.process(chunk_24k)  # bytes in, bytes out
tail = resampler.flush()  # the audio the filter is delaying

stereo = Resampler(48000, 16000, channels=2)
frames_16k = stereo.process(frames_48k)  # interleaved frames in and out
```

## Platforms

The extension uses Python's stable ABI, so one wheel per platform covers Python 3.11 and later.

| Platform | Wheels |
|---|---|
| Linux x86-64 and ARM64 | manylinux (glibc 2.17 or later) and musllinux (Alpine 3.13 or later) |
| macOS | x86-64 and ARM64 |
| Windows | x86-64 |

## Why it exists

A real-time voice pipeline has to resample whenever its parts run at different rates. WebRTC audio arrives at 48 kHz while a speech recognizer might want 16 kHz, or a TTS voice that only comes at 24 kHz has to go out on an 8 kHz phone call. When that happens, the conversion runs chunk by chunk, every 10 to 40 ms, for every stream the process is handling.

We wanted a resampler built for that:

- **Sinc quality.** A band-limited polyphase sinc filter, flat to 90% of the band, with about 90 dB of aliasing rejection.
- **Fast.** C with SIMD, so one process can carry many streams at once.
- **Small.** About 0.1 MB installed, with no runtime dependencies.
- **Steady output.** The same input size gives the same output size on every call, so nothing downstream has to absorb bursts of audio or wait for audio that's held back.
- **Low latency.** A fixed delay of a few milliseconds.
- **Easy to ship.** BSD 2-Clause licensed, like Pipecat, with wheels for Linux (glibc 2.17 or later, and Alpine), macOS and Windows.

Each library we tried covers part of this list, and none covered all of it.

## Comparison

Each resampler converted 24 kHz noise to 16 kHz in 40 ms chunks, called from Python, on an Intel i9-10920X. Aliasing is how loud a 10 kHz tone comes out after it folds down to 6 kHz; lower is better.

| Resampler | Samples per 40 ms chunk (640 expected) | Aliasing | CPU per chunk | Install size |
|---|---|---|---|---|
| soxr VHQ | 0 to 1100 | -93 dB | 13.5 µs | 0.5 MB |
| soxr QQ | 639 to 640 | -3 dB | 6.5 µs | 0.5 MB |
| samplerate (libsamplerate) `sinc_medium` | 640, first chunk 594 | -100 dB | 198 µs | 1.8 MB |
| PyAV (FFmpeg) | 640, first chunk 624 | -88 dB | 33.7 µs | 127 MB |
| Speex resampler, quality 7 | 640 | -100 dB | 26.9 µs | no PyPI binding |
| **pipecat-resampler** | 640 | -98 dB | 9.1 µs | 0.1 MB |

How each one fits this use:

- **soxr** reaches its quality cheaply by filtering with FFTs over large blocks. Fed small chunks, that means its output arrives in bursts. Its quick setting, QQ, is steady and the fastest option here; it uses cubic interpolation instead of a long filter, so content above the new rate folds back into the speech. soxr is licensed under the LGPL 2.1, and its Linux wheels need glibc 2.27 or later (2.26 on ARM), with none for Alpine.
- **samplerate** (libsamplerate) is steady after its first chunk and very clean. Its sinc converters compute the filter for each output sample, which handles any ratio and costs about 15 times soxr's CPU here.
- **PyAV** wraps FFmpeg's resampler, which is steady after its first chunk and fast. It brings FFmpeg along, 127 MB installed, and each call from Python costs about 30 µs whatever the chunk size.
- **Speex** has a steady, clean resampler. It has no Python binding on PyPI (the `speexdsp` package covers its echo canceller), so we measured the system library through ctypes. It uses about twice soxr's CPU here, and more at 44.1 kHz, where it computes the filter on the fly.

## Performance

Called from Python, in microseconds per chunk, from the same run as the comparison; lower is better.

At 24 → 16 kHz, by chunk size:

| Resampler | 10 ms | 20 ms | 40 ms |
|---|---|---|---|
| **pipecat-resampler** | 2.4 | 4.7 | 9.1 |
| soxr VHQ | 5.7 | 8.1 | 13.5 |
| soxr QQ | 3.3 | 4.6 | 6.5 |
| samplerate `sinc_medium` | 55.0 | 103.0 | 198.1 |
| PyAV | 29.0 | 31.3 | 33.7 |
| Speex quality 7 | 9.2 | 14.9 | 26.9 |

At 40 ms chunks, the size Pipecat's output transport sends, by rate pair:

| Resampler | 24 → 16 kHz | 24 → 8 kHz | 16 → 8 kHz | 8 → 16 kHz | 48 → 16 kHz | 44.1 → 16 kHz |
|---|---|---|---|---|---|---|
| **pipecat-resampler** | 9.2 | 8.6 | 6.1 | 7.9 | 17.0 | 17.0 |
| soxr VHQ | 13.8 | 9.2 | 6.1 | 6.9 | 15.1 | 18.0 |
| soxr QQ | 7.0 | 4.8 | 4.6 | 6.9 | 7.2 | 7.1 |
| samplerate `sinc_medium` | 198.8 | 184.0 | 127.8 | 145.1 | 357.0 | 388.8 |
| PyAV | 33.5 | 32.8 | 31.8 | 31.2 | 36.0 | 35.6 |
| Speex quality 7 | 27.4 | 29.7 | 19.6 | 19.0 | 55.5 | 217.3 |

At 24 → 16 kHz, pipecat-resampler is the fastest of the filtered options at every chunk size, and its lead grows as chunks get smaller, because each call costs less. With 40 ms chunks, soxr VHQ is a little faster on 8 → 16 and 48 → 16 kHz, where its multistage FFT design is very cheap. Even pipecat-resampler's slowest case here, 17 µs per 40 ms chunk, is under 0.05% of one core per stream.

Run `uv run python scripts/bench.py` to measure on your machine. It times soxr, and also samplerate, PyAV and the Speex library when they're installed.

## How it works

It's a polyphase Kaiser-windowed sinc filter, the design WebRTC and Speex use. The filter keeps the lower rate's band flat up to 90% of its Nyquist frequency and attenuates aliasing and images by about 90 dB.

Each call works out how many output samples its input is worth and carries any fraction of a sample into the next call. That's what keeps the output steady, for any pair of rates and any chunk size. Instead of holding output back, the filter delays the audio by a few milliseconds; `flush()` returns that delayed end, and `reset()` discards it.

With several channels, the audio is interleaved frames, and each channel is filtered on its own with the same filter. The cost grows with the channel count, so stereo costs twice what mono does.

The inner loop picks the fastest kernel the CPU supports when a resampler is created: AVX2 with FMA or SSE2 on x86-64, NEON on ARM64, and plain C elsewhere. Filter phases with symmetric taps add mirrored samples before multiplying, which halves the multiplications. Set `PIPECAT_RESAMPLER_KERNEL` to `generic`, `sse2`, `avx2` or `neon` to force a kernel.

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
