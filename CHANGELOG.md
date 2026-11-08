# Changelog

All notable changes to **pipecat-resampler** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-08

### Added

- `Resampler`, a one-shot resampler for whole clips with the same filter.
  Its output lines up with its input and keeps the clip's length, rounded to
  the nearest sample, and it releases the GIL while it works.

### Changed

- The streaming resampler is now `StreamResampler`. `Resampler` is the new
  one-shot resampler.

- Filters have an odd number of taps, which can shift streaming output by
  less than a sample. Output sizes and the delay stay the same.

- Resamplers between the same two rates share one filter, so each is built
  and held once per process.

- Resampling between rates like 48 → 16, 24 → 16, 24 → 8 and 8 → 16 kHz
  takes a quarter to two fifths less CPU with AVX2, and a tenth to a fifth
  less with SSE2.

- With NEON, on ARM64, rates like 48 → 16, 24 → 16, 24 → 8 and 8 → 16 kHz
  take a third to nearly half less CPU. The 44.1 kHz family takes about half
  as much with 40 ms chunks and whole clips, and up to a quarter less with 10
  and 20 ms chunks.

## [0.1.1] - 2026-11-07

### Fixed

- `flush()` returns nothing when no audio has gone in since the resampler was
  created, flushed or reset. It returned a few milliseconds of silence, which
  a caller that flushes at the end of every stream could send as audio.

## [0.1.0] - 2026-11-07

### Added

- Initial release of `Resampler`, a streaming polyphase windowed-sinc resampler
  for 16-bit PCM whose output keeps pace with its input: equal chunks in give
  equal chunks out, the first one included.

- Mono or any number of interleaved channels, with `channels=`.

- SIMD kernels picked at runtime: AVX2 with FMA or SSE2 on x86-64, NEON on
  ARM64, and plain C elsewhere.

- Stable-ABI wheels for Python 3.11 and later on Linux (glibc 2.17 or later,
  and Alpine), macOS and Windows.
