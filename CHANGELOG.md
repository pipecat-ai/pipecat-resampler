# Changelog

All notable changes to **pipecat-resampler** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
