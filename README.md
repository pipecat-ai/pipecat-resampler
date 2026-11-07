# pipecat-resampler

A streaming audio resampler for mono 16-bit PCM whose output keeps pace with its input. Every call returns exactly the audio its input amounts to at the output rate, so 40 ms in gives 40 ms out, the first chunk included. That makes it safe to use on audio that has already been paced, such as the chunks a real-time voice pipeline sends to a phone call.

Block resamplers like soxr give back their output in bursts instead: nothing for one chunk, then about 70 ms at once. A provider with a small playout buffer plays that back choppy.

```python
from pipecat_resampler import Resampler

resampler = Resampler(24000, 16000)
chunk_16k = resampler.process(chunk_24k)  # bytes in, bytes out
tail = resampler.flush()  # the audio the filter is delaying
```

## How it works

It's a polyphase Kaiser-windowed sinc filter. The filter keeps the lower rate's band flat up to 90% of its Nyquist frequency and attenuates aliasing and images by about 90 dB. It works for any pair of rates and any chunk size, and it delays the audio by a few milliseconds instead of holding output back.

The inner loop picks the fastest kernel the CPU supports when a resampler is created: AVX2 with FMA or SSE2 on x86-64, NEON on ARM64, and plain C elsewhere. Set `PIPECAT_RESAMPLER_KERNEL` to `generic`, `sse2`, `avx2` or `neon` to force one.

## Development

```bash
uv sync
uv run pytest
uv run python scripts/bench.py
```

Wheels are built with [cibuildwheel](https://cibuildwheel.pypa.io). The extension uses Python's stable ABI, so one wheel per platform covers Python 3.11 and later.
