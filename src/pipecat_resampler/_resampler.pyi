from typing_extensions import Buffer

class Resampler:
    """Streaming resampler for mono 16-bit PCM whose output keeps pace with its input."""

    def __init__(self, in_rate: int, out_rate: int) -> None: ...
    def process(self, audio: Buffer) -> bytes:
        """Resample a chunk of mono 16-bit PCM and return the resampled bytes."""
    def flush(self) -> bytes:
        """Return the audio the filter is delaying and reset for the next stream."""
    def reset(self) -> None:
        """Discard the audio held in the filter."""
    @property
    def in_rate(self) -> int:
        """Input sample rate in Hz."""
    @property
    def out_rate(self) -> int:
        """Output sample rate in Hz."""
    @property
    def delay(self) -> int:
        """Input samples it takes for the last input to reach the output."""
    @property
    def kernel(self) -> str:
        """Name of the dot-product kernel in use: avx2, sse2, neon or generic."""
