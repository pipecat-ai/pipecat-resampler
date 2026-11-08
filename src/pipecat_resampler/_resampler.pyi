from typing_extensions import Buffer

class Resampler:
    """One-shot resampler for whole clips of 16-bit PCM."""

    def __init__(self, in_rate: int, out_rate: int, channels: int = 1) -> None: ...
    def resample(self, audio: Buffer) -> bytes:
        """Resample a whole clip of 16-bit PCM, interleaved frames when there are several channels.

        The output lines up with the input and holds the clip's length at the output rate,
        rounded to the nearest sample.
        """
    @property
    def in_rate(self) -> int:
        """Input sample rate in Hz."""
    @property
    def out_rate(self) -> int:
        """Output sample rate in Hz."""
    @property
    def channels(self) -> int:
        """Number of interleaved channels."""
    @property
    def kernel(self) -> str:
        """Name of the dot-product kernel in use: avx2, sse2, neon or generic."""

class StreamResampler:
    """Streaming resampler for 16-bit PCM whose output keeps pace with its input."""

    def __init__(self, in_rate: int, out_rate: int, channels: int = 1) -> None: ...
    def process(self, audio: Buffer) -> bytes:
        """Resample a chunk of 16-bit PCM, interleaved frames when there are several channels."""
    def flush(self) -> bytes:
        """Return the audio the filter is delaying and reset for the next stream.

        Empty when no audio has gone in since the resampler was created, flushed or reset.
        """
    def reset(self) -> None:
        """Discard the audio held in the filter."""
    @property
    def in_rate(self) -> int:
        """Input sample rate in Hz."""
    @property
    def out_rate(self) -> int:
        """Output sample rate in Hz."""
    @property
    def channels(self) -> int:
        """Number of interleaved channels."""
    @property
    def delay(self) -> int:
        """Input frames it takes for the last input to reach the output."""
    @property
    def kernel(self) -> str:
        """Name of the dot-product kernel in use: avx2, sse2, neon or generic."""
