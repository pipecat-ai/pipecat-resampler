/*
 * Copyright (c) 2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/*
 * Streaming polyphase windowed-sinc resampler for 16-bit PCM, mono or
 * interleaved multichannel.
 *
 * Every call returns exactly the audio its input amounts to at the output
 * rate, carrying any fraction of a sample into the next call, so equal input
 * chunks come out as equal output chunks, the first one included. The filter
 * delays the audio by a few milliseconds instead of holding output back.
 *
 * Lengths are in frames: one sample for every channel.
 */

#ifndef PIPECAT_RESAMPLER_KERNEL_H
#define PIPECAT_RESAMPLER_KERNEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct resampler resampler_t;

typedef enum {
    RESAMPLER_OK = 0,
    RESAMPLER_BAD_RATE,
    RESAMPLER_BAD_CHANNELS,
    RESAMPLER_TOO_LARGE,
    RESAMPLER_NO_MEMORY,
} resampler_status_t;

/* Creates a resampler between two rates, or returns NULL and sets status. */
resampler_t *resampler_new(int in_rate, int out_rate, int channels, resampler_status_t *status);

void resampler_free(resampler_t *r);

int resampler_channels(const resampler_t *r);

/* Number of output frames the next call with in_frames input frames returns. */
size_t resampler_output_count(const resampler_t *r, size_t in_frames);

/*
 * Resamples in_frames frames of interleaved, native-endian int16 audio, which
 * need not be aligned, into out, which holds resampler_output_count(r,
 * in_frames) frames. Returns RESAMPLER_OK or RESAMPLER_NO_MEMORY.
 */
resampler_status_t resampler_process(resampler_t *r, const void *in, size_t in_frames,
                                     int16_t *out);

/* Input frames it takes for the last input to reach the output. */
size_t resampler_delay(const resampler_t *r);

/* Discards the audio held in the filter. */
void resampler_reset(resampler_t *r);

/* Name of the dot-product kernel this resampler uses. */
const char *resampler_kernel(const resampler_t *r);

#endif
