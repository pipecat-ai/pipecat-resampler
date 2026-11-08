/*
 * Copyright (c) 2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/*
 * Polyphase windowed-sinc resampler for 16-bit PCM, mono or interleaved
 * multichannel, streaming or one-shot.
 *
 * A filter holds everything a pair of rates needs and never changes once
 * made, so any number of resamplers and threads can share one.
 *
 * Streaming, a resampler keeps the recent input of each channel. Every
 * resampler_process() call returns exactly the audio its input amounts to at
 * the output rate, carrying any fraction of a sample into the next call, so
 * equal input chunks come out as equal output chunks, the first one included.
 * The filter delays the audio by a few milliseconds instead of holding output
 * back.
 *
 * One-shot, filter_resample() converts a whole clip. Its output lines up with
 * its input, and the clip is taken to be silent before and after.
 *
 * Lengths are in frames: one sample for every channel.
 */

#ifndef PIPECAT_RESAMPLER_KERNEL_H
#define PIPECAT_RESAMPLER_KERNEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct filter filter_t;
typedef struct resampler resampler_t;

typedef enum {
    RESAMPLER_OK = 0,
    RESAMPLER_BAD_RATE,
    RESAMPLER_BAD_CHANNELS,
    RESAMPLER_TOO_LARGE,
    RESAMPLER_NO_MEMORY,
} resampler_status_t;

/* ---- Filters ---- */

/* Creates the filter between two rates, or returns NULL and sets status. */
filter_t *filter_new(int in_rate, int out_rate, resampler_status_t *status);

void filter_free(filter_t *f);

/* Input frames it takes for the last input to reach the output. */
size_t filter_delay(const filter_t *f);

/* Name of the dot-product kernel this filter uses. */
const char *filter_kernel(const filter_t *f);

/* Number of output frames filter_resample() returns for in_frames input frames:
 * the clip's length at the output rate, rounded to the nearest frame. */
size_t filter_resample_count(const filter_t *f, size_t in_frames);

/*
 * Resamples a whole clip of in_frames frames of interleaved, native-endian
 * int16 audio, which need not be aligned, into out, which holds
 * filter_resample_count(f, in_frames) frames. Output frame n is the clip at
 * input frame n * in_rate / out_rate. Returns RESAMPLER_OK or
 * RESAMPLER_NO_MEMORY.
 */
resampler_status_t filter_resample(const filter_t *f, int channels, const void *in,
                                   size_t in_frames, int16_t *out);

/* ---- Streaming resamplers ---- */

/* Creates a streaming resampler that filters with f, which must outlive it,
 * or returns NULL and sets status. */
resampler_t *resampler_new(const filter_t *f, int channels, resampler_status_t *status);

void resampler_free(resampler_t *r);

/* Number of output frames the next call with in_frames input frames returns. */
size_t resampler_output_count(const resampler_t *r, size_t in_frames);

/*
 * Resamples in_frames frames of interleaved, native-endian int16 audio, which
 * need not be aligned, into out, which holds resampler_output_count(r,
 * in_frames) frames. Returns RESAMPLER_OK or RESAMPLER_NO_MEMORY.
 */
resampler_status_t resampler_process(resampler_t *r, const void *in, size_t in_frames,
                                     int16_t *out);

/* Discards the audio held in the filter. */
void resampler_reset(resampler_t *r);

/* Whether audio has gone in since the resampler was created or last reset. */
int resampler_pending(const resampler_t *r);

#endif
