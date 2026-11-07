/*
 * Copyright (c) 2026, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

#if !defined(_MSC_VER) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#include "_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define RS_SSE2 1
#include <emmintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define RS_AVX2 1
#include <immintrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define RS_NEON 1
#include <arm_neon.h>
#endif

/* How much the filter attenuates what the output rate can't carry, and how
 * much of the lower rate's band it keeps flat. */
#define ATTENUATION_DB 90.0
#define PASSBAND 0.9

/* Largest filter, in taps across all phases, a pair of rates may need. */
#define MAX_TAPS (1L << 22)

#define RS_PI 3.14159265358979323846

typedef float (*dot_fn)(const float *a, const float *t, int n);
typedef float (*fold_fn)(const float *w, int span, const float *t, int hp);

typedef struct {
    int fold; /* symmetric taps, computed on mirrored pairs */
    int len;  /* dense: taps, padded to a multiple of 8; folded: taps spanned */
    int hp;   /* folded: taps of the first half, padded to a multiple of 8 */
    float *taps;
} phase_t;

struct resampler {
    int in_rate, out_rate;
    int channels;
    int passthrough;
    int up, down, width;
    phase_t *phases;
    float *arena;
    /* One buffer per channel: width - 1 samples of history, the input, then
     * padding. */
    float **bufs;
    size_t cap;
    int pending; /* audio has gone in since the last reset */
    int64_t offset;
    int64_t step_whole;
    int step_frac;
    dot_fn dot;
    fold_fn fold;
    const char *kernel;
};

/* ---- Dot products. n and hp are multiples of 8; t is 32-byte aligned. ---- */

static float dot_generic(const float *a, const float *t, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int j = 0; j < n; j += 4) {
        s0 += a[j] * t[j];
        s1 += a[j + 1] * t[j + 1];
        s2 += a[j + 2] * t[j + 2];
        s3 += a[j + 3] * t[j + 3];
    }
    return (s0 + s1) + (s2 + s3);
}

static float fold_generic(const float *w, int span, const float *t, int hp) {
    float s0 = 0.0f, s1 = 0.0f;
    for (int j = 0; j < hp; j += 2) {
        s0 += t[j] * (w[j] + w[span - 1 - j]);
        s1 += t[j + 1] * (w[j + 1] + w[span - 2 - j]);
    }
    return s0 + s1;
}

#if RS_SSE2
static inline float hsum_sse(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}

static float dot_sse2(const float *a, const float *t, int n) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(a + j), _mm_load_ps(t + j)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(a + j + 4), _mm_load_ps(t + j + 4)));
        a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(a + j + 8), _mm_load_ps(t + j + 8)));
        a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(a + j + 12), _mm_load_ps(t + j + 12)));
    }
    for (; j < n; j += 8) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(a + j), _mm_load_ps(t + j)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(a + j + 4), _mm_load_ps(t + j + 4)));
    }
    return hsum_sse(_mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3)));
}

static float fold_sse2(const float *w, int span, const float *t, int hp) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    for (int j = 0; j < hp; j += 8) {
        __m128 r0 = _mm_loadu_ps(w + span - 4 - j);
        __m128 r1 = _mm_loadu_ps(w + span - 8 - j);
        r0 = _mm_shuffle_ps(r0, r0, _MM_SHUFFLE(0, 1, 2, 3));
        r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 1, 2, 3));
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_add_ps(_mm_loadu_ps(w + j), r0), _mm_load_ps(t + j)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_add_ps(_mm_loadu_ps(w + j + 4), r1), _mm_load_ps(t + j + 4)));
    }
    return hsum_sse(_mm_add_ps(a0, a1));
}
#endif

#if RS_AVX2
#define RS_AVX2_TARGET __attribute__((target("avx2,fma")))

RS_AVX2_TARGET static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

RS_AVX2_TARGET static float dot_avx2(const float *a, const float *t, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_load_ps(t + j), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 8), _mm256_load_ps(t + j + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 16), _mm256_load_ps(t + j + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 24), _mm256_load_ps(t + j + 24), a3);
    }
    for (; j < n; j += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_load_ps(t + j), a0);
    return hsum_avx(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}

RS_AVX2_TARGET static float fold_avx2(const float *w, int span, const float *t, int hp) {
    const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= hp; j += 16) {
        __m256 r0 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(w + span - 8 - j), rev);
        __m256 r1 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(w + span - 16 - j), rev);
        a0 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(w + j), r0), _mm256_load_ps(t + j), a0);
        a1 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(w + j + 8), r1), _mm256_load_ps(t + j + 8), a1);
    }
    for (; j < hp; j += 8) {
        __m256 r0 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(w + span - 8 - j), rev);
        a0 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(w + j), r0), _mm256_load_ps(t + j), a0);
    }
    return hsum_avx(_mm256_add_ps(a0, a1));
}

static int has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

#if RS_NEON
static float dot_neon(const float *a, const float *t, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + j), vld1q_f32(t + j));
        a1 = vfmaq_f32(a1, vld1q_f32(a + j + 4), vld1q_f32(t + j + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + j + 8), vld1q_f32(t + j + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + j + 12), vld1q_f32(t + j + 12));
    }
    for (; j < n; j += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + j), vld1q_f32(t + j));
        a1 = vfmaq_f32(a1, vld1q_f32(a + j + 4), vld1q_f32(t + j + 4));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

static inline float32x4_t reverse_neon(float32x4_t v) {
    v = vrev64q_f32(v);
    return vextq_f32(v, v, 2);
}

static float fold_neon(const float *w, int span, const float *t, int hp) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    for (int j = 0; j < hp; j += 8) {
        float32x4_t r0 = reverse_neon(vld1q_f32(w + span - 4 - j));
        float32x4_t r1 = reverse_neon(vld1q_f32(w + span - 8 - j));
        a0 = vfmaq_f32(a0, vaddq_f32(vld1q_f32(w + j), r0), vld1q_f32(t + j));
        a1 = vfmaq_f32(a1, vaddq_f32(vld1q_f32(w + j + 4), r1), vld1q_f32(t + j + 4));
    }
    return vaddvq_f32(vaddq_f32(a0, a1));
}
#endif

static void choose_kernel(resampler_t *r) {
    const char *wanted = getenv("PIPECAT_RESAMPLER_KERNEL");
    r->dot = dot_generic;
    r->fold = fold_generic;
    r->kernel = "generic";
    if (wanted && strcmp(wanted, "generic") == 0)
        return;
#if RS_SSE2
    r->dot = dot_sse2;
    r->fold = fold_sse2;
    r->kernel = "sse2";
    if (wanted && strcmp(wanted, "sse2") == 0)
        return;
#endif
#if RS_AVX2
    if (has_avx2()) {
        r->dot = dot_avx2;
        r->fold = fold_avx2;
        r->kernel = "avx2";
    }
#endif
#if RS_NEON
    r->dot = dot_neon;
    r->fold = fold_neon;
    r->kernel = "neon";
#endif
}

/* ---- Filter design ---- */

static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 500; k++) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < sum * 1e-17)
            break;
    }
    return sum;
}

static int gcd(int a, int b) {
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int round8(int n) { return (n + 7) & ~7; }

static void *aligned_alloc32(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, 32);
#else
    void *p = NULL;
    return posix_memalign(&p, 32, size) ? NULL : p;
#endif
}

static void aligned_free32(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* A Kaiser-windowed sinc low-pass split into one reversed filter per phase:
 * phase p, tap j is h[(width - 1 - j) * up + p]. */
static double *design_taps(const resampler_t *r, long length) {
    double up_rate = (double)r->in_rate * r->up;
    double nyquist = (r->in_rate < r->out_rate ? r->in_rate : r->out_rate) / 2.0;
    double transition = nyquist * (1.0 - PASSBAND);
    double cutoff = nyquist - transition / 2.0;
    double beta = 0.1102 * (ATTENUATION_DB - 8.7);
    double i0_beta = bessel_i0(beta);
    double alpha = (length - 1) / 2.0;
    double *h = malloc(sizeof(double) * length);
    if (!h)
        return NULL;
    double sum = 0.0;
    for (long k = 0; k < length; k++) {
        double x = 2.0 * cutoff / up_rate * (k - alpha);
        double sinc = x == 0.0 ? 1.0 : sin(RS_PI * x) / (RS_PI * x);
        double ratio = alpha > 0.0 ? (k - alpha) / alpha : 0.0;
        double window = bessel_i0(beta * sqrt(fmax(0.0, 1.0 - ratio * ratio))) / i0_beta;
        h[k] = sinc * window;
        sum += h[k];
    }
    double scale = r->up / sum;
    for (long k = 0; k < length; k++)
        h[k] *= scale;
    return h;
}

static void set_up_phase(phase_t *ph, const float *taps, int width, float *store) {
    float peak = 0.0f;
    for (int j = 0; j < width; j++)
        if (fabsf(taps[j]) > peak)
            peak = fabsf(taps[j]);
    int symmetric = width >= 16;
    for (int j = 0; j < width / 2 && symmetric; j++)
        if (fabsf(taps[j] - taps[width - 1 - j]) > 1e-6f * peak)
            symmetric = 0;

    ph->taps = store;
    if (symmetric) {
        int half = (width + 1) / 2;
        ph->fold = 1;
        ph->len = width;
        ph->hp = round8(half);
        memset(store, 0, sizeof(float) * ph->hp);
        memcpy(store, taps, sizeof(float) * half);
        /* The middle tap of an odd filter pairs with itself. */
        if (width % 2)
            store[half - 1] *= 0.5f;
    } else {
        ph->fold = 0;
        ph->len = round8(width);
        memset(store, 0, sizeof(float) * ph->len);
        memcpy(store, taps, sizeof(float) * width);
    }
}

resampler_t *resampler_new(int in_rate, int out_rate, int channels, resampler_status_t *status) {
    *status = RESAMPLER_OK;
    if (in_rate <= 0 || out_rate <= 0) {
        *status = RESAMPLER_BAD_RATE;
        return NULL;
    }
    if (channels <= 0) {
        *status = RESAMPLER_BAD_CHANNELS;
        return NULL;
    }
    resampler_t *r = calloc(1, sizeof *r);
    if (!r) {
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    r->in_rate = in_rate;
    r->out_rate = out_rate;
    r->channels = channels;
    choose_kernel(r);
    if (in_rate == out_rate) {
        r->passthrough = 1;
        return r;
    }

    int divisor = gcd(in_rate, out_rate);
    r->up = out_rate / divisor;
    r->down = in_rate / divisor;

    double up_rate = (double)in_rate * r->up;
    double nyquist = (in_rate < out_rate ? in_rate : out_rate) / 2.0;
    double transition = nyquist * (1.0 - PASSBAND);
    double estimate = (ATTENUATION_DB - 7.95) / (2.285 * 2.0 * RS_PI * transition / up_rate) + 1.0;
    double whole = ceil(estimate);
    if (whole + r->up > (double)MAX_TAPS) {
        free(r);
        *status = RESAMPLER_TOO_LARGE;
        return NULL;
    }
    long length = (long)ceil(whole / r->up) * r->up;
    r->width = (int)(length / r->up);

    double *h = design_taps(r, length);
    float *row = malloc(sizeof(float) * r->width);
    size_t stride = (size_t)round8(r->width + 8);
    r->phases = calloc((size_t)r->up, sizeof(phase_t));
    r->arena = aligned_alloc32(sizeof(float) * stride * r->up);
    r->bufs = calloc((size_t)channels, sizeof(float *));
    if (!h || !row || !r->phases || !r->arena || !r->bufs) {
        free(h);
        free(row);
        resampler_free(r);
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    for (int p = 0; p < r->up; p++) {
        for (int j = 0; j < r->width; j++)
            row[j] = (float)h[(size_t)(r->width - 1 - j) * r->up + p];
        set_up_phase(&r->phases[p], row, r->width, r->arena + stride * p);
    }
    free(h);
    free(row);

    r->step_whole = r->down / r->up;
    r->step_frac = r->down % r->up;
    return r;
}

void resampler_free(resampler_t *r) {
    if (!r)
        return;
    if (r->bufs)
        for (int c = 0; c < r->channels; c++)
            free(r->bufs[c]);
    free(r->bufs);
    aligned_free32(r->arena);
    free(r->phases);
    free(r);
}

int resampler_channels(const resampler_t *r) { return r->channels; }

size_t resampler_output_count(const resampler_t *r, size_t in_frames) {
    if (r->passthrough)
        return in_frames;
    int64_t total = (int64_t)in_frames * r->up - r->offset;
    return total > 0 ? (size_t)((total + r->down - 1) / r->down) : 0;
}

size_t resampler_delay(const resampler_t *r) {
    if (r->passthrough)
        return 0;
    int64_t taps = (int64_t)r->up * r->width - 1;
    return (size_t)((taps + 2 * r->up - 1) / (2 * r->up));
}

void resampler_reset(resampler_t *r) {
    if (r->bufs)
        for (int c = 0; c < r->channels; c++)
            if (r->bufs[c])
                memset(r->bufs[c], 0, sizeof(float) * (r->width - 1));
    r->offset = 0;
    r->pending = 0;
}

int resampler_pending(const resampler_t *r) { return r->pending; }

const char *resampler_kernel(const resampler_t *r) { return r->kernel; }

static inline int16_t to_int16(float v) {
    long rounded = lrintf(v);
    return (int16_t)(rounded > 32767 ? 32767 : (rounded < -32768 ? -32768 : rounded));
}

/* Filters one channel's buffer into every channels-th sample of out. */
static void filter_channel(const resampler_t *r, const float *b, size_t count, int16_t *out) {
    size_t stride = (size_t)r->channels;
    /* Output n sits at n * down on the input upsampled by up. The offset is
     * where the next output sits relative to the next input there. */
    int64_t start = r->offset / r->up;
    int phase = (int)(r->offset % r->up);
    for (size_t n = 0; n < count; n++) {
        const phase_t *ph = &r->phases[phase];
        const float *w = b + start;
        float v = ph->fold ? r->fold(w, ph->len, ph->taps, ph->hp) : r->dot(w, ph->taps, ph->len);
        out[n * stride] = to_int16(v);
        start += r->step_whole;
        phase += r->step_frac;
        if (phase >= r->up) {
            phase -= r->up;
            start++;
        }
    }
}

resampler_status_t resampler_process(resampler_t *r, const void *in, size_t in_frames,
                                     int16_t *out) {
    const unsigned char *bytes = in;
    size_t channels = (size_t)r->channels;
    if (r->passthrough) {
        memcpy(out, bytes, sizeof(int16_t) * in_frames * channels);
        return RESAMPLER_OK;
    }
    if (!in_frames)
        return RESAMPLER_OK;

    size_t hist = (size_t)r->width - 1;
    /* The padding lets the last dot product read whole SIMD registers. */
    size_t need = hist + in_frames + 8;
    if (need > r->cap) {
        for (size_t c = 0; c < channels; c++) {
            float *grown = realloc(r->bufs[c], sizeof(float) * need);
            if (!grown)
                return RESAMPLER_NO_MEMORY;
            if (!r->bufs[c])
                memset(grown, 0, sizeof(float) * hist);
            r->bufs[c] = grown;
        }
        r->cap = need;
    }

    for (size_t c = 0; c < channels; c++) {
        float *b = r->bufs[c];
        for (size_t i = 0; i < in_frames; i++) {
            int16_t sample;
            memcpy(&sample, bytes + sizeof sample * (i * channels + c), sizeof sample);
            b[hist + i] = (float)sample;
        }
        memset(b + hist + in_frames, 0, sizeof(float) * 8);
    }

    size_t count = resampler_output_count(r, in_frames);
    for (size_t c = 0; c < channels; c++) {
        filter_channel(r, r->bufs[c], count, out + c);
        memmove(r->bufs[c], r->bufs[c] + in_frames, sizeof(float) * hist);
    }
    r->offset += (int64_t)count * r->down - (int64_t)in_frames * r->up;
    r->pending = 1;
    return RESAMPLER_OK;
}
