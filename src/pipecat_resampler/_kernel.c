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

/* Filters with at most this many phases compute four outputs per pass, where
 * the kernel can. Every output of a filter is computed the same way, so how
 * the input is chunked doesn't change the result, and with more phases too
 * few outputs in a chunk share each phase for four at a time to pay off.
 * Where the four-wide pass gives each output exactly what the one-wide pass
 * does, any filter computes four at a time where four outputs in a chunk
 * share a phase, and one at a time elsewhere. */
#define BLOCK_MAX_UP 4

typedef float (*dot_fn)(const float *a, const float *t, int n);
typedef float (*fold_fn)(const float *w, int span, const float *t, int hp);
/* Four dot products with the same taps, over windows stride samples apart. */
typedef void (*dot4_fn)(const float *a, ptrdiff_t stride, const float *t, int n, float *out);

typedef struct {
    int fold; /* symmetric taps, computed on mirrored pairs */
    int len;  /* dense: taps, padded to a multiple of 8; folded: taps spanned */
    int hp;   /* folded: taps of the first half, padded to a multiple of 8 */
    float *taps;
} phase_t;

struct filter {
    int in_rate, out_rate;
    int passthrough;
    int up, down, width;
    /* The middle tap, on the input upsampled by up. */
    int64_t center;
    int64_t step_whole;
    int step_frac;
    phase_t *phases;
    float *arena;
    dot_fn dot;
    fold_fn fold; /* NULL where the plain dot product is faster */
    dot4_fn dot4; /* NULL where outputs are computed one at a time */
    int exact4;   /* dot4 gives each window exactly what dot does */
    int pad;      /* samples past the input that dot4 may read */
    const char *kernel;
};

struct resampler {
    const filter_t *f;
    int channels;
    /* One buffer per channel: width - 1 samples of history, the input, then
     * padding. */
    float **bufs;
    size_t cap;
    int pending; /* audio has gone in since the last reset */
    int64_t offset;
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

/* Loads each tap once for all four windows. Each window keeps its own two
 * sums, so its result doesn't depend on the other three. */
static void dot4_sse2(const float *a, ptrdiff_t stride, const float *t, int n, float *out) {
    const float *a0 = a, *a1 = a + stride, *a2 = a + 2 * stride, *a3 = a + 3 * stride;
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    __m128 u0 = _mm_setzero_ps(), u1 = _mm_setzero_ps();
    __m128 u2 = _mm_setzero_ps(), u3 = _mm_setzero_ps();
    for (int j = 0; j < n; j += 8) {
        __m128 t0 = _mm_load_ps(t + j), t1 = _mm_load_ps(t + j + 4);
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a0 + j), t0));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(a1 + j), t0));
        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(a2 + j), t0));
        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(a3 + j), t0));
        u0 = _mm_add_ps(u0, _mm_mul_ps(_mm_loadu_ps(a0 + j + 4), t1));
        u1 = _mm_add_ps(u1, _mm_mul_ps(_mm_loadu_ps(a1 + j + 4), t1));
        u2 = _mm_add_ps(u2, _mm_mul_ps(_mm_loadu_ps(a2 + j + 4), t1));
        u3 = _mm_add_ps(u3, _mm_mul_ps(_mm_loadu_ps(a3 + j + 4), t1));
    }
    out[0] = hsum_sse(_mm_add_ps(s0, u0));
    out[1] = hsum_sse(_mm_add_ps(s1, u1));
    out[2] = hsum_sse(_mm_add_ps(s2, u2));
    out[3] = hsum_sse(_mm_add_ps(s3, u3));
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

/* Loads each tap once for all four windows. Each window keeps its own two
 * sums, so its result doesn't depend on the other three. */
RS_AVX2_TARGET static void dot4_avx2(const float *a, ptrdiff_t stride, const float *t, int n,
                                     float *out) {
    const float *a0 = a, *a1 = a + stride, *a2 = a + 2 * stride, *a3 = a + 3 * stride;
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    __m256 u0 = _mm256_setzero_ps(), u1 = _mm256_setzero_ps();
    __m256 u2 = _mm256_setzero_ps(), u3 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        __m256 t0 = _mm256_load_ps(t + j), t1 = _mm256_load_ps(t + j + 8);
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + j), t0, s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + j), t0, s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + j), t0, s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + j), t0, s3);
        u0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + j + 8), t1, u0);
        u1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + j + 8), t1, u1);
        u2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + j + 8), t1, u2);
        u3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + j + 8), t1, u3);
    }
    for (; j < n; j += 8) {
        __m256 t0 = _mm256_load_ps(t + j);
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + j), t0, s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + j), t0, s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + j), t0, s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + j), t0, s3);
    }
    out[0] = hsum_avx(_mm256_add_ps(s0, u0));
    out[1] = hsum_avx(_mm256_add_ps(s1, u1));
    out[2] = hsum_avx(_mm256_add_ps(s2, u2));
    out[3] = hsum_avx(_mm256_add_ps(s3, u3));
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

/* Loads each tap once for all four windows. Each window keeps four sums and
 * adds them up the way dot_neon does, so its result is exactly dot_neon's. */
static void dot4_neon(const float *a, ptrdiff_t stride, const float *t, int n, float *out) {
    const float *a0 = a, *a1 = a + stride, *a2 = a + 2 * stride, *a3 = a + 3 * stride;
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    float32x4_t u0 = vdupq_n_f32(0.0f), u1 = vdupq_n_f32(0.0f);
    float32x4_t u2 = vdupq_n_f32(0.0f), u3 = vdupq_n_f32(0.0f);
    float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f), v3 = vdupq_n_f32(0.0f);
    float32x4_t w0 = vdupq_n_f32(0.0f), w1 = vdupq_n_f32(0.0f);
    float32x4_t w2 = vdupq_n_f32(0.0f), w3 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        float32x4_t t0 = vld1q_f32(t + j), t1 = vld1q_f32(t + j + 4);
        float32x4_t t2 = vld1q_f32(t + j + 8), t3 = vld1q_f32(t + j + 12);
        s0 = vfmaq_f32(s0, vld1q_f32(a0 + j), t0);
        s1 = vfmaq_f32(s1, vld1q_f32(a1 + j), t0);
        s2 = vfmaq_f32(s2, vld1q_f32(a2 + j), t0);
        s3 = vfmaq_f32(s3, vld1q_f32(a3 + j), t0);
        u0 = vfmaq_f32(u0, vld1q_f32(a0 + j + 4), t1);
        u1 = vfmaq_f32(u1, vld1q_f32(a1 + j + 4), t1);
        u2 = vfmaq_f32(u2, vld1q_f32(a2 + j + 4), t1);
        u3 = vfmaq_f32(u3, vld1q_f32(a3 + j + 4), t1);
        v0 = vfmaq_f32(v0, vld1q_f32(a0 + j + 8), t2);
        v1 = vfmaq_f32(v1, vld1q_f32(a1 + j + 8), t2);
        v2 = vfmaq_f32(v2, vld1q_f32(a2 + j + 8), t2);
        v3 = vfmaq_f32(v3, vld1q_f32(a3 + j + 8), t2);
        w0 = vfmaq_f32(w0, vld1q_f32(a0 + j + 12), t3);
        w1 = vfmaq_f32(w1, vld1q_f32(a1 + j + 12), t3);
        w2 = vfmaq_f32(w2, vld1q_f32(a2 + j + 12), t3);
        w3 = vfmaq_f32(w3, vld1q_f32(a3 + j + 12), t3);
    }
    for (; j < n; j += 8) {
        float32x4_t t0 = vld1q_f32(t + j), t1 = vld1q_f32(t + j + 4);
        s0 = vfmaq_f32(s0, vld1q_f32(a0 + j), t0);
        s1 = vfmaq_f32(s1, vld1q_f32(a1 + j), t0);
        s2 = vfmaq_f32(s2, vld1q_f32(a2 + j), t0);
        s3 = vfmaq_f32(s3, vld1q_f32(a3 + j), t0);
        u0 = vfmaq_f32(u0, vld1q_f32(a0 + j + 4), t1);
        u1 = vfmaq_f32(u1, vld1q_f32(a1 + j + 4), t1);
        u2 = vfmaq_f32(u2, vld1q_f32(a2 + j + 4), t1);
        u3 = vfmaq_f32(u3, vld1q_f32(a3 + j + 4), t1);
    }
    out[0] = vaddvq_f32(vaddq_f32(vaddq_f32(s0, u0), vaddq_f32(v0, w0)));
    out[1] = vaddvq_f32(vaddq_f32(vaddq_f32(s1, u1), vaddq_f32(v1, w1)));
    out[2] = vaddvq_f32(vaddq_f32(vaddq_f32(s2, u2), vaddq_f32(v2, w2)));
    out[3] = vaddvq_f32(vaddq_f32(vaddq_f32(s3, u3), vaddq_f32(v3, w3)));
}
#endif

static void choose_kernel(filter_t *f) {
    const char *wanted = getenv("PIPECAT_RESAMPLER_KERNEL");
    f->dot = dot_generic;
    f->fold = fold_generic;
    f->dot4 = NULL;
    f->kernel = "generic";
    if (wanted && strcmp(wanted, "generic") == 0)
        return;
#if RS_SSE2
    f->dot = dot_sse2;
    f->fold = fold_sse2;
    f->dot4 = dot4_sse2;
    f->kernel = "sse2";
    if (wanted && strcmp(wanted, "sse2") == 0)
        return;
#endif
#if RS_AVX2
    if (has_avx2()) {
        f->dot = dot_avx2;
        f->fold = NULL;
        f->dot4 = dot4_avx2;
        f->kernel = "avx2";
    }
#endif
#if RS_NEON
    f->dot = dot_neon;
    f->fold = NULL;
    f->dot4 = dot4_neon;
    f->exact4 = 1;
    f->kernel = "neon";
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

/* A Kaiser-windowed sinc low-pass with taps coefficients, padded with zeros
 * to length, to be split into one reversed filter per phase: phase p, tap j
 * is h[(width - 1 - j) * up + p]. */
static double *design_taps(const filter_t *f, long length, long taps) {
    double up_rate = (double)f->in_rate * f->up;
    double nyquist = (f->in_rate < f->out_rate ? f->in_rate : f->out_rate) / 2.0;
    double transition = nyquist * (1.0 - PASSBAND);
    double cutoff = nyquist - transition / 2.0;
    double beta = 0.1102 * (ATTENUATION_DB - 8.7);
    double i0_beta = bessel_i0(beta);
    double alpha = (taps - 1) / 2.0;
    double *h = calloc((size_t)length, sizeof(double));
    if (!h)
        return NULL;
    double sum = 0.0;
    for (long k = 0; k < taps; k++) {
        double x = 2.0 * cutoff / up_rate * (k - alpha);
        double sinc = x == 0.0 ? 1.0 : sin(RS_PI * x) / (RS_PI * x);
        double ratio = alpha > 0.0 ? (k - alpha) / alpha : 0.0;
        double window = bessel_i0(beta * sqrt(fmax(0.0, 1.0 - ratio * ratio))) / i0_beta;
        h[k] = sinc * window;
        sum += h[k];
    }
    double scale = f->up / sum;
    for (long k = 0; k < taps; k++)
        h[k] *= scale;
    return h;
}

static void set_up_phase(phase_t *ph, const float *taps, int width, int can_fold, float *store) {
    float peak = 0.0f;
    for (int j = 0; j < width; j++)
        if (fabsf(taps[j]) > peak)
            peak = fabsf(taps[j]);
    int symmetric = can_fold && width >= 16;
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

/* ---- Filters ---- */

filter_t *filter_new(int in_rate, int out_rate, resampler_status_t *status) {
    *status = RESAMPLER_OK;
    if (in_rate <= 0 || out_rate <= 0) {
        *status = RESAMPLER_BAD_RATE;
        return NULL;
    }
    filter_t *f = calloc(1, sizeof *f);
    if (!f) {
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    f->in_rate = in_rate;
    f->out_rate = out_rate;
    choose_kernel(f);
    if (in_rate == out_rate) {
        f->passthrough = 1;
        return f;
    }

    int divisor = gcd(in_rate, out_rate);
    f->up = out_rate / divisor;
    f->down = in_rate / divisor;
    if (f->up > BLOCK_MAX_UP && !f->exact4)
        f->dot4 = NULL;
    f->pad = f->dot4 && !f->exact4 ? 3 * f->down : 0;

    double up_rate = (double)in_rate * f->up;
    double nyquist = (in_rate < out_rate ? in_rate : out_rate) / 2.0;
    double transition = nyquist * (1.0 - PASSBAND);
    double estimate = (ATTENUATION_DB - 7.95) / (2.285 * 2.0 * RS_PI * transition / up_rate) + 1.0;
    double whole = ceil(estimate);
    if (whole + f->up > (double)MAX_TAPS) {
        free(f);
        *status = RESAMPLER_TOO_LARGE;
        return NULL;
    }
    /* An odd number of taps puts the filter's middle, where each output sits,
     * on a whole upsampled sample. Zeros pad it to whole phases. */
    long taps = (long)whole | 1;
    f->width = (int)((taps + f->up - 1) / f->up);
    long length = (long)f->width * f->up;
    f->center = (taps - 1) / 2;

    double *h = design_taps(f, length, taps);
    float *row = malloc(sizeof(float) * f->width);
    size_t stride = (size_t)round8(f->width + 8);
    f->phases = calloc((size_t)f->up, sizeof(phase_t));
    f->arena = aligned_alloc32(sizeof(float) * stride * f->up);
    if (!h || !row || !f->phases || !f->arena) {
        free(h);
        free(row);
        filter_free(f);
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    for (int p = 0; p < f->up; p++) {
        for (int j = 0; j < f->width; j++)
            row[j] = (float)h[(size_t)(f->width - 1 - j) * f->up + p];
        set_up_phase(&f->phases[p], row, f->width, f->fold && !f->dot4, f->arena + stride * p);
    }
    free(h);
    free(row);

    f->step_whole = f->down / f->up;
    f->step_frac = f->down % f->up;
    return f;
}

void filter_free(filter_t *f) {
    if (!f)
        return;
    aligned_free32(f->arena);
    free(f->phases);
    free(f);
}

size_t filter_delay(const filter_t *f) {
    if (f->passthrough)
        return 0;
    return (size_t)((f->center + f->up - 1) / f->up);
}

const char *filter_kernel(const filter_t *f) { return f->kernel; }

size_t filter_resample_count(const filter_t *f, size_t in_frames) {
    if (f->passthrough)
        return in_frames;
    return (size_t)(((int64_t)in_frames * 2 * f->up + f->down) / (2 * f->down));
}

static inline int16_t to_int16(float v) {
    long rounded = lrintf(v);
    return (int16_t)(rounded > 32767 ? 32767 : (rounded < -32768 ? -32768 : rounded));
}

/* Copies one channel of interleaved int16 frames, which need not be aligned,
 * into b as floats. */
static void load_channel(float *b, const unsigned char *bytes, size_t frames, size_t channels,
                         size_t c) {
    for (size_t i = 0; i < frames; i++) {
        int16_t sample;
        memcpy(&sample, bytes + sizeof sample * (i * channels + c), sizeof sample);
        b[i] = (float)sample;
    }
}

/* Filters one channel's buffer into every stride-th sample of out, four
 * outputs per pass. Outputs n, n + up, n + 2 up and n + 3 up use the same
 * phase, with windows down samples apart, so each block of 4 up outputs
 * starts 4 down samples after the last, on the same phase. Where fewer than
 * four outputs on a phase are left, an exact kernel computes them one at a
 * time; otherwise lanes past count read the padding, and their results are
 * dropped. */
static void filter_channel4(const filter_t *f, const float *b, int64_t offset, size_t count,
                            size_t stride, int16_t *out) {
    size_t up = (size_t)f->up;
    int64_t block_start = offset / f->up;
    int block_phase = (int)(offset % f->up);
    for (size_t first = 0; first < count; first += 4 * up, block_start += 4 * f->down) {
        size_t left = count - first;
        int64_t start = block_start;
        int phase = block_phase;
        for (size_t k = 0; k < up && k < left; k++) {
            const phase_t *ph = &f->phases[phase];
            float v[4];
            if (k + 3 * up < left || !f->exact4)
                f->dot4(b + start, f->down, ph->taps, ph->len, v);
            else
                for (size_t i = 0; k + i * up < left; i++)
                    v[i] = f->dot(b + start + (int64_t)i * f->down, ph->taps, ph->len);
            for (size_t i = 0; i < 4 && k + i * up < left; i++)
                out[(first + k + i * up) * stride] = to_int16(v[i]);
            start += f->step_whole;
            phase += f->step_frac;
            if (phase >= f->up) {
                phase -= f->up;
                start++;
            }
        }
    }
}

/* Filters one channel's buffer into every stride-th sample of out. */
static void filter_channel(const filter_t *f, const float *b, int64_t offset, size_t count,
                           size_t stride, int16_t *out) {
    if (f->dot4) {
        filter_channel4(f, b, offset, count, stride, out);
        return;
    }
    /* Output n sits at n * down on the input upsampled by up. The offset is
     * where the first output sits relative to the first input there. */
    int64_t start = offset / f->up;
    int phase = (int)(offset % f->up);
    for (size_t n = 0; n < count; n++) {
        const phase_t *ph = &f->phases[phase];
        const float *w = b + start;
        float v = ph->fold ? f->fold(w, ph->len, ph->taps, ph->hp) : f->dot(w, ph->taps, ph->len);
        out[n * stride] = to_int16(v);
        start += f->step_whole;
        phase += f->step_frac;
        if (phase >= f->up) {
            phase -= f->up;
            start++;
        }
    }
}

resampler_status_t filter_resample(const filter_t *f, int channels, const void *in,
                                   size_t in_frames, int16_t *out) {
    const unsigned char *bytes = in;
    size_t stride = (size_t)channels;
    if (f->passthrough) {
        memcpy(out, bytes, sizeof(int16_t) * in_frames * stride);
        return RESAMPLER_OK;
    }
    size_t count = filter_resample_count(f, in_frames);
    if (!count)
        return RESAMPLER_OK;

    /* Silence before the clip fills the first window, and silence after it
     * the windows that reach past its end. The padding lets the last dot
     * product read whole SIMD registers, and four-wide passes read further. */
    size_t hist = (size_t)f->width - 1;
    size_t reach = (size_t)((f->center + (int64_t)(count - 1) * f->down) / f->up) + f->width;
    size_t len = (reach > hist + in_frames ? reach : hist + in_frames) + 8 + f->pad;
    float *b = malloc(sizeof(float) * len);
    if (!b)
        return RESAMPLER_NO_MEMORY;
    memset(b, 0, sizeof(float) * hist);
    memset(b + hist + in_frames, 0, sizeof(float) * (len - hist - in_frames));
    for (size_t c = 0; c < stride; c++) {
        load_channel(b + hist, bytes, in_frames, stride, c);
        filter_channel(f, b, f->center, count, stride, out + c);
    }
    free(b);
    return RESAMPLER_OK;
}

/* ---- Streaming resamplers ---- */

resampler_t *resampler_new(const filter_t *f, int channels, resampler_status_t *status) {
    *status = RESAMPLER_OK;
    if (channels <= 0) {
        *status = RESAMPLER_BAD_CHANNELS;
        return NULL;
    }
    resampler_t *r = calloc(1, sizeof *r);
    if (r)
        r->bufs = calloc((size_t)channels, sizeof(float *));
    if (!r || !r->bufs) {
        free(r);
        *status = RESAMPLER_NO_MEMORY;
        return NULL;
    }
    r->f = f;
    r->channels = channels;
    return r;
}

void resampler_free(resampler_t *r) {
    if (!r)
        return;
    for (int c = 0; c < r->channels; c++)
        free(r->bufs[c]);
    free(r->bufs);
    free(r);
}

size_t resampler_output_count(const resampler_t *r, size_t in_frames) {
    const filter_t *f = r->f;
    if (f->passthrough)
        return in_frames;
    int64_t total = (int64_t)in_frames * f->up - r->offset;
    return total > 0 ? (size_t)((total + f->down - 1) / f->down) : 0;
}

resampler_status_t resampler_process(resampler_t *r, const void *in, size_t in_frames,
                                     int16_t *out) {
    const filter_t *f = r->f;
    const unsigned char *bytes = in;
    size_t channels = (size_t)r->channels;
    if (f->passthrough) {
        memcpy(out, bytes, sizeof(int16_t) * in_frames * channels);
        return RESAMPLER_OK;
    }
    if (!in_frames)
        return RESAMPLER_OK;

    size_t hist = (size_t)f->width - 1;
    /* The padding lets the last dot product read whole SIMD registers, and
     * four-wide passes read further. */
    size_t tail = 8 + (size_t)f->pad;
    size_t need = hist + in_frames + tail;
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
        load_channel(r->bufs[c] + hist, bytes, in_frames, channels, c);
        memset(r->bufs[c] + hist + in_frames, 0, sizeof(float) * tail);
    }

    size_t count = resampler_output_count(r, in_frames);
    for (size_t c = 0; c < channels; c++) {
        filter_channel(f, r->bufs[c], r->offset, count, channels, out + c);
        memmove(r->bufs[c], r->bufs[c] + in_frames, sizeof(float) * hist);
    }
    r->offset += (int64_t)count * f->down - (int64_t)in_frames * f->up;
    r->pending = 1;
    return RESAMPLER_OK;
}

void resampler_reset(resampler_t *r) {
    for (int c = 0; c < r->channels; c++)
        if (r->bufs[c])
            memset(r->bufs[c], 0, sizeof(float) * (r->f->width - 1));
    r->offset = 0;
    r->pending = 0;
}

int resampler_pending(const resampler_t *r) { return r->pending; }
