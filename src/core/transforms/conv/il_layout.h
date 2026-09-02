/**
 * il_layout.h -- interleaved-complex boundary support (v1.1 P1a).
 *
 * TWO PIECES:
 *
 * 1. CONVERSION KERNELS: il2sp / sp2il full-array converters between the
 *    world's interleaved pairs z[2f]=re, z[2f+1]=im and the engine's split
 *    planes. AVX2 4-complex-per-iteration (unpacklo/hi + permute2f128 --
 *    4 shuffles per 4 complex, load/store-bound in practice) with scalar
 *    tails; exact (moves, no arithmetic).
 *
 * 2. THE UNIVERSAL IL WRAPPER: stride_il_t wraps ANY stride plan -- a 1D
 *    lane-batched plan (flat split arrays), or the fft2d/fft3d/fftnd
 *    override wraps -- and exposes fwd/bwd on a single interleaved buffer.
 *    The wrapper owns a split working cube; fwd = il2sp sweep -> plan fwd
 *    -> sp2il sweep, bwd mirrored. That is TWO explicit conversion sweeps
 *    per direction: the honest P1a cost ceiling, measured below in
 *    v1_0_results (§2b addendum). The roadmap removes them in two steps
 *    (interleaved_design.md): P1b fuses the output/input boundary into the
 *    tiled pass's own gather/scatter (which the strided/natural work made
 *    cheaper still), and P2's emitter flags fuse the remaining native-pass
 *    boundary. The wrapper's API is the stable contract; the sweeps are an
 *    implementation stage.
 *
 * BATCH GEOMETRY NOTE (design doc pitfall #1): this wrapper converts the
 * COMPONENT layout only. It serves multi-dim transforms (no batch
 * dimension), 1D K=1, and lane-major-interleaved batches (element-major
 * pairs -- the cheap-for-us geometry). Transform-major batched-1D
 * (z[k*2N + 2i], FFTW/MKL idist convention) is the corner-turn problem --
 * explicitly P3, not silently mishandled here.
 *
 * Order contract unchanged: whatever the wrapped plan emits (scrambled, or
 * natural-per-axis under strided rows), the wrapper reproduces in
 * interleaved pairs. Conversions are exact, so all bit-level gates carry.
 */
#ifndef VFFT_IL_LAYOUT_H
#define VFFT_IL_LAYOUT_H

#include <stdlib.h>
#include <string.h>
#include "executor.h"
#include "proto_stride_compat.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ── converters ─────────────────────────────────────────────────── */

/** z[2f],z[2f+1] -> re[f],im[f], n complex elements. */
static void vfft_il2sp(const double *z, double *re, double *im, size_t n) {
    size_t f = 0;
#if defined(__AVX2__)
    for (; f + 4 <= n; f += 4) {
        __m256d a = _mm256_loadu_pd(z + 2 * f);        /* r0 i0 r1 i1 */
        __m256d b = _mm256_loadu_pd(z + 2 * f + 4);    /* r2 i2 r3 i3 */
        __m256d lo = _mm256_unpacklo_pd(a, b);         /* r0 r2 r1 r3 */
        __m256d hi = _mm256_unpackhi_pd(a, b);         /* i0 i2 i1 i3 */
        _mm256_storeu_pd(re + f, _mm256_permute4x64_pd(lo, 0xD8)); /* r0 r1 r2 r3 */
        _mm256_storeu_pd(im + f, _mm256_permute4x64_pd(hi, 0xD8));
    }
#endif
    for (; f < n; f++) { re[f] = z[2*f]; im[f] = z[2*f + 1]; }
}

/** re[f],im[f] -> z[2f],z[2f+1], n complex elements. */
static void vfft_sp2il(const double *re, const double *im, double *z, size_t n) {
    size_t f = 0;
#if defined(__AVX2__)
    for (; f + 4 <= n; f += 4) {
        __m256d r = _mm256_loadu_pd(re + f);           /* r0 r1 r2 r3 */
        __m256d i = _mm256_loadu_pd(im + f);           /* i0 i1 i2 i3 */
        __m256d rp = _mm256_permute4x64_pd(r, 0xD8);   /* r0 r2 r1 r3 */
        __m256d ip = _mm256_permute4x64_pd(i, 0xD8);   /* i0 i2 i1 i3 */
        _mm256_storeu_pd(z + 2*f,     _mm256_unpacklo_pd(rp, ip)); /* r0 i0 r1 i1 */
        _mm256_storeu_pd(z + 2*f + 4, _mm256_unpackhi_pd(rp, ip)); /* r2 i2 r3 i3 */
    }
#endif
    for (; f < n; f++) { z[2*f] = re[f]; z[2*f + 1] = im[f]; }
}

/* ── the universal wrapper ──────────────────────────────────────── */

typedef struct {
    stride_plan_t *plan;       /* wrapped transform (owned iff own_plan) */
    int own_plan;
    size_t n;                  /* complex elements = plan->N * plan->K   */
    double *cre, *cim;         /* owned split working planes             */
} stride_il_t;

/** Wrap any plan for interleaved-buffer execution. */
static stride_il_t *stride_il_wrap(stride_plan_t *plan, int take_ownership) {
    if (!plan) return NULL;
    stride_il_t *w = (stride_il_t *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->plan = plan;
    w->own_plan = take_ownership;
    w->n = (size_t)plan->N * (plan->K ? plan->K : 1);
    w->cre = (double *)STRIDE_ALIGNED_ALLOC(64, w->n * sizeof(double));
    w->cim = (double *)STRIDE_ALIGNED_ALLOC(64, w->n * sizeof(double));
    if (!w->cre || !w->cim) {
        STRIDE_ALIGNED_FREE(w->cre); STRIDE_ALIGNED_FREE(w->cim);
        free(w);
        return NULL;
    }
    return w;
}

/** In-place on the interleaved buffer z (2n doubles). */
static void stride_il_fwd(stride_il_t *w, double *z) {
    vfft_il2sp(z, w->cre, w->cim, w->n);
    stride_execute_fwd(w->plan, w->cre, w->cim);
    vfft_sp2il(w->cre, w->cim, z, w->n);
}
static void stride_il_bwd(stride_il_t *w, double *z) {
    vfft_il2sp(z, w->cre, w->cim, w->n);
    stride_execute_bwd(w->plan, w->cre, w->cim);
    vfft_sp2il(w->cre, w->cim, z, w->n);
}

static void stride_il_destroy(stride_il_t *w) {
    if (!w) return;
    if (w->own_plan && w->plan) stride_plan_destroy(w->plan);
    STRIDE_ALIGNED_FREE(w->cre);
    STRIDE_ALIGNED_FREE(w->cim);
    free(w);
}


/* ── THE PADDED-ROW FORM (migration step 14) ─────────────────────────────
 * The converters above take a whole array. These take a PLANE: the same
 * interleaved<->split move applied per row, with the destination row pitch Kp
 * differing from the source count K. That is what the pad-vs-tail path needs -
 * a padded working buffer is Kp-strided while the caller's data is K-strided,
 * so the conversion and the re-stride happen in one pass rather than two.
 *
 * _vfft_z_dein / _vfft_z_inter are HAND-WRITTEN INTRINSICS (8 complex per
 * iteration on AVX-512, 4 on AVX2) rather than a loop left to the compiler.
 * The reason is compiler independence, not a measured win over the scalar form:
 * these are pure data movement and must stay bit-identical to the scalar path
 * across toolchains, so the shuffle sequence is written out rather than hoped
 * for. Exact - moves only, no arithmetic.
 * ──────────────────────────────────────────────────────────────────────── */

/* z<->split converts: pure data movement, bit-identical to the scalar epilogue by construction.
 * Hand intrinsics for compiler independence (8 cx/iter AVX-512, 4 AVX2); scalar tail, no masks.
 * See docs/design/vfft_front_door.md. */
static void _vfft_z_dein(const double *z, double *re, double *im, size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i ir = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ii = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (; i + 8 <= n; i += 8)
    {
        __m512d v0 = _mm512_loadu_pd(z + 2 * i);
        __m512d v1 = _mm512_loadu_pd(z + 2 * i + 8);
        _mm512_storeu_pd(re + i, _mm512_permutex2var_pd(v0, ir, v1));
        _mm512_storeu_pd(im + i, _mm512_permutex2var_pd(v0, ii, v1));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4)
    {
        __m256d v0 = _mm256_loadu_pd(z + 2 * i);
        __m256d v1 = _mm256_loadu_pd(z + 2 * i + 4);
        __m256d t0 = _mm256_permute2f128_pd(v0, v1, 0x20);
        __m256d t1 = _mm256_permute2f128_pd(v0, v1, 0x31);
        _mm256_storeu_pd(re + i, _mm256_unpacklo_pd(t0, t1));
        _mm256_storeu_pd(im + i, _mm256_unpackhi_pd(t0, t1));
    }
#endif
    for (; i < n; i++)
    {
        re[i] = z[2 * i];
        im[i] = z[2 * i + 1];
    }
}
static void _vfft_z_inter(const double *re, const double *im, double *z,
                          size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (; i + 8 <= n; i += 8)
    {
        __m512d r = _mm512_loadu_pd(re + i);
        __m512d m = _mm512_loadu_pd(im + i);
        _mm512_storeu_pd(z + 2 * i, _mm512_permutex2var_pd(r, lo, m));
        _mm512_storeu_pd(z + 2 * i + 8, _mm512_permutex2var_pd(r, hi, m));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4)
    {
        __m256d r = _mm256_loadu_pd(re + i);
        __m256d m = _mm256_loadu_pd(im + i);
        __m256d l2 = _mm256_unpacklo_pd(r, m);
        __m256d h2 = _mm256_unpackhi_pd(r, m);
        _mm256_storeu_pd(z + 2 * i, _mm256_permute2f128_pd(l2, h2, 0x20));
        _mm256_storeu_pd(z + 2 * i + 4, _mm256_permute2f128_pd(l2, h2, 0x31));
    }
#endif
    for (; i < n; i++)
    {
        z[2 * i] = re[i];
        z[2 * i + 1] = im[i];
    }
}

static void _il_pad_dein(const double *z, double *wr, double *wi,
                         int N, size_t K, size_t Kp)
{
    for (int p = 0; p < N; p++)
        _vfft_z_dein(z + 2 * (size_t)p * K,
                     wr + (size_t)p * Kp, wi + (size_t)p * Kp, K);
}
static void _il_pad_inter(const double *wr, const double *wi, double *z,
                          int N, size_t K, size_t Kp)
{
    for (int p = 0; p < N; p++)
        _vfft_z_inter(wr + (size_t)p * Kp, wi + (size_t)p * Kp,
                      z + 2 * (size_t)p * K, K);
}

#endif /* VFFT_IL_LAYOUT_H */
