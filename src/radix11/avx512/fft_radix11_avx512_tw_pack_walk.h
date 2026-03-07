/**
 * @file fft_radix11_avx512_tw_pack_walk.h
 * @brief DFT-11 AVX-512 packed data + walked twiddles (zero tw table)
 *
 * Walks 2 base accumulators (W^1, W^2), derives 8 more via tree:
 *   W^3  = W^1 * W^2
 *   W^4  = (W^2)^2
 *   W^5  = W^1 * W^4
 *   W^6  = W^2 * W^4
 *   W^7  = W^3 * W^4
 *   W^8  = (W^4)^2
 *   W^9  = W^1 * W^8
 *   W^10 = W^2 * W^8
 *
 * Per block: 6 cmuls + 2 csquares (derivation) + 10 cmuls (apply)
 *            + 2 cmuls (walk step) = 20 cmul-equivalents
 *
 * Eliminates the 10*K twiddle table entirely.
 *
 * Requires: fft_radix11_genfft.h included first (for notw kernel)
 */

#ifndef FFT_RADIX11_AVX512_TW_PACK_WALK_H
#define FFT_RADIX11_AVX512_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WALK PLAN (AVX-512, T=8)
 *
 * step[i] = W_{11K}^{pow[i]*8}    (advance 8 lanes per block)
 * init[i][j] = W_{11K}^{pow[i]*j}  j=0..7
 * pow = {1, 2}
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double step_re[2];
    double step_im[2];
    double init_re[2][8];
    double init_im[2][8];
} radix11_walk_plan_t;

static inline void radix11_walk_plan_init(
    radix11_walk_plan_t *plan, size_t K)
{
    const size_t NN = 11 * K;
    const int pows[2] = {1, 2};
    for (int i = 0; i < 2; i++) {
        double a_step = -2.0 * M_PI * (double)(pows[i] * 8) / (double)NN;
        plan->step_re[i] = cos(a_step);
        plan->step_im[i] = sin(a_step);
        for (int j = 0; j < 8; j++) {
            double a = -2.0 * M_PI * (double)(pows[i] * j) / (double)NN;
            plan->init_re[i][j] = cos(a);
            plan->init_im[i][j] = sin(a);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * COMPLEX MULTIPLY HELPERS
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R11W_CMUL_DEFINED
#define R11W_CMUL_DEFINED
#define R11W_CMUL_RE(ar,ai,br,bi) \
    _mm512_fnmadd_pd(ai, bi, _mm512_mul_pd(ar, br))
#define R11W_CMUL_IM(ar,ai,br,bi) \
    _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br))
#define R11W_CSQ_RE(ar,ai) \
    _mm512_fmsub_pd(ar, ar, _mm512_mul_pd(ai, ai))
#define R11W_CSQ_IM(ar,ai) \
    _mm512_mul_pd(_mm512_add_pd(ar, ar), ai)
#endif

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK
 *
 * Data must be in packed layout: in[block*11*8 + n*8 + j]
 * Derives all 10 twiddle vectors from 2 walked bases,
 * applies them to a scratch buffer, then calls the notw
 * DFT-11 kernel on the twiddled block.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix11_tw_pack_walk_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix11_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t T = 8, bs = 11 * T, nb = K / T;

    /* Load step broadcasts */
    __m512d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    /* Initialize base accumulators */
    __m512d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    /* Scratch for twiddled input (11*8 = 88 doubles × 2 = 1408 bytes) */
    double __attribute__((aligned(64))) tw_blk_re[11 * 8];
    double __attribute__((aligned(64))) tw_blk_im[11 * 8];

    for (size_t blk = 0; blk < nb; blk++) {
        const double *blk_ir = in_re + blk * bs;
        const double *blk_ii = in_im + blk * bs;

        /* ── Derive 10 twiddle vectors from 2 bases ── */
        /* tw[0]=W^1, tw[1]=W^2 (direct from bases) */
        __m512d tw_r[10], tw_i[10];
        tw_r[0] = b_re[0]; tw_i[0] = b_im[0];  /* W^1 */
        tw_r[1] = b_re[1]; tw_i[1] = b_im[1];  /* W^2 */

        /* W^3 = W^1 * W^2 */
        tw_r[2] = R11W_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
        tw_i[2] = R11W_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);

        /* W^4 = (W^2)^2 */
        __m512d w4r = R11W_CSQ_RE(b_re[1],b_im[1]);
        __m512d w4i = R11W_CSQ_IM(b_re[1],b_im[1]);
        tw_r[3] = w4r; tw_i[3] = w4i;

        /* W^5 = W^1 * W^4 */
        tw_r[4] = R11W_CMUL_RE(b_re[0],b_im[0],w4r,w4i);
        tw_i[4] = R11W_CMUL_IM(b_re[0],b_im[0],w4r,w4i);

        /* W^6 = W^2 * W^4 */
        tw_r[5] = R11W_CMUL_RE(b_re[1],b_im[1],w4r,w4i);
        tw_i[5] = R11W_CMUL_IM(b_re[1],b_im[1],w4r,w4i);

        /* W^7 = W^3 * W^4 */
        tw_r[6] = R11W_CMUL_RE(tw_r[2],tw_i[2],w4r,w4i);
        tw_i[6] = R11W_CMUL_IM(tw_r[2],tw_i[2],w4r,w4i);

        /* W^8 = (W^4)^2 */
        __m512d w8r = R11W_CSQ_RE(w4r,w4i);
        __m512d w8i = R11W_CSQ_IM(w4r,w4i);
        tw_r[7] = w8r; tw_i[7] = w8i;

        /* W^9 = W^1 * W^8 */
        tw_r[8] = R11W_CMUL_RE(b_re[0],b_im[0],w8r,w8i);
        tw_i[8] = R11W_CMUL_IM(b_re[0],b_im[0],w8r,w8i);

        /* W^10 = W^2 * W^8 */
        tw_r[9] = R11W_CMUL_RE(b_re[1],b_im[1],w8r,w8i);
        tw_i[9] = R11W_CMUL_IM(b_re[1],b_im[1],w8r,w8i);

        /* ── Apply twiddles: x'[0] = x[0] (no twiddle), x'[n] = x[n] * tw[n-1] ── */
        _mm512_store_pd(&tw_blk_re[0], _mm512_load_pd(&blk_ir[0]));
        _mm512_store_pd(&tw_blk_im[0], _mm512_load_pd(&blk_ii[0]));

        for (int n = 0; n < 10; n++) {
            __m512d xr = _mm512_load_pd(&blk_ir[(n+1) * T]);
            __m512d xi = _mm512_load_pd(&blk_ii[(n+1) * T]);
            _mm512_store_pd(&tw_blk_re[(n+1) * T],
                R11W_CMUL_RE(xr, xi, tw_r[n], tw_i[n]));
            _mm512_store_pd(&tw_blk_im[(n+1) * T],
                R11W_CMUL_IM(xr, xi, tw_r[n], tw_i[n]));
        }

        /* ── DFT-11 butterfly on twiddled block ── */
        radix11_genfft_fwd_avx512(tw_blk_re, tw_blk_im,
                                  out_re + blk * bs, out_im + blk * bs, T);

        /* ── Walk: advance 2 bases by step ── */
        for (int i = 0; i < 2; i++) {
            __m512d nr = R11W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R11W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK
 *
 * Backward DFT-11 = forward with re↔im swap.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix11_tw_pack_walk_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix11_walk_plan_t * __restrict__ plan,
    size_t K)
{
    /* Backward = forward with swapped re/im on both input and output.
     * But twiddles must also be conjugated for backward direction.
     * We create a conjugated plan and call forward. */

    /* Actually, for backward: the twiddle is conj(W^{nk}) = W^{-nk}.
     * In our convention, forward twiddles are W^{-nk} (negative exponent),
     * so backward twiddles are W^{+nk}. The walk plan was initialized with
     * negative exponent. For backward, we negate all imaginary parts.
     *
     * Simplest correct approach: swap re/im on input, call forward with
     * conjugated twiddles, swap re/im on output. But the twiddle conjugation
     * is the tricky part.
     *
     * Cleanest approach: build the twiddled block with conjugate twiddles,
     * then call backward butterfly. */

    const size_t T = 8, bs = 11 * T, nb = K / T;

    __m512d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(64))) tw_blk_re[11 * 8];
    double __attribute__((aligned(64))) tw_blk_im[11 * 8];

    for (size_t blk = 0; blk < nb; blk++) {
        const double *blk_ir = in_re + blk * bs;
        const double *blk_ii = in_im + blk * bs;

        /* Derive twiddles (same tree, conjugated: use conj(W) = (re, -im) ) */
        __m512d tw_r[10], tw_i[10];
        /* Conjugate bases for backward */
        __m512d cb_re0 = b_re[0], cb_im0 = _mm512_xor_pd(b_im[0], _mm512_set1_pd(-0.0));
        __m512d cb_re1 = b_re[1], cb_im1 = _mm512_xor_pd(b_im[1], _mm512_set1_pd(-0.0));

        tw_r[0] = cb_re0; tw_i[0] = cb_im0;
        tw_r[1] = cb_re1; tw_i[1] = cb_im1;

        tw_r[2] = R11W_CMUL_RE(cb_re0,cb_im0,cb_re1,cb_im1);
        tw_i[2] = R11W_CMUL_IM(cb_re0,cb_im0,cb_re1,cb_im1);

        __m512d w4r = R11W_CSQ_RE(cb_re1,cb_im1);
        __m512d w4i = R11W_CSQ_IM(cb_re1,cb_im1);
        tw_r[3] = w4r; tw_i[3] = w4i;

        tw_r[4] = R11W_CMUL_RE(cb_re0,cb_im0,w4r,w4i);
        tw_i[4] = R11W_CMUL_IM(cb_re0,cb_im0,w4r,w4i);

        tw_r[5] = R11W_CMUL_RE(cb_re1,cb_im1,w4r,w4i);
        tw_i[5] = R11W_CMUL_IM(cb_re1,cb_im1,w4r,w4i);

        tw_r[6] = R11W_CMUL_RE(tw_r[2],tw_i[2],w4r,w4i);
        tw_i[6] = R11W_CMUL_IM(tw_r[2],tw_i[2],w4r,w4i);

        __m512d w8r = R11W_CSQ_RE(w4r,w4i);
        __m512d w8i = R11W_CSQ_IM(w4r,w4i);
        tw_r[7] = w8r; tw_i[7] = w8i;

        tw_r[8] = R11W_CMUL_RE(cb_re0,cb_im0,w8r,w8i);
        tw_i[8] = R11W_CMUL_IM(cb_re0,cb_im0,w8r,w8i);

        tw_r[9] = R11W_CMUL_RE(cb_re1,cb_im1,w8r,w8i);
        tw_i[9] = R11W_CMUL_IM(cb_re1,cb_im1,w8r,w8i);

        /* Apply conjugate twiddles */
        _mm512_store_pd(&tw_blk_re[0], _mm512_load_pd(&blk_ir[0]));
        _mm512_store_pd(&tw_blk_im[0], _mm512_load_pd(&blk_ii[0]));

        for (int n = 0; n < 10; n++) {
            __m512d xr = _mm512_load_pd(&blk_ir[(n+1) * T]);
            __m512d xi = _mm512_load_pd(&blk_ii[(n+1) * T]);
            _mm512_store_pd(&tw_blk_re[(n+1) * T],
                R11W_CMUL_RE(xr, xi, tw_r[n], tw_i[n]));
            _mm512_store_pd(&tw_blk_im[(n+1) * T],
                R11W_CMUL_IM(xr, xi, tw_r[n], tw_i[n]));
        }

        /* Backward butterfly */
        radix11_genfft_bwd_avx512(tw_blk_re, tw_blk_im,
                                  out_re + blk * bs, out_im + blk * bs, T);

        /* Walk (unconjugated — we conjugate per-block above) */
        for (int i = 0; i < 2; i++) {
            __m512d nr = R11W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R11W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R11W_CMUL_RE
#undef R11W_CMUL_IM
#undef R11W_CSQ_RE
#undef R11W_CSQ_IM

#endif /* FFT_RADIX11_AVX512_TW_PACK_WALK_H */