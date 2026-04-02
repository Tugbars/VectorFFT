/**
 * bench_stride_executor.c — Stride-based in-place FFT executor prototype
 *
 * Proves the architecture: single buffer, multi-pass stride-based execution,
 * no transpose, no permutation, output in natural order.
 *
 * N=12 = 3×4:
 *   Stage 1: 4× DFT-3 at stride 4K (no twiddles)
 *   Stage 2: 3× DFT-4 at stride  K (with twiddles W_12^{j*g})
 *
 * Data layout: data[n*K + k]  for n=0..N-1, k=0..K-1
 * Each SIMD load reads K contiguous doubles regardless of stride.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include "bench_compat.h"

/* R=3 stride-based n1 codelet (is/os/vl parameters) */
#include "fft_radix3_avx2_ct_n1.h"

/* R=4 t1_dit codelet (ios/me parameters) + notw K-loop */
#include "fft_radix4_avx2.h"

/* R=4 stride-based n1 codelet — inline since gen_radix4.py doesn't generate one */
__attribute__((target("avx2,fma")))
static void radix4_n1_stride_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    size_t is, size_t vl)
{
    for (size_t k = 0; k < vl; k += 4) {
        __m256d r0 = _mm256_load_pd(&rio_re[k + 0*is]);
        __m256d i0 = _mm256_load_pd(&rio_im[k + 0*is]);
        __m256d r1 = _mm256_load_pd(&rio_re[k + 1*is]);
        __m256d i1 = _mm256_load_pd(&rio_im[k + 1*is]);
        __m256d r2 = _mm256_load_pd(&rio_re[k + 2*is]);
        __m256d i2 = _mm256_load_pd(&rio_im[k + 2*is]);
        __m256d r3 = _mm256_load_pd(&rio_re[k + 3*is]);
        __m256d i3 = _mm256_load_pd(&rio_im[k + 3*is]);
        __m256d sr = _mm256_add_pd(r0,r2), si = _mm256_add_pd(i0,i2);
        __m256d dr = _mm256_sub_pd(r0,r2), di = _mm256_sub_pd(i0,i2);
        __m256d tr = _mm256_add_pd(r1,r3), ti = _mm256_add_pd(i1,i3);
        __m256d ur = _mm256_sub_pd(r1,r3), ui = _mm256_sub_pd(i1,i3);
        _mm256_store_pd(&rio_re[k + 0*is], _mm256_add_pd(sr,tr));
        _mm256_store_pd(&rio_im[k + 0*is], _mm256_add_pd(si,ti));
        _mm256_store_pd(&rio_re[k + 2*is], _mm256_sub_pd(sr,tr));
        _mm256_store_pd(&rio_im[k + 2*is], _mm256_sub_pd(si,ti));
        _mm256_store_pd(&rio_re[k + 1*is], _mm256_add_pd(dr,ui));
        _mm256_store_pd(&rio_im[k + 1*is], _mm256_sub_pd(di,ur));
        _mm256_store_pd(&rio_re[k + 3*is], _mm256_sub_pd(dr,ui));
        _mm256_store_pd(&rio_im[k + 3*is], _mm256_add_pd(di,ur));
    }
}

__attribute__((target("avx2,fma")))
static void radix4_n1_stride_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    size_t is, size_t vl)
{
    for (size_t k = 0; k < vl; k += 4) {
        __m256d r0 = _mm256_load_pd(&rio_re[k + 0*is]);
        __m256d i0 = _mm256_load_pd(&rio_im[k + 0*is]);
        __m256d r1 = _mm256_load_pd(&rio_re[k + 1*is]);
        __m256d i1 = _mm256_load_pd(&rio_im[k + 1*is]);
        __m256d r2 = _mm256_load_pd(&rio_re[k + 2*is]);
        __m256d i2 = _mm256_load_pd(&rio_im[k + 2*is]);
        __m256d r3 = _mm256_load_pd(&rio_re[k + 3*is]);
        __m256d i3 = _mm256_load_pd(&rio_im[k + 3*is]);
        __m256d sr = _mm256_add_pd(r0,r2), si = _mm256_add_pd(i0,i2);
        __m256d dr = _mm256_sub_pd(r0,r2), di = _mm256_sub_pd(i0,i2);
        __m256d tr = _mm256_add_pd(r1,r3), ti = _mm256_add_pd(i1,i3);
        __m256d ur = _mm256_sub_pd(r1,r3), ui = _mm256_sub_pd(i1,i3);
        _mm256_store_pd(&rio_re[k + 0*is], _mm256_add_pd(sr,tr));
        _mm256_store_pd(&rio_im[k + 0*is], _mm256_add_pd(si,ti));
        _mm256_store_pd(&rio_re[k + 2*is], _mm256_sub_pd(sr,tr));
        _mm256_store_pd(&rio_im[k + 2*is], _mm256_sub_pd(si,ti));
        /* bwd: j -> -j */
        _mm256_store_pd(&rio_re[k + 1*is], _mm256_sub_pd(dr,ui));
        _mm256_store_pd(&rio_im[k + 1*is], _mm256_add_pd(di,ur));
        _mm256_store_pd(&rio_re[k + 3*is], _mm256_add_pd(dr,ui));
        _mm256_store_pd(&rio_im[k + 3*is], _mm256_sub_pd(di,ur));
    }
}

/* R=5 stride-based n1 codelet */
#include "fft_radix5_avx2_ct_n1.h"

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE PRECOMPUTATION
 *
 * For N=12=3×4, stage 2 (DFT-4 with twiddles):
 *   3 groups (g=0..2), 3 twiddle legs each (j=1..3)
 *   Group g, leg j: W_12^{j*g}
 *
 * The t1 codelet reads: W[(j-1)*me + m] for j=1..3, m=0..me-1
 * Since W_12^{j*g} is constant across m, we broadcast to K entries.
 *
 * Layout: tw[g * (R2-1) * K + (j-1) * K + k]
 * ═══════════════════════════════════════════════════════════════ */

static void init_stride_twiddles_12(double *tw_re, double *tw_im, size_t K) {
    const int R1 = 3, R2 = 4, N = 12;
    /* For each group g (n1=g) in stage 2: */
    for (int g = 0; g < R1; g++) {
        for (int j = 1; j < R2; j++) {
            double angle = -2.0 * M_PI * (double)(j * g) / (double)N;
            double wr = cos(angle), wi = sin(angle);
            for (size_t k = 0; k < K; k++) {
                tw_re[g * (R2 - 1) * K + (j - 1) * K + k] = wr;
                tw_im[g * (R2 - 1) * K + (j - 1) * K + k] = wi;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDE-BASED EXECUTOR: N=12 = 3×4
 *
 * Stage 1: 4 groups of DFT-3, stride = R2*K = 4K
 *   Group g (n2=g): base = g*K
 *   n1_R3(data+base, data+base, is=4K, os=4K, vl=K)
 *
 * Stage 2: 3 groups of DFT-4, stride = K
 *   Group g (k1=g): base = g*R2*K = g*4K
 *   t1_R4(data+base, tw+g*3K, ios=K, me=K)
 * ═══════════════════════════════════════════════════════════════ */

static void execute_fwd_12(double *re, double *im,
                           const double *tw_re, const double *tw_im,
                           size_t K) {
    const size_t R2K = 4 * K;

    /* Stage 1: 4× DFT-3 at stride 4K (in-place, no twiddles) */
    for (int g = 0; g < 4; g++) {
        radix3_n1_fwd_avx2(
            re + g * K, im + g * K,     /* input */
            re + g * K, im + g * K,     /* output (in-place) */
            R2K, R2K, K);               /* is=4K, os=4K, vl=K */
    }

    /* Stage 2: 3× DFT-4 at stride K (in-place with twiddles) */
    for (int g = 0; g < 3; g++) {
        radix4_t1_dit_fwd_avx2(
            re + g * R2K, im + g * R2K,                  /* rio */
            tw_re + g * 3 * K, tw_im + g * 3 * K,       /* twiddles for this group */
            K, K);                                        /* ios=K, me=K */
    }
}

/* ── Manual DIF backward t1 for R=4 ──
 * DIF = butterfly FIRST, then conjugate twiddles.
 * t1_dit_bwd applies conj(tw) BEFORE butterfly — wrong for DIF.
 * This is a small manual AVX2 implementation for the prototype. */
__attribute__((target("avx2,fma")))
static void radix4_t1_dif_bwd_avx2_manual(
    double *rio_re, double *rio_im,
    const double *W_re, const double *W_im,
    size_t ios, size_t me)
{
    for (size_t m = 0; m < me; m += 4) {
        /* Load 4 inputs at stride ios */
        __m256d r0 = _mm256_load_pd(&rio_re[m + 0*ios]);
        __m256d i0 = _mm256_load_pd(&rio_im[m + 0*ios]);
        __m256d r1 = _mm256_load_pd(&rio_re[m + 1*ios]);
        __m256d i1 = _mm256_load_pd(&rio_im[m + 1*ios]);
        __m256d r2 = _mm256_load_pd(&rio_re[m + 2*ios]);
        __m256d i2 = _mm256_load_pd(&rio_im[m + 2*ios]);
        __m256d r3 = _mm256_load_pd(&rio_re[m + 3*ios]);
        __m256d i3 = _mm256_load_pd(&rio_im[m + 3*ios]);

        /* DFT-4 backward butterfly (j -> -j vs forward) */
        __m256d sr = _mm256_add_pd(r0, r2), si = _mm256_add_pd(i0, i2);
        __m256d dr = _mm256_sub_pd(r0, r2), di = _mm256_sub_pd(i0, i2);
        __m256d tr = _mm256_add_pd(r1, r3), ti = _mm256_add_pd(i1, i3);
        __m256d ur = _mm256_sub_pd(r1, r3), ui = _mm256_sub_pd(i1, i3);

        r0 = _mm256_add_pd(sr, tr); i0 = _mm256_add_pd(si, ti);
        r2 = _mm256_sub_pd(sr, tr); i2 = _mm256_sub_pd(si, ti);
        /* bwd: y1 = d - j*u -> (dr-ui, di+ur);  y3 = d + j*u -> (dr+ui, di-ur) */
        r1 = _mm256_sub_pd(dr, ui); i1 = _mm256_add_pd(di, ur);
        r3 = _mm256_add_pd(dr, ui); i3 = _mm256_sub_pd(di, ur);

        /* Store y0 untwiddle */
        _mm256_store_pd(&rio_re[m + 0*ios], r0);
        _mm256_store_pd(&rio_im[m + 0*ios], i0);

        /* Apply conjugate twiddles AFTER butterfly (DIF convention) */
        /* conj(W) * y: yr*wr + yi*wi, yi*wr - yr*wi */
        for (int j = 1; j < 4; j++) {
            __m256d wr = _mm256_load_pd(&W_re[(j-1)*me + m]);
            __m256d wi = _mm256_load_pd(&W_im[(j-1)*me + m]);
            __m256d *yr, *yi;
            if (j == 1) { yr = &r1; yi = &i1; }
            else if (j == 2) { yr = &r2; yi = &i2; }
            else { yr = &r3; yi = &i3; }
            __m256d t = *yr;
            *yr = _mm256_fmadd_pd(*yr, wr, _mm256_mul_pd(*yi, wi));
            *yi = _mm256_fmsub_pd(*yi, wr, _mm256_mul_pd(t, wi));
            _mm256_store_pd(&rio_re[m + j*ios], *yr);
            _mm256_store_pd(&rio_im[m + j*ios], *yi);
        }
    }
}

/* DIF backward: reverse stage order from DIT forward.
 * DIT fwd: stage1=DFT-3 cols(no tw) -> stage2=DFT-4 rows(DIT: tw before bfly)
 * DIF bwd: stage1=DFT-4 rows(DIF: bfly then conj-tw) -> stage2=DFT-3 cols(no tw) */
static void execute_bwd_12(double *re, double *im,
                           const double *tw_re, const double *tw_im,
                           size_t K) {
    const size_t R2K = 4 * K;

    /* DIF bwd stage 1: 3x DFT-4 (butterfly then conj-twiddle) */
    for (int g = 0; g < 3; g++) {
        radix4_t1_dif_bwd_avx2_manual(
            re + g * R2K, im + g * R2K,
            tw_re + g * 3 * K, tw_im + g * 3 * K,
            K, K);
    }

    /* DIF bwd stage 2: 4x DFT-3 (no twiddles) */
    for (int g = 0; g < 4; g++) {
        radix3_n1_bwd_avx2(
            re + g * K, im + g * K,
            re + g * K, im + g * K,
            R2K, R2K, K);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * CORRECTNESS: compare against FFTW
 *
 * FFTW guru split DFT: K simultaneous N-point DFTs.
 * Data layout: data[n*K + k], transform dim stride=K, howmany stride=1.
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * STRIDE-BASED EXECUTOR: N=60 = 3×4×5
 *
 * Separate twiddle passes (no twiddle folded into codelet):
 *   Stage 1: 20× DFT-3 at stride 20K (n1, no twiddle)
 *   Twiddle 1: W_60^{k1*(5*n2+n3)} at each position
 *   Stage 2: 15× DFT-4 at stride 5K (n1, no twiddle)
 *   Twiddle 2: W_20^{k2*n3} at each position
 *   Stage 3: 12× DFT-5 at stride K (n1, no twiddle)
 * ═══════════════════════════════════════════════════════════════ */

/* Twiddle pass 1: after stage 1, apply W_60^{k1*(5*n2+n3)} */
__attribute__((target("avx2,fma")))
static void twiddle_pass1_60(double *re, double *im, size_t K) {
    /* k1=0: W^0=1, skip. k1=1,2: apply twiddles. */
    for (int k1 = 1; k1 < 3; k1++) {
        for (int n2 = 0; n2 < 4; n2++) {
            for (int n3 = 0; n3 < 5; n3++) {
                int exp = k1 * (5*n2 + n3);
                if (exp == 0) continue;
                double angle = -2.0 * M_PI * (double)exp / 60.0;
                double wr = cos(angle), wi = sin(angle);
                size_t base = (size_t)(k1*20 + n2*5 + n3) * K;
                for (size_t k = 0; k + 3 < K; k += 4) {
                    __m256d xr = _mm256_load_pd(&re[base+k]);
                    __m256d xi = _mm256_load_pd(&im[base+k]);
                    __m256d cwr = _mm256_set1_pd(wr), cwi = _mm256_set1_pd(wi);
                    __m256d tr = xr;
                    xr = _mm256_fmsub_pd(tr, cwr, _mm256_mul_pd(xi, cwi));
                    xi = _mm256_fmadd_pd(tr, cwi, _mm256_mul_pd(xi, cwr));
                    _mm256_store_pd(&re[base+k], xr);
                    _mm256_store_pd(&im[base+k], xi);
                }
            }
        }
    }
}

/* Twiddle pass 2: after stage 2, apply W_20^{k2*n3} */
__attribute__((target("avx2,fma")))
static void twiddle_pass2_60(double *re, double *im, size_t K) {
    for (int k1 = 0; k1 < 3; k1++) {
        for (int k2 = 1; k2 < 4; k2++) {  /* k2=0: W^0=1, skip */
            for (int n3 = 1; n3 < 5; n3++) {  /* n3=0: W^0=1, skip */
                int exp = k2 * n3;
                double angle = -2.0 * M_PI * (double)exp / 20.0;
                double wr = cos(angle), wi = sin(angle);
                size_t base = (size_t)(k1*20 + k2*5 + n3) * K;
                for (size_t k = 0; k + 3 < K; k += 4) {
                    __m256d xr = _mm256_load_pd(&re[base+k]);
                    __m256d xi = _mm256_load_pd(&im[base+k]);
                    __m256d cwr = _mm256_set1_pd(wr), cwi = _mm256_set1_pd(wi);
                    __m256d tr = xr;
                    xr = _mm256_fmsub_pd(tr, cwr, _mm256_mul_pd(xi, cwi));
                    xi = _mm256_fmadd_pd(tr, cwi, _mm256_mul_pd(xi, cwr));
                    _mm256_store_pd(&re[base+k], xr);
                    _mm256_store_pd(&im[base+k], xi);
                }
            }
        }
    }
}

/* Backward twiddle passes: conjugate (negate wi) */
__attribute__((target("avx2,fma")))
static void twiddle_pass2_60_bwd(double *re, double *im, size_t K) {
    for (int k1 = 0; k1 < 3; k1++) {
        for (int k2 = 1; k2 < 4; k2++) {
            for (int n3 = 1; n3 < 5; n3++) {
                int exp = k2 * n3;
                double angle = +2.0 * M_PI * (double)exp / 20.0;
                double wr = cos(angle), wi = sin(angle);
                size_t base = (size_t)(k1*20 + k2*5 + n3) * K;
                for (size_t k = 0; k + 3 < K; k += 4) {
                    __m256d xr = _mm256_load_pd(&re[base+k]);
                    __m256d xi = _mm256_load_pd(&im[base+k]);
                    __m256d cwr = _mm256_set1_pd(wr), cwi = _mm256_set1_pd(wi);
                    __m256d tr = xr;
                    xr = _mm256_fmsub_pd(tr, cwr, _mm256_mul_pd(xi, cwi));
                    xi = _mm256_fmadd_pd(tr, cwi, _mm256_mul_pd(xi, cwr));
                    _mm256_store_pd(&re[base+k], xr);
                    _mm256_store_pd(&im[base+k], xi);
                }
            }
        }
    }
}

__attribute__((target("avx2,fma")))
static void twiddle_pass1_60_bwd(double *re, double *im, size_t K) {
    for (int k1 = 1; k1 < 3; k1++) {
        for (int n2 = 0; n2 < 4; n2++) {
            for (int n3 = 0; n3 < 5; n3++) {
                int exp = k1 * (5*n2 + n3);
                if (exp == 0) continue;
                double angle = +2.0 * M_PI * (double)exp / 60.0;
                double wr = cos(angle), wi = sin(angle);
                size_t base = (size_t)(k1*20 + n2*5 + n3) * K;
                for (size_t k = 0; k + 3 < K; k += 4) {
                    __m256d xr = _mm256_load_pd(&re[base+k]);
                    __m256d xi = _mm256_load_pd(&im[base+k]);
                    __m256d cwr = _mm256_set1_pd(wr), cwi = _mm256_set1_pd(wi);
                    __m256d tr = xr;
                    xr = _mm256_fmsub_pd(tr, cwr, _mm256_mul_pd(xi, cwi));
                    xi = _mm256_fmadd_pd(tr, cwi, _mm256_mul_pd(xi, cwr));
                    _mm256_store_pd(&re[base+k], xr);
                    _mm256_store_pd(&im[base+k], xi);
                }
            }
        }
    }
}

static void execute_fwd_60(double *re, double *im, size_t K) {
    /* Stage 1: 20× DFT-3 at stride 20K */
    for (int g = 0; g < 20; g++) {
        radix3_n1_fwd_avx2(
            re + g*K, im + g*K,
            re + g*K, im + g*K,
            20*K, 20*K, K);
    }

    /* Twiddle 1: W_60^{k1*(5*n2+n3)} */
    twiddle_pass1_60(re, im, K);

    /* Stage 2: 15× DFT-4 at stride 5K */
    for (int k1 = 0; k1 < 3; k1++) {
        for (int n3 = 0; n3 < 5; n3++) {
            size_t base = (size_t)(k1*20 + n3) * K;
            radix4_n1_stride_fwd_avx2(
                re + base, im + base,
                5*K, K);
        }
    }

    /* Twiddle 2: W_20^{k2*n3} */
    twiddle_pass2_60(re, im, K);

    /* Stage 3: 12× DFT-5 at stride K */
    for (int k1 = 0; k1 < 3; k1++) {
        for (int k2 = 0; k2 < 4; k2++) {
            size_t base = (size_t)(k1*4 + k2) * 5 * K;
            radix5_n1_fwd_avx2(
                re + base, im + base,
                re + base, im + base,
                K, K, K);
        }
    }
}

static void execute_bwd_60(double *re, double *im, size_t K) {
    /* DIF backward: reverse order */
    /* Stage 3 bwd: 12× DFT-5 */
    for (int k1 = 0; k1 < 3; k1++) {
        for (int k2 = 0; k2 < 4; k2++) {
            size_t base = (size_t)(k1*4 + k2) * 5 * K;
            radix5_n1_bwd_avx2(
                re + base, im + base,
                re + base, im + base,
                K, K, K);
        }
    }

    /* Un-twiddle 2: conj(W_20^{k2*n3}) */
    twiddle_pass2_60_bwd(re, im, K);

    /* Stage 2 bwd: 15× DFT-4 */
    for (int k1 = 0; k1 < 3; k1++) {
        for (int n3 = 0; n3 < 5; n3++) {
            size_t base = (size_t)(k1*20 + n3) * K;
            radix4_n1_stride_bwd_avx2(
                re + base, im + base,
                5*K, K);
        }
    }

    /* Un-twiddle 1: conj(W_60^{k1*(5*n2+n3)}) */
    twiddle_pass1_60_bwd(re, im, K);

    /* Stage 1 bwd: 20× DFT-3 */
    for (int g = 0; g < 20; g++) {
        radix3_n1_bwd_avx2(
            re + g*K, im + g*K,
            re + g*K, im + g*K,
            20*K, 20*K, K);
    }
}


/* Digit-reversal permutation for N = R1*R2.
 * In-place CT produces X[k1 + R1*k2] at position R2*k1 + k2.
 * This permutation maps natural index m to storage position p.
 * perm[m] = position where X[m] is stored after executor. */
static void build_digit_rev_perm(int *perm, int R1, int R2) {
    int N = R1 * R2;
    for (int k1 = 0; k1 < R1; k1++)
        for (int k2 = 0; k2 < R2; k2++) {
            int pos = R2 * k1 + k2;
            int dft_idx = k1 + R1 * k2;
            perm[dft_idx] = pos;   /* X[dft_idx] is at position pos */
        }
}

/* Apply digit-reversal permutation: out[m] = in[perm[m]] for each K-batch */
static void apply_perm(const int *perm, int N, size_t K,
                       const double *in_re, const double *in_im,
                       double *out_re, double *out_im) {
    for (int m = 0; m < N; m++) {
        int p = perm[m];
        memcpy(out_re + (size_t)m * K, in_re + (size_t)p * K, K * 8);
        memcpy(out_im + (size_t)m * K, in_im + (size_t)p * K, K * 8);
    }
}

static int verify_fwd_12(size_t K, int verbose) {
    const int R1 = 3, R2 = 4, N = 12;
    const size_t total = (size_t)N * K;

    double *data_re = aligned_alloc(64, total * 8);
    double *data_im = aligned_alloc(64, total * 8);
    double *ref_re  = fftw_malloc(total * 8);
    double *ref_im  = fftw_malloc(total * 8);
    double *orig_re = aligned_alloc(64, total * 8);
    double *orig_im = aligned_alloc(64, total * 8);
    double *sorted_re = aligned_alloc(64, total * 8);
    double *sorted_im = aligned_alloc(64, total * 8);

    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* FFTW reference */
    double *fftw_tmp_re = fftw_malloc(total * 8);
    double *fftw_tmp_im = fftw_malloc(total * 8);
    memcpy(fftw_tmp_re, orig_re, total * 8);
    memcpy(fftw_tmp_im, orig_im, total * 8);
    fftw_iodim dim  = { .n = N, .is = (int)K, .os = (int)K };
    fftw_iodim howm = { .n = (int)K, .is = 1, .os = 1 };
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm,
                                           fftw_tmp_re, fftw_tmp_im,
                                           ref_re, ref_im, FFTW_ESTIMATE);
    fftw_execute_split_dft(p, orig_re, orig_im, ref_re, ref_im);
    fftw_destroy_plan(p);
    fftw_free(fftw_tmp_re); fftw_free(fftw_tmp_im);

    /* Our executor */
    memcpy(data_re, orig_re, total * 8);
    memcpy(data_im, orig_im, total * 8);
    double *tw_re = aligned_alloc(64, R1 * (R2 - 1) * K * 8);
    double *tw_im = aligned_alloc(64, R1 * (R2 - 1) * K * 8);
    init_stride_twiddles_12(tw_re, tw_im, K);
    execute_fwd_12(data_re, data_im, tw_re, tw_im, K);

    /* Test 1: direct comparison (no permutation) */
    double max_err_direct = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(data_re[i] - ref_re[i]);
        double ei = fabs(data_im[i] - ref_im[i]);
        if (er > max_err_direct) max_err_direct = er;
        if (ei > max_err_direct) max_err_direct = ei;
    }

    /* Test 2: comparison WITH digit-reversal permutation */
    int perm[12];
    build_digit_rev_perm(perm, R1, R2);
    apply_perm(perm, N, K, data_re, data_im, sorted_re, sorted_im);

    double max_err_perm = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(sorted_re[i] - ref_re[i]);
        double ei = fabs(sorted_im[i] - ref_im[i]);
        if (er > max_err_perm) max_err_perm = er;
        if (ei > max_err_perm) max_err_perm = ei;
    }

    if (verbose) {
        printf("  K=%-4zu  direct=%.2e  permuted=%.2e  %s\n", K,
               max_err_direct, max_err_perm,
               max_err_perm < 1e-10 ? "OK" : "FAIL");
    }

    aligned_free(data_re); aligned_free(data_im);
    aligned_free(orig_re); aligned_free(orig_im);
    aligned_free(sorted_re); aligned_free(sorted_im);
    aligned_free(tw_re); aligned_free(tw_im);
    fftw_free(ref_re); fftw_free(ref_im);

    return max_err_perm < 1e-10 ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════
 * BENCHMARK: our executor vs FFTW_MEASURE
 * ═══════════════════════════════════════════════════════════════ */

static double bench_ours_12(size_t K, int reps) {
    const int N = 12;
    const size_t total = (size_t)N * K;

    double *re = aligned_alloc(64, total * 8);
    double *im = aligned_alloc(64, total * 8);
    double *tw_re = aligned_alloc(64, 3 * 3 * K * 8);
    double *tw_im = aligned_alloc(64, 3 * 3 * K * 8);

    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    init_stride_twiddles_12(tw_re, tw_im, K);

    /* Warmup */
    for (int i = 0; i < 20; i++)
        execute_fwd_12(re, im, tw_re, tw_im, K);

    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            execute_fwd_12(re, im, tw_re, tw_im, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    aligned_free(re); aligned_free(im);
    aligned_free(tw_re); aligned_free(tw_im);
    return best;
}

static double bench_fftw_12(size_t K, int reps) {
    const int N = 12;
    const size_t total = (size_t)N * K;

    double *ri = fftw_malloc(total * 8), *ii = fftw_malloc(total * 8);
    double *ro = fftw_malloc(total * 8), *io = fftw_malloc(total * 8);

    for (size_t i = 0; i < total; i++) {
        ri[i] = (double)rand() / RAND_MAX - 0.5;
        ii[i] = (double)rand() / RAND_MAX - 0.5;
    }

    fftw_iodim dim  = { .n = N, .is = (int)K, .os = (int)K };
    fftw_iodim howm = { .n = (int)K, .is = 1, .os = 1 };
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm,
                                           ri, ii, ro, io, FFTW_MEASURE);
    /* Re-randomize after MEASURE planning */
    for (size_t i = 0; i < total; i++) {
        ri[i] = (double)rand() / RAND_MAX - 0.5;
        ii[i] = (double)rand() / RAND_MAX - 0.5;
    }

    for (int i = 0; i < 20; i++) fftw_execute(p);

    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            fftw_execute_split_dft(p, ri, ii, ro, io);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    fftw_destroy_plan(p);
    fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io);
    return best;
}


int main(void) {
    srand(42);

    printf("VectorFFT Stride-Based Executor Prototype\n");
    printf("N=12 = 3x4, stride-based in-place, no transpose, no permutation\n\n");

    /* ── Correctness: single direction (with known digit-reversal) ── */
    printf("Correctness vs FFTW (digit-reversal permuted):\n");
    int fail = 0;
    size_t test_Ks[] = { 4, 8, 16, 32, 64, 128, 256 };
    int n_test = (int)(sizeof(test_Ks) / sizeof(test_Ks[0]));
    for (int i = 0; i < n_test; i++) {
        size_t K = test_Ks[i];
        int err = verify_fwd_12(K, 1);
        fail |= err;
    }
    if (fail) {
        printf("\n*** CORRECTNESS FAILURE ***\n");
        return 1;
    }
    printf("  All correct.\n\n");

    /* ── Correctness: DIT fwd + DIF bwd roundtrip = identity ── */
    printf("Roundtrip (DIT fwd + DIF bwd) = identity:\n");
    for (int i = 0; i < n_test; i++) {
        size_t K = test_Ks[i];
        const int N = 12;
        const size_t total = (size_t)N * K;
        double *re = aligned_alloc(64, total * 8);
        double *im = aligned_alloc(64, total * 8);
        double *orig_re = aligned_alloc(64, total * 8);
        double *orig_im = aligned_alloc(64, total * 8);
        double *tw_re = aligned_alloc(64, 3 * 3 * K * 8);
        double *tw_im = aligned_alloc(64, 3 * 3 * K * 8);

        for (size_t j = 0; j < total; j++) {
            orig_re[j] = (double)rand() / RAND_MAX - 0.5;
            orig_im[j] = (double)rand() / RAND_MAX - 0.5;
        }
        memcpy(re, orig_re, total * 8);
        memcpy(im, orig_im, total * 8);
        init_stride_twiddles_12(tw_re, tw_im, K);

        execute_fwd_12(re, im, tw_re, tw_im, K);
        execute_bwd_12(re, im, tw_re, tw_im, K);

        /* Scale by 1/N */
        double scale = 1.0 / N;
        for (size_t j = 0; j < total; j++) {
            re[j] *= scale;
            im[j] *= scale;
        }

        double max_err = 0;
        for (size_t j = 0; j < total; j++) {
            double er = fabs(re[j] - orig_re[j]);
            double ei = fabs(im[j] - orig_im[j]);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }

        printf("  K=%-4zu  err=%.2e  %s\n", K, max_err,
               max_err < 1e-10 ? "OK" : "FAIL");
        if (max_err >= 1e-10) fail = 1;

        aligned_free(re); aligned_free(im);
        aligned_free(orig_re); aligned_free(orig_im);
        aligned_free(tw_re); aligned_free(tw_im);
    }
    if (fail) {
        printf("\n*** ROUNDTRIP FAILURE ***\n");
        return 1;
    }
    printf("  All correct.\n\n");

    /* ── Performance ── */
    printf("Performance: stride executor vs FFTW_MEASURE (N=12, batched K transforms)\n\n");
    printf("%-5s %-7s %10s %10s %8s\n",
           "K", "N*K", "FFTW_M", "stride", "ratio");
    printf("%-5s %-7s %10s %10s %8s\n",
           "-----", "-------", "----------", "----------", "--------");

    size_t bench_Ks[] = { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
    int n_bench = (int)(sizeof(bench_Ks) / sizeof(bench_Ks[0]));

    for (int i = 0; i < n_bench; i++) {
        size_t K = bench_Ks[i];
        size_t total = 12 * K;

        int reps = (int)(2e6 / (total + 1));
        if (reps < 200) reps = 200;
        if (reps > 2000000) reps = 2000000;

        double fftw_ns = bench_fftw_12(K, reps);
        double ours_ns = bench_ours_12(K, reps);

        printf("%-5zu %-7zu %10.1f %10.1f %7.2fx\n",
               K, total, fftw_ns, ours_ns,
               fftw_ns > 0 ? fftw_ns / ours_ns : 0);
    }

    printf("\nDone.\n");
    return fail;
}
