/**
 * bench_r3_three_way.c — Old strided vs New strided vs New packed (kern only)
 *
 * Three contestants:
 *   OLD:  Hand-written DIF kernel (prefetch, loadu, pointer offsets)
 *   NEW:  Generated straight-line kernel (clean, no prefetch)
 *   PACK: Generated kernel on packed layout (K=T=8 per block, zero strides)
 *
 * Timing: min-of-5 runs, ns per call (fwd only)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Old kernel (uses struct twiddles) ── */
#include "fft_radix3_avx512_old.h"

/* ── New kernel ── */
#define R3_512_LD(p)    _mm512_loadu_pd(p)
#define R3_512_ST(p,v)  _mm512_storeu_pd((p),(v))
#include "fft_radix3_avx512.h"

/* ═══════════════════════════════════════════════════════════════ */

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double *alloc64(size_t n) {
    double *p = NULL;
    posix_memalign((void**)&p, 64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

/* ── SIMD pack/unpack ── */

__attribute__((target("avx512f")))
static void r3_pack_avx512(
    const double *sr, const double *si,
    double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K/8; b++) {
        size_t sk = b*8, dk = b*24;
        for (int n = 0; n < 3; n++) {
            _mm512_storeu_pd(&dr[dk+n*8], _mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8], _mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}

__attribute__((target("avx512f")))
static void r3_pack_tw_avx512(
    const double *sr, const double *si,
    double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K/8; b++) {
        size_t sk = b*8, dk = b*16;
        for (int n = 0; n < 2; n++) {
            _mm512_storeu_pd(&dr[dk+n*8], _mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8], _mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}

__attribute__((target("avx512f")))
static void r3_unpack_avx512(
    const double *sr, const double *si,
    double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K/8; b++) {
        size_t sk = b*24, dk = b*8;
        for (int n = 0; n < 3; n++) {
            _mm512_storeu_pd(&dr[n*K+dk], _mm512_loadu_pd(&sr[sk+n*8]));
            _mm512_storeu_pd(&di[n*K+dk], _mm512_loadu_pd(&si[sk+n*8]));
        }
    }
}

/* ── Packed kernel driver (loop over blocks, K=T=8) ── */

__attribute__((target("avx512f,fma")))
static void r3_packed_kern_fwd(
    const double *ir, const double *ii,
    double *or_, double *oi,
    const double *twr, const double *twi,
    size_t K)
{
    const size_t nb = K / 8;
    for (size_t b = 0; b < nb; b++)
        radix3_tw_dit_kernel_fwd_avx512(
            ir + b*24, ii + b*24,
            or_ + b*24, oi + b*24,
            twr + b*16, twi + b*16, 8);
}

__attribute__((target("avx512f,fma")))
static void r3_packed_kern_bwd(
    const double *ir, const double *ii,
    double *or_, double *oi,
    const double *twr, const double *twi,
    size_t K)
{
    const size_t nb = K / 8;
    for (size_t b = 0; b < nb; b++)
        radix3_tw_dit_kernel_bwd_avx512(
            ir + b*24, ii + b*24,
            or_ + b*24, oi + b*24,
            twr + b*16, twi + b*16, 8);
}

/* ── Twiddle generation ── */
static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 3 * K;
    for (int n = 1; n < 3; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n*k) / (double)NN;
            re[(n-1)*K+k] = cos(a);
            im[(n-1)*K+k] = sin(a);
        }
}

/* ── Reference DFT-3 ── */
static void ref_tw_dft3(const double *ir, const double *ii,
                        double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 3*K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[3], xi[3];
        for (int n = 0; n < 3; n++) {
            double dr = ir[n*K+k], di = ii[n*K+k];
            if (n > 0) {
                double a = sign*2.0*M_PI*(double)(n*k)/(double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr*wr - di*wi;
                di = dr*wi + di*wr; dr = tr;
            }
            xr[n] = dr; xi[n] = di;
        }
        for (int m = 0; m < 3; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 3; n++) {
                double a = sign*2.0*M_PI*(double)(m*n)/3.0;
                double wr = cos(a), wi = sin(a);
                sr += xr[n]*wr - xi[n]*wi;
                si += xr[n]*wi + xi[n]*wr;
            }
            or_[m*K+k] = sr; oi[m*K+k] = si;
        }
    }
}

static double maxerr(const double *ar, const double *ai,
                     const double *br, const double *bi, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double dr = fabs(ar[i]-br[i]), di = fabs(ai[i]-bi[i]);
        if (dr > mx) mx = dr; if (di > mx) mx = di;
    }
    return mx;
}

/* ═══════════════════════════════════════════════════════════════ */

static void bench(size_t K, int nruns, int warmup, int iters) {
    const size_t NN = 3*K;
    if (K < 8 || (K & 7)) return;

    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *or_old = alloc64(NN), *oi_old = alloc64(NN);
    double *or_new = alloc64(NN), *oi_new = alloc64(NN);
    double *or_pk  = alloc64(NN), *oi_pk  = alloc64(NN);
    double *ref_r  = alloc64(NN), *ref_i  = alloc64(NN);
    double *twr = alloc64(2*K), *twi = alloc64(2*K);

    /* Packed buffers */
    double *pk_ir = alloc64(NN), *pk_ii = alloc64(NN);
    double *pk_or = alloc64(NN), *pk_oi = alloc64(NN);
    double *pk_twr = alloc64(2*K), *pk_twi = alloc64(2*K);

    srand(42+(unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    gen_flat_tw(twr, twi, K);

    /* Old kernel twiddle struct */
    radix3_stage_twiddles_t old_tw;
    old_tw.re = twr;
    old_tw.im = twi;

    /* Pre-pack */
    r3_pack_avx512(ir, ii, pk_ir, pk_ii, K);
    r3_pack_tw_avx512(twr, twi, pk_twr, pk_twi, K);

    /* Reference */
    ref_tw_dft3(ir, ii, ref_r, ref_i, K, 1);

    /* Correctness: old */
    radix3_stage_forward_avx512(K, ir, ii, or_old, oi_old, &old_tw);
    double err_old = maxerr(ref_r, ref_i, or_old, oi_old, NN);

    /* Correctness: new strided */
    radix3_tw_dit_kernel_fwd_avx512(ir, ii, or_new, oi_new, twr, twi, K);
    double err_new = maxerr(ref_r, ref_i, or_new, oi_new, NN);

    /* Correctness: packed (unpack to compare) */
    r3_packed_kern_fwd(pk_ir, pk_ii, pk_or, pk_oi, pk_twr, pk_twi, K);
    r3_unpack_avx512(pk_or, pk_oi, or_pk, oi_pk, K);
    double err_pk = maxerr(ref_r, ref_i, or_pk, oi_pk, NN);

    /* Timing: min-of-N runs */
    double best_old = 1e18, best_new = 1e18, best_pk = 1e18;

    for (int r = 0; r < nruns; r++) {
        /* OLD */
        for (int i = 0; i < warmup; i++)
            radix3_stage_forward_avx512(K, ir, ii, or_old, oi_old, &old_tw);
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            radix3_stage_forward_avx512(K, ir, ii, or_old, oi_old, &old_tw);
        double ns = (now_ns() - t0) / iters;
        if (ns < best_old) best_old = ns;

        /* NEW strided */
        for (int i = 0; i < warmup; i++)
            radix3_tw_dit_kernel_fwd_avx512(ir, ii, or_new, oi_new, twr, twi, K);
        t0 = now_ns();
        for (int i = 0; i < iters; i++)
            radix3_tw_dit_kernel_fwd_avx512(ir, ii, or_new, oi_new, twr, twi, K);
        ns = (now_ns() - t0) / iters;
        if (ns < best_new) best_new = ns;

        /* PACKED kernel only */
        for (int i = 0; i < warmup; i++)
            r3_packed_kern_fwd(pk_ir, pk_ii, pk_or, pk_oi, pk_twr, pk_twi, K);
        t0 = now_ns();
        for (int i = 0; i < iters; i++)
            r3_packed_kern_fwd(pk_ir, pk_ii, pk_or, pk_oi, pk_twr, pk_twi, K);
        ns = (now_ns() - t0) / iters;
        if (ns < best_pk) best_pk = ns;
    }

    double r_new_vs_old = best_old / best_new;
    double r_pk_vs_old  = best_old / best_pk;
    double r_pk_vs_new  = best_new / best_pk;

    const char *winner;
    double best_all = best_old;
    winner = "OLD";
    if (best_new < best_all) { best_all = best_new; winner = "NEW"; }
    if (best_pk  < best_all) { best_all = best_pk;  winner = "PACKED"; }

    printf("  K=%-5zu | %7.1f  %7.1f  %7.1f | new/old=%.2f  pk/old=%.2f  pk/new=%.2f | %-6s  err=%.0e/%.0e/%.0e\n",
           K, best_old, best_new, best_pk,
           r_new_vs_old, r_pk_vs_old, r_pk_vs_new,
           winner, err_old, err_new, err_pk);

    free(ir);free(ii);free(or_old);free(oi_old);free(or_new);free(oi_new);
    free(or_pk);free(oi_pk);free(ref_r);free(ref_i);free(twr);free(twi);
    free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);free(pk_twr);free(pk_twi);
}

int main(void) {
    printf("═══ Radix-3 AVX-512 Three-Way: Old vs New vs Packed (min-of-5, fwd) ═══\n");
    printf("  OLD    = hand-written DIF (prefetch, loadu, pointer offsets)\n");
    printf("  NEW    = generated straight-line (clean, no prefetch)\n");
    printf("  PACKED = generated kernel on packed layout (K=T=8 per block)\n");
    printf("  ratio >1 = numerator faster\n\n");
    printf("  %-7s | %7s  %7s  %7s | %-10s %-10s %-10s | winner\n",
           "K", "old ns", "new ns", "pk ns", "new/old", "pk/old", "pk/new");
    printf("  ────────┼─────────────────────────┼──────────────────────────────────┼──────\n");

    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    for (int i = 0; i < 9; i++) {
        int it = (int)(2000000.0 / Ks[i]);
        if (it < 200) it = 200;
        bench(Ks[i], 5, it/10, it);
    }

    printf("\n  Notes:\n");
    printf("  - PACKED kernel processes K/8 blocks × DFT-3 at K=8 each\n");
    printf("  - No pack/unpack overhead included (data stays packed between stages)\n");
    printf("  - Twiddles pre-packed at plan time (one-time cost)\n");

    return 0;
}
