/**
 * test_radix11_walk.c — Radix-11 pack+walk correctness + benchmark
 * ISA-adaptive: uses AVX-512 if available, falls back to AVX2.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix11_dispatch.h"

/* Portable twiddle table builder — r11_build_tw_table in fft_radix11_genfft.h
 * may be inside an __AVX512F__ guard, so provide a fallback here. */
#ifndef R11_BUILD_TW_TABLE_DEFINED
static void r11_build_tw_table_portable(size_t K, double *tw_re, double *tw_im)
{
    const size_t NN = 11 * K;
    for (size_t n = 1; n < 11; n++)
        for (size_t k = 0; k < K; k++)
        {
            double a = 2.0 * M_PI * (double)(n * k) / (double)NN;
            tw_re[(n - 1) * K + k] = cos(a);
            tw_im[(n - 1) * K + k] = -sin(a);
        }
}
#define r11_build_tw_table r11_build_tw_table_portable
#endif

/* Portable timer */
#ifdef _WIN32
#include <windows.h>
static double now_ns(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#endif

/* Portable aligned alloc */
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n)
{
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (p)
        memset(p, 0, n * sizeof(double));
    return p;
}
static void free64(void *p) { _aligned_free(p); }
#else
static double *alloc64(size_t n)
{
    double *p = NULL;
    posix_memalign((void **)&p, 64, n * sizeof(double));
    if (p)
        memset(p, 0, n * sizeof(double));
    return p;
}
static void free64(void *p) { free(p); }
#endif

static double maxerr(const double *ar, const double *ai,
                     const double *br, const double *bi, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i++)
    {
        double dr = fabs(ar[i] - br[i]), di = fabs(ai[i] - bi[i]);
        if (dr > mx)
            mx = dr;
        if (di > mx)
            mx = di;
    }
    return mx;
}

/* Reference DFT-11 with twiddles */
static void ref_tw_dft11(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K, int fwd)
{
    const size_t NN = 11 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++)
    {
        double xr[11], xi[11];
        for (size_t n = 0; n < 11; n++)
        {
            double dr = ir[n * K + k], di = ii[n * K + k];
            if (n > 0)
            {
                double a = sign * 2.0 * M_PI * (double)(n * k) / (double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr * wr - di * wi;
                di = dr * wi + di * wr;
                dr = tr;
            }
            xr[n] = dr;
            xi[n] = di;
        }
        for (size_t m = 0; m < 11; m++)
        {
            double sr = 0, si = 0;
            for (size_t n2 = 0; n2 < 11; n2++)
            {
                double a = sign * 2.0 * M_PI * (double)(m * n2) / 11.0;
                sr += xr[n2] * cos(a) - xi[n2] * sin(a);
                si += xr[n2] * sin(a) + xi[n2] * cos(a);
            }
            or_[m * K + k] = sr;
            oi[m * K + k] = si;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * ISA-adaptive pack/unpack/walk wrappers
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#define WALK_T 8
#define WALK_ISA "AVX-512"

__attribute__((target("avx512f"))) static void r11_pack_simd(const double *sr, const double *si,
                                                             double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
        for (size_t n = 0; n < 11; n++)
        {
            _mm512_store_pd(&dr[b * 88 + n * 8], _mm512_loadu_pd(&sr[n * K + b * 8]));
            _mm512_store_pd(&di[b * 88 + n * 8], _mm512_loadu_pd(&si[n * K + b * 8]));
        }
}

__attribute__((target("avx512f"))) static void r11_unpack_simd(const double *sr, const double *si,
                                                               double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
        for (size_t n = 0; n < 11; n++)
        {
            _mm512_storeu_pd(&dr[n * K + b * 8], _mm512_load_pd(&sr[b * 88 + n * 8]));
            _mm512_storeu_pd(&di[n * K + b * 8], _mm512_load_pd(&si[b * 88 + n * 8]));
        }
}

__attribute__((target("avx512f"))) static void r11_pack_tw_simd(const double *sr, const double *si,
                                                                double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
        for (size_t n = 0; n < 10; n++)
        {
            _mm512_store_pd(&dr[b * 80 + n * 8], _mm512_loadu_pd(&sr[n * K + b * 8]));
            _mm512_store_pd(&di[b * 80 + n * 8], _mm512_loadu_pd(&si[n * K + b * 8]));
        }
}

typedef radix11_walk_plan_t walk_plan_type;
static void walk_plan_create(walk_plan_type *wp, size_t K)
{
    radix11_walk_plan_init(wp, K);
}

__attribute__((target("avx512f,fma"))) static void walk_fwd(const double *ir, const double *ii,
                                                            double *or_, double *oi, walk_plan_type *wp, size_t K)
{
    radix11_tw_pack_walk_fwd_avx512(ir, ii, or_, oi, wp, K);
}

__attribute__((target("avx512f,fma"))) static void walk_bwd(const double *ir, const double *ii,
                                                            double *or_, double *oi, walk_plan_type *wp, size_t K)
{
    radix11_tw_pack_walk_bwd_avx512(ir, ii, or_, oi, wp, K);
}

#elif defined(__AVX2__)
#define WALK_T 4
#define WALK_ISA "AVX2"

__attribute__((target("avx2"))) static void r11_pack_simd(const double *sr, const double *si,
                                                          double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 4; b++)
        for (size_t n = 0; n < 11; n++)
        {
            _mm256_store_pd(&dr[b * 44 + n * 4], _mm256_loadu_pd(&sr[n * K + b * 4]));
            _mm256_store_pd(&di[b * 44 + n * 4], _mm256_loadu_pd(&si[n * K + b * 4]));
        }
}

__attribute__((target("avx2"))) static void r11_unpack_simd(const double *sr, const double *si,
                                                            double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 4; b++)
        for (size_t n = 0; n < 11; n++)
        {
            _mm256_storeu_pd(&dr[n * K + b * 4], _mm256_load_pd(&sr[b * 44 + n * 4]));
            _mm256_storeu_pd(&di[n * K + b * 4], _mm256_load_pd(&si[b * 44 + n * 4]));
        }
}

__attribute__((target("avx2"))) static void r11_pack_tw_simd(const double *sr, const double *si,
                                                             double *dr, double *di, size_t K)
{
    for (size_t b = 0; b < K / 4; b++)
        for (size_t n = 0; n < 10; n++)
        {
            _mm256_store_pd(&dr[b * 40 + n * 4], _mm256_loadu_pd(&sr[n * K + b * 4]));
            _mm256_store_pd(&di[b * 40 + n * 4], _mm256_loadu_pd(&si[n * K + b * 4]));
        }
}

typedef radix11_walk_plan_avx2_t walk_plan_type;
static void walk_plan_create(walk_plan_type *wp, size_t K)
{
    radix11_walk_plan_avx2_init(wp, K);
}

__attribute__((target("avx2,fma"))) static void walk_fwd(const double *ir, const double *ii,
                                                         double *or_, double *oi, walk_plan_type *wp, size_t K)
{
    radix11_tw_pack_walk_fwd_avx2(ir, ii, or_, oi, wp, K);
}

__attribute__((target("avx2,fma"))) static void walk_bwd(const double *ir, const double *ii,
                                                         double *or_, double *oi, walk_plan_type *wp, size_t K)
{
    radix11_tw_pack_walk_bwd_avx2(ir, ii, or_, oi, wp, K);
}

#else
#error "test_radix11_walk.c requires AVX2 or AVX-512"
#endif

/* Packed-table twiddle apply + notw kernel (scalar, always works) */
static void r11_tw_packed_table_fwd(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K)
{
    const size_t T = WALK_T;
    const size_t dbs = 11 * T, tbs = 10 * T, nb = K / T;

    double __attribute__((aligned(64))) tmp_re[11 * 8];
    double __attribute__((aligned(64))) tmp_im[11 * 8];

    for (size_t blk = 0; blk < nb; blk++)
    {
        const double *bir = in_re + blk * dbs, *bii = in_im + blk * dbs;
        const double *btr = tw_re + blk * tbs, *bti = tw_im + blk * tbs;

        for (size_t j = 0; j < T; j++)
        {
            tmp_re[j] = bir[j];
            tmp_im[j] = bii[j];
        }

        for (size_t n = 0; n < 10; n++)
        {
            for (size_t j = 0; j < T; j++)
            {
                double xr = bir[(n + 1u) * T + j];
                double xi = bii[(n + 1u) * T + j];
                double tr = btr[n * T + j];
                double ti = bti[n * T + j];
                tmp_re[(n + 1u) * T + j] = xr * tr - xi * ti;
                tmp_im[(n + 1u) * T + j] = xr * ti + xi * tr;
            }
        }

        r11_dispatch_fwd(tmp_re, tmp_im,
                         out_re + blk * dbs, out_im + blk * dbs, T);
    }
}

int main(void)
{
    int total = 0, passed = 0;

    printf("=== Radix-11 Pack+Walk Test [%s, T=%d] ===\n\n", WALK_ISA, WALK_T);

    /* ── Correctness ── */
    printf("-- Correctness --\n");
    size_t cKs[] = {
        (size_t)WALK_T,
        (size_t)WALK_T * 8,
        (size_t)WALK_T * 32,
        (size_t)WALK_T * 128,
        (size_t)WALK_T * 512};

    for (size_t ki = 0; ki < 5; ki++)
    {
        size_t K = cKs[ki], NN = 11 * K;

        double *ir = alloc64(NN), *ii = alloc64(NN);
        double *ref_r = alloc64(NN), *ref_i = alloc64(NN);
        double *out_r = alloc64(NN), *out_i = alloc64(NN);
        double *pk_ir = alloc64(NN), *pk_ii = alloc64(NN);
        double *pk_or = alloc64(NN), *pk_oi = alloc64(NN);

        srand(42 + (unsigned)K);
        for (size_t i = 0; i < NN; i++)
        {
            ir[i] = (double)rand() / RAND_MAX - 0.5;
            ii[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* Forward */
        ref_tw_dft11(ir, ii, ref_r, ref_i, K, 1);
        r11_pack_simd(ir, ii, pk_ir, pk_ii, K);
        walk_plan_type wp;
        walk_plan_create(&wp, K);
        walk_fwd(pk_ir, pk_ii, pk_or, pk_oi, &wp, K);
        r11_unpack_simd(pk_or, pk_oi, out_r, out_i, K);
        double ef = maxerr(ref_r, ref_i, out_r, out_i, NN);

        /* Backward */
        ref_tw_dft11(ir, ii, ref_r, ref_i, K, 0);
        r11_pack_simd(ir, ii, pk_ir, pk_ii, K);
        walk_plan_create(&wp, K);
        walk_bwd(pk_ir, pk_ii, pk_or, pk_oi, &wp, K);
        r11_unpack_simd(pk_or, pk_oi, out_r, out_i, K);
        double eb = maxerr(ref_r, ref_i, out_r, out_i, NN);

        int ok = (ef < 1e-9 && eb < 1e-9);
        total++;
        passed += ok;
        printf("  K=%-5zu fwd=%.1e bwd=%.1e %s\n", K, ef, eb, ok ? "PASS" : "FAIL");

        free64(ir);
        free64(ii);
        free64(ref_r);
        free64(ref_i);
        free64(out_r);
        free64(out_i);
        free64(pk_ir);
        free64(pk_ii);
        free64(pk_or);
        free64(pk_oi);
    }
    printf("\n=== %d / %d passed ===\n\n", passed, total);

    /* ── Benchmark ── */
    printf("-- Benchmark: Packed-table vs Pack+Walk (min-of-5, fwd only) --\n");
    printf("  %-5s | %8s %8s | %7s | winner\n", "K", "pk-tbl", "pk+walk", "pw/pk");
    printf("  ------+-------------------+---------+-------\n");

    size_t bKs[] = {
        (size_t)WALK_T * 8,
        (size_t)WALK_T * 32,
        (size_t)WALK_T * 64,
        (size_t)WALK_T * 128,
        (size_t)WALK_T * 256,
        (size_t)WALK_T * 512};

    for (size_t ki = 0; ki < 6; ki++)
    {
        size_t K = bKs[ki], NN = 11 * K;

        double *ir = alloc64(NN), *ii = alloc64(NN);
        double *pk_ir = alloc64(NN), *pk_ii = alloc64(NN);
        double *pk_or = alloc64(NN), *pk_oi = alloc64(NN);
        double *twr = alloc64(10 * K), *twi = alloc64(10 * K);
        double *pk_twr = alloc64(10 * K), *pk_twi = alloc64(10 * K);

        srand(42 + (unsigned)K);
        for (size_t i = 0; i < NN; i++)
        {
            ir[i] = (double)rand() / RAND_MAX - 0.5;
            ii[i] = (double)rand() / RAND_MAX - 0.5;
        }

        r11_build_tw_table(K, twr, twi);
        r11_pack_simd(ir, ii, pk_ir, pk_ii, K);
        r11_pack_tw_simd(twr, twi, pk_twr, pk_twi, K);

        walk_plan_type wp;
        int iters = (int)(500000.0 / (double)K);
        if (iters < 100)
            iters = 100;
        int warmup = iters / 10;
        double best_t = 1e18, best_w = 1e18;

        for (int run = 0; run < 5; run++)
        {
            /* Packed table */
            for (int i = 0; i < warmup; i++)
                r11_tw_packed_table_fwd(pk_ir, pk_ii, pk_or, pk_oi,
                                        pk_twr, pk_twi, K);
            double t0 = now_ns();
            for (int i = 0; i < iters; i++)
                r11_tw_packed_table_fwd(pk_ir, pk_ii, pk_or, pk_oi,
                                        pk_twr, pk_twi, K);
            double ns = (now_ns() - t0) / iters;
            if (ns < best_t)
                best_t = ns;

            /* Pack+walk */
            walk_plan_create(&wp, K);
            for (int i = 0; i < warmup; i++)
            {
                walk_plan_create(&wp, K);
                walk_fwd(pk_ir, pk_ii, pk_or, pk_oi, &wp, K);
            }
            walk_plan_create(&wp, K);
            t0 = now_ns();
            for (int i = 0; i < iters; i++)
            {
                walk_plan_create(&wp, K);
                walk_fwd(pk_ir, pk_ii, pk_or, pk_oi, &wp, K);
            }
            ns = (now_ns() - t0) / iters;
            if (ns < best_w)
                best_w = ns;
        }

        double r = best_t / best_w;
        printf("  K=%-4zu | %7.0f  %7.0f | %6.2fx | %s\n",
               K, best_t, best_w, r,
               r > 1.03 ? "PK+WALK" : (r < 0.97 ? "PK-TBL" : "~tie"));

        free64(ir);
        free64(ii);
        free64(pk_ir);
        free64(pk_ii);
        free64(pk_or);
        free64(pk_oi);
        free64(twr);
        free64(twi);
        free64(pk_twr);
        free64(pk_twi);
    }

    return (passed == total) ? 0 : 1;
}