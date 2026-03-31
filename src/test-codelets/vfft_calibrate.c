/**
 * vfft_calibrate.c -- Runtime calibrator for optimal CT dispatch
 *
 * Benchmarks all valid 1-level factorizations for each N with three
 * call strategies (indirect, fused-inline, fused-noinline), verifies
 * correctness against FFTW, and writes a wisdom file with the best
 * factorization + strategy per N.
 *
 * Usage: vfft_calibrate [output_file] [min_N] [max_N]
 *   Default: vfft_wisdom.txt, N=16..4096
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_t1_dit.h"
#include "r64_unified_avx2.h"

/* ================================================================
 * Function pointer types
 * ================================================================ */

typedef void (*n1_ovs_fn)(const double*, const double*, double*, double*,
                          size_t, size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);
typedef void (*fused_fn)(const double*, const double*,
                         double*, double*,
                         const double*, const double*);

/* ================================================================
 * Codelet registry
 * ================================================================ */

typedef struct { size_t R; n1_ovs_fn n1; t1_fn t1; const char *t1_tag; } codelet_t;

static const codelet_t CODELETS[] = {
    {  4, (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  (t1_fn)radix4_t1_dit_fwd_avx2,  "" },
    {  8, (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,  (t1_fn)radix8_t1_dit_fwd_avx2,  "" },
    { 16, (n1_ovs_fn)radix16_n1_ovs_fwd_avx2, (t1_fn)radix16_t1_dit_fwd_avx2, "" },
    { 32, (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, (t1_fn)radix32_t1_dit_fwd_avx2, "" },
    { 64, (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, (t1_fn)radix64_t1_dit_fwd_avx2, "" },
    /* Log3 t1: fewer twiddle loads, wins when tw table exceeds L1 */
    { 64, (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, (t1_fn)radix64_t1_dit_log3_fwd_avx2, "L" },
    {  0, NULL, NULL, NULL }
};

static const codelet_t *find(size_t R) {
    for (const codelet_t *c = CODELETS; c->R; c++)
        if (c->R == R && c->t1_tag[0] == '\0') return c;
    return NULL;
}

/* ================================================================
 * Twiddle init
 * ================================================================ */

static void init_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* ================================================================
 * Variant 1: Indirect (function pointer calls)
 * ================================================================ */

static void ct_indirect(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *W_re, const double *W_im, size_t R, size_t M)
{
    n1(ir, ii, or_, oi, R, 1, R, M);
    t1(or_, oi, W_re, W_im, M, M);
}

/* ================================================================
 * Variant 2: Fused inline (direct calls, ICX can inline)
 * ================================================================ */

#define FUSED_1LEVEL(name, n1_func, t1_func, R_val, M_val)       \
static void name(                                                  \
    const double * __restrict__ ir, const double * __restrict__ ii,\
    double * __restrict__ or_, double * __restrict__ oi,           \
    const double * __restrict__ W_re, const double * __restrict__ W_im) \
{                                                                  \
    n1_func(ir, ii, or_, oi, R_val, 1, R_val, M_val);             \
    t1_func(or_, oi, W_re, W_im, M_val, M_val);                   \
}

FUSED_1LEVEL(fi_4x4,   radix4_n1_ovs_fwd_avx2,  radix4_t1_dit_fwd_avx2,   4,  4)
FUSED_1LEVEL(fi_4x8,   radix8_n1_ovs_fwd_avx2,  radix4_t1_dit_fwd_avx2,   4,  8)
FUSED_1LEVEL(fi_4x16,  radix16_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 16)
FUSED_1LEVEL(fi_8x8,   radix8_n1_ovs_fwd_avx2,  radix8_t1_dit_fwd_avx2,   8,  8)
FUSED_1LEVEL(fi_4x32,  radix32_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 32)
FUSED_1LEVEL(fi_8x16,  radix16_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 16)
FUSED_1LEVEL(fi_4x64,  radix64_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 64)
FUSED_1LEVEL(fi_8x32,  radix32_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 32)
FUSED_1LEVEL(fi_16x16, radix16_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 16)
FUSED_1LEVEL(fi_8x64,  radix64_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 64)
FUSED_1LEVEL(fi_16x32, radix32_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 32)
FUSED_1LEVEL(fi_16x64, radix64_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 64)
FUSED_1LEVEL(fi_32x32, radix32_n1_ovs_fwd_avx2, radix32_t1_dit_fwd_avx2, 32, 32)
FUSED_1LEVEL(fi_32x64, radix64_n1_ovs_fwd_avx2, radix32_t1_dit_fwd_avx2, 32, 64)
FUSED_1LEVEL(fi_64x64, radix64_n1_ovs_fwd_avx2, radix64_t1_dit_fwd_avx2, 64, 64)

/* ================================================================
 * Variant 3: Fused noinline (direct calls, ICX cannot inline)
 * ================================================================ */

#define NOINLINE_N1(name, func) \
__attribute__((noinline)) static void name( \
    const double *ir, const double *ii, double *or_, double *oi, \
    size_t is, size_t os, size_t vl, size_t ovs) { \
    func(ir, ii, or_, oi, is, os, vl, ovs); }

#define NOINLINE_T1(name, func) \
__attribute__((noinline)) static void name( \
    double *rio_re, double *rio_im, \
    const double *W_re, const double *W_im, \
    size_t ios, size_t me) { \
    func(rio_re, rio_im, W_re, W_im, ios, me); }

NOINLINE_N1(r4_n1_ni,  radix4_n1_ovs_fwd_avx2)
NOINLINE_T1(r4_t1_ni,  radix4_t1_dit_fwd_avx2)
NOINLINE_N1(r8_n1_ni,  radix8_n1_ovs_fwd_avx2)
NOINLINE_T1(r8_t1_ni,  radix8_t1_dit_fwd_avx2)
NOINLINE_N1(r16_n1_ni, radix16_n1_ovs_fwd_avx2)
NOINLINE_T1(r16_t1_ni, radix16_t1_dit_fwd_avx2)
NOINLINE_N1(r32_n1_ni, radix32_n1_ovs_fwd_avx2)
NOINLINE_T1(r32_t1_ni, radix32_t1_dit_fwd_avx2)
NOINLINE_N1(r64_n1_ni, radix64_n1_ovs_fwd_avx2)
NOINLINE_T1(r64_t1_ni, radix64_t1_dit_fwd_avx2)

FUSED_1LEVEL(fni_4x4,   r4_n1_ni,  r4_t1_ni,   4,  4)
FUSED_1LEVEL(fni_4x8,   r8_n1_ni,  r4_t1_ni,   4,  8)
FUSED_1LEVEL(fni_4x16,  r16_n1_ni, r4_t1_ni,   4, 16)
FUSED_1LEVEL(fni_8x8,   r8_n1_ni,  r8_t1_ni,   8,  8)
FUSED_1LEVEL(fni_4x32,  r32_n1_ni, r4_t1_ni,   4, 32)
FUSED_1LEVEL(fni_8x16,  r16_n1_ni, r8_t1_ni,   8, 16)
FUSED_1LEVEL(fni_4x64,  r64_n1_ni, r4_t1_ni,   4, 64)
FUSED_1LEVEL(fni_8x32,  r32_n1_ni, r8_t1_ni,   8, 32)
FUSED_1LEVEL(fni_16x16, r16_n1_ni, r16_t1_ni, 16, 16)
FUSED_1LEVEL(fni_8x64,  r64_n1_ni, r8_t1_ni,   8, 64)
FUSED_1LEVEL(fni_16x32, r32_n1_ni, r16_t1_ni, 16, 32)
FUSED_1LEVEL(fni_16x64, r64_n1_ni, r16_t1_ni, 16, 64)
FUSED_1LEVEL(fni_32x32, r32_n1_ni, r32_t1_ni, 32, 32)
FUSED_1LEVEL(fni_32x64, r64_n1_ni, r32_t1_ni, 32, 64)
FUSED_1LEVEL(fni_64x64, r64_n1_ni, r64_t1_ni, 64, 64)

/* ================================================================
 * Fused variant lookup tables
 * ================================================================ */

typedef struct { size_t R; size_t M; fused_fn fi; fused_fn fni; } fused_pair;

static const fused_pair FUSED_PAIRS[] = {
    {  4,   4, fi_4x4,   fni_4x4   },
    {  4,   8, fi_4x8,   fni_4x8   },
    {  4,  16, fi_4x16,  fni_4x16  },
    {  8,   8, fi_8x8,   fni_8x8   },
    {  4,  32, fi_4x32,  fni_4x32  },
    {  8,  16, fi_8x16,  fni_8x16  },
    {  4,  64, fi_4x64,  fni_4x64  },
    {  8,  32, fi_8x32,  fni_8x32  },
    { 16,  16, fi_16x16, fni_16x16 },
    {  8,  64, fi_8x64,  fni_8x64  },
    { 16,  32, fi_16x32, fni_16x32 },
    { 16,  64, fi_16x64, fni_16x64 },
    { 32,  32, fi_32x32, fni_32x32 },
    { 32,  64, fi_32x64, fni_32x64 },
    { 64,  64, fi_64x64, fni_64x64 },
    {  0,   0, NULL,     NULL      }
};

static const fused_pair *find_fused(size_t R, size_t M) {
    for (const fused_pair *p = FUSED_PAIRS; p->R; p++)
        if (p->R == R && p->M == M) return p;
    return NULL;
}

/* ================================================================
 * Benchmark helpers
 * ================================================================ */

static double bench_fn(fused_fn fn, double *or_re, double *or_im,
                       const double *in_re, const double *in_im,
                       const double *W_re, const double *W_im, int reps)
{
    for (int i = 0; i < 20; i++)
        fn(in_re, in_im, or_re, or_im, W_re, W_im);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            fn(in_re, in_im, or_re, or_im, W_re, W_im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static double bench_indirect(n1_ovs_fn n1, t1_fn t1,
                              double *or_re, double *or_im,
                              const double *in_re, const double *in_im,
                              const double *W_re, const double *W_im,
                              size_t R, size_t M, int reps)
{
    for (int i = 0; i < 20; i++)
        ct_indirect(n1, t1, in_re, in_im, or_re, or_im, W_re, W_im, R, M);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            ct_indirect(n1, t1, in_re, in_im, or_re, or_im, W_re, W_im, R, M);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static double check_err(const double *out_re, const double *out_im,
                         const double *ref_re, const double *ref_im, size_t N)
{
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - ref_re[i]) + fabs(out_im[i] - ref_im[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

/* ================================================================
 * Wisdom entry
 * ================================================================ */

typedef struct {
    size_t N;
    size_t R;
    size_t M;
    char   variant;   /* 'I'=indirect, 'F'=fused-inline, 'N'=fused-noinline */
    double ns;
    double fftw_ns;
} wisdom_entry;

/* ================================================================
 * Calibrate one N: test all factorizations x 3 variants
 * ================================================================ */

static wisdom_entry calibrate_N(size_t N) {
    int reps = (int)(2e6 / (N + 1));
    if (reps < 200) reps = 200;
    if (reps > 200000) reps = 200000;

    /* Allocate input */
    double *in_re = (double*)aligned_alloc(32, N * 8);
    double *in_im = (double*)aligned_alloc(32, N * 8);
    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand() / RAND_MAX - 0.5;
        in_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* FFTW reference */
    double *fre = fftw_malloc(N * 8), *fim = fftw_malloc(N * 8);
    double *fro = fftw_malloc(N * 8), *fio = fftw_malloc(N * 8);
    memcpy(fre, in_re, N * 8);
    memcpy(fim, in_im, N * 8);
    fftw_iodim d = {.n = (int)N, .is = 1, .os = 1};
    fftw_iodim h = {.n = 1, .is = (int)N, .os = (int)N};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &d, 1, &h, fre, fim, fro, fio, FFTW_MEASURE);
    /* FFTW_MEASURE destroys input — restore before executing for reference */
    memcpy(fre, in_re, N * 8);
    memcpy(fim, in_im, N * 8);
    fftw_execute(fp);

    /* Bench FFTW */
    for (int i = 0; i < 20; i++) fftw_execute(fp);
    double fftw_best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            fftw_execute_split_dft(fp, fre, fim, fro, fio);
        double ns = (now_ns() - t0) / reps;
        if (ns < fftw_best) fftw_best = ns;
    }

    /* Scratch buffers */
    double *out_re = (double*)aligned_alloc(32, N * 8);
    double *out_im = (double*)aligned_alloc(32, N * 8);

    wisdom_entry best = {.N = N, .R = 0, .M = 0, .variant = '?', .ns = 1e18, .fftw_ns = fftw_best};

    /* Test all valid 1-level factorizations */
    for (const codelet_t *cr = CODELETS; cr->R; cr++) {
        size_t R = cr->R;
        if (N % R != 0) continue;
        size_t M = N / R;
        if (M < 4) continue;

        const codelet_t *cm = find(M);
        if (!cm) continue;

        double *W_re = (double*)aligned_alloc(32, (R - 1) * M * 8);
        double *W_im = (double*)aligned_alloc(32, (R - 1) * M * 8);
        init_tw(W_re, W_im, R, M);

        const fused_pair *fp2 = find_fused(R, M);

        /* --- Variant I: indirect (or L for log3) --- */
        ct_indirect(cm->n1, cr->t1, in_re, in_im, out_re, out_im,
                    W_re, W_im, R, M);
        double err = check_err(out_re, out_im, fro, fio, N);
        if (err < 1e-10) {
            double ns = bench_indirect(cm->n1, cr->t1, out_re, out_im,
                                       in_re, in_im, W_re, W_im, R, M, reps);
            char v = cr->t1_tag[0] == 'L' ? 'L' : 'I';
            if (ns < best.ns) {
                best.R = R; best.M = M; best.variant = v; best.ns = ns;
            }
        }

        if (fp2) {
            /* --- Variant F: fused inline --- */
            fp2->fi(in_re, in_im, out_re, out_im, W_re, W_im);
            err = check_err(out_re, out_im, fro, fio, N);
            if (err < 1e-10) {
                double ns = bench_fn(fp2->fi, out_re, out_im,
                                     in_re, in_im, W_re, W_im, reps);
                if (ns < best.ns) {
                    best.R = R; best.M = M; best.variant = 'F'; best.ns = ns;
                }
            }

            /* --- Variant NI: fused noinline --- */
            fp2->fni(in_re, in_im, out_re, out_im, W_re, W_im);
            err = check_err(out_re, out_im, fro, fio, N);
            if (err < 1e-10) {
                double ns = bench_fn(fp2->fni, out_re, out_im,
                                     in_re, in_im, W_re, W_im, reps);
                if (ns < best.ns) {
                    best.R = R; best.M = M; best.variant = 'N'; best.ns = ns;
                }
            }
        }

        aligned_free(W_re);
        aligned_free(W_im);
    }

    fftw_destroy_plan(fp);
    fftw_free(fre); fftw_free(fim); fftw_free(fro); fftw_free(fio);
    aligned_free(out_re); aligned_free(out_im);
    aligned_free(in_re); aligned_free(in_im);

    return best;
}

/* ================================================================
 * Main: calibrate range of N, write wisdom file
 * ================================================================ */

int main(int argc, char **argv) {
    const char *outfile = "vfft_wisdom.txt";
    size_t min_N = 16, max_N = 4096;

    if (argc > 1) outfile = argv[1];
    if (argc > 2) min_N = (size_t)atoi(argv[2]);
    if (argc > 3) max_N = (size_t)atoi(argv[3]);

    printf("VectorFFT Calibrator\n");
    printf("  Output: %s\n", outfile);
    printf("  Range:  N=%zu..%zu\n", min_N, max_N);
    printf("  Variants: I=indirect, F=fused-inline, N=fused-noinline\n\n");
    fflush(stdout);

    /* Enumerate all power-of-2 sizes and composite sizes that
       factor into our available radixes {4,8,16,32,64} */
    size_t sizes[256];
    int nsizes = 0;

    /* Generate all N = product of radixes in [min_N, max_N] */
    size_t radixes[] = {4, 8, 16, 32, 64, 0};
    for (size_t *r1 = radixes; *r1; r1++) {
        /* 1-level: R * M where M is also a radix */
        for (size_t *r2 = radixes; *r2; r2++) {
            size_t N = (*r1) * (*r2);
            if (N < min_N || N > max_N) continue;
            /* Allow R>M factorizations too */
            /* Check not already in list */
            int dup = 0;
            for (int i = 0; i < nsizes; i++)
                if (sizes[i] == N) { dup = 1; break; }
            if (!dup) sizes[nsizes++] = N;
        }
    }

    /* Sort */
    for (int i = 0; i < nsizes - 1; i++)
        for (int j = i + 1; j < nsizes; j++)
            if (sizes[i] > sizes[j]) {
                size_t tmp = sizes[i]; sizes[i] = sizes[j]; sizes[j] = tmp;
            }

    /* Calibrate each N */
    wisdom_entry results[256];
    int nresults = 0;

    for (int i = 0; i < nsizes; i++) {
        size_t N = sizes[i];
        printf("  Calibrating N=%-6zu ... ", N);
        fflush(stdout);

        wisdom_entry w = calibrate_N(N);
        results[nresults++] = w;

        const char *vname = w.variant == 'I' ? "indirect" :
                            w.variant == 'F' ? "fused-inline" :
                            w.variant == 'N' ? "fused-noinline" :
                            w.variant == 'L' ? "indirect-log3" : "???";
        printf("%2zux%-3zu %-14s %6.0f ns  (%.2fx vs FFTW)\n",
               w.R, w.M, vname, w.ns, w.fftw_ns / w.ns);
        fflush(stdout);
    }

    /* Write wisdom file */
    FILE *f = fopen(outfile, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", outfile);
        return 1;
    }

    fprintf(f, "# VectorFFT Wisdom File\n");
    fprintf(f, "# Auto-generated by vfft_calibrate\n");
    fprintf(f, "# Format: N R M variant time_ns fftw_ns speedup\n");
    fprintf(f, "# variant: I=indirect F=fused-inline N=fused-noinline\n");
    fprintf(f, "#\n");

    for (int i = 0; i < nresults; i++) {
        wisdom_entry *w = &results[i];
        fprintf(f, "%zu %zu %zu %c %.0f %.0f %.2f\n",
                w->N, w->R, w->M, w->variant, w->ns, w->fftw_ns,
                w->fftw_ns / w->ns);
    }

    fclose(f);
    printf("\nWisdom written to %s (%d entries)\n", outfile, nresults);

    return 0;
}
