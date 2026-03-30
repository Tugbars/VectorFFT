/**
 * bench_from_wisdom.c -- Benchmark using calibrated wisdom
 *
 * Reads vfft_wisdom.txt, runs only the optimal factorization+variant
 * per N, and prints a clean comparison table vs FFTW.
 *
 * Usage: bench_from_wisdom [wisdom_file]
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

typedef void (*n1_ovs_fn)(const double*, const double*, double*, double*,
                          size_t, size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);
typedef void (*fused_fn)(const double*, const double*,
                         double*, double*,
                         const double*, const double*);

/* ================================================================
 * Codelet registry (same as calibrator)
 * ================================================================ */

typedef struct { size_t R; n1_ovs_fn n1; t1_fn t1; } codelet_t;

static const codelet_t CODELETS[] = {
    {  4, (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  (t1_fn)radix4_t1_dit_fwd_avx2  },
    {  8, (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,  (t1_fn)radix8_t1_dit_fwd_avx2  },
    { 16, (n1_ovs_fn)radix16_n1_ovs_fwd_avx2, (t1_fn)radix16_t1_dit_fwd_avx2 },
    { 32, (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, (t1_fn)radix32_t1_dit_fwd_avx2 },
    { 64, (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, (t1_fn)radix64_t1_dit_fwd_avx2 },
    {  0, NULL, NULL }
};

static const codelet_t *find(size_t R) {
    for (const codelet_t *c = CODELETS; c->R; c++)
        if (c->R == R) return c;
    return NULL;
}

/* ================================================================
 * Fused variants (inline + noinline)
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

/* Fused-inline */
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

/* Noinline wrappers */
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

/* Fused-noinline */
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
 * Lookup: given R, M, variant → function to call
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
 * Helpers
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

static void ct_indirect(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *W_re, const double *W_im, size_t R, size_t M)
{
    n1(ir, ii, or_, oi, R, 1, R, M);
    t1(or_, oi, W_re, W_im, M, M);
}

/* ================================================================
 * Wisdom entry
 * ================================================================ */

typedef struct {
    size_t N, R, M;
    char   variant;  /* I, F, N */
} wisdom_entry;

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    const char *wisfile = "vfft_wisdom.txt";
    if (argc > 1) wisfile = argv[1];

    FILE *wf = fopen(wisfile, "r");
    if (!wf) {
        fprintf(stderr, "Cannot open wisdom file: %s\n", wisfile);
        fprintf(stderr, "Run vfft_calibrate first.\n");
        return 1;
    }

    /* Parse wisdom */
    wisdom_entry entries[256];
    int nentries = 0;
    char line[256];
    while (fgets(line, sizeof(line), wf)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        size_t N, R, M;
        char var;
        double ns, fns, spd;
        if (sscanf(line, "%zu %zu %zu %c %lf %lf %lf", &N, &R, &M, &var, &ns, &fns, &spd) >= 4) {
            entries[nentries].N = N;
            entries[nentries].R = R;
            entries[nentries].M = M;
            entries[nentries].variant = var;
            nentries++;
        }
    }
    fclose(wf);

    printf("================================================================\n");
    printf("  VectorFFT vs FFTW -- Wisdom-guided benchmark\n");
    printf("  Wisdom: %s (%d entries)\n", wisfile, nentries);
    printf("================================================================\n\n");
    printf("  %-8s  %-7s %-4s  %8s  %8s  %7s\n",
           "N", "RxM", "var", "VecFFT", "FFTW", "ratio");
    printf("  %-8s  %-7s %-4s  %8s  %8s  %7s\n",
           "--------", "-------", "----", "--------", "--------", "-------");
    fflush(stdout);

    for (int e = 0; e < nentries; e++) {
        size_t N = entries[e].N;
        size_t R = entries[e].R;
        size_t M = entries[e].M;
        char   var = entries[e].variant;

        int reps = (int)(2e6 / (N + 1));
        if (reps < 200) reps = 200;
        if (reps > 200000) reps = 200000;

        /* Input */
        double *in_re = (double*)aligned_alloc(32, N * 8);
        double *in_im = (double*)aligned_alloc(32, N * 8);
        srand(42);
        for (size_t i = 0; i < N; i++) {
            in_re[i] = (double)rand() / RAND_MAX - 0.5;
            in_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* FFTW */
        double *fri = fftw_malloc(N * 8), *fii = fftw_malloc(N * 8);
        double *fro = fftw_malloc(N * 8), *fio = fftw_malloc(N * 8);
        memcpy(fri, in_re, N * 8); memcpy(fii, in_im, N * 8);
        fftw_iodim d = {.n = (int)N, .is = 1, .os = 1};
        fftw_iodim h = {.n = 1, .is = (int)N, .os = (int)N};
        fftw_plan fp = fftw_plan_guru_split_dft(1, &d, 1, &h,
                        fri, fii, fro, fio, FFTW_MEASURE);
        for (int i = 0; i < 20; i++) fftw_execute(fp);
        double fftw_best = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                fftw_execute_split_dft(fp, fri, fii, fro, fio);
            double ns = (now_ns() - t0) / reps;
            if (ns < fftw_best) fftw_best = ns;
        }

        /* VectorFFT — run the wisdom-selected variant */
        double *W_re = (double*)aligned_alloc(32, (R - 1) * M * 8);
        double *W_im = (double*)aligned_alloc(32, (R - 1) * M * 8);
        double *out_re = (double*)aligned_alloc(32, N * 8);
        double *out_im = (double*)aligned_alloc(32, N * 8);
        init_tw(W_re, W_im, R, M);

        const codelet_t *cr = find(R);
        const codelet_t *cm = find(M);
        const fused_pair *fp2 = find_fused(R, M);

        double vfft_best = 1e18;
        int ok = 0;

        if (var == 'I' && cm && cr) {
            /* Indirect */
            for (int i = 0; i < 20; i++)
                ct_indirect(cm->n1, cr->t1, in_re, in_im, out_re, out_im,
                            W_re, W_im, R, M);
            for (int t = 0; t < 7; t++) {
                double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    ct_indirect(cm->n1, cr->t1, in_re, in_im, out_re, out_im,
                                W_re, W_im, R, M);
                double ns = (now_ns() - t0) / reps;
                if (ns < vfft_best) vfft_best = ns;
            }
            ok = 1;
        } else if (var == 'F' && fp2) {
            for (int i = 0; i < 20; i++)
                fp2->fi(in_re, in_im, out_re, out_im, W_re, W_im);
            for (int t = 0; t < 7; t++) {
                double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    fp2->fi(in_re, in_im, out_re, out_im, W_re, W_im);
                double ns = (now_ns() - t0) / reps;
                if (ns < vfft_best) vfft_best = ns;
            }
            ok = 1;
        } else if (var == 'N' && fp2) {
            for (int i = 0; i < 20; i++)
                fp2->fni(in_re, in_im, out_re, out_im, W_re, W_im);
            for (int t = 0; t < 7; t++) {
                double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    fp2->fni(in_re, in_im, out_re, out_im, W_re, W_im);
                double ns = (now_ns() - t0) / reps;
                if (ns < vfft_best) vfft_best = ns;
            }
            ok = 1;
        }

        if (ok) {
            double ratio = fftw_best / vfft_best;
            const char *tag = ratio >= 1.0 ? " WINS" : "";
            char rxm[16];
            snprintf(rxm, sizeof(rxm), "%zux%zu", R, M);
            const char *vname = var == 'I' ? "I " : var == 'F' ? "F " : "NI";
            printf("  %-8zu  %-7s %-4s  %7.0f   %7.0f   %5.2fx%s\n",
                   N, rxm, vname, vfft_best, fftw_best, ratio, tag);
        } else {
            printf("  %-8zu  SKIPPED (variant %c not available)\n", N, var);
        }
        fflush(stdout);

        fftw_destroy_plan(fp);
        fftw_free(fri); fftw_free(fii); fftw_free(fro); fftw_free(fio);
        aligned_free(W_re); aligned_free(W_im);
        aligned_free(out_re); aligned_free(out_im);
        aligned_free(in_re); aligned_free(in_im);
    }

    printf("\n");
    return 0;
}
