/* test_fft3d_roundtrip.c — validate the new 3D c2c (core/fft3d.h) on dag.
 *
 * Correctness protocol (order-agnostic — output is digit-scrambled per axis):
 *   1. ROUNDTRIP  bwd(fwd(x)) == N1*N2*N3 * x   (definitive)
 *   2. PARSEVAL   sum|X|^2 == N1*N2*N3 * sum|x|^2   (fwd alone is a real DFT)
 *   3. DC         x == const -> fwd has exactly one nonzero bin (== Ntot*c)
 *
 * Matrix: sizes x pass-A mode {FLAT, BLOCKED} x threads {1, 2, 4}.
 * Sizes include non-pow2 (60 = 2^2*3*5), anisotropy, and a prime on each
 * axis position (61) to exercise the Rader/Bluestein override paths.
 *
 * Inner plans via vfft_proto_auto_plan_dispatch (fast heuristic) wrapped by
 * stride_plan_3d_from — the wisdom-shaped path; stride_plan_3d's exhaustive
 * search is exercised once at the smallest size.
 *
 * Build: cd build_tuned && python build.py --src benches/test_fft3d_roundtrip.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft3d.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static double frand(void) { return 2.0 * ((double)rand() / RAND_MAX) - 1.0; }

static int g_fail = 0;

/* Build a 3D plan with auto inners + explicit pass-A mode. */
static stride_plan_t *make_plan(int N1, int N2, int N3, size_t a_block,
                                const vfft_proto_registry_t *reg) {
    size_t K0 = (size_t)N2 * (size_t)N3;
    size_t NR = (size_t)N1 * (size_t)N2;
    size_t B  = _fft3d_choose_tile(N3, NR);
    stride_plan_t *p0 = vfft_proto_auto_plan_dispatch(N1, K0, reg, NULL);
    stride_plan_t *p1 = vfft_proto_auto_plan_dispatch(N2, (size_t)N3, reg, NULL);
    stride_plan_t *pr = vfft_proto_auto_plan_dispatch(N3, B, reg, NULL);
    if (!p0 || !p1 || !pr) {
        if (p0) stride_plan_destroy(p0);
        if (p1) stride_plan_destroy(p1);
        if (pr) stride_plan_destroy(pr);
        return NULL;
    }
    return stride_plan_3d_from(N1, N2, N3, B, a_block, p0, p1, pr);
}

static void run_cell(int N1, int N2, int N3, size_t a_block, int T,
                     const vfft_proto_registry_t *reg) {
    size_t n = (size_t)N1 * N2 * N3;
    double scale = (double)n;

    stride_set_num_threads(T);
    stride_plan_t *plan = make_plan(N1, N2, N3, a_block, reg);
    if (!plan) {
        printf("  %4dx%-4dx%-4d ablk=%-5zu T=%d  PLAN FAILED\n",
               N1, N2, N3, a_block, T);
        g_fail++;
        return;
    }
    stride_fft3d_data_t *d = (stride_fft3d_data_t *)plan->override_data;

    double *re  = (double *)AALLOC(n * sizeof(double));
    double *im  = (double *)AALLOC(n * sizeof(double));
    double *rr  = (double *)AALLOC(n * sizeof(double));
    double *ri  = (double *)AALLOC(n * sizeof(double));

    /* ── 1. roundtrip on random data ── */
    srand(42);
    double in_energy = 0.0;
    for (size_t i = 0; i < n; i++) {
        rr[i] = re[i] = frand();
        ri[i] = im[i] = frand();
        in_energy += re[i]*re[i] + im[i]*im[i];
    }

    stride_execute_fwd(plan, re, im);

    /* ── 2. Parseval on the forward spectrum ── */
    double out_energy = 0.0;
    for (size_t i = 0; i < n; i++)
        out_energy += re[i]*re[i] + im[i]*im[i];
    double pars_err = fabs(out_energy - scale * in_energy) / (scale * in_energy);

    stride_execute_bwd(plan, re, im);

    double max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        double er = fabs(re[i] - scale * rr[i]);
        double ei = fabs(im[i] - scale * ri[i]);
        double m  = fabs(scale * rr[i]) + fabs(scale * ri[i]) + 1e-300;
        double rel = (er + ei) / m;
        if (rel > max_rel) max_rel = rel;
    }

    /* ── 3. DC impulse: constant input -> one nonzero bin == n*c ── */
    for (size_t i = 0; i < n; i++) { re[i] = 1.0; im[i] = 0.0; }
    stride_execute_fwd(plan, re, im);
    int nonzero = 0; double dc_err = 1e300;
    for (size_t i = 0; i < n; i++) {
        double mag = fabs(re[i]) + fabs(im[i]);
        if (mag > 1e-6 * (double)n) {
            nonzero++;
            double e = fabs(re[i] - (double)n) + fabs(im[i]);
            if (e < dc_err) dc_err = e;
        }
    }

    int ok = (max_rel < 1e-11) && (pars_err < 1e-12) &&
             (nonzero == 1) && (dc_err < 1e-8 * (double)n);
    if (!ok) g_fail++;

    printf("  %4dx%-4dx%-4d ablk=%-5zu(eff %-5zu) T=%d  rt=%.2e pars=%.2e "
           "dcbins=%d  %s\n",
           N1, N2, N3, a_block, d->a_block, T,
           max_rel, pars_err, nonzero, ok ? "OK" : "**FAIL**");

    AFREE(re); AFREE(im); AFREE(rr); AFREE(ri);
    stride_plan_destroy(plan);
}

int main(void) {
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    struct { int n1, n2, n3; } sizes[] = {
        { 32, 32, 32 },   /* pow2 iso                          */
        { 16, 32, 64 },   /* pow2 aniso, small outer           */
        { 64, 32, 16 },   /* pow2 aniso, small inner           */
        { 60, 20, 12 },   /* smooth non-pow2 (2^2*3*5, ...)    */
        { 61, 16, 16 },   /* prime axis 0 -> override pass A   */
        { 16, 61, 16 },   /* prime axis 1 -> override pass B   */
        { 16, 16, 61 },   /* prime axis 2 -> override rows     */
        {  8,  8,  8 },   /* tiny (tiles/planes < threads)     */
    };
    size_t ablocks[] = { 0, 64 };     /* FLAT and forced-BLOCKED */
    int threads[] = { 1, 2, 4 };

    printf("fft3d roundtrip/parseval/dc matrix\n");
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++)
        for (size_t a = 0; a < sizeof(ablocks)/sizeof(ablocks[0]); a++)
            for (size_t t = 0; t < sizeof(threads)/sizeof(threads[0]); t++)
                run_cell(sizes[s].n1, sizes[s].n2, sizes[s].n3,
                         ablocks[a], threads[t], &reg);

    /* Exhaustive-builder smoke (measures at create; smallest size only). */
    printf("stride_plan_3d exhaustive-builder smoke:\n");
    stride_set_num_threads(1);
    {
        stride_plan_t *p = stride_plan_3d(16, 16, 16, &reg);
        if (!p) { printf("  BUILD FAILED\n"); g_fail++; }
        else {
            size_t n = 16*16*16;
            double *re = (double *)AALLOC(n*sizeof(double));
            double *im = (double *)AALLOC(n*sizeof(double));
            double *rr = (double *)AALLOC(n*sizeof(double));
            double *ri = (double *)AALLOC(n*sizeof(double));
            srand(7);
            for (size_t i = 0; i < n; i++) { rr[i]=re[i]=frand(); ri[i]=im[i]=frand(); }
            stride_execute_fwd(p, re, im);
            stride_execute_bwd(p, re, im);
            double mx = 0.0;
            for (size_t i = 0; i < n; i++) {
                double rel = (fabs(re[i]-n*rr[i]) + fabs(im[i]-n*ri[i]))
                           / (fabs(n*rr[i]) + fabs(n*ri[i]) + 1e-300);
                if (rel > mx) mx = rel;
            }
            int ok = mx < 1e-11;
            if (!ok) g_fail++;
            printf("  16x16x16 exhaustive  rt=%.2e  %s\n", mx, ok?"OK":"**FAIL**");
            AFREE(re); AFREE(im); AFREE(rr); AFREE(ri);
            stride_plan_destroy(p);
        }
    }

    stride_set_num_threads(1);
    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
