/* bench_oddk_tail.c — arbitrary-K tail vs MKL, ONE CELL per process (canonical
 * MKLBench methodology, mirrors bench_1d_vs_mkl measure_ab):
 *   - best-of-5 min per engine; cachebust + cool_ms idle BETWEEN engines;
 *   - order-flip (argv) so the driver alternates and averages out order bias;
 *   - one cell per PROCESS (cross-cell thermal/cache carryover can't be cachebusted).
 * Plan: N=1024 [4,4,4,4,4] T1S. Whether the tail is scalar or masked is decided
 * by the codelet build (-DVFFT_TAIL_MASKED on r4_n1_fwd / r4_t1s_dit_fwd).
 *
 * Usage: bench_oddk_tail <K> <flip 0|1> <cool_ms>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include <mkl_dfti.h>
#include <mkl_service.h>

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
    {
        fprintf(stderr, "alloc\n");
        exit(1);
    }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }
static void pace(int ms)
{
    if (ms <= 0)
        return;
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = ad(s);
    for (size_t i = 0; i < s; i++)
        j[i] = (double)i;
    volatile double a = 0;
    for (size_t i = 0; i < s; i++)
        a += j[i];
    (void)a;
    afree(j);
}

static double best_proto(stride_plan_t *plan, double *re, double *im, size_t K, int reps)
{
    for (int w = 0; w < 10; w++)
        vfft_proto_execute_fwd(plan, re, im, K);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static double best_mkl(DFTI_DESCRIPTOR_HANDLE d, double *re, double *im, int reps)
{
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, re, im);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    DftiCommitDescriptor(d);
    return d;
}
static double correctness(int N, int *f, int *v, int nf, size_t K, vfft_proto_registry_t *reg)
{
    size_t Kp = (K + 7) & ~(size_t)7;
    stride_plan_t *pk = vfft_proto_plan_create(N, K, f, v, nf, reg), *pp = vfft_proto_plan_create(N, Kp, f, v, nf, reg);
    if (!pk || !pp)
        return -1.0;
    double *rk = ad((size_t)N * K), *ik = ad((size_t)N * K), *rp = ad((size_t)N * Kp), *ip = ad((size_t)N * Kp);
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++)
        {
            double a = (double)rand() / RAND_MAX - 0.5, b = (double)rand() / RAND_MAX - 0.5;
            rk[e * K + l] = a;
            ik[e * K + l] = b;
            rp[e * Kp + l] = a;
            ip[e * Kp + l] = b;
        }
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = K; l < Kp; l++)
        {
            rp[e * Kp + l] = 0;
            ip[e * Kp + l] = 0;
        }
    vfft_proto_execute_fwd(pk, rk, ik, K);
    vfft_proto_execute_fwd(pp, rp, ip, Kp);
    double md = 0;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++)
        {
            double dr = fabs(rk[e * K + l] - rp[e * Kp + l]), di = fabs(ik[e * K + l] - ip[e * Kp + l]);
            double d = dr > di ? dr : di;
            if (d > md)
                md = d;
        }
    afree(rk);
    afree(ik);
    afree(rp);
    afree(ip);
    return md;
}

int main(int argc, char **argv)
{
    size_t K = (argc > 1) ? (size_t)atoll(argv[1]) : 31;
    int flip = (argc > 2) ? atoi(argv[2]) : 0;
    int cool_ms = (argc > 3) ? atoi(argv[3]) : 80;
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    int N = 1024, nf = 5, factors[5] = {4, 4, 4, 4, 4}, variants[5] = {0, 2, 2, 2, 2};

    double md = correctness(N, factors, variants, nf, K, &reg);
    stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, variants, nf, &reg);
    if (!plan)
    {
        printf("K=%-4zu f%d  plan FAILED\n", K, flip);
        return 1;
    }
    size_t total = (size_t)N * K;
    int reps = (int)(50000000ull / total);
    if (reps < 200)
        reps = 200;
    double *re = ad(total), *im = ad(total), *rm = ad(total), *imk = ad(total);
    srand(42 + (int)K);
    for (size_t i = 0; i < total; i++)
    {
        double a = (double)rand() / RAND_MAX - 0.5, b = (double)rand() / RAND_MAX - 0.5;
        re[i] = a;
        im[i] = b;
        rm[i] = a;
        imk[i] = b;
    }
    DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);

    double v, m;
    if (flip)
    {
        m = best_mkl(d, rm, imk, reps);
        cachebust();
        pace(cool_ms);
        v = best_proto(plan, re, im, K, reps);
    }
    else
    {
        v = best_proto(plan, re, im, K, reps);
        cachebust();
        pace(cool_ms);
        m = best_mkl(d, rm, imk, reps);
    }
    printf("K=%-4zu rem=%zu flip=%d  corr=%.1e  vfft=%9.0f  mkl=%9.0f  ratio=%.3f\n",
           K, K % 4, flip, md, v, m, m / v);
    DftiFreeDescriptor(&d);
    afree(re);
    afree(im);
    afree(rm);
    afree(imk);
    return 0;
}
