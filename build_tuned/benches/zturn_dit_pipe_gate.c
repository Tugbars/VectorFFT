/* zturn_dit_pipe_gate.c — Phase C gate 2: the wired DIT-forward PIPELINE.
 *
 * vfft_zturn2_execute_dit_fwd (dtsn ingest -> msg fwd mids in bwd stage
 * order -> dtt finisher) gated per chain (mixed-radix mandatory — the rho
 * involution masks table mix-ups on uniform chains) x both ingest radices:
 *
 *   1. REF:      DIT fwd (natord plan, natural x in) == naive O(N^2) DFT
 *                elementwise IN ORDER, tolerance. The independent reference.
 *   2. CROSS:    DIT fwd vs DIF-natural fwd (stfn path) — SAME operator,
 *                different summation order, so TOLERANCE, never memcmp.
 *   3. IN-PLACE: execute_dit_fwd(buf, buf) == the OOP output, memcmp EXACT
 *                (zin is fully consumed by the ingest before dtt writes).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_dit_pipe_gate.c
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "zturn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *az(size_t doubles)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(doubles * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, doubles * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void naive_dft(const double *x, double *X, long N)
{
    double *wr = (double *)malloc(sizeof(double) * (size_t)N);
    double *wi = (double *)malloc(sizeof(double) * (size_t)N);
    for (long j = 0; j < N; j++)
    {
        const double a = -2.0 * M_PI * (double)j / (double)N;
        wr[j] = cos(a);
        wi[j] = sin(a);
    }
    for (long k = 0; k < N; k++)
    {
        double sr = 0.0, si = 0.0;
        long idx = 0;
        for (long j = 0; j < N; j++)
        {
            const double xr = x[2 * j], xi = x[2 * j + 1];
            sr += xr * wr[idx] - xi * wi[idx];
            si += xr * wi[idx] + xi * wr[idx];
            idx += k;
            if (idx >= N) idx -= N;
        }
        X[2 * k] = sr;
        X[2 * k + 1] = si;
    }
    free(wr);
    free(wi);
}

typedef struct { int N, nf, chain[8]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4} },
    { 4096,  6, {4,4,4,4,4,4} },
    { 8192,  6, {4,8,4,4,4,4} },
    { 16384, 7, {4,4,4,4,4,4,4} },
    { 32768, 7, {4,8,4,4,4,4,4} },
    { 16384, 6, {4,8,4,4,4,8} },   /* r8 ingest form */
};

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    int fails = 0;
    printf("\n=== Phase C gate 2: DIT-forward pipeline ===\n");
    printf("%-7s %-16s %-3s | %-10s %-10s %-8s\n",
           "N", "chain", "Rt", "vs naive", "vs DIF-nat", "in-place");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *pd =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pn =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!pd || !pn || !vfft_zturn2_set_natord(pd, 1)
            || !vfft_zturn2_set_natord(pn, 1))
        { printf("%-7d %-16s REFUSED\n", N, cs); fails++; continue; }

        srand(9091 + N + Rt);
        double *x  = az(2 * (size_t)N), *X = az(2 * (size_t)N);
        double *yd = az(2 * (size_t)N), *yf = az(2 * (size_t)N);
        double *ip = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;

        vfft_zturn2_execute_dit_fwd(pd, x, yd);
        naive_dft(x, X, N);
        double xm = 0.0, e1 = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            const double m = fabs(X[i]);
            if (m > xm) xm = m;
            const double d = fabs(yd[i] - X[i]);
            if (d > e1) e1 = d;
        }
        const double r1 = e1 / xm;
        const int a1 = r1 < 1e-9;

        vfft_zturn2_execute_fwd(pn, x, yf);      /* DIF-natural (stfn) */
        double e2 = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            const double d = fabs(yd[i] - yf[i]);
            if (d > e2) e2 = d;
        }
        const double r2 = e2 / xm;
        const int a2 = r2 < 1e-9;

        memcpy(ip, x, 2 * (size_t)N * sizeof(double));
        vfft_zturn2_execute_dit_fwd(pd, ip, ip);
        const int a3 = memcmp(ip, yd, 2 * (size_t)N * sizeof(double)) == 0;

        const int ok = a1 && a2 && a3;
        if (!ok) fails++;
        printf("%-7d %-16s %-3d | %.1e   %.1e   %-8s%s\n",
               N, cs, Rt, r1, r2,
               a3 ? "EXACT" : "DIFF!", ok ? "" : "   *** FAIL ***");

        fz(x); fz(X); fz(yd); fz(yf); fz(ip);
        vfft_zturn2_destroy(pd);
        vfft_zturn2_destroy(pn);
    }

    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
