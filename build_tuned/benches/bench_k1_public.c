/* bench_k1_public.c — the K=1 engine through the PUBLIC vfft API (§13 step 4).
 * Per N: create (C2C, OOP, howmany=1), gate fwd vs naive + roundtrip for BOTH
 * buffer contracts (split re/im pairs; interleaved z via sim==dim==NULL),
 * then hot-loop timing (best-of-5, cachebust) for split-fwd and il-fwd.
 * Honors VFFT_WISDOM_DIR (kind-3 K1 wisdom) — without it, exercises the
 * heuristic-miss path. Kill-switch check: VFFT_NO_K1=1 must still be correct
 * (old routes).
 * Build: python build.py --src benches/bench_k1_public.c --vfft --jit
 * Usage: bench_k1_public <wisdom_dir|-> [N ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free(j);
}
static void naive_dft(int N, const double *xr, const double *xi, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
            double c = cos(ang), s = sin(ang);
            sr += xr[n] * c - xi[n] * s;
            si += xr[n] * s + xi[n] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    const char *wis = argc > 1 ? argv[1] : "-";
    if (strcmp(wis, "-") != 0) {
        static char env[700];
        snprintf(env, sizeof env, "VFFT_WISDOM_DIR=%s", wis);
        putenv(env);
    }
    int Nd[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
    int nN = argc > 2 ? argc - 2 : 8;
    int fails = 0;

    printf("# K=1 public API (wisdom=%s, VFFT_NO_K1=%s)\n", wis,
           getenv("VFFT_NO_K1") ? "SET" : "unset");
    printf("%-6s %10s %10s   %s\n", "N", "split(ns)", "il(ns)", "gates");

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 2 ? atoi(argv[ni + 2]) : Nd[ni];
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%-6d create=NULL <FAIL>\n", N); fails++; continue; }

        double *xr = malloc((size_t)N * 8), *xi = malloc((size_t)N * 8);
        double *nr = malloc((size_t)N * 8), *nsi = malloc((size_t)N * 8);
        double *dr = malloc((size_t)N * 8), *di = malloc((size_t)N * 8);
        double *rr = malloc((size_t)N * 8), *ri = malloc((size_t)N * 8);
        double *z = malloc((size_t)2 * N * 8), *zo = malloc((size_t)2 * N * 8);
        srand(5 + N);
        for (int n = 0; n < N; n++) {
            xr[n] = (double)rand() / RAND_MAX - 0.5;
            xi[n] = (double)rand() / RAND_MAX - 0.5;
            z[2*n] = xr[n]; z[2*n+1] = xi[n];
        }
        naive_dft(N, xr, xi, nr, nsi);
        double tol = 1e-9 * (double)N, rt_tol = tol * (double)N;

        /* split fwd + roundtrip */
        vfft_execute(h, VFFT_FORWARD, xr, xi, dr, di);
        double ef = 0;
        for (int k = 0; k < N; k++) {
            double c1 = fabs(dr[k] - nr[k]), c2 = fabs(di[k] - nsi[k]);
            if (c1 > ef) ef = c1;
            if (c2 > ef) ef = c2;
        }
        vfft_execute(h, VFFT_BACKWARD, dr, di, rr, ri);
        double ert = 0;
        for (int n = 0; n < N; n++) {
            double c1 = fabs(rr[n] - (double)N * xr[n]);
            double c2 = fabs(ri[n] - (double)N * xi[n]);
            if (c1 > ert) ert = c1;
            if (c2 > ert) ert = c2;
        }
        /* il fwd + roundtrip (sim==dim==NULL contract); may be unsupported
         * for some N (no IL route) — detect by output unchanged */
        double eilf = -1, eilrt = -1;
        memset(zo, 0, (size_t)2 * N * 8);
        vfft_execute(h, VFFT_FORWARD, z, NULL, zo, NULL);
        {
            int touched = 0;
            for (int k = 0; k < 2 * N && !touched; k++) if (zo[k] != 0.0) touched = 1;
            if (touched) {
                eilf = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(zo[2*k] - nr[k]), c2 = fabs(zo[2*k+1] - nsi[k]);
                    if (c1 > eilf) eilf = c1;
                    if (c2 > eilf) eilf = c2;
                }
                vfft_execute(h, VFFT_BACKWARD, zo, NULL, zo, NULL);
                eilrt = 0;
                for (int n = 0; n < N; n++) {
                    double c1 = fabs(zo[2*n] - (double)N * xr[n]);
                    double c2 = fabs(zo[2*n+1] - (double)N * xi[n]);
                    if (c1 > eilrt) eilrt = c1;
                    if (c2 > eilrt) eilrt = c2;
                }
            }
        }
        int bad = (ef > tol || ert > rt_tol ||
                   (eilf >= 0 && eilf > tol) || (eilrt >= 0 && eilrt > rt_tol));
        if (bad) fails++;

        /* timing */
        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;
        double bsp = 1e18, bil = 1e18;
        for (int t = 0; t < 5; t++) {
            if (t) cachebust();
            for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, xr, xi, dr, di);
            double t0 = now_ms();
            for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, xr, xi, dr, di);
            double ns = (now_ms() - t0) * 1e6 / reps;
            if (ns < bsp) bsp = ns;
            if (eilf >= 0) {
                for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, z, NULL, zo, NULL);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, z, NULL, zo, NULL);
                ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < bil) bil = ns;
            }
        }
        printf("%-6d %10.0f %10.0f   fwd=%.1e rt=%.1e ilf=%.1e ilrt=%.1e%s\n",
               N, bsp, eilf >= 0 ? bil : -1.0, ef, ert, eilf, eilrt,
               bad ? "  <FAIL>" : "");

        free(xr); free(xi); free(nr); free(nsi);
        free(dr); free(di); free(rr); free(ri); free(z); free(zo);
        vfft_destroy(h);
    }
    printf("\n%s (%d fail)\n", fails ? "FAILURES" : "K1 PUBLIC API: ALL GREEN", fails);
    return fails ? 1 : 0;
}
