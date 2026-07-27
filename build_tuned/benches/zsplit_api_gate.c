/* zsplit_api_gate.c — PUBLIC-API gate + front-door bench for the K=1
 * SCRAMBLED block-split cascade route (src/core/oop/zsplit.h wired in
 * vfft.c). Everything through vfft.h ONLY:
 *   gate 1: fwd spectrum == naive DFT under the cascade's drev permutation
 *           (also proves the ROUTE served — the classic path's scramble
 *           would fail this compare);
 *   gate 2: roundtrip bwd(fwd(x)) == N*x through the API;
 *   bench : paced fwd/bwd vs MKL (in the same process).
 *
 * Build: python build.py --src benches/zsplit_api_gate.c --mkl
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include <mkl_dfti.h>
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

/* the route's chains (must match vfft_zsplit_default_chain) + mixed drev */
static int def_chain(int N, int *c)
{
    switch (N) {
    case 2048:  c[0]=4; c[1]=8; c[2]=8; c[3]=8; return 4;
    case 4096:  c[0]=4; c[1]=4; c[2]=4; c[3]=8; c[4]=8; return 5;
    case 8192:  c[0]=4; c[1]=4; c[2]=8; c[3]=8; c[4]=8; return 5;
    case 16384: c[0]=4; c[1]=8; c[2]=8; c[3]=8; c[4]=8; return 5;
    default: return 0;
    }
}
static long drev_full(long x, const int *r, int nf)
{ long v = 0; for (int i = nf - 1; i >= 0; i--) { v = v * r[i] + (x % r[i]); x /= r[i]; } return v; }

/* ROUTE-AGNOSTIC comb check (route axis, 2026-07-27): the kind-4 wisdom may
 * serve LEGACY zsplit or ZTURN-S, whose scrambled combs DIFFER (both legal —
 * the public SCRAMBLED contract promises a fixed route-defined permutation,
 * not WHICH one; bwd inverts the same route's comb). So the fixed drev
 * compare only applies to a route-pinned legacy plan; the wisdom-routed plan
 * is gated as: fwd output must be a BIJECTIVE relabeling of the reference
 * spectrum (extracted by nearest-match), plus determinism + roundtrip. */
static const double *g_sort_key;
static int idx_cmp(const void *a, const void *b)
{
    double x = g_sort_key[*(const int *)a], y = g_sort_key[*(const int *)b];
    return (x > y) - (x < y);
}
static int spec_bijection(const double *S, const double *Rr, const double *Ri,
                          int N, double mag, double *errout)
{
    int *ord = (int *)malloc((size_t)N * sizeof(int));
    unsigned char *used = (unsigned char *)calloc((size_t)N, 1);
    if (!ord || !used) { free(ord); free(used); return 0; }
    for (int i = 0; i < N; i++) ord[i] = i;
    g_sort_key = Rr;
    qsort(ord, (size_t)N, sizeof(int), idx_cmp);
    const double win = 1e-9 * mag; /* candidate window on re; (re,im) decides */
    double err = 0;
    int ok = 1;
    for (long idx = 0; idx < N && ok; idx++) {
        double sr = S[2 * idx], si = S[2 * idx + 1];
        /* first sorted position with Rr >= sr - win */
        int lo = 0, hi = N;
        while (lo < hi) { int mid = (lo + hi) >> 1;
            if (Rr[ord[mid]] < sr - win) lo = mid + 1; else hi = mid; }
        int best = -1; double bd = 0;
        for (int j = lo; j < N && Rr[ord[j]] <= sr + win; j++) {
            int m = ord[j];
            if (used[m]) continue;
            double d = fabs(sr - Rr[m]) + fabs(si - Ri[m]);
            if (best < 0 || d < bd) { best = m; bd = d; }
        }
        if (best < 0 || bd / mag >= 1e-11) { ok = 0; break; }
        used[best] = 1;
        if (bd > err) err = bd;
    }
    *errout = err / mag;
    free(ord); free(used);
    return ok;
}

static int run_cell(int N)
{
    int ch[6], nf = def_chain(N, ch);
    if (!nf) return 1;

    vfft_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; /* the cascade serves the committed z contract */
    cfg.nthreads = 1;
    vfft_plan h = vfft_create(&cfg);
    if (!h) { printf("N=%d: create FAILED\n", N); return 1; }

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *S  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *xr = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    srand(N);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    for (int m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)(((long)n * m) % N) / (double)N;
            double cc = cos(a), ss = sin(a);
            sr += z0[2 * n] * cc - z0[2 * n + 1] * ss;
            si += z0[2 * n] * ss + z0[2 * n + 1] * cc;
        }
        Rr[m] = sr; Ri[m] = si;
    }
    double mag = 0;
    for (int m = 0; m < N; m++) {
        double g = fabs(Rr[m]) + fabs(Ri[m]);
        if (g > mag) mag = g;
    }

    int rc = 0;
    /* gate 1: fwd drev compare on a route-PINNED legacy plan (the fixed comb
     * below is the LEGACY zsplit one; VFFT_FORCE_ZROUTE is the documented
     * gate/test hook — the wisdom-routed plan is gated route-agnostically
     * as gate 1b). */
    {
        _putenv("VFFT_FORCE_ZROUTE=legacy");
        vfft_plan hL = vfft_create(&cfg);
        _putenv("VFFT_FORCE_ZROUTE=");
        if (!hL) { printf("N=%d: legacy-pinned create FAILED\n", N); rc = 1; }
        else {
            vfft_execute(hL, VFFT_FORWARD, z0, NULL, S, NULL);
            long NR = (long)N / 8;
            double err = 0;
            for (long idx = 0; idx < N; idx++) {
                long l = idx / NR, g = idx % NR;
                long m = drev_full(g * 8 + l, ch, nf);
                double d = fabs(S[2 * idx] - Rr[m]) + fabs(S[2 * idx + 1] - Ri[m]);
                if (d > err) err = d;
            }
            printf("N=%-6d GATE api-fwd(drev) relerr=%.3e %s   (route-pinned legacy)\n",
                   N, err / mag, (err / mag < 1e-11) ? "PASS" : "FAIL");
            if (err / mag >= 1e-11) rc = 1;
            vfft_destroy(hL);
        }
    }
    /* gate 1b: the WISDOM-ROUTED plan's fwd comb — bijective relabeling of
     * the reference spectrum + deterministic (two executes bit-identical). */
    vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
    {
        double berr = 0;
        int bij = spec_bijection(S, Rr, Ri, N, mag, &berr);
        vfft_execute(h, VFFT_FORWARD, z0, NULL, xr, NULL);
        int det = memcmp(S, xr, (size_t)2 * N * 8) == 0;
        printf("N=%-6d GATE api-fwd(comb) bijection=%s relerr=%.3e det=%s %s\n",
               N, bij ? "yes" : "NO", berr, det ? "yes" : "NO",
               (bij && det) ? "PASS" : "FAIL");
        if (!bij || !det) rc = 1;
    }
    /* gate 2: roundtrip through the API (matched-permutation contract) */
    vfft_execute(h, VFFT_BACKWARD, S, NULL, xr, NULL);
    {
        double err = 0, m2 = 0;
        for (long i = 0; i < 2 * (long)N; i++) {
            double d = fabs(xr[i] - (double)N * z0[i]);
            if (d > err) err = d;
            double g = fabs((double)N * z0[i]);
            if (g > m2) m2 = g;
        }
        printf("N=%-6d GATE api-rtrip     relerr=%.3e %s\n", N, err / m2,
               (err / m2 < 1e-11) ? "PASS" : "FAIL");
        if (err / m2 >= 1e-11) rc = 1;
    }

    if (!rc && !getenv("VFFT_GATE_ONLY")) { /* front-door bench (paced): API fwd/bwd vs MKL.
                                             * VFFT_GATE_ONLY=1 = gates only, no timing. */
        DFTI_DESCRIPTOR_HANDLE mh = NULL;
        DftiCreateDescriptor(&mh, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiCommitDescriptor(mh);
        int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
        double bf = 1e18, bb = 1e18, mf = 1e18, mb = 1e18;
        for (int t = 0; t < 5; t++) {
            cachebust(); Sleep(150);
            double t0;
            for (int w = 0; w < 6; w++) vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
            t0 = now_ms();
            for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
            { double ns = (now_ms() - t0) * 1e6 / reps; if (ns < bf) bf = ns; }
            for (int w = 0; w < 6; w++) vfft_execute(h, VFFT_BACKWARD, S, NULL, xr, NULL);
            t0 = now_ms();
            for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_BACKWARD, S, NULL, xr, NULL);
            { double ns = (now_ms() - t0) * 1e6 / reps; if (ns < bb) bb = ns; }
            for (int w = 0; w < 6; w++) DftiComputeForward(mh, S);
            t0 = now_ms();
            for (int i = 0; i < reps; i++) DftiComputeForward(mh, S);
            { double ns = (now_ms() - t0) * 1e6 / reps; if (ns < mf) mf = ns; }
            for (int w = 0; w < 6; w++) DftiComputeBackward(mh, S);
            t0 = now_ms();
            for (int i = 0; i < reps; i++) DftiComputeBackward(mh, S);
            { double ns = (now_ms() - t0) * 1e6 / reps; if (ns < mb) mb = ns; }
        }
        printf("N=%-6d API fwd %9.1f ns (vsMKL %.2f)   bwd %9.1f ns (vsMKLbwd %.2f)\n",
               N, bf, mf / bf, bb, mb / bb);
        DftiFreeDescriptor(&mh);
    }

    vfft_destroy(h);
    _mm_free(z0); _mm_free(S); _mm_free(xr); _mm_free(Rr); _mm_free(Ri);
    return rc;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    int rc = 0;
    rc |= run_cell(2048);
    rc |= run_cell(4096);
    rc |= run_cell(8192);
    rc |= run_cell(16384);
    printf(rc ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return rc;
}
