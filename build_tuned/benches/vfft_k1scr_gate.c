/* vfft_k1scr_gate.c — Phase A gate: explicit-SCRAMBLED sub-2048 K=1
 * interleaved OOP now routes to the native K=1 IL engines (identity
 * permutation — contract-legal, il_coverage_plan.md Phase A).
 *
 * ARMS per sub-2048 cell (128/256/512/1024), OOP INTERLEAVED, scratch W:
 *   1. IDENTITY == ROUTE PROOF: fwd(SCRAMBLED handle) memcmp-EXACT ==
 *      fwd(NATURAL handle) on the same input. The old route (convert
 *      fallback -> split MODEB) emits a genuinely permuted comb, so a
 *      routing regression CANNOT pass this arm.
 *   2. REFERENCE: the NATURAL handle vs naive DFT IN ORDER (tolerance) —
 *      anchors arm 1 to ground truth (🔴 roundtrip cannot gate ordering).
 *   3. MATCHED ROUNDTRIP on the SCRAMBLED handle: bwd(fwd(x)) == N·x.
 *   4. SPEED: scrambled/natural paced ratio ≈ 1.0 (same engine, same plan;
 *      the convert path would be visibly slower). Informational + <1.25
 *      green line (thermal).
 *
 * ≥2048 regression (A3): a 4096 SCRAMBLED handle must NOT be identity-
 * served (its cascade comb is a real permutation) and must roundtrip.
 * (Structural note: execute prefers an attached cascade over k1_on, so
 * the only ≥2048 failure mode of the A1 gate change is dead weight — the
 * non-identity check plus the standing k1zip gates cover behavior.)
 *
 * Run:   vfft_k1scr_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/vfft_k1scr_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ns(void)
{
#ifdef _WIN32
    static double f = 0.0;
    LARGE_INTEGER t;
    if (f == 0.0) { LARGE_INTEGER q; QueryPerformanceFrequency(&q);
                    f = 1e9 / (double)q.QuadPart; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * f;
#else
    return 0.0;
#endif
}
static void pace(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#endif
}
static double *az(size_t n)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(2 * n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
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
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static vfft_plan mk(vfft_wisdom *W, int N, int scrambled)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = scrambled ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    return vfft_create(&cfg);
}

static void naive_dft(const double *x, double *X, long N)
{
    for (long k = 0; k < N; k++)
    {
        double sr = 0, si = 0;
        for (long j = 0; j < N; j++)
        {
            const double a = -2.0 * M_PI * (double)((j * k) % N) / (double)N;
            const double c = cos(a), s = sin(a);
            sr += x[2 * j] * c - x[2 * j + 1] * s;
            si += x[2 * j] * s + x[2 * j + 1] * c;
        }
        X[2 * k] = sr;
        X[2 * k + 1] = si;
    }
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    printf("\n=== Phase A: explicit-SCRAMBLED sub-2048 IL -> native K=1 engine ===\n");
    printf("%-7s | %-9s %-10s %-10s %-9s\n",
           "N", "scr==nat", "nat vs ref", "scr rt", "scr/nat");
    int fails = 0;

    static const int NS[] = { 128, 256, 512, 1024 };
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        vfft_plan hs = mk(W, N, 1), hn = mk(W, N, 0);
        if (!hs || !hn)
        {
            printf("%-7d create FAILED\n", N);
            fails++;
            continue;
        }
        srand(77 + N);
        double *x = az((size_t)N), *X = az((size_t)N);
        double *ys = az((size_t)N), *yn = az((size_t)N), *rt = az((size_t)N);
        for (long j = 0; j < 2L * N; j++)
            x[j] = (double)rand() / RAND_MAX - 0.5;

        /* arm 1: identity == route proof */
        vfft_execute(hs, VFFT_FORWARD, x, NULL, ys, NULL);
        vfft_execute(hn, VFFT_FORWARD, x, NULL, yn, NULL);
        const int eq = memcmp(ys, yn, 2 * (size_t)N * sizeof(double)) == 0;

        /* arm 2: natural vs naive IN ORDER */
        naive_dft(x, X, N);
        double m = 0, e = 0;
        for (long j = 0; j < 2L * N; j++)
        {
            if (fabs(X[j]) > m) m = fabs(X[j]);
            if (fabs(yn[j] - X[j]) > e) e = fabs(yn[j] - X[j]);
        }
        const double ref = e / m;

        /* arm 3: matched roundtrip on the scrambled handle */
        vfft_execute(hs, VFFT_BACKWARD, ys, NULL, rt, NULL);
        double m3 = 0, e3 = 0;
        const double inv = 1.0 / N;
        for (long j = 0; j < 2L * N; j++)
        {
            if (fabs(x[j]) > m3) m3 = fabs(x[j]);
            if (fabs(rt[j] * inv - x[j]) > e3) e3 = fabs(rt[j] * inv - x[j]);
        }
        const double rte = e3 / m3;

        /* arm 4: speed ratio, 9 paced rounds, medians */
        double ss[9], sn[9];
        const int reps = 2000;
        for (int r = 0; r < 9; r++)
        {
            double t0;
            if (r & 1)
            {
                t0 = now_ns();
                for (int k = 0; k < reps; k++)
                    vfft_execute(hn, VFFT_FORWARD, x, NULL, yn, NULL);
                sn[r] = (now_ns() - t0) / reps;
                t0 = now_ns();
                for (int k = 0; k < reps; k++)
                    vfft_execute(hs, VFFT_FORWARD, x, NULL, ys, NULL);
                ss[r] = (now_ns() - t0) / reps;
            }
            else
            {
                t0 = now_ns();
                for (int k = 0; k < reps; k++)
                    vfft_execute(hs, VFFT_FORWARD, x, NULL, ys, NULL);
                ss[r] = (now_ns() - t0) / reps;
                t0 = now_ns();
                for (int k = 0; k < reps; k++)
                    vfft_execute(hn, VFFT_FORWARD, x, NULL, yn, NULL);
                sn[r] = (now_ns() - t0) / reps;
            }
            pace(100);
        }
        qsort(ss, 9, sizeof(double), dcmp);
        qsort(sn, 9, sizeof(double), dcmp);
        const double ratio = ss[4] / sn[4];

        const int ok = eq && ref < 1e-9 && rte < 1e-9 && ratio < 1.25;
        if (!ok) fails++;
        printf("%-7d | %-9s %.2e   %.2e   %5.3fx%s\n",
               N, eq ? "EXACT" : "DIFF!", ref, rte, ratio,
               ok ? "" : "   *** FAIL ***");

        fz(x); fz(X); fz(ys); fz(yn); fz(rt);
        vfft_destroy(hs);
        vfft_destroy(hn);
    }

    /* A3: ≥2048 SCRAMBLED must stay on the cascade (non-identity comb). */
    {
        const int N = 4096;
        vfft_plan hs = mk(W, N, 1), hn = mk(W, N, 0);
        if (!hs || !hn)
        {
            printf("4096    create FAILED (A3)\n");
            fails++;
        }
        else
        {
            srand(77 + N);
            double *x = az((size_t)N), *ys = az((size_t)N),
                   *yn = az((size_t)N), *rt = az((size_t)N);
            for (long j = 0; j < 2L * N; j++)
                x[j] = (double)rand() / RAND_MAX - 0.5;
            vfft_execute(hs, VFFT_FORWARD, x, NULL, ys, NULL);
            vfft_execute(hn, VFFT_FORWARD, x, NULL, yn, NULL);
            const int ident =
                memcmp(ys, yn, 2 * (size_t)N * sizeof(double)) == 0;
            vfft_execute(hs, VFFT_BACKWARD, ys, NULL, rt, NULL);
            double m = 0, e = 0;
            const double inv = 1.0 / N;
            for (long j = 0; j < 2L * N; j++)
            {
                if (fabs(x[j]) > m) m = fabs(x[j]);
                if (fabs(rt[j] * inv - x[j]) > e) e = fabs(rt[j] * inv - x[j]);
            }
            const int ok = !ident && (e / m) < 1e-9;
            if (!ok) fails++;
            printf("%-7d | %-9s rt=%.2e  (A3: cascade comb must be a REAL "
                   "permutation)%s\n",
                   N, ident ? "IDENT(!)" : "permuted", e / m,
                   ok ? "" : "   *** FAIL ***");
            fz(x); fz(ys); fz(yn); fz(rt);
        }
        if (hs) vfft_destroy(hs);
        if (hn) vfft_destroy(hn);
    }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
