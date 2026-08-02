/* zturn_natscatter_probe.c — P0c: what does the NATURAL-order terminator's
 * permuted access actually cost? The GO/NO-GO number for Phase B.
 *
 * A natural-writing DIF terminator must apply rho^{-1} (P0b: conv0 MSF digit
 * reversal over the middle radices, CONFIRMED) at 64 B-record granularity, on
 * one side or the other:
 *   arm A: contiguous reads + contiguous stores       (today's terminator)
 *   arm B: contiguous reads + PERMUTED 64 B stores    (store-side natural)
 *   arm C: PERMUTED 64 B reads + contiguous stores    (load-side natural —
 *          iterate columns in rho order; MKL's lesson says loads tolerate
 *          striding, stores do not)
 *   arm D: memcpy(N*16) anchor
 *
 * Traffic model = the r4 terminator's real shape: per column k' in [0, N/16),
 * read one 64 B record from each of 4 section streams, combine (radix-2-ish,
 * enough math that nothing collapses), store one 64 B record to each of 4
 * lane streams. A and B/C differ ONLY in one side's index table — exactly how
 * a table-driven natural kernel would realize it.
 *
 * WHY THIS DECIDES: the incumbent natural mechanism (PURE-cycle) costs an
 * extra full pass over the data (+16–36% measured on real cells). The natural
 * terminator instead perturbs ONE existing pass. If Δ(min(B,C)) is small
 * against the full-FFT time, the terminator wins and Phase B proceeds; if the
 * permuted side costs anything like the old 0.40x scattered-store number,
 * DIF-natural is dead and the answer is the DIT cascade.
 *
 * Protocol: core 2 (mask 0x4, P-core), HIGH priority, 17 rounds, arms rotated
 * per round, 200 ms pace between arms, medians + p10-p90, warm buffers,
 * same-run comparisons only. Thermally noisy machine: nothing under ~3% is a
 * result.
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_natscatter_probe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#ifdef _WIN32
#include <windows.h>
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
static void pace(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#endif
}
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
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* conv0 (MSF) digit reversal — the P0b-confirmed convention. */
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

/* one terminator-shaped pass: 4 section reads -> combine -> 4 lane stores.
 * rd[k]/wr[k] are 64 B-record indices (units of 8 doubles). */
static void term_pass(const double *plane, double *out, long M, long secd,
                      long land, const long *rd, const long *wr)
{
    for (long i = 0; i < M; i++)
    {
        const double *s0 = plane + rd[i] * 8;
        const __m256d a0 = _mm256_load_pd(s0);
        const __m256d a1 = _mm256_load_pd(s0 + 4);
        const double *s1 = plane + secd + rd[i] * 8;
        const __m256d b0 = _mm256_load_pd(s1);
        const __m256d b1 = _mm256_load_pd(s1 + 4);
        const double *s2 = plane + 2 * secd + rd[i] * 8;
        const __m256d c0 = _mm256_load_pd(s2);
        const __m256d c1 = _mm256_load_pd(s2 + 4);
        const double *s3 = plane + 3 * secd + rd[i] * 8;
        const __m256d d0 = _mm256_load_pd(s3);
        const __m256d d1 = _mm256_load_pd(s3 + 4);

        double *o0 = out + wr[i] * 8;
        _mm256_store_pd(o0,     _mm256_add_pd(a0, b0));
        _mm256_store_pd(o0 + 4, _mm256_add_pd(a1, b1));
        double *o1 = out + land + wr[i] * 8;
        _mm256_store_pd(o1,     _mm256_sub_pd(a0, b0));
        _mm256_store_pd(o1 + 4, _mm256_sub_pd(a1, b1));
        double *o2 = out + 2 * land + wr[i] * 8;
        _mm256_store_pd(o2,     _mm256_add_pd(c0, d0));
        _mm256_store_pd(o2 + 4, _mm256_add_pd(c1, d1));
        double *o3 = out + 3 * land + wr[i] * 8;
        _mm256_store_pd(o3,     _mm256_sub_pd(c0, d0));
        _mm256_store_pd(o3 + 4, _mm256_sub_pd(c1, d1));
    }
}

typedef struct { int N, nf, chain[8]; double fft_ns; } cell_t;
/* fft_ns = this cell's banked FORWARD time (~half the calibrator's joint ns;
 * bench-measured fwd where available) — ONLY used to express deltas as a % of
 * a real transform, so the verdict line is readable. Rough by design. */
static const cell_t CELLS[] = {
    { 4096,  6, {4,4,4,4,4,4},     3940.0  },
    { 16384, 7, {4,4,4,4,4,4,4},   18042.0 },
    { 32768, 7, {4,8,4,4,4,4,4},   44662.0 },
};

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    printf("\n=== P0c: natural-terminator permuted-access cost (GO/NO-GO) ===\n");
    printf("rounds=17, paced 200ms, core 2, medians; B=store-side perm, "
           "C=load-side perm\n\n");
    printf("%-7s %10s %10s %10s %10s | %7s %7s | %s\n",
           "N", "A ns", "B ns", "C ns", "memcpy", "B/A", "C/A",
           "min-delta as % of one fwd FFT");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N;
        const long M = N / 16;               /* r4 terminator: N/16 columns  */
        const long secd = (long)N / 4 * 2;   /* doubles per section          */
        const long land = (long)N / 4 * 2;   /* doubles per output lane      */

        double *plane = az(2 * (size_t)N);
        double *out   = az(2 * (size_t)N);
        double *mc    = az(2 * (size_t)N);
        long *idx  = (long *)malloc(sizeof(long) * (size_t)M);
        long *prm  = (long *)malloc(sizeof(long) * (size_t)M);
        for (long i = 0; i < 2L * N; i++) plane[i] = (double)(i & 1023) * 0.5;
        memset(out, 0, 2 * (size_t)N * sizeof(double));
        for (long i = 0; i < M; i++) idx[i] = i;
        for (long i = 0; i < M; i++)
            prm[rho0(i, c->chain + 1, c->nf - 2)] = i;   /* rho^{-1} table    */

        /* rep count for ~300us blocks */
        int reps = (int)(300000.0 / ((double)N * 0.06));
        if (reps < 4) reps = 4;

        enum { NARM = 4 };
        double smp[NARM][17];
        for (int r = 0; r < 17; r++)
        {
            for (int k = 0; k < NARM; k++)
            {
                const int arm = (k + r) % NARM;
                /* warm */
                term_pass(plane, out, M, secd, land, idx, idx);
                double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    switch (arm)
                    {
                    case 0: term_pass(plane, out, M, secd, land, idx, idx); break;
                    case 1: term_pass(plane, out, M, secd, land, idx, prm); break;
                    case 2: term_pass(plane, out, M, secd, land, prm, idx); break;
                    case 3: memcpy(mc, plane, 2 * (size_t)N * sizeof(double)); break;
                    }
                smp[arm][r] = (now_ns() - t0) / reps;
                pace(200);
            }
        }
        double med[NARM], lo[NARM], hi[NARM];
        for (int a = 0; a < NARM; a++)
        {
            qsort(smp[a], 17, sizeof(double), dcmp);
            med[a] = smp[a][8];
            lo[a] = smp[a][1]; hi[a] = smp[a][15];
        }
        const double dB = med[1] - med[0], dC = med[2] - med[0];
        const double dmin = dB < dC ? dB : dC;
        printf("%-7d %10.0f %10.0f %10.0f %10.0f | %7.3f %7.3f | "
               "+%.0f ns = %.1f%% of fwd (PURE-cycle incumbent: +16..36%%)\n",
               N, med[0], med[1], med[2], med[3],
               med[1] / med[0], med[2] / med[0],
               dmin, 100.0 * dmin / c->fft_ns);
        printf("        spreads p10-p90: A %.0f-%.0f  B %.0f-%.0f  C %.0f-%.0f\n",
               lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);

        free(idx); free(prm);
#ifdef _WIN32
        _aligned_free(plane); _aligned_free(out); _aligned_free(mc);
#endif
    }
    printf("\nGO if the min-delta is far below the incumbent's +16-36%%; "
           "NO-GO -> the answer is the DIT cascade, not a faster scatter.\n");
    return 0;
}
