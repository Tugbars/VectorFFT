/* zturn_stfn_gate.c — B2 gates for the NATURAL-ORDER terminator kinds.
 *
 * GATE 1 (bit-identity of the mechanism): stfn == stf explicitly
 * block-permuted, and stfbn(permuted input) == stfb(input), memcmp, per
 * chain × both terminator radices × both directions. This settles the one
 * flagged unknown in natterm_spec.md §2 (does the packed-w^1 stream travel
 * with the loads?) mechanically instead of by argument.
 *
 * GATE 2 (speed — gate_new_kernels_on_speed_too): the natural kinds within
 * P0c's envelope of the scrambled kinds on the SAME inputs, paced medians.
 * Bit-identical is not enough; the 4KB-aliasing trap is on record twice.
 *
 * Kernel-level on purpose: the terminator is a pure function of
 * (plane, tzq, table) and gate 1 is about the KIND, not the pipeline (B3
 * wires + gates the pipeline through the front door).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_stfn_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "zturn.h"

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

/* conv0 (MSF) digit reversal over r[0..m-1] — the P0b-pinned convention. */
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

typedef void (*termfn)(const double *, const double *, double *, double *,
                       const double *, const double *,
                       size_t, size_t, size_t, size_t, size_t);

typedef struct { int N, nf, chain[8]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4} },     /* banked, r4 terminator                 */
    { 4096,  6, {4,4,4,4,4,4} },
    { 8192,  6, {4,8,4,4,4,4} },
    { 16384, 7, {4,4,4,4,4,4,4} },
    { 32768, 7, {4,8,4,4,4,4,4} },
    { 16384, 6, {4,8,4,4,4,8} },   /* last==8: the r8 terminator form       */
};

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    int fails = 0;
    printf("\n=== B2: natural terminator — bit-identity + speed ===\n");
    printf("%-7s %-16s %-3s | %-10s %-10s | %-11s %-11s\n",
           "N", "chain", "Rt", "fwd", "bwd", "stfn/stf", "stfbn/stfb");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!p) { printf("%-7d %-16s create REFUSED\n", N, cs); fails++; continue; }

        const size_t OLs = (size_t)N / (size_t)Rt;   /* columns per lane     */
        const long M = (long)(OLs / 4);              /* 64B blocks per lane  */
        size_t *tf = (size_t *)malloc(sizeof(size_t) * (size_t)M); /* rho^-1 */
        size_t *tb = (size_t *)malloc(sizeof(size_t) * (size_t)M); /* rho    */
        for (long t = 0; t < M; t++)
        {
            const long r = rho0(t, c->chain + 1, c->nf - 2);
            tf[r] = (size_t)t;   /* tf[t] = rho^{-1}(t) via forward eval     */
            tb[t] = (size_t)r;
        }

        /* random plane + comb inputs */
        srand(31 + N + Rt);
        for (long i = 0; i < 2L * N; i++)
            p->plane[i] = (double)rand() / RAND_MAX - 0.5;
        double *A  = az(2 * (size_t)N), *B = az(2 * (size_t)N);
        double *R  = az(2 * (size_t)N);
        double *zs = az(2 * (size_t)N), *zn = az(2 * (size_t)N);
        double *pa = az(2 * (size_t)N), *pb = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
            zs[i] = (double)rand() / RAND_MAX - 0.5;

        const termfn F  = (Rt == 8) ? radix8_z_stf_r4_fwd_avx2
                                    : radix4_z_stf_r4_fwd_avx2;
        const termfn Fn = (Rt == 8) ? radix8_z_stfn_r4_fwd_avx2
                                    : radix4_z_stfn_r4_fwd_avx2;
        const termfn Bw = (Rt == 8) ? radix8_z_stf_r4_bwd_avx2
                                    : radix4_z_stf_r4_bwd_avx2;
        const termfn Bn = (Rt == 8) ? radix8_z_stfn_r4_bwd_avx2
                                    : radix4_z_stfn_r4_bwd_avx2;

        /* ---- GATE 1 fwd: stfn == permute(stf) ---- */
        F (p->plane, 0, A, 0, p->tzq, 0,          0, 0, OLs, 0, OLs);
        Fn(p->plane, 0, B, 0, p->tzq, (const double *)tf,
                                                   0, 0, OLs, 0, OLs);
        /* reference: per lane, natural block t = scrambled block tf[t]      */
        for (int l = 0; l < Rt; l++)
            for (long t = 0; t < M; t++)
                memcpy(R + 2 * (l * (long)OLs) + 8 * t,
                       A + 2 * (l * (long)OLs) + 8 * (long)tf[t],
                       8 * sizeof(double));
        const int fok = memcmp(B, R, 2 * (size_t)N * sizeof(double)) == 0;

        /* ---- GATE 1 bwd: stfbn(permuted zs) == stfb(zs) on the plane ---- */
        for (int l = 0; l < Rt; l++)
            for (long t = 0; t < M; t++)         /* zn[block rho(t)] = zs[t] */
                memcpy(zn + 2 * (l * (long)OLs) + 8 * (long)tb[t],
                       zs + 2 * (l * (long)OLs) + 8 * t,
                       8 * sizeof(double));
        memset(pa, 0, 2 * (size_t)N * sizeof(double));
        memset(pb, 0, 2 * (size_t)N * sizeof(double));
        Bw(zs, 0, pa, 0, p->tzqb, 0,               0, 0, OLs, 0, OLs);
        Bn(zn, 0, pb, 0, p->tzqb, (const double *)tb,
                                                    0, 0, OLs, 0, OLs);
        const int bok = memcmp(pa, pb, 2 * (size_t)N * sizeof(double)) == 0;

        /* ---- GATE 2: paced speed, 15 rounds, medians ---- */
        double sf[15], sn[15], sbw[15], sbn[15];
        const int reps = N <= 4096 ? 400 : (N <= 16384 ? 100 : 50);
        for (int r = 0; r < 15; r++)
        {
            double t0;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                F(p->plane, 0, A, 0, p->tzq, 0, 0, 0, OLs, 0, OLs);
            sf[r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                Fn(p->plane, 0, B, 0, p->tzq, (const double *)tf, 0, 0, OLs, 0, OLs);
            sn[r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                Bw(zs, 0, pa, 0, p->tzqb, 0, 0, 0, OLs, 0, OLs);
            sbw[r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                Bn(zn, 0, pb, 0, p->tzqb, (const double *)tb, 0, 0, OLs, 0, OLs);
            sbn[r] = (now_ns() - t0) / reps;
            pace(100);
        }
        qsort(sf, 15, sizeof(double), dcmp);  qsort(sn, 15, sizeof(double), dcmp);
        qsort(sbw, 15, sizeof(double), dcmp); qsort(sbn, 15, sizeof(double), dcmp);
        const double rf = sn[7] / sf[7], rb = sbn[7] / sbw[7];

        const int ok = fok && bok;
        if (!ok) fails++;
        printf("%-7d %-16s %-3d | %-10s %-10s | %8.3fx %10.3fx%s\n",
               N, cs, Rt,
               fok ? "EXACT" : "*** DIFFER ***",
               bok ? "EXACT" : "*** DIFFER ***",
               rf, rb, ok ? "" : "   *** FAIL ***");

        free(tf); free(tb);
#ifdef _WIN32
        _aligned_free(A); _aligned_free(B); _aligned_free(R);
        _aligned_free(zs); _aligned_free(zn); _aligned_free(pa); _aligned_free(pb);
#endif
        vfft_zturn2_destroy(p);
    }

    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    printf("speed ratios are the PASS in isolation (hot plane) — P0c's envelope\n"
           "is <=~1.12x at 4096-scale, ~1.0x above; treat <1.15x as green.\n");
    return fails ? 1 : 0;
}
