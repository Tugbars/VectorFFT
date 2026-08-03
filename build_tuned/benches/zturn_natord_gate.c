/* zturn_natord_gate.c — B3 front-door gate for the NATURAL-ORDER zturn mode.
 *
 * Where zturn_stfn_gate.c (B2) gated the terminator KERNELS in isolation,
 * this gates the WIRED PLAN: vfft_zturn2_set_natord() -> full cascade
 * execute, per chain (mixed-radix chains mandatory — rho is an involution on
 * uniform chains, which MASKS ntf/ntb mix-ups; the B2 gate's own first
 * failure) x both terminator radices x both directions.
 *
 * ARMS (each independent; roundtrip is NOT among them — 🔴 roundtrip cannot
 * gate a permuted transform, and the scrambled control IS permuted):
 *   1. fwd EXACT:   natord fwd == scrambled fwd explicitly block-permuted
 *                   (P0b map), memcmp. Catches wiring, not just math.
 *   2. fwd REF:     natord fwd == naive O(N^2) DFT elementwise IN ORDER,
 *                   tolerance. Independent of arm 1 — catches a shared wrong
 *                   permutation in both plans.
 *   3. bwd REF:     natord bwd(naive DFT spectrum) == N*x elementwise,
 *                   tolerance. The bwd input is an EXTERNAL reference
 *                   spectrum, never our own fwd output.
 *   4. bwd EXACT:   natord bwd(natural arrangement of zs) == scrambled
 *                   bwd(zs), memcmp on the final output (both end natural
 *                   time domain).
 *   5. COMPOSE:     natord + tiled mids == natord untiled, memcmp EXACT;
 *                   and tfuse requested alongside natord must come back 0
 *                   (rho spans the section — per-tile terminator illegal).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_natord_gate.c
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

/* conv0 (MSF) digit reversal over r[0..m-1] — the P0b-pinned convention.
 * Deliberately a LOCAL copy: the gate's reference permutation must not share
 * code with the driver it is gating. */
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

/* naive DFT via a precomputed w^(jk mod N) table: X[k] = sum_j x[j] w^{jk},
 * w = e^{-2pi i/N} (the library convention, pinned by the P0b tone probe). */
static void naive_dft(const double *x, double *X, long N, int sign)
{
    double *wr = (double *)malloc(sizeof(double) * (size_t)N);
    double *wi = (double *)malloc(sizeof(double) * (size_t)N);
    for (long j = 0; j < N; j++)
    {
        const double a = (double)sign * 2.0 * M_PI * (double)j / (double)N;
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
    { 2048,  5, {4,8,4,4,4} },     /* mixed middle, r4 terminator           */
    { 4096,  6, {4,4,4,4,4,4} },   /* uniform (the masking case, kept as    */
                                   /* control alongside the mixed chains)   */
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
    printf("\n=== B3: natural-order zturn — front-door plan gate ===\n");
    printf("%-7s %-16s %-3s | %-6s %-10s %-10s %-6s | %-9s\n",
           "N", "chain", "Rt", "fwdEQ", "fwdREF", "bwdREF", "bwdEQ", "compose");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *ps =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pn =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!ps || !pn || !vfft_zturn2_set_natord(pn, 1))
        {
            printf("%-7d %-16s create/setter REFUSED\n", N, cs);
            fails++;
            vfft_zturn2_destroy(ps);
            vfft_zturn2_destroy(pn);
            continue;
        }

        const size_t OLs = (size_t)N / (size_t)Rt;
        const long M = (long)(OLs / 4);
        size_t *tf = (size_t *)malloc(sizeof(size_t) * (size_t)M); /* rho^-1 */
        size_t *tb = (size_t *)malloc(sizeof(size_t) * (size_t)M); /* rho    */
        for (long t = 0; t < M; t++)
        {
            const long r = rho0(t, c->chain + 1, c->nf - 2);
            tf[r] = (size_t)t;
            tb[t] = (size_t)r;
        }

        srand(1009 + N + Rt);
        double *x  = az(2 * (size_t)N);   /* time-domain input              */
        double *X  = az(2 * (size_t)N);   /* naive DFT reference (natural)  */
        double *ys = az(2 * (size_t)N);   /* scrambled fwd out              */
        double *yn = az(2 * (size_t)N);   /* natural fwd out                */
        double *R  = az(2 * (size_t)N);   /* permuted-scrambled reference   */
        double *zs = az(2 * (size_t)N);   /* random scrambled spectrum      */
        double *zn = az(2 * (size_t)N);   /* its natural arrangement        */
        double *bs = az(2 * (size_t)N), *bn = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
        {
            x[i]  = (double)rand() / RAND_MAX - 0.5;
            zs[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* ---- arm 1: natord fwd == permute(scrambled fwd), EXACT -------- */
        vfft_zturn2_execute_fwd(ps, x, ys);
        vfft_zturn2_execute_fwd(pn, x, yn);
        /* scrambled block t holds natural block rho(t):
         * natural[u] = scrambled[rho^{-1}(u)] = scrambled[tf[u]], per lane  */
        for (int l = 0; l < Rt; l++)
            for (long u = 0; u < M; u++)
                memcpy(R  + 2 * (l * (long)OLs) + 8 * u,
                       ys + 2 * (l * (long)OLs) + 8 * (long)tf[u],
                       8 * sizeof(double));
        const int a1 = memcmp(yn, R, 2 * (size_t)N * sizeof(double)) == 0;

        /* ---- arm 2: natord fwd == naive DFT IN ORDER, tolerance -------- */
        naive_dft(x, X, N, -1);
        double xm = 0.0, e2 = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            const double m = fabs(X[i]);
            if (m > xm) xm = m;
            const double d = fabs(yn[i] - X[i]);
            if (d > e2) e2 = d;
        }
        const double r2 = e2 / xm;
        const int a2 = r2 < 1e-9;

        /* ---- arm 3: natord bwd(naive spectrum) == N*x, tolerance ------- */
        vfft_zturn2_execute_bwd(pn, X, bn);
        double e3 = 0.0, x3 = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            const double ref = (double)N * x[i];
            const double m = fabs(ref);
            if (m > x3) x3 = m;
            const double d = fabs(bn[i] - ref);
            if (d > e3) e3 = d;
        }
        const double r3 = e3 / x3;
        const int a3 = r3 < 1e-9;

        /* ---- arm 4: natord bwd(natural arr of zs) == scrambled bwd(zs) - */
        for (int l = 0; l < Rt; l++)
            for (long t = 0; t < M; t++)   /* zn[block rho(t)] = zs[block t] */
                memcpy(zn + 2 * (l * (long)OLs) + 8 * (long)tb[t],
                       zs + 2 * (l * (long)OLs) + 8 * t,
                       8 * sizeof(double));
        vfft_zturn2_execute_bwd(ps, zs, bs);
        vfft_zturn2_execute_bwd(pn, zn, bn);
        const int a4 = memcmp(bn, bs, 2 * (size_t)N * sizeof(double)) == 0;

        /* ---- arm 5: compose with tiled mids + tfuse-refusal fence ------ */
        int a5 = 1;
        char comp[10] = "n/a";
        {
            long tws[3] = { 2048, 1024, 512 };
            long got = 0;
            for (int i = 0; i < 3 && !got; i++)
                if (vfft_zturn2_set_tile_w(pn, 1, tws[i], /*tfuse*/ 1,
                                           /*thonest*/ 0))
                    got = tws[i];
            if (got)
            {
                if (pn->tfuse != 0) a5 = 0;          /* fence must hold      */
                vfft_zturn2_execute_fwd(pn, x, bs);  /* bs reused as scratch */
                if (memcmp(bs, yn, 2 * (size_t)N * sizeof(double)) != 0)
                    a5 = 0;
                vfft_zturn2_execute_bwd(pn, zn, bs);
                {
                    double *bs2 = az(2 * (size_t)N);
                    vfft_zturn2_execute_bwd(ps, zs, bs2);
                    if (memcmp(bs, bs2, 2 * (size_t)N * sizeof(double)) != 0)
                        a5 = 0;
                    fz(bs2);
                }
                snprintf(comp, sizeof comp, a5 ? "w%ld OK" : "w%ld BAD", got);
                vfft_zturn2_set_tile_w(pn, 0, 0, 0, 0);
            }
        }

        const int ok = a1 && a2 && a3 && a4 && a5;
        if (!ok) fails++;
        printf("%-7d %-16s %-3d | %-6s %.1e   %.1e   %-6s | %-9s%s\n",
               N, cs, Rt,
               a1 ? "EXACT" : "DIFF!",
               r2, r3,
               a4 ? "EXACT" : "DIFF!",
               comp, ok ? "" : "   *** FAIL ***");

        free(tf); free(tb);
        fz(x); fz(X); fz(ys); fz(yn); fz(R); fz(zs); fz(zn); fz(bs); fz(bn);
        vfft_zturn2_destroy(ps);
        vfft_zturn2_destroy(pn);
    }

    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
