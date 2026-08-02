/* zturn_natmap_probe.c — P0b: verify the cascade's OUTPUT PERMUTATION closed
 * form before anything derives store maps from it.
 *
 * zturn.h claims (SCRAMBLED class, Rt = chain[nf-1]):
 *     out_z[l*(N/Rt) + 4*k' + j] = X[l*(N/Rt) + 4*rho(k') + j]
 * with rho = digit reversal over the MIDDLE radices chain[1..nf-2],
 * l in [0,Rt), j in [0,4), k' in [0, N/(4*Rt)).
 *
 * The header does NOT pin the digit-order convention of "digit reversal", so
 * this probe tests BOTH (most-significant-first vs least-significant-first
 * digit decomposition) and reports which one the kernels actually implement.
 * The natural-order terminator (Phase B) will bake this map into stores;
 * getting the convention wrong there would produce a correct-looking permuted
 * transform that silently isn't natural — hence: verified empirically, per
 * chain, before the spec is written.
 *
 * METHOD (the discovered-permutation technique from zturn_tcut_gate.c):
 * feed x[n] = e^{+2*pi*i*f*n/N}; the exact DFT is N*delta[k=f]; the peak's
 * position in the cascade output IS the scrambled position of natural bin f.
 * Full rank-N for N<=8192; a fixed 4096-f sample above. Peak must be isolated
 * (|peak| ~ N, worst sidelobe small) and the map a bijection on the tested f.
 *
 * Correctness probe — no timing. Plan-level (pre-front-door, like P0a).
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_natmap_probe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "zturn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* digit reversal of v over radices r[0..m-1].
 * conv 0 (MSF): v = d0*(r1*..*r_{m-1}) + d1*(r2*..) + ... + d_{m-1};
 *               result = d_{m-1}*(r_{m-2}*..*r0) + ... + d0.
 * conv 1 (LSF): v decomposed least-significant-first over r[0..m-1] and
 *               reassembled most-significant-first — the mirror convention. */
static long rho(long v, const int *r, int m, int conv)
{
    long d[16];
    if (conv == 0)
    {
        for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
        /* d[0] is the MSF digit (radix r[0]); reverse */
        long out = 0;
        for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
        return out;
    }
    for (int i = 0; i < m; i++) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = 0; i < m; i++) out = out * r[i] + d[i];
    return out;
}

typedef struct { int N, nf, chain[8]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4} },      /* banked, r4 terminator            */
    { 4096,  6, {4,4,4,4,4,4} },    /* banked 4^6                        */
    { 8192,  6, {4,8,4,4,4,4} },    /* banked                            */
    { 16384, 7, {4,4,4,4,4,4,4} },  /* banked 4^7                        */
    { 16384, 6, {4,8,4,4,4,8} },    /* last==8: the r8 terminator form   */
    { 32768, 7, {4,8,4,4,4,4,4} },  /* banked                            */
};

int main(void)
{
    printf("\n=== P0b: output-permutation closed form, verified per chain ===\n");
    printf("%-7s %-16s %-4s %-10s %-10s %-10s %s\n",
           "N", "chain", "Rt", "tested f", "conv0 ok", "conv1 ok", "verdict");

    int total_fail = 0;

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        const long BLK = N / Rt, M = N / (4L * Rt);
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!p) { printf("%-7d %-16s create REFUSED\n", N, cs); total_fail++; continue; }

        /* rho^{-1} tables for both conventions: nat position q -> k' with
         * rho(k') == q. Built by forward evaluation (reversal is a bijection). */
        long *inv0 = (long *)malloc(sizeof(long) * (size_t)M);
        long *inv1 = (long *)malloc(sizeof(long) * (size_t)M);
        for (long k = 0; k < M; k++)
        {
            inv0[rho(k, c->chain + 1, c->nf - 2, 0)] = k;
            inv1[rho(k, c->chain + 1, c->nf - 2, 1)] = k;
        }

        double *x = az((size_t)N), *y = az((size_t)N);
        const int full = (N <= 8192);
        const int nf_test = full ? N : 4096;
        long ok0 = 0, ok1 = 0, badpeak = 0;
        char *seen = (char *)calloc((size_t)N, 1);
        int bijection = 1;

        for (int t = 0; t < nf_test; t++)
        {
            const long f = full ? t : ((long)t * 40503 + 11) % N; /* spread */
            for (int n = 0; n < N; n++)
            {
                const double a = 2.0 * M_PI * (double)((f * (long)n) % N)
                                 / (double)N;
                x[2 * n] = cos(a);
                x[2 * n + 1] = sin(a);
            }
            vfft_zturn2_execute_fwd(p, x, y);

            long best = -1; double bm = -1.0, second = 0.0;
            for (int k = 0; k < N; k++)
            {
                const double m2 = fabs(y[2 * k]) + fabs(y[2 * k + 1]);
                if (m2 > bm) { second = bm > 0 ? bm : second; bm = m2; best = k; }
                else if (m2 > second) second = m2;
            }
            if (bm < 0.9 * N || second > 0.01 * N) { badpeak++; continue; }
            if (seen[best]) bijection = 0;
            seen[best] = 1;

            /* closed-form predictions */
            const long l = f / BLK, rem = f % BLK, q = rem / 4, j = rem % 4;
            if (best == l * BLK + 4 * inv0[q] + j) ok0++;
            if (best == l * BLK + 4 * inv1[q] + j) ok1++;
        }

        const char *verdict =
            (badpeak || !bijection)            ? "*** PEAK/BIJECTION FAIL ***" :
            (ok0 == nf_test && ok1 == nf_test) ? "BOTH (chain is palindromic-ish)" :
            (ok0 == nf_test)                   ? "conv0 (MSF) CONFIRMED" :
            (ok1 == nf_test)                   ? "conv1 (LSF) CONFIRMED" :
                                                 "*** NEITHER — closed form WRONG ***";
        if (!(ok0 == nf_test || ok1 == nf_test) || badpeak || !bijection)
            total_fail++;

        printf("%-7d %-16s %-4d %-10d %-10ld %-10ld %s\n",
               N, cs, Rt, nf_test, ok0, ok1, verdict);

        free(seen); free(inv0); free(inv1);
        fz(x); fz(y);
        vfft_zturn2_destroy(p);
    }

    printf("\n=== %s ===\n", total_fail
        ? "*** FAIL — do NOT derive store maps from the header's formula ***"
        : "closed form CONFIRMED — Phase B may derive store maps from it");
    return total_fail ? 1 : 0;
}
