/* zturn_dit_diffprobe.c — diagnose the conj-identity DIFFs: few-ulp
 * rounding (sign flip re-scheduled the DAG — acceptable, gate becomes
 * tolerance) vs gross error (wrong table/addressing — emission bug).
 * Prints per-identity: #differing doubles, max abs diff, max rel diff.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zturn.h"

#define VFFT_DIT_DECL(fn) extern void fn(const double *, const double *, \
    double *, double *, const double *, const double *,                  \
    size_t, size_t, size_t, size_t, size_t);
VFFT_DIT_DECL(radix8_z_dts_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dts_r4_fwd_avx2)
VFFT_DIT_DECL(radix8_z_dtsn_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dtsn_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dtt_r4_fwd_avx2)
#undef VFFT_DIT_DECL

static double *az(size_t d)
{
    return (double *)_aligned_malloc(d * sizeof(double), 64);
}
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}
static void conj_il(double *z, long n)
{
    for (long i = 0; i < n; i++) z[2 * i + 1] = -z[2 * i + 1];
}
static void conj_blk(double *p, size_t doubles)
{
    for (size_t b = 0; b + 8 <= doubles; b += 8)
        for (int j = 4; j < 8; j++) p[b + j] = -p[b + j];
}
static void stats(const char *tag, const double *a, const double *b, long n)
{
    long nd = 0;
    double mad = 0.0, mrd = 0.0, mag = 0.0;
    for (long i = 0; i < n; i++)
    {
        if (fabs(a[i]) > mag) mag = fabs(a[i]);
        if (a[i] != b[i])
        {
            nd++;
            const double d = fabs(a[i] - b[i]);
            if (d > mad) mad = d;
            const double den = fabs(a[i]) > 1e-300 ? fabs(a[i]) : 1.0;
            if (d / den > mrd) mrd = d / den;
        }
    }
    printf("  %-6s ndiff=%ld/%ld  maxabs=%.3e  maxrel=%.3e  maxmag=%.3e\n",
           tag, nd, n, mad, mrd, mag);
}

typedef void (*termfn)(const double *, const double *, double *, double *,
                       const double *, const double *,
                       size_t, size_t, size_t, size_t, size_t);
typedef struct { int N, nf, chain[8]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4} },     /* the dtt-DIFF cell                     */
    { 16384, 6, {4,8,4,4,4,8} },   /* r8 ingest form                        */
};

int main(void)
{
    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        const size_t OLs = (size_t)N / (size_t)Rt;
        const long M = (long)(OLs / 4);
        size_t *tb = (size_t *)malloc(sizeof(size_t) * (size_t)M);
        for (long t = 0; t < M; t++)
            tb[t] = (size_t)rho0(t, c->chain + 1, c->nf - 2);

        const termfn DTS  = (Rt == 8) ? radix8_z_dts_r4_fwd_avx2
                                      : radix4_z_dts_r4_fwd_avx2;
        const termfn DTSN = (Rt == 8) ? radix8_z_dtsn_r4_fwd_avx2
                                      : radix4_z_dtsn_r4_fwd_avx2;
        const termfn STFB = (Rt == 8) ? radix8_z_stf_r4_bwd_avx2
                                      : radix4_z_stf_r4_bwd_avx2;
        const termfn STFBN = (Rt == 8) ? radix8_z_stfn_r4_bwd_avx2
                                       : radix4_z_stfn_r4_bwd_avx2;

        srand(7717 + N + Rt);
        double *x = az(2 * (size_t)N), *xc = az(2 * (size_t)N);
        double *pa = az(2 * (size_t)N), *pb = az(2 * (size_t)N);
        double *wc = az(2 * OLs), *plc = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
        {
            x[i] = (double)rand() / RAND_MAX - 0.5;
            p->plane[i] = (double)rand() / RAND_MAX - 0.5;
        }
        memcpy(xc, x, 2 * (size_t)N * sizeof(double));
        conj_il(xc, N);
        memcpy(wc, p->tzq, 2 * OLs * sizeof(double));
        conj_blk(wc, 2 * OLs);

        printf("N=%d Rt=%d:\n", N, Rt);
        memset(pa, 0, 2 * (size_t)N * sizeof(double));
        memset(pb, 0, 2 * (size_t)N * sizeof(double));
        DTS (x,  0, pa, 0, p->tzq, 0, 0, 0, OLs, 0, OLs);
        STFB(xc, 0, pb, 0, wc,     0, 0, 0, OLs, 0, OLs);
        conj_blk(pb, 2 * (size_t)N);
        stats("dts", pa, pb, 2L * N);

        memset(pa, 0, 2 * (size_t)N * sizeof(double));
        memset(pb, 0, 2 * (size_t)N * sizeof(double));
        DTSN (x,  0, pa, 0, p->tzq, (const double *)tb, 0, 0, OLs, 0, OLs);
        STFBN(xc, 0, pb, 0, wc,     (const double *)tb, 0, 0, OLs, 0, OLs);
        conj_blk(pb, 2 * (size_t)N);
        stats("dtsn", pa, pb, 2L * N);

        memcpy(plc, p->plane, 2 * (size_t)N * sizeof(double));
        conj_blk(plc, 2 * (size_t)N);
        radix4_z_dtt_r4_fwd_avx2(p->plane, 0, pa, 0, 0, 0,
                                 (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
        radix4_z_s0t_r4_bwd_avx2(plc, 0, pb, 0, 0, 0,
                                 (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
        conj_il(pb, N);
        stats("dtt", pa, pb, 2L * N);

        free(tb);
        vfft_zturn2_destroy(p);
    }
    return 0;
}
