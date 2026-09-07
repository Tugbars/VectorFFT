/* zturn_r8_gate.c — the r0 = 8 INGEST GEOMETRY through the zturn driver.
 *
 * chain[0] = 8: 8 sections of N/8, the radix-8 DIF ingest (s0t8), mids on
 * the linear plane (Gp = G[s]/8, the +4h lane term), the terminator PER HALF
 * (sections 4h..4h+3 -> bins 4h..4h+3 of every 8-bin column group). Per
 * cell x chain, every (order, terminator form) the plan can run:
 *
 *   1. FWD NATURAL:   execute_fwd(natord) == naive O(N^2) DFT, tolerance.
 *   2. FWD SCRAMBLED: execute_fwd == the r0 = 8 comb of the DFT (bin
 *                     m*N/Rt + 8*k' + 4h + j holds X[m*N/Rt + 8*rho(k') +
 *                     4h + j], rho = the middle-digit reversal), tolerance.
 *   3. BWD:           execute_bwd(DFT) == N*x (natural), execute_bwd(comb)
 *                     == N*x (scrambled), tolerance — the bwd twins consume
 *                     exactly what the fwd produces.
 *   4. ROUNDTRIP:     bwd(fwd(x)) == N*x, both orders, both forms.
 *   5. IN PLACE:      fwd(buf, buf) and bwd(buf, buf) == the OOP outputs,
 *                     memcmp EXACT (the terminator reads only the plane; the
 *                     bwd ingest consumes zin half by half before s0tb8).
 *   6. FENCES:        the unordered-lane twin is REFUSED at r0 = 8 (NULL),
 *                     the tiled axis enumerates ZERO widths, t2q pins 0.
 *   7. CONTROL:       the r0 = 4 chain at the same N passes 1 and 4 (the
 *                     driver's r0 = 4 path is untouched by the wiring).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_r8_gate.c
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
        double re = 0.0, im = 0.0;
        for (long n = 0; n < N; n++)
        {
            const long j = (k * n) % N;
            re += x[2 * n] * wr[j] - x[2 * n + 1] * wi[j];
            im += x[2 * n] * wi[j] + x[2 * n + 1] * wr[j];
        }
        X[2 * k] = re;
        X[2 * k + 1] = im;
    }
    free(wr);
    free(wi);
}

static double relerr(const double *a, const double *b, long n2)
{
    double m = 0.0, e = 0.0;
    for (long i = 0; i < n2; i++)
    {
        if (fabs(b[i]) > m) m = fabs(b[i]);
        if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]);
    }
    return m > 0.0 ? e / m : e;
}

/* the plan's scrambled output law, r0 = 8: bin m*N/Rt + r0*k' + 4h + j holds
 * X[m*N/Rt + r0*rho(k') + 4h + j] (rho = _vfft_zs_brev over the middle
 * digits, k' in [0, N/(r0*Rt))); r0 = 4 is the same law with h = 0. */
static void comb(const vfft_zturn2_plan_t *p, const double *X, double *out)
{
    const long N = p->N, Rt = p->chain[p->nf - 1], r0 = p->r0;
    const long K2 = N / (r0 * Rt);
    for (long m = 0; m < Rt; m++)
        for (long h = 0; h < r0 / 4; h++)
            for (long k2 = 0; k2 < K2; k2++)
            {
                const long br = _vfft_zs_brev(k2, p->nf - 2, p->chain + 1);
                for (long j = 0; j < 4; j++)
                {
                    const long o = m * (N / Rt) + r0 * k2 + 4 * h + j;
                    const long i = m * (N / Rt) + r0 * br + 4 * h + j;
                    out[2 * o] = X[2 * i];
                    out[2 * o + 1] = X[2 * i + 1];
                }
            }
}

static int g_fail = 0;
static void check(int ok, const char *what, int N, const char *cs, const char *arm, double v)
{
    if (!ok) g_fail++;
    printf("  %-4s N=%-5d %-12s %-22s %-9s %.2e\n", ok ? "ok" : "FAIL", N, cs, what, arm, v);
}

static void chain_str(const int *c, int nf, char *out, size_t cap)
{
    int off = 0;
    for (int s = 0; s < nf && off < (int)cap; s++)
        off += snprintf(out + off, cap - (size_t)off, "%s%d", s ? "." : "", c[s]);
}

/* one (chain, natord, form) arm: fwd vs reference, bwd of reference, roundtrip, in place */
static void gate_arm(vfft_zturn2_plan_t *p, int natord, int form, const double *x,
                     const double *X, const double *nx, const char *cs)
{
    const int N = p->N;
    const long n2 = 2L * N;
    const double TOL = 1e-11;
    double *ref = az((size_t)n2), *y = az((size_t)n2), *z = az((size_t)n2), *b = az((size_t)n2);
    char arm[16];
    snprintf(arm, sizeof arm, "%s/%s", natord ? "nat" : "scr", form ? "loaded" : "packed");
    if (!vfft_zturn2_set_natord(p, natord)) { check(0, "set_natord", N, cs, arm, 0); goto out; }
    if (!vfft_zturn2_set_tforms(p, form, form)) { check(0, "set_tforms", N, cs, arm, 0); goto out; }
    if (natord) memcpy(ref, X, (size_t)n2 * sizeof(double));
    else comb(p, X, ref);
    vfft_zturn2_execute_fwd(p, x, y);
    check(relerr(y, ref, n2) <= TOL, "fwd vs reference", N, cs, arm, relerr(y, ref, n2));
    vfft_zturn2_execute_bwd(p, ref, z);
    check(relerr(z, nx, n2) <= TOL, "bwd(reference)==N*x", N, cs, arm, relerr(z, nx, n2));
    vfft_zturn2_execute_bwd(p, y, z);
    check(relerr(z, nx, n2) <= TOL, "roundtrip", N, cs, arm, relerr(z, nx, n2));
    memcpy(b, x, (size_t)n2 * sizeof(double));
    vfft_zturn2_execute_fwd(p, b, b);
    check(memcmp(b, y, (size_t)n2 * sizeof(double)) == 0, "in-place fwd exact", N, cs, arm, relerr(b, y, n2));
    vfft_zturn2_execute_bwd(p, b, b);
    check(memcmp(b, z, (size_t)n2 * sizeof(double)) == 0, "in-place bwd exact", N, cs, arm, relerr(b, z, n2));
out:
    fz(ref); fz(y); fz(z); fz(b);
}

int main(void)
{
    static const struct { int N; int chain[VFFT_ZSPLIT_MAX_NF]; int nf; } cells[] = {
        { 128,  { 8, 4, 4 }, 3 },
        { 256,  { 8, 8, 4 }, 3 },
        { 256,  { 8, 4, 8 }, 3 },
        { 512,  { 8, 8, 8 }, 3 },
        { 512,  { 8, 4, 4, 4 }, 4 },
        { 1024, { 8, 8, 4, 4 }, 4 },
        { 1024, { 8, 4, 8, 4 }, 4 },
        { 2048, { 8, 8, 8, 4 }, 4 },
        { 2048, { 8, 4, 4, 4, 4 }, 5 },
        { 4096, { 8, 8, 8, 8 }, 4 },
        { 4096, { 8, 8, 4, 4, 4 }, 5 },
        /* r0 = 4 CONTROLS at the same N (the untouched path) */
        { 128,  { 4, 8, 4 }, 3 },
        { 512,  { 4, 4, 8, 4 }, 4 },
        { 2048, { 4, 8, 8, 8 }, 4 },
        { 4096, { 4, 4, 4, 4, 4, 4 }, 6 },
    };
    printf("zturn_r8_gate: the r0 = 8 ingest geometry through vfft_zturn2\n");
    for (size_t c = 0; c < sizeof cells / sizeof cells[0]; c++)
    {
        const int N = cells[c].N, nf = cells[c].nf;
        const long n2 = 2L * N;
        char cs[32];
        chain_str(cells[c].chain, nf, cs, sizeof cs);
        vfft_zturn2_plan_t *p = vfft_zturn2_create_chain(N, cells[c].chain, nf);
        if (!p) { check(0, "create_chain", N, cs, "-", 0); continue; }
        double *x = az((size_t)n2), *X = az((size_t)n2), *nx = az((size_t)n2);
        srand(17 + N + nf);
        for (long i = 0; i < n2; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);
        for (long i = 0; i < n2; i++) nx[i] = (double)N * x[i];
        if (p->r0 == 8)
        {
            /* fences: no unordered-lane twin, no tiled axis, t2q pinned 0 */
            vfft_zt_tile_cand_t cand[8];
            int dropped = 0;
            check(vfft_zturn2_create_chain_u(N, cells[c].chain, nf, 1) == NULL, "lanes_u refused", N, cs, "-", 0);
            check(vfft_zturn2_tile_candidates(p, cand, 8, &dropped) == 0 && dropped == 0, "tile widths == 0", N, cs, "-", 0);
            check(p->t2q == 0, "t2q pinned 0", N, cs, "-", 0);
            for (int natord = 0; natord < 2; natord++)
                for (int form = 0; form < 2; form++)
                    gate_arm(p, natord, form, x, X, nx, cs);
        }
        else
        {
            gate_arm(p, 1, 1, x, X, nx, cs);   /* the control: natural, loaded */
            gate_arm(p, 0, 0, x, X, nx, cs);   /* the control: scrambled, packed */
        }
        vfft_zturn2_destroy(p);
        fz(x); fz(X); fz(nx);
    }
    printf(g_fail ? "=== %d FAIL ===\n" : "=== ALL PASS ===\n", g_fail);
    return g_fail ? 1 : 0;
}
