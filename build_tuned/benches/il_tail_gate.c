/* il_tail_gate.c — the odd-COUNT tail gate (il_odd_count_tail.md §7.1):
 * every monolithic cil kind at counts 1..8, separating WRONG from
 * NEVER-WRITTEN via canary prefill. Kinds:
 *   n1  (straight, twiddle-free)     — out[l*OLs + k]
 *   n1t (corner-turn store)          — out[k*OLs + l]
 *   t2  (streamed VTW2, fwd pre/bwd post twiddle) — straight store
 * Oracle: scalar per-column DFT/IDFT with the same twiddle convention.
 * Pow2 radices are IN on purpose — they never had a tail, so they are the
 * regression surface. Build: python build.py --src benches/il_tail_gate.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*zk_fn)(const double *, const double *, double *, double *,
                      const double *, const double *,
                      size_t, size_t, size_t, size_t, size_t);

#define DECL(sym) extern void sym(const double *, const double *, double *, \
    double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
DECL(radix4_z_n1t_fwd_avx2)  DECL(radix8_z_n1t_fwd_avx2)
DECL(radix5_z_n1t_fwd_avx2)  DECL(radix9_z_n1t_fwd_avx2)
DECL(radix6_z_n1t_fwd_avx2)  DECL(radix10_z_n1t_fwd_avx2)
DECL(radix3_z_t2_fwd_avx2)   DECL(radix5_z_t2_fwd_avx2)
DECL(radix8_z_t2_fwd_avx2)   DECL(radix9_z_t2_fwd_avx2)
DECL(radix3_z_t2_bwd_avx2)   DECL(radix8_z_t2_bwd_avx2)
DECL(radix5_z_n1_bwd_avx2)   DECL(radix8_z_n1_bwd_avx2)
#undef DECL

/* scalar oracle: per column k, X[l] = sum_j x[j]*e^{sgn*2pi i jl/R}; t2 fwd
 * pre-twiddles leg j by record (j,k); t2 bwd post-twiddles leg l. */
static void oracle(int R, int cnt, int kind /*0 n1,1 n1t,2 t2*/, int bwd,
                   const double *zin, double *zout, size_t Ls, size_t OLs,
                   const double *tw)
{
    double sgn = bwd ? 1.0 : -1.0;
    for (int k = 0; k < cnt; k++) {
        double ir[64], ii[64];
        for (int j = 0; j < R; j++) {
            double xr = zin[2 * (j * Ls + k)], xi = zin[2 * (j * Ls + k) + 1];
            if (kind == 2 && !bwd && j > 0) {
                const double *rec = tw + ((size_t)(k / 2) * (R - 1) + (j - 1)) * 8u;
                int lane = k & 1;
                double c = rec[2 * lane], s = rec[4 + 2 * lane];
                /* kernel BYTW2 = fmadd([c,c], x, mul([s0,s1], cflip x)) with
                 * s0 = rec[4+2*lane], s1 = -s0: re' = c*xr + s0*xi,
                 * im' = c*xi - s0*xr — the same form il2p's F-DIAG uses. */
                double nr = c * xr + s * xi;
                double ni = c * xi - s * xr;
                xr = nr; xi = ni;
            }
            ir[j] = xr; ii[j] = xi;
        }
        for (int l = 0; l < R; l++) {
            double sr = 0, si = 0;
            for (int j = 0; j < R; j++) {
                double a = sgn * 2.0 * M_PI * (double)j * l / R;
                double c = cos(a), s = sin(a);
                sr += ir[j] * c - ii[j] * s;
                si += ir[j] * s + ii[j] * c;
            }
            if (kind == 2 && bwd && l > 0) {
                const double *rec = tw + ((size_t)(k / 2) * (R - 1) + (l - 1)) * 8u;
                int lane = k & 1;
                double c = rec[2 * lane], s = rec[4 + 2 * lane];
                double nr = c * sr + s * si;
                double ni = c * si - s * sr;
                sr = nr; si = ni;
            }
            size_t off = (kind == 1) ? 2 * ((size_t)k * OLs + l)
                                     : 2 * ((size_t)l * OLs + k);
            zout[off] = sr; zout[off + 1] = si;
        }
    }
}

static double *twtab(int R, int cols, int conj)
{
    double *tw = malloc(((size_t)cols / 2 + 1) * (R - 1) * 8 * sizeof(double));
    for (int pp = 0; pp <= cols / 2; pp++)
        for (int l = 1; l < R; l++) {
            double *rf = tw + ((size_t)pp * (R - 1) + (l - 1)) * 8;
            for (int j = 0; j < 2; j++) {
                double a = -2.0 * M_PI * l * (2 * pp + j) / (double)(R * 8);
                double s = conj ? sin(a) : -sin(a);
                rf[2 * j] = cos(a); rf[2 * j + 1] = cos(a);
                rf[4 + 2 * j] = s; rf[4 + 2 * j + 1] = -s;
            }
        }
    return tw;
}

static int cell(const char *name, zk_fn fn, int R, int kind, int bwd)
{
    int bad = 0;
    for (int cnt = 1; cnt <= 8; cnt++) {
        size_t Ls = (size_t)cnt, OLs = (kind == 1) ? (size_t)R : (size_t)cnt;
        size_t nin = (size_t)2 * R * cnt;
        size_t nout = nin + 16; /* canary band */
        double *zi = malloc(nin * 8), *zo = malloc(nout * 8), *ref = malloc(nin * 8);
        double *tw = twtab(R, cnt + 2, bwd);
        srand(11 * R + cnt);
        for (size_t i = 0; i < nin; i++) zi[i] = (double)rand() / RAND_MAX - 0.5;
        for (size_t i = 0; i < nout; i++) zo[i] = 7777.0; /* canary */
        fn(zi, 0, zo, 0, tw, 0, Ls, 0, OLs, 0, (size_t)cnt);
        oracle(R, cnt, kind, bwd, zi, ref, Ls, OLs, tw);
        double worst = 0, scale = 0;
        int unwritten = 0;
        for (size_t i = 0; i < nin; i++) {
            if (zo[i] == 7777.0) unwritten = 1;
            double d = fabs(zo[i] - ref[i]);
            if (d > worst) worst = d;
            if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
        }
        int oob = 0;
        for (size_t i = nin; i < nout; i++) if (zo[i] != 7777.0) oob = 1;
        double rel = worst / (scale > 0 ? scale : 1);
        if (rel > 1e-12 || unwritten || oob) {
            printf("  %-24s cnt=%d rel=%.2e%s%s  *** FAIL ***\n", name, cnt, rel,
                   unwritten ? " NEVER-WRITTEN" : "", oob ? " OOB" : "");
            bad = 1;
        }
        free(zi); free(zo); free(ref); free(tw);
    }
    if (!bad) printf("  %-24s counts 1..8 ok\n", name);
    return bad;
}

int main(void)
{
    int bad = 0;
    printf("-- IL odd-count tail gate: counts 1..8, canary-checked --\n");
    bad |= cell("n1t(4) fwd",  radix4_z_n1t_fwd_avx2, 4, 1, 0);
    bad |= cell("n1t(8) fwd",  radix8_z_n1t_fwd_avx2, 8, 1, 0);
    bad |= cell("n1t(5) fwd",  radix5_z_n1t_fwd_avx2, 5, 1, 0);
    bad |= cell("n1t(9) fwd",  radix9_z_n1t_fwd_avx2, 9, 1, 0);
    bad |= cell("n1t(6) fwd",  radix6_z_n1t_fwd_avx2, 6, 1, 0);
    bad |= cell("n1t(10) fwd", radix10_z_n1t_fwd_avx2, 10, 1, 0);
    bad |= cell("t2(3) fwd",   radix3_z_t2_fwd_avx2, 3, 2, 0);
    bad |= cell("t2(5) fwd",   radix5_z_t2_fwd_avx2, 5, 2, 0);
    bad |= cell("t2(8) fwd",   radix8_z_t2_fwd_avx2, 8, 2, 0);
    bad |= cell("t2(9) fwd",   radix9_z_t2_fwd_avx2, 9, 2, 0);
    bad |= cell("t2(3) bwd",   radix3_z_t2_bwd_avx2, 3, 2, 1);
    bad |= cell("t2(8) bwd",   radix8_z_t2_bwd_avx2, 8, 2, 1);
    bad |= cell("n1(5) bwd",   radix5_z_n1_bwd_avx2, 5, 0, 1);
    bad |= cell("n1(8) bwd",   radix8_z_n1_bwd_avx2, 8, 0, 1);
    printf(bad ? "IL TAIL GATE FAILED\n" : "IL TAIL GATE PASSED\n");
    return bad;
}
