/* Correctness harness for the ZTURN-T stage kernels at VW = VFFT_IL_VW:
 * every legal chain (natural + plain), staged walk, fwd/bwd, out-of-place /
 * in-place, untiled + every legal tile, vs a naive long-double DFT. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"

static long double *CS, *SN; static int CSN = 0;
static void ref_dft(const double *x, double *y, int N, int sgn)
{
    int j, k;
    if (CSN != N) { free(CS); free(SN); CS = malloc(N * sizeof *CS); SN = malloc(N * sizeof *SN);
        for (k = 0; k < N; k++) { CS[k] = cosl(2.0L * 3.14159265358979323846264338327950288L * k / N); SN[k] = sinl(2.0L * 3.14159265358979323846264338327950288L * k / N); } CSN = N; }
    for (k = 0; k < N; k++) {
        long double re = 0, im = 0; long idx = 0;
        for (j = 0; j < N; j++) {
            const long double c = CS[idx], s = sgn * SN[idx];      /* exp(sgn*i*2pi*jk/N), sgn=-1 fwd */
            re += x[2*j] * c - x[2*j+1] * s; im += x[2*j] * s + x[2*j+1] * c;
            idx += k; if (idx >= N) idx -= N;
        }
        y[2*k] = (double)re; y[2*k+1] = (double)im;
    }
}
static double relerr(const double *a, const double *b, int n2)
{ double m = 0, e = 0; int i; for (i = 0; i < n2; i++) { if (fabs(b[i]) > m) m = fabs(b[i]); if (fabs(a[i]-b[i]) > e) e = fabs(a[i]-b[i]); } return e / (m > 0 ? m : 1); }

static int ntest = 0, nfail = 0, nplans = 0;
static double worst = 0;
static void check(const char *what, int N, const int *ch, int nf, int scr, size_t tile, double e)
{
    ntest++; if (e > worst) worst = e;
    if (e > 1e-12 || e != e) { int i; nfail++; printf("FAIL %-12s N=%d scr=%d tile=%zu chain=", what, N, scr, tile); for (i = 0; i < nf; i++) printf("%d%s", ch[i], i < nf-1 ? "." : ""); printf(" err=%.3g\n", e); }
}
static void run_chain(int N, const int *ch, int nf, int scr, double *x, double *Xref, double *xinv)
{
    static const size_t tiles[] = { 0, 128, 256, 512, 1024, 2048, 3072 };
    double *y = aligned_alloc(64, 2 * N * sizeof(double) + 64), *z = aligned_alloc(64, 2 * N * sizeof(double) + 64);
    vfft_ztt_plan_t *p = _ztt_create(N, ch, nf, scr, 1);
    size_t ti; long k;
    if (!p) { free(y); free(z); return; }
    nplans++;
    for (ti = 0; ti < sizeof tiles / sizeof tiles[0]; ti++) {
        if (!vfft_ztt_set_tile(p, tiles[ti])) continue;
        /* forward, out of place */
        vfft_ztt_bind(p, 0);
        vfft_ztt_execute_fwd(p, x, y);
        for (k = 0; k < N; k++) { z[2*k] = y[2*vfft_ztt_perm(p, k)]; z[2*k+1] = y[2*vfft_ztt_perm(p, k)+1]; }
        check("fwd-oop", N, ch, nf, scr, tiles[ti], relerr(z, Xref, 2*N));
        /* forward, in place */
        vfft_ztt_bind(p, 1);
        memcpy(y, x, 2 * N * sizeof(double));
        vfft_ztt_execute_fwd(p, y, y);
        for (k = 0; k < N; k++) { z[2*k] = y[2*vfft_ztt_perm(p, k)]; z[2*k+1] = y[2*vfft_ztt_perm(p, k)+1]; }
        check("fwd-ip", N, ch, nf, scr, tiles[ti], relerr(z, Xref, 2*N));
        /* backward: input = x laid out in the plan's order class (natural, or the
           scrambled positions), expected = the unnormalised inverse DFT of x */
        for (k = 0; k < N; k++) { z[2*vfft_ztt_perm(p, k)] = x[2*k]; z[2*vfft_ztt_perm(p, k)+1] = x[2*k+1]; }
        vfft_ztt_bind(p, 0);
        vfft_ztt_execute_bwd(p, z, y);
        check("bwd-oop", N, ch, nf, scr, tiles[ti], relerr(y, xinv, 2*N));
        vfft_ztt_bind(p, 1);
        vfft_ztt_execute_bwd(p, z, z);
        check("bwd-ip", N, ch, nf, scr, tiles[ti], relerr(z, xinv, 2*N));
    }
    vfft_ztt_destroy(p); free(y); free(z);
}
/* every ordered chain over {4,8} U odd mids, product N, 2 <= nf <= 7 */
static int cnt_rej[2];
static void enum_chains(int N, int *ch, int d, long prod, double *x, double *Xr, double *xi)
{
    static const int R[] = { 4, 8, 3, 5, 7, 9, 15 };
    int i, scr;
    if (prod == N && d >= 2) {
        for (scr = 0; scr < 2; scr++) { int before = nplans; run_chain(N, ch, d, scr, x, Xr, xi); if (nplans == before) cnt_rej[scr]++; }
        return;
    }
    if (prod >= N || d >= 7) return;
    for (i = 0; i < 7; i++) if (N % (prod * R[i]) == 0) { ch[d] = R[i]; enum_chains(N, ch, d + 1, prod * R[i], x, Xr, xi); }
}
int main(int argc, char **argv)
{
    static const int Ns[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2048*3, 2048*5, 4096*3, 2048*9, 2048*15, 2048*7 };
    int ni, i, ch[8];
    int maxN = argc > 1 ? atoi(argv[1]) : 1 << 30;
    for (ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++) {
        const int N = Ns[ni];
        double *x, *Xr, *xi;
        int t0 = ntest, p0 = nplans;
        if (N > maxN) continue;
        x = malloc(2 * N * sizeof(double)); Xr = malloc(2 * N * sizeof(double)); xi = malloc(2 * N * sizeof(double));
        srand(N); for (i = 0; i < 2 * N; i++) x[i] = rand() / (double)RAND_MAX - 0.5;
        ref_dft(x, Xr, N, -1); ref_dft(x, xi, N, +1);
        cnt_rej[0] = cnt_rej[1] = 0;
        enum_chains(N, ch, 0, 1, x, Xr, xi);
        printf("N=%-6d plans=%-4d checks=%-5d refused(nat/plain)=%d/%d\n", N, nplans - p0, ntest - t0, cnt_rej[0], cnt_rej[1]);
        fflush(stdout);
        free(x); free(Xr); free(xi);
    }
    printf("VW=%d: %d plans, %d checks, %d FAIL, worst rel err %.3g\n", VFFT_IL_VW, nplans, ntest, nfail, worst);
    return nfail != 0;
}
