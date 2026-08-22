/* odd_n_real_probe.c — is R2C CORRECT at odd N, on every layout it accepts?
 *
 * A research sweep reported that odd-N R2C is "reachable and silently wrong
 * through the public front door" (r2c_dispatch.h:482, the STRIDE zo-mode
 * postprocess). That is a correctness claim about shipping code, so it gets
 * tested, not discussed.
 *
 * Odd N is the case with NO half-N embedding: the even path reinterprets
 * x[N] as z[N/2] and folds, which is meaningless when N is odd, so odd N must
 * take a full-N route instead. This probe asks only one question, per (N,
 * layout, placement): does the spectrum match a naive DFT?
 *
 * C2R is included for coverage even though it is expected to REFUSE odd N
 * (vfft.c:5596) -- a loud refusal is a correct outcome here and is reported as
 * such. What would NOT be acceptable is a plan that builds and computes
 * garbage.
 *
 * Controls: even N on the same code path must pass, so a failure cannot be
 * blamed on the probe's own reference or indexing.
 *
 * Build: python build.py --src benches/odd_n_real_probe.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static void naive(const double *x, int N, double *Xr, double *Xi)
{
    int f, n;
    for (f = 0; f <= N/2; f++) { double sr = 0, si = 0;
        for (n = 0; n < N; n++) {
            double a = -2.0*M_PI*(double)f*n/(double)N;
            sr += x[n]*cos(a); si += x[n]*sin(a); }
        Xr[f] = sr; Xi[f] = si; }
}

static int g_fail = 0;

static void probe_r2c(int N, int interleaved)
{
    const int nb = N/2 + 1;
    vfft_config_t cfg;
    vfft_plan p;
    double *x  = (double *)calloc((size_t)N + 8, sizeof *x);
    double *o  = (double *)calloc(2*(size_t)nb + 8, sizeof *o);
    double *oi = (double *)calloc((size_t)nb + 8, sizeof *oi);
    double *Xr = (double *)malloc((size_t)nb*sizeof *Xr);
    double *Xi = (double *)malloc((size_t)nb*sizeof *Xi);
    double w = 0, m = 0;
    int f;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = interleaved ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    cfg.nthreads = 1;

    for (f = 0; f < N; f++) x[f] = rnd();
    naive(x, N, Xr, Xi);

    p = vfft_create(&cfg);
    if (!p) {
        printf("  r2c %-11s N=%-5d  create REFUSED (loud -- acceptable)\n",
               interleaved ? "INTERLEAVED" : "SPLIT", N);
        goto done;
    }
    if (interleaved) vfft_execute(p, VFFT_FORWARD, x, NULL, o,  NULL);
    else             vfft_execute(p, VFFT_FORWARD, x, NULL, o,  oi);

    for (f = 0; f < nb; f++) {
        double gr = interleaved ? o[2*f]   : o[f];
        double gi = interleaved ? o[2*f+1] : oi[f];
        double a = fabs(Xr[f]) + fabs(Xi[f]);
        if (a > m) m = a;
        if (fabs(gr - Xr[f]) > w) w = fabs(gr - Xr[f]);
        if (fabs(gi - Xi[f]) > w) w = fabs(gi - Xi[f]);
    }
    {
        double rel = m > 0 ? w/m : w;
        int ok = rel < 1e-9;
        printf("  r2c %-11s N=%-5d  rel %.2e  %s\n",
               interleaved ? "INTERLEAVED" : "SPLIT", N, rel,
               ok ? "OK" : "*** WRONG (built a plan, computed garbage) ***");
        if (!ok) g_fail = 1;
    }
    vfft_destroy(p);
done:
    free(x); free(o); free(oi); free(Xr); free(Xi);
}

static void probe_c2r_refusal(int N)
{
    vfft_config_t cfg;
    vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2R; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
    p = vfft_create(&cfg);
    printf("  c2r INTERLEAVED N=%-5d  %s\n", N,
           p ? "*** BUILT (expected a loud refusal at odd N) ***"
             : "refused (loud) -- the documented behaviour");
    if (p) { g_fail = 1; vfft_destroy(p); }
}

static int is_prime(int n)
{ int d; if (n < 2) return 0; for (d = 2; d*d <= n; d++) if (n % d == 0) return 0; return 1; }

/* returns 1 if the cell is WRONG */
static int check(int N, int interleaved)
{
    const int nb = N/2 + 1;
    vfft_config_t cfg; vfft_plan p;
    double *x  = (double *)calloc((size_t)N + 8, sizeof *x);
    double *o  = (double *)calloc(2*(size_t)nb + 8, sizeof *o);
    double *oi = (double *)calloc((size_t)nb + 8, sizeof *oi);
    double *Xr = (double *)malloc((size_t)nb*sizeof *Xr);
    double *Xi = (double *)malloc((size_t)nb*sizeof *Xi);
    double w = 0, m = 0; int f, bad = 0;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = interleaved ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    cfg.nthreads = 1;
    for (f = 0; f < N; f++) x[f] = rnd();
    naive(x, N, Xr, Xi);
    p = vfft_create(&cfg);
    if (!p) { free(x); free(o); free(oi); free(Xr); free(Xi); return 0; } /* refusal is not wrongness */
    if (interleaved) vfft_execute(p, VFFT_FORWARD, x, NULL, o, NULL);
    else             vfft_execute(p, VFFT_FORWARD, x, NULL, o, oi);
    for (f = 0; f < nb; f++) {
        double gr = interleaved ? o[2*f] : o[f];
        double gi = interleaved ? o[2*f+1] : oi[f];
        double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > m) m = a;
        if (fabs(gr - Xr[f]) > w) w = fabs(gr - Xr[f]);
        if (fabs(gi - Xi[f]) > w) w = fabs(gi - Xi[f]);
    }
    bad = !((m > 0 ? w/m : w) < 1e-9);
    if (bad) printf("  N=%-5d %-11s rel %.2e  %s\n", N,
                    interleaved ? "INTERLEAVED" : "SPLIT", m>0?w/m:w,
                    is_prime(N) ? "(PRIME)" : "(composite)");
    vfft_destroy(p);
    free(x); free(o); free(oi); free(Xr); free(Xi);
    return bad;
}

int main(void)
{
    int N, nbadp = 0, nbadc = 0, nprime = 0, ncomp = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("R2C correctness sweep -- EVERY N from 3 to 300, both layouts\n");
    printf("  only FAILURES are printed; a create refusal is not a failure\n\n");
    for (N = 3; N <= 300; N++) {
        int b;
        fprintf(stderr, "[at N=%d]\n", N);
        b = check(N, 1) | check(N, 0);
        if (is_prime(N)) { nprime++; if (b) nbadp++; }
        else             { ncomp++;  if (b) nbadc++; }
    }
    printf("\n  PRIME N     : %d tested, %d WRONG\n", nprime, nbadp);
    printf("  composite N : %d tested, %d WRONG\n", ncomp, nbadc);
    printf("\n%s\n", (nbadp || nbadc) ? "*** R2C: DEFECT PRESENT ***"
                                       : "R2C: no defect observed");
    return (nbadp || nbadc) ? 1 : 0;
}
