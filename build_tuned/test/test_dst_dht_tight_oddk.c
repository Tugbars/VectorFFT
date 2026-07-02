/* test_dst_dht_tight_oddk.c — close the last sliver: TIGHT (non-padded) odd-K DST-II and DHT
 * through vfft.h. They ride the SAME stride r2c inner that DCT-II uses (now odd-B safe), and
 * their Makhoul pre/post are all-scalar, so they should just work. Checks per kind:
 *   (A) forward vs naive reference (scale-normalized: fit s=<a,b>/<a,a>, residual = max|b-s*a|;
 *       the fitted scale should be ~2 for DST-II RODFT10 and ~1 for DHT — a sanity tag).
 *   (B) fwd->inverse roundtrip recovers scale*x (DST-II via DST-III; DHT is involutory).
 * Build: python build.py --src test/test_dst_dht_tight_oddk.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;
static const double PI = 3.14159265358979323846;

/* FFTW RODFT10 (DST-II): X[k] = 2 sum_n x[n] sin(pi (n+1/2)(k+1)/N), k=0..N-1 */
static void naive_dst2(const double *x, double *X, int N) {
    for (int k = 0; k < N; k++) { double s = 0;
        for (int n = 0; n < N; n++) s += x[n] * sin(PI * (n + 0.5) * (k + 1) / N);
        X[k] = 2.0 * s; }
}
/* FFTW DHT: X[k] = sum_n x[n] (cos(2*pi*n*k/N) + sin(2*pi*n*k/N)) */
static void naive_dht(const double *x, double *X, int N) {
    for (int k = 0; k < N; k++) { double s = 0;
        for (int n = 0; n < N; n++) { double a = 2.0 * PI * n * k / N; s += x[n] * (cos(a) + sin(a)); }
        X[k] = s; }
}

/* scale-normalized residual of b vs a: fit s minimizing ||b - s*a||, return (err, *scale). */
static double fit_err(const double *a, const double *b, size_t n, double *scale) {
    double sab = 0, saa = 0, denom = 0;
    for (size_t i = 0; i < n; i++) { sab += a[i]*b[i]; saa += a[i]*a[i]; }
    double s = saa > 0 ? sab/saa : 0, e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs(b[i] - s*a[i]); if (d > e) e = d; if (fabs(s*a[i]) > denom) denom = fabs(s*a[i]); }
    *scale = s;
    return denom > 0 ? e/denom : e;
}

static void probe(vfft_transform_t xf, const char *nm, void (*naive)(const double*, double*, int), int N, int K) {
    size_t nk = (size_t)N * K;
    double *x = malloc(nk*8), *X = calloc(nk, 8), *y = calloc(nk, 8);
    srand(4 + N + K);
    for (size_t i = 0; i < nk; i++) x[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = xf; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = (size_t)K;   /* tight (no batch) */
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  %s N=%-4d K=%-3d rem%d  create NULL\n", nm, N, K, K%4); fails++; free(x);free(X);free(y); return; }

    vfft_execute(p, VFFT_FORWARD,  x, NULL, X, NULL);
    vfft_execute(p, VFFT_BACKWARD, X, NULL, y, NULL);

    /* (A) forward vs naive, per lane, scale-normalized (one shared scale across all lanes) */
    double *Xr = malloc((size_t)N*K*8), *xl = malloc(N*8), *rl = malloc(N*8);
    for (int k = 0; k < K; k++) { for (int n = 0; n < N; n++) xl[n] = x[n*K+k];
        naive(xl, rl, N); for (int kk = 0; kk < N; kk++) Xr[(size_t)kk*K+k] = rl[kk]; }
    double fscale = 0, fe = fit_err(Xr, X, nk, &fscale);
    free(Xr); free(xl); free(rl);
    /* (B) roundtrip vs x */
    double rscale = 0, re = fit_err(x, y, nk, &rscale);

    int bad = (fe > 1e-10) || (re > 1e-9);
    if (bad) fails++;
    printf("  %s N=%-4d K=%-3d rem%d  fwd_shape=%8.1e (s=%.2f)  roundtrip=%8.1e  %s\n",
           nm, N, K, K%4, fe, fscale, re, bad ? "*** FAIL ***" : "ok");
    vfft_destroy(p); free(x); free(X); free(y);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=dst_dht_tight_probe");
    system("mkdir dst_dht_tight_probe 2>nul");
    printf("# TIGHT (non-padded) odd-K DST-II + DHT through vfft.h (same fixed stride r2c inner as DCT-II)\n");
    int cells[][2] = {{64,8},{64,7},{64,11},{128,7},{256,7},{256,15},{256,13},{512,23}};
    printf("# -- DST-II (fwd vs RODFT10 s~2; roundtrip via DST-III) --\n");
    for (int i=0;i<8;i++) probe(VFFT_DST2, "DST2", naive_dst2, cells[i][0], cells[i][1]);
    printf("# -- DHT (fwd vs cas s~1; involutory roundtrip) --\n");
    for (int i=0;i<8;i++) probe(VFFT_DHT,  "DHT ", naive_dht,  cells[i][0], cells[i][1]);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: tight odd-K DST-II + DHT work (fwd vs naive + roundtrip)\n", fails);
    return fails ? 1 : 0;
}
