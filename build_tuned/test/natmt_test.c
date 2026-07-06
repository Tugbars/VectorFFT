/* natmt_test.c — MT (T=8) vs ST (T=1) BIT-EXACT gate for 1D in-place c2c, both orders.
 *
 * Isolates the T=8 wrong-output bug found by natorder_vs_mkl (128/32, 512/32):
 *   - DEFAULT (scrambled): times only the FFT MT (_c2c_mt K-split).  If DEFAULT MT != DEFAULT ST -> the
 *     FFT split is the culprit.
 *   - NATURAL: FFT MT + reorder MT (_natorder_reorder_mt).  If only NATURAL diverges -> the reorder split.
 * Same plan is built for ST and MT (nthreads only changes the execute path), so any diff is an MT bug.
 * Noise-immune (compares OUTPUT equality, not timing) — safe on a busy CPU. Caller pins core 0 (MT gotcha).
 *
 * Build: python build.py --src test/natmt_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static int run(int N, size_t K, int order, int threads,
               const double *xr, const double *xi, double *or_, double *oi) {
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = threads; c.order = order;
    vfft_plan h = vfft_create(&c);
    if (!h) return -1;
    size_t tot = (size_t)N * K;
    memcpy(or_, xr, tot * 8); memcpy(oi, xi, tot * 8);
    vfft_execute(h, VFFT_FORWARD, or_, oi, or_, oi);
    vfft_destroy(h);
    return 0;
}

static double maxdiff(const double *ar, const double *ai, const double *br, const double *bi, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) { double e = fabs(ar[i] - br[i]) + fabs(ai[i] - bi[i]); if (e > m) m = e; }
    return m;
}
/* naive O(N^2) natural DFT of lane 0 (absolute correctness reference). */
static void naive(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi) {
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * 3.14159265358979323846 * k * n / N, c = cos(a), s = sin(a);
            sr += re[(size_t)n * K] * c - im[(size_t)n * K] * s;
            si += re[(size_t)n * K] * s + im[(size_t)n * K] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}
/* natural fwd (lane 0) vs naive; err normalized by peak. */
static double vs_naive(const double *fr, const double *fi, const double *Xr, const double *Xi, int N, size_t K) {
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);
    double e = 0;
    for (int k = 0; k < N; k++) { double d = fabs(fr[(size_t)k*K] - Xr[k]) + fabs(fi[(size_t)k*K] - Xi[k]); if (d > e) e = d; }
    return e / (sc > 0 ? sc : 1);
}

static int cell(int N, size_t K, int T) {
    size_t tot = (size_t)N * K;
    double *xr = malloc(tot*8), *xi = malloc(tot*8);
    double *a = malloc(tot*8), *b = malloc(tot*8), *c = malloc(tot*8), *d = malloc(tot*8);
    for (size_t i = 0; i < tot; i++) { xr[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }

    run(N, K, VFFT_ORDER_DEFAULT, 1, xr, xi, a, b);   /* scrambled ST */
    run(N, K, VFFT_ORDER_DEFAULT, T, xr, xi, c, d);   /* scrambled MT */
    double ds = maxdiff(a, b, c, d, tot);

    run(N, K, VFFT_ORDER_NATURAL, 1, xr, xi, a, b);   /* natural ST (calibrates @nat on miss) */
    run(N, K, VFFT_ORDER_NATURAL, T, xr, xi, c, d);   /* natural MT (consumes the same @nat) */
    double dn = maxdiff(a, b, c, d, tot);             /* MT-vs-ST consistency */

    /* ABSOLUTE correctness vs naive DFT — catches a bug present in BOTH MT and ST (which dn misses). */
    double *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    naive(xr, xi, N, K, Xr, Xi);
    double stE = vs_naive(a, b, Xr, Xi, N, K);        /* natural ST vs naive */
    double mtE = vs_naive(c, d, Xr, Xi, N, K);        /* natural MT vs naive */

    int ok = ds < 1e-12 && dn < 1e-12 && stE < 1e-9 && mtE < 1e-9;
    printf("  N=%-5d K=%-3zu  scr MT-ST=%.0e  nat MT-ST=%.0e  | nat-ST/naive=%.0e  nat-MT/naive=%.0e   %s\n",
           N, K, ds, dn, stE, mtE, ok ? "ok" : "*** FAIL");
    free(xr); free(xi); free(a); free(b); free(c); free(d); free(Xr); free(Xi);
    return ok;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1);  /* core 0 = MT caller (workers pin 1..T-1) */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natmt_wis");
    int T = 8;
    printf("# MT(T=%d) vs ST(T=1) bit-exact, 1D in-place c2c (scrambled + natural)\n", T);
    int Ns[] = { 128, 256, 512, 1024 };
    size_t Ks[] = { 32, 64, 128, 256 };
    int all = 1;
    for (int ki = 0; ki < 4; ki++)
        for (int ni = 0; ni < 4; ni++)
            all &= cell(Ns[ni], Ks[ki], T);
    printf("\n%s\n", all ? "ALL PASS (MT == ST both orders)" : "*** SOME MT != ST ***");
    return all ? 0 : 1;
}
