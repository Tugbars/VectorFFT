/* natorder_opp_pswap_test.c — validate the OPPORTUNISTIC-PSWAP 1D create fix.
 *
 * The fix: when a cell's calibrated chain is palindromic (its perm M is an involution),
 * NATURAL create uses pair_pass on the SAME calibrated plan (deterministic, no race, no
 * generic-vs-JIT bias) instead of injecting a separate uniform-T1S palindrome. Wisdom marker
 * is nat_mode=5 (PSWAP) with nat_nf=0. 256/4 is THE flip cell (16·16 palindrome; the old race
 * mis-picked PURE ~8/10 due to generic-vs-JIT bias).
 *
 * Two things must hold, and a permutation bug can only be caught by the first:
 *   (1) NATURAL forward output == reference naive DFT in natural bin order  (catches wrong perm)
 *   (2) NATURAL fwd then NATURAL inv == scaled identity                     (roundtrip)
 * Plus: a SECOND create (wisdom LOOKUP) must reproduce (1)+(2) — exercises the stored-verdict
 * reload path (reader must keep nat_mode=PSWAP with nat_nf=0, not downgrade to UNSET).
 *
 * Build: python build.py --src test/natorder_opp_pswap_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static void fill(double *re, double *im, size_t n) {
    for (size_t i = 0; i < n; i++) {
        re[i] = (double)((i * 2654435761u) & 4095) / 4096.0 - 0.5;
        im[i] = (double)((i * 40503u)       & 4095) / 4096.0 - 0.5;
    }
}

/* Reference: naive length-N DFT of each of the K lanes (split, interleaved as [n*K+lane]).
 * Forward sign convention e^{-2pi i k n / N}, natural bin order k=0..N-1. */
static void ref_dft(int N, size_t K, const double *re, const double *im,
                    double *ore, double *oim) {
    for (size_t lane = 0; lane < K; lane++) {
        for (int k = 0; k < N; k++) {
            double ar = 0, ai = 0;
            for (int n = 0; n < N; n++) {
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)N;
                double c = cos(ang), s = sin(ang);
                double xr = re[(size_t)n * K + lane], xi = im[(size_t)n * K + lane];
                ar += xr * c - xi * s;
                ai += xr * s + xi * c;
            }
            ore[(size_t)k * K + lane] = ar;
            oim[(size_t)k * K + lane] = ai;
        }
    }
}

static double maxdiff(const double *ar, const double *ai,
                      const double *br, const double *bi, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double dr = ar[i] - br[i], di = ai[i] - bi[i];
        double d = sqrt(dr * dr + di * di);
        if (d > m) m = d;
    }
    return m;
}

static vfft_plan mk(int N, size_t K, int order) {
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = order;
    return vfft_create(&c);
}

/* One create+check pass on the current wisdom state. pass_label distinguishes
 * MEASURE (first create) from LOOKUP (second create, stored verdict). */
static int check(int N, size_t K, const char *pass_label) {
    size_t n = (size_t)N * K;
    double *re = malloc(n * 8), *im = malloc(n * 8);
    double *fr = malloc(n * 8), *fi = malloc(n * 8);   /* our natural forward   */
    double *gr = malloc(n * 8), *gi = malloc(n * 8);   /* reference naive DFT   */
    fill(re, im, n);
    ref_dft(N, K, re, im, gr, gi);

    vfft_plan pn = mk(N, K, VFFT_ORDER_NATURAL);
    if (!pn) { printf("  N=%d K=%zu [%s] plan NULL\n", N, K, pass_label);
               free(re);free(im);free(fr);free(fi);free(gr);free(gi); return 0; }

    /* (1) forward vs reference DFT */
    memcpy(fr, re, n * 8); memcpy(fi, im, n * 8);
    vfft_execute(pn, VFFT_FORWARD, fr, fi, fr, fi);
    double efwd = maxdiff(fr, fi, gr, gi, n);

    /* (2) roundtrip: inv of the forward == scaled identity (vfft inverse is unnormalized -> /N) */
    vfft_execute(pn, VFFT_BACKWARD, fr, fi, fr, fi);
    for (size_t i = 0; i < n; i++) { fr[i] /= (double)N; fi[i] /= (double)N; }
    double ertt = maxdiff(fr, fi, re, im, n);

    vfft_destroy(pn);
    int ok = (efwd < 1e-9) && (ertt < 1e-11);
    printf("  N=%-4d K=%-3zu [%-7s]  fwd-vs-DFT %.2e   roundtrip %.2e   %s\n",
           N, K, pass_label, efwd, ertt, ok ? "PASS" : "*** FAIL ***");
    free(re);free(im);free(fr);free(fi);free(gr);free(gi);
    return ok;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_opp_wis");

    /* Cells: 256/4 = the flip cell (palindromic 16·16 -> opportunistic PSWAP);
     * 128/64 = injected-PSWAP (4·8·4, nat_nf>0); 64/64 = PURE; 512/4 = another palindrome-candidate. */
    int Ns[] = {256, 128,  64, 512};
    int Ks[] = {  4,  64,  64,   4};
    int all = 1;
    printf("# opportunistic-PSWAP 1D natural validation (fresh wisdom = MEASURE, then LOOKUP)\n");
    for (int i = 0; i < 4; i++) {
        printf("cell %d/%d:\n", Ns[i], Ks[i]);
        all &= check(Ns[i], (size_t)Ks[i], "MEASURE");   /* first create: derive + bank verdict */
        all &= check(Ns[i], (size_t)Ks[i], "LOOKUP");    /* second create: reload stored verdict */
    }
    printf("\n%s\n", all ? "ALL PASS" : "*** SOME FAILED ***");
    return all ? 0 : 1;
}
