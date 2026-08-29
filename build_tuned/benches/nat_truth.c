/* nat_truth.c - is the natural-order cell's run-to-run variation ROUNDING or a BUG?
 *
 * c2c.split.ip.natural (N=256, K=1, split, in-place) produced two different
 * output digests across repeated processes. A digest cannot tell rounding from
 * a wrong answer, so this compares the FORWARD output against a naive O(N^2)
 * DFT computed in long double - ground truth that shares no code with the
 * library - and reports which branch was taken alongside its error.
 *
 * Forward only, per direction, on purpose: a roundtrip cannot gate a permuted
 * transform, because an inverse that undoes its own permutation hides a wrong
 * permutation entirely.
 *
 * Build: python build.py --src benches/nat_truth.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "vfft.h"

#define N 256

static void fill(double *re, double *im)
{
    unsigned s = 12345u;
    int i;
    for (i = 0; i < N; i++) {
        s = s * 1103515245u + 12345u; re[i] = (double)((s >> 16) & 0x7fff) / 32768.0 - 0.5;
        s = s * 1103515245u + 12345u; im[i] = (double)((s >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
}

static unsigned long long digest(const double *v, int n)
{
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)v;
    size_t i, nb = (size_t)n * sizeof(double);
    for (i = 0; i < nb; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(void)
{
    static double re[N], im[N], re0[N], im0[N];
    static long double tr[N], ti[N];
    vfft_config_t cfg;
    vfft_plan p;
    int k, j;
    double worst = 0.0, mag = 0.0;

    fill(re0, im0);
    /* naive DFT, forward, exp(-2*pi*i*j*k/N) - the same sign convention the
     * library documents for VFFT_FORWARD. */
    for (k = 0; k < N; k++) {
        long double sr = 0.0L, si = 0.0L;
        for (j = 0; j < N; j++) {
            long double a = -2.0L * 3.14159265358979323846264338327950288L * (long double)j * (long double)k / (long double)N;
            long double c = cosl(a), s = sinl(a);
            sr += (long double)re0[j] * c - (long double)im0[j] * s;
            si += (long double)re0[j] * s + (long double)im0[j] * c;
        }
        tr[k] = sr; ti[k] = si;
    }

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = VFFT_LAYOUT_SPLIT;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;

    p = vfft_create(&cfg);
    if (!p) { printf("CREATE_FAILED\n"); return 2; }
    memcpy(re, re0, sizeof re); memcpy(im, im0, sizeof im);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);

    for (k = 0; k < N; k++) {
        double dr = re[k] - (double)tr[k], di = im[k] - (double)ti[k];
        double e = sqrt(dr * dr + di * di);
        double m = sqrt((double)(tr[k] * tr[k] + ti[k] * ti[k]));
        if (e > worst) worst = e;
        if (m > mag) mag = m;
    }
    printf("digest=%016llx  max_abs_err=%.3e  max_mag=%.3e  rel=%.3e  %s\n",
           digest(re, N) ^ digest(im, N), worst, mag, worst / mag,
           (worst / mag < 1e-12) ? "CORRECT" : "*** WRONG ***");
    vfft_destroy(p);
    return 0;
}
