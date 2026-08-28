/* TEMPORARY: is the K=8 nondeterminism in the FFT RESULT, or only in the bytes
 * my digest happens to cover? Reports both, plus the plan's own stride. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static void fill(double *p, size_t n, unsigned seed)
{
    size_t i; unsigned s = seed * 2654435761u + 1u;
    for (i = 0; i < n; i++) { s = s * 1664525u + 1013904223u;
        p[i] = (double)(s >> 8) / (double)(1u << 24) - 0.5; }
}
static unsigned long long dig(const double *p, size_t n)
{
    unsigned long long h = 1469598103934665603ULL; size_t i;
    const unsigned char *b = (const unsigned char *)p;
    for (i = 0; i < n * sizeof(double); i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
int main(void)
{
    const int N = 256; const size_t K = 8; const size_t n = (size_t)N * K;
    vfft_config_t cfg; vfft_plan p;
    double *a, *b, *a0, *b0;
    size_t i; double maxrt = 0.0, maxabs = 0.0;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_SPLIT; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K; cfg.rigor = VFFT_MEASURE;
    p = vfft_create(&cfg);
    if (!p) { printf("create failed\n"); return 1; }
    printf("plan stride = %zu   (K = %zu)\n", vfft_plan_stride(p), K);

    a  = malloc(n * sizeof(double)); b  = malloc(n * sizeof(double));
    a0 = malloc(n * sizeof(double)); b0 = malloc(n * sizeof(double));
    fill(a, n, 264); fill(b, n, 263);
    memcpy(a0, a, n * sizeof(double)); memcpy(b0, b, n * sizeof(double));

    vfft_execute(p, VFFT_FORWARD, a, b, a, b);
    printf("fwd digest  = %016llx %016llx\n", dig(a, n), dig(b, n));
    for (i = 0; i < n; i++) { double m = fabs(a[i]); if (m > maxabs) maxabs = m; }
    printf("fwd max|re| = %.6f\n", maxabs);

    vfft_execute(p, VFFT_BACKWARD, a, b, a, b);
    for (i = 0; i < n; i++) {
        double er = fabs(a[i] / (double)N - a0[i]);
        double ei = fabs(b[i] / (double)N - b0[i]);
        if (er > maxrt) maxrt = er;
        if (ei > maxrt) maxrt = ei;
    }
    printf("roundtrip max err = %.3e   <- the MATH\n", maxrt);
    vfft_destroy(p); free(a); free(b); free(a0); free(b0);
    return 0;
}
