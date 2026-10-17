/* ch06_frequencies.c: a complex signal of frequencies +5 and -3 through a 1D
 * VFFT_C2C plan of N = 64 points, forward and back. Each frequency lands in
 * the bin that frequency() names, at N times its amplitude.
 */
#include <math.h>
#include <stdio.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum { N = 64 };
static const double TOL = 1e-12;

/* region frequency */
/* the frequency that bin k of an n-point spectrum holds */
static int frequency(int k, int n)
{
    return k <= n / 2 ? k : k - n;
}
/* end region frequency */

int main(void)
{
    vfft_config_t cfg = {0};        /* 1D VFFT_C2C, out of place, natural */
    cfg.n[0] = N;
    vfft_plan plan = vfft_create(&cfg);
    if (!plan)
    {
        fprintf(stderr, "vfft_create: %s\n", vfft_error());
        return 1;
    }

    vfft_info_t info;
    vfft_info(plan, &info);
    double *x = vfft_malloc(info.in_doubles * sizeof(double));
    double *X = vfft_malloc(info.out_doubles * sizeof(double));
    double *y = vfft_malloc(info.in_doubles * sizeof(double));
    if (!x || !X || !y)
    {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    /* x[n] = e^(j 2 pi 5 n / N) + 0.5 e^(-j 2 pi 3 n / N) */
    for (int n = 0; n < N; n++)
    {
        double a = 2 * PI * 5 * n / N, b = -2 * PI * 3 * n / N;
        x[2 * n] = cos(a) + 0.5 * cos(b);           /* Re x[n] */
        x[2 * n + 1] = sin(a) + 0.5 * sin(b);       /* Im x[n] */
    }

    /* region check */
    vfft_execute(plan, VFFT_FORWARD, x, X);

    /* expected: N at frequency +5, N/2 at frequency -3, 0 elsewhere */
    double err = 0.0;
    for (int k = 0; k < N; k++)
    {
        int f = frequency(k, N);
        double want = f == 5 ? N : f == -3 ? 0.5 * N : 0.0;
        double d = hypot(X[2 * k] - want, X[2 * k + 1]);
        if (d > err || isnan(d))
            err = d;
        if (want != 0.0)
            printf("bin %d holds frequency %+d: Re X[%d] = %.3f\n",
                   k, f, k, X[2 * k]);
    }
    /* end region check */

    vfft_execute(plan, VFFT_BACKWARD, X, y);        /* y = N x */
    double back = 0.0;
    for (size_t i = 0; i < info.in_doubles; i++)
    {
        double d = fabs(y[i] / N - x[i]);
        if (d > back || isnan(d))
            back = d;
    }

    int pass = err <= TOL && back <= TOL;           /* false for a NaN */
    printf("spectrum: max |X[k] - expected| = %.1e\n", err);
    printf("roundtrip: max |y[n]/N - x[n]| = %.1e\n", back);
    printf("tolerance %.0e: %s\n", TOL, pass ? "PASS" : "FAIL");

    vfft_free(y);
    vfft_free(X);
    vfft_free(x);
    vfft_destroy(plan);
    return pass ? 0 : 1;
}
