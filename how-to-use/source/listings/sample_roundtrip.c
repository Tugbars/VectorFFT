/* sample_roundtrip.c: one 1D complex transform of N = 64 points, forward and
 * back, checked against the input times N. The listing of the sample chapter.
 */
#include <math.h>
#include <stdio.h>
#include "vfft.h"

enum { N = 64 };
static const double TOL = 1e-12;

int main(void)
{
    /* region create */
    vfft_config_t cfg = {0};        /* zeroed: 1D C2C, out of place */
    cfg.n[0] = N;
    vfft_plan plan = vfft_create(&cfg);
    if (!plan)
    {
        fprintf(stderr, "vfft_create: %s\n", vfft_error());
        return 1;
    }
    /* end region create */

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
    for (int n = 0; n < N; n++)
    {
        x[2 * n] = cos(0.1 * n);        /* Re x[n] */
        x[2 * n + 1] = sin(0.3 * n);    /* Im x[n] */
    }

    vfft_execute(plan, VFFT_FORWARD, x, X);
    vfft_execute(plan, VFFT_BACKWARD, X, y);    /* y = N x */

    double err = 0.0;
    for (size_t i = 0; i < info.in_doubles; i++)
    {
        double d = fabs(y[i] / N - x[i]);
        if (d > err || isnan(d))
            err = d;
    }
    int pass = err <= TOL;          /* false for a NaN */
    printf("roundtrip of %d points: max |y/N - x| = %.1e, "
           "tolerance %.0e: %s\n", N, err, TOL, pass ? "PASS" : "FAIL");

    vfft_free(y);
    vfft_free(X);
    vfft_free(x);
    vfft_destroy(plan);
    return pass ? 0 : 1;
}
