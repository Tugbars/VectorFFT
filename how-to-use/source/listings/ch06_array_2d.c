/* ch06_array_2d.c: a 2D VFFT_C2C plan of N1 x N2 = 16 x 32 points, out of
 * place, checked against the direct 2D sum and by a backward transform.
 */
#include <math.h>
#include <stdio.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum { N1 = 16, N2 = 32 };
static const double TOL = 1e-12;

int main(void)
{
    /* region create */
    vfft_config_t cfg = {0};        /* VFFT_C2C, out of place, natural */
    cfg.rank = 2;
    cfg.n[0] = N1;                  /* N1 rows: the slow axis */
    cfg.n[1] = N2;                  /* N2 columns: the contiguous axis */
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
    printf("2D VFFT_C2C of %d x %d points: in_doubles %zu, out_doubles %zu\n",
           N1, N2, info.in_doubles, info.out_doubles);

    /* region fill */
    for (int n1 = 0; n1 < N1; n1++)
        for (int n2 = 0; n2 < N2; n2++)
        {
            size_t l = (size_t)n1 * N2 + n2;        /* row-major */
            x[2 * l] = cos(0.3 * n1 - 0.1 * n2 * n2);     /* Re x[n1,n2] */
            x[2 * l + 1] = sin(0.2 * n1 * n2 + 1.0);      /* Im x[n1,n2] */
        }
    /* end region fill */

    vfft_execute(plan, VFFT_FORWARD, x, X);

    /* X[k1,k2] against the sum over n1, n2 of x[n1,n2] W_N1^(k1 n1)
     * W_N2^(k2 n2), with the angles reduced modulo one turn */
    double err = 0.0;
    for (int k1 = 0; k1 < N1; k1++)
        for (int k2 = 0; k2 < N2; k2++)
        {
            double re = 0.0, im = 0.0;
            for (int n1 = 0; n1 < N1; n1++)
                for (int n2 = 0; n2 < N2; n2++)
                {
                    size_t l = (size_t)n1 * N2 + n2;
                    double t = -2 * PI * ((double)(k1 * n1 % N1) / N1 +
                                          (double)(k2 * n2 % N2) / N2);
                    re += x[2 * l] * cos(t) - x[2 * l + 1] * sin(t);
                    im += x[2 * l] * sin(t) + x[2 * l + 1] * cos(t);
                }
            size_t b = (size_t)k1 * N2 + k2;        /* bin [k1,k2] */
            double d = hypot(X[2 * b] - re, X[2 * b + 1] - im);
            if (d > err || isnan(d))
                err = d;
        }

    vfft_execute(plan, VFFT_BACKWARD, X, y);        /* y = N1 N2 x */
    double back = 0.0;
    for (size_t i = 0; i < info.in_doubles; i++)
    {
        double d = fabs(y[i] / (N1 * N2) - x[i]);
        if (d > back || isnan(d))
            back = d;
    }

    int pass = err <= TOL && back <= TOL;
    printf("forward: max |X - direct sum| = %.1e\n", err);
    printf("roundtrip: max |y/(N1 N2) - x| = %.1e\n", back);
    printf("tolerance %.0e: %s\n", TOL, pass ? "PASS" : "FAIL");

    vfft_free(y);
    vfft_free(X);
    vfft_free(x);
    vfft_destroy(plan);
    return pass ? 0 : 1;
}
