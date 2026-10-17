/* ch06_volume_in_place.c: a plane wave through a 3D VFFT_C2C plan of
 * N1 x N2 x N3 = 8 x 16 x 32 points, in place. The spectrum holds N1 N2 N3
 * in the bin of the wave's frequencies and 0 elsewhere; the backward
 * transform returns N1 N2 N3 times the wave.
 */
#include <math.h>
#include <stdio.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum { N1 = 8, N2 = 16, N3 = 32, F1 = 3, F2 = -5, F3 = 7 };
static const double TOL = 1e-12;

/* region index */
/* where x[n1,n2,n3] is stored: element l, doubles 2 l and 2 l + 1 */
static size_t element(int n1, int n2, int n3)
{
    return ((size_t)n1 * N2 + n2) * N3 + n3;
}
/* end region index */

/* the wave e^(j 2 pi (F1 n1 / N1 + F2 n2 / N2 + F3 n3 / N3)) */
static void wave(int n1, int n2, int n3, double *re, double *im)
{
    double t = 2 * PI * ((double)F1 * n1 / N1 + (double)F2 * n2 / N2 +
                         (double)F3 * n3 / N3);
    *re = cos(t);
    *im = sin(t);
}

int main(void)
{
    /* region create */
    vfft_config_t cfg = {0};
    cfg.rank = 3;
    cfg.n[0] = N1;                  /* the slowest axis */
    cfg.n[1] = N2;
    cfg.n[2] = N3;                  /* the contiguous axis */
    cfg.placement = VFFT_PLACEMENT_IN;
    vfft_plan plan = vfft_create(&cfg);
    if (!plan)
    {
        fprintf(stderr, "vfft_create: %s\n", vfft_error());
        return 1;
    }
    /* end region create */

    vfft_info_t info;
    vfft_info(plan, &info);
    double *z = vfft_malloc(info.in_doubles * sizeof(double));
    if (!z)
    {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    printf("3D VFFT_C2C in place, %d x %d x %d points: one buffer of %zu "
           "doubles\n", N1, N2, N3, info.in_doubles);

    for (int n1 = 0; n1 < N1; n1++)
        for (int n2 = 0; n2 < N2; n2++)
            for (int n3 = 0; n3 < N3; n3++)
            {
                size_t l = element(n1, n2, n3);
                wave(n1, n2, n3, &z[2 * l], &z[2 * l + 1]);
            }

    /* region execute */
    vfft_execute(plan, VFFT_FORWARD, z, z);         /* the spectrum over x */
    /* end region execute */

    /* region bin */
    /* frequency f of an axis of n points is bin f, or n + f when f < 0 */
    int k1 = F1 < 0 ? N1 + F1 : F1;
    int k2 = F2 < 0 ? N2 + F2 : F2;
    int k3 = F3 < 0 ? N3 + F3 : F3;
    size_t peak = element(k1, k2, k3);
    /* end region bin */
    const double S = (double)N1 * N2 * N3;
    printf("frequency (%d, %d, %d) is bin [%d, %d, %d], element %zu: "
           "Re X = %.3f\n", F1, F2, F3, k1, k2, k3, peak, z[2 * peak]);

    double err = 0.0;                   /* relative to N1 N2 N3 */
    for (size_t l = 0; l < info.in_doubles / 2; l++)
    {
        double want = l == peak ? 1.0 : 0.0;
        double d = hypot(z[2 * l] / S - want, z[2 * l + 1] / S);
        if (d > err || isnan(d))
            err = d;
    }

    vfft_execute(plan, VFFT_BACKWARD, z, z);        /* N1 N2 N3 x */
    double back = 0.0;
    for (int n1 = 0; n1 < N1; n1++)
        for (int n2 = 0; n2 < N2; n2++)
            for (int n3 = 0; n3 < N3; n3++)
            {
                double re, im;
                size_t l = element(n1, n2, n3);
                wave(n1, n2, n3, &re, &im);
                double d = hypot(z[2 * l] / S - re, z[2 * l + 1] / S - im);
                if (d > back || isnan(d))
                    back = d;
            }

    int pass = err <= TOL && back <= TOL;
    printf("spectrum: max |X/(N1 N2 N3) - expected| = %.1e\n", err);
    printf("roundtrip: max |y/(N1 N2 N3) - x| = %.1e\n", back);
    printf("tolerance %.0e: %s\n", TOL, pass ? "PASS" : "FAIL");

    vfft_free(z);
    vfft_destroy(plan);
    return pass ? 0 : 1;
}
