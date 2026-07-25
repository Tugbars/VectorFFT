/* CIL-1 multi-radix gate: interleaved n1 (shared scheduler) vs legacy
 * hand-scheduled kernel AND vs scalar DFT, for radix 4 / 8 / 16. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix4_z_n1ref_fwd_avx2)  DECL(radix4_z_n1_fwd_avx2)
DECL(radix8_z_n1ref_fwd_avx2)  DECL(radix8_z_n1_fwd_avx2)
DECL(radix16_z_n1ref_fwd_avx2) DECL(radix16_z_n1_fwd_avx2)

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long,
                    unsigned long long);

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

static int run(int R, kfn leg, kfn pipe)
{
    const size_t count = 64, Ls = count, OLs = count;
    const size_t nd = 2 * (size_t)R * count;
    double *in = (double *)_aligned_malloc(nd * 8, 64);
    double *o_l = (double *)_aligned_malloc(nd * 8, 64);
    double *o_p = (double *)_aligned_malloc(nd * 8, 64);
    unsigned seed = 11 + R;
    int ok = 1;

    for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);
    leg(in, 0, o_l, 0, 0, 0, Ls, 0, OLs, 0, count);
    pipe(in, 0, o_p, 0, 0, 0, Ls, 0, OLs, 0, count);
    {
        double dmax = 0.0, amax = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(o_l[i] - o_p[i]), a = fabs(o_l[i]);
            if (d > dmax) dmax = d;
            if (a > amax) amax = a;
        }
        printf("r%-3d pipeline-vs-legacy  max|d|=%.3e  %s%s\n", R, dmax,
               dmax <= 1e-14 * amax ? "PASS" : "FAIL",
               dmax == 0.0 ? "  (BIT-IDENTICAL)" : "");
        if (dmax > 1e-14 * amax) ok = 0;
    }
    {
        double dmax = 0.0;
        for (size_t k = 0; k < count; k++)
            for (int l = 0; l < R; l++) {
                double sr = 0.0, si = 0.0;
                for (int m = 0; m < R; m++) {
                    double th = -2.0 * 3.14159265358979323846 * (double)(m * l) / (double)R;
                    double c = cos(th), s = sin(th);
                    double xr = in[2 * ((size_t)m * Ls + k)];
                    double xi = in[2 * ((size_t)m * Ls + k) + 1];
                    sr += xr * c - xi * s;
                    si += xr * s + xi * c;
                }
                double gr = o_p[2 * ((size_t)l * OLs + k)];
                double gi = o_p[2 * ((size_t)l * OLs + k) + 1];
                if (fabs(gr - sr) > dmax) dmax = fabs(gr - sr);
                if (fabs(gi - si) > dmax) dmax = fabs(gi - si);
            }
        printf("r%-3d pipeline-vs-scalar  max|d|=%.3e  %s\n", R, dmax,
               dmax <= 1e-12 ? "PASS" : "FAIL");
        if (dmax > 1e-12) ok = 0;
    }
    _aligned_free(in); _aligned_free(o_l); _aligned_free(o_p);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= run(4, radix4_z_n1ref_fwd_avx2, radix4_z_n1_fwd_avx2);
    ok &= run(8, radix8_z_n1ref_fwd_avx2, radix8_z_n1_fwd_avx2);
    ok &= run(16, radix16_z_n1ref_fwd_avx2, radix16_z_n1_fwd_avx2);
    printf("\n%s\n", ok ? "CIL-1 MULTI-RADIX GATE: PASS" : "CIL-1 MULTI-RADIX GATE: FAIL");
    return ok ? 0 : 1;
}
