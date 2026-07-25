/* blocked (Cooley-Tukey split) vs monolithic: same DAG, different
 * materialization, so they must agree exactly; and blocked must match a
 * scalar DFT. m=2 for r16/r32 (halving), m=8 for r64 (the 8x8 form). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix16_z_n1_fwd_avx2) DECL(radix16_z_n1b_fwd_avx2)
DECL(radix32_z_n1_fwd_avx2) DECL(radix32_z_n1b_fwd_avx2)
DECL(radix64_z_n1_fwd_avx2) DECL(radix64_z_n1b_fwd_avx2)
typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long,
                    unsigned long long);
#define PI 3.14159265358979323846
static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}
static int run(int R, kfn mono, kfn blk)
{
    const size_t count = 32, Ls = count, OLs = count, nd = 2 * (size_t)R * count;
    double *in = (double *)_aligned_malloc(nd * 8, 64);
    double *a = (double *)_aligned_malloc(nd * 8, 64);
    double *b = (double *)_aligned_malloc(nd * 8, 64);
    unsigned seed = 5 + R; int ok = 1;
    for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);
    mono(in, 0, a, 0, 0, 0, Ls, 0, OLs, 0, count);
    blk (in, 0, b, 0, 0, 0, Ls, 0, OLs, 0, count);
    double dm = 0, sc = 0;
    for (size_t i = 0; i < nd; i++) {
        double d = fabs(a[i] - b[i]); if (d > dm) dm = d;
        if (fabs(a[i]) > sc) sc = fabs(a[i]);
    }
    printf("r%-3d blocked-vs-monolithic  max|d|=%.3e  %s%s\n", R, dm,
           dm <= 1e-14 * sc ? "PASS" : "FAIL", dm == 0.0 ? "  (BIT-IDENTICAL)" : "");
    if (dm > 1e-14 * sc) ok = 0;
    dm = 0;
    for (size_t c = 0; c < count; c++)
        for (int k = 0; k < R; k++) {
            double sr = 0, si = 0;
            for (int m = 0; m < R; m++) {
                double th = -2.0 * PI * (double)(m * k) / (double)R;
                double xr = in[2 * ((size_t)m * Ls + c)], xi = in[2 * ((size_t)m * Ls + c) + 1];
                sr += xr * cos(th) - xi * sin(th);
                si += xr * sin(th) + xi * cos(th);
            }
            double gr = b[2 * ((size_t)k * OLs + c)], gi = b[2 * ((size_t)k * OLs + c) + 1];
            if (fabs(gr - sr) > dm) dm = fabs(gr - sr);
            if (fabs(gi - si) > dm) dm = fabs(gi - si);
        }
    printf("r%-3d blocked-vs-scalar      max|d|=%.3e  %s\n", R, dm, dm <= 1e-12 ? "PASS" : "FAIL");
    if (dm > 1e-12) ok = 0;
    _aligned_free(in); _aligned_free(a); _aligned_free(b);
    return ok;
}
int main(void)
{
    int ok = 1;
    ok &= run(16, radix16_z_n1_fwd_avx2, radix16_z_n1b_fwd_avx2);
    ok &= run(32, radix32_z_n1_fwd_avx2, radix32_z_n1b_fwd_avx2);
    ok &= run(64, radix64_z_n1_fwd_avx2, radix64_z_n1b_fwd_avx2);
    printf("\n%s\n", ok ? "BLOCKED GATE: PASS" : "BLOCKED GATE: FAIL");
    return ok ? 0 : 1;
}
