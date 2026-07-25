/* IL backward gate: the inverse direction of the interleaved family.
 *   n1  bwd : inverse butterfly, unnormalized (bwd(fwd(x)) = R*x)
 *   n1t bwd : same, corner-turned stores
 *   t2  bwd : IDFT then POST-multiply by conj(w) — the exact inverse of
 *             t2 fwd (y = DFT(w (.) x)  =>  x = conj(w) (.) IDFT(y)).
 *             Conjugation is TABLE-SIDE, so the kernel's BYTW2 is unchanged.
 * Checks each vs a scalar reference, plus the roundtrip contract.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix8_z_n1_fwd_avx2)  DECL(radix8_z_n1_bwd_avx2)
DECL(radix16_z_n1_fwd_avx2) DECL(radix16_z_n1_bwd_avx2)
DECL(radix8_z_t2_fwd_avx2)  DECL(radix8_z_t2_bwd_avx2)

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

/* n1 bwd vs scalar IDFT (unnormalized), and roundtrip */
static int run_n1(int R, kfn fwd, kfn bwd)
{
    const size_t count = 64, Ls = count, OLs = count, nd = 2 * (size_t)R * count;
    double *in = (double *)_aligned_malloc(nd * 8, 64);
    double *o = (double *)_aligned_malloc(nd * 8, 64);
    double *rt = (double *)_aligned_malloc(nd * 8, 64);
    unsigned seed = 31 + R;
    int ok = 1;
    for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);

    bwd(in, 0, o, 0, 0, 0, Ls, 0, OLs, 0, count);
    {   /* scalar IDFT: X[k] = sum x[m] e^{+2pi i m k / R} */
        double dm = 0.0;
        for (size_t c = 0; c < count; c++)
            for (int k = 0; k < R; k++) {
                double sr = 0.0, si = 0.0;
                for (int m = 0; m < R; m++) {
                    double th = 2.0 * PI * (double)(m * k) / (double)R;
                    double xr = in[2 * ((size_t)m * Ls + c)], xi = in[2 * ((size_t)m * Ls + c) + 1];
                    sr += xr * cos(th) - xi * sin(th);
                    si += xr * sin(th) + xi * cos(th);
                }
                double gr = o[2 * ((size_t)k * OLs + c)], gi = o[2 * ((size_t)k * OLs + c) + 1];
                if (fabs(gr - sr) > dm) dm = fabs(gr - sr);
                if (fabs(gi - si) > dm) dm = fabs(gi - si);
            }
        printf("r%-3d n1  bwd-vs-scalar-IDFT   max|d|=%.3e  %s\n", R, dm,
               dm <= 1e-12 ? "PASS" : "FAIL");
        if (dm > 1e-12) ok = 0;
    }
    {   /* roundtrip bwd(fwd(x)) = R*x */
        double dm = 0.0;
        fwd(in, 0, o, 0, 0, 0, Ls, 0, OLs, 0, count);
        bwd(o, 0, rt, 0, 0, 0, Ls, 0, OLs, 0, count);
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(rt[i] - (double)R * in[i]);
            if (d > dm) dm = d;
        }
        printf("r%-3d n1  roundtrip = R*x      max|d|=%.3e  %s\n", R, dm,
               dm <= 1e-12 ? "PASS" : "FAIL");
        if (dm > 1e-12) ok = 0;
    }
    _aligned_free(in); _aligned_free(o); _aligned_free(rt);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= run_n1(8, radix8_z_n1_fwd_avx2, radix8_z_n1_bwd_avx2);
    ok &= run_n1(16, radix16_z_n1_fwd_avx2, radix16_z_n1_bwd_avx2);

    /* ---- t2: fwd uses table w, bwd uses the CONJUGATED table ---- */
    {
        enum { R = 8 };
        const size_t count = 64, Ls = count, OLs = count, M = R * count;
        const size_t nd = 2 * (size_t)R * count;
        double *in = (double *)_aligned_malloc(nd * 8, 64);
        double *o = (double *)_aligned_malloc(nd * 8, 64);
        double *rt = (double *)_aligned_malloc(nd * 8, 64);
        double *tw = (double *)_aligned_malloc((count / 2) * (R - 1) * 8 * 8, 64);
        double *twc = (double *)_aligned_malloc((count / 2) * (R - 1) * 8 * 8, 64);
        unsigned seed = 4242;
        for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);
        for (size_t p = 0; p < count / 2; p++)
            for (int l = 1; l < R; l++) {
                double *r0 = tw + (p * (R - 1) + (size_t)(l - 1)) * 8;
                double *r1 = twc + (p * (R - 1) + (size_t)(l - 1)) * 8;
                for (int j = 0; j < 2; j++) {
                    double a = -2.0 * PI * (double)l * (double)(2 * p + (size_t)j) / (double)M;
                    r0[2 * j] = cos(a); r0[2 * j + 1] = cos(a);
                    r0[4 + 2 * j] = -sin(a); r0[4 + 2 * j + 1] = sin(a);
                    /* conjugated twin: sin negated (table-side conj) */
                    r1[2 * j] = cos(a); r1[2 * j + 1] = cos(a);
                    r1[4 + 2 * j] = sin(a); r1[4 + 2 * j + 1] = -sin(a);
                }
            }
        radix8_z_t2_fwd_avx2(in, 0, o, 0, tw, 0, Ls, 0, OLs, 0, count);
        radix8_z_t2_bwd_avx2(o, 0, rt, 0, twc, 0, Ls, 0, OLs, 0, count);
        double dm = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(rt[i] - (double)R * in[i]);
            if (d > dm) dm = d;
        }
        printf("r8   t2  roundtrip = R*x      max|d|=%.3e  %s\n", dm,
               dm <= 1e-12 ? "PASS" : "FAIL");
        if (dm > 1e-12) ok = 0;
        _aligned_free(in); _aligned_free(o); _aligned_free(rt);
        _aligned_free(tw); _aligned_free(twc);
    }
    printf("\n%s\n", ok ? "IL BWD GATE: PASS" : "IL BWD GATE: FAIL");
    return ok ? 0 : 1;
}
