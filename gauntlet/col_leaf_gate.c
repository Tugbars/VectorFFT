/* col_leaf_gate.c -- the blocked column leaves (radixN_z_n1cb*_avx2: the
 * N-point complex DFT down every column of a row-major plane, two columns per
 * vector) against a naive DFT: the two-pass leaves at 32 and 64 in both
 * directions, their half-store twins, and the 128 leaves (two-pass b816,
 * three-pass b448). Every remainder path of the column loop -- 1..5, 8, 9 and
 * 17 columns: the wide iteration and the lone column, which re-runs the
 * blocked passes at VEX-128 -- at the tight pitch (the count) and at a padded
 * one with guard values; out of place and in place (the column pass runs
 * zin == zout). The bins against the reference, nothing written outside the
 * counted columns. The kernels link as their own objects (each file declares
 * its own static constants).
 * Build: gcc -O2 -mavx2 -mfma col_leaf_gate.c
 *          src/dag-fft-compiler/codelets/zil/avx2/shared/col/blocked/radix*_z_n1cb*_avx2.c
 *          src/dag-fft-compiler/codelets/zil/avx2/shared/col/half/radix*_z_n1cb*h_avx2.c -o col_leaf_gate */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#define DECL(sym) \
    void sym(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
DECL(radix32_z_n1cb48_fwd_avx2)
DECL(radix32_z_n1cb48_bwd_avx2)
DECL(radix32_z_n1cb84_fwd_avx2)
DECL(radix32_z_n1cb84_bwd_avx2)
DECL(radix64_z_n1cb88_fwd_avx2)
DECL(radix64_z_n1cb88_bwd_avx2)
DECL(radix32_z_n1cb48h_fwd_avx2)
DECL(radix32_z_n1cb84h_fwd_avx2)
DECL(radix64_z_n1cb88h_fwd_avx2)
DECL(radix128_z_n1cb816_fwd_avx2)
DECL(radix128_z_n1cb448_fwd_avx2)
#undef DECL

typedef struct { const char *name; int n, bwd; kfn f; } cell_t;
static const cell_t cells[] = {
    { "32 b48 fwd", 32, 0, radix32_z_n1cb48_fwd_avx2 },
    { "32 b48 bwd", 32, 1, radix32_z_n1cb48_bwd_avx2 },
    { "32 b84 fwd", 32, 0, radix32_z_n1cb84_fwd_avx2 },
    { "32 b84 bwd", 32, 1, radix32_z_n1cb84_bwd_avx2 },
    { "64 b88 fwd", 64, 0, radix64_z_n1cb88_fwd_avx2 },
    { "64 b88 bwd", 64, 1, radix64_z_n1cb88_bwd_avx2 },
    { "32 b48h fwd", 32, 0, radix32_z_n1cb48h_fwd_avx2 },
    { "32 b84h fwd", 32, 0, radix32_z_n1cb84h_fwd_avx2 },
    { "64 b88h fwd", 64, 0, radix64_z_n1cb88h_fwd_avx2 },
    { "128 b816 fwd", 128, 0, radix128_z_n1cb816_fwd_avx2 },
    { "128 b448 fwd", 128, 0, radix128_z_n1cb448_fwd_avx2 },
};

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
/* column k of an n-row plane at pitch p (complex): X[f] = sum_j x[j] e^{-/+ 2 pi i f j / n} */
static void naive_col(const double *x, double *X, int n, size_t p, size_t k, int bwd)
{
    for (int f = 0; f < n; f++)
    {
        long double re = 0, im = 0;
        for (int j = 0; j < n; j++)
        {
            const long double a = (bwd ? 2.0L : -2.0L) * 3.14159265358979323846264338327950288L *
                                  (long double)((long long)f * j % n) / (long double)n;
            const long double c = cosl(a), s = sinl(a);
            const long double xr = x[2 * ((size_t)j * p + k)], xi = x[2 * ((size_t)j * p + k) + 1];
            re += xr * c - xi * s; im += xr * s + xi * c;
        }
        X[2 * ((size_t)f * p + k)] = (double)re; X[2 * ((size_t)f * p + k) + 1] = (double)im;
    }
}

#define GUARD 7.25e77
int main(void)
{
    static const int counts[] = { 1, 2, 3, 4, 5, 8, 9, 17 };
    int fails = 0, checks = 0;
    double worst = 0;
    for (size_t ci = 0; ci < sizeof cells / sizeof cells[0]; ci++)
    {
        const int n = cells[ci].n;
        double cw = 0;
        int cf = 0;
        for (size_t ki = 0; ki < sizeof counts / sizeof counts[0]; ki++)
            for (int pad = 0; pad < 2; pad++)
                for (int inplace = 0; inplace < 2; inplace++)
                {
                    const size_t cnt = (size_t)counts[ki], p = cnt + (pad ? 3 : 0), nz = 2 * (size_t)n * p;
                    double *x = (double *)_mm_malloc((nz + 8) * sizeof(double), 64);
                    double *y = (double *)_mm_malloc((nz + 8) * sizeof(double), 64);
                    double *r = (double *)_mm_malloc((nz + 8) * sizeof(double), 64);
                    unsigned sd = 0x1234567u + (unsigned)(n * 131 + (int)cnt * 17 + pad);
                    double e = 0, m = 0;
                    int bad = 0;
                    for (size_t i = 0; i < nz; i++) x[i] = urand(&sd);
                    for (size_t i = 0; i < nz; i++) r[i] = GUARD;
                    for (size_t k = 0; k < cnt; k++) naive_col(x, r, n, p, k, cells[ci].bwd);
                    if (inplace)
                    {   /* the pad columns hold the input's own values: they must come back untouched */
                        memcpy(y, x, nz * sizeof(double));
                        cells[ci].f(y, NULL, y, NULL, NULL, NULL, p, 0, p, 0, cnt);
                    }
                    else
                    {
                        for (size_t i = 0; i < nz; i++) y[i] = GUARD;
                        cells[ci].f(x, NULL, y, NULL, NULL, NULL, p, 0, p, 0, cnt);
                    }
                    for (size_t j = 0; j < (size_t)n; j++)
                        for (size_t k = 0; k < p; k++)
                            for (int q = 0; q < 2; q++)
                            {
                                const size_t i = 2 * (j * p + k) + (size_t)q;
                                if (k < cnt)
                                {
                                    const double d = fabs(y[i] - r[i]);
                                    if (d > e) e = d;
                                    if (fabs(r[i]) > m) m = fabs(r[i]);
                                }
                                else if (y[i] != (inplace ? x[i] : GUARD))
                                    bad = 1; /* a store outside the counted columns */
                            }
                    e = m > 0 ? e / m : e;
                    checks++;
                    if (e > cw) cw = e;
                    if (!(e < 1e-13) || bad)
                    {
                        cf++;
                        printf("  FAIL %-13s count %2zu pitch %2zu %s: rel %.2e%s\n", cells[ci].name, cnt, p,
                               inplace ? "in place" : "out of place", e, bad ? ", WROTE OUTSIDE ITS COLUMNS" : "");
                    }
                    _mm_free(x); _mm_free(y); _mm_free(r);
                }
        printf("%-13s worst rel %.2e  %s\n", cells[ci].name, cw, cf ? "FAIL" : "ok");
        fails += cf;
        if (cw > worst) worst = cw;
    }
    printf("%d checks, %d failed, worst rel %.2e -> %s\n", checks, fails, worst, fails ? "FAIL" : "ALL PASS");
    return fails != 0;
}
