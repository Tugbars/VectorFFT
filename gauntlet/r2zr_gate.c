/* r2zr_gate.c -- the real rows kernels (radixN_z_r2zr_fwd_avx2: the real
 * N-point DFT of every row of a row-major plane, four rows per vector) against
 * a naive real DFT. Every remainder path of the row loop (2..9, 13 and 64
 * rows: the wide iteration, the two-row step, the lone last row run with the
 * row before it), at the tight pitches (Ls = N, OLs = N/2 + 1) and at padded
 * ones (Ls = N + 3, OLs = N/2 + 4) with guard values: the bins against the
 * reference, the DC and Nyquist imaginary parts exactly zero, nothing written
 * outside a row's bins. The kernels link as their own objects (each file
 * declares its own static constants).
 * Build: gcc -O2 -mavx2 -mfma r2zr_gate.c src/dag-fft-compiler/codelets/zil/avx2/real/rows/radix*_z_r2zr_avx2.c -o r2zr_gate */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define DECL(n) \
    void radix##n##_z_r2zr_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
DECL(4)
DECL(6)
DECL(8)
DECL(10)
DECL(12)
DECL(16)
DECL(32)
#undef DECL

typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
typedef struct { int n; kfn f; } cell_t;
static const cell_t cells[] = {
    { 4, radix4_z_r2zr_fwd_avx2 },
    { 6, radix6_z_r2zr_fwd_avx2 },
    { 8, radix8_z_r2zr_fwd_avx2 },
    { 10, radix10_z_r2zr_fwd_avx2 },
    { 12, radix12_z_r2zr_fwd_avx2 },
    { 16, radix16_z_r2zr_fwd_avx2 },
    { 32, radix32_z_r2zr_fwd_avx2 },
};

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
static void naive_r2c(const double *x, double *X, int n)
{
    for (int k = 0; k <= n / 2; k++)
    {
        long double re = 0, im = 0;
        for (int j = 0; j < n; j++)
        {
            long double a = -2.0L * 3.14159265358979323846264338327950288L * (long double)k * (long double)j / (long double)n;
            re += x[j] * cosl(a); im += x[j] * sinl(a);
        }
        X[2 * k] = (double)re; X[2 * k + 1] = (double)im;
    }
}

#define GUARD 7.25e77
int main(void)
{
    static const int counts[] = { 2, 3, 4, 5, 6, 7, 8, 9, 13, 64 };
    unsigned seed = 0x2468u;
    int fails = 0;
    for (size_t c = 0; c < sizeof cells / sizeof cells[0]; c++)
    {
        const int n = cells[c].n, h = n / 2;
        double worst = 0;
        int zero_ok = 1, guard_ok = 1;
        for (int pad = 0; pad < 2; pad++)
            for (size_t ci = 0; ci < sizeof counts / sizeof counts[0]; ci++)
            {
                const int cnt = counts[ci];
                const size_t Ls = (size_t)n + (pad ? 3 : 0), OLs = (size_t)h + 1 + (pad ? 3 : 0);
                const size_t nx = (size_t)cnt * Ls + 8, nz = 2 * ((size_t)cnt * OLs + 4);
                double *x = (double *)malloc(nx * sizeof(double)), *Z = (double *)malloc(nz * sizeof(double));
                double ref[66];
                for (size_t i = 0; i < nx; i++) x[i] = urand(&seed);
                for (size_t i = 0; i < nz; i++) Z[i] = GUARD;
                cells[c].f(x, 0, Z, 0, 0, 0, Ls, 0, OLs, 0, (size_t)cnt);
                for (int r = 0; r < cnt; r++)
                {
                    double sc = 0, e = 0;
                    naive_r2c(x + (size_t)r * Ls, ref, n);
                    for (int k = 0; k <= h; k++)
                    {
                        const double *z = Z + 2 * ((size_t)r * OLs + (size_t)k);
                        const double dr = fabs(z[0] - ref[2 * k]), di = fabs(z[1] - ((k == 0 || k == h) ? 0.0 : ref[2 * k + 1]));
                        if (dr > e) e = dr;
                        if (di > e) e = di;
                        if (fabs(ref[2 * k]) > sc) sc = fabs(ref[2 * k]);
                        if (fabs(ref[2 * k + 1]) > sc) sc = fabs(ref[2 * k + 1]);
                    }
                    e /= sc > 0 ? sc : 1;
                    if (e > worst) worst = e;
                    if (Z[2 * ((size_t)r * OLs) + 1] != 0.0 || Z[2 * ((size_t)r * OLs + (size_t)h) + 1] != 0.0) zero_ok = 0;
                    for (size_t k = (size_t)h + 1; k < OLs; k++)
                        if (Z[2 * ((size_t)r * OLs + k)] != GUARD || Z[2 * ((size_t)r * OLs + k) + 1] != GUARD) guard_ok = 0;
                }
                for (size_t i = 2 * (size_t)cnt * OLs; i < nz; i++)
                    if (Z[i] != GUARD) guard_ok = 0;
                free(x); free(Z);
            }
        const int ok = worst < 1e-13 && zero_ok && guard_ok;
        if (!ok) fails++;
        printf("N=%-3d rows 2..64, tight and padded pitches: r2c %.1e | DC/Nyquist imaginary %s | guards %s  %s\n",
               n, worst, zero_ok ? "zero" : "NONZERO", guard_ok ? "intact" : "WRITTEN", ok ? "PASS" : "FAIL");
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
