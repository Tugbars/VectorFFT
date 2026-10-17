/*
 * convolve3d.c -- linear convolution by zero padding: 3D R2C, multiply, C2R.
 *
 * A volume x[n1,n2,n3], L = 56 samples on each axis, is filtered by a point-spread function
 * h[n1,n2,n3], P = 9 samples on each axis. Their linear convolution
 *
 *     y[n1,n2,n3] = sum_{m1,m2,m3=0}^{P-1} h[m1,m2,m3] x[n1-m1, n2-m2, n3-m3]
 *
 * has length L + P - 1 = 64 on each axis: it is zero outside 0 <= n1, n2, n3 < 64. A product
 * of DFTs gives the N-point CIRCULAR convolution instead, whose indices are taken modulo N,
 * x[((n - m))_N]. The two agree when both sequences are zero-padded to N >= L + P - 1 samples
 * on each axis, so that no term wraps around. Here N = 64.
 *
 * The library's transforms, with W_N = e^{-j 2 pi / N} and k.n = k1 n1 + k2 n2 + k3 n3:
 *
 *     VFFT_FORWARD    X[k1,k2,k3] = sum_{n1,n2,n3=0}^{N-1} x[n1,n2,n3] W_N^{k.n}
 *     VFFT_BACKWARD   sum_{k1,k2,k3=0}^{N-1} X[k1,k2,k3] W_N^{-k.n} = N^3 x[n1,n2,n3]
 *
 * VFFT_BACKWARD is unnormalized, so the program folds the 1/N^3 into the product:
 *
 *     y[n1,n2,n3] = (1/N^3) sum_{k1,k2,k3=0}^{N-1} X[k1,k2,k3] H[k1,k2,k3] W_N^{-k.n}.
 *
 * x, h and y are real, so R2C keeps only k3 = 0..N/2 (N/2 + 1 = 33 bins of (re, im) pairs,
 * the spectrum row-major N x N x 33); the other bins are X[k] = X*[((-k))_N]. The product of
 * two such spectra has the same symmetry, and C2R returns the real y.
 *
 * The check compares all N^3 outputs with the direct sum. h is not symmetric, so the check
 * also tells convolution from correlation: X*[k] H[k] gives the circular cross-correlation
 * sum_n x[n] h[((n + m))_N] instead.
 *
 * Build from the repository root (the program lands beside this file), then run it:
 *
 *     python gauntlet/build.py --src examples/convolve3d.c --vfft --compile
 *     examples/convolve3d.exe        (examples/convolve3d on Linux)
 *
 * A create that misses wisdom measures the cell's candidates once and saves the winner before
 * it returns; later runs are served from the store. VFFT_WISDOM_DIR=<dir> makes <dir> the
 * store instead of the library's own.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vfft.h"

enum
{
    L = 56,        /* the volume's samples per axis */
    P = 9,         /* the point-spread function's */
    N = 64,        /* the DFT's: N >= L + P - 1 */
    NB = N / 2 + 1 /* the bins R2C keeps on the last axis */
};

/* rounding in the three transforms is of order eps log2(N^3) max|y| = 2.2e-16 x 18 max|y|,
 * about 4e-15 max|y|; a wrap-around, a correlation or a missing 1/N^3 errs by the order of
 * max|y| itself. The tolerance is relative to max|y|. */
static const double TOL = 1e-12;

/* the row-major index of (n1, n2, n3) on the N^3 grid, the last axis contiguous */
static size_t idx(int n1, int n2, int n3)
{
    return ((size_t)n1 * N + (size_t)n2) * N + (size_t)n3;
}

/* a uniform draw in [-1, 1): a 32-bit LCG, the same sequence on every platform */
static double urand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / 8388608.0 - 1.0;
}

/* a[n] random on 0 <= n1, n2, n3 < len and zero on the rest of the N^3 grid: the padding */
static void fill_padded(double *a, int len, uint32_t *s)
{
    int n1, n2, n3;
    for (n1 = 0; n1 < N; n1++)
        for (n2 = 0; n2 < N; n2++)
            for (n3 = 0; n3 < N; n3++)
                a[idx(n1, n2, n3)] = (n1 < len && n2 < len && n3 < len) ? urand(s) : 0.0;
}

/* one N^3 real transform; the zeroed config is interleaved, out of place, natural order,
 * one thread, one transform */
static vfft_plan create3d(vfft_transform_t t)
{
    vfft_config_t c;
    vfft_plan p;
    memset(&c, 0, sizeof c);
    c.transform = t;
    c.rank = 3;
    c.n[0] = c.n[1] = c.n[2] = N;
    p = vfft_create(&c);
    if (!p)
        fprintf(stderr, "convolve3d: vfft_create(%s): %s\n",
                t == VFFT_R2C ? "VFFT_R2C" : "VFFT_C2R", vfft_error());
    return p;
}

int main(void)
{
    const size_t nr = (size_t)N * N * N, nc = (size_t)N * N * NB;
    vfft_plan r2c = create3d(VFFT_R2C);
    vfft_plan c2r = r2c ? create3d(VFFT_C2R) : NULL;
    double *xp = vfft_malloc(nr * sizeof(double)), *hp = vfft_malloc(nr * sizeof(double));
    double *X = vfft_malloc(2 * nc * sizeof(double)), *H = vfft_malloc(2 * nc * sizeof(double));
    double *y = vfft_malloc(nr * sizeof(double));
    double err = 0.0, ymax = 0.0;
    uint32_t seed = 2026u;
    int pass = 0, n1, n2, n3, m1, m2, m3;
    size_t i;

    if (r2c && c2r && xp && hp && X && H && y)
    {
        fill_padded(xp, L, &seed);
        fill_padded(hp, P, &seed);
        vfft_execute(r2c, VFFT_FORWARD, xp, X);
        vfft_execute(r2c, VFFT_FORWARD, hp, H);
        for (i = 0; i < nc; i++)
        {
            /* Y[k] = X[k] H[k] / N^3, into X: the inverse DFT's 1/N^3, which VFFT_BACKWARD
             * leaves out */
            const double a = X[2 * i], b = X[2 * i + 1], c = H[2 * i], d = H[2 * i + 1];
            X[2 * i] = (a * c - b * d) / (double)nr;
            X[2 * i + 1] = (a * d + b * c) / (double)nr;
        }
        vfft_execute(c2r, VFFT_BACKWARD, X, y);

        /* the direct sum, from xp and hp: an out-of-place R2C leaves its input as it was. The
         * terms with n_i < m_i are left out and x[n - m] is zero past L, so no index wraps */
        for (n1 = 0; n1 < N; n1++)
            for (n2 = 0; n2 < N; n2++)
                for (n3 = 0; n3 < N; n3++)
                {
                    double s = 0.0, e;
                    for (m1 = 0; m1 < P && m1 <= n1; m1++)
                        for (m2 = 0; m2 < P && m2 <= n2; m2++)
                            for (m3 = 0; m3 < P && m3 <= n3; m3++)
                                s += hp[idx(m1, m2, m3)] * xp[idx(n1 - m1, n2 - m2, n3 - m3)];
                    e = fabs(y[idx(n1, n2, n3)] - s);
                    if (e > err || isnan(e)) /* a NaN must fail, not vanish from the max */
                        err = e;
                    if (fabs(s) > ymax)
                        ymax = fabs(s);
                }
        pass = err <= TOL * ymax;
        printf("convolve3d %d^3 * %d^3 on %d^3: max |y - direct| = %.2e, max |y| = %.1f, "
               "ratio %.2e, tolerance %.0e: %s\n",
               L, P, N, err, ymax, err / ymax, TOL, pass ? "PASS" : "FAIL");
    }
    else if (r2c && c2r)
        fprintf(stderr, "convolve3d: vfft_malloc: out of memory\n");

    vfft_free(xp);
    vfft_free(hp);
    vfft_free(X);
    vfft_free(H);
    vfft_free(y);
    vfft_destroy(r2c);
    vfft_destroy(c2r);
    return pass ? 0 : 1;
}
