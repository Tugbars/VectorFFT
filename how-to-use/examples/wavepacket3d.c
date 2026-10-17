/*
 * wavepacket3d.c -- a free quantum wave packet advanced by one kinetic step of a split-step
 * Fourier solver: a rank-3 C2C forward transform, an all-pass phase filter and the backward
 * transform of the same plan, in place in one 64 x 64 x 64 complex buffer.
 *
 * The sequence. A Gaussian packet of width s = 1 and momentum p = (1, -1/2, 1/4), centred at c
 * in a periodic box of side L = 24,
 *
 *     psi_0(r) = exp(-|r - c|^2 / (2 s^2) + j p.(r - c)),
 *
 * is sampled at N = 64 points a side, on the grid r = n L / N with n = (n1, n2, n3):
 *
 *     x[n1,n2,n3] = psi_0(n L / N),   0 <= n1, n2, n3 <= N - 1.                                 (1)
 *
 * The buffer is row-major, n3 fastest: x[n1,n2,n3] at (n1 N + n2) N + n3, each sample an
 * adjacent (re, im) pair, which is the layout of a C double complex array. In the code, I is
 * the imaginary unit of <complex.h>, the j of these equations.
 *
 * The transforms. With W_N = e^{-j 2 pi / N} and k.n = k1 n1 + k2 n2 + k3 n3, VFFT_FORWARD is
 *
 *     X[k1,k2,k3] = sum_{n1=0}^{N-1} sum_{n2=0}^{N-1} sum_{n3=0}^{N-1} x[n1,n2,n3] W_N^{k.n},   (2)
 *
 * the full spectrum (x[n] is complex, so X[k] has no symmetry). VFFT_BACKWARD is the sum of
 * X[k] W_N^{-k.n} over k without the 1/N^3 of the inverse DFT: a forward and a backward return
 * N^3 x[n]. The program folds the 1/N^3 into the filter.
 *
 * The step. The free Schroedinger equation j d(psi)/dt = -(1/2) lap(psi) (hbar = m = 1) is
 * diagonal in k. Bin k_i <= N/2 is the frequency kappa_i = 2 pi k_i / L, bin k_i > N/2 the
 * negative frequency kappa_i = 2 pi (k_i - N) / L (H depends on kappa_i^2 only, so the sign
 * given to the Nyquist bin k_i = N/2 does not matter), and a step of length t is the filter
 *
 *     Y[k] = H[k] X[k],   H[k] = e^{-j t |kappa|^2 / 2},   |H[k]| = 1,                          (3)
 *
 * and the packet after the step is the inverse DFT y[n] = (1/N^3) sum_k Y[k] W_N^{-k.n}.
 *
 * The checks. H is all-pass, so the energy is unchanged (Parseval's relation):
 *
 *     sum_n |y[n]|^2 = (1/N^3) sum_k |Y[k]|^2 = (1/N^3) sum_k |X[k]|^2 = sum_n |x[n]|^2,        (4)
 *
 * and the free Gaussian packet has a closed form, with a = 1 + j t / s^2,
 *
 *     psi_t(r) = a^{-3/2} exp(-|r - c - p t|^2 / (2 s^2 a) + j p.(r - c) - j |p|^2 t / 2),      (5)
 *
 * the packet moved by p t and spread: y[n] must equal psi_t(n L / N) at every grid point.
 *
 * Build it against an installed VectorFFT (how-to-use/ is the guide), with CMake:
 *
 *     cmake -S how-to-use/examples -B build-examples -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>
 *     cmake --build build-examples
 *
 * or with one compiler line, the compiler that built the library (MinGW: add -static, or
 * ship libwinpthread-1.dll beside the program; Linux: add -pthread):
 *
 *     gcc -O2 wavepacket3d.c -I<prefix>/include -L<prefix>/lib -lvfft -lm -o wavepacket3d
 *
 * Run it with a store of its own, an existing directory (PowerShell: $env:VFFT_WISDOM_DIR="<dir>"):
 *
 *     VFFT_WISDOM_DIR=<dir> ./wavepacket3d
 *
 * A create that misses wisdom measures the cell's candidates once and saves the winner in that
 * store before it returns; later runs are served from it. In the repository the root CMake build
 * compiles it as well, and python gauntlet/build.py --src how-to-use/examples/wavepacket3d.c --vfft
 * --compile builds it beside this file.
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum
{
    N = 64 /* points a side: N^3 = 262144 complex samples, 4 MB */
};

static const double L = 24.0;                 /* the box side */
static const double S = 1.0;                  /* the packet's width s */
static const double P[3] = {1.0, -0.5, 0.25}; /* its momentum p */
static const double T = 1.0;                  /* the step t */

/* the tolerance of both checks. (4) is exact for any all-pass H: the energy moves only by the
 * rounding of the two transforms and of the two sums of N^3 terms, a few 1e-14. (5) describes
 * the packet in free space; the box adds its periodic images and the sampling aliases its
 * spectrum beyond |kappa_i| = pi N / L into the band, together 2.9e-14 at these parameters.
 * Each convention error (the bins above N/2 read as positive frequencies, the forward run
 * twice, the 1/N^3 left out) is an error of order one. */
static const double TOL = 1e-12;

static double mag2(double complex z) /* |z|^2 */
{
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

/* kappa_i^2 of bin k on one axis: k~ = k for k <= N/2 and k - N above */
static double kappa2(int k)
{
    const double kt = 2 * PI * (k <= N / 2 ? k : k - N) / L;
    return kt * kt;
}

/* eq. (5) at grid point n: the samples (1) at t = 0, the closed form after a step t */
static double complex packet(const int n[3], double t)
{
    const double complex a = 1.0 + I * t / (S * S), inv_a = conj(a) / mag2(a); /* 1/a */
    double r2 = 0.0, phase = 0.0;
    int i;
    for (i = 0; i < 3; i++)
    {
        const double u = n[i] * L / N - L / 2; /* r - c */
        r2 += (u - P[i] * t) * (u - P[i] * t);
        phase += P[i] * u - P[i] * P[i] * t / 2;
    }
    return cpow(a, -1.5) * cexp(-r2 * inv_a / (2 * S * S) + I * phase);
}

int main(void)
{
    const size_t vol = (size_t)N * N * N;
    double e0 = 0.0, e1 = 0.0, err = 0.0, de;
    double complex *psi;
    vfft_config_t c;
    vfft_plan plan;
    int n[3], k[3], pass;
    size_t i;

    /* one rank-3 C2C plan, in place; the zeroed fields are interleaved, natural order, one
     * transform, one thread and the library's wisdom store */
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.rank = 3;
    c.n[0] = c.n[1] = c.n[2] = N;
    c.placement = VFFT_PLACEMENT_IN;
    plan = vfft_create(&c);
    if (!plan)
    {
        fprintf(stderr, "wavepacket3d: vfft_create: %s\n", vfft_error());
        return 1;
    }
    psi = vfft_malloc(vol * sizeof *psi);
    if (!psi)
    {
        fprintf(stderr, "wavepacket3d: vfft_malloc: out of memory\n");
        vfft_destroy(plan);
        return 1;
    }

    for (n[0] = 0, i = 0; n[0] < N; n[0]++) /* x[n], eq. (1) */
        for (n[1] = 0; n[1] < N; n[1]++)
            for (n[2] = 0; n[2] < N; n[2]++, i++)
            {
                psi[i] = packet(n, 0.0);
                e0 += mag2(psi[i]);
            }

    vfft_execute(plan, VFFT_FORWARD, psi, psi); /* X[k], eq. (2) */
    for (k[0] = 0, i = 0; k[0] < N; k[0]++)     /* Y[k] / N^3, eq. (3) */
        for (k[1] = 0; k[1] < N; k[1]++)
            for (k[2] = 0; k[2] < N; k[2]++, i++)
            {
                const double kk = kappa2(k[0]) + kappa2(k[1]) + kappa2(k[2]); /* |kappa|^2 */
                psi[i] *= cexp(-I * T * kk / 2) / (double)vol;
            }
    vfft_execute(plan, VFFT_BACKWARD, psi, psi); /* y[n] */

    for (n[0] = 0, i = 0; n[0] < N; n[0]++) /* the checks (4) and (5) */
        for (n[1] = 0; n[1] < N; n[1]++)
            for (n[2] = 0; n[2] < N; n[2]++, i++)
            {
                const double d = cabs(psi[i] - packet(n, T));
                e1 += mag2(psi[i]);
                if (d > err || isnan(d)) /* a NaN must fail, not vanish from the max */
                    err = d;
            }

    de = fabs(e1 / e0 - 1.0);
    pass = de < TOL && err < TOL;
    printf("wavepacket3d %d^3 C2C in place, t = %g: max |y[n] - psi_t(nL/N)| = %.2e, "
           "energy change %.1e, tolerance %.0e: %s\n",
           N, T, err, de, TOL, pass ? "PASS" : "FAIL");

    vfft_free(psi);
    vfft_destroy(plan);
    return pass ? 0 : 1;
}
