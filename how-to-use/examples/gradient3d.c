/*
 * gradient3d.c -- the spectral gradient of a periodic field on a 64^3 grid, in place with
 * padded rows: one 3D R2C, then for each axis a multiply by j kappa and one 3D C2R.
 *
 * A pseudo-spectral flow solver keeps each real field in one padded array and takes the
 * derivatives of its nonlinear term this way: one forward transform of the field, one
 * backward transform per derivative. The program checks all three derivatives against the
 * analytic ones.
 *
 * Convention (include/vfft.h). For x[n1,n2,n3], 0 <= n_i < N, and W_N = e^{-j 2pi/N}, the
 * forward VFFT_R2C computes the half spectrum of the last axis,
 *
 *     X[k1,k2,k3] = sum_{n1,n2,n3} x[n1,n2,n3] W_N^{k1 n1 + k2 n2 + k3 n3},  0 <= k3 <= N/2;
 *
 * the other bins follow from X[((-k1))_N, ((-k2))_N, ((-k3))_N] = X*[k1,k2,k3]. The backward
 * VFFT_C2R computes sum_k X[k] W_N^{-(k1 n1 + k2 n2 + k3 n3)} over the whole spectrum: it is
 * unnormalized, N^3 times the inverse DFT, so the factor 1/N^3 is the caller's.
 *
 * Layout. In place (VFFT_PLACEMENT_IN, out == in) each row of N reals is padded to
 * 2 (N/2 + 1) doubles, and the row's N/2 + 1 bins land over its reals as (re, im) pairs:
 * row (n1, n2) of the field and row (k1, k2) of the spectrum begin (n1 N + n2) stride and
 * (k1 N + k2) stride doubles into the buffer. vfft_info() gives the pitch (stride) and the
 * buffer's size (in_doubles) of each plan; read them, never compute them. An in-place C2R
 * overwrites its input, so each derivative runs on a copy of the spectrum; a plan holds none
 * of the caller's buffers, so the copy is a second buffer.
 *
 * Derivative. With kappa = k~ (k~ = k for k <= N/2, k - N above), d/dx_a multiplies bin k by
 * j kappa_a. A real derivative needs G_a[((-k))_N] = G_a*[k], as U has; the imaginary
 * multiplier keeps it by being odd, but the plane k_a = N/2 is its own mirror
 * (-N/2 = N/2 mod N), so there the multiplier must equal its own conjugate, that is, be real,
 * and j kappa_a is real only as 0. With that Nyquist plane zeroed,
 *
 *     G_a[k] = j kappa_a U[k]  (k_a != N/2),   G_a[k] = 0  (k_a = N/2),
 *     d_a u[n1,n2,n3] = (1/N^3) sum_k G_a[k] W_N^{-(k1 n1 + k2 n2 + k3 n3)}.
 *
 * Check. u[n1,n2,n3] = sin x1 cos 2x2 sin 3x3 + (1/4)(-1)^{n2} cos x3 at the grid points
 * x_i = n_i h, h = 2pi/N. The second term lives on the Nyquist plane k2 = N/2: on the grid
 * its derivative along x2 is 0, which the zeroed plane gives, and along x3 it is
 * -(1/4)(-1)^{n2} sin x3. Keeping the plane, with kappa = N/2 there, would add
 * -(N/8)(-1)^{n2} sin x3 to d/dx2 (its negative with kappa = -N/2): an error of 8. Since
 * j kappa is odd, a flipped transform sign negates each derivative, and a wrong pitch
 * scrambles them.
 *
 * Build it against an installed VectorFFT (how-to-use/ is the guide), with CMake:
 *
 *     cmake -S how-to-use/examples -B build-examples -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>
 *     cmake --build build-examples
 *
 * or with one compiler line, the compiler that built the library (MinGW: add -static, or
 * ship libwinpthread-1.dll beside the program; Linux: add -pthread):
 *
 *     gcc -O3 gradient3d.c -I<prefix>/include -L<prefix>/lib -lvfft -lm -o gradient3d
 *
 * Run it with a store of its own, an existing directory (PowerShell: $env:VFFT_WISDOM_DIR="<dir>"):
 *
 *     VFFT_WISDOM_DIR=<dir> ./gradient3d
 *
 * A create that misses wisdom measures the cell's candidates once and saves the winner in that
 * store before it returns; later runs are served from it. In the repository the root CMake build
 * compiles it as well, and python gauntlet/build.py --src how-to-use/examples/gradient3d.c --vfft
 * --compile builds it beside this file.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum
{
    N = 64 /* N1 = N2 = N3 */
};

/* the rounding of two 64^3 transforms, raised up to N/2 times by the multiply, is a few
 * 1e-14; each defect the field is built to catch is an error of order 1 */
static const double TOL = 1e-12;

/* u[n1,n2,n3]; du[a] = its exact derivative along x_{a+1} */
static double field(int n1, int n2, int n3, double du[3])
{
    const double h = 2 * PI / N, x1 = n1 * h, x2 = n2 * h, x3 = n3 * h;
    const double nyq = (n2 & 1) ? -0.25 : 0.25; /* (1/4)(-1)^{n2} */
    du[0] = cos(x1) * cos(2 * x2) * sin(3 * x3);
    du[1] = -2 * sin(x1) * sin(2 * x2) * sin(3 * x3);
    du[2] = 3 * sin(x1) * cos(2 * x2) * cos(3 * x3) - nyq * sin(x3);
    return sin(x1) * cos(2 * x2) * sin(3 * x3) + nyq * cos(x3);
}

/* one in-place plan of the cube; the zeroed rest of the config is interleaved, natural
 * order, one transform, one thread */
static vfft_plan plan_in_place(vfft_transform_t t)
{
    vfft_config_t c;
    vfft_plan p;
    memset(&c, 0, sizeof c);
    c.transform = t;
    c.rank = 3;
    c.n[0] = c.n[1] = c.n[2] = N;
    c.placement = VFFT_PLACEMENT_IN;
    p = vfft_create(&c);
    if (!p)
        fprintf(stderr, "gradient3d: vfft_create(%s): %s\n",
                t == VFFT_R2C ? "VFFT_R2C" : "VFFT_C2R", vfft_error());
    return p;
}

/* the half spectrum times j kappa_a / N^3, and 0 on the Nyquist plane k_a = N/2 */
static void times_j_kappa(double *G, size_t stride, int a)
{
    const double scale = 1.0 / ((double)N * N * N);
    int k[3];
    for (k[0] = 0; k[0] < N; k[0]++)
        for (k[1] = 0; k[1] < N; k[1]++)
            for (k[2] = 0; k[2] <= N / 2; k[2]++)
            {
                double *z = G + ((size_t)k[0] * N + k[1]) * stride + 2 * (size_t)k[2];
                const double g = k[a] == N / 2 ? 0.0 : (k[a] <= N / 2 ? k[a] : k[a] - N) * scale;
                const double re = z[0];
                z[0] = -g * z[1]; /* (re + j im) j g */
                z[1] = g * re;
            }
}

int main(void)
{
    vfft_plan r2c = plan_in_place(VFFT_R2C), c2r = plan_in_place(VFFT_C2R);
    vfft_info_t fwd, bwd;
    double *U = NULL, *G = NULL, err[3] = {0, 0, 0}, du[3], e;
    size_t s = 0, row;
    int a, n1, n2, n3, pass = 0;

    if (r2c && c2r)
    {
        vfft_info(r2c, &fwd);
        vfft_info(c2r, &bwd);
        if (fwd.stride != bwd.stride || fwd.in_doubles != bwd.in_doubles)
            fprintf(stderr, "gradient3d: the plans disagree: pitch %zu and %zu, size %zu and %zu\n",
                    fwd.stride, bwd.stride, fwd.in_doubles, bwd.in_doubles);
        else
        {
            s = fwd.stride; /* no VFFT_FLAG_OWN_PITCH: the tight pitch, 2 (N/2 + 1) */
            U = (double *)vfft_malloc(fwd.in_doubles * sizeof(double));
            G = (double *)vfft_malloc(fwd.in_doubles * sizeof(double));
            if (!U || !G)
                fprintf(stderr, "gradient3d: vfft_malloc: out of memory\n");
        }
    }
    if (U && G)
    {
        /* the N reals of each row; the pad doubles from n3 = N to s - 1 are not read by the
         * R2C, so they need no value */
        for (n1 = 0; n1 < N; n1++)
            for (n2 = 0; n2 < N; n2++)
                for (n3 = 0; n3 < N; n3++)
                    U[((size_t)n1 * N + n2) * s + n3] = field(n1, n2, n3, du);
        vfft_execute(r2c, VFFT_FORWARD, U, U);
        for (a = 0; a < 3; a++)
        {
            memcpy(G, U, fwd.in_doubles * sizeof(double)); /* the C2R consumes its input */
            times_j_kappa(G, s, a);
            vfft_execute(c2r, VFFT_BACKWARD, G, G);
            for (n1 = 0; n1 < N; n1++)
                for (n2 = 0; n2 < N; n2++)
                    for (n3 = 0, row = ((size_t)n1 * N + n2) * s; n3 < N; n3++)
                    {
                        field(n1, n2, n3, du);
                        e = fabs(G[row + n3] - du[a]);
                        if (e > err[a] || isnan(e)) /* a NaN must fail, not vanish from the max */
                            err[a] = e;
                    }
        }
        pass = err[0] <= TOL && err[1] <= TOL && err[2] <= TOL;
        printf("gradient3d %dx%dx%d in place, row pitch %zu doubles: max |error| d/dx1 %.2e, "
               "d/dx2 %.2e, d/dx3 %.2e, tolerance %.0e: %s\n",
               N, N, N, s, err[0], err[1], err[2], TOL, pass ? "PASS" : "FAIL");
    }
    vfft_free(U);
    vfft_free(G);
    vfft_destroy(r2c);
    vfft_destroy(c2r);
    return pass ? 0 : 1;
}
