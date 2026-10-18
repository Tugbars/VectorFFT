/*
 * poisson3d.c -- a periodic Poisson solve: one 3D R2C, a divide in the half spectrum, one 3D C2R.
 *
 * A simulation holds a periodic source f on a grid (the mass density of particle-mesh gravity,
 * the charge on a particle-mesh Ewald grid, the divergence in a pressure projection) and needs
 * phi with lap phi = f. The box here is [0, 2pi) x [0, 2pi) x [0, pi), sampled on
 * N1 x N2 x N3 = 128 x 128 x 64 points at one spacing h = 2pi/128: not a cube, and the
 * contiguous axis, the one the real transforms halve, is the short one.
 *
 * The transforms (include/vfft.h). With W_N = e^{-j2pi/N}, VFFT_FORWARD computes
 *
 *     X[k1,k2,k3] = sum_{n1=0}^{N1-1} sum_{n2=0}^{N2-1} sum_{n3=0}^{N3-1}
 *                   x[n1,n2,n3] W_N1^{k1 n1} W_N2^{k2 n2} W_N3^{k3 n3},
 *
 * and VFFT_BACKWARD is the unnormalized inverse: the same sum over k with W_N^{-kn} returns
 * N1 N2 N3 x[n1,n2,n3]. A real x has X[k1,k2,k3] = X*[((-k1))_N1, ((-k2))_N2, ((-k3))_N3], so
 * the R2C plan writes only 0 <= k3 <= N3/2: N1 x N2 x (N3/2 + 1) (re, im) pairs, row-major,
 * bin k at index k. The C2R plan reads them back and, out of place and without
 * VFFT_FLAG_DESTROY_INPUT, leaves them intact.
 *
 * The solve. Bin k of an axis of N points over a length L has the angular wavenumber
 * kappa = 2pi k~/L (k~ = k for k <= N/2, k - N above), and lap becomes a product with -|kappa|^2:
 *
 *     Phi[k1,k2,k3] = -F[k1,k2,k3] / |kappa|^2,    Phi[0,0,0] = 0.
 *
 * A periodic phi exists only when F[0,0,0] = sum f[n] = 0, and Phi[0,0,0] = 0 picks the one
 * of zero mean. The 1/(N1 N2 N3) the C2R leaves out is folded into the multiplier. The
 * multiplier is real and even in k, so Phi keeps the symmetry of a real sequence's spectrum
 * and the Nyquist bins need no rule of their own (an odd multiplier, j kappa, would); only
 * kappa^2 enters, so the sign given to the Nyquist bin k = N/2 does not matter.
 *
 * The check. f[n1,n2,n3] = (lap phi_e)(n1 h, n2 h, n3 h) for the band-limited
 *
 *     phi_e(x) = sin x1 cos 2x2 sin 2x3 + (1/2) cos(3x1 - x2 + 4x3),
 *
 * so the computed phi[n1,n2,n3] must equal phi_e(n1 h, n2 h, n3 h) to rounding.
 *
 * Build it against an installed VectorFFT (how-to-use/ is the guide), with CMake:
 *
 *     cmake -S how-to-use/examples -B build-examples -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>
 *     cmake --build build-examples
 *
 * or with one compiler line, the compiler that built the library (MinGW: add -static, or
 * ship libwinpthread-1.dll beside the program; Linux: add -pthread):
 *
 *     gcc -O3 poisson3d.c -I<prefix>/include -L<prefix>/lib -lvfft -lm -o poisson3d
 *
 * Run it with a store of its own, an existing directory (PowerShell: $env:VFFT_WISDOM_DIR="<dir>"):
 *
 *     VFFT_WISDOM_DIR=<dir> ./poisson3d
 *
 * A create that misses wisdom measures the cell's candidates once and saves the winner in that
 * store before it returns; later runs are served from it. In the repository the root CMake build
 * compiles it as well, and python gauntlet/build.py --src how-to-use/examples/poisson3d.c --vfft
 * --compile builds it beside this file.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "vfft.h"

#define PI 3.14159265358979323846

enum
{
    N1 = 128,        /* n[0], the slowest axis */
    N2 = 128,        /* n[1] */
    N3 = 64,         /* n[2], contiguous */
    H3 = N3 / 2 + 1 /* the bins kept on the halved axis */
};

/* phi_e lies on a few bins far below Nyquist, so the discrete solve is exact and the error is
 * rounding alone, of order eps log2(N1 N2 N3) max|f| = 2.2e-16 x 20 x 22 = 1e-13. A wrong bin
 * index, a missed wrap of k > N/2, a wrong axis length or a missing scale costs 0.1 or more. */
static const double TOL = 1e-12;

static const double L1 = 2 * PI, L2 = 2 * PI, L3 = PI; /* the box */
static const double h = 2 * PI / N1;                   /* = L1/N1 = L2/N2 = L3/N3 */

/* the row-major index of (n1, n2, n3) on the real grid, the last axis contiguous */
static size_t idx(int n1, int n2, int n3)
{
    return ((size_t)n1 * N2 + (size_t)n2) * N3 + (size_t)n3;
}

/* the manufactured solution and its Laplacian: -(1 + 4 + 4) and -(9 + 1 + 16)/2 */
static double phi_e(double x1, double x2, double x3)
{
    return sin(x1) * cos(2 * x2) * sin(2 * x3) + 0.5 * cos(3 * x1 - x2 + 4 * x3);
}
static double lap_phi_e(double x1, double x2, double x3)
{
    return -9.0 * sin(x1) * cos(2 * x2) * sin(2 * x3) - 13.0 * cos(3 * x1 - x2 + 4 * x3);
}

/* the angular wavenumber of bin k on an axis of n points over the length len */
static double kappa(int k, int n, double len)
{
    return 2 * PI * (k <= n / 2 ? k : k - n) / len;
}

/* a rank-3 plan; the zeroed config is interleaved, out of place, natural order, one thread */
static vfft_plan plan3d(vfft_transform_t transform)
{
    vfft_config_t c;
    vfft_plan p;
    memset(&c, 0, sizeof c);
    c.transform = transform;
    c.rank = 3;
    c.n[0] = N1;
    c.n[1] = N2;
    c.n[2] = N3;
    p = vfft_create(&c);
    if (!p) /* read it now: the next create clears it */
        fprintf(stderr, "poisson3d: vfft_create: %s\n", vfft_error());
    return p;
}

/* Phi = -F / |kappa|^2 in place over the half spectrum, the C2R's 1/(N1 N2 N3) folded in */
static void divide_by_laplacian(double *F)
{
    const double scale = 1.0 / ((double)N1 * N2 * N3);
    int k1, k2, k3;
    for (k1 = 0; k1 < N1; k1++)
        for (k2 = 0; k2 < N2; k2++)
            for (k3 = 0; k3 < H3; k3++) /* k3 <= N3/2: this axis never wraps */
            {
                const double a = kappa(k1, N1, L1), b = kappa(k2, N2, L2);
                const double c = kappa(k3, N3, L3), kk = a * a + b * b + c * c;
                const double g = kk > 0.0 ? -scale / kk : 0.0; /* Phi[0,0,0] = 0 */
                double *z = F + 2 * (((size_t)k1 * N2 + k2) * H3 + k3);
                z[0] *= g;
                z[1] *= g;
            }
}

int main(void)
{
    const size_t nr = (size_t)N1 * N2 * N3, nc = (size_t)N1 * N2 * H3; /* reals, bins */
    vfft_plan r2c = plan3d(VFFT_R2C), c2r = plan3d(VFFT_C2R);
    double *f = vfft_malloc(nr * sizeof(double));
    double *F = vfft_malloc(2 * nc * sizeof(double));
    double *phi = vfft_malloc(nr * sizeof(double));
    double err = 0.0;
    int n1, n2, n3, pass = 0;

    if (r2c && c2r && f && F && phi)
    {
        for (n1 = 0; n1 < N1; n1++)
            for (n2 = 0; n2 < N2; n2++)
                for (n3 = 0; n3 < N3; n3++)
                    f[idx(n1, n2, n3)] = lap_phi_e(n1 * h, n2 * h, n3 * h);

        vfft_execute(r2c, VFFT_FORWARD, f, F);
        printf("poisson3d: mean of f = F[0,0,0]/(N1 N2 N3) = %.1e (a periodic phi needs 0)\n",
               F[0] / (double)nr);
        divide_by_laplacian(F);
        vfft_execute(c2r, VFFT_BACKWARD, F, phi);

        for (n1 = 0; n1 < N1; n1++)
            for (n2 = 0; n2 < N2; n2++)
                for (n3 = 0; n3 < N3; n3++)
                {
                    const double d = fabs(phi[idx(n1, n2, n3)] - phi_e(n1 * h, n2 * h, n3 * h));
                    if (d > err || isnan(d)) /* a NaN must fail, not vanish from the max */
                        err = d;
                }
        pass = err <= TOL;
        printf("poisson3d %dx%dx%d: max |phi[n] - phi_e(nh)| = %.2e, tolerance %.0e: %s\n",
               N1, N2, N3, err, TOL, pass ? "PASS" : "FAIL");
    }
    else if (r2c && c2r)
        fprintf(stderr, "poisson3d: vfft_malloc: out of memory\n");

    vfft_free(phi);
    vfft_free(F);
    vfft_free(f);
    vfft_destroy(c2r);
    vfft_destroy(r2c);
    return pass ? 0 : 1;
}
