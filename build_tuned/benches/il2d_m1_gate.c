/* il2d_m1_gate.c — M1 gate for the native IL 2D c2c tier
 * (docs/roadmap/fft2d_il_c2c_design.md; driver in vfft.c, kernels = n1c).
 *
 * FORWARD elementwise vs a naive separable DFT, PER DIRECTION (roundtrip
 * never gates a permuted transform — and here it also could not detect a
 * wrong-but-consistent permutation). The gate is SELF-PROVING on
 * engagement: the native tier serves natural x natural while the convert
 * wrapper serves scrambled, so an elementwise pass IS proof the native
 * tier ran (mt_results_need_engagement_proof, adapted). Cells whose row
 * child (1D K=1 IL in-place NATURAL at N2) cannot be created are SKIPPED
 * loudly — that is the designed fallback, not a failure.
 *
 * Build: python build.py --src benches/il2d_m1_gate.c --vfft --compile
 * Run  : il2d_m1_gate.exe <SCRATCH wisdir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static void naive_axis_i(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int k, i, j;
    for (k = 0; k < N1; k++)
        for (j = 0; j < N2; j++) {
            double sr = 0.0, si = 0.0;
            for (i = 0; i < N1; i++) {
                const double a = sgn * 2.0 * pi * (double)(k * i) / (double)N1;
                const double c = cos(a), s = sin(a);
                const size_t x = 2 * ((size_t)i * N2 + j);
                sr += in[x] * c - in[x + 1] * s;
                si += in[x] * s + in[x + 1] * c;
            }
            out[2 * ((size_t)k * N2 + j)] = sr;
            out[2 * ((size_t)k * N2 + j) + 1] = si;
        }
}

static void naive_axis_j(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int i, k, j;
    for (i = 0; i < N1; i++)
        for (k = 0; k < N2; k++) {
            double sr = 0.0, si = 0.0;
            for (j = 0; j < N2; j++) {
                const double a = sgn * 2.0 * pi * (double)(k * j) / (double)N2;
                const double c = cos(a), s = sin(a);
                const size_t x = 2 * ((size_t)i * N2 + j);
                sr += in[x] * c - in[x + 1] * s;
                si += in[x] * s + in[x + 1] * c;
            }
            out[2 * ((size_t)i * N2 + k)] = sr;
            out[2 * ((size_t)i * N2 + k) + 1] = si;
        }
}

/* can the row child exist? probe the exact plan the driver builds */
static int row_child_ok(vfft_wisdom *W, int N2)
{
    vfft_config_t rc;
    vfft_plan p;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_INPLACE;
    rc.rigor = VFFT_MEASURE;
    rc.dims = 1;
    rc.n[0] = N2;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = W;
    p = vfft_create(&rc);
    if (!p) return 0;
    vfft_destroy(p);
    return 1;
}

int main(int argc, char **argv)
{
    static const int CELLS[][2] = {
        { 4, 16 },  { 8, 64 },   { 16, 16 },  { 16, 100 }, { 32, 64 },
        { 64, 64 }, { 64, 256 }, { 32, 5 },   { 64, 100 },
    };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    int fails = 0, skips = 0, ci, dir, oop;
    vfft_wisdom *W;
#ifdef _WIN32
    _putenv("VFFT_IL2D_NATIVE=1");
#else
    putenv("VFFT_IL2D_NATIVE=1");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d M1 gate (native tier, elementwise vs naive, wisdom=%s %s) ===\n",
           wisdir, W ? "loaded" : "MISSING");
    srand(20260825);
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const int N1 = CELLS[ci][0], N2 = CELLS[ci][1];
        const size_t T = (size_t)N1 * N2;
        double *x = malloc(2 * T * 8), *z = malloc(2 * T * 8);
        double *oz = malloc(2 * T * 8);
        double *t1 = malloc(2 * T * 8), *ref = malloc(2 * T * 8);
        size_t i;
        if (!x || !z || !oz || !t1 || !ref) { printf("OOM\n"); return 2; }
        fprintf(stderr, "[m1] cell %dx%d: probing row child...\n", N1, N2);
        if (!row_child_ok(W, N2)) {
            printf("  %3dx%-4d SKIP (row child at N2=%d unavailable — tier falls back)\n",
                   N1, N2, N2);
            skips++;
            free(x); free(z); free(oz); free(t1); free(ref);
            continue;
        }
        for (i = 0; i < 2 * T; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        for (dir = 0; dir < 2; dir++) {
            const double sgn = dir ? 1.0 : -1.0;
            naive_axis_i(x, t1, N1, N2, sgn);
            naive_axis_j(t1, ref, N1, N2, sgn);
            for (oop = 0; oop < 2; oop++) {
                vfft_config_t cfg;
                vfft_plan h;
                double maxe = 0.0, maxr = 0.0, rel;
                memset(&cfg, 0, sizeof cfg);
                cfg.transform = VFFT_C2C;
                cfg.placement = oop ? VFFT_OUTOFPLACE : VFFT_INPLACE;
                cfg.rigor = VFFT_MEASURE;
                cfg.dims = 2;
                cfg.n[0] = N1;
                cfg.n[1] = N2;
                cfg.howmany = 1;
                cfg.order = VFFT_ORDER_DEFAULT;
                cfg.layout = VFFT_LAYOUT_INTERLEAVED;
                cfg.nthreads = 1;
                cfg.wisdom = W;
                cfg.wisdom_write = 0;
                h = vfft_create(&cfg);
                if (!h) {
                    printf("  %3dx%-4d %s %s create FAIL\n", N1, N2,
                           dir ? "bwd" : "fwd", oop ? "oop" : "ip ");
                    fails++;
                    continue;
                }
                memcpy(z, x, 2 * T * 8);
                if (oop) {
                    memset(oz, 0, 2 * T * 8);
                    vfft_execute(h, dir ? VFFT_BACKWARD : VFFT_FORWARD,
                                 z, NULL, oz, NULL);
                } else {
                    vfft_execute(h, dir ? VFFT_BACKWARD : VFFT_FORWARD,
                                 z, NULL, z, NULL);
                }
                {
                    const double *got = oop ? oz : z;
                    for (i = 0; i < 2 * T; i++) {
                        double d = fabs(got[i] - ref[i]);
                        double m = fabs(ref[i]);
                        if (d > maxe) maxe = d;
                        if (m > maxr) maxr = m;
                    }
                }
                rel = maxe / (maxr > 0 ? maxr : 1.0);
                printf("  %3dx%-4d %s %s rel %.2e  %s\n", N1, N2,
                       dir ? "bwd" : "fwd", oop ? "oop" : "ip ", rel,
                       rel < 1e-10 ? "PASS" : "*** FAIL ***");
                if (rel >= 1e-10) fails++;
                vfft_destroy(h);
            }
        }
        free(x); free(z); free(oz); free(t1); free(ref);
    }
    if (W) vfft_wisdom_free(W);
    printf("\n%s (%d fail, %d skip)\n",
           fails ? "*** FAIL ***" : "=== ALL PASS ===", fails, skips);
    return fails ? 1 : 0;
}
